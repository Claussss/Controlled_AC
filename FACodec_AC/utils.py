import os
import glob
import torch
import torchaudio
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm
from FACodec_AC.config import Config, ASRConfig
import csv
import random
import string
import torch.nn.functional as F
from einops import rearrange
from torchaudio.functional import forced_align
import re
from num2words import num2words
from phonemizer import phonemize
from phonemizer.separator import Separator
from phonemizer.backend.espeak.wrapper import EspeakWrapper
SCRIPT_LOCATION = os.environ.get("location")

# This is just because delta cannot install espeak properly
if SCRIPT_LOCATION == "server":
    # absolute path to the library you compiled
    lib_path = "/u/yurii/.local/lib/libespeak-ng.so.1"
    # 1) let phonemizer know
    EspeakWrapper.set_library(lib_path)          # python-only
    # 2) make sure the dynamic loader can also find it
    os.environ["LD_LIBRARY_PATH"] = (
        os.path.dirname(lib_path) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    )
    os.environ["ESPEAK_DATA_PATH"] = "/u/yurii/.local/share/espeak-ng-data"


def pad_token_sequence(seq, target_len, pad_id):
    """
    Pads seq along its last dimension to target_len.
    
    If seq is one-dimensional (shape (T,)), padding is done with pad_id.
    If seq is multi-dimensional (e.g. shape (..., T)), padded vectors are zeros.
    
    Returns:
      padded_seq: tensor with the same shape as seq except the last dimension is target_len.
      mask: Boolean tensor of the same shape as padded_seq, where True indicates padded positions.
    """
    # Determine sequence length from the last dimension.
    seq_len = seq.shape[-1]
    
    # Create shape for padded sequence: same as seq except last dimension = target_len.
    new_shape = list(seq.shape)
    new_shape[-1] = target_len
    
    # If the sequence is one-dimensional, pad with pad_id, otherwise pad with zeros.
    if seq.dim() == 1:
        padded_seq = torch.full(new_shape, pad_id, dtype=seq.dtype, device=seq.device)
    else:
        padded_seq = torch.zeros(new_shape, dtype=seq.dtype, device=seq.device)
    
    # Create mask with the same shape: padded positions marked as True.
    mask = torch.zeros(new_shape, dtype=torch.bool, device=seq.device)
    
    # Copy original values into padded_seq along the last dimension.
    indices = [slice(None)] * (seq.dim() - 1) + [slice(0, seq_len)]
    padded_seq[tuple(indices)] = seq
    
    # For padded positions, mark True in the mask.
    if seq_len < target_len:
        indices_pad = [slice(None)] * (seq.dim() - 1) + [slice(seq_len, target_len)]
        mask[tuple(indices_pad)] = True

    # Remove the extra first dimension if present with size 1.
    if padded_seq.dim() > 1 and padded_seq.size(0) == 1:
        padded_seq = padded_seq.squeeze(0)
        mask = mask.squeeze(0)
    
    return padded_seq, mask

def interpolate_alignment(predicted_ids, num_zeros, mode='nearest'):
    """
    Interpolates the predicted_ids (alignment) to match the expected sequence length.
    """
    phone_ids = (
        F.interpolate(predicted_ids.unsqueeze(0).float(),
                      size=num_zeros,
                      mode=mode)
        .long()
        .squeeze(0)
    )
    return phone_ids


def process_wav_facodec(filepath, fa_encoder, fa_decoder, out_dir, device):
    """
    Processes an audio file using the FACODec to extract and pad zc1, zc2, prosody, and acoustic features.
    Specifically, it extracts:
        - zc1 tokens (from vq_id[1]), along with a corresponding padding mask,
        - zc2 tokens (from vq_id[2]),
        - a prosody vector (from quantized_arr[0]), and
        - an acoustic vector (from quantized_arr[2]).
    Each of these components is padded to a fixed maximum sequence length using a provided padding function.
    Finally, the function saves the padded tokens, mask, prosody, and acoustic vectors as a PyTorch file in the specified output directory.

    Parameters:
            filepath (str): The path to the input WAV audio file.
            fa_encoder (Callable): The FACODec encoder model function to process the audio waveform.
            fa_decoder (Callable): The FACODec decoder model function to extract latent features.
            out_dir (str): The directory path where the processed output will be saved.
            device (torch.device or str): The device on which the processing will be performed.

    Returns:
            tuple: A tuple containing the input filepath and a status string ("success" if processing and saving completed successfully; otherwise, an error message indicating the failure reason).
    """
    try:
        wav_waveform, wav_sr = torchaudio.load(filepath)
        if wav_sr != 16000:
            resample = torchaudio.transforms.Resample(orig_freq=wav_sr, new_freq=16000)
            wav_waveform = resample(wav_waveform)
        wav_waveform = wav_waveform.to(device)
        with torch.no_grad():
            h_input = fa_encoder(wav_waveform[None, :, :])
            _, vq_id, _, quantized_arr, _ = fa_decoder(h_input, eval_vq=False, vq=True)
        tokens, mask = pad_token_sequence(vq_id[1], Config.max_seq_len, Config.PAD_ID)
        tokens_zc2, _ = pad_token_sequence(vq_id[2], Config.max_seq_len, Config.PAD_ID)
        prosody_vector, _ = pad_token_sequence(quantized_arr[0], Config.max_seq_len, Config.PAD_ID)
        acoustic_vector, _ = pad_token_sequence(quantized_arr[2], Config.max_seq_len, Config.PAD_ID)
        base = os.path.splitext(os.path.basename(filepath))[0]
        torch.save({'tokens': tokens, 'mask': mask, 'tokens_zc2': tokens_zc2, 'prosody':prosody_vector, 'acoustic':acoustic_vector}, os.path.join(out_dir, f"{base}.pt"))
        return filepath, "success"
    except Exception as e:
        return filepath, f"error: {str(e)}"

def load_transcript_metadata(transcript_file):
    """
    Loads transcript metadata from a CSV file.
    Each row in the CSV file should be formatted as:
        file_id|transcript|...
    Returns a dict mapping file_id to transcript.
    """
    metadata = {}
    with open(transcript_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) >= 2:
                file_id = row[0].strip()
                transcript = row[1].strip()
                metadata[file_id] = transcript
    return metadata

def clean_transcript(transcript: str) -> str:
    """
    Converts currency numbers prefixed by '£' so that the number is converted to words
    and 'pounds' is appended (e.g., "£81,791" -> "eighty-one thousand seven hundred and ninety-one pounds").
    Then, replaces other plain numbers with words, maps additional symbols, removes punctuation,
    and lowercases the transcript.
    """
    # Function to convert currency numbers.
    def replace_currency(match):
        num_str = match.group(1)
        # Remove commas -- this handles thousands separators.
        num_clean = num_str.replace(",", "")
        try:
            # Attempt to convert to a number; if there's a decimal point, cast as float.
            if "." in num_clean:
                number = float(num_clean)
                # Convert only the integer portion since num2words expects an integer.
                number = int(round(number))
            else:
                number = int(num_clean)
            words = num2words(number)
        except Exception:
            # If conversion fails, leave the original text.
            words = num_str
        return f"{words} pounds"

    # Handle currency numbers: find patterns like "£81,791" or "£ 81,791.50"
    transcript = re.sub(
        r"£\s*((?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)",
        replace_currency,
        transcript
    )
    
    # Replace remaining standalone numbers with words.
    transcript = re.sub(
        r'\b\d+\b',
        lambda m: num2words(int(m.group(0))),
        transcript
    )
    
    # Map additional symbols to their desired spoken forms.
    symbol_replacements = {
        "Â": "a",   
        "É": "e",
        "À": "a",
        "Ê": "e",
        "Ü": "u",
        "“": " ",
        "”": " ",
        "’": "'"
    }
    for symbol, replacement in symbol_replacements.items():
        transcript = transcript.replace(symbol, replacement)
    
    # Remove remaining punctuation.
    translator = str.maketrans('', '', string.punctuation)
    cleaned = transcript.translate(translator)
    
    # Normalize whitespace and convert to lowercase.
    cleaned = " ".join(cleaned.split()).lower()

    #phoneme_seq = phonemize(cleaned, language="en-us", backend="espeak", strip=True, njobs=1)
    return cleaned

def get_grapheme_forced_alignment(embedding_path, audio_folder, transcript_metadata, device, model, bundle, target_sr, inference=False):
    """
    Loads the embedding (to count zeros from the mask) and retrieves the transcript 
    from transcript_metadata (after cleaning), then performs forced alignment 
    on the corresponding audio file, returning predicted token IDs and num_zeros.
    
    Args:
        embedding_path (str): Path to the embedding (.pt) file.
        audio_folder (str): Directory containing the audio (.wav) files.
        transcript_metadata (dict): Mapping from file_id to transcript.
        device (torch.device or str): Device for computation.
        model: Wav2Vec2 model.
        bundle: Torchaudio pipeline bundle.
        target_sr (int): Target sample rate.
        inference (bool): During inference, I don't have saved embeddings because I extract those on the fly.
        
    Returns:
        predicted_ids (Tensor): Forced-aligned predicted token IDs.
        num_zeros (int): Number of non-padded (non-zero) frames based on the mask.
    """
    # Load embedding and count zeros in the mask.
    num_zeros = 0
    if not inference:
        embedding = torch.load(embedding_path)
        mask = embedding.get("mask", None)
        if mask is None:
            raise ValueError(f"No 'mask' key found in {embedding_path}")
        num_zeros = int((mask == 0).sum().item())
    
    # Extract file_id (assumes embedding filename is like "LJ001-0002.pt")
    file_id = os.path.splitext(os.path.basename(embedding_path))[0]
    
    # Retrieve and clean the transcript from metadata.
    if file_id not in transcript_metadata:
        raise ValueError(f"Transcript for {file_id} not found in transcript metadata.")
    raw_transcript = transcript_metadata[file_id]
    cleaned_transcript = clean_transcript(raw_transcript)  # lower and remove punctuation
    
    # Load the corresponding audio file.
    audio_path = os.path.join(audio_folder, f"{file_id}.wav")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    wav, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    # Get frame-level log-probs.
    with torch.inference_mode():
        emissions, _ = model(wav.to(device))
    log_probs = torch.log_softmax(emissions, dim=-1).cpu()
    
    # Build transcript → token IDs.
    labels = bundle.get_labels()  # e.g. ['-', '|', 'E', 'T', …]
    blank_id = 0  # CTC blank is assumed to be index 0
    # Uppercase and replace spaces with pipes.
    tx = cleaned_transcript.upper().replace(" ", "|")
    char2idx = {c: i for i, c in enumerate(labels)}
    try:
        token_ids = torch.tensor([[char2idx[c] for c in tx]], dtype=torch.int64)
    except KeyError as e:
        raise ValueError(f"Character {e} not found in labels")
    
    input_lens = torch.tensor([log_probs.size(1)])
    target_lens = torch.tensor([token_ids.size(1)])
    
    # Forced alignment (Viterbi).
    predicted_ids, frame_scores = forced_align(
        log_probs,    # (1, T', V)
        token_ids,    # (1, N)
        input_lens,   # (1,)
        target_lens,  # (1,)
        blank=blank_id,
    )
    
    return predicted_ids, num_zeros, frame_scores

def greedy_split(token: str, vocab):
    """
    Split the given token into substrings found in the vocab using the longest possible prefix match.
    If no match is found, the token is split into single characters.

    This function is used only for get_phone_forced_alignment as the phonemizer sometimes returns merged phones.

    Args:
        token (str): The input token to be split.
        vocab (iterable): Collection of valid substrings.

    Returns:
        list: A list of substrings (or individual characters) from the token.
    """
    splits = []
    idx    = 0
    L      = len(token)
    while idx < L:
        # Try longest possible match from the remaining string
        for end in range(L, idx, -1):
            piece = token[idx:end]
            if piece in vocab:
                splits.append(piece)
                idx = end
                break
        else:
            # No substring match—just emit the single character
            splits.append(token[idx])
            idx += 1
    return splits


def get_phone_forced_alignment(embedding_path, audio_folder, transcript_metadata, device, model, proc, target_sr, inference=False):
    """
    Same args as in get_grapheme_forced_alignment, but this uses phonemizer and returns ids of phones not graphemes.
    """
    num_zeros = 0
    if not inference:
        embedding = torch.load(embedding_path)
        mask = embedding.get("mask", None)
        if mask is None:
            raise ValueError(f"No 'mask' key found in {embedding_path}")
        num_zeros = int((mask == 0).sum().item())


    # Extract file_id (assumes embedding filename is like "LJ001-0002.pt")
    file_id = os.path.splitext(os.path.basename(embedding_path))[0]
    # Retrieve and clean the transcript from metadata.
    if file_id not in transcript_metadata:
        raise ValueError(f"Transcript for {file_id} not found in transcript metadata.")
    raw = transcript_metadata[file_id]
    clean = clean_transcript(raw)

    # 1) Get the raw phoneme tokens
    phone_str    = phonemize([clean], language="en-us", backend="espeak",
                            strip=True, separator=Separator(phone=" ", word="", syllable=""))
    raw_seq      = phone_str[0].split()

    # 2) Load the target model’s phoneme vocab
    model_vocab  = set(proc.tokenizer.get_vocab().keys())

    # 3) Fix any invalid tokens
    phone_seq = []
    for tok in raw_seq:
        if tok in model_vocab:
            phone_seq.append(tok)
        else:
            phone_seq.extend(greedy_split(tok, model_vocab))

    # 3) map phonemes → token IDs with the CTC tokenizer
    #    any missing phonemes must be added or mapped to <unk>
    vocab   = proc.tokenizer.get_vocab()
    token_ids = [vocab.get(p, proc.tokenizer.unk_token_id) for p in phone_seq]
    token_ids = torch.tensor([token_ids], dtype=torch.int64, device=device)

    # 4) compute emissions
    audio_path = os.path.join(audio_folder, f"{file_id}.wav")
    wav, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    with torch.inference_mode():
        feats = proc.feature_extractor(wav.to(device).squeeze(0), sampling_rate=target_sr, return_tensors="pt")
        feats = { k: v.to(device) for k, v in feats.items() }
        logits = model(**feats).logits          # (1, T', V)
    logp = torch.log_softmax(logits, dim=-1)

    # 5) forced alignment on phoneme IDs
    input_lens  = torch.tensor([logp.size(1)])
    target_lens = torch.tensor([token_ids.size(1)])
    blank_id    = 0  # CTC blank is assumed to be index 0

    aligned_ids, frame_scores = forced_align(
        logp,        # (1, T', V)
        token_ids,   # (1, N)
        input_lens,  # (1,)
        target_lens, # (1,)
        blank=blank_id,
    )
    # aligned_ids is a tensor of phoneme‐CTC IDs (with repeats & blanks)

    return aligned_ids, num_zeros, frame_scores, logits

def get_asr_alignment(embedding_path, audio_folder, device, w2v_model, w2v_processor):
    """
    Transcribes the audio corresponding to the embedding, producing an aligned transcription.
    """
    embedding = torch.load(embedding_path)
    mask = embedding.get("mask", None)
    if mask is None:
        raise ValueError(f"No 'mask' key found in {embedding_path}")
    num_zeros = int((mask == 0).sum().item())

    # Recover audio filename from embedding filename: sample.pt -> sample.wav
    audio_filename = os.path.basename(embedding_path).replace('.pt', '.wav')
    audio_path = os.path.join(audio_folder, audio_filename)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Inline prepare_wav2vec_input functionality:
    audio, sample_rate = torchaudio.load(audio_path)
    target_rate = 16000
    if sample_rate != target_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_rate)
        audio = resampler(audio)
    # Preprocess audio for model input
    inputs = w2v_processor(audio.squeeze(0), sampling_rate=target_rate, return_tensors="pt", padding=True)
    w2v_input = inputs.input_values.to(device)

    with torch.no_grad():
        w2v_outputs = w2v_model(w2v_input).logits
    predicted_ids = torch.argmax(w2v_outputs, dim=-1)
    return predicted_ids, num_zeros


def get_zc1_from_indx(tokens, mask, fa_decoder):
    """
    Convert token indexes to continuous zc1 representations.
    
    tokens: LongTensor of shape [B, T] (pad token = 1025)
    mask: BooleanTensor of shape [B, T] where True indicates padding.
    
    Replaces pad tokens with 0, embeds tokens via the codebook and projection,
    then zeros out the padded positions.
    """
    tokens_mod = tokens.clone()
    tokens_mod[mask] = 0
    with torch.no_grad():
        # Get codebook from the FACodec decoder.
        codebook = fa_decoder.quantizer[1].layers[0].codebook.weight  # [num_codes, code_dim]
        e_q = torch.nn.functional.embedding(tokens_mod, codebook)     # [B, T, code_dim]
        z_c1 = fa_decoder.quantizer[1].layers[0].out_proj(e_q)          # [B, T, 256]
    pad_mask = mask.unsqueeze(-1).expand_as(z_c1)
    z_c1[pad_mask] = 0
    z_c1 = rearrange(z_c1, "b t d -> b d t")
    return z_c1

def get_mask_positions(x: torch.Tensor, r_range: tuple = (0.3, 0.4), p_drop: float = 0.1) -> torch.Tensor:
    """
    Args:
        x      : LongTensor of token IDs of shape [B, T]
        r_range: tuple (default: (0.3, 0.4)). Defines the lower and upper bound for the percentage of tokens 
                 to drop in a contiguous segment when not dropping the entire sequence.
        p_drop : float. With probability p_drop, the entire sequence is masked.
                 Otherwise, a single contiguous segment is masked based on r_range.
                 
    Returns:
        mask   : BoolTensor of shape [B, T] where masked positions are True and unmasked are False,
                 excluding PAD tokens (those with value Config.PAD_ID).
    """
    B, T = x.shape
    device = x.device
    mask = torch.zeros((B, T), dtype=torch.bool, device=device)
    
    for b in range(B):
        if random.random() < p_drop:
            mask[b] = True
        else:
            r = random.uniform(r_range[0], r_range[1])
            seg_len = max(1, int(round(r * T)))
            start = random.randint(0, T - seg_len)
            mask[b, start:start + seg_len] = True

    # Exclude PAD tokens from being masked.
    mask = mask & (x != Config.PAD_ID)
    return mask