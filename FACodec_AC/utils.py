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

def process_wav_facodec(filepath, fa_encoder, fa_decoder, out_dir, device):
    try:
        wav_waveform, wav_sr = torchaudio.load(filepath)
        if wav_sr != 16000:
            resample = torchaudio.transforms.Resample(orig_freq=wav_sr, new_freq=16000)
            wav_waveform = resample(wav_waveform)
        wav_waveform = wav_waveform.to(device)
        with torch.no_grad():
            h_input = fa_encoder(wav_waveform[None, :, :])
            _, vq_id, _, quantized_arr, _ = fa_decoder(h_input, eval_vq=False, vq=True)
        tokens, mask = pad_token_sequence(vq_id[1], Config.time_frames, Config.PAD_ID)
        tokens_zc2, _ = pad_token_sequence(vq_id[2], Config.time_frames, Config.PAD_ID)
        prosody_vector, _ = pad_token_sequence(quantized_arr[0], Config.time_frames, Config.PAD_ID)
        acoustic_vector, _ = pad_token_sequence(quantized_arr[2], Config.time_frames, Config.PAD_ID)
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

def get_wav2vec_forced_predicted_ids(embedding_path, audio_folder, transcript_metadata, device, model, bundle, target_sr, inference=False):
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
    predicted_ids, _ = forced_align(
        log_probs,    # (1, T', V)
        token_ids,    # (1, N)
        input_lens,   # (1,)
        target_lens,  # (1,)
        blank=blank_id,
    )
    
    return predicted_ids, num_zeros

def greedy_split(token: str, vocab):
    """
    Split `token` into a sequence of substrings found in `vocab`
    by always taking the longest possible prefix match.
    If no match is found, falls back to single‐char tokens.
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


def get_text_phoneme_forced_ids(embedding_path, audio_folder, transcript_metadata, device, model, proc, target_sr, inference=False):
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

    return aligned_ids, num_zeros, frame_scores

def prepare_wav_wav2vec(audio_path, device, w2v_processor):
    # Load audio and resample if needed
    audio, sample_rate = torchaudio.load(audio_path)
    target_rate = 16000
    if sample_rate != target_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_rate)
        audio = resampler(audio)
    # Preprocess audio for model input
    inputs = w2v_processor(audio.squeeze(0), sampling_rate=target_rate, return_tensors="pt", padding=True)
    return inputs.input_values.to(device)

def get_wav2vec_predicted_ids(embedding_path, audio_folder, device, w2v_model, w2v_processor):
    """
    Loads the embedding, recovers the corresponding audio,
    and returns the predicted token ids along with the number of non-padded positions.
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

    w2v_input = prepare_wav_wav2vec(audio_path, device, w2v_processor)
    with torch.no_grad():
        w2v_outputs = w2v_model(w2v_input).logits
    predicted_ids = torch.argmax(w2v_outputs, dim=-1)
    return predicted_ids, num_zeros

def interpolate_and_pad_wav2vec_predicted_ids(predicted_ids, num_zeros):
    """
    Interpolates the predicted_ids to match the expected sequence length and pads them.
    """
    phone_ids = (
        F.interpolate(predicted_ids.unsqueeze(0).float(),
                      size=num_zeros,
                      mode="nearest")
        .long()
        .squeeze(0)
    )
    padded_phone_ids, pad_mask = pad_token_sequence(
        phone_ids, Config.time_frames, ASRConfig.PAD_ID
    )
    return padded_phone_ids, pad_mask

def process_files_wav2vec(audio_folder, embeddings_folder, output_folder, device, w2v_model, w2v_processor):
    # Create the output folder if missing
    os.makedirs(output_folder, exist_ok=True)
    for file in tqdm.tqdm(os.listdir(embeddings_folder), desc="Processing files"):
        if file.endswith(".pt"):
            embedding_path = os.path.join(embeddings_folder, file)
            try:
                predicted_ids, num_zeros = get_wav2vec_predicted_ids(embedding_path, audio_folder, device, w2v_model, w2v_processor)
                padded_phone_ids, pad_mask = interpolate_and_pad_wav2vec_predicted_ids(predicted_ids, num_zeros)
                # Save the processed data
                output_path = os.path.join(output_folder, file)
                torch.save(padded_phone_ids, output_path)
            except Exception as e:
                print(f"Error processing {file}: {e}")

def process_files_wav2vec_forced(audio_folder, embeddings_folder, transcript_metadata, output_folder, device, model, bundle, target_sr):
    """
    Processes each embedding file in embeddings_folder by performing forced alignment.

    For each embedding file (.pt), this function:
      1. Loads the embedding, counts zeros in the "mask",
      2. Retrieves and cleans the transcript from transcript_metadata,
      3. Loads the corresponding audio file from audio_folder,
      4. Computes frame-level log-probs and forced aligns with the transcript to get predicted token IDs,
      5. Interpolates and pads the predicted_ids to match the expected length,
      6. Saves the padded predicted_ids to output_folder.
    
    Args:
        audio_folder (str): Directory containing the audio (.wav) files.
        embeddings_folder (str): Directory containing embedding (.pt) files.
        transcript_metadata (dict): Mapping from file_id to transcript.
        output_folder (str): Directory where processed files will be saved.
        device (torch.device or str): Device for computation.
        model: Wav2Vec2 model.
        bundle: Torchaudio pipeline bundle.
        target_sr (int): Target sample rate.
    """

    os.makedirs(output_folder, exist_ok=True)
    
    for file in tqdm.tqdm(os.listdir(embeddings_folder), desc="Processing forced alignment files"):
        if file.endswith(".pt"):
            embedding_path = os.path.join(embeddings_folder, file)
            try:
                predicted_ids, num_zeros = get_wav2vec_forced_predicted_ids(
                    embedding_path,
                    audio_folder,
                    transcript_metadata,
                    device,
                    model,
                    bundle,
                    target_sr
                )
                padded_phone_ids, _ = interpolate_and_pad_wav2vec_predicted_ids(predicted_ids, num_zeros)
                output_path = os.path.join(output_folder, file)
                torch.save(padded_phone_ids, output_path)
                #print(f"{file}: success")
            except Exception as e:
                print(f"Error processing {file}: {e}")

def process_files_text_phoneme_forced(audio_folder, embeddings_folder, transcript_metadata, output_folder, device, model, proc, target_sr):
    """
    Processes each embedding file in embeddings_folder by performing forced alignment.

    For each embedding file (.pt), this function:
      1. Loads the embedding, counts zeros in the "mask",
      2. Retrieves and cleans the transcript from transcript_metadata,
      3. Loads the corresponding audio file from audio_folder,
      4. Computes frame-level log-probs and forced aligns with the transcript to get predicted token IDs,
      5. Interpolates and pads the predicted_ids to match the expected length,
      6. Saves the padded predicted_ids to output_folder.
    
    Args:
        audio_folder (str): Directory containing the audio (.wav) files.
        embeddings_folder (str): Directory containing embedding (.pt) files.
        transcript_metadata (dict): Mapping from file_id to transcript.
        output_folder (str): Directory where processed files will be saved.
        device (torch.device or str): Device for computation.
        model: Wav2Vec2 model.
        bundle: Torchaudio pipeline bundle.
        target_sr (int): Target sample rate.
    """

    os.makedirs(output_folder, exist_ok=True)
    for file in tqdm.tqdm(os.listdir(embeddings_folder), desc="Processing forced alignment files"):
        if file.endswith(".pt"):
            embedding_path = os.path.join(embeddings_folder, file)
            try:
                predicted_ids, num_zeros = get_text_phoneme_forced_ids(
                    embedding_path,
                    audio_folder,
                    transcript_metadata,
                    device,
                    model,
                    proc,
                    target_sr
                )
                padded_phone_ids, _ = interpolate_and_pad_wav2vec_predicted_ids(predicted_ids, num_zeros)
                output_path = os.path.join(output_folder, file)
                torch.save(padded_phone_ids, output_path)
                #print(f"{file}: success")
            except Exception as e:
                print(f"Error processing {file}: {e}")

def process_files_facodec(file_list, fa_encoder, fa_decoder, out_dir, device):
    results = []
    for f in tqdm.tqdm(file_list, desc="Processing files"):
        fp, status = process_wav_facodec(f, fa_encoder, fa_decoder, out_dir, device)
        print(f"{fp}: {status}")
        results.append((fp, status))
    return results

def normalize_transcript(transcript):
    """Remove punctuation and lowercase the transcript."""
    return transcript.translate(str.maketrans('', '', string.punctuation)).lower()

def load_metadata(metadata_path):
    """Return a dict mapping file_id to normalized transcript."""
    meta = {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) < 2:
                continue
            file_id = row[0].strip()
            transcript = row[1].strip()
            meta[file_id] = normalize_transcript(transcript)
    return meta

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

def collate_fn(batch, pad_token_id=0):
    """
    Collate function assumes that the latent tokens and masks are fixed-length.
    Pads the variable-length target_ids.
    """
    latent_tokens = torch.stack([b["latent_tokens"] for b in batch], dim=0)  # [B, T]
    latent_mask = torch.stack([b["latent_mask"] for b in batch], dim=0)      # [B, T]
    # 2) collect true input lengths
    input_lengths = (latent_mask == 0).sum(dim=1)                             # [B]

    # 3) grab the raw target lists, no padding
    target_lists   = [b["target_ids"] for b in batch]                  # list of [L_i]
    target_lengths = torch.tensor([t.size(0) for t in target_lists])  # [B]
    targets_flat   = torch.cat(target_lists, dim=0)                    # [sum(L_i)]

    return {
      "latent_tokens":  latent_tokens,
      "latent_mask":    latent_mask,
      "input_lengths":  input_lengths,
      "targets":        targets_flat,
      "target_lengths": target_lengths,
    }