import os
import glob
import torch
import torchaudio
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm
from FACodec_AC.config import Config, ASRConfig, PitchConfig
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
from enum import Enum
import torchcrepe
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

SCRIPT_LOCATION = os.environ.get("location")

sep = Separator(phone=" ", word="", syllable="")

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

def normalise(txt, pipeline):
	return pipeline["normaliser"].normalize(txt)

def clean(txt, pipeline):
	txt = txt.lower().replace("’", "'")
	txt = pipeline["regex"].sub(" ", txt)
	return re.sub(r"\s+", " ", txt).strip()

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

def g2p(words, proc):
	# tagged = [f"{pipeline['LANG_TAG']}{w}" for w in words]
	# inputs = pipeline["g2p_tok"](tagged, padding=True, return_tensors="pt").to(pipeline["dev_g2p"])
	# with torch.no_grad():
	# 	out = pipeline["g2p_net"].generate(**inputs, num_beams=1, max_length=50)
	# phones = pipeline["g2p_tok"].batch_decode(out, skip_special_tokens=True)
	# tidy = [p.replace("ˈ", "").replace("ˌ", "").replace("▁", "").replace(" ", "") for p in phones]
	# return " ".join(tidy)
    phone_str    = phonemize(words, language="en-us", backend="espeak",
                            strip=True, separator=sep)
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
    #token_ids = torch.tensor([token_ids], dtype=torch.int64, device=device)
    return token_ids


def text2ids(raw_txt, pipeline, proc):
	#ipa = g2p([clean(normalise(raw_txt, pipeline), pipeline)], pipeline)
	#ids = tokenizer(ipa, add_special_tokens=False).input_ids
    ids = g2p([clean(normalise(raw_txt, pipeline), pipeline)], proc)
    return ids, None


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
            tuple: A tuple containing the input filepath, a status string ("success" if processing and saving completed successfully; otherwise, an error message indicating the failure reason), and the standard deviations of content and prosody vectors.
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
        _, mask = pad_token_sequence(vq_id[1], Config.max_seq_len, Config.PAD_ID)
        # Process zc1 and zc2 instead of content_vector
        zc1_vector = get_z_from_indx(vq_id[1], torch.zeros(vq_id[1].shape[1], dtype=torch.long), fa_decoder, layer=0)
        zc1_vector = pad_token_sequence(zc1_vector, Config.max_seq_len, Config.PAD_ID)[0]
        zc2_vector = get_z_from_indx(vq_id[2], torch.zeros(vq_id[2].shape[1], dtype=torch.long), fa_decoder, layer=1)
        zc2_vector = pad_token_sequence(zc2_vector, Config.max_seq_len, Config.PAD_ID)[0]
        prosody_vector, _ = pad_token_sequence(quantized_arr[0], Config.max_seq_len, Config.PAD_ID)
        acoustic_vector, _ = pad_token_sequence(quantized_arr[2], Config.max_seq_len, Config.PAD_ID)
        base = os.path.splitext(os.path.basename(filepath))[0]
        torch.save({'zc1': zc1_vector, 'zc2': zc2_vector, 'mask': mask, 'prosody': prosody_vector, 'acoustic': acoustic_vector},
                   os.path.join(out_dir, f"{base}.pt"))
        # Compute per-dimension std (unbiased) for each tensor (each std is 256-dim)
        std_zc1    = torch.std(zc1_vector.float(), dim=1, unbiased=True)
        std_zc2    = torch.std(zc2_vector.float(), dim=1, unbiased=True)
        std_prosody= torch.std(prosody_vector.float(), dim=1, unbiased=True)
        # Compute running aggregates per channel and move them to CPU
        sum_zc1   = zc1_vector.float().sum(dim=1).cpu()           # shape [256]
        sumsq_zc1 = (zc1_vector.float() ** 2).sum(dim=1).cpu()      # shape [256]
        count_zc1 = zc1_vector.size(1)                              # scalar (same for all channels)
        sum_zc2   = zc2_vector.float().sum(dim=1).cpu()
        sumsq_zc2 = (zc2_vector.float() ** 2).sum(dim=1).cpu()
        count_zc2 = zc2_vector.size(1)
        sum_prosody   = prosody_vector.float().sum(dim=1).cpu()
        sumsq_prosody = (prosody_vector.float() ** 2).sum(dim=1).cpu()
        count_prosody = prosody_vector.size(1)
        return (filepath, "success", std_zc1, std_zc2, std_prosody,
                sum_zc1, sumsq_zc1, count_zc1,
                sum_zc2, sumsq_zc2, count_zc2,
                sum_prosody, sumsq_prosody, count_prosody)
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



def get_phone_forced_alignment(embedding_path, audio_folder, transcript_metadata, device, model, proc, target_sr, pipeline, inference=False):
	num_zeros = 0
	if not inference:
		embedding = torch.load(embedding_path)
		mask = embedding.get("mask", None)
		if mask is None:
			raise ValueError(f"No 'mask' key found in {embedding_path}")
		num_zeros = int((mask == 0).sum().item())
	
	file_id = os.path.splitext(os.path.basename(embedding_path))[0]
	
	# --- Use text processing functions with pipeline ---
	if file_id not in transcript_metadata:
		raise ValueError(f"Transcript for {file_id} not found in transcript metadata.")
	raw = transcript_metadata[file_id]
	token_ids_list, ipa = text2ids(raw, pipeline, proc)
	token_ids = torch.tensor([token_ids_list], dtype=torch.int64, device=device)
	# --- End text processing ---

	# Load audio and perform forced alignment.
	audio_path = os.path.join(audio_folder, f"{file_id}.wav")
	if not os.path.exists(audio_path):
		raise FileNotFoundError(f"Audio file not found: {audio_path}")
	wav, sr = torchaudio.load(audio_path)
	if sr != target_sr:
		wav = torchaudio.functional.resample(wav, sr, target_sr)
	with torch.inference_mode():
		feats = proc.feature_extractor(wav.to(device).squeeze(0), sampling_rate=target_sr, return_tensors="pt")
		feats = {k: v.to(device) for k, v in feats.items()}
		logits = model(**feats).logits
	logp = torch.log_softmax(logits, dim=-1)
	
	input_lens  = torch.tensor([logp.size(1)])
	target_lens = torch.tensor([token_ids.size(1)])
	blank_id    = 0  # CTC blank
	
	aligned_ids, frame_scores = forced_align(
		logp,
		token_ids,
		input_lens,
		target_lens,
		blank=blank_id,
	)
	if not inference:
		return aligned_ids, num_zeros, frame_scores,
	else:
		return aligned_ids, num_zeros, frame_scores, logits

def normalize_per_utt(f0_hz):
    voiced = f0_hz[~np.isnan(f0_hz)]
    if len(voiced):
        offset = np.median(np.log2(voiced))          # log‑Hz median
        f0_hz  = f0_hz / (2.0 ** offset)             # divide → median = 1 ×
    return f0_hz

def get_pitched_aligned(embedding_path, audio_folder, device, inference=False):
    """
    Extract pitch-aligned data from an audio file and embedding.
    """
    # Load embedding and count zeros in the mask
    num_zeros = 0
    if not inference:
        embedding = torch.load(embedding_path)
        mask = embedding.get("mask", None)
        if mask is None:
            raise ValueError(f"No 'mask' key found in {embedding_path}")
        num_zeros = int((mask == 0).sum().item())

    # Extract file ID and corresponding audio path
    file_id = os.path.splitext(os.path.basename(embedding_path))[0]
    audio_path = os.path.join(audio_folder, f"{file_id}.wav")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load audio and extract pitch
    wav_waveform, wav_sr = torchaudio.load(audio_path)
    if wav_sr != 16000:
        resample = torchaudio.transforms.Resample(orig_freq=wav_sr, new_freq=16000)
        wav_waveform = resample(wav_waveform)
    wav_waveform = wav_waveform.to(device)

    t, f0 = extract_pitch(wav_waveform, wav_sr, hop_ms=PitchConfig.hop_ms, fmin=50, fmax=500, crepe_model="full")
    t_coarse, f0_coarse = coarse_emotion_f0(t, f0, hop_ms=PitchConfig.hop_ms, median_ms=PitchConfig.median_ms, downsample_factor=PitchConfig.downsample_factor)
    f0_lp = lowpass(f0_coarse, cutoff_hz=PitchConfig.lowpass_cutoff, sr=1000.0 / PitchConfig.hop_ms) 
    f0_norm = normalize_per_utt(f0_lp)
    f0_quant = quantize_f0(f0_hz=f0_norm, n_bins=PitchConfig.n_bins, f0_min=0.5, f0_max=2.0)
    f0_quant = torch.tensor(f0_quant).unsqueeze(0).to(device)

    return f0_quant, num_zeros

class QuantizerNames(Enum):
    prosody = 0
    content = 1
    acoustic = 2

def get_z_from_indx(tokens, mask, fa_decoder, layer=0, quantizer_num=QuantizerNames.content):
    """
    Convert token indexes to continuous zc1 representations.
    
    tokens: LongTensor of shape [B, T] (pad token = 1025)
    mask: BooleanTensor of shape [B, T] where True indicates padding.
    layer: int, 0 for zc1 1 for zc2. Prosody has only 1 layer, content 2, acoustic 3.
    
    Replaces pad tokens with 0, embeds tokens via the codebook and projection,
    then zeros out the padded positions.
    """
    tokens_mod = tokens.clone()
    #tokens_mod[:,mask] = 0
    with torch.no_grad():
        quantizer_index = quantizer_num.value
        # Get codebook from the FACodec decoder.
        codebook = fa_decoder.quantizer[quantizer_index].layers[layer].codebook.weight  # [num_codes, code_dim]
        e_q = torch.nn.functional.embedding(tokens_mod, codebook)     # [B, T, code_dim]
        z_c1 = fa_decoder.quantizer[quantizer_index].layers[layer].out_proj(e_q)          # [B, T, 256]
    #pad_mask = mask.unsqueeze(-1).expand_as(z_c1)
    #z_c1[pad_mask] = 0
    z_c1 = rearrange(z_c1, "b t d -> b d t")
    return z_c1



def lowpass(x, cutoff_hz, sr, order=4):
    """
    Low‑pass filter that is NaN‑aware.
    - x:        1‑D numpy array (may contain NaNs)
    - cutoff_hz: desired cutoff
    - sr:       sample‑rate of the sequence (Hz)
    """
    voiced = ~np.isnan(x)
    if voiced.sum() < order * 3:      # not enough voiced frames to filter
        return x                      # just return as‑is

    # --- 1) fill gaps by linear interpolation
    x_filled = x.copy()
    idx      = np.flatnonzero(voiced)
    x_filled[~voiced] = np.interp(np.flatnonzero(~voiced), idx, x[idx])

    # --- 2) ordinary Butterworth low‑pass
    nyq = sr / 2.0
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    x_smooth = filtfilt(b, a, x_filled, method="pad")

    # --- 3) put NaNs back where the gaps were
    x_smooth[~voiced] = np.nan
    return x_smooth

def extract_pitch(wav_waveform: torch.Tensor,
                  wav_sr: int = 16000,
                  hop_ms: float = 10.0,
                  fmin: int = 50,
                  fmax: int = 500,
                  crepe_model: str = "full",
                  unvoiced_treshold=0.3) -> tuple[np.ndarray, np.ndarray]:
    """
    Params
    ----
    wav_path    : path to .wav (any sr / bit‑depth)
    sr_target   : resample rate for analysis (CREPE is trained at 16 kHz)
    hop_ms      : frame hop in milliseconds
    fmin, fmax  : analysis range in Hz
    crepe_model : "full" (robust) or "tiny" (faster) – only for torchcrepe
    """

    

    hop_length = int(round(hop_ms * 1e-3 * wav_sr))      # samples

    with torch.inference_mode():
        f0, pd = torchcrepe.predict(
            wav_waveform,
            wav_sr,
            hop_length,
            fmin,
            fmax,
            model=crepe_model,
            batch_size=1024,
            device=Config.device,
            return_periodicity=True,
        )
        f0 = f0.squeeze(0).cpu().numpy()          # [frames]
        pd = pd.squeeze(0).cpu().numpy()
        f0[pd < unvoiced_treshold] = np.nan                     # simple voicing mask   
        times = np.arange(len(f0)) * hop_length / wav_sr

        return times, f0
    

def coarse_emotion_f0(times: np.ndarray,
                      f0: np.ndarray,
                      hop_ms: float = 10.0,
                      median_ms: float = 100.0,
                      downsample_factor: int = 4):
    """
    Given a 10 ms-hop CREPE pitch track, produce a coarse emotion envelope:
      1) 100 ms median-filter (low-pass)
      2) down-sample by factor 4 (→ 40 ms hop)

    Args:
    -----
    times               : 1D array of frame center times (s), length N
    f0                  : 1D array of pitch values (Hz or NaN), length N
    hop_ms              : CREPE hop in ms (default 10)
    median_ms           : window size for median LP in ms (default 100)
    downsample_factor   : how many frames to skip when down-sampling

    Returns:
    --------
    times_coarse        : 1D array of coarse frame times (s), length ≈ N/downsample_factor
    f0_coarse           : 1D array of coarse pitch (Hz or NaN), same length
    """
    # 1) median-filter window in frames
    window_frames = int(np.round(median_ms / hop_ms))
    # ensure odd kernel for symmetry
    if window_frames % 2 == 0:
        window_frames += 1

    # 2) apply median-filter (centered)
    f0_lp = pd.Series(f0).rolling(window=window_frames,
                                  center=True,
                                  min_periods=1).median().values

    # 3) down-sample
    f0_coarse   = f0_lp[::downsample_factor]
    times_coarse = times[::downsample_factor]

    return times_coarse, f0_coarse

def quantize_f0(f0_hz, n_bins=32, f0_min=50, f0_max=500):
    """
    Converts an f0 vector (Hz, NaN=unvoiced) to discrete bin IDs.
    Returns:
        ids  – int array, 0..n_bins‑1 for voiced, n_bins for unvoiced
    """
    # Log‑scale edges
    log_f0 = np.log2(f0_hz)
    log_min, log_max = np.log2([f0_min, f0_max])

    # Clip & normalize
    voiced = ~np.isnan(log_f0)
    cents = (log_f0[voiced] - log_min) / (log_max - log_min)
    cents = np.clip(cents, 0, 0.9999)

    ids = np.full_like(f0_hz, fill_value=n_bins, dtype=np.int16)  # default=UNVOICED
    ids[voiced] = (cents * n_bins).astype(np.int16)
    return ids