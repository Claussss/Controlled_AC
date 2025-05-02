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

SCRIPT_LOCATION = os.environ.get("location")

def normalise(txt, pipeline):
	return pipeline["normaliser"].normalize(txt)

def clean(txt, pipeline):
	txt = txt.lower().replace("’", "'")
	txt = pipeline["regex"].sub(" ", txt)
	return re.sub(r"\s+", " ", txt).strip()

def g2p(words, pipeline):
	tagged = [f"{pipeline['LANG_TAG']}{w}" for w in words]
	inputs = pipeline["g2p_tok"](tagged, padding=True, return_tensors="pt").to(pipeline["dev_g2p"])
	with torch.no_grad():
		out = pipeline["g2p_net"].generate(**inputs, num_beams=1, max_length=50)
	phones = pipeline["g2p_tok"].batch_decode(out, skip_special_tokens=True)
	tidy = [p.replace("ˈ", "").replace("ˌ", "").replace("▁", "").replace(" ", "") for p in phones]
	return " ".join(tidy)

def text2ids(raw_txt, tokenizer, pipeline):
	ipa = g2p([clean(normalise(raw_txt, pipeline), pipeline)], pipeline)
	ids = tokenizer(ipa, add_special_tokens=False).input_ids
	return ids, ipa
# --- End global text processing functions ---

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
        #content_vector, _ = pad_token_sequence(quantized_arr[1], Config.max_seq_len, Config.PAD_ID)
        content_vector = get_zc1_from_indx(vq_id[1], torch.zeros(vq_id[1].shape[1], dtype=torch.long), fa_decoder)
        content_vector = pad_token_sequence(content_vector, Config.max_seq_len, Config.PAD_ID)[0]
        prosody_vector, _ = pad_token_sequence(quantized_arr[0], Config.max_seq_len, Config.PAD_ID)
        acoustic_vector, _ = pad_token_sequence(quantized_arr[2], Config.max_seq_len, Config.PAD_ID)
        base = os.path.splitext(os.path.basename(filepath))[0]
        torch.save({'content': content_vector, 'mask': mask, 'prosody': prosody_vector, 'acoustic': acoustic_vector},
                   os.path.join(out_dir, f"{base}.pt"))
        # Compute standard deviations (for reporting only)
        std_content = torch.std(content_vector.float())
        std_prosody = torch.std(prosody_vector.float())
        # Compute running aggregates for dynamic global stat computation
        sum_content   = content_vector.float().sum()
        sumsq_content = (content_vector.float() ** 2).sum()
        count_content = content_vector.numel()
        sum_prosody   = prosody_vector.float().sum()
        sumsq_prosody = (prosody_vector.float() ** 2).sum()
        count_prosody = prosody_vector.numel()
        return (filepath, "success", std_content, std_prosody,
                sum_content, sumsq_content, count_content,
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
	token_ids_list, ipa = text2ids(raw, proc.tokenizer, pipeline)
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