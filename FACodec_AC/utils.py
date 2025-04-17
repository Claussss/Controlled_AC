import os
import glob
import torch
import torchaudio
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm
from FACodec_AC.config import Config
import csv
import random
import string

def pad_token_sequence(seq, target_len, pad_id):
    """
    Pads seq up to target_len with pad_id, no truncation.
    Mask: False for original tokens, True for padded.
    """
    seq = seq.squeeze(0) if seq.dim() == 2 else seq  # ensure shape (T,)
    seq_len = seq.size(0)
    padded_seq = torch.full((target_len,), pad_id, dtype=seq.dtype)
    padded_seq[:seq_len] = seq
    
    mask = torch.zeros(target_len, dtype=torch.bool)
    if seq_len < target_len:
        mask[seq_len:] = True
    
    return padded_seq, mask

def process_wav(filepath, fa_encoder, fa_decoder, out_dir, device):
    try:
        wav_waveform, wav_sr = torchaudio.load(filepath)
        if wav_sr != 16000:
            resample = torchaudio.transforms.Resample(orig_freq=wav_sr, new_freq=16000)
            wav_waveform = resample(wav_waveform)
        wav_waveform = wav_waveform.to(device)
        with torch.no_grad():
            h_input = fa_encoder(wav_waveform[None, :, :])
            _, vq_id, _, _, _ = fa_decoder(h_input, eval_vq=False, vq=True)
        tokens, mask = pad_token_sequence(vq_id[1], Config.time_frames, Config.PAD_ID)
        base = os.path.splitext(os.path.basename(filepath))[0]
        torch.save({'tokens': tokens, 'mask': mask}, os.path.join(out_dir, f"{base}.pt"))
        return filepath, "success"
    except Exception as e:
        return filepath, f"error: {str(e)}"

def process_files(file_list, fa_encoder, fa_decoder, out_dir, device, workers=4, sequential=False):
    results = []
    if sequential:
        for f in tqdm.tqdm(file_list, desc="Processing sequentially"):
            fp, status = process_wav(f, fa_encoder, fa_decoder, out_dir, device)
            print(f"{fp}: {status}")
            results.append((fp, status))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_wav, f, fa_encoder, fa_decoder, out_dir, device)
                       for f in file_list]
            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                fp, status = future.result()
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
    return z_c1

def collate_fn(batch, pad_token_id=0):
    """
    Collate function assumes that the latent tokens and masks are fixed-length.
    Pads the variable-length target_ids.
    """
    latent_tokens = torch.stack([b["latent_tokens"] for b in batch], dim=0)  # [B, T]
    latent_mask = torch.stack([b["latent_mask"] for b in batch], dim=0)      # [B, T]
    transcripts = [b["transcript"] for b in batch]
    target_ids = [b["target_ids"] for b in batch]
    target_ids_padded = torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=pad_token_id)
    return {
        "latent_tokens": latent_tokens,
        "latent_mask": latent_mask,
        "transcripts": transcripts,
        "target_ids": target_ids_padded,
    }