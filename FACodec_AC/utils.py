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

def process_wav_facodec(filepath, fa_encoder, fa_decoder, out_dir, device):
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

def process_wav_wav2vec(embedding_path, audio_folder, output_folder, device, w2v_model, w2v_processor):
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

    # Interpolate predicted_ids to match the embedding sequence length
    phone_ids = (
        F.interpolate(predicted_ids.unsqueeze(0).float(),
                      size=num_zeros,
                      mode="nearest")
        .long()
        .squeeze(0)
    )

        # Optional: pad phone_ids up to a fixed length defined in ASRConfig.
    padded_phone_ids, pad_mask = pad_token_sequence(
        phone_ids, Config.time_frames, ASRConfig.PAD_ID
    )

    output_filename = os.path.basename(embedding_path)
    output_path = os.path.join(output_folder, output_filename)
    torch.save(padded_phone_ids, output_path)
    #print(f"Saved processed data to {output_path}")

def process_files_wav2vec(audio_folder, embeddings_folder, output_folder, device, w2v_model, w2v_processor):
    # Create the output folder if missing
    os.makedirs(output_folder, exist_ok=True)
    for file in tqdm.tqdm(os.listdir(embeddings_folder), desc="Processing files"):
        if file.endswith(".pt"):
            embedding_path = os.path.join(embeddings_folder, file)
            try:
                process_wav_wav2vec(embedding_path, audio_folder, output_folder, device, w2v_model, w2v_processor)
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