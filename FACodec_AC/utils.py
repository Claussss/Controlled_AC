import os
import glob
import torch
import torchaudio
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm
from FACodec_AC.config import Config

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