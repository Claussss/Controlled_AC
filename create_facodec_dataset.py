import os
import glob
import random
from sklearn.model_selection import train_test_split
from FACodec_AC.utils import process_wav_facodec
from huggingface_hub import hf_hub_download
import torch
import torchaudio
import tqdm
from FACodec_AC.config import Config

# Device configuration
device = Config.device

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Amphion'))

# Imports for FACodec
from models.codec.ns3_codec import FACodecEncoder, FACodecDecoder
SCRIPT_LOCATION = os.environ.get("location")

# Setup FACodec models
fa_encoder = FACodecEncoder(ngf=32, up_ratios=[2,4,5,5], out_channels=256)
fa_decoder = FACodecDecoder(
    in_channels=256,
    upsample_initial_channel=1024,
    ngf=32,
    up_ratios=[5,5,4,2],
    vq_num_q_c=2,
    vq_num_q_p=1,
    vq_num_q_r=3,
    vq_dim=256,
    codebook_dim=8,
    codebook_size_prosody=10,
    codebook_size_content=10,
    codebook_size_residual=10,
    use_gr_x_timbre=True,
    use_gr_residual_f0=True,
    use_gr_residual_phone=True,
)
encoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder.bin")
decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder.bin")
fa_encoder.load_state_dict(torch.load(encoder_ckpt))
fa_decoder.load_state_dict(torch.load(decoder_ckpt))
fa_encoder.eval()
fa_decoder.eval()
if device == 'cuda':
    fa_encoder = fa_encoder.to(device)
    fa_decoder = fa_decoder.to(device)

if SCRIPT_LOCATION == "server":
    wav_dir = '/u/yurii/Projects/datasets/LJSpeech-1.1/wavs'
else:
    wav_dir = '/home/yurii/Projects/AC/ljspeech/LJSpeech-1.1/wavs'

all_wavs = glob.glob(os.path.join(wav_dir, '*.wav'))
print(f"Found {len(all_wavs)} wav files.")

random.seed(42)
train_files, test_files = train_test_split(all_wavs, test_size=0.1, random_state=42)
print(f"Train files: {len(train_files)}, Test files: {len(test_files)}")

output_dir = Config.facodec_dataset_dir
train_out = os.path.join(output_dir, 'train')
test_out = os.path.join(output_dir, 'test')
os.makedirs(train_out, exist_ok=True)
os.makedirs(test_out, exist_ok=True)

if __name__ == "__main__":
    # Initialize accumulators for content and prosody
    global_sum_content = 0.0
    global_sumsq_content = 0.0
    global_count_content = 0
    global_sum_prosody = 0.0
    global_sumsq_prosody = 0.0
    global_count_prosody = 0

    print("Processing train set...")
    for f in tqdm.tqdm(train_files, desc="Processing train files"):
        try:
            (fp, status, std_content, std_prosody,
             sum_content, sumsq_content, count_content,
             sum_prosody, sumsq_prosody, count_prosody) = process_wav_facodec(f, fa_encoder, fa_decoder, train_out, device)
            print(f"{fp}: {status}")
            if "success" in status:
                global_sum_content   += sum_content
                global_sumsq_content += sumsq_content
                global_count_content += count_content
                global_sum_prosody   += sum_prosody
                global_sumsq_prosody += sumsq_prosody
                global_count_prosody += count_prosody
        except Exception as e:
            print(f"Error processing {f}: {e}")

    print("Processing test set...")
    for f in tqdm.tqdm(test_files, desc="Processing test files"):
        try:
            fp, status, _, _, _, _, _, _, _, _ = process_wav_facodec(f, fa_encoder, fa_decoder, test_out, device)
            print(f"{fp}: {status}")
        except Exception as e:
            print(f"Error processing {f}: {e}")

    try:
        # Compute global standard deviations dynamically
        if global_count_content > 1 and global_count_prosody > 1:
            var_content = (global_sumsq_content - (global_sum_content ** 2) / global_count_content) / (global_count_content - 1)
            var_prosody = (global_sumsq_prosody - (global_sum_prosody ** 2) / global_count_prosody) / (global_count_prosody - 1)
            global_std_content = var_content.sqrt()
            global_std_prosody = var_prosody.sqrt()
            # Save stats in a new sub-directory "stats"
            stats_dir = os.path.join(output_dir, "stats")
            os.makedirs(stats_dir, exist_ok=True)
            torch.save(global_std_content, os.path.join(stats_dir, "std_content.pt"))
            torch.save(global_std_prosody, os.path.join(stats_dir, "std_prosody.pt"))
            print(f"Saved global std stats to {stats_dir}")
        else:
            print("Insufficient data to compute global stats.")
    except KeyboardInterrupt:
        print("Interrupted by user. Computing stats with data processed so far...")