import os
import glob
import random
from sklearn.model_selection import train_test_split
from FACodec_AC.utils import process_files
from huggingface_hub import hf_hub_download
import torch
import torchaudio

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Amphion'))

# Imports for FACodec
from models.codec.ns3_codec import FACodecEncoder, FACodecDecoder

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

wav_dir = '/home/yurii/Projects/AC/ljspeech/LJSpeech-1.1/wavs'
all_wavs = glob.glob(os.path.join(wav_dir, '*.wav'))
print(f"Found {len(all_wavs)} wav files.")

random.seed(42)
train_files, test_files = train_test_split(all_wavs, test_size=0.1, random_state=42)
print(f"Train files: {len(train_files)}, Test files: {len(test_files)}")

output_dir = '/home/yurii/Projects/AC/ljspeech/zc1_dataset'
train_out = os.path.join(output_dir, 'train')
test_out = os.path.join(output_dir, 'test')
os.makedirs(train_out, exist_ok=True)
os.makedirs(test_out, exist_ok=True)

if __name__ == "__main__":
    parallel = False
    print("Processing train set...")
    process_files(train_files, fa_encoder, fa_decoder, train_out, device, workers=4, sequential=not parallel)
    print("Processing test set...")
    process_files(test_files, fa_encoder, fa_decoder, test_out, device, workers=4, sequential=not parallel)