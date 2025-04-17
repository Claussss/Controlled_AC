import os
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Append FACodec_AC and Amphion directories to sys.path.
#sys.path.append(os.path.join(os.path.dirname(__file__), 'FACodec_AC'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Amphion'))

from FACodec_AC.dataset import Zc1DatasetASR
from FACodec_AC.models import ProjectionHead, ASRModel
from FACodec_AC.utils  import collate_fn, get_zc1_from_indx
from FACodec_AC.config import ASRConfig

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)
torch.manual_seed(42)

# Load processor for tokenization, etc.
from transformers import Wav2Vec2Processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
pad_token_id = processor.tokenizer.pad_token_id

# Load the FACodec decoder (from Amphion).
from models.codec.ns3_codec import FACodecDecoder
from huggingface_hub import hf_hub_download
fa_decoder = FACodecDecoder(
    in_channels=256,
    upsample_initial_channel=1024,
    ngf=32,
    up_ratios=[5, 5, 4, 2],
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
).to(device)
decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder.bin")
fa_decoder.load_state_dict(torch.load(decoder_ckpt, map_location=device))
fa_decoder.eval()
for param in fa_decoder.phone_predictor.parameters():
    param.requires_grad = False

# Create Projection Head and wrap into ASRModel.
proj_head = ProjectionHead(in_features=5003, out_features=392).to(device)
asr_model = ASRModel(fa_decoder, proj_head).to(device)

# Only the projection head parameters are trainable.
optimizer = optim.Adam(proj_head.parameters(), lr=ASRConfig.lr)

# Create dataset and dataloader.
dataset = Zc1DatasetASR(
    zc1_dir=ASRConfig.zc1_dir,
    metadata_path=ASRConfig.metadata_path,
    processor=processor
)
dataloader = DataLoader(dataset, batch_size=ASRConfig.batch_size, shuffle=True,
                        collate_fn=lambda batch: collate_fn(batch, pad_token_id))

# Define CTC loss.
ctc_loss_fn = nn.CTCLoss(blank=pad_token_id, reduction="mean", zero_infinity=True)

# TensorBoard writer.
writer = SummaryWriter(log_dir=ASRConfig.tensorboard_dir)

global_step = 0
num_epochs = ASRConfig.num_epochs
for epoch in range(num_epochs):
    asr_model.train()
    running_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        latent_tokens = batch["latent_tokens"].to(device)        # [B, T]
        latent_mask = batch["latent_mask"].to(device).bool()        # [B, T]
        bsz = latent_tokens.size(0)
        # Convert latent tokens to continuous zc1 representations.
        zc1 = get_zc1_from_indx(latent_tokens, latent_mask, fa_decoder)  # [B, T, 256]
        # Forward pass through ASR model.
        proj_logits = asr_model(zc1)  # [B, T, 392]
        log_probs = proj_logits.log_softmax(dim=-1).transpose(0, 1)  # [T, B, 392]
        #input_lengths = torch.full((bsz,), log_probs.size(0), dtype=torch.long, device=device)
        #targets = batch["target_ids"].to(device)
        #target_lengths = (targets != pad_token_id).sum(dim=1)
        loss = ctc_loss_fn(log_probs, 
                            batch["targets"].to(device),
                            batch["input_lengths"].to(device),
                            batch["target_lengths"].to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        writer.add_scalar("Loss/train_step", loss.item(), global_step)
        global_step += 1
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs} Loss: {avg_loss:.4f}")
    writer.add_scalar("Loss/epoch", avg_loss, epoch)

# Save the projection head weights (ASR model mapping).
torch.save(proj_head.state_dict(), os.path.join(ASRConfig.checkpoint_path,"asr_projection_head.pt"))
writer.close()
print(f"Training complete. Weights saved to {os.path.join(ASRConfig.checkpoint_path,'asr_projection_head.pt')}")