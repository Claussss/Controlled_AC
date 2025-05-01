import os
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from FACodec_AC.dataset import CodebookSequenceDataset
from FACodec_AC.models import DiffusionTransformerModel
from FACodec_AC.config import Config
from huggingface_hub import hf_hub_download
import sys

SCRIPT_LOCATION = os.environ.get("location")

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Amphion'))
from models.codec.ns3_codec import FACodecDecoder

def main():
    # Seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    # Create train and test datasets/dataloaders
    train_dataset = CodebookSequenceDataset(
        os.path.join(Config.facodec_dataset_dir, 'train'),
        os.path.join(Config.phoneme_cond_dir, 'train')
    )
    test_dataset  = CodebookSequenceDataset(
        os.path.join(Config.facodec_dataset_dir, 'test'),
        os.path.join(Config.phoneme_cond_dir, 'test')
    )
    
    dataloader_train = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    dataloader_test  = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)
    
    # Initialize the transformer model with the pretrained components.
    model = DiffusionTransformerModel(
        std_file_path=Config.std_content_path,
        d_model=Config.d_model,
        nhead=Config.nhead,
        num_layers=Config.num_layers,
        d_ff=Config.d_ff,
        dropout=Config.dropout,
        max_seq_len=Config.max_seq_len
    )
    
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    model.to(Config.device)
    model.train()

    writer = SummaryWriter(log_dir=Config.tensorboard_dir)
    best_eval_loss = float('inf')

    for epoch in range(Config.epochs):
        total_loss = 0.0
        num_batches = 0

        # --- Training ---
        for batch, padding_mask, padded_phone_ids, prosody_cond in dataloader_train:
            optimizer.zero_grad()
            x0 = batch.to(Config.device)  # clean continuous input, shape: [B, T, 256]
            padding_mask = padding_mask.to(Config.device)
            padded_phone_ids = padded_phone_ids.to(Config.device)
            prosody_cond = prosody_cond.to(Config.device)
            bsz, feature_dim, seq_len = x0.shape

            # Generate a raw noise scalar
            noise_raw = random.uniform(Config.NOISE_MIN, Config.NOISE_MAX)

            # Normalize the noise level (noise_scaled for conditioning)
            noise_norm = (noise_raw - Config.NOISE_MIN) / (Config.NOISE_MAX - Config.NOISE_MIN)
            # Expand noise_norm to a tensor shape [B, T, 256] for conditioning
            noise_scaled = torch.full((bsz, seq_len, feature_dim), noise_norm, device=Config.device, dtype=torch.float)

            # Generate input noise scaled by precomputed_std
            input_noise = torch.randn_like(x0) * (noise_raw * model.precomputed_std)
            # Create noisy input by adding the scaled noise to the clean input
            x_noisy = x0 + input_noise

            # Forward pass: pass the noisy input with the conditioning noise
            pred = model(
                x=x_noisy,
                padded_phone_ids=padded_phone_ids,
                noise_scaled=noise_scaled,
                padding_mask=padding_mask,
                prosody_cond=prosody_cond
            )

            # Target remains the clean input x0
            target = x0.reshape(bsz, seq_len, feature_dim)
            loss = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{Config.epochs}, Loss={avg_loss:.4f}")
        writer.add_scalar("Loss/Train", avg_loss, epoch+1)

        # --- Evaluation ---
        if (epoch+1) % Config.eval_epochs == 0:
            model.eval()
            total_test_loss = 0.0
            test_batches = 0
            with torch.no_grad():
                for test_batch, padding_mask, test_phone_ids, prosody_cond_test in dataloader_test:
                    x0 = test_batch.to(Config.device)
                    padding_mask = padding_mask.to(Config.device)
                    padded_phone_ids = test_phone_ids.to(Config.device)
                    prosody_cond = prosody_cond_test.to(Config.device)
                    bsz, feature_dim, seq_len = x0.shape
                    noise_raw = random.uniform(Config.NOISE_MIN, Config.NOISE_MAX)
                    noise_norm = (noise_raw - Config.NOISE_MIN) / (Config.NOISE_MAX - Config.NOISE_MIN)
                    noise_scaled = torch.full((bsz, seq_len, feature_dim), noise_norm, device=Config.device, dtype=torch.float)

                    input_noise = torch.randn_like(x0) * (noise_raw * model.precomputed_std)
                    x_noisy = x0 + input_noise

                    pred = model(
                        x=x_noisy,
                        padded_phone_ids=padded_phone_ids,
                        noise_scaled=noise_scaled,
                        padding_mask=padding_mask,
                        prosody_cond=prosody_cond
                    )
                    target = x0
                    loss_test = F.mse_loss(pred, target)
                    total_test_loss += loss_test.item()
                    test_batches += 1

            avg_test_loss = total_test_loss / max(test_batches, 1)
            print(f"Epoch {epoch+1}/{Config.epochs}, Eval Loss={avg_test_loss:.4f}")
            writer.add_scalar("Loss/Eval", avg_test_loss, epoch+1)

            if avg_test_loss < best_eval_loss:
                checkpoint_full_path = Config.checkpoint_path
                checkpoint_dir = os.path.dirname(checkpoint_full_path)
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_full_path)
                print(f"Checkpoint saved at {checkpoint_full_path} at epoch {epoch+1} with Eval Loss={avg_test_loss:.4f}")
                best_eval_loss = avg_test_loss
            model.train()
    
    writer.close()

if __name__ == "__main__":
    main()