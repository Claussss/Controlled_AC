import os
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from FACodec_AC.dataset import FACodecProsodyDataset
from FACodec_AC.models import DiffusionTransformerModel
from FACodec_AC.config import Config, PitchConfig

def main():
    # Seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # Create train and test datasets/dataloaders
    train_dataset = FACodecProsodyDataset(
        os.path.join(Config.facodec_dataset_dir, 'train'),
        os.path.join(PitchConfig.pitch_cond_dir, 'train')
    )
    test_dataset = FACodecProsodyDataset(
        os.path.join(Config.facodec_dataset_dir, 'test'),
        os.path.join(PitchConfig.pitch_cond_dir, 'test')
    )

    dataloader_train = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    dataloader_test = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

    # Initialize the transformer model with the pretrained components.
    model = DiffusionTransformerModel(
        std_file_path=PitchConfig.std_path,
        d_model=Config.d_model,
        nhead=Config.nhead,
        num_layers=Config.num_layers,
        d_ff=Config.d_ff,
        dropout=Config.dropout,
        max_seq_len=Config.max_seq_len,
        prosody_model=True
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
        for prosody, padding_mask, pitch_cond in dataloader_train:
            optimizer.zero_grad()
            x0 = prosody.to(Config.device)
            padding_mask = padding_mask.to(Config.device)
            padded_pitch_ids = pitch_cond.to(Config.device)

            bsz, feature_dim, seq_len = x0.shape

            noise_raw = random.uniform(PitchConfig.NOISE_MIN, PitchConfig.NOISE_MAX)
            noise_norm = (noise_raw - PitchConfig.NOISE_MIN) / (PitchConfig.NOISE_MAX - PitchConfig.NOISE_MIN)
            noise_scaled = torch.full((bsz, seq_len, feature_dim), noise_norm, device=Config.device, dtype=torch.float)

            input_noise = torch.randn_like(x0) * (noise_raw * model.precomputed_std.view(1, -1, 1))
            x_noisy = x0 + input_noise

            # Forward pass
            prosody_pred = model(
                x=x_noisy,
                padded_global_cond_ids=padded_pitch_ids,
                noise_scaled=noise_scaled,
                padding_mask=padding_mask,
                prosody_cond=None
            )

            loss = F.mse_loss(prosody_pred.transpose(1, 2), x0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1}/{Config.epochs}, Loss={avg_loss:.4f}")
        writer.add_scalar("Loss/Train", avg_loss, epoch + 1)

        # --- Evaluation ---
        if (epoch + 1) % Config.eval_epochs == 0:
            model.eval()
            total_test_loss = 0.0
            test_batches = 0
            with torch.no_grad():
                for prosody_val, padding_mask, pitch_cond_test in dataloader_test:
                    x0 = prosody_val.to(Config.device)
                    padding_mask = padding_mask.to(Config.device)
                    padded_pitch_ids = pitch_cond_test.to(Config.device)

                    bsz, feature_dim, seq_len = x0.shape
                    noise_raw = random.uniform(PitchConfig.NOISE_MIN, PitchConfig.NOISE_MAX)
                    noise_norm = (noise_raw - PitchConfig.NOISE_MIN) / (PitchConfig.NOISE_MAX - PitchConfig.NOISE_MIN)
                    noise_scaled = torch.full((bsz, seq_len, feature_dim), noise_norm, device=Config.device, dtype=torch.float)

                    input_noise = torch.randn_like(x0) * (noise_raw * model.precomputed_std.view(1, -1, 1))
                    x_noisy = x0 + input_noise

                    prosody_pred = model(
                        x=x_noisy,
                        padded_global_cond_ids=padded_pitch_ids,
                        noise_scaled=noise_scaled,
                        padding_mask=padding_mask,
                        prosody_cond=None
                    )
                    loss = F.mse_loss(prosody_pred.transpose(1, 2), x0)
                    total_test_loss += loss.item()
                    test_batches += 1

            avg_test_loss = total_test_loss / max(test_batches, 1)
            print(f"Epoch {epoch + 1}/{Config.epochs}, Eval Loss={avg_test_loss:.4f}")
            writer.add_scalar("Loss/Eval", avg_test_loss, epoch + 1)

            if (epoch + 1) % Config.checkpoint_epochs == 0:
                checkpoint_full_path = Config.checkpoint_path
                checkpoint_dir = os.path.dirname(checkpoint_full_path)
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_full_path)
                print(f"Checkpoint saved at {checkpoint_full_path} at epoch {epoch + 1} with Eval Loss={avg_test_loss:.4f}")
                best_eval_loss = avg_test_loss
            model.train()

    writer.close()

if __name__ == "__main__":
    main()