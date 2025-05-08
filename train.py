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
    
    # Define learnable log‑variance parameters for each loss term
    log_var_zc1    = torch.nn.Parameter(torch.zeros(1, device=Config.device))
    log_var_zc2    = torch.nn.Parameter(torch.zeros(1, device=Config.device))
    log_var_prosody= torch.nn.Parameter(torch.zeros(1, device=Config.device))
    log_var_acoustic = torch.nn.Parameter(torch.zeros(1, device=Config.device))
    
    # Update the optimizer to include the log‑variance parameters
    optimizer = optim.Adam(list(model.parameters()) + [
        log_var_zc1, log_var_zc2, log_var_prosody, log_var_acoustic
    ], lr=Config.lr)
    
    model.to(Config.device)
    model.train()

    writer = SummaryWriter(log_dir=Config.tensorboard_dir)
    best_eval_loss = float('inf')
    
    for epoch in range(Config.epochs):
        total_loss = 0.0
        total_loss_zc1 = 0.0
        total_loss_zc2 = 0.0
        total_loss_prosody = 0.0
        total_loss_acoustic = 0.0
        num_batches = 0

        # --- Training ---
        for zc1, zc2, padding_mask, padded_phone_ids, prosody, acoustic in dataloader_train:
            optimizer.zero_grad()
            x0 = zc1.to(Config.device)
            zc2 = zc2.to(Config.device)
            prosody = prosody.to(Config.device)
            acoustic = acoustic.to(Config.device)
            padding_mask = padding_mask.to(Config.device)
            padded_phone_ids = padded_phone_ids.to(Config.device)
            bsz, feature_dim, seq_len = x0.shape

            noise_raw = random.uniform(Config.NOISE_MIN, Config.NOISE_MAX)
            noise_norm = (noise_raw - Config.NOISE_MIN) / (Config.NOISE_MAX - Config.NOISE_MIN)
            noise_scaled = torch.full((bsz, seq_len, feature_dim), noise_norm, device=Config.device, dtype=torch.float)

            input_noise = torch.randn_like(x0) * (noise_raw * model.precomputed_std.view(1, -1, 1))
            x_noisy = x0 + input_noise

            # Forward pass with teacher forcing
            zc1_pred, zc2_pred, prosody_pred, acoustic_pred = model(
                x=x_noisy,
                padded_phone_ids=padded_phone_ids,
                noise_scaled=noise_scaled,
                real_zc1=x0,
                real_zc2=zc2,
                real_prosody=prosody,
                padding_mask=padding_mask
            )

            loss_zc1 = F.mse_loss(zc1_pred.transpose(1,2), x0)
            loss_zc2 = F.mse_loss(zc2_pred.transpose(1,2), zc2)
            loss_prosody = F.mse_loss(prosody_pred.transpose(1,2), prosody)
            loss_acoustic = F.mse_loss(acoustic_pred.transpose(1,2), acoustic)
            # Use the learnable log‑variance weights
            loss = (torch.exp(-log_var_zc1)*loss_zc1 + log_var_zc1 +
                    torch.exp(-log_var_zc2)*loss_zc2 + log_var_zc2 +
                    torch.exp(-log_var_prosody)*loss_prosody + log_var_prosody +
                    torch.exp(-log_var_acoustic)*loss_acoustic + log_var_acoustic)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_loss_zc1 += loss_zc1.item()
            total_loss_zc2 += loss_zc2.item()
            total_loss_prosody += loss_prosody.item()
            total_loss_acoustic += loss_acoustic.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_loss_zc1 = total_loss_zc1 / max(num_batches, 1)
        avg_loss_zc2 = total_loss_zc2 / max(num_batches, 1)
        avg_loss_prosody = total_loss_prosody / max(num_batches, 1)
        avg_loss_acoustic = total_loss_acoustic / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{Config.epochs}, Loss={avg_loss:.4f}, zc1 Loss={avg_loss_zc1:.4f}, zc2 Loss={avg_loss_zc2:.4f}, Prosody Loss={avg_loss_prosody:.4f}, Acoustic Loss={avg_loss_acoustic:.4f}")
        writer.add_scalar("Loss/Train_zc1", avg_loss_zc1, epoch+1)
        writer.add_scalar("Loss/Train_zc2", avg_loss_zc2, epoch+1)
        writer.add_scalar("Loss/Train_Prosody", avg_loss_prosody, epoch+1)
        writer.add_scalar("Loss/Train_Acoustic", avg_loss_acoustic, epoch+1)

        # --- Evaluation ---
        if (epoch+1) % Config.eval_epochs == 0:
            model.eval()
            total_test_loss = 0.0
            total_test_loss_zc1 = 0.0
            total_test_loss_zc2 = 0.0
            total_test_loss_prosody = 0.0
            total_test_loss_acoustic = 0.0
            test_batches = 0
            with torch.no_grad():
                for zc1_val, zc2_val, padding_mask, test_phone_ids, prosody, acoustic in dataloader_test:
                    x0 = zc1_val.to(Config.device)
                    zc2_val = zc2_val.to(Config.device)
                    prosody = prosody.to(Config.device)
                    acoustic = acoustic.to(Config.device)
                    padding_mask = padding_mask.to(Config.device)
                    padded_phone_ids = test_phone_ids.to(Config.device)
                    bsz, feature_dim, seq_len = x0.shape
                    noise_raw = random.uniform(Config.NOISE_MIN, Config.NOISE_MAX)
                    noise_norm = (noise_raw - Config.NOISE_MIN) / (Config.NOISE_MAX - Config.NOISE_MIN)
                    noise_scaled = torch.full((bsz, seq_len, feature_dim), noise_norm, device=Config.device, dtype=torch.float)

                    input_noise = torch.randn_like(x0) * (noise_raw * model.precomputed_std.view(1, -1, 1))
                    x_noisy = x0 + input_noise

                    # Forward pass with teacher forcing
                    zc1_pred, zc2_pred, prosody_pred, acoustic_pred = model(
                        x=x_noisy,
                        padded_phone_ids=padded_phone_ids,
                        noise_scaled=noise_scaled,
                        real_zc1=x0,
                        real_zc2=zc2_val,
                        real_prosody=prosody,
                        padding_mask=padding_mask
                    )
                    loss_zc1 = F.mse_loss(zc1_pred.transpose(1,2), x0)
                    loss_zc2 = F.mse_loss(zc2_pred.transpose(1,2), zc2_val)
                    loss_prosody = F.mse_loss(prosody_pred.transpose(1,2), prosody)
                    loss_acoustic = F.mse_loss(acoustic_pred.transpose(1,2), acoustic)
                    # Compute combined evaluation loss using learned weights
                    loss_eval = (torch.exp(-log_var_zc1)*loss_zc1 + log_var_zc1 +
                                 torch.exp(-log_var_zc2)*loss_zc2 + log_var_zc2 +
                                 torch.exp(-log_var_prosody)*loss_prosody + log_var_prosody +
                                 torch.exp(-log_var_acoustic)*loss_acoustic + log_var_acoustic)
                    total_test_loss += loss_eval.item()
                    total_test_loss_zc1 += loss_zc1.item()
                    total_test_loss_zc2 += loss_zc2.item()
                    total_test_loss_prosody += loss_prosody.item()
                    total_test_loss_acoustic += loss_acoustic.item()
                    test_batches += 1

            avg_test_loss = total_test_loss / max(test_batches, 1)
            avg_test_loss_zc1 = total_test_loss_zc1 / max(test_batches, 1)
            avg_test_loss_zc2 = total_test_loss_zc2 / max(test_batches, 1)
            avg_test_loss_prosody = total_test_loss_prosody / max(test_batches, 1)
            avg_test_loss_acoustic = total_test_loss_acoustic / max(test_batches, 1)
            print(f"Epoch {epoch+1}/{Config.epochs}, Eval Loss={avg_test_loss:.4f}, zc1 Loss={avg_test_loss_zc1:.4f}, zc2 Loss={avg_test_loss_zc2:.4f}, Prosody Loss={avg_test_loss_prosody:.4f}, Acoustic Loss={avg_test_loss_acoustic:.4f}")
            writer.add_scalar("Loss/Eval_zc1", avg_test_loss_zc1, epoch+1)
            writer.add_scalar("Loss/Eval_zc2", avg_test_loss_zc2, epoch+1)
            writer.add_scalar("Loss/Eval_Prosody", avg_test_loss_prosody, epoch+1)
            writer.add_scalar("Loss/Eval_Acoustic", avg_test_loss_acoustic, epoch+1)

            if (epoch+1) % Config.checkpoint_epochs == 0:
                checkpoint_full_path = Config.checkpoint_path
                checkpoint_dir = os.path.dirname(checkpoint_full_path)
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_full_path)
                print(f"Checkpoint saved at {checkpoint_full_path} at epoch {epoch+1} with Eval Loss={avg_test_loss:.4f}")
            model.train()
    
    writer.close()

if __name__ == "__main__":
    main()