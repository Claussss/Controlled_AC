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
from FACodec_AC.utils import get_mask_positions
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
        os.path.join(Config.zc1_data_dir, 'train'),
        os.path.join(Config.phoneme_cond_dir, 'train')
    )
    test_dataset  = CodebookSequenceDataset(
        os.path.join(Config.zc1_data_dir, 'test'),
        os.path.join(Config.phoneme_cond_dir, 'test')
    )
    
    dataloader_train = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    dataloader_test  = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

    # Initialize the FACodecDecoder and load its pretrained weights
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
    )
    decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec",
                                   filename="ns3_facodec_decoder.bin")
    fa_decoder.load_state_dict(torch.load(decoder_ckpt))
    fa_decoder.eval()

    pretrained_codebook = fa_decoder.quantizer[1].layers[0].codebook
    pretrained_proj_layer = fa_decoder.quantizer[1].layers[0].out_proj
    
    # Initialize the transformer model with the pretrained components.
    model = DiffusionTransformerModel(
        pretrained_codebook=pretrained_codebook,
        pretrained_proj_layer=pretrained_proj_layer,
        std_file_path=os.path.join(Config.zc1_data_dir, 'stats', 'std.pt'),
        vocab_size=Config.VOCAB_SIZE,
        d_model=Config.d_model,
        nhead=Config.nhead,
        num_layers=Config.num_layers,
        d_ff=Config.d_ff,
        dropout=Config.dropout,
        max_seq_len=Config.max_seq_len
    )
    
    # Inline training loop (integrating the content of train_diffusion_model)
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
            x0 = batch.to(Config.device)             # discrete tokens
            padding_mask = padding_mask.to(Config.device)
            padded_phone_ids = padded_phone_ids.to(Config.device)
            prosody_cond = prosody_cond.to(Config.device)
            bsz, seq_len = x0.shape

            # Determine which positions to mask (true = mask, false = no mask)
            mask_positions = get_mask_positions(x0, r_range=Config.r_range, p_drop=Config.p_drop)

            # Generate noise scaled by the precomputed std:
            noise_level_value = random.uniform(Config.NOISE_MIN, Config.NOISE_MAX)
            noise_level_value_norm = (noise_level_value - Config.NOISE_MIN) / (Config.NOISE_MAX - Config.NOISE_MIN)
            noise_level = torch.full((bsz, 1), noise_level_value_norm, device=Config.device, dtype=torch.float)
            feature_dim = model.proj_to_256.out_features
            noise_scaled = torch.randn(bsz, seq_len, feature_dim, device=x0.device) * (noise_level_value * model.precomputed_std)

            # Forward pass returns (prediction, target_clean)
            pred, target = model(
                x=x0, 
                padded_phone_ids=padded_phone_ids, 
                noise_level=noise_level,
                mask_positions=mask_positions, 
                padding_mask=padding_mask, 
                noise_scaled=noise_scaled,
                prosody_cond=prosody_cond
            )

            # Use MSE loss between prediction and clean target representation
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
                    bsz, seq_len = x0.shape

                    mask_positions = get_mask_positions(x0, r_range=Config.r_range, p_drop=Config.p_drop)

                    noise_val = random.uniform(Config.NOISE_MIN, Config.NOISE_MAX)
                    noise_level = torch.full((bsz, 1),
                                             (noise_val - Config.NOISE_MIN) / (Config.NOISE_MAX - Config.NOISE_MIN),
                                             device=Config.device)
                    feat_dim = model.proj_to_256.out_features
                    noise_scaled = torch.randn(bsz, seq_len, feat_dim, device=Config.device) * (noise_val * model.precomputed_std)

                    pred, target = model(
                        x=x0, 
                        padded_phone_ids=padded_phone_ids, 
                        noise_level=noise_level,
                        mask_positions=mask_positions, 
                        padding_mask=padding_mask, 
                        noise_scaled=noise_scaled,
                        prosody_cond=prosody_cond
                    )
                    # Compute MSE loss on the continuous predictions.
                    loss_test = F.mse_loss(pred, target)
                    total_test_loss += loss_test.item()

                    test_batches += 1

            avg_test_loss = total_test_loss / max(test_batches, 1)
            print(f"Epoch {epoch+1}/{Config.epochs}, Eval Loss={avg_test_loss:.4f}")
            writer.add_scalar("Loss/Eval", avg_test_loss, epoch+1)

            # Save checkpoint if evaluation loss improves
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