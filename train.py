import os
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from FACodec_AC.config import Config
from huggingface_hub import hf_hub_download
import sys
from FACodec_AC.dataset import ZContentDataset, LengthSortedBatchSampler, collate_fn_zcontent
from FACodec_AC.utils import init_facodec_models
from FACodec_AC.models import DenoisingTransformerModel 

SCRIPT_LOCATION = os.environ.get("location")
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Amphion'))


def lr_lambda(step):
    warmup_steps = 1000
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0  # constant after warmup

def main():
    # Seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    # Initialize FACodec models.
    fa_encoder, fa_decoder = init_facodec_models(Config.device)
    # It is needed only in get_item in the Dataset for looking up indexes, and workers work on CPU anyway.
    fa_decoder = fa_decoder.to('cpu') 
    # Extract codebooks and projection layers from the content quantizer.
    codebook_zc1 = fa_decoder.quantizer[1].layers[0].codebook.weight
    out_proj_zc1 = fa_decoder.quantizer[1].layers[0].out_proj
    codebook_zc2 = fa_decoder.quantizer[1].layers[1].codebook.weight
    out_proj_zc2 = fa_decoder.quantizer[1].layers[1].out_proj

    # Use ZContentDataset instead of CodebookSequenceDataset.
    train_dataset = ZContentDataset(
        os.path.join(Config.facodec_dataset_dir, 'train'),
        os.path.join(Config.phoneme_cond_dir,'train'),
        codebook_zc1, out_proj_zc1,
        codebook_zc2, out_proj_zc2,
        facodec_dim=Config.FACodec_dim
    )
    test_dataset  = ZContentDataset(
        os.path.join(Config.facodec_dataset_dir, 'test'),
        os.path.join(Config.phoneme_cond_dir,'test'),
        codebook_zc1, out_proj_zc1,
        codebook_zc2, out_proj_zc2,
        facodec_dim=Config.FACodec_dim
    )
    
    # Use LengthSortedBatchSampler for bucketing.
    train_batch_sampler = LengthSortedBatchSampler(train_dataset, batch_size=Config.batch_size, drop_last=False)
    test_batch_sampler  = LengthSortedBatchSampler(test_dataset, batch_size=Config.batch_size, drop_last=False)
    
    dataloader_train = DataLoader(train_dataset, batch_sampler=train_batch_sampler, collate_fn=collate_fn_zcontent)
    dataloader_test  = DataLoader(test_dataset, batch_sampler=test_batch_sampler, collate_fn=collate_fn_zcontent)
    
    # Initialize the transformer model.
    model = DenoisingTransformerModel(
        d_model=Config.d_model,
        nhead=Config.nhead,
        num_layers=Config.num_layers,
        d_ff=Config.d_ff,
        dropout=Config.dropout,
        max_seq_len=Config.max_seq_len,
        FACodec_dim=Config.FACodec_dim,
        phone_vocab_size=Config.PHONE_VOCAB_SIZE
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    model.to(Config.device)

    # -- Initialize from Checkpoint if available --
    if os.path.exists(Config.checkpoint_path):
        print(f"Loading checkpoint from {Config.checkpoint_path}")
        checkpoint = torch.load(Config.checkpoint_path)
        model.load_state_dict(checkpoint)
    model.train()
    
    writer = SummaryWriter(log_dir=Config.tensorboard_dir)
    best_eval_loss = float('inf')
    
    for epoch in range(Config.epochs):
        total_loss = 0.0
        total_loss_zc1 = 0.0
        total_loss_zc2 = 0.0
        num_batches = 0

        # --- Training ---
        for zc1, zc2, phone_cond, mask in dataloader_train:
            optimizer.zero_grad()
            x0 = zc1.to(Config.device)
            zc2 = zc2.to(Config.device)
            padded_phone_ids = phone_cond.to(Config.device)
            padding_mask = mask.to(Config.device)
            bsz, feature_dim, seq_len = x0.shape
            
            noise_raw = random.uniform(Config.NOISE_MIN, Config.NOISE_MAX)
            noise_norm = (noise_raw - Config.NOISE_MIN) / (Config.NOISE_MAX - Config.NOISE_MIN)
            noise_scaled = torch.full((bsz, seq_len, feature_dim), noise_norm, device=Config.device, dtype=torch.float)
            
            input_noise = noise_raw * torch.randn_like(x0)
            x_noisy = x0 + input_noise
            
            zc1_pred, zc2_pred = model(
                zc1_noisy=x_noisy,
                zc1_ground_truth=x0,
                padded_phone_ids=padded_phone_ids,
                noise_scaled=noise_scaled,
                padding_mask=padding_mask
            )
            
            loss_zc1 = F.mse_loss(zc1_pred, x0)
            loss_zc2 = F.mse_loss(zc2_pred, zc2)
            loss = loss_zc1 + loss_zc2
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            total_loss_zc1 += loss_zc1.item()
            total_loss_zc2 += loss_zc2.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_loss_zc1 = total_loss_zc1 / max(num_batches, 1)
        avg_loss_zc2 = total_loss_zc2 / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{Config.epochs}, Loss={avg_loss:.4f}, zc1 Loss={avg_loss_zc1:.4f}, zc2 Loss={avg_loss_zc2:.4f}")
        writer.add_scalar("Loss/Train_zc1", avg_loss_zc1, epoch+1)
        writer.add_scalar("Loss/Train_zc2", avg_loss_zc2, epoch+1)

        # --- Evaluation ---
        if (epoch+1) % Config.eval_epochs == 0:
            model.eval()
            total_test_loss = 0.0
            total_test_loss_zc1 = 0.0
            total_test_loss_zc2 = 0.0
            test_batches = 0
            with torch.no_grad():
                for zc1_val, zc2_val, test_phone_ids, mask in dataloader_test:
                    x0 = zc1_val.to(Config.device)
                    padding_mask = mask.to(Config.device)
                    padded_phone_ids = test_phone_ids.to(Config.device)
                    zc2_val = zc2_val.to(Config.device)
                    bsz, feature_dim, seq_len = x0.shape
                    noise_raw = random.uniform(Config.NOISE_MIN, Config.NOISE_MAX)
                    noise_norm = (noise_raw - Config.NOISE_MIN) / (Config.NOISE_MAX - Config.NOISE_MIN)
                    noise_scaled = torch.full((bsz, seq_len, feature_dim), noise_norm, device=Config.device, dtype=torch.float)

                    input_noise = noise_raw * torch.randn_like(x0)
                    x_noisy = x0 + input_noise

                    zc1_pred, zc2_pred = model(
                        zc1_noisy=x_noisy,
                        zc1_ground_truth=x0,
                        padded_phone_ids=padded_phone_ids,
                        noise_scaled=noise_scaled,
                        padding_mask=padding_mask
                    )
                    loss_zc1 = F.mse_loss(zc1_pred, x0)
                    loss_zc2 = F.mse_loss(zc2_pred, zc2_val)
                    total_test_loss += (loss_zc1.item() + loss_zc2.item())
                    total_test_loss_zc1 += loss_zc1.item()
                    total_test_loss_zc2 += loss_zc2.item()
                    test_batches += 1

            avg_test_loss = total_test_loss / max(test_batches, 1)
            avg_test_loss_zc1 = total_test_loss_zc1 / max(test_batches, 1)
            avg_test_loss_zc2 = total_test_loss_zc2 / max(test_batches, 1)
            print(f"Epoch {epoch+1}/{Config.epochs}, Eval Loss={avg_test_loss:.4f}, zc1 Loss={avg_test_loss_zc1:.4f}, zc2 Loss={avg_test_loss_zc2:.4f}")
            writer.add_scalar("Loss/Eval_zc1", avg_test_loss_zc1, epoch+1)
            writer.add_scalar("Loss/Eval_zc2", avg_test_loss_zc2, epoch+1)

            # Save checkpoint only if current eval loss is lower than best so far and epoch has passed Config.checkpoint_epochs.
            if (epoch+1) >= Config.checkpoint_epochs and avg_test_loss < best_eval_loss:
                best_eval_loss = avg_test_loss
                checkpoint_full_path = Config.checkpoint_path
                checkpoint_dir = os.path.dirname(checkpoint_full_path)
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_full_path)
                print(f"Epoch {epoch+1}: New best eval loss: {avg_test_loss:.4f}. Checkpoint saved at {checkpoint_full_path}")
            model.train()
    
    writer.close()

if __name__ == "__main__":
    main()