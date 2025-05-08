import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from FACodec_AC.dataset import CPCDataset
from FACodec_AC.models import CPC
from FACodec_AC.losses import cpc_loss
from FACodec_AC.config import Config
from torch.utils.tensorboard import SummaryWriter  # added import

def main():
    # Use the same dataset directory as facodec data
    dataset = CPCDataset(Config.facodec_dataset_dir+'/train')
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    crop_len = 120 
    model = CPC(dim=256, hidden=128, steps=3)
    model.to(Config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9,0.98))
    sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1,
                                          total_iters=1000) 

    model.train()
    writer = SummaryWriter(log_dir=Config.tensorboard_dir)  # initialize SummaryWriter
    epochs = Config.epochs  # reusing the epochs parameter from config
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches  = 0

        for latent, mask in dataloader:
            # latent: (B, 256, T)    mask: (B, T)  (True = padding)
            latent = latent.to(Config.device)
            mask   = mask.to(Config.device)
            x      = latent.transpose(1, 2).contiguous()      # (B, T, 256)

            B, T, _ = x.size()
            x_crops     = []
            mask_crops  = []

            # ---- random crop that is guaranteed to hit at least one real frame ----
            for i in range(B):
                valid_mask  = ~mask[i]                         # True = real frame
                valid_idx   = valid_mask.nonzero(as_tuple=False).squeeze(-1)

                if valid_idx.numel() == 0:                     # fully padded (rare)
                    start = 0
                else:
                    pivot = valid_idx[torch.randint(0, valid_idx.numel(), (1,))].item()
                    start = max(0, min(pivot - crop_len // 2, T - crop_len))

                x_crops.append(   x[i,    start:start+crop_len, :] )
                mask_crops.append(mask[i, start:start+crop_len   ])

            x_crop    = torch.stack(x_crops,    dim=0)         # (B, 120, 256)
            mask_crop = torch.stack(mask_crops, dim=0)         # (B, 120)

            # -----------------------------------------------------------------------

            ctx  = model(x_crop, mask_crop)
            loss = cpc_loss(ctx, x_crop, mask_crop,
                            model.steps, model, temperature=0.07)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sched.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/CPC_Train", avg_loss, epoch+1)

        if (epoch + 1) % 1 == 0:
            ckpt_path = Config.checkpoint_path
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved to {ckpt_path} at epoch {epoch+1}")

        writer.close()  # close the writer

if __name__ == "__main__":
    main()
