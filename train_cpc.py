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

    model = CPC(dim=256, hidden=128, steps=3)
    model.to(Config.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    writer = SummaryWriter(log_dir=Config.tensorboard_dir)  # initialize SummaryWriter
    epochs = Config.epochs  # reusing the epochs parameter from config
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for latent, mask in dataloader:
            # latent: (B, 256, T) --> transpose to (B, T, 256)
            latent = latent.to(Config.device)
            mask = mask.to(Config.device)
            x = latent.transpose(1, 2)
            
            # Forward pass through CPC model
            ctx = model(x, mask)
            loss = cpc_loss(ctx, x, mask, model.steps, model, temperature=0.07)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/CPC_Train", avg_loss, epoch+1)  # log train loss
        if (epoch+1) % Config.checkpoint_epochs == 0:
            checkpoint_full_path = Config.checkpoint_path
            checkpoint_dir = os.path.dirname(checkpoint_full_path)
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_full_path)
            print(f"Checkpoint saved at {checkpoint_full_path} at epoch {epoch+1}")

    writer.close()  # close the writer

if __name__ == "__main__":
    main()
