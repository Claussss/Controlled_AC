import os
import random
import torch
from torch.utils.data import DataLoader
from FACodec_AC.dataset import CodebookSequenceDataset, pad_collate
from FACodec_AC.models import DiffusionTransformerModel, train_diffusion_model
from FACodec_AC.config import Config

def main():
    # Seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    # Create train and test datasets/dataloaders
    train_dataset = CodebookSequenceDataset(os.path.join(Config.data_dir, 'train'))
    test_dataset  = CodebookSequenceDataset(os.path.join(Config.data_dir, 'test'))
    
    dataloader_train = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, collate_fn=pad_collate)
    dataloader_test  = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, collate_fn=pad_collate)
    
    # Initialize the model.
    model = DiffusionTransformerModel(
        vocab_size=Config.vocab_size,
        d_model=Config.d_model,
        nhead=Config.nhead,
        num_layers=Config.num_layers,
        d_ff=Config.d_ff,
        dropout=Config.dropout,
        max_seq_len=Config.time_frames
    )
    
    train_diffusion_model(
        model,
        dataloader_train,
        dataloader_test,
        T=Config.T,
        epochs=Config.epochs,
        lr=Config.lr,
        device=Config.device,
        eval_epochs=Config.eval_epochs
    )

if __name__ == "__main__":
    main()