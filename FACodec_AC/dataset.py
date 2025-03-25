import os
import glob
import torch
from torch.utils.data import Dataset
from FACodec_AC.config import Config


class CodebookSequenceDataset(Dataset):
    """
    Loads .pt files from a directory.
    Each file is shaped (1, time_frames) containing integer IDs.
    """
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, "*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        x = torch.load(path)  # shape: (1, time_frames)
        return x.squeeze(0)   # shape: (time_frames,)

def pad_collate(batch):
    lengths = [x.size(0) for x in batch]
    max_len = max(lengths)
    # Use PAD_ID for padding.
    padded_batch = torch.full((len(batch), max_len), Config.PAD_ID, dtype=torch.long)
    # Padding mask: True for padded positions.
    padding_mask = torch.ones((len(batch), max_len), dtype=torch.bool)
    for i, x in enumerate(batch):
        padded_batch[i, :x.size(0)] = x
        padding_mask[i, :x.size(0)] = False
    return padded_batch, padding_mask