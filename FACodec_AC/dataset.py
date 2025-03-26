import os
import glob
import torch
from torch.utils.data import Dataset
from FACodec_AC.config import Config


class CodebookSequenceDataset(Dataset):
    """
    Loads .pt files (containing {'tokens': ..., 'mask': ...}) from a directory.
    """
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, "*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        # data['tokens'] shape: (time_frames,)
        # data['mask']   shape: (time_frames,)
        return data['tokens'], data['mask']
