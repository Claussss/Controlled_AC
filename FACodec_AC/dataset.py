import os
import glob
import torch
from torch.utils.data import Dataset
from FACodec_AC.config import Config
import random


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

class ClassificationDataset(Dataset):
    """
    Dataset for classification tasks using .pt files from two classes: 'foreign' and 'native'.
    The directory structure should be:
    
        data_dir/
            foreign/
                *.pt
            native/
                *.pt

    Each .pt file should contain a dict with keys 'tokens' and 'mask'.
    The dataset returns (tokens, mask, label) where label is:
        1 for foreign
        0 for native

    The dataset is balanced by sampling an equal number of files from each class.
    """
    def __init__(self, data_dir_native, data_dir_foreign):
        # Get file lists for each class.
        foreign_files = glob.glob(os.path.join(data_dir_foreign, "*.pt"))
        native_files = glob.glob(os.path.join(data_dir_native, "*.pt"))
        
        # Balance the dataset by sampling the same number of files from each class.
        min_count = min(len(foreign_files), len(native_files))
        if len(foreign_files) > min_count:
            foreign_files = random.sample(foreign_files, min_count)
        if len(native_files) > min_count:
            native_files = random.sample(native_files, min_count)
        
        self.samples = []
        # Add foreign samples with label 1
        for file in foreign_files:
            self.samples.append((file, 1))
        # Add native samples with label 0
        for file in native_files:
            self.samples.append((file, 0))
        
        # Shuffle the combined dataset.
        random.shuffle(self.samples)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        data = torch.load(file_path)
        tokens = data['tokens']
        mask = data['mask']
        return tokens, mask, label