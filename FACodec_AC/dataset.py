import os
import glob
import torch
from torch.utils.data import Dataset
from FACodec_AC.config import Config
import random


class FACodecContentDataset(Dataset):
    """
    Loads .pt files (containing {'tokens': ..., 'mask': ...}) from a directory.
    """
    def __init__(self, data_dir, cond_dir=None):
        # raise an error if data_dir does not exist
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory {data_dir} does not exist.")
        # if wav2vec condition directory is provided, check that it exists
        if cond_dir is not None and not os.path.isdir(cond_dir):
            raise FileNotFoundError(f"Condition directory {cond_dir} does not exist.")
        
        self.files = glob.glob(os.path.join(data_dir, "*.pt"))
        self.cond_dir = cond_dir  # store wav2vec condition directory if provided
        if cond_dir is not None:
            wav2vec_files = glob.glob(os.path.join(cond_dir, "*.pt"))
            wav2vec_file_names = {os.path.basename(f) for f in wav2vec_files}
            # Filter self.files to only include files present in wav2vec_cond_dir.
            self.files = [f for f in self.files if os.path.basename(f) in wav2vec_file_names]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        
        # load the corresponding wav2vec conditioning file with the same basename
        phone_cond_path = os.path.join(self.cond_dir, os.path.basename(self.files[idx]))
        phone_cond_data = torch.load(phone_cond_path)
        return data['zc1'], data['zc2'], data['mask'], phone_cond_data, data['prosody']
    

class FACodecProsodyDataset(Dataset):
    """
    Loads .pt files (containing {'tokens': ..., 'mask': ...}) from a directory.
    """
    def __init__(self, data_dir, cond_dir=None):
        # raise an error if data_dir does not exist
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory {data_dir} does not exist.")
        # if wav2vec condition directory is provided, check that it exists
        if cond_dir is not None and not os.path.isdir(cond_dir):
            raise FileNotFoundError(f"Condition directory {cond_dir} does not exist.")
        
        self.files = glob.glob(os.path.join(data_dir, "*.pt"))
        self.cond_dir = cond_dir 
        if cond_dir is not None:
            wav2vec_files = glob.glob(os.path.join(cond_dir, "*.pt"))
            wav2vec_file_names = {os.path.basename(f) for f in wav2vec_files}
            # Filter self.files to only include files present in wav2vec_cond_dir.
            self.files = [f for f in self.files if os.path.basename(f) in wav2vec_file_names]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        
        # load the corresponding wav2vec conditioning file with the same basename
        pitch_cond_path = os.path.join(self.cond_dir, os.path.basename(self.files[idx]))
        pitch_cond_data = torch.load(pitch_cond_path)
        return data['prosody'], data['mask'], pitch_cond_data