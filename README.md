# Controlled Accent Conversion (AC) Project

This repository provides code to create the zc1 dataset and train a diffusion transformer model on that dataset. It is organized into modules under the `FACodec_AC/` directory.

## Directory Structure
- **create_zc1_dataset.py**: Prepares the dataset by converting wav files into codebook index files (.py).  
- **train.py**: Trains the diffusion transformer model on the dataset.  
- **config.py**: Configuration definitions for paths, hyperparameters, etc.  
- **dataset.py**: Logic for loading and preprocessing the dataset.  
- **models.py**: Core model definition and training loop.  
- **utils.py**: Additional utility functions.  

## Installation

1. Create a conda environment with Python 3.11:  
   ```bash
   conda create -n facodec python=3.11.11
   ```
2. Activate the environment:  
   ```bash
   conda activate facodec
   ```
3. Install the required packages:  
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Creation

To build your dataset, first download and unzip LJSpeech:
```bash
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar xfj LJSpeech-1.1.tar.bz2
```

Then edit `create_zc1_dataset.py` to point to the correct input wav directory (e.g., `LJSpeech-1.1/wavs`) and set the desired output path. Finally, run:
```bash
python create_zc1_dataset.py
```
Each wav file will produce a `.pt` file containing a dictionary with:
- `"token"`: indices of the content codebook (excluding residual).  
- `"mask"`: a boolean mask marking the padding tokens.

## Usage

1. (Optional) Generate the dataset using the steps above.
2. Train the model locally:
   ```bash
   python train.py
   ```