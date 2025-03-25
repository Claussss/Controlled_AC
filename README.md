# Controlled Accent Conversion (AC) Project

This repo contains code to create the zc1 dataset and to train a diffusion transformer model on that dataset. The project is organized into modules under `FACoder_AC/`. A SLURM submission script is provided in `run_slurm.sh`.

## Directory Structure

AC/  
├── README.md  
├── requirements.txt  
├── run_slurm.sh  
├── create_zc1_dataset.py  
├── train.py  
└── FACoder_AC  
   ├── __init__.py  
   ├── config.py  
   ├── dataset.py  
   ├── models.py  
   └── utils.py  

## Installation

1. Create a conda environment with Python 3.11:  
   `conda create -n facodec python=3.11.11`

2. Activate the environment:  
   `conda activate facodec`

3. Install the required packages:  
   `pip install -r requirements.txt`

## Usage

1. Create the content codebook indexes dataset (zc1) from wavs. Specify the input dirs  inside the script:  
   `python create_zc1_dataset.py`

2. Train the model locally:  
   `python train.py`