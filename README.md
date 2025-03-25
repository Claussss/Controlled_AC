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

## Usage

1. Create the dataset:  
   `python create_zc1_dataset.py`

2. Train the model locally:  
   `python train.py`

3. On a SLURM cluster, submit the job with:  
   `sbatch run_slurm.sh`