# Controlled Accent Conversion (ControlledAC)

This repository provides the code for preparing datasets and training a non-autoregressive denoising transformer model for the Controlled Accent Conversion (AC) project. The model is conditioned on prosody, and phone features derived from LJSpeech.

## Directory Structure

- **create_facodec_dataset.py**  
  Processes WAV files from an input folder and generates a `.pt` file for each WAV. Each file contains a dictionary with the following keys:
  - `prosody_indx`: indices from the prosody quantizer codebook.
  - `zc1_indx`: indices from the primary content quantizer (first-stage).
  - `zc2_indx`: indices from the residual content quantizer (second-stage).
  - `acoustic1_indx`: indices from the first acoustic residual quantizer.
  - `acoustic2_indx`: indices from the second acoustic residual quantizer.
  - `acoustic3_indx`: indices from the third acoustic residual quantizer.
  
- **create_phone_dataset.py**  
  Processes WAV files (and their corresponding transcript) to create forced alignment data that the model will use for conditioning. Here, the ASR model used is `wav2vec2-xlsr-53-espeak-cv-ft`, which produces phone outputs (IPA), so the ASR vocabulary size is 392.  


- **train.py**  
  Trains the diffusion transformer model using the dataset created by `create_facodec_dataset.py` and conditions it on phone outputs (the best results have been obtained using the phone dataset). The training loop is integrated directly in this file.

- **FACodec_AC/config.py**  
  Contains the main configuration settings including paths, training parameters, model hyperparameters, and data constants. Adjust these as needed for your dataset location and training preferences.

- **FACodec_AC/models.py**  
  Contains the transformer model definition along with any custom layers and the integrated training code (if desired).

- **FACodec_AC/dataset.py** and **FACodec_AC/utils.py**  
  Include dataset loading routines and various utility functions (e.g., forced alignment, token padding, etc.).

## Installation

1. **Clone Amphion** (FACodec implementation)  
   Outside of the ControlledAC folder (at the same directory level), clone the Amphion repository:  
   ```
   git clone https://github.com/open-mmlab/Amphion.git
   ```

2. **Create and Activate the Conda Environment**  
   ```
   conda create -n facodec python=3.11.11
   conda activate facodec
   ```

3. **Install Dependencies**  
   Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

## Dataset Creation

1. **Download LJSpeech**  
   Download and extract LJSpeech:
   ```
   wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
   tar xfj LJSpeech-1.1.tar.bz2
   ```

2. **Generate FACodec Dataset**  
   Edit the input and output directories in `config.py` as needed and run:
   ```
   python create_facodec_dataset.py
   ```
   Each WAV file will produce a `.pt` file containing a dictionary with:
   - `prosody_indx`
   - `zc1_indx`
   - `zc2_indx`
   - `acoustic1_indx`
   - `acoustic2_indx`
   - `acoustic3_indx`

3. **Generate Phone Forced Alignment Data**  
   For phoneme conditioning, open `config.py` and adjust the paths or passes as needed. Then run:
   ```
   python create_phone_dataset.py
   ```
   This script processes the audio files along with transcript metadata and produces forced alignment data for conditioning (using phone outputs from wav2vec2-xlsr-53-espeak-cv-ft).


## Training the Model

1. **Update Configurations**  
   Open `FACodec_AC/config.py` to update paths (such as `facodec_dataset_dir`, `phoneme_cond_dir`, and `checkpoint_path`) and training hyperparameters (e.g., `epochs`, `batch_size`, etc.) to suit your environment.

2. **Run the Training Script**  
   Once the dataset has been created and the configuration updated, run:
   ```
   python train.py
   ```
   This will train the transformer model. Training progress and checkpoints are saved as specified in the config file. The model is conditioned on the forced alignment data (phone/grapheme) generated earlier.

## Additional Notes

- **Checkpointing and Logs**:  
  Model checkpoints are saved in the `checkpoints/` folder and training logs are written to the `tensorboard/` directory. Use TensorBoard to monitor training progress.



