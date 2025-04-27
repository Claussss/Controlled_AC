import os
SCRIPT_LOCATION = os.environ.get("location")
class Config:
    # Data and training parameters
    if SCRIPT_LOCATION == "server":
        exp_num = os.getenv('EXP_NUM', 0) # This is specified in my slurm script
    else:
        exp_num = 777 # Default experiment name

    zc1_data_dir = '/home/yurii/Projects/AC/ljspeech/zc1_dataset'
    phoneme_cond_dir = '/home/yurii/Projects/AC/ljspeech/wav2vec_dataset_forced'
    checkpoint_path = f'./checkpoints/model_exp_{exp_num}.pt'
    tensorboard_dir = f'./tensorboard/exp_{exp_num}'
    
    max_seq_len = 807
    r_range = (0.3, 0.4) # Defines the lower and upper bound for the percentage of tokens 
                  # to drop in a contiguous segment when not dropping the entire sequence.
    p_drop = 0.1 # With probability p_drop, the entire sequence is masked.
    NOISE_MIN = 1.0
    NOISE_MAX = 3.5 # How much noise will be added to the input(soft masking)
    lambda_unmasked = 0.01 # small weight on the unmasked penalty

    batch_size = 4
    epochs = 100
    eval_epochs = 10
    checkpoint_epochs = 10
    lr = 1e-4
    device = 'cuda'

    # Model constants
    VOCAB_SIZE = 1026 # For Facodec 1024 codebook + 1 MASK_TOKEN + 1 PAD_TOKEN
    MASK_ID = 1024
    PAD_ID = 1025

    # Model Architecture
    d_model=1024
    nhead=8
    num_layers=6
    d_ff=2048
    dropout=0.1


class ASRConfig:
    # Separate configuration for ASR training.
    zc1_dir = "/home/yurii/Projects/AC/ljspeech/zc1_dataset"
    metadata_path = "/home/yurii/Projects/AC/ljspeech/LJSpeech-1.1/metadata.csv"
    checkpoint_path = "/home/yurii/Projects/AC/Controlled_AC/checkpoints"
    tensorboard_dir = "/home/yurii/Projects/AC/Controlled_AC/tensorboard/asr"
    #PAD_ID = 29
    #VOCAB_SIZE = 29 # For grapheme-based ASR
    PAD_ID = 392
    VOCAB_SIZE = 392