import os
SCRIPT_LOCATION = os.environ.get("location")

# Skip this, this is for my ssh, go to the else branch
if SCRIPT_LOCATION == "server":
    class Config:
        exp_num = os.getenv('EXP_NUM', 0)  # This is specified in my slurm script
        # Data and training parameters
        wav_dir = '/u/yurii/Projects/datasets/LJSpeech-1.1/wavs'
        facodec_dataset_dir = '/u/yurii/Projects/datasets/LJSpeech-1.1/facodec_dataset_zc1_zc2'
        std_content_path = os.path.join(facodec_dataset_dir, 'stats', 'std_zc1.pt')
        std_prosody_path = os.path.join(facodec_dataset_dir, 'stats', 'std_prosody.pt') 
        phoneme_cond_dir = '/u/yurii/Projects/datasets/LJSpeech-1.1/wav2vec_dataset_forced_phoneme'
        checkpoint_path = f'/u/yurii/Projects/Controlled_AC/checkpoints/model_exp_{exp_num}.pt'
        tensorboard_dir = f'/u/yurii/Projects/Controlled_AC/tensorboard/exp_{exp_num}'
        
        max_seq_len = 807 # aka max_seq_len
        r_range = (0.3, 0.4) # Defines the lower and upper bound for the percentage of tokens 
                    # to drop in a contiguous segment when not dropping the entire sequence.
        p_drop = 0.1 # With probability p_drop, the entire sequence is masked.
        NOISE_MIN = 1.0
        NOISE_MAX = 20 # How much noise will be added to the input(soft masking)
        
        lambda_unmasked = 0.01 # small weight on the unmasked penalty

        batch_size = 32
        epochs = 300
        eval_epochs = 2
        checkpoint_epochs = 5
        lr = 1e-4
        device = 'cuda'

        # Model constants
        VOCAB_SIZE = 1026
        MASK_ID = 1024
        PAD_ID = 1025

        # Model Architecture
        d_model=1024
        nhead=8
        num_layers=6
        d_ff=2048
        dropout=0.1

    class ASRConfig:
        metadata_path = "/u/yurii/Projects/datasets/LJSpeech-1.1/metadata.csv"
        PAD_ID = 392
        VOCAB_SIZE = 392

    class PitchConfig:
        hop_ms = 10
        median_ms = 210
        downsample_factor = 10
        lowpass_cutoff = 20
        n_bins = 16 # 0 - 15 for pitch, 16 unvoiced silence, 17 pad, 18 tokens in total
        PAD_ID = 17
        VOCAB_SIZE = 18
        std_path = '/u/yurii/Projects/datasets/LJSpeech-1.1/facodec_dataset_zc1_zc2/stats/std_prosody.pt'
        pitch_cond_dir = '/u/yurii/Projects/datasets/LJSpeech-1.1/pitch_dataset'

        NOISE_MIN = 1.0
        NOISE_MAX = 20
else:
    # Local configuration. THIS IS WHAT YOU GUYS CHANGE
    class Config:

        exp_num = 777 # Default experiment name
        wav_dir = '/home/yurii/Projects/AC/ljspeech/LJSpeech-1.1/wavs'
        facodec_dataset_dir = '/home/yurii/Projects/AC/ljspeech/facodec_dataset_zc1_zc2'
        std_content_path = os.path.join(facodec_dataset_dir, 'stats', 'std_zc1.pt')
        std_prosody_path = os.path.join(facodec_dataset_dir, 'stats', 'std_prosody.pt') 
        phoneme_cond_dir = '/home/yurii/Projects/AC/ljspeech/phone_dataset_zc1_zc2'
        checkpoint_path = f'./checkpoints/model_exp_{exp_num}.pt'
        tensorboard_dir = f'./tensorboard/exp_{exp_num}'
        
        max_seq_len = 807
        r_range = (0.3, 0.4) # Defines the lower and upper bound for the percentage of tokens 
                    # to drop in a contiguous segment when not dropping the entire sequence.
        p_drop = 0.1 # With probability p_drop, the entire sequence is masked.
        NOISE_MIN = 1.0
        NOISE_MAX = 20 # How much noise will be added to the input(soft masking)
        lambda_unmasked = 0.01 # small weight on the unmasked penalty

        batch_size = 2
        epochs = 100
        eval_epochs = 1
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
        metadata_path = "/home/yurii/Projects/AC/ljspeech/LJSpeech-1.1/metadata.csv"
        PAD_ID = 392
        VOCAB_SIZE = 392

    class PitchConfig:
        hop_ms = 10
        median_ms = 210
        downsample_factor = 10
        lowpass_cutoff = 20
        n_bins = 16 # 0 - 15 for pitch, 16 unvoiced silence, 17 pad, 18 tokens in total
        PAD_ID = 17
        VOCAB_SIZE = 18
        std_path = '/home/yurii/Projects/AC/ljspeech/zc1_dataset/stats/std.pt' # TODO fix it, it is refering to cotnent
        pitch_cond_dir = '/home/yurii/Projects/AC/ljspeech/pitch_dataset'

        NOISE_MIN = 1.0
        NOISE_MAX = 20
