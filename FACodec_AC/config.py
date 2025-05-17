class Config:
    exp_num = 'default_exp' # Default experiment name
    wav_dir = '/home/yurii/Projects/AC/ljspeech/LJSpeech-1.1/wavs'
    facodec_dataset_dir = '/home/yurii/Projects/AC/ljspeech/facodec_dataset_indx_only'
    phoneme_cond_dir = '/home/yurii/Projects/AC/ljspeech/phone_dataset_indx_only'
    transcription_file = '/home/yurii/Projects/AC/ljspeech/LJSpeech-1.1/metadata.csv'
    checkpoint_path = f'./checkpoints/model_exp_{exp_num}.pt'
    tensorboard_dir = f'./tensorboard/exp_{exp_num}'
    
    max_seq_len = 807
    NOISE_MIN = 0
    NOISE_MAX = 8 # How much noise will be added to the input

    FACodec_dim = 8
    # Model constants
    PHONE_PAD_ID = 392
    PHONE_VOCAB_SIZE = 392


    # Training Params
    batch_size = 4
    batches_per_bucket = 10 # Number of batches per bucket. 
    epochs = 100
    eval_epochs = 5
    checkpoint_epochs = 5
    lr = 3e-4
    device = 'cuda'

    # Model Architecture
    d_model=1024
    nhead=8
    num_layers=6
    d_ff=2048
    dropout=0.1

