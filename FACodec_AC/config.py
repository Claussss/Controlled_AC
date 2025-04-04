class Config:
    # Data and training parameters
    data_dir = '/home/yurii/Projects/AC/ljspeech/zc1_dataset'
    checkpoint_dir = './checkpoints'
    tensorboard_dir = './tensorboard'
    
    time_frames = 807 # aka max_seq_len
    T = 1
    max_token_fraction = 0.5 # How much tokens will be masked will be sampled from uniform(0, 0.5)
    NOISE_MIN = 1.0
    NOISE_MAX = 3.5 # How much noise will be added to the input(soft masking)

    batch_size = 8
    epochs = 100
    eval_epochs = 10
    checkpoint_epochs = 10
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

    # Classifier parameters
    data_dir_foreign_classifier = '/home/yurii/Projects/AC/l2_arctic/zc1_dataset'
    data_dir_native_classifier = '/home/yurii/Projects/AC/ljspeech/zc1_dataset'
    epochs_classifier = 10
    lr_classifier = 1e-4
    eval_epochs_classifier = 1