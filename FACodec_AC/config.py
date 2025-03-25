class Config:
    # Data and training parameters
    data_dir = '/home/yurii/Projects/AC/ljspeech/zc1_dataset'
    batch_size = 8
    time_frames = 807
    T = 1
    epochs = 100
    eval_epochs = 10
    lr = 1e-4
    device = 'cuda'

    # Model constants
    VOCAB_SIZE = 1026
    MASK_ID = 1024
    PAD_ID = 1025

    # Model Architecture
    vocab_size=1026
    d_model=1024
    nhead=8
    num_layers=6
    d_ff=2048
    dropout=0.1