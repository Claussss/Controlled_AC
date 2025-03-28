import os
import random
import torch
from torch.utils.data import DataLoader
from FACodec_AC.dataset import CodebookSequenceDataset
from FACodec_AC.models import DiffusionTransformerModel, train_diffusion_model
from FACodec_AC.config import Config
from huggingface_hub import hf_hub_download
import sys
# Append Amphion directory to sys.path so that the import works.
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Amphion'))

from models.codec.ns3_codec import FACodecDecoder

def main():
    # Seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    # Create train and test datasets/dataloaders
    train_dataset = CodebookSequenceDataset(os.path.join(Config.data_dir, 'train'))
    test_dataset  = CodebookSequenceDataset(os.path.join(Config.data_dir, 'test'))
    
    dataloader_train = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    dataloader_test  = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

        # Initialize the FACodecDecoder and load pretrained weights
    fa_decoder = FACodecDecoder(
        in_channels=256,
        upsample_initial_channel=1024,
        ngf=32,
        up_ratios=[5, 5, 4, 2],
        vq_num_q_c=2,
        vq_num_q_p=1,
        vq_num_q_r=3,
        vq_dim=256,
        codebook_dim=8,
        codebook_size_prosody=10,
        codebook_size_content=10,
        codebook_size_residual=10,
        use_gr_x_timbre=True,
        use_gr_residual_f0=True,
        use_gr_residual_phone=True,
    )
    decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder.bin")
    fa_decoder.load_state_dict(torch.load(decoder_ckpt))
    fa_decoder.eval()

    pretrained_codebook = fa_decoder.quantizer[1].layers[0].codebook
    pretrained_proj_layer = fa_decoder.quantizer[1].layers[0].out_proj
    
    # Initialize the diffusion transformer model with the pretrained args
    model = DiffusionTransformerModel(
        pretrained_codebook=pretrained_codebook,
        pretrained_proj_layer=pretrained_proj_layer,
        std_file_path=os.path.join(Config.data_dir, 'stats', 'std.pt'),
        vocab_size=Config.VOCAB_SIZE,
        d_model=Config.d_model,
        nhead=Config.nhead,
        num_layers=Config.num_layers,
        d_ff=Config.d_ff,
        dropout=Config.dropout,
        max_seq_len=Config.time_frames
    )
    
    train_diffusion_model(
        model,
        dataloader_train,
        dataloader_test,
        T=Config.T,
        epochs=Config.epochs,
        lr=Config.lr,
        device=Config.device,
        eval_epochs=Config.eval_epochs
    )

if __name__ == "__main__":
    main()