import os
import random
import torch
from torch.utils.data import DataLoader
from FACodec_AC.dataset import ClassificationDataset
from FACodec_AC.models import SelfAttentionPoolingClassifier, train_classifier
from FACodec_AC.config import Config
from huggingface_hub import hf_hub_download
import sys

# Append Amphion directory to sys.path for FACodecDecoder import.
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Amphion'))
from models.codec.ns3_codec import FACodecDecoder

def main():
    # Seed for reproducibility.
    torch.manual_seed(42)
    random.seed(42)
    
    # Create training and evaluation datasets.
    # The expected directory structure for each dataset is:
    #   <dataset_dir>/foreign/*.pt
    #   <dataset_dir>/native/*.pt
    # Adjust the paths below as needed.
    train_dataset = ClassificationDataset(data_dir_native=os.path.join(Config.data_dir_native_classifier, 'train'),
                                          data_dir_foreign=os.path.join(Config.data_dir_foreign_classifier, 'train'))
    eval_dataset = ClassificationDataset(data_dir_native=os.path.join(Config.data_dir_native_classifier, 'test'),
                                          data_dir_foreign=os.path.join(Config.data_dir_foreign_classifier, 'test'))
    
    dataloader_train = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    dataloader_eval = DataLoader(eval_dataset, batch_size=Config.batch_size, shuffle=False)
    
    # Initialize the FACodecDecoder and load pretrained weights to extract codebook and projection layer.
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

    # Get the pretrained codebook and projection layer.
    pretrained_codebook = fa_decoder.quantizer[1].layers[0].codebook
    pretrained_proj_layer = fa_decoder.quantizer[1].layers[0].out_proj

    # Initialize the classifier model.
    model = SelfAttentionPoolingClassifier(
        pretrained_codebook=pretrained_codebook,
        pretrained_proj_layer=pretrained_proj_layer,
        num_classes=2,           # 1 for foreign, 0 for native
        input_channels=256,
        attention_hidden_dim=128
    )
    
    # Train the classifier.
    train_classifier(
        model,
        dataloader_train,
        dataloader_eval,
        epochs=Config.epochs_classifier,
        lr=Config.lr_classifier,
        device=Config.device,
        eval_epochs=Config.eval_epochs_classifier
    )

if __name__ == "__main__":
    main()