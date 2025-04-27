import torch
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)
from FACodec_AC.utils import process_files_wav2vec
import os

def main():
    # Initialize device, processor, and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fe = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        )
    tok = Wav2Vec2CTCTokenizer.from_pretrained(
    "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
    use_fast=False
    )
    w2v_processor = Wav2Vec2Processor(feature_extractor=fe, tokenizer=tok)
    w2v_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft").to(device)
    w2v_model.eval()

    # Define directories (update the paths as needed)
    audio_folder = "/home/yurii/Projects/AC/ljspeech/LJSpeech-1.1/wavs"
    embeddings_folder = "/home/yurii/Projects/AC/ljspeech/zc1_dataset"  # Contains train and test subdirectories
    output_folder = "/home/yurii/Projects/AC/ljspeech/wav2vec_dataset"

    # Process train files
    train_embeddings_folder = f"{embeddings_folder}/train"
    output_train_folder = f"{output_folder}/train"
    os.makedirs(output_train_folder, exist_ok=True)
    process_files_wav2vec(audio_folder, train_embeddings_folder, output_train_folder, device, w2v_model, w2v_processor)

    # Process test files
    test_embeddings_folder = f"{embeddings_folder}/test"
    output_test_folder = f"{output_folder}/test"
    os.makedirs(output_test_folder, exist_ok=True)
    process_files_wav2vec(audio_folder, test_embeddings_folder, output_test_folder, device, w2v_model, w2v_processor)

if __name__ == "__main__":
    main()