import os
import csv
import torch
import torchaudio
from FACodec_AC.utils import process_files_text_phoneme_forced, load_transcript_metadata
from phonemizer import phonemize
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)


def main():
    # Initialize device and forced alignment model pipeline.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1) Load phoneme‐CTC Wav2Vec2
    fe  = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    tok = Wav2Vec2CTCTokenizer .from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft", use_fast=False)
    proc= Wav2Vec2Processor   (feature_extractor=fe, tokenizer=tok)
    model= Wav2Vec2ForCTC     .from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft").to(device)
    model.eval()
    target_sr = fe.sampling_rate # 16000

    # Define directories (update paths as needed)
    audio_folder = "/home/yurii/Projects/AC/ljspeech/LJSpeech-1.1/wavs"
    embeddings_folder = "/home/yurii/Projects/AC/ljspeech/zc1_dataset"  # Contains train and test subdirectories
    output_folder = "/home/yurii/Projects/AC/ljspeech/wav2vec_dataset_forced_phoneme"
    transcript_file = "/home/yurii/Projects/AC/ljspeech/LJSpeech-1.1/metadata.csv"  # Updated to metadata.csv

    # Load transcript metadata.
    transcript_metadata = load_transcript_metadata(transcript_file)

    # Process train embeddings.
    train_embeddings_folder = os.path.join(embeddings_folder, "train")
    output_train_folder = os.path.join(output_folder, "train")
    os.makedirs(output_train_folder, exist_ok=True)
    process_files_text_phoneme_forced(
        audio_folder,
        train_embeddings_folder,
        transcript_metadata,
        output_train_folder,
        device,
        model,
        proc,
        target_sr
    )

    # Process test embeddings.
    test_embeddings_folder = os.path.join(embeddings_folder, "test")
    output_test_folder = os.path.join(output_folder, "test")
    os.makedirs(output_test_folder, exist_ok=True)
    process_files_text_phoneme_forced(
        audio_folder,
        test_embeddings_folder,
        transcript_metadata,
        output_test_folder,
        device,
        model,
        proc,
        target_sr
    )

if __name__ == "__main__":
    main()