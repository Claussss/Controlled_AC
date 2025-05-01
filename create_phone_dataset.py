import os
import csv
import torch
import torchaudio
import tqdm
from FACodec_AC.config import Config, ASRConfig
from FACodec_AC.utils import load_transcript_metadata, get_phone_forced_alignment, interpolate_alignment, pad_token_sequence
from phonemizer import phonemize
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)

SCRIPT_LOCATION = os.environ.get("location")


def main():
    # Initialize device and the forced alignment model pipeline.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1) Load phoneme‐CTC Wav2Vec2
    fe  = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    tok = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft", use_fast=False)
    proc = Wav2Vec2Processor(feature_extractor=fe, tokenizer=tok)
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft").to(device)
    model.eval()
    target_sr = fe.sampling_rate  # typically 16000


    audio_folder = Config.wav_dir
    embeddings_folder = Config.facodec_dataset_dir
    output_folder = Config.phoneme_cond_dir
    transcript_file = ASRConfig.metadata_path


    # Load transcript metadata.
    transcript_metadata = load_transcript_metadata(transcript_file)

    # Process train embeddings forced alignment inline:
    train_embeddings_folder = os.path.join(embeddings_folder, "train")
    output_train_folder = os.path.join(output_folder, "train")
    os.makedirs(output_train_folder, exist_ok=True)
    print("Processing train forced alignment files...")
    for file in tqdm.tqdm(os.listdir(train_embeddings_folder), desc="Processing forced alignment files"):
        if file.endswith(".pt"):
            embedding_path = os.path.join(train_embeddings_folder, file)
            try:
                # Inline functionality of process_files_text_phoneme_forced:
                predicted_ids, num_zeros, _ = get_phone_forced_alignment(
                    embedding_path,
                    audio_folder,
                    transcript_metadata,
                    device,
                    model,
                    proc,
                    target_sr
                )
                interpolated_phone_ids = interpolate_alignment(predicted_ids, num_zeros)
                padded_phone_ids, pad_mask = pad_token_sequence(interpolated_phone_ids, Config.max_seq_len, ASRConfig.PAD_ID)
                output_path = os.path.join(output_train_folder, file)
                torch.save(padded_phone_ids, output_path)
                # Uncomment the line below to print progress for each file.
                # print(f"{file}: success")
            except Exception as e:
                print(f"Error processing {file}: {e}")

    # Process test embeddings forced alignment inline:
    test_embeddings_folder = os.path.join(embeddings_folder, "test")
    output_test_folder = os.path.join(output_folder, "test")
    os.makedirs(output_test_folder, exist_ok=True)
    print("Processing test forced alignment files...")
    for file in tqdm.tqdm(os.listdir(test_embeddings_folder), desc="Processing forced alignment files"):
        if file.endswith(".pt"):
            embedding_path = os.path.join(test_embeddings_folder, file)
            try:
                predicted_ids, num_zeros, _ = get_phone_forced_alignment(
                    embedding_path,
                    audio_folder,
                    transcript_metadata,
                    device,
                    model,
                    proc,
                    target_sr
                )
                interpolated_phone_ids = interpolate_alignment(predicted_ids, num_zeros)
                padded_phone_ids, pad_mask = pad_token_sequence(interpolated_phone_ids, Config.max_seq_len, ASRConfig.PAD_ID)
                output_path = os.path.join(output_test_folder, file)
                torch.save(padded_phone_ids, output_path)
            except Exception as e:
                print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    main()