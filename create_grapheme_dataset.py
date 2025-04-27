import os
import csv
import torch
import torchaudio
import tqdm
from FACodec_AC.config import Config, ASRConfig
from FACodec_AC.utils import load_transcript_metadata, get_grapheme_forced_alignment, interpolate_alignment, pad_token_sequence

SCRIPT_LOCATION = os.environ.get("location")

def main():
    # Initialize device and forced alignment model pipeline.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H
    model = bundle.get_model().eval().to(device)
    target_sr = bundle.sample_rate


    if SCRIPT_LOCATION == "server":
        audio_folder = "/u/yurii/Projects/datasets/LJSpeech-1.1/wavs"
        embeddings_folder = "/u/yurii/Projects/datasets/LJSpeech-1.1/zc1_dataset"
        output_folder = "/u/yurii/Projects/datasets/LJSpeech-1.1/wav2vec_dataset_forced_60k"
        transcript_file = "/u/yurii/Projects/datasets/LJSpeech-1.1/metadata.csv" 
    else:
        # Define directories (update paths as needed)
        audio_folder = "/home/yurii/Projects/AC/ljspeech/LJSpeech-1.1/wavs"
        embeddings_folder = "/home/yurii/Projects/AC/ljspeech/zc1_dataset"  # Contains train and test subdirectories
        output_folder = "/home/yurii/Projects/AC/ljspeech/wav2vec_dataset_forced"
        transcript_file = "/home/yurii/Projects/AC/ljspeech/LJSpeech-1.1/metadata.csv"  # Updated to metadata.csv

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
                predicted_ids, num_zeros, _ = get_grapheme_forced_alignment(
                    embedding_path,
                    audio_folder,
                    transcript_metadata,
                    device,
                    model,
                    bundle,
                    target_sr
                )
                interpolated_phone_ids = interpolate_alignment(predicted_ids, num_zeros)
                padded_phone_ids, pad_mask = pad_token_sequence(interpolated_phone_ids, Config.max_seq_len, ASRConfig.PAD_ID)
                output_path = os.path.join(output_train_folder, file)
                torch.save(padded_phone_ids, output_path)
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
                predicted_ids, num_zeros = get_grapheme_forced_alignment(
                    embedding_path,
                    audio_folder,
                    transcript_metadata,
                    device,
                    model,
                    bundle,
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