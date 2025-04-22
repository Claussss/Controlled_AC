import os
import csv
import torch
import torchaudio
from FACodec_AC.utils import process_files_wav2vec_forced

def load_transcript_metadata(transcript_file):
    """
    Loads transcript metadata from a CSV file.
    Each row in the CSV file should be formatted as:
        file_id|transcript|...
    Returns a dict mapping file_id to transcript.
    """
    metadata = {}
    with open(transcript_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) >= 2:
                file_id = row[0].strip()
                transcript = row[1].strip()
                metadata[file_id] = transcript
    return metadata

def main():
    # Initialize device and forced alignment model pipeline.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().eval().to(device)
    target_sr = bundle.sample_rate

    # Define directories (update paths as needed)
    audio_folder = "/home/yurii/Projects/AC/ljspeech/LJSpeech-1.1/wavs"
    embeddings_folder = "/home/yurii/Projects/AC/ljspeech/zc1_dataset"  # Contains train and test subdirectories
    output_folder = "/home/yurii/Projects/AC/ljspeech/wav2vec_dataset_forced"
    transcript_file = "/home/yurii/Projects/AC/ljspeech/LJSpeech-1.1/metadata.csv"  # Updated to metadata.csv

    # Load transcript metadata.
    transcript_metadata = load_transcript_metadata(transcript_file)

    # Process train embeddings.
    train_embeddings_folder = os.path.join(embeddings_folder, "train")
    output_train_folder = os.path.join(output_folder, "train")
    os.makedirs(output_train_folder, exist_ok=True)
    process_files_wav2vec_forced(
        audio_folder,
        train_embeddings_folder,
        transcript_metadata,
        output_train_folder,
        device,
        model,
        bundle,
        target_sr
    )

    # Process test embeddings.
    test_embeddings_folder = os.path.join(embeddings_folder, "test")
    output_test_folder = os.path.join(output_folder, "test")
    os.makedirs(output_test_folder, exist_ok=True)
    process_files_wav2vec_forced(
        audio_folder,
        test_embeddings_folder,
        transcript_metadata,
        output_test_folder,
        device,
        model,
        bundle,
        target_sr
    )

if __name__ == "__main__":
    main()