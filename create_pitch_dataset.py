import os
import torch
import torchaudio
import tqdm
from FACodec_AC.config import Config, PitchConfig
from FACodec_AC.utils import get_pitched_aligned, interpolate_alignment, pad_token_sequence

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_folder = Config.wav_dir
    embeddings_folder = Config.facodec_dataset_dir
    output_folder = PitchConfig.pitch_cond_dir  # Reuse phoneme_cond_dir for pitch-aligned data

    # Process train embeddings
    train_embeddings_folder = os.path.join(embeddings_folder, "train")
    output_train_folder = os.path.join(output_folder, "train")
    os.makedirs(output_train_folder, exist_ok=True)
    print("Processing train pitch-aligned files...")
    for file in tqdm.tqdm(os.listdir(train_embeddings_folder), desc="Processing pitch-aligned files"):
        if file.endswith(".pt"):
            embedding_path = os.path.join(train_embeddings_folder, file)
            try:
                f0_quant, num_zeros = get_pitched_aligned(embedding_path, audio_folder, device)
                interpolated_f0 = interpolate_alignment(f0_quant, num_zeros)
                padded_f0, pad_mask = pad_token_sequence(interpolated_f0, Config.max_seq_len, PitchConfig.PAD_ID)
                output_path = os.path.join(output_train_folder, file)
                torch.save(padded_f0, output_path)
            except Exception as e:
                print(f"Error processing {file}: {e}")

    # Process test embeddings
    test_embeddings_folder = os.path.join(embeddings_folder, "test")
    output_test_folder = os.path.join(output_folder, "test")
    os.makedirs(output_test_folder, exist_ok=True)
    print("Processing test pitch-aligned files...")
    for file in tqdm.tqdm(os.listdir(test_embeddings_folder), desc="Processing pitch-aligned files"):
        if file.endswith(".pt"):
            embedding_path = os.path.join(test_embeddings_folder, file)
            try:
                f0_quant, num_zeros = get_pitched_aligned(embedding_path, audio_folder, device)
                interpolated_f0 = interpolate_alignment(f0_quant, num_zeros)
                padded_f0, pad_mask = pad_token_sequence(interpolated_f0, Config.max_seq_len, PitchConfig.PAD_ID)
                output_path = os.path.join(output_test_folder, file)
                torch.save(padded_f0, output_path)
            except Exception as e:
                print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    main()