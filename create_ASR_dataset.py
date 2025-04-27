import os
import torch
import tqdm
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)
from FACodec_AC.config import Config, ASRConfig
from FACodec_AC.utils import get_asr_alignment, interpolate_alignment, pad_token_sequence

SCRIPT_LOCATION = os.environ.get("location")

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

    if SCRIPT_LOCATION == "server":
            audio_folder = "/u/yurii/Projects/datasets/LJSpeech-1.1/wavs"
            embeddings_folder = "/u/yurii/Projects/datasets/LJSpeech-1.1/zc1_dataset" 
            output_folder = "/u/yurii/Projects/datasets/LJSpeech-1.1/wav2vec_dataset"
    else:
        # Define directories (update the paths as needed)
        audio_folder = "/home/yurii/Projects/AC/ljspeech/LJSpeech-1.1/wavs"
        embeddings_folder = "/home/yurii/Projects/AC/ljspeech/zc1_dataset"
        output_folder = "/home/yurii/Projects/AC/ljspeech/wav2vec_dataset"

    # Process train files inline:
    train_embeddings_folder = os.path.join(embeddings_folder, "train")
    output_train_folder = os.path.join(output_folder, "train")
    os.makedirs(output_train_folder, exist_ok=True)
    print("Processing train files...")
    for file in tqdm.tqdm(os.listdir(train_embeddings_folder), desc="Processing train files"):
        if file.endswith(".pt"):
            embedding_path = os.path.join(train_embeddings_folder, file)
            try:
                predicted_ids, num_zeros = get_asr_alignment(embedding_path, audio_folder, device, w2v_model, w2v_processor)
                interpolated_phone_ids = interpolate_alignment(predicted_ids, num_zeros)
                padded_phone_ids, pad_mask = pad_token_sequence(interpolated_phone_ids, Config.max_seq_len, ASRConfig.PAD_ID)
                # Save the processed data
                output_path = os.path.join(output_train_folder, file)
                torch.save(padded_phone_ids, output_path)
            except Exception as e:
                print(f"Error processing {file}: {e}")

    # Process test files inline:
    test_embeddings_folder = os.path.join(embeddings_folder, "test")
    output_test_folder = os.path.join(output_folder, "test")
    os.makedirs(output_test_folder, exist_ok=True)
    print("Processing test files...")
    for file in tqdm.tqdm(os.listdir(test_embeddings_folder), desc="Processing test files"):
        if file.endswith(".pt"):
            embedding_path = os.path.join(test_embeddings_folder, file)
            try:
                predicted_ids, num_zeros = get_asr_alignment(embedding_path, audio_folder, device, w2v_model, w2v_processor)
                interpolated_phone_ids = interpolate_alignment(predicted_ids, num_zeros)
                padded_phone_ids, pad_mask = pad_token_sequence(interpolated_phone_ids, Config.max_seq_len, ASRConfig.PAD_ID)
                # Save the processed data
                output_path = os.path.join(output_test_folder, file)
                torch.save(padded_phone_ids, output_path)
            except Exception as e:
                print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    main()