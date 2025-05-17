import os
import sys
import glob

import torch
import torchaudio
from torchaudio.functional import forced_align
import torch.nn.functional as F

import csv
from einops import rearrange
import re
from enum import Enum

from phonemizer import phonemize
from phonemizer.separator import Separator
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from phonemizer.backend import EspeakBackend



from huggingface_hub import hf_hub_download
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Amphion'))
from models.codec.ns3_codec import FACodecEncoder, FACodecDecoder

from FACodec_AC.config import Config

SCRIPT_LOCATION = os.environ.get("location")

sep = Separator(phone=" ", word="", syllable="")
backend = EspeakBackend('en-us') 

# This is just because delta cannot install espeak properly
if SCRIPT_LOCATION == "server":
    # absolute path to the library you compiled
    lib_path = "/u/yurii/.local/lib/libespeak-ng.so.1"
    # 1) let phonemizer know
    EspeakWrapper.set_library(lib_path)          # python-only
    # 2) make sure the dynamic loader can also find it
    os.environ["LD_LIBRARY_PATH"] = (
        os.path.dirname(lib_path) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    )
    os.environ["ESPEAK_DATA_PATH"] = "/u/yurii/.local/share/espeak-ng-data"

def normalise(txt, pipeline):
	return pipeline["normaliser"].normalize(txt)

def clean(txt, pipeline):
	txt = txt.lower().replace("’", "'")
	txt = pipeline["regex"].sub(" ", txt)
	return re.sub(r"\s+", " ", txt).strip()

def greedy_split(token: str, vocab):
    """
    Split the given token into substrings found in the vocab using the longest possible prefix match.
    If no match is found, the token is split into single characters.

    This function is used only for get_phone_forced_alignment as the phonemizer sometimes returns merged phones.

    Args:
        token (str): The input token to be split.
        vocab (iterable): Collection of valid substrings.

    Returns:
        list: A list of substrings (or individual characters) from the token.
    """
    splits = []
    idx    = 0
    L      = len(token)
    while idx < L:
        # Try longest possible match from the remaining string
        for end in range(L, idx, -1):
            piece = token[idx:end]
            if piece in vocab:
                splits.append(piece)
                idx = end
                break
        else:
            # No substring match—just emit the single character
            splits.append(token[idx])
            idx += 1
    return splits

def g2p(words, proc):
	# tagged = [f"{pipeline['LANG_TAG']}{w}" for w in words]
	# inputs = pipeline["g2p_tok"](tagged, padding=True, return_tensors="pt").to(pipeline["dev_g2p"])
	# with torch.no_grad():
	# 	out = pipeline["g2p_net"].generate(**inputs, num_beams=1, max_length=50)
	# phones = pipeline["g2p_tok"].batch_decode(out, skip_special_tokens=True)
	# tidy = [p.replace("ˈ", "").replace("ˌ", "").replace("▁", "").replace(" ", "") for p in phones]
	# return " ".join(tidy)
    phone_str = backend.phonemize(words, separator=sep, strip=True)
    raw_seq      = phone_str[0].split()

        # 2) Load the target model’s phoneme vocab
    model_vocab  = set(proc.tokenizer.get_vocab().keys())

        # 3) Fix any invalid tokens
    phone_seq = []
    for tok in raw_seq:
        if tok in model_vocab:
            phone_seq.append(tok)
        else:
            phone_seq.extend(greedy_split(tok, model_vocab))

    # 3) map phonemes → token IDs with the CTC tokenizer
    #    any missing phonemes must be added or mapped to <unk>
    vocab   = proc.tokenizer.get_vocab()
    token_ids = [vocab.get(p, proc.tokenizer.unk_token_id) for p in phone_seq]
    #token_ids = torch.tensor([token_ids], dtype=torch.int64, device=device)
    return token_ids


def text2ids(raw_txt, pipeline, proc):
	#ipa = g2p([clean(normalise(raw_txt, pipeline), pipeline)], pipeline)
	#ids = tokenizer(ipa, add_special_tokens=False).input_ids
    ids = g2p([clean(normalise(raw_txt, pipeline), pipeline)], proc)
    return ids, None


def pad_token_sequence(seq, target_len, pad_id):
    """
    Pads seq along its last dimension to target_len.
    
    If seq is one-dimensional (shape (T,)), padding is done with pad_id.
    If seq is multi-dimensional (e.g. shape (..., T)), padded vectors are zeros.
    
    Returns:
      padded_seq: tensor with the same shape as seq except the last dimension is target_len.
      mask: Boolean tensor of the same shape as padded_seq, where True indicates padded positions.
    """
    # Determine sequence length from the last dimension.
    seq_len = seq.shape[-1]
    
    # Create shape for padded sequence: same as seq except last dimension = target_len.
    new_shape = list(seq.shape)
    new_shape[-1] = target_len
    
    # If the sequence is one-dimensional, pad with pad_id, otherwise pad with zeros.
    if seq.dim() == 1:
        padded_seq = torch.full(new_shape, pad_id, dtype=seq.dtype, device=seq.device)
    else:
        padded_seq = torch.zeros(new_shape, dtype=seq.dtype, device=seq.device)
    
    # Create mask with the same shape: padded positions marked as True.
    mask = torch.zeros(new_shape, dtype=torch.bool, device=seq.device)
    
    # Copy original values into padded_seq along the last dimension.
    indices = [slice(None)] * (seq.dim() - 1) + [slice(0, seq_len)]
    padded_seq[tuple(indices)] = seq
    
    # For padded positions, mark True in the mask.
    if seq_len < target_len:
        indices_pad = [slice(None)] * (seq.dim() - 1) + [slice(seq_len, target_len)]
        mask[tuple(indices_pad)] = True

    # Remove the extra first dimension if present with size 1.
    if padded_seq.dim() > 1 and padded_seq.size(0) == 1:
        padded_seq = padded_seq.squeeze(0)
        mask = mask.squeeze(0)
    
    return padded_seq, mask

def interpolate_alignment(predicted_ids, num_zeros, mode='nearest'):
    """
    Interpolates the predicted_ids (alignment) to match the expected sequence length.
    """
    phone_ids = (
        F.interpolate(predicted_ids.unsqueeze(0).float(),
                      size=num_zeros,
                      mode=mode)
        .long()
        .squeeze(0)
    )
    return phone_ids


def process_wav_facodec(filepath, fa_encoder, fa_decoder, out_dir, device):
    """
    Processes an audio file using the FACODec to extract indices.
    Saves:
        - prosody_indx: vq_id[0],
        - zc1_indx: vq_id[1],
        - zc2_indx: vq_id[2],
        - acoustic1_indx: vq_id[3],
        - acoustic2_indx: vq_id[4],
        - acoustic3_indx: vq_id[5]
    without any padding.
    """
    try:
        wav_waveform, wav_sr = torchaudio.load(filepath)
        if wav_sr != 16000:
            resample = torchaudio.transforms.Resample(orig_freq=wav_sr, new_freq=16000)
            wav_waveform = resample(wav_waveform)
        wav_waveform = wav_waveform.to(device)
        with torch.no_grad():
            h_input = fa_encoder(wav_waveform[None, :, :])
            _, vq_id, _, _, _ = fa_decoder(h_input, eval_vq=False, vq=True)
        # Extract indices directly (assumes vq_id has at least six elements)
        prosody_indx   = vq_id[0].cpu()
        zc1_indx       = vq_id[1].cpu()
        zc2_indx       = vq_id[2].cpu()
        acoustic1_indx = vq_id[3].cpu()
        acoustic2_indx = vq_id[4].cpu()
        acoustic3_indx = vq_id[5].cpu()
        base = os.path.splitext(os.path.basename(filepath))[0]
        torch.save({
            "prosody_indx": prosody_indx,
            "zc1_indx": zc1_indx,
            "zc2_indx": zc2_indx,
            "acoustic1_indx": acoustic1_indx,
            "acoustic2_indx": acoustic2_indx,
            "acoustic3_indx": acoustic3_indx
        }, os.path.join(out_dir, f"{base}.pt"))
        return filepath, "success"
    except Exception as e:
        return filepath, f"error: {str(e)}"


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



def get_phone_forced_alignment(embedding_path, audio_folder, transcript_metadata, device, model, proc, target_sr, pipeline, inference=False):
	num_zeros = 0
	if not inference:
		embedding = torch.load(embedding_path)
		zc1_indx = embedding.get("zc1_indx", None)
		if zc1_indx is None:
			raise ValueError(f"No 'zc1_indx' key found in {embedding_path}")
		num_zeros = zc1_indx.shape[1]
	
	file_id = os.path.splitext(os.path.basename(embedding_path))[0]
	
	# --- Use text processing functions with pipeline ---
	if file_id not in transcript_metadata:
		raise ValueError(f"Transcript for {file_id} not found in transcript metadata.")
	raw = transcript_metadata[file_id]
	token_ids_list, ipa = text2ids(raw, pipeline, proc)
	token_ids = torch.tensor([token_ids_list], dtype=torch.int64, device=device)
	# --- End text processing ---

	# Load audio and perform forced alignment.
	audio_path = os.path.join(audio_folder, f"{file_id}.wav")
	if not os.path.exists(audio_path):
		raise FileNotFoundError(f"Audio file not found: {audio_path}")
	wav, sr = torchaudio.load(audio_path)
	if sr != target_sr:
		wav = torchaudio.functional.resample(wav, sr, target_sr)
	with torch.inference_mode():
		feats = proc.feature_extractor(wav.to(device).squeeze(0), sampling_rate=target_sr, return_tensors="pt")
		feats = {k: v.to(device) for k, v in feats.items()}
		logits = model(**feats).logits
	logp = torch.log_softmax(logits, dim=-1)
	
	input_lens  = torch.tensor([logp.size(1)])
	target_lens = torch.tensor([token_ids.size(1)])
	blank_id    = 0  # CTC blank
	
	aligned_ids, frame_scores = forced_align(
		logp,
		token_ids,
		input_lens,
		target_lens,
		blank=blank_id,
	)
	if not inference:
		return aligned_ids, num_zeros
	else:
		return aligned_ids, num_zeros, frame_scores, logits

class QuantizerNames(Enum):
    prosody = 0
    content = 1
    acoustic = 2

def get_z_from_indx(tokens, fa_decoder, layer=0, quantizer_num=QuantizerNames.content, dim=256):
    """
    Convert token indexes to z representations (content, prosody, or acoustic).
    
    tokens: LongTensor of shape [B, T] (pad token = 1025)
    layer: int. Prosody has only 1 layer, content 2, acoustic 3.
    """
    with torch.no_grad():
        quantizer_index = quantizer_num.value
        codebook = fa_decoder.quantizer[quantizer_index].layers[layer].codebook.weight  # [num_codes, code_dim]
        z_c1 = torch.nn.functional.embedding(tokens, codebook)     # [B, T, code_dim]
        if dim==256:
            z_c1 = fa_decoder.quantizer[quantizer_index].layers[layer].out_proj(z_c1)          # [B, T, 256]
    z_c1 = rearrange(z_c1, "b t d -> b d t")
    return z_c1


def compute_stats(dataset_dir, stats_dir, fa_decoder):
    """
    Computes channel-wise mean and std for each variable over all .pt files in dataset_dir.
    For each file, loads the index, converts it to its continuous representation using get_z_from_indx,
    and accumulates sums and sum of squares channel-wise.
    Uses different layer and quantizer arguments according to the key:
      - 'prosody_indx': layer=0, quantizer=QuantizerNames.prosody
      - 'zc1_indx': layer=0, quantizer=QuantizerNames.content
      - 'zc2_indx': layer=1, quantizer=QuantizerNames.content
      - 'acoustic1_indx': layer=0, quantizer=QuantizerNames.acoustic
      - 'acoustic2_indx': layer=1, quantizer=QuantizerNames.acoustic
      - 'acoustic3_indx': layer=2, quantizer=QuantizerNames.acoustic
    Saves the per-channel mean and std for each key in stats_dir as torch tensors and returns the stats.
    """
    # Initialize dictionaries to accumulate per-channel sums, sumsq and counts (each will be a tensor)
    sums = {}
    sumsqs = {}
    counts = {}
    keys = ["prosody_indx", "zc1_indx", "zc2_indx", "acoustic1_indx", "acoustic2_indx", "acoustic3_indx"]

    # For each key, we will initialize the accumulation tensors when we first see a file
    pt_files = glob.glob(os.path.join(dataset_dir, "*.pt"))
    for pt in pt_files:
        try:
            data = torch.load(pt)
        except Exception as e:
            print(f"Error loading {pt}: {e} -- skipping.")
            continue

        for k in keys:
            if k in data:
                # Determine layer and quantizer based on the key
                if k == "prosody_indx":
                    layer = 0
                    quant = QuantizerNames.prosody
                elif k == "zc1_indx":
                    layer = 0
                    quant = QuantizerNames.content
                elif k == "zc2_indx":
                    layer = 1
                    quant = QuantizerNames.content
                elif k == "acoustic1_indx":
                    layer = 0
                    quant = QuantizerNames.acoustic
                elif k == "acoustic2_indx":
                    layer = 1
                    quant = QuantizerNames.acoustic
                elif k == "acoustic3_indx":
                    layer = 2
                    quant = QuantizerNames.acoustic

                # Convert indices to continuous representations using get_z_from_indx
                # Each stored tensor is of shape [1, T]
                indices = data[k]
                rep = get_z_from_indx(indices, fa_decoder, layer=layer, quantizer_num=quant, dim=Config.FACodec_dim).cpu()  
                # rep has shape [B, d, T], B==1
                rep = rep.squeeze(0)  # now shape [d, T]

                # Initialize accumulation tensors if not exist
                if k not in sums:
                    d = rep.size(0)  # number of channels
                    sums[k] = torch.zeros(d, dtype=torch.float64)
                    sumsqs[k] = torch.zeros(d, dtype=torch.float64)
                    counts[k] = torch.zeros(d, dtype=torch.float64)

                # Accumulate sums over time dimension for each channel
                sums[k] += rep.sum(dim=1).double()
                sumsqs[k] += (rep ** 2).sum(dim=1).double()
                counts[k] += rep.size(1)

    means = {}
    stds = {}
    for k in keys:
        if k in counts and torch.all(counts[k] > 0):
            mean = sums[k] / counts[k]
            # Compute unbiased sample variance channel-wise
            variance = (sumsqs[k] - counts[k] * (mean ** 2)) / (counts[k] - 1)
            std = torch.sqrt(variance)
            means[k] = mean
            stds[k] = std
            torch.save(mean.float(), os.path.join(stats_dir, f"mean_{k}.pt"))
            torch.save(std.float(), os.path.join(stats_dir, f"std_{k}.pt"))
    return means, stds

def init_facodec_models(device):
    """
    Initializes FACodec encoder and decoder, loads checkpoints,
    and returns them.
    """

    # Initialize FACCodec models
    fa_encoder = FACodecEncoder(ngf=32, up_ratios=[2,4,5,5], out_channels=256)
    fa_decoder = FACodecDecoder(
        in_channels=256,
        upsample_initial_channel=1024,
        ngf=32,
        up_ratios=[5,5,4,2],
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
    encoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder.bin")
    decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder.bin")
    fa_encoder.load_state_dict(torch.load(encoder_ckpt))
    fa_decoder.load_state_dict(torch.load(decoder_ckpt))
    fa_encoder.eval()
    fa_decoder.eval()

    fa_encoder = fa_encoder.to(device)
    fa_decoder = fa_decoder.to(device)
    
    return fa_encoder, fa_decoder

def snap_latent(z, fa_decoder, layer: int = 1, quantizer_num: QuantizerNames = QuantizerNames.content):
    """
    For a given latent tensor z of shape [B, T, D],
    replaces each latent vector with its closest codebook vector 
    from the specified layer (default layer 1) of the quantizer specified by quantizer_num (default content).

    Parameters:
        z (torch.Tensor): Latent tensor with shape [B, T, D].
        fa_decoder: Instance of FACodecDecoder.
        layer (int): Layer index to use (default is 1).
        quantizer_num (QuantizerNames): Quantizer to use (default is QuantizerNames.content).

    Returns:
        torch.Tensor: Snapped latent tensor with shape [B, T, D].
    """
    codebook = fa_decoder.quantizer[quantizer_num.value].layers[layer].codebook.weight
    B, T, D = z.shape
    z_flat = z.view(-1, D)
    distances = torch.cdist(z_flat, codebook, p=2)
    closest_idxs = torch.argmin(distances, dim=1)
    snapped = codebook[closest_idxs].view(B, T, D)
    return snapped