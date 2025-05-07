import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from FACodec_AC.config import Config, ASRConfig, PitchConfig


class ConvFeedForward(nn.Module):
    """
    A feed-forward block with 1D convolution (kernel_size=3)
    to simulate a "filter size 2048" notion.
    """
    def __init__(self, d_model: int = 1024, d_ff: int = 2048, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size, padding=(kernel_size // 2))
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size, padding=(kernel_size // 2))
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x (Tensor [batch_size, d_model, seq_len])
        
        Returns:
            Tensor [batch_size, d_model, seq_len]
        """
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        return self.dropout(out)

class CondLayerNorm(nn.Module):
    """
    Normalizes input tensor x and applies affine modulation using parameters derived from a conditioning tensor.

    Args:
        d_model (int): Dimensionality of input features.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x (torch.Tensor): Tensor with shape [B, T, D] to be normalized.
            cond (torch.Tensor): Tensor with shape [B, T, 2D], split into gamma and beta for modulation.

        Returns:
            torch.Tensor: Normalized and modulated tensor with shape [B, T, D].
        """
        gamma, beta = cond.chunk(2, dim=-1)
        x_norm = self.norm(x)
        return x_norm * (1 + gamma) + beta

class CustomTransformerEncoderLayer(nn.Module):
    """
    A custom Transformer encoder layer with ConvFeedForward and conditional LayerNorm.
    """
    def __init__(self, d_model: int=1024, nhead: int=8, d_ff: int=2048, dropout: float=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        # conditional LayerNorm for post-FFN
        self.norm2 = CondLayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.conv_ff = ConvFeedForward(d_model=d_model, d_ff=d_ff, kernel_size=3, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        src_key_padding_mask: torch.BoolTensor = None
    ) -> torch.Tensor:
        """
        Parameters:
            x (torch.Tensor): Input tensor of shape [B, T, D]
            cond (torch.Tensor): Conditioning tensor used for the conditional layer normalization (FiLM parameters).
            src_key_padding_mask (torch.BoolTensor, optional): Boolean mask indicating positions to ignore in the attention mechanism. Shape should conform to [B, T].

        Returns:
            torch.Tensor: Output tensor of shape [B, T, D] after applying self-attention, feed-forward operations, and conditional normalization.
        """
        # Self-attention block
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Conv feed-forward block
        x_t = x.transpose(1, 2)  # [B, D, T]
        ff_out = self.conv_ff(x_t)
        ff_out = ff_out.transpose(1, 2)  # [B, T, D]
        x = x + self.dropout(ff_out)

        # Conditional LayerNorm with FiLM parameters
        return self.norm2(x, cond)

class CustomTransformerEncoder(nn.Module):
    """
    Stacks multiple Conditional Transformer encoder layers.
    """
    def __init__(self, num_layers: int=12, d_model: int=1024, nhead: int=8, d_ff: int=2048, dropout: float=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        src_key_padding_mask: torch.BoolTensor = None
    ) -> torch.Tensor:
        # cond: [B, T, 2D]
        for layer in self.layers:
            x = layer(x, cond, src_key_padding_mask=src_key_padding_mask)
        return x

class DiffusionTransformerModel(nn.Module):
    """        
    Parameters:
        pretrained_codebook (nn.Embedding): FACodec Embedding of 1024 codes (dim 8) for input indices.
        pretrained_proj_layer (nn.Module): Projects codebook vectors from 8D to 256D for model and FACodec decoder.
        std_file_path (str): Path to tensor with standard deviations for FACodec content embeddings.
        vocab_size (int, optional): Vocabulary size (default: 1024).
    """
    def __init__(
        self,
        std_file_path: str,
        d_model: int = 1024,
        nhead: int = 8,
        num_layers: int = 12,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
        prosody_model: bool = False,
    ):
        super().__init__()

        self.prosody_model = prosody_model
        self.d_model = d_model

        # Input feature dim is fixed as 256 (removed self.proj_to_256)
        self.proj_to_d_model = nn.Linear(256, d_model)

        if self.prosody_model:
            self.proj_zc1_to_d_model = nn.Linear(256, d_model)

        # positional embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # conditional FiLM MLP: input [mask_bit, noise_level] (old branch, unused now)
        self.cond_mlp = nn.Sequential(
            nn.Linear(2, d_model * 2),
            nn.ReLU(),
        )

        # prosody conditioning projection
        self.prosody_proj = nn.Linear(256, d_model * 2)

        # phone conditioning
        if self.prosody_model:
            self.global_cond_embedding = nn.Embedding(PitchConfig.VOCAB_SIZE + 1, d_model)
        else:
            self.global_cond_embedding = nn.Embedding(ASRConfig.VOCAB_SIZE + 1, d_model)
        self.global_cond_proj = nn.Linear(d_model, d_model * 2)

        # Extra dropout for conditioning signals
        self.dropout_cond = nn.Dropout(0.1)

        # encoder & output; output feature dim is now 256
        self.encoder = CustomTransformerEncoder(num_layers, d_model, nhead, d_ff, dropout)
        feature_dim = 256
        self.fc_out = nn.Linear(d_model, feature_dim)

        # Add noise_proj to process noise_scaled as conditioning input
        self.noise_proj = nn.Linear(feature_dim, d_model * 2)

        if not self.prosody_model:
            # fc_zc2 head, concatenating encoder output h and zc1 prediction
            self.fc_zc2 = nn.Linear(d_model + feature_dim, feature_dim)

        # load std
        self.register_buffer("precomputed_std", torch.load(std_file_path))

    def forward(self,
                x: torch.Tensor,  # x is always continuous with dim 256
                padded_global_cond_ids: torch.LongTensor,
                noise_scaled: torch.Tensor,
                padding_mask: torch.BoolTensor = None,
                prosody_cond: torch.Tensor = None  
    ) -> tuple:
        x = x.transpose(1, 2)
        bsz, seq_len = x.size() if x.dim() == 2 else (x.shape[0], x.shape[1])
        device = x.device
        
        # position IDs
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        
        # Use continuous input x directly (dim=256)
        code_up = x
        
        # Project to model dimension and add positional & global cond embeddings
        token_emb = self.proj_to_d_model(code_up)      # [B, T, D]
        pos_emb   = self.pos_embedding(pos_ids)         # [1, T, D]
        
        
        if self.prosody_model:
            global_cond_emb = self.proj_zc1_to_d_model(prosody_cond.transpose(1, 2))
        else:
            global_cond_emb = self.global_cond_embedding(padded_global_cond_ids)

        global_cond_emb = self.dropout_cond(global_cond_emb)

        h = token_emb + pos_emb + global_cond_emb
        
        # Build conditioning using noise_scaled and phone/prosody cues, with dropout applied
        noise_cond = self.dropout_cond(self.noise_proj(noise_scaled))  # [B, T, 2*d_model]

        if self.prosody_model:
            phone_cond = torch.zeros_like(noise_cond)
        else:
            phone_cond = self.dropout_cond(self.global_cond_proj(global_cond_emb))       # [B, T, 2*d_model]
        cond = noise_cond + phone_cond

        if prosody_cond is not None:
            prosody_repr = self.dropout_cond(self.prosody_proj(prosody_cond.transpose(1, 2)))
            cond += prosody_repr
        
        # Forward through encoder
        h = self.encoder(h, cond, src_key_padding_mask=padding_mask)
        
        # zc1 prediction head
        zc1_pred = self.fc_out(h)
        
        if self.prosody_model:
            return zc1_pred
        else:
            zc2_input = torch.cat([h, zc1_pred], dim=-1)
            zc2_pred = self.fc_zc2(zc2_input)
            return zc1_pred, zc2_pred


