import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from FACodec_AC.config import Config, ASRConfig
from FACodec_AC.utils import get_mask_positions
from torch.utils.tensorboard import SummaryWriter


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
        # projections to generate gamma and beta from cond input
        # TODO: remove it; (now the trained weights use it) 
        self.gamma_proj = nn.Linear(d_model, d_model)
        self.beta_proj  = nn.Linear(d_model, d_model)

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
        pretrained_codebook: nn.Embedding,
        pretrained_proj_layer: nn.Module,
        std_file_path: str,
        vocab_size: int = Config.VOCAB_SIZE,
        d_model: int = 1024,
        nhead: int = 8,
        num_layers: int = 12,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 4096
    ):
        super().__init__()
        self.d_model = d_model

        # frozen codebook & projection
        self.codebook = nn.Embedding.from_pretrained(pretrained_codebook.weight.clone(), freeze=True)
        self.proj_to_256 = pretrained_proj_layer
        for p in self.proj_to_256.parameters(): p.requires_grad = False
        self.proj_to_d_model = nn.Linear(self.proj_to_256.out_features, d_model)

        # positional embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # conditional FiLM MLP: input [mask_bit, noise_level]
        self.cond_mlp = nn.Sequential(
            nn.Linear(2, d_model * 2),
            nn.ReLU(),
        )

        # prosody conditioning projection: prosody_cond is (B, 256, seq_len)
        self.prosody_proj = nn.Linear(256, d_model * 2)
        # TODO: remove later, not needed. (now the trained weights use it)
        self.acoustic_proj = nn.Linear(256, d_model * 2)

        # phone conditioning
        self.phone_embedding = nn.Embedding(ASRConfig.VOCAB_SIZE + 1, d_model)
        self.phone_proj = nn.Linear(d_model, d_model * 2)

        # Extra dropout for conditioning signals
        self.dropout_cond = nn.Dropout(0.1)

        # encoder & output: update output layer to produce continuous output matching proj_to_256 output shape.
        self.encoder = CustomTransformerEncoder(num_layers, d_model, nhead, d_ff, dropout)
        feature_dim = self.proj_to_256.out_features
        self.fc_out = nn.Linear(d_model, feature_dim)

        # load std (TODO, later we can normalize inputs instead of applying std to noise directly)
        self.register_buffer("precomputed_std", torch.load(std_file_path))

    def forward(self,
                x: torch.Tensor,  # changed annotation to allow both Long and Float tensors
                padded_phone_ids: torch.LongTensor,
                noise_level: torch.FloatTensor,
                mask_positions: torch.BoolTensor = None,
                padding_mask: torch.BoolTensor = None,
                noise_scaled: torch.Tensor = None,
                prosody_cond: torch.Tensor = None  
    ) -> torch.Tensor:
        bsz, seq_len = x.size() if x.dim() == 2 else (x.shape[0], x.shape[1])
        device = x.device

        # position IDs
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        # If x is provided as indexes, perform codebook lookup; otherwise assume x already holds continuous vectors.
        if x.dtype == torch.long:
            x_mod = x.clone()
            if padding_mask is not None:
                x_mod = x_mod.masked_fill(padding_mask, 0)
            code_vecs = self.codebook(x_mod)
            code_up   = self.proj_to_256(code_vecs)  # continuous representation from lookup
        else:
            code_up = x  # use the provided continuous vectors directly

        # add noise on a copy of code_up
        code_noisy = code_up.clone()
        if noise_scaled is None:
            noise_scaled = torch.zeros_like(code_up)
        if mask_positions is not None:
            code_noisy[mask_positions] += noise_scaled[mask_positions] # TODO: +=

        # project to model dim
        token_emb = self.proj_to_d_model(code_noisy)      # [B,T,D]
        pos_emb   = self.pos_embedding(pos_ids)             # [1,T,D]

        # Early Fusion: add phone embeddings
        phone_emb = self.phone_embedding(padded_phone_ids)  # [B, T, D]
        phone_emb = self.dropout_cond(phone_emb)
        h = token_emb + pos_emb + phone_emb

        # Build FiLM conditioning tensor
        m = mask_positions.float().unsqueeze(-1) if mask_positions is not None else torch.zeros(bsz, seq_len, 1, device=device)
        n = noise_level.unsqueeze(-1).expand(-1, seq_len, -1)
        cond_input = torch.cat([m, n], dim=-1)            # [B,T,2]
        γβ = self.cond_mlp(cond_input)                      # [B,T,2D]

        if prosody_cond is not None:
            prosody_in = prosody_cond.transpose(1, 2)       # (B, seq_len, 256)
            prosody_in = self.dropout_cond(prosody_in)
            prosody_γβ = self.prosody_proj(prosody_in)       # [B, T, 2D]
            γβ = γβ + prosody_γβ

        phone_film = self.phone_proj(phone_emb)             # [B, T, 2D]
        γβ = γβ + phone_film

        # Pass through transformer encoder using combined FiLM parameters
        h = self.encoder(h, γβ, src_key_padding_mask=padding_mask)
        reconstruction = self.fc_out(h)

        # Return both the prediction and the clean target for MSE loss.
        return reconstruction, code_up


