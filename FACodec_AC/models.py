import torch
import torch.nn as nn
from FACodec_AC.utils import snap_latent, QuantizerNames

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

class DenoisingTransformerModel(nn.Module):
    def __init__(
        self,
        d_model: int = 1024,
        nhead: int = 8,
        num_layers: int = 12,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 807,
        FACodec_dim: int = 8,
        phone_vocab_size: int = 392
    ):
        super().__init__()
        self.d_model = d_model

        self.proj_to_d_model = nn.Linear(FACodec_dim, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.encoder = CustomTransformerEncoder(num_layers, d_model, nhead, d_ff, dropout)
        self.fc_out = nn.Linear(d_model, FACodec_dim)
        # fc_zc2 head, concatenating encoder output h and zc1 prediction
        #self.fc_zc2 = nn.Linear(d_model + FACodec_dim, FACodec_dim)
        self.fc_zc2 = nn.Sequential(
                            nn.Linear(d_model + FACodec_dim, 4 * FACodec_dim),
                            nn.GELU(),
                            nn.Linear(4 * FACodec_dim, FACodec_dim)
                        )


        # phone conditioning
        self.phone_embedding = nn.Embedding(phone_vocab_size + 1, d_model)
        self.phone_proj = nn.Linear(d_model, d_model * 2)
        # Add noise_proj to process noise_scaled as conditioning input
        self.noise_proj = nn.Linear(FACodec_dim, d_model * 2)

        # Extra dropout for conditioning signals
        self.dropout_cond = nn.Dropout(0.1)


    def forward(self,
                zc1_noisy: torch.Tensor,
                zc1_ground_truth: torch.Tensor,
                padded_phone_ids: torch.LongTensor,
                noise_scaled: torch.Tensor,
                padding_mask: torch.BoolTensor,
    ) -> tuple:
        zc1_noisy = zc1_noisy.transpose(1, 2)
        zc1_ground_truth = zc1_ground_truth.transpose(1,2)
        bsz, seq_len =  zc1_noisy.shape[0], zc1_noisy.shape[1]
        device = zc1_noisy.device
        

        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)        
        # Project to model dimension and add positional & phone embeddings
        token_emb = self.proj_to_d_model(zc1_noisy)      # [B, T, D]
        pos_emb   = self.pos_embedding(pos_ids)         # [1, T, D]
        phone_emb = self.phone_embedding(padded_phone_ids)
        phone_emb = self.dropout_cond(phone_emb)
        h = token_emb + pos_emb + phone_emb
        
        # Build conditioning using noise_scaled and phone/prosody cues, with dropout applied
        noise_cond = self.dropout_cond(self.noise_proj(noise_scaled))  # [B, T, 2*d_model]
        phone_cond = self.dropout_cond(self.phone_proj(phone_emb))       # [B, T, 2*d_model]
        cond = noise_cond + phone_cond
        # t
        # Forward through encoder
        h = self.encoder(h, cond, src_key_padding_mask=padding_mask)
        
        # zc1 prediction head
        zc1_pred = self.fc_out(h)
        
        # Predict zc2 by concatenating h and zc1_pred
        zc2_input = torch.cat([h, zc1_pred.detach()], dim=-1)
        zc2_pred = self.fc_zc2(zc2_input)
        
        return zc1_pred.transpose(1, 2), zc2_pred.transpose(1, 2)

    def inference(self,
                  zc1_noisy: torch.Tensor,
                  padded_phone_ids: torch.LongTensor,
                  noise_scaled: torch.Tensor,
                  padding_mask: torch.BoolTensor,
                  fa_decoder  # pass the FACodecDecoder instance
                  ) -> tuple:
        """
        Inference method similar to forward, but uses the snapped predicted zc1 instead of ground truth.
        In inference we do not apply dropout on conditioning signals.
        """
        # Transpose input: [B, T, FACodec_dim]
        zc1_noisy = zc1_noisy.transpose(1, 2)
        bsz, seq_len = zc1_noisy.shape[0], zc1_noisy.shape[1]
        device = zc1_noisy.device
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        
        # Projection and embeddings (no dropout applied here)
        token_emb = self.proj_to_d_model(zc1_noisy)       # [B, T, d_model]
        pos_emb   = self.pos_embedding(pos_ids)            # [1, T, d_model]
        phone_emb = self.phone_embedding(padded_phone_ids)   # [B, T, d_model]
        h = token_emb + pos_emb + phone_emb
        
        # Build conditioning without using dropout_cond
        noise_cond = self.noise_proj(noise_scaled)           # [B, T, 2*d_model]
        phone_cond = self.phone_proj(phone_emb)              # [B, T, 2*d_model]
        cond = noise_cond + phone_cond
        
        # Forward through encoder with the given padding mask
        h = self.encoder(h, cond, src_key_padding_mask=padding_mask)
        
        # zc1 prediction head
        zc1_pred = self.fc_out(h)  # [B, T, FACodec_dim]
        
        # Snap predicted zc1 using the provided FACodecDecoder
       
        snapped_zc1 = snap_latent(zc1_pred, fa_decoder, layer=1, quantizer_num=QuantizerNames.content)
        
        # Predict zc2 by concatenating h and snapped zc1
        zc2_input = torch.cat([h, snapped_zc1], dim=-1)
        zc2_pred = self.fc_zc2(zc2_input)
        
        return zc1_pred.transpose(1, 2), zc2_pred.transpose(1, 2)


