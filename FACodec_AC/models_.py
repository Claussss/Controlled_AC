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
from einops import rearrange
from torch import einsum


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
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_attn = Attention(dim=d_model, 
                                   dim_heads=d_model // nhead, 
                                   dim_context=None,
                                   causal=False, 
                                   zero_init_output=True)
        self.cross_attn = Attention(dim=d_model, 
                                    dim_heads=d_model // nhead, 
                                    dim_context=d_model,
                                    causal=False, 
                                    zero_init_output=True)
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
        # attn_out, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        attn_out = self.self_attn(x, mask=src_key_padding_mask)

        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        x = x + self.cross_attn(x, context = cond, mask=src_key_padding_mask)


        # Conv feed-forward block
        x_t = x.transpose(1, 2)  # [B, D, T]
        ff_out = self.conv_ff(x_t)
        ff_out = ff_out.transpose(1, 2)  # [B, T, D]
        x = x + self.dropout(ff_out)

        # Conditional LayerNorm with FiLM parameters
        # return self.norm2(x, cond)
        return x
    
def or_reduce(masks):
    head, *body = masks
    for rest in body:
        head = head | rest
    return head
    
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_heads = 64,
        dim_context = None,
        causal = False,
        zero_init_output=True,
        qk_norm = False,
        natten_kernel_size = None
    ):
        super().__init__()
        self.dim = dim
        self.dim_heads = dim_heads
        self.causal = causal

        dim_kv = dim_context if dim_context is not None else dim
        
        self.num_heads = dim // dim_heads
        self.kv_heads = dim_kv // dim_heads

        if dim_context is not None:
            self.to_q = nn.Linear(dim, dim, bias=False)
            self.to_kv = nn.Linear(dim_kv, dim_kv * 2, bias=False)
        else:
            self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.to_out = nn.Linear(dim, dim, bias=False)

        if zero_init_output:
            nn.init.zeros_(self.to_out.weight)

        self.qk_norm = qk_norm

        # Using 1d neighborhood attention
        self.natten_kernel_size = natten_kernel_size
        if natten_kernel_size is not None:
            return

        # self.use_pt_flash = torch.cuda.is_available() and version.parse(torch.__version__) >= version.parse('2.0.0')

        # self.use_fa_flash = torch.cuda.is_available() and flash_attn_func is not None
        
        self.use_pt_flash = True
        self.use_fa_flash = False
        
        self.sdp_kwargs = dict(
            enable_flash = True,
            enable_math = True,
            enable_mem_efficient = True
        )

    def flash_attn(
            self,
            q, 
            k, 
            v,
            mask = None,
            causal = None
    ):
        batch, heads, q_len, _, k_len, device = *q.shape, k.shape[-2], q.device
        kv_heads = k.shape[1]
        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        if heads != kv_heads:
            # Repeat interleave kv_heads to match q_heads
            heads_per_kv_head = heads // kv_heads
            k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim = 1), (k, v))

        if k.ndim == 3:
            k = rearrange(k, 'b ... -> b 1 ...').expand_as(q)

        if v.ndim == 3:
            v = rearrange(v, 'b ... -> b 1 ...').expand_as(q)

        causal = self.causal if causal is None else causal

        if q_len == 1 and causal:
            causal = False
        
        if mask is not None:
            assert mask.ndim == 4
            mask = mask.expand(batch, heads, q_len, k_len)

        # handle kv cache - this should be bypassable in updated flash attention 2

        if k_len > q_len and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device = device)
            if mask is None:
                mask = ~causal_mask
            else:
                mask = mask & ~causal_mask
            causal = False

        # manually handle causal mask, if another mask was given

        row_is_entirely_masked = None

        if mask is not None and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device = device)
            mask = mask & ~causal_mask

            # protect against an entire row being masked out

            row_is_entirely_masked = ~mask.any(dim = -1)
            mask[..., 0] = mask[..., 0] | row_is_entirely_masked

            causal = False
        
        # print('q shape:', q.shape, 'k shape:', k.shape, 'v shape:', v.shape)
        with torch.backends.cuda.sdp_kernel(**self.sdp_kwargs):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                is_causal = causal
            )

        # for a row that is entirely masked out, should zero out the output of that row token

        if row_is_entirely_masked is not None:
            out = out.masked_fill(row_is_entirely_masked[..., None], 0.)

        return out

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None,
        rotary_pos_emb = None,
        causal = None
    ):
        # print('x shape:', x.shape, 'context shape:', context.shape if context is not None else None)
        h, kv_h, has_context = self.num_heads, self.kv_heads, context is not None

        kv_input = context if has_context else x

        if hasattr(self, 'to_q'):
            # Use separate linear projections for q and k/v
            q = self.to_q(x)
            q = rearrange(q, 'b n (h d) -> b h n d', h = h)

            k, v = self.to_kv(kv_input).chunk(2, dim=-1)

            k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = kv_h), (k, v))
        else:
            # Use fused linear projection
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        
        # Normalize q and k for cosine sim attention
        if self.qk_norm:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)

        if rotary_pos_emb is not None and not has_context:
            freqs, _ = rotary_pos_emb

            q_dtype = q.dtype
            k_dtype = k.dtype

            q = q.to(torch.float32)
            k = k.to(torch.float32)
            freqs = freqs.to(torch.float32)

            q = apply_rotary_pos_emb(q, freqs)
            k = apply_rotary_pos_emb(k, freqs)

            q = q.to(q_dtype)
            k = k.to(k_dtype)
        
        input_mask = context_mask 

        if input_mask is None and not has_context:
            input_mask = mask

        # determine masking
        masks = []
        final_attn_mask = None # The mask that will be applied to the attention matrix, taking all masks into account

        if input_mask is not None:
            input_mask = rearrange(input_mask, 'b j -> b 1 1 j')
            masks.append(~input_mask)

        # Other masks will be added here later

        if len(masks) > 0:
            final_attn_mask = ~or_reduce(masks)

        n, device = q.shape[-2], q.device

        causal = self.causal if causal is None else causal

        if n == 1 and causal:
            causal = False

        if self.natten_kernel_size is not None:
            if natten is None:
                raise ImportError('natten not installed, please install natten to use neighborhood attention')
            
            dtype_in = q.dtype
            q, k, v = map(lambda t: t.to(torch.float32), (q, k, v))

            attn = natten.functional.natten1dqk(q, k, kernel_size = self.natten_kernel_size, dilation=1)

            if final_attn_mask is not None:
                attn = attn.masked_fill(final_attn_mask, -torch.finfo(attn.dtype).max)

            print('natten_kernel is used')
            
            attn = F.softmax(attn, dim=-1, dtype=torch.float32)

            out = natten.functional.natten1dav(attn, v, kernel_size = self.natten_kernel_size, dilation=1).to(dtype_in)

        # Prioritize Flash Attention 2
        elif False:
            # print('flash attention 2 is used')
            assert final_attn_mask is None, 'masking not yet supported for Flash Attention 2'
            # Flash Attention 2 requires FP16 inputs
            fa_dtype_in = q.dtype
            q, k, v = map(lambda t: rearrange(t, 'b h n d -> b n h d').to(torch.float16), (q, k, v))
            
            out = flash_attn_func(q, k, v, causal = causal)
            
            out = rearrange(out.to(fa_dtype_in), 'b n h d -> b h n d')
            print(out.shape)

        # Fall back to PyTorch implementation
        elif self.use_pt_flash:
            out = self.flash_attn(q, k, v, causal = causal, mask = final_attn_mask)

        else:
            print('custom attention is used')
            # Fall back to custom implementation

            if h != kv_h:
                # Repeat interleave kv_heads to match q_heads
                heads_per_kv_head = h // kv_h
                k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim = 1), (k, v))

            scale = 1. / (q.shape[-1] ** 0.5)

            kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

            dots = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k) * scale
            
            i, j, dtype = *dots.shape[-2:], dots.dtype

            mask_value = -torch.finfo(dots.dtype).max

            if final_attn_mask is not None:
                dots = dots.masked_fill(~final_attn_mask, mask_value)

            if causal:
                causal_mask = self.create_causal_mask(i, j, device = device)
                dots = dots.masked_fill(causal_mask, mask_value)

            attn = F.softmax(dots, dim=-1, dtype=torch.float32)
            attn = attn.type(dtype)
            
            print(attn.shape)

            out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, ' b h n d -> b n (h d)')

        # Communicate between heads
        
        # with autocast(enabled = False):
        #     out_dtype = out.dtype
        #     out = out.to(torch.float32)
        #     out = self.to_out(out).to(out_dtype)
        out = self.to_out(out)
        

        if mask is not None:
            mask = rearrange(mask, 'b n -> b n 1')
            out = out.masked_fill(~mask, 0.)

        return out
    

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
        # self.cond_mlp = nn.Sequential(
        #     nn.Linear(2, d_model * 2),
        #     nn.ReLU(),
        # )
        self.cond_mlp = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        

        # prosody conditioning projection: prosody_cond is (B, 256, seq_len)
        # self.prosody_proj = nn.Linear(256, d_model * 2)
        # self.prosody_proj = nn.Linear(256, d_model)
        # TODO: remove later, not needed. (now the trained weights use it)
        # self.acoustic_proj = nn.Linear(256, d_model * 2)

        # phone conditioning
        self.phone_embedding = nn.Embedding(ASRConfig.VOCAB_SIZE + 1, d_model)
        # self.phone_proj = nn.Linear(d_model, d_model * 2)
        # self.phone_proj = nn.Linear(d_model, 1)
        self.phone_proj = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.ReLU(),
            nn.Linear(d_model*2, 512),
        )
        # self.prosody_cond_proj = nn.Linear(256, 254)
        self.prosody_cond_proj = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # Extra dropout for conditioning signals
        self.dropout_cond = nn.Dropout(0.1)

        # encoder & output
        self.encoder = CustomTransformerEncoder(num_layers, d_model, nhead, d_ff, dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)

        # load std (TODO, later we can normalize inputs instead of applying std to noise directly)
        self.register_buffer("precomputed_std", torch.load(std_file_path))

    def forward(self,
                x: torch.LongTensor,
                padded_phone_ids: torch.LongTensor,
                noise_level: torch.FloatTensor,
                mask_positions: torch.BoolTensor = None,
                padding_mask: torch.BoolTensor = None,
                noise_scaled: torch.Tensor = None,
                prosody_cond: torch.Tensor = None  
    ) -> torch.Tensor:
        """
        Parameters:
            x (torch.LongTensor):
                Input sequence of token indices.
            padded_phone_ids (torch.LongTensor):
                Sequence of phone ids for conditioning, which are embedded and added to the token representations.
            noise_level (torch.FloatTensor):
                Global noise level indicators, which are incorporated into the FiLM conditioning.
            mask_positions (torch.BoolTensor, optional):
                Boolean mask indicating the positions in the input to which noise should be applied.
            padding_mask (torch.BoolTensor, optional):
                Mask indicating padded positions in the input sequence.
            noise_scaled (torch.Tensor, optional):
                Noise tensor scaled by the precomputed standard deviation, applied at masked positions.
            prosody_cond (torch.Tensor, optional):
                Additional prosody conditioning information provided per time step.
        """
        bsz, seq_len = x.size()
        device = x.device

        # position IDs
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        # sanitize pad tokens by replacing them with 0, which is valid token in Facodec codebook
        # After codebook lookup, those vectors will be zeroed out by the padding mask anyway
        x_mod = x.clone()
        if padding_mask is not None:
            x_mod = x_mod.masked_fill(padding_mask, 0)

        # codebook & upsample
        code_vecs = self.codebook(x_mod)
        code_up   = self.proj_to_256(code_vecs)
        if padding_mask is not None:
            code_up = code_up.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # add noise
        if noise_scaled is None:
            noise_scaled = torch.zeros_like(code_up)
        code_noisy = code_up.clone()
        if mask_positions is not None:
            code_noisy[mask_positions] += noise_scaled[mask_positions]

        # project to model dim
        token_emb = self.proj_to_d_model(code_noisy)      # [B,T,D]
        pos_emb   = self.pos_embedding(pos_ids)           # [1,T,D]

        # --- Early Fusion: Add phone embeddings directly ---
        # Obtain phone embeddings and apply dropout
        phone_emb = self.phone_embedding(padded_phone_ids)  # [B, T, D]
        phone_emb = self.dropout_cond(phone_emb)
        
        h = token_emb + pos_emb + phone_emb
        
        # print(h.shape)
        

        # --- Build FiLM conditioning tensor ---
        m = mask_positions.float().unsqueeze(-1) if mask_positions is not None else torch.zeros(bsz, seq_len, 1, device=device)
        # TODO: test whether FiLM is good for mask_pos
        n = noise_level.unsqueeze(-1).expand(-1, seq_len, -1)
        cond_input = torch.cat([m, n], dim=-1)            # [B,T,2]
        # γβ = self.cond_mlp(cond_input)                    # [B,T,2D]
        
       
        cond_input = self.cond_mlp(cond_input)                     # [B, 2D, T]  
        # n = n.transpose(1, 2)
        
        # padded_phone_ids = padded_phone_ids.unsqueeze(1)  # [B, 1, T] 
        phone_cross = self.phone_proj(phone_emb) 
        # phone_cross = phone_cross.transpose(1, 2)              
        # prosody_in = prosody_cond.transpose(1, 2)                   
        # print(n.shape)
        # print(prosody_cond.shape)
        # print(phone_cross.shape)
        # cross_cond = prosody_cond.transpose(1, 2)
        prosody_cond = self.prosody_cond_proj(prosody_cond.transpose(1, 2))  # [B, 256, T] -> [B, 254, T]
        cross_cond = torch.cat([cond_input, prosody_cond, phone_cross], dim=-1)  # [B, 2D+256+1, T]
        # print(cross_cond.shape)
        # print(cross_cond.shape)
        # exit()
        # cross_cond = cross_cond.transpose(1, 2)  # [B, T, 2D+256+1]
        # print(cross_cond.shape)

        # Incorporate prosody conditioning if provided:
        # if prosody_cond is not None:
        #     # prosody_cond: (B, 256, seq_len) -> transpose to (B, seq_len, 256)
        #     prosody_in = prosody_cond.transpose(1, 2)
        #     # Apply dropout to prosody input
        #     prosody_in = self.dropout_cond(prosody_in)
        #     prosody_γβ = self.prosody_proj(prosody_in)         # [B, T, 2D]
        #     γβ = γβ + prosody_γβ

        # phone_film = self.phone_proj(phone_emb)                # [B, T, 2D]
        # γβ = γβ + phone_film

        # Pass through conditional encoder using the combined FiLM parameters:
        h = self.encoder(h, cross_cond, src_key_padding_mask=padding_mask)

        return self.fc_out(h)