import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from FACodec_AC.config import Config, ASRConfig
from torch.utils.tensorboard import SummaryWriter


class ConvFeedForward(nn.Module):
    """
    A feed-forward block with 1D convolution (kernel_size=3)
    to simulate a "filter size 2048" notion.
    """
    def __init__(self, d_model=1024, d_ff=2048, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size, padding=(kernel_size // 2))
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size, padding=(kernel_size // 2))
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Parameters:
            x (Tensor [batch_size, d_model, seq_len])
        
        Returns:
            Tensor [batch_size, d_model, seq_len]
        """
        # x: [batch_size, d_model, seq_len]
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        return self.dropout(out)

# Conditional LayerNorm for FiLM-style gating
class CondLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # base layernorm without affine
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        # projections to generate gamma and beta from cond input
        self.gamma_proj = nn.Linear(d_model, d_model)
        self.beta_proj  = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D], cond: [B, T, 2D] -> split into gamma, beta
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
        # After transpose it becomes (B, seq_len, 256); we want (B, seq_len, 2*d_model)
        self.prosody_proj = nn.Linear(256, d_model * 2)

        self.acoustic_proj = nn.Linear(256, d_model * 2)


        # phone conditioning
        self.phone_embedding = nn.Embedding(ASRConfig.VOCAB_SIZE + 1, d_model)
        self.phone_proj = nn.Linear(d_model, d_model * 2)

        # Extra dropout for conditioning signals (set to 0.1)
        self.dropout_cond = nn.Dropout(0.1)

        # encoder & output
        self.encoder = CustomTransformerEncoder(num_layers, d_model, nhead, d_ff, dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)

        # load std
        self.register_buffer("precomputed_std", torch.load(std_file_path))

    def forward(self,
                x: torch.LongTensor,
                padded_phone_ids: torch.LongTensor,
                t: torch.FloatTensor,
                noise_level: torch.FloatTensor,
                mask_positions: torch.BoolTensor = None,
                padding_mask: torch.BoolTensor = None,
                noise_scaled: torch.Tensor = None,
                prosody_cond: torch.Tensor = None  
    ) -> torch.Tensor:
        bsz, seq_len = x.size()
        device = x.device

        # position IDs
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        # sanitize pad tokens
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

        # --- Build FiLM conditioning tensor ---
        m = mask_positions.float().unsqueeze(-1) if mask_positions is not None else torch.zeros(bsz, seq_len, 1, device=device)
        n = noise_level.unsqueeze(-1).expand(-1, seq_len, -1)
        cond_input = torch.cat([m, n], dim=-1)            # [B,T,2]
        γβ = self.cond_mlp(cond_input)                    # [B,T,2D]

        # Incorporate prosody conditioning if provided:
        if prosody_cond is not None:
            # prosody_cond: (B, 256, seq_len) -> transpose to (B, seq_len, 256)
            prosody_in = prosody_cond.transpose(1, 2)
            # Apply dropout to prosody input
            prosody_in = self.dropout_cond(prosody_in)
            prosody_γβ = self.prosody_proj(prosody_in)         # [B, T, 2D]
            γβ = γβ + prosody_γβ

        phone_film = self.phone_proj(phone_emb)                # [B, T, 2D]
        γβ = γβ + phone_film

        # Pass through conditional encoder using the combined FiLM parameters:
        h = self.encoder(h, γβ, src_key_padding_mask=padding_mask)

        return self.fc_out(h)


def train_diffusion_model(model,
                          dataloader,
                          eval_dataloader,
                          T=10,
                          epochs=1,
                          lr=1e-4,
                          device='cuda',
                          eval_epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_ID, reduction='none')
    model.to(device)
    model.train()

    writer = SummaryWriter(log_dir=Config.tensorboard_dir)
    best_eval_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0.0
        correct_train = 0
        total_masked_train = 0
        num_batches = 0

        for batch, padding_mask, padded_phone_ids, prosody_cond in dataloader:
            optimizer.zero_grad()
            x0 = batch.to(device)   # discrete tokens
            padding_mask = padding_mask.to(device)  # True/False for padding
            padded_phone_ids = padded_phone_ids.to(device)  # condition: shape [bsz, seq_len]
            prosody_cond = prosody_cond.to(device)  # condition: shape [bsz, 256, seq_len]
            bsz, seq_len = x0.shape
            # Generate random time t and choose mask fraction
            t_value = random.uniform(0, Config.max_token_fraction)
            t_value_norm = t_value / Config.max_token_fraction
            t = torch.full((bsz,1), t_value_norm, device=device, dtype=torch.float)
            m_prob = diffusion_mask_schedule(t_value, T)
            

            # 3) Find which positions to mask; do NOT replace with mask_id
            mask_positions = get_mask_positions(x0, mask_prob=m_prob)

            # 4) Generate noise scaled by std
            noise_level_value = random.uniform(Config.NOISE_MIN, Config.NOISE_MAX)
            noise_level_value_norm = (noise_level_value - Config.NOISE_MIN) \
                   / (Config.NOISE_MAX - Config.NOISE_MIN)
            noise_level = torch.full((bsz,1), noise_level_value_norm, device=device, dtype=torch.float)

            feature_dim = model.proj_to_256.out_features
            noise_scaled = torch.randn(bsz, seq_len, feature_dim, device=x0.device) \
                * (noise_level_value * model.precomputed_std)

            # Forward pass: add noise to masked positions
            logits = model(x0, padded_phone_ids, t, noise_level,
                           mask_positions=mask_positions, 
                           padding_mask=padding_mask, 
                           noise_scaled=noise_scaled,
                           prosody_cond=prosody_cond)
            bsz, seq_len, vocab_sz = logits.shape

            # flatten everything
            logits_flat   = logits.view(-1, vocab_sz)      # [B⋅T, V]
            x0_flat       = x0.view(-1)                   # [B⋅T]
            mask_flat     = mask_positions.view(-1).float()  # 1.0 = masked, 0.0 = unmasked

            # compute per-token cross-entropy
            all_ce        = F.cross_entropy(logits_flat, x0_flat, reduction='none')  # [B⋅T]

            # masked loss
            masked_sum    = (all_ce * mask_flat       ).sum()
            num_masked    = mask_flat.sum().clamp_min(1.0)
            masked_loss   = masked_sum / num_masked

            # unmasked “anchor” loss
            unmask_sum    = (all_ce * (1.0 - mask_flat)).sum()
            num_unmasked  = ((1.0 - mask_flat).sum()).clamp_min(1.0)
            unmasked_loss = unmask_sum / num_unmasked

            # combine
            lambda_u      = 0.01   # small weight on the unmasked penalty
            loss          = masked_loss + lambda_u * unmasked_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Compute training accuracy over masked tokens
            predicted = logits.argmax(dim=-1)
            correct = ((predicted == x0) & mask_positions).sum().item()
            total = mask_positions.sum().item()
            correct_train += correct
            total_masked_train += total

            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        train_accuracy = correct_train / (total_masked_train + 1e-9)
        print(f"Epoch {epoch+1}/{epochs}, Loss={avg_loss:.4f}, Train Accuracy={train_accuracy:.4f}")
        writer.add_scalar("Loss/Train", avg_loss, epoch+1)
        writer.add_scalar("Accuracy/Train", train_accuracy, epoch+1)
        if (epoch+1) % eval_epochs == 0:
            model.eval()
            total_test_loss = 0.0
            correct_eval = 0
            total_masked_eval = 0
            test_batches = 0
            with torch.no_grad():
                for test_batch, padding_mask, test_phone_ids, prosody_cond_test in eval_dataloader:
                    x0 = test_batch.to(device)
                    pad_mask = padding_mask.to(device)
                    phone_ids = test_phone_ids.to(device)
                    prosody_cond = prosody_cond_test.to(device)  # condition: shape [bsz, 256, seq_len]
                    bsz, T = x0.shape

                    # 1) Sample t, mask, noise_level, noise_scaled exactly as in train
                    t_value = random.uniform(0, Config.max_token_fraction)
                    t = torch.full((bsz,1), t_value/Config.max_token_fraction, device=device)
                    m_prob = diffusion_mask_schedule(t_value, T)
                    mask_positions = get_mask_positions(x0, mask_prob=m_prob)

                    noise_val = random.uniform(Config.NOISE_MIN, Config.NOISE_MAX)
                    noise_level = torch.full((bsz,1),
                        (noise_val - Config.NOISE_MIN)/(Config.NOISE_MAX-Config.NOISE_MIN),
                        device=device)
                    feat_dim = model.proj_to_256.out_features
                    noise_scaled = (torch.randn(bsz, T, feat_dim, device=device)
                                    * (noise_val * model.precomputed_std))

                    # 2) Forward pass
                    logits = model(x0, phone_ids, t, noise_level,
                                mask_positions=mask_positions,
                                padding_mask=pad_mask,
                                noise_scaled=noise_scaled,
                                prosody_cond=prosody_cond)
                    V = logits.size(-1)

                    # 3) Compute per-token CE
                    logits_flat = logits.view(-1, V)
                    x0_flat     = x0.view(-1)
                    mask_flat   = mask_positions.view(-1).float()
                    ce_flat     = F.cross_entropy(logits_flat, x0_flat, reduction='none')

                    # 4) Masked loss
                    masked_loss   = (ce_flat * mask_flat).sum() / (mask_flat.sum().clamp_min(1.0))
                    # 5) Unmasked‐token anchor
                    unmask_loss   = (ce_flat * (1-mask_flat)).sum() / ((1-mask_flat).sum().clamp_min(1.0))
                    loss_test     = masked_loss + lambda_u * unmask_loss

                    total_test_loss += loss_test.item()

                    # Compute test accuracy for masked tokens
                    predicted_test = logits.argmax(dim=-1)
                    correct_batch = ((predicted_test == x0) & mask_positions).sum().item()
                    total_batch = mask_positions.sum().item()
                    correct_eval += correct_batch
                    total_masked_eval += total_batch

                    test_batches += 1
            
            avg_test_loss = total_test_loss / max(test_batches, 1)
            test_accuracy = correct_eval / (total_masked_eval + 1e-9)
            print(f"Epoch {epoch+1}/{epochs}, Eval Test Loss={avg_test_loss:.4f}, Eval Accuracy={test_accuracy:.4f}")
            writer.add_scalar("Loss/Eval", avg_test_loss, epoch+1)
            writer.add_scalar("Accuracy/Eval", test_accuracy, epoch+1)
            
            # Save checkpoint only if evaluation loss improves
            if avg_test_loss < best_eval_loss:
                best_eval_loss = avg_test_loss
                os.makedirs(Config.checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(Config.checkpoint_dir, f"diffusion_transformer.pt")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path} at epoch {epoch+1} with Eval Loss={avg_test_loss:.4f}")
            model.train()
    
    writer.close()

def diffusion_mask_schedule(t, T):
    """
    Returns a masking fraction based on time t.
    """
    return math.sin(math.pi * t / (2.0 * T))

def mask_tokens(x, mask_prob, mask_id=Config.MASK_ID):
    """
    x: tensor [batch_size, seq_len] of token IDs.
    Returns a masked version of x, where a fraction `mask_prob` of non-PAD tokens are replaced by mask_id.
    Also returns a binary mask indicating which tokens were masked.
    """
    mask_positions = get_mask_positions(x, mask_prob)
    x_masked = x.clone()
    x_masked[mask_positions] = mask_id
    return x_masked, mask_positions

# def get_mask_positions(
#         x: torch.Tensor,
#         mask_prob: float,
#         *,
#         min_size: int = 40,
#         max_size: int = 100,
#         max_global_attempts: int = 1000,
# ) -> torch.Tensor:
#     """
#     Args
#     ----
#     x          : LongTensor [B, L] – token IDs
#     mask_prob  : float        – target fraction of *each* sequence to mask
#     min_size   : int          – minimum contiguous-chunk length
#     max_size   : int          – maximum contiguous-chunk length
#     Returns
#     -------
#     mask : BoolTensor [B, L] – True where x should be masked (never on PAD)
#     """

#     B, L = x.shape
#     device = x.device
#     mask   = torch.zeros_like(x, dtype=torch.bool, device=device)

#     # pre–compute once, avoids re-allocating each sequence
#     idx = torch.arange(L, device=device)

#     for b in range(B):
#         target        = int(round(mask_prob * L))
#         if target == 0:
#             continue

#         remaining     = target
#         occupied      = torch.zeros(L, dtype=torch.bool, device=device)
#         global_trials = 0

#         # keep trying until we hit the budget or run out of space / attempts
#         while remaining > 0 and global_trials < max_global_attempts:
#             global_trials += 1

#             # 1. sample size (never less than 1, at most remaining, capped by bounds)
#             chunk_len = random.randint(min_size, max_size)
#             chunk_len = max(1, min(chunk_len, remaining, L))

#             # 2. find all contiguous gaps large enough to fit `chunk_len`
#             free = (~occupied).nonzero(as_tuple=True)[0]        # 1-D tensor of indices
#             if free.numel() == 0:
#                 break                                           # nothing left to mask

#             # find split points where the run is no longer consecutive
#             diffs   = torch.diff(free)
#             breaks  = torch.where(diffs != 1)[0] + 1            # 1-based indices

#             # convert break-indices → segment-sizes
#             #   e.g. len=10, breaks=[3,7]  →  sizes=[3,4,3]
#             starts  = torch.cat([free.new_tensor([0]), breaks])
#             ends    = torch.cat([breaks, free.new_tensor([free.numel()])])
#             sizes   = (ends - starts).tolist()

#             # finally slice into contiguous gaps
#             gaps = torch.split(free, sizes)

#             # keep only those gaps that can fit the chunk
#             gaps = [g for g in gaps if g.numel() >= chunk_len]
#             if not gaps:
#                 break

#             # 3. pick a gap & starting offset uniformly
#             g        = gaps[random.randrange(len(gaps))]
#             offset   = random.randrange(0, g.numel() - chunk_len + 1)
#             start    = int(g[offset])
#             end      = start + chunk_len

#             # 4. apply chunk
#             occupied[start:end] = True
#             remaining          -= chunk_len

#         # write result, excluding PAD tokens
#         mask[b] = occupied & (x[b] != Config.PAD_ID)

#     return mask

def get_mask_positions(x: torch.Tensor, mask_prob, p_drop: float = 0.1) -> torch.Tensor:
    """
    Args:
        x      : LongTensor of token IDs of shape [B, T]
        p_drop : float. With probability p_drop, the entire sequence is masked.
                 Otherwise, a single contiguous segment is masked.
                 For audio, r ~ Uniform(0.7, 1.0) determines the segment length as r * T.
                 
    Returns:
        mask   : BoolTensor of shape [B, T] where masked positions are True and unmasked are False,
                 excluding PAD tokens (those with value Config.PAD_ID).
    """
    B, T = x.shape
    device = x.device
    mask = torch.zeros((B, T), dtype=torch.bool, device=device)
    
    for b in range(B):
        if random.random() < p_drop:
            mask[b] = True
        else:
            r = random.uniform(0.3, 0.4)
            seg_len = max(1, int(round(r * T)))
            start = random.randint(0, T - seg_len)
            mask[b, start:start + seg_len] = True

    # Exclude PAD tokens from being masked.
    mask = mask & (x != Config.PAD_ID)
    return mask