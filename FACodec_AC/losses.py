import torch
import torch.nn.functional as F

def cpc_loss(ctx, x, mask, K, model, temperature=0.07):
    mask = ~mask
    B, T, H = ctx.size(); D = x.size(-1)
    loss = 0.0; n = 0
    z = F.normalize(x, dim=-1)            # (B,T,D)
    for k, W in enumerate(model.pred, 1):
        if k > K: break
        valid = mask[:, :-k] & mask[:, k:]
        # Check
        num_total   = mask[:, :-k].numel()
        num_valid_k = valid.sum().item()
        #print(f"[k={k}] valid frames {num_valid_k}/{num_total} "
        #    f"({100*num_valid_k/num_total:.1f} %)")
        if not valid.any(): continue
        c = ctx[:, :-k][valid]            # (M,H)
        p = F.normalize(W(c), dim=-1)     # (M,D)
        z_pos = z[:, k:][valid]           # (M,D)
        # Debug: Ensure prediction and positive shapes match.
        assert p.shape == z_pos.shape, f"Shape mismatch: {p.shape} vs {z_pos.shape}"
        # negatives: use all z from time offset k across batch, flattened.
        mask_k     = mask[:, k:]                       # (B, T‑k)
        z_flat     = z[:, k:].reshape(-1, D)           # (B*(T‑k), D)
        valid_flat = mask_k.reshape(-1)                # (B*(T‑k),)

        # 1) negatives pool: keep only valid frames
        z_all      = z_flat[valid_flat]   
        
        flat_mask_k = mask_k.reshape(-1)
        raw_pad = (~flat_mask_k).float().mean()
        neg_pad = (~flat_mask_k)[valid_flat].float().mean()   # expect 0 %
        #print(f"[k={k}] pad in raw {raw_pad:.2%}   pad in negatives {neg_pad:.2%}")

        # 2) remap each positive pair to its row in the *compressed* pool
        #    build a prefix‑sum map: old_index -> new_index
        remap = torch.zeros_like(valid_flat, dtype=torch.long)
        remap[valid_flat] = torch.arange(valid_flat.sum(), device=valid_flat.device)

        idx       = torch.nonzero(valid, as_tuple=False)        # (M,2)
        old_index = idx[:,0]*(T-k) + idx[:,1]                   # index in the *uncompressed* view
        flat_idx  = remap[old_index]      
        # Debug: Ensure target indices are within valid range.
        assert flat_idx.max().item() < z_all.shape[0], "Target index out of range"
        logits = p @ z_all.T / temperature            # (M, B*(T-k))
        # New debug info: check for NaNs and log stats.
        if torch.isnan(logits).any():
            raise ValueError(f"NaN detected in logits at k={k}")
        # Debug: print mean and std (could later be switched to logging)
        # (Remove or comment out print statements once debugging is complete)
        #print(f"k={k}: logits mean {logits.mean().item():.4f}, std {logits.std().item():.4f}, targets range {flat_idx.min().item()}-{flat_idx.max().item()}")
        targets = flat_idx.to(x.device)
        loss += F.cross_entropy(logits, targets)
        n += 1
    return loss / n if n > 0 else loss