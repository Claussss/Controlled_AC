import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from FACodec_AC.config import Config
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
        # x: [batch_size, d_model, seq_len]
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        return self.dropout(out)

class CustomTransformerEncoderLayer(nn.Module):
    """
    A custom Transformer encoder layer that uses our ConvFeedForward.
    """
    def __init__(self, d_model=1024, nhead=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.conv_ff = ConvFeedForward(d_model=d_model, d_ff=d_ff, kernel_size=3, dropout=dropout)

    def forward(self, x, src_key_padding_mask=None):
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        # Conv feed-forward
        x_t = x.transpose(1, 2)  # [batch, d_model, seq_len]
        ff_out = self.conv_ff(x_t)
        ff_out = ff_out.transpose(1, 2)  # back to [batch, seq_len, d_model]
        x = x + self.dropout(ff_out)
        return self.norm2(x)

class CustomTransformerEncoder(nn.Module):
    """
    Stacks multiple CustomTransformerEncoderLayers.
    """
    def __init__(self, num_layers=12, d_model=1024, nhead=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x

class DiffusionTransformerModel(nn.Module):
    def __init__(self,
                 vocab_size=Config.VOCAB_SIZE,
                 d_model=1024,
                 nhead=8,
                 num_layers=12,
                 d_ff=2048,
                 dropout=0.1,
                 max_seq_len=4096):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.encoder = CustomTransformerEncoder(num_layers, d_model, nhead, d_ff, dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, src_key_padding_mask=None):
        bsz, seq_len = x.size()
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        token_emb = self.embedding(x)       # [bsz, seq_len, d_model]
        pos_emb = self.pos_embedding(pos_ids) # [1, seq_len, d_model]
        h = token_emb + pos_emb
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        return self.fc_out(h)

def train_diffusion_model(model,
                          dataloader,
                          eval_dataloader,
                          T=10,
                          epochs=1,
                          lr=1e-4,
                          device='cuda',
                          eval_epochs=5):
    """
    Train the diffusion transformer model.
    Every `eval_epochs` epochs, it evaluates on the test set.
    Logs both loss and accuracy to TensorBoard.
    Saves a checkpoint only when evaluation loss improves.
    """
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_ID, reduction='none')
    model.to(device)
    model.train()

    writer = SummaryWriter(log_dir=Config.tensorboard_dir)  # Logs saved in Config.tensorboard_dir
    best_eval_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0.0
        correct_train = 0
        total_masked_train = 0
        num_batches = 0

        for batch, mask in dataloader:
            optimizer.zero_grad()
            x0 = batch.to(device)
            mask = mask.to(device)
            
            # Generate random time and corresponding masking probability
            t = torch.tensor(random.uniform(0, 0.5))
            m_prob = diffusion_mask_schedule(t.item(), T)
            
            # Mask tokens in x0
            x_t, mask_positions = mask_tokens(x0, mask_prob=m_prob, mask_id=Config.MASK_ID)
            
            logits = model(x_t, src_key_padding_mask=mask)
            bsz, seq_len, vocab_sz = logits.shape
            logits_flat = logits.view(bsz * seq_len, vocab_sz)
            x0_flat = x0.view(-1)
            mask_flat = mask_positions.view(-1)
            ce = criterion(logits_flat, x0_flat)
            loss = ce.mul(mask_flat).sum() / (mask_flat.sum() + 1e-9)
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
                for test_batch, test_mask in eval_dataloader:
                    x0_test = test_batch.to(device)
                    test_mask = test_mask.to(device)
                    logits_test = model(x0_test, src_key_padding_mask=test_mask)
                    bsz, seq_len, vocab_sz = logits_test.shape
                    logits_test_flat = logits_test.view(bsz * seq_len, vocab_sz)
                    x0_test_flat = x0_test.view(-1)
                    ce_test = criterion(logits_test_flat, x0_test_flat)
                    loss_test = ce_test.mean()
                    total_test_loss += loss_test.item()
                    
                    # Compute test accuracy for masked tokens
                    predicted_test = logits_test.argmax(dim=-1)
                    correct_batch = ((predicted_test == x0_test) & (test_mask)).sum().item()
                    total_batch = (test_mask).sum().item()
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
    bsz, seq_len = x.size()
    valid = x != Config.PAD_ID
    rand = torch.rand(bsz, seq_len, device=x.device)
    mask_positions = (rand < mask_prob) & valid
    x_masked = x.clone()
    x_masked[mask_positions] = mask_id
    return x_masked, mask_positions