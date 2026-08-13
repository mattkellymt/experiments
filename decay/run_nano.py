import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)

# Detect device (Use MPS on Apple Silicon)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon MPS device for acceleration!", flush=True)
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device!", flush=True)
else:
    device = torch.device("cpu")
    print("Using CPU device.", flush=True)

# Hyperparameters
batch_size = 32
block_size = 64
max_iters = 1500
eval_interval = 100
eval_iters = 50
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.0

# -----------------------------------------------------------------------------
# 0. Load Shakespeare Dataset & Vocab
# -----------------------------------------------------------------------------
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
full_data = torch.tensor(encode(text), dtype=torch.long)

# -----------------------------------------------------------------------------
# 1. Stochastic Distributional Decay Mechanism
# -----------------------------------------------------------------------------
class StochasticDistributionalDecay:
    def __init__(self, gamma: float = 0.001):
        self.gamma = gamma
        
    def step(self, model: nn.Module):
        if self.gamma <= 0:
            return
            
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and param.ndim > 1 and "embedding" not in name:
                    shape = param.shape
                    
                    R = torch.randn(shape, device=param.device, dtype=param.dtype)
                    r_min, r_max = R.min(), R.max()
                    R_norm = 2.0 * (R - r_min) / (r_max - r_min + 1e-8) - 1.0
                    
                    W_gamma = self.gamma * R_norm
                    W_inv = 1.0 - W_gamma
                    
                    p_mean = param.mean()
                    p_std = param.std(unbiased=False) + 1e-8
                    
                    S = torch.randn(shape, device=param.device, dtype=param.dtype) * p_std + p_mean
                    
                    param.copy_(param * W_inv + S * W_gamma)

# -----------------------------------------------------------------------------
# 2. NanoGPT Architecture (from Karpathy's tutorials)
# -----------------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * (C**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class NanoGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

# -----------------------------------------------------------------------------
# 3. Training Script
# -----------------------------------------------------------------------------
n = int(0.9 * len(full_data))
train_data = full_data[:n]
val_data = full_data[n:]

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def run_experiment():
    print(f"Dataset sizes - Train: {len(train_data)} | Val: {len(val_data)}")
    
    decay_configs = {
        "No Decay": {"wd": 0.0, "stoch_gamma": 0.0, "color": "#7f7f7f", "ls": "--"},
        "L2 Decay (λ=1e-2)": {"wd": 1e-2, "stoch_gamma": 0.0, "color": "#d62728", "ls": "-"},
        "Stoch Decay (γ=1e-4)": {"wd": 0.0, "stoch_gamma": 1e-4, "color": "#1f77b4", "ls": "-"},
    }
    
    results = {}
    
    for config_name, cfg in decay_configs.items():
        print(f"\nInitializing NanoGPT with {config_name}...", flush=True)
        torch.manual_seed(SEED)
        
        model = NanoGPT().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=cfg["wd"])
        stochastic_decay = StochasticDistributionalDecay(gamma=cfg["stoch_gamma"])
        
        train_losses = []
        val_losses = []
        iters = []
        
        for iter_num in range(max_iters):
            if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
                losses = estimate_loss(model)
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}", flush=True)
                train_losses.append(losses['train'])
                val_losses.append(losses['val'])
                iters.append(iter_num)
                
            xb, yb = get_batch('train')
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            stochastic_decay.step(model)
            
        results[config_name] = {
            "iters": iters,
            "train_loss": train_losses,
            "val_loss": val_losses,
            "color": cfg["color"],
            "ls": cfg["ls"]
        }

    # -----------------------------------------------------------------------------
    # 4. Plotting
    # -----------------------------------------------------------------------------
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300, facecolor='white')
    fig.subplots_adjust(wspace=0.25)
    
    for config_name, res in results.items():
        ax1.plot(res["iters"], res["val_loss"], label=f'{config_name} (Val: {res["val_loss"][-1]:.3f})', 
                 color=res["color"], linestyle=res["ls"], linewidth=2.0)
                 
    ax1.set_title('(A) Validation Cross-Entropy Loss', fontsize=12, fontweight='bold', color='#111111')
    ax1.set_xlabel('Training Iteration', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Loss (Lower is Better)', fontsize=10, fontweight='bold')
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax1.legend(fontsize=8.5, loc='upper right', frameon=True)
    
    for config_name, res in results.items():
        gap = np.array(res["val_loss"]) - np.array(res["train_loss"])
        ax2.plot(res["iters"], gap, label=config_name, color=res["color"], linestyle=res["ls"], linewidth=2.0)
        
    ax2.set_title('(B) Generalization Gap (Val Loss - Train Loss)', fontsize=12, fontweight='bold', color='#111111')
    ax2.set_xlabel('Training Iteration', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Generalization Gap', fontsize=10, fontweight='bold')
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax2.legend(fontsize=8.5, loc='upper left', frameon=True)

    plt.suptitle('NanoGPT Training Dynamics: Decay Evaluation (MPS Accelerated)', fontsize=14, fontweight='bold', y=0.98)
    
    output_plot_path = os.path.join(os.path.dirname(__file__), "plot_nano.png")
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot successfully saved to: {output_plot_path}", flush=True)

if __name__ == "__main__":
    run_experiment()
