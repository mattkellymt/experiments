import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon MPS device!", flush=True)
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

vocab_size = 32
block_size = 32
n_embd = 64
n_head = 4
n_layer = 3
dropout = 0.0

micro_batch_size = 8
grad_accum_steps = vocab_size // micro_batch_size
max_iters = 2000
eval_interval = 100

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
# 2. Hard Noisy Synthetic Dataset
# -----------------------------------------------------------------------------
print("Generating noisy algorithmic dataset (25% Label Noise)...", flush=True)
seq = [np.random.randint(0, vocab_size) for _ in range(5)]
for i in range(5, 12000):
    if np.random.rand() < 0.25:
        seq.append(np.random.randint(0, vocab_size))
    else:
        val = (seq[i-3] * 2 + seq[i-5]) % vocab_size
        seq.append(val)
    
full_data = torch.tensor(seq, dtype=torch.long)
n = int(0.9 * len(full_data))
train_data = full_data[:n]
val_data = full_data[n:]

class_indices_train = defaultdict(list)
for i in range(block_size, len(train_data)):
    v = train_data[i].item()
    class_indices_train[v].append(i)

class_indices_val = defaultdict(list)
for i in range(block_size, len(val_data)):
    v = val_data[i].item()
    class_indices_val[v].append(i)

def get_uniform_batch(split='train'):
    d = train_data if split == 'train' else val_data
    indices_dict = class_indices_train if split == 'train' else class_indices_val
    
    batch_x, batch_y = [], []
    for v in range(vocab_size):
        idx = np.random.choice(indices_dict[v])
        batch_x.append(d[idx - block_size : idx])
        batch_y.append(d[idx - block_size + 1 : idx + 1])
        
    X = torch.stack(batch_x)
    Y = torch.stack(batch_y)
    
    perm = torch.randperm(vocab_size)
    return X[perm], Y[perm]

# -----------------------------------------------------------------------------
# 3. NanoGPT Architecture
# -----------------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * (C**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
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
            logits_last = logits[:, -1, :]  # (B, C)
            targets_last = targets[:, -1]   # (B)
            loss = F.cross_entropy(logits_last, targets_last)
        return logits, loss

# -----------------------------------------------------------------------------
# 4. Training Loop (Gamma Sweep)
# -----------------------------------------------------------------------------
def run_experiment():
    print(f"Starting Empirical Distributional Weight Noise Sweep", flush=True)
    
    # We test only Stochastic Decay variants
    gamma_values = [5e-5, 1e-4, 5e-4]
    colors = plt.cm.viridis(np.linspace(0, 1, len(gamma_values)))
    
    results = {}
    
    for i, gamma in enumerate(gamma_values):
        config_name = f"γ={gamma:.1e}"
        print(f"\nTraining with Stochastic Decay ({config_name})...", flush=True)
        torch.manual_seed(SEED)
        
        model = NanoGPT().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
        stochastic_decay = StochasticDistributionalDecay(gamma=gamma)
        
        train_losses = []
        val_losses = []
        iters = []
        
        for iter_num in range(max_iters):
            if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
                model.eval()
                with torch.no_grad():
                    val_loss_accum = 0.0
                    for _ in range(10):
                        X_val, Y_val = get_uniform_batch('val')
                        _, loss = model(X_val.to(device), Y_val.to(device))
                        val_loss_accum += loss.item()
                    val_losses.append(val_loss_accum / 10)
                    
                    tr_loss_accum = 0.0
                    for _ in range(10):
                        X_tr, Y_tr = get_uniform_batch('train')
                        _, loss = model(X_tr.to(device), Y_tr.to(device))
                        tr_loss_accum += loss.item()
                    train_losses.append(tr_loss_accum / 10)
                    
                print(f"step {iter_num}: train loss {train_losses[-1]:.4f}, val loss {val_losses[-1]:.4f}", flush=True)
                model.train()
                iters.append(iter_num)
                
            X, Y = get_uniform_batch('train')
            optimizer.zero_grad(set_to_none=True)
            
            for micro_step in range(grad_accum_steps):
                start = micro_step * micro_batch_size
                end = start + micro_batch_size
                
                xb = X[start:end].to(device)
                yb = Y[start:end].to(device)
                
                logits, loss = model(xb, yb)
                loss = loss / grad_accum_steps
                loss.backward()
                
            optimizer.step()
            stochastic_decay.step(model)
            
        results[config_name] = {
            "iters": iters,
            "train_loss": train_losses,
            "val_loss": val_losses,
            "color": colors[i],
            "gamma": gamma
        }
        
    # Write summary for README
    summary_path = os.path.join(os.path.dirname(__file__), "gamma_sweep_results.txt")
    with open(summary_path, "w") as f:
        for config_name, res in results.items():
            f.write(f"{config_name}: Final Val Loss = {res['val_loss'][-1]:.4f}\n")
        
    # Plotting
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300, facecolor='white')
    fig.subplots_adjust(wspace=0.25)
    
    for config_name, res in results.items():
        ax1.plot(res["iters"], res["val_loss"], label=f'{config_name} (Val: {res["val_loss"][-1]:.3f})', 
                 color=res["color"], linewidth=2.0)
                 
    ax1.set_title('(A) Validation Cross-Entropy Loss', fontsize=12, fontweight='bold', color='#111111')
    ax1.set_xlabel('Training Iteration', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Validation Loss', fontsize=10, fontweight='bold')
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax1.legend(fontsize=8.5, loc='upper right', frameon=True)
    
    for config_name, res in results.items():
        gap = np.array(res["val_loss"]) - np.array(res["train_loss"])
        ax2.plot(res["iters"], gap, label=config_name, color=res["color"], linewidth=2.0)
        
    ax2.set_title('(B) Generalization Gap (Val Loss - Train Loss)', fontsize=12, fontweight='bold', color='#111111')
    ax2.set_xlabel('Training Iteration', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Generalization Gap', fontsize=10, fontweight='bold')
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax2.legend(fontsize=8.5, loc='upper left', frameon=True)

    plt.suptitle('Empirical Distributional Weight Noise (EDWN): Gamma Hyperparameter Sweep', fontsize=14, fontweight='bold', y=0.98)
    
    output_plot_path = os.path.join(os.path.dirname(__file__), "plot_gamma_sweep.png")
    plt.savefig(output_plot_path, bbox_inches='tight')
    plt.close()
    print(f"\nPlot successfully saved to: {output_plot_path}", flush=True)

if __name__ == "__main__":
    run_experiment()
