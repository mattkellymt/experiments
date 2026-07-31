import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. DATASET GENERATION
# ==========================================
dim = 32
num_samples = 30000

a = torch.randn(num_samples, dim)
b = torch.randn(num_samples, dim)

a = a / torch.norm(a, dim=1, keepdim=True)
b = b / torch.norm(b, dim=1, keepdim=True)

cos_sim = (a * b).sum(dim=1)

is_subtraction = (cos_sim < 0)
targets = torch.where(is_subtraction.unsqueeze(1), a - b, a * b)

train_size = int(0.8 * num_samples)
train_a, test_a = a[:train_size], a[train_size:]
train_b, test_b = b[:train_size], b[train_size:]
train_y, test_y = targets[:train_size], targets[train_size:]
train_sub, test_sub = is_subtraction[:train_size], is_subtraction[train_size:]

train_dataset = TensorDataset(train_a, train_b, train_y, train_sub)
test_dataset = TensorDataset(test_a, test_b, test_y, test_sub)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# ==========================================
# 2. 3-PHASE ARCHITECTURES (Phase 1 & 3 identical, Phase 2 varied)
#    All Phase 2 variants are ~4700 params
# ==========================================

# --- Phase 2 Modules ---

class DensePhase2(nn.Module):
    """Standard Dense: Linear(32, 72) -> GELU -> Linear(72, 32)"""
    def __init__(self, in_dim=32, hidden=72):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, in_dim))
    def forward(self, h):
        return self.net(h)

class GLUPhase2(nn.Module):
    """Gated Linear Unit: per-dimension gating enables emergent non-intersecting activation.
    Different inputs activate different dimension subsets -> non-intersecting subspaces emerge.
    h2 = Linear_out( Linear_value(h1) ⊙ σ(Linear_gate(h1)) )
    """
    def __init__(self, in_dim=32, hidden=48):
        super().__init__()
        self.value = nn.Linear(in_dim, hidden)
        self.gate = nn.Linear(in_dim, hidden)
        self.out = nn.Linear(hidden, in_dim)
    def forward(self, h):
        return self.out(self.value(h) * torch.sigmoid(self.gate(h)))

class GatedSkipPhase2(nn.Module):
    """Per-Dimension Gated Skip: each dimension independently routes to either
    pass-through (non-intersecting highway) or nonlinear transform.
    h2 = mask ⊙ h1 + (1-mask) ⊙ MLP(h1)  where mask = σ(Linear(h1))
    """
    def __init__(self, in_dim=32, hidden=56):
        super().__init__()
        self.mask_proj = nn.Linear(in_dim, in_dim)  # Per-dim mask
        self.transform = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, in_dim))
    def forward(self, h):
        mask = torch.sigmoid(self.mask_proj(h))  # (batch, 32) per-dim gate
        return mask * h + (1.0 - mask) * self.transform(h)

class BlockDiagPhase2(nn.Module):
    """Block-Diagonal (4 heads of 8D): Phase 1 LEARNS to organize its output
    so related features land in the same group for independent processing.
    Split 32D into 4 groups of 8, each through independent 8->70->8 MLP, concat back.
    """
    def __init__(self, in_dim=32, n_heads=4, head_hidden=70):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = in_dim // n_heads  # 8
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(self.head_dim, head_hidden), nn.GELU(), nn.Linear(head_hidden, self.head_dim))
            for _ in range(n_heads)
        ])
    def forward(self, h):
        chunks = h.chunk(self.n_heads, dim=1)  # 4 chunks of (batch, 8)
        processed = [head(chunk) for head, chunk in zip(self.heads, chunks)]
        return torch.cat(processed, dim=1)  # (batch, 32)

# --- Full 3-Phase Models (Phase 1 & 3 are identical across all) ---

class ThreePhaseModel(nn.Module):
    def __init__(self, phase2_module, in_dim=32, D=64):
        super().__init__()
        self.in_proj = nn.Linear(in_dim * 2, in_dim)
        self.phase1 = nn.Sequential(nn.Linear(in_dim, D), nn.GELU(), nn.Linear(D, in_dim))
        self.phase2 = phase2_module
        self.phase3 = nn.Sequential(nn.Linear(in_dim, D), nn.GELU(), nn.Linear(D, in_dim))

    def forward(self, a, b):
        x = self.in_proj(torch.cat([a, b], dim=1))
        h1 = self.phase1(x)
        h2 = self.phase2(h1)
        h3 = self.phase3(h2)
        return h3

# Build the 4 models
models = {
    'Standard Dense':   ThreePhaseModel(DensePhase2(dim, hidden=72)),
    'GLU (Per-Dim Gate)': ThreePhaseModel(GLUPhase2(dim, hidden=48)),
    'Gated Skip':       ThreePhaseModel(GatedSkipPhase2(dim, hidden=56)),
    'Block-Diagonal (4h)': ThreePhaseModel(BlockDiagPhase2(dim, n_heads=4, head_hidden=70)),
}

print("=" * 70)
print("PARAMETER COUNTS")
print("=" * 70)
for name, model in models.items():
    total = sum(p.numel() for p in model.parameters())
    p2_params = sum(p.numel() for p in model.phase2.parameters())
    print(f"  {name:25s} -> Total: {total:6d}  (Phase 2: {p2_params:5d})")

# ==========================================
# 3. TRAINING FUNCTION (200 EPOCHS for better convergence)
# ==========================================
def train_model(model, train_loader, test_loader, epochs=200):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    train_losses, test_losses = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_a, batch_b, batch_y, _ in train_loader:
            optimizer.zero_grad()
            pred = model(batch_a, batch_b)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_a.size(0)

        scheduler.step()
        train_losses.append(running_loss / len(train_loader.dataset))

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_a, batch_b, batch_y, _ in test_loader:
                pred = model(batch_a, batch_b)
                loss = criterion(pred, batch_y)
                test_loss += loss.item() * batch_a.size(0)
        test_losses.append(test_loss / len(test_loader.dataset))

    return train_losses, test_losses

num_epochs = 200

results = {}
for name, model in models.items():
    print(f"\nTraining {name} ({num_epochs} epochs)...")
    train_l, test_l = train_model(model, train_loader, test_loader, epochs=num_epochs)
    results[name] = {'train': train_l, 'test': test_l, 'model': model}

# ==========================================
# 4. SUBSET EVALUATION & ACTIVATION ANALYSIS
# ==========================================
criterion_none = nn.MSELoss(reduction='none')

subset_results = {}
for name, res in results.items():
    model = res['model']
    model.eval()
    sub_mse, mult_mse = [], []

    with torch.no_grad():
        for batch_a, batch_b, batch_y, batch_sub in test_loader:
            pred = model(batch_a, batch_b)
            mse = criterion_none(pred, batch_y).mean(dim=1)
            sub_mse.extend(mse[batch_sub].tolist())
            mult_mse.extend(mse[~batch_sub].tolist())

    subset_results[name] = {
        'sub_mse': np.mean(sub_mse),
        'mult_mse': np.mean(mult_mse),
        'total_mse': res['test'][-1]
    }

# --- Analyze GLU gate activations ---
glu_model = results['GLU (Per-Dim Gate)']['model']
glu_model.eval()
glu_gates_sub, glu_gates_mult = [], []
with torch.no_grad():
    for batch_a, batch_b, batch_y, batch_sub in test_loader:
        x = glu_model.in_proj(torch.cat([batch_a, batch_b], dim=1))
        h1 = glu_model.phase1(x)
        gate_vals = torch.sigmoid(glu_model.phase2.gate(h1))  # (batch, 48) per-dim gates
        for g, is_s in zip(gate_vals, batch_sub):
            if is_s:
                glu_gates_sub.append(g.numpy())
            else:
                glu_gates_mult.append(g.numpy())

glu_gates_sub = np.array(glu_gates_sub)   # (N_sub, 48)
glu_gates_mult = np.array(glu_gates_mult) # (N_mult, 48)

# --- Analyze Gated Skip mask activations ---
gs_model = results['Gated Skip']['model']
gs_model.eval()
gs_masks_sub, gs_masks_mult = [], []
with torch.no_grad():
    for batch_a, batch_b, batch_y, batch_sub in test_loader:
        x = gs_model.in_proj(torch.cat([batch_a, batch_b], dim=1))
        h1 = gs_model.phase1(x)
        mask_vals = torch.sigmoid(gs_model.phase2.mask_proj(h1))  # (batch, 32)
        for m, is_s in zip(mask_vals, batch_sub):
            if is_s:
                gs_masks_sub.append(m.numpy())
            else:
                gs_masks_mult.append(m.numpy())

gs_masks_sub = np.array(gs_masks_sub)
gs_masks_mult = np.array(gs_masks_mult)

# ==========================================
# 5. PRINT RESULTS
# ==========================================
print(f"\n{'='*75}")
print(f"FINAL RESULTS ({num_epochs} EPOCHS)")
print(f"{'='*75}")
for name, sr in subset_results.items():
    n_params = sum(p.numel() for p in models[name].parameters())
    p2_params = sum(p.numel() for p in models[name].phase2.parameters())
    print(f"  {name:25s} ({n_params}p, P2:{p2_params}p) -> Test: {sr['total_mse']:.6f} | Sub: {sr['sub_mse']:.6f} | Mult: {sr['mult_mse']:.6f}")

print(f"\nGLU Gate Analysis (48 dims):")
print(f"  Mean gate (subtraction):    {glu_gates_sub.mean():.4f}")
print(f"  Mean gate (multiplication): {glu_gates_mult.mean():.4f}")
print(f"  Per-dim gate std (sub):     {glu_gates_sub.mean(axis=0).std():.4f}")
print(f"  Per-dim gate std (mult):    {glu_gates_mult.mean(axis=0).std():.4f}")

# Count dimensions that are differentially active
gate_diff = np.abs(glu_gates_sub.mean(axis=0) - glu_gates_mult.mean(axis=0))
n_diff_dims = (gate_diff > 0.05).sum()
print(f"  Dims with >5% gate difference between tasks: {n_diff_dims}/48")

print(f"\nGated Skip Mask Analysis (32 dims):")
print(f"  Mean mask (subtraction):    {gs_masks_sub.mean():.4f} (higher = more pass-through)")
print(f"  Mean mask (multiplication): {gs_masks_mult.mean():.4f}")
mask_diff = np.abs(gs_masks_sub.mean(axis=0) - gs_masks_mult.mean(axis=0))
n_diff_mask = (mask_diff > 0.05).sum()
print(f"  Dims with >5% mask difference between tasks: {n_diff_mask}/32")

# ==========================================
# 6. VISUALIZATION (4-panel, no text blocks)
# ==========================================
plt.style.use('default')
fig, axs = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

colors = {
    'Standard Dense': '#ff7f0e',
    'GLU (Per-Dim Gate)': '#1f77b4',
    'Gated Skip': '#2ca02c',
    'Block-Diagonal (4h)': '#9467bd',
}

epochs_range = range(1, num_epochs + 1)

# Plot 1: Test Loss Convergence
for name, res in results.items():
    n_params = sum(p.numel() for p in models[name].parameters())
    axs[0, 0].plot(epochs_range, res['test'], color=colors[name], linewidth=2.5,
                   label=f'{name} ({n_params}p)')
axs[0, 0].set_title('Test Loss Convergence (All Phase 2 Variants)', fontsize=12, fontweight='bold')
axs[0, 0].set_xlabel('Epochs', fontsize=10)
axs[0, 0].set_ylabel('Mean Squared Error', fontsize=10)
axs[0, 0].grid(True, linestyle='--', alpha=0.5)
axs[0, 0].legend(frameon=True, fontsize=9)

# Plot 2: Subset Error Breakdown
categories = ['Subtraction (a - b)', 'Multiplication (a * b)']
x = np.arange(len(categories))
width = 0.2
offsets = [-1.5, -0.5, 0.5, 1.5]

for i, (name, sr) in enumerate(subset_results.items()):
    rects = axs[0, 1].bar(x + offsets[i] * width, [sr['sub_mse'], sr['mult_mse']], width,
                           label=name, color=colors[name], alpha=0.85)
    for rect in rects:
        h = rect.get_height()
        axs[0, 1].annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width()/2, h),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)

axs[0, 1].set_title('Subset Error Breakdown by Phase 2 Type', fontsize=12, fontweight='bold')
axs[0, 1].set_xticks(x)
axs[0, 1].set_xticklabels(categories, fontsize=10, fontweight='bold')
axs[0, 1].set_ylabel('MSE (Lower is better)', fontsize=10)
axs[0, 1].grid(True, linestyle='--', alpha=0.5, axis='y')
axs[0, 1].legend(frameon=True, fontsize=8)

# Plot 3: GLU Per-Dimension Gate Activation Heatmap (sorted by difference)
gate_mean_sub = glu_gates_sub.mean(axis=0)  # (48,)
gate_mean_mult = glu_gates_mult.mean(axis=0)  # (48,)
sort_idx = np.argsort(gate_mean_sub - gate_mean_mult)[::-1]

x_dims = np.arange(48)
axs[1, 0].bar(x_dims - 0.2, gate_mean_sub[sort_idx], 0.4, color='#2ca02c', alpha=0.75, label='Subtraction (a-b)')
axs[1, 0].bar(x_dims + 0.2, gate_mean_mult[sort_idx], 0.4, color='#9467bd', alpha=0.75, label='Multiplication (a*b)')
axs[1, 0].set_title('GLU: Per-Dimension Gate Activation (sorted by diff)', fontsize=12, fontweight='bold')
axs[1, 0].set_xlabel('Hidden Dimension (sorted)', fontsize=10)
axs[1, 0].set_ylabel('Mean Gate Activation σ(g)', fontsize=10)
axs[1, 0].legend(frameon=True)
axs[1, 0].grid(True, linestyle='--', alpha=0.5, axis='y')

# Plot 4: Gated Skip Per-Dimension Mask Comparison
mask_mean_sub = gs_masks_sub.mean(axis=0)  # (32,)
mask_mean_mult = gs_masks_mult.mean(axis=0)  # (32,)
sort_idx_m = np.argsort(mask_mean_sub - mask_mean_mult)[::-1]

x_dims_m = np.arange(32)
axs[1, 1].bar(x_dims_m - 0.2, mask_mean_sub[sort_idx_m], 0.4, color='#2ca02c', alpha=0.75, label='Subtraction (a-b)')
axs[1, 1].bar(x_dims_m + 0.2, mask_mean_mult[sort_idx_m], 0.4, color='#9467bd', alpha=0.75, label='Multiplication (a*b)')
axs[1, 1].set_title('Gated Skip: Per-Dim Mask (1=passthrough, 0=transform)', fontsize=12, fontweight='bold')
axs[1, 1].set_xlabel('Dimension (sorted)', fontsize=10)
axs[1, 1].set_ylabel('Mean Mask Value', fontsize=10)
axs[1, 1].legend(frameon=True)
axs[1, 1].grid(True, linestyle='--', alpha=0.5, axis='y')

plt.suptitle('Phase 2 Architecture Search: Enabling Emergent Non-Intersecting Subspaces', fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout()

output_dir = 'experiments/whitney_2'
plot_path = os.path.join(output_dir, 'plot.png')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()

print(f"\nSuccessfully generated and saved plot to {plot_path}")
