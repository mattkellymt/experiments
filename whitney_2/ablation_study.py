import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. DATASET
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
# 2. ABLATION: 4 Phase 2 variants that isolate skip vs. per-dim routing
# ==========================================

class DensePhase2(nn.Module):
    """Control: Standard Dense MLP, no skip connection."""
    def __init__(self, d=32, h=72):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))
    def forward(self, x):
        return self.net(x)

class SimpleResidualPhase2(nn.Module):
    """Ablation A: Fixed additive skip. h2 = h1 + MLP(h1).
    Tests: does the skip connection alone explain the gain?"""
    def __init__(self, d=32, h=72):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))
    def forward(self, x):
        return x + self.net(x)

class ScalarGatedSkipPhase2(nn.Module):
    """Ablation B: Single learned scalar blend. h2 = α*h1 + (1-α)*MLP(h1).
    Tests: does LEARNED blending help beyond a fixed skip?"""
    def __init__(self, d=32, h=72):
        super().__init__()
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))
    def forward(self, x):
        alpha = torch.sigmoid(self.alpha_logit)
        return alpha * x + (1.0 - alpha) * self.net(x)

class PerDimGatedSkipPhase2(nn.Module):
    """Full Model: Per-dimension input-dependent gating.
    h2 = σ(W·h1) ⊙ h1 + (1-σ(W·h1)) ⊙ MLP(h1).
    Tests: does per-dim INPUT-DEPENDENT routing (Whitney) help beyond scalar blend?"""
    def __init__(self, d=32, h=56):
        super().__init__()
        self.mask_proj = nn.Linear(d, d)  # 1056 extra params, MLP smaller to compensate
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))
    def forward(self, x):
        mask = torch.sigmoid(self.mask_proj(x))
        return mask * x + (1.0 - mask) * self.net(x)

class StaticPerDimGatedSkipPhase2(nn.Module):
    """Ablation C: Per-dimension gating but NOT input-dependent (learned static mask).
    h2 = σ(bias) ⊙ h1 + (1-σ(bias)) ⊙ MLP(h1).
    Tests: is it the per-dim structure or the input-dependence that matters?"""
    def __init__(self, d=32, h=70):
        super().__init__()
        self.mask_bias = nn.Parameter(torch.zeros(d))  # Per-dim but static
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))
    def forward(self, x):
        mask = torch.sigmoid(self.mask_bias).unsqueeze(0)  # (1, d) broadcast
        return mask * x + (1.0 - mask) * self.net(x)

# --- Full model wrapper ---
class ThreePhaseModel(nn.Module):
    def __init__(self, phase2_module, d=32, D=64):
        super().__init__()
        self.in_proj = nn.Linear(d * 2, d)
        self.phase1 = nn.Sequential(nn.Linear(d, D), nn.GELU(), nn.Linear(D, d))
        self.phase2 = phase2_module
        self.phase3 = nn.Sequential(nn.Linear(d, D), nn.GELU(), nn.Linear(D, d))
    def forward(self, a, b):
        x = self.in_proj(torch.cat([a, b], dim=1))
        return self.phase3(self.phase2(self.phase1(x)))

# Build 5 models
models = {
    '1. No Skip (Dense)':        ThreePhaseModel(DensePhase2(dim, 72)),
    '2. Fixed Skip (+residual)': ThreePhaseModel(SimpleResidualPhase2(dim, 72)),
    '3. Scalar Gated Skip':      ThreePhaseModel(ScalarGatedSkipPhase2(dim, 72)),
    '4. Static Per-Dim Skip':    ThreePhaseModel(StaticPerDimGatedSkipPhase2(dim, 70)),
    '5. Input-Dep Per-Dim Skip': ThreePhaseModel(PerDimGatedSkipPhase2(dim, 56)),
}

print("=" * 75)
print("ABLATION: SEPARATING SKIP CONNECTION vs. WHITNEY NON-INTERSECTION")
print("=" * 75)
for name, model in models.items():
    total = sum(p.numel() for p in model.parameters())
    p2 = sum(p.numel() for p in model.phase2.parameters())
    print(f"  {name:30s} -> Total: {total:6d}  (Phase 2: {p2:5d})")

# ==========================================
# 3. TRAINING
# ==========================================
def train_model(model, epochs=200):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        model.train()
        rl = 0.0
        for ba, bb, by, _ in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(ba, bb), by)
            loss.backward()
            optimizer.step()
            rl += loss.item() * ba.size(0)
        scheduler.step()
        train_losses.append(rl / len(train_loader.dataset))
        model.eval()
        tl = 0.0
        with torch.no_grad():
            for ba, bb, by, _ in test_loader:
                tl += criterion(model(ba, bb), by).item() * ba.size(0)
        test_losses.append(tl / len(test_loader.dataset))
    return train_losses, test_losses

results = {}
for name, model in models.items():
    print(f"\nTraining {name} ({200} epochs)...")
    tr, te = train_model(model, 200)
    results[name] = {'train': tr, 'test': te, 'model': model}

# ==========================================
# 4. EVALUATION
# ==========================================
criterion_none = nn.MSELoss(reduction='none')
subset_results = {}
for name, res in results.items():
    model = res['model']
    model.eval()
    sub_mse, mult_mse = [], []
    with torch.no_grad():
        for ba, bb, by, bs in test_loader:
            mse = criterion_none(model(ba, bb), by).mean(dim=1)
            sub_mse.extend(mse[bs].tolist())
            mult_mse.extend(mse[~bs].tolist())
    subset_results[name] = {'sub': np.mean(sub_mse), 'mult': np.mean(mult_mse), 'total': res['test'][-1]}

# Extract learned parameters from ablation models
# Scalar gate
scalar_model = results['3. Scalar Gated Skip']['model']
scalar_alpha = torch.sigmoid(scalar_model.phase2.alpha_logit).item()

# Static per-dim mask
static_model = results['4. Static Per-Dim Skip']['model']
static_mask = torch.sigmoid(static_model.phase2.mask_bias).detach().numpy()

# Input-dependent per-dim mask analysis
perdim_model = results['5. Input-Dep Per-Dim Skip']['model']
perdim_model.eval()
masks_sub, masks_mult = [], []
with torch.no_grad():
    for ba, bb, by, bs in test_loader:
        x = perdim_model.in_proj(torch.cat([ba, bb], dim=1))
        h1 = perdim_model.phase1(x)
        mask = torch.sigmoid(perdim_model.phase2.mask_proj(h1))
        for m, s in zip(mask, bs):
            if s:
                masks_sub.append(m.numpy())
            else:
                masks_mult.append(m.numpy())
masks_sub = np.array(masks_sub)
masks_mult = np.array(masks_mult)

# ==========================================
# 5. PRINT RESULTS
# ==========================================
print(f"\n{'='*80}")
print(f"ABLATION RESULTS")
print(f"{'='*80}")
for name, sr in subset_results.items():
    n = sum(p.numel() for p in models[name].parameters())
    print(f"  {name:30s} ({n}p) -> Test: {sr['total']:.6f} | Sub: {sr['sub']:.6f} | Mult: {sr['mult']:.6f}")

dense_mse = subset_results['1. No Skip (Dense)']['total']
for name, sr in subset_results.items():
    pct = (sr['total'] - dense_mse) / dense_mse * 100
    print(f"    {name:30s} -> {pct:+.1f}% vs Dense")

print(f"\nLearned Parameters:")
print(f"  Scalar gate α = {scalar_alpha:.4f} (0=all MLP, 1=all skip)")
print(f"  Static per-dim mask: mean={static_mask.mean():.4f}, std={static_mask.std():.4f}")
print(f"  Input-dep mask (sub):  mean={masks_sub.mean():.4f}")
print(f"  Input-dep mask (mult): mean={masks_mult.mean():.4f}")
print(f"  Mask difference: {masks_sub.mean() - masks_mult.mean():.4f}")

diff = np.abs(masks_sub.mean(axis=0) - masks_mult.mean(axis=0))
print(f"  Dims with >5% task-dependent difference: {(diff > 0.05).sum()}/32")
print(f"  Dims with >10% task-dependent difference: {(diff > 0.10).sum()}/32")

# ==========================================
# 6. VISUALIZATION
# ==========================================
fig, axs = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

colors = {
    '1. No Skip (Dense)':        '#ff7f0e',
    '2. Fixed Skip (+residual)': '#d62728',
    '3. Scalar Gated Skip':      '#8c564b',
    '4. Static Per-Dim Skip':    '#9467bd',
    '5. Input-Dep Per-Dim Skip': '#1f77b4',
}

# --- Panel 1: Convergence ---
for name, res in results.items():
    n = sum(p.numel() for p in models[name].parameters())
    axs[0, 0].plot(range(1, 201), res['test'], color=colors[name], linewidth=2.5,
                   label=f'{name} ({n}p)')
axs[0, 0].set_title('Ablation: Test Loss Convergence', fontsize=12, fontweight='bold')
axs[0, 0].set_xlabel('Epochs', fontsize=10)
axs[0, 0].set_ylabel('MSE', fontsize=10)
axs[0, 0].grid(True, linestyle='--', alpha=0.5)
axs[0, 0].legend(frameon=True, fontsize=8)

# --- Panel 2: Final MSE bar chart ---
names_short = ['No Skip\n(Dense)', 'Fixed\nSkip', 'Scalar\nGated', 'Static\nPer-Dim', 'Input-Dep\nPer-Dim']
total_mses = [sr['total'] for sr in subset_results.values()]
color_list = list(colors.values())
bars = axs[0, 1].bar(names_short, total_mses, color=color_list, alpha=0.85, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, total_mses):
    axs[0, 1].annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, val),
                       xytext=(0, 5), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
axs[0, 1].set_title('Final Test MSE: Isolating Each Component', fontsize=12, fontweight='bold')
axs[0, 1].set_ylabel('MSE (Lower is Better)', fontsize=10)
axs[0, 1].grid(True, linestyle='--', alpha=0.5, axis='y')

# Draw arrows/annotations showing the incremental effect
arrow_y = max(total_mses) * 1.15
axs[0, 1].set_ylim(0, max(total_mses) * 1.35)

# --- Panel 3: Static mask vs Input-Dependent mask per dimension ---
sort_idx = np.argsort(masks_sub.mean(axis=0) - masks_mult.mean(axis=0))[::-1]
x_dims = np.arange(32)

axs[1, 0].bar(x_dims - 0.3, masks_sub.mean(axis=0)[sort_idx], 0.3, color='#2ca02c', alpha=0.75, label='Sub (input-dep)')
axs[1, 0].bar(x_dims, masks_mult.mean(axis=0)[sort_idx], 0.3, color='#9467bd', alpha=0.75, label='Mult (input-dep)')
axs[1, 0].bar(x_dims + 0.3, static_mask[sort_idx], 0.3, color='#8c564b', alpha=0.6, label='Static mask')
axs[1, 0].set_title('Per-Dim Mask: Input-Dependent vs Static', fontsize=12, fontweight='bold')
axs[1, 0].set_xlabel('Dimension (sorted by sub-mult diff)', fontsize=10)
axs[1, 0].set_ylabel('Mask Value (1=skip, 0=transform)', fontsize=10)
axs[1, 0].legend(frameon=True, fontsize=9)
axs[1, 0].grid(True, linestyle='--', alpha=0.5, axis='y')

# --- Panel 4: Contribution breakdown ---
# Show the % improvement attributable to each component
baseline = subset_results['1. No Skip (Dense)']['total']
improvements = {}
prev = baseline
labels_contrib = []
values_contrib = []
colors_contrib = []

steps = [
    ('Skip Connection\nEffect', '2. Fixed Skip (+residual)', '#d62728'),
    ('+ Learned Scalar\nBlend', '3. Scalar Gated Skip', '#8c564b'),
    ('+ Per-Dim Static\nStructure', '4. Static Per-Dim Skip', '#9467bd'),
    ('+ Input-Dependent\nRouting (Whitney)', '5. Input-Dep Per-Dim Skip', '#1f77b4'),
]

cumulative = []
prev_val = baseline
for label, key, color in steps:
    curr = subset_results[key]['total']
    delta = prev_val - curr
    pct = delta / baseline * 100
    labels_contrib.append(label)
    values_contrib.append(pct)
    colors_contrib.append(color)
    cumulative.append(curr)
    prev_val = curr

bars = axs[1, 1].bar(labels_contrib, values_contrib, color=colors_contrib, alpha=0.85, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, values_contrib):
    sign = '+' if val > 0 else ''
    axs[1, 1].annotate(f'{sign}{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, max(val, 0)),
                       xytext=(0, 5), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')
axs[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axs[1, 1].set_title('Incremental Improvement Attribution', fontsize=12, fontweight='bold')
axs[1, 1].set_ylabel('% Improvement vs No Skip (Cumulative)', fontsize=10)
axs[1, 1].grid(True, linestyle='--', alpha=0.5, axis='y')

plt.suptitle('Ablation Study: Is It the Skip Connection or Whitney Non-Intersection?', fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout()

plot_path = 'experiments/whitney_2/ablation_plot.png'
plt.savefig(plot_path, bbox_inches='tight')
plt.close()
print(f"\nPlot saved to {plot_path}")
