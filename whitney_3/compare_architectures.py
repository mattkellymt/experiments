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
dim = 16
num_samples = 30000

# Input a and b are 16d
a = torch.randn(num_samples, dim)
b = torch.randn(num_samples, dim)

# Normalize
a = a / torch.norm(a, dim=1, keepdim=True)
b = b / torch.norm(b, dim=1, keepdim=True)

cos_sim = (a * b).sum(dim=1)
is_subtraction = (cos_sim < 0)

# Target is 16d: a-b or a*b based on cosine similarity
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
# 2. ARCHITECTURES
# ==========================================
class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, out_dim),
            nn.GELU()
        )
    def forward(self, x):
        return self.net(x)

class ArchDeep(nn.Module):
    """
    3 MLPs in series, all 32->32->32. 
    Total 6 blocks of 32x32 weights.
    Maintains 32d representation space throughout.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            MLP(32, 32, 32),
            MLP(32, 32, 32),
            MLP(32, 32, 32)
        )
        self.out = nn.Linear(32, 16)
        
    def forward(self, a, b):
        x = torch.cat([a, b], dim=1) # 16d + 16d = 32d input
        return self.out(self.net(x))

class ArchWide(nn.Module):
    """
    1 MLP (32->32->32) + 1 MLP (32->64->32).
    Total 2 blocks of 32x32 + 2 blocks of 32x64 weights.
    Expands representation space to 64d to allow non-intersecting features.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            MLP(32, 32, 32),
            MLP(32, 64, 32)
        )
        self.out = nn.Linear(32, 16)
        
    def forward(self, a, b):
        x = torch.cat([a, b], dim=1)
        return self.out(self.net(x))

models = {
    'Deep (Constant 32d)': ArchDeep(),
    'Wide (Expands to 64d)': ArchWide()
}

print("=" * 70)
print("PARAMETER COUNTS")
print("=" * 70)
for name, model in models.items():
    total = sum(p.numel() for p in model.parameters())
    print(f"  {name:25s} -> Total: {total:6d}")

# ==========================================
# 3. TRAINING
# ==========================================
def train_model(model, train_loader, test_loader, epochs=150):
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

num_epochs = 150
results = {}
for name, model in models.items():
    print(f"\nTraining {name} ({num_epochs} epochs)...")
    train_l, test_l = train_model(model, train_loader, test_loader, epochs=num_epochs)
    results[name] = {'train': train_l, 'test': test_l, 'model': model}

# ==========================================
# 4. SUBSET EVALUATION
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
        'sub': np.mean(sub_mse),
        'mult': np.mean(mult_mse),
        'total': res['test'][-1]
    }

print(f"\n{'='*75}")
print(f"FINAL RESULTS ({num_epochs} EPOCHS)")
print(f"{'='*75}")
for name, sr in subset_results.items():
    n_params = sum(p.numel() for p in models[name].parameters())
    print(f"  {name:25s} ({n_params}p) -> Test: {sr['total']:.6f} | Sub: {sr['sub']:.6f} | Mult: {sr['mult']:.6f}")

# ==========================================
# 5. VISUALIZATION
# ==========================================
plt.style.use('default')
fig, axs = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

colors = ['#1f77b4', '#ff7f0e']
epochs_range = range(1, num_epochs + 1)

# Plot 1: Convergence
for (name, res), color in zip(results.items(), colors):
    n_params = sum(p.numel() for p in res['model'].parameters())
    axs[0].plot(epochs_range, res['test'], color=color, linewidth=2.5,
                   label=f'{name} ({n_params}p)')
axs[0].set_title('Test Loss Convergence', fontsize=12, fontweight='bold')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Mean Squared Error')
axs[0].grid(True, linestyle='--', alpha=0.5)
axs[0].legend(frameon=True)

# Plot 2: Subset Breakdown
categories = ['Subtraction (a - b)', 'Multiplication (a * b)']
x = np.arange(len(categories))
width = 0.35

for i, ((name, sr), color) in enumerate(zip(subset_results.items(), colors)):
    rects = axs[1].bar(x + (i - 0.5) * width, [sr['sub'], sr['mult']], width, label=name, color=color, alpha=0.85)
    for rect in rects:
        h = rect.get_height()
        axs[1].annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width()/2, h),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

axs[1].set_title('Subset Error Breakdown', fontsize=12, fontweight='bold')
axs[1].set_xticks(x)
axs[1].set_xticklabels(categories, fontweight='bold')
axs[1].set_ylabel('MSE (Lower is better)')
axs[1].grid(True, linestyle='--', alpha=0.5, axis='y')
axs[1].legend(frameon=True)

plt.suptitle('Architecture Comparison: Constant 32d vs Wide 64d', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

output_dir = '/Users/matt/projects/techstack/experiments/whitney_3'
plot_path = os.path.join(output_dir, 'architectures_comparison.png')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()

print(f"\nSuccessfully generated and saved plot to {plot_path}")
