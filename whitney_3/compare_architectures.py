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
    def __init__(self, num_blocks):
        super().__init__()
        layers = [MLP(32, 32, 32) for _ in range(num_blocks)]
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(32, 16)
        
    def forward(self, a, b):
        x = torch.cat([a, b], dim=1)
        return self.out(self.net(x))

class ArchWide(nn.Module):
    def __init__(self, wide_dim):
        super().__init__()
        self.net = nn.Sequential(
            MLP(32, 32, 32),
            MLP(32, wide_dim, 32)
        )
        self.out = nn.Linear(32, 16)
        
    def forward(self, a, b):
        x = torch.cat([a, b], dim=1)
        return self.out(self.net(x))

models = {
    'Deep (3 Blocks)': ArchDeep(num_blocks=3),
    'Wide (64d)': ArchWide(wide_dim=64),
    'Deep (5 Blocks)': ArchDeep(num_blocks=5),
    'Extra Wide (128d)': ArchWide(wide_dim=128)
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
    criterion_none = nn.MSELoss(reduction='none')
    
    train_losses, test_losses = [], []
    test_sub_losses, test_mult_losses = [], []
    
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
        sub_mse, mult_mse = [], []
        
        with torch.no_grad():
            for batch_a, batch_b, batch_y, batch_sub in test_loader:
                pred = model(batch_a, batch_b)
                loss = criterion(pred, batch_y)
                test_loss += loss.item() * batch_a.size(0)
                
                mse = criterion_none(pred, batch_y).mean(dim=1)
                sub_mse.extend(mse[batch_sub].tolist())
                mult_mse.extend(mse[~batch_sub].tolist())
                
        test_losses.append(test_loss / len(test_loader.dataset))
        test_sub_losses.append(np.mean(sub_mse))
        test_mult_losses.append(np.mean(mult_mse))
        
    return train_losses, test_losses, test_sub_losses, test_mult_losses

num_epochs = 150
results = {}
for name, model in models.items():
    print(f"\nTraining {name} ({num_epochs} epochs)...")
    train_l, test_l, sub_l, mult_l = train_model(model, train_loader, test_loader, epochs=num_epochs)
    results[name] = {
        'test': test_l, 
        'test_sub': sub_l, 
        'test_mult': mult_l,
        'model': model
    }

# ==========================================
# 4. FULL TEST SET EVALUATION
# ==========================================
criterion_none = nn.MSELoss(reduction='none')
subset_results = {}
raw_errors = {}

for name, res in results.items():
    model = res['model']
    model.eval()
    all_mse, all_sub, all_cos = [], [], []
    
    with torch.no_grad():
        for batch_a, batch_b, batch_y, batch_sub in test_loader:
            pred = model(batch_a, batch_b)
            mse = criterion_none(pred, batch_y).mean(dim=1)
            cos = (batch_a * batch_b).sum(dim=1)
            
            all_mse.extend(mse.tolist())
            all_sub.extend(batch_sub.tolist())
            all_cos.extend(cos.tolist())
            
    all_mse = np.array(all_mse)
    all_sub = np.array(all_sub, dtype=bool)
    all_cos = np.array(all_cos)
    
    subset_results[name] = {
        'sub': np.mean(all_mse[all_sub]),
        'mult': np.mean(all_mse[~all_sub]),
        'total': np.mean(all_mse)
    }
    
    raw_errors[name] = {
        'mse': all_mse,
        'is_sub': all_sub,
        'cos': all_cos
    }

# ==========================================
# 5. VISUALIZATION (4-Pane)
# ==========================================
plt.style.use('default')
fig, axs = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

names = list(models.keys())
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
epochs_range = range(1, num_epochs + 1)

# --- Panel 1: Subset Test Loss Convergence (Log Scale) ---
axs[0, 0].set_yscale('log')

for name, color in zip(names, colors):
    axs[0, 0].plot(epochs_range, results[name]['test_sub'], color=color, linestyle='-', linewidth=2, alpha=1.0, label=f'{name} (Sub)')
    axs[0, 0].plot(epochs_range, results[name]['test_mult'], color=color, linestyle='-', linewidth=2, alpha=0.3, label=f'{name} (Mult)')

axs[0, 0].set_title('Test Loss Convergence (Log Scale)', fontsize=12, fontweight='bold')
axs[0, 0].set_xlabel('Epochs')
axs[0, 0].set_ylabel('Mean Squared Error (Log Scale)')
axs[0, 0].grid(True, linestyle='--', alpha=0.5, which='both')
axs[0, 0].legend(frameon=True, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, ncol=1) # Move legend outside to declutter
# Actually wait, moving legend outside might clip it. I'll put it lower left but make font small.
axs[0, 0].legend(frameon=True, loc='lower left', fontsize=8, ncol=2)

# --- Panel 2: Final Subset Error Breakdown ---
categories = ['Subtraction (a - b)', 'Multiplication (a * b)']
x = np.arange(len(categories))
width = 0.2
offsets = [-1.5, -0.5, 0.5, 1.5]

for i, ((name, sr), color) in enumerate(zip(subset_results.items(), colors)):
    rects = axs[0, 1].bar(x + offsets[i] * width, [sr['sub'], sr['mult']], width, label=name, color=color, alpha=0.85)
    for rect in rects:
        h = rect.get_height()
        axs[0, 1].annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width()/2, h),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

axs[0, 1].set_title('Final Subset Error Breakdown', fontsize=12, fontweight='bold')
axs[0, 1].set_xticks(x)
axs[0, 1].set_xticklabels(categories, fontweight='bold')
axs[0, 1].set_ylabel('MSE (Lower is better)')
axs[0, 1].grid(True, linestyle='--', alpha=0.5, axis='y')
axs[0, 1].legend(frameon=True, fontsize=9)

# --- Panel 3: Binned Average Error vs Cosine Similarity ---
bins = np.linspace(-1, 1, 31)
bin_centers = (bins[:-1] + bins[1:]) / 2

for name, color in zip(names, colors):
    re = raw_errors[name]
    cos_vals = re['cos']
    mse_vals = re['mse']
    
    binned_mse = []
    for i in range(len(bins)-1):
        mask = (cos_vals >= bins[i]) & (cos_vals < bins[i+1])
        if mask.sum() > 0:
            binned_mse.append(mse_vals[mask].mean())
        else:
            binned_mse.append(np.nan)
            
    axs[1, 0].plot(bin_centers, binned_mse, color=color, marker='o', markersize=4, linestyle='-', linewidth=2, label=name)

axs[1, 0].axvline(0, color='red', linestyle='--', linewidth=1.5, label='Decision Boundary (0)')
axs[1, 0].set_title('Average Error vs Cosine Similarity (Binned)', fontsize=12, fontweight='bold')
axs[1, 0].set_xlabel('Cosine Similarity Bins')
axs[1, 0].set_ylabel('Mean MSE inside Bin')
axs[1, 0].grid(True, linestyle='--', alpha=0.5)
axs[1, 0].legend(frameon=True, fontsize=9)

# Cap the Y-axis to ignore extreme single-point spikes if they happen
max_binned_mse = np.nanmax(binned_mse) * 1.2
if np.isnan(max_binned_mse) or max_binned_mse <= 0:
    max_binned_mse = 0.02
axs[1, 0].set_ylim(0, max_binned_mse)

# --- Panel 4: Error vs Cosine Similarity (Scatter Plot) ---
for name, color in zip(names, colors):
    re = raw_errors[name]
    axs[1, 1].scatter(re['cos'], re['mse'], color=color, alpha=0.1, s=4, label=name)

axs[1, 1].axvline(0, color='red', linestyle='--', linewidth=1.5, label='Decision Boundary (0)')
axs[1, 1].set_title('Prediction Error vs. Input Cosine Similarity (Scatter)', fontsize=12, fontweight='bold')
axs[1, 1].set_xlabel('Cosine Similarity')
axs[1, 1].set_ylabel('Mean Squared Error (Per Sample)')

# Dynamically set y-limit to match roughly the 98th percentile to exclude massive outliers
all_mse_combined = np.concatenate([raw_errors[n]['mse'] for n in names])
max_val = np.percentile(all_mse_combined, 98) * 1.5
axs[1, 1].set_ylim(-0.001, max_val) 
axs[1, 1].grid(True, linestyle='--', alpha=0.5)

# Match Panel 3 Y-axis to Panel 4 for direct comparison
axs[1, 0].set_ylim(-0.001, max_val)

# Fix scatter legend opacity
leg = axs[1, 1].legend(frameon=True, fontsize=9)
for lh in leg.legend_handles:
    if hasattr(lh, 'set_alpha'): 
        lh.set_alpha(1.0)
        
plt.suptitle('Architecture Comparison: Depth vs Width Scaling', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

output_dir = '/Users/matt/projects/techstack/experiments/whitney_3'
plot_path = os.path.join(output_dir, 'architectures_comparison.png')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()

print(f"\nSuccessfully generated and saved 4-pane plot to {plot_path}")
