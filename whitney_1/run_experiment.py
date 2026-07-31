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
num_samples = 25000

# Generate pairs of 32D vectors
a = torch.randn(num_samples, dim)
b = torch.randn(num_samples, dim)

# Normalize vectors to unit length
a = a / torch.norm(a, dim=1, keepdim=True)
b = b / torch.norm(b, dim=1, keepdim=True)

# Compute cosine similarity
cos_sim = (a * b).sum(dim=1)

# Target Function:
# If cos_sim < 0 (dissimilar) -> Target = a - b (linear subtraction, non-intersecting bypass)
# If cos_sim >= 0 (similar) -> Target = a * b (element-wise product, requires intersecting mix)
is_subtraction = (cos_sim < 0)
targets = torch.where(is_subtraction.unsqueeze(1), a - b, a * b)

# Create train/test splits (80% train, 20% test)
train_size = int(0.8 * num_samples)
train_a, test_a = a[:train_size], a[train_size:]
train_b, test_b = b[:train_size], b[train_size:]
train_y, test_y = targets[:train_size], targets[train_size:]
train_sub, test_sub = is_subtraction[:train_size], is_subtraction[train_size:]
test_cos_sim = cos_sim[train_size:]

train_dataset = TensorDataset(train_a, train_b, train_y, train_sub)
test_dataset = TensorDataset(test_a, test_b, test_y, test_sub)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# ==========================================
# 2. ARCHITECTURES (2 PARAMETER-MATCHED 64D MODELS)
# ==========================================

# Model A: Standard Dense 64D MLP (14,917 parameters)
class StandardDense64D(nn.Module):
    def __init__(self, dim=32, hidden_dim=65):
        super().__init__()
        self.fc1 = nn.Linear(dim * 2, hidden_dim)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.GELU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.act3 = nn.GELU()
        self.out = nn.Linear(hidden_dim, dim)

    def forward(self, a, b):
        x = torch.cat([a, b], dim=1) # (batch, 64)
        h1 = self.act1(self.fc1(x))
        h2 = self.act2(self.fc2(h1))
        h3 = self.act3(self.fc3(h2))
        return self.out(h3)

# Model B: Whitney 64D Model (14,689 parameters - Half-Split Initialization & Channel Routing)
class Whitney64DModel(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        # Independent lower/upper stream layers
        self.stream_a1 = nn.Linear(dim, dim)
        self.stream_a2 = nn.Linear(dim, dim)
        self.stream_b1 = nn.Linear(dim, dim)
        self.stream_b2 = nn.Linear(dim, dim)
        
        # Cross-channel gating & mixing
        self.cross_gate = nn.Linear(dim * 2, 1)
        self.cross_mix1 = nn.Linear(dim * 2, dim * 2)
        self.cross_mix2 = nn.Linear(dim * 2, dim * 2)
        
        self.act = nn.GELU()
        self.out = nn.Linear(dim * 2, dim)

    def forward(self, a, b, return_gate=False):
        # Half-split non-intersecting initial channels
        ha1 = self.act(self.stream_a1(a))
        hb1 = self.act(self.stream_b1(b))
        ha2 = self.act(self.stream_a2(ha1))
        hb2 = self.act(self.stream_b2(hb1))
        
        x_unmixed = torch.cat([ha2, hb2], dim=1)
        x_concat = torch.cat([a, b], dim=1)
        gate = torch.sigmoid(self.cross_gate(x_concat))
        x_mix = self.act(self.cross_mix2(self.act(self.cross_mix1(x_concat))))
        
        x_64 = x_unmixed + gate * x_mix
        out = self.out(x_64)
        
        if return_gate:
            return out, gate
        return out

mA = StandardDense64D(dim, hidden_dim=65)
mB = Whitney64DModel(dim)

n_paramsA = sum(p.numel() for p in mA.parameters())
n_paramsB = sum(p.numel() for p in mB.parameters())

print(f"Model A (Standard Dense 64D MLP) Parameters: {n_paramsA}")
print(f"Model B (Whitney 64D Model) Parameters     : {n_paramsB}")

# ==========================================
# 3. TRAINING FUNCTION (120 EPOCHS)
# ==========================================
def train_model(model, train_loader, test_loader, epochs=120):
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
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_a, batch_b, batch_y, _ in test_loader:
                pred = model(batch_a, batch_b)
                loss = criterion(pred, batch_y)
                test_loss += loss.item() * batch_a.size(0)
        test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(test_loss)
        
    return train_losses, test_losses

num_epochs = 120

print(f"\nTraining Model A: Standard Dense 64D MLP ({num_epochs} epochs)...")
tA_train, tA_test = train_model(mA, train_loader, test_loader, epochs=num_epochs)

print(f"Training Model B: Whitney 64D Model ({num_epochs} epochs)...")
tB_train, tB_test = train_model(mB, train_loader, test_loader, epochs=num_epochs)

# ==========================================
# 4. EVALUATION & PROBING
# ==========================================
mA.eval()
mB.eval()

criterion_none = nn.MSELoss(reduction='none')

sub_mse_A, mult_mse_A = [], []
sub_mse_B, mult_mse_B = [], []
gates_sub, gates_mult = [], []
all_gates, all_cos_sims = [], []

with torch.no_grad():
    for batch_a, batch_b, batch_y, batch_sub in test_loader:
        # Model A
        predA = mA(batch_a, batch_b)
        mseA = criterion_none(predA, batch_y).mean(dim=1)
        sub_mse_A.extend(mseA[batch_sub].tolist())
        mult_mse_A.extend(mseA[~batch_sub].tolist())
        
        # Model B & Gate extraction
        predB, gB = mB(batch_a, batch_b, return_gate=True)
        mseB = criterion_none(predB, batch_y).mean(dim=1)
        sub_mse_B.extend(mseB[batch_sub].tolist())
        mult_mse_B.extend(mseB[~batch_sub].tolist())
        
        gB_flat = gB.squeeze().tolist()
        batch_cos = (batch_a * batch_b).sum(dim=1).tolist()
        
        all_gates.extend(gB_flat)
        all_cos_sims.extend(batch_cos)
        
        for gate_val, is_s in zip(gB_flat, batch_sub.tolist()):
            if is_s:
                gates_sub.append(gate_val)
            else:
                gates_mult.append(gate_val)

# Averages
mA_sub, mA_mult = np.mean(sub_mse_A), np.mean(mult_mse_A)
mB_sub, mB_mult = np.mean(sub_mse_B), np.mean(mult_mse_B)

g_sub_mean, g_mult_mean = np.mean(gates_sub), np.mean(gates_mult)

print(f"\n================ FINAL 2-MODEL COMPARISON RESULTS ================")
print(f"Model A (Standard Dense 64D MLP, {n_paramsA} params) -> Test Loss: {tA_test[-1]:.6f} | Sub MSE: {mA_sub:.6f} | Mult MSE: {mA_mult:.6f}")
print(f"Model B (Whitney 64D Model, {n_paramsB} params)     -> Test Loss: {tB_test[-1]:.6f} | Sub MSE: {mB_sub:.6f} | Mult MSE: {mB_mult:.6f}")

# ==========================================
# 5. PURE PLOTS VISUALIZATION (NO TEXT BLOCKS)
# ==========================================
plt.style.use('default')
fig, axs = plt.subplots(2, 2, figsize=(15, 11), dpi=300)

epochs_range = range(1, len(tA_test) + 1)

# Plot 1: Test Loss Convergence (MSE)
axs[0, 0].plot(epochs_range, tA_test, color='#ff7f0e', linewidth=2.5, label=f'Standard Dense 64D ({n_paramsA} params)')
axs[0, 0].plot(epochs_range, tB_test, color='#1f77b4', linewidth=2.5, label=f'Whitney 64D Model ({n_paramsB} params)')
axs[0, 0].set_title('Test Loss Convergence (MSE)', fontsize=12, fontweight='bold')
axs[0, 0].set_xlabel('Epochs', fontsize=10)
axs[0, 0].set_ylabel('Mean Squared Error', fontsize=10)
axs[0, 0].grid(True, linestyle='--', alpha=0.5)
axs[0, 0].legend(frameon=True)

# Plot 2: Subset Error Breakdown (Subtraction vs Multiplication)
categories = ['Subtraction (a - b)', 'Multiplication (a * b)']
x = np.arange(len(categories))
width = 0.35

rects1 = axs[0, 1].bar(x - width/2, [mA_sub, mA_mult], width, label=f'Standard Dense 64D ({n_paramsA}p)', color='#ff7f0e', alpha=0.85)
rects2 = axs[0, 1].bar(x + width/2, [mB_sub, mB_mult], width, label=f'Whitney 64D ({n_paramsB}p)', color='#1f77b4', alpha=0.85)

axs[0, 1].set_title('Subset Error Breakdown', fontsize=12, fontweight='bold')
axs[0, 1].set_xticks(x)
axs[0, 1].set_xticklabels(categories, fontsize=10, fontweight='bold')
axs[0, 1].set_ylabel('MSE (Lower is better)', fontsize=10)
axs[0, 1].grid(True, linestyle='--', alpha=0.5, axis='y')
axs[0, 1].legend(frameon=True)

for rect in rects1 + rects2:
    h = rect.get_height()
    axs[0, 1].annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width()/2, h),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

# Plot 3: Distribution of Learned Intersection Gate Values in Whitney 64D Model
axs[1, 0].hist(gates_sub, bins=40, alpha=0.65, color='#2ca02c', density=True, label='a - b (Linear Subtraction)')
axs[1, 0].hist(gates_mult, bins=40, alpha=0.65, color='#9467bd', density=True, label='a * b (Element-wise Product)')
axs[1, 0].axvline(g_sub_mean, color='#2ca02c', linestyle='--', linewidth=2, label=f'Sub Mean: {g_sub_mean:.3f}')
axs[1, 0].axvline(g_mult_mean, color='#9467bd', linestyle='--', linewidth=2, label=f'Mult Mean: {g_mult_mean:.3f}')

axs[1, 0].set_title('Learned 64D Intersection Gate Distribution', fontsize=12, fontweight='bold')
axs[1, 0].set_xlabel('Learned Gate Value (0 = Non-Intersecting Bypass, 1 = Full Intersection)', fontsize=10)
axs[1, 0].set_ylabel('Density', fontsize=10)
axs[1, 0].grid(True, linestyle='--', alpha=0.5)
axs[1, 0].legend(frameon=True)

# Plot 4: Scatter Plot of Learned Gate vs Cosine Similarity cos(a, b)
idx_sample = np.random.choice(len(all_cos_sims), size=min(2000, len(all_cos_sims)), replace=False)
sample_cos = np.array(all_cos_sims)[idx_sample]
sample_gate = np.array(all_gates)[idx_sample]

axs[1, 1].scatter(sample_cos[sample_cos < 0], sample_gate[sample_cos < 0], color='#2ca02c', alpha=0.4, s=15, label='a - b (cos < 0)')
axs[1, 1].scatter(sample_cos[sample_cos >= 0], sample_gate[sample_cos >= 0], color='#9467bd', alpha=0.4, s=15, label='a * b (cos >= 0)')
axs[1, 1].axvline(0, color='gray', linestyle=':', alpha=0.7)

axs[1, 1].set_title('Learned Intersection Gate vs Cosine Similarity cos(a, b)', fontsize=12, fontweight='bold')
axs[1, 1].set_xlabel('Vector Cosine Similarity cos(a, b)', fontsize=10)
axs[1, 1].set_ylabel('Learned 64D Intersection Gate Value', fontsize=10)
axs[1, 1].grid(True, linestyle='--', alpha=0.5)
axs[1, 1].legend(frameon=True)

plt.suptitle(f'64D Up-Projection Comparison: Standard Dense ({n_paramsA}p) vs Whitney Model ({n_paramsB}p)', fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout()

output_dir = 'experiments/whitney_1'
plot_path = os.path.join(output_dir, 'plot.png')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()

print(f"\nSuccessfully generated and saved plot to {plot_path}")
