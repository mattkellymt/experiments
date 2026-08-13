import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cpu")
print(f"Using compute device: {device}", flush=True)

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
            for param in model.parameters():
                if param.requires_grad and param.ndim > 1:  # Apply to weight matrices
                    shape = param.shape
                    
                    # 1. Random normal sample
                    R = torch.randn(shape, device=param.device, dtype=param.dtype)
                    
                    # 2. Normalize R to [-1, 1] globally
                    r_min, r_max = R.min(), R.max()
                    R_norm = 2.0 * (R - r_min) / (r_max - r_min + 1e-8) - 1.0
                        
                    # 3. Scale down by gamma
                    W_gamma = self.gamma * R_norm
                    
                    # 4. Complementary weight matrix
                    W_inv = 1.0 - W_gamma
                    
                    # 5. Measure mean and std of actual parameters
                    p_mean = param.mean()
                    p_std = param.std(unbiased=False) + 1e-8
                    
                    # 6. Sample matching distribution (Actual, Global)
                    S = torch.randn(shape, device=param.device, dtype=param.dtype) * p_std + p_mean
                    
                    # 7. Slow weighted combination parameter update
                    param.copy_(param * W_inv + S * W_gamma)

# -----------------------------------------------------------------------------
# 2. Hard Dataset Generation (High dimension, lots of noise, irrelevant features)
# -----------------------------------------------------------------------------
def generate_hard_dataset(num_samples=10000, in_dim=100):
    # 100 features, but only the first 10 contain the actual signal.
    X = torch.randn(num_samples, in_dim) * 2.0
    
    # Highly non-linear, chaotic target function
    Y = (
        torch.sin(X[:, 0] * X[:, 1]) * 2.0 +
        torch.cos(X[:, 2] ** 2) +
        torch.tanh(X[:, 3] + X[:, 4]) * 3.0 +
        (X[:, 5:10].sum(dim=1) ** 2) / 15.0 + 
        torch.exp(X[:, 10] / 4.0)
    )
    
    # Add significant label noise to induce severe overfitting in complex models
    noise = torch.randn(num_samples) * 3.0
    Y = Y + noise
    
    return X, Y.unsqueeze(1)

# -----------------------------------------------------------------------------
# 3. Over-parameterized Deep MLP
# -----------------------------------------------------------------------------
class OverparameterizedMLP(nn.Module):
    """A wide and deep network prone to overfitting on noisy datasets."""
    def __init__(self, in_dim=100, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, out_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# -----------------------------------------------------------------------------
# 4. Main Experiment
# -----------------------------------------------------------------------------
def run_experiment():
    print("Generating hard synthetic dataset (100D, 10 signal features, heavy noise)...", flush=True)
    X, Y = generate_hard_dataset(num_samples=12000, in_dim=100)
    
    train_X, train_Y = X[:8000], Y[:8000]
    val_X, val_Y = X[8000:], Y[8000:]
    
    train_dataset = TensorDataset(train_X, train_Y)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    decay_configs = {
        "No Decay": {"wd": 0.0, "stochastic_gamma": 0.0, "color": "#7f7f7f", "ls": "--"},
        "L2 Decay (λ=1e-2)": {"wd": 1e-2, "stochastic_gamma": 0.0, "color": "#d62728", "ls": "-"},
        "Stoch Decay (γ=1e-4)": {"wd": 0.0, "stochastic_gamma": 1e-4, "color": "#1f77b4", "ls": "-"},
        "Stoch Decay (γ=5e-4)": {"wd": 0.0, "stochastic_gamma": 5e-4, "color": "#2ca02c", "ls": "-"},
    }
    
    epochs = 150
    results = {}
    
    for config_name, cfg in decay_configs.items():
        print(f"\nTraining Model with {config_name}...", flush=True)
        torch.manual_seed(SEED)
        model = OverparameterizedMLP(in_dim=100, out_dim=1).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=cfg["wd"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        stochastic_decay = StochasticDistributionalDecay(gamma=cfg["stochastic_gamma"])
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * bx.size(0)
                
            stochastic_decay.step(model)
            scheduler.step()
            
            epoch_train_loss = running_loss / len(train_X)
            
            model.eval()
            with torch.no_grad():
                val_pred = model(val_X.to(device))
                epoch_val_loss = criterion(val_pred, val_Y.to(device)).item()
                
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            
            if epoch % 25 == 0 or epoch == epochs:
                print(f"  Epoch {epoch:03d}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}", flush=True)
                
        results[config_name] = {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "color": cfg["color"],
            "ls": cfg["ls"]
        }

    # -----------------------------------------------------------------------------
    # 5. Visualization Generation
    # -----------------------------------------------------------------------------
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300, facecolor='white')
    fig.subplots_adjust(wspace=0.25)
    
    # Panel A: Validation Loss
    for config_name, res in results.items():
        final_val = res["val_loss"][-1]
        label = f"{config_name} (Val MSE: {final_val:.2f})"
        ax1.plot(res["val_loss"], label=label, color=res["color"], linestyle=res["ls"], linewidth=2.0)
    
    ax1.set_yscale('log')
    ax1.set_title('(A) Validation MSE (Hard Overfitting Task)', fontsize=12, fontweight='bold', color='#111111')
    ax1.set_xlabel('Training Epoch', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Validation MSE Loss (Log Scale)', fontsize=10, fontweight='bold')
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax1.legend(fontsize=8.5, loc='upper right', frameon=True)
    
    # Panel B: Train vs Val Gap (Overfitting measure)
    for config_name, res in results.items():
        gap = np.array(res["val_loss"]) - np.array(res["train_loss"])
        ax2.plot(gap, label=config_name, color=res["color"], linestyle=res["ls"], linewidth=2.0)
        
    ax2.set_title('(B) Generalization Gap (Val Loss - Train Loss)', fontsize=12, fontweight='bold', color='#111111')
    ax2.set_xlabel('Training Epoch', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Generalization Gap', fontsize=10, fontweight='bold')
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax2.legend(fontsize=8.5, loc='upper left', frameon=True)

    plt.suptitle('Regularization Under Severe Overfitting Conditions', fontsize=14, fontweight='bold', y=0.98)
    
    output_plot_path = os.path.join(os.path.dirname(__file__), "plot_hard.png")
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot successfully saved to: {output_plot_path}", flush=True)

if __name__ == "__main__":
    run_experiment()
