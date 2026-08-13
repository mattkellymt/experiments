import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Replicability and Seed Configuration
# -----------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cpu")
print(f"Using compute device: {device}", flush=True)

# -----------------------------------------------------------------------------
# 1. Non-Linear Target Function Definition (32 -> 32)
# -----------------------------------------------------------------------------
def target_function(x: torch.Tensor) -> torch.Tensor:
    """
    Non-linear coupled vector transformation f: R^32 -> R^32.
    
    Combines trigonometric, hyperbolic, and rational non-linearities
    with circular spatial shift coupling across the 32 dimensions.
    """
    shift_right = torch.roll(x, shifts=1, dims=-1)
    shift_left = torch.roll(x, shifts=-1, dims=-1)
    
    y = (
        torch.sin(2.0 * x) +
        0.5 * torch.cos(3.0 * shift_right) +
        torch.tanh(shift_left * x) +
        (shift_right / (1.0 + x**2))
    )
    return y

# -----------------------------------------------------------------------------
# 2. Deep Dense Neural Network (Input 32 -> Output 32, Depth = ceil(log2(32)) = 5)
# -----------------------------------------------------------------------------
class DenseDecayNet(nn.Module):
    """
    Dense Neural Network with depth = ceil(log2(32)) = 5 layers.
    
    Layer 1: Linear(32, 64)
    Layer 2: Linear(64, 128)
    Layer 3: Linear(128, 64)
    Layer 4: Linear(64, 32)
    Layer 5: Linear(32, 32)
    """
    def __init__(self, in_dim=32, out_dim=32):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.depth = math.ceil(math.log2(in_dim))  # ceil(log2(32)) = 5
        
        self.layers = nn.ModuleList([
            nn.Linear(32, 64),
            nn.Linear(64, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 32)
        ])
        self.act = nn.GELU()
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Activation for all but final layer
                x = self.act(x)
        return x

# -----------------------------------------------------------------------------
# 3. Stochastic Distributional Decay Mechanism
# -----------------------------------------------------------------------------
class StochasticDistributionalDecay:
    """
    Applies Stochastic Distributional Decay to neural network weights.
    
    Formula & Process:
    1. Draw a random normal sample R ~ N(0, I) matching parameter shape.
    2. Min-max normalize R to [-1, 1]: R_norm = 2 * (R - min(R)) / (max(R) - min(R)) - 1.
    3. Scale down by gamma: W_gamma = gamma * R_norm (capping delta to [-gamma, gamma]).
    4. Compute complementary inverse matrix: W_inv = 1 - W_gamma (so W_gamma + W_inv = 1).
    5. Measure mean mu_W and std sigma_W of actual learnable parameter tensor.
    6. Draw random sample S ~ N(mu_W, sigma_W^2) matching actual parameter distribution.
    7. Update parameters: W_new = W * W_inv + S * W_gamma.
    """
    def __init__(self, gamma: float = 0.001, symmetric: bool = True, sample_source: str = "actual", granularity: str = "global"):
        self.gamma = gamma
        self.symmetric = symmetric
        self.sample_source = sample_source
        self.granularity = granularity
        
    def step(self, model: nn.Module):
        if self.gamma <= 0:
            return
            
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad and param.ndim > 1:  # Apply to weight matrices
                    shape = param.shape
                    
                    # 1. Random normal sample
                    R = torch.randn(shape, device=param.device, dtype=param.dtype)
                    
                    # Compute min/max for normalization based on granularity
                    if self.granularity == "global":
                        r_min, r_max = R.min(), R.max()
                    elif self.granularity == "row":
                        r_min = R.min(dim=1, keepdim=True)[0]
                        r_max = R.max(dim=1, keepdim=True)[0]
                    elif self.granularity == "col":
                        r_min = R.min(dim=0, keepdim=True)[0]
                        r_max = R.max(dim=0, keepdim=True)[0]
                    else:
                        raise ValueError(f"Unknown granularity: {self.granularity}")
                    
                    # 2. Normalize R
                    if self.symmetric:
                        R_norm = 2.0 * (R - r_min) / (r_max - r_min + 1e-8) - 1.0
                    else:
                        R_norm = (R - r_min) / (r_max - r_min + 1e-8)
                        
                    # 3. Scale down by gamma
                    W_gamma = self.gamma * R_norm
                    
                    # 4. Complementary weight matrix (W_gamma + W_inv = 1)
                    W_inv = 1.0 - W_gamma
                    
                    # 5. Measure mean and std of actual parameters
                    if self.sample_source == "actual":
                        if self.granularity == "global":
                            p_mean = param.mean()
                            p_std = param.std(unbiased=False) + 1e-8
                        elif self.granularity == "row":
                            p_mean = param.mean(dim=1, keepdim=True)
                            p_std = param.std(dim=1, unbiased=False, keepdim=True) + 1e-8
                        elif self.granularity == "col":
                            p_mean = param.mean(dim=0, keepdim=True)
                            p_std = param.std(dim=0, unbiased=False, keepdim=True) + 1e-8
                        S = torch.randn(shape, device=param.device, dtype=param.dtype) * p_std + p_mean
                    elif self.sample_source == "normal":
                        S = torch.randn(shape, device=param.device, dtype=param.dtype)
                    else:
                        raise ValueError(f"Unknown sample source: {self.sample_source}")
                    
                    # 7. Slow weighted combination parameter update
                    param.copy_(param * W_inv + S * W_gamma)

# -----------------------------------------------------------------------------
# 4. Synthetic Data Generation
# -----------------------------------------------------------------------------
def generate_dataset(num_samples=10000, in_dim=32):
    X = torch.randn(num_samples, in_dim) * 1.5
    Y = target_function(X)
    return X, Y

def compute_model_param_stats(model: nn.Module):
    """Computes total parameter standard deviation across weight matrices."""
    all_weights = torch.cat([p.detach().flatten() for p in model.parameters() if p.ndim > 1])
    return all_weights.mean().item(), all_weights.std().item(), all_weights.norm().item()

# -----------------------------------------------------------------------------
# 5. Main Experiment & Training Execution
# -----------------------------------------------------------------------------
def run_experiment():
    print(f"Initializing Dense Architecture: input_dim=32, output_dim=32, depth={math.ceil(math.log2(32))}", flush=True)
    print("Generating synthetic 32->32 dataset...", flush=True)
    X, Y = generate_dataset(num_samples=10000, in_dim=32)
    
    train_X, train_Y = X[:8000], Y[:8000]
    val_X, val_Y = X[8000:], Y[8000:]
    
    train_dataset = TensorDataset(train_X, train_Y)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    # Weight decay configurations to benchmark
    decay_configs = {
        "No Decay": {"wd": 0.0, "stochastic_gamma": 0.0, "src": "actual", "gran": "global", "color": "#7f7f7f", "ls": "--"},
        "L2 Decay (λ=1e-2)": {"wd": 1e-2, "stochastic_gamma": 0.0, "src": "actual", "gran": "global", "color": "#d62728", "ls": "-"},
        "Stoch (Actual, Global)": {"wd": 0.0, "stochastic_gamma": 1e-4, "src": "actual", "gran": "global", "color": "#1f77b4", "ls": "-"},
        "Stoch (Normal, Global)": {"wd": 0.0, "stochastic_gamma": 1e-4, "src": "normal", "gran": "global", "color": "#ff7f0e", "ls": "-"},
        "Stoch (Actual, Row)": {"wd": 0.0, "stochastic_gamma": 1e-4, "src": "actual", "gran": "row", "color": "#2ca02c", "ls": "-"},
        "Stoch (Normal, Row)": {"wd": 0.0, "stochastic_gamma": 1e-4, "src": "normal", "gran": "row", "color": "#9467bd", "ls": "-"},
        "Stoch (Actual, Col)": {"wd": 0.0, "stochastic_gamma": 1e-4, "src": "actual", "gran": "col", "color": "#e377c2", "ls": "-"},
        "Stoch (Normal, Col)": {"wd": 0.0, "stochastic_gamma": 1e-4, "src": "normal", "gran": "col", "color": "#8c564b", "ls": "-"},
    }
    
    epochs = 100
    results = {}
    best_model = None
    best_config = None
    best_val_loss = float("inf")
    
    for config_name, cfg in decay_configs.items():
        print(f"\nTraining Model with {config_name}...", flush=True)
        torch.manual_seed(SEED)
        model = DenseDecayNet(in_dim=32, out_dim=32).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=cfg["wd"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        stochastic_decay = StochasticDistributionalDecay(
            gamma=cfg["stochastic_gamma"],
            sample_source=cfg["src"],
            granularity=cfg["gran"]
        )
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        param_means = []
        param_stds = []
        weight_norms = []
        
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
                
            # Apply stochastic distributional decay per epoch
            stochastic_decay.step(model)
            
            scheduler.step()
            epoch_train_loss = running_loss / len(train_X)
            
            model.eval()
            with torch.no_grad():
                val_pred = model(val_X.to(device))
                epoch_val_loss = criterion(val_pred, val_Y.to(device)).item()
                
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            
            p_mean, p_std, p_norm = compute_model_param_stats(model)
            param_means.append(p_mean)
            param_stds.append(p_std)
            weight_norms.append(p_norm)
            
            if epoch % 25 == 0 or epoch == epochs:
                print(f"  Epoch {epoch:03d}/{epochs} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f} | Param Std: {p_std:.4f}", flush=True)
                
        results[config_name] = {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "param_means": param_means,
            "param_stds": param_stds,
            "weight_norms": weight_norms,
            "model": model,
            "color": cfg["color"],
            "ls": cfg["ls"]
        }
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model = model
            best_config = config_name

    # Final Evaluation of Best Model
    best_model.eval()
    with torch.no_grad():
        val_X_dev, val_Y_dev = val_X.to(device), val_Y.to(device)
        test_pred = best_model(val_X_dev)
        test_mse = criterion(test_pred, val_Y_dev).item()
        
    print(f"\nFinal Best Model ({best_config}) Validation MSE: {test_mse:.6f}", flush=True)
    
    # -----------------------------------------------------------------------------
    # 6. Visualization Generation (Clean Light Mode, High Contrast & High Legibility)
    # -----------------------------------------------------------------------------
    plt.style.use('default')
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 7), dpi=300, facecolor='white')
    
    # Validation Loss across Epochs
    for config_name, res in results.items():
        final_val = res["val_loss"][-1]
        label = f"{config_name} (Val MSE: {final_val:.4f})"
        ax1.plot(res["val_loss"], label=label, color=res["color"], linestyle=res["ls"], linewidth=2.0)
    
    ax1.set_yscale('log')
    ax1.set_title('Validation MSE Loss Curves (Lower is Better)', fontsize=12, fontweight='bold', color='#111111', pad=10)
    ax1.set_xlabel('Training Epoch', fontsize=10, fontweight='bold', color='#333333')
    ax1.set_ylabel('Validation MSE Loss (Log Scale)', fontsize=10, fontweight='bold', color='#333333')
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7, color="#cccccc")
    ax1.legend(fontsize=8.5, loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#dddddd')
    ax1.set_facecolor('#fcfcfc')
    
    plt.suptitle('Deep Learning Decay Dynamics: Stochastic Distributional Decay vs. L2 Weight Decay', fontsize=14, fontweight='bold', color='#111111', y=0.96)
    
    output_plot_path = os.path.join(os.path.dirname(__file__), "plot.png")
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot successfully saved to: {output_plot_path}", flush=True)

if __name__ == "__main__":
    run_experiment()
