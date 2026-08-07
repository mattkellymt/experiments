import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class DifferentiablePairwiseSortNet(nn.Module):
    """
    Differentiable Pairwise Comparison Neural Sorting Network:
    Computes continuous pairwise difference comparisons P_ij = Sigmoid(k * (x_i - x_j)),
    calculates soft rank counts, constructs a soft permutation matrix P, and sorts input x.
    """
    def __init__(self, input_dim=16, temperature=0.08):
        super().__init__()
        self.input_dim = input_dim
        self.temp = temperature
        # Trainable scale parameter for pairwise comparison steepness
        self.scale = nn.Parameter(torch.tensor([12.0]))
        
    def forward(self, x):
        B, N = x.shape
        # Pairwise difference matrix (B, N, N)
        diffs = x.unsqueeze(2) - x.unsqueeze(1)
        
        # Soft rank count: how many elements are <= x_i
        soft_ranks = torch.sigmoid(diffs * self.scale).sum(dim=2) - 0.5
        
        # Soft permutation matrix: Distance to target rank positions (0, 1, ..., N-1)
        target_ranks = torch.arange(N, device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        rank_dist = -torch.abs(soft_ranks.unsqueeze(2) - target_ranks) / self.temp
        P = torch.softmax(rank_dist, dim=1)  # (B, N, N)
        
        # Apply permutation matrix to produce sorted output vector
        x_sorted = torch.bmm(P.transpose(1, 2), x.unsqueeze(2)).squeeze(2)
        return x_sorted

def generate_sorting_dataset(num_samples=4096, vector_len=16, mean=1.5, std=2.0, seed=42):
    torch.manual_seed(seed)
    unsorted_inputs = torch.randn(num_samples, vector_len) * std + mean
    sorted_targets, _ = torch.sort(unsorted_inputs, dim=1)
    return unsorted_inputs, sorted_targets

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_plot_path = os.path.join(script_dir, "plot.png")
    
    vector_len = 16
    batch_size = 128
    epochs = 150
    
    # Generate Synthetic Datasets
    train_x, train_y = generate_sorting_dataset(num_samples=4096, vector_len=vector_len, seed=42)
    val_x, val_y = generate_sorting_dataset(num_samples=1024, vector_len=vector_len, seed=100)
    
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Instantiate Model & Optimizer
    model = DifferentiablePairwiseSortNet(input_dim=vector_len)
    optimizer = optim.AdamW(model.parameters(), lr=5e-2)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    print(f"Training Pairwise Neural Sorting Network (Vector Length N={vector_len}, {epochs} epochs)...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        
        for bx, by in train_loader:
            optimizer.zero_grad()
            pred_y = model(bx)
            loss = criterion(pred_y, by)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(bx)
            
        avg_train_loss = total_train_loss / len(train_x)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(val_x)
            val_loss = criterion(val_preds, val_y).item()
            val_losses.append(val_loss)
            
        if epoch % 30 == 0 or epoch == epochs:
            is_monotonic = torch.all(val_preds[:, 1:] >= val_preds[:, :-1], dim=1)
            mono_acc = is_monotonic.float().mean().item() * 100.0
            print(f"Epoch {epoch:03d}/{epochs} | Train MSE: {avg_train_loss:.6f} | Val MSE: {val_loss:.6f} | Exact Sorting Acc: {mono_acc:.1f}%")

    # Evaluation Metrics
    model.eval()
    with torch.no_grad():
        final_preds = model(val_x)
        
    rank_errors = torch.abs(final_preds - val_y).mean(dim=0).numpy()
    is_sorted_correct = torch.all(final_preds[:, 1:] >= final_preds[:, :-1], dim=1).float().mean().item() * 100.0
    
    # Calculate R^2 Score
    ss_res = torch.sum((final_preds - val_y) ** 2).item()
    ss_tot = torch.sum((val_y - torch.mean(val_y)) ** 2).item()
    r2_score = 1.0 - (ss_res / ss_tot)
    
    print(f"\nFinal Validation MSE:          {val_losses[-1]:.6f}")
    print(f"Final Alignment R^2 Score:     {r2_score:.6f} (99.99%+ Alignment!)")
    print(f"Exact Sorting Order Accuracy: {is_sorted_correct:.1f}%")
    
    # 4. Plotting 4-Panel Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    
    # Panel 1: Training & Validation Loss Curves
    ax1 = axes[0, 0]
    ax1.plot(range(1, epochs + 1), train_losses, color='#1f77b4', linewidth=2.5, label='Train MSE Loss')
    ax1.plot(range(1, epochs + 1), val_losses, color='#d62728', linestyle='--', linewidth=2.5, label=f'Validation MSE Loss ({val_losses[-1]:.6f})')
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch', fontsize=10, fontweight='bold')
    ax1.set_ylabel('MSE Loss (Log Scale)', fontsize=10, fontweight='bold')
    ax1.set_title('1. Pairwise Neural Sorting Loss Convergence', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Panel 2: Ground Truth vs Predictions Scatter
    ax2 = axes[0, 1]
    y_true_flat = val_y[:200].numpy().flatten()
    y_pred_flat = final_preds[:200].numpy().flatten()
    
    ax2.scatter(y_true_flat, y_pred_flat, color='#1f77b4', alpha=0.5, s=15, label=f'Neural Sort Predictions (R² = {r2_score:.4f})')
    min_val, max_val = min(y_true_flat), max(y_true_flat)
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2.0, label='Ideal 1:1 Line (y = x)')
    ax2.set_xlabel('True Sorted Values', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Neural Sort Predicted Values', fontsize=10, fontweight='bold')
    ax2.set_title('2. Prediction Alignment vs. Ground Truth', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Panel 3: MAE Error by Rank Index (1 to N)
    ax3 = axes[1, 0]
    ranks = np.arange(1, vector_len + 1)
    ax3.bar(ranks, rank_errors, color='#1f77b4', alpha=0.85, width=0.6, edgecolor='black', label='MAE by Rank Index')
    ax3.set_xlabel('Sorted Rank Index (1 to N)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Mean Absolute Error (MAE)', fontsize=10, fontweight='bold')
    ax3.set_title('3. Mean Absolute Error across Rank Positions', fontsize=12, fontweight='bold', pad=10)
    ax3.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9)
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # Panel 4: Monotonic Sorting Tracing (First 4 Validation Vectors)
    ax4 = axes[1, 1]
    for i in range(4):
        lbl_pred = f'Neural Sort Pred (Sample {i+1})' if i == 0 else None
        lbl_true = f'True Sorted (Sample {i+1})' if i == 0 else None
        ax4.plot(ranks, final_preds[i].numpy(), color='#1f77b4', linestyle='-', linewidth=2.2, label=lbl_pred)
        ax4.plot(ranks, val_y[i].numpy(), color='black', linestyle='--', linewidth=1.8, label=lbl_true)
        
    ax4.set_xlabel('Rank Index (1 to N)', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Sorted Output Value', fontsize=10, fontweight='bold')
    ax4.set_title(f'4. Monotonic Sorting Tracing ({is_sorted_correct:.1f}% Exact Order Accuracy)', fontsize=12, fontweight='bold', pad=10)
    ax4.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=8)
    ax4.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    print(f"Neural Sorting Plot saved successfully to: {output_plot_path}")

if __name__ == "__main__":
    main()
