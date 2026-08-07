import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE

# Set random seeds for complete reproducibility
np.random.seed(42)
torch.manual_seed(42)

# -----------------------------------------------------------------------------
# 1. High-Dimensional Dynamic On-The-Fly Generator
# -----------------------------------------------------------------------------
class DynamicHighDClusterGenerator:
    def __init__(self, K=8, d_in=8, seed=42):
        np.random.seed(seed)
        self.K = K
        self.d_in = d_in
        self.centroids = np.random.normal(loc=0.0, scale=3.0, size=(K, d_in))
        self.stds = np.clip(np.abs(np.random.normal(loc=0.6, scale=0.25, size=(K, d_in))), 0.25, 1.2)

    def sample_unsupervised_batch(self, batch_size=32):
        """
        Sample batch_size random points from random classes WITHOUT returning labels.
        """
        class_ids = np.random.choice(self.K, size=batch_size)
        X_list = []
        for k in class_ids:
            pt = np.random.normal(loc=self.centroids[k], scale=self.stds[k], size=(1, self.d_in))
            X_list.append(pt)
        X = np.vstack(X_list)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(class_ids, dtype=torch.long)

    def sample_test_dataset(self, num_per_class=32, seed=999):
        np.random.seed(seed)
        X_list, y_list = [], []
        for k in range(self.K):
            pts = np.random.normal(loc=self.centroids[k], scale=self.stds[k], size=(num_per_class, self.d_in))
            X_list.append(pts)
            y_list.append(np.full(num_per_class, k))
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        idx = np.arange(len(y))
        np.random.shuffle(idx)
        return torch.tensor(X[idx], dtype=torch.float32), torch.tensor(y[idx], dtype=torch.long)

# Data Augmentation Function for 8D Features (Creating Positive Pair Views)
def augment_views(X, aug_noise_std=0.25):
    view1 = X + torch.randn_like(X) * aug_noise_std
    view2 = X + torch.randn_like(X) * aug_noise_std
    return view1, view2

# -----------------------------------------------------------------------------
# 2. Pure Unsupervised SimCLR-Style InfoNCE Loss (ZERO LABELS)
# -----------------------------------------------------------------------------
def unsupervised_infonce_loss(z1, z2, temperature=0.15):
    """
    Pure Unsupervised InfoNCE Loss (SimCLR):
    Pulls 2 augmented views of the same sample together.
    Pushes all other samples in the batch apart.
    NO CLASS LABELS ARE USED.
    """
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0) # (2B, D)
    
    sim = torch.matmul(z, z.T) / temperature # (2B, 2B)
    mask_self = torch.eye(2 * B, device=z.device).bool()
    
    pos_sim = torch.cat([torch.diag(sim, B), torch.diag(sim, -B)], dim=0) # (2B,)
    
    logits = sim.masked_fill(mask_self, -1e9)
    exp_logits = torch.exp(logits).sum(dim=1)
    
    loss = -torch.log(torch.exp(pos_sim) / (exp_logits + 1e-8)).mean()
    return loss

class Unsupervised2DEncoder(nn.Module):
    def __init__(self, d_in=8, hidden_dim=64, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, x):
        z = self.encoder(x)
        return F.normalize(z, dim=-1) # Project to 2D unit hypersphere

# -----------------------------------------------------------------------------
# 3. Experiment Pipeline
# -----------------------------------------------------------------------------
def run_unsupervised_infonce_demo():
    K = 8
    d_in = 8
    generator = DynamicHighDClusterGenerator(K=K, d_in=d_in, seed=42)
    
    print("==========================================================================")
    print("PURE UNSUPERVISED INFONCE EXPERIMENT (SimCLR-Style, ZERO CLASS LABELS)")
    print("Input Dimension: 8D  |  Classes: K=8  |  Latent Output: 2D Unit Circle")
    print("==========================================================================")

    torch.manual_seed(42)
    encoder = Unsupervised2DEncoder(d_in=d_in, hidden_dim=64, latent_dim=2)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.015)
    
    loss_history = []
    steps = 300
    
    print("Pre-Training Encoder with Unsupervised InfoNCE (Zero Human Labels)...")
    for step in range(1, steps + 1):
        # Sample batch WITHOUT labels
        X_b, _ = generator.sample_unsupervised_batch(batch_size=32)
        v1, v2 = augment_views(X_b)
        
        optimizer.zero_grad()
        z1 = encoder(v1)
        z2 = encoder(v2)
        loss = unsupervised_infonce_loss(z1, z2, temperature=0.15)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        if step % 50 == 0 or step == 1:
            print(f"  Step {step:3d}/300 | Unsupervised InfoNCE Loss: {loss.item():.4f}")

    # Evaluate Self-Discovered Features via Linear Probe
    X_test, y_test = generator.sample_test_dataset(num_per_class=32, seed=999)
    encoder.eval()
    with torch.no_grad():
        z_test = encoder(X_test)
        
    print("\nTraining Linear Probe on Frozen Unsupervised Features...")
    linear_probe = nn.Linear(2, K)
    opt_probe = torch.optim.Adam(linear_probe.parameters(), lr=0.1)
    
    # Train linear probe on frozen features for 200 steps
    for _ in range(200):
        opt_probe.zero_grad()
        probe_loss = F.cross_entropy(linear_probe(z_test), y_test)
        probe_loss.backward()
        opt_probe.step()
        
    linear_probe.eval()
    with torch.no_grad():
        probe_acc = (linear_probe(z_test).argmax(dim=1) == y_test).float().mean().item() * 100.0
        
    print(f"\nLinear Probe Test Accuracy on Frozen Unsupervised Features: {probe_acc:.2f}%\n")

    # -------------------------------------------------------------------------
    # 4. Multi-Panel Visual Graphic Generation
    # -------------------------------------------------------------------------
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 12), dpi=300)
    
    class_colors_8 = ['#FF5376', '#00F5D4', '#FFEE55', '#7B2CBF', '#4D96FF', '#FF9F43', '#00BBF9', '#F15BB5']

    # -------------------------------------------------------------------------
    # Panel 1: Unsupervised InfoNCE Loss Curve
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(range(1, steps+1), loss_history, color='#00F5D4', linewidth=2.5, label='Unsupervised InfoNCE Loss')
    ax1.set_title("1. Unsupervised InfoNCE Loss Curve\n(Zero Class Labels Used During Pre-Training)", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Pre-Training Step (Batch Size = 32)", fontsize=10)
    ax1.set_ylabel("InfoNCE Loss", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.2)
    ax1.legend(loc='upper right', framealpha=0.85, fontsize=9)

    # -------------------------------------------------------------------------
    # Panel 2: t-SNE Projection of Raw 8D Input Space
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(2, 2, 2)
    tsne = TSNE(n_components=2, perplexity=15, random_state=42)
    X_test_tsne = tsne.fit_transform(X_test.numpy())
    
    for k in range(K):
        mask = (y_test.numpy() == k)
        ax2.scatter(X_test_tsne[mask, 0], X_test_tsne[mask, 1], color=class_colors_8[k], 
                    s=55, edgecolors='white', linewidth=0.6, alpha=0.9, label=f'Class {k}')
        
    ax2.set_title("2. Raw 8D Input Space (t-SNE Projection)\n(Before InfoNCE Unsupervised Pre-Training)", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("t-SNE Dimension 1", fontsize=10)
    ax2.set_ylabel("t-SNE Dimension 2", fontsize=10)
    ax2.legend(loc='upper right', framealpha=0.85, fontsize=7.5, ncol=2)
    ax2.grid(True, linestyle='--', alpha=0.2)

    # -------------------------------------------------------------------------
    # Panel 3: Self-Discovered 2D Latent Representation (Zero Labels!)
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(2, 2, 3)
    z_test_np = z_test.numpy()
    
    for k in range(K):
        mask = (y_test.numpy() == k)
        ax3.scatter(z_test_np[mask, 0], z_test_np[mask, 1], color=class_colors_8[k], 
                    s=60, edgecolors='white', linewidth=0.8, alpha=0.95, label=f'Class {k}')
        
    circle = plt.Circle((0, 0), 1.0, color='#888888', fill=False, linestyle='--', alpha=0.5)
    ax3.add_patch(circle)

    ax3.set_title("3. Self-Discovered 2D Latent Representation (Zero Labels!)\n(InfoNCE Automatically Organized 8 Classes on Unit Circle)", fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel("Latent Feature $z_1$", fontsize=10)
    ax3.set_ylabel("Latent Feature $z_2$", fontsize=10)
    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(-1.2, 1.2)
    ax3.legend(loc='upper right', framealpha=0.85, fontsize=7.5, ncol=2)
    ax3.grid(True, linestyle='--', alpha=0.2)

    # -------------------------------------------------------------------------
    # Panel 4: Linear Probe Confusion Matrix on Unsupervised Features
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(2, 2, 4)
    probe_preds = linear_probe(z_test).argmax(dim=1).numpy()
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y_test.numpy(), probe_preds):
        cm[t, p] += 1
        
    im4 = ax4.imshow(cm, cmap='viridis')
    cbar4 = fig.colorbar(im4, ax=ax4)
    cbar4.set_label("Number of Test Samples", fontsize=10)
    
    for i in range(K):
        for j in range(K):
            ax4.text(j, i, f"{cm[i, j]}", ha="center", va="center", 
                     color="white" if cm[i, j] < cm.max()/2 else "black",
                     fontsize=10, fontweight='bold')
            
    ax4.set_title(f"4. Linear Probe Test Set Confusion Matrix\n(Zero-Label Unsupervised Test Acc: {probe_acc:.2f}%)", fontsize=12, fontweight='bold', pad=10)
    ax4.set_xticks(range(K))
    ax4.set_yticks(range(K))
    ax4.set_xticklabels([f"C{k}" for k in range(K)])
    ax4.set_yticklabels([f"C{k}" for k in range(K)])

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "infonce_unsupervised_story.png")
    plt.savefig(output_path, dpi=300)
    print(f"Unsupervised visual story graphic saved successfully to {output_path}")

if __name__ == "__main__":
    run_unsupervised_infonce_demo()
