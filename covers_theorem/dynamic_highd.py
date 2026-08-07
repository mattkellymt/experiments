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
# 1. Dynamic Generator with Null Class (Class 0 = Standard Normal Noise N(0, I))
# -----------------------------------------------------------------------------
class DynamicNullHighDClusterGenerator:
    """
    On-The-Fly Data Generator for K High-Dimensional Clusters:
    - Class 0: NULL CLASS (Unstructured Standard Normal Noise N(0, I8) at Origin).
    - Classes 1 to K-1: Structured Signal Clusters centered at random centroids.
    """
    def __init__(self, K=8, d_in=8, seed=42):
        np.random.seed(seed)
        self.K = K
        self.d_in = d_in
        
        # Centroids and Stds initialization
        self.centroids = np.zeros((K, d_in))
        self.stds = np.ones((K, d_in)) # Class 0 Null Class: Centroid=0, Std=1.0
        
        # Classes 1 to K-1: Structured Signal Clusters
        self.centroids[1:] = np.random.normal(loc=0.0, scale=3.5, size=(K-1, d_in))
        self.stds[1:] = np.clip(np.abs(np.random.normal(loc=0.6, scale=0.25, size=(K-1, d_in))), 0.25, 1.2)

    def sample_balanced_batch(self, batch_size_per_class=1):
        """
        Sample mini-batch with exactly 1 instance per class (including 1 Null Class sample).
        """
        X_list, y_list = [], []
        for k in range(self.K):
            pt = np.random.normal(loc=self.centroids[k], scale=self.stds[k], size=(batch_size_per_class, self.d_in))
            X_list.append(pt)
            y_list.append(np.full(batch_size_per_class, k))
            
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        
        idx = np.arange(len(y))
        np.random.shuffle(idx)
        return torch.tensor(X[idx], dtype=torch.float32), torch.tensor(y[idx], dtype=torch.long)

    def sample_test_dataset(self, num_per_class=32, seed=999):
        """
        Sample test dataset dynamically (256 total samples: 32 Null + 224 Signal samples).
        """
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

# -----------------------------------------------------------------------------
# 2. Modern Supervised Contrastive / Parametric 2D Latent Encoder
# -----------------------------------------------------------------------------
class SupConLatentEncoder(nn.Module):
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
        return F.normalize(z, dim=-1)

def supervised_contrastive_loss(z, y, temperature=0.1):
    sim_matrix = torch.matmul(z, z.T) / temperature
    labels_eq = (y.unsqueeze(0) == y.unsqueeze(1)).float()
    mask_self = torch.eye(z.shape[0], device=z.device).bool()
    labels_eq.masked_fill_(mask_self, 0)
    
    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    logits = sim_matrix - logits_max.detach()
    exp_logits = torch.exp(logits) * (~mask_self).float()
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (labels_eq * log_prob).sum(1) / (labels_eq.sum(1) + 1e-8)
    return -mean_log_prob_pos.mean()

# -----------------------------------------------------------------------------
# 3. Dynamic Training Pipeline & Graphic Generator
# -----------------------------------------------------------------------------
def run_dynamic_highd_experiment():
    K = 8       # 8 Classes (1 Null Class + 7 Signal Classes)
    d_in = 8    # 8D Input Space
    batch_size = K # Batch size = 8
    
    generator = DynamicNullHighDClusterGenerator(K=K, d_in=d_in, seed=42)
    
    print("==========================================================================")
    print("DYNAMIC HIGH-DIMENSIONAL EXPERIMENT WITH NULL CLASS INJECTION")
    print(f"Number of Classes K = {K} (Class 0: NULL N(0, I8), Classes 1-7: Signal)")
    print(f"Batch Size = {batch_size} (1 Null + 7 Signal per mini-batch)")
    print(f"Input Feature Dimension d_in = {d_in}D Space")
    print("==========================================================================")
    print(f"  Class 0 (NULL CLASS): Centroid = [0, 0, 0...] | Std = 1.00 (Unstructured Background Noise)")
    for k in range(1, K):
        print(f"  Class {k} (Signal Class): Centroid = [{generator.centroids[k, :3].round(2)}...] | Mean Std = {generator.stds[k].mean():.2f}")
    print("--------------------------------------------------------------------------")

    X_test, y_test = generator.sample_test_dataset(num_per_class=32, seed=999)

    # 1. Dynamic Classifier Training Loop
    torch.manual_seed(42)
    classifier_net = nn.Sequential(
        nn.Linear(d_in, 64),
        nn.GELU(),
        nn.Linear(64, K)
    )
    optimizer = torch.optim.Adam(classifier_net.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    steps = 100
    loss_history = []
    test_acc_history = []
    
    print("Training Classifier with Null Class Injected On-The-Fly...")
    for step in range(1, steps + 1):
        X_b, y_b = generator.sample_balanced_batch(batch_size_per_class=1)
        
        optimizer.zero_grad()
        loss = criterion(classifier_net(X_b), y_b)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        classifier_net.eval()
        with torch.no_grad():
            t_logits = classifier_net(X_test)
            t_acc = (t_logits.argmax(dim=1) == y_test).float().mean().item() * 100.0
            test_acc_history.append(t_acc)
        classifier_net.train()
        
        if step % 10 == 0 or step == 1:
            print(f"  Step {step:3d}/100 | Dynamic Loss: {loss.item():.4f} | Test Acc (Inc. Null): {t_acc:5.1f}%")

    classifier_net.eval()
    with torch.no_grad():
        test_logits = classifier_net(X_test)
        test_preds = torch.argmax(test_logits, dim=1)
        test_acc = (test_preds == y_test).float().mean().item() * 100.0
        
        null_mask = (y_test == 0)
        null_acc = (test_preds[null_mask] == 0).float().mean().item() * 100.0
        signal_acc = (test_preds[~null_mask] == y_test[~null_mask]).float().mean().item() * 100.0
        
    print(f"\nFinal Test Accuracy (Overall): {test_acc:.2f}%")
    print(f"  - Null Class (Class 0) Accuracy: {null_acc:.2f}%")
    print(f"  - Signal Classes (1-7) Accuracy: {signal_acc:.2f}%\n")

    # 2. Train Modern Supervised Contrastive 2D Encoder
    print("Training InfoNCE 2D Latent Encoder with Null Class Injected...")
    latent_encoder = SupConLatentEncoder(d_in=d_in, hidden_dim=64, latent_dim=2)
    opt_enc = torch.optim.Adam(latent_encoder.parameters(), lr=0.02)
    
    for enc_step in range(250):
        X_enc_b, y_enc_b = generator.sample_balanced_batch(batch_size_per_class=4)
        opt_enc.zero_grad()
        z_b = latent_encoder(X_enc_b)
        c_loss = supervised_contrastive_loss(z_b, y_enc_b)
        c_loss.backward()
        opt_enc.step()
        
    print("Contrastive 2D Encoder Training Complete!\n")

    # -------------------------------------------------------------------------
    # 3. Multi-Panel Visual Graphic Generation (dynamic_highd.png)
    # -------------------------------------------------------------------------
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 12), dpi=300)
    
    # Color scheme: Class 0 (Null Class) is distinct Bright White / Gray '#E0E0E0'
    class_colors = ['#FFFFFF', '#FF5376', '#00F5D4', '#FFEE55', '#7B2CBF', '#4D96FF', '#FF9F43', '#00BBF9']
    class_labels = ['Class 0 (NULL Noise)', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7']

    # Panel 1: Dynamic Loss & Test Accuracy
    ax1 = fig.add_subplot(2, 2, 1)
    window = 5
    loss_smooth = np.convolve(loss_history, np.ones(window)/window, mode='valid')
    
    ax1.plot(range(1, steps+1), loss_history, color='#FF5376', alpha=0.3, label='Raw Step Loss')
    ax1.plot(range(window, steps+1), loss_smooth, color='#FF5376', linewidth=2.5, label='Smoothed Loss (MA-5)')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(range(1, steps+1), test_acc_history, color='#00F5D4', linewidth=2.5, linestyle='--', label='Dynamic Test Accuracy')
    ax1_twin.set_ylabel("Test Accuracy (%)", color='#00F5D4', fontsize=10)
    ax1_twin.tick_params(axis='y', labelcolor='#00F5D4')
    ax1_twin.set_ylim(40, 105)

    ax1.set_title("1. Dynamic Training Dynamics (Null Class Injected)\n(Loss Reduction & Overall Test Accuracy Growth)", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Dynamic Step (Batch Size K=8)", fontsize=10)
    ax1.set_ylabel("Cross-Entropy Loss", color='#FF5376', fontsize=10)
    ax1.tick_params(axis='y', labelcolor='#FF5376')
    ax1.grid(True, linestyle='--', alpha=0.2)

    # Panel 2: t-SNE 2D Projection
    ax2 = fig.add_subplot(2, 2, 2)
    tsne = TSNE(n_components=2, perplexity=15, random_state=42)
    X_test_tsne = tsne.fit_transform(X_test.numpy())
    
    for k in range(K):
        mask = (y_test.numpy() == k)
        ax2.scatter(X_test_tsne[mask, 0], X_test_tsne[mask, 1], color=class_colors[k], 
                    s=65 if k==0 else 50, edgecolors='black' if k==0 else 'white', 
                    linewidth=1.2 if k==0 else 0.6, alpha=0.95, label=class_labels[k])
        
    ax2.set_title("2. Raw 8D Input Space with Null Class (t-SNE 2D Projection)\n(White = Null N(0, I) Noise Class at Center)", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("t-SNE Dimension 1", fontsize=10)
    ax2.set_ylabel("t-SNE Dimension 2", fontsize=10)
    ax2.legend(loc='upper right', framealpha=0.85, fontsize=7.5, ncol=2)
    ax2.grid(True, linestyle='--', alpha=0.2)

    # Panel 3: InfoNCE 2D Latent Representation (Null Anchor)
    ax3 = fig.add_subplot(2, 2, 3)
    latent_encoder.eval()
    with torch.no_grad():
        z_test = latent_encoder(X_test).numpy()
        
    for k in range(K):
        mask = (y_test.numpy() == k)
        ax3.scatter(z_test[mask, 0], z_test[mask, 1], color=class_colors[k], 
                    s=70 if k==0 else 55, edgecolors='black' if k==0 else 'white', 
                    linewidth=1.2 if k==0 else 0.8, alpha=0.95, label=class_labels[k])
        
    circle = plt.Circle((0, 0), 1.0, color='#888888', fill=False, linestyle='--', alpha=0.5)
    ax3.add_patch(circle)

    ax3.set_title("3. InfoNCE 2D Latent Unit Circle with Null Anchor\n(Null Class Equiangularly Equidistant on Sphere)", fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel("Latent Feature $z_1$", fontsize=10)
    ax3.set_ylabel("Latent Feature $z_2$", fontsize=10)
    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(-1.2, 1.2)
    ax3.legend(loc='upper right', framealpha=0.85, fontsize=7.5, ncol=2)
    ax3.grid(True, linestyle='--', alpha=0.2)

    # Panel 4: Confusion Matrix
    ax4 = fig.add_subplot(2, 2, 4)
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y_test.numpy(), test_preds.numpy()):
        cm[t, p] += 1
        
    im4 = ax4.imshow(cm, cmap='viridis')
    cbar4 = fig.colorbar(im4, ax=ax4)
    cbar4.set_label("Number of Test Samples", fontsize=10)
    
    for i in range(K):
        for j in range(K):
            ax4.text(j, i, f"{cm[i, j]}", ha="center", va="center", 
                     color="white" if cm[i, j] < cm.max()/2 else "black",
                     fontsize=10, fontweight='bold')
            
    ax4.set_title(f"4. Test Confusion Matrix (Null + 7 Signal Classes)\n(Overall Test Accuracy: {test_acc:.2f}%)", fontsize=12, fontweight='bold', pad=10)
    ax4.set_xticks(range(K))
    ax4.set_yticks(range(K))
    ax4.set_xticklabels(['Null'] + [f"S{k}" for k in range(1, K)])
    ax4.set_yticklabels(['Null'] + [f"S{k}" for k in range(1, K)])

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "dynamic_highd.png")
    plt.savefig(output_path, dpi=300)
    print(f"Visual graphic saved successfully to {output_path}")

if __name__ == "__main__":
    run_dynamic_highd_experiment()
