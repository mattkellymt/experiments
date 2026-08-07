import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# -----------------------------------------------------------------------------
# 1. Dynamic On-The-Fly High-Dimensional Cluster Generator
# -----------------------------------------------------------------------------
class DynamicHighDClusterGenerator:
    """
    On-The-Fly Data Generator for K High-Dimensional Clusters.
    Parameters (Centroids & Std Devs in d_in space) are generated from Normal distributions.
    Data samples are generated dynamically on demand without static dataset storage.
    """
    def __init__(self, K=8, d_in=8, seed=42):
        np.random.seed(seed)
        self.K = K
        self.d_in = d_in
        
        # 1. Draw 8D Centroids for each class from Normal(0, 3.0)
        self.centroids = np.random.normal(loc=0.0, scale=3.0, size=(K, d_in))
        
        # 2. Draw 8D Std Devs for each class from |Normal(0.6, 0.25)| bounded in [0.25, 1.2]
        self.stds = np.clip(np.abs(np.random.normal(loc=0.6, scale=0.25, size=(K, d_in))), 0.25, 1.2)

    def sample_balanced_batch(self, batch_size_per_class=1):
        """
        Dynamically sample a balanced mini-batch with exactly batch_size_per_class 
        instances per class (Total batch size = K * batch_size_per_class).
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
        Dynamically sample a test dataset on the fly using the exact same class parameters.
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
# 2. Modern Supervised Contrastive / Parametric 2D Latent Representation Encoder
# -----------------------------------------------------------------------------
class SupConLatentEncoder(nn.Module):
    """
    Modern Parametric 2D Encoder trained with Supervised Contrastive (InfoNCE) Loss.
    Maps high-dimensional inputs (d_in=8) into a clean, separated 2D Latent Space.
    """
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
    loss = -mean_log_prob_pos.mean()
    return loss

# -----------------------------------------------------------------------------
# 3. Dynamic Training Loop & Visual Graphic Generator
# -----------------------------------------------------------------------------
def run_dynamic_highd_experiment():
    K = 8       # 8 Classes (Power of 2)
    d_in = 8    # 8D Input Space
    batch_size = K # Batch size = 8 (1 instance per class)
    
    generator = DynamicHighDClusterGenerator(K=K, d_in=d_in, seed=42)
    
    print("==========================================================================")
    print("DYNAMIC HIGH-DIMENSIONAL ON-THE-FLY CLUSTER EXPERIMENT")
    print(f"Number of Classes K = {K} (Power of 2)")
    print(f"Batch Size = {batch_size} (1 instance per class per batch)")
    print(f"Input Feature Dimension d_in = {d_in}D Space")
    print("==========================================================================")
    for k in range(K):
        print(f"  Class {k}: Centroid = [{generator.centroids[k, :3].round(2)}...] | Mean Std = {generator.stds[k].mean():.2f}")
    print("--------------------------------------------------------------------------")

    # Sample a fixed dynamic test set for step-by-step evaluation
    X_test, y_test = generator.sample_test_dataset(num_per_class=32, seed=999)

    # 1. Dynamic On-The-Fly Training Loop for Classifier
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
    
    print("Training Classifier on Dynamically Generated Mini-Batches On-The-Fly...")
    for step in range(1, steps + 1):
        X_b, y_b = generator.sample_balanced_batch(batch_size_per_class=1)
        
        optimizer.zero_grad()
        loss = criterion(classifier_net(X_b), y_b)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        # Evaluate current test accuracy
        classifier_net.eval()
        with torch.no_grad():
            t_logits = classifier_net(X_test)
            t_acc = (t_logits.argmax(dim=1) == y_test).float().mean().item() * 100.0
            test_acc_history.append(t_acc)
        classifier_net.train()
        
        if step % 10 == 0 or step == 1:
            print(f"  Step {step:3d}/100 | Dynamic Loss: {loss.item():.4f} | Dynamic Test Acc: {t_acc:5.1f}%")

    # Final Classifier Evaluation
    classifier_net.eval()
    with torch.no_grad():
        test_logits = classifier_net(X_test)
        test_preds = torch.argmax(test_logits, dim=1)
        test_acc = (test_preds == y_test).float().mean().item() * 100.0
    print(f"\nClassifier Final Test Set Accuracy (Dynamically Sampled N=256): {test_acc:.2f}%\n")

    # 2. Train Modern Parametric 2D Supervised Contrastive Encoder
    print("Training Modern Parametric 2D Latent Representation Encoder (InfoNCE)...")
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
    # 3. Multi-Panel Visual Graphic Generation
    # -------------------------------------------------------------------------
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 12), dpi=300)
    
    class_colors_8 = ['#FF5376', '#00F5D4', '#FFEE55', '#7B2CBF', '#4D96FF', '#FF9F43', '#00BBF9', '#F15BB5']

    # -------------------------------------------------------------------------
    # Panel 1: Smooth Dynamic Training Dynamics (Loss & Test Accuracy)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(2, 2, 1)
    
    # Smooth moving average for loss
    window = 5
    loss_smooth = np.convolve(loss_history, np.ones(window)/window, mode='valid')
    
    ax1.plot(range(1, steps+1), loss_history, color='#FF5376', alpha=0.3, label='Raw Step Loss')
    ax1.plot(range(window, steps+1), loss_smooth, color='#FF5376', linewidth=2.5, label='Smoothed Loss (MA-5)')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(range(1, steps+1), test_acc_history, color='#00F5D4', linewidth=2.5, linestyle='--', label='Dynamic Test Accuracy')
    ax1_twin.set_ylabel("Test Accuracy (%)", color='#00F5D4', fontsize=10)
    ax1_twin.tick_params(axis='y', labelcolor='#00F5D4')
    ax1_twin.set_ylim(40, 105)

    ax1.set_title("1. Smooth Dynamic Training Dynamics\n(Loss Reduction & Test Accuracy Growth)", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Dynamic Step (Batch Size K=8)", fontsize=10)
    ax1.set_ylabel("Cross-Entropy Loss", color='#FF5376', fontsize=10)
    ax1.tick_params(axis='y', labelcolor='#FF5376')
    ax1.grid(True, linestyle='--', alpha=0.2)

    # -------------------------------------------------------------------------
    # Panel 2: t-SNE 2D Manifold Projection of Raw 8D Input Space
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(2, 2, 2)
    tsne = TSNE(n_components=2, perplexity=15, random_state=42)
    X_test_tsne = tsne.fit_transform(X_test.numpy())
    
    for k in range(K):
        mask = (y_test.numpy() == k)
        ax2.scatter(X_test_tsne[mask, 0], X_test_tsne[mask, 1], color=class_colors_8[k], 
                    s=55, edgecolors='white', linewidth=0.6, alpha=0.9, label=f'Class {k}')
        
    ax2.set_title("2. Raw 8D Input Space (t-SNE 2D Manifold Projection)\n(Overlapping 8D Gaussian Clusters)", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("t-SNE Dimension 1", fontsize=10)
    ax2.set_ylabel("t-SNE Dimension 2", fontsize=10)
    ax2.legend(loc='upper right', framealpha=0.85, fontsize=7.5, ncol=2)
    ax2.grid(True, linestyle='--', alpha=0.2)

    # -------------------------------------------------------------------------
    # Panel 3: Modern Supervised Contrastive 2D Latent Sphere Representation
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(2, 2, 3)
    latent_encoder.eval()
    with torch.no_grad():
        z_test = latent_encoder(X_test).numpy()
        
    for k in range(K):
        mask = (y_test.numpy() == k)
        ax3.scatter(z_test[mask, 0], z_test[mask, 1], color=class_colors_8[k], 
                    s=60, edgecolors='white', linewidth=0.8, alpha=0.95, label=f'Class {k}')
        
    circle = plt.Circle((0, 0), 1.0, color='#888888', fill=False, linestyle='--', alpha=0.5)
    ax3.add_patch(circle)

    ax3.set_title("3. Modern Supervised Contrastive 2D Latent Space (InfoNCE)\n(Parametric Neural Latent Unit Circle)", fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel("Latent Feature $z_1$", fontsize=10)
    ax3.set_ylabel("Latent Feature $z_2$", fontsize=10)
    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(-1.2, 1.2)
    ax3.legend(loc='upper right', framealpha=0.85, fontsize=7.5, ncol=2)
    ax3.grid(True, linestyle='--', alpha=0.2)

    # -------------------------------------------------------------------------
    # Panel 4: Test Confusion Matrix & Final Metrics
    # -------------------------------------------------------------------------
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
            
    ax4.set_title(f"4. Dynamically Sampled Test Confusion Matrix\n(Overall Test Accuracy: {test_acc:.2f}%)", fontsize=12, fontweight='bold', pad=10)
    ax4.set_xticks(range(K))
    ax4.set_yticks(range(K))
    ax4.set_xticklabels([f"C{k}" for k in range(K)])
    ax4.set_yticklabels([f"C{k}" for k in range(K)])

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "dynamic_highd_story.png")
    plt.savefig(output_path, dpi=300)
    print(f"Visual graphic saved successfully to {output_path}")

if __name__ == "__main__":
    run_dynamic_highd_experiment()
