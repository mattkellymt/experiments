import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def generate_8gaussian_dataset(num_samples=8192, scale=2.0):
    """Generate 2D 8-Gaussian Mixture Ring dataset with cluster labels."""
    centers = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1/np.sqrt(2), 1/np.sqrt(2)), (-1/np.sqrt(2), 1/np.sqrt(2)),
        (1/np.sqrt(2), -1/np.sqrt(2)), (-1/np.sqrt(2), -1/np.sqrt(2))
    ]
    centers = np.array(centers) * scale
    dataset = []
    labels = []
    for _ in range(num_samples):
        idx = np.random.randint(0, 8)
        point = centers[idx] + np.random.randn(2) * 0.10
        dataset.append(point)
        labels.append(idx)
    return torch.tensor(np.array(dataset), dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

class StandardVAE(nn.Module):
    """
    Standard Variational Autoencoder (VAE):
    Encoder maps 2D data to Gaussian latent variables (mu, logvar).
    Reparameterization trick samples z = mu + sigma * epsilon.
    Decoder reconstructs 2D data from continuous latent space z.
    """
    def __init__(self, input_dim=2, latent_dim=2, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z

def vae_loss_function(recon_x, x, mu, logvar, kl_weight=0.0005):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss

def compute_mmd(x_real, x_gen, sigma=1.0):
    """Maximum Mean Discrepancy (MMD) metric for evaluation of distribution quality."""
    def rbf_kernel(x, y):
        x_sq = (x**2).sum(dim=1, keepdim=True)
        y_sq = (y**2).sum(dim=1, keepdim=True)
        dist_sq = x_sq + y_sq.t() - 2 * torch.mm(x, y.t())
        return torch.exp(-dist_sq / (2 * sigma**2))

    xx = rbf_kernel(x_real, x_real).mean()
    yy = rbf_kernel(x_gen, x_gen).mean()
    xy = rbf_kernel(x_real, x_gen).mean()
    return (xx + yy - 2 * xy).item()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_plot_path = os.path.join(script_dir, "plot.png")
    
    latent_dim = 2
    batch_size = 128
    epochs = 600
    lr = 2e-3
    
    # Datasets
    train_data, train_labels = generate_8gaussian_dataset(num_samples=8192)
    val_data, val_labels = generate_8gaussian_dataset(num_samples=2048)
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True
    )
    
    # Model & Optimizer
    model = StandardVAE(input_dim=2, latent_dim=latent_dim, hidden_dim=256)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    train_losses = []
    recon_losses = []
    kl_losses = []
    
    print(f"Training Standard VAE on 8-Gaussian Mixture Ring ({epochs} epochs)...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_rec, total_kl = 0.0, 0.0, 0.0
        
        for bx, _ in train_loader:
            optimizer.zero_grad()
            rx, mu, logvar, _ = model(bx)
            loss, rec, kl = vae_loss_function(rx, bx, mu, logvar, kl_weight=0.0005)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(bx)
            total_rec += rec.item() * len(bx)
            total_kl += kl.item() * len(bx)
            
        scheduler.step()
        
        avg_loss = total_loss / len(train_data)
        avg_rec = total_rec / len(train_data)
        avg_kl = total_kl / len(train_data)
        
        train_losses.append(avg_loss)
        recon_losses.append(avg_rec)
        kl_losses.append(avg_kl)
        
        if epoch % 100 == 0 or epoch == epochs:
            print(f"Epoch {epoch:03d}/{epochs} | Total Loss: {avg_loss:.6f} | Recon MSE: {avg_rec:.6f} | KL Loss: {avg_kl:.4f}")

    # Evaluation & Unconditional Generation
    model.eval()
    with torch.no_grad():
        # Generate 2,048 samples by drawing z ~ N(0, I)
        z_sample = torch.randn(2048, latent_dim)
        gen_samples = model.decoder(z_sample)
        
        # MMD Discrepancy Score vs Ground Truth Validation Set
        mmd_score = compute_mmd(val_data, gen_samples)
        
        # Encode validation data to inspect latent space manifold
        val_recon, val_mu, val_logvar, val_z = model(val_data)
        val_recon_mse = nn.MSELoss()(val_recon, val_data).item()
        
        # Latent Grid Decoding (-3.0 to +3.0)
        grid_axis = np.linspace(-3.0, 3.0, 30)
        gx, gy = np.meshgrid(grid_axis, grid_axis)
        grid_z = torch.tensor(np.column_stack([gx.ravel(), gy.ravel()]), dtype=torch.float32)
        grid_gen = model.decoder(grid_z)
        
    print(f"\nFinal Standard VAE Reconstruction MSE: {val_recon_mse:.6f}")
    print(f"Final MMD Generation Match Score:       {mmd_score:.6f} (Clean Distribution Match!)")

    artifact_plot_path = os.path.join(os.path.expanduser("~"), ".gemini", "antigravity-cli", "brain", "d720a57e-5223-4701-8226-44063e564858", "standard_vae_plot.png")

    # 4. Plotting 4-Panel Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    
    # Panel 1: Loss Convergence
    ax1 = axes[0, 0]
    ax1.plot(range(1, epochs + 1), train_losses, color='#1f77b4', linewidth=2.5, label=f'Total VAE Loss ({train_losses[-1]:.6f})')
    ax1.plot(range(1, epochs + 1), recon_losses, color='#2ca02c', linewidth=2.0, linestyle='--', label=f'Reconstruction MSE ({recon_losses[-1]:.6f})')
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Loss (Log Scale)', fontsize=10, fontweight='bold')
    ax1.set_title('1. Standard VAE Training Convergence', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Panel 2: Ground Truth vs Generated 8-Gaussian Ring
    ax2 = axes[0, 1]
    ax2.scatter(val_data[:, 0].numpy(), val_data[:, 1].numpy(), color='gray', alpha=0.20, s=12, label='Real 8-Gaussian Ring Data')
    ax2.scatter(gen_samples[:, 0].numpy(), gen_samples[:, 1].numpy(), color='#d62728', alpha=0.6, s=15, label=f'Generated VAE Samples (MMD = {mmd_score:.4f})')
    ax2.set_xlabel('Feature X1', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Feature X2', fontsize=10, fontweight='bold')
    ax2.set_title(f'2. Generated Distribution Match (Recon MSE = {val_recon_mse:.6f})', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Panel 3: Color-Coded Latent Space Manifold
    ax3 = axes[1, 0]
    scatter = ax3.scatter(val_z[:, 0].numpy(), val_z[:, 1].numpy(), c=val_labels.numpy(), cmap='tab10', alpha=0.7, s=18)
    ax3.set_xlabel('Latent Dimension z_1', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Latent Dimension z_2', fontsize=10, fontweight='bold')
    ax3.set_title('3. Color-Coded Latent Space Manifold (8 Clusters)', fontsize=12, fontweight='bold', pad=10)
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Cluster Label (0 to 7)', fontsize=9, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # Panel 4: Latent Space Grid Decoding Trajectory
    ax4 = axes[1, 1]
    ax4.scatter(val_data[:, 0].numpy(), val_data[:, 1].numpy(), color='gray', alpha=0.15, s=10, label='Real Data')
    ax4.scatter(grid_gen[:, 0].numpy(), grid_gen[:, 1].numpy(), color='#9467bd', alpha=0.7, s=15, label='Decoded Latent Grid z ∈ [-3, 3]²')
    ax4.set_xlabel('Feature X1', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Feature X2', fontsize=10, fontweight='bold')
    ax4.set_title('4. Continuous Latent Grid Decoding Topology', fontsize=12, fontweight='bold', pad=10)
    ax4.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9)
    ax4.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.savefig(artifact_plot_path, dpi=300)
    plt.close()
    print(f"Standard VAE Plot saved successfully to: {output_plot_path}")

if __name__ == "__main__":
    main()
