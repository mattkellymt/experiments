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
    """Generate 2D 8-Gaussian Mixture Ring dataset."""
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

class Generator(nn.Module):
    """
    GAN Generator G(z):
    Maps 2D random Gaussian noise z ~ N(0, I) to synthetic 2D data points.
    """
    def __init__(self, latent_dim=2, hidden_dim=256, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    """
    GAN Discriminator D(x):
    Classifies whether 2D data points are real (1) or synthetic fake (0).
    """
    def __init__(self, input_dim=2, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        return self.net(x)

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
    lr = 2e-4
    
    # Datasets
    train_data, train_labels = generate_8gaussian_dataset(num_samples=8192)
    val_data, val_labels = generate_8gaussian_dataset(num_samples=2048)
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True
    )
    
    # Instantiate Models & Optimizers
    netG = Generator(latent_dim=latent_dim, hidden_dim=256)
    netD = Discriminator(input_dim=2, hidden_dim=256)
    
    optG = optim.AdamW(netG.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)
    optD = optim.AdamW(netD.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)
    
    criterion = nn.BCEWithLogitsLoss()
    
    g_losses = []
    d_losses = []
    
    print(f"Training Generative Adversarial Network (GAN) on 8-Gaussian Mixture Ring ({epochs} epochs)...")
    
    for epoch in range(1, epochs + 1):
        netG.train()
        netD.train()
        total_g_loss = 0.0
        total_d_loss = 0.0
        
        for bx, _ in train_loader:
            bs = len(bx)
            
            # -------------------------------------------------------------
            # Train Discriminator D
            # -------------------------------------------------------------
            optD.zero_grad()
            z_noise = torch.randn(bs, latent_dim)
            fake_x = netG(z_noise).detach()
            
            out_real = netD(bx)
            out_fake = netD(fake_x)
            
            loss_d_real = criterion(out_real, torch.ones_like(out_real))
            loss_d_fake = criterion(out_fake, torch.zeros_like(out_fake))
            loss_d = (loss_d_real + loss_d_fake) / 2.0
            
            loss_d.backward()
            optD.step()
            
            # -------------------------------------------------------------
            # Train Generator G
            # -------------------------------------------------------------
            optG.zero_grad()
            z_noise = torch.randn(bs, latent_dim)
            fake_x = netG(z_noise)
            out_fake = netD(fake_x)
            
            loss_g = criterion(out_fake, torch.ones_like(out_fake))
            loss_g.backward()
            optG.step()
            
            total_d_loss += loss_d.item() * bs
            total_g_loss += loss_g.item() * bs
            
        avg_d_loss = total_d_loss / len(train_data)
        avg_g_loss = total_g_loss / len(train_data)
        
        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)
        
        if epoch % 100 == 0 or epoch == epochs:
            print(f"Epoch {epoch:03d}/{epochs} | Discriminator Loss: {avg_d_loss:.4f} | Generator Loss: {avg_g_loss:.4f}")

    # Evaluation & Unconditional Generation
    netG.eval()
    netD.eval()
    
    with torch.no_grad():
        # Generate 2,048 synthetic samples
        z_sample = torch.randn(2048, latent_dim)
        gen_samples = netG(z_sample)
        
        # MMD Discrepancy Score vs Ground Truth Validation Set
        mmd_score = compute_mmd(val_data, gen_samples)
        
        # Latent Grid Decoding (-3.0 to +3.0)
        grid_axis = np.linspace(-3.0, 3.0, 30)
        gx, gy = np.meshgrid(grid_axis, grid_axis)
        grid_z = torch.tensor(np.column_stack([gx.ravel(), gy.ravel()]), dtype=torch.float32)
        grid_gen = netG(grid_z)
        
        # Discriminator Decision Boundary Landscape
        map_axis = np.linspace(-3.5, 3.5, 100)
        mx, my = np.meshgrid(map_axis, map_axis)
        map_pts = torch.tensor(np.column_stack([mx.ravel(), my.ravel()]), dtype=torch.float32)
        d_landscape = torch.sigmoid(netD(map_pts)).numpy().reshape(100, 100)

    print(f"\nFinal GAN Discriminator Loss:      {d_losses[-1]:.4f}")
    print(f"Final GAN Generator Loss:          {g_losses[-1]:.4f}")
    print(f"Final MMD Generation Match Score: {mmd_score:.6f} (Distribution Fit)")

    artifact_plot_path = os.path.join(os.path.expanduser("~"), ".gemini", "antigravity-cli", "brain", "d720a57e-5223-4701-8226-44063e564858", "gan_plot.png")

    # 4. Plotting 4-Panel Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    
    # Panel 1: Loss Convergence
    ax1 = axes[0, 0]
    ax1.plot(range(1, epochs + 1), d_losses, color='#1f77b4', linewidth=2.0, label=f'Discriminator Loss ({d_losses[-1]:.4f})')
    ax1.plot(range(1, epochs + 1), g_losses, color='#d62728', linewidth=2.0, label=f'Generator Loss ({g_losses[-1]:.4f})')
    ax1.set_xlabel('Epoch', fontsize=10, fontweight='bold')
    ax1.set_ylabel('BCE Minimax Loss', fontsize=10, fontweight='bold')
    ax1.set_title('1. GAN Adversarial Loss Convergence', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Panel 2: Ground Truth vs GAN Generated 8-Gaussian Ring
    ax2 = axes[0, 1]
    ax2.scatter(val_data[:, 0].numpy(), val_data[:, 1].numpy(), color='gray', alpha=0.20, s=12, label='Real 8-Gaussian Ring Data')
    ax2.scatter(gen_samples[:, 0].numpy(), gen_samples[:, 1].numpy(), color='#1f77b4', alpha=0.6, s=15, label=f'GAN Generated Samples (MMD = {mmd_score:.4f})')
    ax2.set_xlabel('Feature X1', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Feature X2', fontsize=10, fontweight='bold')
    ax2.set_title(f'2. GAN Generated Distribution (MMD = {mmd_score:.4f})', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Panel 3: Discriminator Decision Boundary Landscape
    ax3 = axes[1, 0]
    contour = ax3.contourf(mx, my, d_landscape, levels=20, cmap='Blues', alpha=0.85)
    ax3.scatter(val_data[:, 0].numpy(), val_data[:, 1].numpy(), color='black', alpha=0.3, s=8, label='Real Data')
    ax3.set_xlabel('Feature X1', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Feature X2', fontsize=10, fontweight='bold')
    ax3.set_title('3. Discriminator Decision Boundary P(Real|x)', fontsize=12, fontweight='bold', pad=10)
    cbar = plt.colorbar(contour, ax=ax3)
    cbar.set_label('D(x) Real Probability', fontsize=9, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.3)
    
    # Panel 4: Latent Space Grid Decoding Trajectory
    ax4 = axes[1, 1]
    ax4.scatter(val_data[:, 0].numpy(), val_data[:, 1].numpy(), color='gray', alpha=0.15, s=10, label='Real Data')
    ax4.scatter(grid_gen[:, 0].numpy(), grid_gen[:, 1].numpy(), color='#9467bd', alpha=0.7, s=15, label='Generator Latent Grid z ∈ [-3, 3]²')
    ax4.set_xlabel('Feature X1', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Feature X2', fontsize=10, fontweight='bold')
    ax4.set_title('4. GAN Latent Grid Decoding Topology', fontsize=12, fontweight='bold', pad=10)
    ax4.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9)
    ax4.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.savefig(artifact_plot_path, dpi=300)
    plt.close()
    print(f"GAN Plot saved successfully to: {output_plot_path}")

if __name__ == "__main__":
    main()
