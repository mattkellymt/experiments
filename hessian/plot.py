import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from hessian import generate_data, compute_loss_and_hessian

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    X, y = generate_data(num_samples=500)
    N = X.shape[0]

    # True optimal weights
    w_true = torch.tensor([2.0, -3.0])

    # Compute Hessian matrix and eigen-decomposition
    H = (1.0 / N) * (X.T @ X)
    eigenvalues, eigenvectors = torch.linalg.eigh(H)

    # Build 2D weight grid for loss surface evaluation
    w1_vals = np.linspace(-1.0, 5.0, 100)
    w2_vals = np.linspace(-6.0, 0.0, 100)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)

    Z = np.zeros_like(W1)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            w_grid = torch.tensor([W1[i, j], W2[i, j]], dtype=torch.float32)
            preds = X @ w_grid
            Z[i, j] = (0.5 / N) * torch.sum((preds - y) ** 2).item()

    # Simulate Gradient Descent Trajectory (lr = 0.15)
    w_gd = torch.tensor([-0.5, -0.5])
    gd_history = [w_gd.clone().numpy()]
    lr = 0.15
    for step in range(25):
        grad = (1.0 / N) * X.T @ (X @ w_gd - y)
        w_gd = w_gd - lr * grad
        gd_history.append(w_gd.clone().numpy())
    gd_history = np.array(gd_history)

    # Compute 1-Step Newton-Hessian Jump
    w_start = torch.tensor([-0.5, -0.5])
    grad_start = (1.0 / N) * X.T @ (X @ w_start - y)
    w_newton = w_start - torch.linalg.inv(H) @ grad_start

    # Build Figure with 3 Panels
    fig = plt.figure(figsize=(18, 6), dpi=300)

    # -------------------------------------------------------------
    # Panel 1: 3D Loss Landscape Surface (z = L(w1, w2))
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    surf = ax1.plot_surface(W1, W2, Z, cmap='viridis', alpha=0.85, edgecolor='none')
    ax1.scatter([w_true[0].item()], [w_true[1].item()], [0], color='red', s=80, label='Global Minimum (w*)', zorder=5)
    ax1.set_xlabel('Weight w1', fontsize=10, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Weight w2', fontsize=10, fontweight='bold', labelpad=8)
    ax1.set_zlabel('Loss L(w1, w2)', fontsize=10, fontweight='bold', labelpad=8)
    ax1.set_title('1. 3D Loss Landscape Surface Bowl', fontsize=12, fontweight='bold', pad=12)
    ax1.legend(loc='upper right')
    ax1.view_init(elev=28, azim=-125)

    # -------------------------------------------------------------
    # Panel 2: 2D Contour Map + Hessian Eigenvectors (Curvature Axes)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(1, 3, 2)
    contours = ax2.contour(W1, W2, Z, levels=25, cmap='viridis', alpha=0.8)
    ax2.clabel(contours, inline=True, fontsize=7)

    origin = w_true.numpy()
    ev1 = eigenvectors[:, 0].numpy() * np.sqrt(eigenvalues[0].item()) * 1.2
    ev2 = eigenvectors[:, 1].numpy() * np.sqrt(eigenvalues[1].item()) * 1.2

    # Draw Hessian Eigenvector Curvature Axes
    ax2.quiver(*origin, *ev1, color='#d62728', scale=1, scale_units='xy', width=0.012, label=f'v1 (Flat Axis, λ={eigenvalues[0].item():.2f})')
    ax2.quiver(*origin, *ev2, color='#1f77b4', scale=1, scale_units='xy', width=0.012, label=f'v2 (Steep Axis, λ={eigenvalues[1].item():.2f})')
    ax2.scatter(*origin, color='red', s=70, zorder=5, label='Minimum w*')

    ax2.set_xlabel('Weight w1', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Weight w2', fontsize=10, fontweight='bold')
    ax2.set_title('2. Loss Contours & Hessian Eigenvectors', fontsize=12, fontweight='bold', pad=12)
    ax2.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # -------------------------------------------------------------
    # Panel 3: Gradient Descent Oscillations vs Newton 1-Step Jump
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.contour(W1, W2, Z, levels=25, cmap='viridis', alpha=0.5)
    
    # Plot Gradient Descent Path
    ax3.plot(gd_history[:, 0], gd_history[:, 1], 'o-', color='#ff7f0e', linewidth=2, markersize=4, label='Gradient Descent (Oscillates)')
    
    # Plot Newton-Hessian 1-Step Jump
    ax3.plot([w_start[0].item(), w_newton[0].item()], [w_start[1].item(), w_newton[1].item()], 
             's--', color='#2ca02c', linewidth=2.5, markersize=7, label='Newton-Hessian (1-Step Direct Jump)')

    ax3.scatter(*origin, color='red', s=80, zorder=5, label='Minimum w*')
    ax3.set_xlabel('Weight w1', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Weight w2', fontsize=10, fontweight='bold')
    ax3.set_title('3. Gradient Descent vs. Newton 1-Step Jump', fontsize=12, fontweight='bold', pad=12)
    ax3.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=8)
    ax3.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    output_path = os.path.join(script_dir, "hessian_visualization.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Hessian visualization saved to: {output_path}")

if __name__ == "__main__":
    main()
