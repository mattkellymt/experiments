import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    A = torch.tensor([[4.0, 1.5],
                      [1.5, 2.0]], dtype=torch.float32)

    true_evals, true_evecs = torch.linalg.eigh(A)
    true_v1 = true_evecs[:, 1].numpy()  # Dominant (λ ≈ 4.80)
    true_v2 = true_evecs[:, 0].numpy()  # Secondary (λ ≈ 1.20)

    # -------------------------------------------------------------
    # Simulate Optimization Trajectories for Phase 1 and Phase 2
    # -------------------------------------------------------------
    # Phase 1: v1 Trajectory
    v1_param = torch.tensor([1.0, 0.1], requires_grad=True)
    opt1 = torch.optim.Adam([v1_param], lr=0.04)
    v1_traj = [(v1_param / torch.norm(v1_param)).detach().numpy()]
    loss_v1_history = []

    for _ in range(80):
        opt1.zero_grad()
        v1_unit = v1_param / torch.norm(v1_param)
        Av1 = A @ v1_unit
        cos_sim = torch.abs(torch.nn.functional.cosine_similarity(v1_unit.unsqueeze(0), Av1.unsqueeze(0)))
        loss = 1.0 - cos_sim
        loss.backward()
        opt1.step()

        v1_traj.append((v1_param / torch.norm(v1_param)).detach().numpy())
        loss_v1_history.append(loss.item())

    v1_learned = (v1_param / torch.norm(v1_param)).detach()
    v1_traj = np.array(v1_traj)

    # Phase 2: v2 Trajectory
    v2_param = torch.tensor([0.1, 1.0], requires_grad=True)
    opt2 = torch.optim.Adam([v2_param], lr=0.04)
    v2_traj = [(v2_param / torch.norm(v2_param)).detach().numpy()]
    loss_v2_history = []

    for _ in range(80):
        opt2.zero_grad()
        v2_unit = v2_param / torch.norm(v2_param)
        Av2 = A @ v2_unit
        cos_sim = torch.abs(torch.nn.functional.cosine_similarity(v2_unit.unsqueeze(0), Av2.unsqueeze(0)))
        ortho_penalty = (torch.dot(v1_learned, v2_unit)) ** 2
        loss = (1.0 - cos_sim) + 10.0 * ortho_penalty
        loss.backward()
        opt2.step()

        v2_traj.append((v2_param / torch.norm(v2_param)).detach().numpy())
        loss_v2_history.append(loss.item())

    v2_traj = np.array(v2_traj)

    # -------------------------------------------------------------
    # Build 3-Panel Figure
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)

    # Panel 1: Matrix Action & Non-Rotation Axes (Eigenvectors)
    ax1 = axes[0]
    angles = np.linspace(0, 2*np.pi, 24, endpoint=False)
    for theta in angles:
        u = np.array([np.cos(theta), np.sin(theta)])
        Au = (A.numpy() @ u)
        Au_norm = Au / np.linalg.norm(Au)
        ax1.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1, color='gray', alpha=0.3)
        ax1.quiver(0, 0, Au_norm[0], Au_norm[1], angles='xy', scale_units='xy', scale=1, color='#1f77b4', alpha=0.25)

    # Highlight Eigenvector Axes
    ax1.plot([-true_v1[0]*1.5, true_v1[0]*1.5], [-true_v1[1]*1.5, true_v1[1]*1.5], '--', color='#d62728', linewidth=2, label='Eigenvector 1 Axis (λ=4.80)')
    ax1.plot([-true_v2[0]*1.5, true_v2[0]*1.5], [-true_v2[1]*1.5, true_v2[1]*1.5], '--', color='#2ca02c', linewidth=2, label='Eigenvector 2 Axis (λ=1.20)')

    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.set_xlabel('x1', fontsize=10, fontweight='bold')
    ax1.set_ylabel('x2', fontsize=10, fontweight='bold')
    ax1.set_title('1. Matrix Action: Rotates All Vectors Except Eigenvectors', fontsize=11, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', fontsize=8, frameon=True, framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Panel 2: Vector Optimization Trajectories
    ax2 = axes[1]
    circle = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle=':', alpha=0.6)
    ax2.add_patch(circle)

    ax2.plot(v1_traj[:, 0], v1_traj[:, 1], 'o-', color='#d62728', linewidth=2, markersize=3, label='v1 Path (Learns Dominant Axis)')
    ax2.plot(v2_traj[:, 0], v2_traj[:, 1], 's-', color='#2ca02c', linewidth=2, markersize=3, label='v2 Path (Learns Secondary + Orthogonal)')

    ax2.quiver(0, 0, true_v1[0], true_v1[1], angles='xy', scale_units='xy', scale=1, color='black', width=0.015, label='Target v1')
    ax2.quiver(0, 0, true_v2[0], true_v2[1], angles='xy', scale_units='xy', scale=1, color='black', width=0.015, label='Target v2')

    ax2.set_xlim(-1.3, 1.3)
    ax2.set_ylim(-1.3, 1.3)
    ax2.set_aspect('equal')
    ax2.set_xlabel('x1', fontsize=10, fontweight='bold')
    ax2.set_ylabel('x2', fontsize=10, fontweight='bold')
    ax2.set_title('2. Optimization Trajectories in Vector Space', fontsize=11, fontweight='bold', pad=10)
    ax2.legend(loc='upper left', fontsize=8, frameon=True, framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Panel 3: Cosine Similarity Loss Curves
    ax3 = axes[2]
    ax3.plot(range(1, 81), loss_v1_history, color='#d62728', linewidth=2, label='Phase 1 Loss (v1 -> Dominant)')
    ax3.plot(range(1, 81), loss_v2_history, color='#2ca02c', linewidth=2, label='Phase 2 Loss (v2 -> Secondary)')

    ax3.set_xlabel('Epochs', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Loss (1.0 - |Cosine Similarity|)', fontsize=10, fontweight='bold')
    ax3.set_title('3. Convergence Curves to Zero Loss (|Cos Sim| = 1.0)', fontsize=11, fontweight='bold', pad=10)
    ax3.legend(loc='upper right', fontsize=9, frameon=True, framealpha=0.9)
    ax3.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    output_path = os.path.join(script_dir, "eigenvector_learning_plot.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Eigenvector learning plot saved to: {output_path}")

if __name__ == "__main__":
    main()
