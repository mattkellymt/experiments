import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from model import RotationModel, CodebookRehydrationLinear, generate_dataset

def evaluate_codebook_model(m_cb: nn.Module, x_test: torch.Tensor, hard: bool = False) -> torch.Tensor:
    m_cb.eval()
    x = x_test.clone()
    for child in m_cb.children():
        if isinstance(child, CodebookRehydrationLinear):
            x = child(x, hard=hard)
        else:
            x = child(x)
    return x

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    num_samples = 512
    x_test, y_test = generate_dataset(num_samples=num_samples, dim=256, seed=999)
    x_test = x_test.to(device)

    fp32_path = os.path.join(script_dir, "model_fp32.pt")
    m_fp32 = RotationModel(dim=256, hidden_dim=1024).to(device)
    m_fp32.load_state_dict(torch.load(fp32_path, weights_only=True))
    m_fp32.eval()

    with torch.no_grad():
        y_ref = m_fp32(x_test).float()

    mag_ref = torch.norm(y_ref, p=2, dim=1).cpu().numpy()

    block_configs = [
        ("32x32 Block (0.18 MB)", 32, 32, "#1f77b4", "model_codebook_32x32.pt"),
        ("16x16 Block (0.22 MB)", 16, 16, "#2ca02c", "model_codebook_16x16.pt"),
        ("8x8 Block (0.32 MB)", 8, 8, "#ff7f0e", "model_codebook_8x8.pt"),
        ("4x4 Block (0.45 MB)", 4, 4, "#d62728", "model_codebook_4x4.pt"),
    ]

    data = {}

    for label, bh, bw, color, fname in block_configs:
        fpath = os.path.join(script_dir, fname)
        if not os.path.exists(fpath):
            continue

        state_dict = torch.load(fpath, weights_only=True)
        m_cb = RotationModel(dim=256, hidden_dim=1024).to(device)

        for name, child in m_fp32.named_children():
            if isinstance(child, nn.Linear):
                k_shape = state_dict[f"{name}.quantizer.codebook"].shape[0]
                cb_layer = CodebookRehydrationLinear(child.in_features, child.out_features, k_codes=k_shape, block_h=bh, block_w=bw).to(device)
                cb_layer.quantizer.codebook = nn.Parameter(torch.randn(k_shape, bh, bw, device=device))
                setattr(m_cb, name, cb_layer)

        m_cb.load_state_dict(state_dict)
        m_cb.eval()

        with torch.no_grad():
            y_var = evaluate_codebook_model(m_cb, x_test, hard=True).float()

        cos_sims = torch.nn.functional.cosine_similarity(y_ref, y_var, dim=1).cpu().numpy()
        mag_var = torch.norm(y_var, p=2, dim=1).cpu().numpy()
        rel_mag_delta = (mag_var - mag_ref) / mag_ref * 100.0

        data[label] = {
            "cos_sims": cos_sims,
            "rel_mag_delta": rel_mag_delta,
            "color": color,
        }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

    # -------------------------------------------------------------
    # ROW 1: SCATTER & ANNEALING PLOTS
    # -------------------------------------------------------------
    # Panel 1 (Row 1 Left): Relative Magnitude Error vs Cosine Similarity Sweep
    ax1 = axes[0, 0]
    for label, d in data.items():
        ax1.scatter(d["rel_mag_delta"], d["cos_sims"], alpha=0.45, color=d["color"], label=f"{label} (Mean Cos: {np.mean(d['cos_sims']):.4f})", s=22)

    ax1.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax1.set_xlabel("Relative Magnitude Error (%): (||y_rehydrated|| - ||y_ref||) / ||y_ref||", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Cosine Similarity vs FP32 Reference", fontsize=10, fontweight='bold')
    ax1.set_title("Block Size Sweep: Relative Magnitude Error vs. Cosine Similarity", fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Panel 2 (Row 1 Right): Temperature Annealing Curve
    ax2 = axes[0, 1]
    epochs = np.arange(1, 36)
    taus = 1.0 * ((0.05 / 1.0) ** (epochs / 35.0))
    ax2.plot(epochs, taus, color='#9467bd', linewidth=2.5, label='Softmax Temperature (τ: 1.0 → 0.05)')
    ax2.set_xlabel("Fine-Tuning Epochs", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Softmax Temperature (τ)", fontsize=10, fontweight='bold')
    ax2.set_title("Softmax Annealing: Mixture → Hard Discrete Selection", fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # -------------------------------------------------------------
    # ROW 2: HISTOGRAM DISTRIBUTION BINS
    # -------------------------------------------------------------
    # Panel 3 (Row 2 Left): Cosine Similarity Distribution Bins
    ax3 = axes[1, 0]
    all_cos = np.concatenate([d["cos_sims"] for d in data.values()])
    cos_bins = np.linspace(np.min(all_cos) - 0.001, 1.0001, 50)
    for label, d in data.items():
        ax3.hist(d["cos_sims"], bins=cos_bins, alpha=0.35, color=d["color"], label=f"{label} (Min: {np.min(d['cos_sims']):.4f})", histtype='stepfilled')

    ax3.set_xlabel("Cosine Similarity", fontsize=10, fontweight='bold')
    ax3.set_ylabel("Sample Count", fontsize=10, fontweight='bold')
    ax3.set_title("Distribution Bins: Cosine Similarity Across Block Sizes", fontsize=12, fontweight='bold', pad=10)
    ax3.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=8)
    ax3.grid(True, linestyle='--', alpha=0.5)

    # Panel 4 (Row 2 Right): Vector Magnitude Shift Bins (%)
    ax4 = axes[1, 1]
    all_mag_deltas = np.concatenate([d["rel_mag_delta"] for d in data.values()])
    mag_bins = np.linspace(np.min(all_mag_deltas) - 0.2, np.max(all_mag_deltas) + 0.2, 50)
    for label, d in data.items():
        ax4.hist(d["rel_mag_delta"], bins=mag_bins, alpha=0.35, color=d["color"], label=f"{label} (Std: {np.std(d['rel_mag_delta']):.2f}%)", histtype='stepfilled')

    ax4.axvline(0, color='black', linestyle='--', alpha=0.7)
    ax4.set_xlabel("Relative Magnitude Error (%)", fontsize=10, fontweight='bold')
    ax4.set_ylabel("Sample Count", fontsize=10, fontweight='bold')
    ax4.set_title("Distribution Bins: Vector Magnitude Shifts (%)", fontsize=12, fontweight='bold', pad=10)
    ax4.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=8)
    ax4.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    output_path = os.path.join(script_dir, "codebook_analysis_plot.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Codebook Plot saved to: {output_path}")

if __name__ == "__main__":
    main()
