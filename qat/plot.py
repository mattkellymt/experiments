import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from model import RotationModel, generate_dataset

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_samples = 1024
    x_test, y_test = generate_dataset(num_samples=num_samples, dim=256, seed=999)
    x_test = x_test.to(device)

    fp32_path = os.path.join(script_dir, "model_fp32.pt")
    m_fp32 = RotationModel(dim=256, hidden_dim=1024).to(device)
    m_fp32.load_state_dict(torch.load(fp32_path, weights_only=True))
    m_fp32.eval()

    with torch.no_grad():
        y_ref = m_fp32(x_test).float()

    mag_ref = torch.norm(y_ref, p=2, dim=1).cpu().numpy()

    variants = [
        ("PTQ FP4", "model_fp4_ptq.pt", "#d62728", "--"),
        ("QAT FP4", "model_fp4_qat.pt", "#2ca02c", "-"),
        ("PTQ FP8", "model_fp8_ptq.pt", "#ff7f0e", "--"),
        ("QAT FP8", "model_fp8_qat.pt", "#1f77b4", "-"),
    ]

    data = {}

    for label, filename, color, ls in variants:
        filepath = os.path.join(script_dir, filename)
        m_var = RotationModel(dim=256, hidden_dim=1024).to(device)
        m_var.load_state_dict(torch.load(filepath, weights_only=True))
        m_var.eval()
        with torch.no_grad():
            y_var = m_var(x_test).float()

        cos_sims = torch.nn.functional.cosine_similarity(y_ref, y_var, dim=1).cpu().numpy()
        mag_var = torch.norm(y_var, p=2, dim=1).cpu().numpy()
        rel_mag_delta = (mag_var - mag_ref) / mag_ref * 100.0

        data[label] = {
            "cos_sims": cos_sims,
            "rel_mag_delta": rel_mag_delta,
            "color": color,
            "ls": ls,
        }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

    # -------------------------------------------------------------
    # ROW 1: SCATTER PLOTS
    # -------------------------------------------------------------
    # Panel 1 (Row 1 Left): FP4 PTQ vs QAT Scatter
    ax1 = axes[0, 0]
    ax1.scatter(data["PTQ FP4"]["rel_mag_delta"], data["PTQ FP4"]["cos_sims"], alpha=0.4, color=data["PTQ FP4"]["color"], label=f"PTQ FP4 (Mean Cos: {np.mean(data['PTQ FP4']['cos_sims']):.4f})", s=22)
    ax1.scatter(data["QAT FP4"]["rel_mag_delta"], data["QAT FP4"]["cos_sims"], alpha=0.4, color=data["QAT FP4"]["color"], label=f"QAT FP4 (Mean Cos: {np.mean(data['QAT FP4']['cos_sims']):.4f})", s=22)
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax1.set_xlabel("Relative Magnitude Error (%): (||y_var|| - ||y_ref||) / ||y_ref||", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Cosine Similarity vs FP32 Reference", fontsize=10, fontweight='bold')
    ax1.set_title("FP4 Scatter: Post-Training (PTQ) vs. QAT Fine-Tuning", fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Panel 2 (Row 1 Right): FP8 PTQ vs QAT Scatter
    ax2 = axes[0, 1]
    ax2.scatter(data["PTQ FP8"]["rel_mag_delta"], data["PTQ FP8"]["cos_sims"], alpha=0.4, color=data["PTQ FP8"]["color"], label=f"PTQ FP8 (Mean Cos: {np.mean(data['PTQ FP8']['cos_sims']):.4f})", s=22)
    ax2.scatter(data["QAT FP8"]["rel_mag_delta"], data["QAT FP8"]["cos_sims"], alpha=0.4, color=data["QAT FP8"]["color"], label=f"QAT FP8 (Mean Cos: {np.mean(data['QAT FP8']['cos_sims']):.4f})", s=22)
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax2.set_xlabel("Relative Magnitude Error (%): (||y_var|| - ||y_ref||) / ||y_ref||", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Cosine Similarity vs FP32 Reference", fontsize=10, fontweight='bold')
    ax2.set_title("FP8 Scatter: Post-Training (PTQ) vs. QAT Fine-Tuning", fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # -------------------------------------------------------------
    # ROW 2: HISTOGRAM BINS
    # -------------------------------------------------------------
    # Panel 3 (Row 2 Left): Cosine Similarity Distribution Bins
    ax3 = axes[1, 0]
    all_cos = np.concatenate([d["cos_sims"] for d in data.values()])
    cos_bins = np.linspace(np.min(all_cos) - 0.001, 1.0001, 50)
    for label, d in data.items():
        ax3.hist(d["cos_sims"], bins=cos_bins, alpha=0.35, color=d["color"], label=f"{label} (Min: {np.min(d['cos_sims']):.4f})", histtype='stepfilled')

    ax3.set_xlabel("Cosine Similarity", fontsize=10, fontweight='bold')
    ax3.set_ylabel("Sample Count", fontsize=10, fontweight='bold')
    ax3.set_title("Distribution Bins: Cosine Similarity Comparison", fontsize=12, fontweight='bold', pad=10)
    ax3.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=9)
    ax3.grid(True, linestyle='--', alpha=0.5)

    # Panel 4 (Row 2 Right): Relative Magnitude Error Bins (%)
    ax4 = axes[1, 1]
    all_mag_deltas = np.concatenate([d["rel_mag_delta"] for d in data.values()])
    mag_bins = np.linspace(np.min(all_mag_deltas) - 0.2, np.max(all_mag_deltas) + 0.2, 50)
    for label, d in data.items():
        ax4.hist(d["rel_mag_delta"], bins=mag_bins, alpha=0.35, color=d["color"], label=f"{label} (Std: {np.std(d['rel_mag_delta']):.2f}%)", histtype='stepfilled')

    ax4.axvline(0, color='black', linestyle='--', alpha=0.7)
    ax4.set_xlabel("Relative Magnitude Error (%)", fontsize=10, fontweight='bold')
    ax4.set_ylabel("Sample Count", fontsize=10, fontweight='bold')
    ax4.set_title("Distribution Bins: Vector Magnitude Shifts (%)", fontsize=12, fontweight='bold', pad=10)
    ax4.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9)
    ax4.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    output_path = os.path.join(script_dir, "qat_analysis_plot.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"QAT Analysis Plot saved to: {output_path}")

if __name__ == "__main__":
    main()
