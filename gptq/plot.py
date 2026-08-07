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
        ("BF16", "model_bf16.pt", "bf16", torch.bfloat16, "#1f77b4"),
        ("GPTQ-FP8", "model_fp8.pt", "fp8", None, "#ff7f0e"),
        ("GPTQ-FP4", "model_fp4.pt", "fp4", None, "#d62728"),
    ]

    data = {}

    for label, filename, kind, dtype, color in variants:
        filepath = os.path.join(script_dir, filename)
        if kind == "bf16":
            m_var = RotationModel(dim=256, hidden_dim=1024).to(device, dtype=dtype)
            m_var.load_state_dict(torch.load(filepath, weights_only=True))
            m_var.eval()
            with torch.no_grad():
                y_var = m_var(x_test.to(dtype)).float()
        elif kind in ("fp8", "fp4"):
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
        }

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)

    # Panel 1: Combined Scatter Plot
    ax1 = axes[0, 0]
    for label, d in data.items():
        ax1.scatter(
            d["rel_mag_delta"],
            d["cos_sims"],
            alpha=0.45,
            s=22,
            color=d["color"],
            label=f"{label} (Mean Cos: {np.mean(d['cos_sims']):.4f})",
            edgecolors='none'
        )
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax1.set_xlabel("Relative Magnitude Error (%): (||y_var|| - ||y_ref||) / ||y_ref||", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Cosine Similarity vs FP32 Reference", fontsize=10, fontweight='bold')
    ax1.set_title("GPTQ Hessian-Guided Error Trade-off: Magnitude Shift vs. Angle Tilt", fontsize=12, fontweight='bold', pad=8)
    ax1.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Panel 2: Cosine Similarity Distribution
    ax2 = axes[0, 1]
    all_cos = np.concatenate([d["cos_sims"] for d in data.values()])
    cos_bins = np.linspace(np.min(all_cos) - 0.001, 1.0001, 50)
    for label, d in data.items():
        ax2.hist(d["cos_sims"], bins=cos_bins, alpha=0.45, color=d["color"], label=f"{label} (Min: {np.min(d['cos_sims']):.4f})")
    ax2.set_xlabel("Cosine Similarity", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Sample Count", fontsize=10, fontweight='bold')
    ax2.set_title("Distribution of Cosine Similarities Under GPTQ", fontsize=12, fontweight='bold', pad=8)
    ax2.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Panel 3: Relative Magnitude Errors (%) Distribution
    ax3 = axes[1, 0]
    all_mag_deltas = np.concatenate([d["rel_mag_delta"] for d in data.values()])
    mag_bins = np.linspace(np.min(all_mag_deltas) - 0.2, np.max(all_mag_deltas) + 0.2, 50)
    for label, d in data.items():
        ax3.hist(d["rel_mag_delta"], bins=mag_bins, alpha=0.45, color=d["color"], label=f"{label} (Std: {np.std(d['rel_mag_delta']):.2f}%)")
    ax3.axvline(0, color='black', linestyle='--', alpha=0.7)
    ax3.set_xlabel("Relative Magnitude Error (%)", fontsize=10, fontweight='bold')
    ax3.set_ylabel("Sample Count", fontsize=10, fontweight='bold')
    ax3.set_title("Distribution of Vector Magnitude Errors (%)", fontsize=12, fontweight='bold', pad=8)
    ax3.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9)
    ax3.grid(True, linestyle='--', alpha=0.5)

    # Panel 4: GPTQ-FP8 vs GPTQ-FP4 Focus Scatter Plot
    ax4 = axes[1, 1]
    ax4.scatter(data["GPTQ-FP8"]["rel_mag_delta"], data["GPTQ-FP8"]["cos_sims"], alpha=0.45, color=data["GPTQ-FP8"]["color"], label="GPTQ-FP8", s=25)
    ax4.scatter(data["GPTQ-FP4"]["rel_mag_delta"], data["GPTQ-FP4"]["cos_sims"], alpha=0.45, color=data["GPTQ-FP4"]["color"], label="GPTQ-FP4", s=25)
    ax4.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax4.set_xlabel("Relative Magnitude Error (%)", fontsize=10, fontweight='bold')
    ax4.set_ylabel("Cosine Similarity", fontsize=10, fontweight='bold')
    ax4.set_title("GPTQ-FP8 vs. GPTQ-FP4 Precision Comparison", fontsize=12, fontweight='bold', pad=8)
    ax4.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=9)
    ax4.grid(True, linestyle='--', alpha=0.5)

    output_path = os.path.join(script_dir, "gptq_analysis_plot.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"GPTQ Plot saved to: {output_path}")

if __name__ == "__main__":
    main()
