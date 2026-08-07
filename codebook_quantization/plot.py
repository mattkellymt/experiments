import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from model import RotationModel, NeuralRehydrationLinear, generate_dataset

def evaluate_rehydration_model(m_cb: nn.Module, x_test: torch.Tensor) -> torch.Tensor:
    m_cb.eval()
    x = x_test.clone()
    for child in m_cb.children():
        if isinstance(child, NeuralRehydrationLinear):
            x = child(x)
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

    # Load Raw FP4 & Rehydrated FP4 Models
    m_raw_fp4 = RotationModel(dim=256, hidden_dim=1024).to(device)
    m_raw_fp4.load_state_dict(torch.load(os.path.join(script_dir, "model_raw_fp4.pt"), weights_only=True))
    m_raw_fp4.eval()

    m_rehydrated_fp4 = RotationModel(dim=256, hidden_dim=1024).to(device)
    for name, child in m_fp32.named_children():
        if isinstance(child, nn.Linear):
            setattr(m_rehydrated_fp4, name, NeuralRehydrationLinear(child.in_features, child.out_features, format_type="fp4").to(device))
    m_rehydrated_fp4.load_state_dict(torch.load(os.path.join(script_dir, "model_rehydrated_fp4.pt"), weights_only=True))
    m_rehydrated_fp4.eval()

    # Load Raw FP8 & Rehydrated FP8 Models
    m_raw_fp8 = RotationModel(dim=256, hidden_dim=1024).to(device)
    m_raw_fp8.load_state_dict(torch.load(os.path.join(script_dir, "model_raw_fp8.pt"), weights_only=True))
    m_raw_fp8.eval()

    m_rehydrated_fp8 = RotationModel(dim=256, hidden_dim=1024).to(device)
    for name, child in m_fp32.named_children():
        if isinstance(child, nn.Linear):
            setattr(m_rehydrated_fp8, name, NeuralRehydrationLinear(child.in_features, child.out_features, format_type="fp8").to(device))
    m_rehydrated_fp8.load_state_dict(torch.load(os.path.join(script_dir, "model_rehydrated_fp8.pt"), weights_only=True))
    m_rehydrated_fp8.eval()

    with torch.no_grad():
        y_raw_fp4 = m_raw_fp4(x_test).float()
        y_rehydrated_fp4 = evaluate_rehydration_model(m_rehydrated_fp4, x_test).float()

        y_raw_fp8 = m_raw_fp8(x_test).float()
        y_rehydrated_fp8 = evaluate_rehydration_model(m_rehydrated_fp8, x_test).float()

    variants = [
        ("Raw FP4 Baseline", y_raw_fp4, "#ff7f0e"),
        ("Rehydrated FP4 Engine", y_rehydrated_fp4, "#2ca02c"),
        ("Raw FP8 Baseline", y_raw_fp8, "#d62728"),
        ("Rehydrated FP8 Engine", y_rehydrated_fp8, "#1f77b4"),
    ]

    data = {}

    for label, y_var, color in variants:
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
    # ROW 1: SCATTER PLOTS
    # -------------------------------------------------------------
    # Panel 1 (Row 1 Left): FP4 Raw vs Rehydrated Scatter
    ax1 = axes[0, 0]
    for label in ["Raw FP4 Baseline", "Rehydrated FP4 Engine"]:
        d = data[label]
        ax1.scatter(d["rel_mag_delta"], d["cos_sims"], alpha=0.55, color=d["color"], label=f"{label} (Mean Cos: {np.mean(d['cos_sims']):.4f})", s=24)

    ax1.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax1.set_xlabel("Relative Magnitude Error (%): (||y_var|| - ||y_ref||) / ||y_ref||", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Cosine Similarity vs FP32 Reference", fontsize=10, fontweight='bold')
    ax1.set_title("FP4 Comparison: Raw FP4 vs. Non-Linear Rehydrated FP4 Engine", fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Panel 2 (Row 1 Right): FP8 Raw vs Rehydrated Scatter
    ax2 = axes[0, 1]
    for label in ["Raw FP8 Baseline", "Rehydrated FP8 Engine"]:
        d = data[label]
        ax2.scatter(d["rel_mag_delta"], d["cos_sims"], alpha=0.55, color=d["color"], label=f"{label} (Mean Cos: {np.mean(d['cos_sims']):.4f})", s=24)

    ax2.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax2.set_xlabel("Relative Magnitude Error (%): (||y_var|| - ||y_ref||) / ||y_ref||", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Cosine Similarity vs FP32 Reference", fontsize=10, fontweight='bold')
    ax2.set_title("FP8 Comparison: Raw FP8 vs. Non-Linear Rehydrated FP8 Engine", fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=9)
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
    ax3.set_title("Distribution Bins: Cosine Similarity Comparison", fontsize=12, fontweight='bold', pad=10)
    ax3.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=8)
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
    ax4.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=8)
    ax4.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    output_path = os.path.join(script_dir, "codebook_analysis_plot.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Codebook Plot saved to: {output_path}")

if __name__ == "__main__":
    main()
