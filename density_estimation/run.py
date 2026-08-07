import os
import torch
import numpy as np
import scipy.stats as stats
import scipy.special as special
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def cdf_to_pdf_algebraic(y_cdf, sigma):
    """
    Closed-form algebraic transform mapping CDF probabilities y in (0, 1) directly to PDF values:
    f(y) = (1 / (sigma * sqrt(2*pi))) * exp(- (erfinv(2y - 1))^2)
    """
    z = special.erfinv(2.0 * y_cdf - 1.0)
    pdf = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-(z ** 2))
    return pdf

def generate_samples(num_samples=2000, mean=1.5, std=2.0):
    samples = np.random.normal(loc=mean, scale=std, size=num_samples)
    return np.sort(samples)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_plot_path = os.path.join(script_dir, "plot.png")
    
    N = 2000
    true_mean, true_std = 1.5, 2.0
    x_samples = generate_samples(num_samples=N, mean=true_mean, std=true_std)
    
    sample_mean = np.mean(x_samples)
    sample_std = np.std(x_samples, ddof=1)
    
    # Empirical CDF ranks y_i in (0, 1)
    y_empirical_cdf = (np.arange(1, N + 1) - 0.5) / N
    y_theoretical_cdf = stats.norm.cdf(x_samples, loc=sample_mean, scale=sample_std)
    
    # Algebraic transform g(y)
    pdf_algebraic_empirical = cdf_to_pdf_algebraic(y_empirical_cdf, sigma=sample_std)
    pdf_true = stats.norm.pdf(x_samples, loc=sample_mean, scale=sample_std)
    
    # Goodness-of-fit
    ss_res = np.sum((pdf_algebraic_empirical - pdf_true) ** 2)
    ss_tot = np.sum((pdf_true - np.mean(pdf_true)) ** 2)
    r2_score = 1.0 - (ss_res / ss_tot)
    rmse_val = np.sqrt(np.mean((pdf_algebraic_empirical - pdf_true) ** 2))
    
    print(f"Sample Mean: {sample_mean:.4f} | Sample Std: {sample_std:.4f}")
    print(f"Algebraic Transform PDF Alignment R^2: {r2_score:.6f} | RMSE: {rmse_val:.6f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    
    # -------------------------------------------------------------
    # Plot 1: CDF (Empirical Ranks vs Theoretical Line)
    # -------------------------------------------------------------
    ax1 = axes[0]
    ax1.scatter(x_samples, y_empirical_cdf, color='#1f77b4', alpha=0.45, s=12, label=f'Sample ECDF (N={N})')
    ax1.plot(x_samples, y_theoretical_cdf, color='#d62728', linewidth=2.5, label=f'Theoretical CDF (μ={sample_mean:.2f}, σ={sample_std:.2f})')
    ax1.axvline(sample_mean, color='gray', linestyle='--', alpha=0.7, label=f'Mean μ = {sample_mean:.2f}')
    ax1.axhline(0.5, color='gray', linestyle=':', alpha=0.7, label='Median (0.5)')
    
    ax1.set_xlabel('Sorted Sample Values (x)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Cumulative Probability P(X ≤ x)', fontsize=11, fontweight='bold')
    ax1.set_title('1. Cumulative Distribution Function (CDF)', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # -------------------------------------------------------------
    # Plot 2: Transformed PDF via g(y)
    # -------------------------------------------------------------
    ax2 = axes[1]
    ax2.plot(x_samples, pdf_algebraic_empirical, color='#1f77b4', linewidth=3.0, label=f'Algebraic PDF g(y_emp) (R² = {r2_score:.4f})')
    ax2.plot(x_samples, pdf_true, color='#d62728', linestyle='--', linewidth=2.5, label='Theoretical Gaussian PDF')
    
    ax2.set_xlabel('Sorted Sample Values (x)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Probability Density f(x)', fontsize=11, fontweight='bold')
    ax2.set_title('2. Transformed Probability Density Function (PDF)', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    print(f"Plot saved to: {output_plot_path}")

if __name__ == "__main__":
    main()
