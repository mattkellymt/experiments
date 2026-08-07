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
    1. Exact Algebraic CDF-to-PDF Transform g(y):
       f(y) = (1 / (sigma * sqrt(2*pi))) * exp(- (erfinv(2y - 1))^2)
    """
    z = special.erfinv(2.0 * y_cdf - 1.0)
    pdf = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-(z ** 2))
    return pdf

def pdf_to_cdf_algebraic(f_pdf, x_positions, mean, sigma):
    """
    2. Exact Algebraic PDF-to-CDF Transform y(f):
       y(f) = 0.5 * [1 +/- erf( sqrt( -ln(f * sigma * sqrt(2*pi)) ) )]
    """
    max_pdf = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    f_clamped = np.minimum(f_pdf, max_pdf * (1.0 - 1e-12))
    z_dist = np.sqrt(np.maximum(0.0, -np.log(f_clamped * sigma * np.sqrt(2.0 * np.pi))))
    
    y_cdf = np.where(x_positions >= mean,
                     0.5 * (1.0 + special.erf(z_dist)),
                     0.5 * (1.0 - special.erf(z_dist)))
    return y_cdf

def generate_samples(num_samples=2000, mean=1.5, std=2.0):
    samples = np.random.normal(loc=mean, scale=std, size=num_samples)
    return np.sort(samples)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_plot_path = os.path.join(script_dir, "plot.png")
    
    N = 2000
    true_mean, true_std = 1.5, 2.0
    x_samples = generate_samples(num_samples=N, mean=true_mean, std=true_std)
    
    # Measured Sample Parameters
    sample_mean = np.mean(x_samples)
    sample_std = np.std(x_samples, ddof=1)
    
    # -------------------------------------------------------------
    # PANE 1: Empirical CDF from Sorted Samples
    # -------------------------------------------------------------
    y_empirical_cdf = (np.arange(1, N + 1) - 0.5) / N
    y_theoretical_cdf = stats.norm.cdf(x_samples, loc=sample_mean, scale=sample_std)
    
    # -------------------------------------------------------------
    # PANE 2: PDF Created Algebraically from CDF
    # -------------------------------------------------------------
    pdf_algebraic_empirical = cdf_to_pdf_algebraic(y_empirical_cdf, sigma=sample_std)
    pdf_true = stats.norm.pdf(x_samples, loc=sample_mean, scale=sample_std)
    
    ss_res_pdf = np.sum((pdf_algebraic_empirical - pdf_true) ** 2)
    ss_tot_pdf = np.sum((pdf_true - np.mean(pdf_true)) ** 2)
    r2_pdf = 1.0 - (ss_res_pdf / ss_tot_pdf)
    
    # -------------------------------------------------------------
    # PANE 3: CDF Reconstructed Algebraically from PDF
    # -------------------------------------------------------------
    cdf_reconstructed = pdf_to_cdf_algebraic(pdf_algebraic_empirical, x_samples, mean=sample_mean, sigma=sample_std)
    
    ss_res_cdf = np.sum((cdf_reconstructed - y_theoretical_cdf) ** 2)
    ss_tot_cdf = np.sum((y_theoretical_cdf - np.mean(y_theoretical_cdf)) ** 2)
    r2_cdf_recon = 1.0 - (ss_res_cdf / ss_tot_cdf)
    
    print(f"Sample Mean: {sample_mean:.4f} | Sample Std: {sample_std:.4f}")
    print(f"Pane 2 (CDF -> PDF Algebraic R^2): {r2_pdf:.6f}")
    print(f"Pane 3 (PDF -> CDF Algebraic R^2): {r2_cdf_recon:.8f}")
    
    # Plotting 3 Panes Side-by-Side (1x3 Subplots)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), dpi=300)
    
    # Pane 1: Empirical CDF
    ax1 = axes[0]
    ax1.scatter(x_samples, y_empirical_cdf, color='#1f77b4', alpha=0.45, s=12, label=f'Sample ECDF (N={N})')
    ax1.plot(x_samples, y_theoretical_cdf, color='#d62728', linewidth=2.5, label=f'Theoretical CDF (μ={sample_mean:.2f}, σ={sample_std:.2f})')
    ax1.axvline(sample_mean, color='gray', linestyle='--', alpha=0.7, label=f'Mean μ = {sample_mean:.2f}')
    ax1.axhline(0.5, color='gray', linestyle=':', alpha=0.7, label='Median (0.5)')
    
    ax1.set_xlabel('Sorted Sample Values (x)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Cumulative Probability P(X ≤ x)', fontsize=11, fontweight='bold')
    ax1.set_title('1. Empirical CDF (Sorted Samples)', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Pane 2: PDF from CDF
    ax2 = axes[1]
    ax2.plot(x_samples, pdf_algebraic_empirical, color='#1f77b4', linewidth=3.0, label=f'Algebraic PDF g(y_emp)\n(R² = {r2_pdf:.4f})')
    ax2.plot(x_samples, pdf_true, color='#d62728', linestyle='--', linewidth=2.5, label='Theoretical PDF')
    
    ax2.set_xlabel('Sorted Sample Values (x)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Probability Density f(x)', fontsize=11, fontweight='bold')
    ax2.set_title('2. PDF Created Algebraically from CDF', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Pane 3: CDF from PDF
    ax3 = axes[2]
    ax3.plot(x_samples, cdf_reconstructed, color='#2ca02c', linewidth=3.0, label=f'Reconstructed CDF y(f)\n(R² = {r2_cdf_recon:.6f})')
    ax3.plot(x_samples, y_theoretical_cdf, color='#d62728', linestyle='--', linewidth=2.5, label='Theoretical CDF')
    ax3.axvline(sample_mean, color='gray', linestyle='--', alpha=0.7, label=f'Mean μ = {sample_mean:.2f}')
    
    ax3.set_xlabel('Sorted Sample Values (x)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Cumulative Probability P(X ≤ x)', fontsize=11, fontweight='bold')
    ax3.set_title('3. CDF Reconstructed Algebraically from PDF', fontsize=12, fontweight='bold', pad=10)
    ax3.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=9)
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    print(f"3-Pane Plot saved to: {output_plot_path}")

if __name__ == "__main__":
    main()
