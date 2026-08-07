import os
import torch
import numpy as np
import scipy.stats as stats
import scipy.special as special
import matplotlib.pyplot as plt

def cdf_to_pdf_algebraic(y_cdf, sigma):
    """1. CDF to PDF: f(y) = (1 / (sigma * sqrt(2pi))) * exp(-(erfinv(2y - 1))^2)."""
    z = special.erfinv(2.0 * y_cdf - 1.0)
    pdf = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-(z ** 2))
    return pdf

def pdf_to_derivative_algebraic(f_pdf, x_positions, mean, sigma):
    """2. PDF to Derivative: f'(f) = -/+ (f / sigma) * sqrt(-2 * ln(f * sigma * sqrt(2pi)))."""
    max_pdf = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    f_clamped = np.minimum(f_pdf, max_pdf * (1.0 - 1e-12))
    log_term = np.maximum(0.0, -2.0 * np.log(f_clamped * sigma * np.sqrt(2.0 * np.pi)))
    slope_mag = (f_pdf / sigma) * np.sqrt(log_term)
    
    # - for right branch (x >= mean), + for left branch (x < mean)
    pdf_deriv = np.where(x_positions >= mean, -slope_mag, slope_mag)
    return pdf_deriv

def cdf_to_integral_algebraic(y_cdf, f_pdf, sigma):
    """3. CDF to CDF Integral: I(y, f) = (sigma * sqrt(2) * erfinv(2y - 1)) * y + sigma^2 * f."""
    z = special.erfinv(2.0 * y_cdf - 1.0)
    cdf_integral = (sigma * np.sqrt(2.0) * z) * y_cdf + (sigma ** 2) * f_pdf
    return cdf_integral

def generate_samples(num_samples=2000, mean=1.5, std=2.0, seed=42):
    np.random.seed(seed)
    samples = np.random.normal(loc=mean, scale=std, size=num_samples)
    return np.sort(samples)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_plot_path = os.path.join(script_dir, "plot.png")
    
    N = 2000
    num_trials = 25
    true_mean, true_std = 1.5, 2.0
    
    # Dense Grid for Ideal Reference Curves
    x_grid = np.linspace(-5.0, 8.0, 1000)
    cdf_ideal_grid = stats.norm.cdf(x_grid, loc=true_mean, scale=true_std)
    pdf_ideal_grid = stats.norm.pdf(x_grid, loc=true_mean, scale=true_std)
    pdf_deriv_ideal_grid = -((x_grid - true_mean) / (true_std ** 2)) * pdf_ideal_grid
    cdf_integral_ideal_grid = (x_grid - true_mean) * cdf_ideal_grid + (true_std ** 2) * pdf_ideal_grid
    
    print(f"Running Ensemble Visualization: {num_trials} independent random sample draws (N={N} per trial)...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    
    # -------------------------------------------------------------
    # LOOP THROUGH 25 INDEPENDENT RANDOM TRIALS
    # -------------------------------------------------------------
    for trial_idx in range(num_trials):
        seed = 100 + trial_idx
        x_samples = generate_samples(num_samples=N, mean=true_mean, std=true_std, seed=seed)
        
        sample_mean = np.mean(x_samples)
        sample_std = np.std(x_samples, ddof=1)
        
        y_empirical_cdf = (np.arange(1, N + 1) - 0.5) / N
        pdf_algebraic_sample = cdf_to_pdf_algebraic(y_empirical_cdf, sigma=sample_std)
        pdf_deriv_sample = pdf_to_derivative_algebraic(pdf_algebraic_sample, x_samples, mean=sample_mean, sigma=sample_std)
        cdf_integral_sample = cdf_to_integral_algebraic(y_empirical_cdf, pdf_algebraic_sample, sigma=sample_std)
        
        # Label only the first trial for legend clarity
        label_cdf = 'Sample Trials (N=2000, 25 draws)' if trial_idx == 0 else None
        label_int = 'Sample Trials (Algebraic Integral)' if trial_idx == 0 else None
        label_pdf = 'Sample Trials (Algebraic PDF)' if trial_idx == 0 else None
        label_drv = 'Sample Trials (Algebraic Derivative)' if trial_idx == 0 else None
        
        # Plot fine, transparent lines (alpha=0.18, linewidth=0.7)
        axes[0, 0].plot(x_samples, y_empirical_cdf, color='#1f77b4', alpha=0.18, linewidth=0.7, label=label_cdf)
        axes[1, 0].plot(x_samples, cdf_integral_sample, color='#2ca02c', alpha=0.18, linewidth=0.7, label=label_int)
        axes[0, 1].plot(x_samples, pdf_algebraic_sample, color='#1f77b4', alpha=0.18, linewidth=0.7, label=label_pdf)
        axes[1, 1].plot(x_samples, pdf_deriv_sample, color='#9467bd', alpha=0.18, linewidth=0.7, label=label_drv)

    # -------------------------------------------------------------
    # OVERLAY BOLD IDEAL REFERENCE CURVES ON TOP
    # -------------------------------------------------------------
    # Top-Left: CDF
    ax1 = axes[0, 0]
    ax1.plot(x_grid, cdf_ideal_grid, color='#d62728', linewidth=2.8, label=f'Ideal CDF Φ(x; μ={true_mean}, σ={true_std})')
    ax1.axvline(true_mean, color='gray', linestyle='--', alpha=0.7, label=f'Mean μ = {true_mean}')
    ax1.set_xlabel('Sample Values (x)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Cumulative Probability P(X ≤ x)', fontsize=10, fontweight='bold')
    ax1.set_title('1. Cumulative Distribution Function (25 Sample Trails vs. Ideal)', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Bottom-Left: CDF Integral
    ax2 = axes[1, 0]
    ax2.plot(x_grid, cdf_integral_ideal_grid, color='#d62728', linewidth=2.8, label='Ideal CDF Integral ∫ Φ(t) dt')
    ax2.axvline(true_mean, color='gray', linestyle='--', alpha=0.7, label=f'Mean μ = {true_mean}')
    ax2.set_xlabel('Sample Values (x)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Integrated Area I(x)', fontsize=10, fontweight='bold')
    ax2.set_title('3. Integral of the CDF (25 Sample Trials vs. Ideal)', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Top-Right: PDF
    ax3 = axes[0, 1]
    ax3.plot(x_grid, pdf_ideal_grid, color='#d62728', linewidth=2.8, label='Ideal Gaussian PDF f(x)')
    ax3.axvline(true_mean, color='gray', linestyle='--', alpha=0.7, label=f'Mean μ = {true_mean}')
    ax3.set_xlabel('Sample Values (x)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Probability Density f(x)', fontsize=10, fontweight='bold')
    ax3.set_title('2. Probability Density Function (25 Sample Trials vs. Ideal)', fontsize=12, fontweight='bold', pad=10)
    ax3.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9)
    ax3.grid(True, linestyle='--', alpha=0.5)

    # Bottom-Right: PDF Derivative
    ax4 = axes[1, 1]
    ax4.plot(x_grid, pdf_deriv_ideal_grid, color='#d62728', linewidth=2.8, label="Ideal PDF Derivative f'(x)")
    ax4.axhline(0, color='black', linestyle='-', alpha=0.7)
    ax4.axvline(true_mean, color='gray', linestyle='--', alpha=0.7, label=f'Mean μ = {true_mean}')
    ax4.set_xlabel('Sample Values (x)', fontsize=10, fontweight='bold')
    ax4.set_ylabel("Slope f'(x) = df/dx", fontsize=10, fontweight='bold')
    ax4.set_title("4. Derivative of the PDF (25 Sample Trials vs. Ideal)", fontsize=12, fontweight='bold', pad=10)
    ax4.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9)
    ax4.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    print(f"25-Trial Ensemble Plot saved successfully to: {output_plot_path}")

if __name__ == "__main__":
    main()
