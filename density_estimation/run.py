import os
import time
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
    
    pdf_deriv = np.where(x_positions >= mean, -slope_mag, slope_mag)
    return pdf_deriv

def cdf_to_integral_algebraic(y_cdf, f_pdf, sigma):
    """3. CDF to CDF Integral: I(y, f) = (sigma * sqrt(2) * erfinv(2y - 1)) * y + sigma^2 * f."""
    z = special.erfinv(2.0 * y_cdf - 1.0)
    cdf_integral = (sigma * np.sqrt(2.0) * z) * y_cdf + (sigma ** 2) * f_pdf
    return cdf_integral

def generate_samples(num_samples=2000, mean=1.5, std=2.0, seed=42):
    """Generate authentic samples with mean = 1.5, std = 2.0."""
    np.random.seed(seed)
    samples = np.random.normal(loc=mean, scale=std, size=num_samples)
    return np.sort(samples)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_plot_path = os.path.join(script_dir, "plot.png")
    
    N = 2000
    M = 4096  # 4,096 independent random sample trials
    true_mean, true_std = 1.5, 2.0
    
    x_grid = np.linspace(-4.0, 7.0, 500)
    num_grid = len(x_grid)
    
    # Real Un-Normalized Ideal Reference Curves
    cdf_ideal = stats.norm.cdf(x_grid, loc=true_mean, scale=true_std)
    pdf_ideal = stats.norm.pdf(x_grid, loc=true_mean, scale=true_std)
    pdf_deriv_ideal = -((x_grid - true_mean) / (true_std ** 2)) * pdf_ideal
    cdf_int_ideal = (x_grid - true_mean) * cdf_ideal + (true_std ** 2) * pdf_ideal
    
    cdf_trials = np.zeros((M, num_grid))
    pdf_trials = np.zeros((M, num_grid))
    pdf_deriv_trials = np.zeros((M, num_grid))
    cdf_int_trials = np.zeros((M, num_grid))
    
    y_empirical_cdf = (np.arange(1, N + 1) - 0.5) / N
    
    print(f"Running M={M} trials with authentic un-normalized values (μ={true_mean}, σ={true_std})...")
    
    for m in range(M):
        seed = 40000 + m
        x_samples = generate_samples(num_samples=N, mean=true_mean, std=true_std, seed=seed)
        
        sample_mean = np.mean(x_samples)
        sample_std = np.std(x_samples, ddof=1)
        
        pdf_sample = cdf_to_pdf_algebraic(y_empirical_cdf, sigma=sample_std)
        pdf_deriv_sample = pdf_to_derivative_algebraic(pdf_sample, x_samples, mean=sample_mean, sigma=sample_std)
        cdf_int_sample = cdf_to_integral_algebraic(y_empirical_cdf, pdf_sample, sigma=sample_std)
        
        cdf_trials[m] = np.interp(x_grid, x_samples, y_empirical_cdf)
        pdf_trials[m] = np.interp(x_grid, x_samples, pdf_sample)
        pdf_deriv_trials[m] = np.interp(x_grid, x_samples, pdf_deriv_sample)
        cdf_int_trials[m] = np.interp(x_grid, x_samples, cdf_int_sample)

    # Real Un-Normalized Residual Standard Deviations in actual physical units
    std_res_cdf = np.std(cdf_trials - cdf_ideal, axis=0)
    std_res_pdf = np.std(pdf_trials - pdf_ideal, axis=0)
    std_res_cdf_int = np.std(cdf_int_trials - cdf_int_ideal, axis=0)
    std_res_pdf_deriv = np.std(pdf_deriv_trials - pdf_deriv_ideal, axis=0)

    min_cdf, max_cdf = np.min(cdf_trials, axis=0), np.max(cdf_trials, axis=0)
    min_pdf, max_pdf = np.min(pdf_trials, axis=0), np.max(pdf_trials, axis=0)
    min_int, max_int = np.min(cdf_int_trials, axis=0), np.max(cdf_int_trials, axis=0)
    min_deriv, max_deriv = np.min(pdf_deriv_trials, axis=0), np.max(pdf_deriv_trials, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    
    color_cdf = '#1f77b4'      # Blue
    color_pdf = '#ff7f0e'      # Orange
    color_int = '#2ca02c'      # Green
    color_deriv = '#9467bd'    # Purple
    
    # -------------------------------------------------------------
    # PANE 1 (Top-Left): CDF (Real Values)
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.fill_between(x_grid, min_cdf, max_cdf, color=color_cdf, alpha=0.25, label='Min-Max Bounds (4096 Trials)')
    ax1.plot(x_grid, cdf_ideal, color='black', linewidth=2.5, label='Ideal CDF Φ(x)')
    ax1.set_xlabel('Sample Values (x)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Cumulative Probability P(X ≤ x)', fontsize=10, fontweight='bold')
    ax1.set_title('1. Cumulative Distribution Function (CDF)', fontsize=12, fontweight='bold', pad=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x_grid, std_res_cdf, color=color_cdf, linewidth=2.2, linestyle='-', label='Residual Std Dev σ_res(x)')
    ax1_twin.set_ylabel('Residual Std Dev σ_res(x)', color=color_cdf, fontsize=10, fontweight='bold')
    ax1_twin.tick_params(axis='y', labelcolor=color_cdf)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_1_twin, labels_1_twin = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_1_twin, labels_1 + labels_1_twin, loc='lower right', frameon=True, framealpha=0.9, fontsize=8)

    # -------------------------------------------------------------
    # PANE 2 (Top-Right): PDF (Real Values)
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.fill_between(x_grid, min_pdf, max_pdf, color=color_pdf, alpha=0.25, label='Min-Max Bounds (4096 Trials)')
    ax2.plot(x_grid, pdf_ideal, color='black', linewidth=2.5, label='Ideal Gaussian PDF f(x)')
    ax2.set_xlabel('Sample Values (x)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Probability Density f(x)', fontsize=10, fontweight='bold')
    ax2.set_title('2. Probability Density Function (PDF)', fontsize=12, fontweight='bold', pad=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x_grid, std_res_pdf, color=color_pdf, linewidth=2.2, linestyle='-', label='Residual Std Dev σ_res(x)')
    ax2_twin.set_ylabel('Residual Std Dev σ_res(x)', color=color_pdf, fontsize=10, fontweight='bold')
    ax2_twin.tick_params(axis='y', labelcolor=color_pdf)
    
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines_2_twin, labels_2_twin = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_2 + lines_2_twin, labels_2 + labels_2_twin, loc='upper right', frameon=True, framealpha=0.9, fontsize=8)

    # -------------------------------------------------------------
    # PANE 3 (Bottom-Left): CDF Integral (Real Values)
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.fill_between(x_grid, min_int, max_int, color=color_int, alpha=0.25, label='Min-Max Bounds (4096 Trials)')
    ax3.plot(x_grid, cdf_int_ideal, color='black', linewidth=2.5, label='Ideal CDF Integral ∫ Φ(t) dt')
    ax3.set_xlabel('Sample Values (x)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Integrated Area I(x)', fontsize=10, fontweight='bold')
    ax3.set_title('3. Integral of the CDF', fontsize=12, fontweight='bold', pad=10)
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(x_grid, std_res_cdf_int, color=color_int, linewidth=2.2, linestyle='-', label='Residual Std Dev σ_res(x)')
    ax3_twin.set_ylabel('Residual Std Dev σ_res(x)', color=color_int, fontsize=10, fontweight='bold')
    ax3_twin.tick_params(axis='y', labelcolor=color_int)
    
    lines_3, labels_3 = ax3.get_legend_handles_labels()
    lines_3_twin, labels_3_twin = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines_3 + lines_3_twin, labels_3 + labels_3_twin, loc='lower right', frameon=True, framealpha=0.9, fontsize=8)

    # -------------------------------------------------------------
    # PANE 4 (Bottom-Right): PDF Derivative (Real Values)
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.fill_between(x_grid, min_deriv, max_deriv, color=color_deriv, alpha=0.25, label='Min-Max Bounds (4096 Trials)')
    ax4.plot(x_grid, pdf_deriv_ideal, color='black', linewidth=2.5, label="Ideal PDF Derivative f'(x)")
    ax4.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax4.set_xlabel('Sample Values (x)', fontsize=10, fontweight='bold')
    ax4.set_ylabel("Slope f'(x) = df/dx", fontsize=10, fontweight='bold')
    ax4.set_title("4. Derivative of the PDF", fontsize=12, fontweight='bold', pad=10)
    ax4.grid(True, linestyle='--', alpha=0.5)
    
    ax4_twin = ax4.twinx()
    ax4_twin.plot(x_grid, std_res_pdf_deriv, color=color_deriv, linewidth=2.2, linestyle='-', label='Residual Std Dev σ_res(x)')
    ax4_twin.set_ylabel('Residual Std Dev σ_res(x)', color=color_deriv, fontsize=10, fontweight='bold')
    ax4_twin.tick_params(axis='y', labelcolor=color_deriv)
    
    lines_4, labels_4 = ax4.get_legend_handles_labels()
    lines_4_twin, labels_4_twin = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines_4 + lines_4_twin, labels_4 + labels_4_twin, loc='upper right', frameon=True, framealpha=0.9, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    print(f"Authentic Un-Normalized Plot saved successfully to: {output_plot_path}")

if __name__ == "__main__":
    main()
