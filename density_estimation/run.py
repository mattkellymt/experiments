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

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_plot_path = os.path.join(script_dir, "plot.png")
    
    N = 2000
    M = 65536  # 2^16 = 65,536 independent random sample trials
    true_mean, true_std = 1.5, 2.0
    
    x_grid = np.linspace(-4.0, 7.0, 500)
    num_grid = len(x_grid)
    
    cdf_ideal = stats.norm.cdf(x_grid, loc=true_mean, scale=true_std)
    pdf_ideal = stats.norm.pdf(x_grid, loc=true_mean, scale=true_std)
    pdf_deriv_ideal = -((x_grid - true_mean) / (true_std ** 2)) * pdf_ideal
    cdf_int_ideal = (x_grid - true_mean) * cdf_ideal + (true_std ** 2) * pdf_ideal
    
    # Online accumulators for 65,536 trials
    min_cdf, max_cdf = np.full(num_grid, 1e9), np.full(num_grid, -1e9)
    min_pdf, max_pdf = np.full(num_grid, 1e9), np.full(num_grid, -1e9)
    min_int, max_int = np.full(num_grid, 1e9), np.full(num_grid, -1e9)
    min_deriv, max_deriv = np.full(num_grid, 1e9), np.full(num_grid, -1e9)
    
    sum_cdf, sum_sq_cdf = np.zeros(num_grid), np.zeros(num_grid)
    sum_pdf, sum_sq_pdf = np.zeros(num_grid), np.zeros(num_grid)
    sum_int, sum_sq_int = np.zeros(num_grid), np.zeros(num_grid)
    sum_deriv, sum_sq_deriv = np.zeros(num_grid), np.zeros(num_grid)
    
    y_empirical_cdf = (np.arange(1, N + 1) - 0.5) / N
    
    print(f"Running M={M} (2^16 = 65,536) trials ({M * N:,} total sample points)...")
    t0 = time.time()
    
    chunk_size = 2048
    num_chunks = M // chunk_size
    
    for c in range(num_chunks):
        np.random.seed(4242 + c)
        samples_chunk = np.sort(np.random.normal(true_mean, true_std, size=(chunk_size, N)), axis=1)
        
        for i in range(chunk_size):
            x_samples = samples_chunk[i]
            sample_mean = np.mean(x_samples)
            sample_std = np.std(x_samples, ddof=1)
            
            pdf_sample = cdf_to_pdf_algebraic(y_empirical_cdf, sigma=sample_std)
            pdf_deriv_sample = pdf_to_derivative_algebraic(pdf_sample, x_samples, mean=sample_mean, sigma=sample_std)
            cdf_int_sample = cdf_to_integral_algebraic(y_empirical_cdf, pdf_sample, sigma=sample_std)
            
            # Interpolate onto x_grid
            c_interp = np.interp(x_grid, x_samples, y_empirical_cdf)
            p_interp = np.interp(x_grid, x_samples, pdf_sample)
            d_interp = np.interp(x_grid, x_samples, pdf_deriv_sample)
            i_interp = np.interp(x_grid, x_samples, cdf_int_sample)
            
            # Min / Max bounds
            min_cdf = np.minimum(min_cdf, c_interp)
            max_cdf = np.maximum(max_cdf, c_interp)
            
            min_pdf = np.minimum(min_pdf, p_interp)
            max_pdf = np.maximum(max_pdf, p_interp)
            
            min_deriv = np.minimum(min_deriv, d_interp)
            max_deriv = np.maximum(max_deriv, d_interp)
            
            min_int = np.minimum(min_int, i_interp)
            max_int = np.maximum(max_int, i_interp)
            
            # Residual accumulators (sample - ideal)
            r_c = c_interp - cdf_ideal
            r_p = p_interp - pdf_ideal
            r_d = d_interp - pdf_deriv_ideal
            r_i = i_interp - cdf_int_ideal
            
            sum_cdf += r_c
            sum_sq_cdf += r_c ** 2
            
            sum_pdf += r_p
            sum_sq_pdf += r_p ** 2
            
            sum_deriv += r_d
            sum_sq_deriv += r_d ** 2
            
            sum_int += r_i
            sum_sq_int += r_i ** 2

    # Compute Residual Standard Deviations across 65,536 trials
    std_res_cdf = np.sqrt(np.maximum(0.0, (sum_sq_cdf / M) - (sum_cdf / M) ** 2))
    std_res_pdf = np.sqrt(np.maximum(0.0, (sum_sq_pdf / M) - (sum_pdf / M) ** 2))
    std_res_pdf_deriv = np.sqrt(np.maximum(0.0, (sum_sq_deriv / M) - (sum_deriv / M) ** 2))
    std_res_cdf_int = np.sqrt(np.maximum(0.0, (sum_sq_int / M) - (sum_int / M) ** 2))
    
    t1 = time.time()
    print(f"Completed 65,536 trials in {t1 - t0:.2f} seconds!")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    
    # Custom Pane Colors
    color_cdf = '#1f77b4'      # Blue for Pane 1 (CDF)
    color_pdf = '#ff7f0e'      # Orange for Pane 2 (PDF)
    color_int = '#2ca02c'      # Green for Pane 3 (CDF Integral)
    color_deriv = '#9467bd'    # Purple for Pane 4 (PDF Derivative)
    
    # -------------------------------------------------------------
    # PANE 1 (Top-Left): CDF (Blue Palette + Shaded Min-Max Band)
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.fill_between(x_grid, min_cdf, max_cdf, color=color_cdf, alpha=0.25, label='Min-Max Bounds (65,536 Trials)')
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
    # PANE 2 (Top-Right): PDF (Orange Palette + Shaded Min-Max Band)
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.fill_between(x_grid, min_pdf, max_pdf, color=color_pdf, alpha=0.25, label='Min-Max Bounds (65,536 Trials)')
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
    # PANE 3 (Bottom-Left): CDF Integral (Green Palette + Shaded Min-Max Band)
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.fill_between(x_grid, min_int, max_int, color=color_int, alpha=0.25, label='Min-Max Bounds (65,536 Trials)')
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
    # PANE 4 (Bottom-Right): PDF Derivative (Purple Palette + Shaded Min-Max Band)
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.fill_between(x_grid, min_deriv, max_deriv, color=color_deriv, alpha=0.25, label='Min-Max Bounds (65,536 Trials)')
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
    print(f"65,536-Trial Shaded Envelope Plot saved successfully to: {output_plot_path}")

if __name__ == "__main__":
    main()
