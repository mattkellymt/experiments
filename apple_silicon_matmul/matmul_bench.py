import os
import sys
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import mlx.core as mx

print("=== Starting Apple Silicon Matrix Multiplication Profiler ===", flush=True)
print(f"PyTorch Version: {torch.__version__}", flush=True)
print(f"MPS Available: {torch.backends.mps.is_available()}", flush=True)
print(f"MLX Default Device: {mx.default_device()}", flush=True)

EXP_DIR = "/Users/matt/projects/techstack/experiments/apple_silicon_matmul"
os.makedirs(EXP_DIR, exist_ok=True)

def benchmark_pytorch_mps(batch_size, offset, grid_size, dtype, warmup=3, iters=8):
    """
    Benchmark PyTorch MPS (GPU) matmul over a grid of A(M, K) and B(K, N).
    M = offset + i, N = offset + j, K = offset.
    Returns 2D grid of mean execution times in milliseconds.
    """
    device = torch.device("mps")
    runtimes = np.zeros((grid_size, grid_size), dtype=np.float64)
    
    torch_dtype = torch.float32 if dtype == "fp32" else torch.float16
    
    # Warmup MPS device
    dummy_a = torch.randn(batch_size, offset, offset, device=device, dtype=torch_dtype)
    dummy_b = torch.randn(batch_size, offset, offset, device=device, dtype=torch_dtype)
    for _ in range(warmup):
        _ = torch.bmm(dummy_a, dummy_b)
        torch.mps.synchronize()
    
    for i in range(grid_size):
        m = offset + i
        for j in range(grid_size):
            n = offset + j
            k = offset
            
            # Create tensors
            a = torch.randn(batch_size, m, k, device=device, dtype=torch_dtype)
            b = torch.randn(batch_size, k, n, device=device, dtype=torch_dtype)
            
            # Warmup for this specific shape
            for _ in range(warmup):
                _ = torch.bmm(a, b)
                torch.mps.synchronize()
                
            # Benchmark
            start_time = time.perf_counter_ns()
            for _ in range(iters):
                _ = torch.bmm(a, b)
                torch.mps.synchronize()
            end_time = time.perf_counter_ns()
            
            avg_ms = ((end_time - start_time) / 1e6) / iters
            runtimes[i, j] = avg_ms
            
    return runtimes

def benchmark_pytorch_cpu(batch_size, offset, grid_size, dtype, warmup=2, iters=5):
    """
    Benchmark PyTorch CPU matmul over a grid of A(M, K) and B(K, N).
    M = offset + i, N = offset + j, K = offset.
    Returns 2D grid of mean execution times in milliseconds.
    """
    device = torch.device("cpu")
    runtimes = np.zeros((grid_size, grid_size), dtype=np.float64)
    torch_dtype = torch.float32 if dtype == "fp32" else torch.float16
    
    for i in range(grid_size):
        m = offset + i
        for j in range(grid_size):
            n = offset + j
            k = offset
            
            a = torch.randn(batch_size, m, k, device=device, dtype=torch_dtype)
            b = torch.randn(batch_size, k, n, device=device, dtype=torch_dtype)
            
            for _ in range(warmup):
                _ = torch.bmm(a, b)
                
            start_time = time.perf_counter_ns()
            for _ in range(iters):
                _ = torch.bmm(a, b)
            end_time = time.perf_counter_ns()
            
            avg_ms = ((end_time - start_time) / 1e6) / iters
            runtimes[i, j] = avg_ms
            
    return runtimes

def benchmark_mlx_gpu(batch_size, offset, grid_size, dtype, warmup=3, iters=8):
    """
    Benchmark Apple MLX GPU matmul over a grid of A(M, K) and B(K, N).
    M = offset + i, N = offset + j, K = offset.
    Returns 2D grid of mean execution times in milliseconds.
    """
    runtimes = np.zeros((grid_size, grid_size), dtype=np.float64)
    mx_dtype = mx.float32 if dtype == "fp32" else mx.float16
    
    # Warmup MLX
    dummy_a = mx.random.normal((batch_size, offset, offset), dtype=mx_dtype)
    dummy_b = mx.random.normal((batch_size, offset, offset), dtype=mx_dtype)
    for _ in range(warmup):
        c = mx.matmul(dummy_a, dummy_b)
        mx.eval(c)
        mx.synchronize()
        
    for i in range(grid_size):
        m = offset + i
        for j in range(grid_size):
            n = offset + j
            k = offset
            
            a = mx.random.normal((batch_size, m, k), dtype=mx_dtype)
            b = mx.random.normal((batch_size, k, n), dtype=mx_dtype)
            
            for _ in range(warmup):
                c = mx.matmul(a, b)
                mx.eval(c)
                mx.synchronize()
                
            start_time = time.perf_counter_ns()
            for _ in range(iters):
                c = mx.matmul(a, b)
                mx.eval(c)
                mx.synchronize()
            end_time = time.perf_counter_ns()
            
            avg_ms = ((end_time - start_time) / 1e6) / iters
            runtimes[i, j] = avg_ms
            
    return runtimes

def normalize_to_minus_one_one(matrix):
    """
    Normalize 2D matrix into [-1, 1] relative to the mean runtime.
    0 represents mean runtime.
    -1 represents the fastest (minimum runtime).
    +1 represents the slowest (maximum runtime).
    """
    mean_val = np.mean(matrix)
    max_dev = max(np.max(matrix) - mean_val, mean_val - np.min(matrix))
    if max_dev == 0:
        return np.zeros_like(matrix)
    norm_matrix = (matrix - mean_val) / max_dev
    return norm_matrix

def plot_heatmap(norm_matrix, raw_matrix, title, filename, offset, grid_size=32):
    """
    Plot coolwarm heatmap with clear labels, grid lines on mod 8/16/32 alignment.
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    im = ax.imshow(norm_matrix, cmap='coolwarm', vmin=-1.0, vmax=1.0, origin='lower')
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('Normalized Relative Runtime (-1: Faster, +1: Slower)', fontsize=11, fontweight='bold')
    
    # Labels and Titles
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel(f'Dimension N Modulo 32 (Matrix B Columns = {offset} + N_mod)', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'Dimension M Modulo 32 (Matrix A Rows = {offset} + M_mod)', fontsize=11, fontweight='bold')
    
    # Ticks
    ticks = np.arange(0, grid_size, 4)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}" for t in ticks])
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t}" for t in ticks])
    
    # Add major grid lines on multiples of 8 and 16 and 32
    for mod_val in range(0, grid_size, 8):
        ax.axhline(mod_val - 0.5, color='black', linewidth=0.8, linestyle='--' if mod_val % 16 != 0 else '-')
        ax.axvline(mod_val - 0.5, color='black', linewidth=0.8, linestyle='--' if mod_val % 16 != 0 else '-')
        
    # Annotate min/max points
    min_idx = np.unravel_index(np.argmin(raw_matrix), raw_matrix.shape)
    max_idx = np.unravel_index(np.argmax(raw_matrix), raw_matrix.shape)
    
    ax.plot(min_idx[1], min_idx[0], 'g*', markersize=14, label=f'Fastest: {raw_matrix[min_idx]:.3f}ms (M+{min_idx[0]}, N+{min_idx[1]})')
    ax.plot(max_idx[1], max_idx[0], 'rX', markersize=12, label=f'Slowest: {raw_matrix[max_idx]:.3f}ms (M+{max_idx[0]}, N+{max_idx[1]})')
    
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    
    # Annotate summary statistics at bottom
    mean_ms = np.mean(raw_matrix)
    min_ms = np.min(raw_matrix)
    max_ms = np.max(raw_matrix)
    diff_pct = ((max_ms - min_ms) / min_ms) * 100
    
    fig.text(0.5, 0.01, 
             f"Mean: {mean_ms:.3f} ms | Min: {min_ms:.3f} ms | Max: {max_ms:.3f} ms | Delta: +{diff_pct:.2f}% variance across alignment",
             ha='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='gainsboro', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    filepath = os.path.join(EXP_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved heatmap to: {filepath}", flush=True)

def run_all_experiments():
    batch_size = 32
    offset = 1024
    grid_size = 32
    
    results = {}
    
    print("\n--- 1. Running PyTorch MPS (Metal GPU) FP32 Benchmark ---", flush=True)
    raw_mps_fp32 = benchmark_pytorch_mps(batch_size, offset, grid_size, dtype="fp32")
    norm_mps_fp32 = normalize_to_minus_one_one(raw_mps_fp32)
    plot_heatmap(norm_mps_fp32, raw_mps_fp32, 
                 f"PyTorch MPS (Metal GPU) FP32 MatMul Latency (Batch={batch_size}, Base={offset})",
                 "heat.png", offset=offset, grid_size=grid_size)
    results["mps_fp32"] = raw_mps_fp32.tolist()
    
    print("\n--- 2. Running PyTorch MPS (Metal GPU) FP16 Benchmark ---", flush=True)
    raw_mps_fp16 = benchmark_pytorch_mps(batch_size, offset, grid_size, dtype="fp16")
    norm_mps_fp16 = normalize_to_minus_one_one(raw_mps_fp16)
    plot_heatmap(norm_mps_fp16, raw_mps_fp16, 
                 f"PyTorch MPS (Metal GPU) FP16 MatMul Latency (Batch={batch_size}, Base={offset})",
                 "heat_mps_fp16.png", offset=offset, grid_size=grid_size)
    results["mps_fp16"] = raw_mps_fp16.tolist()

    print("\n--- 3. Running MLX GPU FP32 Benchmark ---", flush=True)
    raw_mlx_fp32 = benchmark_mlx_gpu(batch_size, offset, grid_size, dtype="fp32")
    norm_mlx_fp32 = normalize_to_minus_one_one(raw_mlx_fp32)
    plot_heatmap(norm_mlx_fp32, raw_mlx_fp32, 
                 f"Apple MLX (Metal GPU) FP32 MatMul Latency (Batch={batch_size}, Base={offset})",
                 "heat_mlx_fp32.png", offset=offset, grid_size=grid_size)
    results["mlx_fp32"] = raw_mlx_fp32.tolist()

    print("\n--- 4. Running PyTorch CPU FP32 Benchmark ---", flush=True)
    raw_cpu_fp32 = benchmark_pytorch_cpu(batch_size, offset, grid_size, dtype="fp32")
    norm_cpu_fp32 = normalize_to_minus_one_one(raw_cpu_fp32)
    plot_heatmap(norm_cpu_fp32, raw_cpu_fp32, 
                 f"PyTorch CPU (Accelerate/NEON) FP32 MatMul Latency (Batch={batch_size}, Base={offset})",
                 "heat_cpu_fp32.png", offset=offset, grid_size=grid_size)
    results["cpu_fp32"] = raw_cpu_fp32.tolist()

    # Create multi-panel comparative figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=300)
    
    plots_config = [
        (axes[0, 0], norm_mps_fp32, raw_mps_fp32, "PyTorch MPS (Metal GPU) FP32"),
        (axes[0, 1], norm_mps_fp16, raw_mps_fp16, "PyTorch MPS (Metal GPU) FP16"),
        (axes[1, 0], norm_mlx_fp32, raw_mlx_fp32, "Apple MLX (Metal GPU) FP32"),
        (axes[1, 1], norm_cpu_fp32, raw_cpu_fp32, "PyTorch CPU (Accelerate) FP32")
    ]
    
    for ax, norm_mat, raw_mat, ptitle in plots_config:
        im = ax.imshow(norm_mat, cmap='coolwarm', vmin=-1.0, vmax=1.0, origin='lower')
        ax.set_title(ptitle, fontsize=12, fontweight='bold')
        ax.set_xlabel('N Modulo 32 (Offset + N)', fontsize=9)
        ax.set_ylabel('M Modulo 32 (Offset + M)', fontsize=9)
        
        # Grid lines for multiples of 8, 16, 32
        for mod_val in range(0, grid_size, 8):
            ax.axhline(mod_val - 0.5, color='black', linewidth=0.5, linestyle='--' if mod_val % 16 != 0 else '-')
            ax.axvline(mod_val - 0.5, color='black', linewidth=0.5, linestyle='--' if mod_val % 16 != 0 else '-')
            
        min_ms = np.min(raw_mat)
        max_ms = np.max(raw_mat)
        mean_ms = np.mean(raw_mat)
        ax.text(0.02, 0.95, f"Mean: {mean_ms:.2f}ms\nMin: {min_ms:.2f}ms\nMax: {max_ms:.2f}ms\nRange: +{((max_ms-min_ms)/min_ms)*100:.1f}%", 
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Normalized Relative Runtime (-1: Faster, +1: Slower)')
    fig.suptitle(f"Apple M4 Pro Matrix Multiplication Profiling: Dimension Modulo 32 Alignment\nBatch Size = {batch_size}, Base Dimension = {offset}", 
                 fontsize=14, fontweight='bold', y=0.98)
    
    multi_filepath = os.path.join(EXP_DIR, "heat_all_backends.png")
    plt.savefig(multi_filepath, bbox_inches='tight')
    plt.close()
    print(f"\nSaved combined comparison heatmap to: {multi_filepath}", flush=True)
    
    # Save raw json output
    json_path = os.path.join(EXP_DIR, "benchmark_data.json")
    with open(json_path, "w") as f:
        json.dump(results, f)
    print(f"Saved benchmark raw data to: {json_path}", flush=True)
    print("=== Matrix Multiplication Profiling Completed Successfully! ===", flush=True)

if __name__ == "__main__":
    run_all_experiments()
