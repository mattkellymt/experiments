import os
import sys
import time
import gc
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import mlx.core as mx

print("=== Starting High-Sample-Size Apple Silicon MatMul Profiler (Batch=64, Iters=8) ===", flush=True)
print(f"PyTorch Version: {torch.__version__}", flush=True)
print(f"MPS Available: {torch.backends.mps.is_available()}", flush=True)
print(f"MLX Default Device: {mx.default_device()}", flush=True)

ART_DIR = "/Users/matt/.gemini/antigravity-cli/brain/56a3ae9f-46e5-4c0b-a4c9-6994d4111270"
EXP_DIR = "/Users/matt/projects/techstack/experiments/apple_silicon_matmul"
os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(ART_DIR, exist_ok=True)

def min_max_norm_minus_one_to_one(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val == min_val:
        return np.zeros_like(arr)
    return 2.0 * (arr - min_val) / (max_val - min_val) - 1.0

def safe_norm_and_plot(arr, title, filename):
    norm_mat = min_max_norm_minus_one_to_one(arr)
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    im = ax.imshow(norm_mat, cmap='coolwarm', vmin=-1.0, vmax=1.0, origin='lower')
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('Normalized Relative Latency (-1.0: Deep Blue/Fastest, +1.0: Deep Red/Slowest)', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Dimension N Modulo 32 (Matrix B Columns)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Dimension M Modulo 32 (Matrix A Rows)', fontsize=10, fontweight='bold')
    
    ticks = np.arange(0, 32, 4)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    for mod_val in range(0, 32, 8):
        ax.axhline(mod_val - 0.5, color='black', linewidth=0.8, linestyle='--' if mod_val % 16 != 0 else '-')
        ax.axvline(mod_val - 0.5, color='black', linewidth=0.8, linestyle='--' if mod_val % 16 != 0 else '-')
        
    min_idx = np.unravel_index(np.argmin(arr), arr.shape)
    max_idx = np.unravel_index(np.argmax(arr), arr.shape)
    ax.plot(min_idx[1], min_idx[0], 'g*', markersize=14, label=f'Fastest (-1.0): {arr[min_idx]:.3f}ms (M+{min_idx[0]}, N+{min_idx[1]})')
    ax.plot(max_idx[1], max_idx[0], 'rX', markersize=12, label=f'Slowest (+1.0): {arr[max_idx]:.3f}ms (M+{max_idx[0]}, N+{max_idx[1]})')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    
    mean_ms = np.mean(arr)
    min_ms = np.min(arr)
    max_ms = np.max(arr)
    diff_pct = ((max_ms - min_ms) / min_ms) * 100
    fig.text(0.5, 0.01, f'Mean: {mean_ms:.3f}ms | Min (-1.0): {min_ms:.3f}ms | Max (+1.0): {max_ms:.3f}ms | Latency Variance: +{diff_pct:.2f}%',
             ha='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.5', facecolor='gainsboro', alpha=0.8))
             
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    p_art = os.path.join(ART_DIR, filename)
    p_exp = os.path.join(EXP_DIR, filename)
    plt.savefig(p_art)
    plt.savefig(p_exp)
    plt.close()
    print(f"Saved heatmap to: {p_art} and {p_exp}", flush=True)

class MatMulBlock(nn.Module):
    def forward(self, a, b):
        return torch.bmm(a, b)

def benchmark_pytorch_mps(batch_size=64, offset=512, grid_size=32, dtype="fp32", use_compile=False, warmup=3, iters=8):
    device = torch.device("mps")
    runtimes = np.zeros((grid_size, grid_size), dtype=np.float64)
    torch_dtype = torch.float32 if dtype == "fp32" else torch.float16
    
    raw_mod = MatMulBlock().to(device)
    mod = torch.compile(raw_mod, backend="aot_eager") if use_compile else raw_mod
    
    dummy_a = torch.randn(batch_size, offset, offset, device=device, dtype=torch_dtype)
    dummy_b = torch.randn(batch_size, offset, offset, device=device, dtype=torch_dtype)
    for _ in range(warmup):
        _ = mod(dummy_a, dummy_b)
        torch.mps.synchronize()
    del dummy_a, dummy_b
    torch.mps.empty_cache()
    
    for i in range(grid_size):
        m = offset + i
        for j in range(grid_size):
            n = offset + j
            k = offset
            
            a = torch.randn(batch_size, m, k, device=device, dtype=torch_dtype)
            b = torch.randn(batch_size, k, n, device=device, dtype=torch_dtype)
            
            for _ in range(warmup):
                _ = mod(a, b)
                torch.mps.synchronize()
                
            t0 = time.perf_counter_ns()
            for _ in range(iters):
                _ = mod(a, b)
                torch.mps.synchronize()
            t1 = time.perf_counter_ns()
            
            runtimes[i, j] = ((t1 - t0) / (iters * 1e6))
            
            del a, b
            time.sleep(0.002)
            
        torch.mps.empty_cache()
        gc.collect()
        
    return runtimes

def benchmark_mlx_gpu(batch_size=64, offset=512, grid_size=32, dtype="fp32", warmup=3, iters=8):
    runtimes = np.zeros((grid_size, grid_size), dtype=np.float64)
    mx_dtype = mx.float32 if dtype == "fp32" else mx.float16
    
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
            
            t0 = time.perf_counter_ns()
            for _ in range(iters):
                c = mx.matmul(a, b)
                mx.eval(c)
                mx.synchronize()
            t1 = time.perf_counter_ns()
            
            runtimes[i, j] = ((t1 - t0) / (iters * 1e6))
            
            del a, b, c
            time.sleep(0.002)
            
        mx.clear_cache()
        gc.collect()
        
    return runtimes

def benchmark_pytorch_cpu(batch_size=64, offset=512, grid_size=32, dtype="fp32", warmup=2, iters=4):
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
                
            t0 = time.perf_counter_ns()
            for _ in range(iters):
                _ = torch.bmm(a, b)
            t1 = time.perf_counter_ns()
            
            runtimes[i, j] = ((t1 - t0) / (iters * 1e6))
            del a, b
            time.sleep(0.001)
            
    return runtimes

def run_safe_benchmarks():
    batch_size = 64
    offset = 512
    grid_size = 32
    
    print("\n--- 1. PyTorch MPS FP32 (High Sample Size: Batch=64, Iters=8) ---", flush=True)
    r_mps_fp32 = benchmark_pytorch_mps(batch_size, offset, grid_size, dtype="fp32", use_compile=False)
    safe_norm_and_plot(r_mps_fp32, f"PyTorch MPS FP32 Latency (Batch={batch_size}, Base={offset})", "heat.png")

    print("\n--- 2. PyTorch MPS FP32 (torch.compile, Batch=64) ---", flush=True)
    r_mps_compiled = benchmark_pytorch_mps(batch_size, offset, grid_size, dtype="fp32", use_compile=True)
    safe_norm_and_plot(r_mps_compiled, f"PyTorch MPS FP32 (torch.compile) Latency (Batch={batch_size}, Base={offset})", "heat_mps_compiled.png")

    print("\n--- 3. PyTorch MPS FP16 (Batch=64) ---", flush=True)
    r_mps_fp16 = benchmark_pytorch_mps(batch_size, offset, grid_size, dtype="fp16", use_compile=False)
    safe_norm_and_plot(r_mps_fp16, f"PyTorch MPS FP16 Latency (Batch={batch_size}, Base={offset})", "heat_mps_fp16.png")

    print("\n--- 4. Apple MLX GPU FP32 (Batch=64) ---", flush=True)
    r_mlx_fp32 = benchmark_mlx_gpu(batch_size, offset, grid_size, dtype="fp32")
    safe_norm_and_plot(r_mlx_fp32, f"Apple MLX FP32 Latency (Batch={batch_size}, Base={offset})", "heat_mlx_fp32.png")

    print("\n--- 5. PyTorch CPU FP32 (Batch=64) ---", flush=True)
    r_cpu_fp32 = benchmark_pytorch_cpu(batch_size, offset, grid_size, dtype="fp32")
    safe_norm_and_plot(r_cpu_fp32, f"PyTorch CPU (Accelerate) FP32 Latency (Batch={batch_size}, Base={offset})", "heat_cpu_fp32.png")

    fig, axes = plt.subplots(2, 2, figsize=(15, 13), dpi=300)
    plots = [
        (axes[0,0], r_mps_fp32, 'PyTorch MPS FP32 (Standard)'),
        (axes[0,1], r_mps_compiled, 'PyTorch MPS FP32 (torch.compile)'),
        (axes[1,0], r_mps_fp16, 'PyTorch MPS FP16 (Standard)'),
        (axes[1,1], r_mlx_fp32, 'Apple MLX GPU FP32')
    ]

    for ax, arr, ptitle in plots:
        norm_mat = min_max_norm_minus_one_to_one(arr)
        im = ax.imshow(norm_mat, cmap='coolwarm', vmin=-1.0, vmax=1.0, origin='lower')
        ax.set_title(ptitle, fontweight='bold', fontsize=11)
        ax.set_xlabel('N Modulo 32', fontsize=9)
        ax.set_ylabel('M Modulo 32', fontsize=9)
        for mod_val in range(0, 32, 8):
            ax.axhline(mod_val - 0.5, color='black', linewidth=0.5, linestyle='--' if mod_val % 16 != 0 else '-')
            ax.axvline(mod_val - 0.5, color='black', linewidth=0.5, linestyle='--' if mod_val % 16 != 0 else '-')
            
        min_ms = np.min(arr)
        max_ms = np.max(arr)
        mean_ms = np.mean(arr)
        ax.text(0.02, 0.95, f'Mean: {mean_ms:.2f}ms\nMin (-1.0): {min_ms:.2f}ms\nMax (+1.0): {max_ms:.2f}ms\nDelta: +{((max_ms-min_ms)/min_ms)*100:.1f}%',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Normalized Dynamic Range (-1.0: Deep Blue, +1.0: Deep Red)')
    fig.suptitle(f"Apple M4 Pro Matrix Multiplication Profiling: High Sample Size (Batch={batch_size})\nBase Dimension = {offset}, 8 Iterations Per Shape Point", fontsize=13, fontweight='bold', y=0.98)
    
    p_art_multi = os.path.join(ART_DIR, 'heat_all_backends.png')
    p_exp_multi = os.path.join(EXP_DIR, 'heat_all_backends.png')
    plt.savefig(p_art_multi, bbox_inches='tight')
    plt.savefig(p_exp_multi, bbox_inches='tight')
    plt.close()

    data = {
        'mps_fp32': r_mps_fp32.tolist(),
        'mps_compiled': r_mps_compiled.tolist(),
        'mps_fp16': r_mps_fp16.tolist(),
        'mlx_fp32': r_mlx_fp32.tolist(),
        'cpu_fp32': r_cpu_fp32.tolist()
    }
    with open(os.path.join(ART_DIR, 'benchmark_data.json'), 'w') as f:
        json.dump(data, f)
    with open(os.path.join(EXP_DIR, 'benchmark_data.json'), 'w') as f:
        json.dump(data, f)

    print("=== HIGH-SAMPLE-SIZE BENCHMARKING COMPLETED SUCCESSFULLY! ===", flush=True)

if __name__ == "__main__":
    run_safe_benchmarks()
