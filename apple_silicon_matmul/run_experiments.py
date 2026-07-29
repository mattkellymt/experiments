import time
import json
import gc
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

# Configuration for 2D Sweep
OFFSET = 1024
SWEEP_RANGE = 64 # 0 to 63 inclusive
K = 1024
M_RANGE = list(range(OFFSET, OFFSET + SWEEP_RANGE))
N_RANGE = list(range(OFFSET, OFFSET + SWEEP_RANGE))

# Reduced iterations due to 4096 combinations per hardware target (64x64)
WARMUP = 1
ITERS = 3

ENGINES = ["MPS", "MLX"]
PRECISIONS = ["FP32", "FP16", "BF16"]

import multiprocessing

# Configure PyTorch to use all available cores for FP32 CPU math
try:
    num_cores = multiprocessing.cpu_count()
    torch.set_num_threads(num_cores)
except:
    pass

def get_torch_dtype(prec, engine="CPU"):
    if prec == "FP32":
        return torch.float32
    elif prec == "FP16":
        if engine == "CPU":
            return None
        return torch.float16
    elif prec == "BF16":
        if engine == "CPU":
            return None
        return torch.bfloat16
    return None

def get_mlx_dtype(prec):
    if prec == "FP32":
        return mx.float32
    elif prec == "FP16":
        return mx.float16
    elif prec == "BF16":
        return mx.bfloat16
    return None

def run_torch_bench(device, dtype, M, N, mm_comp):
    try:
        a = torch.randn(M, K, dtype=dtype, device=device)
        b = torch.randn(K, N, dtype=dtype, device=device)

        # Warmup
        for _ in range(WARMUP):
            res = mm_comp(a, b)
        
        if device == "mps":
            torch.mps.synchronize()
            
        start = time.perf_counter()
        for _ in range(ITERS):
            res = mm_comp(a, b)
        
        if device == "mps":
            torch.mps.synchronize()
            
        end = time.perf_counter()
        
        return ((end - start) / ITERS) * 1000.0  # ms
    except Exception as e:
        return None

def run_mlx_bench(dtype, M, N, mm_comp):
    try:
        a = mx.random.normal((M, K), dtype=dtype)
        b = mx.random.normal((K, N), dtype=dtype)
        
        # Warmup
        for _ in range(WARMUP):
            res = mm_comp(a, b)
            mx.eval(res)
            
        start = time.perf_counter()
        for _ in range(ITERS):
            res = mm_comp(a, b)
        mx.eval(res)
        end = time.perf_counter()
        
        return ((end - start) / ITERS) * 1000.0  # ms
    except Exception as e:
        return None

def get_compiled_torch(device, dtype):
    def mm(x, y):
        return torch.matmul(x, y)
    return mm

def get_compiled_mlx():
    def mm(x, y):
        return mx.matmul(x, y)
    return mx.compile(mm)

def main():
    print("Starting clean-room 2D matrix multiplication benchmarking...")
    results = {engine: {prec: {} for prec in PRECISIONS} for engine in ENGINES}

    for engine in ENGINES:
        for prec in PRECISIONS:
            print(f"Benchmarking {engine} | {prec}...")
            
            if engine in ["CPU", "MPS"] and not HAS_TORCH:
                print("  Skipped (Torch not available)")
                continue
            if engine == "MLX" and not HAS_MLX:
                print("  Skipped (MLX not available)")
                continue
            
            torch_comp = None
            mlx_comp = None
            if engine == "CPU":
                dtype = get_torch_dtype(prec, engine)
                if dtype is not None:
                    torch_comp = get_compiled_torch("cpu", dtype)
            elif engine == "MPS":
                dtype = get_torch_dtype(prec, engine)
                if dtype is not None and torch.backends.mps.is_available():
                    torch_comp = get_compiled_torch("mps", dtype)
            elif engine == "MLX":
                dtype = get_mlx_dtype(prec)
                if dtype is not None:
                    mlx_comp = get_compiled_mlx()

            for M in M_RANGE:
                for N in N_RANGE:
                    val = None
                    if engine == "CPU":
                        dtype = get_torch_dtype(prec, engine)
                        if dtype is not None and torch_comp is not None:
                            val = run_torch_bench("cpu", dtype, M, N, torch_comp)
                    elif engine == "MPS":
                        if not torch.backends.mps.is_available():
                            val = None
                        else:
                            dtype = get_torch_dtype(prec, engine)
                            if dtype is not None and torch_comp is not None:
                                val = run_torch_bench("mps", dtype, M, N, torch_comp)
                    elif engine == "MLX":
                        dtype = get_mlx_dtype(prec)
                        if dtype is not None and mlx_comp is not None:
                            val = run_mlx_bench(dtype, M, N, mlx_comp)
                            
                    results[engine][prec][f"{M}_{N}"] = val
                
            gc.collect()

    print("Benchmarking complete. Saving data to benchmark_data.json...")
    with open("benchmark_data.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Plotting results...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Matrix Multiplication Cache Efficiency (Latency / $M \\times N \\times K$)\nInner Dim (K) = 1024, Outer Dims (M, N) = 1024 to 1087\nNormalized Per-Framework (Row)", fontsize=16)
    
    # We will have one colorbar per row (engine)
    cbar_ax_mps = fig.add_axes([0.92, 0.55, 0.02, 0.35])
    cbar_ax_mlx = fig.add_axes([0.92, 0.1, 0.02, 0.35])
    
    for i, engine in enumerate(ENGINES):
        # Calculate row-specific min and max for the normalized values
        row_vals = []
        for prec in PRECISIONS:
            data_dict = results.get(engine, {}).get(prec, {})
            for key, latency in data_dict.items():
                if latency is not None:
                    m_val, n_val = map(int, key.split('_'))
                    flops = m_val * n_val * K
                    normalized_latency = (latency / flops) * 1e9 # convert ms/FLOP to picoseconds/FLOP
                    row_vals.append(normalized_latency)
            
        if not row_vals:
            print(f"No valid data for {engine}")
            continue
            
        row_min = min(row_vals)
        row_max = max(row_vals)
        print(f"{engine} Normalization: Min = {row_min:.3f} ps/FLOP, Max = {row_max:.3f} ps/FLOP")
        
        cbar_ax = cbar_ax_mps if engine == "MPS" else cbar_ax_mlx
        
        for j, prec in enumerate(PRECISIONS):
            ax = axes[i, j]
            
            data_dict = results.get(engine, {}).get(prec, {})
            valid_data = {k: v for k, v in data_dict.items() if v is not None}
            
            if not valid_data:
                ax.text(0.5, 0.5, "Unsupported\nor Failed", ha='center', va='center', transform=ax.transAxes, color="red")
                ax.set_title(f"{engine} | {prec}")
                ax.set_xticks([])
                ax.set_yticks([])
                continue
                
            heat_data = np.full((len(M_RANGE), len(N_RANGE)), np.nan)
            valid_scaled_vals = []
            for m_idx, m_val in enumerate(M_RANGE):
                for n_idx, n_val in enumerate(N_RANGE):
                    key = f"{m_val}_{n_val}"
                    if key in valid_data:
                        latency = valid_data[key]
                        flops = m_val * n_val * K
                        val_per_flop = (latency / flops) * 1e9
                        valid_scaled_vals.append(val_per_flop)
                        
                        # Normalize to [-1, 1] relative to THIS ENGINE'S min/max
                        if row_max > row_min:
                            norm_val = 2 * ((val_per_flop - row_min) / (row_max - row_min)) - 1
                        else:
                            norm_val = 0
                        heat_data[m_idx, n_idx] = norm_val
                        
            sns.heatmap(heat_data, ax=ax, cmap="coolwarm", vmin=-1.0, vmax=1.0,
                        cbar=(j == 2), cbar_ax=cbar_ax if j == 2 else None,
                        yticklabels=[str(m) if m % 8 == 0 else "" for m in M_RANGE], 
                        xticklabels=[str(n) if n % 8 == 0 else "" for n in N_RANGE])
            
            # Add subtitle with absolute speeds for context
            ax.set_title(f"{engine} | {prec}\nMin: {min(valid_scaled_vals):.3f} ps/FLOP  Max: {max(valid_scaled_vals):.3f} ps/FLOP")
            ax.set_xlabel("N Dimension (Cols of B)")
            if j == 0:
                ax.set_ylabel("M Dimension (Rows of A)")
                
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig("heat.png", dpi=300)
    print("Saved heat.png successfully.")

if __name__ == "__main__":
    main()
