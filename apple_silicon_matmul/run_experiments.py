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

    print("Benchmarking complete. Saving data to benchmark_data_2d.json...")
    with open("benchmark_data_2d.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Plotting results...")
    
    all_vals = []
    for engine in ENGINES:
        for prec in PRECISIONS:
            vals = results[engine][prec].values()
            all_vals.extend([v for v in vals if v is not None])
            
    if not all_vals:
        print("No valid data collected.")
        return
        
    global_min = min(all_vals)
    global_max = max(all_vals)
    
    print(f"Global Normalization: Min = {global_min:.3f} ms, Max = {global_max:.3f} ms")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Matrix Multiplication 2D Sweep Latency (ms)\nInner Dim (K) = 1024, Outer Dims (M, N) = 1024 to 1087\nGlobal Min: {global_min:.3f} ms, Global Max: {global_max:.3f} ms", fontsize=16)
    
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    
    for i, engine in enumerate(ENGINES):
        for j, prec in enumerate(PRECISIONS):
            ax = axes[i, j]
            
            data_dict = results[engine][prec]
            valid_data = {k: v for k, v in data_dict.items() if v is not None}
            
            if not valid_data:
                ax.text(0.5, 0.5, "Unsupported\nor Failed", ha='center', va='center', transform=ax.transAxes, color="red")
                ax.set_title(f"{engine} | {prec}")
                ax.set_xticks([])
                ax.set_yticks([])
                continue
                
            # Create a 2D array for heatmap: shape (len(M_RANGE), len(N_RANGE))
            heat_data = np.full((len(M_RANGE), len(N_RANGE)), np.nan)
            for m_idx, m_val in enumerate(M_RANGE):
                for n_idx, n_val in enumerate(N_RANGE):
                    key = f"{m_val}_{n_val}"
                    if key in valid_data:
                        norm_val = 2 * ((valid_data[key] - global_min) / (global_max - global_min)) - 1
                        heat_data[m_idx, n_idx] = norm_val
                        
            sns.heatmap(heat_data, ax=ax, cmap="coolwarm", vmin=-1.0, vmax=1.0,
                        cbar=(i == 0 and j == 0), cbar_ax=cbar_ax if (i==0 and j==0) else None,
                        yticklabels=[str(m) if m % 8 == 0 else "" for m in M_RANGE], 
                        xticklabels=[str(n) if n % 8 == 0 else "" for n in N_RANGE])
            
            ax.set_title(f"{engine} | {prec}")
            ax.set_xlabel("N Dimension (Cols of B)")
            ax.set_ylabel("M Dimension (Rows of A)")
                
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig("heat_2d.png", dpi=300)
    print("Saved heat_2d.png successfully.")

if __name__ == "__main__":
    main()
