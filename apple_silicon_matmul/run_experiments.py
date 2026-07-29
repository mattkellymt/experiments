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

# Configuration
M = 2048
N = 2048
K_RANGE = list(range(2000, 2101))
WARMUP = 5
ITERS = 32

ENGINES = ["CPU", "MPS", "MLX"]
PRECISIONS = ["FP32", "FP16", "BF16"]

def get_torch_dtype(prec):
    if prec == "FP32":
        return torch.float32
    elif prec == "FP16":
        return torch.float16
    elif prec == "BF16":
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

def run_torch_bench(device, dtype, K, mm_comp):
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

def run_mlx_bench(dtype, K, mm_comp):
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
    # PyTorch's inductor compiler causes a deadlock on macOS (0% CPU, hanging).
    # torch.matmul already dispatches to highly-optimized, pre-compiled Accelerate/Metal kernels.
    def mm(x, y):
        return torch.matmul(x, y)
    return mm

def get_compiled_mlx():
    def mm(x, y):
        return mx.matmul(x, y)
    return mx.compile(mm)

def main():
    print("Starting clean-room matrix multiplication benchmarking...")
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
            
            # Create compiled functions once per engine/precision to avoid recompilation overhead
            torch_comp = None
            mlx_comp = None
            if engine == "CPU":
                dtype = get_torch_dtype(prec)
                if dtype is not None:
                    torch_comp = get_compiled_torch("cpu", dtype)
            elif engine == "MPS":
                dtype = get_torch_dtype(prec)
                if dtype is not None and torch.backends.mps.is_available():
                    torch_comp = get_compiled_torch("mps", dtype)
            elif engine == "MLX":
                dtype = get_mlx_dtype(prec)
                if dtype is not None:
                    mlx_comp = get_compiled_mlx()

            for K in K_RANGE:
                val = None
                if engine == "CPU":
                    dtype = get_torch_dtype(prec)
                    if dtype is not None and torch_comp is not None:
                        val = run_torch_bench("cpu", dtype, K, torch_comp)
                elif engine == "MPS":
                    if not torch.backends.mps.is_available():
                        val = None
                    else:
                        dtype = get_torch_dtype(prec)
                        if dtype is not None and torch_comp is not None:
                            val = run_torch_bench("mps", dtype, K, torch_comp)
                elif engine == "MLX":
                    dtype = get_mlx_dtype(prec)
                    if dtype is not None and mlx_comp is not None:
                        val = run_mlx_bench(dtype, K, mlx_comp)
                        
                results[engine][prec][str(K)] = val
                
            gc.collect()

    print("Benchmarking complete. Saving data to benchmark_data.json...")
    with open("benchmark_data.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Plotting results...")
    
    # Calculate global min and max for normalization
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
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f"Matrix Multiplication Latency (ms): M=2048, N=2048, K=(2000-2100)\nGlobal Min: {global_min:.3f} ms, Global Max: {global_max:.3f} ms", fontsize=16)
    
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    
    for i, engine in enumerate(ENGINES):
        for j, prec in enumerate(PRECISIONS):
            ax = axes[i, j]
            
            data_dict = results[engine][prec]
            valid_data = {int(k): v for k, v in data_dict.items() if v is not None}
            
            if not valid_data:
                ax.text(0.5, 0.5, "Unsupported\nor Failed", ha='center', va='center', transform=ax.transAxes, color="red")
                ax.set_title(f"{engine} | {prec}")
                ax.set_xticks([])
                ax.set_yticks([])
                continue
                
            # Create a 2D array for heatmap: shape (1, len(K_RANGE))
            heat_data = np.full((1, len(K_RANGE)), np.nan)
            for idx, k in enumerate(K_RANGE):
                if k in valid_data:
                    # Normalize to [-1, 1]
                    norm_val = 2 * ((valid_data[k] - global_min) / (global_max - global_min)) - 1
                    heat_data[0, idx] = norm_val
                    
            sns.heatmap(heat_data, ax=ax, cmap="coolwarm", vmin=-1.0, vmax=1.0,
                        cbar=(i == 0 and j == 0), cbar_ax=cbar_ax if (i==0 and j==0) else None,
                        yticklabels=False, 
                        xticklabels=[str(k) if k % 20 == 0 else "" for k in K_RANGE])
            
            ax.set_title(f"{engine} | {prec}")
            ax.set_xlabel("K Dimension")
            if j == 0:
                ax.set_ylabel("Latency (ms)")
                
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig("heat.png", dpi=300)
    print("Saved heat.png successfully.")

if __name__ == "__main__":
    main()
