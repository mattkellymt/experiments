import os
os.environ["MLX_ALLOW_CACHE"] = "0"

import time
import json
import gc
import torch
import mlx.core as mx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuration for 2D Sweep
OFFSET = 2048
SWEEP_RANGE = 33 # 0 to 32 inclusive (SIMD group size is 32)
K = 2048
M_RANGE = list(range(OFFSET, OFFSET + SWEEP_RANGE))
N_RANGE = list(range(OFFSET, OFFSET + SWEEP_RANGE))

# Bumped iterations to compensate for tiny matrix sizes (otherwise python overhead dominates)
WARMUP = 5
ITERS = 20

ENGINES = ["MPS", "MLX"]
PRECISIONS = ["FP16", "BF16"]

import multiprocessing

def get_compiled_torch(device="mps", dtype=torch.float16):
    def mm(x, y):
        return torch.matmul(x, y)
    return mm

def get_compiled_mlx():
    def mm(x, y):
        return mx.matmul(x, y)
    return mx.compile(mm)

def main():
    print("Starting clean-room 2D matrix multiplication benchmarking...")
    
    results = {
        "MPS": {"FP16": {}, "BF16": {}},
        "MLX": {"FP16": {}, "BF16": {}}
    }
    
    for engine in ENGINES:
        for prec in PRECISIONS:
            print(f"Benchmarking {engine} | {prec}...")
            
            if prec == "FP16":
                dtype = torch.float16
                mlx_dtype = mx.float16
            elif prec == "BF16":
                dtype = torch.bfloat16
                mlx_dtype = mx.bfloat16
                
            for M in M_RANGE:
                for N in N_RANGE:
                    key = f"{M}_{N}"
                    try:
                        if engine == "MPS":
                            a = torch.randn(M, K, dtype=dtype, device="mps")
                            b = torch.randn(K, N, dtype=dtype, device="mps")
                            mm_comp = get_compiled_torch("mps", dtype)
                            
                            for _ in range(WARMUP):
                                res = mm_comp(a, b)
                            torch.mps.synchronize()
                            
                            start = time.perf_counter()
                            for _ in range(ITERS):
                                res = mm_comp(a, b)
                            torch.mps.synchronize()
                            end = time.perf_counter()
                            
                            val = ((end - start) / ITERS) * 1000
                            results[engine][prec][key] = val
                            
                        elif engine == "MLX":
                            a = mx.random.normal((M, K), dtype=mlx_dtype)
                            b = mx.random.normal((K, N), dtype=mlx_dtype)
                            
                            mm_comp = get_compiled_mlx()
                            
                            for _ in range(WARMUP):
                                res = mm_comp(a, b)
                                mx.eval(res)
                                
                            start = time.perf_counter()
                            for _ in range(ITERS):
                                res = mm_comp(a, b)
                            mx.eval(res)
                            end = time.perf_counter()
                            
                            val = ((end - start) / ITERS) * 1000
                            results[engine][prec][key] = val
                    except Exception as e:
                        results[engine][prec][key] = None
                
            gc.collect()

    print("Benchmarking complete. Saving data to benchmark_data.json...")
    with open("benchmark_data.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Plotting results...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Matrix Multiplication Cache Efficiency (Latency / $M \\times N \\times K$)\nInner Dim (K) = {K}, Outer Dims (M, N) = {OFFSET} to {OFFSET + SWEEP_RANGE - 1}\nNormalized Per-Framework (Row)", fontsize=16)
    
    cbar_ax_mps = fig.add_axes([0.92, 0.55, 0.02, 0.35])
    cbar_ax_mlx = fig.add_axes([0.92, 0.1, 0.02, 0.35])
    
    for i, engine in enumerate(ENGINES):
        row_vals = []
        for prec in PRECISIONS:
            data_dict = results.get(engine, {}).get(prec, {})
            for key, latency in data_dict.items():
                if latency is not None:
                    m_val, n_val = map(int, key.split('_'))
                    flops = m_val * n_val * K
                    normalized_latency = (latency / flops) * 1e9
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
                        
                        if row_max > row_min:
                            norm_val = 2 * ((val_per_flop - row_min) / (row_max - row_min)) - 1
                        else:
                            norm_val = 0
                        heat_data[m_idx, n_idx] = norm_val
                        
            sns.heatmap(heat_data, ax=ax, cmap="coolwarm", vmin=-1.0, vmax=1.0,
                        cbar=(j == 1), cbar_ax=cbar_ax if j == 1 else None,
                        yticklabels=[str(m) if m % 4 == 0 else "" for m in M_RANGE], 
                        xticklabels=[str(n) if n % 4 == 0 else "" for n in N_RANGE])
            
            ax.set_title(f"{engine} | {prec}\nMin: {min(valid_scaled_vals):.3f} ps/FLOP  Max: {max(valid_scaled_vals):.3f} ps/FLOP")
            ax.set_xlabel("N Dimension (Cols of B)")
            if j == 0:
                ax.set_ylabel("M Dimension (Rows of A)")
                
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig("heat.png", dpi=300)
    print("Saved heat.png successfully.")

if __name__ == "__main__":
    main()
