import os
os.environ["MLX_ALLOW_CACHE"] = "0"

import time
import json
import gc
import torch
import mlx.core as mx
import numpy as np

# Configuration
BASE = 2048
SWEEP_RANGE = 256
WARMUP = 2
ITERS = 10
DTYPE_TORCH = torch.float16
DTYPE_MLX = mx.float16

ENGINES = ["MPS", "MLX"]
SWEEP_AXES = ["M_vs_N", "M_vs_K", "N_vs_K"]

def get_compiled_torch():
    return lambda x, y: torch.matmul(x, y)

def get_compiled_mlx():
    return mx.compile(lambda x, y: mx.matmul(x, y))

def run_bench(engine, m, k, n, warmup, iters):
    if engine == "MPS":
        a = torch.randn(m, k, dtype=DTYPE_TORCH, device="mps")
        b = torch.randn(k, n, dtype=DTYPE_TORCH, device="mps")
        mm = get_compiled_torch()
        for _ in range(warmup):
            res = mm(a, b)
        torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            res = mm(a, b)
        torch.mps.synchronize()
        end = time.perf_counter()
    else:
        a = mx.random.normal((m, k), dtype=DTYPE_MLX)
        b = mx.random.normal((k, n), dtype=DTYPE_MLX)
        mm = get_compiled_mlx()
        for _ in range(warmup):
            res = mm(a, b)
            mx.eval(res)
        start = time.perf_counter()
        for _ in range(iters):
            res = mm(a, b)
        mx.eval(res)
        end = time.perf_counter()
        
    return ((end - start) / iters) * 1000

def main():
    print("Starting Advanced Sweeps...")
    
    # 1. 1D Sweeps
    results_1d = {"MPS": {"M": {}, "N": {}, "K": {}}, "MLX": {"M": {}, "N": {}, "K": {}}}
    # We sweep up to 512 for 1D
    for engine in ENGINES:
        print(f"Running 1D sweeps for {engine}...")
        for val in range(BASE, BASE + 513):
            # Sweep M
            results_1d[engine]["M"][val] = run_bench(engine, val, BASE, BASE, WARMUP, ITERS)
            # Sweep N
            results_1d[engine]["N"][val] = run_bench(engine, BASE, BASE, val, WARMUP, ITERS)
            # Sweep K
            results_1d[engine]["K"][val] = run_bench(engine, BASE, val, BASE, WARMUP, ITERS)
            if val % 64 == 0:
                print(f"  {engine} 1D progress: {val}/{BASE + 512}")
                
    # 2. 2D Sweeps
    results_2d = {e: {ax: {} for ax in SWEEP_AXES} for e in ENGINES}
    for engine in ENGINES:
        for ax in SWEEP_AXES:
            print(f"Running 2D sweep {ax} for {engine}...")
            for i in range(BASE, BASE + SWEEP_RANGE, 4):
                for j in range(BASE, BASE + SWEEP_RANGE, 4):
                    if ax == "M_vs_N":
                        m, k, n = i, BASE, j
                    elif ax == "M_vs_K":
                        m, k, n = i, j, BASE
                    else: # N_vs_K
                        m, k, n = BASE, j, i
                        
                    key = f"{i}_{j}"
                    results_2d[engine][ax][key] = run_bench(engine, m, k, n, WARMUP, ITERS)
                if i % 32 == 0:
                    print(f"  {engine} {ax} progress: {i - BASE}/{SWEEP_RANGE}")
                    
    with open("advanced_benchmark_data.json", "w") as f:
        json.dump({"1d": results_1d, "2d": results_2d}, f)
        
    print("Benchmarking completed.")

if __name__ == "__main__":
    main()
