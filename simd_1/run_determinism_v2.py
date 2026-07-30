import json
import time
import torch
import mlx.core as mx
import numpy as np
import random
import statistics

DTYPE_TORCH = torch.float16
DTYPE_MLX = mx.float16

def get_compiled_torch():
    return lambda x, y: torch.matmul(x, y)

def get_compiled_mlx():
    return mx.compile(lambda x, y: mx.matmul(x, y))

def bench(engine, m, k, n):
    if engine == "MPS":
        a = torch.randn(m, k, dtype=DTYPE_TORCH, device="mps")
        b = torch.randn(k, n, dtype=DTYPE_TORCH, device="mps")
        mm = get_compiled_torch()
        # Warmup
        for _ in range(5):
            res = mm(a, b)
        torch.mps.synchronize()
        start = time.perf_counter()
        res = mm(a, b)
        torch.mps.synchronize()
        end = time.perf_counter()
    else:
        a = mx.random.normal((m, k), dtype=DTYPE_MLX)
        b = mx.random.normal((k, n), dtype=DTYPE_MLX)
        mm = get_compiled_mlx()
        for _ in range(5):
            res = mm(a, b)
            mx.eval(res)
        start = time.perf_counter()
        res = mm(a, b)
        mx.eval(res)
        end = time.perf_counter()
    return (end - start) * 1000

def main():
    print("Loading data to find hot/cold/median spots...")
    with open("advanced_benchmark_data.json", "r") as f:
        data = json.load(f)
    
    results = {}
    
    for engine in ["MPS", "MLX"]:
        m_vs_n = data["2d"][engine]["M_vs_N"]
        items = list(m_vs_n.items())
        items.sort(key=lambda x: x[1])
        
        # Pick top 5, bottom 5, and median 5 to save time but keep it rigorous
        coldest_5 = [k for k, v in items[:5]]
        hottest_5 = [k for k, v in items[-5:]]
        mid_idx = len(items) // 2
        median_5 = [k for k, v in items[mid_idx-2 : mid_idx+3]]
        
        # Build the run list
        run_plan = []
        for key in coldest_5:
            run_plan.extend([("cold", key)] * 150)
        for key in hottest_5:
            run_plan.extend([("hot", key)] * 150)
        for key in median_5:
            run_plan.extend([("median", key)] * 150)
            
        # Shuffle to prevent temporal/thermal bias
        random.shuffle(run_plan)
        
        results[engine] = {"cold": {}, "hot": {}, "median": {}}
        for key in coldest_5 + hottest_5 + median_5:
            if key in coldest_5: results[engine]["cold"][key] = []
            elif key in hottest_5: results[engine]["hot"][key] = []
            else: results[engine]["median"][key] = []
            
        print(f"Running randomized {engine} interleaved benchmark (Total: {len(run_plan)} runs)...")
        for i, (cond, key) in enumerate(run_plan):
            m, n = map(int, key.split('_'))
            latency = bench(engine, m, 2048, n)
            results[engine][cond][key].append(latency)
            
            if (i+1) % 500 == 0:
                print(f"  {engine} progress: {i+1}/{len(run_plan)}")
                
        # Calculate CV
        print(f"\n{engine} Results (Median Latency | CV %):")
        for cond in ["cold", "median", "hot"]:
            print(f"  {cond.capitalize()}:")
            for key, lats in results[engine][cond].items():
                med = statistics.median(lats)
                cv = (statistics.stdev(lats) / statistics.mean(lats)) * 100
                print(f"    {key}: {med:.4f} ms | CV: {cv:.2f}%")

    with open("determinism_data_v2.json", "w") as f:
        json.dump(results, f)
        
    print("Determinism v2 test complete.")

if __name__ == "__main__":
    main()
