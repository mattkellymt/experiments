import json
import time
import torch
import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt

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
    print("Loading data to find hot/cold spots...")
    with open("advanced_benchmark_data.json", "r") as f:
        data = json.load(f)
    
    # Analyze M_vs_N for MPS and MLX
    results = {}
    
    for engine in ["MPS", "MLX"]:
        m_vs_n = data["2d"][engine]["M_vs_N"]
        # Convert to list of (key, latency)
        items = list(m_vs_n.items())
        # Sort by latency
        items.sort(key=lambda x: x[1])
        
        coldest_20 = items[:20]
        hottest_20 = items[-20:]
        
        print(f"\n{engine} Coldest 20 (Fastest):")
        for k, v in coldest_20[:5]: print(f"  {k}: {v:.4f} ms")
        print(f"{engine} Hottest 20 (Slowest):")
        for k, v in hottest_20[-5:]: print(f"  {k}: {v:.4f} ms")
        
        results[engine] = {"cold": {}, "hot": {}}
        
        print(f"Rerunning {engine} coldest 20 spots 1000 times each...")
        for key, _ in coldest_20:
            m, n = map(int, key.split('_'))
            latencies = [bench(engine, m, 2048, n) for _ in range(1000)]
            results[engine]["cold"][key] = latencies
            
        print(f"Rerunning {engine} hottest 20 spots 1000 times each...")
        for key, _ in hottest_20:
            m, n = map(int, key.split('_'))
            latencies = [bench(engine, m, 2048, n) for _ in range(1000)]
            results[engine]["hot"][key] = latencies

    with open("determinism_data.json", "w") as f:
        json.dump(results, f)
        
    print("Determinism test complete. Saved to determinism_data.json.")

if __name__ == "__main__":
    main()
