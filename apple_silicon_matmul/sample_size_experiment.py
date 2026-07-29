import torch
import mlx.core as mx
import time
import numpy as np
import gc

def test_mps(M, K, N, iterations):
    a = torch.randn(M, K, dtype=torch.float32, device="mps")
    b = torch.randn(K, N, dtype=torch.float32, device="mps")
    
    # Warmup
    for _ in range(5):
        c = torch.matmul(a, b)
    torch.mps.synchronize()
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        c = torch.matmul(a, b)
        torch.mps.synchronize()
        times.append(time.perf_counter() - start)
        
    return np.mean(times), np.std(times)

def test_mlx(M, K, N, iterations):
    a = mx.random.normal((M, K), dtype=mx.float32)
    b = mx.random.normal((K, N), dtype=mx.float32)
    
    # Warmup
    for _ in range(5):
        c = mx.matmul(a, b)
        mx.eval(c)
        
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        c = mx.matmul(a, b)
        mx.eval(c)
        times.append(time.perf_counter() - start)
        
    return np.mean(times), np.std(times)

if __name__ == "__main__":
    M, N = 2048, 2048
    K = 2048
    sample_sizes = [5, 10, 20, 50, 100, 200]
    
    print("Running sample size experiment on M4 Pro...")
    print(f"Matrix size: {M}x{K} @ {K}x{N}")
    
    for engine, func in [("MPS", test_mps), ("MLX", test_mlx)]:
        print(f"\nEngine: {engine}")
        for size in sample_sizes:
            gc.collect()
            mean, std = func(M, K, N, size)
            cv = (std / mean) * 100 if mean > 0 else 0
            print(f"Sample Size: {size:3d} | Mean: {mean:.5f}s | Std: {std:.5f}s | CV: {cv:.2f}%")
