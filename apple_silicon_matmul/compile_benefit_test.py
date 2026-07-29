import time
import torch
import mlx.core as mx

M, K, N = 2048, 2048, 2048
WARMUP = 10
ITERS = 50

def profile_torch(device, compiled):
    try:
        a = torch.randn(M, K, dtype=torch.float32, device=device)
        b = torch.randn(K, N, dtype=torch.float32, device=device)
        
        def mm(x, y):
            return torch.matmul(x, y)
            
        if compiled:
            mm_func = torch.compile(mm)
        else:
            mm_func = mm
            
        # Warmup
        for _ in range(WARMUP):
            res = mm_func(a, b)
            
        if device == "mps":
            torch.mps.synchronize()
            
        start = time.perf_counter()
        for _ in range(ITERS):
            res = mm_func(a, b)
            
        if device == "mps":
            torch.mps.synchronize()
            
        end = time.perf_counter()
        latency = ((end - start) / ITERS) * 1000.0
        return f"{latency:.3f} ms"
    except Exception as e:
        return f"FAILED ({str(e)})"

def profile_mlx(compiled):
    try:
        a = mx.random.normal((M, K), dtype=mx.float32)
        b = mx.random.normal((K, N), dtype=mx.float32)
        
        def mm(x, y):
            return mx.matmul(x, y)
            
        if compiled:
            mm_func = mx.compile(mm)
        else:
            mm_func = mm
            
        # Warmup
        for _ in range(WARMUP):
            res = mm_func(a, b)
            mx.eval(res)
            
        start = time.perf_counter()
        for _ in range(ITERS):
            res = mm_func(a, b)
        mx.eval(res)
        end = time.perf_counter()
        
        latency = ((end - start) / ITERS) * 1000.0
        return f"{latency:.3f} ms"
    except Exception as e:
        return f"FAILED ({str(e)})"

if __name__ == "__main__":
    print("--- Compilation Benefit Test ---")
    print(f"Matrix Size: {M}x{K} x {K}x{N}, FP32")
    
    print("\n1. PyTorch CPU")
    print("   Eager mode:    ", profile_torch("cpu", compiled=False))
    print("   Compiled mode: ", profile_torch("cpu", compiled=True))
    
    if torch.backends.mps.is_available():
        print("\n2. PyTorch MPS (GPU)")
        print("   Eager mode:    ", profile_torch("mps", compiled=False))
        print("   Compiled mode: ", profile_torch("mps", compiled=True))
    
    print("\n3. Apple MLX (GPU)")
    print("   Eager mode:    ", profile_mlx(compiled=False))
    print("   Compiled mode: ", profile_mlx(compiled=True))
