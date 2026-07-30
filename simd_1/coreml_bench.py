import torch
import coremltools as ct
import time
import numpy as np
import os

class MatMulModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.matmul(x, y)

def benchmark_coreml():
    M, K, N = 2048, 2048, 2048
    model = MatMulModel().eval()
    
    # Trace the model
    example_x = torch.randn(M, K)
    example_y = torch.randn(K, N)
    traced_model = torch.jit.trace(model, (example_x, example_y))
    
    print(f"Compiling Core ML model for shapes ({M},{K}) x ({K},{N})...")
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="x", shape=(M, K)),
                ct.TensorType(name="y", shape=(K, N))],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16
    )
    
    # Save model to disk temporarily
    model_path = "matmul.mlpackage"
    coreml_model.save(model_path)
    
    # Generate test inputs
    inputs = {
        "x": np.random.randn(M, K).astype(np.float32),
        "y": np.random.randn(K, N).astype(np.float32)
    }
    
    compute_units = [
        ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
        ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
        ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
        ("ALL", ct.ComputeUnit.ALL)
    ]
    
    print("\n--- Benchmarking Compute Units ---")
    for name, cu in compute_units:
        print(f"\nLoading model with Compute Unit: {name}")
        try:
            # Reload model with specific compute unit
            loaded_model = ct.models.MLModel(model_path, compute_units=cu)
            
            # Warmup
            for _ in range(2):
                _ = loaded_model.predict(inputs)
                
            # Benchmark
            start = time.perf_counter()
            iters = 5
            for _ in range(iters):
                _ = loaded_model.predict(inputs)
            end = time.perf_counter()
            
            latency = ((end - start) / iters) * 1000
            print(f"SUCCESS: {name} executed successfully. Latency: {latency:.2f} ms")
        except Exception as e:
            print(f"FAILED: {name} threw an error: {e}")

if __name__ == "__main__":
    benchmark_coreml()
