import torch
import coremltools as ct
import time
import numpy as np

class MatMulModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.matmul(x, y)

def test_coreml():
    M, N, K = 128, 128, 128
    model = MatMulModel().eval()
    
    # Trace the model
    example_input_x = torch.randn(M, K)
    example_input_y = torch.randn(K, N)
    traced_model = torch.jit.trace(model, (example_input_x, example_input_y))
    
    print("Compiling Core ML model...")
    t0 = time.time()
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="x", shape=(M, K)),
                ct.TensorType(name="y", shape=(K, N))],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16
    )
    t1 = time.time()
    print(f"Compilation took {t1 - t0:.2f}s")
    
    # Run prediction
    inputs = {
        "x": np.random.randn(M, K).astype(np.float32),
        "y": np.random.randn(K, N).astype(np.float32)
    }
    
    print("Running prediction...")
    t0 = time.time()
    res = coreml_model.predict(inputs)
    t1 = time.time()
    print(f"Prediction took {t1 - t0:.4f}s")
    
    # Can we dynamically change shapes easily without recompiling?
    # Core ML allows flexible shapes, but the API overhead for predict() is generally dictionary-based and very high.

if __name__ == "__main__":
    test_coreml()
