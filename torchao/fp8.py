import os
import torch
from torchao.quantization import quantize_, Float8WeightOnlyConfig
from model import RotationModel, generate_dataset, evaluate_predictions

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fp32_path = os.path.join(script_dir, "model_fp32.pt")
    fp8_path = os.path.join(script_dir, "model_fp8.pt")

    if not os.path.exists(fp32_path):
        raise FileNotFoundError(f"Source FP32 model checkpoint not found at '{fp32_path}'. Please run fp32.py first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[FP8] Loading source FP32 model from '{fp32_path}' on device '{device}'...")

    # Load FP32 reference model
    fp32_state = torch.load(fp32_path, weights_only=True)
    model_fp32 = RotationModel(dim=256, hidden_dim=1024).to(device=device, dtype=torch.float32)
    model_fp32.load_state_dict(fp32_state)
    model_fp32.eval()

    # Create FP8 quantized model using torchao Float8WeightOnlyConfig
    model_fp8 = RotationModel(dim=256, hidden_dim=1024).to(device=device, dtype=torch.float32)
    model_fp8.load_state_dict(fp32_state)
    quantize_(model_fp8, Float8WeightOnlyConfig())
    model_fp8.eval()

    # Generate benchmark test dataset
    x_test, y_test = generate_dataset(num_samples=512, dim=256, seed=999)
    x_test, y_test = x_test.to(device), y_test.to(device)

    # Evaluate outputs
    with torch.no_grad():
        y_pred_fp32 = model_fp32(x_test)
        y_pred_fp8 = model_fp8(x_test)

    metrics = evaluate_predictions(y_pred_fp32, y_pred_fp8, y_test)

    # Save FP8 model checkpoint
    torch.save(model_fp8.state_dict(), fp8_path)

    file_size_bytes = os.path.getsize(fp8_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    fp32_size_bytes = os.path.getsize(fp32_path)
    reduction = (1.0 - file_size_bytes / fp32_size_bytes) * 100.0

    print(f"[FP8] Model quantized and saved successfully to: {fp8_path}")
    print(f"[FP8] File Size: {file_size_bytes:,} bytes ({file_size_mb:.2f} MB) - Reduction: {reduction:.2f}%")
    print(f"[FP8] Layer fc1 Weight Class: {type(model_fp8.fc1.weight)}")
    print(f"[FP8] Test MSE: {metrics['test_mse']:.6f}")
    print(f"[FP8] Mean Cosine Similarity: {metrics['mean_cos_sim']:.6f}")
    print(f"[FP8] Worst Case  -> Cos Sim: {metrics['worst_cos_sim']:.6f} | Ref Mag: {metrics['worst_ref_mag']:.4f} | FP8 Mag: {metrics['worst_var_mag']:.4f}")
    print(f"[FP8] Median Case -> Cos Sim: {metrics['median_cos_sim']:.6f} | Ref Mag: {metrics['median_ref_mag']:.4f} | FP8 Mag: {metrics['median_var_mag']:.4f}")

if __name__ == "__main__":
    main()