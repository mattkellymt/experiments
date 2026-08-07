import os
import torch
from model import RotationModel, generate_dataset, evaluate_predictions

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fp32_path = os.path.join(script_dir, "model_fp32.pt")
    bf16_path = os.path.join(script_dir, "model_bf16.pt")

    if not os.path.exists(fp32_path):
        raise FileNotFoundError(f"Source FP32 model checkpoint not found at '{fp32_path}'. Please run fp32.py first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[BF16] Loading source FP32 model from '{fp32_path}' on device '{device}'...")

    # Load FP32 reference model
    fp32_state = torch.load(fp32_path, weights_only=True)
    model_fp32 = RotationModel(dim=256, hidden_dim=1024).to(device=device, dtype=torch.float32)
    model_fp32.load_state_dict(fp32_state)
    model_fp32.eval()

    # Create BF16 model variant by casting state dict tensors to bfloat16
    bf16_state = {k: v.bfloat16() for k, v in fp32_state.items()}
    model_bf16 = RotationModel(dim=256, hidden_dim=1024).to(device=device, dtype=torch.bfloat16)
    model_bf16.load_state_dict(bf16_state)
    model_bf16.eval()

    # Generate benchmark test dataset
    x_test, y_test = generate_dataset(num_samples=512, dim=256, seed=999)
    x_test, y_test = x_test.to(device), y_test.to(device)

    # Evaluate outputs
    with torch.no_grad():
        y_pred_fp32 = model_fp32(x_test)
        y_pred_bf16 = model_bf16(x_test.bfloat16())

    metrics = evaluate_predictions(y_pred_fp32, y_pred_bf16, y_test)

    # Save BF16 model checkpoint
    torch.save(model_bf16.state_dict(), bf16_path)

    file_size_bytes = os.path.getsize(bf16_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    fp32_size_bytes = os.path.getsize(fp32_path)
    reduction = (1.0 - file_size_bytes / fp32_size_bytes) * 100.0

    print(f"[BF16] Model converted and saved successfully to: {bf16_path}")
    print(f"[BF16] File Size: {file_size_bytes:,} bytes ({file_size_mb:.2f} MB) - Reduction: {reduction:.2f}%")
    print(f"[BF16] Test MSE: {metrics['test_mse']:.6f}")
    print(f"[BF16] Mean Cosine Similarity: {metrics['mean_cos_sim']:.6f}")
    print(f"[BF16] Worst Case  -> Cos Sim: {metrics['worst_cos_sim']:.6f} | Ref Mag: {metrics['worst_ref_mag']:.4f} | BF16 Mag: {metrics['worst_var_mag']:.4f}")
    print(f"[BF16] Median Case -> Cos Sim: {metrics['median_cos_sim']:.6f} | Ref Mag: {metrics['median_ref_mag']:.4f} | BF16 Mag: {metrics['median_var_mag']:.4f}")

if __name__ == "__main__":
    main()
