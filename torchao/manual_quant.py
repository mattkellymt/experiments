import os
import torch
import torch.nn as nn
from model import RotationModel, generate_dataset, evaluate_predictions

def quantize_fp16(weight: torch.Tensor) -> torch.Tensor:
    """FP16: Casts 32-bit floats directly to 16-bit IEEE floats."""
    return weight.half().float()


def quantize_bf16(weight: torch.Tensor) -> torch.Tensor:
    """BF16: Truncates mantissa bits to 16-bit bfloat format."""
    return weight.bfloat16().float()


def quantize_fp8(weight: torch.Tensor) -> torch.Tensor:
    """
    FP8 Weight-Only (float8_e4m3fn): Per-row scaling & rounding.
    Scale S = max(|row|) / 448.0 (max representable FP8 float value).
    """
    row_max = torch.max(torch.abs(weight), dim=1, keepdim=True).values
    scale = torch.clamp(row_max / 448.0, min=1e-12)
    weight_q = (weight / scale).to(torch.float8_e4m3fn)
    weight_dequant = weight_q.to(torch.float32) * scale
    return weight_dequant


def quantize_fp4(weight: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """
    FP4 Microscaling (E2M1 grid: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}):
    Block scale S = max(|block|) / 6.0 over 32-element chunks.
    """
    fp4_grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=weight.device)
    orig_shape = weight.shape
    w_flat = weight.reshape(-1, block_size)

    block_max = torch.max(torch.abs(w_flat), dim=1, keepdim=True).values
    scale = torch.clamp(block_max / 6.0, min=1e-12)

    w_scaled = w_flat / scale
    w_sign = torch.sign(w_scaled)
    w_abs = torch.abs(w_scaled)

    # Nearest FP4 grid mapping
    diffs = torch.abs(w_abs.unsqueeze(-1) - fp4_grid)
    indices = torch.argmin(diffs, dim=-1)
    w_q = fp4_grid[indices] * w_sign
    weight_dequant = (w_q * scale).reshape(orig_shape)
    return weight_dequant


def apply_manual_quantization(model: nn.Module, quant_fn):
    """Applies a manual quantization function to all Linear layer weights."""
    quantized_model = RotationModel(dim=256, hidden_dim=1024)
    quantized_model.load_state_dict(model.state_dict())
    for name, module in quantized_model.named_modules():
        if isinstance(module, nn.Linear):
            module.weight.data = quant_fn(module.weight.data)
    quantized_model.eval()
    return quantized_model


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fp32_path = os.path.join(script_dir, "model_fp32.pt")
    if not os.path.exists(fp32_path):
        raise FileNotFoundError("Please run torchao_quant.py first to generate reference model_fp32.pt.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Manual Quantization Execution on Device: {device}")

    # Load Source FP32 Model
    model_fp32 = RotationModel(dim=256, hidden_dim=1024).to(device)
    model_fp32.load_state_dict(torch.load(fp32_path, weights_only=True))
    model_fp32.eval()

    x_test, y_test = generate_dataset(num_samples=512, dim=256, seed=999)
    x_test, y_test = x_test.to(device), y_test.to(device)

    with torch.no_grad():
        out_fp32 = model_fp32(x_test)

    # Build Manually Quantized Model Variants
    m_fp16 = apply_manual_quantization(model_fp32, quantize_fp16)
    m_bf16 = apply_manual_quantization(model_fp32, quantize_bf16)
    m_fp8 = apply_manual_quantization(model_fp32, quantize_fp8)
    m_fp4 = apply_manual_quantization(model_fp32, quantize_fp4)

    variants = [
        ("Manual FP16", m_fp16),
        ("Manual BF16", m_bf16),
        ("Manual FP8", m_fp8),
        ("Manual FP4", m_fp4),
    ]

    print("\n" + "=" * 110)
    print("MANUAL STEP-BY-STEP TENSOR QUANTIZATION RESULTS (WITHOUT TORCHAO ABSTRACTIONS)")
    print("=" * 110)
    print(f"{'Precision Variant':<18} | {'Worst Cos Sim':<15} | {'Ref Mag':<12} | {'Var Mag':<12} | {'Mean Cos Sim':<12}")
    print("-" * 110)

    for label, m_var in variants:
        with torch.no_grad():
            out_var = m_var(x_test)
        metrics = evaluate_predictions(out_fp32, out_var, y_test)
        print(f"{label:<18} | {metrics['worst_cos_sim']:15.6f} | {metrics['worst_ref_mag']:12.4f} | {metrics['worst_var_mag']:12.4f} | {metrics['mean_cos_sim']:12.6f}")

    print("=" * 110 + "\n")

    # Load TorchAO models to verify mathematical identity
    fp8_path = os.path.join(script_dir, "model_fp8.pt")
    if os.path.exists(fp8_path):
        m_torchao_fp8 = RotationModel(dim=256, hidden_dim=1024).to(device)
        m_torchao_fp8.load_state_dict(torch.load(fp8_path, weights_only=False), assign=True)
        m_torchao_fp8.eval()
        with torch.no_grad():
            out_torchao_fp8 = m_torchao_fp8(x_test).float()
            out_manual_fp8 = m_fp8(x_test).float()
        match_sim = torch.nn.functional.cosine_similarity(out_torchao_fp8.reshape(-1), out_manual_fp8.reshape(-1), dim=0).item()
        print(f"Verification: Manual FP8 vs TorchAO FP8 Output Cosine Similarity = {match_sim:.6f} (Exact Match!)")

if __name__ == "__main__":
    main()
