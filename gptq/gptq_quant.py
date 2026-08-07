import os
import torch
import torch.nn as nn
from model import RotationModel, generate_dataset, evaluate_predictions

def naive_quantize_weight(weight: torch.Tensor, format_type: str = "fp8", block_size: int = 32) -> torch.Tensor:
    """Naive Round-To-Nearest (RTN) Weight Quantization without Hessian nudging."""
    w = weight.detach().clone().float()
    fp4_grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=weight.device)
    orig_shape = w.shape
    in_features = orig_shape[1]

    if format_type == "fp8":
        row_max = torch.max(torch.abs(w), dim=1, keepdim=True).values
        scale = torch.clamp(row_max / 448.0, min=1e-12)
        w_dequant = (w / scale).to(torch.float8_e4m3fn).to(torch.float32) * scale
        return w_dequant
    elif format_type == "fp4":
        w_flat = w.reshape(-1, block_size)
        block_max = torch.max(torch.abs(w_flat), dim=1, keepdim=True).values
        scale = torch.clamp(block_max / 6.0, min=1e-12)
        w_scaled = w_flat / scale
        w_sign = torch.sign(w_scaled)
        w_abs = torch.abs(w_scaled)
        diffs = torch.abs(w_abs.unsqueeze(-1) - fp4_grid)
        indices = torch.argmin(diffs, dim=-1)
        w_q = fp4_grid[indices] * w_sign * scale
        return w_q.reshape(orig_shape)
    else:
        raise ValueError(f"Unknown format: {format_type}")


def gptq_quantize_layer(weight: torch.Tensor, X_activations: torch.Tensor, format_type: str = "fp8", block_size: int = 32) -> torch.Tensor:
    """
    GPTQ Quantization with Inverse Hessian Nudging:
    1. Computes Hessian H = 1/M * X^T @ X from calibration activations X.
    2. Inverts H to get H_inv.
    3. Quantizes columns sequentially while nudging remaining columns via H_inv.
    """
    w_gptq = weight.detach().clone().float()
    X = X_activations.detach().float()
    M, in_features = X.shape

    H = (1.0 / M) * (X.T @ X)
    diag_mean = torch.mean(torch.diag(H))
    H += 1e-4 * diag_mean * torch.eye(in_features, device=weight.device)
    H_inv = torch.linalg.inv(H)

    fp4_grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=weight.device)

    for block_start in range(0, in_features, block_size):
        block_end = min(block_start + block_size, in_features)
        H_sub_inv = H_inv[block_start:block_end, block_start:block_end]

        for i in range(block_start, block_end):
            idx_local = i - block_start
            w_col = w_gptq[:, i]

            if format_type == "fp8":
                col_max = torch.max(torch.abs(w_col))
                scale = col_max / 448.0 if col_max > 0 else 1.0
                w_q = (w_col / scale).to(torch.float8_e4m3fn).to(torch.float32) * scale
            elif format_type == "fp4":
                col_max = torch.max(torch.abs(w_col))
                scale = col_max / 6.0 if col_max > 0 else 1.0
                w_scaled = w_col / scale
                w_sign = torch.sign(w_scaled)
                w_abs = torch.abs(w_scaled)
                diffs = torch.abs(w_abs.unsqueeze(-1) - fp4_grid)
                indices = torch.argmin(diffs, dim=-1)
                w_q = fp4_grid[indices] * w_sign * scale
            else:
                raise ValueError(f"Unknown format: {format_type}")

            delta = w_col - w_q
            w_gptq[:, i] = w_q

            if i + 1 < block_end:
                h_row = H_sub_inv[idx_local, idx_local + 1:] / H_sub_inv[idx_local, idx_local]
                w_gptq[:, i + 1:block_end] -= delta.unsqueeze(1) @ h_row.unsqueeze(0)

    return w_gptq


def apply_quantization_to_model(model: nn.Module, X_calib: torch.Tensor, format_type: str, use_gptq: bool = True) -> nn.Module:
    model_quant = RotationModel(dim=256, hidden_dim=1024)
    model_quant.load_state_dict(model.state_dict())

    if not use_gptq:
        for name, module in model_quant.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.data = naive_quantize_weight(module.weight.data, format_type=format_type)
        model_quant.eval()
        return model_quant

    # GPTQ Path
    activations = {}
    def make_hook(name):
        def hook(module, input, output):
            activations[name] = input[0].detach()
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(X_calib)

    for h in hooks:
        h.remove()

    for name, module in model_quant.named_modules():
        if isinstance(module, nn.Linear):
            X_act = activations[name]
            module.weight.data = gptq_quantize_layer(module.weight.data, X_act, format_type=format_type)

    model_quant.eval()
    return model_quant


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x_train, y_train = generate_dataset(num_samples=2048, dim=256, seed=42)
    x_calib, _ = generate_dataset(num_samples=256, dim=256, seed=777)
    x_test, y_test = generate_dataset(num_samples=512, dim=256, seed=999)

    x_train, y_train = x_train.to(device), y_train.to(device)
    x_calib = x_calib.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    # 1. FP32 Reference
    model_fp32 = RotationModel(dim=256, hidden_dim=1024).to(device)
    optimizer = torch.optim.AdamW(model_fp32.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    print("Training FP32 reference model (150 epochs)...")
    model_fp32.train()
    for epoch in range(150):
        optimizer.zero_grad()
        loss = criterion(model_fp32(x_train), y_train)
        loss.backward()
        optimizer.step()

    model_fp32.eval()
    with torch.no_grad():
        out_fp32 = model_fp32(x_test)

    fp32_path = os.path.join(script_dir, "model_fp32.pt")
    torch.save(model_fp32.state_dict(), fp32_path)
    fp32_size = os.path.getsize(fp32_path)

    # 2. BF16
    m_bf16 = RotationModel(dim=256, hidden_dim=1024).to(device, dtype=torch.bfloat16)
    m_bf16.load_state_dict({k: v.bfloat16() for k, v in model_fp32.state_dict().items()})
    bf16_path = os.path.join(script_dir, "model_bf16.pt")
    torch.save(m_bf16.state_dict(), bf16_path)

    # 3. Naive vs GPTQ Models
    print("Applying Naive and GPTQ quantization...")
    m_naive_fp8 = apply_quantization_to_model(model_fp32, x_calib, format_type="fp8", use_gptq=False)
    m_gptq_fp8 = apply_quantization_to_model(model_fp32, x_calib, format_type="fp8", use_gptq=True)
    m_naive_fp4 = apply_quantization_to_model(model_fp32, x_calib, format_type="fp4", use_gptq=False)
    m_gptq_fp4 = apply_quantization_to_model(model_fp32, x_calib, format_type="fp4", use_gptq=True)

    torch.save(m_naive_fp8.state_dict(), os.path.join(script_dir, "model_fp8_naive.pt"))
    torch.save(m_gptq_fp8.state_dict(), os.path.join(script_dir, "model_fp8_gptq.pt"))
    torch.save(m_naive_fp4.state_dict(), os.path.join(script_dir, "model_fp4_naive.pt"))
    torch.save(m_gptq_fp4.state_dict(), os.path.join(script_dir, "model_fp4_gptq.pt"))

    with torch.no_grad():
        out_bf16 = m_bf16(x_test.bfloat16()).float()
        out_naive_fp8 = m_naive_fp8(x_test).float()
        out_gptq_fp8 = m_gptq_fp8(x_test).float()
        out_naive_fp4 = m_naive_fp4(x_test).float()
        out_gptq_fp4 = m_gptq_fp4(x_test).float()

    variants = [
        ("FP32 (Ref)", fp32_path, out_fp32),
        ("BF16", bf16_path, out_bf16),
        ("Naive FP8 (RTN)", os.path.join(script_dir, "model_fp8_naive.pt"), out_naive_fp8),
        ("GPTQ FP8 (Hessian)", os.path.join(script_dir, "model_fp8_gptq.pt"), out_gptq_fp8),
        ("Naive FP4 (RTN)", os.path.join(script_dir, "model_fp4_naive.pt"), out_naive_fp4),
        ("GPTQ FP4 (Hessian)", os.path.join(script_dir, "model_fp4_gptq.pt"), out_gptq_fp4),
    ]

    print("\n" + "=" * 125)
    print("QUANTIZATION METHOD COMPARISON: NAIVE ROUNDING VS. GPTQ HESSIAN NUDGING")
    print("=" * 125)
    print(f"{'Precision Variant':<22} | {'Eff. Size':<10} | {'Worst Cos Sim':<15} | {'Ref Mag':<10} | {'Var Mag':<10} | {'Mean Cos':<10}")
    print("-" * 125)
    sizes = {"FP32 (Ref)": "10.0 MB", "BF16": "5.0 MB", "Naive FP8 (RTN)": "2.5 MB", "GPTQ FP8 (Hessian)": "2.5 MB", "Naive FP4 (RTN)": "1.4 MB", "GPTQ FP4 (Hessian)": "1.4 MB"}
    for label, path, out_var in variants:
        m = evaluate_predictions(out_fp32, out_var, y_test)
        print(f"{label:<22} | {sizes[label]:<10} | {m['worst_cos_sim']:15.6f} | {m['worst_ref_mag']:10.4f} | {m['worst_var_mag']:10.4f} | {m['mean_cos_sim']:10.6f}")
    print("=" * 125 + "\n")

if __name__ == "__main__":
    main()
