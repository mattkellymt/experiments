import os
import torch
import torch.nn as nn
from model import (
    RotationModel,
    QATLinear,
    STEQuantizeFP8,
    STEQuantizeFP4,
    generate_dataset,
    evaluate_predictions,
)

def naive_quantize_weight(weight: torch.Tensor, format_type: str = "fp8", block_size: int = 32) -> torch.Tensor:
    """Naive Round-To-Nearest (RTN) Weight Quantization."""
    w = weight.detach().clone().float()
    orig_shape = w.shape
    fp4_grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=weight.device)

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


def gptq_quantize_layer(weight: torch.Tensor, X_activations: torch.Tensor, format_type: str = "fp4", block_size: int = 32, damping: float = 0.01) -> torch.Tensor:
    """GPTQ Quantization with Damped Inverse Hessian Error Nudging."""
    w_gptq = weight.detach().clone().float()
    X = X_activations.detach().float()
    M, in_features = X.shape

    H = (1.0 / M) * (X.T @ X)
    diag_mean = torch.mean(torch.diag(H))
    H += damping * diag_mean * torch.eye(in_features, device=weight.device)
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

            delta = w_col - w_q
            w_gptq[:, i] = w_q

            if i + 1 < block_end:
                h_row = H_sub_inv[idx_local, idx_local + 1:] / H_sub_inv[idx_local, idx_local]
                w_gptq[:, i + 1:block_end] -= delta.unsqueeze(1) @ h_row.unsqueeze(0)

    return w_gptq


def awq_quantize_layer(weight: torch.Tensor, X_activations: torch.Tensor, format_type: str = "fp4", block_size: int = 32) -> torch.Tensor:
    """
    AWQ (Activation-aware Weight Quantization):
    Protects salient activation channels by optimizing per-channel scale factors S_X.
    Scales weights W_scaled = W * S_X before quantization, reducing salient channel error!
    """
    w = weight.detach().clone().float()
    X = X_activations.detach().float()
    out_features, in_features = w.shape

    act_means = torch.mean(torch.abs(X), dim=0)

    # Grid search optimal scale exponent alpha in [0.0, 1.0]
    best_alpha = 0.5
    best_error = float("inf")
    fp4_grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=weight.device)

    for alpha in torch.linspace(0.0, 1.0, 11):
        S = torch.pow(act_means, alpha)
        S = torch.clamp(S / torch.mean(S), min=1e-4)

        W_scaled = w * S.unsqueeze(0)
        w_flat = W_scaled.reshape(-1, block_size)
        b_max = torch.max(torch.abs(w_flat), dim=1, keepdim=True).values
        scale = torch.clamp(b_max / 6.0, min=1e-12)
        w_s = w_flat / scale
        w_sign = torch.sign(w_s)
        w_abs = torch.abs(w_s)
        diffs = torch.abs(w_abs.unsqueeze(-1) - fp4_grid)
        indices = torch.argmin(diffs, dim=-1)
        w_q_scaled = (fp4_grid[indices] * w_sign * scale).reshape(out_features, in_features)

        W_awq = w_q_scaled / S.unsqueeze(0)
        err = torch.nn.functional.mse_loss(X @ W_awq.T, X @ w.T).item()

        if err < best_error:
            best_error = err
            best_alpha = alpha.item()

    # Apply best scale factor
    S = torch.pow(act_means, best_alpha)
    S = torch.clamp(S / torch.mean(S), min=1e-4)

    W_scaled = w * S.unsqueeze(0)
    w_flat = W_scaled.reshape(-1, block_size)
    b_max = torch.max(torch.abs(w_flat), dim=1, keepdim=True).values
    scale = torch.clamp(b_max / 6.0, min=1e-12)
    w_s = w_flat / scale
    w_sign = torch.sign(w_s)
    w_abs = torch.abs(w_s)
    diffs = torch.abs(w_abs.unsqueeze(-1) - fp4_grid)
    indices = torch.argmin(diffs, dim=-1)
    w_q_scaled = (fp4_grid[indices] * w_sign * scale).reshape(out_features, in_features)

    return w_q_scaled / S.unsqueeze(0)


def build_qat_model(model_fp32: nn.Module, format_type: str) -> nn.Module:
    m_qat = RotationModel(dim=256, hidden_dim=1024)
    ste_fn = STEQuantizeFP8.apply if format_type == "fp8" else STEQuantizeFP4.apply

    for name, child in model_fp32.named_children():
        if isinstance(child, nn.Linear):
            qat_layer = QATLinear(child.in_features, child.out_features, ste_fn)
            qat_layer.weight.data = child.weight.data.clone()
            setattr(m_qat, name, qat_layer)

    return m_qat


def export_qat_checkpoint(m_qat: nn.Module, format_type: str) -> nn.Module:
    m_export = RotationModel(dim=256, hidden_dim=1024)
    ste_fn = STEQuantizeFP8.apply if format_type == "fp8" else STEQuantizeFP4.apply

    for name, child in m_qat.named_children():
        if isinstance(child, QATLinear):
            w_q = ste_fn(child.weight.data)
            getattr(m_export, name).weight.data = w_q

    m_export.eval()
    return m_export


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Master Quantization Suite Execution Device: {device}")

    # Generate Datasets
    x_train, y_train = generate_dataset(num_samples=2048, dim=256, seed=42)
    x_calib, _ = generate_dataset(num_samples=256, dim=256, seed=777)
    x_test, y_test = generate_dataset(num_samples=512, dim=256, seed=999)

    x_train, y_train = x_train.to(device), y_train.to(device)
    x_calib = x_calib.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    # 1. Train Baseline FP32 Reference Model
    model_fp32 = RotationModel(dim=256, hidden_dim=1024).to(device)
    optimizer = torch.optim.AdamW(model_fp32.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    print("1. Training baseline FP32 reference model (150 epochs)...")
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

    # 2. BF16
    m_bf16 = RotationModel(dim=256, hidden_dim=1024).to(device, dtype=torch.bfloat16)
    m_bf16.load_state_dict({k: v.bfloat16() for k, v in model_fp32.state_dict().items()})
    bf16_path = os.path.join(script_dir, "model_bf16.pt")
    torch.save(m_bf16.state_dict(), bf16_path)

    # 3. Naive RTN FP8 & FP4
    print("2. Applying Naive Round-To-Nearest (RTN) PTQ...")
    m_naive_fp8 = RotationModel(dim=256, hidden_dim=1024).to(device)
    m_naive_fp8.load_state_dict(model_fp32.state_dict())
    for mod in m_naive_fp8.modules():
        if isinstance(mod, nn.Linear):
            mod.weight.data = naive_quantize_weight(mod.weight.data, format_type="fp8")

    m_naive_fp4 = RotationModel(dim=256, hidden_dim=1024).to(device)
    m_naive_fp4.load_state_dict(model_fp32.state_dict())
    for mod in m_naive_fp4.modules():
        if isinstance(mod, nn.Linear):
            mod.weight.data = naive_quantize_weight(mod.weight.data, format_type="fp4")

    torch.save(m_naive_fp8.state_dict(), os.path.join(script_dir, "model_fp8_naive.pt"))
    torch.save(m_naive_fp4.state_dict(), os.path.join(script_dir, "model_fp4_naive.pt"))

    # 4. GPTQ FP4 (Hessian Error Nudged)
    print("3. Applying GPTQ (Hessian Error Nudge)...")
    m_gptq_fp4 = RotationModel(dim=256, hidden_dim=1024).to(device)
    m_gptq_fp4.load_state_dict(model_fp32.state_dict())
    m_gptq_fp4.eval()

    curr_calib = x_calib.clone().float()
    for name, module in m_gptq_fp4.named_children():
        if isinstance(module, nn.Linear):
            module.weight.data = gptq_quantize_layer(module.weight.data, curr_calib, format_type="fp4")
            with torch.no_grad():
                curr_calib = module(curr_calib)

    torch.save(m_gptq_fp4.state_dict(), os.path.join(script_dir, "model_fp4_gptq.pt"))

    # 5. AWQ FP4 (Activation-aware Channel Protection)
    print("4. Applying AWQ (Activation-aware Weight Quantization)...")
    m_awq_fp4 = RotationModel(dim=256, hidden_dim=1024).to(device)
    m_awq_fp4.load_state_dict(model_fp32.state_dict())
    m_awq_fp4.eval()

    curr_calib = x_calib.clone().float()
    for name, module in m_awq_fp4.named_children():
        if isinstance(module, nn.Linear):
            module.weight.data = awq_quantize_layer(module.weight.data, curr_calib, format_type="fp4")
            with torch.no_grad():
                curr_calib = module(curr_calib)

    torch.save(m_awq_fp4.state_dict(), os.path.join(script_dir, "model_fp4_awq.pt"))

    # 6. Fine-Tune FP8 QAT Model
    print("5. Fine-tuning FP8 QAT model with Straight-Through Estimator (50 epochs)...")
    m_qat_fp8_train = build_qat_model(model_fp32, format_type="fp8").to(device)
    opt_qat_fp8 = torch.optim.AdamW(m_qat_fp8_train.parameters(), lr=1e-4, weight_decay=1e-4)

    m_qat_fp8_train.train()
    for epoch in range(50):
        opt_qat_fp8.zero_grad()
        loss = criterion(m_qat_fp8_train(x_train), y_train)
        loss.backward()
        opt_qat_fp8.step()

    m_qat_fp8_export = export_qat_checkpoint(m_qat_fp8_train, format_type="fp8").to(device)
    torch.save(m_qat_fp8_export.state_dict(), os.path.join(script_dir, "model_fp8_qat.pt"))

    # 7. Fine-Tune FP4 QAT Model
    print("6. Fine-tuning FP4 QAT model with Straight-Through Estimator (50 epochs)...")
    m_qat_fp4_train = build_qat_model(model_fp32, format_type="fp4").to(device)
    opt_qat_fp4 = torch.optim.AdamW(m_qat_fp4_train.parameters(), lr=1e-4, weight_decay=1e-4)

    m_qat_fp4_train.train()
    for epoch in range(50):
        opt_qat_fp4.zero_grad()
        loss = criterion(m_qat_fp4_train(x_train), y_train)
        loss.backward()
        opt_qat_fp4.step()

    m_qat_fp4_export = export_qat_checkpoint(m_qat_fp4_train, format_type="fp4").to(device)
    torch.save(m_qat_fp4_export.state_dict(), os.path.join(script_dir, "model_fp4_qat.pt"))

    # Evaluate all outputs
    with torch.no_grad():
        out_bf16 = m_bf16(x_test.bfloat16()).float()
        out_naive_fp8 = m_naive_fp8(x_test).float()
        out_qat_fp8 = m_qat_fp8_export(x_test).float()
        out_naive_fp4 = m_naive_fp4(x_test).float()
        out_gptq_fp4 = m_gptq_fp4(x_test).float()
        out_awq_fp4 = m_awq_fp4(x_test).float()
        out_qat_fp4 = m_qat_fp4_export(x_test).float()

    variants = [
        ("FP32 (Ref Ground Truth)", fp32_path, out_fp32),
        ("BF16 (IEEE Truncation)", bf16_path, out_bf16),
        ("Naive FP8 (RTN)", os.path.join(script_dir, "model_fp8_naive.pt"), out_naive_fp8),
        ("QAT FP8 (STE Fine-Tuned)", os.path.join(script_dir, "model_fp8_qat.pt"), out_qat_fp8),
        ("Naive FP4 (RTN)", os.path.join(script_dir, "model_fp4_naive.pt"), out_naive_fp4),
        ("GPTQ FP4 (Hessian Nudge)", os.path.join(script_dir, "model_fp4_gptq.pt"), out_gptq_fp4),
        ("AWQ FP4 (Salient Channel)", os.path.join(script_dir, "model_fp4_awq.pt"), out_awq_fp4),
        ("QAT FP4 (STE Fine-Tuned)", os.path.join(script_dir, "model_fp4_qat.pt"), out_qat_fp4),
    ]

    sizes = {
        "FP32 (Ref Ground Truth)": "10.0 MB",
        "BF16 (IEEE Truncation)": "5.0 MB",
        "Naive FP8 (RTN)": "2.5 MB",
        "QAT FP8 (STE Fine-Tuned)": "2.5 MB",
        "Naive FP4 (RTN)": "1.4 MB",
        "GPTQ FP4 (Hessian Nudge)": "1.4 MB",
        "AWQ FP4 (Salient Channel)": "1.4 MB",
        "QAT FP4 (STE Fine-Tuned)": "1.4 MB",
    }

    print("\n" + "=" * 130)
    print("MASTER QUANTIZATION SUITE RESULTS BREAKDOWN")
    print("=" * 130)
    print(f"{'Quantization Method / Format':<27} | {'Eff. Size':<10} | {'Worst Cos Sim':<15} | {'Ref Mag':<10} | {'Var Mag':<10} | {'Mean Cos':<10}")
    print("-" * 130)
    for label, path, out_var in variants:
        m = evaluate_predictions(out_fp32, out_var, y_test)
        print(f"{label:<27} | {sizes[label]:<10} | {m['worst_cos_sim']:15.6f} | {m['worst_ref_mag']:10.4f} | {m['worst_var_mag']:10.4f} | {m['mean_cos_sim']:10.6f}")
    print("=" * 130 + "\n")

if __name__ == "__main__":
    main()
