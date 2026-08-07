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

def ptq_quantize_weight(weight: torch.Tensor, format_type: str = "fp8", block_size: int = 32) -> torch.Tensor:
    """Post-Training Quantization (PTQ) without training."""
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


def build_qat_model(model_fp32: nn.Module, format_type: str) -> nn.Module:
    """Converts a standard model into a QAT model with STE Fake Quantization wrappers."""
    m_qat = RotationModel(dim=256, hidden_dim=1024)
    ste_fn = STEQuantizeFP8.apply if format_type == "fp8" else STEQuantizeFP4.apply

    for name, child in model_fp32.named_children():
        if isinstance(child, nn.Linear):
            qat_layer = QATLinear(child.in_features, child.out_features, ste_fn)
            qat_layer.weight.data = child.weight.data.clone()
            setattr(m_qat, name, qat_layer)

    return m_qat


def export_qat_checkpoint(m_qat: nn.Module, format_type: str) -> nn.Module:
    """Exports a fine-tuned QAT model to a clean eval-ready model with quantized weights."""
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
    print(f"QAT Execution Device: {device}")

    # Generate Datasets
    x_train, y_train = generate_dataset(num_samples=2048, dim=256, seed=42)
    x_test, y_test = generate_dataset(num_samples=512, dim=256, seed=999)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    # 1. Train Baseline FP32 Model
    model_fp32 = RotationModel(dim=256, hidden_dim=1024).to(device)
    optimizer = torch.optim.AdamW(model_fp32.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    print("Training baseline FP32 model (150 epochs)...")
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

    # 2. Build Post-Training Quantization (PTQ) Baselines
    print("Building Post-Training Quantization (PTQ) baselines...")
    m_ptq_fp8 = RotationModel(dim=256, hidden_dim=1024).to(device)
    m_ptq_fp8.load_state_dict(model_fp32.state_dict())
    for mod in m_ptq_fp8.modules():
        if isinstance(mod, nn.Linear):
            mod.weight.data = ptq_quantize_weight(mod.weight.data, format_type="fp8")

    m_ptq_fp4 = RotationModel(dim=256, hidden_dim=1024).to(device)
    m_ptq_fp4.load_state_dict(model_fp32.state_dict())
    for mod in m_ptq_fp4.modules():
        if isinstance(mod, nn.Linear):
            mod.weight.data = ptq_quantize_weight(mod.weight.data, format_type="fp4")

    torch.save(m_ptq_fp8.state_dict(), os.path.join(script_dir, "model_fp8_ptq.pt"))
    torch.save(m_ptq_fp4.state_dict(), os.path.join(script_dir, "model_fp4_ptq.pt"))

    # 3. Fine-Tune FP8 QAT Model
    print("Fine-tuning FP8 QAT model with Straight-Through Estimator (50 epochs)...")
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

    # 4. Fine-Tune FP4 QAT Model
    print("Fine-tuning FP4 QAT model with Straight-Through Estimator (50 epochs)...")
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

    # Evaluate all outputs against test dataset
    with torch.no_grad():
        out_ptq_fp8 = m_ptq_fp8(x_test).float()
        out_qat_fp8 = m_qat_fp8_export(x_test).float()
        out_ptq_fp4 = m_ptq_fp4(x_test).float()
        out_qat_fp4 = m_qat_fp4_export(x_test).float()

    variants = [
        ("FP32 (Ref)", fp32_path, out_fp32),
        ("PTQ FP8 (Post-Train)", os.path.join(script_dir, "model_fp8_ptq.pt"), out_ptq_fp8),
        ("QAT FP8 (Fine-Tuned)", os.path.join(script_dir, "model_fp8_qat.pt"), out_qat_fp8),
        ("PTQ FP4 (Post-Train)", os.path.join(script_dir, "model_fp4_ptq.pt"), out_ptq_fp4),
        ("QAT FP4 (Fine-Tuned)", os.path.join(script_dir, "model_fp4_qat.pt"), out_qat_fp4),
    ]

    sizes = {
        "FP32 (Ref)": "10.0 MB",
        "PTQ FP8 (Post-Train)": "2.5 MB",
        "QAT FP8 (Fine-Tuned)": "2.5 MB",
        "PTQ FP4 (Post-Train)": "1.4 MB",
        "QAT FP4 (Fine-Tuned)": "1.4 MB",
    }

    print("\n" + "=" * 125)
    print("QUANTIZATION-AWARE TRAINING (QAT) VS. POST-TRAINING QUANTIZATION (PTQ) RESULTS")
    print("=" * 125)
    print(f"{'Precision Variant':<24} | {'Eff. Size':<10} | {'Worst Cos Sim':<15} | {'Ref Mag':<10} | {'Var Mag':<10} | {'Mean Cos':<10}")
    print("-" * 125)
    for label, path, out_var in variants:
        m = evaluate_predictions(out_fp32, out_var, y_test)
        print(f"{label:<24} | {sizes[label]:<10} | {m['worst_cos_sim']:15.6f} | {m['worst_ref_mag']:10.4f} | {m['worst_var_mag']:10.4f} | {m['mean_cos_sim']:10.6f}")
    print("=" * 125 + "\n")

if __name__ == "__main__":
    main()
