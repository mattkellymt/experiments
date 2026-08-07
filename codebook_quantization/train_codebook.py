import os
import torch
import torch.nn as nn
from model import (
    RotationModel,
    QuantizedKeyTransformLinear,
    generate_dataset,
    evaluate_predictions,
)

def build_key_transform_model(model_fp32: nn.Module, method_type: str = "method1_kv_router", format_type: str = "fp4") -> nn.Module:
    m_cb = RotationModel(dim=256, hidden_dim=1024)

    for name, child in model_fp32.named_children():
        if isinstance(child, nn.Linear):
            layer = QuantizedKeyTransformLinear(child.in_features, child.out_features, method_type=method_type, format_type=format_type)
            layer.weight.data = child.weight.data.clone()
            setattr(m_cb, name, layer)

    return m_cb


def evaluate_key_transform_model(m_cb: nn.Module, x_test: torch.Tensor) -> torch.Tensor:
    m_cb.eval()
    x = x_test.clone()
    for child in m_cb.children():
        if isinstance(child, QuantizedKeyTransformLinear):
            x = child(x)
        else:
            x = child(x)
    return x


def train_key_transform_variant(model_fp32: nn.Module, x_train: torch.Tensor, y_train: torch.Tensor, out_train_fp32: torch.Tensor, method_type: str = "method1_kv_router", format_type: str = "fp4", device="cpu"):
    m_cb = build_key_transform_model(model_fp32, method_type=method_type, format_type=format_type).to(device)
    opt_cb = torch.optim.AdamW(m_cb.parameters(), lr=3e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    num_epochs = 50

    for epoch in range(1, num_epochs + 1):
        m_cb.train()
        opt_cb.zero_grad()

        out_pred = evaluate_key_transform_model(m_cb, x_train)
        loss = criterion(out_pred, y_train) + criterion(out_pred, out_train_fp32)
        loss.backward()
        opt_cb.step()

    if device.type == "mps":
        torch.mps.empty_cache()

    return m_cb


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Quantized Key Transform Methods Device: {device}", flush=True)

    # Datasets
    x_train, y_train = generate_dataset(num_samples=1024, dim=256, seed=42)
    x_test, y_test = generate_dataset(num_samples=512, dim=256, seed=999)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    # 1. Train Baseline FP32 Model
    model_fp32 = RotationModel(dim=256, hidden_dim=1024).to(device)
    optimizer = torch.optim.AdamW(model_fp32.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    print("1. Training baseline FP32 model (150 epochs)...", flush=True)
    model_fp32.train()
    for epoch in range(150):
        optimizer.zero_grad()
        loss = criterion(model_fp32(x_train), y_train)
        loss.backward()
        optimizer.step()

    model_fp32.eval()
    with torch.no_grad():
        out_fp32 = model_fp32(x_test)
        out_train_fp32 = model_fp32(x_train)

    fp32_path = os.path.join(script_dir, "model_fp32.pt")
    torch.save(model_fp32.state_dict(), fp32_path)

    # 2. Benchmark 3 Quantized Key Transform Methods
    configs = [
        ("Method 1: Key-Value Codebook Router (FP4)", "method1_kv_router", "fp4", "1.41 MB + NN", "model_m1_fp4.pt"),
        ("Method 2: Multi-Head Key Attention (FP4)", "method2_mh_attn", "fp4", "1.41 MB + NN", "model_m2_fp4.pt"),
        ("Method 3: Deep Key Projection Net (FP4)", "method3_deep_kpn", "fp4", "1.41 MB + NN", "model_m3_fp4.pt"),
        ("Method 1: Key-Value Codebook Router (FP8)", "method1_kv_router", "fp8", "2.62 MB + NN", "model_m1_fp8.pt"),
        ("Method 2: Multi-Head Key Attention (FP8)", "method2_mh_attn", "fp8", "2.62 MB + NN", "model_m2_fp8.pt"),
        ("Method 3: Deep Key Projection Net (FP8)", "method3_deep_kpn", "fp8", "2.62 MB + NN", "model_m3_fp8.pt"),
    ]

    results = []

    for label, method, fmt, size_str, fname in configs:
        print(f"Training {label}...", flush=True)
        m_variant = train_key_transform_variant(model_fp32, x_train, y_train, out_train_fp32, method_type=method, format_type=fmt, device=device)
        save_path = os.path.join(script_dir, fname)
        torch.save(m_variant.state_dict(), save_path)

        with torch.no_grad():
            out_var = evaluate_key_transform_model(m_variant, x_test).float()

        m_eval = evaluate_predictions(out_fp32, out_var, y_test)
        results.append((label, size_str, m_eval))

    print("\n" + "=" * 135, flush=True)
    print("QUANTIZED KEY TRANSFORM METHODS BENCHMARK RESULTS")
    print("=" * 135, flush=True)
    print(f"{'Quantized Key Transform Method':<42} | {'Eff. Size':<12} | {'Worst Cos Sim':<15} | {'Ref Mag':<10} | {'Var Mag':<10} | {'Mean Cos':<10}", flush=True)
    print("-" * 135, flush=True)
    for label, size_str, m in results:
        print(f"{label:<42} | {size_str:<12} | {m['worst_cos_sim']:15.6f} | {m['worst_ref_mag']:10.4f} | {m['worst_var_mag']:10.4f} | {m['mean_cos_sim']:10.6f}", flush=True)
    print("=" * 135 + "\n", flush=True)

if __name__ == "__main__":
    main()
