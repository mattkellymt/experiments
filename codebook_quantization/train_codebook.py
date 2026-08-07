import os
import torch
import torch.nn as nn
from model import (
    RotationModel,
    STENeuralRehydrationLinear,
    generate_dataset,
    evaluate_predictions,
)

def build_ste_model(model_fp32: nn.Module, format_type: str = "fp4") -> nn.Module:
    """Converts a standard FP32 model into an STE Binning + Non-Linear Neural Codebook Rehydration model."""
    m_cb = RotationModel(dim=256, hidden_dim=1024)

    for name, child in model_fp32.named_children():
        if isinstance(child, nn.Linear):
            cb_layer = STENeuralRehydrationLinear(child.in_features, child.out_features, format_type=format_type)
            cb_layer.quantizer.weight_master.data = child.weight.data.clone()
            setattr(m_cb, name, cb_layer)

    return m_cb


def evaluate_ste_model(m_cb: nn.Module, x_test: torch.Tensor) -> torch.Tensor:
    m_cb.eval()
    x = x_test.clone()
    for child in m_cb.children():
        if isinstance(child, STENeuralRehydrationLinear):
            x = child(x)
        else:
            x = child(x)
    return x


def train_ste_variant(model_fp32: nn.Module, x_train: torch.Tensor, y_train: torch.Tensor, out_train_fp32: torch.Tensor, format_type: str = "fp4", device="cpu"):
    m_cb = build_ste_model(model_fp32, format_type=format_type).to(device)
    opt_cb = torch.optim.AdamW(m_cb.parameters(), lr=4e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    num_epochs = 60

    for epoch in range(1, num_epochs + 1):
        m_cb.train()
        opt_cb.zero_grad()

        out_pred = evaluate_ste_model(m_cb, x_train)
        loss = criterion(out_pred, y_train) + criterion(out_pred, out_train_fp32)
        loss.backward()
        opt_cb.step()

    if device.type == "mps":
        torch.mps.empty_cache()

    return m_cb


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"STE Binning + Non-Linear Neural Rehydration Device: {device}", flush=True)

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

    # 2. Train STE FP4 Binned + Non-Linear Rehydrator
    print("2. Fine-tuning STE FP4 Binned + Non-Linear Rehydrator (60 epochs)...", flush=True)
    m_ste_fp4 = train_ste_variant(model_fp32, x_train, y_train, out_train_fp32, format_type="fp4", device=device)
    torch.save(m_ste_fp4.state_dict(), os.path.join(script_dir, "model_ste_fp4.pt"))

    # 3. Train STE FP8 Binned + Non-Linear Rehydrator
    print("3. Fine-tuning STE FP8 Binned + Non-Linear Rehydrator (60 epochs)...", flush=True)
    m_ste_fp8 = train_ste_variant(model_fp32, x_train, y_train, out_train_fp32, format_type="fp8", device=device)
    torch.save(m_ste_fp8.state_dict(), os.path.join(script_dir, "model_ste_fp8.pt"))

    # Evaluate outputs
    with torch.no_grad():
        out_ste_fp4 = evaluate_ste_model(m_ste_fp4, x_test).float()
        out_ste_fp8 = evaluate_ste_model(m_ste_fp8, x_test).float()

    variants = [
        ("FP32 (Ref Ground Truth)", fp32_path, out_fp32),
        ("STE FP4 Binned + Neural Rehydrator", os.path.join(script_dir, "model_ste_fp4.pt"), out_ste_fp4),
        ("STE FP8 Binned + Neural Rehydrator", os.path.join(script_dir, "model_ste_fp8.pt"), out_ste_fp8),
    ]

    sizes = {
        "FP32 (Ref Ground Truth)": "10.00 MB",
        "STE FP4 Binned + Neural Rehydrator": "1.41 MB + NN",
        "STE FP8 Binned + Neural Rehydrator": "2.62 MB + NN",
    }

    print("\n" + "=" * 135, flush=True)
    print("STE BINNING + NON-LINEAR NEURAL CODEBOOK REHYDRATION BENCHMARK RESULTS")
    print("=" * 135, flush=True)
    print(f"{'Quantization Method / Format':<37} | {'Eff. Size':<12} | {'Worst Cos Sim':<15} | {'Ref Mag':<10} | {'Var Mag':<10} | {'Mean Cos':<10}", flush=True)
    print("-" * 135, flush=True)
    for label, path, out_var in variants:
        m = evaluate_predictions(out_fp32, out_var, y_test)
        print(f"{label:<37} | {sizes[label]:<12} | {m['worst_cos_sim']:15.6f} | {m['worst_ref_mag']:10.4f} | {m['worst_var_mag']:10.4f} | {m['mean_cos_sim']:10.6f}", flush=True)
    print("=" * 135 + "\n", flush=True)

if __name__ == "__main__":
    main()
