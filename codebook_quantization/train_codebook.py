import os
import torch
import torch.nn as nn
from model import (
    RotationModel,
    FP8NeuralRehydrationLinear,
    generate_dataset,
    evaluate_predictions,
    quantize_fp8_block,
)

def raw_fp8_quantize_model(model_fp32: nn.Module) -> nn.Module:
    """Quantizes model weights to raw FP8 without neural rehydration."""
    m_fp8 = RotationModel(dim=256, hidden_dim=1024)
    m_fp8.load_state_dict(model_fp32.state_dict())
    for mod in m_fp8.modules():
        if isinstance(mod, nn.Linear):
            w = mod.weight.data
            out_f, in_f = w.shape
            w_b = w.view(out_f // 32, 32, in_f // 32, 32).permute(0, 2, 1, 3).reshape(-1, 32, 32)
            w_q_b = quantize_fp8_block(w_b)
            mod.weight.data = w_q_b.view(out_f // 32, in_f // 32, 32, 32).permute(0, 2, 1, 3).reshape(out_f, in_f)
    return m_fp8


def build_nonlinear_fp8_model(model_fp32: nn.Module, k_codes: int = 256) -> nn.Module:
    """Converts a standard model into an FP8 Non-Linear Neural Codebook Rehydration model."""
    m_cb = RotationModel(dim=256, hidden_dim=1024)

    for name, child in model_fp32.named_children():
        if isinstance(child, nn.Linear):
            cb_layer = FP8NeuralRehydrationLinear(child.in_features, child.out_features, k_codes=k_codes)
            cb_layer.weight.data = child.weight.data.clone()
            setattr(m_cb, name, cb_layer)

    return m_cb


def evaluate_codebook_model(m_cb: nn.Module, x_test: torch.Tensor, hard: bool = False) -> torch.Tensor:
    m_cb.eval()
    x = x_test.clone()
    for child in m_cb.children():
        if isinstance(child, FP8NeuralRehydrationLinear):
            x = child(x, hard=hard)
        else:
            x = child(x)
    return x


def train_nonlinear_fp8_model(model_fp32: nn.Module, x_train: torch.Tensor, y_train: torch.Tensor, out_train_fp32: torch.Tensor, k_codes: int = 256, device="cpu"):
    m_cb = build_nonlinear_fp8_model(model_fp32, k_codes=k_codes).to(device)
    opt_cb = torch.optim.AdamW(m_cb.parameters(), lr=4e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    num_epochs = 40
    tau_start, tau_end = 1.0, 0.05

    for epoch in range(1, num_epochs + 1):
        current_tau = tau_start * ((tau_end / tau_start) ** (epoch / num_epochs))
        for mod in m_cb.modules():
            if isinstance(mod, FP8NeuralRehydrationLinear):
                mod.quantizer.tau = current_tau

        m_cb.train()
        opt_cb.zero_grad()

        out_pred = evaluate_codebook_model(m_cb, x_train, hard=False)
        loss = criterion(out_pred, y_train) + criterion(out_pred, out_train_fp32)
        loss.backward()
        opt_cb.step()

    if device.type == "mps":
        torch.mps.empty_cache()

    return m_cb


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Non-Linear FP8 Neural Codebook Rehydration Device: {device}", flush=True)

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

    # 2. Raw Naive FP8 Baseline
    print("2. Building Raw Naive FP8 baseline (no neural rehydration)...", flush=True)
    m_raw_fp8 = raw_fp8_quantize_model(model_fp32).to(device)
    torch.save(m_raw_fp8.state_dict(), os.path.join(script_dir, "model_raw_fp8.pt"))

    # 3. Fine-Tune Non-Linear FP8 Neural Rehydrator (K=256)
    print("3. Fine-tuning Non-Linear FP8 Neural Rehydrator (K=256)...", flush=True)
    m_nonlinear_fp8 = train_nonlinear_fp8_model(model_fp32, x_train, y_train, out_train_fp32, k_codes=256, device=device)
    torch.save(m_nonlinear_fp8.state_dict(), os.path.join(script_dir, "model_nonlinear_fp8.pt"))

    # Evaluate outputs
    with torch.no_grad():
        out_raw_fp8 = m_raw_fp8(x_test).float()
        out_nonlinear_fp8 = evaluate_codebook_model(m_nonlinear_fp8, x_test, hard=True).float()

    variants = [
        ("FP32 (Ref Ground Truth)", fp32_path, out_fp32),
        ("Raw Naive FP8 Baseline", os.path.join(script_dir, "model_raw_fp8.pt"), out_raw_fp8),
        ("Non-Linear Neural FP8 Rehydrator", os.path.join(script_dir, "model_nonlinear_fp8.pt"), out_nonlinear_fp8),
    ]

    sizes = {
        "FP32 (Ref Ground Truth)": "10.00 MB",
        "Raw Naive FP8 Baseline": "2.62 MB",
        "Non-Linear Neural FP8 Rehydrator": "2.62 MB + NN",
    }

    print("\n" + "=" * 130, flush=True)
    print("NON-LINEAR FP8 NEURAL CODEBOOK REHYDRATION BENCHMARK RESULTS")
    print("=" * 130, flush=True)
    print(f"{'Quantization Method / Format':<35} | {'Eff. Size':<12} | {'Worst Cos Sim':<15} | {'Ref Mag':<10} | {'Var Mag':<10} | {'Mean Cos':<10}", flush=True)
    print("-" * 130, flush=True)
    for label, path, out_var in variants:
        m = evaluate_predictions(out_fp32, out_var, y_test)
        print(f"{label:<35} | {sizes[label]:<12} | {m['worst_cos_sim']:15.6f} | {m['worst_ref_mag']:10.4f} | {m['worst_var_mag']:10.4f} | {m['mean_cos_sim']:10.6f}", flush=True)
    print("=" * 130 + "\n", flush=True)

if __name__ == "__main__":
    main()
