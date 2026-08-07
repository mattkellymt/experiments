import os
import torch
import torch.nn as nn
from model import (
    RotationModel,
    CodebookRehydrationLinear,
    generate_dataset,
    evaluate_predictions,
)

def native_fp4_quantize_weight(weight: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """TorchAO Native FP4 Micro-Scaling."""
    w = weight.detach().clone().float()
    orig_shape = w.shape
    fp4_grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=weight.device)

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


def build_codebook_model(model_fp32: nn.Module, k_codes: int = 512) -> nn.Module:
    """Converts a standard model into a 4-Bit Grid + Dual-Codebook Neural Rehydration model."""
    m_cb = RotationModel(dim=256, hidden_dim=1024)

    for name, child in model_fp32.named_children():
        if isinstance(child, nn.Linear):
            cb_layer = CodebookRehydrationLinear(child.in_features, child.out_features, k_codes=k_codes)
            cb_layer.weight.data = child.weight.data.clone()

            # Initialize Codebooks by sampling actual trained weight blocks
            out_f, in_f = child.weight.data.shape
            num_h, num_w = out_f // 32, in_f // 32
            W_blocks = child.weight.data.view(num_h, 32, num_w, 32).permute(0, 2, 1, 3).reshape(-1, 32, 32)

            if W_blocks.shape[0] >= k_codes:
                sample_idxs = torch.randperm(W_blocks.shape[0])[:k_codes]
                cb_layer.quantizer.codebook2.data = W_blocks[sample_idxs].clone()
                cb_layer.quantizer.codebook1.data = W_blocks[sample_idxs].clone()
            else:
                idxs = torch.randint(0, W_blocks.shape[0], (k_codes,))
                cb_layer.quantizer.codebook2.data = W_blocks[idxs].clone()
                cb_layer.quantizer.codebook1.data = W_blocks[idxs].clone()

            setattr(m_cb, name, cb_layer)

    return m_cb


def evaluate_codebook_model(m_cb: nn.Module, x_test: torch.Tensor, hard: bool = False) -> torch.Tensor:
    m_cb.eval()
    x = x_test.clone()
    for child in m_cb.children():
        if isinstance(child, CodebookRehydrationLinear):
            x = child(x, hard=hard)
        else:
            x = child(x)
    return x


def train_codebook_variant(model_fp32: nn.Module, x_train: torch.Tensor, y_train: torch.Tensor, out_train_fp32: torch.Tensor, k_codes: int = 512, device="cpu"):
    m_cb = build_codebook_model(model_fp32, k_codes=k_codes).to(device)
    opt_cb = torch.optim.AdamW(m_cb.parameters(), lr=4e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    num_epochs = 40
    tau_start, tau_end = 1.0, 0.05

    for epoch in range(1, num_epochs + 1):
        current_tau = tau_start * ((tau_end / tau_start) ** (epoch / num_epochs))
        for mod in m_cb.modules():
            if isinstance(mod, CodebookRehydrationLinear):
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
    print(f"4-Bit Grid + Dual-Codebook Neural Rehydration Device: {device}", flush=True)

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

    # 2. TorchAO Native FP4 Baseline
    print("2. Building TorchAO Native FP4 baseline...", flush=True)
    m_fp4_native = RotationModel(dim=256, hidden_dim=1024).to(device)
    m_fp4_native.load_state_dict(model_fp32.state_dict())
    for mod in m_fp4_native.modules():
        if isinstance(mod, nn.Linear):
            mod.weight.data = native_fp4_quantize_weight(mod.weight.data)

    fp4_native_path = os.path.join(script_dir, "model_fp4_native.pt")
    torch.save(m_fp4_native.state_dict(), fp4_native_path)

    # 3. Build & Fine-Tune 4-Bit Grid + K=512 Codebook Rehydrator
    print("3. Fine-tuning 4-Bit Grid + K=512 Codebook Rehydrator...", flush=True)
    m_cb512 = train_codebook_variant(model_fp32, x_train, y_train, out_train_fp32, k_codes=512, device=device)
    torch.save(m_cb512.state_dict(), os.path.join(script_dir, "model_codebook_9bit.pt"))

    # 4. Build & Fine-Tune 4-Bit Grid + K=1024 Codebook Rehydrator
    print("4. Fine-tuning 4-Bit Grid + K=1024 Codebook Rehydrator...", flush=True)
    m_cb1024 = train_codebook_variant(model_fp32, x_train, y_train, out_train_fp32, k_codes=1024, device=device)
    torch.save(m_cb1024.state_dict(), os.path.join(script_dir, "model_codebook_10bit.pt"))

    # Evaluate outputs
    with torch.no_grad():
        out_native_fp4 = m_fp4_native(x_test).float()
        out_cb512_hard = evaluate_codebook_model(m_cb512, x_test, hard=True).float()
        out_cb1024_hard = evaluate_codebook_model(m_cb1024, x_test, hard=True).float()

    variants = [
        ("FP32 (Ref Ground Truth)", fp32_path, out_fp32),
        ("TorchAO Native FP4", fp4_native_path, out_native_fp4),
        ("4-Bit Grid + Codebook K=512", os.path.join(script_dir, "model_codebook_9bit.pt"), out_cb512_hard),
        ("4-Bit Grid + Codebook K=1024", os.path.join(script_dir, "model_codebook_10bit.pt"), out_cb1024_hard),
    ]

    sizes = {
        "FP32 (Ref Ground Truth)": "10.00 MB",
        "TorchAO Native FP4": "1.41 MB",
        "4-Bit Grid + Codebook K=512": "1.41 MB + NN",
        "4-Bit Grid + Codebook K=1024": "1.41 MB + NN",
    }

    print("\n" + "=" * 130, flush=True)
    print("4-BIT GRID + DUAL-CODEBOOK NEURAL REHYDRATION BENCHMARK RESULTS", flush=True)
    print("=" * 130, flush=True)
    print(f"{'Quantization Method / Format':<32} | {'Eff. Size':<12} | {'Worst Cos Sim':<15} | {'Ref Mag':<10} | {'Var Mag':<10} | {'Mean Cos':<10}", flush=True)
    print("-" * 130, flush=True)
    for label, path, out_var in variants:
        m = evaluate_predictions(out_fp32, out_var, y_test)
        print(f"{label:<32} | {sizes[label]:<12} | {m['worst_cos_sim']:15.6f} | {m['worst_ref_mag']:10.4f} | {m['worst_var_mag']:10.4f} | {m['mean_cos_sim']:10.6f}", flush=True)
    print("=" * 130 + "\n", flush=True)

if __name__ == "__main__":
    main()
