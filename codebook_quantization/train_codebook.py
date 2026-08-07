import os
import torch
import torch.nn as nn
from model import (
    RotationModel,
    CodebookRehydrationLinear,
    generate_dataset,
    evaluate_predictions,
)

def build_codebook_model(model_fp32: nn.Module, k_codes: int = 1024, block_h: int = 32, block_w: int = 32) -> nn.Module:
    """Converts a standard FP32 model into a Single Codebook Neural Rehydration model with custom block size."""
    m_cb = RotationModel(dim=256, hidden_dim=1024)

    for name, child in model_fp32.named_children():
        if isinstance(child, nn.Linear):
            cb_layer = CodebookRehydrationLinear(child.in_features, child.out_features, k_codes=k_codes, block_h=block_h, block_w=block_w)
            cb_layer.weight.data = child.weight.data.clone()

            # Initialize Single Codebook by sampling actual trained weight blocks
            out_f, in_f = child.weight.data.shape
            num_h, num_w = out_f // block_h, in_f // block_w
            W_blocks = child.weight.data.view(num_h, block_h, num_w, block_w).permute(0, 2, 1, 3).reshape(-1, block_h, block_w)

            k_c = min(k_codes, W_blocks.shape[0])
            sample_idxs = torch.randperm(W_blocks.shape[0])[:k_c]
            scales = torch.norm(W_blocks[sample_idxs], p=2, dim=(-2, -1), keepdim=True) / ((block_h * block_w) ** 0.5)
            cb_layer.quantizer.codebook = nn.Parameter((W_blocks[sample_idxs] / torch.clamp(scales, min=1e-6)).clone())

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


def train_codebook_variant(model_fp32: nn.Module, x_train: torch.Tensor, y_train: torch.Tensor, out_train_fp32: torch.Tensor, k_codes: int = 1024, block_h: int = 32, block_w: int = 32, device="cpu"):
    m_cb = build_codebook_model(model_fp32, k_codes=k_codes, block_h=block_h, block_w=block_w).to(device)
    opt_cb = torch.optim.AdamW(m_cb.parameters(), lr=4e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    num_epochs = 35
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
    print(f"Single Codebook Block Size Sweep Device: {device}", flush=True)

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

    # 2. Block Size Sweep (32x32 vs 16x16 vs 8x8 vs 4x4)
    block_configs = [
        ("32x32 Block (1024 params/block)", 32, 32, "0.18 MB", "model_codebook_32x32.pt"),
        ("16x16 Block (256 params/block)", 16, 16, "0.22 MB", "model_codebook_16x16.pt"),
        ("8x8 Block (64 params/block)", 8, 8, "0.32 MB", "model_codebook_8x8.pt"),
        ("4x4 Block (16 params/block)", 4, 4, "0.45 MB", "model_codebook_4x4.pt"),
    ]

    results = []

    for label, bh, bw, size_str, fname in block_configs:
        print(f"Training {label}...", flush=True)
        m_variant = train_codebook_variant(model_fp32, x_train, y_train, out_train_fp32, k_codes=1024, block_h=bh, block_w=bw, device=device)
        save_path = os.path.join(script_dir, fname)
        torch.save(m_variant.state_dict(), save_path)

        with torch.no_grad():
            out_hard = evaluate_codebook_model(m_variant, x_test, hard=True).float()

        m_eval = evaluate_predictions(out_fp32, out_hard, y_test)
        results.append((label, size_str, m_eval))

    print("\n" + "=" * 110, flush=True)
    print("BLOCK SIZE SWEEP BENCHMARK RESULTS (K=1024)")
    print("=" * 110, flush=True)
    print(f"{'Block Size Configuration':<35} | {'Eff. Size':<10} | {'Worst Cos Sim':<15} | {'Ref Mag':<10} | {'Var Mag':<10} | {'Mean Cos':<10}")
    print("-" * 110, flush=True)
    for label, size_str, m in results:
        print(f"{label:<35} | {size_str:<10} | {m['worst_cos_sim']:15.6f} | {m['worst_ref_mag']:10.4f} | {m['worst_var_mag']:10.4f} | {m['mean_cos_sim']:10.6f}")
    print("=" * 110 + "\n", flush=True)

if __name__ == "__main__":
    main()
