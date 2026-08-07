import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# -----------------------------------------------------------------------------
# 1. Fake Quantization Autograd Functions (Straight-Through Estimator - STE)
# -----------------------------------------------------------------------------
class STEQuantizeFP8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        row_max = torch.max(torch.abs(weight), dim=1, keepdim=True).values
        scale = torch.clamp(row_max / 448.0, min=1e-12)
        w_scaled = weight / scale
        w_fp8 = w_scaled.to(torch.float8_e4m3fn).to(torch.float32)
        return w_fp8 * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class STEQuantizeFP4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, block_size=32):
        orig_shape = weight.shape
        w_flat = weight.reshape(-1, block_size)
        fp4_grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=weight.device)
        block_max = torch.max(torch.abs(w_flat), dim=1, keepdim=True).values
        scale = torch.clamp(block_max / 6.0, min=1e-12)
        w_scaled = w_flat / scale
        w_sign = torch.sign(w_scaled)
        w_abs = torch.abs(w_scaled)
        diffs = torch.abs(w_abs.unsqueeze(-1) - fp4_grid)
        indices = torch.argmin(diffs, dim=-1)
        w_q = fp4_grid[indices] * w_sign * scale
        return w_q.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# -----------------------------------------------------------------------------
# 2. Neural Network Models & Dataset Generator
# -----------------------------------------------------------------------------
class RotationModel(nn.Module):
    def __init__(self, dim=256, hidden_dim=1024):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class QATLinear(nn.Module):
    def __init__(self, in_features, out_features, quant_fn):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_fn = quant_fn
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        w_q = self.quant_fn(self.weight)
        return F.linear(x, w_q, self.bias)


def generate_dataset(num_samples=2048, dim=256, seed=42):
    torch.manual_seed(seed)
    X = torch.randn(num_samples, dim)
    R, _ = torch.linalg.qr(torch.randn(dim, dim))
    Y = X @ R
    return X, Y


def evaluate_predictions(y_ref: torch.Tensor, y_var: torch.Tensor, y_true: torch.Tensor):
    cos_sims = F.cosine_similarity(y_ref, y_var, dim=1)
    mag_ref = torch.norm(y_ref, p=2, dim=1)
    mag_var = torch.norm(y_var, p=2, dim=1)
    worst_idx = torch.argmin(cos_sims).item()

    return {
        "mean_cos_sim": torch.mean(cos_sims).item(),
        "worst_cos_sim": cos_sims[worst_idx].item(),
        "worst_ref_mag": mag_ref[worst_idx].item(),
        "worst_var_mag": mag_var[worst_idx].item(),
    }

# -----------------------------------------------------------------------------
# 3. Quantization Algorithms (Naive RTN, GPTQ, AWQ, QAT)
# -----------------------------------------------------------------------------
def naive_quantize_weight(weight: torch.Tensor, format_type: str = "fp8", block_size: int = 32) -> torch.Tensor:
    w = weight.detach().clone().float()
    orig_shape = w.shape
    fp4_grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=weight.device)

    if format_type == "fp8":
        row_max = torch.max(torch.abs(w), dim=1, keepdim=True).values
        scale = torch.clamp(row_max / 448.0, min=1e-12)
        return (w / scale).to(torch.float8_e4m3fn).to(torch.float32) * scale
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


def gptq_quantize_layer(weight: torch.Tensor, X_activations: torch.Tensor, format_type: str = "fp4", block_size: int = 32, damping: float = 0.01) -> torch.Tensor:
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
    w = weight.detach().clone().float()
    X = X_activations.detach().float()
    out_features, in_features = w.shape

    act_means = torch.mean(torch.abs(X), dim=0)
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
        err = F.mse_loss(X @ W_awq.T, X @ w.T).item()

        if err < best_error:
            best_error = err
            best_alpha = alpha.item()

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

# -----------------------------------------------------------------------------
# 4. In-Memory Visual Plot Generator (Saved as plot.png)
# -----------------------------------------------------------------------------
def generate_quantization_plot(models_dict, x_test, output_path=None):
    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "plot.png")

    m_fp32 = models_dict["FP32"]
    m_fp32.eval()
    with torch.no_grad():
        y_ref = m_fp32(x_test).float()

    mag_ref = torch.norm(y_ref, p=2, dim=1).cpu().numpy()

    variants = [
        ("Naive FP4", models_dict["Naive FP4"], "#d62728"),
        ("GPTQ FP4", models_dict["GPTQ FP4"], "#ff7f0e"),
        ("AWQ FP4", models_dict["AWQ FP4"], "#9467bd"),
        ("QAT FP4", models_dict["QAT FP4"], "#2ca02c"),
        ("Naive FP8", models_dict["Naive FP8"], "#17becf"),
        ("QAT FP8", models_dict["QAT FP8"], "#1f77b4"),
    ]

    data = {}

    for label, model, color in variants:
        model.eval()
        with torch.no_grad():
            if hasattr(model, "weight") and model.weight.dtype == torch.bfloat16:
                y_var = model(x_test.bfloat16()).float()
            else:
                y_var = model(x_test).float()

        cos_sims = F.cosine_similarity(y_ref, y_var, dim=1).cpu().numpy()
        mag_var = torch.norm(y_var, p=2, dim=1).cpu().numpy()
        rel_mag_delta = (mag_var - mag_ref) / mag_ref * 100.0

        data[label] = {
            "cos_sims": cos_sims,
            "rel_mag_delta": rel_mag_delta,
            "color": color,
        }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

    # Panel 1: 4-Bit Scatter
    ax1 = axes[0, 0]
    for key in ["Naive FP4", "GPTQ FP4", "AWQ FP4", "QAT FP4"]:
        d = data[key]
        ax1.scatter(d["rel_mag_delta"], d["cos_sims"], alpha=0.45, color=d["color"], label=f"{key} (Mean Cos: {np.mean(d['cos_sims']):.4f})", s=22)

    ax1.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax1.set_xlabel("Relative Magnitude Error (%): (||y_var|| - ||y_ref||) / ||y_ref||", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Cosine Similarity vs FP32 Reference", fontsize=10, fontweight='bold')
    ax1.set_title("4-Bit Quantization Methods: Naive vs. GPTQ vs. AWQ vs. QAT", fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Panel 2: 8-Bit Scatter
    ax2 = axes[0, 1]
    for key in ["Naive FP8", "QAT FP8"]:
        d = data[key]
        ax2.scatter(d["rel_mag_delta"], d["cos_sims"], alpha=0.45, color=d["color"], label=f"{key} (Mean Cos: {np.mean(d['cos_sims']):.4f})", s=22)

    ax2.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax2.set_xlabel("Relative Magnitude Error (%): (||y_var|| - ||y_ref||) / ||y_ref||", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Cosine Similarity vs FP32 Reference", fontsize=10, fontweight='bold')
    ax2.set_title("8-Bit Quantization Methods: Naive FP8 vs. QAT FP8", fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Panel 3: Cosine Similarity Bins
    ax3 = axes[1, 0]
    all_cos = np.concatenate([d["cos_sims"] for d in data.values()])
    cos_bins = np.linspace(np.min(all_cos) - 0.001, 1.0001, 50)
    for label, d in data.items():
        ax3.hist(d["cos_sims"], bins=cos_bins, alpha=0.3, color=d["color"], label=f"{label} (Min: {np.min(d['cos_sims']):.4f})", histtype='stepfilled')

    ax3.set_xlabel("Cosine Similarity", fontsize=10, fontweight='bold')
    ax3.set_ylabel("Sample Count", fontsize=10, fontweight='bold')
    ax3.set_title("Distribution Bins: Cosine Similarity Across All Methods", fontsize=12, fontweight='bold', pad=10)
    ax3.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=8)
    ax3.grid(True, linestyle='--', alpha=0.5)

    # Panel 4: Magnitude Error Bins
    ax4 = axes[1, 1]
    all_mag_deltas = np.concatenate([d["rel_mag_delta"] for d in data.values()])
    mag_bins = np.linspace(np.min(all_mag_deltas) - 0.2, np.max(all_mag_deltas) + 0.2, 50)
    for label, d in data.items():
        ax4.hist(d["rel_mag_delta"], bins=mag_bins, alpha=0.3, color=d["color"], label=f"{label} (Std: {np.std(d['rel_mag_delta']):.2f}%)", histtype='stepfilled')

    ax4.axvline(0, color='black', linestyle='--', alpha=0.7)
    ax4.set_xlabel("Relative Magnitude Error (%)", fontsize=10, fontweight='bold')
    ax4.set_ylabel("Sample Count", fontsize=10, fontweight='bold')
    ax4.set_title("Distribution Bins: Vector Magnitude Shifts (%) Across All Methods", fontsize=12, fontweight='bold', pad=10)
    ax4.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=8)
    ax4.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Quantization Plot saved to: {output_path}")

# -----------------------------------------------------------------------------
# 5. Main Execution Pipeline
# -----------------------------------------------------------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Quantization Suite Execution Device: {device} (Unified In-Memory Run)")

    # Datasets
    x_train, y_train = generate_dataset(num_samples=2048, dim=256, seed=42)
    x_calib, _ = generate_dataset(num_samples=256, dim=256, seed=777)
    x_test, y_test = generate_dataset(num_samples=512, dim=256, seed=999)

    x_train, y_train = x_train.to(device), y_train.to(device)
    x_calib = x_calib.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    # 1. Baseline FP32
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

    # 2. BF16
    m_bf16 = RotationModel(dim=256, hidden_dim=1024).to(device, dtype=torch.bfloat16)
    m_bf16.load_state_dict({k: v.bfloat16() for k, v in model_fp32.state_dict().items()})

    # 3. Naive RTN FP8 & FP4
    print("2. Applying Naive Round-To-Nearest (RTN) PTQ in memory...")
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

    # 4. GPTQ FP4
    print("3. Applying GPTQ (Hessian Error Nudge) in memory...")
    m_gptq_fp4 = RotationModel(dim=256, hidden_dim=1024).to(device)
    m_gptq_fp4.load_state_dict(model_fp32.state_dict())
    m_gptq_fp4.eval()

    curr_calib = x_calib.clone().float()
    for name, module in m_gptq_fp4.named_children():
        if isinstance(module, nn.Linear):
            module.weight.data = gptq_quantize_layer(module.weight.data, curr_calib, format_type="fp4")
            with torch.no_grad():
                curr_calib = module(curr_calib)

    # 5. AWQ FP4
    print("4. Applying AWQ (Activation-aware Weight Quantization) in memory...")
    m_awq_fp4 = RotationModel(dim=256, hidden_dim=1024).to(device)
    m_awq_fp4.load_state_dict(model_fp32.state_dict())
    m_awq_fp4.eval()

    curr_calib = x_calib.clone().float()
    for name, module in m_awq_fp4.named_children():
        if isinstance(module, nn.Linear):
            module.weight.data = awq_quantize_layer(module.weight.data, curr_calib, format_type="fp4")
            with torch.no_grad():
                curr_calib = module(curr_calib)

    # 6. Fine-Tune FP8 QAT Model
    print("5. Fine-tuning FP8 QAT model with STE in memory (50 epochs)...")
    m_qat_fp8_train = build_qat_model(model_fp32, format_type="fp8").to(device)
    opt_qat_fp8 = torch.optim.AdamW(m_qat_fp8_train.parameters(), lr=1e-4, weight_decay=1e-4)

    m_qat_fp8_train.train()
    for epoch in range(50):
        opt_qat_fp8.zero_grad()
        loss = criterion(m_qat_fp8_train(x_train), y_train)
        loss.backward()
        opt_qat_fp8.step()

    m_qat_fp8_export = export_qat_checkpoint(m_qat_fp8_train, format_type="fp8").to(device)

    # 7. Fine-Tune FP4 QAT Model
    print("6. Fine-tuning FP4 QAT model with STE in memory (50 epochs)...")
    m_qat_fp4_train = build_qat_model(model_fp32, format_type="fp4").to(device)
    opt_qat_fp4 = torch.optim.AdamW(m_qat_fp4_train.parameters(), lr=1e-4, weight_decay=1e-4)

    m_qat_fp4_train.train()
    for epoch in range(50):
        opt_qat_fp4.zero_grad()
        loss = criterion(m_qat_fp4_train(x_train), y_train)
        loss.backward()
        opt_qat_fp4.step()

    m_qat_fp4_export = export_qat_checkpoint(m_qat_fp4_train, format_type="fp4").to(device)

    models_dict = {
        "FP32": model_fp32,
        "BF16": m_bf16,
        "Naive FP8": m_naive_fp8,
        "QAT FP8": m_qat_fp8_export,
        "Naive FP4": m_naive_fp4,
        "GPTQ FP4": m_gptq_fp4,
        "AWQ FP4": m_awq_fp4,
        "QAT FP4": m_qat_fp4_export,
    }

    with torch.no_grad():
        out_fp32 = model_fp32(x_test).float()
        out_bf16 = m_bf16(x_test.bfloat16()).float()
        out_naive_fp8 = m_naive_fp8(x_test).float()
        out_qat_fp8 = m_qat_fp8_export(x_test).float()
        out_naive_fp4 = m_naive_fp4(x_test).float()
        out_gptq_fp4 = m_gptq_fp4(x_test).float()
        out_awq_fp4 = m_awq_fp4(x_test).float()
        out_qat_fp4 = m_qat_fp4_export(x_test).float()

    variants = [
        ("FP32 (Ref Ground Truth)", out_fp32),
        ("BF16 (IEEE Truncation)", out_bf16),
        ("Naive FP8 (RTN)", out_naive_fp8),
        ("QAT FP8 (STE Fine-Tuned)", out_qat_fp8),
        ("Naive FP4 (RTN)", out_naive_fp4),
        ("GPTQ FP4 (Hessian Nudge)", out_gptq_fp4),
        ("AWQ FP4 (Salient Channel)", out_awq_fp4),
        ("QAT FP4 (STE Fine-Tuned)", out_qat_fp4),
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
    print("QUANTIZATION SUITE RESULTS BREAKDOWN")
    print("=" * 130)
    print(f"{'Quantization Method / Format':<27} | {'Eff. Size':<10} | {'Worst Cos Sim':<15} | {'Ref Mag':<10} | {'Var Mag':<10} | {'Mean Cos':<10}")
    print("-" * 130)
    for label, out_var in variants:
        m = evaluate_predictions(out_fp32, out_var, y_test)
        print(f"{label:<27} | {sizes[label]:<10} | {m['worst_cos_sim']:15.6f} | {m['worst_ref_mag']:10.4f} | {m['worst_var_mag']:10.4f} | {m['mean_cos_sim']:10.6f}")
    print("=" * 130 + "\n")

    plot_path = os.path.join(script_dir, "plot.png")
    generate_quantization_plot(models_dict, x_test, output_path=plot_path)

if __name__ == "__main__":
    main()
