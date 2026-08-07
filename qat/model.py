import torch
import torch.nn as nn

torch.manual_seed(42)

class STEQuantizeFP8(torch.autograd.Function):
    """
    Fake Quantization for FP8 with Straight-Through Estimator (STE):
    Forward pass: Quantizes weights to Float8 (float8_e4m3fn) with per-row scaling.
    Backward pass: Passes gradients straight through to continuous FP32 master weights.
    """
    @staticmethod
    def forward(ctx, weight):
        row_max = torch.max(torch.abs(weight), dim=1, keepdim=True).values
        scale = torch.clamp(row_max / 448.0, min=1e-12)
        w_q = (weight / scale).to(torch.float8_e4m3fn).to(torch.float32) * scale
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class STEQuantizeFP4(torch.autograd.Function):
    """
    Fake Quantization for FP4 (E2M1 Grid) with Straight-Through Estimator (STE):
    Forward pass: Quantizes weights to 4-bit E2M1 grid over 32-element blocks.
    Backward pass: Passes gradients straight through to continuous FP32 master weights.
    """
    @staticmethod
    def forward(ctx, weight, block_size=32):
        fp4_grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=weight.device)
        orig_shape = weight.shape
        w_flat = weight.reshape(-1, block_size)
        block_max = torch.max(torch.abs(w_flat), dim=1, keepdim=True).values
        scale = torch.clamp(block_max / 6.0, min=1e-12)
        w_scaled = w_flat / scale
        w_sign = torch.sign(w_scaled)
        w_abs = torch.abs(w_scaled)
        diffs = torch.abs(w_abs.unsqueeze(-1) - fp4_grid)
        indices = torch.argmin(diffs, dim=-1)
        w_q = (fp4_grid[indices] * w_sign * scale).reshape(orig_shape)
        return w_q

    @staticmethod
    def backward(ctx, grad_output, block_size=None):
        return grad_output, None


class QATLinear(nn.Module):
    """Linear layer wrapper that applies Fake Quantization during forward pass."""
    def __init__(self, in_features: int, out_features: int, ste_fn):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.ste_fn = ste_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_fake_q = self.ste_fn(self.weight)
        return torch.nn.functional.linear(x, w_fake_q)


class RotationModel(nn.Module):
    def __init__(self, dim: int = 256, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.act2 = nn.GELU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.act3 = nn.GELU()
        self.fc4 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.fc1(x))
        h = self.act2(self.fc2(h))
        h = self.act3(self.fc3(h))
        return self.fc4(h)


def generate_dataset(num_samples: int = 1024, dim: int = 256, seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    W_target = torch.randn(dim, dim, generator=g) * 0.1
    U, _, V = torch.linalg.svd(torch.randn(dim, dim, generator=g))
    R_target = U @ V.T

    x_in = torch.randn(num_samples, dim, generator=g)
    y_target = torch.nn.functional.gelu(x_in @ W_target) @ R_target
    return x_in, y_target


def evaluate_predictions(y_ref: torch.Tensor, y_var: torch.Tensor, y_gt: torch.Tensor):
    pred_ref = y_ref.detach().float()
    pred_var = y_var.detach().float()
    gt = y_gt.detach().float()

    cos_sims = torch.nn.functional.cosine_similarity(pred_ref, pred_var, dim=1)
    mag_ref = torch.norm(pred_ref, p=2, dim=1)
    mag_var = torch.norm(pred_var, p=2, dim=1)

    worst_idx = torch.argmin(cos_sims).item()
    sorted_idxs = torch.argsort(cos_sims)
    median_idx = sorted_idxs[len(sorted_idxs) // 2].item()

    return {
        "test_mse": torch.nn.functional.mse_loss(pred_var, gt).item(),
        "mean_cos_sim": torch.mean(cos_sims).item(),
        "worst_cos_sim": cos_sims[worst_idx].item(),
        "worst_ref_mag": mag_ref[worst_idx].item(),
        "worst_var_mag": mag_var[worst_idx].item(),
        "median_cos_sim": cos_sims[median_idx].item(),
        "median_ref_mag": mag_ref[median_idx].item(),
        "median_var_mag": mag_var[median_idx].item(),
    }
