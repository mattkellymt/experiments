import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# STE Autograd Function for FP4 Binning
class STEFP4Grid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight_blocks: torch.Tensor, block_elements: int = 1024) -> torch.Tensor:
        w = weight_blocks.detach().clone()
        fp4_grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=weight_blocks.device)

        w_flat = w.view(-1, block_elements)
        scale = torch.clamp(torch.max(torch.abs(w_flat), dim=1, keepdim=True).values / 6.0, min=1e-12)
        w_scaled = w_flat / scale
        w_sign = torch.sign(w_scaled)
        w_abs = torch.abs(w_scaled)
        diffs = torch.abs(w_abs.unsqueeze(-1) - fp4_grid)
        indices = torch.argmin(diffs, dim=-1)
        w_q = fp4_grid[indices] * w_sign * scale
        return w_q.view_as(weight_blocks)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None  # Straight-Through Estimator Identity Pass-Through Gradient!

ste_fp4_quantize = STEFP4Grid.apply


# STE Autograd Function for FP8 Binning
class STEFP8Grid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight_blocks: torch.Tensor, block_elements: int = 1024) -> torch.Tensor:
        w = weight_blocks.detach().clone()
        w_flat = w.view(-1, block_elements)
        scale = torch.clamp(torch.max(torch.abs(w_flat), dim=1, keepdim=True).values / 448.0, min=1e-12)
        w_scaled = w_flat / scale
        w_q = torch.clamp(torch.round(w_scaled), -448.0, 448.0)
        return (w_q * scale).view_as(weight_blocks)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None  # Straight-Through Estimator Identity Pass-Through Gradient!

ste_fp8_quantize = STEFP8Grid.apply


class STENonLinearCodebookRehydrator(nn.Module):
    """
    STE (Straight-Through Estimator) Binning + Non-Linear Neural Codebook Rehydrator:
    1. FP32 Master Weights (W_master): Fine-tuned continuously via STE gradients.
    2. STE Discrete Grid Binning: In forward pass, force W_master into FP4 or FP8 bins (W_binned).
       In backward pass, STE passes gradients directly to W_master!
    3. Non-Linear Feature Extractor (MLP + GELU):
       h = GELU(Linear(W_binned_flat)) -> maps discrete bin steps into smooth feature space.
    4. Gated Non-Linear Neural Rehydration:
       W_rehydrated = W_binned + gamma * RefinementHead(h)
    5. Allows master weights to settle into optimal bins while the neural engine cleans up step noise!
    """
    def __init__(self, out_features: int, in_features: int, format_type: str = "fp4", block_h: int = 32, block_w: int = 32, hidden_dim: int = 512):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.format_type = format_type.lower()
        self.block_h = block_h
        self.block_w = block_w
        self.block_elements = block_h * block_w

        self.num_h = out_features // block_h
        self.num_w = in_features // block_w
        self.num_blocks = self.num_h * self.num_w

        # Master FP32 Weights
        self.weight_master = nn.Parameter(torch.randn(out_features, in_features) * 0.05)

        # Non-Linear Neural Feature Extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.block_elements, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        # Non-Linear Neural Refinement Head
        self.refinement_head = nn.Linear(hidden_dim, self.block_elements)
        self.gamma = nn.Parameter(torch.zeros(1))  # Start at zero gate to guarantee stability!

    def forward(self, hard: bool = False) -> torch.Tensor:
        # Slices master weights into 32x32 blocks
        W_blocks = self.weight_master.view(self.num_h, self.block_h, self.num_w, self.block_w).permute(0, 2, 1, 3).reshape(self.num_blocks, self.block_h, self.block_w)

        # 1. STE Binning: Force master weights into discrete bins
        if self.format_type == "fp4":
            W_binned_blocks = ste_fp4_quantize(W_blocks, self.block_elements)
        else:
            W_binned_blocks = ste_fp8_quantize(W_blocks, self.block_elements)

        W_binned_flat = W_binned_blocks.reshape(self.num_blocks, self.block_elements)

        # 2. Non-Linear Neural Feature Extraction: h ∈ R^(num_blocks x hidden_dim)
        h = self.feature_extractor(W_binned_flat)

        # 3. Non-Linear Neural Rehydration
        non_linear_delta = self.refinement_head(h)
        W_rehydrated_flat = W_binned_flat + self.gamma * non_linear_delta
        W_rehydrated_blocks = W_rehydrated_flat.reshape(self.num_blocks, self.block_h, self.block_w)

        # Reconstruct full weight tensor shape (out_features, in_features)
        W_rehydrated = W_rehydrated_blocks.view(self.num_h, self.num_w, self.block_h, self.block_w).permute(0, 2, 1, 3).reshape(self.out_features, self.in_features)
        return W_rehydrated


class STENeuralRehydrationLinear(nn.Module):
    """Linear layer wrapper applying STE Binning + Non-Linear Neural Rehydration."""
    def __init__(self, in_features: int, out_features: int, format_type: str = "fp4"):
        super().__init__()
        self.quantizer = STENonLinearCodebookRehydrator(out_features, in_features, format_type=format_type)

    def forward(self, x: torch.Tensor, hard: bool = False) -> torch.Tensor:
        w_rehydrated = self.quantizer(hard=hard)
        return F.linear(x, w_rehydrated)


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
    y_target = F.gelu(x_in @ W_target) @ R_target
    return x_in, y_target


def evaluate_predictions(y_ref: torch.Tensor, y_var: torch.Tensor, y_gt: torch.Tensor):
    pred_ref = y_ref.detach().float()
    pred_var = y_var.detach().float()
    gt = y_gt.detach().float()

    cos_sims = F.cosine_similarity(pred_ref, pred_var, dim=1)
    mag_ref = torch.norm(pred_ref, p=2, dim=1)
    mag_var = torch.norm(pred_var, p=2, dim=1)

    worst_idx = torch.argmin(cos_sims).item()
    sorted_idxs = torch.argsort(cos_sims)
    median_idx = sorted_idxs[len(sorted_idxs) // 2].item()

    return {
        "test_mse": F.mse_loss(pred_var, gt).item(),
        "mean_cos_sim": torch.mean(cos_sims).item(),
        "worst_cos_sim": cos_sims[worst_idx].item(),
        "worst_ref_mag": mag_ref[worst_idx].item(),
        "worst_var_mag": mag_var[worst_idx].item(),
        "median_cos_sim": cos_sims[median_idx].item(),
        "median_ref_mag": mag_ref[median_idx].item(),
        "median_var_mag": mag_var[median_idx].item(),
    }
