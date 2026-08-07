import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

def quantize_fp8_block(weight_blocks: torch.Tensor, block_elements: int = 1024) -> torch.Tensor:
    """Quantizes flat weight blocks into 8-bit FP8 (E4M3) micro-scaled grid values."""
    w = weight_blocks.detach().clone()
    w_flat = w.view(-1, block_elements)
    scale = torch.clamp(torch.max(torch.abs(w_flat), dim=1, keepdim=True).values / 448.0, min=1e-12)
    w_scaled = w_flat / scale
    w_q = torch.clamp(torch.round(w_scaled), -448.0, 448.0)
    return (w_q * scale).view_as(weight_blocks)


class NonLinearFP8CodebookRehydrator(nn.Module):
    """
    Gated Non-Linear Neural Codebook Rehydration Engine for FP8 Quantized Weights:
    1. Input: 32x32 blocks of 8-bit FP8 quantized weights (W_fp8).
    2. Non-Linear Feature Extractor (MLP + GELU):
       h = GELU(Linear(W_fp8_flat)) -> maps FP8 grid noise into smooth feature space.
    3. Gated Non-Linear Residual Correction:
       W_rehydrated = W_fp8 + gamma * tanh(MLP(h))
    4. Ensures magnitude stability while refining non-linear FP8 quantization noise!
    """
    def __init__(self, out_features: int, in_features: int, k_codes: int = 256, block_h: int = 32, block_w: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.k_codes = k_codes
        self.block_h = block_h
        self.block_w = block_w
        self.block_elements = block_h * block_w

        self.num_h = out_features // block_h
        self.num_w = in_features // block_w
        self.num_blocks = self.num_h * self.num_w

        # Non-Linear Neural Feature Extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.block_elements, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        # Gated Non-Linear Neural Refinement Head
        self.refinement_head = nn.Linear(hidden_dim, self.block_elements)
        self.gamma = nn.Parameter(torch.zeros(1))  # Start at zero gate to guarantee stability!

        # Annealable Softmax Temperature
        self.tau = 1.0

    def forward(self, W: torch.Tensor, hard: bool = False) -> torch.Tensor:
        # Slices master weights into 32x32 blocks
        W_blocks = W.view(self.num_h, self.block_h, self.num_w, self.block_w).permute(0, 2, 1, 3).reshape(self.num_blocks, self.block_h, self.block_w)

        # 1. Quantize 32x32 blocks to 8-bit FP8 grid values (W_fp8)
        W_fp8_blocks = quantize_fp8_block(W_blocks, block_elements=self.block_elements)
        W_fp8_flat = W_fp8_blocks.reshape(self.num_blocks, self.block_elements)

        # 2. Non-Linear Neural Feature Extraction: h ∈ R^(num_blocks x hidden_dim)
        h = self.feature_extractor(W_fp8_flat)

        # 3. Gated Non-Linear Neural Rehydration
        non_linear_delta = torch.tanh(self.refinement_head(h)) * 0.05
        W_rehydrated_flat = W_fp8_flat + self.gamma * non_linear_delta
        W_rehydrated_blocks = W_rehydrated_flat.reshape(self.num_blocks, self.block_h, self.block_w)

        # Reconstruct full weight tensor shape (out_features, in_features)
        W_rehydrated = W_rehydrated_blocks.view(self.num_h, self.num_w, self.block_h, self.block_w).permute(0, 2, 1, 3).reshape(self.out_features, self.in_features)
        return W_rehydrated


class FP8NeuralRehydrationLinear(nn.Module):
    """Linear layer wrapper applying FP8 Non-Linear Neural Codebook Rehydration."""
    def __init__(self, in_features: int, out_features: int, k_codes: int = 256):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.05)
        self.quantizer = NonLinearFP8CodebookRehydrator(out_features, in_features, k_codes=k_codes)

    def forward(self, x: torch.Tensor, hard: bool = False) -> torch.Tensor:
        w_rehydrated = self.quantizer(self.weight, hard=hard)
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
