import torch
import torch.nn as nn

torch.manual_seed(42)

def native_fp4_quantize_block(weight_flat: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Quantizes flat weight blocks into 4-bit FP4 micro-scaled grid values."""
    w = weight_flat.detach().clone()
    fp4_grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=weight_flat.device)

    w_reshaped = w.view(-1, block_size)
    block_max = torch.max(torch.abs(w_reshaped), dim=1, keepdim=True).values
    scale = torch.clamp(block_max / 6.0, min=1e-12)
    w_scaled = w_reshaped / scale
    w_sign = torch.sign(w_scaled)
    w_abs = torch.abs(w_scaled)
    diffs = torch.abs(w_abs.unsqueeze(-1) - fp4_grid)
    indices = torch.argmin(diffs, dim=-1)
    w_q = fp4_grid[indices] * w_sign * scale
    return w_q.view_as(weight_flat)


class GridCodebookRehydrator(nn.Module):
    """
    4-Bit Spatial Grid + Dual-Codebook Neural Rehydration Engine:
    1. Base Weights: 32x32 blocks of 4-bit quantized values (W_fp4).
    2. Convert W_fp4 to FP32 and matmul with Codebook 1 (K x 32 x 32 prototypes).
    3. Softmax (fine-tuning) or Hard Argmax (inference) across K=1024 dimension produces alpha.
    4. Weight Codebook 2 (K x 32 x 32 full FP32 basis tensors) using alpha:
       W_rehydrated = alpha @ Codebook2
    5. Rehydrates 4-bit 32x32 spatial grids into full 32-bit FP32 weight matrices at inference!
    """
    def __init__(self, out_features: int, in_features: int, k_codes: int = 1024, block_h: int = 32, block_w: int = 32):
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

        # Codebook 1: 1024 x 32 x 32 FP32 Selector Basis
        self.codebook1 = nn.Parameter(torch.randn(k_codes, block_h, block_w) * 0.1)
        # Codebook 2: 1024 x 32 x 32 FP32 Basis Expansion Tensors
        self.codebook2 = nn.Parameter(torch.randn(k_codes, block_h, block_w) * 0.1)
        # Annealable Softmax Temperature
        self.tau = 1.0

    def forward(self, W: torch.Tensor, hard: bool = False) -> torch.Tensor:
        # Slices master weights into 32x32 blocks
        W_blocks = W.view(self.num_h, self.block_h, self.num_w, self.block_w).permute(0, 2, 1, 3).reshape(self.num_blocks, self.block_h, self.block_w)

        # 1. Quantize 32x32 blocks into 4-bit grid values (W_fp4)
        W_fp4_blocks = native_fp4_quantize_block(W_blocks, block_size=self.block_elements)

        # 2. Matmul 4-bit FP32-converted values with Codebook 1 (num_blocks, K)
        W_fp4_flat = W_fp4_blocks.reshape(self.num_blocks, self.block_elements)
        cb1_flat = self.codebook1.reshape(self.k_codes, self.block_elements)
        sim = (W_fp4_flat @ cb1_flat.T) / max(self.tau, 0.05)

        cb2_flat = self.codebook2.reshape(self.k_codes, self.block_elements)

        if hard:
            # Hard Argmax selection along 1024 dimension
            best_idx = torch.argmax(sim, dim=1)
            W_rehydrated_flat = cb2_flat[best_idx]
        else:
            # Softmax mixture during fine-tuning (differentiable)
            alpha = torch.softmax(sim, dim=1)
            W_rehydrated_flat = alpha @ cb2_flat

        W_rehydrated_blocks = W_rehydrated_flat.reshape(self.num_blocks, self.block_h, self.block_w)

        # Reconstruct full weight matrix shape (out_features, in_features)
        W_rehydrated = W_rehydrated_blocks.view(self.num_h, self.num_w, self.block_h, self.block_w).permute(0, 2, 1, 3).reshape(self.out_features, self.in_features)
        return W_rehydrated


class CodebookRehydrationLinear(nn.Module):
    """Linear layer wrapper applying 4-Bit Grid + Dual-Codebook Neural Rehydration."""
    def __init__(self, in_features: int, out_features: int, k_codes: int = 1024):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.05)
        self.quantizer = GridCodebookRehydrator(out_features, in_features, k_codes=k_codes)

    def forward(self, x: torch.Tensor, hard: bool = False) -> torch.Tensor:
        w_rehydrated = self.quantizer(self.weight, hard=hard)
        return torch.nn.functional.linear(x, w_rehydrated)


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
