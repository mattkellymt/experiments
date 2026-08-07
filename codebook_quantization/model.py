import torch
import torch.nn as nn

torch.manual_seed(42)

class PureIndexCodebookQuantizer(nn.Module):
    """
    Softmax Codebook Mixture Quantization (SCMQ) / Single Codebook Neural Rehydration:
    - Weight matrix W is divided into 32x32 blocks (1024 params / block).
    - Single Codebook (K=1024 x 32 x 32): Acts as BOTH the prototype selector basis AND basis expansion output.
    - Softmax with temperature annealing (tau: 1.0 -> 0.05) sharpens continuous mixture into discrete Argmax.
    - At inference time: W_hat_block = S_block * Codebook[k]
    - Stored on disk: ONLY 10-bit integer index k per block (0.97 bits/param) + 1 single shared Codebook!
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

        # SINGLE CODEBOOK TENSOR: 1024 x 32 x 32 FP32 Basis Grids
        self.codebook = nn.Parameter(torch.randn(k_codes, block_h, block_w) * 0.1)
        # Annealable Softmax Temperature
        self.tau = 1.0

    def forward(self, W: torch.Tensor, hard: bool = False) -> torch.Tensor:
        # Reshape W into 32x32 blocks: (num_blocks, 32, 32)
        W_blocks = W.view(self.num_h, self.block_h, self.num_w, self.block_w).permute(0, 2, 1, 3).reshape(self.num_blocks, self.block_h, self.block_w)

        # Per-block norm scale factor (preserves magnitude)
        block_scales = torch.norm(W_blocks, p=2, dim=(-2, -1), keepdim=True) / 5.0
        W_norm = W_blocks / torch.clamp(block_scales, min=1e-6)

        # Fast GEMM similarity matching with Single Codebook: (num_blocks, K)
        W_norm_flat = W_norm.reshape(self.num_blocks, self.block_elements)
        cb_flat = self.codebook.reshape(self.codebook.shape[0], self.block_elements)
        sim = (W_norm_flat @ cb_flat.T) / max(self.tau, 0.05)

        if hard:
            # Inference Time: Discrete 10-bit integer index lookup into SINGLE codebook
            best_idx = torch.argmax(sim, dim=1)
            W_rehydrated_flat = block_scales.squeeze(-1) * cb_flat[best_idx]
        else:
            # Training Time: Differentiable Softmax mixture
            alpha = torch.softmax(sim, dim=1)
            W_rehydrated_flat = block_scales.squeeze(-1) * (alpha @ cb_flat)

        W_rehydrated_blocks = W_rehydrated_flat.reshape(self.num_blocks, self.block_h, self.block_w)

        # Reconstruct full weight tensor (out_features, in_features)
        W_rehydrated = W_rehydrated_blocks.view(self.num_h, self.num_w, self.block_h, self.block_w).permute(0, 2, 1, 3).reshape(self.out_features, self.in_features)
        return W_rehydrated


class CodebookRehydrationLinear(nn.Module):
    """Linear layer wrapper applying Single Codebook Neural Rehydration."""
    def __init__(self, in_features: int, out_features: int, k_codes: int = 1024):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.05)
        self.quantizer = PureIndexCodebookQuantizer(out_features, in_features, k_codes=k_codes)

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
