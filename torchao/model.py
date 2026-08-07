import torch
import torch.nn as nn

# Ensure deterministic execution
torch.manual_seed(42)

class RotationModel(nn.Module):
    """
    A 4-layer MLP designed to learn a high-dimensional non-linear transformation / rotation mapping.
    Architecture: 256 -> 1024 -> 1024 -> 1024 -> 256 (~2.62 Million parameters).
    """
    def __init__(self, dim: int = 256, hidden_dim: int = 1024):
        super().__init__()
        self.dim = dim
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
        out = self.fc4(h)
        return out


def generate_dataset(num_samples: int = 1024, dim: int = 256, seed: int = 42):
    """
    Generates synthetic target mapping data based on a fixed random orthonormal target matrix R.
    y = GELU(x @ W_target) @ R_target
    """
    g = torch.Generator().manual_seed(seed)
    # Fixed random target transformation matrices
    W_target = torch.randn(dim, dim, generator=g, dtype=torch.float32) * 0.1
    U, _, V = torch.linalg.svd(torch.randn(dim, dim, generator=g, dtype=torch.float32))
    R_target = U @ V.T

    x_in = torch.randn(num_samples, dim, generator=g, dtype=torch.float32)
    y_target = torch.nn.functional.gelu(x_in @ W_target) @ R_target

    return x_in, y_target


def evaluate_predictions(y_pred_fp32: torch.Tensor, y_pred_variant: torch.Tensor, y_ground_truth: torch.Tensor):
    """
    Calculates detailed per-sample comparison metrics between FP32 reference predictions and variant predictions.
    Focuses on Cosine Similarity decomposition alongside Reference & Variant Vector Magnitudes.
    """
    # Cast predictions to float32 for metric computation
    pred_ref = y_pred_fp32.detach().to(torch.float32)
    pred_var = y_pred_variant.detach().to(torch.float32)
    gt = y_ground_truth.detach().to(torch.float32)

    test_mse = torch.nn.functional.mse_loss(pred_var, gt).item()
    diff_mse = torch.nn.functional.mse_loss(pred_var, pred_ref).item()

    # Per-sample metrics (across dimension 1)
    cos_sims = torch.nn.functional.cosine_similarity(pred_ref, pred_var, dim=1) # (num_samples,)
    mag_ref = torch.norm(pred_ref, p=2, dim=1)                                # (num_samples,)
    mag_var = torch.norm(pred_var, p=2, dim=1)                                # (num_samples,)

    # 1. Worst Cosine Similarity Case (Sample with minimum cosine similarity)
    worst_idx = torch.argmin(cos_sims).item()
    worst_cos = cos_sims[worst_idx].item()
    worst_ref_mag = mag_ref[worst_idx].item()
    worst_var_mag = mag_var[worst_idx].item()

    # 2. Median Cosine Similarity Case (Sample at median cosine similarity)
    sorted_indices = torch.argsort(cos_sims)
    median_idx = sorted_indices[len(sorted_indices) // 2].item()
    median_cos = cos_sims[median_idx].item()
    median_ref_mag = mag_ref[median_idx].item()
    median_var_mag = mag_var[median_idx].item()

    # Mean Cosine Similarity across dataset
    mean_cos = torch.mean(cos_sims).item()

    # 3. Worst Dot Product Error Case (Sample with max |y_ref . y_ref - y_ref . y_var|)
    dot_ref_ref = torch.sum(pred_ref * pred_ref, dim=1)
    dot_ref_var = torch.sum(pred_ref * pred_var, dim=1)
    dot_diff = torch.abs(dot_ref_ref - dot_ref_var)
    worst_dot_idx = torch.argmax(dot_diff).item()
    worst_dot_cos = cos_sims[worst_dot_idx].item()
    worst_dot_ref_mag = mag_ref[worst_dot_idx].item()
    worst_dot_var_mag = mag_var[worst_dot_idx].item()

    return {
        "test_mse": test_mse,
        "diff_mse": diff_mse,
        "mean_cos_sim": mean_cos,
        # Worst Cosine Sim Case
        "worst_cos_sim": worst_cos,
        "worst_ref_mag": worst_ref_mag,
        "worst_var_mag": worst_var_mag,
        # Median Cosine Sim Case
        "median_cos_sim": median_cos,
        "median_ref_mag": median_ref_mag,
        "median_var_mag": median_var_mag,
        # Worst Dot Product Error Case
        "worst_dot_cos_sim": worst_dot_cos,
        "worst_dot_ref_mag": worst_dot_ref_mag,
        "worst_dot_var_mag": worst_dot_var_mag,
    }
