import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

def quantize_fp4_block(weight_blocks: torch.Tensor, block_elements: int = 1024) -> torch.Tensor:
    """Quantizes flat weight blocks into 4-bit FP4 micro-scaled grid values."""
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


def quantize_fp8_block(weight_blocks: torch.Tensor, block_elements: int = 1024) -> torch.Tensor:
    """Quantizes flat weight blocks into 8-bit FP8 (E4M3) micro-scaled grid values."""
    w = weight_blocks.detach().clone()
    w_flat = w.view(-1, block_elements)
    scale = torch.clamp(torch.max(torch.abs(w_flat), dim=1, keepdim=True).values / 448.0, min=1e-12)
    w_scaled = w_flat / scale
    w_q = torch.clamp(torch.round(w_scaled), -448.0, 448.0)
    return (w_q * scale).view_as(weight_blocks)


# -----------------------------------------------------------------------------
# METHOD 1: Key-Value Codebook Router (KVCodebookRouter)
# -----------------------------------------------------------------------------
class KVCodebookRouter(nn.Module):
    """
    Method 1: Key-Value Codebook Router
    - Uses quantized block W_q as a key query into a Key Encoder E_phi.
    - Computes Softmax similarity scores against a Key Codebook K.
    - Rehydrates high-fidelity FP32 weights from a Value Codebook V.
    """
    def __init__(self, block_elements: int = 1024, k_codes: int = 256, key_dim: int = 128):
        super().__init__()
        self.block_elements = block_elements
        self.key_encoder = nn.Sequential(
            nn.Linear(block_elements, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, key_dim)
        )
        self.key_codebook = nn.Parameter(torch.randn(k_codes, key_dim) * 0.1)
        self.value_codebook = nn.Parameter(torch.randn(k_codes, block_elements) * 0.1)
        self.scale_head = nn.Linear(key_dim, 1)

    def forward(self, W_q_flat: torch.Tensor) -> torch.Tensor:
        scales = torch.norm(W_q_flat, p=2, dim=1, keepdim=True)
        W_norm = W_q_flat / torch.clamp(scales, min=1e-6)

        keys = self.key_encoder(W_norm)
        sim = keys @ self.key_codebook.T
        alpha = F.softmax(sim, dim=1)
        scale_pred = F.softplus(self.scale_head(keys))

        W_rehydrated = scale_pred * (alpha @ self.value_codebook)
        return W_rehydrated


# -----------------------------------------------------------------------------
# METHOD 2: Multi-Head Quantized Key Attention Rehydrator (MHKeyAttentionRehydrator)
# -----------------------------------------------------------------------------
class MHKeyAttentionRehydrator(nn.Module):
    """
    Method 2: Multi-Head Quantized Key Attention Rehydrator
    - Partitions quantized block W_q into H=4 heads.
    - Each head acts as a key to attend over H independent codebook memories.
    - Combines multi-head basis outputs into a composite FP32 weight matrix.
    """
    def __init__(self, block_elements: int = 1024, num_heads: int = 4, k_codes: int = 128):
        super().__init__()
        self.block_elements = block_elements
        self.num_heads = num_heads
        self.head_dim = block_elements // num_heads

        self.key_codebook = nn.Parameter(torch.randn(num_heads, k_codes, self.head_dim) * 0.1)
        self.value_codebook = nn.Parameter(torch.randn(num_heads, k_codes, self.head_dim) * 0.1)

    def forward(self, W_q_flat: torch.Tensor) -> torch.Tensor:
        scales = torch.norm(W_q_flat, p=2, dim=1, keepdim=True)
        W_norm = W_q_flat / torch.clamp(scales, min=1e-6)

        w_heads = W_norm.view(-1, self.num_heads, self.head_dim)
        sim = torch.einsum('bhd,hkd->bhk', w_heads, self.key_codebook)
        alpha = F.softmax(sim, dim=-1)

        v_out = torch.einsum('bhk,hkd->bhd', alpha, self.value_codebook)
        W_rehydrated = scales * v_out.reshape(-1, self.block_elements)
        return W_rehydrated


# -----------------------------------------------------------------------------
# METHOD 3: Deep Non-Linear Key Projection Network (DeepKeyProjectionNetwork)
# -----------------------------------------------------------------------------
class DeepKeyProjectionNetwork(nn.Module):
    """
    Method 3: Deep Non-Linear Key Projection Network
    - Treats quantized block W_q as a key vector passed through a deep non-linear MLP.
    - Maps noisy low-precision key patterns directly into continuous FP32 basis matrices.
    """
    def __init__(self, block_elements: int = 1024, hidden_dim: int = 512):
        super().__init__()
        self.block_elements = block_elements
        self.key_projector = nn.Sequential(
            nn.Linear(block_elements, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, block_elements)
        )

    def forward(self, W_q_flat: torch.Tensor) -> torch.Tensor:
        scales = torch.norm(W_q_flat, p=2, dim=1, keepdim=True)
        W_norm = W_q_flat / torch.clamp(scales, min=1e-6)

        proj = self.key_projector(W_norm)
        W_rehydrated = scales * proj
        return W_rehydrated


# -----------------------------------------------------------------------------
# Master Linear Layer Wrapper Supporting All 3 Quantized Key Transform Methods
# -----------------------------------------------------------------------------
class QuantizedKeyTransformLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, method_type: str = "method1_kv_router", format_type: str = "fp4", block_h: int = 32, block_w: int = 32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.method_type = method_type.lower()
        self.format_type = format_type.lower()
        self.block_h = block_h
        self.block_w = block_w
        self.block_elements = block_h * block_w

        self.num_h = out_features // block_h
        self.num_w = in_features // block_w
        self.num_blocks = self.num_h * self.num_w

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.05)

        if "method1" in self.method_type:
            self.transform_engine = KVCodebookRouter(block_elements=self.block_elements)
        elif "method2" in self.method_type:
            self.transform_engine = MHKeyAttentionRehydrator(block_elements=self.block_elements)
        else:
            self.transform_engine = DeepKeyProjectionNetwork(block_elements=self.block_elements)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_blocks = self.weight.view(self.num_h, self.block_h, self.num_w, self.block_w).permute(0, 2, 1, 3).reshape(self.num_blocks, self.block_h, self.block_w)

        if self.format_type == "fp4":
            W_q_blocks = quantize_fp4_block(W_blocks, block_elements=self.block_elements)
        else:
            W_q_blocks = quantize_fp8_block(W_blocks, block_elements=self.block_elements)

        W_q_flat = W_q_blocks.reshape(self.num_blocks, self.block_elements)
        W_rehydrated_flat = self.transform_engine(W_q_flat)
        W_rehydrated_blocks = W_rehydrated_flat.reshape(self.num_blocks, self.block_h, self.block_w)

        W_rehydrated = W_rehydrated_blocks.view(self.num_h, self.num_w, self.block_h, self.block_w).permute(0, 2, 1, 3).reshape(self.out_features, self.in_features)
        return F.linear(x, W_rehydrated)


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
