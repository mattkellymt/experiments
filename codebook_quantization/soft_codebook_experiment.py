import os
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

def quantize_to_fp4(tensor: torch.Tensor) -> torch.Tensor:
    """Simulates 4-bit FP4 E2M1 quantization with per-channel scaling."""
    w = tensor.detach().clone()
    orig_shape = w.shape
    fp4_grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=w.device)

    w_flat = w.view(-1, 16)
    scale = torch.clamp(torch.max(torch.abs(w_flat), dim=1, keepdim=True).values / 6.0, min=1e-12)
    w_scaled = w_flat / scale
    w_sign = torch.sign(w_scaled)
    w_abs = torch.abs(w_scaled)

    diffs = torch.abs(w_abs.unsqueeze(-1) - fp4_grid)
    indices = torch.argmin(diffs, dim=-1)
    w_q = fp4_grid[indices] * w_sign * scale
    return w_q.view_as(tensor)


def quantize_to_fp8(tensor: torch.Tensor) -> torch.Tensor:
    """Simulates 8-bit FP8 E4M3 quantization."""
    scale = torch.clamp(torch.max(torch.abs(tensor)) / 448.0, min=1e-12)
    q = torch.clamp(torch.round(tensor / scale), -448.0, 448.0)
    return q * scale


class SoftCodebookMatrices(nn.Module):
    """
    Dual-codebook soft routing for linear maps (single value codebook).

    For each task t:
        scores = K @ Q_t          # (M, d, d)
        W      = softmax_m(scores)
        R_t    = sum_m W_m ⊙ V_m  # predicted d×d matrix
    """
    def __init__(self, num_tasks: int, dim: int = 4, codebook_size: int = 16):
        super().__init__()
        self.T = num_tasks
        self.d = dim
        self.M = codebook_size

        # Task-specific routing matrices (T, d, d)
        self.Q = nn.Parameter(torch.randn(num_tasks, dim, dim) * 0.05)

        # Shared key codebook (M, d, d)
        self.key = nn.Parameter(torch.randn(codebook_size, dim, dim) * 0.05)

        # Shared value codebook (M, d, d) — single value book
        self.value = nn.Parameter(torch.randn(codebook_size, dim, dim) * 0.05)

    def forward(self, task_ids=None, precision: str = "fp32"):
        """
        Returns predicted matrices R of shape (T, d, d) or (len(task_ids), d, d).
        """
        Q = self.Q if task_ids is None else self.Q[task_ids]  # (B, d, d)

        # Simulate low-precision quantization of Query matrix Q before FP32 matmul
        if precision == "fp16":
            Q = Q.half().float()
        elif precision == "fp8":
            Q = quantize_to_fp8(Q)
        elif precision == "fp4":
            Q = quantize_to_fp4(Q)

        # scores[b, m, i, j] = sum_k key[m, i, k] * Q[b, k, j]
        scores = torch.einsum('mik,bkj->bmij', self.key, Q)   # (B, M, d, d)

        W = F.softmax(scores, dim=1)                          # soft assignment over M

        # R[b, i, j] = sum_m W[b, m, i, j] * value[m, i, j]
        R = torch.einsum('bmij,mij->bij', W, self.value)      # (B, d, d)
        return R


def random_rotation(d):
    A = torch.randn(d, d)
    Q, R = torch.linalg.qr(A)
    Q = Q @ torch.diag(torch.sign(torch.diag(R)))
    if torch.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def run_experiment_precision(precision: str = "fp32", steps: int = 3000):
    d, M, T = 4, 16, 64
    torch.manual_seed(42)

    model = SoftCodebookMatrices(num_tasks=T, dim=d, codebook_size=M).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    R_true = torch.stack([random_rotation(d) for _ in range(T)]).to(device)  # (T, d, d)

    print(f"\n--- Running Experiment: Precision = {precision.upper()} ---")
    for step in range(steps):
        R_pred = model(precision=precision)                    # (T, d, d)
        x = torch.randn(T, 32, d, device=device)               # probe vectors

        # Simulate precision conversion on probe vectors
        if precision == "fp16":
            x_probe = x.half().float()
        elif precision == "fp8":
            x_probe = quantize_to_fp8(x)
        elif precision == "fp4":
            x_probe = quantize_to_fp4(x)
        else:
            x_probe = x

        y_pred = torch.einsum('tij,tbj->tbi', R_pred, x_probe)
        y_true = torch.einsum('tij,tbj->tbi', R_true, x)
        loss = ((y_pred - y_true) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 1000 == 0 or step == steps - 1:
            print(f"  Step {step:4d} | MSE: {loss.item():.4e}")

    # Evaluate Cosine Similarity between R_true and R_pred
    with torch.no_grad():
        R_pred_final = model(precision=precision)
        cos_sim = F.cosine_similarity(R_true.view(T, -1), R_pred_final.view(T, -1), dim=1).mean().item()

    return loss.item(), cos_sim


def main():
    print(f"Soft Codebook Matrices Task Routing Device: {device}")
    precisions = ["fp32", "fp16", "fp8", "fp4"]
    results = {}

    for p in precisions:
        mse, cos = run_experiment_precision(precision=p, steps=3000)
        results[p] = (mse, cos)

    print("\n" + "=" * 80)
    print("SOFT CODEBOOK MATRIX ROUTING BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Precision':<12} | {'Final Probe MSE':<20} | {'Rehydrated Cosine Similarity':<30}")
    print("-" * 80)
    for p, (mse, cos) in results.items():
        print(f"{p.upper():<12} | {mse:<20.4e} | {cos:<30.6f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
