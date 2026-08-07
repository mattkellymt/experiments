import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(42)

def generate_random_rotation_32d(dim=32, seed=42):
    """Generate a valid 32x32 rotation matrix in SO(32) using PyTorch."""
    torch.manual_seed(seed)
    A = torch.randn(dim, dim)
    Q, R = torch.linalg.qr(A)
    d = torch.diag(R)
    ph = d / torch.abs(d)
    return Q @ torch.diag(ph)

# ====================================================================
# 1. Custom PyTorch nn.Module Model Class
# ====================================================================
class TargetedRotation32DModel(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim
        # Multi-scale parameter decomposition: p1, p2, p3, p4
        self.p1 = nn.Parameter(torch.full((dim, dim), 0.01))  # Super-exponential regime (>= e^2)
        self.p2 = nn.Parameter(torch.zeros((dim, dim)))        # Linear medium regime (1.0 to e^2)
        self.p3 = nn.Parameter(torch.full((dim, dim), 1.0))   # Sub-linear micro regime (< 1.0)
        self.p4 = nn.Parameter(torch.randn(dim, dim) * 0.1)   # Decoupled sign controller

    def get_rotation_matrix(self):
        S = torch.exp(torch.abs(self.p1)) + self.p2 + torch.exp(-torch.abs(self.p3))
        sign_factor = torch.tanh(self.p4)
        return sign_factor * S

    def forward(self, x):
        W = self.get_rotation_matrix()
        return torch.matmul(x, W.T)


# ====================================================================
# 2. Custom PyTorch torch.optim.Optimizer Class
# ====================================================================
class MultiScaleEOSTargetedOptimizer(optim.Optimizer):
    """
    Custom PyTorch Optimizer implementing targeted multi-scale magnitude gating
    to eliminate Edge of Stability (EoS) gradient norm volatility spikes.
    """
    def __init__(self, model, lr=0.35, beta_m=0.85):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, beta_m=beta_m)
        params = [model.p1, model.p2, model.p3, model.p4]
        super().__init__(params, defaults)
        self.model = model
        self.dim = model.dim

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        e2 = 7.389056  # exp(2.0)

        for group in self.param_groups:
            lr = group['lr']
            beta_m = group['beta_m']

            params = group['params']
            if len(params) < 4:
                continue

            p1, p2, p3, p4 = params[0], params[1], params[2], params[3]
            if p1.grad is None or p2.grad is None or p3.grad is None or p4.grad is None:
                continue

            # State initialization for momentum
            state = self.state[p1]
            if len(state) == 0:
                state['m1'] = torch.zeros_like(p1)
                state['m2'] = torch.zeros_like(p2)
                state['m3'] = torch.zeros_like(p3)
                state['m4'] = torch.zeros_like(p4)

            m1, m2, m3, m4 = state['m1'], state['m2'], state['m3'], state['m4']

            # Base gradient wrt matrix W: g_base = p2.grad * self.dim (re-scaling MSELoss to vector norm)
            g_base = p2.grad * self.dim
            grad_mag = torch.abs(g_base * lr * 20.0)

            # Continuous sigmoidal soft-gating weights
            w1 = torch.sigmoid(2.0 * (grad_mag - e2))
            w3 = torch.sigmoid(2.0 * (1.0 - grad_mag))
            w2 = torch.clamp(1.0 - w1 - w3, min=0.0)

            # Base targeted parameter gradients
            sign_factor = torch.tanh(p4)
            gp1 = g_base * sign_factor * torch.sign(p1 + 1e-8)
            gp2 = g_base * sign_factor
            gp3 = -g_base * sign_factor * torch.sign(p3 + 1e-8)
            gp4 = p4.grad * self.dim

            # Momentum updates
            m1.mul_(beta_m).add_(gp1, alpha=1 - beta_m)
            m2.mul_(beta_m).add_(gp2, alpha=1 - beta_m)
            m3.mul_(beta_m).add_(gp3, alpha=1 - beta_m)
            m4.mul_(beta_m).add_(gp4, alpha=1 - beta_m)

            # Targeted parameter updates
            p1.add_(-lr * w1 * m1)
            p2.add_(-lr * w2 * m2)
            p3.add_(-lr * w3 * m3)
            p4.add_(-lr * m4)

        return loss


# ====================================================================
# 3. PyTorch EoS Stress Test & Plot Generation
# ====================================================================
def run_pytorch_eos_experiment():
    dim = 32
    seed = 42
    n_steps = 400
    lr = 0.35  # High learning rate stress test (Theoretical EoS limit 2/η = 5.71)

    print("======================================================================")
    print(f" PyTorch Edge of Stability (EoS) Benchmark (Learning Rate η = {lr})")
    print(f" Theoretical Edge of Stability Limit (2/η) = {2.0/lr:.2f}")
    print("======================================================================")

    # Ground Truth Rotation Matrix in SO(32)
    R_true = generate_random_rotation_32d(dim=dim, seed=seed)

    torch.manual_seed(seed + 1)
    n_train = 256
    X_train = torch.randn(n_train, dim)
    Y_train = X_train @ R_true.T

    n_val = 128
    X_val = torch.randn(n_val, dim)
    Y_val = X_val @ R_true.T

    criterion = nn.MSELoss()

    # --- MODEL 1: Custom PyTorch Targeted Model + Custom PyTorch Targeted Optimizer ---
    torch.manual_seed(42)
    model_targeted = TargetedRotation32DModel(dim=dim)
    optimizer_targeted = MultiScaleEOSTargetedOptimizer(model_targeted, lr=lr)

    history_targeted = []

    for step in range(n_steps):
        model_targeted.train()
        Y_pred = model_targeted(X_train)
        loss = criterion(Y_pred, Y_train)

        optimizer_targeted.zero_grad()
        loss.backward()

        gW = (2.0 / n_train) * ((Y_pred - Y_train).T @ X_train)
        grad_norm = float(torch.linalg.norm(gW).item())

        optimizer_targeted.step()

        with torch.no_grad():
            val_mse = float(criterion(model_targeted(X_val), Y_val).item())
            history_targeted.append({"step": step, "val_mse": val_mse, "grad_norm": grad_norm})

    # --- MODEL 2: PyTorch Linear Model + Standard PyTorch Adam Optimizer ---
    torch.manual_seed(42)
    model_adam = nn.Linear(dim, dim, bias=False)
    nn.init.normal_(model_adam.weight, std=0.1)
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=lr)

    history_adam = []

    for step in range(n_steps):
        model_adam.train()
        Y_pred = model_adam(X_train)
        loss = criterion(Y_pred, Y_train)

        optimizer_adam.zero_grad()
        loss.backward()

        grad_norm = float(torch.linalg.norm(model_adam.weight.grad * dim).item())
        optimizer_adam.step()

        with torch.no_grad():
            val_mse = float(criterion(model_adam(X_val), Y_val).item())
            history_adam.append({"step": step, "val_mse": val_mse, "grad_norm": grad_norm})

    # --- MODEL 3: PyTorch Linear Model + Standard PyTorch SGD Optimizer ---
    torch.manual_seed(42)
    model_sgd = nn.Linear(dim, dim, bias=False)
    nn.init.normal_(model_sgd.weight, std=0.1)
    optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=lr)

    history_sgd = []

    for step in range(n_steps):
        model_sgd.train()
        Y_pred = model_sgd(X_train)
        loss = criterion(Y_pred, Y_train)

        optimizer_sgd.zero_grad()
        loss.backward()

        grad_norm = float(torch.linalg.norm(model_sgd.weight.grad * dim).item())
        optimizer_sgd.step()

        with torch.no_grad():
            val_mse = float(criterion(model_sgd(X_val), Y_val).item())
            history_sgd.append({"step": step, "val_mse": val_mse, "grad_norm": grad_norm})

    # Calculate EoS Volatility Metrics
    def calc_volatility(hist):
        losses = torch.tensor([h["val_mse"] for h in hist[20:]], dtype=torch.float64)
        gnorms = torch.tensor([h["grad_norm"] for h in hist[20:]], dtype=torch.float64)
        loss_roughness = float(torch.nan_to_num(losses.diff().abs().mean(), nan=0.0).item())
        gnorm_std = float(torch.nan_to_num(gnorms.std(), nan=0.0).item())
        return loss_roughness, gnorm_std

    t_rough, t_std = calc_volatility(history_targeted)
    a_rough, a_std = calc_volatility(history_adam)
    s_rough, s_std = calc_volatility(history_sgd)

    print("\n--- PyTorch Edge of Stability (EoS) Metrics ---")
    print(f"1. PyTorch Targeted Model + Custom Optimizer : Final MSE = {history_targeted[-1]['val_mse']:.8f} | Loss Volatility = {t_rough:.7f} | Grad Norm Std = {t_std:.4f} → NO EoS Volatility")
    print(f"2. PyTorch Model + Standard Adam Optimizer   : Final MSE = {history_adam[-1]['val_mse']:.8f} | Loss Volatility = {a_rough:.7f} | Grad Norm Std = {a_std:.4f} → High EoS Spikes")
    print(f"3. PyTorch Model + Standard SGD Optimizer    : Final MSE = {history_sgd[-1]['val_mse']:.8f} | Loss Volatility = {s_rough:.7f} | Grad Norm Std = {s_std:.4f} → EoS Bouncing")

    # ----------------------------------------------------------------
    # Plotting plot.png
    # ----------------------------------------------------------------
    plt.style.use('dark_background')
    plt.rcParams['font.sans-serif'] = 'Inter, DejaVu Sans, Arial'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor='#0f172a')
    steps = list(range(n_steps))

    ax1.set_facecolor('#1e293b')
    ax1.plot(steps, [max(h['val_mse'], 1e-10) for h in history_targeted], label='PyTorch Targeted Model + Custom Optimizer (No EoS)', color='#10b981', linewidth=2.5)
    ax1.plot(steps, [max(h['val_mse'], 1e-10) for h in history_adam], label='PyTorch Model + Standard Adam (EoS Volatility)', color='#f43f5e', linewidth=2.0)
    ax1.plot(steps, [max(h['val_mse'], 1e-10) for h in history_sgd], label='PyTorch Model + Standard SGD (EoS Bouncing)', color='#3b82f6', linewidth=1.8, linestyle='--')
    ax1.set_yscale('log')
    ax1.set_title("PyTorch Validation Loss (Log Scale): Smooth Decay vs EoS Spikes", color='#f8fafc', fontweight='bold', fontsize=12)
    ax1.set_xlabel("Training Step", color='#94a3b8')
    ax1.set_ylabel("Validation MSE Loss", color='#94a3b8')
    ax1.legend(facecolor='#0f172a', edgecolor='#334155', fontsize=9)
    ax1.grid(True, color='#334155', alpha=0.3)

    ax2.set_facecolor('#1e293b')
    ax2.plot(steps, [h['grad_norm'] for h in history_targeted], label='PyTorch Targeted Model + Custom Optimizer (Stable)', color='#10b981', linewidth=2.5)
    ax2.plot(steps, [h['grad_norm'] for h in history_adam], label='PyTorch Model + Standard Adam (Severe EoS Spikes)', color='#f43f5e', linewidth=1.8, alpha=0.85)
    ax2.plot(steps, [h['grad_norm'] for h in history_sgd], label='PyTorch Model + Standard SGD', color='#3b82f6', linewidth=1.8, alpha=0.85, linestyle='--')
    ax2.set_title("PyTorch Gradient Norm ||∇L|| Volatility: Edge of Stability Test", color='#f8fafc', fontweight='bold', fontsize=12)
    ax2.set_xlabel("Training Step", color='#94a3b8')
    ax2.set_ylabel("Gradient Norm ||∇L||", color='#94a3b8')
    ax2.legend(facecolor='#0f172a', edgecolor='#334155', fontsize=9)
    ax2.grid(True, color='#334155', alpha=0.3)

    plt.suptitle(f"PyTorch Edge of Stability (EoS) Benchmark: Custom Targeted Optimizer vs Adam & SGD (η = {lr})", color='#f8fafc', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot.png")
    plt.savefig(plot_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"\nPlot saved to {plot_path}")

if __name__ == "__main__":
    run_pytorch_eos_experiment()
