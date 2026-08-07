import torch

torch.manual_seed(42)

def generate_data(num_samples: int = 500):
    """
    Generates 2D input data with unequal variance across dimensions.
    x1 has high variance (5.0), x2 has low variance (0.5).
    This creates an elongated 'elliptical' loss bowl with strong Hessian curvature contrast.
    """
    g = torch.Generator().manual_seed(42)
    x1 = torch.randn(num_samples, 1, generator=g) * 2.24  # std dev sqrt(5.0)
    x2 = torch.randn(num_samples, 1, generator=g) * 0.71  # std dev sqrt(0.5)
    X = torch.cat([x1, x2], dim=1)

    w_true = torch.tensor([2.0, -3.0])
    noise = torch.randn(num_samples, generator=g) * 0.1
    y = X @ w_true + noise
    return X, y


def compute_loss_and_hessian(X: torch.Tensor, y: torch.Tensor, w: torch.Tensor):
    """
    Computes Loss, Gradient vector, and the 2x2 Hessian matrix for linear regression.
    Loss: L(w) = 1/(2N) * ||X w - y||^2
    Gradient: ∇L = 1/N * X^T (X w - y)
    Hessian: H = 1/N * X^T X
    """
    N = X.shape[0]
    preds = X @ w
    loss = (0.5 / N) * torch.sum((preds - y) ** 2).item()
    gradient = (1.0 / N) * X.T @ (preds - y)
    hessian = (1.0 / N) * (X.T @ X)

    # Eigen-decomposition of the Hessian
    eigenvalues, eigenvectors = torch.linalg.eigh(hessian)

    return {
        "loss": loss,
        "gradient": gradient,
        "hessian": hessian,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
    }


def main():
    X, y = generate_data(num_samples=500)
    w_initial = torch.tensor([-1.0, 1.0])

    stats = compute_loss_and_hessian(X, y, w_initial)

    print("=" * 60)
    print("HESSIAN COMPUTATION DEMO (2D Weights -> 3D Loss Bowl)")
    print("=" * 60)
    print(f"Initial Weights: {w_initial.tolist()}")
    print(f"Initial Loss:    {stats['loss']:.4f}")
    print(f"Gradient Vector: {stats['gradient'].tolist()}")
    print("\nHessian Matrix (H = 1/N * X^T X):")
    print(stats['hessian'])
    print(f"\nEigenvalues (Curvatures λ1, λ2): {stats['eigenvalues'].tolist()}")
    print("Eigenvectors (Principal Curvature Axes):")
    print(stats['eigenvectors'])
    print("=" * 60)

if __name__ == "__main__":
    main()
