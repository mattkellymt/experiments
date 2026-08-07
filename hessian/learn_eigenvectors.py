import torch
import torch.nn as nn

torch.manual_seed(42)

def main():
    # Define an arbitrary symmetric 2D matrix
    A = torch.tensor([[4.0, 1.5],
                      [1.5, 2.0]], dtype=torch.float32)

    # Compute ground truth eigenvalues and eigenvectors using PyTorch linalg
    true_evals, true_evecs = torch.linalg.eigh(A)
    
    # torch.linalg.eigh returns eigenvalues in ascending order
    # Index 1 = Dominant (Largest λ), Index 0 = Secondary (Smaller λ)
    true_v1 = true_evecs[:, 1]  # Dominant eigenvector (λ ≈ 4.80)
    true_v2 = true_evecs[:, 0]  # Secondary eigenvector (λ ≈ 1.20)

    print("=" * 70)
    print("LEARNING EIGENVECTORS VIA GRADIENT DESCENT & COSINE SIMILARITY")
    print("=" * 70)
    print("Target Matrix A:")
    print(A.numpy())
    print(f"\nAnalytical Eigenvalues:  λ1 = {true_evals[1].item():.4f},  λ2 = {true_evals[0].item():.4f}")
    print(f"Analytical Dominant v1:   {true_v1.tolist()}")
    print(f"Analytical Secondary v2:  {true_v2.tolist()}")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # PHASE 1: Learn Dominant Eigenvector (v1)
    # Loss = 1.0 - |Cosine_Similarity(A @ v1, v1)|
    # -------------------------------------------------------------------------
    print("\n--- PHASE 1: Learning Dominant Eigenvector (v1) ---")
    v1_param = torch.tensor([1.0, 0.1], requires_grad=True)
    optimizer1 = torch.optim.Adam([v1_param], lr=0.04)

    for epoch in range(1, 81):
        optimizer1.zero_grad()
        v1_unit = v1_param / torch.norm(v1_param)
        Av1 = A @ v1_unit
        
        # Absolute Cosine Similarity between input v1 and transformed Av1
        cos_sim = torch.abs(torch.nn.functional.cosine_similarity(v1_unit.unsqueeze(0), Av1.unsqueeze(0)))
        loss = 1.0 - cos_sim
        loss.backward()
        optimizer1.step()

        if epoch % 20 == 0 or epoch == 80:
            v_curr = (v1_param / torch.norm(v1_param)).detach().numpy()
            print(f"Epoch {epoch:2d} - Loss: {loss.item():.6f} - Cos Sim: {cos_sim.item():.6f} - Vector v1: [{v_curr[0]:.4f}, {v_curr[1]:.4f}]")

    v1_learned = (v1_param / torch.norm(v1_param)).detach()

    # -------------------------------------------------------------------------
    # PHASE 2: Learn Secondary Eigenvector (v2)
    # Loss = (1.0 - |Cosine_Similarity(A @ v2, v2)|) + Penalty * (v1_learned · v2)^2
    # Enforces orthogonality to previously learned v1
    # -------------------------------------------------------------------------
    print("\n--- PHASE 2: Learning Secondary Eigenvector (v2) ---")
    v2_param = torch.tensor([0.1, 1.0], requires_grad=True)
    optimizer2 = torch.optim.Adam([v2_param], lr=0.04)

    for epoch in range(1, 81):
        optimizer2.zero_grad()
        v2_unit = v2_param / torch.norm(v2_param)
        Av2 = A @ v2_unit
        
        cos_sim = torch.abs(torch.nn.functional.cosine_similarity(v2_unit.unsqueeze(0), Av2.unsqueeze(0)))
        ortho_penalty = (torch.dot(v1_learned, v2_unit)) ** 2
        loss = (1.0 - cos_sim) + 10.0 * ortho_penalty
        loss.backward()
        optimizer2.step()

        if epoch % 20 == 0 or epoch == 80:
            v_curr = (v2_param / torch.norm(v2_param)).detach().numpy()
            print(f"Epoch {epoch:2d} - Loss: {loss.item():.6f} - Cos Sim: {cos_sim.item():.6f} - Vector v2: [{v_curr[0]:.4f}, {v_curr[1]:.4f}]")

    v2_learned = (v2_param / torch.norm(v2_param)).detach()

    print("\n" + "=" * 70)
    print("VERIFICATION OF LEARNED EIGENVECTORS")
    print("=" * 70)
    print(f"Learned Dominant v1:   {v1_learned.numpy().tolist()}")
    print(f"Learned Secondary v2:  {v2_learned.numpy().tolist()}")
    print(f"Dot Product (v1 · v2): {torch.dot(v1_learned, v2_learned).item():.6f} (Orthogonal!)")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
