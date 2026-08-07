import os
import torch
import fp32, fp16, bf16, fp8, fp4
from model import RotationModel, generate_dataset, evaluate_predictions

def run_all():
    print("=" * 110)
    print("STEP 1: Executing Model Training and Precision Variant Conversions...")
    print("=" * 110)

    fp32.main()
    print("-" * 60)
    fp16.main()
    print("-" * 60)
    bf16.main()
    print("-" * 60)
    fp8.main()
    print("-" * 60)
    fp4.main()
    print("-" * 60)

    print("\n" + "=" * 110)
    print("STEP 2: Evaluating All Variants & Generating Detailed Cosine Similarity & Magnitude Breakdown Table...")
    print("=" * 110)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Generate benchmark dataset
    x_test, y_test = generate_dataset(num_samples=512, dim=256, seed=999)
    x_test, y_test = x_test.to(device), y_test.to(device)

    # 1. Load FP32 Reference
    fp32_path = os.path.join(script_dir, "model_fp32.pt")
    m_fp32 = RotationModel(dim=256, hidden_dim=1024).to(device=device)
    m_fp32.load_state_dict(torch.load(fp32_path, weights_only=True))
    m_fp32.eval()
    with torch.no_grad():
        out_fp32 = m_fp32(x_test)

    fp32_size = os.path.getsize(fp32_path)

    variants = [
        ("FP32 (Reference)", "model_fp32.pt", "fp32", torch.float32),
        ("FP16 (Half Precision)", "model_fp16.pt", "fp16", torch.float16),
        ("BF16 (Bfloat16)", "model_bf16.pt", "bf16", torch.bfloat16),
        ("FP8 (TorchAO Float8 W-Only)", "model_fp8.pt", "fp8", None),
        ("FP4 (TorchAO NVFP4 W-Only)", "model_fp4.pt", "fp4", None),
    ]

    results = []

    for label, filename, kind, dtype in variants:
        filepath = os.path.join(script_dir, filename)
        size_bytes = os.path.getsize(filepath)
        size_mb = size_bytes / (1024 * 1024)
        size_reduction = (1.0 - size_bytes / fp32_size) * 100.0

        if kind == "fp32":
            out_var = out_fp32
        elif kind in ("fp16", "bf16"):
            m_var = RotationModel(dim=256, hidden_dim=1024).to(device=device, dtype=dtype)
            m_var.load_state_dict(torch.load(filepath, weights_only=True))
            m_var.eval()
            with torch.no_grad():
                out_var = m_var(x_test.to(dtype)).float()
        elif kind in ("fp8", "fp4"):
            m_var = RotationModel(dim=256, hidden_dim=1024).to(device=device)
            m_var.load_state_dict(torch.load(filepath, weights_only=False), assign=True)
            m_var.eval()
            with torch.no_grad():
                out_var = m_var(x_test).float()

        metrics = evaluate_predictions(out_fp32, out_var, y_test)

        results.append({
            "label": label,
            "filename": filename,
            "size_mb": size_mb,
            "size_reduction": size_reduction,
            "test_mse": metrics["test_mse"],
            "mean_cos_sim": metrics["mean_cos_sim"],
            "worst_cos_sim": metrics["worst_cos_sim"],
            "worst_ref_mag": metrics["worst_ref_mag"],
            "worst_var_mag": metrics["worst_var_mag"],
            "median_cos_sim": metrics["median_cos_sim"],
            "median_ref_mag": metrics["median_ref_mag"],
            "median_var_mag": metrics["median_var_mag"],
            "worst_dot_cos_sim": metrics["worst_dot_cos_sim"],
            "worst_dot_ref_mag": metrics["worst_dot_ref_mag"],
            "worst_dot_var_mag": metrics["worst_dot_var_mag"],
        })

    # Print Summary Table 1: Cosine Similarity & Vector Magnitude Decomposition
    print("\n" + "#" * 140)
    print("PRIMARY BREAKDOWN: PER-SAMPLE COSINE SIMILARITY & VECTOR MAGNITUDE DECOMPOSITION")
    print("#" * 140)
    header1 = f"{'Precision Variant':<28} | {'Size (MB)':<9} | {'Worst Cos Sim':<14} | {'Ref Mag (Worst)':<15} | {'Var Mag (Worst)':<15} | {'Median Cos Sim':<14} | {'Ref Mag (Med)':<13} | {'Var Mag (Med)':<13}"
    print(header1)
    print("-" * 140)
    for r in results:
        line = (
            f"{r['label']:<28} | "
            f"{r['size_mb']:6.2f} MB | "
            f"{r['worst_cos_sim']:14.6f} | "
            f"{r['worst_ref_mag']:15.4f} | "
            f"{r['worst_var_mag']:15.4f} | "
            f"{r['median_cos_sim']:14.6f} | "
            f"{r['median_ref_mag']:13.4f} | "
            f"{r['median_var_mag']:13.4f}"
        )
        print(line)
    print("#" * 140 + "\n")

    # Print Summary Table 2: Dot Product Error Decomposition
    print("#" * 120)
    print("SECONDARY BREAKDOWN: SAMPLE WITH MOST DISSIMILAR DOT PRODUCT / MAGNITUDE DISTANCE")
    print("#" * 120)
    header2 = f"{'Precision Variant':<28} | {'Dot-Worst Cos Sim':<18} | {'Ref Mag (Dot-Worst)':<20} | {'Var Mag (Dot-Worst)':<20} | {'Mean Cos Sim':<12}"
    print(header2)
    print("-" * 120)
    for r in results:
        line2 = (
            f"{r['label']:<28} | "
            f"{r['worst_dot_cos_sim']:18.6f} | "
            f"{r['worst_dot_ref_mag']:20.4f} | "
            f"{r['worst_dot_var_mag']:20.4f} | "
            f"{r['mean_cos_sim']:12.6f}"
        )
        print(line2)
    print("#" * 120 + "\n")

    return results

if __name__ == "__main__":
    run_all()
