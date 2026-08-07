import os
import torch
import torch.nn as nn
from model import RotationModel, generate_dataset, evaluate_predictions
from torchao.quantization import quantize_, Float8WeightOnlyConfig
from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Train FP32 Reference Model
    model_fp32 = RotationModel(dim=256, hidden_dim=1024).to(device)
    x_train, y_train = generate_dataset(num_samples=2048, dim=256, seed=42)
    x_test, y_test = generate_dataset(num_samples=512, dim=256, seed=999)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    optimizer = torch.optim.AdamW(model_fp32.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    print("Training FP32 reference model (150 epochs)...")
    model_fp32.train()
    for epoch in range(150):
        optimizer.zero_grad()
        loss = criterion(model_fp32(x_train), y_train)
        loss.backward()
        optimizer.step()

    model_fp32.eval()
    with torch.no_grad():
        out_fp32 = model_fp32(x_test)

    fp32_path = os.path.join(script_dir, "model_fp32.pt")
    torch.save(model_fp32.state_dict(), fp32_path)
    fp32_size = os.path.getsize(fp32_path)

    # 2. FP16 Variant
    m_fp16 = RotationModel(dim=256, hidden_dim=1024).to(device, dtype=torch.float16)
    m_fp16.load_state_dict({k: v.half() for k, v in model_fp32.state_dict().items()})
    fp16_path = os.path.join(script_dir, "model_fp16.pt")
    torch.save(m_fp16.state_dict(), fp16_path)

    # 3. BF16 Variant
    m_bf16 = RotationModel(dim=256, hidden_dim=1024).to(device, dtype=torch.bfloat16)
    m_bf16.load_state_dict({k: v.bfloat16() for k, v in model_fp32.state_dict().items()})
    bf16_path = os.path.join(script_dir, "model_bf16.pt")
    torch.save(m_bf16.state_dict(), bf16_path)

    # 4. TorchAO FP8 Variant
    m_fp8 = RotationModel(dim=256, hidden_dim=1024).to(device)
    m_fp8.load_state_dict(model_fp32.state_dict())
    quantize_(m_fp8, Float8WeightOnlyConfig())
    fp8_path = os.path.join(script_dir, "model_fp8.pt")
    torch.save(m_fp8.state_dict(), fp8_path)

    # 5. TorchAO FP4 Variant
    m_fp4 = RotationModel(dim=256, hidden_dim=1024).to(device)
    m_fp4.load_state_dict(model_fp32.state_dict())
    quantize_(m_fp4, NVFP4WeightOnlyConfig())
    fp4_path = os.path.join(script_dir, "model_fp4.pt")
    torch.save(m_fp4.state_dict(), fp4_path)

    # Print Summary Table
    variants = [
        ("FP32 (Ref)", fp32_path, out_fp32),
        ("FP16", fp16_path, m_fp16(x_test.half()).float()),
        ("BF16", bf16_path, m_bf16(x_test.bfloat16()).float()),
        ("FP8 (TorchAO)", fp8_path, m_fp8(x_test).float()),
        ("FP4 (TorchAO)", fp4_path, m_fp4(x_test).float()),
    ]

    print("\n" + "=" * 115)
    print(f"{'Precision Variant':<18} | {'Size (MB)':<10} | {'Reduction':<10} | {'Worst Cos Sim':<15} | {'Ref Mag':<10} | {'Var Mag':<10} | {'Mean Cos':<10}")
    print("-" * 115)
    for label, path, out_var in variants:
        size_bytes = os.path.getsize(path)
        size_mb = size_bytes / (1024 * 1024)
        reduction = (1.0 - size_bytes / fp32_size) * 100.0
        m = evaluate_predictions(out_fp32, out_var, y_test)
        print(f"{label:<18} | {size_mb:7.2f} MB | {reduction:8.1f}% | {m['worst_cos_sim']:15.6f} | {m['worst_ref_mag']:10.4f} | {m['worst_var_mag']:10.4f} | {m['mean_cos_sim']:10.6f}")
    print("=" * 115 + "\n")

if __name__ == "__main__":
    main()
