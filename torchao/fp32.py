import os
import torch
import torch.nn as nn
from model import RotationModel, generate_dataset

def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[FP32] Using device: {device}")

    # Initialize model in FP32
    model_fp32 = RotationModel(dim=256, hidden_dim=1024).to(device=device, dtype=torch.float32)

    # Generate dataset (train: 2048 samples, test: 512 samples)
    x_train, y_train = generate_dataset(num_samples=2048, dim=256, seed=42)
    x_test, y_test = generate_dataset(num_samples=512, dim=256, seed=999)

    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    # Training loop
    optimizer = torch.optim.AdamW(model_fp32.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    epochs = 150
    print(f"[FP32] Starting training for {epochs} epochs...")
    model_fp32.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        y_pred = model_fp32(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:3d}/{epochs} - Loss: {loss.item():.6f}")

    # Evaluate on Test Set
    model_fp32.eval()
    with torch.no_grad():
        y_test_pred = model_fp32(x_test)
        test_mse = criterion(y_test_pred, y_test).item()

    print(f"[FP32] Training Complete. Test MSE: {test_mse:.6f}")

    # Save reference FP32 model checkpoint
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model_fp32.pt")
    
    # Save state dict
    torch.save(model_fp32.state_dict(), model_path)
    file_size_bytes = os.path.getsize(model_path)
    file_size_mb = file_size_bytes / (1024 * 1024)

    print(f"[FP32] Reference FP32 model saved to: {model_path}")
    print(f"[FP32] File Size: {file_size_bytes:,} bytes ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
