import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)

dim = 32
num_samples = 20000

a = torch.randn(num_samples, dim)
b = torch.randn(num_samples, dim)
a = a / torch.norm(a, dim=1, keepdim=True)
b = b / torch.norm(b, dim=1, keepdim=True)

cos_sim = (a * b).sum(dim=1)
is_subtraction = (cos_sim < 0)
targets = torch.where(is_subtraction.unsqueeze(1), a - b, a * b)

# Gated Residual Whitney Architecture:
# Lower 32 processes a, Upper 32 processes b, with a learned cross-attention/gating block
class GatedWhitney64DModel(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        # Parallel stream layers
        self.stream_a = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        self.stream_b = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        
        # Cross-interaction gate (computes how much lower and upper channels mix)
        self.cross_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
        self.cross_mix = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.GELU()
        )
        
        self.out_proj = nn.Linear(dim * 2, dim)

    def forward(self, a, b, return_gate=False):
        ha = self.stream_a(a) # (batch, 32)
        hb = self.stream_b(b) # (batch, 32)
        
        # Compute dynamic cross-channel mixing gate
        x_concat = torch.cat([a, b], dim=1) # (batch, 64)
        gate = self.cross_gate(x_concat)    # (batch, 1) - 0 = no intersection, 1 = full intersection
        
        # Mixed features
        x_mix = self.cross_mix(x_concat)    # (batch, 64)
        
        # Unmixed orthogonal representation
        x_unmixed = torch.cat([ha, hb], dim=1) # (batch, 64)
        
        # Blend based on gate: if gate -> 0, x_64 remains unmixed (non-intersecting); if gate -> 1, x_64 mixes
        x_64 = x_unmixed + gate * x_mix
        
        out = self.out_proj(x_64)
        
        if return_gate:
            return out, gate
        return out

model = GatedWhitney64DModel(dim=32)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()

train_a, train_b, train_y = a[:16000], b[:16000], targets[:16000]
test_a, test_b, test_y = a[16000:], b[16000:], targets[16000:]
test_sub = is_subtraction[16000:]

for epoch in range(40):
    model.train()
    for i in range(0, 16000, 128):
        ba, bb, by = train_a[i:i+128], train_b[i:i+128], train_y[i:i+128]
        opt.zero_grad()
        loss = criterion(model(ba, bb), by)
        loss.backward()
        opt.step()

model.eval()
with torch.no_grad():
    out, gates = model(test_a, test_b, return_gate=True)
    test_mse = criterion(out, test_y).item()
    gates = gates.squeeze().numpy()
    sub_gates = gates[test_sub.numpy()]
    mult_gates = gates[~test_sub.numpy()]
    
    sub_mse = criterion(out[test_sub], test_y[test_sub]).item()
    mult_mse = criterion(out[~test_sub], test_y[~test_sub]).item()

print(f"Test MSE: {test_mse:.6f} (Sub MSE: {sub_mse:.6f}, Mult MSE: {mult_mse:.6f})")
print(f"Mean Gate for Subtraction (a-b): {sub_gates.mean():.4f}")
print(f"Mean Gate for Multiplication (a*b): {mult_gates.mean():.4f}")
