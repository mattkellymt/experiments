import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)

dim = 32

# Model 1: Unconstrained 64D Dense MLP (All-to-All Mixing)
class UnconstrainedDense64D(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        # Input concatenation [a ; b] -> 64D
        self.fc1 = nn.Linear(dim * 2, dim * 2)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(dim * 2, dim * 2)
        self.act2 = nn.GELU()
        self.out = nn.Linear(dim * 2, dim)

    def forward(self, a, b):
        x = torch.cat([a, b], dim=1) # (batch, 64)
        h1 = self.act1(self.fc1(x))
        h2 = self.act2(self.fc2(h1))
        return self.out(h2)

# Model 2: Enforced Half-Split Non-Intersecting Whitney 64D
# Lower 32 processes a, Upper 32 processes b independently, plus a learned cross-channel interaction path
class EnforcedWhitney64D(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        # Independent stream for lower 32 (a)
        self.stream_a1 = nn.Linear(dim, dim)
        self.stream_a2 = nn.Linear(dim, dim)
        
        # Independent stream for upper 32 (b)
        self.stream_b1 = nn.Linear(dim, dim)
        self.stream_b2 = nn.Linear(dim, dim)
        
        # Cross-channel interaction gate & mixer (allows optional intersection)
        self.cross_gate = nn.Linear(dim * 2, 1)
        self.cross_mix1 = nn.Linear(dim * 2, dim * 2)
        self.cross_mix2 = nn.Linear(dim * 2, dim * 2)
        
        self.act = nn.GELU()
        self.out = nn.Linear(dim * 2, dim)

    def forward(self, a, b, return_gate=False):
        # Guaranteed half-split non-intersecting initial streams
        ha1 = self.act(self.stream_a1(a)) # (batch, 32)
        hb1 = self.act(self.stream_b1(b)) # (batch, 32)
        
        ha2 = self.act(self.stream_a2(ha1)) # (batch, 32)
        hb2 = self.act(self.stream_b2(hb1)) # (batch, 32)
        
        x_unmixed = torch.cat([ha2, hb2], dim=1) # Pure non-intersecting 64D representation
        
        # Cross-channel path
        x_concat = torch.cat([a, b], dim=1)
        gate = torch.sigmoid(self.cross_gate(x_concat))
        x_mix = self.act(self.cross_mix2(self.act(self.cross_mix1(x_concat))))
        
        # Blend: non-intersecting by default + optional gated intersection
        x_64 = x_unmixed + gate * x_mix
        out = self.out(x_64)
        
        if return_gate:
            return out, gate
        return out

m1 = UnconstrainedDense64D(dim)
m2 = EnforcedWhitney64D(dim)

params1 = sum(p.numel() for p in m1.parameters())
params2 = sum(p.numel() for p in m2.parameters())

print(f"Unconstrained 64D Dense Parameters: {params1}")
print(f"Enforced Whitney 64D Parameters   : {params2}")
