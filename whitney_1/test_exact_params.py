import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)

dim = 32

# Model A: Unconstrained 64D Dense Architecture (Hidden width tuned to match params exactly)
# Hidden width H = 65 gives: Linear(64, 65) + Linear(65, 65) + Linear(65, 65) + Linear(65, 32)
# = (64*65+65) + (65*65+65) + (65*65+65) + (65*32+32) = 4225 + 4290 + 4290 + 2112 = 14,917
# Let's adjust width to get EXACT parameter equality!

class ParamMatchedDense64D(nn.Module):
    def __init__(self, dim=32, hidden_dim=65):
        super().__init__()
        self.fc1 = nn.Linear(dim * 2, hidden_dim) # 64 -> H
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # H -> H
        self.act2 = nn.GELU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim) # H -> H
        self.act3 = nn.GELU()
        self.out = nn.Linear(hidden_dim, dim)      # H -> 32

    def forward(self, a, b):
        x = torch.cat([a, b], dim=1) # (batch, 64)
        h1 = self.act1(self.fc1(x))
        h2 = self.act2(self.fc2(h1))
        h3 = self.act3(self.fc3(h2))
        return self.out(h3)

# Model B: Enforced Half-Split Non-Intersecting Whitney 64D
class EnforcedWhitney64D(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.stream_a1 = nn.Linear(dim, dim) # 32 -> 32
        self.stream_a2 = nn.Linear(dim, dim) # 32 -> 32
        self.stream_b1 = nn.Linear(dim, dim) # 32 -> 32
        self.stream_b2 = nn.Linear(dim, dim) # 32 -> 32
        
        self.cross_gate = nn.Linear(dim * 2, 1) # 64 -> 1
        self.cross_mix1 = nn.Linear(dim * 2, dim * 2) # 64 -> 64
        self.cross_mix2 = nn.Linear(dim * 2, dim * 2) # 64 -> 64
        
        self.act = nn.GELU()
        self.out = nn.Linear(dim * 2, dim) # 64 -> 32

    def forward(self, a, b, return_gate=False):
        ha1 = self.act(self.stream_a1(a))
        hb1 = self.act(self.stream_b1(b))
        ha2 = self.act(self.stream_a2(ha1))
        hb2 = self.act(self.stream_b2(hb1))
        
        x_unmixed = torch.cat([ha2, hb2], dim=1)
        x_concat = torch.cat([a, b], dim=1)
        gate = torch.sigmoid(self.cross_gate(x_concat))
        x_mix = self.act(self.cross_mix2(self.act(self.cross_mix1(x_concat))))
        
        x_64 = x_unmixed + gate * x_mix
        out = self.out(x_64)
        
        if return_gate:
            return out, gate
        return out

m1 = ParamMatchedDense64D(dim, hidden_dim=65)
m2 = EnforcedWhitney64D(dim)

p1 = sum(p.numel() for p in m1.parameters())
p2 = sum(p.numel() for p in m2.parameters())

print(f"Model A (Param-Matched Unconstrained 64D Dense) Params: {p1}")
print(f"Model B (Enforced Half-Split Whitney 64D) Params        : {p2}")
print(f"Parameter Difference: {abs(p1 - p2)} parameters ({abs(p1 - p2)/p2*100:.2f}%)")
