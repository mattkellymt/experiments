import sys
import os
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from experiments.whitney_1.run_experiment import train_loader, test_loader, mA, mB, criterion_none

mA.eval()
mB.eval()

train_mse_A, test_mse_A = [], []
train_mse_B, test_mse_B = [], []

with torch.no_grad():
    for ba, bb, by, _ in train_loader:
        pA = mA(ba, bb)
        pB = mB(ba, bb)
        train_mse_A.append(criterion_none(pA, by).mean().item())
        train_mse_B.append(criterion_none(pB, by).mean().item())

    for ba, bb, by, _ in test_loader:
        pA = mA(ba, bb)
        pB = mB(ba, bb)
        test_mse_A.append(criterion_none(pA, by).mean().item())
        test_mse_B.append(criterion_none(pB, by).mean().item())

print(f"Model A (Standard Dense 64D)  -> Train MSE: {np.mean(train_mse_A):.6f} | Test MSE: {np.mean(test_mse_A):.6f} | Gap: {abs(np.mean(train_mse_A) - np.mean(test_mse_A)):.6f}")
print(f"Model B (Whitney 64D Model)   -> Train MSE: {np.mean(train_mse_B):.6f} | Test MSE: {np.mean(test_mse_B):.6f} | Gap: {abs(np.mean(train_mse_B) - np.mean(test_mse_B)):.6f}")
