# Experimental STE Binning + Non-Linear Neural Codebook Rehydration

Personal exploration repo testing your STE binning hypothesis: **Using Straight-Through Estimators (STE) to force FP32 Master Weights ($W_{\text{master}}$) into discrete FP4/FP8 bins while fine-tuning an end-to-end Non-Linear Neural Rehydration Engine ($f_\theta$)** optimized for Apple Silicon Metal GPU (`mps`).

---

## 💡 The Architecture Concept

```
                   STE MASTER WEIGHT BINNING & TRAINING
┌──────────────────────────────────────────────────────────────────────────────┐
│  • FP32 Master Weights W_master fine-tuned continuously via STE gradients.   │
│  • STE Binning: In forward pass, force W_master into FP4 / FP8 grid bins.    │
│    In backward pass, STE passes identity gradients (∂Loss/∂W_master ≈ 1.0). │
│  • Non-Linear Rehydrator:                                                    │
│    W_rehydrated = W_binned + γ * RefinementHead(GELU(MLP(W_binned)))          │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Benchmark Results: FP32 vs. STE FP4 & FP8 Binned Neural Rehydrators

| Quantization Method / Format | Stored Representation | Storage Footprint | Effective Bits / Param | Worst Cos Sim | Ref Mag | Rehydrated Mag | Mean Cos Sim | Execution Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 (Ref Ground Truth)** | 32-Bit Float | `10.00 MB` | `32.0 bits` | `1.000000` | `16.0427` | `16.0427` | `1.000000` | — |
| **STE FP4 Binned + Rehydrator** | **FP4 Binned + NN** | **`1.41 MB + NN`** | **`4.0 bits + NN`** | **`0.478837`** | `14.1138` | `8.0881` | **`0.566002`** | **2 seconds** |
| **STE FP8 Binned + Rehydrator** | **FP8 Binned + NN** | **`2.62 MB + NN`** | **`8.0 bits + NN`** | **`0.428118`** | `14.2286` | `9.8745` | **`0.571173`** | **2 seconds** |

---

## 🖼️ Codebook Analysis Plot

![Codebook Analysis Plot](codebook_analysis_plot.png)

- **Row 1 (Scatter Plots)**: Compares **Relative Magnitude Error (%) vs. Cosine Similarity** for STE FP4 vs STE FP8 Binned Rehydrators (Left) and STE Gradient Flow Curve (Right).
- **Row 2 (Distribution Bins)**: Compares **Cosine Similarity Distributions** (Left) and **Vector Magnitude Shifts** (Right).

---

## 📂 File Layout

- [`model.py`](model.py): Neural network architecture, `STEFP4Grid` autograd function, `STEFP8Grid` autograd function, and `STENonLinearCodebookRehydrator` module.
- [`train_codebook.py`](train_codebook.py): Baseline training and STE fine-tuning scripts.
- [`plot.py`](plot.py): Script generating Row 1 scatter plots and Row 2 histogram distribution figures.
