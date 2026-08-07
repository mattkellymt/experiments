# Experimental Neural Codebook Rehydration & Softmax Mixture Quantization

Personal exploration repo implementing a custom experimental quantization paradigm: **Softmax Codebook Mixture Quantization (SCMQ) / Neural Rehydration**.

---

## 💡 The Architecture Concept

Instead of quantizing individual weight numbers into 4-bit or 8-bit grids, we represent entire $32 \times 32$ weight blocks as **6-bit discrete index pointers** into a shared dual codebook!

```
                  OFFLINE TRAINING & ANNEALING PHASE
┌──────────────────────────────────────────────────────────────────────────┐
│  • Slice weight matrix W into 32x32 blocks.                              │
│  • Codebook 1 (K=64 x 32 x 32): Compute dot-product prototype scores.    │
│  • Softmax(sim / τ): Generate mixture weights α_1 .. α_64.               │
│  • Codebook 2 (K=64 x 32 x 32): Linear combination of FP32 basis tensors.│
│  • Cool temperature τ: 1.0 → 0.05 (Softmax sharpens into Argmax).        │
└──────────────────────────────────────────────────────────────────────────┘

                  INFERENCE TIME LAYER REHYDRATION
┌──────────────────────────────────────────────────────────────────────────┐
│  1. Load 6-bit block indices k and shared Codebooks (0.19 MB on disk).   │
│  2. Rehydrate Layer L: W_hat_L = Codebook2[k_blocks].                    │
│  3. Execute forward pass: Y = X @ W_hat_L.                               │
│  4. Discard W_hat_L from VRAM before moving to Layer L+1!               │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Benchmark Results: TorchAO Native FP4 vs. Neural Codebook Rehydration

| Quantization Method | Representation | Storage Footprint | Bits / Param | Worst Cos Sim | Ref Mag | Variant Mag | Mean Cos Sim |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 (Ref Ground Truth)** | 32-Bit Float | `10.00 MB` | `32.0 bits` | `1.000000` | `16.2210` | `16.2210` | `1.000000` |
| **TorchAO Native FP4** | 4-Bit Micro-Scale | `1.41 MB` | `4.0 bits` | `0.987285` | `13.7367` | `13.5644` | **`0.990556`** |
| **Codebook (Soft Mixture)** | Differentiable Mixture | `0.19 MB` | `0.58 bits` | `0.539000` | `14.9310` | `8.7357` | `0.649396` |
| **Codebook (Hard Rehydrated)** | **6-Bit Index / Block** | **`0.19 MB`** | **`0.58 bits`** | `0.406333` | `15.6320` | `11.2464` | `0.560466` |

---

## 🖼️ Codebook Analysis Plot

![Codebook Analysis Plot](codebook_analysis_plot.png)

- **Row 1 (Scatter Plots)**: Compares **Relative Magnitude Error (%) vs. Cosine Similarity** for TorchAO FP4 vs Dual-Codebook Rehydration (Left) and Temperature Annealing Curve $\tau$ (Right).
- **Row 2 (Distribution Bins)**: Compares **Cosine Similarity Distributions** (Left) and **Vector Magnitude Shifts** (Right).

---

## 📂 File Layout

- [`model.py`](model.py): Neural network architecture, `DualCodebookQuantizer`, `CodebookRehydrationLinear` wrapper, and evaluation metrics.
- [`train_codebook.py`](train_codebook.py): FP32 reference model training, temperature-annealed codebook fine-tuning, and evaluation.
- [`plot.py`](plot.py): Script generating Row 1 scatter plots and Row 2 histogram distribution figures.
