# Experimental Neural Codebook Rehydration & Softmax Mixture Quantization

Personal exploration repo implementing a custom experimental quantization paradigm: **Softmax Codebook Mixture Quantization (SCMQ) / Neural Rehydration** optimized for Apple Silicon Metal GPU (`mps`).

---

## 💡 The Architecture Concept

Instead of quantizing individual weight numbers into 4-bit or 8-bit grids, we represent entire $32 \times 32$ weight blocks as **discrete index pointers** ($K=256 \rightarrow$ 8 bits, $K=512 \rightarrow$ 9 bits) into a shared dual codebook!

```
                  OFFLINE TRAINING & ANNEALING PHASE
┌──────────────────────────────────────────────────────────────────────────┐
│  1. Slice weight matrix W into 32x32 blocks.                              │
│  2. Compute per-block norm scale factor S_block = ||W_b|| / 5.0.          │
│  3. Codebook 1 (K x 32 x 32): Compute fast GEMM prototype similarity.    │
│  4. Softmax(sim / τ): Generate mixture weights α_1 .. α_K.               │
│  5. Codebook 2 (K x 32 x 32): Linear combination of FP32 basis tensors.  │
│  6. Cool temperature τ: 1.0 → 0.05 (Softmax sharpens into Argmax).        │
└──────────────────────────────────────────────────────────────────────────┘

                  INFERENCE TIME LAYER REHYDRATION
┌──────────────────────────────────────────────────────────────────────────┐
│  1. Store ONLY 8-bit / 9-bit block indices k and Codebooks (0.21 MB).     │
│  2. Rehydrate Layer L: W_hat_L = S_block * Codebook2[k_blocks].          │
│  3. Execute forward pass: Y = X @ W_hat_L.                               │
│  4. Discard W_hat_L from VRAM before moving to Layer L+1!               │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Benchmark Results: TorchAO Native FP4 vs. K=256 & K=512 Codebooks

| Quantization Method | Representation | Storage Footprint | Effective Bits / Param | Worst Cos Sim | Ref Mag | Variant Mag | Mean Cos Sim | Execution Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 (Ref Ground Truth)** | 32-Bit Float | `10.00 MB` | `32.0 bits` | `1.000000` | `16.0427` | `16.0427` | `1.000000` | — |
| **TorchAO Native FP4** | 4-Bit Micro-Scale | `1.41 MB` | `4.0 bits` | **`0.988580`** | `12.8681` | `12.7213` | **`0.991338`** | Instant |
| **Codebook K=256 (8-bit Index)** | 8-Bit Index / Block | `0.20 MB` | `0.78 bits` | `0.786355` | `14.5336` | `12.0069` | `0.843481` | 2 seconds |
| **Codebook K=512 (9-bit Index)** | **9-Bit Index / Block** | **`0.21 MB`** | **`0.88 bits`** | **`0.861997`** | `13.8130` | `13.1562` | **`0.903540`** | **5 seconds $\star$** |

---

## 🖼️ Codebook Analysis Plot

![Codebook Analysis Plot](codebook_analysis_plot.png)

- **Row 1 (Scatter Plots)**: Compares **Relative Magnitude Error (%) vs. Cosine Similarity** for TorchAO FP4 vs. $K=256$ & $K=512$ Codebooks (Left) and Temperature Annealing Curve $\tau$ (Right).
- **Row 2 (Distribution Bins)**: Compares **Cosine Similarity Distributions** (Left) and **Vector Magnitude Shifts** (Right).

---

## 📂 File Layout

- [`model.py`](model.py): Neural network architecture, `DualCodebookQuantizer` module with per-block scale factors and fast GEMM matrix products.
- [`train_codebook.py`](train_codebook.py): FP32 reference model training, temperature-annealed codebook fine-tuning ($K=256$ and $K=512$) with Apple Silicon MPS GPU cache management.
- [`plot.py`](plot.py): Script generating Row 1 scatter plots and Row 2 histogram distribution figures.
