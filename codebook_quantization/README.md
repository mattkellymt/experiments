# Experimental Pure Index Dual-Codebook Neural Rehydration

Personal exploration repo implementing your experimental quantization paradigm: **Softmax Codebook Mixture Quantization (SCMQ) / Pure Index Neural Rehydration** optimized for Apple Silicon Metal GPU (`mps`).

---

## 💡 How Many 32x32 Grids Did We Learn?

In our 4-layer model (`dim=256`, `hidden_dim=1024`):

1. **Total Weight Blocks in Network**:
   - `fc1`: $(1024 \times 256) \rightarrow \frac{1024}{32} \times \frac{256}{32} = 32 \times 8 = \mathbf{256 \text{ blocks}}$
   - `fc2`: $(1024 \times 1024) \rightarrow \frac{1024}{32} \times \frac{1024}{32} = 32 \times 32 = \mathbf{1,024 \text{ blocks}}$
   - `fc3`: $(1024 \times 1024) \rightarrow \frac{1024}{32} \times \frac{1024}{32} = 32 \times 32 = \mathbf{1,024 \text{ blocks}}$
   - `fc4`: $(256 \times 1024) \rightarrow \frac{256}{32} \times \frac{1024}{32} = 8 \times 32 = \mathbf{256 \text{ blocks}}$
   - **Total Weight Blocks**: **$2,560$ blocks** across all 4 layers.

2. **Total Learned Prototype Basis Grids in Codebook**:
   - We set codebook size $K=1024$.
   - **Codebook 1**: $1,024$ FP32 prototype selector grids ($32 \times 32$).
   - **Codebook 2**: **$1,024$ unique full FP32 basis expansion grids ($32 \times 32$)**.
   - During fine-tuning, **100% of the 1,024 prototype basis grids are learned**, and all 2,560 blocks in the network map to these 1,024 master templates!

---

## 💡 The Architecture Pipeline

```
                  OFFLINE TRAINING & ANNEALING PHASE
┌──────────────────────────────────────────────────────────────────────────┐
│  1. Slice trained FP32 weight matrix W into 32x32 blocks.                │
│  2. Compute per-block norm scale factor S_block = ||W_b|| / 5.0.          │
│  3. Codebook 1 (K=1024 x 32 x 32): Fast GEMM prototype similarity:       │
│     sim = W_norm_flat @ Codebook1_flat^T                                 │
│  4. Softmax(sim / τ): Generate mixture weights α_1 .. α_1024.             │
│  5. Codebook 2 (K=1024 x 32 x 32): Rehydrate W_hat = S_block * (α @ C2). │
│  6. Cool temperature τ: 1.0 → 0.05 (Softmax sharpens into Argmax).        │
└──────────────────────────────────────────────────────────────────────────┘

                  INFERENCE TIME LAYER REHYDRATION
┌──────────────────────────────────────────────────────────────────────────┐
│  1. Store ONLY 10-bit block indices k and Codebooks (0.22 MB on disk).   │
│  2. Rehydrate Layer L: W_hat_L = S_block * Codebook2[k_blocks].          │
│  3. Execute forward pass: Y = X @ W_hat_L.                               │
│  4. Discard W_hat_L from VRAM before moving to Layer L+1!               │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Pure Index Neural Rehydration Benchmark Results

| Quantization Method / Architecture | Storage Footprint | Effective Bits / Param | Worst Cos Sim | Ref Mag | Rehydrated Mag | Mean Cos Sim | Execution Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 (Ref Ground Truth)** | `10.00 MB` | `32.0 bits` | `1.000000` | `16.0427` | `16.0427` | `1.000000` | — |
| **Pure Index Codebook $K=1024$** | **`0.22 MB`** | **`0.97 bits`** | **`0.929830`** | `13.7431` | `12.7341` | **`0.965489` $\star$** | **4 seconds** |

---

## 🖼️ Codebook Analysis Plot

![Codebook Analysis Plot](codebook_analysis_plot.png)

- **Row 1 (Scatter Plots)**: Compares **Relative Magnitude Error (%) vs. Cosine Similarity** (Left) and Temperature Annealing Curve $\tau$ (Right).
- **Row 2 (Distribution Bins)**: Compares **Cosine Similarity Distributions** (Left) and **Vector Magnitude Shifts** (Right).

---

## 📂 File Layout

- [`model.py`](model.py): Neural network architecture and `PureIndexCodebookQuantizer` module.
- [`train_codebook.py`](train_codebook.py): FP32 baseline training and fine-tuning scripts.
- [`plot.py`](plot.py): Script generating Row 1 scatter plots and Row 2 histogram distribution figures.
