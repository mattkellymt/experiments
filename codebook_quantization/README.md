# Experimental Single Codebook Neural Rehydration

Personal exploration repo implementing your experimental quantization paradigm: **Softmax Codebook Mixture Quantization (SCMQ) / Single Codebook Neural Rehydration** optimized for Apple Silicon Metal GPU (`mps`).

---

## 💡 Do We Need 2 Codebooks or 1 Single Codebook Matrix?

**We only need 1 SINGLE Codebook matrix (`1024 x 32 x 32`)!**

Using a single codebook matrix makes the architecture cleaner, simpler, and cuts stored codebook parameters in half (`0.18 MB` total model size):

- **Single Codebook Tensor ($C \in \mathbb{R}^{1024 \times 32 \times 32}$)**:
  - Acts as BOTH the prototype selector vector during similarity matching:
    $$\mathbf{s} = W_{\text{norm, flat}} \cdot C_{\text{flat}}^T \quad \in \mathbb{R}^{1024}$$
  - AND the basis expansion output tensor during rehydration:
    $$W_{\text{rehydrated, block}} = S_{\text{block}} \times C[\text{argmax}(\mathbf{s})]$$

---

## 💡 The Single Codebook Pipeline

```
                  OFFLINE TRAINING & ANNEALING PHASE
┌──────────────────────────────────────────────────────────────────────────┐
│  1. Slice trained FP32 weight matrix W into 32x32 blocks.                │
│  2. Compute per-block norm scale factor S_block = ||W_b|| / 5.0.          │
│  3. SINGLE Codebook (K=1024 x 32 x 32): Fast GEMM similarity matching:   │
│     sim = W_norm_flat @ Codebook_flat^T                                  │
│  4. Softmax(sim / τ): Generate mixture weights α_1 .. α_1024.             │
│  5. Rehydrate W_hat = S_block * (α @ Codebook_flat).                     │
│  6. Cool temperature τ: 1.0 → 0.05 (Softmax sharpens into Argmax).        │
└──────────────────────────────────────────────────────────────────────────┘

                  INFERENCE TIME LAYER REHYDRATION
┌──────────────────────────────────────────────────────────────────────────┐
│  1. Store ONLY 10-bit block indices k and 1 Single Codebook (0.18 MB).   │
│  2. Rehydrate Layer L: W_hat_L = S_block * Codebook[k_blocks].           │
│  3. Execute forward pass: Y = X @ W_hat_L.                               │
│  4. Discard W_hat_L from VRAM before moving to Layer L+1!               │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Single Codebook Neural Rehydration Benchmark Results

| Quantization Method / Architecture | Storage Footprint | Effective Bits / Param | Worst Cos Sim | Ref Mag | Rehydrated Mag | Mean Cos Sim | Execution Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 (Ref Ground Truth)** | `10.00 MB` | `32.0 bits` | `1.000000` | `16.0427` | `16.0427` | `1.000000` | — |
| **Single Codebook $K=1024$** | **`0.18 MB`** | **`0.97 bits`** | **`0.900191`** | `13.7431` | `12.7341` | **`0.937675` $\star$** | **3 seconds** |

---

## 🖼️ Codebook Analysis Plot

![Codebook Analysis Plot](codebook_analysis_plot.png)

- **Row 1 (Scatter Plots)**: Compares **Relative Magnitude Error (%) vs. Cosine Similarity** (Left) and Temperature Annealing Curve $\tau$ (Right).
- **Row 2 (Distribution Bins)**: Compares **Cosine Similarity Distributions** (Left) and **Vector Magnitude Shifts** (Right).

---

## 📂 File Layout

- [`model.py`](model.py): Neural network architecture and `PureIndexCodebookQuantizer` module using 1 single codebook.
- [`train_codebook.py`](train_codebook.py): FP32 baseline training and fine-tuning scripts.
- [`plot.py`](plot.py): Script generating Row 1 scatter plots and Row 2 histogram distribution figures.
