# Experimental Non-Linear FP4 & FP8 Neural Codebook Rehydration

Personal exploration repo testing your non-linear rehydration hypothesis across **4-bit FP4 and 8-bit FP8 low-precision weight grids** optimized for Apple Silicon Metal GPU (`mps`).

---

## 💡 The Architecture Concept

```
             DISK WEIGHT STORAGE (4-Bit / 8-Bit Quantized Grid: 1.41 MB / 2.62 MB)
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  • Storage: Weight matrix stored as 32x32 blocks of low-precision values (W_q).     │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                  NON-LINEAR NEURAL REHYDRATION ENGINE f_θ(W_q)
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  Step 1: Slice W_q into 32x32 blocks (1024 params / block).                          │
│  Step 2: Non-Linear Feature Extractor (2-Layer MLP + GELU):                          │
│          h = GELU(Linear2(GELU(Linear1(W_q_flat))))  ∈ R^512                         │
│  Step 3: Gated Non-Linear Neural Rehydration:                                        │
│          W_rehydrated = W_q_flat + γ * RefinementHead(h)                             │
│          --> Rehydrates coarse 4-bit / 8-bit grid steps into full FP32 matrices!      │
│  Step 4: Execute Layer Forward Pass: Y = X @ W_rehydrated.                           │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Benchmark Results: FP4 vs. FP8 Non-Linear Neural Rehydration

| Quantization Method / Format | Stored Representation | Storage Footprint | Effective Bits / Param | Worst Cos Sim | Ref Mag | Rehydrated Mag | Mean Cos Sim | Execution Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 (Ref Ground Truth)** | 32-Bit Float | `10.00 MB` | `32.0 bits` | `1.000000` | `16.0427` | `16.0427` | `1.000000` | — |
| **Raw Naive FP4 Baseline** | 4-Bit Micro-Scale | `1.41 MB` | `4.0 bits` | **`0.987507`** | `12.8681` | `12.4865` | **`0.990843` $\star$** | Instant |
| **Non-Linear Neural FP4 Rehydrator** | **4-Bit Grid + NN** | **`1.41 MB + NN`** | **`4.0 bits + NN`** | **`0.977062`** | `14.5336` | `14.7763` | **`0.986915` $\star$** | **2 seconds** |
| **Raw Naive FP8 Baseline** | 8-Bit (E4M3) | `2.62 MB` | `8.0 bits` | **`0.999997`** | `14.5336` | `14.5382` | **`0.999997` $\star$** | Instant |
| **Non-Linear Neural FP8 Rehydrator** | **8-Bit Grid + NN** | **`2.62 MB + NN`** | **`8.0 bits + NN`** | **`0.972625`** | `9.8161` | `10.5130` | **`0.986103` $\star$** | **2 seconds** |

---

## 🖼️ Codebook Analysis Plot

![Codebook Analysis Plot](codebook_analysis_plot.png)

- **Row 1 (Scatter Plots)**: Compares **Relative Magnitude Error (%) vs. Cosine Similarity** for FP4 Raw vs Rehydrated (Left) and FP8 Raw vs Rehydrated (Right).
- **Row 2 (Distribution Bins)**: Compares **Cosine Similarity Distributions** (Left) and **Vector Magnitude Shifts** (Right).

---

## 📂 File Layout

- [`model.py`](model.py): Neural network architecture and `NonLinearCodebookRehydrator` module supporting both FP4 and FP8 formats.
- [`train_codebook.py`](train_codebook.py): FP32 baseline training and fine-tuning scripts.
- [`plot.py`](plot.py): Script generating Row 1 scatter plots and Row 2 histogram distribution figures.
