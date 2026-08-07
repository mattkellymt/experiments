# Experimental Neural Codebook Rehydration & Softmax Mixture Quantization

Personal exploration repo implementing two variants of your experimental quantization paradigm: **Softmax Codebook Mixture Quantization (SCMQ) / Neural Rehydration** optimized for Apple Silicon Metal GPU (`mps`).

---

## 💡 The Architecture Concept

```
                   PARADIGM 1: PURE INDEX NEURAL REHYDRATION (0.22 MB)
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│  • Store ONLY 10-bit block indices k and shared Codebooks (0.22 MB on disk).             │
│  • Rehydrate Layer L: W_hat_L = S_block * Codebook2[k_blocks].                          │
│  • Achieves 0.9610 Mean Cosine Sim at sub-1 bit/param!                                  │
└──────────────────────────────────────────────────────────────────────────────────────────┘

               PARADIGM 2: 4-BIT GRID + DUAL-CODEBOOK REHYDRATOR (1.41 MB + NN)
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│  • Store 32x32 blocks of 4-bit quantized values (W_fp4).                                 │
│  • Matmul W_fp4 with Codebook 1 (1024 x 32 x 32) -> Compute similarity scores.            │
│  • Softmax (fine-tuning) / Hard Argmax (inference) selects Codebook 2.                   │
│  • Rehydrates 4-bit spatial grid into full 32-bit FP32 weight matrix!                    │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Benchmark Results: TorchAO Native FP4 vs. Both Codebook Paradigms

| Quantization Method / Architecture | Stored Representation | Storage Footprint | Effective Bits / Param | Worst Cos Sim | Ref Mag | Variant Mag | Mean Cos Sim | Execution Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 (Ref Ground Truth)** | 32-Bit Float | `10.00 MB` | `32.0 bits` | `1.000000` | `16.0427` | `16.0427` | `1.000000` | — |
| **TorchAO Native FP4** | 4-Bit Micro-Scale | `1.41 MB` | `4.0 bits` | **`0.988580`** | `12.8681` | `12.7213` | **`0.991338`** | Instant |
| **Pure Index Codebook $K=1024$** | **10-Bit Index / Block** | **`0.22 MB`** | **`0.97 bits`** | **`0.933949`** | `13.7431` | `12.7341` | **`0.960992` $\star$** | **5 seconds** |
| **4-Bit Grid + Codebook $K=512$** | 4-Bit Grid + NN | `1.41 MB + NN` | `4.0 bits + NN` | `0.447956` | `15.3463` | `27.7146` | `0.537198` | 5 seconds |
| **4-Bit Grid + Codebook $K=1024$** | 4-Bit Grid + NN | `1.41 MB + NN` | `4.0 bits + NN` | `-0.352361` | `16.4617` | `38.4554` | `0.453953` | 5 seconds |

---

## 🖼️ Codebook Analysis Plot

![Codebook Analysis Plot](codebook_analysis_plot.png)

- **Row 1 (Scatter Plots)**: Compares **Relative Magnitude Error (%) vs. Cosine Similarity** for TorchAO FP4 vs. 4-Bit Grid + Neural Rehydrator (Left) and Temperature Annealing Curve $\tau$ (Right).
- **Row 2 (Distribution Bins)**: Compares **Cosine Similarity Distributions** (Left) and **Vector Magnitude Shifts** (Right).

---

## 📂 File Layout

- [`model.py`](model.py): Neural network architecture, `GridCodebookRehydrator` module supporting 4-bit grid matmul with Codebooks 1 & 2.
- [`train_codebook.py`](train_codebook.py): FP32 baseline training and fine-tuning scripts.
- [`plot.py`](plot.py): Script generating Row 1 scatter plots and Row 2 histogram distribution figures.
