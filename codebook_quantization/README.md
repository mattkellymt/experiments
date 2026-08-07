# Experimental Single Codebook Neural Rehydration: Block Size Sweep

Personal exploration repo evaluating the impact of **Spatial Block Size ($32 \times 32 \rightarrow 4 \times 4$)** on Neural Codebook Rehydration optimized for Apple Silicon Metal GPU (`mps`).

---

## 💡 How Does Block Size Affect Approximation Quality?

Surprisingly, **larger $32 \times 32$ blocks significantly OUTPERFORM smaller $4 \times 4$ blocks** when codebook size $K=1024$ is fixed!

### Why Larger $32 \times 32$ Blocks Win:

1. **Codebook Capacity Bottleneck**:
   - A $1024 \times 1024$ hidden layer contains **$1,024$ blocks of size $32 \times 32$**. When $K=1024$, the codebook size $K$ matches the number of blocks $1:1$!
   - Shrinking the block size to $4 \times 4$ increases the number of blocks to **$65,536$ blocks**. Squeezing 65,536 blocks into $K=1024$ forces $65$ blocks to share 1 entry ($65:1$ compression bottleneck!).
2. **Global Spatial Orientation**:
   - Larger $32 \times 32$ blocks capture the low-rank eigenstructure and directional orientations of weight matrices, allowing 1,024 master templates to rehydrate full layers with **$99.5\%$ Cosine Similarity**!

---

## 📊 Block Size Sweep Benchmark Results ($K=1024$)

| Block Size Configuration | Params / Block | Storage Footprint | Effective Bits / Param | Worst Cos Sim | Ref Mag | Rehydrated Mag | Mean Cos Sim | Execution Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 (Ref Ground Truth)** | — | `10.00 MB` | `32.0 bits` | `1.000000` | `19.1901` | `19.1901` | `1.000000` | — |
| **`32x32` Block** | **1024 params** | **`0.18 MB`** | **`0.97 bits`** | **`0.992744`** | `19.1901` | `19.4900` | **`0.995008` $\star$** | **2 seconds** |
| **`16x16` Block** | 256 params | `0.22 MB` | `1.18 bits` | `0.759902` | `12.2570` | `10.7454` | `0.829787` | 2 seconds |
| **`8x8` Block** | 64 params | `0.32 MB` | `1.72 bits` | `0.533746` | `14.7816` | `8.5104` | `0.609840` | 2 seconds |
| **`4x4` Block** | 16 params | `0.45 MB` | `2.42 bits` | `0.578909` | `15.3463` | `9.3311` | `0.661644` | 2 seconds |

---

## 🖼️ Codebook Analysis Plot

![Codebook Analysis Plot](codebook_analysis_plot.png)

- **Row 1 (Scatter Plots)**: Compares **Relative Magnitude Error (%) vs. Cosine Similarity** across block sizes (Left) and Temperature Annealing Curve $\tau$ (Right).
- **Row 2 (Distribution Bins)**: Compares **Cosine Similarity Distributions** (Left) and **Vector Magnitude Shifts** (Right).

---

## 📂 File Layout

- [`model.py`](model.py): Neural network architecture and `PureIndexCodebookQuantizer` module with configurable `block_h` and `block_w`.
- [`train_codebook.py`](train_codebook.py): FP32 baseline training and block size sweep scripts ($32 \times 32 \rightarrow 4 \times 4$).
- [`plot.py`](plot.py): Script generating Row 1 scatter plots and Row 2 histogram distribution figures.
