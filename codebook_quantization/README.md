# Experimental Quantized Key Transform Methods

Personal exploration repo testing your vision: **Using Low-Precision Quantized Weight Blocks ($W_{\text{q}}$) as KEYS to Route and Transform Non-Linear Feature Spaces into FP32 Rehydrated Matrices** optimized for Apple Silicon Metal GPU (`mps`).

---

## 💡 The 3 Architectural Methods Tested

```
                 1. KEY-VALUE CODEBOOK ROUTER (Method 1)
┌──────────────────────────────────────────────────────────────────────────┐
│  Keys: E_phi(W_q)  --> Softmax Sim against Key Codebook K.               │
│  Output: Rehydrates FP32 matrix from Value Codebook V.                   │
└──────────────────────────────────────────────────────────────────────────┘

              2. MULTI-HEAD QUANTIZED KEY ATTENTION (Method 2)
┌──────────────────────────────────────────────────────────────────────────┐
│  Keys: W_q partitioned into H=4 heads.                                   │
│  Output: Multi-head attention mixture over H codebook memories.         │
└──────────────────────────────────────────────────────────────────────────┘

             3. DEEP NON-LINEAR KEY PROJECTION NET (Method 3)
┌──────────────────────────────────────────────────────────────────────────┐
│  Keys: W_q passed through Deep MLP Key Projector f_theta(W_q).            │
│  Output: Maps key representations directly into FP32 basis matrices.     │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Benchmark Results: 3 Quantized Key Transform Methods

| Quantized Key Transform Method | Stored Representation | Storage Footprint | Effective Bits / Param | Worst Cos Sim | Ref Mag | Rehydrated Mag | Mean Cos Sim | Execution Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 (Ref Ground Truth)** | 32-Bit Float | `10.00 MB` | `32.0 bits` | `1.000000` | `16.0427` | `16.0427` | `1.000000` | — |
| **Method 1: Key-Value Codebook Router (FP4)** | **FP4 Grid + NN** | **`1.41 MB + NN`** | **`4.0 bits + NN`** | **`0.475211`** | `16.6830` | `10.1923` | **`0.572495` $\star$** | **2 seconds** |
| **Method 2: Multi-Head Key Attention (FP4)** | FP4 Grid + NN | `1.41 MB + NN` | `4.0 bits + NN` | `0.408168` | `15.3939` | `4.6857` | `0.549664` | 2 seconds |
| **Method 3: Deep Key Projection Net (FP4)** | FP4 Grid + NN | `1.41 MB + NN` | `4.0 bits + NN` | `-0.049274` | `14.0640` | `28.7974` | `0.001230` | 2 seconds |
| **Method 1: Key-Value Codebook Router (FP8)** | **FP8 Grid + NN** | **`2.62 MB + NN`** | **`8.0 bits + NN`** | **`0.482111`** | `12.1971` | `8.2196` | **`0.573451` $\star$** | **2 seconds** |
| **Method 2: Multi-Head Key Attention (FP8)** | FP8 Grid + NN | `2.62 MB + NN` | `8.0 bits + NN` | `-0.154820` | `16.3205` | `5.8949` | `0.412585` | 2 seconds |
| **Method 3: Deep Key Projection Net (FP8)** | FP8 Grid + NN | `2.62 MB + NN` | `8.0 bits + NN` | `-0.119939` | `13.9336` | `0.0010` | `-0.000299` | 2 seconds |

---

## 🖼️ Codebook Analysis Plot

![Codebook Analysis Plot](codebook_analysis_plot.png)

- **Row 1 (Scatter Plots)**: Compares **Relative Magnitude Error (%) vs. Cosine Similarity** for FP4 Methods (Left) and FP8 Methods (Right).
- **Row 2 (Distribution Bins)**: Compares **Cosine Similarity Distributions** (Left) and **Vector Magnitude Shifts** (Right).

---

## 📂 File Layout

- [`model.py`](model.py): Neural network architecture implementing Method 1 (`KVCodebookRouter`), Method 2 (`MHKeyAttentionRehydrator`), and Method 3 (`DeepKeyProjectionNetwork`).
- [`train_codebook.py`](train_codebook.py): FP32 baseline training and fine-tuning scripts across all 3 methods.
- [`plot.py`](plot.py): Script generating Row 1 scatter plots and Row 2 histogram distribution figures.
