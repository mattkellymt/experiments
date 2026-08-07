# Neural Network Quantization Experiment

## 1. What is Quantization?

**Quantization:** Compressing neural network floating-point weights (FP32) down to 8-bit (FP8) or 4-bit (FP4) formats to reduce memory footprint by **up to 85%** with minimal loss in model quality.

---

## 2. Experiment Setup

* **Paradigms Benchmark (In-Memory):**
  * **RTN (Round-To-Nearest) [PTQ]:** Snaps FP32 weights to nearest FP8/FP4 grid point (0 training epochs).
  * **GPTQ (Hessian-Guided) [PTQ]:** Nudges unquantized weight columns using inverse Hessian $H^{-1}$ (0 training epochs).
  * **AWQ (Activation-Aware) [PTQ]:** Scales salient activation weight channels before quantization (0 training epochs).
  * **QAT (Quantization-Aware Training) [QAT]:** Fine-tunes FP32 weights with Straight-Through Estimators STE (50 training epochs).
* **Execution:** 100% in-memory (no `.pt` checkpoint files saved to disk).

---

## 3. Results & Experimental Fairness

| Quantization Method / Format | Paradigm | Memory Footprint | Worst Cosine Similarity | Mean Cosine Similarity | Winner |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FP32 (Ref Ground Truth)** | Reference | `10.0 MB` | `1.000000` | `1.000000` | Reference |
| **BF16 (16-Bit)** | 16-Bit | `5.0 MB` | `0.999992` | `0.999994` | — |
| **RTN FP8 (Round-To-Nearest)** | **PTQ** | `2.5 MB` | `0.999557` | `0.999664` | — |
| **GPTQ FP8 (Hessian)** | **PTQ** | `2.5 MB` | `0.999468` | `0.999622` | — |
| **AWQ FP8 (Channel)** | **PTQ** | `2.5 MB` | `0.999557` | **`0.999671`** | 🥇 8-Bit Winner |
| **QAT FP8 (STE)** | **QAT** | `2.5 MB` | `0.997823` | `0.998451` | — |
| **RTN FP4 (Round-To-Nearest)** | **PTQ** | `1.4 MB` | `0.993284` | `0.994953` | — |
| **GPTQ FP4 (Hessian)** | **PTQ** | `1.4 MB` | `0.991496` | `0.993804` | — |
| **AWQ FP4 (Channel)** | **PTQ** | `1.4 MB` | `0.992161` | `0.994903` | — |
| **QAT FP4 (STE)** | **QAT** | `1.4 MB` | **`0.994146`** | **`0.995625`** | 🥇 4-Bit Winner |

* **Experimental Bias Note:** On this small 2-layer MLP, activations lack extreme outliers ($>100\sigma$), so RTN artificially performs almost as well as AWQ/QAT. On real 70B+ LLMs, RTN fails without AWQ/GPTQ.

---

## 4. Visual Summary

![Visual Summary Plot](plot.png)

- **Left Column (All 4-Bit Methods):** Top: Scatter Plot (RTN, GPTQ, AWQ, QAT) | Bottom: Cosine Similarity Histogram Bins.
- **Right Column (All 8-Bit Methods):** Top: Scatter Plot (RTN, GPTQ, AWQ, QAT) | Bottom: Cosine Similarity Histogram Bins.

---

## Files

- [`run.py`](run.py) — Unified in-memory Python script
- [`plot.png`](plot.png) — 4-panel visual graphic
- [`README.md`](README.md) — Summary report
