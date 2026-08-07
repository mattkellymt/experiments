# GPTQ (Hessian-Guided) Quantization Experiments

Personal exploration repo analyzing post-training quantization using **GPTQ (Generalized Post-Training Quantization)** with Inverse Hessian nudging across **BF16**, **FP8**, and **FP4**.

---

## 🔍 How GPTQ Hessian Quantization Works Under the Hood

### 1. Calibration Pass ($X$)
Pass a small calibration dataset ($X$, 256 samples) through the model to record activation signals entering each `nn.Linear` layer.

### 2. Hessian Matrix Computation ($H$)
Compute the $N \times N$ Hessian matrix measuring feature sensitivity & co-activation:
$$H = \frac{1}{M} X^T X + \epsilon I$$

### 3. Inverse Hessian Matrix ($H^{-1}$)
Invert $H$ (via Cholesky decomposition or direct inversion) to determine the exact step sizes required to cancel out quantization errors:
$$H^{-1} = \text{inv}(H)$$

### 4. Sequential Quantization + Error Nudging ($\Delta W$)
For each column $q$ in the weight matrix:
1. Quantize column $w_q \rightarrow w_q^{\text{quant}}$ (FP8 or FP4 E2M1 grid).
2. Calculate rounding error $\delta_q = w_q - w_q^{\text{quant}}$.
3. Nudge remaining unquantized columns $j > q$ using the Inverse Hessian:
   $$w_j \longleftarrow w_j - \delta_q \cdot \left( \frac{H^{-1}_{q, j}}{H^{-1}_{q, q}} \right)$$

---

## 📊 Results Breakdown

| Variant | Precision | Effective Weight Size | Memory Reduction | Worst Cos Sim | Ref Mag | Variant Mag | Mean Cos Sim |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 (Ref)** | `32-bit` | `10.00 MB` | `0.0%` | `1.000000` | `16.2210` | `16.2210` | `1.000000` |
| **BF16** | `16-bit` | `5.00 MB` | `50.0%` | `0.999985` | `13.7367` | `13.7547` | `0.999989` |
| **GPTQ-FP8** | `8-bit` | `2.50 MB` | `75.0%` | `0.998483` | `17.3092` | `17.3167` | `0.999056` |
| **GPTQ-FP4** | `4-bit` | `1.41 MB` | `85.9%` | `0.975038` | `14.4973` | `14.4478` | `0.984627` |

---

## 🖼️ GPTQ Quantization Analysis Plot

![GPTQ Analysis Plot](gptq_analysis_plot.png)

---

## 📂 File Layout

- [`model.py`](model.py): Neural network model architecture & dataset generator.
- [`gptq_quant.py`](gptq_quant.py): FP32 training, BF16 conversion, and GPTQ quantization loops for FP8 and FP4.
- [`plot.py`](plot.py): Generates scatter plots and probability density distributions.
