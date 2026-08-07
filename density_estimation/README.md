# Authentic Un-Normalized Transformation & Residual Standard Deviation Analysis

## 1. Experimental Methodology ($M=4096$ Trials $\times N=2000$ Samples)

All functions and residual standard deviations are presented in their **authentic, un-normalized physical units** ($\mathcal{N}(\mu=1.5, \sigma=2.0)$).

For each grid point $x_j$, we computed the **Residual Standard Deviation** $\sigma_{\text{res}}(x_j)$ across all 4,096 trials in actual physical units on a **secondary right-hand Y-axis**:

$$\sigma_{\text{res}}(x_j) = \text{std}_{m=1 \dots 4096} \left( y_{\text{sample}}^{(m)}(x_j) - y_{\text{ideal}}(x_j) \right)$$

---

## 2. Physical Metrics per Pane

* **Pane 1 (Top-Left): Cumulative Distribution Function (CDF)**
  * **Function Y-Axis:** $P(X \le x) \in [0, 1.0]$
  * **Residual StdDev Y-Axis:** $\sigma_{\text{res}} \in [0, 0.0112]$ (Actual cumulative probability units)
* **Pane 2 (Top-Right): Probability Density Function (PDF)**
  * **Function Y-Axis:** $f(x) \in [0, 0.20]$
  * **Residual StdDev Y-Axis:** $\sigma_{\text{res}} \in [0, 0.0035]$ (Actual density height units)
* **Pane 3 (Bottom-Left): Integral of the CDF**
  * **Function Y-Axis:** $I(x) \in [0, 5.5]$
  * **Residual StdDev Y-Axis:** $\sigma_{\text{res}} \in [0.0082, 0.0450]$ (Actual integrated area units)
* **Pane 4 (Bottom-Right): First Derivative of the PDF**
  * **Function Y-Axis:** $f'(x) \in [-0.0619, +0.0619]$
  * **Residual StdDev Y-Axis:** $\sigma_{\text{res}} \in [0, 0.0028]$ (Actual slope units)

---

## 3. Visual Summary

![Visual Summary Plot](plot.png)

- **Shaded Color Band:** Min-Max sample bounds across 4,096 trials in real physical units.
- **Solid Black Line:** True ideal function across all 4 panels.
- **Solid Color Line (Right Y-Axis):** Residual Standard Deviation $\sigma_{\text{res}}(x)$ in authentic physical units.

---

## Files

- [`run.py`](run.py) — Entrypoint Python script
- [`plot.png`](plot.png) — Authentic physical units dual Y-axis 2x2 graphic
- [`README.md`](README.md) — Documentation report
