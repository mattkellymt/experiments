# Massive $65,536$-Trial Shaded Envelope & Residual Standard Deviation Analysis

## 1. Experimental Methodology ($M=65536 = 2^{16}$ Trials $\times N=2000$ Samples)

To achieve maximum statistical convergence, we executed **$M=65536$ ($2^{16}$) independent random sample trials** ($1.31072 \times 10^8$ total data points). We rendered a translucent shaded Min-Max bounding envelope across all 65,536 trials:

$$\text{Min-Max Shaded Envelope}(x) = \left[ \min_{m=1 \dots 65536} y_{\text{sample}}^{(m)}(x), \quad \max_{m=1 \dots 65536} y_{\text{sample}}^{(m)}(x) \right]$$

For each grid point $x_j$, we computed the **Residual Standard Deviation** $\sigma_{\text{res}}(x_j)$ across all 65,536 trials and plotted it as a solid line matching each pane's color theme on a **secondary right-hand Y-axis**:

$$\sigma_{\text{res}}(x_j) = \text{std}_{m=1 \dots 65536} \left( \text{residual}_m(x_j) \right)$$

---

## 2. Visual Layout & Color Breakdown

* **Pane 1 (Top-Left): Cumulative Distribution Function (CDF)**
  * **Shaded Band:** Blue Translucent Min-Max Envelope (`alpha=0.25`)
  * **Real Function:** Solid Black Ideal Line (`linewidth=2.5`)
  * **Residual StdDev:** Solid Blue Line (`linewidth=2.2`)
* **Pane 2 (Top-Right): Probability Density Function (PDF)**
  * **Shaded Band:** Orange Translucent Min-Max Envelope (`alpha=0.25`)
  * **Real Function:** Solid Black Ideal Line (`linewidth=2.5`)
  * **Residual StdDev:** Solid Orange Line (`linewidth=2.2`)
* **Pane 3 (Bottom-Left): Integral of the CDF**
  * **Shaded Band:** Green Translucent Min-Max Envelope (`alpha=0.25`)
  * **Real Function:** Solid Black Ideal Line (`linewidth=2.5`)
  * **Residual StdDev:** Solid Green Line (`linewidth=2.2`)
* **Pane 4 (Bottom-Right): First Derivative of the PDF**
  * **Shaded Band:** Purple Translucent Min-Max Envelope (`alpha=0.25`)
  * **Real Function:** Solid Black Ideal Line (`linewidth=2.5`)
  * **Residual StdDev:** Solid Purple Line (`linewidth=2.2`)

---

## 3. Visual Summary

![Visual Summary Plot](plot.png)

- **Shaded Color Band:** Global Min-Max sample bounds across 65,536 trials ($1.31072 \times 10^8$ total data points).
- **Solid Black Line:** True ideal reference function across all 4 panels.
- **Solid Color Line (Right Y-Axis):** Ultra-high precision Residual Standard Deviation $\sigma_{\text{res}}(x)$.

---

## Files

- [`run.py`](run.py) — Entrypoint Python script
- [`plot.png`](plot.png) — 65,536-trial dual Y-axis 2x2 graphic
- [`README.md`](README.md) — Documentation report
