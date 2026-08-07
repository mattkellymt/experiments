# Clean Shaded Envelope & Residual Standard Deviation Analysis Across 1,024 Trials

## 1. Experimental Methodology ($M=1024$ Trials $\times N=2000$ Samples)

To visualize sample bounds cleanly without cluttering the plot with individual trial lines, we used **`plt.fill_between()`** to render a translucent shaded Min-Max bounding band across all 1,024 independent random sample trials:

$$\text{Min-Max Shaded Envelope}(x) = \left[ \min_{m=1 \dots 1024} y_{\text{sample}}^{(m)}(x), \quad \max_{m=1 \dots 1024} y_{\text{sample}}^{(m)}(x) \right]$$

For each grid point $x_j$, we also computed the **Residual Standard Deviation** $\sigma_{\text{res}}(x_j)$ across all 1,024 trials and plotted it as a solid line matching each pane's color theme on a **secondary right-hand Y-axis**:

$$\sigma_{\text{res}}(x_j) = \text{std}_{m=1 \dots 1024} \left( \text{residual}_m(x_j) \right)$$

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

- **Shaded Color Band:** Ultra-clean sample bounds showing the full extent of sample variation across 1,024 trials.
- **Solid Black Line:** True ideal function across all 4 panels.
- **Solid Color Line (Right Y-Axis):** Residual Standard Deviation $\sigma_{\text{res}}(x)$ tracking cross-trial variance.

---

## Files

- [`run.py`](run.py) — Entrypoint Python script
- [`plot.png`](plot.png) — Clean shaded envelope dual Y-axis 2x2 graphic
- [`README.md`](README.md) — Documentation report
