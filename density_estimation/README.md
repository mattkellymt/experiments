# Closed-Form Algebraic Transformation Suite (Ensemble Tracing)

## 1. Closed-Form Algebraic Formulas

1. **CDF to PDF Transformation $g(y)$:**
   $$g(y) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left( - \left[ \text{erfinv}(2y - 1) \right]^2 \right)$$

2. **CDF to CDF Integral Transformation $I(y, f)$:**
   $$I(y, f) = \int_{-\infty}^{x} \Phi(t) dt = \left[ \sigma \sqrt{2} \cdot \text{erfinv}(2y - 1) \right] \cdot y + \sigma^2 \cdot f$$

3. **PDF to Derivative Transformation $h(f)$:**
   $$h(f) = \mp \frac{f}{\sigma} \sqrt{ -2 \ln\left( f \cdot \sigma \sqrt{2\pi} \right) }$$

---

## 2. Ensemble Tracing Setup ($M=25$ Trials $\times N=2000$ Samples)

* **Multi-Trial Ensemble:** 25 independent random sample draws from $\mathcal{N}(\mu=1.5, \sigma=2.0)$.
* **Visual Rendering:** Each trial is rendered as a fine, transparent tracing line (`alpha=0.18`, `linewidth=0.7`) to visualize variance clouds around the bold ideal theoretical curve.

---

## 3. Visual Summary

![Visual Summary Plot](plot.png)

- **Top-Left (CDF):** 25 fine transparent ECDF sample trials tracing the bold ideal CDF curve.
- **Bottom-Left (CDF Integral):** 25 fine transparent algebraic CDF integral trials tracing the ideal integral curve.
- **Top-Right (PDF):** 25 fine transparent algebraic PDF trials tracing the bold ideal Gaussian PDF bell curve.
- **Bottom-Right (PDF Derivative):** 25 fine transparent algebraic derivative trials tracing the bold ideal PDF derivative wave.

---

## Files

- [`run.py`](run.py) — Entrypoint Python script
- [`plot.png`](plot.png) — 2x2 ensemble visual graphic
- [`README.md`](README.md) — Documentation report
