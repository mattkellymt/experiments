# Closed-Loop Algebraic Transformation Suite (CDF ↔ PDF)

## 1. Closed-Form Algebraic Formulas

1. **CDF to PDF Transformation $g(y)$:**
   $$g(y) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left( - \left[ \text{erfinv}(2y - 1) \right]^2 \right)$$

2. **PDF to CDF Transformation $y(f)$:**
   $$y(f) = \frac{1}{2} \left[ 1 \pm \text{erf}\left( \sqrt{ -\ln\left( f \cdot \sigma \sqrt{2\pi} \right) } \right) \right]$$

---

## 2. Experiment Setup ($N=2000$)

* **Sample Generation:** Drawn from $\mathcal{N}(\mu=1.5, \sigma=2.0)$.
* **Measured Parameters:** Sample Mean $\hat{\mu} = 1.5902$, Sample Std $\hat{\sigma} = 1.9769$.

---

## 3. Results & Closed-Loop Alignment

| Pipeline Stage | Mathematical Transformation | Alignment $R^2$ Score |
| :--- | :---: | :---: |
| **Pane 1: Empirical CDF** | $y_{\text{emp}} = \frac{i - 0.5}{N}$ (Sorted Ranks) | Reference Input |
| **Pane 2: CDF $\rightarrow$ PDF** | $g(y) = \frac{1}{\sigma \sqrt{2\pi}} e^{-[\text{erfinv}(2y-1)]^2}$ | **`R² = 0.999288`** 🥇 |
| **Pane 3: PDF $\rightarrow$ CDF** | $y(f) = \frac{1}{2}\left[1 \pm \text{erf}\left(\sqrt{-\ln(f \sigma \sqrt{2\pi})}\right)\right]$ | **`R² = 0.999817`** 🥇 |

---

## 4. Visual Summary

![Visual Summary Plot](plot.png)

- **Pane 1 (Left):** Empirical CDF dots $y_{\text{emp}} = \frac{i - 0.5}{N}$ vs. Theoretical CDF Line $y_{\text{theo}} = \Phi(x)$.
- **Pane 2 (Middle):** PDF Created Algebraically from CDF ($R^2 = 0.9993$).
- **Pane 3 (Right):** CDF Reconstructed Algebraically from PDF ($R^2 = 0.9998$).

---

## Files

- [`run.py`](run.py) — Entrypoint Python script
- [`plot.png`](plot.png) — 3-panel visual graphic
- [`README.md`](README.md) — Documentation report
