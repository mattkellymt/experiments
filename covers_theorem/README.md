# Dynamic High-Dimensional Experiment with Null Class Injection (Clean Inputs)

This experiment evaluates **Dynamic On-The-Fly Data Generation**, **Balanced Mini-Batch Training**, and **Modern 2D Representation Learning (InfoNCE)** when a **Null Class (Class 0: Standard Normal Background Noise $\mathcal{N}(0, I_8)$)** is injected into every mini-batch:

---

## 1. System Specifications

* **Number of Classes ($K$):** $K = 8$ total classes:
  * **Class 0 (Null Noise Class):** Unstructured Standard Normal Noise $\mathcal{N}(0, I_8)$ centered at origin ($c_0 = 0$, $\sigma_0 = 1.0$).
  * **Classes 1 to 7 (Signal Classes):** 7 clean structured Gaussian signal clusters ($c_k \sim \mathcal{N}(0, 3.5^2)$, $\sigma_k \sim |\mathcal{N}(0.6, 0.25)|$).
* **Input Feature Dimension ($d_{\text{in}}$):** $d_{\text{in}} = 8$ dimensions.
* **Balanced Mini-Batch:** Batch size = $K = 8$ (1 Null sample + 7 Signal samples per mini-batch).
* **Dynamically Sampled Test Set:** $N_{\text{test}} = 256$ samples (32 Null samples + 224 Signal samples).

---

## 2. Dynamic Training Dynamics Progression

| Step Range | Dynamic Mini-Batch Loss | Dynamic Test Accuracy (%) | Training State |
| :---: | :---: | :---: | :--- |
| **Step 1** | 2.4584 | 37.5% | Initial state |
| **Step 10** | 0.3059 | 89.8% | Learning background vs signal boundaries |
| **Step 20** | 0.2006 | 94.5% | High signal-to-noise separability |
| **Step 50** | 0.0189 | **100.0%** | **100.0% Test Acc (Null & Signal)** |
| **Step 100** | 0.0018 | **100.0%** | Final convergence plateau |

---

## 3. Results Summary

* **Overall Test Accuracy (Dynamically Sampled $N=256$):** **`100.00%`**
  * Null Class (Class 0) Accuracy: **`100.00%`** (32/32 correct)
  * Signal Classes (Classes 1–7) Accuracy: **`100.00%`** (224/224 correct)

---

## 4. Visual Graphic

![Dynamic High-D Graphic](dynamic_highd.png)

- **Panel 1:** Dynamic loss reduction (Red curve) and Test Accuracy growth (Cyan dashed line hitting 100.0% at Step 50).
- **Panel 2:** t-SNE 2D manifold projection of raw 8D input space, with the **White dots representing the Null Class $\mathcal{N}(0, I_8)$ at the origin**.
- **Panel 3:** **InfoNCE 2D Latent Space with Null Anchor**, showing the Null Class (White dot) and 7 Signal Classes cleanly organized around the unit circle.
- **Panel 4:** Confusion matrix on 256 dynamically sampled test points (100.00% Test Acc across Null and Signal classes).

---

## Files

- [`dynamic_highd.py`](dynamic_highd.py) — Python script for dynamic 8D Null Class experiment
- [`dynamic_highd.png`](dynamic_highd.png) — Generated 4-panel visual graphic
- [`README.md`](README.md) — Documentation report
