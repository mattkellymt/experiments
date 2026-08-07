# Visualizing the Hessian Matrix & Eigenvector Learning

Personal exploration repo detailing the geometry of the Hessian matrix ($H = \frac{1}{N} X^T X$), loss landscapes, 2nd-order curvature, and learning eigenvectors via Gradient Descent.

---

## 📐 1. Why 2D Input $\rightarrow$ 3D Loss Bowl?

Consider a simple linear projection mapping 2D inputs $x = [x_1, x_2]^T$ to a 1D scalar output $y$:
$$y = w_1 x_1 + w_2 x_2 = \mathbf{w}^T \mathbf{x}$$

- The parameter space is **2D** $(w_1, w_2)$.
- The Loss $L(w_1, w_2)$ is a **scalar (1D)**:
  $$L(w_1, w_2) = \frac{1}{2N} \sum_{i=1}^N (w_1 x_{i1} + w_2 x_{i2} - y_i)^2$$
- Plotting $z = L(w_1, w_2)$ yields a **3D Surface Bowl** (a paraboloid).

---

## 🔬 2. Calculus: Gradient vs. Hessian

### Gradient Vector $\nabla L$ (First Derivatives)
Indicates the **steepest direction of increase** on the 3D loss surface:
$$\nabla L = \frac{1}{N} X^T (X \mathbf{w} - \mathbf{y})$$

### Hessian Matrix $H$ (Second Derivatives)
Measures the **3D Curvature (rate of change of the gradient)**:
$$H = \begin{bmatrix} \frac{\partial^2 L}{\partial w_1^2} & \frac{\partial^2 L}{\partial w_1 \partial w_2} \\[4pt] \frac{\partial^2 L}{\partial w_2 \partial w_1} & \frac{\partial^2 L}{\partial w_2^2} \end{bmatrix} = \frac{1}{N} X^T X$$

---

## 🖼️ 3. Loss Surface & Curvature Visualization

![Hessian Visualization](hessian_visualization.png)

1. **Panel 1 (3D Surface Bowl)**: Visualizes the 3D paraboloid $z = L(w_1, w_2)$ over weight space.
2. **Panel 2 (Contour Map & Eigenvectors)**: Displays elliptical loss contours with overlaid Hessian eigenvectors ($\mathbf{v}_1, \mathbf{v}_2$), scaled by their curvature eigenvalues ($\lambda_1, \lambda_2$).
3. **Panel 3 (Gradient Descent vs. Newton 1-Step Jump)**:
   - **Gradient Descent (Orange)**: Oscillates back and forth across the steep valley wall ($\lambda_2 = 4.75$) while making slow progress along the flat floor ($\lambda_1 = 0.47$).
   - **Newton-Hessian Jump (Green)**: Using $H^{-1} \nabla L$ accounts for both steepness and cross-coupling, jumping **straight to the global minimum in 1 step**:
     $$\mathbf{w}_{\text{newton}} = \mathbf{w}_0 - H^{-1} \nabla L$$

---

## 🔍 4. Learning Eigenvectors Directly via Gradient Descent

Can you learn/discover eigenvectors directly using PyTorch and Gradient Descent? **Yes!**

### The Invariant Direction Loss
Since an eigenvector satisfies $A \mathbf{v} = \lambda \mathbf{v}$, its direction **does not change** when transformed by $A$. Thus, $|\text{cosine\_similarity}(A \mathbf{v}, \mathbf{v})| = 1.0$.

$$\text{Loss}(\mathbf{v}) = 1.0 - \left| \text{cosine\_similarity}(A \mathbf{v}, \mathbf{v}) \right|$$

### Finding All $N$ Eigenvectors:
1. **Dominant Eigenvector ($\mathbf{v}_1$)**: Unconstrained optimization of $\text{Loss}(\mathbf{v}_1)$ naturally converges to the dominant eigenvector (largest eigenvalue $\lambda_1$).
2. **Secondary Eigenvector ($\mathbf{v}_2$)**: Optimize $\mathbf{v}_2$ while enforcing orthogonality to $\mathbf{v}_1$:
   $$\text{Loss}_2 = \big(1.0 - |\text{CosSim}(A \mathbf{v}_2, \mathbf{v}_2)|\big) + 10.0 \cdot (\mathbf{v}_1^T \mathbf{v}_2)^2$$

![Eigenvector Learning Visualization](eigenvector_learning_plot.png)

---

## 📂 File Layout

- [`hessian.py`](hessian.py): Script generating data, computing loss, gradient, Hessian matrix $H = \frac{1}{N} X^T X$, and eigenvalues/eigenvectors.
- [`plot.py`](plot.py): Script generating the 3D surface plot, contour maps, and optimization trajectories.
- [`learn_eigenvectors.py`](learn_eigenvectors.py): Script demonstrating learning both eigenvectors of a 2D matrix via Gradient Descent.
- [`plot_eigen_learning.py`](plot_eigen_learning.py): Script generating the eigenvector rotation vector field & convergence plots.
