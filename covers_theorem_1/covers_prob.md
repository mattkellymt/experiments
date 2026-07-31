## 1. The Math: Cover's Probability Formula

Cover solved the exact question of how likely a dataset of $N$ random points is to become **linearly separable** when projected into a $d$-dimensional space.

The probability $P(N, d)$ is given by the cumulative binomial distribution:

$$P(N, d) = \frac{1}{2^{N-1}} \sum_{k=0}^{d-1} \binom{N-1}{k}$$

When $N$ and $d$ become reasonably large, this sum transitions into a **Gaussian Cumulative Distribution Function (CDF)**:

$$P(N, d) \approx \Phi\left( \frac{d - N/2}{\sqrt{N/4}} \right)$$

where $\Phi(z)$ is the standard normal CDF.

---

## 2. The Sharp Phase Transition at $2\times$

Look closely at the term inside the CDF: $(d - N/2)$.

This tells you that the probability $P(N, d)$ undergoes a **sharp phase transition** around the ratio:

$$\frac{N}{d} = 2$$

```
Probability of Linear Separability P(N, d)
1.0 |-------------============== (d >= N or 2x/3x/4x)
    |            /
0.5 |           /  <-- Sharp Phase Transition occurs at N = 2d
    |          /
0.0 |=========-------------
    +------------------------------------> Ratio of Space to Data

```

* **If $d < \frac{N}{2}$:** The probability drops exponentially fast toward **0**.
* **At $d = \frac{N}{2}$:** The probability is **exactly $0.5$** (a 50/50 coin flip).
* **If $d > \frac{N}{2}$:** The probability jumps rapidly toward **1.0**.

---

## 3. Difference Between $2\times$, $3\times$, $4\times$, and $8\times$

The probability curve follows a Gaussian CDF, the jump from $0.5$ to $1.0$ is **steep**. You do not need an infinitely large space to get practically guaranteed linear separability.

Assuming $N$ data points in $d$-dimensional hidden space:

| Multiplier ($\frac{d}{N}$) | Standard Deviations ($\sigma$) from Mean | Probability of Separability $P(N, d)$ | Practical Result |
| --- | --- | --- | --- |
| **$0.5\times$** | $-N/2$ | **$\approx 0\%$** | Completely tangled; non-separable. |
| **$1.0\times$** (Critical Threshold) | $0\sigma$ | **$50.0\%$** | Right at the cliff. Half of random tasks work, half fail. |
| **$2.0\times$** | $+1.5\sigma$ to $+2\sigma$ | **$\approx 97.7\%$ to $99.9\%$** | **Virtually guaranteed.** This is why $2\times$ is the baseline. |
| **$3.0\times$** | $+3\sigma$ to $+4\sigma$ | **$\approx 99.999\%$** | Essentially $1.0$ for almost all practical purposes. |
| **$4.0\times$** | $> +5\sigma$ | **$> 99.99999\%$** | Standard in modern Transformer FFNs ($d_{ffn} = 4 \times d_{model}$). |
| **$8\times$** | Massive overkill | **$1.0 - \epsilon$** | Diminishing returns on separability; used only when raw capacity/memory storage is needed. |

If $2\times$ gets you to $\sim 99\%$ separability, why do modern models (like Transformers, SwiGLU, and MLPs) expand by **$4\times$** in their feed-forward layers?

1. **Real Data Is Not Random:** Cover's theorem assumes data is distributed in "general position" (randomly scattered). Real-world data lies on tight, highly curved manifolds. To untangle real-world manifolds, you need extra head room above the theoretical random baseline.
2. **Memorization & Capacity:** As shown by MacKay, the capacity limit of a perceptron is $2d$ items. Expanding to $4\times$ gives the network enough room to untangle multiple overlapping features *simultaneously* without them interfering with one another.
3. **The "Squeeze" (Linear Bottlenecks):** When you project back down (e.g., $4d \to d$), the network compresses the now-linearly-separated representations back into a dense space for the next layer to consume.