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

## SwiGLU
The classic $4 \times d_{model}$ rule was designed for the standard Feed-Forward Network (FFN) layout using activations like ReLU or GELU. [1] 
However, almost all popular open models today (like Meta's Llama 3, Mistral, and Qwen) use an architectural upgrade called SwiGLU (Gated Linear Units). Because SwiGLU splits the up-projection into two separate matrices ($W_{gate}$ and $W_{up}$), it is mathematically more expressive. To prevent the model from becoming bloated with too many parameters, researchers scaled down the intermediate dimension factor. [2] 
## The Actual Ratios in Popular Ollama Models
Instead of a clean 4x multiplier, modern open models calculate their intermediate FFN dimension ($d_{ffn}$) using a formula like:
$$d_{ffn} \approx \frac{8}{3} \times d_{model} \approx 2.67 \times d_{model}$$ 
Here is exactly how the math breaks down for the most popular models you pull on [Ollama](https://ollama.com/blog/vision-models): [3] 

* Llama 3 / 3.1 / 3.2 (8B):
* Hidden size ($d_{model}$): 4,096
   * FFN size ($d_{ffn}$): 14,336
   * Actual Ratio: ~3.5x [4] 
* Llama 3 (70B):
* Hidden size ($d_{model}$): 8,192
   * FFN size ($d_{ffn}$): 28,672
   * Actual Ratio: ~3.5x
* Mistral (7B) & Qwen 2.5 (7B):
* Hidden size ($d_{model}$): 4,096
   * FFN size ($d_{ffn}$): 11,008
   * Actual Ratio: ~2.68x (The exact $\frac{8}{3}$ ratio) [1] 

