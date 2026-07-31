# Apple Silicon M4 Pro: Empirical GPU Optimization Guide

This whitepaper details an empirical, black-box reverse-engineering study of Apple Silicon's GPU performance.

## 1. Core ML Feasibility & API Overhead

We first investigated the feasibility of using Core ML via `coremltools` for sub-millisecond, low-level matrix multiplication (GEMM) dispatch. A baseline $2048 \times 2048$ matmul was compiled to an `.mlpackage` and evaluated against four explicit Apple Compute Units:

*   **`CPU_ONLY`**: `6.11 ms`
*   **`CPU_AND_NE`**: `3.96 ms`
*   **`ALL`**: `3.72 ms`
*   **`CPU_AND_GPU`**: `12.57 ms`

### Findings & Limitations
The Core ML prediction API overhead is massive for isolated matrix math. Explicitly targeting the GPU via `CPU_AND_GPU` performed worse than `CPU_ONLY`. This overhead (cache syncs, dispatch translation, metal command buffering) functionally cripples the hardware. 
While `CPU_AND_NE` was the fastest, this merely indicates the *scheduler allowed* Neural Engine usage; it does not definitively prove the ANE executed the pure matmul isolated from the CPU's AMX coprocessor. Because Core ML enforces static graph compilation and abstracts low-level dispatch, it was abandoned in favor of PyTorch MPS and Apple MLX for granular dimension sweeping.

## 2. Matrix Dimension Sweeps

We executed a massive parameter sweep, independently modifying the $M$, $N$, and $K$ dimensions of the GEMM operation.

*   **1D Sweeps**: High-resolution (step size = 1) sweeps holding two dimensions at 2048 and extending the third to 2560.
*   **2D Sweeps**: Bivariate grids (step size = 4) comparing $M \times N$, $M \times K$, and $N \times K$ across a 256-element range.

See Cache 1D
See Cache 2D

### Key Observations & Competing Hypotheses
1.  **Framework Uniformity**: MLX demonstrates profoundly smoother, more uniform execution latencies than PyTorch MPS across virtually all unaligned shapes.
2.  **Structural Periodicity**: The PyTorch MPS grid exhibits severe, hard-edged alignment penalties. However, in the $M \times K$ grid, we observe that shifting $K$ fundamentally alters the optimal alignment mapping of $M$ (exhibiting diagonal or shifted banding rather than pure vertical/horizontal stripes). 
3.  **Hypotheses**: 
    *   *Cache Geometry*: The penalties could be mapping directly to physical cache line limits or threadgroup memory (shared memory) bank conflicts.
    *   *Kernel Heuristics*: Given the shifting hot-spots when interacting with $K$, it is highly likely that the PyTorch compiler is falling back to poorly-optimized, generalized tiling kernels when the dimensions fail to divide cleanly by the Apple SIMD group size (32 threads). MLX's compiler likely JIT-compiles tighter, specialized kernels for these edge cases.

## 3. Hot/Cold Determinism & Variance Analysis

To prove that the observed cache penalties were structural to the matrix shape and not temporal artifacts (e.g., thermal throttling, OS scheduling), we executed a highly rigorous interleaved determinism study.
We extracted the top 5 fastest ("Cold"), top 5 slowest ("Hot"), and 5 median ("Median") shape configurations. We interleaved and randomized 150 executions per shape to eliminate temporal bias.

See Determinism Distributions

### Results
The distributions are phenomenally tight. 
*   **MPS Coefficient of Variation (CV)**: Ranged between $1.8\%$ and $6.8\%$.
*   **MLX Coefficient of Variation (CV)**: Ranged between $1.0\%$ and $4.9\%$.

The randomized test proved definitively that the latency gaps (e.g., jumping from ~3.0 ms to ~3.8 ms) are strictly deterministic properties of the matrix dimensionality.

## Conclusion

When optimizing workloads for the M4 Pro GPU:
1.  **Framework Choice**: MLX is demonstrably superior at handling arbitrary, unaligned matrix dimensions compared to PyTorch MPS.
2.  **Alignment**: If using PyTorch MPS, developers must rigorously pad operations to multiples of 32 or 64 to avoid triggering severely unoptimized fallback kernels or catastrophic cache penalties.
3.  **Performance Metrics**: When evaluating Apple Silicon throughput, GEMM math complexity must be calculated as $(2 \times M \times N \times K)$ to accurately convert latency into true TFLOP/s or ps/FLOP.
