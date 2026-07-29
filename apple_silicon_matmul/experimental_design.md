# Apple Silicon Matrix Multiplication: Clean-Room Experimental Design

## 1. Abstract
This document outlines a rigorous, single-threaded profiling methodology to evaluate the performance characteristics of matrix multiplication on Apple Silicon (M4 Pro). The focus is on strictly serializing execution across different compute engines (CPU, MPS, MLX) and numerical precisions (FP32, FP16, BF16) to avoid any hardware resource contention, thread clobbering, or artificial thermal throttling caused by asynchronous or parallel processes. The output will be a 9-panel, globally normalized heatmap demonstrating absolute and relative latency differences.

## 2. Experimental Objectives
1. **Engine Comparison:** Compare the execution latency of matrix multiplication on CPU (PyTorch), MPS (PyTorch GPU), and MLX (native Apple MLX GPU).
2. **Precision Comparison:** Assess the performance impact of FP32, FP16, and BF16 (if available/supported) across the available engines.
3. **Alignment & Coalescing Profiling:** Sweep the inner dimension ($K$) of matrix multiplication ($M \times K$ and $K \times N$) from unaligned shapes (e.g., odds, non-powers-of-two) to perfectly aligned shapes (multiples of 32/64) to visualize cache alignment and SIMD register effects.
4. **Global Normalization:** Present the final results in a globally normalized color space ($[-1, +1]$ scaling over the entire dataset) to visually contrast cross-engine and cross-precision performance fairly on a single chart.

## 3. Hardware Profiling & Architectural Mapping
*   **Target Hardware:** Apple M4 Pro (ARM64), Unified Memory Architecture.
    *   **GPU Cores:** 16 Cores (as profiled via `system_profiler`)
    *   **Memory:** 24 GB Unified Memory
*   **CUDA Mapping for M4 Pro:**
    *   **SIMD Group** = **Warp** (Size: 32 threads)
    *   **Threadgroup** = **Thread Block** (synchronization unit with shared memory)
    *   **Execution Unit / ALU** = **CUDA Core**
    *   **Core** = **Streaming Multiprocessor (SM)**
*   **Concurrency Prohibition:** **Crucial Constraint.** All operations MUST be executed purely sequentially. Parallel processing is strictly prohibited to prevent thermal throttling and memory bandwidth contention.
*   **Compilation:** All PyTorch code utilizes native MPS execution. MLX relies on `mx.compile()` to JIT compile optimized kernels.

## 4. Test Matrix
The test suite consists of 9 discrete experiments (3 Engines $\times$ 3 Precisions):

| Engine (Framework) | FP32 Supported | FP16 Supported | BF16 Supported |
| :--- | :---: | :---: | :---: |
| **CPU** (PyTorch) | Yes | Yes (Bfloat16 favored, FP16 fallback) | Yes |
| **MPS** (PyTorch) | Yes | Yes | Yes (M-series support) |
| **MLX** (Apple) | Yes | Yes | Yes |

*Note: The script will dynamically probe capability and safely fallback or mark as "Unsupported" if a specific precision (e.g., MLX BF16) is not natively supported by the framework/hardware combination, displaying an empty or grayed-out subplot.*

## 5. Statistical Side-Experiment & Matrix Dimensions
To empirically determine the necessary sample size, a statistical side-experiment was conducted:
*   **Methodology:** Matrix multiplication ($2048 \times 2048$) was run for sample sizes $N \in \{5, 10, 20, 50, 100, 200\}$.
*   **Finding:** The Coefficient of Variation (CV) drops well below 1.0% around 20 iterations (e.g., 0.59% for MPS, 0.55% for MLX). Running more iterations (e.g., 100+) slightly increased the standard deviation due to thermal variance and background OS noise.
*   **Conclusion:** The optimal batch size is $B=20$ iterations per matrix size.

### Architecturally Informed Sweeping Strategy
To accurately capture memory coalescing effects around the SIMD group boundary (32 threads):
*   Fixed inner dimension: $K = 2048$ (A common size for LLM layers).
*   Variable outer dimensions: $M$ and $N$ swept from $2048$ to $2048 + 32$ (a full SIMD group, length 33).
*   This sweep range crosses the exact 32-thread SIMD alignment boundary, visualizing unaligned cache penalties.

## 6. Execution Methodology
To guarantee clean measurements:
1.  **Warm-up:** For every combination of (Engine, Precision, $K$), the function will execute 5 warm-up iterations. This ensures caching, dynamic compilation (e.g., `torch.compile`), and JIT processes are completed before the stopwatch begins.
2.  **Synchronization:**
    *   **CPU:** Inherently synchronous.
    *   **MPS:** `torch.mps.synchronize()` MUST be called before starting the timer and immediately before stopping the timer.
    *   **MLX:** `mx.eval(result)` MUST be used to force evaluation of the lazy computational graph before stopping the timer.
3.  **Garbage Collection:** `gc.collect()` will be called between major engine transitions.

## 7. Visualization Strategy
*   Data will be collected into a centralized structured format (e.g., pandas DataFrame or nested JSON).
*   A Python plotting script (using `matplotlib` and `seaborn`) will generate a $3 \times 3$ grid of heatmaps.
*   The minimum and maximum recorded latencies across ALL 9 experiments will establish the global `vmin` and `vmax`.
*   A single shared colorbar will represent the global scale.
*   Missing or unsupported datatypes will render as a distinct hatched or empty region with an explicit "Unsupported" label.

## 8. Implementation Steps (Clean-Room Plan)
1.  **Probe & Setup:** Write utility functions to verify engine availability (MPS, MLX) and determine precision support.
2.  **Benchmarking Core:** Implement the `run_benchmark` function with strict synchronization rules as defined in section 6.
3.  **Experiment Runner:** Implement the nested loops (Engine $\rightarrow$ Precision $\rightarrow$ K-dimension) ensuring strict serial ordering.
4.  **Data Serialization:** Save the raw benchmark results to a JSON file.
5.  **Plotting:** Read the JSON, compute global min/max, and render the 9-panel PNG.
