# Experimental Goal

To systematically benchmark matrix multiplication performance across Apple Silicon hardware engines (CPU, MPS, MLX) and precision types (FP32, FP16, BF16) using purely serial, synchronized execution. 

The core objectives are to:
1. Isolate the effects of matrix dimension alignment, memory coalescing, and precision on latency.
2. Produce a single globally normalized heatmap visualization that contrasts all hardware/precision permutations.
3. Establish a pristine "clean-room" methodology free of background concurrency or parallel resource contention.
