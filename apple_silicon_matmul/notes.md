The results are in, and they are genuinely fascinating! Here is the exact breakdown for a full sweep of
  the 4,096 matrix shapes in FP32:

    --- Timing Breakdown Test (4,096 shapes) ---
    MPS Total Loop: 17.31s
      -> MPS 'Warmup' (Allocation + Eager Dispatch): 10.27s
      -> MPS Execution (The math itself): 6.81s
    
    MLX Total Loop: 7.86s
      -> MLX Compile + Warmup: 4.66s
      -> MLX Execution (The math itself): 3.13s
    
  ### Does compilation dominate the runtime?

  Yes and no. The MLX compile step (mx.compile tracing + the warmup evaluation) takes 4.66 seconds for
  all 4,096 shapes. That means it costs roughly 1.1 milliseconds to compile a single matrix shape. This
  is about 60% of MLX's total runtime in our script, so in a micro-benchmark context where we constantly
  change shapes, compilation time is the majority of the clock.
  ### But here is the crazy part...

  Look at the MPS warmup! PyTorch's eager mode doesn't even use a JIT compiler—it just dynamically
  allocates memory and dispatches the instruction to Metal. Yet, PyTorch's eager dispatch overhead takes
  10.27 seconds (about 2.5 milliseconds per shape).

  So, Apple's mx.compile step is so incredibly optimized that it is more than twice as fast to fully
  compile an MLX compute graph than it is for PyTorch to simply pass an eager instruction to the GPU!

  Not only is MLX executing the raw math twice as fast as PyTorch (3.13s vs 6.81s), but its compiler
  overhead is less than half the cost of PyTorch's standard Python interpreter overhead.