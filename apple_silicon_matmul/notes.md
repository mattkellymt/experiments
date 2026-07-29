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

  ## fp32 vs fp16 runtime

    That is a fantastic observation, and it highlights one of the most interesting quirks of Apple Silicon
  compared to traditional Nvidia GPUs!

  You are completely right to expect that FP16 should be twice as fast (because it's half the data size,
  meaning 2x the memory bandwidth and potentially 2x the SIMD vector packing). On an Nvidia GPU, FP16 is
  often 2x to 4x faster because of specialized Tensor Cores.

  So why doesn't FP16 crush FP32 on your M4 Pro? There are three main reasons:

  ### 1. The AMX Does Not Penalize FP32
  On traditional GPUs, chip manufacturers often artificially limit FP32 performance (to upsell enterprise
  cards) or dedicate massive silicon space purely to lower-precision AI. Apple took a different route.
  The Apple Matrix Coprocessor (AMX) on your M4 chip has massive, unthrottled FP32 execution units.
  Because the silicon is already so brutally efficient at blasting through 32-bit math, the theoretical
  "headroom" for 16-bit math to be faster is much smaller.

  ### 2. FP32 Accumulation (The Secret Math)
  Even when you explicitly tell PyTorch or MLX to multiply FP16 matrices, they almost never actually do
  pure FP16 math. To prevent catastrophic rounding errors when adding thousands of numbers together, the
  kernel will read the FP16 values, multiply them, but accumulate the running sum in FP32 hardware
  registers. Because the core execution units are still ultimately forced to do 32-bit math for the
  accumulation step, the throughput difference shrinks dramatically.

  ### 3. API Dispatch Overhead

  Our matrix is a 1024x1024 square. At these blistering speeds (around 0.179 milliseconds on MLX), a
  significant chunk of that time isn't actually doing math—it's the fixed API overhead of the CPU telling
  the GPU what to do and waiting for the pointer to return. Because this fixed overhead takes the exact
  same amount of time for FP32 as it does for FP16, it dilutes any percentage speedup you'd get from the
  math itself.

  If we tested a gigantic 16,384 x 16,384 matrix, the API overhead would become irrelevant, the memory
  bandwidth would become the primary bottleneck, and you would likely see FP16 pull further ahead of FP32
  simply because it uses half as much RAM!

  ## flops

    "ps" stands for picoseconds.

  To give you an idea of how small that is:

  • 1 second = 1,000 milliseconds (ms)
  • 1 millisecond = 1,000 microseconds (μs)
  • 1 microsecond = 1,000 nanoseconds (ns)
  • 1 nanosecond = 1,000 picoseconds (ps)
  So, 1 picosecond is one-trillionth of a second (10⁻¹² seconds)!
  ### What does "ps/FLOP" mean?

  FLOP stands for Floating-Point Operation (a single piece of math, like multiplying two decimals
  together).

  When our chart says the MLX benchmark ran at a minimum of ~0.083 ps/FLOP, it means that, on average, it
  took your M4 Pro chip 0.083 trillionths of a second to complete a single math operation.
  Obviously, a single silicon core can't physically do math that fast (the speed of light alone makes
  that tough!). But because your GPU has thousands of tiny cores all doing math simultaneously in
  parallel, the average time it takes to spit out one finished piece of math averages down to fractions
  of a picosecond.

  It's essentially the inverse of TFLOPs (Teraflops). Computing math at 0.083 ps/FLOP is mathematically
  identical to saying your chip is outputting 12 Trillion Operations Per Second (12 TFLOPs)!

  In 0.083 picoseconds, light travels 0.02488 mm in a vacuum and 0.01244 to 0.01493 mm as an electrical impulse inside a silicon chip, the Velocity of Propagation (VoP).Distance Estimates in MillimetersIn a Vacuum: 0.02488 mmInside a Chip: 0.01244 to 0.01493 mm