import torch
import triton
import triton.language as tl

@triton.jit
def adamw_kernel(
    param_ptr,       # Pointer to Model Weights (W)
    grad_ptr,        # Pointer to Gradients (dL/dW)
    exp_avg_ptr,     # Pointer to Adam 1st Moment (m)
    exp_avg_sq_ptr,  # Pointer to Adam 2nd Moment (v)
    lr, beta1, beta2, eps, weight_decay, step,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 1. Read EVERYTHING from VRAM in one single memory transaction
    w = tl.load(param_ptr + offsets, mask=mask)
    g = tl.load(grad_ptr + offsets, mask=mask)
    m = tl.load(exp_avg_ptr + offsets, mask=mask)
    v = tl.load(exp_avg_sq_ptr + offsets, mask=mask)

    # 2. Apply Weight Decay
    w = w * (1.0 - lr * weight_decay)

    # 3. Update 1st and 2nd Moments (Adam equations)
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * (g * g)

    # 4. Compute Bias Corrections
    bias_correction1 = 1.0 - tl.exp(step * tl.log(beta1))
    bias_correction2 = 1.0 - tl.exp(step * tl.log(beta2))
    
    denom = (tl.sqrt(v) / tl.sqrt(bias_correction2)) + eps
    step_size = lr / bias_correction1

    # 5. Compute new weight
    w = w - step_size * (m / denom)

    # 6. Store updated values back in-place!
    tl.store(param_ptr + offsets, w, mask=mask)
    tl.store(exp_avg_ptr + offsets, m, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, v, mask=mask)
