import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.1, momentum=0.95, nesterov=True, eps=1e-7, ns_steps=5):
        super().__init__(params, dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            eps=eps,
            ns_steps=ns_steps,
        ))

    def step_newton_schulz(self, update, a, b, c):
        g = update @ update.T
        g_upd = torch.addmm(g, g, g, beta=b, alpha=c)
        update_next = torch.addmm(update, g_upd, update, beta=a, alpha=1.0)
        return update_next

    def newton_schulz(self, grad, eps, steps):
        a, b, c = 3.4445, -4.7750, 2.0315
        update = grad.bfloat16()
        is_transposed = grad.size(0) > grad.size(1)
        if is_transposed:
            update = update.T
        update.div_(update.norm().clamp(min=eps))
        for step_idx in range(steps):
            update = self.step_newton_schulz(update, a, b, c)
        if is_transposed:
            update = update.T
        return update

    def step_param(self, p, group):
        if p.ndim != 2:
            raise ValueError("Muon only supports 2D parameters")
        if p.grad is None:
            return
        lr = group['lr']
        wd = group['weight_decay']
        mom = group['momentum']
        nest = group['nesterov']
        eps = group['eps']
        steps = group['ns_steps']
        grad = p.grad
        state = self.state[p]
        if 'buf' not in state:
            state['buf'] = torch.zeros_like(grad)
        buf = state['buf']
        buf.lerp_(grad, 1 - mom)
        update = grad.lerp(buf, mom) if nest else buf
        update = self.newton_schulz(update, eps, steps)
        adj_lr = lr * math.sqrt(max(1, p.shape[0] / p.shape[1]))
        p.mul_(1 - lr * wd)
        p.add_(update, alpha=-adj_lr)

    def step_group(self, group):
        for p in group['params']:
            self.step_param(p, group)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            self.step_group(group)


def create_param(*shape):
    param = nn.ParameterDict({'weight': nn.Parameter(torch.empty(*shape))})
    return param


class RMSNorm(nn.Module):
    def __init__(self, **config):
        super().__init__()
        vocab_dim = config['vocab_dim']
        eps = config['eps']
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(vocab_dim))

    def forward(self, x):
        var = x.pow(2).mean(dim=-1, keepdim=True)
        out = (x * torch.rsqrt(var + self.eps)) * self.weight
        return out


class Attention(nn.Module):
    def __init__(self, **config):
        super().__init__()
        vocab_dim = config['vocab_dim']
        kv_dim = config['kv_dim']
        n_heads = config['n_heads']
        n_kv_heads = config['n_kv_heads']
        head_dim = config['head_dim']
        rope_theta = config['rope_theta']

        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")
        expected_kv_dim = n_kv_heads * head_dim
        if kv_dim != expected_kv_dim:
            raise ValueError("kv_dim must equal n_kv_heads * head_dim")

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.q_dim = n_heads * head_dim
        self.rope_theta = rope_theta

        self.q_proj = create_param(vocab_dim, self.q_dim)
        self.k_proj = create_param(vocab_dim, expected_kv_dim)
        self.v_proj = create_param(vocab_dim, expected_kv_dim)
        self.o_proj = create_param(self.q_dim, vocab_dim)

    def rope(self, x):
        seq_len = x.shape[-2]
        theta = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2, device=x.device, dtype=torch.float32) / self.head_dim))
        seq_idx = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        idx_theta = torch.outer(seq_idx, theta)
        cos = idx_theta.cos()
        sin = idx_theta.sin()

        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos
        return out

    def gqa(self, q, k, v):
        batch_size, num_heads, seq_len, head_dim = q.shape
        n_rep = self.n_heads // self.n_kv_heads
        q_gqa = q.view(batch_size, self.n_kv_heads, n_rep, seq_len, self.head_dim)
        k_gqa = k.unsqueeze(2)
        v_gqa = v.unsqueeze(2)
        out = F.scaled_dot_product_attention(q_gqa, k_gqa, v_gqa, is_causal=True)
        out = out.reshape(batch_size, self.n_heads, seq_len, self.head_dim)
        return out

    def forward(self, x):
        batch_size, seq_len, vocab_dim = x.shape
        q = torch.matmul(x, self.q_proj.weight).reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = torch.matmul(x, self.k_proj.weight).reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = torch.matmul(x, self.v_proj.weight).reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        out = self.gqa(q, k, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.q_dim)
        out = torch.matmul(out, self.o_proj.weight)
        return out


class MLP(nn.Module):
    def __init__(self, **config):
        super().__init__()
        vocab_dim = config['vocab_dim']
        hidden_dim = config['hidden_dim']

        self.gate_proj = create_param(vocab_dim, hidden_dim)
        self.up_proj = create_param(vocab_dim, hidden_dim)
        self.down_proj = create_param(hidden_dim, vocab_dim)

    def forward(self, x):
        gate = torch.matmul(x, self.gate_proj.weight)
        up = torch.matmul(x, self.up_proj.weight)
        out = torch.matmul(F.silu(gate) * up, self.down_proj.weight)
        return out


class Block(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.input_layernorm = RMSNorm(**config)
        self.self_attn = Attention(**config)
        self.post_attention_layernorm = RMSNorm(**config)
        self.mlp = MLP(**config)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Model(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.config = config
        vocab_size = config['vocab_size']
        vocab_dim = config['vocab_dim']
        n_layers = config['n_layers']

        self.model = nn.ModuleDict({
            'embed_tokens': create_param(vocab_size, vocab_dim),
            'norm': RMSNorm(**config),
            'layers': nn.ModuleList(Block(**config) for _ in range(n_layers))
        })
        self.lm_head = create_param(vocab_dim, vocab_size)

    def forward(self, inputs):
        x = self.model.embed_tokens.weight[inputs]
        for layer in self.model.layers:
            x = layer(x)
        x = self.model.norm(x)
        logits = torch.matmul(x, self.lm_head.weight)
        return logits

    @torch.no_grad()
    def init_params(self):
        for p in self.parameters():
            if p.ndim > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)
            else:
                nn.init.ones_(p)

    def save(self, path):
        save_safetensors(self.state_dict(), path)

    def load(self, path, device=None):
        dev = device or next(self.parameters()).device
        sd = load_safetensors(path, device=str(dev))
        self.load_state_dict(sd, strict=False)


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dtype = torch.bfloat16

    config = {
        'n_layers': 2,
        'batch_size': 4,
        'n_heads': 8,
        'n_kv_heads': 2,
        'head_dim': 16,
        'seq_len': 32,
        'vocab_dim': 128,
        'kv_dim': 32,
        'hidden_dim': 256,
        'vocab_size': 512,
        'eps': 1 / 1024,
        'rope_theta': 10_000.0,
        'lr': 0.01,
        'momentum': 0.95,
        'weight_decay': 0.1,
        'loss_target': 1 / 4096,
        'dtype': dtype,
        'device': device,
    }

    model = Model(**config).to(device=device, dtype=dtype)


if __name__ == "__main__":
    main()