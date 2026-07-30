import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=None, weight_decay=None, momentum=None, nesterov=None, eps=None, ns_steps=None):
        lr = 1e-3 if lr is None else lr
        weight_decay = 0.1 if weight_decay is None else weight_decay
        momentum = 0.95 if momentum is None else momentum
        nesterov = True if nesterov is None else nesterov
        eps = 1e-7 if eps is None else eps
        ns_steps = 5 if ns_steps is None else ns_steps
        super().__init__(params, dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            eps=eps,
            ns_steps=ns_steps,
        ))

    def addmm(self, inp, mat1, mat2, beta=None, alpha=None):
        beta = 1 if beta is None else beta
        alpha = 1 if alpha is None else alpha
        return beta * inp + alpha * (mat1 @ mat2)

    def newton_schulz_step(self, upd, a, b, c):
        g = upd @ upd.T
        g_upd = self.addmm(g, g, g, beta=b, alpha=c)
        return self.addmm(upd, g_upd, upd, beta=a)

    def newton_schulz(self, grad, eps, steps):
        a, b, c = 3.4445, -4.7750, 2.0315
        upd = grad.bfloat16()
        is_transposed = grad.size(0) > grad.size(1)
        if is_transposed:
            upd = upd.T
        upd.div_(upd.norm().clamp(min=eps))
        for step_idx in range(steps):
            upd = self.newton_schulz_step(upd, a, b, c)
        if is_transposed:
            upd = upd.T
        return upd

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
        upd = grad.lerp(buf, mom) if nest else buf
        upd = self.newton_schulz(upd, eps, steps)
        adj_lr = lr * math.sqrt(max(1, p.shape[0] / p.shape[1]))
        p.mul_(1 - lr * wd)
        p.add_(upd, alpha=-adj_lr)

    def step_group(self, group):
        for p in group['params']:
            self.step_param(p, group)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            self.step_group(group)


def create_param(*shape):
    return nn.ParameterDict({'weight': nn.Parameter(torch.empty(*shape))})


class RMSNorm(nn.Module):
    def __init__(self, vocab_dim, eps=None, **kwargs):
        eps = 1e-5 if eps is None else eps
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(vocab_dim))

    def forward(self, x):
        var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(var + self.eps).to(x.dtype)) * self.weight


class RoPE(nn.Module):
    def __init__(self, head_dim, rope_theta=None, rope_scaling=None, seq_len=None, **kwargs):
        rope_theta = 500000.0 if rope_theta is None else rope_theta
        seq_len = 2048 if seq_len is None else seq_len
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        if rope_scaling and rope_scaling.get("rope_type") == "llama3":
            factor = rope_scaling.get("factor", 32.0)
            low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
            high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
            orig_max = rope_scaling.get("original_max_position_embeddings", 8192)

            low_wavelen = orig_max / low_freq_factor
            high_wavelen = orig_max / high_freq_factor

            new_inv_freq = []
            for freq in inv_freq:
                wavelen = 2 * math.pi / freq.item()
                if wavelen < high_wavelen:
                    new_inv_freq.append(freq.item())
                elif wavelen > low_wavelen:
                    new_inv_freq.append(freq.item() / factor)
                else:
                    smooth = (orig_max / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                    new_freq = (1 - smooth) * (freq.item() / factor) + smooth * freq.item()
                    new_inv_freq.append(new_freq)
            inv_freq = torch.tensor(new_inv_freq, dtype=torch.float32)

        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer('cos', emb.cos(), persistent=False)
        self.register_buffer('sin', emb.sin(), persistent=False)

    def forward(self, position_ids):
        cos = self.cos[position_ids].unsqueeze(2)
        sin = self.sin[position_ids].unsqueeze(2)
        return cos, sin


class Attention(nn.Module):
    def __init__(self, vocab_dim, kv_dim, n_heads, n_kv_heads, head_dim, rope_theta=None, rope_scaling=None, seq_len=None, **kwargs):
        rope_theta = 500000.0 if rope_theta is None else rope_theta
        seq_len = 2048 if seq_len is None else seq_len
        super().__init__()
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")
        expected_kv_dim = n_kv_heads * head_dim
        if kv_dim != expected_kv_dim:
            raise ValueError(f"kv_dim ({kv_dim}) must equal n_kv_heads * head_dim ({expected_kv_dim})")

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.q_dim = n_heads * head_dim

        self.q_proj = create_param(vocab_dim, self.q_dim)
        self.k_proj = create_param(vocab_dim, expected_kv_dim)
        self.v_proj = create_param(vocab_dim, expected_kv_dim)
        self.o_proj = create_param(self.q_dim, vocab_dim)

    def forward(self, x, position_embeddings=None, attention_mask=None, past_key_value=None):
        b, s, d = x.shape
        q = torch.matmul(x, self.q_proj.weight).reshape(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = torch.matmul(x, self.k_proj.weight).reshape(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = torch.matmul(x, self.v_proj.weight).reshape(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            def apply_rope(x_to_rotate):
                x1 = x_to_rotate[..., :self.head_dim // 2]
                x2 = x_to_rotate[..., self.head_dim // 2:]
                rotate_x = torch.cat((-x2, x1), dim=-1)
                return (x_to_rotate * cos) + (rotate_x * sin)
            
            q = apply_rope(q.transpose(1, 2)).transpose(1, 2)
            k = apply_rope(k.transpose(1, 2)).transpose(1, 2)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v)
        s_total = k.shape[2]
        n_rep = self.n_heads // self.n_kv_heads
        
        q_gqa = q.view(b, self.n_kv_heads, n_rep, s, self.head_dim)
        k_gqa = k.unsqueeze(2)
        v_gqa = v.unsqueeze(2)

        out = F.scaled_dot_product_attention(
            q_gqa, k_gqa, v_gqa, 
            attn_mask=attention_mask,
            is_causal=(attention_mask is None and s == s_total)
        )
        
        out = out.reshape(b, self.n_heads, s, self.head_dim).transpose(1, 2).reshape(b, s, self.q_dim)
        return torch.matmul(out, self.o_proj.weight), new_kv


class MLP(nn.Module):
    def __init__(self, vocab_dim, hidden_dim, **kwargs):
        super().__init__()
        self.gate_proj = create_param(vocab_dim, hidden_dim)
        self.up_proj = create_param(vocab_dim, hidden_dim)
        self.down_proj = create_param(hidden_dim, vocab_dim)

    def forward(self, x):
        gate = torch.matmul(x, self.gate_proj.weight)
        up = torch.matmul(x, self.up_proj.weight)
        return torch.matmul(F.silu(gate) * up, self.down_proj.weight)


class Block(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_layernorm = RMSNorm(**kwargs)
        self.self_attn = Attention(**kwargs)
        self.post_attention_layernorm = RMSNorm(**kwargs)
        self.mlp = MLP(**kwargs)

    def forward(self, x, position_embeddings=None, attention_mask=None, past_key_value=None):
        attn_out, new_kv = self.self_attn(self.input_layernorm(x), position_embeddings=position_embeddings, attention_mask=attention_mask, past_key_value=past_key_value)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_kv


class Model(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.config = config
        vocab_size = config['vocab_size']
        vocab_dim = config['vocab_dim']
        n_layers = config['n_layers']

        self.model = nn.ModuleDict({
            'embed_tokens': create_param(vocab_size, vocab_dim),
            'rotary_emb': RoPE(**config),
            'norm': RMSNorm(**config),
            'layers': nn.ModuleList(Block(**config) for _ in range(n_layers))
        })

    def forward(self, inputs, position_ids=None, attention_mask=None, past_key_values=None, use_cache=False):
        b, s = inputs.shape
        if position_ids is None:
            past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            position_ids = torch.arange(past_length, past_length + s, dtype=torch.long, device=inputs.device).unsqueeze(0).expand(b, -1)
            
        position_embeddings = self.model.rotary_emb(position_ids)
        
        x = self.model.embed_tokens.weight[inputs]
        new_past_key_values = []
        for i, layer in enumerate(self.model.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, next_kv = layer(x, position_embeddings=position_embeddings, attention_mask=attention_mask, past_key_value=past_kv)
            new_past_key_values.append(next_kv)
            
        x = self.model.norm(x)
        logits = torch.matmul(x, self.model.embed_tokens.weight.T)
        if use_cache:
            return logits, new_past_key_values
        return logits

    def init_params(self):
        with torch.no_grad():
            for p in self.parameters():
                if p.ndim > 1:
                    nn.init.normal_(p, mean=0.0, std=0.02)
                else:
                    nn.init.ones_(p)

    def save(self, path):
        save_safetensors(self.state_dict(), path)

    def load(self, path, device=None):
        if not os.path.exists(path):
            return False
        dev = device or next(self.parameters()).device
        sd = load_safetensors(path, device=str(dev))
        
        if "lm_head.weight" in sd:
            sd.pop("lm_head.weight")
            
        self.load_state_dict(sd, strict=False)
        return True
