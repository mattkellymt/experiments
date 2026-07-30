from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors


def create_param(shape, dtype, device):
    param = nn.Parameter(torch.empty(shape, device=device, dtype=dtype))
    param = nn.ParameterDict({'weight': param})
    return param


class RMSNorm(nn.Module):
    def __init__(self, **config):
        super().__init__()
        hidden_size = config['hidden_size']
        rms_norm_eps = config['rms_norm_eps']
        dtype = config['dtype']
        device = config['device']
        self.rms_norm_eps = rms_norm_eps
        shape = (hidden_size,)
        self.weight = create_param(shape, dtype, device)

    def forward(self, x):
        var = x.pow(2).mean(-1, True)
        out = (x * torch.rsqrt(var + self.rms_norm_eps)) * self.weight.weight
        return out


class Attention(nn.Module):
    def __init__(self, **config):
        super().__init__()
        hidden_size = config['hidden_size']
        num_attention_heads = config['num_attention_heads']
        num_key_value_heads = config['num_key_value_heads']
        head_dim = config['head_dim']
        rope_theta = config['rope_theta']
        dtype = config['dtype']
        device = config['device']

        expected_kv_dim = num_key_value_heads * head_dim
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")

        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.q_dim = num_attention_heads * head_dim
        self.rope_theta = rope_theta
        self.device = device

        q_shape = (self.q_dim, hidden_size)
        kv_shape = (expected_kv_dim, hidden_size)
        o_shape = (hidden_size, self.q_dim)

        self.q_proj = create_param(q_shape, dtype, device)
        self.k_proj = create_param(kv_shape, dtype, device)
        self.v_proj = create_param(kv_shape, dtype, device)
        self.o_proj = create_param(o_shape, dtype, device)

    def rope(self, x):
        batch_size, num_heads, seq_len, head_dim = x.shape
        step_val = 2
        float32_dtype = torch.float32
        freq_exponents = torch.arange(0, self.head_dim, step_val, device=self.device, dtype=float32_dtype) / self.head_dim
        theta = 1.0 / (self.rope_theta ** freq_exponents)
        seq_idx = torch.arange(seq_len, device=self.device, dtype=float32_dtype)
        idx_theta = torch.outer(seq_idx, theta)
        cos = torch.cat((idx_theta.cos(), idx_theta.cos()), -1).to(x.dtype)
        sin = torch.cat((idx_theta.sin(), idx_theta.sin()), -1).to(x.dtype)

        half_dim = self.head_dim // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        neg_x2 = -x2
        rotate_half = torch.cat((neg_x2, x1), -1)
        out = (x * cos) + (rotate_half * sin)
        return out

    def gqa(self, q, k, v):
        batch_size, num_heads, seq_len, head_dim = q.shape
        n_rep = self.num_attention_heads // self.num_key_value_heads
        q_gqa = q.view(batch_size, self.num_key_value_heads, n_rep, seq_len, self.head_dim)
        k_gqa = k.unsqueeze(2)
        v_gqa = v.unsqueeze(2)
        is_causal = True
        attn_out = F.scaled_dot_product_attention(q_gqa, k_gqa, v_gqa, None, 0.0, is_causal)
        out = attn_out.reshape(batch_size, self.num_attention_heads, seq_len, self.head_dim)
        return out

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        q = F.linear(x, self.q_proj.weight).reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = F.linear(x, self.k_proj.weight).reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = F.linear(x, self.v_proj.weight).reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        out = self.gqa(q, k, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.q_dim)
        out = F.linear(out, self.o_proj.weight)
        return out


class MLP(nn.Module):
    def __init__(self, **config):
        super().__init__()
        hidden_size = config['hidden_size']
        intermediate_size = config['intermediate_size']
        dtype = config['dtype']
        device = config['device']

        gate_shape = (intermediate_size, hidden_size)
        up_shape = (intermediate_size, hidden_size)
        down_shape = (hidden_size, intermediate_size)

        self.gate_proj = create_param(gate_shape, dtype, device)
        self.up_proj = create_param(up_shape, dtype, device)
        self.down_proj = create_param(down_shape, dtype, device)

    def forward(self, x):
        gate = F.linear(x, self.gate_proj.weight)
        up = F.linear(x, self.up_proj.weight)
        out = F.linear(F.silu(gate) * up, self.down_proj.weight)
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
        hidden_size = config['hidden_size']
        num_hidden_layers = config['num_hidden_layers']
        tie_word_embeddings = config['tie_word_embeddings']
        dtype = config['dtype']
        device = config['device']

        embed_shape = (vocab_size, hidden_size)

        self.model = nn.ModuleDict({
            'embed_tokens': create_param(embed_shape, dtype, device),
            'norm': RMSNorm(**config),
            'layers': nn.ModuleList(Block(**config) for layer_idx in range(num_hidden_layers))
        })
        if tie_word_embeddings:
            self.lm_head = self.model['embed_tokens']
        else:
            self.lm_head = create_param(embed_shape, dtype, device)

    def forward(self, inputs):
        x = self.model.embed_tokens.weight[inputs]
        for layer in self.model.layers:
            x = layer(x)
        x = self.model.norm(x)
        logits = F.linear(x, self.lm_head.weight)
        return logits

    @torch.no_grad()
    def init_params(self):
        std_val = 0.02
        mean_val = 0.0
        one_val = 1.0
        for p in self.parameters():
            if p.ndim > 1:
                nn.init.normal_(p, mean_val, std_val)
            else:
                nn.init.ones_(p)


class Adam(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        defaults = dict(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def step_param(self, p, group):
        if p.grad is None:
            return
        lr = group['lr']
        b1, b2 = group['beta1'], group['beta2']
        eps = group['eps']
        wd = group['weight_decay']
        grad = p.grad
        state = self.state[p]
        if 'step' not in state:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p)
            state['exp_avg_sq'] = torch.zeros_like(p)

        state['step'] += 1
        t = state['step']
        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

        alpha_val = 1 - b1
        beta_val = 1 - b2
        exp_avg.mul_(b1).add_(grad, alpha=alpha_val)
        exp_avg_sq.mul_(b2).addcmul_(grad, grad, value=beta_val)

        step_size = lr * (math.sqrt(1 - b2 ** t) / (1 - b1 ** t))
        denom = exp_avg_sq.sqrt().add_(eps)

        neg_step_size = -step_size
        p.mul_(1 - lr * wd)
        p.addcdiv_(exp_avg, denom, value=neg_step_size)

    def step_group(self, group):
        for p in group['params']:
            self.step_param(p, group)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            self.step_group(group)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, momentum, nesterov, eps, ns_steps, adam_lr, adam_betas, adam_eps, adam_wd):
        params_list = list(params)
        muon_params = [p for p in params_list if p.ndim == 2]
        adam_params = [p for p in params_list if p.ndim != 2]
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            eps=eps,
            ns_steps=ns_steps,
        )
        super().__init__(muon_params, defaults)
        self.adam = Adam(adam_params, adam_lr, adam_betas, adam_eps, adam_wd)

    def step_newton_schulz(self, update, a, b, c):
        g = update @ update.T
        g_upd = torch.addmm(g, g, g, beta=b, alpha=c)
        alpha_one = 1.0
        update_next = torch.addmm(update, g_upd, update, beta=a, alpha=alpha_one)
        return update_next

    def newton_schulz(self, grad, eps, steps):
        a, b, c = 3.4445, -4.7750, 2.0315
        update = grad.bfloat16()
        is_transposed = grad.size(0) > grad.size(1)
        if is_transposed:
            update = update.T
        update.div_(update.norm().clamp(eps))
        for step_idx in range(steps):
            update = self.step_newton_schulz(update, a, b, c)
        if is_transposed:
            update = update.T
        return update

    def step_param(self, p, group):
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
        mom_weight = 1 - mom
        buf.lerp_(grad, mom_weight)
        update = grad.lerp(buf, mom) if nest else buf
        update = self.newton_schulz(update, eps, steps)
        ratio = max(1, p.shape[0] / p.shape[1])
        adj_lr = lr * math.sqrt(ratio)
        neg_adj_lr = -adj_lr
        p.mul_(1 - lr * wd)
        p.add_(update, alpha=neg_adj_lr)

    def step_group(self, group):
        for p in group['params']:
            self.step_param(p, group)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            self.step_group(group)
        self.adam.step()


def save_config(config, path):
    config_dict = {k: str(v) for k, v in config.items()}
    indent_val = 2
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=indent_val)


def load_config(path):
    with open(path, 'r') as f:
        config = json.load(f)
    return config


def save_model(model, path):
    save_safetensors(model.state_dict(), path)


def load_model(model, path, device):
    dev_str = str(device)
    sd = load_safetensors(path, dev_str)
    model_sd = model.state_dict()
    new_sd = {}
    for k, v in sd.items():
        if f"{k}.weight" in model_sd:
            new_sd[f"{k}.weight"] = v
        elif k in model_sd:
            new_sd[k] = v
    strict_flag = False
    model.load_state_dict(new_sd, strict_flag)


def train_step(model, optimizer, inputs, targets):
    optimizer.zero_grad()
    logits = model(inputs)
    vocab_size = logits.shape[-1]
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
    loss.backward()
    optimizer.step()
    loss_val = loss.item()
    return loss_val


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, temperature):
    device = next(model.parameters()).device
    messages = [{'role': 'user', 'content': prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(formatted_prompt)
    prompt_tokens = torch.tensor([prompt_ids], device=device)
    prompt_len = prompt_tokens.shape[1]

    input_shape = (1, prompt_len + max_new_tokens)
    long_dtype = torch.long
    input_ids = torch.empty(input_shape, long_dtype, device)
    input_ids[:, :prompt_len] = prompt_tokens

    for step_idx in range(max_new_tokens):
        current_len = prompt_len + step_idx
        logits = model(input_ids[:, :current_len])
        dim_val = -1
        if temperature <= 0.0:
            next_token = torch.argmax(logits[:, -1, :], dim_val)
        else:
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim_val)
            num_samples_val = 1
            next_token = torch.multinomial(probs, num_samples_val).squeeze(dim_val)

        input_ids[0, current_len] = next_token

    response_tokens = input_ids[0, prompt_len:].tolist()
    skip_special_val = True
    response_text = tokenizer.decode(response_tokens, skip_special_val)
    return response_text


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    name = "Llama-3.2-1B-Instruct"
    repo_id = f"unsloth/{name}"
    
    config_path = hf_hub_download(repo_id, "config.json")
    weights_path = hf_hub_download(repo_id, "model.safetensors")

    config = load_config(config_path)
    config.update({
        'device': device,
        'dtype': torch.bfloat16,
        'lr': 0.01,
        'weight_decay': 0.1,
        'momentum': 0.95,
        'max_new_tokens': 40,
        'temperature': 0.0,
        'config_path': config_path,
        'weights_path': weights_path,
    })

    model = Model(**config)
    load_model(model, weights_path, device)

    nesterov_val = True
    eps_val = 1e-7
    ns_steps_val = 5
    betas_val = (0.9, 0.999)
    adam_eps_val = 1e-8
    muon = Muon(
        model.parameters(),
        config['lr'],
        config['weight_decay'],
        config['momentum'],
        nesterov_val,
        eps_val,
        ns_steps_val,
        config['lr'],
        betas_val,
        adam_eps_val,
        config['weight_decay'],
    )
    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    prompt = "Explain how a transformer model uses multi-head self-attention to process text."
    print(f"Prompt: {prompt}")
    response_before = generate(model, tokenizer, prompt, config['max_new_tokens'], config['temperature'])
    print(f"Response:\n{response_before}\n")

    batch_size, seq_len = 2, 16
    vocab_size = config['vocab_size']
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    loss_val = train_step(model, muon, inputs, targets)

    response_after = generate(model, tokenizer, prompt, config['max_new_tokens'], config['temperature'])
    print(f"Response:\n{response_after}")


if __name__ == "__main__":
    main()