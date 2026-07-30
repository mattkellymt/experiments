import argparse
import json
import math
import os
import random
import time
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

from architecture import Model, Muon

# ==========================================
# 1. Configuration & Setup
# ==========================================

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

dtype = torch.bfloat16

DEFAULT_CONFIG = {
    "n_layers": 16,
    "n_heads": 32,
    "n_kv_heads": 8,
    "head_dim": 64,
    "seq_len": 2048,
    "vocab_dim": 2048,
    "kv_dim": 512,
    "hidden_dim": 8192,
    "vocab_size": 128256,
    "eps": 1e-5,
    "rope_theta": 500000.0,
    "muon_lr": 0.01,
    "momentum": 0.95,
    "weight_decay": 0.01,
    "loss_target": 1e-4,
    "rope_scaling": {
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    }
}

HF_REPO_ID = "unsloth/Llama-3.2-1B-Instruct"
TEACHER_CHECKPOINT_PATH = "teacher_llama3_2_1b.safetensors"
STUDENT_CHECKPOINT_PATH = "student_llama3_2_1b.safetensors"
CONFIG_PATH = "llama3_2_1b.json"

META_PROMPTS = [
    "Ask a complex, fascinating question about cell biology, genetics, or molecular biology.",
    "Ask a substantive question about biochemistry, enzymes, or metabolic pathways.",
    "Ask an intriguing question about evolutionary biology, speciation, or phylogenetics.",
    "Ask a detailed question about neurobiology, synaptic transmission, or neuroscience.",
    "Ask a challenging question about microbiology, extremophiles, or immunology.",
    "Ask a deep question about developmental biology, morphogenesis, or stem cells."
]


# ==========================================
# 2. Optimized Model Helpers
# ==========================================

def ensure_model_weights():
    if not os.path.exists(TEACHER_CHECKPOINT_PATH) and os.path.exists("llama3_2_1b.safetensors"):
        os.rename("llama3_2_1b.safetensors", TEACHER_CHECKPOINT_PATH)

    if os.path.exists(TEACHER_CHECKPOINT_PATH) and os.path.exists(CONFIG_PATH):
        return

    print(f"Preparing transparent Teacher model weights for {HF_REPO_ID}...", flush=True)
    hf_path = hf_hub_download(repo_id=HF_REPO_ID, filename="model.safetensors")
    sd = load_safetensors(hf_path)

    mapped_sd = {
        "model.embed_tokens.weight": sd["model.embed_tokens.weight"].to(dtype).contiguous(),
        "model.norm.weight": sd["model.norm.weight"].to(dtype).contiguous(),
        "lm_head.weight": sd.get("lm_head.weight", sd["model.embed_tokens.weight"]).T.to(dtype).contiguous(),
    }

    for i in range(16):
        mapped_sd[f"model.layers.{i}.input_layernorm.weight"] = sd[f"model.layers.{i}.input_layernorm.weight"].to(dtype).contiguous()
        mapped_sd[f"model.layers.{i}.self_attn.q_proj.weight"] = sd[f"model.layers.{i}.self_attn.q_proj.weight"].T.to(dtype).contiguous()
        mapped_sd[f"model.layers.{i}.self_attn.k_proj.weight"] = sd[f"model.layers.{i}.self_attn.k_proj.weight"].T.to(dtype).contiguous()
        mapped_sd[f"model.layers.{i}.self_attn.v_proj.weight"] = sd[f"model.layers.{i}.self_attn.v_proj.weight"].T.to(dtype).contiguous()
        mapped_sd[f"model.layers.{i}.self_attn.o_proj.weight"] = sd[f"model.layers.{i}.self_attn.o_proj.weight"].T.to(dtype).contiguous()
        mapped_sd[f"model.layers.{i}.post_attention_layernorm.weight"] = sd[f"model.layers.{i}.post_attention_layernorm.weight"].T.to(dtype).contiguous()
        mapped_sd[f"model.layers.{i}.mlp.gate_proj.weight"] = sd[f"model.layers.{i}.mlp.gate_proj.weight"].T.to(dtype).contiguous()
        mapped_sd[f"model.layers.{i}.mlp.up_proj.weight"] = sd[f"model.layers.{i}.mlp.up_proj.weight"].T.to(dtype).contiguous()
        mapped_sd[f"model.layers.{i}.mlp.down_proj.weight"] = sd[f"model.layers.{i}.mlp.down_proj.weight"].T.to(dtype).contiguous()

    save_safetensors(mapped_sd, TEACHER_CHECKPOINT_PATH)
    with open(CONFIG_PATH, "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
    print(f"Converted and saved Teacher weights to '{TEACHER_CHECKPOINT_PATH}'.", flush=True)


@torch.no_grad()
def generate_dynamic_prompt(teacher, tokenizer, temperature=0.8):
    """Step A: Generate a dynamic biology prompt using Teacher with non-zero temperature"""
    meta_prompt = random.choice(META_PROMPTS)
    prompt_encoding = tokenizer.apply_chat_template(
        [{"role": "user", "content": meta_prompt}],
        add_generation_prompt=True,
        return_tensors="pt"
    )
    input_ids = prompt_encoding["input_ids"].to(device=device)
    curr = input_ids
    eos_ids = {128009, 128001}

    for _ in range(32):
        logits = teacher(curr)
        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        curr = torch.cat([curr, next_token], dim=1)
        if next_token.item() in eos_ids:
            break

    gen_ids = curr[0][input_ids.shape[1]:].tolist()
    prompt_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return prompt_text if len(prompt_text) > 10 else "What is the process by which enzymes catalyze biological reactions?"


@torch.no_grad()
def run_inference_zero_temp(prompt, model, tokenizer, max_new_tokens=96):
    """Step B: Run inference on prompt using temperature 0.0 (greedy decoding)"""
    encoding = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device=device)
    curr_tokens = input_ids
    eos_token_ids = {128009, 128001}

    for _ in range(max_new_tokens):
        logits = model(curr_tokens)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        curr_tokens = torch.cat([curr_tokens, next_token], dim=1)
        if next_token.item() in eos_token_ids:
            break

    gen_token_ids = curr_tokens[0][input_ids.shape[1]:].tolist()
    return tokenizer.decode(gen_token_ids, skip_special_tokens=True).strip()


# ==========================================
# 3. Continuous Training & Metric Logging Loop
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Continuous Dynamic Biology Prompt JSD + Lowercase Doc Loss Trainer")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate for Muon (default: 0.001)")
    args = parser.parse_args()

    ensure_model_weights()

    tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID)
    vocab_size = DEFAULT_CONFIG["vocab_size"]
    JSD_MAX = math.log(2)  # ln(2) ~ 0.693147

    print("\n--- Loading Unmodified Teacher Model (Frozen Reference) ---", flush=True)
    teacher = Model(**DEFAULT_CONFIG).to(device=device, dtype=dtype)
    teacher.load(TEACHER_CHECKPOINT_PATH, device=device)
    teacher.train()

    print("--- Loading Unmodified Student Model (Fresh Restart from Teacher Checkpoint) ---", flush=True)
    student = Model(**DEFAULT_CONFIG).to(device=device, dtype=dtype)
    student.load(TEACHER_CHECKPOINT_PATH, device=device)
    student.train()

    optimizer = Muon(
        (p for p in student.parameters() if p.ndim == 2),
        lr=args.lr,
        momentum=DEFAULT_CONFIG["momentum"],
        weight_decay=DEFAULT_CONFIG["weight_decay"],
    )

    print("\n=================================================================", flush=True)
    print("Starting Biology Domain Training with Dynamic Prompts & 256-Char Truncation", flush=True)
    print(f"JSD_MAX Cap: {JSD_MAX:.6f} | Muon LR: {args.lr}", flush=True)
    print("=================================================================\n", flush=True)

    step = 1
    try:
        while True:
            # 1. Step A: Generate dynamic prompt from Teacher with Temp = 0.8
            dynamic_prompt = generate_dynamic_prompt(teacher, tokenizer, temperature=0.8)

            # 2. Step B: Build target sequence with lower-cased response target
            messages = [
                {"role": "user", "content": dynamic_prompt},
                {"role": "assistant", "content": dynamic_prompt.lower()}
            ]
            full_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")["input_ids"].to(device=device)
            inputs, targets = full_ids[:, :-1], full_ids[:, 1:]

            optimizer.zero_grad()

            # Forward passes (Step B: Temp = 0.0)
            t_logits = teacher(inputs)
            s_logits = student(inputs)

            # 1. JSD Loss calculation
            log_probs_s = F.log_softmax(s_logits.float().reshape(-1, vocab_size), dim=-1)
            log_probs_t = F.log_softmax(t_logits.float().reshape(-1, vocab_size), dim=-1)
            log_probs_m = torch.logsumexp(torch.stack([log_probs_s, log_probs_t], dim=0), dim=0) - math.log(2)

            kl_s_m = F.kl_div(log_probs_s, log_probs_m, reduction='batchmean', log_target=True)
            kl_t_m = F.kl_div(log_probs_t, log_probs_m, reduction='batchmean', log_target=True)
            jsd_loss = 0.5 * kl_s_m + 0.5 * kl_t_m

            # 2. Raw Doc Loss (Cross Entropy to Lowercase Target)
            raw_doc_loss = F.cross_entropy(s_logits.float().reshape(-1, vocab_size), targets.reshape(-1))

            # 3. Capped / Bounded Doc Loss: doc_loss = (raw_doc_loss / (1 + raw_doc_loss.abs())) * JSD_MAX
            bounded_doc_loss = (raw_doc_loss / (1.0 + raw_doc_loss.abs())) * JSD_MAX

            # Total Loss
            total_loss = jsd_loss + bounded_doc_loss
            total_loss.backward()

            optimizer.step()

            # Measure Logit Gap on first token position of response
            first_pos_logits = s_logits[0, -1, :]
            top_token_id = torch.argmax(first_pos_logits).item()
            top_token_str = tokenizer.decode([top_token_id])
            lower_str = top_token_str.lower()
            lower_token_ids = tokenizer.encode(lower_str, add_special_tokens=False)
            lower_token_id = lower_token_ids[-1] if lower_token_ids else top_token_id
            
            gap = (first_pos_logits[top_token_id] - first_pos_logits[lower_token_id]).item()

            # Generate zero-temp inference outputs for Teacher and Student
            teacher_output = run_inference_zero_temp(dynamic_prompt, teacher, tokenizer)
            student_output = run_inference_zero_temp(dynamic_prompt, student, tokenizer)

            # Max 256 chars formatting
            prompt_disp = dynamic_prompt
            teacher_disp = teacher_output
            student_disp = student_output

            # Print 5-line output format with max 256 chars per text section
            print(
                f"Step {step:04d} | JSD: {jsd_loss.item():.6f} | "
                f"Raw Doc: {raw_doc_loss.item():.6f} | "
                f"Bounded Doc: {bounded_doc_loss.item():.6f} | "
                f"Total Loss: {total_loss.item():.6f}",
                flush=True
            )
            print(f"Logit Gap ('{top_token_str.strip()}' vs '{lower_str.strip()}'): {gap:+.4f}", flush=True)
            print(f"Prompt: \"{prompt_disp}\"", flush=True)
            print(f"Teacher: \"{teacher_disp}\"", flush=True)
            print(f"Student: \"{student_disp}\"", flush=True)
            print("", flush=True)
            print("-" * 80, flush=True)
            print("", flush=True)
            step += 1

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving final Student checkpoint...", flush=True)
        student.save(STUDENT_CHECKPOINT_PATH)
        print("Shutdown complete.", flush=True)


if __name__ == "__main__":
    main()
