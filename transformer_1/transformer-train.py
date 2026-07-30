import argparse
import json
import math
import os
import time
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

from models.experiments.transformer_1.architecture import Model, Muon

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
    "muon_lr": 0.0001,
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

PROMPTS = [
    "What is the principle of conservation of energy?",
    "Explain how a hash table works in computer science.",
    "What is the theory of general relativity?",
    "How does the model-view-controller pattern work?",
    "What caused the fall of the Western Roman Empire?",
    "Explain the concept of recursion with an example.",
    "What is the function of mitochondria in a cell?",
    "How does a binary search tree maintain order?",
    "What is the difference between synchronous and asynchronous execution?",
    "Explain the laws of thermodynamics.",
    "What is quantum entanglement?",
    "How does gradient descent optimize neural networks?",
    "What is the Turing test?",
    "Explain the difference between TCP and UDP.",
    "What is the role of natural selection in evolution?",
    "How does key-value storage differ from relational databases?",
    "What is the significance of the Magna Carta?",
    "Explain how attention mechanisms work in transformers.",
    "What is the Doppler effect?",
    "How does garbage collection work in modern programming languages?",
    "What is the difference between a process and a thread?",
    "Explain the concept of time complexity and Big O notation.",
    "What is photosynthesis and how does it convert light into energy?",
    "How does public-key cryptography enable secure communication?",
    "What is the Fermi paradox?",
    "Explain the concept of deadlocks in multi-threaded programming.",
    "What are the primary functions of an operating system kernel?",
    "How does object-oriented programming promote code reuse?",
    "What is the difference between precision and recall in machine learning?",
    "Explain the structure and purpose of DNA.",
    "What is the role of a compiler in software development?",
    "How does backpropagation update weights in a deep neural network?"
]


# ==========================================
# 2. Model Helpers
# ==========================================

def ensure_model_weights():
    # If legacy checkpoint exists, rename/ensure teacher checkpoint
    if not os.path.exists(TEACHER_CHECKPOINT_PATH) and os.path.exists("llama3_2_1b.safetensors"):
        os.rename("llama3_2_1b.safetensors", TEACHER_CHECKPOINT_PATH)

    if os.path.exists(TEACHER_CHECKPOINT_PATH) and os.path.exists(CONFIG_PATH):
        return

    print(f"Preparing transparent Teacher model weights for {HF_REPO_ID}...")
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
        mapped_sd[f"model.layers.{i}.post_attention_layernorm.weight"] = sd[f"model.layers.{i}.post_attention_layernorm.weight"].to(dtype).contiguous()
        mapped_sd[f"model.layers.{i}.mlp.gate_proj.weight"] = sd[f"model.layers.{i}.mlp.gate_proj.weight"].T.to(dtype).contiguous()
        mapped_sd[f"model.layers.{i}.mlp.up_proj.weight"] = sd[f"model.layers.{i}.mlp.up_proj.weight"].T.to(dtype).contiguous()
        mapped_sd[f"model.layers.{i}.mlp.down_proj.weight"] = sd[f"model.layers.{i}.mlp.down_proj.weight"].T.to(dtype).contiguous()

    save_safetensors(mapped_sd, TEACHER_CHECKPOINT_PATH)
    with open(CONFIG_PATH, "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
    print(f"Converted and saved Teacher weights to '{TEACHER_CHECKPOINT_PATH}'.")


# ==========================================
# 3. Main Execution & JSD Training
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Teacher vs Student Jensen-Shannon Divergence (JSD) Trainer")
    parser.add_argument("--temperature", type=float, default=0.0, help="Training temperature (default: 0.0)")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for optimizer (default: 0.0)")
    args = parser.parse_args()

    ensure_model_weights()

    tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID)
    vocab_size = DEFAULT_CONFIG["vocab_size"]

    print("\n--- Loading Unmodified Teacher Model (from Hugging Face) ---", flush=True)
    teacher = Model(**DEFAULT_CONFIG).to(device=device, dtype=dtype)
    teacher.load(TEACHER_CHECKPOINT_PATH, device=device)
    teacher.train()  # Matching execution mode ensures identical PyTorch backend dispatch

    print("--- Loading Unmodified Student Model (from Hugging Face) ---", flush=True)
    student = Model(**DEFAULT_CONFIG).to(device=device, dtype=dtype)
    student.load(TEACHER_CHECKPOINT_PATH, device=device)
    student.train()

    optimizer = Muon(
        (p for p in student.parameters() if p.ndim == 2),
        lr=DEFAULT_CONFIG["muon_lr"],
        momentum=DEFAULT_CONFIG["momentum"],
        weight_decay=args.weight_decay,
    )

    print(f"\n--- Processing {len(PROMPTS)} Fixed Prompts with Jensen-Shannon Divergence (JSD) Loss ---", flush=True)
    start_time = time.time()
    total_jsd_loss = 0.0

    for idx, prompt in enumerate(PROMPTS, 1):
        encoding = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device=device)

        # Teacher forward pass (frozen weights)
        teacher_logits = teacher(input_ids)

        # Student forward pass
        student_logits = student(input_ids)

        if args.temperature > 0.0:
            teacher_logits = teacher_logits / args.temperature
            student_logits = student_logits / args.temperature

        # Exact log_softmax for both models
        log_probs_student = F.log_softmax(student_logits.float().reshape(-1, vocab_size), dim=-1)
        log_probs_teacher = F.log_softmax(teacher_logits.float().reshape(-1, vocab_size), dim=-1)

        # Log-space Midpoint distribution M = 0.5 * (P_Student + P_Teacher)
        log_probs_mid = torch.logsumexp(torch.stack([log_probs_student, log_probs_teacher], dim=0), dim=0) - math.log(2)

        # JSD = 0.5 * KL(P_Student || M) + 0.5 * KL(P_Teacher || M)
        kl_student_m = F.kl_div(log_probs_student, log_probs_mid, reduction='batchmean', log_target=True)
        kl_teacher_m = F.kl_div(log_probs_teacher, log_probs_mid, reduction='batchmean', log_target=True)

        jsd_loss = 0.5 * kl_student_m + 0.5 * kl_teacher_m

        loss_val = jsd_loss.item()
        total_jsd_loss += loss_val
        max_logits_diff = (teacher_logits - student_logits).abs().max().item()

        print(
            f"Sample {idx:02d}/32 | Diff: {max_logits_diff:.6f} | "
            f"KL(S||M): {kl_student_m.item():.8f} | KL(T||M): {kl_teacher_m.item():.8f} | "
            f"JSD Loss: {loss_val:.8f} | Prompt: \"{prompt[:35]}...\"",
            flush=True
        )

    avg_jsd_loss = total_jsd_loss / len(PROMPTS)
    elapsed = time.time() - start_time

    print(f"\n=======================================================", flush=True)
    print(f"Completed pass over all {len(PROMPTS)} prompts in {elapsed:.2f}s", flush=True)
    print(f"Final Average JSD Loss: {avg_jsd_loss:.8f}", flush=True)
    print(f"=======================================================\n", flush=True)


if __name__ == "__main__":
    main()
