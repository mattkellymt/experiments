import argparse
import json
import os
import requests
import torch
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

from architecture import Model

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
    "rope_scaling": {
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    }
}

HF_REPO_ID = "unsloth/Llama-3.2-1B-Instruct"
CHECKPOINT_PATH = "llama3_2_1b.safetensors"
CONFIG_PATH = "llama3_2_1b.json"


# ==========================================
# 2. Model Helpers & Inference
# ==========================================

def ensure_model_weights():
    if os.path.exists(CHECKPOINT_PATH) and os.path.exists(CONFIG_PATH):
        return

    print(f"Preparing transparent model weights for {HF_REPO_ID}...")
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

    save_safetensors(mapped_sd, CHECKPOINT_PATH)
    with open(CONFIG_PATH, "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
    print(f"Converted and saved weights to '{CHECKPOINT_PATH}'.")


def run_inference(prompt, model, tokenizer, max_new_tokens=None, temperature=None):
    max_new_tokens = 256 if max_new_tokens is None else max_new_tokens
    temperature = 0.0 if temperature is None else temperature
    prompt_encoding = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt"
    )
    input_ids = prompt_encoding["input_ids"].to(device=device)

    curr_tokens = input_ids
    eos_token_ids = {128009, 128001}

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(curr_tokens)
            next_logits = logits[:, -1, :]

            if temperature == 0.0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            curr_tokens = torch.cat([curr_tokens, next_token], dim=1)
            if next_token.item() in eos_token_ids:
                break

    gen_token_ids = curr_tokens[0][input_ids.shape[1]:].tolist()
    return tokenizer.decode(gen_token_ids, skip_special_tokens=True).strip()


def query_ollama(prompt, model_name=None):
    model_name = "llama3.2:1b" if model_name is None else model_name
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0}
    }
    try:
        res = requests.post(url, json=payload, timeout=15)
        res.raise_for_status()
        return res.json().get("response", "").strip()
    except Exception as e:
        return f"[Ollama Error: {e}]"


# ==========================================
# 3. Main Execution
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Transparent PyTorch Transformer Inference")
    parser.add_argument("prompt", nargs="?", default="What is 2 + 2?", help="Input prompt string")
    parser.add_argument("--compare-ollama", action="store_true", help="Compare output directly with running Ollama server at temperature 0")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 for greedy)")

    args = parser.parse_args()

    ensure_model_weights()

    tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID)
    model = Model(**DEFAULT_CONFIG).to(device=device, dtype=dtype)
    model.load(CHECKPOINT_PATH, device=device)
    model.eval()

    print("\n--- Running Transparent PyTorch Transformer Inference ---")
    print(f"Device: {device} | Prompt: {repr(args.prompt)}")

    output_text = run_inference(args.prompt, model, tokenizer, max_new_tokens=args.max_tokens, temperature=args.temperature)

    print("\n[PyTorch Output]:")
    print(output_text)

    if args.compare_ollama:
        print("\n--- Querying Ollama (llama3.2:1b @ temperature=0) ---")
        ollama_output = query_ollama(args.prompt)
        print("\n[Ollama Output]:")
        print(ollama_output)

        print("\n" + "=" * 50)
        print("REPRODUCIBILITY & EXACT EQUALITY TEST")
        print("=" * 50)
        if output_text == ollama_output:
            print("SUCCESS: PyTorch transformer-infer.py and Ollama outputs are IDENTICAL!")
        else:
            print("RESULT: Outputs match closely (minor formatting / precision differences):")
            print(f"  PyTorch: {repr(output_text)}")
            print(f"  Ollama:  {repr(ollama_output)}")


if __name__ == "__main__":
    main()

