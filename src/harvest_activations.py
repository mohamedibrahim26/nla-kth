"""
Step 1 of the NLA pipeline: harvest activations from a frozen target model.

What this does, in plain terms
------------------------------
We run many short text snippets through a frozen language model. For each snippet
we truncate it at a random point and record TWO things:

  1. the residual-stream activation vector at a chosen layer, for the *final* token
     (this is the model's "state of mind" at that point), and
  2. the exact (truncated) text the model had read up to that point.

We keep the text because later a "teacher" model will describe what the target model
was thinking about at that token -- and to do that fairly, the teacher must see the
same text the target model saw.

The base model is FROZEN. We never train it. We only read activations out of it.

Outputs (written to <data_dir>/):
  - activations.npy   : float16 array of shape (N, d_model)
  - metadata.jsonl    : one JSON line per activation: {idx, text, n_tokens}
  - harvest_meta.json : run settings (model, layer, etc.) for reproducibility

Run from the command line:
    python src/harvest_activations.py --num_samples 8000 --data_dir data
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import CONFIG


def parse_args():
    p = argparse.ArgumentParser(description="Harvest activations from a frozen LM.")
    p.add_argument("--model_name", default=CONFIG.model_name)
    p.add_argument("--layer", type=int, default=CONFIG.layer)
    p.add_argument("--dataset_name", default=CONFIG.dataset_name)
    p.add_argument("--dataset_config", default=CONFIG.dataset_config)
    p.add_argument("--dataset_split", default=CONFIG.dataset_split)
    p.add_argument("--num_samples", type=int, default=CONFIG.num_samples)
    p.add_argument("--min_tokens", type=int, default=CONFIG.min_tokens)
    p.add_argument("--max_tokens", type=int, default=CONFIG.max_tokens)
    p.add_argument("--seed", type=int, default=CONFIG.seed)
    p.add_argument("--data_dir", default=CONFIG.data_dir)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs(args.data_dir, exist_ok=True)

    # --- Load the frozen target model ---
    print(f"Loading {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        output_hidden_states=True,
    ).to(device)
    model.eval()  # frozen: no training, no dropout

    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    assert 0 < args.layer <= n_layers, f"layer must be in 1..{n_layers}"
    print(f"Model has {n_layers} layers, hidden size {d_model}. Reading layer {args.layer}.")

    # --- Load text corpus ---
    print(f"Loading dataset {args.dataset_name}/{args.dataset_config} ...")
    ds = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    # Keep only non-trivial lines (wikitext has many blank/heading lines).
    texts = [t for t in ds["text"] if len(t.strip()) > 80]
    random.shuffle(texts)
    print(f"{len(texts)} usable text snippets available.")

    activations = np.zeros((args.num_samples, d_model), dtype=np.float16)
    meta_path = os.path.join(args.data_dir, "metadata.jsonl")
    collected = 0

    with open(meta_path, "w", encoding="utf-8") as meta_f:
        pbar = tqdm(total=args.num_samples, desc="Harvesting")
        for text in texts:
            if collected >= args.num_samples:
                break

            # Tokenize, then truncate at a RANDOM point so we sample the model's
            # "thinking" at many different positions, not just the ends of texts.
            ids = tokenizer(text, return_tensors="pt").input_ids[0]
            if ids.shape[0] < args.min_tokens:
                continue
            cut = random.randint(args.min_tokens, min(args.max_tokens, ids.shape[0]))
            ids = ids[:cut].unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(input_ids=ids)

            # hidden_states[layer] has shape (1, seq_len, d_model); take the last token.
            act = out.hidden_states[args.layer][0, -1, :].float().cpu().numpy()
            activations[collected] = act.astype(np.float16)

            truncated_text = tokenizer.decode(ids[0], skip_special_tokens=True)
            meta_f.write(json.dumps({
                "idx": collected,
                "text": truncated_text,
                "n_tokens": int(cut),
            }) + "\n")

            collected += 1
            pbar.update(1)
        pbar.close()

    if collected < args.num_samples:
        activations = activations[:collected]
        print(f"Note: only {collected} samples collected (corpus exhausted).")

    np.save(os.path.join(args.data_dir, "activations.npy"), activations)

    with open(os.path.join(args.data_dir, "harvest_meta.json"), "w") as f:
        json.dump({
            "model_name": args.model_name,
            "layer": args.layer,
            "d_model": int(d_model),
            "n_layers": int(n_layers),
            "num_samples": int(collected),
            "dataset": f"{args.dataset_name}/{args.dataset_config}",
            "min_tokens": args.min_tokens,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        }, f, indent=2)

    # Quick sanity stats so we can eyeball that nothing is degenerate.
    norms = np.linalg.norm(activations.astype(np.float32), axis=1)
    print("\nDone.")
    print(f"  activations shape : {activations.shape}")
    print(f"  L2 norm  mean/std : {norms.mean():.2f} / {norms.std():.2f}")
    print(f"  saved to          : {args.data_dir}/activations.npy")


if __name__ == "__main__":
    main()
