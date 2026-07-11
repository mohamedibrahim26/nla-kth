"""
End-to-end NLA evaluation.

For each validation activation we:
  1. Run the VERBALIZER to generate a natural-language description from the activation.
  2. Run the RECONSTRUCTOR on that generated text to predict the activation back.
  3. Compute FVE (in the reconstructor's standardized space) on:
       - generated_text  : the real NLA roundtrip number (the headline)
       - oracle_text     : feed the teacher summary instead -> upper-bound ceiling
       - random_text     : feed a random OTHER teacher summary -> negative control
       - predict_mean    : zeros (FVE = 0 by construction)

Also saves a small CSV of (original_text, teacher_summary, generated_summary) for
qualitative inspection in the README.

Run:
    python src/evaluate_nla.py --data_dir data --output_dir data
"""

import argparse
import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from config import CONFIG
from models import compute_fve


WRAP_RECON = "A language model's internal state is described as follows: {s}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--model_name", default=CONFIG.model_name)
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--batch_size_eval", type=int, default=8)
    p.add_argument("--max_len_recon", type=int, default=200)
    p.add_argument("--n_qualitative", type=int, default=20)
    p.add_argument("--verbalizer_pt",   default=None,
                   help="path to verbalizer .pt file (default: <output_dir>/verbalizer.pt); "
                        "set to data/verbalizer_rl_best.pt to evaluate the RL model")
    p.add_argument("--verbalizer_lora",  default=None,
                   help="path to verbalizer LoRA dir (default: <output_dir>/verbalizer_lora); "
                        "set to data/verbalizer_rl_lora_best to evaluate the RL model")
    return p.parse_args()


def last_token(hidden, attn):
    idx = attn.sum(1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = args.output_dir or args.data_dir
    os.makedirs(out_dir, exist_ok=True)
    if args.verbalizer_pt   is None: args.verbalizer_pt   = os.path.join(out_dir, "verbalizer.pt")
    if args.verbalizer_lora is None: args.verbalizer_lora = os.path.join(out_dir, "verbalizer_lora")
    print(f"Device: {device}")
    print(f"Verbalizer: {args.verbalizer_pt}  |  {args.verbalizer_lora}")

    # --- Load data + split ---
    activations = np.load(os.path.join(args.data_dir, "activations.npy")).astype(np.float32)
    rows = [json.loads(l) for l in open(os.path.join(args.data_dir, "summaries.jsonl"), encoding="utf-8")]
    meta_rows = {json.loads(l)["idx"]: json.loads(l)
                 for l in open(os.path.join(args.data_dir, "metadata.jsonl"), encoding="utf-8")}
    idx_to_summary = {r["idx"]: r["summary"] for r in rows}
    idxs_all = [r["idx"] for r in rows]

    split = json.load(open(os.path.join(out_dir, "split.json")))
    val_idx = split["val_idx"]
    val_idx = [i for i in val_idx if i in idx_to_summary]
    print(f"Val set: {len(val_idx)} samples")

    targets_raw = torch.tensor(activations[val_idx], dtype=torch.float32)
    summaries = [idx_to_summary[i] for i in val_idx]
    snippets = [meta_rows[i]["text"] for i in val_idx]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load RECONSTRUCTOR (base + LoRA + head + stats) ---
    print("Loading reconstructor...")
    base_r = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32).to(device)
    recon_model = PeftModel.from_pretrained(base_r, os.path.join(out_dir, "reconstructor_lora")).to(device)
    recon_model.eval()
    recon_pt = torch.load(os.path.join(out_dir, "reconstructor.pt"), map_location=device)
    d = recon_pt["d"]
    # Rebuild head (Linear(d, d))
    head = nn.Linear(d, d).to(device)
    head.load_state_dict(recon_pt["head"])
    head.eval()
    tmean = recon_pt["tmean"].to(device); tstd = recon_pt["tstd"].to(device)
    Y_std = ((targets_raw.to(device) - tmean) / tstd).cpu()

    @torch.no_grad()
    def recon_predict(texts):
        preds = []
        for i in range(0, len(texts), args.batch_size_eval):
            batch = [WRAP_RECON.format(s=t) for t in texts[i:i + args.batch_size_eval]]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=args.max_len_recon).to(device)
            out = recon_model(**enc, output_hidden_states=True)
            h = last_token(out.hidden_states[-1], enc["attention_mask"])
            preds.append(head(h.float()).cpu())
        return torch.cat(preds, dim=0)

    # --- Quick FVE on ORACLE text (teacher summaries) ---
    print("Computing oracle-text FVE...")
    oracle_pred = recon_predict(summaries)
    fve_oracle = compute_fve(oracle_pred, Y_std)
    print(f"  oracle FVE: {fve_oracle:.4f}")

    # --- FVE on RANDOM (shuffled) text -> control ---
    print("Computing random-text FVE...")
    random.seed(0)
    shuffled = summaries[:]; random.shuffle(shuffled)
    rand_pred = recon_predict(shuffled)
    fve_random = compute_fve(rand_pred, Y_std)
    print(f"  random FVE: {fve_random:.4f}")

    # Free recon model memory before loading verbalizer
    del recon_model, base_r
    torch.cuda.empty_cache()

    # --- Load VERBALIZER (base + LoRA + projection) ---
    print("Loading verbalizer...")
    vp = torch.load(args.verbalizer_pt, map_location=device)
    base_v = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32).to(device)
    verb_model = PeftModel.from_pretrained(base_v, args.verbalizer_lora).to(device)
    verb_model.eval()
    projection = nn.Linear(vp["act_dim"], vp["embed_dim"]).to(device)
    projection.load_state_dict(vp["projection"])
    projection.eval()
    embed_layer = verb_model.get_input_embeddings()

    suffix_ids = tokenizer(vp["prompt_after"], return_tensors="pt",
                           add_special_tokens=False).input_ids[0].to(device)

    # Unit-normalize the activations the same way we did during training
    targets_norm = targets_raw / (targets_raw.norm(dim=1, keepdim=True) + 1e-8)

    @torch.no_grad()
    def generate_texts():
        gen = []
        for i, act in enumerate(targets_norm):
            act_emb = projection(act.to(device).unsqueeze(0))      # (1, E)
            suffix_emb = embed_layer(suffix_ids)                   # (P, E)
            seq_embeds = torch.cat([act_emb, suffix_emb], dim=0).unsqueeze(0)  # (1, 1+P, E)
            attn = torch.ones(seq_embeds.shape[:2], dtype=torch.long, device=device)
            out = verb_model.generate(
                inputs_embeds=seq_embeds, attention_mask=attn,
                max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            gen.append(text.strip())
            if (i + 1) % 50 == 0:
                print(f"  generated {i + 1}/{len(targets_norm)}")
        return gen

    print("Generating verbalizer text for val set...")
    generated = generate_texts()

    # Free verbalizer, reload reconstructor for final scoring
    del verb_model, base_v, projection, embed_layer
    torch.cuda.empty_cache()
    base_r = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32).to(device)
    recon_model = PeftModel.from_pretrained(base_r, os.path.join(out_dir, "reconstructor_lora")).to(device)
    recon_model.eval()

    print("Computing generated-text FVE...")
    gen_pred = recon_predict(generated)
    fve_gen = compute_fve(gen_pred, Y_std)
    print(f"  generated FVE: {fve_gen:.4f}")

    # --- Save results ---
    results = {
        "fve_generated_text": fve_gen,
        "fve_oracle_text": fve_oracle,
        "fve_random_text": fve_random,
        "fve_predict_mean": 0.0,
        "n_val": len(val_idx),
    }
    with open(os.path.join(out_dir, "nla_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults:", json.dumps(results, indent=2))

    # Qualitative samples
    n = min(args.n_qualitative, len(val_idx))
    with open(os.path.join(out_dir, "nla_samples.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "snippet", "teacher_summary", "generated_summary"])
        for i in range(n):
            w.writerow([val_idx[i], snippets[i], summaries[i], generated[i]])
    print(f"Saved {n} qualitative samples to {out_dir}/nla_samples.csv")


if __name__ == "__main__":
    main()
