"""
analyze_bug_descriptions.py — The central research question for the code extension.

Do NLA verbalizer descriptions shift in a measurable way when a code model reads
buggy code versus correct code?

What this script does
---------------------
1. Loads code_activations.pkl and code_teacher_summaries.pkl.
2. Runs the trained verbalizer on every code activation to generate a description.
3. Computes keyword frequency for a bug-concept vocabulary across buggy vs correct
   descriptions.
4. Computes FVE (Fraction of Variance Explained) separately for buggy vs correct
   using the trained reconstructor.
5. Runs a per-bug-type breakdown.
6. Saves results/bug_analysis.json with the full results table.

The falsifiable prediction: if the NLA bottleneck carries code-semantic content,
bug-concept keywords should appear significantly more often in descriptions of
buggy activations than correct activations.

Usage
-----
    python src/analyze_bug_descriptions.py \
        --data_dir  data \
        --out_dir   results \
        --verbalizer_pt   data/verbalizer_rl_best.pt \
        --verbalizer_lora data/verbalizer_rl_lora_best \
        --reconstructor_pt   data/reconstructor.pt \
        --reconstructor_lora data/reconstructor_lora

(Defaults fall back to warm-start checkpoints if RL checkpoints are absent.)
"""

import argparse
import json
import os
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

# ── bug-concept keyword vocabulary ────────────────────────────────────────────
# Grouped into conceptual clusters for richer analysis.
BUG_KEYWORDS = {
    "off_by_one":   ["off.by.one", "off by one", "boundary", "bounds check",
                     "fencepost", "fence.post", "one.too", "one too", "range"],
    "wrong_op":     ["wrong operator", "incorrect operator", "wrong operation",
                     "sign error", "arithmetic error", "subtracted", "added",
                     "multiplied", "divided"],
    "wrong_var":    ["wrong variable", "swapped", "transposed", "incorrect variable",
                     "mixed up", "confused"],
    "general_bug":  ["bug", "error", "issue", "incorrect", "mistake", "fault",
                     "problem", "defect", "flaw", "broken"],
    "logic":        ["logic", "logical", "condition", "conditional", "guard",
                     "check", "null", "none", "edge case", "corner case"],
}

ALL_KEYWORDS = [kw for group in BUG_KEYWORDS.values() for kw in group]

WRAP_RECON = "A language model's internal state for the following code is described as: {s}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",          default="data")
    p.add_argument("--out_dir",           default="results")
    p.add_argument("--verbalizer_pt",     default=None)
    p.add_argument("--verbalizer_lora",   default=None)
    p.add_argument("--reconstructor_pt",  default=None)
    p.add_argument("--reconstructor_lora",default=None)
    p.add_argument("--max_new_tokens",    type=int, default=150)
    p.add_argument("--batch_size",        type=int, default=8)
    p.add_argument("--layer",             type=int, default=16)
    p.add_argument("--limit",             type=int, default=None)
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

def standardise(acts: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (acts - mean) / np.where(std < 1e-8, 1.0, std)


def fve(pred: np.ndarray, true: np.ndarray) -> float:
    mse = np.mean((pred - true) ** 2)
    var = np.var(true)
    return float(1.0 - mse / var) if var > 1e-10 else 0.0


def keyword_hits(text: str, keywords: list) -> dict:
    text_lower = text.lower()
    return {kw: int(bool(re.search(re.escape(kw), text_lower))) for kw in keywords}


def load_checkpoint(base_model_name, pt_path, lora_path, device):
    """Load a LoRA-adapted causal LM checkpoint."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    if lora_path and os.path.exists(lora_path):
        model = PeftModel.from_pretrained(model, lora_path)
    if pt_path and os.path.exists(pt_path):
        ckpt = torch.load(pt_path, map_location="cpu")
        # load projection weights if present
        if "projection" in ckpt:
            return model.to(device), tokenizer, ckpt
    model = model.to(device)
    return model, tokenizer, None


# ── verbalizer inference ──────────────────────────────────────────────────────

@torch.no_grad()
def verbalize_batch(
    model, tokenizer, projection,
    acts_norm: torch.Tensor,       # (B, D)
    suffix_ids: torch.Tensor,      # (P,)
    max_new_tokens: int,
    device,
) -> list:
    B = acts_norm.shape[0]
    act_embs   = projection(acts_norm).unsqueeze(1)           # (B, 1, E)
    suffix_emb = model.get_input_embeddings()(suffix_ids)     # (P, E)
    suffix_emb = suffix_emb.unsqueeze(0).expand(B, -1, -1)   # (B, P, E)
    prompt_emb = torch.cat([act_embs, suffix_emb], dim=1)    # (B, P+1, E)

    out = model.generate(
        inputs_embeds=prompt_emb,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    texts = [tokenizer.decode(o, skip_special_tokens=True).strip() for o in out]
    return texts


# ── reconstructor inference ───────────────────────────────────────────────────

@torch.no_grad()
def reconstruct_batch(
    model, tokenizer, head,
    texts: list,
    device,
) -> np.ndarray:
    """Return predicted standardised activations, shape (B, D)."""
    wrapped = [WRAP_RECON.format(s=t) for t in texts]
    enc = tokenizer(
        wrapped, return_tensors="pt", padding=True,
        truncation=True, max_length=256,
    ).to(device)
    hidden = model(**enc, output_hidden_states=True).hidden_states[-1]  # (B, T, D)
    last   = hidden[:, -1, :]                                           # (B, D)
    pred   = head(last.float())                                          # (B, D_act)
    return pred.cpu().numpy()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # ── resolve checkpoint paths ──────────────────────────────────────────────
    d = args.data_dir
    verb_pt   = args.verbalizer_pt   or _first_exists(
        os.path.join(d, "verbalizer_rl_best.pt"),
        os.path.join(d, "verbalizer.pt"),
    )
    verb_lora = args.verbalizer_lora or _first_exists(
        os.path.join(d, "verbalizer_rl_lora_best"),
        os.path.join(d, "verbalizer_lora"),
    )
    recon_pt   = args.reconstructor_pt   or _first_exists(
        os.path.join(d, "reconstructor.pt"),
    )
    recon_lora = args.reconstructor_lora or _first_exists(
        os.path.join(d, "reconstructor_lora"),
    )

    if not verb_pt:
        sys.exit("[error] No verbalizer checkpoint found. Train the verbalizer first.")
    if not recon_pt:
        sys.exit("[error] No reconstructor checkpoint found. Train the reconstructor first.")

    # ── load activations ──────────────────────────────────────────────────────
    act_path = os.path.join(d, "code_activations.pkl")
    if not os.path.exists(act_path):
        sys.exit(f"[error] {act_path} not found. Run harvest_code_activations.py first.")
    with open(act_path, "rb") as f:
        records = pickle.load(f)
    if args.limit:
        records = records[: args.limit]

    acts_raw = np.stack([r["activation"] for r in records]).astype(np.float32)
    mean_ = acts_raw.mean(axis=0)
    std_  = acts_raw.std(axis=0)
    acts_std = standardise(acts_raw, mean_, std_)

    # ── load verbalizer ───────────────────────────────────────────────────────
    print(f"Loading verbalizer from {verb_pt} …")
    BASE_MODEL = "Qwen/Qwen2.5-Coder-0.5B"
    verb_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    verb_model     = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    if verb_lora and os.path.exists(verb_lora):
        verb_model = PeftModel.from_pretrained(verb_model, verb_lora)
    verb_model = verb_model.to(device).eval()

    ckpt = torch.load(verb_pt, map_location="cpu")
    D    = acts_raw.shape[1]
    E    = verb_model.config.hidden_size
    projection = torch.nn.Linear(D, E, bias=False).to(device)
    projection.load_state_dict(ckpt["projection"])
    projection.eval()

    SUFFIX = "Describe what this code model is currently focusing on:"
    suffix_ids = verb_tokenizer(SUFFIX, return_tensors="pt",
                                add_special_tokens=False)["input_ids"][0].to(device)

    # ── load reconstructor ────────────────────────────────────────────────────
    print(f"Loading reconstructor from {recon_pt} …")
    recon_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    recon_model     = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    if recon_lora and os.path.exists(recon_lora):
        recon_model = PeftModel.from_pretrained(recon_model, recon_lora)
    recon_model = recon_model.to(device).eval()

    recon_ckpt = torch.load(recon_pt, map_location="cpu")
    head = torch.nn.Linear(E, D, bias=True).to(device)
    head.load_state_dict(recon_ckpt["head"])
    head.eval()

    # ── generate descriptions + compute metrics ───────────────────────────────
    results = []
    BS = args.batch_size

    for start in tqdm(range(0, len(records), BS), desc="Analysing"):
        batch_recs = records[start: start + BS]
        batch_acts = torch.tensor(
            acts_std[start: start + BS], dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)

        texts = verbalize_batch(
            verb_model, verb_tokenizer, projection,
            batch_acts, suffix_ids, args.max_new_tokens, device,
        )

        preds_std = reconstruct_batch(recon_model, recon_tokenizer, head, texts, device)
        true_std  = acts_std[start: start + BS]

        for i, (rec, text, pred, true) in enumerate(
            zip(batch_recs, texts, preds_std, true_std)
        ):
            hits = keyword_hits(text, ALL_KEYWORDS)
            results.append({
                "func_id":    rec["func_id"],
                "variant":    rec["variant"],
                "bug_type":   rec.get("bug_type"),
                "description":text,
                "fve":        fve(pred.reshape(1, -1), true.reshape(1, -1)),
                "kw_hits":    hits,
            })

    # ── aggregate ─────────────────────────────────────────────────────────────
    correct_res = [r for r in results if r["variant"] == "correct"]
    buggy_res   = [r for r in results if r["variant"] == "buggy"]

    def mean_kw(res_list):
        if not res_list:
            return {}
        out = {}
        for kw in ALL_KEYWORDS:
            out[kw] = np.mean([r["kw_hits"][kw] for r in res_list])
        return out

    correct_kw = mean_kw(correct_res)
    buggy_kw   = mean_kw(buggy_res)

    keyword_shift = {
        kw: {
            "correct_freq": correct_kw.get(kw, 0.0),
            "buggy_freq":   buggy_kw.get(kw, 0.0),
            "shift":        buggy_kw.get(kw, 0.0) - correct_kw.get(kw, 0.0),
        }
        for kw in ALL_KEYWORDS
    }

    # Sort by absolute shift descending
    top_shifts = sorted(
        keyword_shift.items(), key=lambda x: abs(x[1]["shift"]), reverse=True
    )[:20]

    # Per-bug-type FVE
    fve_by_type = {}
    for bt in set(r["bug_type"] for r in buggy_res if r["bug_type"]):
        subset = [r for r in buggy_res if r["bug_type"] == bt]
        fve_by_type[bt] = float(np.mean([r["fve"] for r in subset]))

    summary = {
        "n_correct":    len(correct_res),
        "n_buggy":      len(buggy_res),
        "fve_correct":  float(np.mean([r["fve"] for r in correct_res])) if correct_res else 0.0,
        "fve_buggy":    float(np.mean([r["fve"] for r in buggy_res]))   if buggy_res   else 0.0,
        "fve_by_bug_type": fve_by_type,
        "top_keyword_shifts": {k: v for k, v in top_shifts},
        "full_keyword_shift": keyword_shift,
        "sample_descriptions": {
            "correct": [r["description"] for r in correct_res[:3]],
            "buggy":   [r["description"] for r in buggy_res[:3]],
        },
    }

    out_path = os.path.join(args.out_dir, "bug_analysis.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {out_path}")

    # ── print headline results ────────────────────────────────────────────────
    print("\n══ HEADLINE RESULTS ══════════════════════════════════════")
    print(f"  FVE correct : {summary['fve_correct']:.4f}")
    print(f"  FVE buggy   : {summary['fve_buggy']:.4f}")
    print("\n  Top keyword shifts (buggy − correct frequency):")
    for kw, vals in top_shifts[:10]:
        bar = "▲" if vals["shift"] > 0 else "▼"
        print(f"    {bar} {kw:<30s}  "
              f"correct={vals['correct_freq']:.3f}  "
              f"buggy={vals['buggy_freq']:.3f}  "
              f"Δ={vals['shift']:+.3f}")
    print("══════════════════════════════════════════════════════════")


def _first_exists(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


if __name__ == "__main__":
    main()
