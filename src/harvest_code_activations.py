"""
harvest_code_activations.py — Harvest residual-stream activations from
Qwen2.5-Coder-0.5B on correct/buggy Python function pairs.

Pipeline
--------
1. Download Python functions from CodeSearchNet (HuggingFace datasets).
2. For each function, attempt to inject one synthetic bug via bug_injection.py.
3. Run both the correct and the buggy version through Qwen2.5-Coder-0.5B.
4. Record the residual-stream activation of the LAST token at a target layer.
5. Save everything to <data_dir>/code_activations.pkl.

Output schema (list of dicts, one per (function, variant) pair)
---------------------------------------------------------------
{
    "func_id":    int,          # index into the function corpus
    "variant":    "correct" | "buggy",
    "bug_type":   str | None,   # e.g. "off_by_one", None for correct
    "bug_line":   int | None,
    "code":       str,          # the actual source fed to the model
    "activation": np.ndarray,   # shape (d_model,) float32
}

Usage
-----
    python src/harvest_code_activations.py \
        --data_dir data \
        --num_functions 2000 \
        --layer 16 \
        --max_tokens 256

The script is safe to re-run; it skips the download if code_activations.pkl
already exists.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ── import local helper ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from bug_injection import try_all_bug_types


# ── config ────────────────────────────────────────────────────────────────────
CODER_MODEL = "Qwen/Qwen2.5-Coder-0.5B"
DATASET_NAME = "code_search_net"
DATASET_LANG = "python"
MIN_FUNC_TOKENS = 30    # skip very short snippets
MAX_FUNC_CHARS  = 2000  # skip huge functions


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",      default="data")
    p.add_argument("--num_functions", type=int, default=2000,
                   help="Number of distinct functions to process")
    p.add_argument("--layer",         type=int, default=16,
                   help="Which residual-stream layer to read (0-indexed)")
    p.add_argument("--max_tokens",    type=int, default=256,
                   help="Truncate code to this many tokens")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--overwrite",     action="store_true")
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

def load_functions(num_functions: int, seed: int):
    """
    Stream CodeSearchNet Python functions and return a list of source strings.
    Falls back to a small hand-written set if the dataset is unavailable.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset(
            DATASET_NAME, DATASET_LANG,
            split="train", streaming=True,
            trust_remote_code=True,
        )
        funcs = []
        rng = np.random.default_rng(seed)
        for ex in ds:
            src = ex.get("func_code_string", "")
            if (
                src
                and len(src) <= MAX_FUNC_CHARS
                and "def " in src
            ):
                funcs.append(src.strip())
            if len(funcs) >= num_functions * 3:   # oversample for bug filter
                break
        # shuffle deterministically
        idx = rng.permutation(len(funcs))
        return [funcs[i] for i in idx]
    except Exception as e:
        print(f"[warn] Could not load CodeSearchNet ({e}). Using fallback functions.")
        return _fallback_functions()


def _fallback_functions():
    """Small hand-written corpus for offline / CI use."""
    return [
        "def add(a, b):\n    return a + b\n",
        "def subtract(a, b):\n    return a - b\n",
        "def multiply(a, b):\n    return a * b\n",
        "def divide(a, b):\n    if b == 0:\n        return None\n    return a / b\n",
        "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n",
        "def is_even(n):\n    return n % 2 == 0\n",
        "def clamp(x, lo, hi):\n    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n    return x\n",
        "def linear_search(lst, target):\n    for i in range(len(lst)):\n        if lst[i] == target:\n            return i\n    return -1\n",
        "def sum_list(lst):\n    total = 0\n    for x in lst:\n        total = total + x\n    return total\n",
        "def count_vowels(s):\n    count = 0\n    for c in s:\n        if c in 'aeiouAEIOU':\n            count = count + 1\n    return count\n",
    ] * 200   # repeat to reach ~2000


@torch.no_grad()
def get_activation(model, tokenizer, code: str, layer: int, max_tokens: int, device):
    """Return (d_model,) residual-stream activation of the last non-padding token."""
    inputs = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    ).to(device)

    seq_len = inputs["input_ids"].shape[1]
    if seq_len < 2:
        return None

    outputs = model(
        **inputs,
        output_hidden_states=True,
        use_cache=False,
    )

    # hidden_states is a tuple: (embedding, layer1, ..., layerN)
    # index layer+1 because index 0 is the embedding layer output
    hidden = outputs.hidden_states[layer + 1]   # (1, T, D)
    last_tok = hidden[0, -1, :].float().cpu().numpy()  # (D,)
    return last_tok


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_path = os.path.join(args.data_dir, "code_activations.pkl")
    os.makedirs(args.data_dir, exist_ok=True)

    if os.path.exists(out_path) and not args.overwrite:
        print(f"[skip] {out_path} already exists. Use --overwrite to regenerate.")
        return

    # ── load model ────────────────────────────────────────────────────────────
    print(f"Loading {CODER_MODEL} …")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(CODER_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        CODER_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    if args.layer >= n_layers:
        raise ValueError(f"--layer {args.layer} out of range (model has {n_layers} layers)")

    # ── load functions ────────────────────────────────────────────────────────
    print(f"Loading functions …")
    all_funcs = load_functions(args.num_functions, args.seed)

    records = []
    func_id = 0

    for src in tqdm(all_funcs, desc="Processing functions"):
        if func_id >= args.num_functions:
            break

        # ── inject bug ────────────────────────────────────────────────────────
        buggy_src, bug_type, bug_line = try_all_bug_types(src, seed=args.seed + func_id)
        if buggy_src is None:
            continue   # no injection site found — skip

        # ── harvest correct activation ────────────────────────────────────────
        act_correct = get_activation(model, tokenizer, src, args.layer, args.max_tokens, device)
        if act_correct is None:
            continue

        # ── harvest buggy activation ──────────────────────────────────────────
        act_buggy = get_activation(model, tokenizer, buggy_src, args.layer, args.max_tokens, device)
        if act_buggy is None:
            continue

        records.append({
            "func_id":    func_id,
            "variant":    "correct",
            "bug_type":   None,
            "bug_line":   None,
            "code":       src,
            "activation": act_correct,
        })
        records.append({
            "func_id":    func_id,
            "variant":    "buggy",
            "bug_type":   bug_type,
            "bug_line":   bug_line,
            "code":       buggy_src,
            "activation": act_buggy,
        })
        func_id += 1

    print(f"\nHarvested {len(records)} activation records ({func_id} function pairs).")

    # ── diagnostics ───────────────────────────────────────────────────────────
    acts = np.stack([r["activation"] for r in records])
    norms = np.linalg.norm(acts, axis=1)
    print(f"Activation L2 norms — mean: {norms.mean():.2f}, std: {norms.std():.2f}")

    bug_type_counts = {}
    for r in records:
        if r["bug_type"]:
            bug_type_counts[r["bug_type"]] = bug_type_counts.get(r["bug_type"], 0) + 1
    print("Bug type distribution:", bug_type_counts)

    with open(out_path, "wb") as f:
        pickle.dump(records, f)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
