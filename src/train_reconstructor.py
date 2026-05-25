"""
Step 3a of the NLA pipeline: train the RECONSTRUCTOR (English -> activation).

Idea
----
The reconstructor reads a natural-language description and predicts the activation
vector it corresponds to. We build it as:

    frozen base model (text encoder)  ->  small trainable MLP head  ->  activation

Because the base model is frozen, we can encode every summary ONCE into a feature
vector, cache those features, and then train only the tiny MLP head. This is fast
and lets us iterate quickly.

Targets are the activations normalised to unit L2 norm (as in the paper). We report
the Fraction of Variance Explained (FVE) on a held-out validation split. Trained on
the *teacher's* text, this number is our "oracle text" reconstruction quality -- an
upper-ish bound on what the verbalizer can achieve once it generates its own text.

Inputs : <data_dir>/activations.npy, <data_dir>/summaries.jsonl
Outputs: <data_dir>/reconstructor.pt   (MLP head weights + config)
         <data_dir>/split.json         (train/val indices, reused by later steps)

Run:
    python src/train_reconstructor.py --data_dir data --epochs 60
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import CONFIG
from models import ReconstructorHead, compute_fve, encode_texts

WRAP = "A language model's internal state is described as follows: {s}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--model_name", default=CONFIG.model_name)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Load activations and the summaries that pair with them ---
    activations = np.load(os.path.join(args.data_dir, "activations.npy")).astype(np.float32)
    rows = [json.loads(l) for l in open(os.path.join(args.data_dir, "summaries.jsonl"), encoding="utf-8")]
    idxs = [r["idx"] for r in rows]
    texts = [WRAP.format(s=r["summary"]) for r in rows]
    targets = activations[idxs]                       # (N, d)
    targets = targets / (np.linalg.norm(targets, axis=1, keepdims=True) + 1e-8)  # unit norm
    targets = torch.tensor(targets, dtype=torch.float32)
    N, d = targets.shape
    print(f"{N} paired examples, activation dim {d}.")

    # --- Encode all summaries once with the frozen base model ---
    print("Loading frozen encoder and extracting text features (one-time)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    base.eval()
    feats = encode_texts(base, tokenizer, texts, device)   # (N, d) CPU tensor
    print(f"features: {tuple(feats.shape)}")

    # --- Train / val split (saved for reuse by later phases) ---
    perm = np.random.permutation(N)
    n_val = int(N * args.val_frac)
    val_pos, train_pos = perm[:n_val], perm[n_val:]
    split = {"train_idx": [int(idxs[i]) for i in train_pos],
             "val_idx": [int(idxs[i]) for i in val_pos]}
    with open(os.path.join(args.data_dir, "split.json"), "w") as f:
        json.dump(split, f)

    Xtr, Ytr = feats[train_pos].to(device), targets[train_pos].to(device)
    Xva, Yva = feats[val_pos].to(device), targets[val_pos].to(device)

    # --- Train the MLP head ---
    head = ReconstructorHead(d, d, hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    best_fve = -1e9
    bs = 128
    for epoch in range(args.epochs):
        head.train()
        order = torch.randperm(Xtr.shape[0], device=device)
        for i in range(0, Xtr.shape[0], bs):
            sel = order[i:i + bs]
            pred = head(Xtr[sel])
            loss = ((pred - Ytr[sel]) ** 2).sum(dim=1).mean()
            opt.zero_grad(); loss.backward(); opt.step()

        head.eval()
        with torch.no_grad():
            fve_tr = compute_fve(head(Xtr), Ytr)
            fve_va = compute_fve(head(Xva), Yva)
        if fve_va > best_fve:
            best_fve = fve_va
            torch.save({"head": head.state_dict(), "d": d, "hidden": args.hidden,
                        "wrap": WRAP, "model_name": args.model_name},
                       os.path.join(args.data_dir, "reconstructor.pt"))
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d} | train FVE {fve_tr:.3f} | val FVE {fve_va:.3f}")

    print(f"\nBest validation FVE (oracle text): {best_fve:.3f}")
    print(f"Saved reconstructor to {args.data_dir}/reconstructor.pt")


if __name__ == "__main__":
    main()
