"""
Step 3a of the NLA pipeline: train the RECONSTRUCTOR (English -> activation).

Idea
----
The reconstructor reads a natural-language description and predicts the activation
vector it corresponds to:

    frozen base model (text encoder)  ->  small trainable MLP head  ->  activation

Because the base model is frozen, we encode every summary ONCE into a feature
vector, cache those, and train only the small MLP head. Fast and easy to iterate.

Important detail: per-dimension standardization
------------------------------------------------
LLM residual streams contain a few "massive activation" dimensions whose values
dwarf all others. If we just normalise each activation to unit norm, those few dims
dominate and every target looks nearly identical -> almost no variance to explain ->
FVE collapses to ~0. The raw text features have the same uneven scale, which also
stalls optimisation. So we z-score BOTH features and targets per dimension (using
training-set statistics). FVE is then reported in this standardized space: the
average fraction of each dimension's variance that we recover. We also print
diagnostics (variance concentration, a linear-regression baseline).

Inputs : <data_dir>/activations.npy, <data_dir>/summaries.jsonl
Outputs: <data_dir>/reconstructor.pt (head weights + standardization stats)
         <data_dir>/split.json       (train/val indices, reused by later steps)

Run:
    python src/train_reconstructor.py --data_dir data --epochs 100
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import CONFIG
from models import ReconstructorHead, compute_fve, encode_texts

WRAP = "A language model's internal state is described as follows: {s}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--model_name", default=CONFIG.model_name)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def standardize_stats(x):
    """Per-dimension mean and (floored) std."""
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True)
    floor = 1e-2 * std.median()
    std = std.clamp(min=max(floor.item(), 1e-6))
    return mean, std


def ridge_baseline(Xtr, Ytr, Xva, Yva, lam=10.0):
    """Closed-form ridge regression -> a quick 'is anything learnable?' ceiling."""
    ones_tr = torch.ones(Xtr.shape[0], 1, device=Xtr.device)
    ones_va = torch.ones(Xva.shape[0], 1, device=Xva.device)
    Xb = torch.cat([Xtr, ones_tr], dim=1)
    A = Xb.T @ Xb + lam * torch.eye(Xb.shape[1], device=Xb.device)
    W = torch.linalg.solve(A, Xb.T @ Ytr)
    pred_va = torch.cat([Xva, ones_va], dim=1) @ W
    return compute_fve(pred_va, Yva)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Load activations and paired summaries ---
    activations = np.load(os.path.join(args.data_dir, "activations.npy")).astype(np.float32)
    rows = [json.loads(l) for l in open(os.path.join(args.data_dir, "summaries.jsonl"), encoding="utf-8")]
    idxs = [r["idx"] for r in rows]
    texts = [WRAP.format(s=r["summary"]) for r in rows]
    targets_raw = torch.tensor(activations[idxs], dtype=torch.float32)
    N, d = targets_raw.shape
    print(f"{N} paired examples, activation dim {d}.")

    # --- Diagnostic: how concentrated is the activation variance? ---
    var_per_dim = targets_raw.var(dim=0)
    total_var = var_per_dim.sum().item()
    top1 = var_per_dim.max().item() / total_var
    top5 = var_per_dim.topk(5).values.sum().item() / total_var
    print(f"[diag] variance share -> top-1 dim: {top1:.1%}, top-5 dims: {top5:.1%}")
    print(f"[diag] max |mean| over dims: {targets_raw.mean(0).abs().max().item():.1f}")

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

    # --- Train / val split (saved for reuse) ---
    perm = np.random.permutation(N)
    n_val = int(N * args.val_frac)
    val_pos, train_pos = perm[:n_val], perm[n_val:]
    with open(os.path.join(args.data_dir, "split.json"), "w") as f:
        json.dump({"train_idx": [int(idxs[i]) for i in train_pos],
                   "val_idx": [int(idxs[i]) for i in val_pos]}, f)

    # --- Standardize features and targets using TRAIN stats only ---
    fmean, fstd = standardize_stats(feats[train_pos])
    tmean, tstd = standardize_stats(targets_raw[train_pos])
    X = ((feats - fmean) / fstd).to(device)
    Y = ((targets_raw - tmean) / tstd).to(device)
    Xtr, Ytr = X[train_pos], Y[train_pos]
    Xva, Yva = X[val_pos], Y[val_pos]

    # --- Linear baseline (sanity ceiling) ---
    lin_fve = ridge_baseline(Xtr, Ytr, Xva, Yva)
    print(f"[diag] ridge linear baseline val FVE: {lin_fve:.3f}")

    # --- Train the MLP head ---
    head = ReconstructorHead(d, d, hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_fve, bs = -1e9, 128
    for epoch in range(args.epochs):
        head.train()
        order = torch.randperm(Xtr.shape[0], device=device)
        for i in range(0, Xtr.shape[0], bs):
            sel = order[i:i + bs]
            loss = ((head(Xtr[sel]) - Ytr[sel]) ** 2).sum(dim=1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            fve_tr = compute_fve(head(Xtr), Ytr)
            fve_va = compute_fve(head(Xva), Yva)
        if fve_va > best_fve:
            best_fve = fve_va
            torch.save({"head": head.state_dict(), "d": d, "hidden": args.hidden,
                        "wrap": WRAP, "model_name": args.model_name,
                        "fmean": fmean, "fstd": fstd, "tmean": tmean, "tstd": tstd},
                       os.path.join(args.data_dir, "reconstructor.pt"))
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d} | train FVE {fve_tr:.3f} | val FVE {fve_va:.3f}")

    print(f"\nBest validation FVE (oracle text): {best_fve:.3f}")
    print(f"Saved reconstructor to {args.data_dir}/reconstructor.pt")


if __name__ == "__main__":
    main()
