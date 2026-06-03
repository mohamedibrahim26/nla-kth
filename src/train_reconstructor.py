"""
Step 3a of the NLA pipeline: train the RECONSTRUCTOR (English -> activation).

The reconstructor reads a natural-language description and predicts the activation
vector it describes. Targets are z-scored per dimension (using TRAIN statistics) so
every dimension contributes comparably; FVE is reported in this standardized space.

Two modes (compare them in the README):
  --mode frozen : freeze the base model, mean-pool its hidden states, train a small
                  MLP head on those fixed features. Cheap baseline.
  --mode lora   : let the transformer actually read the text. Fine-tune the base with
                  LoRA adapters + a linear head on the last-token final hidden state,
                  trained end-to-end. Much stronger, closer to the paper.

Inputs : <data_dir>/activations.npy, <data_dir>/summaries.jsonl
Outputs: <data_dir>/reconstructor.pt          (head + standardization stats + meta)
         <data_dir>/reconstructor_lora/        (LoRA adapter, lora mode only)
         <data_dir>/split.json                 (train/val indices, reused later)

Run:
    python src/train_reconstructor.py --data_dir data --mode lora --epochs 8
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
    p.add_argument("--mode", choices=["frozen", "lora"], default="lora")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--head_hidden", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=2048)  # used by frozen mode
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_len", type=int, default=160)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def standardize_stats(x):
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True)
    floor = max((1e-2 * std.median()).item(), 1e-6)
    return mean, std.clamp(min=floor)


def last_token(hidden, attn):
    idx = attn.sum(1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | mode: {args.mode}")

    # --- Load activations + paired summaries ---
    activations = np.load(os.path.join(args.data_dir, "activations.npy")).astype(np.float32)
    rows = [json.loads(l) for l in open(os.path.join(args.data_dir, "summaries.jsonl"), encoding="utf-8")]
    idxs = [r["idx"] for r in rows]
    texts = [WRAP.format(s=r["summary"]) for r in rows]
    targets_raw = torch.tensor(activations[idxs], dtype=torch.float32)
    N, d = targets_raw.shape
    print(f"{N} paired examples, activation dim {d}.")

    # --- Train / val split (saved for reuse) ---
    perm = np.random.permutation(N)
    n_val = int(N * args.val_frac)
    val_pos, train_pos = perm[:n_val], perm[n_val:]
    with open(os.path.join(args.data_dir, "split.json"), "w") as f:
        json.dump({"train_idx": [int(idxs[i]) for i in train_pos],
                   "val_idx": [int(idxs[i]) for i in val_pos]}, f)

    # --- Standardize targets using TRAIN stats ---
    tmean, tstd = standardize_stats(targets_raw[train_pos])
    Y = ((targets_raw - tmean) / tstd)
    Ytr_cpu, Yva_cpu = Y[train_pos], Y[val_pos]
    texts_tr = [texts[i] for i in train_pos]
    texts_va = [texts[i] for i in val_pos]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.mode == "frozen":
        # ---- Baseline: frozen encoder + MLP head on cached mean-pooled features ----
        base = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
        base.eval()
        feats = encode_texts(base, tokenizer, texts, device)
        fmean, fstd = standardize_stats(feats[train_pos])
        X = ((feats - fmean) / fstd).to(device)
        Xtr, Xva = X[train_pos], X[val_pos]
        Ytr, Yva = Y[train_pos].to(device), Y[val_pos].to(device)

        head = ReconstructorHead(d, d, hidden=args.hidden).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=args.weight_decay)
        best = -1e9
        for epoch in range(max(args.epochs, 60)):
            head.train(); order = torch.randperm(Xtr.shape[0], device=device)
            for i in range(0, Xtr.shape[0], 128):
                sel = order[i:i + 128]
                loss = ((head(Xtr[sel]) - Ytr[sel]) ** 2).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            head.eval()
            with torch.no_grad():
                fve_va = compute_fve(head(Xva), Yva)
            best = max(best, fve_va)
        print(f"\n[frozen baseline] best val FVE: {best:.3f}")
        return

    # ---- Main: LoRA end-to-end ----
    from peft import LoraConfig, get_peft_model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32).to(device)
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    head = nn.Sequential(
        nn.Linear(d, args.head_hidden), nn.GELU(),
        nn.Linear(args.head_hidden, d),
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    def encode_batch(texts_b):
        return tokenizer(list(texts_b), return_tensors="pt", padding=True,
                         truncation=True, max_length=args.max_len).to(device)

    @torch.no_grad()
    def eval_fve(texts_e, Y_cpu):
        model.eval(); head.eval(); preds = []
        for i in range(0, len(texts_e), args.batch_size):
            enc = encode_batch(texts_e[i:i + args.batch_size])
            h = last_token(model(**enc, output_hidden_states=True).hidden_states[-1], enc["attention_mask"])
            preds.append(head(h.float()).cpu())
        return compute_fve(torch.cat(preds), Y_cpu)

    best = -1e9
    n_tr = len(texts_tr)
    for epoch in range(args.epochs):
        model.train(); head.train()
        order = np.random.permutation(n_tr)
        running = 0.0
        for i in range(0, n_tr, args.batch_size):
            sel = order[i:i + args.batch_size]
            enc = encode_batch([texts_tr[j] for j in sel])
            h = last_token(model(**enc, output_hidden_states=True).hidden_states[-1], enc["attention_mask"])
            pred = head(h.float())
            tgt = Ytr_cpu[sel].to(device)
            loss = ((pred - tgt) ** 2).sum(1).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            running += loss.item()
        fve_va = eval_fve(texts_va, Yva_cpu)
        fve_tr = eval_fve(texts_tr[:500], Ytr_cpu[:500])
        print(f"epoch {epoch} | train-FVE(500) {fve_tr:.3f} | val FVE {fve_va:.3f} | loss {running:.1f}")
        if fve_va > best:
            best = fve_va
            model.save_pretrained(os.path.join(args.data_dir, "reconstructor_lora"))
            torch.save({"head": head.state_dict(), "d": d, "wrap": WRAP, "mode": "lora",
                        "model_name": args.model_name, "tmean": tmean, "tstd": tstd},
                       os.path.join(args.data_dir, "reconstructor.pt"))

    print(f"\nBest validation FVE (oracle text, LoRA): {best:.3f}")
    print(f"Saved reconstructor to {args.data_dir}/reconstructor.pt (+ reconstructor_lora/)")


if __name__ == "__main__":
    main()
