"""
Step 3b of the NLA pipeline: train the VERBALIZER (activation -> English).

Idea
----
We take an activation vector h and use a LoRA-adapted copy of the base model to
GENERATE a short English description of what the model is thinking. The way we
feed the activation to the model is the trick from the paper: we project it into
the embedding space with a learned linear map, then place that vector as the FIRST
input embedding in the prompt (a "soft token"), followed by a fixed instruction.

The model then continues writing the description token by token. We train both
the LoRA adapters and the projection by next-token cross-entropy against the
teacher summaries (this is the warm-start; RL would come next but is out of scope).

Inputs : <data_dir>/activations.npy, <data_dir>/summaries.jsonl, [split.json]
Outputs: <output_dir>/verbalizer.pt          (projection weights + meta)
         <output_dir>/verbalizer_lora/       (LoRA adapter)

Run:
    python src/train_verbalizer.py --data_dir data --output_dir data --epochs 3
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import CONFIG

PROMPT_AFTER = (
    "\n\nDescribe in 80-120 words what a language model is focusing on at this "
    "point: the main topic, the key entities, and what it might predict next.\n"
    "Description: "
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--model_name", default=CONFIG.model_name)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_len", type=int, default=160)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = args.output_dir or args.data_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Device: {device}")

    # --- Load activations + paired summaries ---
    activations = np.load(os.path.join(args.data_dir, "activations.npy")).astype(np.float32)
    rows = [json.loads(l) for l in open(os.path.join(args.data_dir, "summaries.jsonl"), encoding="utf-8")]
    idxs = [r["idx"] for r in rows]
    summaries = [r["summary"] for r in rows]
    acts = activations[idxs]
    # Unit-norm each activation -- stable injection scale
    acts = acts / (np.linalg.norm(acts, axis=1, keepdims=True) + 1e-8)
    acts = torch.tensor(acts, dtype=torch.float32)
    N, d = acts.shape
    print(f"{N} pairs, activation dim {d}.")

    # --- Reuse the reconstructor's split if present, else create one ---
    split_path = os.path.join(args.data_dir, "split.json")
    if os.path.exists(split_path):
        split = json.load(open(split_path))
        train_set = set(split["train_idx"])
        train_pos = [i for i, x in enumerate(idxs) if x in train_set]
        val_pos = [i for i, x in enumerate(idxs) if x not in train_set]
        print(f"Loaded existing split: {len(train_pos)} train, {len(val_pos)} val")
    else:
        perm = np.random.permutation(N)
        n_val = int(N * args.val_frac)
        val_pos = list(perm[:n_val]); train_pos = list(perm[n_val:])
        with open(split_path, "w") as f:
            json.dump({"train_idx": [int(idxs[i]) for i in train_pos],
                       "val_idx": [int(idxs[i]) for i in val_pos]}, f)
        print(f"Created new split: {len(train_pos)} train, {len(val_pos)} val")

    # --- Load model + LoRA ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

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

    embed_dim = model.config.hidden_size
    embed_layer = model.get_input_embeddings()
    projection = nn.Linear(d, embed_dim).to(device)

    # The fixed instruction tokens that always follow the injected activation
    suffix_ids = tokenizer(PROMPT_AFTER, return_tensors="pt",
                           add_special_tokens=False).input_ids[0].to(device)
    P = suffix_ids.shape[0]
    pad_emb = embed_layer(torch.tensor([pad_id], device=device))[0]

    params = [p for p in model.parameters() if p.requires_grad] + list(projection.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    def encode_summary(text):
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False,
                        truncation=True, max_length=args.max_len).input_ids[0]
        return torch.cat([ids, torch.tensor([eos_id])]).to(device)

    def build_batch(positions):
        """Build a padded batch of (inputs_embeds, attention_mask, labels)."""
        seqs = []
        for pos in positions:
            act = acts[pos].to(device)
            sids = encode_summary(summaries[pos])
            act_emb = projection(act.unsqueeze(0))            # (1, E)
            suffix_emb = embed_layer(suffix_ids)              # (P, E)
            summary_emb = embed_layer(sids)                   # (S, E)
            seq_emb = torch.cat([act_emb, suffix_emb, summary_emb], dim=0)
            labels = torch.cat([
                torch.full((1 + P,), -100, dtype=torch.long, device=device),
                sids,
            ])
            seqs.append((seq_emb, labels))
        B = len(seqs)
        L = max(s[0].shape[0] for s in seqs)
        E = embed_dim
        embeds = pad_emb.expand(B, L, E).contiguous().clone()
        labs = torch.full((B, L), -100, dtype=torch.long, device=device)
        attn = torch.zeros(B, L, dtype=torch.long, device=device)
        for k, (emb, lab) in enumerate(seqs):
            ln = emb.shape[0]
            embeds[k, :ln] = emb
            labs[k, :ln] = lab
            attn[k, :ln] = 1
        return embeds, attn, labs

    @torch.no_grad()
    def eval_loss(positions, bs=4, max_batches=50):
        model.eval()
        total, n = 0.0, 0
        for i in range(0, min(len(positions), bs * max_batches), bs):
            sel = positions[i:i + bs]
            if not sel:
                break
            embeds, attn, labs = build_batch(sel)
            out = model(inputs_embeds=embeds, attention_mask=attn, labels=labs)
            total += out.loss.item() * len(sel)
            n += len(sel)
        return total / max(n, 1)

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        order = np.random.permutation(len(train_pos))
        running, n = 0.0, 0
        n_batches = len(train_pos) // args.batch_size
        for i, start in enumerate(range(0, len(train_pos), args.batch_size)):
            sel = [train_pos[j] for j in order[start:start + args.batch_size]]
            embeds, attn, labs = build_batch(sel)
            out = model(inputs_embeds=embeds, attention_mask=attn, labels=labs)
            loss = out.loss
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * len(sel); n += len(sel)
            if i % 50 == 0:
                print(f"  epoch {epoch} step {i}/{n_batches} loss {loss.item():.3f}")
        train_ce = running / max(n, 1)
        val_ce = eval_loss(val_pos)
        print(f"epoch {epoch} | train CE {train_ce:.3f} | val CE {val_ce:.3f}")
        if val_ce < best_val:
            best_val = val_ce
            model.save_pretrained(os.path.join(output_dir, "verbalizer_lora"))
            torch.save({"projection": projection.state_dict(),
                        "embed_dim": embed_dim, "act_dim": d,
                        "prompt_after": PROMPT_AFTER,
                        "model_name": args.model_name},
                       os.path.join(output_dir, "verbalizer.pt"))
    print(f"\nBest val CE: {best_val:.3f}")
    print(f"Saved verbalizer to {output_dir}/verbalizer.pt (+ verbalizer_lora/)")


if __name__ == "__main__":
    main()
