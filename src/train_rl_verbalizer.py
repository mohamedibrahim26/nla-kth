"""
train_rl_verbalizer.py — GRPO RL phase for the NLA verbalizer.

After the warm-start (train_verbalizer.py) the verbalizer has learned to
*imitate* teacher summaries via cross-entropy.  This script closes the
loop: reward = reconstruction quality under the frozen reconstructor.
We use Group Relative Policy Optimization (GRPO; Shao et al. 2024)
because it avoids a separate value-function head while still providing
low-variance advantage estimates through within-group reward normalisation.

Algorithm (one gradient step)
──────────────────────────────
1. Sample B activations from the training split.
2. Generate G candidate descriptions per activation (on-policy, do_sample=True).
3. Pass each description through the frozen reconstructor AR.
   reward_i = −MSE(AR(d_i), z_standardised)   (higher = better reconstruction).
4. Normalise rewards within the group of G:
   Â_i = (R_i − mean_G(R)) / (std_G(R) + ε).
5. Compute sequence log-probs under the current policy π_θ (with gradient)
   and the warm-start reference policy π_ref (frozen snapshot, no gradient).
6. GRPO loss:
     L = −mean(Â_i · log π_θ(o_i | z))       ← policy-gradient term
       + β · mean(log π_θ(o_i | z) − log π_ref(o_i | z))  ← KL penalty
7. Backward + gradient clip + AdamW step.

The reference policy is realised via weight-swapping (no second model copy):
we save the initial LoRA weights, restore them temporarily for the reference
forward-pass, then reload the current weights.

Outputs
───────
  <output_dir>/verbalizer_rl_lora/      LoRA adapter (final epoch)
  <output_dir>/verbalizer_rl_lora_best/ LoRA adapter (best val FVE)
  <output_dir>/verbalizer_rl.pt         projection weights (final)
  <output_dir>/verbalizer_rl_best.pt    projection weights (best)
  <output_dir>/rl_training_log.jsonl    per-step metrics

Run
───
  python src/train_rl_verbalizer.py --data_dir data --output_dir data

Typical GPU run (A100 / L4 / T4):
  python src/train_rl_verbalizer.py \\
      --data_dir data --output_dir data \\
      --n_epochs 3 --batch_size 4 --G 8 \\
      --temperature 0.9 --lr 5e-5 --beta_kl 0.05
"""

import argparse
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import PeftModel

from config import CONFIG
from models import compute_fve


# ── constants ────────────────────────────────────────────────────────────────
WRAP_RECON    = "A language model's internal state is described as follows: {s}"
LOG_INTERVAL  = 10    # print training metrics every N steps
EVAL_INTERVAL = 100   # run validation FVE every N steps


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="GRPO RL fine-tuning of the NLA verbalizer")
    p.add_argument("--data_dir",    default="data")
    p.add_argument("--output_dir",  default=None)
    p.add_argument("--model_name",  default=CONFIG.model_name)
    # RL hyper-parameters
    p.add_argument("--n_epochs",       type=int,   default=3)
    p.add_argument("--batch_size",     type=int,   default=4,
                   help="activations per gradient step (B)")
    p.add_argument("--G",              type=int,   default=8,
                   help="candidate descriptions sampled per activation")
    p.add_argument("--temperature",    type=float, default=0.9,
                   help="sampling temperature for candidate generation")
    p.add_argument("--top_p",          type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int,   default=80)
    p.add_argument("--max_len_recon",  type=int,   default=200,
                   help="max input tokens for the reconstructor")
    # Optimisation
    p.add_argument("--lr",             type=float, default=5e-5)
    p.add_argument("--warmup_ratio",   type=float, default=0.05)
    p.add_argument("--beta_kl",        type=float, default=0.05,
                   help="coefficient for the KL-divergence penalty term")
    p.add_argument("--max_grad_norm",  type=float, default=1.0)
    # Evaluation
    p.add_argument("--recon_batch",    type=int,   default=16,
                   help="batch size used when scoring texts with reconstructor")
    p.add_argument("--n_eval_samples", type=int,   default=200,
                   help="number of validation activations used for periodic eval")
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


# ── helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def last_token_hidden(hidden_states, attention_mask):
    """Return the last non-padding token's hidden state for each sequence."""
    idx = attention_mask.sum(1) - 1
    B   = hidden_states.shape[0]
    return hidden_states[torch.arange(B, device=hidden_states.device), idx]


def pad_sequences(seqs, pad_id, device):
    """
    Right-pad a list of 1-D tensors to the same length.
    Returns (padded: N×L int64, mask: N×L bool).
    """
    max_len = max(s.numel() for s in seqs)
    N       = len(seqs)
    padded  = torch.full((N, max_len), pad_id, dtype=torch.long,  device=device)
    mask    = torch.zeros(N, max_len,           dtype=torch.bool,  device=device)
    for i, s in enumerate(seqs):
        L = s.numel()
        padded[i, :L] = s
        mask[i,   :L] = True
    return padded, mask


def sequence_log_probs(model, prompt_embeds, gen_ids, gen_mask):
    """
    Sum of per-token log-probabilities for each generated sequence.

    prompt_embeds : (N, P, E)  — may carry gradients
    gen_ids       : (N, L)     — padded token IDs, int64
    gen_mask      : (N, L)     — bool, True for real (non-padding) tokens
    returns       : (N,)       — total log-prob per sequence

    The prompt is supplied as embeddings so we cannot invert it to discrete
    IDs.  Full input = [prompt_embeds || embed(gen_ids)].  The logit at
    position P+j-1 predicts gen_ids[:, j], so the relevant slice is
    logits[:, P-1 : P+L-1, :].
    """
    N, P, E = prompt_embeds.shape
    _, L    = gen_ids.shape

    embed_fn  = model.get_input_embeddings()
    gen_embs  = embed_fn(gen_ids)                                      # (N, L, E)
    full_embs = torch.cat([prompt_embeds, gen_embs], dim=1)            # (N, P+L, E)
    attn_mask = torch.ones(N, P + L, dtype=torch.long,
                           device=full_embs.device)

    logits        = model(inputs_embeds=full_embs,
                          attention_mask=attn_mask).logits              # (N, P+L, V)
    rel_logits    = logits[:, P - 1 : P + L - 1, :].contiguous()      # (N, L, V) – own storage
    del logits                                                           # free (N, P+L, V) early
    # Memory-efficient log-prob: avoid materialising (N, L, V) log_softmax output
    sel_logits    = rel_logits.gather(
                        2, gen_ids.unsqueeze(-1)).squeeze(-1)           # (N, L)
    log_Z         = torch.logsumexp(rel_logits, dim=-1)                 # (N, L)
    del rel_logits                                                       # free (N, L, V) ~742 MB
    token_lp      = sel_logits - log_Z                                  # (N, L)
    token_lp      = token_lp * gen_mask.float()                        # zero padding
    return token_lp.sum(dim=1)                                          # (N,)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    set_seed(args.seed)
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = args.output_dir or args.data_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"Device: {device}  |  B={args.batch_size}  G={args.G}  beta_kl={args.beta_kl}")

    # ── data ─────────────────────────────────────────────────────────────────
    acts_all  = np.load(os.path.join(args.data_dir, "activations.npy")).astype(np.float32)
    split     = json.load(open(os.path.join(out_dir, "split.json")))
    train_idx = split["train_idx"]
    val_idx   = split["val_idx"][: args.n_eval_samples]
    print(f"Train: {len(train_idx)}  |  Val subset: {len(val_idx)}")

    acts_train = torch.tensor(acts_all[train_idx], dtype=torch.float32)
    acts_val   = torch.tensor(acts_all[val_idx],   dtype=torch.float32)

    # ── tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    # ── frozen reconstructor ─────────────────────────────────────────────────
    print("Loading reconstructor (frozen)...")
    recon_pt   = torch.load(os.path.join(out_dir, "reconstructor.pt"),
                            map_location=device)
    tmean      = recon_pt["tmean"].to(device)
    tstd       = recon_pt["tstd"].to(device)
    d          = recon_pt["d"]
    recon_head = nn.Linear(d, d).to(device)
    recon_head.load_state_dict(recon_pt["head"])
    recon_head.eval()
    for p in recon_head.parameters():
        p.requires_grad = False

    base_r   = AutoModelForCausalLM.from_pretrained(
                   args.model_name, torch_dtype=torch.float32).to(device)
    recon_lm = PeftModel.from_pretrained(
                   base_r, os.path.join(out_dir, "reconstructor_lora")).to(device)
    recon_lm.eval()
    for p in recon_lm.parameters():
        p.requires_grad = False

    @torch.no_grad()
    def score_texts(texts):
        """
        Encode each text with the frozen reconstructor.
        Returns predicted activations in standardised space: (N, D).
        """
        preds = []
        for i in range(0, len(texts), args.recon_batch):
            batch = [WRAP_RECON.format(s=t) for t in texts[i : i + args.recon_batch]]
            enc   = tokenizer(batch, return_tensors="pt", padding=True,
                              truncation=True,
                              max_length=args.max_len_recon).to(device)
            out   = recon_lm(**enc, output_hidden_states=True)
            h     = last_token_hidden(out.hidden_states[-1], enc["attention_mask"])
            preds.append(recon_head(h.float()))
        return torch.cat(preds, dim=0)

    # ── verbalizer policy ────────────────────────────────────────────────────
    print("Loading verbalizer (policy)...")
    vp          = torch.load(os.path.join(out_dir, "verbalizer.pt"),
                             map_location=device)
    act_dim     = vp["act_dim"]
    embed_dim   = vp["embed_dim"]
    prompt_after = vp["prompt_after"]

    base_v = AutoModelForCausalLM.from_pretrained(
                 args.model_name, torch_dtype=torch.float32).to(device)
    for p in base_v.parameters():
        p.requires_grad = False

    verb_lora_path = os.path.join(out_dir, "verbalizer_lora")
    policy_model   = PeftModel.from_pretrained(
                         base_v, verb_lora_path).to(device)

    for name, param in policy_model.named_parameters():
        param.requires_grad = "lora_" in name

    embed_layer = policy_model.get_input_embeddings()

    projection = nn.Linear(act_dim, embed_dim, bias=False).to(device)
    projection.load_state_dict(vp["projection"])
    projection.requires_grad_(True)

    suffix_ids = tokenizer(
        prompt_after, return_tensors="pt", add_special_tokens=False
    ).input_ids[0].to(device)
    P = 1 + suffix_ids.numel()   # prompt length = 1 soft token + suffix

    # ── reference snapshot (weight-swap approach) ─────────────────────────────
    # Keep a frozen copy of the initial LoRA + projection weights.
    # For reference log-probs we swap in these weights, forward-pass,
    # then swap back to the current weights.
    ref_lora_state = {
        k: v.clone().detach()
        for k, v in policy_model.named_parameters() if v.requires_grad
    }
    ref_proj_state = {
        k: v.clone().detach()
        for k, v in projection.named_parameters()
    }

    @torch.no_grad()
    def reference_log_probs(prompt_e, gen_ids, gen_mask):
        """Return sequence log-probs under the frozen warm-start reference policy."""
        # Save current weights
        curr_lora = {k: v.data.clone()
                     for k, v in policy_model.named_parameters() if v.requires_grad}
        curr_proj = {k: v.data.clone()
                     for k, v in projection.named_parameters()}
        # Swap in reference weights
        for k, v in policy_model.named_parameters():
            if k in ref_lora_state:
                v.data.copy_(ref_lora_state[k])
        for k, v in projection.named_parameters():
            if k in ref_proj_state:
                v.data.copy_(ref_proj_state[k])
        # Forward pass (no grad)
        policy_model.eval()
        ref_lp = sequence_log_probs(policy_model, prompt_e, gen_ids, gen_mask)
        # Restore current weights
        for k, v in policy_model.named_parameters():
            if k in curr_lora:
                v.data.copy_(curr_lora[k])
        for k, v in projection.named_parameters():
            if k in curr_proj:
                v.data.copy_(curr_proj[k])
        return ref_lp

    # ── optimiser + scheduler ────────────────────────────────────────────────
    trainable_params = (
        [p for p in policy_model.parameters() if p.requires_grad]
        + list(projection.parameters())
    )
    n_steps_per_epoch = math.ceil(len(train_idx) / args.batch_size)
    total_steps       = n_steps_per_epoch * args.n_epochs
    warmup_steps      = int(total_steps * args.warmup_ratio)
    print(f"Total steps: {total_steps}  |  Warmup: {warmup_steps}")

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
                    optimizer, warmup_steps, total_steps)

    # ── validation helper ─────────────────────────────────────────────────────
    def run_eval(step):
        policy_model.eval()
        gen_texts = []
        with torch.no_grad():
            acts_n = acts_val / (acts_val.norm(dim=1, keepdim=True) + 1e-8)
            for i in range(0, len(acts_val), args.batch_size):
                chunk    = acts_n[i : i + args.batch_size].to(device)
                B2       = chunk.shape[0]
                act_emb  = projection(chunk).unsqueeze(1)
                suf_emb  = embed_layer(suffix_ids).unsqueeze(0).expand(B2, -1, -1)
                prompt_e = torch.cat([act_emb, suf_emb], dim=1)
                attn     = torch.ones(B2, P, dtype=torch.long, device=device)
                gen_out  = policy_model.generate(
                               inputs_embeds=prompt_e, attention_mask=attn,
                               max_new_tokens=args.max_new_tokens,
                               do_sample=False, pad_token_id=pad_id)
                for row in gen_out:
                    gen_texts.append(
                        tokenizer.decode(row, skip_special_tokens=True).strip())

        acts_std = (acts_val.to(device) - tmean) / tstd
        preds    = score_texts(gen_texts)
        fve      = compute_fve(preds.cpu(), acts_std.cpu())
        print(f"  [step {step:5d}]  val FVE (generated): {fve:.4f}")
        policy_model.train()
        return float(fve)

    # ── training loop ─────────────────────────────────────────────────────────
    log_path = os.path.join(out_dir, "rl_training_log.jsonl")
    log_fh   = open(log_path, "w", encoding="utf-8")
    global_step = 0
    best_fve    = -float("inf")

    for epoch in range(args.n_epochs):
        perm = torch.randperm(len(train_idx)).tolist()

        for batch_start in range(0, len(perm), args.batch_size):
            batch_pos  = perm[batch_start : batch_start + args.batch_size]
            if not batch_pos:
                break
            B          = len(batch_pos)
            acts_batch = acts_train[batch_pos].to(device)

            # 1. Generate G candidates per activation ──────────────────────
            policy_model.eval()
            with torch.no_grad():
                acts_n   = acts_batch / (acts_batch.norm(dim=1, keepdim=True) + 1e-8)
                acts_rep = acts_n.repeat_interleave(args.G, dim=0)    # (B*G, D)

                act_embs = projection(acts_rep).unsqueeze(1)           # (B*G, 1, E)
                suf_embs = (embed_layer(suffix_ids)
                            .unsqueeze(0).expand(B * args.G, -1, -1)) # (B*G, S, E)
                prompt_e = torch.cat([act_embs, suf_embs], dim=1)     # (B*G, P, E)
                attn_gen = torch.ones(B * args.G, P,
                                      dtype=torch.long, device=device)

                gen_out  = policy_model.generate(
                               inputs_embeds=prompt_e, attention_mask=attn_gen,
                               max_new_tokens=args.max_new_tokens,
                               do_sample=True,
                               temperature=args.temperature, top_p=args.top_p,
                               pad_token_id=pad_id)

            # Trim each sequence at the first EOS token (inclusive)
            gen_ids_list = []
            for row in gen_out:
                eos_pos = (row == eos_id).nonzero(as_tuple=True)[0]
                end     = eos_pos[0].item() + 1 if eos_pos.numel() > 0 else row.numel()
                gen_ids_list.append(row[:end].detach())

            texts = [tokenizer.decode(g, skip_special_tokens=True).strip()
                     for g in gen_ids_list]

            # 2. Score with frozen reconstructor ───────────────────────────
            with torch.no_grad():
                preds_std    = score_texts(texts)                       # (B*G, D)
                acts_std     = (acts_batch - tmean) / tstd              # (B, D)
                acts_std_rep = acts_std.repeat_interleave(args.G, dim=0)
                mse          = ((preds_std - acts_std_rep) ** 2).mean(dim=1)
                rewards      = (-mse).reshape(B, args.G)               # (B, G)

            # 3. Group-normalise advantages ────────────────────────────────
            r_mean    = rewards.mean(dim=1, keepdim=True)
            r_std     = rewards.std(dim=1,  keepdim=True).clamp(min=1e-8)
            advantages = (rewards - r_mean) / r_std                    # (B, G)
            adv_flat   = advantages.reshape(B * args.G)

            # 4. Current-policy log-probs (with gradient) ──────────────────
            policy_model.train()

            acts_rep_d = acts_n.detach().repeat_interleave(args.G, dim=0)
            act_embs_g = projection(acts_rep_d).unsqueeze(1)           # (B*G, 1, E)
            suf_embs_d = (embed_layer(suffix_ids).detach()
                          .unsqueeze(0).expand(B * args.G, -1, -1))    # (B*G, S, E)
            prompt_e_g = torch.cat([act_embs_g, suf_embs_d], dim=1)   # (B*G, P, E)

            gen_padded, gen_mask = pad_sequences(gen_ids_list, pad_id, device)

            policy_lp = sequence_log_probs(
                            policy_model, prompt_e_g, gen_padded, gen_mask)

            # 5. Reference-policy log-probs (no gradient) ──────────────────
            ref_lp = reference_log_probs(
                         prompt_e_g.detach(),
                         gen_padded.detach(),
                         gen_mask.detach())

            # 6. GRPO loss ─────────────────────────────────────────────────
            pg_loss = -(adv_flat.detach() * policy_lp).mean()
            kl_loss = (policy_lp - ref_lp.detach()).mean()
            loss    = pg_loss + args.beta_kl * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1

            # Logging ──────────────────────────────────────────────────────
            if global_step % LOG_INTERVAL == 0:
                entry = {
                    "step":        global_step,
                    "epoch":       epoch,
                    "loss":        round(loss.item(),    6),
                    "pg_loss":     round(pg_loss.item(), 6),
                    "kl_loss":     round(kl_loss.item(), 6),
                    "mean_reward": round(rewards.mean().item(), 6),
                    "lr":          scheduler.get_last_lr()[0],
                }
                log_fh.write(json.dumps(entry) + "\n"); log_fh.flush()
                print(f"[{global_step:5d}] loss={loss.item():.4f}  "
                      f"pg={pg_loss.item():.4f}  kl={kl_loss.item():.4f}  "
                      f"R={rewards.mean().item():.4f}")

            # Periodic validation ──────────────────────────────────────────
            if global_step % EVAL_INTERVAL == 0:
                val_fve = run_eval(global_step)
                log_fh.write(json.dumps(
                    {"step": global_step, "val_fve": val_fve}) + "\n")
                log_fh.flush()
                if val_fve > best_fve:
                    best_fve = val_fve
                    best_lora = os.path.join(out_dir, "verbalizer_rl_lora_best")
                    policy_model.save_pretrained(best_lora)
                    torch.save({
                        "act_dim":      act_dim,
                        "embed_dim":    embed_dim,
                        "prompt_after": prompt_after,
                        "projection":   projection.state_dict(),
                        "step":         global_step,
                        "best_fve":     best_fve,
                    }, os.path.join(out_dir, "verbalizer_rl_best.pt"))
                    print(f"    * new best  val FVE: {best_fve:.4f}")

        # End-of-epoch validation
        epoch_fve = run_eval(global_step)
        log_fh.write(json.dumps({"epoch": epoch, "val_fve": epoch_fve}) + "\n")
        log_fh.flush()

    log_fh.close()

    # Save final checkpoint ──────────────────────────────────────────────────
    rl_lora_dir = os.path.join(out_dir, "verbalizer_rl_lora")
    policy_model.save_pretrained(rl_lora_dir)
    torch.save({
        "act_dim":      act_dim,
        "embed_dim":    embed_dim,
        "prompt_after": prompt_after,
        "projection":   projection.state_dict(),
        "step":         global_step,
        "best_fve":     best_fve,
    }, os.path.join(out_dir, "verbalizer_rl.pt"))

    print(f"\nRL training complete.")
    print(f"  Best val FVE : {best_fve:.4f}")
    print(f"  LoRA (final) : {rl_lora_dir}")
    print(f"  Log          : {log_path}")


if __name__ == "__main__":
    main()
