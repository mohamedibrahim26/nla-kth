"""
layer_ablation.py — Reconstructor FVE across transformer layers.

Trains one reconstructor per candidate layer and compares the oracle FVE
(upper bound: teacher summaries → reconstructor) to find which layer's
activations are most linearly predictable from natural language.

We reuse the SAME teacher summaries (from the default data_dir) and only
re-harvest activations at each candidate layer, so the comparison is fair.

Outputs (under <output_dir>/ablation_layer_<L>/ for each L):
  activations.npy         activations at layer L
  reconstructor.pt        trained head + stats
  reconstructor_lora/     LoRA adapter
  ablation_results.json   oracle and random FVE at this layer

A summary table is written to <output_dir>/layer_ablation_summary.json.

Run
───
  # Fast: use cached summaries from data/, only re-harvest + re-train reconstructor
  python src/layer_ablation.py --data_dir data --output_dir data \\
      --layers 8 12 16 20

  # Re-harvest activations for all layers (slow, ~30 min per layer on CPU)
  python src/layer_ablation.py --data_dir data --output_dir data \\
      --layers 8 12 16 20 --force_reharvest
"""

import argparse
import json
import os
import subprocess
import sys

from config import CONFIG


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",        default="data",
                   help="directory that already contains summaries.jsonl, split.json, etc.")
    p.add_argument("--output_dir",      default=None)
    p.add_argument("--layers",          type=int, nargs="+",
                   default=[8, 12, 16, 20],
                   help="layers to ablate (1-indexed, same as harvest_activations --layer)")
    p.add_argument("--model_name",      default=CONFIG.model_name)
    p.add_argument("--num_samples",     type=int, default=CONFIG.num_samples)
    p.add_argument("--force_reharvest", action="store_true",
                   help="re-harvest activations even if activations.npy already exists")
    p.add_argument("--lora_r",          type=int, default=8)
    p.add_argument("--lora_alpha",      type=int, default=16)
    p.add_argument("--n_epochs_recon",  type=int, default=5,
                   help="epochs for reconstructor training at each layer")
    p.add_argument("--seed",            type=int, default=CONFIG.seed)
    return p.parse_args()


def run(cmd, **kwargs):
    """Run a subprocess command and raise on failure."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def eval_reconstructor(layer_dir, data_dir, model_name, recon_batch=16):
    """
    Compute oracle and random FVE for the reconstructor trained at this layer.
    Loads reconstructor.pt + reconstructor_lora/ from layer_dir,
    uses teacher summaries from data_dir.
    Returns dict with fve_oracle, fve_random.
    """
    import numpy as np
    import json as _json
    import random
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    WRAP = "A language model's internal state is described as follows: {s}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load activations + split
    acts = np.load(os.path.join(layer_dir, "activations.npy")).astype("float32")
    rows = [_json.loads(l) for l in open(os.path.join(data_dir, "summaries.jsonl"))]
    idx_to_sum = {r["idx"]: r["summary"] for r in rows}
    split = _json.load(open(os.path.join(layer_dir, "split.json")))
    val_idx = [i for i in split["val_idx"] if i in idx_to_sum][:400]

    targets_raw = torch.tensor(acts[val_idx], dtype=torch.float32)
    summaries   = [idx_to_sum[i] for i in val_idx]

    # Load tokenizer + reconstructor
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    recon_pt = torch.load(os.path.join(layer_dir, "reconstructor.pt"), map_location=device)
    tmean = recon_pt["tmean"].to(device)
    tstd  = recon_pt["tstd"].to(device)
    d     = recon_pt["d"]
    head  = nn.Linear(d, d).to(device)
    head.load_state_dict(recon_pt["head"])
    head.eval()

    base_r   = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    recon_lm = PeftModel.from_pretrained(base_r, os.path.join(layer_dir, "reconstructor_lora")).to(device)
    recon_lm.eval()

    Y_std = ((targets_raw.to(device) - tmean) / tstd).cpu()

    def last_tok(h, m):
        idx = m.sum(1) - 1
        return h[torch.arange(h.shape[0], device=h.device), idx]

    @torch.no_grad()
    def predict(texts):
        preds = []
        for i in range(0, len(texts), recon_batch):
            batch = [WRAP.format(s=t) for t in texts[i:i+recon_batch]]
            enc   = tokenizer(batch, return_tensors="pt", padding=True,
                              truncation=True, max_length=200).to(device)
            out   = recon_lm(**enc, output_hidden_states=True)
            h     = last_tok(out.hidden_states[-1], enc["attention_mask"])
            preds.append(head(h.float()).cpu())
        return torch.cat(preds, 0)

    def fve(preds, targets):
        res_var = (preds - targets).var(dim=0).sum().item()
        tot_var = targets.var(dim=0).sum().item()
        return 1.0 - res_var / (tot_var + 1e-12)

    oracle_pred = predict(summaries)
    fve_oracle  = fve(oracle_pred, Y_std)

    rng = random.Random(0)
    shuffled = summaries[:]
    rng.shuffle(shuffled)
    rand_pred  = predict(shuffled)
    fve_random = fve(rand_pred, Y_std)

    return {"fve_oracle": round(fve_oracle, 4), "fve_random": round(fve_random, 4),
            "n_val": len(val_idx)}


def main():
    args    = parse_args()
    out_dir = args.output_dir or args.data_dir
    os.makedirs(out_dir, exist_ok=True)

    src_dir = os.path.dirname(os.path.abspath(__file__))

    summary_results = {}

    for layer in args.layers:
        print(f"\n{'='*60}")
        print(f"  Layer {layer}")
        print(f"{'='*60}")

        layer_dir = os.path.join(out_dir, f"ablation_layer_{layer}")
        os.makedirs(layer_dir, exist_ok=True)

        acts_path = os.path.join(layer_dir, "activations.npy")

        # Step 1: harvest activations at this layer
        if args.force_reharvest or not os.path.exists(acts_path):
            print(f"[1/3] Harvesting activations at layer {layer}...")
            run([sys.executable,
                 os.path.join(src_dir, "harvest_activations.py"),
                 "--layer",        str(layer),
                 "--model_name",   args.model_name,
                 "--num_samples",  str(args.num_samples),
                 "--data_dir",     layer_dir,
                 "--seed",         str(args.seed)])
        else:
            print(f"[1/3] Activations already exist, skipping harvest.")

        # Copy shared files from data_dir (summaries, metadata, split if already created)
        for fname in ("summaries.jsonl", "metadata.jsonl"):
            src_f = os.path.join(args.data_dir, fname)
            dst_f = os.path.join(layer_dir, fname)
            if os.path.exists(src_f) and not os.path.exists(dst_f):
                import shutil
                shutil.copy2(src_f, dst_f)
                print(f"  Copied {fname} from {args.data_dir}")

        # Step 2: train reconstructor at this layer
        recon_done = os.path.exists(os.path.join(layer_dir, "reconstructor.pt"))
        if not recon_done:
            print(f"[2/3] Training reconstructor at layer {layer}...")
            run([sys.executable,
                 os.path.join(src_dir, "train_reconstructor.py"),
                 "--data_dir",    layer_dir,
                 "--output_dir",  layer_dir,
                 "--model_name",  args.model_name,
                 "--lora_r",      str(args.lora_r),
                 "--lora_alpha",  str(args.lora_alpha),
                 "--n_epochs",    str(args.n_epochs_recon)])
        else:
            print(f"[2/3] Reconstructor already trained, skipping.")

        # Step 3: evaluate oracle FVE
        print(f"[3/3] Evaluating oracle FVE at layer {layer}...")
        result = eval_reconstructor(layer_dir, args.data_dir, args.model_name)
        result["layer"] = layer

        result_path = os.path.join(layer_dir, "ablation_results.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  oracle FVE: {result['fve_oracle']:.4f}  |  "
              f"random FVE: {result['fve_random']:.4f}")

        summary_results[str(layer)] = result

    # Write summary table
    summary_path = os.path.join(out_dir, "layer_ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_results, f, indent=2)
    print(f"\nAblation summary written to: {summary_path}")
    print("\nLayer | Oracle FVE | Random FVE")
    print("-" * 35)
    for layer in args.layers:
        r = summary_results.get(str(layer), {})
        print(f"  {layer:2d}  |   {r.get('fve_oracle', '?'):.4f}   |   {r.get('fve_random', '?'):.4f}")


if __name__ == "__main__":
    main()
