"""
Build the two figures used in the README:

  results/plots/fve_bar.png            - FVE bar chart across all eval conditions
  results/plots/reconstructor_curve.png - reconstructor train vs val FVE per epoch

Inputs (auto-detected; both optional):
  <data_dir>/nla_results.json         - from evaluate_nla.py
  <data_dir>/paraphrase_results.json  - from paraphrase_test.py  (optional)

Run:
    python src/plot_results.py --data_dir data --output_dir results/plots
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


# Reconstructor training trajectory captured from the final Kaggle run.
# (Hardcoded so the plot is reproducible without re-running training.)
RECON_TRAIN_FVE = [0.019, 0.064, 0.103, 0.157, 0.221, 0.285, 0.349, 0.409, 0.468, 0.517, 0.560, 0.597]
RECON_VAL_FVE   = [0.012, 0.040, 0.050, 0.045, 0.031, 0.020, 0.009, -0.012, -0.029, -0.056, -0.079, -0.097]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--output_dir", default="results/plots")
    return p.parse_args()


def plot_fve_bar(data_dir, out_dir):
    """Bar chart of every FVE condition we measured, ordered to match the README table."""
    nla_path = os.path.join(data_dir, "nla_results.json")
    para_path = os.path.join(data_dir, "paraphrase_results.json")
    nla = json.load(open(nla_path)) if os.path.exists(nla_path) else None
    para = json.load(open(para_path)) if os.path.exists(para_path) else None

    bars = []
    if nla:
        bars.append(("Oracle text\n(teacher summary)", nla["fve_oracle_text"], "#3b82f6"))
    if para:
        bars.append(("Paraphrased\n(meaning preserved)", para["fve_paraphrased"], "#06b6d4"))
    if nla:
        bars.append(("Generated text\n(verbalizer NLA)", nla["fve_generated_text"], "#10b981"))
    if para:
        bars.append(("Sentence-shuffled\ngenerated", para["fve_sentence_shuffled"], "#f59e0b"))
        bars.append(("Word-shuffled\ngenerated", para["fve_word_shuffled"], "#dc2626"))
    if nla:
        bars.append(("Predict mean\n(baseline)", nla["fve_predict_mean"], "#a3a3a3"))
        bars.append(("Random text\n(shuffled summary)", nla["fve_random_text"], "#ef4444"))

    labels = [b[0] for b in bars]
    values = [b[1] for b in bars]
    colors = [b[2] for b in bars]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars_drawn = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Fraction of Variance Explained (FVE)")
    ax.set_title("Reconstruction quality across conditions (n = 400 held-out activations)")
    for bar, v in zip(bars_drawn, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + (0.003 if v >= 0 else -0.006),
                f"{v:+.3f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=0, fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fve_bar.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_recon_curve(out_dir):
    """Reconstructor train vs val FVE over epochs (oracle-text training)."""
    epochs = list(range(len(RECON_TRAIN_FVE)))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, RECON_TRAIN_FVE, marker="o", label="Train FVE", color="#10b981")
    ax.plot(epochs, RECON_VAL_FVE, marker="s", label="Val FVE (oracle text)", color="#3b82f6")
    best_e = int(np.argmax(RECON_VAL_FVE))
    ax.axvline(best_e, linestyle="--", color="black", alpha=0.4)
    ax.text(best_e + 0.1, max(RECON_VAL_FVE),
            f"best val = {max(RECON_VAL_FVE):.3f} @ epoch {best_e}",
            fontsize=9, va="top")
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("FVE")
    ax.set_title("Reconstructor training (Qwen2.5-0.5B, layer 16, LoRA r=32)")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "reconstructor_curve.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    plot_fve_bar(args.data_dir, args.output_dir)
    plot_recon_curve(args.output_dir)


if __name__ == "__main__":
    main()
