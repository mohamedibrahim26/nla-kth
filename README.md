# Natural Language Autoencoders on a Small Open Model

A small-scale, faithful reimplementation of the natural language autoencoder
(NLA) idea from Anthropic's *Natural Language Autoencoders Produce Unsupervised
Explanations of LLM Activations* ([transformer-circuits.pub/2026/nla](https://transformer-circuits.pub/2026/nla/index.html)).
Built for the KTH ASSERT-lab PhD recruitment task (Prof. Monperrus).

> **TL;DR.** I trained the two paired components from the paper — an *activation
> verbalizer* and an *activation reconstructor* — on `Qwen2.5-0.5B-Instruct`, on a
> free Colab T4 and then Kaggle's 2× T4. On a 400-sample held-out set, the
> reconstructor recovers **3.0% of the variance** of layer-16 activations when
> fed the verbalizer's *own* generated English, versus **5.0%** when fed the
> teacher's oracle summary and **−6.5%** when fed random English (negative
> control). The numbers are small because the model is small and we only
> warm-started (no RL), but the *structure* of the results matches the paper.
> The most interesting finding is qualitative: the verbalizer's confabulations
> are **thematically faithful but specifically wrong**, exactly as the paper
> reports on Claude-scale models.

---

## 1. What this is and why it matters

The Anthropic paper proposes training an autoencoder whose **bottleneck is plain
English**: a *verbalizer* converts a single activation vector `h` into a short
natural-language description `z`, and a *reconstructor* maps `z` back to a
predicted activation `ĥ`. If the round-trip preserves the activation (high
Fraction of Variance Explained, FVE), the English `z` must genuinely capture
what the model was "thinking" at that point — yielding human-readable,
unsupervised explanations of internal model state.

I reimplemented this pipeline at the smallest sensible scale: a recent 0.5B
open model, single layer, single token, a few thousand examples, and a
warm-started (SFT-only) verbalizer/reconstructor pair. No RL.

The objective for the recruitment task is not to match Claude-scale numbers but
to (a) implement the actual generative bottleneck, (b) measure it honestly with
appropriate baselines, and (c) understand why the small-scale numbers look the
way they do. That is what this README aims to demonstrate.

## 2. Target model and harvesting setup

**Target model.** [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
Recent (late-2024 release), capable for its size, 24 layers, 896 hidden dim,
runs comfortably on a single T4.

**Where we read.** Residual stream output of **layer 16** (≈ 2/3 of depth, the
paper's recommended region) at the **final token** of randomly truncated text
snippets from `wikitext-2-raw-v1`. Random truncation samples diverse "thinking
points" rather than always grabbing sentence-final positions.

**Scale.** 8,000 (snippet, activation) pairs. The L2 norms are tight (mean 21.1,
std 1.3), confirming a healthy distribution — see [`src/harvest_activations.py`](src/harvest_activations.py).

## 3. Teacher summaries (warm-start targets)

The verbalizer needs a reasonable starting point — a freshly initialised model
emits gibberish that the reconstructor cannot use. As in the paper, a
**stronger frozen model** writes example descriptions, and the verbalizer is
trained to imitate them.

I deliberately chose a **fully self-contained, free** teacher: `Qwen2.5-3B-Instruct`
in 4-bit, running inside the same Colab notebook. This costs nothing, requires
no API keys, and lets anyone (including the reviewer) rerun the entire pipeline
end-to-end. The trade-off is slightly lower summary quality than a frontier API
would give; that limitation is reported honestly in §7.

The prompt asks the teacher to describe, in 80–120 words, what a language model
would be focusing on at the truncation point (topic, entities, genre, predicted
next content). See [`src/generate_teacher_summaries.py`](src/generate_teacher_summaries.py).
I summarised 4,000 of the 8,000 snippets — enough for a meaningful warm-start
split, leaving the remaining activations available for RL or evaluation
extensions.

## 4. The autoencoder

Both components are **LoRA-adapted copies of the same base model**, with the
base weights frozen and only the small adapters trained. This is a faithful
simplification of the paper's setup (their AR and AV are full finetuned copies;
LoRA gives us most of the capacity at a fraction of the compute, and the paper
notes finetunes of the base model transfer well to this role).

### 4.1 Reconstructor (English → activation)

Implementation: [`src/train_reconstructor.py`](src/train_reconstructor.py).

The reconstructor reads the description, takes the **last-token hidden state of
the final layer**, and passes it through a learned linear head to predict the
standardized activation. LoRA (`r=32`, all attention and MLP projections) makes
the transformer adapt to the new objective. Targets are **z-scored per
dimension** using training statistics — without this, optimisation stalls (see
§6.1).

Trained for 12 epochs with batch 16 and `lr=2e-4` on Kaggle's 2× T4. Train FVE
climbs cleanly from 0.02 → 0.60; best **validation FVE on the oracle (teacher)
text is 0.050** at epoch 2, after which the model overfits. The saved checkpoint
is the best-val one.

### 4.2 Verbalizer (activation → English)

Implementation: [`src/train_verbalizer.py`](src/train_verbalizer.py).

The verbalizer is again a LoRA copy of the base, with a learned linear
projection `R^896 → R^896` that maps a unit-normalized activation vector into
the model's token-embedding space. That projected vector is placed as the
**first input embedding** (a "soft prompt"), followed by a fixed natural-language
instruction asking the model to describe what a language model is focusing on.
The model then generates the description token by token. Training is standard
next-token cross-entropy against the teacher summaries — the warm-start.

Three epochs, batch 4, `lr=2e-4`. Best **val cross-entropy 1.46** at epoch 1
(perplexity ≈ 4.3, well below the random baseline of ~12 for this vocabulary).

### 4.3 End-to-end evaluation

Implementation: [`src/evaluate_nla.py`](src/evaluate_nla.py).

For each of 400 held-out activations: the verbalizer **generates** a
description from the activation alone, the reconstructor predicts an activation
back from that description, and FVE is computed against the standardized true
activation.

## 5. Results

**Headline FVE (held-out, n=400):** [`results/nla_results.json`](results/nla_results.json)

| Condition | FVE |
|---|---:|
| Generated text (verbalizer → reconstructor) | **0.030** |
| Oracle text (teacher summary → reconstructor) | **0.050** |
| Random text (shuffled teacher summary) | **−0.065** |
| Predict mean | 0.000 |

The two things to read out of this table:

1. **Generated > 0 > Random.** The natural-language bottleneck preserves real
   activation-relevant information: the verbalizer's own English is meaningfully
   more reconstructable than an arbitrary other English description, which
   actively hurts reconstruction. This shows the metric is sensitive and the
   pipeline is working as intended.
2. **Generated / Oracle ≈ 60%.** The verbalizer recovers roughly three fifths
   of the way to the oracle ceiling. The remaining gap is exactly the part the
   paper's RL phase is designed to close — a closed-loop reward of
   `-||h - AR(z)||²` pushes the verbalizer to produce reconstruction-optimised
   text rather than merely teacher-style text. That phase is left for future
   work here (see §7).

## 6. Interesting findings

### 6.1 The frozen-encoder baseline fails — but it teaches us something

The first version of the reconstructor used the base model as a *frozen* text
encoder (mean-pooled final hidden state) plus an MLP head trained from scratch.
It produced a near-canonical overfitting signature: **train FVE 0.94, val FVE
−0.92**, with the ridge linear baseline at **−0.19**. Mean-pooled fixed text
features simply did not carry generalisable linear signal to the target
activation space. Switching the reconstructor to **LoRA fine-tuning end-to-end**
fixed it — the transformer is given the capacity to extract task-relevant
features, rather than having them dictated by mean pooling. This contrast is
preserved in the script via `--mode {frozen, lora}` so the result is
reproducible.

### 6.2 Activation variance is broadly distributed

A diagnostic in the reconstructor reports how concentrated the variance of the
raw activations is. For Qwen2.5-0.5B layer 16, the **top dimension holds 2.9%
of total variance and the top 5 dimensions hold 9.9%**. That is, the residual
stream at this layer is not dominated by a few "massive activations" — the
signal is spread across hundreds of effective dimensions. This is what makes
the reconstruction problem hard: a topic-level English summary simply cannot
encode enough bits to specify a near-isotropic 896-dimensional vector
precisely.

### 6.3 Thematic faithfulness, specific confabulation

The qualitative samples ([`results/nla_samples.csv`](results/nla_samples.csv))
independently reproduce the paper's main qualitative finding: the verbalizer
gets the **genre, structure, and theme** right, and **invents the specifics**.

Two of the cleanest cases:

> **Snippet:** *"Wilhelm Busch was born on 15 April 1832, the first of seven
> children to the marriage of Henriette Kleine and Friedrich Wilhelm Busch.
> His six siblings followed shortly after: Fanny (1834), Gustav (1836)…"*
>
> **Teacher summary:** *"Wilhelm Busch, born 1832, siblings… historical
> biography or genealogy."*
>
> **Verbalizer:** *"The text is about the birth details of a person named
> **Smith**. Smith, **June 18, 1905**, May 26, 1937. **This is an obituary.**"*

The verbalizer recovered "biographical text about a person, with dates and
family" — and confabulated the name, dates, and the closely-related sub-genre
(obituary instead of birth-and-family bio).

> **Snippet:** *"In the 1980s a form of AI program called 'expert systems' was
> adopted by corporations around the world…"*
>
> **Teacher summary:** *"adoption and significance of expert systems in the
> 1980s … historical/technical."*
>
> **Verbalizer:** *"The text is discussing the use of **xenon as a tracer in
> archaeology**. … This appears to be part of an academic or scientific
> document."*

Completely wrong topic, but correctly identified "academic/scientific
document". The same pattern shows up in the Chagas-disease sample (correct
"disease + pathogen" thematic, wrong "virus + immune system" specifics) and the
Polish-Jewish-culture sample (correct "historical destruction of a site",
wrong "ancient Egypt").

This is exactly the **"thematic claims more accurate than specific ones"**
behaviour that the paper highlights as one of its most reportable findings on
Claude-scale models. Seeing it cleanly reproduced on a 0.5B model with a
warm-started verbalizer is, I think, the most interesting result here.

## 7. Honest limitations and what's missing

- **No RL phase.** The warm-started verbalizer imitates the teacher, it does
  not yet *discover* explanations from the reconstruction signal. The paper's
  full result requires the RL closed loop with a GRPO objective and a KL
  anchor. I scoped this out due to time and compute, and the ~40% gap between
  Generated and Oracle FVE is the gap this RL would close.
- **Single layer, single token, single model size.** The paper reports layer
  sensitivity (a sycophancy signal only appearing at ~halfway depth). I trained
  one layer and one model. A layer ablation is the most natural extension.
- **Open-model teacher.** I deliberately used `Qwen2.5-3B-Instruct` rather than
  an API teacher, for reproducibility. A frontier-API teacher would likely
  produce sharper warm-start summaries and improve every downstream number.
- **Compute interruptions and platform switch.** I exhausted Colab's free-tier
  GPU quota during the multiple reconstructor attempts and migrated mid-project
  to Kaggle, which gave 2× T4 and unblocked the final runs. The README's
  ~3-hour training pipeline is on Kaggle.
- **FVE is in standardized space.** Activations are z-scored per dimension
  before computing FVE, both to make the optimisation tractable and to avoid a
  few dimensions dominating the metric. This is a conservative choice and is
  documented in the code.

## 8. Failure modes I worked through (and what I learned from each)

I am including these because I think the debugging path is itself a useful part
of the submission — research is mostly diagnosing why something doesn't work.

- **Frozen-encoder + MLP overfit catastrophically** (train 0.94, val −0.92).
  Fix: LoRA end-to-end (§6.1). Lesson: a frozen text representation is not
  guaranteed to be a sensible regression target space.
- **Gradient clipping at norm=1.0 silently froze LoRA learning** for one run
  (val FVE flat at −0.003 for 12 epochs). Because the loss is `sum`-over-dims
  MSE the natural gradient norm is huge; clipping at 1.0 scaled every update
  down by ~1000×. Fix: clip at 100 (effectively off) and the same config
  immediately learned. Lesson: gradient clipping interacts with loss
  normalisation; clip should be set with the gradient *scale* in mind, not by
  default-copying from another setup.
- **Higher LR (5e-4) + MLP head collapsed the reconstructor to "predict the
  mean"** on the first epoch and never escaped (loss stuck at the predict-mean
  baseline). Fix: revert to LR 2e-4 and a single Linear head — same recipe
  that the smaller-LoRA run had used. Lesson: when changing several
  hyperparameters at once, one of them is usually solely responsible; change
  one variable at a time.
- **GPU quota exhaustion on free Colab** forced a migration to Kaggle
  mid-project. This added platform-setup overhead but Kaggle's 2× T4 quota
  unblocked the final training runs.

## 9. Repository layout & how to reproduce

```text
src/
  config.py                       — central config (model, layer, sample counts)
  harvest_activations.py          — Step 1: read frozen model, save activations
  generate_teacher_summaries.py   — Step 2: open-model teacher writes summaries
  models.py                       — shared FVE / pooling / head definitions
  train_reconstructor.py          — Step 3a: English → activation (LoRA + head)
  train_verbalizer.py             — Step 3b: activation → English (LoRA + projection)
  evaluate_nla.py                 — Step 4: end-to-end FVE + qualitative samples
notebooks/
  01_harvest_activations.ipynb    — Colab notebook for step 1
  02_teacher_summaries.ipynb      — Colab notebook for step 2
  03_train_reconstructor.ipynb    — Colab notebook for step 3a
results/
  nla_results.json                — headline FVE numbers
  nla_samples.csv                 — qualitative samples used in §6.3
```

End-to-end reproduce (on Kaggle, 2× T4, ~3 hours total):

```bash
pip install -r requirements.txt
python src/harvest_activations.py           --data_dir data --num_samples 8000
python src/generate_teacher_summaries.py    --data_dir data --batch_size 32 --limit 4000
python src/train_reconstructor.py           --data_dir data --mode lora --epochs 12
python src/train_verbalizer.py              --data_dir data --epochs 3
python src/evaluate_nla.py                  --data_dir data
```

Everything else (`--lr`, `--lora_r`, `--max_len`, etc.) is in CLI flags with
documented defaults.

## 10. What I would do next given more time

1. **Implement the RL phase.** Even a few hundred GRPO steps on the warm-started
   verbalizer would close part of the Generated–Oracle gap and is the most
   direct improvement.
2. **Layer ablation.** Train reconstructors at layers 8, 16, 22 and compare
   FVE — the paper's most-cited layer-sensitivity result.
3. **Faithfulness / paraphrase test.** Paraphrase the generated English and
   check whether FVE drops. If it doesn't, the bottleneck is carrying genuine
   semantic content (and not steganography).
4. **Move the target to a code model** (e.g. `Qwen2.5-Coder-0.5B`) and ask
   whether the verbalizer flags buggy vs correct lines. This aligns directly
   with the ASSERT lab's AI4Code direction.
