# Natural Language Autoencoders on a Small Open Model

A small-scale reimplementation of the natural language autoencoder idea from Anthropic's *Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations* ([transformer-circuits.pub/2026/nla](https://transformer-circuits.pub/2026/nla/index.html)). Built for the KTH ASSERT-lab PhD recruitment task with Prof. Monperrus.

> **TL;DR.** I trained the two paired components from the paper, an activation verbalizer and an activation reconstructor, on `Qwen2.5-0.5B-Instruct`. The whole experiment ran on a free Colab T4 first, and then on Kaggle's 2x T4 after I hit Colab's daily GPU limit. On a 400-sample held-out set, the reconstructor recovers about **3.0% of the activation variance** when fed the verbalizer's own generated English, **5.0%** when fed the teacher's oracle summary, and **-6.5%** when fed random English as a negative control. The numbers are small because the model is small and I only did the warm-start phase, but the shape of the results matches the paper. The most interesting finding is qualitative: the verbalizer keeps the theme of each snippet right while confabulating the specific entities. That is exactly what the paper reports for Claude-scale models.

---

## 1. What this is and why I cared about it

The Anthropic paper proposes an autoencoder where the bottleneck is plain English. A verbalizer turns one activation vector `h` into a short natural-language description `z`. A reconstructor maps that description back to a predicted activation. If the round-trip preserves the original activation well (high Fraction of Variance Explained, FVE), then the English `z` must really capture what the model was thinking at that point. That gives you human-readable explanations of internal model state, with no human labels involved.

I reimplemented this pipeline at the smallest scale that still made sense. One small recent open model, a single layer, a single token position, a few thousand examples, and only the warm-start phase. No RL.

The recruitment task is explicit that they are not looking for Claude-scale numbers. What they want to see is a faithful reimplementation of the actual generative bottleneck, an honest measurement with proper baselines, and a clear explanation of why the small-scale numbers look the way they do. That is what this README tries to show.

## 2. Target model and how I harvested activations

**Target model.** [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct). It is recent (late 2024), capable for its size, has 24 layers and a hidden dimension of 896, and fits comfortably on a single T4.

**Where I read activations.** Residual stream output of **layer 16**, which is about two-thirds of the way through the model. The paper recommends reading at roughly that depth. I take the activation of the **last token** of randomly truncated text snippets from `wikitext-2-raw-v1`. Random truncation matters here. If you always grab sentence ends, you only ever see "thinking points" at one kind of position. Random cuts give you a more diverse sample.

**Scale.** 8,000 (snippet, activation) pairs. The L2 norms of the resulting activations are tight (mean 21.1, std 1.3), which says the distribution looks healthy. Code lives in [`src/harvest_activations.py`](src/harvest_activations.py).

## 3. Teacher summaries (the warm-start targets)

The verbalizer needs a sensible starting point. A randomly initialised verbalizer would just emit gibberish, and the reconstructor cannot learn anything from gibberish. The paper handles this by having a stronger frozen model write example descriptions, and then training the verbalizer to imitate them.

For the teacher I picked `Qwen2.5-3B-Instruct` in 4-bit, running inside the same Colab notebook. This was a deliberate choice. It costs nothing, needs no API keys, and means the whole pipeline can be rerun end to end by anyone (including the reviewer) without paying for anything. The trade-off is that the summaries are a bit lower quality than a frontier API would produce. I am honest about that in section 7.

The prompt asks the teacher to write an 80 to 120 word description of what a language model is focusing on at the truncation point: topic, key entities, kind of text, what it might predict next. See [`src/generate_teacher_summaries.py`](src/generate_teacher_summaries.py). I summarised 4,000 of the 8,000 snippets. That gave me a comfortable warm-start split and left the remaining activations free for future RL or extra evaluation.

## 4. The autoencoder

Both components are LoRA-adapted copies of the same base model. The base weights are frozen the whole time, and only the small LoRA adapters get trained. This is a faithful simplification of the paper's setup, which uses full finetuned copies of the base. LoRA gives me most of the capacity at a fraction of the compute. The paper also notes that finetunes of the base model transfer well to this role, which made me confident in the choice.

### 4.1 Reconstructor (English to activation)

Code: [`src/train_reconstructor.py`](src/train_reconstructor.py).

The reconstructor reads the description, takes the last-token hidden state of the final layer, and passes it through a learned linear head to predict the standardised activation. LoRA with rank 32 on all attention and MLP projections gives the transformer room to adapt to the new objective. Targets are z-scored per dimension using the training-set statistics. Without that standardisation the optimisation just stalls (more on this in section 6).

Trained for 12 epochs with batch size 16 and learning rate 2e-4 on Kaggle's 2x T4. The train FVE climbs cleanly from 0.02 to 0.60 over the epochs. The best validation FVE on the oracle (teacher) text is **0.050** at epoch 2. After that it overfits and val FVE comes back down. The saved checkpoint is the best-val one.

### 4.2 Verbalizer (activation to English)

Code: [`src/train_verbalizer.py`](src/train_verbalizer.py).

The verbalizer is also a LoRA copy of the base model. The trick is how the activation gets fed in. A small learned linear projection maps the unit-normalised activation vector into the model's token-embedding space. That projected vector goes at the very front of the input embeddings, like a "soft prompt". After it comes a fixed instruction telling the model to describe what a language model is focusing on at this point. The model then generates the description token by token.

Training is standard next-token cross-entropy against the teacher summaries. This is the warm-start phase only, no RL.

Three epochs, batch size 4, learning rate 2e-4. Best validation cross-entropy was **1.46** at epoch 1, which is a perplexity of about 4.3. The random baseline for this vocabulary is about 12, so the verbalizer is actually predicting teacher-style tokens, not noise.

### 4.3 End-to-end evaluation

Code: [`src/evaluate_nla.py`](src/evaluate_nla.py).

For each of the 400 held-out activations: the verbalizer generates a description from the activation alone, the reconstructor predicts an activation back from that description, and FVE is computed against the standardised true activation.

## 5. Results

The headline FVE numbers on the held-out set (n=400) are saved in [`results/nla_results.json`](results/nla_results.json):

| Condition | FVE |
|---|---:|
| Generated text (verbalizer → reconstructor) | **0.030** |
| Oracle text (teacher summary → reconstructor) | **0.050** |
| Random text (shuffled teacher summary) | **−0.065** |
| Predict mean | 0.000 |

Two things to read out of this table.

First, Generated is above zero and Random is well below zero. That means the natural-language bottleneck preserves real activation-relevant information. The verbalizer's own English is meaningfully more reconstructable than an arbitrary unrelated English description, which actively hurts reconstruction. The metric is sensitive, and the pipeline is doing what it is supposed to do.

Second, the Generated number is about 60% of the Oracle number. The verbalizer recovers most of the way toward the oracle ceiling, but not all of it. The remaining gap is the part the paper's RL phase is designed to close. That closed loop, with a reward of `-||h - AR(z)||^2` and a KL anchor, pushes the verbalizer to produce text that is reconstruction-friendly rather than just teacher-style. That phase is left as future work here (section 7).

## 6. Interesting findings

### 6.1 The frozen-encoder baseline fails, and that tells us something

My first reconstructor used the base model as a frozen text encoder (mean-pooled final hidden state) plus an MLP head trained from scratch. It produced a clean overfitting signature: train FVE 0.94, val FVE -0.92. A closed-form ridge linear baseline on the same features came out at -0.19, which is worse than predicting the mean. Mean-pooled frozen text features simply did not carry generalisable linear signal to the target activation space.

Switching the reconstructor to LoRA fine-tuning end-to-end fixed it. The transformer gets the capacity to extract task-relevant features instead of being stuck with whatever mean pooling produced. I kept the original frozen mode reachable through `--mode frozen` so this contrast is reproducible.

### 6.2 The activation variance is broadly distributed

A diagnostic in the reconstructor prints how concentrated the variance of the raw activations is. For Qwen2.5-0.5B at layer 16, the top dimension holds 2.9% of total variance and the top 5 dimensions hold 9.9%. So the residual stream here is not dominated by a few huge "massive activations". The signal is spread across hundreds of effective dimensions.

This is part of why the reconstruction problem is hard at small scale. A topic-level English summary just cannot encode enough bits to specify a nearly isotropic 896-dimensional vector with precision.

### 6.3 Thematic faithfulness, specific confabulation

The qualitative samples ([`results/nla_samples.csv`](results/nla_samples.csv)) independently reproduce the paper's most interesting qualitative finding. The verbalizer gets the genre, the structure, and the theme right, and then makes up the specifics.

Two of the cleanest examples:

> **Snippet:** *"Wilhelm Busch was born on 15 April 1832, the first of seven children to the marriage of Henriette Kleine and Friedrich Wilhelm Busch. His six siblings followed shortly after: Fanny (1834), Gustav (1836)..."*
>
> **Teacher summary:** *"Wilhelm Busch, born 1832, siblings... historical biography or genealogy."*
>
> **Verbalizer:** *"The text is about the birth details of a person named **Smith**. Smith, **June 18, 1905**, May 26, 1937. **This is an obituary.**"*

The verbalizer correctly recovered "biographical text about a person, with dates and family". It then confabulated the name, the dates, and a close but wrong sub-genre (obituary instead of birth-and-family bio).

> **Snippet:** *"In the 1980s a form of AI program called 'expert systems' was adopted by corporations around the world..."*
>
> **Teacher summary:** *"adoption and significance of expert systems in the 1980s ... historical/technical."*
>
> **Verbalizer:** *"The text is discussing the use of **xenon as a tracer in archaeology**. ... This appears to be part of an academic or scientific document."*

The topic is completely wrong, but the genre call ("academic or scientific document") is right. The same pattern shows up in the Chagas-disease sample (right "disease and pathogen" thematic call, wrong "virus and immune system" specifics) and the Polish-Jewish-culture sample (right "historical destruction of a site", wrong "ancient Egypt").

The paper calls this out as one of its main qualitative findings on Claude-scale models: "thematic claims are more accurate than specific ones." Seeing the same behaviour show up cleanly on a 0.5B model with only a warm-started verbalizer is, in my opinion, the most interesting thing in this whole project.

## 7. Honest limitations and what is missing

- **No RL phase.** The warm-started verbalizer imitates the teacher. It does not yet discover explanations from the reconstruction signal. The paper's full result needs the RL closed loop with GRPO and a KL anchor. I scoped this out because of time and compute, and the roughly 40% gap between Generated and Oracle FVE is the gap RL would close.
- **One layer, one token position, one model size.** The paper reports layer sensitivity, like a sycophancy signal that only appears at about halfway depth. I only trained one layer on one model. A layer-by-layer ablation is the most natural extension.
- **Open-model teacher.** I used `Qwen2.5-3B-Instruct` instead of an API teacher, on purpose, for reproducibility. A frontier-API teacher would probably give sharper warm-start summaries and improve every downstream number a bit.
- **Compute interruptions and platform switch.** I burned through Colab's free-tier GPU quota while debugging the reconstructor and had to migrate to Kaggle, which gave 2x T4 and unblocked the final runs. The 3-hour pipeline described above runs on Kaggle.
- **FVE is in standardised space.** Activations are z-scored per dimension before computing FVE. This is partly to make optimisation tractable and partly to keep a few large-variance dimensions from dominating the metric. The choice is documented in the script.

## 8. Failure modes I worked through and what I learned

I am including this section because the debugging path was itself a useful part of the project. ML work is mostly diagnosing why something is not working.

- **Frozen-encoder + MLP overfit catastrophically** (train 0.94, val -0.92). Fix: LoRA end-to-end, section 6.1. Lesson: a frozen text representation is not automatically a sensible regression target space.
- **Gradient clipping at norm 1.0 silently froze LoRA learning** for one run (val FVE stuck at -0.003 for 12 epochs). The loss is sum-over-dimensions MSE, so the natural gradient norm is huge. Clipping at 1.0 scaled every update down by something like 1000 times. Fix: clip at 100 (effectively off), and the same config immediately started learning. Lesson: gradient clipping interacts with loss normalisation, and the clip value should match the gradient scale you actually expect.
- **Higher LR (5e-4) plus an MLP head collapsed the reconstructor to "predict the mean"** on the very first epoch, and it never escaped. Loss stayed pinned at the predict-mean baseline. Fix: roll LR back to 2e-4 and go back to a single Linear head, the recipe that had worked for the smaller-LoRA run. Lesson: when you change several hyperparameters at the same time, one of them is usually the only one that mattered. Change one variable at a time.
- **GPU quota exhaustion on free Colab** forced a switch to Kaggle in the middle of the project. It added some setup overhead, but Kaggle's 2x T4 quota is what got the final training runs done.

## 9. Repository layout and how to reproduce

```text
src/
  config.py                       - central config (model, layer, sample counts)
  harvest_activations.py          - Step 1: read frozen model, save activations
  generate_teacher_summaries.py   - Step 2: open-model teacher writes summaries
  models.py                       - shared FVE / pooling / head definitions
  train_reconstructor.py          - Step 3a: English to activation (LoRA + head)
  train_verbalizer.py             - Step 3b: activation to English (LoRA + projection)
  evaluate_nla.py                 - Step 4: end-to-end FVE + qualitative samples
notebooks/
  01_harvest_activations.ipynb    - Colab notebook for step 1
  02_teacher_summaries.ipynb      - Colab notebook for step 2
  03_train_reconstructor.ipynb    - Colab notebook for step 3a
results/
  nla_results.json                - headline FVE numbers
  nla_samples.csv                 - qualitative samples used in section 6.3
```

End-to-end reproduction (on Kaggle, 2x T4, about 3 hours total):

```bash
pip install -r requirements.txt
python src/harvest_activations.py           --data_dir data --num_samples 8000
python src/generate_teacher_summaries.py    --data_dir data --batch_size 32 --limit 4000
python src/train_reconstructor.py           --data_dir data --mode lora --epochs 12
python src/train_verbalizer.py              --data_dir data --epochs 3
python src/evaluate_nla.py                  --data_dir data
```

Everything else (`--lr`, `--lora_r`, `--max_len`, and so on) is exposed as a CLI flag with a sensible default.

## 10. What I would do next with more time

1. Implement the RL phase. Even a few hundred GRPO steps on the warm-started verbalizer would close part of the Generated to Oracle gap. This is the most direct improvement.
2. Run a layer ablation. Train reconstructors at layers 8, 16, and 22 and compare FVE. This is the paper's most cited layer-sensitivity result.
3. Run a paraphrase / faithfulness test. Paraphrase the generated English and check whether FVE drops. If it does not drop much, the bottleneck is carrying real semantic content and not steganography.
4. Switch the target to a code model like `Qwen2.5-Coder-0.5B` and ask whether the verbalizer flags buggy versus correct lines. This direction connects directly to the ASSERT lab's AI4Code focus.
