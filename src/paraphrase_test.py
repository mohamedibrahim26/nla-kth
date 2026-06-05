"""
Faithfulness / paraphrase test.

The idea
--------
If our natural-language bottleneck is genuinely carrying SEMANTIC content (and not
some hidden steganographic encoding in the choice of tokens), then paraphrasing
the verbalizer's text should preserve the reconstruction quality. Same meaning,
different surface words -> the reconstructor should still recover roughly the
same activation, because what got encoded was the meaning, not the surface form.

If, instead, the bottleneck is steganographic (the AV has learned to abuse a
narrow vocabulary to fool the AR, like the "feet feet feet" failure mode in some
small-scale reimplementations), paraphrasing will destroy that hidden encoding
and FVE will collapse.

So we compare four conditions for the SAME 400 val activations:

  1. ORIGINAL generated text (the headline number from evaluate_nla.py)
  2. PARAPHRASED version of that generated text (semantic transform)
  3. SENTENCE-SHUFFLED version (re-orders sentences inside the text)
  4. WORD-SHUFFLED version (destroys grammar but keeps tokens) -> hard negative

A small drop from 1 -> 2 means the bottleneck is semantic. A big collapse from
1 -> 4 confirms the metric is sensitive to structure.

Inputs : <data_dir>/activations.npy, <data_dir>/summaries.jsonl, split.json,
         reconstructor.pt + reconstructor_lora/, verbalizer.pt + verbalizer_lora/
Outputs: <out>/paraphrase_results.json
         <out>/paraphrase_samples.csv   (original/paraphrase/shuffled side by side)
"""

import argparse
import csv
import json
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import CONFIG
from models import compute_fve


WRAP_RECON = "A language model's internal state is described as follows: {s}"
PARAPHRASE_PROMPT = (
    "Rewrite the following description in completely different words. Keep the "
    "meaning, the topic, and the structure (Main Topic / Key Entities / Genre) "
    "the same, but change the wording as much as possible. Do not add any "
    "preamble like 'Here is'; just write the rewritten description.\n\n"
    "Description:\n\"\"\"\n{text}\n\"\"\"\n\nRewritten description:"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--model_name", default=CONFIG.model_name)
    p.add_argument("--teacher_model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--max_new_tokens_gen", type=int, default=120)
    p.add_argument("--max_new_tokens_para", type=int, default=200)
    p.add_argument("--batch_size_eval", type=int, default=8)
    p.add_argument("--max_len_recon", type=int, default=200)
    p.add_argument("--n_qualitative", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def last_token(hidden, attn):
    idx = attn.sum(1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]


def word_shuffle(text, rng):
    words = text.split()
    rng.shuffle(words)
    return " ".join(words)


def sentence_shuffle(text, rng):
    sents = re.split(r"(?<=[\.\?\!])\s+", text)
    sents = [s for s in sents if s.strip()]
    rng.shuffle(sents)
    return " ".join(sents)


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = args.output_dir or args.data_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"Device: {device}")

    # --- Load data + split ---
    activations = np.load(os.path.join(args.data_dir, "activations.npy")).astype(np.float32)
    rows = [json.loads(l) for l in open(os.path.join(args.data_dir, "summaries.jsonl"), encoding="utf-8")]
    idx_to_summary = {r["idx"]: r["summary"] for r in rows}
    split = json.load(open(os.path.join(out_dir, "split.json")))
    val_idx = [i for i in split["val_idx"] if i in idx_to_summary]
    print(f"Val set: {len(val_idx)} samples")

    targets_raw = torch.tensor(activations[val_idx], dtype=torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load RECONSTRUCTOR (we'll keep this loaded for all FVE computations) ---
    print("Loading reconstructor...")
    base_r = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32).to(device)
    recon_model = PeftModel.from_pretrained(base_r, os.path.join(out_dir, "reconstructor_lora")).to(device)
    recon_model.eval()
    recon_pt = torch.load(os.path.join(out_dir, "reconstructor.pt"), map_location=device)
    d = recon_pt["d"]
    head = nn.Linear(d, d).to(device)
    head.load_state_dict(recon_pt["head"])
    head.eval()
    tmean = recon_pt["tmean"].to(device); tstd = recon_pt["tstd"].to(device)
    Y_std = ((targets_raw.to(device) - tmean) / tstd).cpu()

    @torch.no_grad()
    def recon_predict(texts):
        preds = []
        for i in range(0, len(texts), args.batch_size_eval):
            batch = [WRAP_RECON.format(s=t) for t in texts[i:i + args.batch_size_eval]]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=args.max_len_recon).to(device)
            out = recon_model(**enc, output_hidden_states=True)
            h = last_token(out.hidden_states[-1], enc["attention_mask"])
            preds.append(head(h.float()).cpu())
        return torch.cat(preds, dim=0)

    # --- Generate verbalizer text for val set (or load cached) ---
    gen_cache = os.path.join(out_dir, "val_generated.jsonl")
    if os.path.exists(gen_cache):
        print(f"Loading cached generated texts from {gen_cache}")
        generated = [json.loads(l)["text"] for l in open(gen_cache, encoding="utf-8")]
        assert len(generated) == len(val_idx), "Cache size mismatch; delete cache and rerun."
    else:
        print("Loading verbalizer to regenerate val texts...")
        # Free recon, load verbalizer (memory)
        del recon_model, base_r
        torch.cuda.empty_cache()
        vp = torch.load(os.path.join(out_dir, "verbalizer.pt"), map_location=device)
        base_v = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32).to(device)
        verb_model = PeftModel.from_pretrained(base_v, os.path.join(out_dir, "verbalizer_lora")).to(device)
        verb_model.eval()
        projection = nn.Linear(vp["act_dim"], vp["embed_dim"]).to(device)
        projection.load_state_dict(vp["projection"])
        projection.eval()
        embed_layer = verb_model.get_input_embeddings()
        suffix_ids = tokenizer(vp["prompt_after"], return_tensors="pt",
                               add_special_tokens=False).input_ids[0].to(device)

        targets_norm = targets_raw / (targets_raw.norm(dim=1, keepdim=True) + 1e-8)
        generated = []
        with torch.no_grad():
            for i, act in enumerate(targets_norm):
                act_emb = projection(act.to(device).unsqueeze(0))
                suffix_emb = embed_layer(suffix_ids)
                seq_embeds = torch.cat([act_emb, suffix_emb], dim=0).unsqueeze(0)
                attn = torch.ones(seq_embeds.shape[:2], dtype=torch.long, device=device)
                out = verb_model.generate(
                    inputs_embeds=seq_embeds, attention_mask=attn,
                    max_new_tokens=args.max_new_tokens_gen, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                text = tokenizer.decode(out[0], skip_special_tokens=True).strip()
                generated.append(text)
                if (i + 1) % 50 == 0:
                    print(f"  generated {i + 1}/{len(targets_norm)}")
        # Cache
        with open(gen_cache, "w", encoding="utf-8") as f:
            for i, t in enumerate(generated):
                f.write(json.dumps({"val_pos": i, "text": t}) + "\n")
        # Reload reconstructor for scoring
        del verb_model, base_v, projection, embed_layer
        torch.cuda.empty_cache()
        base_r = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32).to(device)
        recon_model = PeftModel.from_pretrained(base_r, os.path.join(out_dir, "reconstructor_lora")).to(device)
        recon_model.eval()

    # --- FVE on ORIGINAL generated text (headline; reproduces evaluate_nla) ---
    print("Computing FVE on original generated text...")
    fve_orig = compute_fve(recon_predict(generated), Y_std)
    print(f"  FVE original generated: {fve_orig:.4f}")

    # --- FVE on SENTENCE-SHUFFLED generated text ---
    print("Computing FVE on sentence-shuffled text...")
    rng_s = random.Random(args.seed)
    sent_shuf = [sentence_shuffle(t, rng_s) for t in generated]
    fve_sent = compute_fve(recon_predict(sent_shuf), Y_std)
    print(f"  FVE sentence-shuffled:  {fve_sent:.4f}")

    # --- FVE on WORD-SHUFFLED generated text (hard negative) ---
    print("Computing FVE on word-shuffled text...")
    rng_w = random.Random(args.seed + 1)
    word_shuf = [word_shuffle(t, rng_w) for t in generated]
    fve_word = compute_fve(recon_predict(word_shuf), Y_std)
    print(f"  FVE word-shuffled:      {fve_word:.4f}")

    # --- Now the big one: PARAPHRASE via teacher model ---
    para_cache = os.path.join(out_dir, "val_paraphrased.jsonl")
    if os.path.exists(para_cache):
        print(f"Loading cached paraphrases from {para_cache}")
        paraphrased = [json.loads(l)["text"] for l in open(para_cache, encoding="utf-8")]
        assert len(paraphrased) == len(generated), "Paraphrase cache size mismatch."
    else:
        print(f"Loading teacher {args.teacher_model} for paraphrasing...")
        del recon_model, base_r
        torch.cuda.empty_cache()
        teacher_tok = AutoTokenizer.from_pretrained(args.teacher_model)
        if teacher_tok.pad_token is None:
            teacher_tok.pad_token = teacher_tok.eos_token
        teacher_tok.padding_side = "left"
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_model, device_map=device,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"),
        )
        teacher.eval()

        paraphrased = []
        bs = 8
        with torch.no_grad():
            for i in range(0, len(generated), bs):
                batch_texts = generated[i:i + bs]
                prompts = []
                for t in batch_texts:
                    msgs = [{"role": "user", "content": PARAPHRASE_PROMPT.format(text=t)}]
                    prompts.append(teacher_tok.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True))
                enc = teacher_tok(prompts, return_tensors="pt", padding=True,
                                  truncation=True, max_length=768).to(device)
                gen = teacher.generate(
                    **enc, max_new_tokens=args.max_new_tokens_para, do_sample=False,
                    pad_token_id=teacher_tok.pad_token_id,
                )
                new_tokens = gen[:, enc["input_ids"].shape[1]:]
                texts = teacher_tok.batch_decode(new_tokens, skip_special_tokens=True)
                paraphrased.extend([t.strip() for t in texts])
                if (i // bs) % 5 == 0:
                    print(f"  paraphrased {len(paraphrased)}/{len(generated)}")
        with open(para_cache, "w", encoding="utf-8") as f:
            for i, t in enumerate(paraphrased):
                f.write(json.dumps({"val_pos": i, "text": t}) + "\n")

        del teacher, teacher_tok
        torch.cuda.empty_cache()
        base_r = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32).to(device)
        recon_model = PeftModel.from_pretrained(base_r, os.path.join(out_dir, "reconstructor_lora")).to(device)
        recon_model.eval()

    # --- FVE on PARAPHRASED text (the key result) ---
    print("Computing FVE on paraphrased text...")
    fve_para = compute_fve(recon_predict(paraphrased), Y_std)
    print(f"  FVE paraphrased:        {fve_para:.4f}")

    results = {
        "n_val": len(val_idx),
        "fve_original_generated": fve_orig,
        "fve_paraphrased": fve_para,
        "fve_sentence_shuffled": fve_sent,
        "fve_word_shuffled": fve_word,
        "interpretation": {
            "fve_orig_vs_paraphrase_ratio": float(fve_para / fve_orig) if abs(fve_orig) > 1e-9 else None,
            "small_drop_from_paraphrase_means": "bottleneck carries SEMANTIC content (not steganography)",
            "big_drop_from_word_shuffle_means": "metric is sensitive to text structure (sanity check)",
        },
    }
    with open(os.path.join(out_dir, "paraphrase_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults:", json.dumps(results, indent=2))

    n = min(args.n_qualitative, len(generated))
    with open(os.path.join(out_dir, "paraphrase_samples.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["val_pos", "original_generated", "paraphrased",
                    "sentence_shuffled", "word_shuffled"])
        for i in range(n):
            w.writerow([i, generated[i], paraphrased[i], sent_shuf[i], word_shuf[i]])
    print(f"Saved {n} qualitative samples to {out_dir}/paraphrase_samples.csv")


if __name__ == "__main__":
    main()
