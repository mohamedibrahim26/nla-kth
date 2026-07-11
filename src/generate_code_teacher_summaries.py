"""
generate_code_teacher_summaries.py — Generate teacher descriptions for each
code activation record using a stronger frozen code model.

For each record in code_activations.pkl the teacher is shown the code snippet
and asked to describe in plain English what a code model is "focusing on" at
the end of that snippet: the algorithm, any logic issues, the data flow, what
it would predict next.

Teacher: Qwen2.5-Coder-3B-Instruct (4-bit, same pattern as the original NLA pipeline).
Input:   data/code_activations.pkl
Output:  data/code_teacher_summaries.pkl   — list of dicts:
             {"record_idx": int, "func_id": int, "variant": str, "summary": str}

Usage
-----
    python src/generate_code_teacher_summaries.py \
        --data_dir data \
        --batch_size 4 \
        --limit 2000          # how many records to summarise (None = all)
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TEACHER_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"

# Prompt asking the teacher to describe the model's internal "focus"
SYSTEM_PROMPT = (
    "You are an expert at reasoning about what a language model trained on code "
    "is focused on as it processes Python source code."
)

USER_TEMPLATE = """\
A code model is processing the following Python snippet and has just reached the end of it.

```python
{code}
```

In 60–100 words, describe what the model appears to be focusing on at this point: \
the algorithm or pattern being implemented, the variables and their relationships, \
the control flow, any edge cases or potential logic issues, and what the model would \
most likely predict comes next. Be specific and technical."""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="data")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--limit",      type=int, default=None,
                   help="Max records to summarise (None = all)")
    p.add_argument("--overwrite",  action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=180)
    return p.parse_args()


def build_prompt(tokenizer, code: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_TEMPLATE.format(code=code[:1500])},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def main():
    args = parse_args()
    in_path  = os.path.join(args.data_dir, "code_activations.pkl")
    out_path = os.path.join(args.data_dir, "code_teacher_summaries.pkl")

    if not os.path.exists(in_path):
        sys.exit(f"[error] {in_path} not found. Run harvest_code_activations.py first.")

    if os.path.exists(out_path) and not args.overwrite:
        print(f"[skip] {out_path} already exists. Use --overwrite to regenerate.")
        return

    with open(in_path, "rb") as f:
        records = pickle.load(f)

    if args.limit:
        records = records[: args.limit]

    print(f"Summarising {len(records)} records with {TEACHER_MODEL} …")

    # ── load teacher in 4-bit ─────────────────────────────────────────────────
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        quantization_config=bnb_cfg if device == "cuda" else None,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float32 if device == "cpu" else None,
    )
    if device == "cpu":
        teacher = teacher.to(device)
    teacher.eval()

    summaries = []

    # Process in batches
    for start in tqdm(range(0, len(records), args.batch_size), desc="Summarising"):
        batch = records[start: start + args.batch_size]
        prompts = [build_prompt(tokenizer, r["code"]) for r in batch]

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=768,
        ).to(device)

        with torch.no_grad():
            out = teacher.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_lens = enc["input_ids"].shape[1]
        for i, (rec, gen_ids) in enumerate(zip(batch, out)):
            gen_text = tokenizer.decode(
                gen_ids[prompt_lens:], skip_special_tokens=True
            ).strip()
            summaries.append({
                "record_idx": start + i,
                "func_id":    rec["func_id"],
                "variant":    rec["variant"],
                "bug_type":   rec.get("bug_type"),
                "summary":    gen_text,
            })

    with open(out_path, "wb") as f:
        pickle.dump(summaries, f)

    print(f"\nGenerated {len(summaries)} summaries → {out_path}")

    # Quick sanity: print one correct and one buggy example side-by-side
    correct_ex = next((s for s in summaries if s["variant"] == "correct"), None)
    buggy_ex   = next((s for s in summaries if s["variant"] == "buggy"),   None)
    if correct_ex and buggy_ex and correct_ex["func_id"] == buggy_ex["func_id"]:
        print("\n── Sample pair ──────────────────────────────────────")
        print(f"[correct] {correct_ex['summary'][:300]}")
        print(f"[buggy/{buggy_ex['bug_type']}] {buggy_ex['summary'][:300]}")


if __name__ == "__main__":
    main()
