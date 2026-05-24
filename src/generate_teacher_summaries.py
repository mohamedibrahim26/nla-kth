"""
Step 2 of the NLA pipeline: generate "teacher" summaries for the warm-start.

Why this exists
---------------
Our verbalizer (activation -> English) and reconstructor (English -> activation)
both need a sensible starting point. A randomly initialised verbalizer would emit
gibberish, and the reconstructor could never learn from gibberish.

So we use a *stronger* open model (default: Qwen2.5-3B-Instruct) as a TEACHER. The
teacher reads each text snippet that the target model read, and writes ~100 words
describing what a language model is most likely "thinking about" at that point:
the topic, the entities, the kind of text, and what it might predict next.

Crucially, the teacher only sees the TEXT, never the activation. Later:
  - the verbalizer is trained to reproduce these summaries from the activation alone,
  - the reconstructor is trained to map these summaries back to the activation.

Design choices for a free Colab T4:
  - 4-bit loading (bitsandbytes) so a 3B model fits comfortably.
  - Batched generation for speed.
  - Resumable: already-completed indices are skipped, and results are flushed to
    disk after every batch, so a disconnect never loses progress.

Input  : <data_dir>/metadata.jsonl   (from harvest_activations.py)
Output : <data_dir>/summaries.jsonl  (one line per snippet: {idx, summary})

Run:
    python src/generate_teacher_summaries.py --data_dir data --batch_size 16
"""

import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROMPT = (
    "Below is a text excerpt that a language model has just finished reading. "
    "The excerpt may stop mid-sentence. In about 80-120 words, describe what the "
    "model is most likely focusing on internally at this exact point. Cover: the "
    "main topic, the key entities or concepts present, the kind/genre of text, and "
    "what the model might predict next. Be specific and factual about the content. "
    "Do not add any preamble like 'The model is thinking'; just write the description.\n\n"
    "Text excerpt:\n\"\"\"\n{snippet}\n\"\"\""
)


def parse_args():
    p = argparse.ArgumentParser(description="Generate teacher summaries for warm-start.")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--teacher_model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=150)
    p.add_argument("--limit", type=int, default=0, help="0 = summarise all snippets")
    p.add_argument("--load_in_4bit", type=int, default=1, help="1=4-bit, 0=fp16")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    meta_path = os.path.join(args.data_dir, "metadata.jsonl")
    out_path = os.path.join(args.data_dir, "summaries.jsonl")

    # --- Load the snippets ---
    snippets = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            snippets.append((row["idx"], row["text"]))
    if args.limit > 0:
        snippets = snippets[: args.limit]
    print(f"{len(snippets)} snippets to consider.")

    # --- Resume: skip indices we've already summarised ---
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["idx"])
                except Exception:
                    pass
    todo = [(i, t) for (i, t) in snippets if i not in done]
    print(f"{len(done)} already done, {len(todo)} remaining.")
    if not todo:
        print("Nothing to do. summaries.jsonl is complete.")
        return

    # --- Load the teacher model ---
    print(f"Loading teacher {args.teacher_model} (4bit={bool(args.load_in_4bit)}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for correct batched generation

    model_kwargs = {}
    if args.load_in_4bit and device == "cuda":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        model_kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, device_map=device, **model_kwargs
    )
    model.eval()

    # --- Generate in batches, flushing after each batch ---
    out_f = open(out_path, "a", encoding="utf-8")
    pbar = tqdm(total=len(todo), desc="Summarising")
    for start in range(0, len(todo), args.batch_size):
        batch = todo[start : start + args.batch_size]
        prompts = []
        for _, text in batch:
            messages = [{"role": "user", "content": PROMPT.format(snippet=text)}]
            prompts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))

        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=768).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=args.max_new_tokens,
                do_sample=False,  # greedy -> reproducible
                pad_token_id=tokenizer.pad_token_id,
            )
        # With left padding, the prompt length is identical across the batch.
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        for (idx, _), summary in zip(batch, texts):
            out_f.write(json.dumps({"idx": idx, "summary": summary.strip()}) + "\n")
        out_f.flush()
        os.fsync(out_f.fileno())  # force write to Drive so a disconnect is safe
        pbar.update(len(batch))
    pbar.close()
    out_f.close()
    print(f"\nDone. Summaries saved to {out_path}")


if __name__ == "__main__":
    main()
