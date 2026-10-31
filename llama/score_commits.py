#!/usr/bin/env python
"""Score prepared commits with a sequence classification checkpoint on a GPU.

Reads the JSON lines written by ``data_extraction/bugbug/build_scoring_set.py``
(``node`` and ``prompt``; rows without a prompt are commits bugbug ignores and
are skipped) and writes one line per scored commit with ``prob`` (both class
probabilities), ``class`` and ``risk_score`` (the positive class probability),
the fields the pipeline's commit risk records carry. Resumable: nodes already
in the output are skipped, so a job that hits its time limit is continued by
the next one.

The checkpoint is either a full Hugging Face sequence classification
directory (CodeBERT, ModernBERT, a merged LoRA model) or a LoRA adapter
directory, in which case ``--base_model_path`` names the base model and the
adapter is merged in memory. Prompts are sorted by length so a batch pads
little; a batch that runs out of GPU memory is retried one prompt at a time.

    python llama/score_commits.py \\
        --model_path llama/training/run_.../model \\
        --input datasets/mozilla_perf_v2/scoring_2026_8k.jsonl \\
        --output datasets/mozilla_perf_v2/scores_2026_mbert.jsonl --batch_size 8
"""

import argparse
import json
import os
import time

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


def load_model(model_path: str, base_model_path: str | None, dtype: torch.dtype, quant: bool):
    is_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    weights_path = base_model_path if is_adapter else model_path
    if is_adapter and not base_model_path:
        raise SystemExit(f"{model_path} is a LoRA adapter; pass --base_model_path")

    tokenizer = AutoTokenizer.from_pretrained(model_path if os.path.exists(os.path.join(model_path, "tokenizer_config.json")) else weights_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"torch_dtype": dtype}
    if quant:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4"
        )
        kwargs["device_map"] = "auto"
    config = AutoConfig.from_pretrained(weights_path)
    config.num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(weights_path, config=config, **kwargs)
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    if is_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, model_path).merge_and_unload()
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    if not quant:
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", required=True, help="Sequence classification checkpoint, or a LoRA adapter directory")
    parser.add_argument("--base_model_path", default=None, help="Base model when --model_path is a LoRA adapter")
    parser.add_argument("--input", required=True, help="JSON lines from build_scoring_set.py")
    parser.add_argument("--output", required=True, help="JSON lines of scores, appended to when resuming")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=None, help="Token truncation. Default: the model's context length.")
    parser.add_argument("--mixed_precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--quant", action="store_true", help="4-bit weights, for large decoders")
    parser.add_argument("--limit", type=int, default=None, help="Score at most this many commits (smoke tests)")
    args = parser.parse_args()

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]
    if dtype is torch.bfloat16 and not torch.cuda.is_available():
        dtype = torch.float32
    tokenizer, model = load_model(args.model_path, args.base_model_path, dtype, args.quant)
    device = next(model.parameters()).device
    max_length = args.max_length or min(int(tokenizer.model_max_length), int(model.config.max_position_embeddings))
    print(f"model on {device}, dtype {dtype}, max_length {max_length}, labels {model.config.id2label}")

    done = set()
    if os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as fh:
            done = {json.loads(line)["node"] for line in fh}
        print(f"resuming: {len(done)} commits already scored")
    rows = []
    skipped = 0
    with open(args.input, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("prompt") is None:
                skipped += 1
            elif row["node"] not in done:
                rows.append((row["node"], row["prompt"]))
    rows.sort(key=lambda row: len(row[1]), reverse=True)
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} commits to score, {skipped} ignored by bugbug's rules")

    def score(prompts: list[str]) -> list[list[float]]:
        encoded = tokenizer(prompts, truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**encoded).logits.float()
        return torch.softmax(logits, dim=-1).cpu().tolist()

    started = time.monotonic()
    scored = 0
    with open(args.output, "a", encoding="utf-8") as out:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            try:
                probabilities = score([prompt for _, prompt in batch])
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                probabilities = [score([prompt])[0] for _, prompt in batch]
            for (node, _), prob in zip(batch, probabilities):
                out.write(
                    json.dumps({"node": node, "prob": prob, "class": int(prob.index(max(prob))), "risk_score": prob[1]})
                    + "\n"
                )
            scored += len(batch)
            if (start // args.batch_size) % 50 == 49:
                out.flush()
                elapsed = time.monotonic() - started
                print(f"  {scored} of {len(rows)} scored, {scored / elapsed:.1f} commits/s, {elapsed / 60:.1f} min")
    elapsed = time.monotonic() - started
    print(f"done: {scored} commits in {elapsed / 60:.1f} min -> {args.output}")


if __name__ == "__main__":
    main()
