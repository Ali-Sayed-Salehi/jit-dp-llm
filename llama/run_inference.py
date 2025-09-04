#!/usr/bin/env python3
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
from datetime import datetime
from typing import Optional, List

import numpy as np
import torch
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
    set_seed,
    pipeline as hf_pipeline,
)

# ---- Reuse your training utilities ----
from utils import (
    load_and_split_dataset,
    determine_format_fn,
    determine_tokenizer_truncation,
    register_custom_llama4_if_needed,
    recall_at_top_k,
)


def compute_seqcls_metrics_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    threshold: Optional[float],
    recall_pcts: List[float]
):
    import evaluate
    from sklearn.metrics import roc_auc_score, average_precision_score

    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    probs = torch.softmax(torch.tensor(logits), dim=-1).cpu().numpy()
    preds = (probs[:, 1] >= threshold).astype(int) if threshold is not None else np.argmax(probs, axis=1)

    rk = recall_at_top_k(probs[:, 1], labels, percentages=recall_pcts)

    try:
        roc_auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(labels, probs[:, 1])
    except ValueError:
        pr_auc = float("nan")

    metrics = {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "precision": precision.compute(predictions=preds, references=labels, average="binary")["precision"],
        "recall": recall.compute(predictions=preds, references=labels, average="binary")["recall"],
        "f1": f1.compute(predictions=preds, references=labels, average="binary")["f1"],
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }
    metrics.update(rk)
    return metrics, preds, probs


def run_inference(
    model_path: str,
    dataset_path: Optional[str],
    output_dir: str,
    mixed_precision: str = "bf16",
    batch_size: int = 1,  # kept for CLI compatibility; not used (we run one-by-one)
    truncation_len: Optional[int] = None,
    threshold: Optional[float] = None,
    recall_pcts: str = "0.05,0.1,0.3",
    quant_4bit: bool = False,
    debug: bool = False,
):
    """
    Runs inference on the held-out 10% split ('final_test') for SEQUENCE CLASSIFICATION only.
    Uses transformers.pipeline (pipeline loads the model), calling it one example at a time.
    Saves results under: <output_dir>/eval_<timestamp>/final_test_results_seq_cls.json
    """
    # Ensure base output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Timestamped eval subdir
    out_eval_dir = os.path.join(output_dir, "eval_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(out_eval_dir, exist_ok=True)

    set_seed(42)

    # ---- Precision / dtype ----
    if mixed_precision == "fp16":
        dtype = torch.float16
    elif mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif mixed_precision == "fp32":
        dtype = torch.float32
    else:
        raise ValueError("mixed_precision must be one of: fp32, fp16, bf16")

    # ---- Dataset (reuses the 80/10/10 split) ----
    REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_dict: DatasetDict = load_and_split_dataset(
        dataset_path=dataset_path,
        repo_path=REPO_PATH,
        slurm_tmpdir=os.environ.get("TMPDIR", ""),
        debug=False,
        format_fn=determine_format_fn("seq_cls"),
    )
    test_ds = dataset_dict["final_test"]

    # ---- Debug subset BEFORE tokenization for speed ----
    if debug:
        n = min(200, len(test_ds))
        test_ds = test_ds.select(range(n))
        print(f"⚙️ Debug mode ON: using only {n} examples from final_test.")

    # ---- Tokenizer / config (pipeline will load the model; we still manage tokenizer/truncation) ----
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)

    # LLaMA-like padding
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Truncation policy (chunking not needed for seq_cls) ----
    should_trunc, tokenizer_max_len = determine_tokenizer_truncation(
        tokenizer=tokenizer,
        config=config,
        truncation_len=truncation_len,
        chunking_len=None,
    )

    # ---- Optional custom registration for your custom LLaMA4 seq-cls head ----
    register_custom_llama4_if_needed(model_path)

    # ---- Build pipeline (let it load the model) ----
    model_kwargs = dict(
        device_map="auto",
        torch_dtype=dtype if not quant_4bit else (torch.bfloat16 if dtype == torch.bfloat16 else torch.float32),
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    if quant_4bit:
        print("🔢 Loading with 4-bit quantization for inference...")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )

    clf = hf_pipeline(
        task="text-classification",
        model=model_path,
        tokenizer=tokenizer,
        model_kwargs=model_kwargs,
    )

    texts = test_ds["text"]
    labels = np.array(test_ds["labels"], dtype=np.int64)

    # Class order based on loaded model's config (handle int OR str keys)
    num_labels = int(getattr(clf.model.config, "num_labels", 2))
    id2label_raw = getattr(clf.model.config, "id2label", {i: f"LABEL_{i}" for i in range(num_labels)})
    ordered_labels = []
    for i in range(num_labels):
        ordered_labels.append(id2label_raw.get(i) or id2label_raw.get(str(i)) or f"LABEL_{i}")

    # ---- Run pipeline ONE BY ONE and collect per-class probabilities ----
    probs_list = []
    for text in texts:
        out = clf(text, truncation=should_trunc, max_length=tokenizer_max_len, top_k=None)
        # Normalize possible shapes: dict → [dict], [dict] → [dict], [[dict]] → [dict]
        if isinstance(out, dict):
            scores = [out]
        elif isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
            scores = out  # expected case for top_k=None on single input
        elif isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
            scores = out[0]  # rare nested shape
        else:
            raise TypeError(f"Unexpected pipeline output format: {type(out)} -> {out}")

        score_map = {d["label"]: float(d["score"]) for d in scores}
        vec = [score_map.get(lbl, 0.0) for lbl in ordered_labels]  # probability vector in id order
        probs_list.append(vec)

    probs = np.array(probs_list, dtype=np.float32)
    logits = np.log(np.clip(probs, 1e-12, 1.0))  # convert to logits for metric helper

    recall_pcts_list = [float(x) for x in recall_pcts.split(",") if x.strip()]
    metrics, preds, probs_again = compute_seqcls_metrics_from_logits(
        logits=logits,
        labels=labels,
        threshold=threshold,
        recall_pcts=recall_pcts_list,
    )

    # Try to pull commit_id from the dataset (gracefully handle absence)
    if "commit_id" in test_ds.column_names:
        commit_ids = list(test_ds["commit_id"])
    else:
        print("⚠️ 'commit_id' column not found in final_test; setting commit_id=None for all samples.")
        commit_ids = [None] * len(labels)

    # Build per-sample dicts with commit_id, true_label, prediction, confidence
    samples = []
    for i in range(len(labels)):
        pred = int(preds[i])
        conf = float(probs_again[i, pred])   # probability of the predicted class
        samples.append({
            "commit_id": commit_ids[i],
            "true_label": int(labels[i]),
            "prediction": pred,
            "confidence": conf,
        })

    results = {
        "task_type": "seq_cls",
        "debug": debug,
        "used_samples": int(len(texts)),
        "label_order": ordered_labels,  # helpful for interpreting class indices
        "metrics": metrics,
        "samples": samples,             # ✅ per-sample dicts live here
    }

    # ---- Save JSON results ----
    out_json = os.path.join(out_eval_dir, "final_test_results_seq_cls.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"📄 Saved inference results on held-out test dataset → {out_json}")


def parse_args():
    p = argparse.ArgumentParser("Run inference on held-out last 10% split (final_test) for sequence classification.")
    p.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned sequence classification model directory.")
    p.add_argument("--dataset_path", type=str, default=None, help="Path to JSON dataset used in training (or 'imdb').")

    # Default to llama/inference relative to this script's directory
    default_out = os.path.join(os.path.dirname(__file__), "inference")
    p.add_argument(
        "--output_dir",
        type=str,
        default=default_out,
        help="Directory to save inference results (default: llama/inference)."
    )

    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--batch_size", type=int, default=1, help="(Ignored) Kept for CLI compatibility.")
    p.add_argument("--truncation_len", type=int, default=None, help="Optional max length for tokenization truncation.")
    p.add_argument("--threshold", type=float, default=None, help="Optional prob threshold for class 1; default uses argmax.")
    p.add_argument("--recall_pcts", type=str, default="0.05,0.1,0.3", help="CSV list for recall@top_k.")
    p.add_argument("--quant", action="store_true", help="Use 4-bit quantization for memory-constrained inference.")
    p.add_argument("--debug", action="store_true", help="Use only 50 samples from final_test for a quick debug run.")
    a = p.parse_args()

    if a.threshold is not None and not (0.0 <= a.threshold <= 1.0):
        raise ValueError("--threshold must be in [0,1].")

    return a


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        mixed_precision=args.mixed_precision,
        batch_size=args.batch_size,
        truncation_len=args.truncation_len,
        threshold=args.threshold,
        recall_pcts=args.recall_pcts,
        quant_4bit=args.quant,
        debug=args.debug,
    )
