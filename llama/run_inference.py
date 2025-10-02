#!/usr/bin/env python3
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
from datetime import datetime
from typing import Optional, List, Tuple, Union

import numpy as np
import torch
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    set_seed,
    pipeline as hf_pipeline,
)
from peft import PeftModel, PeftConfig
from tqdm.auto import tqdm  # progress bar

# ---- Reuse your training utilities ----
from utils import (
    load_and_split_dataset,
    determine_format_fn,
    determine_tokenizer_truncation,
    register_custom_llama4_if_needed,
    recall_at_top_k,
    _compute_metrics_core,
)


def _is_peft_adapter(path: str) -> bool:
    try:
        _ = PeftConfig.from_pretrained(path, local_files_only=True)
        return True
    except Exception:
        return False


def _build_model_and_tokenizer_for_pipeline(
    model_or_adapter_path: str,
    base_model_path: Optional[str],
    model_kwargs: dict,
    local_files_only: bool = True,
) -> Tuple[Union[str, torch.nn.Module], AutoTokenizer, AutoConfig, bool]:
    """
    Returns (model_for_pipeline, tokenizer, config, used_adapter)

    - If `model_or_adapter_path` is a PEFT adapter:
        * Requires `base_model_path` (offline).
        * Tokenizer is loaded from the ADAPTER path.
        * Config is ALWAYS loaded from the BASE model path.
        * Loads base (optionally quantized), attaches adapter → returns model object.
        * used_adapter=True
    - Else (full model path):
        * Loads tokenizer & config from that model path.
        * Returns the path (pipeline lazy loads).
        * used_adapter=False
    """
    if _is_peft_adapter(model_or_adapter_path):
        if not base_model_path:
            raise ValueError("--base_model_path is required when --model_path points to a PEFT/LoRA adapter.")

        tokenizer = AutoTokenizer.from_pretrained(
            model_or_adapter_path, use_fast=True, local_files_only=local_files_only
        )

        # Config must come from BASE (never from adapter)
        config = AutoConfig.from_pretrained(base_model_path, local_files_only=local_files_only)

        base_model = AutoModelForSequenceClassification.from_pretrained(base_model_path, **model_kwargs)

        # make the embedding matrix match tokenizer size
        new_vocab = len(tokenizer)
        if getattr(base_model.config, "vocab_size", None) != new_vocab:
            base_model.resize_token_embeddings(new_vocab, mean_resizing=False, pad_to_multiple_of=8)
            base_model.config.vocab_size = new_vocab  # keep config consistent

        peft_model = PeftModel.from_pretrained(
            base_model,
            model_id=model_or_adapter_path,
            is_trainable=False,
            local_files_only=local_files_only,
        )
        peft_model.eval()

        return peft_model, tokenizer, base_model.config, True

    # Full model path
    tokenizer = AutoTokenizer.from_pretrained(model_or_adapter_path, use_fast=True, local_files_only=local_files_only)
    config = AutoConfig.from_pretrained(model_or_adapter_path, local_files_only=local_files_only)
    return model_or_adapter_path, tokenizer, config, False


def run_inference(
    model_path: str,
    dataset_path: Optional[str],
    output_dir: str,
    mixed_precision: str = "bf16",
    truncation_len: Optional[int] = None,
    threshold: Optional[float] = None,
    recall_pcts: str = "0.05,0.1,0.3",
    quant_4bit: bool = False,
    debug: bool = False,
    base_model_path: Optional[str] = None,
):
    """
    Runs inference on the held-out 10% split ('final_test') for SEQUENCE CLASSIFICATION only.
    - If `model_path` is a full model dir: use it directly (tokenizer/config from it).
    - If `model_path` is a PEFT/LoRA adapter: attach it to `base_model_path` (required),
      tokenizer comes from adapter, config comes from BASE ONLY.
    Saves results under: <output_dir>/eval_<timestamp>/final_test_results_seq_cls.json
    """
    os.makedirs(output_dir, exist_ok=True)
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

    # ---- Dataset ----
    REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_dict: DatasetDict = load_and_split_dataset(
        dataset_path=dataset_path,
        repo_path=REPO_PATH,
        slurm_tmpdir=os.environ.get("TMPDIR", ""),
        debug=False,
        format_fn=determine_format_fn("seq_cls"),
    )
    test_ds = dataset_dict["final_test"]

    if debug:
        n = min(200, len(test_ds))
        test_ds = test_ds.select(range(n))
        print(f"⚙️ Debug mode ON: using only {n} examples from final_test.")

    # ---- Model kwargs ----
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

    # ---- Build model/tokenizer/config ----
    model_for_pipeline, tokenizer, config, used_adapter = _build_model_and_tokenizer_for_pipeline(
        model_or_adapter_path=model_path,
        base_model_path=base_model_path,
        model_kwargs=model_kwargs,
        local_files_only=True,
    )

    # LLaMA-like padding
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Truncation policy ----
    should_trunc, tokenizer_max_len = determine_tokenizer_truncation(
        tokenizer=tokenizer,
        config=config,
        truncation_len=truncation_len,
        chunking_len=None,
    )

    # ---- Use BASE for registration when adapter is used ----
    config_source = base_model_path if used_adapter else model_path
    register_custom_llama4_if_needed(config_source)

    # ---- Pipeline ----
    clf = hf_pipeline(
        task="text-classification",
        model=model_for_pipeline,
        tokenizer=tokenizer,
        model_kwargs=model_kwargs if isinstance(model_for_pipeline, str) else {},
    )

    texts = test_ds["text"]
    labels = np.array(test_ds["labels"], dtype=np.int64)

    # Class order in id order
    num_labels = int(getattr(clf.model.config, "num_labels", 2))
    id2label_raw = getattr(clf.model.config, "id2label", {i: f"LABEL_{i}" for i in range(num_labels)})
    ordered_labels = []
    for i in range(num_labels):
        ordered_labels.append(id2label_raw.get(i) or id2label_raw.get(str(i)) or f"LABEL_{i}")

    # ---- Inference with tqdm ----
    probs_list = []
    for text in tqdm(texts, desc="Running inference", unit="sample"):
        out = clf(text, truncation=should_trunc, max_length=tokenizer_max_len, top_k=None)
        if isinstance(out, dict):
            scores = [out]
        elif isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
            scores = out
        elif isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
            scores = out[0]
        else:
            raise TypeError(f"Unexpected pipeline output format: {type(out)} -> {out}")

        score_map = {d["label"]: float(d["score"]) for d in scores}
        vec = [score_map.get(lbl, 0.0) for lbl in ordered_labels]
        probs_list.append(vec)

    probs = np.array(probs_list, dtype=np.float32)

    # Convert probabilities to "logits" for the offline core (stable via log)
    logits = np.log(np.clip(probs, 1e-12, 1.0))

    recall_pcts_list = [float(x) for x in recall_pcts.split(",") if x.strip()]

    metrics, preds, probs_again = _compute_metrics_core(
        logits,
        labels,
        repo_root=REPO_PATH,                 # ensures local metric loading via _load_metric_local_first
        threshold=threshold,                 # optional threshold for class 1; else argmax
        percentages=recall_pcts_list,        # recall@top_k percentages
        average="binary",
        recall_at_top_k_fn=recall_at_top_k,
    )

    # Optional commit_id
    if "commit_id" in test_ds.column_names:
        commit_ids = list(test_ds["commit_id"])
    else:
        print("⚠️ 'commit_id' column not found in final_test; setting commit_id=None for all samples.")
        commit_ids = [None] * len(labels)

    # Per-sample results
    samples = []
    for i in range(len(labels)):
        pred = int(preds[i])
        conf = float(probs_again[i, pred])
        samples.append({
            "commit_id": commit_ids[i],
            "true_label": int(labels[i]),
            "prediction": pred,
            "confidence": conf,
        })

    results = {
        "task_type": "seq_cls",
        "debug": debug,
        "base_model_path": base_model_path,
        "model_path": model_path,
        "dataset_path": dataset_path,
        "quant": quant_4bit,
        "used_samples": int(len(texts)),
        "label_order": ordered_labels,
        "metrics": metrics,
        "samples": samples,
    }

    out_json = os.path.join(out_eval_dir, "final_test_results_seq_cls.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"📄 Saved inference results on held-out test dataset → {out_json}")


def parse_args():
    p = argparse.ArgumentParser("Run inference on held-out last 10% split (final_test) for sequence classification.")
    p.add_argument(
        "--model_path",
        type=str,
        required=True,
        help=(
            "Either a full fine-tuned model directory OR a LoRA/PEFT adapter directory. "
            "If an adapter is passed, you must also supply --base_model_path."
        ),
    )
    p.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Offline local path to the base model (required if --model_path is a PEFT/LoRA adapter).",
    )
    p.add_argument("--dataset_path", type=str, default=None, help="Path to JSON dataset used in training (or 'imdb').")

    default_out = os.path.join(os.path.dirname(__file__), "inference")
    p.add_argument("--output_dir", type=str, default=default_out, help="Directory to save inference results.")
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--truncation_len", type=int, default=None, help="Optional max length for tokenization truncation.")
    p.add_argument("--threshold", type=float, default=None, help="Optional prob threshold for class 1; default uses argmax.")
    p.add_argument("--recall_pcts", type=str, default="0.05,0.1,0.3", help="CSV list for recall@top_k.")
    p.add_argument("--quant", action="store_true", help="Use 4-bit quantization for memory-constrained inference.")
    p.add_argument("--debug", action="store_true", help="Use only 200 samples from final_test for a quick debug run.")
    a = p.parse_args()

    if a.threshold is not None and not (0.0 <= a.threshold <= 1.0):
        raise ValueError("--threshold must be in [0,1].")

    if _is_peft_adapter(a.model_path) and not a.base_model_path:
        raise ValueError("--base_model_path is required when --model_path points to a PEFT/LoRA adapter.")

    return a


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        mixed_precision=args.mixed_precision,
        truncation_len=args.truncation_len,
        threshold=args.threshold,
        recall_pcts=args.recall_pcts,
        quant_4bit=args.quant,
        debug=args.debug,
    )
