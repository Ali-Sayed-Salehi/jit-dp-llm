#!/usr/bin/env python3
"""
merge_lora.py

Merge a LoRA adapter into a base Hugging Face model and save the merged model.
Works for:
  - causal language modeling (causal-lm)
  - sequence classification (seq-cls)

Key features:
  - Validates adapter folder (peft_type=LORA, r>0, non-empty tensors)
  - Loads tokenizer from adapter (if present) else falls back to base model
  - Merges in float32 for numerical stability, then (optionally) casts to bf16/fp16
  - For seq-cls: fixed defaults for num_labels/problem_type/label maps
  - Reties weights and resizes embeddings for added tokens when needed
"""

import argparse
import json
import os
import sys
from typing import Optional

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    PreTrainedModel,
)
from peft import PeftModel
from safetensors import safe_open  # validate adapter tensors


# =========================
# Defaults (edit these)
# =========================
DEFAULTS = {
    # Sequence classification defaults
    "num_labels": 2,
    "problem_type": "single_label_classification",
    "id2label": {0: "NEGATIVE", 1: "POSITIVE"},
    "label2id": {"NEGATIVE": 0, "POSITIVE": 1},

    # Loading behavior
    "local_files_only": False,
    "trust_remote_code": True,
}
# =========================

TASK_TO_CLASS = {
    "causal-lm": AutoModelForCausalLM,
    "seq-cls": AutoModelForSequenceClassification,
}


# ----------------------------- Utilities ----------------------------- #

def select_dtype(dtype_str: str):
    if dtype_str == "fp32":
        return torch.float32
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def validate_adapter_folder(adapter_source: str):
    cfg_path = os.path.join(adapter_source, "adapter_config.json")
    weights_path = os.path.join(adapter_source, "adapter_model.safetensors")

    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Missing adapter_config.json at: {cfg_path}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Missing adapter_model.safetensors at: {weights_path}")

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    peft_type = (cfg.get("peft_type") or "").upper()
    if peft_type != "LORA":
        raise ValueError(f"Adapter peft_type is '{cfg.get('peft_type')}', expected 'LORA'.")

    r = cfg.get("r", None)
    if r in (None, 0):
        raise ValueError(f"Adapter rank r={r}. This indicates zero-sized LoRA tensors. Re-train/export with r>0.")

    empties = []
    with safe_open(weights_path, framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if t.numel() == 0 or 0 in t.shape:
                empties.append((k, tuple(t.shape)))

    if empties:
        sample = ", ".join([f"{k}:{s}" for k, s in empties[:5]])
        raise ValueError(
            f"Adapter contains {len(empties)} zero-sized tensors (e.g., {sample}). "
            f"This adapter is invalid or was saved with r=0."
        )

    print("✅ Adapter folder validated: peft_type=LORA, r>0, tensors non-empty.")
    return cfg


def load_tokenizer(adapter_source: str, base_model: str, trust_remote_code: bool, local_files_only: bool):
    tok = None
    for src, local_only in [(adapter_source, True), (base_model, local_files_only)]:
        try:
            tok = AutoTokenizer.from_pretrained(
                src, use_fast=True, trust_remote_code=trust_remote_code, local_files_only=local_only
            )
            break
        except Exception:
            continue

    if tok is None:
        raise RuntimeError("Failed to load tokenizer from adapter or base model.")

    if tok.pad_token_id is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    return tok


def maybe_resize_embeddings(model: PreTrainedModel, tokenizer) -> None:
    try:
        current_vocab = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) != current_vocab:
            model.resize_token_embeddings(len(tokenizer))
            print(f"-> Resized token embeddings: {current_vocab} -> {len(tokenizer)}")
    except Exception as e:
        print(f"Warning: failed to check/resize embeddings: {e}")


def ensure_seqcls_head(model: PreTrainedModel, expected_num_labels: Optional[int]) -> None:
    if expected_num_labels is None:
        return
    head = getattr(model, "score", None)
    if head is None:
        head = getattr(model, "classifier", None)
        if hasattr(head, "out_proj"):
            head = head.out_proj
    out_features = getattr(head, "out_features", None)
    if head is None or out_features != expected_num_labels:
        raise RuntimeError(
            f"Classifier head mismatch after merge: got {out_features}, expected {expected_num_labels}."
        )


# ----------------------------- CLI ----------------------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model; save a standalone merged model ready for further training."
    )
    p.add_argument("--base_model", required=True,
                   help="Base model ID or local path (e.g., meta-llama/Llama-3.1-8B).")
    p.add_argument("--adapter_path", required=True,
                   help="Local LoRA adapter folder.")
    p.add_argument("--save_path", required=True,
                   help="Output directory for merged model.")
    p.add_argument("--task", choices=list(TASK_TO_CLASS.keys()), default="causal-lm",
                   help="Which head to load for the base model.")
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16",
                   help="Final dtype to save (merge in fp32, then cast).")
    return p.parse_args()


# ----------------------------- Main ----------------------------- #

def main():
    args = parse_args()

    num_labels = DEFAULTS["num_labels"] if args.task == "seq-cls" else None
    problem_type = DEFAULTS["problem_type"] if args.task == "seq-cls" else None
    id2label = DEFAULTS["id2label"] if args.task == "seq-cls" else None
    label2id = DEFAULTS["label2id"] if args.task == "seq-cls" else None
    local_only = DEFAULTS["local_files_only"]
    trust_remote = DEFAULTS["trust_remote_code"]

    final_dtype = select_dtype(args.dtype)

    validate_adapter_folder(args.adapter_path)

    tokenizer = load_tokenizer(args.adapter_path, args.base_model, trust_remote, local_only)

    config = AutoConfig.from_pretrained(
        args.base_model,
        trust_remote_code=trust_remote,
        local_files_only=local_only,
    )
    if args.task == "seq-cls":
        if num_labels is not None:
            config.num_labels = num_labels
        if problem_type:
            config.problem_type = problem_type
        if id2label and label2id:
            config.id2label = id2label
            config.label2id = label2id

    ModelClass = TASK_TO_CLASS[args.task]
    print(f"-> Loading base model in fp32 for merge: {args.base_model}")
    model = ModelClass.from_pretrained(
        args.base_model,
        config=config if args.task == "seq-cls" else None,
        torch_dtype=torch.float32,
        trust_remote_code=trust_remote,
        local_files_only=local_only,
        low_cpu_mem_usage=True,
    )

    maybe_resize_embeddings(model, tokenizer)

    print(f"-> Loading LoRA adapter from: {args.adapter_path} ...")
    peft_model = PeftModel.from_pretrained(
        model,
        args.adapter_path,
        is_trainable=False,
        trust_remote_code=trust_remote,
        local_files_only=False,
    )

    if tokenizer.pad_token_id is None and hasattr(peft_model.config, "eos_token_id"):
        tokenizer.pad_token = tokenizer.eos_token
        if hasattr(peft_model.config, "pad_token_id"):
            peft_model.config.pad_token_id = tokenizer.pad_token_id

    print("-> Merging LoRA weights into base model (fp32)...")
    merged_model = peft_model.merge_and_unload(safe_merge =True)

    if final_dtype != torch.float32:
        merged_model = merged_model.to(final_dtype)

    if args.task == "seq-cls":
        ensure_seqcls_head(merged_model, getattr(merged_model.config, "num_labels", None))

    if args.task == "causal-lm" and hasattr(merged_model, "tie_weights"):
        merged_model.tie_weights()
    if hasattr(merged_model.config, "use_cache"):
        merged_model.config.use_cache = True
    try:
        merged_model.config.torch_dtype = str(final_dtype).split(".")[-1]
    except Exception:
        pass

    os.makedirs(args.save_path, exist_ok=True)
    print(f"-> Saving merged model to: {args.save_path} ...")
    merged_model.save_pretrained(args.save_path, safe_serialization=True)
    tokenizer.save_pretrained(args.save_path)

    print("✅ Done. The directory now contains a standalone, merged model ready for further training.")


if __name__ == "__main__":
    sys.exit(main())
