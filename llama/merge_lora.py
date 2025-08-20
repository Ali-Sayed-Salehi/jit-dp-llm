"""
merge_lora.py

Merge LoRA adapter weights into a base Hugging Face model and save the merged model.
Optionally:
  * upload the LoRA adapter folder to the Hugging Face Hub
  * load the LoRA adapter directly from the Hub instead of a local directory

Hugging Face token is read from the HF_TOKEN environment variable for pushes.
"""

import argparse
import json
import os
import sys

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from peft import PeftModel, PeftConfig, PeftType
from huggingface_hub import create_repo, upload_folder

from safetensors import safe_open  # for validating adapter tensors

from utils import login_to_huggingface

REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

TASK_TO_CLASS = {
    "causal-lm": AutoModelForCausalLM,
    "seq-cls": AutoModelForSequenceClassification,
}


def parse_args():
    p = argparse.ArgumentParser(description="Merge LoRA adapter into base model; optionally push/pull LoRA from Hugging Face Hub.")
    # Core
    p.add_argument("--base_model", required=True,
                   help="Base model ID or local path (e.g., meta-llama/Llama-3.1-8B).")
    p.add_argument("--save_path", required=True,
                   help="Where to save the merged (final) model.")
    p.add_argument("--task", choices=list(TASK_TO_CLASS.keys()),
                   default="causal-lm",
                   help="Model task/head to load.")
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"],
                   default="bf16",
                   help="Torch dtype to load/merge with.")

    # Adapter source: local or hub
    p.add_argument("--adapter_path",
                   help="Local path to LoRA adapter dir (contains adapter_config.json + adapter_model.safetensors).")
    p.add_argument("--adapter_from_hub", action="store_true",
                   help="If set, load LoRA adapter from the Hub repo specified by --adapter_repo_id.")
    p.add_argument("--adapter_repo_id",
                   help="Hub repo id for the adapter (e.g., username/repo). Used for pulling (--adapter_from_hub) or pushing (--push_adapter_to_hub).")

    # Push to hub
    p.add_argument("--push_adapter_to_hub", action="store_true",
                   help="If set, upload the adapter at --adapter_path to Hugging Face Hub (--adapter_repo_id required).")

    return p.parse_args()


def select_dtype(dtype_str: str):
    if dtype_str == "fp32":
        return torch.float32
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def push_adapter_to_hub(adapter_path: str, repo_id: str, private: bool):
    if not adapter_path or not os.path.isdir(adapter_path):
        raise ValueError("--push_adapter_to_hub requires a valid --adapter_path (local folder).")
    if not repo_id:
        raise ValueError("--push_adapter_to_hub requires --adapter_repo_id.")

    login_to_huggingface(REPO_PATH)
    create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    upload_folder(
        folder_path=adapter_path,
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=".",
    )
    print(f"✅ Pushed adapter to hub: {repo_id}")


def validate_adapter_folder(adapter_source: str):
    """
    Validate the adapter folder has:
      - adapter_config.json present and peft_type == LORA
      - r > 0
      - adapter_model.safetensors present and contains no zero-sized tensors
    """
    cfg_path = os.path.join(adapter_source, "adapter_config.json")
    weights_path = os.path.join(adapter_source, "adapter_model.safetensors")

    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Missing adapter_config.json at: {cfg_path}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Missing adapter_model.safetensors at: {weights_path}")

    cfg = json.load(open(cfg_path))
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


def main():
    args = parse_args()

    if args.adapter_from_hub:
        if not args.adapter_repo_id:
            raise ValueError("--adapter_from_hub requires --adapter_repo_id.")
    else:
        if not args.adapter_path:
            raise ValueError("Provide --adapter_path or set --adapter_from_hub with --adapter_repo_id.")

    if args.push_adapter_to_hub:
        push_adapter_to_hub(
            adapter_path=args.adapter_path,
            repo_id=args.adapter_repo_id,
            private=False,  # default
        )

    dtype = select_dtype(args.dtype)
    ModelClass = TASK_TO_CLASS[args.task]

    print(f"-> Loading tokenizer from base: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=False,
    )

    print(f"-> Loading base model: {args.base_model}")
    model = ModelClass.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=False,
    )

    adapter_source = args.adapter_repo_id if args.adapter_from_hub else args.adapter_path
    print(f"-> Loading LoRA adapter from: {adapter_source}")

    # Preflight checks
    validate_adapter_folder(adapter_source)

    peft_cfg = PeftConfig.from_pretrained(
        adapter_source,
        trust_remote_code=True,
    )
    if peft_cfg.peft_type != PeftType.LORA:
        raise ValueError(f"Adapter at {adapter_source} is '{peft_cfg.peft_type}', not LoRA.")

    peft_model = PeftModel.from_pretrained(
        model,
        adapter_source,
        is_trainable=False,
        trust_remote_code=True,
        local_files_only=False,
    )

    if tokenizer.pad_token_id is None and hasattr(peft_model.config, "eos_token_id"):
        tokenizer.pad_token = tokenizer.eos_token
        if hasattr(peft_model.config, "pad_token_id"):
            peft_model.config.pad_token_id = tokenizer.pad_token_id

    print("-> Merging LoRA weights into the base model...")
    merged_model = peft_model.merge_and_unload()

    if hasattr(merged_model.config, "use_cache"):
        merged_model.config.use_cache = True

    os.makedirs(args.save_path, exist_ok=True)
    print(f"-> Saving merged model to: {args.save_path}")
    merged_model.save_pretrained(args.save_path, safe_serialization=True)
    tokenizer.save_pretrained(args.save_path)

    print("✅ Done. The directory now contains a standalone, merged model.")


if __name__ == "__main__":
    sys.exit(main())