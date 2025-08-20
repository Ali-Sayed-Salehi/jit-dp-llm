#!/usr/bin/env python
import os
import argparse
import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)

from utils import login_to_huggingface


def enable_hf_transfer_if_available():
    """Enable accelerated transfers if hf_transfer is installed."""
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "") == "1":
        print("⚡ HF Transfer already enabled via env.")
        return
    try:
        import hf_transfer
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        print("⚡ Enabled HF Transfer (HF_HUB_ENABLE_HF_TRANSFER=1).")
    except Exception:
        print("ℹ️ hf_transfer not installed; using standard downloader."
              " Install with: pip install -U hf-transfer")


def main():
    # ----------------- Arg parser -----------------
    p = argparse.ArgumentParser(
        description="Download (snapshot) and optionally instantiate a model + tokenizer."
    )
    p.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B",
                   help="Hugging Face model ID")
    p.add_argument("--model_head", choices=["causal-lm", "sequence-classification"],
                   default="causal-lm", help="Model head to use if instantiating")
    p.add_argument("--save_path", type=str, default=None,
                   help="Optional: override default save path")
    p.add_argument("--instantiate", action="store_true",
                   help="Instantiate the model object after downloading (requires significant RAM/VRAM)")
    args = p.parse_args()

    # ----------------- Paths -----------------
    REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_save_path = os.path.join(REPO_PATH, "LLMs", "snapshots", args.model_id)
    save_path = args.save_path or default_save_path
    os.makedirs(save_path, exist_ok=True)

    print(f"✅ REPO_PATH: {REPO_PATH}")
    print(f"✅ Model will be saved to: {save_path}")

    # ----------------- HF login -----------------
    login_to_huggingface(REPO_PATH)
    enable_hf_transfer_if_available()

    # ----------------- Snapshot (always) -----------------
    print(f"📦 Downloading snapshot of {args.model_id} to {save_path} ...")
    snapshot_download(
        repo_id=args.model_id,
        local_dir=save_path,
    )
    print("✅ Snapshot complete.")

    if not args.instantiate:
        print("🧊 Skipping model instantiation (default). Files are on disk.")
        return

    # ----------------- Optional: Instantiate model object -----------------
    print(f"⚠️  Instantiating {args.model_head} model object. This may require significant memory.")

    common_kwargs = dict(
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map="auto",
    )

    if args.model_head == "causal-lm":
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **common_kwargs)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_id, **common_kwargs)

    print("💾 Saving instantiated model to save_path...")
    model.save_pretrained(save_path)
    print(f"✅ Model saved to {save_path}")


if __name__ == "__main__":
    main()