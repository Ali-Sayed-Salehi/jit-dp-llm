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
    p.add_argument("--model_id", type=str, nargs="+", default=["meta-llama/Llama-3.1-8B"],
                   help="One or more Hugging Face model IDs; each is saved under LLMs/snapshots/<id>")
    p.add_argument("--model_head", choices=["causal-lm", "sequence-classification"],
                   default="causal-lm", help="Model head to use if instantiating")
    p.add_argument("--save_path", type=str, default=None,
                   help="Optional: override default save path")
    p.add_argument("--instantiate", action="store_true",
                   help="Instantiate the model object after downloading (requires significant RAM/VRAM)")
    args = p.parse_args()

    # ----------------- Paths -----------------
    REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print(f"✅ REPO_PATH: {REPO_PATH}")
    if args.save_path and len(args.model_id) > 1:
        raise SystemExit("--save_path only applies to a single --model_id")

    # ----------------- HF login -----------------
    login_to_huggingface(REPO_PATH)
    enable_hf_transfer_if_available()

    for model_id in args.model_id:
        save_path = args.save_path or os.path.join(REPO_PATH, "LLMs", "snapshots", model_id)
        os.makedirs(save_path, exist_ok=True)
        print(f"✅ Model will be saved to: {save_path}")

        # ----------------- Snapshot (always) -----------------
        print(f"📦 Downloading snapshot of {model_id} to {save_path} ...")
        snapshot_download(
            repo_id=model_id,
            local_dir=save_path,
        )
        print("✅ Snapshot complete.")

        if not args.instantiate:
            print("🧊 Skipping model instantiation (default). Files are on disk.")
            continue

        # ----------------- Optional: Instantiate model object -----------------
        print(f"⚠️  Instantiating {args.model_head} model object. This may require significant memory.")

        common_kwargs = dict(
            trust_remote_code=True,
            attn_implementation="sdpa",
            device_map="auto",
        )

        if args.model_head == "causal-lm":
            model = AutoModelForCausalLM.from_pretrained(model_id, **common_kwargs)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_id, **common_kwargs)

        print("💾 Saving instantiated model to save_path...")
        model.save_pretrained(save_path)
        print(f"✅ Model saved to {save_path}")


if __name__ == "__main__":
    main()