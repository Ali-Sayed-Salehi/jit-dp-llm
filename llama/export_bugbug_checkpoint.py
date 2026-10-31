#!/usr/bin/env python
"""Package a trained sequence classification checkpoint for bugbug.

bugbug's ``PerfRegressionPredictorModel.load`` wants a Hugging Face
directory named ``perfregressionpredictormodel`` with a two-label model whose
label 1 is called POSITIVE, a tokenizer that holds the eight structural tokens
of the prompts, and an optional ``bugbug_model.json`` whose ``model_version``
names the checkpoint in every score it produces. The service downloads it as
a ``.tar.zst`` of that single directory.

This script takes a trained run (merged with ``merge_lora.py --task seq-cls``
first if it was LoRA), rewrites the label names, checks the tokenizer, writes
the metadata and the archive, and with ``--smoke-test`` loads the result
through bugbug itself when bugbug is importable.

    python llama/export_bugbug_checkpoint.py \\
        --model_path llama/training/run_.../model \\
        --output_dir LLMs/trained/sequence-classification/perf_mbert_v2 \\
        --model_version perf-regression-predictor-v2-mbert-2026-11 \\
        --dataset datasets/mozilla_perf_v2/perf_bugbug_struc.jsonl \\
        --training_config llama/configs/templates/train_config_speed_mbert_perf_v2.yaml
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone

from transformers import AutoConfig, AutoTokenizer

BUGBUG_DIRECTORY = "perfregressionpredictormodel"
STRUCTURAL_TOKENS = {
    "<COMMIT_MESSAGE>",
    "</COMMIT_MESSAGE>",
    "<FILE>",
    "</FILE>",
    "<ADDED>",
    "</ADDED>",
    "<REMOVED>",
    "</REMOVED>",
}


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path: str) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def compress(directory: str, archive: str) -> None:
    tar_path = archive[: -len(".zst")]
    with tarfile.open(tar_path, "w") as tar:
        tar.add(directory, arcname=os.path.basename(directory))
    try:
        import zstandard

        with open(tar_path, "rb") as src, open(archive, "wb") as dst:
            zstandard.ZstdCompressor(level=10).copy_stream(src, dst)
    except ImportError:
        subprocess.run(["zstd", "-10", "--rm", "-f", "-o", archive, tar_path], check=True)
    if os.path.exists(tar_path):
        os.remove(tar_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", required=True, help="Trained Hugging Face sequence classification directory (merged, not a LoRA adapter)")
    parser.add_argument("--output_dir", required=True, help="Where perfregressionpredictormodel/ and the .tar.zst are written")
    parser.add_argument("--model_version", required=True, help="Recorded in bugbug_model.json and in every score, e.g. perf-regression-predictor-v2-mbert-2026-11")
    parser.add_argument("--dataset", help="Training JSON lines file, recorded with its hash")
    parser.add_argument("--training_config", help="Training YAML, copied into the metadata")
    parser.add_argument("--notes", default="", help="Free text recorded in the metadata")
    parser.add_argument("--smoke-test", action="store_true", help="Load the result with bugbug and classify a dummy commit (needs bugbug importable)")
    args = parser.parse_args()

    if os.path.exists(os.path.join(args.model_path, "adapter_config.json")):
        raise SystemExit("This is a LoRA adapter; merge it first with llama/merge_lora.py --task seq-cls")

    source_config = AutoConfig.from_pretrained(args.model_path)
    target = os.path.join(args.output_dir, BUGBUG_DIRECTORY)
    if os.path.exists(target):
        shutil.rmtree(target)
    shutil.copytree(args.model_path, target, ignore=shutil.ignore_patterns("checkpoint-*", "optimizer.pt", "scheduler.pt", "rng_state*", "trainer_state.json", "training_args.bin"))

    config = AutoConfig.from_pretrained(target)
    if int(config.num_labels) != 2:
        raise SystemExit(f"bugbug needs exactly two labels, the checkpoint has {config.num_labels}")
    config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    config.label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    config.save_pretrained(target)

    tokenizer = AutoTokenizer.from_pretrained(target)
    missing = STRUCTURAL_TOKENS - set(tokenizer.get_added_vocab())
    if missing:
        raise SystemExit(f"the tokenizer lacks the structural tokens {sorted(missing)}; was it saved from the training run?")

    metadata = {
        "model_name": "Perf Regression Predictor",
        "model_version": args.model_version,
        "base_model": getattr(source_config, "_name_or_path", None),
        "model_type": config.model_type,
        "max_position_embeddings": int(getattr(config, "max_position_embeddings", 0)),
        "tokenizer_model_max_length": int(tokenizer.model_max_length) if tokenizer.model_max_length < 10**9 else None,
        "trained_from": os.path.abspath(args.model_path),
        "dataset": None,
        "training_config": None,
        "jit_dp_llm_revision": git_revision(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "notes": args.notes,
    }
    if args.dataset:
        meta_path = args.dataset + ".meta.json"
        metadata["dataset"] = {
            "path": os.path.abspath(args.dataset),
            "sha256": sha256_of(args.dataset),
            **(json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}),
        }
    if args.training_config:
        with open(args.training_config, encoding="utf-8") as fh:
            metadata["training_config"] = {"path": os.path.abspath(args.training_config), "content": fh.read()}
    with open(os.path.join(target, "bugbug_model.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    archive = os.path.join(args.output_dir, f"{BUGBUG_DIRECTORY}.tar.zst")
    compress(target, archive)
    print(f"packaged {target} as {archive} ({os.path.getsize(archive) / 1e6:.0f} MB), version {args.model_version}")

    if args.smoke_test:
        from bugbug.models.perf_regression_predictor import PerfRegressionPredictorModel

        model = PerfRegressionPredictorModel.load(target)
        probabilities = model.classify(
            [{"commit_message": "Bug 1 - Speed up layout", "diff": "diff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -1 +1 @@\n-a\n+b\n"}],
            probabilities=True,
        )
        print(f"bugbug loads it: version {model.version}, max_length {model.max_length}, dummy risk {probabilities[0][1]:.4f}")


if __name__ == "__main__":
    main()
