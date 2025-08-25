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

import hashlib
from typing import Optional, Tuple
from safetensors import safe_open

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

def get_added_token_ids(tokenizer):
    """Return the list of ids for tokens added after the base vocab."""
    try:
        added = tokenizer.get_added_vocab()  # {token_str: id}
        return sorted(added.values())
    except Exception:
        return []

def _normalize_adapter_key_for_embed(k: str) -> str:
    # Strip PEFT prefixes
    for p in ("base_model.model.", "base_model."):
        if k.startswith(p):
            k = k[len(p):]
    return k

def _find_adapter_embedding_tensor(adapter_path: str):
    """
    Try to retrieve an embedding matrix from the adapter. Preference order:
      1) Dense saved embedding (modules_to_save): *embed_tokens.weight
      2) Token adapter tensor from trainable_token_indices: *embed_tokens.token_adapter.base_layer.weight
    Returns (tensor, key, kind) or (None, None, None).
    """
    weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.isfile(weights_path):
        return None, None, None

    dense_candidate = None
    dense_key = None
    adapter_candidate = None
    adapter_key = None

    with safe_open(weights_path, framework="pt", device="cpu") as fh:
        for raw_k in fh.keys():
            if "lora_" in raw_k:
                continue  # we want dense/adapter, not LoRA
            k = _normalize_adapter_key_for_embed(raw_k)
            # 1) Dense embed (modules_to_save)
            if (".weight" in k) and ("embed_tokens" in k) and ("token_adapter" not in k):
                dense_candidate = fh.get_tensor(raw_k).cpu()
                dense_key = raw_k
            # 2) Token adapter base layer
            if "embed_tokens.token_adapter.base_layer.weight" in k:
                adapter_candidate = fh.get_tensor(raw_k).cpu()
                adapter_key = raw_k

    if dense_candidate is not None:
        return dense_candidate, dense_key, "dense"
    if adapter_candidate is not None:
        return adapter_candidate, adapter_key, "token_adapter"
    return None, None, None

def verify_adapter_embeddings_applied(model: PreTrainedModel, tokenizer, adapter_path: str):
    """
    Compare adapter's embedding matrix (dense or token_adapter) with the merged model's input embeddings
    on the ADDED TOKEN IDS. Prints max|Δ| and L2 norms.
    """
    added_ids = get_added_token_ids(tokenizer)
    if not added_ids:
        print("[embed-verify] No added tokens detected; nothing to verify.")
        return

    adapter_w, adapter_key, kind = _find_adapter_embedding_tensor(adapter_path)
    if adapter_w is None:
        print("[embed-verify] Adapter does not contain a dense embedding or token_adapter tensor.")
        return

    model_in = model.get_input_embeddings()
    if model_in is None or not hasattr(model_in, "weight"):
        print("[embed-verify] Could not access model input embeddings.")
        return
    model_w = model_in.weight.detach().cpu()

    if adapter_w.shape != model_w.shape:
        # Token adapters usually store a full-sized matrix after training those rows.
        # If shapes differ, we can't row-compare directly.
        print(f"[embed-verify] Shape mismatch: adapter {tuple(adapter_w.shape)} vs model {tuple(model_w.shape)}")
        return

    # Compare only added rows
    a = adapter_w.index_select(0, torch.tensor(added_ids))
    b = model_w.index_select(0, torch.tensor(added_ids))
    diff = (b.float() - a.float())
    max_abs = float(diff.abs().max())
    l2 = float(torch.linalg.vector_norm(diff))
    print(f"[embed-verify] Compared {len(added_ids)} added rows from adapter '{adapter_key}' ({kind}) "
          f"to merged model embeddings -> max|Δ|={max_abs:.3e}, L2={l2:.3e}")

    # Optional per-row summary: count exact/near matches
    near_matches = (diff.abs().amax(dim=1) < 1e-6).sum().item()
    print(f"[embed-verify] Rows with max|Δ| < 1e-6: {near_matches}/{len(added_ids)}")


def inspect_adapter_heads(adapter_path: str, adapter_cfg: dict):
    """
    Returns a dict with:
      - lora_for_lm_head / lora_for_score / lora_for_classifier (bool)
      - dense_lm_head_saved / dense_score_saved / dense_classifier_saved (bool)
      - modules_to_save (list from adapter_config)
      - keys_scanned (int)
    """
    weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Missing weights file: {weights_path}")

    modules_to_save = adapter_cfg.get("modules_to_save", []) or []
    keys = []
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())

    # Heuristics:
    # - LoRA deltas typically have "lora_A" or "lora_B" (and sometimes "lora_embedding_A/B").
    # - Dense weights saved via modules_to_save will appear as plain ".weight"/".bias" entries without "lora_".
    def is_lora_key(k: str) -> bool:
        return "lora_A" in k or "lora_B" in k or "lora_embedding_A" in k or "lora_embedding_B" in k

    def has_lora_for(token: str) -> bool:
        return any((token in k) and is_lora_key(k) for k in keys)

    def has_dense_for(token: str) -> bool:
        return any((token in k) and (".weight" in k or ".bias" in k) and not is_lora_key(k) for k in keys)

    # Check for CLM head
    lora_for_lm_head = has_lora_for("lm_head")
    dense_lm_head_saved = has_dense_for("lm_head") or ("lm_head" in modules_to_save)

    # Check for sequence-classification heads (score / classifier)
    lora_for_score = has_lora_for("score")
    dense_score_saved = has_dense_for("score") or ("score" in modules_to_save)

    lora_for_classifier = has_lora_for("classifier")
    dense_classifier_saved = has_dense_for("classifier") or ("classifier" in modules_to_save)

    report = {
        "modules_to_save": modules_to_save,
        "keys_scanned": len(keys),

        "lora_for_lm_head": lora_for_lm_head,
        "dense_lm_head_saved": dense_lm_head_saved,

        "lora_for_score": lora_for_score,
        "dense_score_saved": dense_score_saved,

        "lora_for_classifier": lora_for_classifier,
        "dense_classifier_saved": dense_classifier_saved,
    }
    return report

def print_adapter_head_report(report: dict):
    print("=== Adapter head coverage report ===")
    print(f"• keys scanned: {report['keys_scanned']}")
    print(f"• modules_to_save: {report['modules_to_save']}")
    print("")
    print(f"CLM head (lm_head):")
    print(f"  - LoRA deltas present: {report['lora_for_lm_head']}")
    print(f"  - Dense weights saved: {report['dense_lm_head_saved']}")
    print("")
    print(f"Seq-cls head (score):")
    print(f"  - LoRA deltas present: {report['lora_for_score']}")
    print(f"  - Dense weights saved: {report['dense_score_saved']}")
    print("")
    print(f"Seq-cls head (classifier):")
    print(f"  - LoRA deltas present: {report['lora_for_classifier']}")
    print(f"  - Dense weights saved: {report['dense_classifier_saved']}")
    print("====================================")


def _sha256_tensor(t: torch.Tensor) -> str:
    # Device/dtype agnostic fingerprint
    a = t.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()
    return hashlib.sha256(a).hexdigest()

def _find_seqcls_head_module(model: PreTrainedModel) -> Optional[torch.nn.Module]:
    # Priority by common naming
    if hasattr(model, "score") and isinstance(getattr(model, "score"), torch.nn.Module):
        return getattr(model, "score")
    if hasattr(model, "classifier") and isinstance(getattr(model, "classifier"), torch.nn.Module):
        head = getattr(model, "classifier")
        # Some heads wrap the last linear as out_proj
        if hasattr(head, "out_proj") and isinstance(head.out_proj, torch.nn.Module):
            return head.out_proj
        return head
    # Heuristic: pick the last Linear named like "*score" or "*classifier"
    candidates = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            last = name.split(".")[-1]
            if last in {"score", "classifier"} or name.endswith(("score", "classifier")):
                candidates.append((name, mod))
    return candidates[-1][1] if candidates else None

def _locate_head_and_name(model: PreTrainedModel, task: str) -> Tuple[str, torch.nn.Module]:
    if task == "causal-lm":
        head = model.get_output_embeddings()
        if head is None:
            raise RuntimeError("Could not locate lm_head via get_output_embeddings().")
        return "lm_head", head
    elif task == "seq-cls":
        head = _find_seqcls_head_module(model)
        if head is None:
            raise RuntimeError("Could not locate classification head (score/classifier).")
        # Prefer 'score' label if present, else 'classifier'
        label = "score" if hasattr(model, "score") else "classifier"
        return label, head
    else:
        raise ValueError(f"Unknown task: {task}")

def _snapshot_head_hash(model: PreTrainedModel, task: str) -> Tuple[str, str]:
    name, head = _locate_head_and_name(model, task)
    if not hasattr(head, "weight"):
        raise RuntimeError(f"Head '{name}' has no '.weight' parameter.")
    return name, _sha256_tensor(head.weight)


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

    adapter_cfg = validate_adapter_folder(args.adapter_path)

    # --- Inspect adapter for lm_head/score/classifier coverage ---
    try:
        report = inspect_adapter_heads(args.adapter_path, adapter_cfg)
        print_adapter_head_report(report)
    except Exception as e:
        print(f"Warning: could not inspect adapter heads: {e}")


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

    # === Merge verification: snapshot head BEFORE merge ===
    try:
        head_name, pre_hash = _snapshot_head_hash(model, args.task)
        print(f"[verify] {head_name} hash BEFORE merge: {pre_hash}")
    except Exception as e:
        print(f"[verify] ⚠️ Could not snapshot head before merge: {e}")
        pre_hash = None

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

    print("-> Merging LoRA weights into base model ...")
    merged_model = peft_model.merge_and_unload(safe_merge=True)

    verify_adapter_embeddings_applied(merged_model, tokenizer, args.adapter_path)

    # === Merge verification: snapshot head AFTER merge and compare ===
    try:
        _, post_hash = _snapshot_head_hash(merged_model, args.task)
        print(f"[verify] {head_name} hash AFTER  merge: {post_hash}")
        if pre_hash is not None:
            if pre_hash != post_hash:
                print(f"[verify] ✅ {head_name} changed after merge (LoRA likely affected this head).")
            else:
                print(f"[verify] ⚠️ {head_name} unchanged after merge.")
    except Exception as e:
        print(f"[verify] ⚠️ Could not snapshot head after merge: {e}")


    if final_dtype != torch.float32:
        merged_model = merged_model.to(final_dtype)

    if args.task == "seq-cls":
        ensure_seqcls_head(merged_model, getattr(merged_model.config, "num_labels", None))

    if hasattr(merged_model.config, "use_cache"):
        merged_model.config.use_cache = True
    try:
        merged_model.config.torch_dtype = str(final_dtype).split(".")[-1]
    except Exception:
        pass

    os.makedirs(args.save_path, exist_ok=True)
    print(f"-> Saving merged model to: {args.save_path}")
    merged_model.save_pretrained(args.save_path, safe_serialization=True)
    tokenizer.save_pretrained(args.save_path)

    print("✅ Done. The directory now contains a standalone, merged model ready for further training.")


if __name__ == "__main__":
    sys.exit(main())
