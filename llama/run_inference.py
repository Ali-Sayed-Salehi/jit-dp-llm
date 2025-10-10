#!/usr/bin/env python3
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
from datetime import datetime
from typing import Optional, Tuple, Union

import numpy as np
import torch
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed,
    pipeline as hf_pipeline,
)
from transformers.pipelines import Pipeline
from peft import PeftModel, PeftConfig
from tqdm.auto import tqdm

from utils import (
    _compute_metrics_core
)
from utils import *

# =========================
# Shared helpers
# =========================

def _is_peft_adapter(path: str) -> bool:
    try:
        _ = PeftConfig.from_pretrained(path, local_files_only=True)
        print("Path is a PEFT adapter.")
        return True
    except Exception:
        print("Path is a full model (not an adapter).")
        return False


def _build_model_and_tokenizer_for_pipeline(
    model_or_adapter_path: str,
    base_model_path: Optional[str],
    model_kwargs: dict,
    local_files_only: bool = True,
) -> Tuple[Union[str, torch.nn.Module], AutoTokenizer, AutoConfig, bool]:
    """
    Returns (model_for_pipeline, tokenizer, config, used_adapter) for SEQ_CLS pipeline usage.
    If adapter: attaches to base model and returns a model object (ready for pipeline).
    Else: returns the model path (pipeline will lazy load).
    """
    if _is_peft_adapter(model_or_adapter_path):
        if not base_model_path:
            raise ValueError("--base_model_path is required when --model_path points to a PEFT/LoRA adapter.")
        tokenizer = AutoTokenizer.from_pretrained(
            model_or_adapter_path, use_fast=True, local_files_only=local_files_only
        )

        config = AutoConfig.from_pretrained(base_model_path, local_files_only=local_files_only)

        base_model = AutoModelForSequenceClassification.from_pretrained(base_model_path, **model_kwargs)

        new_vocab = len(tokenizer)
        if getattr(base_model.config, "vocab_size", None) != new_vocab:
            base_model.resize_token_embeddings(new_vocab, mean_resizing=False, pad_to_multiple_of=8)
            base_model.config.vocab_size = new_vocab

        peft_model = PeftModel.from_pretrained(
            base_model,
            model_id=model_or_adapter_path,
            is_trainable=False,
            local_files_only=local_files_only,
        )
        peft_model.eval()
        return peft_model, tokenizer, base_model.config, True

    tokenizer = AutoTokenizer.from_pretrained(model_or_adapter_path, use_fast=True, local_files_only=local_files_only)
    config = AutoConfig.from_pretrained(model_or_adapter_path, local_files_only=local_files_only)
    return model_or_adapter_path, tokenizer, config, False


def _build_causal_model_and_tokenizer(
    model_or_adapter_path: str,
    base_model_path: Optional[str],
    model_kwargs: dict,
    local_files_only: bool = True,
) -> Tuple[torch.nn.Module, AutoTokenizer, AutoConfig, bool]:
    """
    Returns (model, tokenizer, config, used_adapter) for CAUSAL_LM inference.
    If adapter: attaches to base causal LM and returns a model object.
    Else: loads a full causal LM.
    """
    if _is_peft_adapter(model_or_adapter_path):
        if not base_model_path:
            raise ValueError("--base_model_path is required when --model_path points to a PEFT/LoRA adapter.")

        tokenizer = AutoTokenizer.from_pretrained(
            model_or_adapter_path, use_fast=True, local_files_only=local_files_only
        )
        config = AutoConfig.from_pretrained(base_model_path, local_files_only=local_files_only)

        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)

        new_vocab = len(tokenizer)
        if getattr(base_model.config, "vocab_size", None) != new_vocab:
            base_model.resize_token_embeddings(new_vocab, mean_resizing=False, pad_to_multiple_of=8)
            base_model.config.vocab_size = new_vocab

        peft_model = PeftModel.from_pretrained(
            base_model,
            model_id=model_or_adapter_path,
            is_trainable=False,
            local_files_only=local_files_only,
        )
        peft_model.eval()
        return peft_model, tokenizer, base_model.config, True

    tokenizer = AutoTokenizer.from_pretrained(model_or_adapter_path, use_fast=True, local_files_only=local_files_only)
    config = AutoConfig.from_pretrained(model_or_adapter_path, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(model_or_adapter_path, **model_kwargs)
    model.eval()

    new_vocab = len(tokenizer)
    if getattr(model.config, "vocab_size", None) != new_vocab:
        model.resize_token_embeddings(new_vocab, mean_resizing=False, pad_to_multiple_of=8)
        model.config.vocab_size = new_vocab

    return model, tokenizer, config, False

# =========================
# Custom HF pipeline for CLM→Seq-Cls
# =========================

class CLMSeqClsPipeline(Pipeline):
    """
    Custom pipeline that:
      - appends [/drs] to the text,
      - runs generate(max_new_tokens=1, output_scores=True),
      - returns 2-class logits for {zero_token, one_token}
        and whether the full-vocab argmax ∈ {ID0, ID1}.
    """
    def __init__(self, *args,
                 drs_token="[/drs]",
                 zero_token="0",
                 one_token="1",
                 strict_single_token=True,
                 truncation=True,
                 max_length=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.drs_token  = drs_token
        self.truncation = truncation
        self.max_length = max_length

        ids0 = self.tokenizer.encode(zero_token, add_special_tokens=False)
        ids1 = self.tokenizer.encode(one_token,  add_special_tokens=False)
        if strict_single_token and not (len(ids0) == 1 and len(ids1) == 1):
            raise ValueError(
                f"Expected single-token labels for {zero_token!r}/{one_token!r}. "
                f"Use single-piece tokens or pre-add dedicated tokens and pass them here."
            )
        self.ID0 = ids0[-1] if len(ids0) >= 1 else self.tokenizer.unk_token_id
        self.ID1 = ids1[-1] if len(ids1) >= 1 else self.tokenizer.unk_token_id

        drs_id = self.tokenizer.convert_tokens_to_ids(drs_token)
        if drs_id == self.tokenizer.unk_token_id:
            raise ValueError(
                f"{drs_token!r} not in tokenizer vocab. Add it (tokenizer.add_tokens(['{drs_token}'])) "
                "and resize embeddings before using this pipeline."
            )

    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, inputs):
        text = inputs.get("text", "") if isinstance(inputs, dict) else str(inputs)
        enc = self.tokenizer(
            text + self.drs_token,
            return_tensors="pt",
            padding=False,
            truncation=self.truncation,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        return enc

    def _forward(self, model_inputs):
        # With device_map='auto', don't forcibly move if model is sharded.
        if not hasattr(self.model, "hf_device_map"):
            model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}
        out = self.model.generate(
            **model_inputs,
            max_new_tokens=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        scores = out.scores[0]  # [B, V]
        return {"scores": scores}

    def postprocess(self, model_outputs):
        scores = model_outputs["scores"]            # [B, V]
        logits_2 = torch.stack([scores[:, self.ID0], scores[:, self.ID1]], dim=1)  # [B,2]
        top_ids = scores.argmax(dim=1)
        top_in_set = (top_ids == self.ID0) | (top_ids == self.ID1)
        return [
            {
                "logits": (float(logits_2[i, 0]), float(logits_2[i, 1])),
                "top_in_set": bool(top_in_set[i].item()),
            }
            for i in range(scores.shape[0])
        ]


# =========================
# Main entrypoint
# =========================

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
    clm_for_seq_cls: bool = False,
    zero_token: str = "0",
    one_token: str = "1",
    drs_token: str = "[/drs]",
    strict_single_token: bool = True,
    per_device_eval_batch_size: int = 1,
):
    """
    Inference on 'final_test'.

    - Default path: SEQUENCE CLASSIFICATION using HF pipeline (offline metrics via _compute_metrics_core).
    - If `clm_for_seq_cls=True`: uses a custom HF pipeline over a Causal LM that appends [/drs],
      runs generate(max_new_tokens=1, output_scores=True), reduces to 2-class logits, and computes offline metrics.

    Outputs:
      <output_dir>/eval_<timestamp>/final_test_results_{seq_cls|clm_seqcls}.json
    """
    os.makedirs(output_dir, exist_ok=True)
    out_eval_dir = os.path.join(output_dir, "eval_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(out_eval_dir, exist_ok=True)
    print("Output directory:", out_eval_dir)

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
    fmt_fn = determine_format_fn("clm", clm_for_seq_cls=True) if clm_for_seq_cls \
             else determine_format_fn("seq_cls", clm_for_seq_cls=False)

    dataset_dict: DatasetDict = load_and_split_dataset(
        dataset_path=dataset_path,
        repo_path=REPO_PATH,
        slurm_tmpdir=os.environ.get("TMPDIR", ""),
        debug=False,
        format_fn=fmt_fn,   # <- expected to yield 'text' and 'orig-labels' for clm_for_seq_cls
    )

    test_ds = dataset_dict["final_test"]
    print("Using final_test split with examples:", len(test_ds))

    if debug:
        n = min(200, len(test_ds))
        test_ds = test_ds.select(range(n))
        print("Debug mode ON: using only", n, "examples from final_test.")

    # ---- Model kwargs ----
    model_kwargs = dict(
        device_map="auto",
        torch_dtype=dtype if not quant_4bit else (torch.bfloat16 if dtype == torch.bfloat16 else torch.float32),
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    print("Model kwargs:", model_kwargs)
    if quant_4bit:
        print("Loading with 4-bit quantization for inference...")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )

    # ---- Build model/tokenizer/config ----
    if clm_for_seq_cls:
        # Build a Causal LM (full or adapter) for our custom pipeline
        model, tokenizer, config, used_adapter = _build_causal_model_and_tokenizer(
            model_or_adapter_path=model_path,
            base_model_path=base_model_path,
            model_kwargs=model_kwargs,
            local_files_only=True,
        )
    else:
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

    # Register custom config (BASE for adapter case)
    config_source = base_model_path if used_adapter else model_path
    register_custom_llama4_if_needed(config_source)

    recall_pcts_list = [float(x) for x in recall_pcts.split(",") if x.strip()]

    # =========================
    # Path A: CLM → Seq-Cls (custom pipeline) — run sequentially per sample
    # =========================
    if clm_for_seq_cls:

        # Ensure required cols exist
        cols = set(test_ds.column_names)
        if "orig-labels" not in cols:
            raise ValueError("Expected 'orig-labels' column; check determine_format_fn/load_and_split_dataset.")
        if "text" not in cols:
            raise ValueError("Expected 'text' column for pipeline input.")

        texts = test_ds["text"]
        y_true = np.array(test_ds["orig-labels"], dtype=np.int64)
        print("Texts:", len(texts), "Labels shape:", y_true.shape)

        # Make custom pipeline (device placement handled by model loaded with device_map='auto')
        pipe = CLMSeqClsPipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",  # label only; our subclass logic drives behavior
            drs_token=drs_token,
            zero_token=zero_token,
            one_token=one_token,
            strict_single_token=strict_single_token,
            truncation=should_trunc,
            max_length=tokenizer_max_len,
        )

        # Ensure Python list of strings for sequential calls
        texts_col = test_ds["text"]
        if isinstance(texts_col, list):
            texts = texts_col
        elif hasattr(texts_col, "to_pylist"):
            texts = texts_col.to_pylist()
        else:
            texts = list(texts_col)

        assert len(texts) > 0, "No texts!"
        assert isinstance(texts[0], str), f"Expected str, got {type(texts[0])}"

        # Sequential pipeline calls (like seq-cls path)
        outs = []
        for idx, text in enumerate(tqdm(texts, desc="Running CLM→seq-cls inference", unit="sample")):
            out = pipe(text)  # returns a list of length 1
            outs.append(out[0])
        print("Pipeline returned", len(outs), "items.")

        # Collect logits and diagnostics
        logits_2 = np.array([o["logits"] for o in outs], dtype=np.float32)   # [N,2]
        top_in_set_flags = np.array([o["top_in_set"] for o in outs], dtype=bool)

        metrics, preds, probs = _compute_metrics_core(
            logits_2,
            y_true,
            repo_root=REPO_PATH,
            threshold=threshold,
            percentages=recall_pcts_list,
            average="binary",
            recall_at_top_k_fn=recall_at_top_k,
        )
        metrics.update({
            "num_samples": int(len(y_true)),
            "valid_rate": float(top_in_set_flags.mean()),
            "num_invalid": int((~top_in_set_flags).sum()),
        })
        print("Metrics computed. num_samples:", metrics.get("num_samples"), "valid_rate:", metrics.get("valid_rate"))

        # Optional commit_id
        if "commit_id" in test_ds.column_names:
            commit_ids = list(test_ds["commit_id"])
        else:
            commit_ids = [None] * len(y_true)

        p_pos = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        p_neg = probs[:, 0] if probs.shape[1] > 1 else (1.0 - p_pos)  # safety if shape==1

        conf_pred = np.where(preds == 1, p_pos, p_neg)

        samples = []
        for i in range(len(y_true)):
            samples.append({
                "commit_id": commit_ids[i],
                "true_label": int(y_true[i]),
                "prediction": int(preds[i]),
                "confidence": float(conf_pred[i]),  # <- predicted-class probability
                "raw_top_in_{0,1}": bool(top_in_set_flags[i]),
            })
        print("Per-sample results built. Total:", len(samples))

        results = {
            "task_type": "clm_seq_cls",
            "debug": debug,
            "base_model_path": base_model_path,
            "model_path": model_path,
            "dataset_path": dataset_path,
            "quant": quant_4bit,
            "threshold": threshold,
            "used_samples": int(len(texts)),
            "metrics": metrics,
            "samples": samples,
            "zero_token": zero_token,
            "one_token": one_token,
            "drs_token": drs_token,
            "strict_single_token": strict_single_token,
        }

        out_json = os.path.join(out_eval_dir, "final_test_results_clm_seqcls.json")
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print("Saved inference results to:", out_json)
        print("Inference complete (CLM → seq-cls via custom pipeline).")
        return

    # =========================
    # Path B: Standard Seq-Cls (pipeline)
    # =========================
    print("Running final inference on held-out test set (seq-cls)...")

    clf = hf_pipeline(
        task="text-classification",
        model=model_for_pipeline,
        tokenizer=tokenizer,
        model_kwargs=model_kwargs if isinstance(model_for_pipeline, str) else {},
    )

    texts = test_ds["text"]
    labels = np.array(test_ds["labels"] if "labels" in test_ds.column_names else test_ds["label"], dtype=np.int64)
    print("Number of test texts:", len(texts), "labels shape:", labels.shape)

    # Class order in id order
    num_labels = int(getattr(clf.model.config, "num_labels", 2))
    id2label_raw = getattr(clf.model.config, "id2label", {i: f"LABEL_{i}" for i in range(num_labels)})
    ordered_labels = []
    for i in range(num_labels):
        ordered_labels.append(id2label_raw.get(i) or id2label_raw.get(str(i)) or f"LABEL_{i}")
    print("Ordered labels:", ordered_labels)

    # Inference with tqdm
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
    logits = np.log(np.clip(probs, 1e-12, 1.0))  # offline core expects logits-like; log-prob is fine

    print("Computing offline metrics...")
    metrics, preds, probs_again = _compute_metrics_core(
        logits,
        labels,
        repo_root=REPO_PATH,
        threshold=threshold,
        percentages=recall_pcts_list,
        average="binary",
        recall_at_top_k_fn=recall_at_top_k,
    )

    # Optional commit_id
    if "commit_id" in test_ds.column_names:
        commit_ids = list(test_ds["commit_id"])
    else:
        commit_ids = [None] * len(labels)

    # Per-sample results
    samples = []
    pos_conf = probs_again.max(axis=1)
    for i in range(len(labels)):
        pred = int(preds[i])
        conf = float(pos_conf[i])
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
        "threshold": threshold,
        "used_samples": int(len(texts)),
        "label_order": ordered_labels,
        "metrics": metrics,
        "samples": samples,
    }

    out_json = os.path.join(out_eval_dir, "final_test_results_seq_cls.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved inference results to:", out_json)
    print("Inference complete (seq-cls).")


def parse_args():
    p = argparse.ArgumentParser("Run inference on held-out last 10% split (final_test).")
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

    # New flags for CLM → seq-cls
    p.add_argument("--clm_for_seq_cls", action="store_true", help="Use CLM-as-seq-cls inference path.")
    p.add_argument("--zero_token", type=str, default="0", help="Label token string for class 0.")
    p.add_argument("--one_token", type=str, default="1", help="Label token string for class 1.")
    p.add_argument("--drs_token", type=str, default="[/drs]", help="Delimiter token appended before the label.")
    p.add_argument("--strict_single_token", action="store_true", default=True,
                   help="Require label strings to encode to single tokens (else error).")
    p.add_argument("--per_device_eval_batch_size", type=int, default=1,
                   help="Batch size for CLM custom pipeline (generate).")

    a = p.parse_args()

    print("Effective arguments:")
    pprint.pprint(vars(a), sort_dicts=False)

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
        clm_for_seq_cls=args.clm_for_seq_cls,
        zero_token=args.zero_token,
        one_token=args.one_token,
        drs_token=args.drs_token,
        strict_single_token=args.strict_single_token,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
    )
