import os
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv
import yaml
from collections import Counter
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import pprint

from huggingface_hub import login as huggingface_hub_login

from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset

import dataclasses
from typing import Optional

from accelerate.utils import is_peft_model
from packaging import version
from transformers.utils import is_peft_available

if is_peft_available():
    import peft
    from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from sklearn.metrics import roc_auc_score, average_precision_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    TrainerCallback,
    PreTrainedModel,
    TrainingArguments
)

import evaluate
from attach_classification_head import CustomLlama4ForSequenceClassification, CustomLlama4TextConfig


def determine_tokenizer_truncation(
    tokenizer,
    config,
    truncation_len=None,
    chunking_len=None
):

    should_truncate = False
    tokenizer_max_len = None

    model_max_len = min(tokenizer.model_max_length, config.max_position_embeddings)

    print(f"✂️ User_specified truncation len: {truncation_len}, chunking len: ({chunking_len}), model max len: {model_max_len}")

    if not truncation_len and not chunking_len:
        should_truncate = True
        tokenizer_max_len = model_max_len
        print(f"No truncation or chunking specified. Tokenizer will truncate to model_max_len ({tokenizer_max_len})")

    elif truncation_len and not chunking_len:
        should_truncate = True
        tokenizer_max_len = min(model_max_len, truncation_len)
        print(f"Tokenizer will truncate to min(model_max_len, truncation_len) ({tokenizer_max_len}). No chunking used")

    elif truncation_len and chunking_len:
        should_truncate = True
        user_specified_max_len = min(truncation_len, chunking_len)

        if user_specified_max_len > model_max_len:
            tokenizer_max_len = model_max_len
            print(f"user_specified_max_len ({user_specified_max_len}) > model_max_len ({chunking_len}). Tokenizer will truncate to model_max_len ({tokenizer_max_len})")
        else:
            tokenizer_max_len = truncation_len
            print(f"Tokenizer will truncate to truncation_len ({tokenizer_max_len})")

    elif not truncation_len and chunking_len:
        if chunking_len > model_max_len:
            should_truncate = True
            tokenizer_max_len = model_max_len
            print(f"chunking_len ({chunking_len}) > model_max_len ({model_max_len}). Tokenizer will truncate to model_max_len ({tokenizer_max_len})")
        else:
            should_truncate = False
            tokenizer_max_len = None
            print(f"chunking_len ({chunking_len}) used. No truncation.")
    
    return should_truncate, tokenizer_max_len


def login_to_huggingface(repo_path: str, env_path: str = "secrets/.env"):
    """
    Loads environment variables and logs into Hugging Face using the token in .env file.
    
    Args:
        repo_path (str): Base path to your repository.
        env_path (str): Relative path to the .env file from the repo path.
    """

    dotenv_file = os.path.join(repo_path, env_path)
    load_dotenv(dotenv_path=dotenv_file)
    
    token = os.getenv("HUGGING_FACE_TOKEN")
    if not token:
        raise ValueError("🚫 HUGGING_FACE_TOKEN not found in environment variables")
    
    huggingface_hub_login(token)
    print("✅ Logged in to Hugging Face.")


def parse_training_args():
    # First, parse only the config file argument
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None, help="Path to YAML config file with default arguments")
    known_args, _ = pre_parser.parse_known_args()

    # Load config file values
    defaults = {}
    if known_args.config:
        if not os.path.isfile(known_args.config):
            raise FileNotFoundError(f"Config file not found: {known_args.config}")
        with open(known_args.config, "r") as f:
            defaults = yaml.safe_load(f) or {}

    # Full parser with defaults from config
    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.set_defaults(**defaults)

    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--live_metrics", action="store_true", help="Enable saving evaluation metrics after each eval step")
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Choose which dataset to use by specifying its absolute path. Default is imdb for sequence classification and eli5 for causal lm."
    )
    parser.add_argument("--model_path", type=str, help="Full path to the local pretrained model directory")
    parser.add_argument(
        "--class_imbalance_fix",
        type=str,
        choices=["oversampling", "weighted_loss", "focal_loss", "undersampling" ,"none"],
        help="Class imbalance handling method. Options: oversampling, weighted_loss, focal_loss, undersampling ,none"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Optional decision threshold for classifying as class 1 (between 0 and 1). If not set, uses argmax."
    )
    parser.add_argument("--quant", action="store_true", help="Enable quantization with BitsAndBytesConfig")
    parser.add_argument("--lora", action="store_true", help="Enable LoRA fine-tuning using PEFT")
    parser.add_argument("--flash_attn_2", action="store_true", help="Use flash attention 2")
    parser.add_argument("--mixed_precision", type=str,
        choices=["fp32", "fp16", "bf16"],
        help="Choose one of: fp32, fp16, bf16"
    )
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, help="weight_decay")
    parser.add_argument("--num_train_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--resampling_ratio", type=float, help="float in (0, 1]: desired minority/majority ratio AFTER resampling (binary)")
    parser.add_argument("--lora_r", type=int, help="lora_r")
    parser.add_argument("--lora_alpha", type=int, help="lora_alpha")
    parser.add_argument("--lora_dropout", type=float, help="lora_dropout")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="gradient_accumulation_steps")
    parser.add_argument("--eval_steps", type=int, help="Run evaluation every x steps")
    parser.add_argument("--logging_steps", type=int, help="Print training metrics every x steps")
    parser.add_argument("--save_steps", type=int, help="Save the model every x steps")
    parser.add_argument("--lr_warmup_ratio", type=float, help="Learning rate warmup ratio")
    parser.add_argument("--train_batch_size", type=int, help="per device training batch size")
    parser.add_argument(
        "--continue_from_dir", 
        type=str, 
        help="""Resume training from this checkpoint directory. 
        Example: '--continue_from_dir /speed-scratch/a_s87063/repos/perf-pilot/llama/training/run_2025-06-10_20-42-03/output'"""
    )
    parser.add_argument(
        "--selection_metric",
        type=str,
        help="Metric to select the best model: recall@top_5%, recall@top_10%, recall@top_30%, f1, precision, recall, accuracy. Default is recall@top_5%"
    )
    parser.add_argument(
        "--truncation_len",
        type=int,
        help="Optional. The length to which the sequences should be truncated using the tokenizer to reduce the size of the dataset."
    )
    parser.add_argument(
        "--chunking_len",
        type=int,
        help="Optional. The length to which the sequences should be chunked in order to reduce sequence length for the model."
    )
    parser.add_argument(
        "--task_type",
        type=str,
        choices=["seq_cls", "clm"],
        help="sequence classification or causal language modelling."
    )
    parser.add_argument("--add_new_tokens", action="store_true", help="Whether to add new tokens like <FILE>")
    parser.add_argument(
        "--slurm_tmpdir_env",
        type=str,
        help="Environment variable that point to slurm temporary directory"
    )

    args = parser.parse_args()

    # --- Path normalization ---
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    for key in ["dataset_path", "model_path", "continue_from_dir"]:
        path_val = getattr(args, key, None)
        if path_val and path_val != "imdb":
            if os.path.isabs(path_val):
                # Leave absolute paths untouched
                abs_path = path_val
            else:
                # Join relative paths to repo_root without resolving symlinks
                abs_path = os.path.normpath(os.path.join(repo_root, path_val))
                # Strip "/nfs" if it's just a symlinked prefix
                if abs_path.startswith("/nfs/"):
                    abs_path = abs_path.replace("/nfs", "", 1)
            setattr(args, key, abs_path)

    # --- Validation Section ---
    if args.threshold is not None and not (0.0 <= args.threshold <= 1.0):
        raise ValueError("Threshold must be between 0 and 1 if specified")

    VALID_SELECTION_METRICS = {
        "recall@top_5%", "recall@top_10%", "recall@top_30%",
        "f1", "precision", "recall", "accuracy", None
    }
    if args.selection_metric not in VALID_SELECTION_METRICS:
        raise ValueError(f"Unsupported selection_metric '{args.selection_metric}'. Must be one of {VALID_SELECTION_METRICS}")

    if args.truncation_len and args.chunking_len and args.truncation_len < args.chunking_len:
        raise ValueError(f"truncation_len ({args.truncation_len}) cannot be less than chunking_len ({args.chunking_len})")

    # --- Print effective args ---
    print("\n===== Effective Training Arguments =====")
    pprint.pprint(vars(args), sort_dicts=False)
    print("=======================================\n")

    return args


def setup_training_directories(repo_root, slurm_tmpdir, continue_from_dir=None):

    if continue_from_dir:
        output_dir = continue_from_dir
        run_timestamp = os.path.basename(os.path.dirname(output_dir)).split("_", 1)[-1]
        print(f"🔁 Resuming from checkpoint in: {output_dir}")
    else:
        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(repo_root, "llama", "training", f"run_{run_timestamp}", "output")

    # Handle SLURM_TMPDIR env variable
    if slurm_tmpdir in os.environ and os.environ[slurm_tmpdir].strip():
        slurm_tmpdir_path = os.environ[slurm_tmpdir]
        offload_dir = os.path.join(slurm_tmpdir_path, "offload")
    else:
        print(f"⚠️ Environment variable '{slurm_tmpdir}' not set.")
        slurm_tmpdir_path = None
        offload_dir = None

    base_run_dir = os.path.dirname(output_dir)
    tensorboard_dir = os.path.join(base_run_dir, "tensorboard")
    metrics_dir = os.path.join(base_run_dir, "metrics")
    model_dir = os.path.join(base_run_dir, "model")
    tokenizer_dir = os.path.join(base_run_dir, "tokenizer")
    config_path = os.path.join(metrics_dir, "config.json")
    live_metrics_path = os.path.join(metrics_dir, "live_metrics.jsonl")

    dirs_to_create = [output_dir, tensorboard_dir, metrics_dir, model_dir, tokenizer_dir]
    if offload_dir:
        dirs_to_create.append(offload_dir)
    if slurm_tmpdir_path:
        dirs_to_create.append(slurm_tmpdir_path)

    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)

    return {
        "output_dir": output_dir,
        "run_timestamp": run_timestamp,
        "tensorboard_dir": tensorboard_dir,
        "metrics_dir": metrics_dir,
        "config_path": config_path,
        "live_metrics_path": live_metrics_path,
        "model_dir": model_dir,
        "tokenizer_dir": tokenizer_dir,
        "all_dirs": dirs_to_create,
        "offload_dir": offload_dir,
        "slurm_tmpdir": slurm_tmpdir_path
    }


class SaveMetricsCallback(TrainerCallback):
    def __init__(self, output_path):
        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def _write_metrics(self, state, metrics, metric_type):
        if metrics is not None:
            with open(self.output_path, "a") as f:
                json.dump({
                    "step": state.global_step,
                    "type": metric_type,
                    "metrics": metrics
                }, f)
                f.write("\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and not any(k.startswith("eval_") for k in logs):
            self._write_metrics(state, logs, "train")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self._write_metrics(state, metrics, "eval")


def setup_live_metrics(live_metrics_enabled: bool, live_metrics_path: str):
    callbacks = []

    if live_metrics_enabled:
        callbacks.append(SaveMetricsCallback(live_metrics_path))
        print(f"📊 Live metrics will be saved to: {live_metrics_path}")
    else:
        print("📊 Live metrics logging disabled.")

    return callbacks


def load_and_split_dataset(
    dataset_path,
    repo_path,
    slurm_tmpdir,
    debug=False,
    seed=42,
    format_fn=None
):

    if dataset_path is None or str(dataset_path).strip().lower() == "imdb":
        print("🎬 Loading IMDb dataset from Hugging Face...")

        dataset = load_dataset("imdb")
        full_dataset = concatenate_datasets([dataset["train"], dataset["test"]])

        # Add length field using raw text length
        full_dataset = full_dataset.map(lambda ex: {"length": len(ex["text"])})

        # Sort and take top 7000
        full_dataset = full_dataset.sort("length", reverse=True)
        final_dataset = full_dataset.select(range(min(7000, len(full_dataset))))

    else:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"❌ Dataset file does not exist: {dataset_path}")

        dataset_copy_path = dataset_path  # default to original path

        # Only copy if slurm_tmpdir is provided
        if slurm_tmpdir and slurm_tmpdir.strip():
            src = Path(dataset_path).resolve()
            repo = Path(repo_path).resolve()

            # Decide how to place the file inside TMPDIR:
            # - If src is inside repo, keep its relative path under repo
            # - Otherwise, flatten (just copy filename into TMPDIR)
            try:
                src_relative_to_repo = src.relative_to(repo)
                rel_parent = src_relative_to_repo.parent  # e.g., datasets/apachejit
            except ValueError:
                # Not under repo; don't try to recreate the whole absolute path
                rel_parent = Path("")

            dest_dir = Path(slurm_tmpdir) / rel_parent
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / src.name

            try:
                # shutil.copy2 is enough here; rsync isn't necessary for single files
                shutil.copy2(src, dest_file)
                print(f"✅ Copied {src} → {dest_file}")
                dataset_copy_path = str(dest_file)
            except OSError as e:
                print(f"⚠️ Could not copy dataset to SLURM tmpdir: {e}")
                print(f"🔄 Falling back to using original dataset path: {dataset_path}")
        else:
            print("⚠️ No SLURM tmpdir provided — using dataset in place.")

        final_dataset = load_dataset("json", data_files=dataset_copy_path, split="train")

    # Split 80% train, 10% eval, 10% test
    n_total = len(final_dataset)
    n_train = int(n_total * 0.8)
    n_eval  = int(n_total * 0.1)

    train_dataset = final_dataset.select(range(0, n_train))
    eval_dataset  = final_dataset.select(range(n_train, n_train + n_eval))
    test_dataset  = final_dataset.select(range(n_train + n_eval, n_total))

    if debug:
        train_dataset = train_dataset.select(range(min(200, len(train_dataset))))
        eval_dataset  = eval_dataset.select(range(min(100, len(eval_dataset))))
        test_dataset  = test_dataset.select(range(min(100, len(test_dataset))))

    train_dataset = train_dataset.shuffle(seed=seed)

    # Apply formatter if provided
    if format_fn is not None:
        train_dataset = train_dataset.map(format_fn, remove_columns=train_dataset.column_names)
        eval_dataset  = eval_dataset.map(format_fn, remove_columns=eval_dataset.column_names)
        # Keep the commit_id column for traceability
        test_dataset  = test_dataset.map(format_fn, remove_columns=["prompt", "response"])

    return DatasetDict({
        "train": train_dataset,
        "test": eval_dataset,
        "final_test": test_dataset
    })


def enable_gradient_checkpointing(
    model: PreTrainedModel, gradient_checkpointing_kwargs: Optional[dict]
) -> PreTrainedModel:
    """Enables gradient checkpointing for the model."""
    # Enable gradient checkpointing on the base model for PEFT
    if is_peft_model(model):
        model.base_model.gradient_checkpointing_enable()
    # Enable gradient checkpointing for non-PEFT models
    else:
        model.gradient_checkpointing_enable()

    gradient_checkpointing_kwargs = gradient_checkpointing_kwargs or {}
    use_reentrant = (
        "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
    )

    if use_reentrant:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    return model


def peft_module_casting_to_bf16(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm) or "norm" in name:
            module = module.to(torch.float32)
        elif any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe", "score"]):
            if hasattr(module, "weight"):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)


def prepare_peft_model(
    model: PreTrainedModel, peft_config: Optional["PeftConfig"], args: TrainingArguments
) -> PreTrainedModel:
    """Prepares a model for PEFT training."""
    if not is_peft_available():
        raise ImportError("PEFT is required to use a peft model. Run `pip install peft`.")

    # If the model is already a PeftModel, we need to merge and unload it.
    # Further information here: https://huggingface.co/docs/trl/dpo_trainer#reference-model-considerations-with-peft
    if isinstance(model, PeftModel) and peft_config is not None:
        model = model.merge_and_unload()

    # Handle quantized models (QLoRA)
    is_qlora = getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)

    is_sharded_qlora = False
    if getattr(model, "is_loaded_in_4bit", False):
        # Check if model is sharded (FSDP/DS-Zero3)
        for _, param in model.named_parameters():
            if param.__class__.__name__ == "Params4bit":
                is_sharded_qlora = param.data.device.type in {"cpu", "meta"}
                break

    # Prepare model for kbit training if needed
    if is_qlora and not is_sharded_qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs or {},
        )
        # Disable gradient checkpointing as it's handled by prepare_model_for_kbit_training
        args = dataclasses.replace(args, gradient_checkpointing=False)
    elif args.gradient_checkpointing:
        model = enable_gradient_checkpointing(model, args.gradient_checkpointing_kwargs)

    # Create PEFT model
    if peft_config is not None:
        if (
            version.parse(peft.__version__) >= version.parse("0.12")  # autocast_adapter_dtype introduced in 0.12
            and getattr(model, "is_loaded_in_4bit", False)
            and is_sharded_qlora
        ):
            model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)
        else:
            model = get_peft_model(model, peft_config)

    # Handle bf16 casting for 4-bit models
    if args.bf16 and getattr(model, "is_loaded_in_4bit", False) and not is_sharded_qlora:
        peft_module_casting_to_bf16(model)

    return model



def set_dtype(mixed_precision):
    """
    Set global dtype and precision flags based on mixed_precision argument.
    """
    if mixed_precision == "fp32":
        DTYPE = torch.float32
        USE_FP16 = False
        USE_BF16 = False

    elif mixed_precision == "fp16":
        DTYPE = torch.float16
        USE_FP16 = True
        USE_BF16 = False

    elif mixed_precision == "bf16":
        DTYPE = torch.bfloat16
        USE_FP16 = False
        USE_BF16 = True

    else:
        raise ValueError(f"Unknown precision type: {mixed_precision}")

    return DTYPE, USE_FP16, USE_BF16


def format_for_lm(example):
    if "text" in example: # imdb dataset
        return {"text": example["text"]}
    if "prompt" in example: # jit dataset
        return {"text": example["prompt"]}
    return {"text": ""}

def format_for_classification(example):
    if "text" in example and "label" in example:
        # IMDb case
        return {"text": example["text"], "labels": int(example["label"])}
    elif "prompt" in example and "response" in example:
        # Custom dataset case
        return {"text": example["prompt"], "labels": int(example["response"])}
    else:
        raise KeyError(f"Unrecognized example keys: {list(example.keys())}")

def determine_format_fn(task):
    format_funct = None

    if task == "seq_cls":
        format_funct = format_for_classification
    elif task == "clm":
        format_funct = format_for_lm
    else:
        raise ValueError(f"Unknown task type: {task}")
    
    return format_funct


def apply_class_imbalance_strategy(
    dataset: DatasetDict,
    strategy: str = "none",
    seed: int = 42,
    alpha: float = 0.25,
    gamma: float = 2.0,
    sampling_strategy=None,
    label_col: str = "labels"
):
    """
    sampling_strategy (used when strategy is 'oversampling' or 'undersampling'):

    - float in (0, 1]: desired minority/majority ratio AFTER resampling (binary).
      Example: 0.3 -> minority_count_after = 0.3 * majority_count_after (not fully balanced).

    - dict: {class_label: target_count_after} for exact per-class totals (multi-class or binary).
      Example: {0: 5000, 1: 2000}.

    - str: one of imbalanced-learn presets, e.g. 'minority', 'not minority', 'all', 'auto'.

    - callable: f(y) -> dict like above.

    Notes:
    - For oversampling, float must be in (0, 1]; use dict/callable if you want a class to exceed the majority.
    - For multiclass with floats, prefer dict/callable for full control.
    """

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    test_dataset = dataset["final_test"]

    before_dist = compute_class_distribution(dataset)
    print("📊 Class distribution before balancing:")
    for split, stats in before_dist.items():
        print(f"  {split}: {stats}")

    if strategy == "focal_loss":
        print("🔥 Using Focal Loss for class imbalance.")
        label_counts = Counter(train_dataset[label_col])
        alpha_dict = {k: alpha for k in label_counts.keys()}
        return dataset, None, {"alpha": alpha_dict, "gamma": gamma}, before_dist, before_dist

    elif strategy == "weighted_loss":
        print("🔄 Using Weighted Cross Entropy Loss.")
        label_counts = Counter(train_dataset[label_col])
        total = sum(label_counts.values())
        # weights[i] = total / count_i  (inverse-frequency)
        weights = [total / label_counts[i] for i in sorted(label_counts)]
        for i, weight in enumerate(weights):
            print(f"  Class {i} weight: {weight:.4f}")
        return dataset, weights, None, before_dist, before_dist

    elif strategy in {"oversampling", "undersampling"}:
        if sampling_strategy is None:
            # default behavior matches your previous code: fully balance classes
            sampling_strategy = "auto"  # imblearn will balance all classes

        print(f"{'📈' if strategy=='oversampling' else '📉'} Applying {strategy} with sampling_strategy={sampling_strategy!r}")

        df = pd.DataFrame(train_dataset)
        if label_col not in df.columns:
            raise KeyError(f"Label column '{label_col}' not found in the training dataset.")

        X = df.drop(columns=[label_col])
        y = df[label_col]

        if strategy == "oversampling":
            sampler = RandomOverSampler(random_state=seed, sampling_strategy=sampling_strategy)
        else:
            sampler = RandomUnderSampler(random_state=seed, sampling_strategy=sampling_strategy)

        X_resampled, y_resampled = sampler.fit_resample(X, y)

        resampled_df = X_resampled.copy()
        resampled_df[label_col] = y_resampled
        train_dataset_balanced = Dataset.from_pandas(resampled_df, preserve_index=False)

        balanced_dataset = DatasetDict({
            "train": train_dataset_balanced,
            "test": eval_dataset,
            "final_test": test_dataset
        })

        after_dist = compute_class_distribution(balanced_dataset)
        print(f"📊 Class distribution after {strategy}:")
        for split, stats in after_dist.items():
            print(f"  {split}: {stats}")

        return balanced_dataset, None, None, before_dist, after_dist

    elif strategy == "none":
        print("🟢 No class imbalance fix applied.")
        return dataset, None, None, before_dist, before_dist

    else:
        raise ValueError(f"Unsupported class imbalance strategy: {strategy}")



def compute_class_distribution(dataset_dict: DatasetDict) -> dict:
    distribution = {}

    for split_name, split_dataset in dataset_dict.items():
        labels = split_dataset["labels"]
        label_counts = Counter(labels)
        total = sum(label_counts.values())
        pos_count = label_counts.get(1, 0)

        counts = {
            str(label): int(count)
            for label, count in sorted(label_counts.items())
        }
        pos_pct = (pos_count / total * 100) if total > 0 else 0.0

        distribution[split_name] = {
            "counts": counts,
            "positive_percentage": round(pos_pct, 2)
        }

    return distribution



class FocalLoss(nn.Module):
    """
    Implementation of the Focal Loss function for classification tasks.
    """

    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        num_classes = logits.size(-1)

        # Match dtype/device of logits
        dtype = logits.dtype
        device = logits.device

        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)

        # One-hot encode targets in the same dtype/device as logits
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).to(dtype=dtype, device=device)

        # Gather probabilities of the true class
        pt = (probs * targets_one_hot).sum(dim=1)

        # Handle alpha per class if it's a dict
        if isinstance(self.alpha, dict):
            alpha_tensor = torch.tensor(
                [self.alpha.get(str(i), 1.0) for i in range(num_classes)],
                dtype=dtype,   # match dtype of logits
                device=device  # match device of logits
            )
            alpha = (alpha_tensor * targets_one_hot).sum(dim=1)
        else:
            # Cast scalar alpha to same dtype/device
            alpha = torch.tensor(self.alpha, dtype=dtype, device=device)

        focal_term = (1 - pt) ** self.gamma
        loss = -alpha * focal_term * log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def determine_data_collator(task, tokenizer):
    collator = None

    if task == "seq_cls":
        collator = DataCollatorWithPadding(tokenizer=tokenizer)
    elif task == "clm":
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
    elif task == "mlm":
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=True, 
            mlm_probability=0.15
        )
    else:
        raise ValueError(f"Unknown task type: {task}")
    
    return collator



def compute_custom_metrics_seq_cls(eval_pred, threshold=None, percentages=None):
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    logits, labels = eval_pred

    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

    if percentages is None:
        raise ValueError("You must specify percentages for recall_at_top_k.")

    recall_at_k_metrics = recall_at_top_k(probs[:, 1], labels, percentages=percentages)

    if threshold is not None:
        predictions = (probs[:, 1] >= threshold).astype(int)
    else:
        predictions = np.argmax(logits, axis=1)

    # Prediction distribution
    class_counts = Counter(predictions)
    output_distribution = {
        f"pred_class_{label}": int(count)
        for label, count in sorted(class_counts.items())
    }

    try:
        roc_auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(labels, probs[:, 1])
    except ValueError:
        pr_auc = float("nan")

    metrics = {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "precision": precision.compute(predictions=predictions, references=labels, average="binary")["precision"],
        "recall": recall.compute(predictions=predictions, references=labels, average="binary")["recall"],
        "f1": f1.compute(predictions=predictions, references=labels, average="binary")["f1"],
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }

    metrics.update(output_distribution)
    metrics.update(recall_at_k_metrics)

    return metrics


def recall_at_top_k(pred_scores, true_labels, percentages=None):
    if percentages is None:
        raise ValueError("You must specify percentages for recall_at_top_k.")

    results = {}
    total_positives = np.sum(true_labels)

    sorted_indices = np.argsort(-pred_scores)
    sorted_labels = true_labels[sorted_indices]

    for pct in percentages:
        k = int(len(pred_scores) * pct)
        top_k_labels = sorted_labels[:k]
        recall_val = np.sum(top_k_labels) / total_positives
        results[f"recall@top_{int(pct * 100)}%"] = recall_val
    return results


def save_training_config(
    config_path,
    run_timestamp,
    args,
    training_args,
    class_distribution,
    original_class_distribution,
    truncation_len,
    chunking_len,
    dtype,
    task,
    dataset=None,
    RECALL_AT_TOP_K_PERCENTAGES=None,
    FL_GAMMA=None,
    FL_ALPHA=None,
    model_config=None
):
    # ------------------ Compute held-out defect rate + max recall@top_k ------------------
    defect_rate = None
    max_recall_at_k = {}

    if dataset is not None and task == "seq_cls":
        if "final_test" in dataset:
            true_labels_test_set = np.array(dataset["final_test"]["labels"])
            
            defect_rate = compute_defect_rate(true_labels_test_set)

            if RECALL_AT_TOP_K_PERCENTAGES:
                max_recall_at_k = compute_max_recall_at_top_k(true_labels_test_set, RECALL_AT_TOP_K_PERCENTAGES)

        else:
            print("⚠️ DatasetDict does not contain 'final_test' split — skipping defect rate and max recall@top_k.")

    # ------------------ Build snapshot ------------------
    config_snapshot = {
        "timestamp": run_timestamp,
        "script arguments": vars(args),
        "class_distribution": class_distribution,
        "original_class_distribution": original_class_distribution,
        "decision_threshold": args.threshold if args.threshold is not None else "argmax",
        "focal_loss_gamma": FL_GAMMA if args.class_imbalance_fix == "focal_loss" else "None",
        "focal_loss_alpha": FL_ALPHA if args.class_imbalance_fix == "focal_loss" else "None",
        "dtype": str(dtype),
        "effective_truncation_len": truncation_len,
        "chunking_len": chunking_len,
        "max_possible_recall@top_k": max_recall_at_k,
        "training_args_full": training_args.to_dict()
    }

    if model_config is not None:
        config_snapshot["model_config"] = model_config

    # ------------------ Save ------------------
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config_snapshot, f, indent=2)

    print(f"⚙️ Logged config to: {config_path}")


def compute_max_recall_at_top_k(true_labels, percentages):
    """
    Computes the maximum achievable recall@top_k% for a given label distribution.
    """
    results = {}
    total_positives = np.sum(true_labels)
    n = len(true_labels)

    for pct in percentages:
        k = int(n * pct)
        max_recall = min(k, total_positives) / total_positives if total_positives > 0 else 0.0
        results[f"max_recall@top_{int(pct * 100)}%"] = max_recall
    return results

def compute_defect_rate(labels):
    """
    Computes the defect rate: the fraction of positive class examples in a label array.
    
    Args:
        labels (list or np.ndarray): Array of binary labels (0 or 1).
    
    Returns:
        float: Ratio of positive examples to total examples.
    """
    labels = np.array(labels)
    if len(labels) == 0:
        return 0.0
    return np.sum(labels) / len(labels)


def register_custom_llama4_if_needed(model_path: str):
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found at: {config_path}")

    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    model_type = config_data.get("model_type", "")
    architectures = config_data.get("architectures", [])

    if model_type == CustomLlama4TextConfig.model_type:
        AutoConfig.register(CustomLlama4TextConfig.model_type, CustomLlama4TextConfig)
        AutoModelForSequenceClassification.register(CustomLlama4TextConfig, CustomLlama4ForSequenceClassification)
        print(f"✅ Registered custom LLaMA: model_type={model_type}, architectures={architectures}")
    else:
        print(f"ℹ️ Skipped custom registration: model_type={model_type}")


class CustomTrainer(Trainer):
    """
    Custom Trainer with class imbalance handling (bf16-only).
    """

    def __init__(self, *args, class_weights=None, focal_loss_fct=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Store class weights directly in bf16
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None else None
        )
        self.focal_loss_fct = focal_loss_fct

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").long()
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.focal_loss_fct:
            # labels = labels.to(device=logits.device)
            loss = self.focal_loss_fct(logits, labels)

        elif self.class_weights is not None:
            # Just move to device, dtype already bf16
            class_weights = self.class_weights.to(device=logits.device)
            loss = F.cross_entropy(logits, labels, weight=class_weights)

        else:
            loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss

def save_training_metrics(trainer, metrics_dir, filename="metrics.json"):
    metrics_save_path = os.path.join(metrics_dir, filename)
    training_metrics = trainer.state.log_history

    os.makedirs(metrics_dir, exist_ok=True)
    with open(metrics_save_path, "w") as f:
        json.dump(training_metrics, f, indent=4)

    print(f"✅ Saved metrics to {metrics_save_path}")
    return metrics_save_path


def run_final_inference(
    trainer,
    test_dataset,
    metrics_dir,
    percentages,
    threshold=None,
):

    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    print("\n🧪 Running final inference on held-out test set...")

    # --- 1) Preserve commit_id order, then drop it before predict ---
    commit_ids = None
    if hasattr(test_dataset, "column_names") and "commit_id" in test_dataset.column_names:
        commit_ids = test_dataset["commit_id"]
        ds_for_pred = test_dataset.remove_columns(["commit_id"])
    else:
        ds_for_pred = test_dataset

    # --- 2) Predict ---
    test_results = trainer.predict(ds_for_pred)
    logits = test_results.predictions
    labels = test_results.label_ids

    # --- 3) Compute predictions & metrics ---
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(probs, axis=1)

    if threshold is not None:
        preds = (probs[:, 1] >= threshold).astype(int)

    recall_at_k = recall_at_top_k(probs[:, 1], labels, percentages)

    try:
        final_roc_auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        final_roc_auc = float("nan")

    try:
        final_pr_auc = average_precision_score(labels, probs[:, 1])
    except ValueError:
        final_pr_auc = float("nan")

    final_metrics = {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "precision": precision.compute(predictions=preds, references=labels, average="binary")["precision"],
        "recall": recall.compute(predictions=preds, references=labels, average="binary")["recall"],
        "f1": f1.compute(predictions=preds, references=labels, average="binary")["f1"],
        "roc_auc": final_roc_auc,
        "pr_auc": final_pr_auc
    }
    final_metrics.update(recall_at_k)

    # --- 4) Build per-sample results ---
    results = []
    for i in range(len(labels)):
        results.append({
            "commit_id": commit_ids[i] if commit_ids is not None else None,
            "true_label": int(labels[i]),
            "prediction": int(preds[i]),
            "confidence": float(probs[i, 1])  # probability for positive class
        })

    # --- 5) Save everything ---
    output_payload = {
        "metrics": final_metrics,
        "results": results
    }

    os.makedirs(metrics_dir, exist_ok=True)
    final_test_metrics_path = os.path.join(metrics_dir, "final_test_results.json")
    with open(final_test_metrics_path, "w") as f:
        json.dump(output_payload, f, indent=4)

    print(f"📄 Final test set results saved to: {final_test_metrics_path}")
    print(json.dumps(final_metrics, indent=4))


def chunk_long_samples(
    dataset_dict,
    max_seq_length,
    overlap_pct=0.0
):
    if not (0.0 <= overlap_pct < 1.0):
        raise ValueError(f"overlap_pct must be in [0, 1), got {overlap_pct}")

    stride = int(max_seq_length * (1.0 - overlap_pct))
    if stride <= 0:
        raise ValueError(f"Calculated stride must be > 0, got stride={stride}")

    print(f"📏 Chunking long samples using max_seq_length={max_seq_length}, "
          f"overlap_pct={overlap_pct:.2f}, stride={stride}")

    chunked_dataset = DatasetDict()

    for split in dataset_dict:
        before_count = len(dataset_dict[split])
        print(f"🔍 Split '{split}': {before_count} samples before chunking...")

        input_ids_chunks = []
        attention_mask_chunks = []
        labels_chunks = []

        for example in dataset_dict[split]:
            tokens = example["input_ids"]
            attn_mask = example["attention_mask"]
            L = len(tokens)

            if L <= max_seq_length:
                input_ids_chunks.append(tokens)
                attention_mask_chunks.append(attn_mask)
                labels_chunks.append(tokens.copy())
            else:
                for i in range(0, L, stride):
                    end = i + max_seq_length
                    chunk = tokens[i:end]
                    chunk_mask = attn_mask[i:end]

                    if len(chunk) > max_seq_length or len(chunk_mask) > max_seq_length:
                        raise AssertionError(f"Chunk or mask length > max_seq_length.")

                    if len(chunk) != len(chunk_mask):
                        raise AssertionError(f"Chunk and mask lengths should match.")

                    input_ids_chunks.append(chunk)
                    attention_mask_chunks.append(chunk_mask)
                    labels_chunks.append(chunk.copy())

        after_count = len(input_ids_chunks)
        print(f"✅ Split '{split}': {after_count} samples after chunking.")

        chunked_dataset[split] = Dataset.from_dict({
            "input_ids": input_ids_chunks,
            "attention_mask": attention_mask_chunks,
            # "labels": labels_chunks
        })

    return chunked_dataset



def add_or_detect_special_tokens(tokenizer, model, task: str, add_new_tokens, use_lora: bool):
    """
    Ensures SPECIAL_TOKENS exist in the tokenizer and the model has the right embedding size.
    Returns:
        info: dict with diagnostics
    """

    SPECIAL_TOKENS = [
        "<COMMIT_MESSAGE>", "</COMMIT_MESSAGE>",
        "<FILE>", "</FILE>",
        "<ADDED>", "</ADDED>",
        "<REMOVED>", "</REMOVED>",
    ]

    # What does the tokenizer already have?
    existing = set(tokenizer.get_added_vocab().keys())
    to_add = [t for t in SPECIAL_TOKENS if t not in existing]

    info = {
        "already_present": sorted(list(existing.intersection(SPECIAL_TOKENS))),
        "added_now": [],
        "resized_embeddings": False,
        "modules_to_save_update": None,
    }

    if not add_new_tokens:
        return info

    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
        info["added_now"] = to_add
        print(f"➕ Added {len(to_add)} special tokens: {to_add}")
    else:
        print("ℹ️ All special tokens already present in tokenizer.")

    added_now = to_add  # list of token strings we added
    added_token_ids = tokenizer.convert_tokens_to_ids(added_now)
    info["added_token_ids"] = added_token_ids 

    # Always ensure PAD is set (we use EOS as PAD)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    # If tokenizer grew, make sure model matches
    model_embed = model.get_input_embeddings()
    if model_embed.num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8, mean_resizing=False)
        info["resized_embeddings"] = True
        print(f"🧩 Resized model embeddings to {len(tokenizer)} to match tokenizer.")

    # LoRA + CLM: if we just added tokens, make sure LM head gets saved with adapters
    # (we set this at LoraConfig time, but we expose a hint so caller can set modules_to_save=['lm_head'])
    if use_lora and task == "clm" and info["added_now"]:
        info["modules_to_save_update"] = True
        print("📎 Detected new tokens with LoRA in CLM. Recommend modules_to_save += ['lm_head'].")

    # Small debug dump
    print("🔎 Special tokens status:", info)
    return info



def resolve_tokenizer_dir(model_path: str, continue_from_dir: str | None) -> str:
    """
    Resolve the directory to load the tokenizer from.
    
    - Defaults to model_path.
    - If continue_from_dir is given, looks for the first checkpoint-* subdirectory.
    """
    tokenizer_load_dir = model_path
    if continue_from_dir and os.path.isdir(continue_from_dir):
        for d in os.listdir(continue_from_dir):
            if d.startswith("checkpoint-"):
                tokenizer_load_dir = os.path.join(continue_from_dir, d)
                break
    print(f"-> Loading tokenizer from {tokenizer_load_dir}")
    return tokenizer_load_dir


def infer_lora_target_modules(model):
    """
    Heuristically select LoRA targets across common encoders/decoders.

    Covers:
    - LLaMA/NeoX/StarCoder/CodeGen/GPT-J/ GPT-2: q_proj,k_proj,v_proj,o_proj,c_attn,c_proj,up/down/gate_proj,c_fc
    - BERT/RoBERTa/CodeBERT: self.query,self.key,self.value, intermediate.dense, output.dense
    Fallback: 'all-linear'.
    """
    # Substrings to probe in module names (ordered by commonality)
    cand_substrings = [
        # decoder-style attention
        "q_proj", "k_proj", "v_proj", "o_proj",
        "c_attn", "c_proj", "attn.c_attn", "attn.c_proj",
        # decoder-style MLPs
        "up_proj", "down_proj", "gate_proj", "c_fc", "mlp.c_fc", "mlp.c_proj",

        # encoder-style attention (BERT/RoBERTa/CodeBERT)
        "self.query", "self.key", "self.value",

        # encoder-style MLPs
        "intermediate.dense", "output.dense",
    ]

    found = set()
    for name, module in model.named_modules():
        # PEFT matches substrings on module/param names; we just collect the substrings that appear.
        # Include Linear and Conv1D (GPT-2 uses Conv1D wrapper).
        cls = module.__class__.__name__
        if cls not in {"Linear", "Conv1D"}:
            continue
        for s in cand_substrings:
            if s in name:
                found.add(s)

    if not found:
        return "all-linear"

    # Normalize to a compact, deduped target list PEFT can match by substring
    priority_order = [
        "q_proj","k_proj","v_proj","o_proj","c_attn","c_proj",
        "up_proj","down_proj","gate_proj","c_fc",
        "self.query","self.key","self.value",
        "intermediate.dense","output.dense",
    ]
    targets = [s for s in priority_order if any(s in f for f in found)]
    return targets or "all-linear"
