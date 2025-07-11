import os
import json
import argparse
from datetime import datetime
from collections import Counter

import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.nn import functional as F
from dotenv import load_dotenv

from datasets import load_dataset, Dataset, DatasetDict
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
from huggingface_hub import login as huggingface_hub_login

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    TrainerCallback,
    Trainer
)

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)

import evaluate


def parse_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--live_metrics", action="store_true", help="Enable saving evaluation metrics after each eval step")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["perf", "jit", "jit_balanced", "eli5", "jit_small", "jit_small_struc_ast_meta"],
        default="eli5",
        help="Choose which dataset to use"
    )
    parser.add_argument("--model_path", type=str, required=True, help="Full path to the local pretrained model directory")
    parser.add_argument("--quant", action="store_true", help="Enable quantization with BitsAndBytesConfig")
    parser.add_argument("--lora", action="store_true", help="Enable LoRA fine-tuning using PEFT")
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16 training")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument(
        "--continue_from_dir", 
        type=str, 
        help="""Resume training from this checkpoint directory. 
        Example: '--continue_from_dir /speed-scratch/a_s87063/repos/perf-pilot/llama/training/run_2025-06-10_20-42-03/output'"""
        )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="Optional. If set, overrides the estimated max sequence length."
    )
    parser.add_argument(
        "--sequence_length_fix",
        choices=["truncate", "chunk"],
        default="truncate",
        help="How to handle inputs longer than max_seq_length: "
            "'truncate' (tokenizer handles it) or 'chunk' (split into overlapping windows). "
            "Default: truncate"
    )
    
    args = parser.parse_args()
    
    return args


def setup_training_directories(repo_root, continue_from_dir=None):
    if continue_from_dir:
        output_dir = continue_from_dir
        run_timestamp = os.path.basename(os.path.dirname(output_dir)).split("_", 1)[-1]
        print(f"🔁 Resuming from checkpoint in: {output_dir}")
    else:
        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(repo_root, "llama", "training", f"run_{run_timestamp}", "output")

    base_run_dir = os.path.dirname(output_dir)
    tensorboard_dir = os.path.join(base_run_dir, "tensorboard")
    metrics_dir = os.path.join(base_run_dir, "metrics")
    model_dir = os.path.join(base_run_dir, "model")
    tokenizer_dir = os.path.join(base_run_dir, "tokenizer")
    config_path = os.path.join(metrics_dir, "config.json")
    live_metrics_path = os.path.join(metrics_dir, "live_metrics.jsonl")

    dirs_to_create = [output_dir, tensorboard_dir, metrics_dir, model_dir, tokenizer_dir]
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
    }


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


def load_and_split_dataset(dataset_name, repo_path, debug=False, seed=42):
    """
    Loads and splits a dataset for training, evaluation, and final testing.

    This function supports both local JSONL datasets and the Hugging Face ELI5 datasets.
    For local datasets and ELI5, it performs a chronological split: 64% for training, 16% for evaluation,
    and 20% for testing.

    Args:
        dataset_name (str): Name of the dataset to load. Supported values:
            - "perf": Performance regression dataset.
            - "jit": ApacheJIT dataset for LLMs.
            - "jit_balanced": Balanced version of the ApacheJIT dataset.
            - "jit_small": A smaller ApacheJIT dataset.
            - "jit_small_struc_ast_meta": Structural ApacheJIT dataset with AST/meta info.
            - "eli5": Loads eli5_category dataset from Hugging Face datasets library.
        repo_path (str): Path to the root of the repository where local datasets are stored.
        debug (bool, optional): If True, reduces the dataset size for faster experimentation. Default is False.
        seed (int, optional): Random seed for shuffling the training dataset. Default is 42.

    Returns:
        DatasetDict: A dictionary with Hugging Face `Dataset` objects:
            - "train": Training split
            - "test": Evaluation split
            - "final_test": Held-out test split
    """
    dataset_file_map = {
        "perf": "dataset.jsonl",
        "jit": "jit_dp/apachejit_llm.jsonl",
        "jit_balanced": "jit_dp/apachejit_llm_balanced.jsonl",
        "jit_small": "jit_dp/apachejit_llm_small.jsonl",
        "jit_small_struc_ast_meta": "jit_dp/apachejit_llm_small_struc_ast_meta.jsonl",
        "eli5": None  # use HF built-in
    }

    print(f"📂 Loading dataset: {dataset_name}")

    if dataset_name == "eli5":
        dataset = load_dataset(
            "eli5_category",
            split="train[:7000]",
            trust_remote_code=True
        )
    else:
        dataset_path = os.path.join(repo_path, "datasets", dataset_file_map[dataset_name])
        dataset = load_dataset("json", data_files=dataset_path, split="train")

    # Chronological split
    n_total = len(dataset)
    n_train = int(n_total * 0.64)
    n_eval = int(n_total * 0.16)

    train_dataset = dataset.select(range(0, n_train))
    eval_dataset = dataset.select(range(n_train, n_train + n_eval))
    test_dataset = dataset.select(range(n_train + n_eval, n_total))

    if debug:
        train_dataset = train_dataset.select(range(200))
        eval_dataset = eval_dataset.select(range(100))
        test_dataset = test_dataset.select(range(100))

    train_dataset = train_dataset.shuffle(seed=seed)

    def format_for_lm(example):
        if dataset_name == "eli5":
            text = " ".join(example.get("answers", {}).get("text", []))
            return {"text": text}
        if "prompt" in example:
            return {"text": example["prompt"]}
        return {"text": example.get("text", "")}

    train_formatted = train_dataset.map(format_for_lm, remove_columns=train_dataset.column_names)
    eval_formatted = eval_dataset.map(format_for_lm, remove_columns=eval_dataset.column_names)
    test_formatted = test_dataset.map(format_for_lm, remove_columns=test_dataset.column_names)

    return DatasetDict({
        "train": train_formatted,
        "test": eval_formatted,
        "final_test": test_formatted
    })


def estimate_max_sequence_length(
    dataset,
    tokenizer,
    config,
    percentile=100,
    text_field="text",
    override_max_seq_length=None
):
    """
    Estimate the maximum sequence length to use for tokenization,
    based on a percentile of tokenized text lengths in the TRAIN split,
    unless an explicit override is provided.

    If `override_max_seq_length` is not None, it will be returned instead
    of calculating the length from the data. This is useful if you want
    consistent or manually-tuned max sequence length for comparison runs.

    Args:
        dataset (DatasetDict): The entire dataset object with 'train' split.
        tokenizer: Hugging Face tokenizer.
        config: Model configuration.
        percentile (float): Percentile to use for max length cutoff.
        text_field (str): The field in the dataset containing text.
        override_max_seq_length (int, optional): If provided, overrides any estimated value.

    Returns:
        int: Final max sequence length to use.
    """
    if override_max_seq_length is not None:
        print(f"⚙️ Overriding max sequence length with user-specified value: {override_max_seq_length}")
        return override_max_seq_length

    if "train" not in dataset:
        raise ValueError("The dataset must contain a 'train' split to estimate sequence length.")

    def get_token_length(example):
        return {"length": len(tokenizer(example[text_field], truncation=False)["input_ids"])}

    lengths_dataset = dataset["train"].map(get_token_length)
    lengths = lengths_dataset["length"]

    calculated_max_length = int(np.percentile(lengths, percentile))
    max_seq_len = min(
        calculated_max_length,
        tokenizer.model_max_length,
        config.max_position_embeddings
    )

    print(f"""✅ Using max_seq_length={max_seq_len}, 
    {percentile}th percentile={calculated_max_length}, 
    tokenizer limit={tokenizer.model_max_length}, 
    model limit={config.max_position_embeddings}""")

    return max_seq_len


def chunk_long_samples(
    dataset_dict,
    max_seq_length,
    overlap_pct=0.0,
    keep_original_size=False
):
    """
    Splits each tokenized sample into fixed-size chunks of input_ids AND attention_mask,
    with optional overlap controlled by overlap_pct.

    Uses an explicit for-loop to guarantee each chunk becomes its own row.
    If keep_original_size=True, each split is truncated to match its original size,
    preserving the order of the commits.

    Args:
        dataset_dict: Hugging Face DatasetDict with splits like 'train', 'test', etc.
        max_seq_length: Max tokens per chunk.
        overlap_pct: Float in [0, 1). E.g., 0.2 means 20% overlap.
        keep_original_size: If True, truncate chunked splits to original split size.
        
    Returns:
        New DatasetDict with chunked splits.
    """
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

        for example in dataset_dict[split]:
            tokens = example["input_ids"]
            attn_mask = example["attention_mask"]
            L = len(tokens)

            if L <= max_seq_length:
                input_ids_chunks.append(tokens)
                attention_mask_chunks.append(attn_mask)
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

        after_count = len(input_ids_chunks)
        print(f"✅ Split '{split}': {after_count} samples after chunking.")

        ds = Dataset.from_dict({
            "input_ids": input_ids_chunks,
            "attention_mask": attention_mask_chunks
        })

        if keep_original_size and after_count > before_count:
            print(f"🔻 Truncating '{split}' back to {before_count} samples (chronological).")
            ds = ds.select(range(before_count))

        chunked_dataset[split] = ds

    return chunked_dataset




def compute_custom_metrics(eval_pred):
    """
    Computes perplexity and token-level accuracy for a causal LM.

    Args:
        eval_pred (tuple): A tuple of (logits, labels):
            - logits: Tensor or np.ndarray of shape (batch_size, seq_len, vocab_size)
            - labels: Tensor or np.ndarray of shape (batch_size, seq_len)

    Returns:
        dict: Dictionary with 'perplexity' and 'token_accuracy'.
    """
    logits, labels = eval_pred

    # Convert to torch if needed
    if not torch.is_tensor(logits):
        logits = torch.tensor(logits)
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels)

    # Shift logits and labels for causal LM next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    perplexity = torch.exp(loss).item()

    # Compute token-level accuracy (optional)
    preds = shift_logits.argmax(dim=-1)
    mask = shift_labels != -100
    correct = (preds == shift_labels) & mask
    token_accuracy = correct.sum().item() / mask.sum().item()

    return {
        "perplexity": perplexity,
        "token_accuracy": token_accuracy
    }


class SaveMetricsCallback(TrainerCallback):
    """
    A custom Hugging Face `TrainerCallback` for saving training and evaluation metrics 
    to a JSON Lines (JSONL) file after each logging or evaluation step.

    Works for **any task**: sequence classification, causal language modeling, etc.

    Appends each set of metrics as a new line in the JSONL file, making it easy to 
    parse and visualize later (e.g., with pandas, seaborn, or custom scripts).

    Example metrics for Causal LM might include:
        - Training loss (`loss`)
        - Evaluation loss (`eval_loss`)
        - Perplexity (`perplexity`) if your `compute_metrics` returns it

    ---
    Args:
        output_path (str):
            Path to the `.jsonl` file where metrics will be saved.
            If the directory does not exist, it will be created automatically.

    Example output file (`metrics.jsonl`):
        {"step": 25, "type": "train", "metrics": {"loss": 2.345, "learning_rate": 2e-5}}
        {"step": 50, "type": "eval", "metrics": {"eval_loss": 2.123, "perplexity": 8.34}}

    ---
    Methods:
        - `on_log()`: Called after each logging event during training. Appends training metrics.
        - `on_evaluate()`: Called after each evaluation event (e.g., validation). Appends eval metrics.

    Usage:
        callback = SaveMetricsCallback("path/to/metrics.jsonl")
        Trainer(..., callbacks=[callback])

    Note:
        This does not replace TensorBoard or WandB — it complements them by giving you
        an easy-to-parse file for custom plots or debugging.
    """

    def __init__(self, output_path):
        """
        Initialize the callback and create the output directory if needed.
        """
        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def _write_metrics(self, state, metrics, metric_type):
        """
        Appends a single metric record to the JSONL file.

        Args:
            state (TrainerState): The current training state object.
            metrics (dict): Dictionary of metric values.
            metric_type (str): One of "train" or "eval" to identify the source.
        """
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
    """
    Sets up Trainer callbacks for live metrics logging.

    Args:
        live_metrics_enabled (bool): Whether to enable live metrics logging.
        live_metrics_path (str): Path to save live metrics JSONL file if enabled.

    Returns:
        list: A list of TrainerCallback instances.
    """
    callbacks = []

    if live_metrics_enabled:
        callbacks.append(SaveMetricsCallback(live_metrics_path))
        print(f"📊 Live metrics will be saved to: {live_metrics_path}")
    else:
        print("📊 Live metrics logging disabled.")

    return callbacks


def save_training_config(
    config_path,
    run_timestamp,
    args,
    training_args,
    MAX_SEQ_LENGTH,
    SEQ_LEN_PERCENTILE,
    DEBUG,
    model_config=None
):
    """
    Saves a snapshot of the key training configuration and metadata for a Causal LM run.

    This version is simplified for language modeling only:
    - No classification-specific stats.
    - No defect rate or recall@top_k.
    - No eval_loss or perplexity — focus is on training config only.

    Args:
        config_path (str): Path to save the configuration file.
        run_timestamp (str): Timestamp of the training run.
        args (argparse.Namespace): Parsed CLI arguments.
        training_args (transformers.TrainingArguments): TrainingArguments used for this run.
        MAX_SEQ_LENGTH (int): Final max sequence length used.
        SEQ_LEN_PERCENTILE (float): Percentile used to determine max sequence length.
        DEBUG (bool): Whether debug mode was enabled.
        model_config (dict, optional): Model config as a dict.
    """

    config_snapshot = {
        "timestamp": run_timestamp,
        "model_path": args.model_path,
        "dataset": args.dataset,
        "learning_rate": training_args.learning_rate,
        "epochs": training_args.num_train_epochs,
        "train_batch_size": training_args.per_device_train_batch_size,
        "eval_batch_size": training_args.per_device_eval_batch_size,
        "weight_decay": training_args.weight_decay,
        "metric_for_best_model": training_args.metric_for_best_model,
        "quantized": args.quant,
        "lora_enabled": args.lora,
        "bf16": args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "max_sequence_length": MAX_SEQ_LENGTH,
        "sequence_length_percentile": SEQ_LEN_PERCENTILE,
        "debug": DEBUG
    }

    if model_config is not None:
        config_snapshot["model_config"] = model_config

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config_snapshot, f, indent=2)

    print(f"⚙️ Logged Causal LM config to: {config_path}")


def evaluate_and_save_best_model(trainer, training_args, metrics_dir, adapter_dir, tokenizer_dir, tokenizer=None):
    """
    Evaluates the best checkpointed model (if enabled), saves the eval metrics,
    and saves ONLY the LoRA adapter weights + tokenizer to disk.
    """
    if training_args.load_best_model_at_end:
        best_eval_metrics = trainer.evaluate()
        best_model_metrics_path = os.path.join(metrics_dir, "best_model_metrics.json")
        with open(best_model_metrics_path, "w") as f:
            json.dump(best_eval_metrics, f, indent=4)
        print(f"✅ Saved best model eval metrics to {best_model_metrics_path}")
    else:
        print("ℹ️ Skipping best model evaluation because load_best_model_at_end=False.")
        best_eval_metrics = None

    print(f"💾 Saving LoRA adapter to {adapter_dir}")
    trainer.model.save_pretrained(adapter_dir)

    if tokenizer is not None:
        print(f"💾 Saving tokenizer to {tokenizer_dir}")
        tokenizer.save_pretrained(tokenizer_dir)
    else:
        print("⚠️ No tokenizer provided; skipping save.")

    print("⚡️ To use this later: load the same base model and attach the adapter with PeftModel.from_pretrained()")

    return best_eval_metrics



def save_training_metrics(trainer, metrics_dir, filename="metrics.json"):
    """
    Saves the full training and evaluation metric history from a Hugging Face Trainer to a JSON file.

    This function extracts the `.state.log_history` attribute from the `trainer`, which contains
    all logged training and evaluation metrics collected over the course of training (e.g., losses,
    learning rates, evaluation scores per step), and writes it to a specified file.

    Args:
        trainer (transformers.Trainer): A Hugging Face Trainer instance with completed training.
        metrics_dir (str): Directory where the metrics file should be saved.
        filename (str, optional): Name of the output file. Default is "metrics.json".

    Returns:
        str: Path to the saved metrics JSON file.

    Side Effects:
        - Creates `metrics_dir` if it does not exist.
        - Saves the trainer’s logged metrics as a JSON file at the specified location.
        - Prints a confirmation message with the file path.

    Example Output (`metrics.json`):
        [
            {"loss": 0.58, "learning_rate": 4e-5, "epoch": 1.0, "step": 50},
            {"eval_loss": 0.45, "eval_accuracy": 0.82, "epoch": 1.0, "step": 50},
            ...
        ]

    Notes:
        - This function is useful for post-training analysis or for generating plots (e.g., loss curves).
        - The returned JSON can be parsed later for plotting training/eval trends over time.
    """

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
):
    """
    Runs final inference on a held-out test dataset for a Causal LM.
    Computes final eval loss, perplexity, and token-level accuracy,
    and saves the results to a JSON file.

    Args:
        trainer (transformers.Trainer): Your fine-tuned Trainer object.
        test_dataset (Dataset): The held-out test dataset.
        metrics_dir (str): Directory to save final metrics as `final_test_results.json`.

    Output:
        Saves a JSON file with:
          - eval_loss
          - perplexity
          - token_accuracy

    Example:
        {
            "eval_loss": 1.23,
            "perplexity": 3.42,
            "token_accuracy": 0.78
        }
    """

    print("\n🧪 Running final inference on held-out test set...")

    test_results = trainer.predict(test_dataset)
    logits, labels = test_results.predictions, test_results.label_ids

    # Some Trainer versions log 'test_loss', some 'eval_loss'
    eval_loss = test_results.metrics.get("test_loss", test_results.metrics.get("eval_loss"))

    # Compute perplexity safely
    try:
        perplexity = float(np.exp(eval_loss))
    except OverflowError:
        perplexity = float("inf")

    # ✅ Compute token-level accuracy (same logic as in compute_custom_metrics)
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    preds = shift_logits.argmax(dim=-1)
    mask = shift_labels != -100
    correct = (preds == shift_labels) & mask
    token_accuracy = correct.sum().item() / mask.sum().item()

    final_metrics = {
        "eval_loss": eval_loss,
        "perplexity": perplexity,
        "token_accuracy": token_accuracy
    }

    os.makedirs(metrics_dir, exist_ok=True)
    final_test_metrics_path = os.path.join(metrics_dir, "final_test_results.json")
    with open(final_test_metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=4)

    print(f"📄 Final test set results saved to: {final_test_metrics_path}")
    print(json.dumps(final_metrics, indent=4))

