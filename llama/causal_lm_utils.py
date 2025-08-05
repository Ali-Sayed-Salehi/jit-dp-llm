import os
import json
import argparse
from datetime import datetime
from subprocess import run, CalledProcessError
import yaml

import torch
import numpy as np

from torch.nn import functional as F

from datasets import load_dataset, Dataset, DatasetDict

from transformers import (
    TrainerCallback
)


def chunk_long_samples(
    dataset_dict,
    max_seq_length,
    overlap_pct=0.0
):
    """
    Splits each tokenized sample into fixed-size chunks of input_ids AND attention_mask,
    with optional overlap controlled by overlap_pct. Also sets labels = input_ids.

    Uses an explicit for-loop to guarantee each chunk becomes its own row.

    Args:
        dataset_dict: Hugging Face DatasetDict with splits like 'train', 'test', etc.
        max_seq_length: Max tokens per chunk.
        overlap_pct: Float in [0, 1). E.g., 0.2 means 20% overlap.
        
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



def save_training_config(
    config_path,
    run_timestamp,
    args,
    training_args,
    truncation_len,
    chunking_len,
    DEBUG,
    FSDP,
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
        truncation_len (float): Max length used to which each sequence will be truncated.
        chunking_len (float): Max length used to which each sequence will be chunked.
        DEBUG (bool): Whether debug mode was enabled.
        model_config (dict, optional): Model config as a dict.
    """

    config_snapshot = {
        "timestamp": run_timestamp,
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
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
        "truncation_len":truncation_len,
        "chunking_len":chunking_len,
        "FSDP": FSDP,
        "debug": DEBUG
    }

    if model_config is not None:
        config_snapshot["model_config"] = model_config

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config_snapshot, f, indent=2)

    print(f"⚙️ Logged Causal LM config to: {config_path}")


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

def safe_predict(trainer, dataset, accum_steps=1):
    orig_eval_accum = trainer.args.eval_accumulation_steps
    try:
        trainer.args.eval_accumulation_steps = accum_steps
        return trainer.predict(dataset)
    finally:
        trainer.args.eval_accumulation_steps = orig_eval_accum


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

    test_results = safe_predict(trainer, test_dataset, accum_steps=2)
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

