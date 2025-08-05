import os
import json
import argparse
from datetime import datetime
from collections import Counter
from subprocess import run, CalledProcessError
import yaml

import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.nn import functional as F

from datasets import load_dataset, Dataset, DatasetDict
from sklearn.metrics import roc_auc_score, average_precision_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    Trainer
)

from accelerate import (
    init_empty_weights,
    infer_auto_device_map, 
)

import evaluate
from attach_classification_head_llama4 import CustomLlama4ForSequenceClassification, CustomLlama4TextConfig


def compute_class_distribution(dataset_dict: DatasetDict) -> dict:
    """
    Compute class distribution for each split in a DatasetDict,
    and the percentage of the positive class ('1').
    """
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


def apply_class_imbalance_strategy(
    dataset,
    strategy="none",
    seed=42,
    alpha=0.25,
    gamma=2.0
):
    """
    Applies a class imbalance correction strategy to the training dataset,
    while keeping the evaluation and final test datasets unchanged.
    Returns distributions before and after balancing (if applied).
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
        label_counts = Counter(train_dataset["labels"])
        alpha_dict = {k: alpha for k in label_counts.keys()}
        return dataset, None, {"alpha": alpha_dict, "gamma": gamma}, before_dist, before_dist

    elif strategy == "weighted_loss":
        print("🔄 Using Weighted Cross Entropy Loss.")
        label_counts = Counter(train_dataset["labels"])
        total = sum(label_counts.values())
        weights = [total / label_counts[i] for i in sorted(label_counts)]
        for i, weight in enumerate(weights):
            print(f"  Class {i} weight: {weight:.4f}")
        return dataset, weights, None, before_dist, before_dist

    elif strategy == "oversampling":
        print("📈 Applying random oversampling to balance dataset.")
        df = pd.DataFrame(train_dataset)
        X = df.drop(columns=["labels"])
        y = df["labels"]
        ros = RandomOverSampler(random_state=seed)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        resampled_df = X_resampled.copy()
        resampled_df["labels"] = y_resampled
        train_dataset_balanced = Dataset.from_pandas(resampled_df, preserve_index=False)

        balanced_dataset = DatasetDict({
            "train": train_dataset_balanced,
            "test": eval_dataset,
            "final_test": test_dataset
        })

        after_dist = compute_class_distribution(balanced_dataset)
        print("📊 Class distribution after oversampling:")
        for split, stats in after_dist.items():
            print(f"  {split}: {stats}")

        return balanced_dataset, None, None, before_dist, after_dist

    elif strategy == "undersampling":
        print("📉 Applying random undersampling to balance dataset.")
        df = pd.DataFrame(train_dataset)
        X = df.drop(columns=["labels"])
        y = df["labels"]
        rus = RandomUnderSampler(random_state=seed)
        X_resampled, y_resampled = rus.fit_resample(X, y)
        resampled_df = X_resampled.copy()
        resampled_df["labels"] = y_resampled
        train_dataset_balanced = Dataset.from_pandas(resampled_df, preserve_index=False)

        balanced_dataset = DatasetDict({
            "train": train_dataset_balanced,
            "test": eval_dataset,
            "final_test": test_dataset
        })

        after_dist = compute_class_distribution(balanced_dataset)
        print("📊 Class distribution after undersampling:")
        for split, stats in after_dist.items():
            print(f"  {split}: {stats}")

        return balanced_dataset, None, None, before_dist, after_dist

    elif strategy == "none":
        print("🟢 No class imbalance fix applied.")
        return dataset, None, None, before_dist, before_dist

    else:
        raise ValueError(f"Unsupported class imbalance strategy: {strategy}")


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
                dtype=dtype,   # 🔑 match dtype of logits
                device=device  # 🔑 match device of logits
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



# Load Hugging Face metrics once
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

def recall_at_top_k(pred_scores, true_labels, percentages=None):
    """
    Computes recall@top-K% for a list of prediction scores, typically used for imbalanced classification
    to measure how many true positives are recovered in the top-ranked examples.

    Args:
        pred_scores (np.ndarray): Array of predicted probabilities for the positive class, shape (N,).
        true_labels (np.ndarray): Array of ground truth binary labels (0 or 1), shape (N,).
        percentages (list of float): List of top-K percentages (as floats) to evaluate recall at. 
                                     For example, 0.1 corresponds to recall@top-10%.

    Returns:
        dict: Dictionary mapping metric names (e.g., "recall@top_10%") to recall values.
              Each recall value is the fraction of total positives recovered in the top-K% examples.

    Example:
        >>> recall_at_top_k(np.array([0.9, 0.8, 0.1]), np.array([1, 0, 1]), percentages=[0.33])
        {'recall@top_33%': 0.5}
    """
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


def compute_custom_metrics(eval_pred, threshold=None, percentages=None):
    """
    Computes a comprehensive set of evaluation metrics for binary classification tasks, including:
    - Accuracy, Precision, Recall, F1 Score (via Hugging Face `evaluate` library)
    - ROC AUC and PR AUC (via scikit-learn)
    - Class prediction distribution
    - Recall@top-K% for 5%, 10%, and 30%

    Args:
        eval_pred (tuple): A tuple of (logits, labels):
            - logits (np.ndarray or torch.Tensor): Model outputs before softmax, shape (batch_size, num_classes)
            - labels (np.ndarray): Ground truth class labels, shape (batch_size,)
        threshold (float, optional): Decision threshold for binary classification. If None, uses `argmax` over logits.
                                     If provided, compares positive class probability against this threshold.

    Returns:
        dict: Dictionary containing all computed metrics:
            - "accuracy", "precision", "recall", "f1"
            - "roc_auc", "pr_auc"
            - "pred_class_0", "pred_class_1", ...
            - "recall@top_5%", "recall@top_10%", "recall@top_30%"

    Raises:
        ValueError: If `roc_auc_score` or `average_precision_score` cannot be computed (e.g., only one class in labels).

    Example:
        >>> compute_custom_metrics((logits, labels), threshold=0.7)
        {
            'accuracy': 0.85,
            'precision': 0.8,
            ...
            'recall@top_10%': 0.6
        }
    """
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


class CustomTrainer(Trainer):
    """
    Custom Trainer with class imbalance handling (bf16-only).
    """

    def __init__(self, *args, class_weights=None, focal_loss_fct=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Store class weights directly in bf16
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.bfloat16)
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



def run_final_inference(
    trainer,
    test_dataset,
    metrics_dir,
    percentages,
    threshold=None,
):
    """
    Runs inference on a held-out test dataset using a fine-tuned Hugging Face Trainer model.
    Computes classification metrics and saves the results to a JSON file.

    Args:
        trainer (transformers.Trainer): The trained Hugging Face `Trainer` object used for inference.
        test_dataset (Dataset): The held-out test dataset to evaluate.
        metrics_dir (str): Directory path where the metrics will be saved as `final_test_results.json`.
        percentages (list of float): Percentages to compute recall@top_k.
        threshold (float, optional): If provided, applies a custom decision threshold to the probability 
                                     of the positive class instead of using argmax for prediction.

    Output:
        Saves a JSON file `final_test_results.json` in `metrics_dir`, containing:
            - All classification metrics (accuracy, precision, recall, f1, ROC AUC, PR AUC)
            - Optional recall@top-{k}% if a function is provided
            - Model predictions and probabilities
            - True labels

    Example JSON Output:
        {
            "metrics": {
                "accuracy": 0.87,
                "precision": 0.85,
                "recall": 0.88,
                "f1": 0.86,
                "roc_auc": 0.91,
                "pr_auc": 0.89,
                "recall@top_5%": 0.52,
                "recall@top_10%": 0.70,
                ...
            },
            "predictions": [0, 1, 0, ...],
            "probabilities": [0.22, 0.85, 0.30, ...],
            "true_labels": [0, 1, 0, ...]
        }

    Notes:
        - The threshold is used to binarize predictions on the positive class probability.
        - ROC AUC and PR AUC are robust to thresholding and reflect ranking performance.
        - Handles exceptions gracefully if AUC metrics cannot be computed (e.g., due to single-class predictions).
    """

    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    print("\n🧪 Running final inference on held-out test set...")

    test_results = trainer.predict(test_dataset)
    logits = test_results.predictions
    labels = test_results.label_ids

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

    output_payload = {
        "metrics": final_metrics,
        "predictions": preds.tolist(),
        "probabilities": probs[:, 1].tolist(),
        "true_labels": labels.tolist()
    }

    os.makedirs(metrics_dir, exist_ok=True)
    final_test_metrics_path = os.path.join(metrics_dir, "final_test_results.json")
    with open(final_test_metrics_path, "w") as f:
        json.dump(output_payload, f, indent=4)

    print(f"📄 Final test set results saved to: {final_test_metrics_path}")
    print(json.dumps(final_metrics, indent=4))


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



def save_training_config(
    config_path,
    run_timestamp,
    args,
    training_args,
    class_distribution,
    original_class_distribution,
    truncation_len,
    chunking_len,
    DEBUG,
    dataset=None,
    RECALL_AT_TOP_K_PERCENTAGES=None,
    FL_GAMMA=None,
    FL_ALPHA=None,
    model_config=None
):
    """
    Saves a snapshot of the key training configuration and metadata to a JSON file.

    This version also computes and logs:
      - Defect rate for the held-out test set (ratio of positive class to total)
      - Maximum possible recall@top_k% based on the true distribution

    Args:
        config_path (str): Path to save the configuration file (e.g., "metrics/config.json").
        run_timestamp (str): Timestamp of the training run.
        args (argparse.Namespace): Parsed CLI arguments.
        training_args (transformers.TrainingArguments): TrainingArguments used for this run.
        class_distribution (dict): Class distribution after any imbalance fix.
        original_class_distribution (dict): Original class distribution before any imbalance fix.
        truncation_len (float): Max sequence length to which every sequence will be truncated.
        chunking_len (float): Max sequence length to which every sequence will be chunked.
        DEBUG (bool): Whether debug mode was enabled.
        dataset (DatasetDict, optional): The full dataset object with splits. Should contain 'final_test'.
        RECALL_AT_TOP_K_PERCENTAGES (List[float], optional): List of percentages for recall@top_k.
        FL_GAMMA (float, optional): Focal loss gamma, if used.
        FL_ALPHA (float or dict, optional): Focal loss alpha, if used.
        model_config (dict, optional): Model config as a dict.

    Side Effects:
        - Creates or overwrites `config.json` at the given path.
        - Prints out defect rate and max recall@top_k% for transparency.

    Example output:
        {
            "timestamp": "...",
            "model_path": "...",
            ...
            "held_out_test_defect_rate": 0.23,
            "max_possible_recall@top_k": {
                "max_recall@top_5%": 0.25,
                ...
            },
            ...
        }
    """
    # ------------------ Compute held-out defect rate + max recall@top_k ------------------
    defect_rate = None
    max_recall_at_k = {}

    if dataset is not None:
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
        "model_path": args.model_path,
        "class_imbalance_fix": args.class_imbalance_fix,
        "dataset_path": args.dataset_path,
        "learning_rate": training_args.learning_rate,
        "epochs": training_args.num_train_epochs,
        "train_batch_size": training_args.per_device_train_batch_size,
        "eval_batch_size": training_args.per_device_eval_batch_size,
        "weight_decay": training_args.weight_decay,
        "metric_for_best_model": training_args.metric_for_best_model,
        "class_distribution": class_distribution,
        "original_class_distribution": original_class_distribution,
        "decision_threshold": args.threshold if args.threshold is not None else "argmax",
        "quantized": args.quant,
        "lora_enabled": args.lora,
        "focal_loss_gamma": FL_GAMMA if args.class_imbalance_fix == "focal_loss" else "None",
        "focal_loss_alpha": FL_ALPHA if args.class_imbalance_fix == "focal_loss" else "None",
        "bf16": args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "truncation_len": truncation_len,
        "chunking_len": chunking_len,
        "debug": DEBUG,
        "held_out_test_defect_rate": defect_rate,
        "max_possible_recall@top_k": max_recall_at_k
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


def copy_model_to_tmpdir(model_path, repo_root, tmpdir_prefix):
    """
    Copies a model directory (which also contains tokenizer files) to local scratch (e.g., $SLURM_TMPDIR),
    preserving its relative structure under repo_root.

    Args:
        model_path (str): Absolute path to the model directory to copy.
        repo_root (str): The root of your repo (used to compute relative paths).
        tmpdir_prefix (str): Path prefix for local scratch, e.g., $SLURM_TMPDIR.

    Returns:
        str: Path to the copied model directory inside tmpdir.
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Source directory does not exist: {model_path}")

    rel_path = os.path.relpath(model_path, start=repo_root)
    dest_dir = os.path.join(tmpdir_prefix, rel_path)
    os.makedirs(dest_dir, exist_ok=True)

    print(f"📂 Copying {model_path} → {dest_dir} ...")
    run(["rsync", "-a", "--delete", model_path.rstrip("/") + "/", dest_dir], check=True)
    print(f"✅ Copy complete: {dest_dir}")

    return dest_dir

def get_mixed_precision_policy():
    """
    Checks whether the current GPU supports BF16 or FP16.
    Returns:
        - 'bf16' if BF16 is supported
        - 'fp16' if FP16 is supported
        - None if no GPU is available
    """
    if not torch.cuda.is_available():
        print("❌ No GPU detected. Using full precision (FP32).")
        return None

    device = torch.device("cuda")
    major, minor = torch.cuda.get_device_capability(device)

    print(f"✅ Detected GPU compute capability: {major}.{minor}")

    if major >= 8:
        print("✨ Your GPU supports BF16 mixed precision.")
        return "bf16"
    else:
        print("⚡️ Your GPU does not support native BF16. Using FP16 instead.")
        return "fp16"
