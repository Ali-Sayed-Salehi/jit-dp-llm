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
from attach_classification_head import CustomLlamaForSequenceClassification, CustomLlamaConfig


def parse_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--live_metrics", action="store_true", help="Enable saving evaluation metrics after each eval step")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["perf", "jit", "jit_balanced", "imdb", "jit_small", "jit_small_struc_ast_meta"],
        default="imdb",
        help="Choose which dataset to use"
    )
    parser.add_argument("--model_path", type=str, required=True, help="Full path to the local pretrained model directory")
    parser.add_argument(
        "--class_imbalance_fix",
        type=str,
        choices=["oversampling", "weighted_loss", "focal_loss", "none"],
        default="none",
        help="Class imbalance handling method"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Optional decision threshold for classifying as class 1 (between 0 and 1). If not set, uses argmax."
    )
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
        "--selection_metric",
        type=str,
        default="recall@top_5%",
        help="Metric to select the best model: recall@top_5%, recall@top_10%, recall@top_30%, f1, precision, recall, accuracy"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="Optional. If set, overrides the estimated max sequence length."
    )
    
    args = parser.parse_args()
    
    # --- Validation Section ---
    if args.threshold is not None and not (0.0 <= args.threshold <= 1.0):
        raise ValueError("Threshold must be between 0 and 1 if specified")
    
    VALID_SELECTION_METRICS = {"recall@top_5%", "recall@top_10%", "recall@top_30%", "f1", "precision", "recall", "accuracy"}
    if args.selection_metric not in VALID_SELECTION_METRICS:
        raise ValueError(f"Unsupported selection_metric '{args.selection_metric}'. Must be one of {VALID_SELECTION_METRICS}")
    
    return args



def setup_training_directories(repo_root, continue_from_dir=None):
    """
    Sets up output, metrics, tensorboard, and model/tokenizer directories.

    Returns:
        dict with keys: output_dir, run_timestamp, metrics_dir, tensorboard_dir,
        config_path, live_metrics_path, model_dir, tokenizer_dir, all_dirs (list of paths created)
    """
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

    This function supports both local JSONL datasets and the Hugging Face IMDb dataset.
    For local datasets, it performs a chronological split: 64% for training, 16% for evaluation,
    and 20% for testing. The IMDb dataset is returned as-is from the Hugging Face hub.

    Args:
        dataset_name (str): Name of the dataset to load. Supported values:
            - "perf": Performance regression dataset.
            - "jit": ApacheJIT dataset for LLMs.
            - "jit_balanced": Balanced version of the ApacheJIT dataset.
            - "jit_small": A smaller ApacheJIT dataset.
            - "jit_small_struc_ast_meta": Structural ApacheJIT dataset with AST/meta info.
            - "imdb": Loads IMDb dataset from Hugging Face datasets library.
        repo_path (str): Path to the root of the repository where local datasets are stored.
        debug (bool, optional): If True, reduces the dataset size to 200 examples for faster experimentation. Default is False.
        seed (int, optional): Random seed for shuffling the training dataset. Default is 42.

    Returns:
        DatasetDict: A dictionary with Hugging Face `Dataset` objects with the following keys:
            - "train": Training split, formatted with 'text' and 'label' fields.
            - "test": Evaluation split, formatted with 'text' and 'label' fields.
            - "final_test": Held-out test split, formatted with 'text' and 'label' fields.
        
        If `dataset_name == "imdb"`, returns the raw Hugging Face IMDb dataset without splitting or formatting.

    Raises:
        KeyError: If an unsupported `dataset_name` is provided.
        FileNotFoundError: If the specified dataset file does not exist.
    """

    dataset_file_map = {
        "perf": "dataset.jsonl",
        "jit": "jit_dp/apachejit_llm.jsonl",
        "jit_balanced": "jit_dp/apachejit_llm_balanced.jsonl",
        "jit_small": "jit_dp/apachejit_llm_small.jsonl",
        "jit_small_struc_ast_meta": "jit_dp/apachejit_llm_small_struc_ast_meta.jsonl",
        "imdb": None  # use HF built-in
    }

    print(f"📂 Loading dataset: {dataset_name}")

    if dataset_name != "imdb":
        dataset_path = os.path.join(repo_path, "datasets", dataset_file_map[dataset_name])
        dataset = load_dataset("json", data_files=dataset_path, split="train")

        # Chronologically split the dataset: 64% train, 16% eval, 20% test
        n_total = len(dataset)
        n_train = int(n_total * 0.64)
        n_eval = int(n_total * 0.16)

        train_dataset = dataset.select(range(0, n_train))
        eval_dataset = dataset.select(range(n_train, n_train + n_eval))
        test_dataset = dataset.select(range(n_train + n_eval, n_total))

        if debug:
            train_dataset = train_dataset.select(range(200))
            eval_dataset = eval_dataset.select(range(200))
            test_dataset = test_dataset.select(range(200))

        train_dataset = train_dataset.shuffle(seed=seed)

        def format_for_classification(example):
            return {
                "text": example['prompt'],
                "label": int(example["response"])
            }

        train_formatted = train_dataset.map(format_for_classification, remove_columns=["prompt", "response"])
        eval_formatted = eval_dataset.map(format_for_classification, remove_columns=["prompt", "response"])
        test_formatted = test_dataset.map(format_for_classification, remove_columns=["prompt", "response"])

        return DatasetDict({
            "train": train_formatted,
            "test": eval_formatted,
            "final_test": test_formatted
        })
    else:
        dataset = load_dataset("imdb")
        return dataset



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

    Args:
        dataset (datasets.DatasetDict): A DatasetDict with keys "train", "test", and "final_test".
        strategy (str): The class imbalance strategy to apply. Supported values:
            - "focal_loss": Computes alpha and gamma for focal loss, no resampling is performed.
            - "weighted_loss": Computes weights for weighted cross-entropy.
            - "oversampling": Uses RandomOverSampler to balance class distribution.
            - "none": No class imbalance fix is applied.
        seed (int): Random seed for reproducibility.
        alpha (float): Alpha value for focal loss.
        gamma (float): Gamma value for focal loss.

    Returns:
        tuple:
            - datasets.DatasetDict: Updated dataset with possibly balanced training split.
            - list or None: Class weights (if "weighted_loss" is used).
            - dict or None: Focal loss params if "focal_loss" is used.

    Raises:
        ValueError: If the strategy is unsupported.
    """
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    test_dataset = dataset["final_test"]

    if strategy == "focal_loss":
        print("🔥 Using Focal Loss for class imbalance.")
        label_counts = Counter(train_dataset["label"])
        alpha_dict = {k: alpha for k in label_counts.keys()}
        return DatasetDict({
            "train": train_dataset,
            "test": eval_dataset,
            "final_test": test_dataset
        }), None, {"alpha": alpha_dict, "gamma": gamma}

    elif strategy == "weighted_loss":
        print("🔄 Using Weighted Cross Entropy Loss.")
        label_counts = Counter(train_dataset["label"])
        total = sum(label_counts.values())
        weights = [total / label_counts[i] for i in sorted(label_counts)]
        for i, weight in enumerate(weights):
            print(f"  Class {i} weight: {weight:.4f}")
        return DatasetDict({
            "train": train_dataset,
            "test": eval_dataset,
            "final_test": test_dataset
        }), weights, None

    elif strategy == "oversampling":
        print("📈 Applying random oversampling to balance dataset.")
        df = pd.DataFrame(train_dataset)
        X = df.drop(columns=["label"])
        y = df["label"]
        ros = RandomOverSampler(random_state=seed)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        resampled_df = X_resampled.copy()
        resampled_df["label"] = y_resampled
        train_dataset_balanced = Dataset.from_pandas(resampled_df)
        return DatasetDict({
            "train": train_dataset_balanced,
            "test": eval_dataset,
            "final_test": test_dataset
        }), None, None

    elif strategy == "none":
        print("🟢 No class imbalance fix applied.")
        return DatasetDict({
            "train": train_dataset,
            "test": eval_dataset,
            "final_test": test_dataset
        }), None, None

    else:
        raise ValueError(f"Unsupported class imbalance strategy: {strategy}")



def compute_class_distribution(labels) -> dict:
    """
    Compute class distribution as a dictionary with string keys.

    Args:
        labels (List[int] or DatasetColumn): A list or dataset column of integer class labels.

    Returns:
        dict: A dictionary where keys are class labels (as strings) and values are counts.
    """
    label_counts = Counter(labels)
    return {
        str(label): int(count)
        for label, count in sorted(label_counts.items())
    }


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


class FocalLoss(nn.Module):
    """
    Implementation of the Focal Loss function for classification tasks, 
    commonly used to address class imbalance by down-weighting easy examples 
    and focusing training on hard examples.

    Args:
        alpha (float or dict): Weighting factor for the classes. Can be:
            - A float: same weight for all classes.
            - A dict: class-specific weights (e.g., {"0": 0.25, "1": 0.75}).
        gamma (float): Focusing parameter that adjusts the rate at which easy examples are down-weighted.
                       A higher value increases focus on hard-to-classify examples. Default is 2.0.
        reduction (str): Specifies the reduction to apply to the output:
                         - 'mean': mean of the loss across the batch (default)
                         - 'sum': sum of the loss across the batch
                         - 'none': no reduction

    Inputs:
        logits (Tensor): Raw output from the model of shape (batch_size, num_classes).
        targets (Tensor): Ground truth class indices of shape (batch_size,).

    Returns:
        Tensor: Scalar loss value if reduction is 'mean' or 'sum'; otherwise, a tensor of per-sample losses.

    Example:
        >>> loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        >>> logits = torch.randn(8, 2)  # batch of 8, binary classification
        >>> targets = torch.randint(0, 2, (8,))
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Computes the focal loss.

        Args:
            logits (Tensor): Predicted logits of shape (batch_size, num_classes).
            targets (Tensor): True class indices of shape (batch_size,).

        Returns:
            Tensor: The focal loss (scalar or per-sample depending on reduction).
        """

        num_classes = logits.size(-1)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)

        # One-hot encode targets
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float().to(logits.device)

        # Gather probabilities of the true class
        pt = (probs * targets_one_hot).sum(dim=1)

        # Handle alpha per class if it's a dict
        if isinstance(self.alpha, dict):
            # Convert dict to tensor of shape (num_classes,) on the correct device
            alpha_tensor = torch.tensor(
                [self.alpha.get(str(i), 1.0) for i in range(num_classes)],
                dtype=torch.float32,
                device=logits.device
            )
            alpha = (alpha_tensor * targets_one_hot).sum(dim=1)  # shape: (batch_size,)
        else:
            alpha = self.alpha  # scalar

        focal_term = (1 - pt) ** self.gamma
        loss = -alpha * focal_term * log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        return loss.mean()



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


class SaveMetricsCallback(TrainerCallback):
    """
    A custom Hugging Face `TrainerCallback` for saving training and evaluation metrics to disk
    after each logging or evaluation step. Metrics are appended to a JSON Lines (JSONL) file,
    enabling later visualization or analysis (e.g., plotting loss/accuracy curves).

    Args:
        output_path (str): Path to the output `.jsonl` file where metrics will be saved.
                           The directory will be created if it does not exist.

    Example JSONL structure:
        {
            "step": 100,
            "type": "train",
            "metrics": {
                "loss": 0.543,
                "learning_rate": 5e-5
            }
        }

        {
            "step": 200,
            "type": "eval",
            "metrics": {
                "eval_accuracy": 0.88,
                "eval_loss": 0.42
            }
        }

    Methods:
        on_log(): Called after each log event (e.g., loss during training).
        on_evaluate(): Called after each evaluation step (e.g., validation metrics).
    """

    def __init__(self, output_path):
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
        """
        Called after logging training metrics. Saves training metrics to file if present.

        Args:
            args (TrainingArguments): The training arguments.
            state (TrainerState): Current state of training.
            control (TrainerControl): Control flow handler.
            logs (dict, optional): Dictionary of logged training metrics.
        """
        if logs and not any(k.startswith("eval_") for k in logs):
            self._write_metrics(state, logs, "train")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Called after evaluation. Saves evaluation metrics to file.

        Args:
            args (TrainingArguments): The training arguments.
            state (TrainerState): Current state of training.
            control (TrainerControl): Control flow handler.
            metrics (dict, optional): Evaluation metrics dictionary.
        """
        self._write_metrics(state, metrics, "eval")



class CustomTrainer(Trainer):
    """
    A custom extension of Hugging Face's `Trainer` class that supports
    class imbalance handling via class weights or focal loss.

    This class overrides the default `compute_loss` method to apply either:
    - Weighted CrossEntropyLoss using provided class weights
    - Focal Loss using a provided `FocalLoss` object
    - Standard CrossEntropyLoss (if no handling specified)

    Args:
        class_weights (list or torch.Tensor, optional): 
            A list or tensor of weights for each class, used to correct for class imbalance.
            If provided, passed as `weight=` to `torch.nn.functional.cross_entropy`.
        focal_loss_fct (nn.Module, optional): 
            A `FocalLoss` object to be used instead of standard cross entropy.

    Example:
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            class_weights=[1.0, 2.5],
        )

    Methods:
        compute_loss(): Computes the loss using the selected class imbalance strategy.
    """

    def __init__(self, *args, class_weights=None, focal_loss_fct=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float32).to(self.args.device)
            if class_weights is not None else None
        )
        self.focal_loss_fct = focal_loss_fct

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Computes the training loss using one of the following:
            - Focal loss (if `focal_loss_fct` is provided)
            - Weighted cross entropy loss (if `class_weights` is provided)
            - Standard cross entropy loss (default)

        Args:
            model (PreTrainedModel): The model to compute loss for.
            inputs (dict): Dictionary containing input tensors (e.g., input_ids, attention_mask, labels).
            return_outputs (bool): If True, also returns model outputs.

        Returns:
            torch.Tensor or (torch.Tensor, dict): Loss value, and optionally model outputs.
        """
        labels = inputs.pop("labels").long()
        outputs = model(**inputs)
        logits = outputs.get('logits')

        if self.focal_loss_fct:
            loss = self.focal_loss_fct(logits, labels)
        elif self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
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


def evaluate_and_save_best_model(trainer, training_args, metrics_dir):
    """
    Evaluates the best checkpointed model after training (if enabled) and saves the evaluation metrics.

    This function checks if `load_best_model_at_end=True` in the training arguments.
    If so, it runs evaluation using `trainer.evaluate()` and saves the resulting metrics to
    a JSON file (`best_model_metrics.json`) inside the given metrics directory.

    Args:
        trainer (transformers.Trainer): A Hugging Face `Trainer` object, expected to have loaded the best model.
        training_args (transformers.TrainingArguments): The training arguments used during fine-tuning.
        metrics_dir (str): Directory where the evaluation results should be saved.

    Returns:
        dict or None: A dictionary of evaluation metrics if evaluated, otherwise `None`.

    Side Effects:
        - Creates (or overwrites) a file named `best_model_metrics.json` in `metrics_dir`.
        - Prints status messages to the console.

    Example Output File (`best_model_metrics.json`):
        {
            "eval_loss": 0.42,
            "eval_accuracy": 0.87,
            "eval_runtime": 5.2,
            ...
        }

    Notes:
        - This function is typically used after training completes, especially if early stopping or
          metric-based checkpointing is used.
        - If `load_best_model_at_end` is `False`, evaluation is skipped.
    """

    if training_args.load_best_model_at_end:
        best_eval_metrics = trainer.evaluate()
        best_model_metrics_path = os.path.join(metrics_dir, "best_model_metrics.json")

        with open(best_model_metrics_path, "w") as f:
            json.dump(best_eval_metrics, f, indent=4)

        print(f"✅ Saved best model eval metrics to {best_model_metrics_path}")
        return best_eval_metrics
    else:
        print("ℹ️ Skipping best model evaluation because load_best_model_at_end=False.")
        return None


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
    MAX_SEQ_LENGTH,
    SEQ_LEN_PERCENTILE,
    DEBUG,
    dataset=None,  # ✅ pass the whole DatasetDict now
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
        MAX_SEQ_LENGTH (int): Final max sequence length used.
        SEQ_LEN_PERCENTILE (float): Percentile used to determine max sequence length.
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
            true_labels_test_set = np.array(dataset["final_test"]["label"])
            
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
        "dataset": args.dataset,
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
        "max_sequence_length": MAX_SEQ_LENGTH,
        "sequence_length_percentile": SEQ_LEN_PERCENTILE,
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


def register_custom_llama_if_needed(model_path: str):
    AutoConfig.register(CustomLlamaConfig.model_type, CustomLlamaConfig)
    AutoModelForSequenceClassification.register(CustomLlamaConfig, CustomLlamaForSequenceClassification)

    config = AutoConfig.from_pretrained(model_path)
    print(f"✅ model and config registered: model_type={config.model_type}, architectures={config.architectures}")


