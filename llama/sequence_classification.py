import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch.nn as nn
import evaluate
import numpy as np
from dotenv import load_dotenv
import os
from datetime import datetime
import json
import argparse
from huggingface_hub import login as huggingface_hub_login
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoConfig,
    TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorWithPadding, 
    TrainerCallback, LlamaForSequenceClassification
)
import random
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training


# ---------------------------- constants  ----------------------------

REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f"✅ Detected REPO_PATH: {REPO_PATH}")
run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

output_dir = f"{REPO_PATH}/llama/training/run_{run_timestamp}/output"
tensorboard_dir = f"{REPO_PATH}/llama/training/run_{run_timestamp}/tensorboard"
metrics_dir = f"{REPO_PATH}/llama/training/run_{run_timestamp}/metrics"
live_metrics_path = os.path.join(metrics_dir, "live_metrics.jsonl")
config_path = os.path.join(metrics_dir, "config.json")
finetuned_model_dir = f"{REPO_PATH}/llama/training/run_{run_timestamp}/model"
finetuned_tokenizer_dir = f"{REPO_PATH}/llama/training/run_{run_timestamp}/tokenizer"

training_dirs = [output_dir, tensorboard_dir, metrics_dir, finetuned_model_dir, finetuned_tokenizer_dir]

for directory in training_dirs:
    os.makedirs(directory, exist_ok=True)

# ---------------------------- Parse Arguments ----------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
parser.add_argument("--live_metrics", action="store_true", help="Enable saving evaluation metrics after each eval step")
parser.add_argument("--perf_data", action="store_true", help="Use performance dataset with oversampling instead of IMDb")
parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Name of the model to use")
parser.add_argument(
    "--class_imbalance_fix",
    type=str,
    choices=["oversampling", "weighted_loss", "focal_loss", "none"],
    default="none",
    help="Class imbalance handling method: 'oversampling' (default), or 'weighted_loss'"
)
parser.add_argument(
    "--threshold",
    type=float,
    help="Optional decision threshold for classifying as class 1 (between 0 and 1). If not set, uses argmax."
)
parser.add_argument("--quant", action="store_true", help="Enable quantization with BitsAndBytesConfig")
parser.add_argument("--lora", action="store_true", help="Enable LoRA fine-tuning using PEFT")

args = parser.parse_args()
DEBUG = args.debug
LLAMA = "llama" in args.model_name.lower()
MAX_SEQ_LENGTH = 8024 if LLAMA else None
focal_loss_fct = None
FL_ALPHA = 2
FL_GAMMA = 5

if args.threshold is not None and not (0.0 <= args.threshold <= 1.0):
    raise ValueError("Threshold must be between 0 and 1 if specified")

# ------------------------- Local model path -------------------------

MODEL_PATH = os.path.join(REPO_PATH, "LLMs", "pretrained", args.model_name)
print(f"🧠 Using model: {args.model_name} from {MODEL_PATH}")

# ------------------------- HF login -------------------------

load_dotenv(dotenv_path= f"{REPO_PATH}/secrets/.env")
hugging_face_token = os.getenv("HUGGING_FACE_TOKEN")
huggingface_hub_login(hugging_face_token)

# ------------------------- focal loss setup -------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(-1)).float()

        focal_term = (1 - probs) ** self.gamma
        loss = -self.alpha * focal_term * log_probs * targets_one_hot

        if self.reduction == 'mean':
            return loss.sum(dim=1).mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ------------------------- Load dataset and fix class imbalance -------------------------

if args.perf_data:
    print("📂 Using performance dataset...")

    dataset = load_dataset("json", data_files=f"{REPO_PATH}/datasets/dataset.jsonl", split="train")
    split_ratio = 0.8
    split_index = int(len(dataset) * split_ratio)

    train_dataset = dataset.select(range(0, split_index))
    eval_dataset = dataset.select(range(split_index, len(dataset)))

    if DEBUG:
        train_dataset = train_dataset.select(range(200))
        eval_dataset = eval_dataset.select(range(200))

    train_dataset = train_dataset.shuffle(seed=42)

    def format_for_classification(example):
        return {
            "text": example['prompt'],
            "label": int(example["response"])
        }

    train_formatted = train_dataset.map(format_for_classification, remove_columns=["prompt", "response"])
    eval_formatted = eval_dataset.map(format_for_classification, remove_columns=["prompt", "response"])

    imbalance_fix = args.class_imbalance_fix

    if imbalance_fix == "weighted_loss":
        print("🔄 Applying weighted loss...")

        train_labels_list = train_formatted['label']
        class_labels = np.array(sorted(set(train_labels_list)))
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=class_labels,
            y=train_labels_list
        )

        class_weights = class_weights.tolist()

        print("📏 Class weights from sklearn (balanced):")
        for label, weight in zip(class_labels, class_weights):
            print(f"  Class {label}: weight = {weight:.4f}")

        dataset = DatasetDict({
            "train": train_formatted,
            "test": eval_formatted
        })

    elif imbalance_fix == "focal_loss":
        print("🔥 Using Focal Loss for class imbalance.")
        focal_loss_fct = FocalLoss(FL_ALPHA, FL_GAMMA)
        dataset = DatasetDict({
            "train": train_formatted,
            "test": eval_formatted
        })

    elif imbalance_fix == "oversampling":
        print("🔄 Applying basic minority oversampling...")
        majority = train_formatted.filter(lambda ex: ex["label"] == 0)
        minority = train_formatted.filter(lambda ex: ex["label"] == 1)
        upsample_size = len(majority)
        minority_upsampled = minority.shuffle(seed=42).select(
            [random.randint(0, len(minority) - 1) for _ in range(upsample_size)]
        )
        balanced_train = concatenate_datasets([majority, minority_upsampled]).shuffle(seed=42)

        dataset = DatasetDict({
            "train": balanced_train,
            "test": eval_formatted
        })
    else:
        dataset = DatasetDict({
            "train": train_formatted,
            "test": eval_formatted
        })

else:
    print("📂 Using IMDb dataset...")
    dataset = load_dataset("imdb")

# Compute class distribution from training data
train_labels = dataset["train"]["label"]
label_counts = Counter(train_labels)
class_distribution = {
    str(label): int(count)
    for label, count in sorted(label_counts.items())
}

# ------------------------- tokenize -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if LLAMA:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_data(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH if MAX_SEQ_LENGTH else tokenizer.model_max_length
    )

tokenized_dataset = dataset.map(tokenize_data, batched=True)
tokenized_dataset.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print("Tokenized dataset features:")
print(tokenized_dataset['train'].features)

# ------------------------- define metrics -------------------------
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

def recall_at_top_k(pred_scores, true_labels, percentages=[0.1, 0.25, 0.5]):
    results = {}
    total_positives = np.sum(true_labels)
    
    sorted_indices = np.argsort(-pred_scores)
    sorted_labels = true_labels[sorted_indices]

    for pct in percentages:
        k = int(len(pred_scores) * pct)
        top_k_labels = sorted_labels[:k]
        recall = np.sum(top_k_labels) / total_positives
        results[f"recall@top_{int(pct * 100)}%"] = recall
    return results

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

    # Simulate gating the top k % of most likely regressions
    recall_at_k_metrics = recall_at_top_k(probs[:, 1], labels, percentages=[0.1, 0.25, 0.5])
    
    if args.threshold is not None:
        threshold = args.threshold
        predictions = (probs[:, 1] >= threshold).astype(int)
    else:
        predictions = np.argmax(logits, axis=1)

    # Class distribution in predictions
    class_counts = Counter(predictions)
    output_distribution = {
        f"pred_class_{label}": int(count)
        for label, count in sorted(class_counts.items())
    }

    metrics = {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "precision": precision.compute(predictions=predictions, references=labels, average="binary")["precision"],
        "recall": recall.compute(predictions=predictions, references=labels, average="binary")["recall"],
        "f1": f1.compute(predictions=predictions, references=labels, average="binary")["f1"],
    }
    metrics.update(output_distribution)
    metrics.update(recall_at_k_metrics)

    return metrics


id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

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

callbacks = []

if args.live_metrics:
    callbacks.append(SaveMetricsCallback(live_metrics_path))
    print(f"📊 Live metrics will be saved to: {live_metrics_path}")
else:
    print("📊 Live metrics logging disabled.")

# ------------------------- Load model and quantization-------------------------
if args.quant and LLAMA:
    print("🧠 Loading model with 4-bit quantization using BitsAndBytesConfig...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        quantization_config=quant_config,
        device_map="auto"
    )
else:
    print("🧠 Loading model without quantization...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.config.pretraining_tp = 1

# ------------------------- LORA -------------------------
if args.lora:
    print("✨ Applying LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=8,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'] if LLAMA else ["query", "value"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

# ------------------------- Training arguments -------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=2 if DEBUG else 32,
    per_device_eval_batch_size=4 if DEBUG else 32,
    gradient_accumulation_steps=2,
    num_train_epochs=1 if DEBUG else 4,
    max_steps=2 if DEBUG else -1,
    weight_decay=0.01,
    logging_strategy="steps",
    logging_steps=1 if DEBUG else 50,
    save_strategy="no" if DEBUG else "steps",
    eval_strategy="steps",
    eval_steps=1 if DEBUG else 50,
    save_steps=1 if DEBUG else 50,
    save_total_limit=1,
    load_best_model_at_end=False if DEBUG else True,
    metric_for_best_model="recall",
    label_names=["labels"],
    max_grad_norm=1.0
)

# ------------------------- Save Config to File -------------------------
config_snapshot = {
    "timestamp": run_timestamp,
    "model_name": args.model_name,
    "class_imbalance_fix": args.class_imbalance_fix,
    "dataset": "perf_data" if args.perf_data else "IMDb",
    "learning_rate": training_args.learning_rate,
    "epochs": training_args.num_train_epochs,
    "train_batch_size": training_args.per_device_train_batch_size,
    "eval_batch_size": training_args.per_device_eval_batch_size,
    "weight_decay": training_args.weight_decay,
    "metric_for_best_model": training_args.metric_for_best_model,
    "class_distribution": class_distribution,
    "decision_threshold": args.threshold if args.threshold is not None else "argmax",
    "quantized": args.quant,
    "lora_enabled": args.lora,
    "focal_loss_gamma": FL_GAMMA if args.class_imbalance_fix == "focal_loss" else "None",
    "focal_loss_alpha": FL_ALPHA if args.class_imbalance_fix == "focal_loss" else "None"
}

with open(config_path, "w") as f:
    json.dump(config_snapshot, f, indent=2)

print(f"⚙️ Logged config to: {config_path}")

# ------------------------- Trainer -------------------------

class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, focal_loss_fct=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float32).to(self.args.device)
            if class_weights is not None else None
        )
        self.focal_loss_fct = focal_loss_fct

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
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

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=callbacks,
    class_weights=class_weights if args.class_imbalance_fix == "weighted_loss" else None,
    focal_loss_fct=focal_loss_fct if args.class_imbalance_fix == "focal_loss" else None
)

trainer.train()

# ---------------------------- Save Metrics and Plot ----------------------------

training_metrics = trainer.state.log_history
metrics_save_path = os.path.join(metrics_dir, "metrics.json")

with open(metrics_save_path, "w") as f:
    json.dump(training_metrics, f, indent=4)

print(f"✅ Saved metrics to {metrics_save_path}")
