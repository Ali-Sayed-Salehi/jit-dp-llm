import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch.nn as nn
import evaluate
import numpy as np
from dotenv import load_dotenv
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

# ---------------------------- Parse Arguments ----------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
parser.add_argument("--live_metrics", action="store_true", help="Enable saving evaluation metrics after each eval step")
parser.add_argument(
    "--dataset",
    type=str,
    choices=["perf", "jit", "jit_balanced", "imdb", "jit_small"],
    default="imdb",
    help="Choose which dataset to use: 'perf', 'jit', 'jit_balanced', 'jit_small', or 'imdb'"
)

parser.add_argument("--model_name", type=str, default="distilbert/distilbert-base-uncased", help="""Name of the model to use. 
                                                                                    Example: meta-llama/Llama-3.2-3B, 
                                                                                    meta-llama/Meta-Llama-3-8B, 
                                                                                    microsoft/codebert-base,
                                                                                    meta-llama/Llama-3.1-8B, 
                                                                                    distilbert/distilbert-base-uncased""")
parser.add_argument(
    "--class_imbalance_fix",
    type=str,
    choices=["oversampling", "weighted_loss", "focal_loss", "none"],
    default="none",
    help="Class imbalance handling method: 'oversampling', or 'weighted_loss. Default is no fix for class imbalance.'"
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
    help="Metric to select the best model: recall@top_5%, recall@top_10%, f1, precision, recall, accuracy"
)

args = parser.parse_args()
DEBUG = args.debug
LLAMA = "llama" in args.model_name.lower()
focal_loss_fct = None
FL_ALPHA = 2
FL_GAMMA = 5
# what percentile of sequence lengths from the data we use as cut-off limit for tokenizer
SEQ_LEN_PERCENTILE = 95

if args.threshold is not None and not (0.0 <= args.threshold <= 1.0):
    raise ValueError("Threshold must be between 0 and 1 if specified")

VALID_SELECTION_METRICS = {"recall@top_5%", "recall@top_10%", "f1", "precision", "recall", "accuracy"}
if args.selection_metric not in VALID_SELECTION_METRICS:
    raise ValueError(f"Unsupported selection_metric '{args.selection_metric}'. Must be one of {VALID_SELECTION_METRICS}")

# ---------------------------- constants  ----------------------------

REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f"✅ Detected REPO_PATH: {REPO_PATH}")

if args.continue_from_dir:
    output_dir = args.continue_from_dir
    run_timestamp = os.path.basename(os.path.dirname(output_dir)).split("_", 1)[-1]
    print(f"🔁 Resuming from checkpoint in: {output_dir}")
else:
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


# ------------------------- Local model path -------------------------

MODEL_PATH = os.path.join(REPO_PATH, "LLMs", "pretrained", args.model_name)
print(f"🧠 Using model: {args.model_name} from {MODEL_PATH}")

if not os.path.isdir(MODEL_PATH):
    raise ValueError(f"🚫 MODEL_PATH does not exist: {MODEL_PATH}")

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

dataset_name = args.dataset
print(f"📂 Loading dataset: {dataset_name}")

dataset_file_map = {
    "perf": "dataset.jsonl",
    "jit": "jit_dp/apachejit_llm.jsonl",
    "jit_balanced": "jit_dp/apachejit_llm_balanced.jsonl",
    "jit_small": "jit_dp/apachejit_llm_small.jsonl",
    "imdb": None  # will use Hugging Face IMDb dataset
}

if dataset_name in ["perf", "jit_balanced", "jit_small"]:
    dataset_path = os.path.join(REPO_PATH, "datasets", dataset_file_map[dataset_name])
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    n_total = len(dataset)
    n_train = int(n_total * 0.64)
    n_eval = int(n_total * 0.16)

    train_dataset = dataset.select(range(0, n_train))
    eval_dataset = dataset.select(range(n_train, n_train + n_eval))
    test_dataset = dataset.select(range(n_train + n_eval, n_total))

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
    test_formatted = test_dataset.map(format_for_classification, remove_columns=["prompt", "response"])

    imbalance_fix = args.class_imbalance_fix

    # Compute class distribution before class imbalance fix
    original_label_counts = Counter(train_formatted["label"])
    original_class_distribution = {
        str(label): int(count)
        for label, count in sorted(original_label_counts.items())
    }

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
            "test": eval_formatted,
            "final_test": test_formatted
        })

    elif imbalance_fix == "focal_loss":
        print("🔥 Using Focal Loss for class imbalance.")
        focal_loss_fct = FocalLoss(FL_ALPHA, FL_GAMMA)
        dataset = DatasetDict({
            "train": train_formatted,
            "test": eval_formatted,
            "final_test": test_formatted
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
            "test": eval_formatted,
            "final_test": test_formatted
        })
    else:
        dataset = DatasetDict({
            "train": train_formatted,
            "test": eval_formatted,
            "final_test": test_formatted
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
config = AutoConfig.from_pretrained(MODEL_PATH)

if LLAMA:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

def get_token_length(example):
    return {"length": len(tokenizer(example["text"], truncation=False)["input_ids"])}

length_sample = dataset["train"]

lengths_dataset = length_sample.map(get_token_length)
lengths = lengths_dataset["length"]

calculated_max_length = int(np.percentile(lengths, SEQ_LEN_PERCENTILE))

MAX_SEQ_LENGTH = min(calculated_max_length, tokenizer.model_max_length, config.max_position_embeddings)

print(f"""✅ Using max_seq_length={MAX_SEQ_LENGTH}, 
    {SEQ_LEN_PERCENTILE}th percentile = {calculated_max_length}, 
    tokenizer limit = {tokenizer.model_max_length}, 
    model limit = {config.max_position_embeddings}""")


def tokenize_data(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH
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

def recall_at_top_k(pred_scores, true_labels, percentages=[0.05, 0.1, 0.5]):
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
    recall_at_k_metrics = recall_at_top_k(probs[:, 1], labels, percentages=[0.05, 0.1, 0.5])
    
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

if args.gradient_checkpointing:
    model.gradient_checkpointing_enable()

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
    per_device_train_batch_size=1 if DEBUG else 1,
    per_device_eval_batch_size=1 if DEBUG else 1,
    gradient_accumulation_steps=16,
    num_train_epochs=1 if DEBUG else 2,
    max_steps=2 if DEBUG else -1,
    weight_decay=0.01,
    logging_strategy="steps",
    logging_steps=1 if DEBUG else 25,
    save_strategy="no" if DEBUG else "steps",
    eval_strategy="steps",
    eval_steps=1 if DEBUG else 25,
    save_steps=1 if DEBUG else 25,
    save_total_limit=2,
    load_best_model_at_end=False if DEBUG else True,
    metric_for_best_model=args.selection_metric,
    greater_is_better=True,
    label_names=["labels"],
    max_grad_norm=1.0,
    bf16=args.bf16,
    gradient_checkpointing=args.gradient_checkpointing
)

# ------------------------- Save Config to File -------------------------
config_snapshot = {
    "timestamp": run_timestamp,
    "model_name": args.model_name,
    "class_imbalance_fix": args.class_imbalance_fix,
    "dataset": dataset_name,
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
    "debug": DEBUG
}

config_snapshot["model_config"] = model.config.to_dict()

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

torch.cuda.empty_cache()

trainer.train(resume_from_checkpoint= True if args.continue_from_dir else False)

# ---------------------------- Evaluate Best Model and Save ----------------------------

if training_args.load_best_model_at_end:
    best_eval_metrics = trainer.evaluate()

    best_model_metrics_path = os.path.join(metrics_dir, "best_model_metrics.json")
    with open(best_model_metrics_path, "w") as f:
        json.dump(best_eval_metrics, f, indent=4)

    print(f"✅ Saved best model eval metrics to {best_model_metrics_path}")

# ---------------------------- Save Metrics and Plot ----------------------------

training_metrics = trainer.state.log_history
metrics_save_path = os.path.join(metrics_dir, "metrics.json")

with open(metrics_save_path, "w") as f:
    json.dump(training_metrics, f, indent=4)

print(f"✅ Saved metrics to {metrics_save_path}")

# ---------------------------- Run inference on held-out test set ----------------------------

print("\n🧪 Running final inference on held-out test set...")

test_results = trainer.predict(tokenized_dataset["final_test"])
logits = test_results.predictions
labels = test_results.label_ids

# Compute predicted class
probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
preds = np.argmax(probs, axis=1)
threshold = args.threshold

if threshold is not None:
    preds = (probs[:, 1] >= threshold).astype(int)

recall_at_k = recall_at_top_k(probs[:, 1], labels, percentages=[0.05, 0.1, 0.5])

final_metrics = {
    "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
    "precision": precision.compute(predictions=preds, references=labels, average="binary")["precision"],
    "recall": recall.compute(predictions=preds, references=labels, average="binary")["recall"],
    "f1": f1.compute(predictions=preds, references=labels, average="binary")["f1"],
}
final_metrics.update(recall_at_k)

output_payload = {
    "metrics": final_metrics,
    "predictions": preds.tolist(),
    "probabilities": probs[:, 1].tolist(),
    "true_labels": labels.tolist()
}

final_test_metrics_path = os.path.join(metrics_dir, "final_test_results.json")
with open(final_test_metrics_path, "w") as f:
    json.dump(output_payload, f, indent=4)

print(f"📄 Final test set results saved to: {final_test_metrics_path}")
print(json.dumps(final_metrics, indent=4))

