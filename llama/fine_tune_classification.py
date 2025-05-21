#!/usr/bin/env python

import argparse
import os
import random
import torch
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
import numpy as np
from collections import Counter
from huggingface_hub import login as huggingface_hub_login
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoConfig,
    TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorWithPadding, 
    TrainerCallback, LlamaForSequenceClassification
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from scipy.special import softmax
import matplotlib.pyplot as plt
from datetime import datetime
import json

# ---------------------------- constants  ----------------------------

REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f"✅ Detected REPO_PATH: {REPO_PATH}")
run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

output_dir = f"{REPO_PATH}/llama/training/run_{run_timestamp}/output"
tensorboard_dir = f"{REPO_PATH}/llama/training/run_{run_timestamp}/tensorboard"
metrics_dir = f"{REPO_PATH}/llama/training/run_{run_timestamp}/metrics"
live_metrics_path = os.path.join(metrics_dir, "live_metrics.jsonl")
finetuned_model_dir = f"{REPO_PATH}/llama/training/run_{run_timestamp}/model"
finetuned_tokenizer_dir = f"{REPO_PATH}/llama/training/run_{run_timestamp}/tokenizer"

training_dirs = [output_dir, tensorboard_dir, metrics_dir, finetuned_model_dir, finetuned_tokenizer_dir]

for directory in training_dirs:
    os.makedirs(directory, exist_ok=True)

# ---------------------------- Parse Arguments ----------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
parser.add_argument("--eval", action="store_true", help="Run evaluation only")
parser.add_argument("--model_path", "--model-path", dest="model_path", type=str, help="Path to the finetuned model folder")
parser.add_argument("--use_weighted_loss", action="store_true", help="Use weighted CrossEntropyLoss instead of oversampling")
parser.add_argument("--live_metrics", action="store_true", help="Enable saving evaluation metrics after each eval step")

args = parser.parse_args()
DEBUG = args.debug

# ---------------------------- GPU and CPU Setup ----------------------------

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("✅ Checking available GPUs...")

if torch.cuda.is_available():
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("🚨 No GPU available! Running on CPU.")

if dist.is_available() and dist.is_initialized():
    print(f"Running distributed training ✅")
    print(f"World size (total processes): {dist.get_world_size()}")
    print(f"My rank: {dist.get_rank()}")
    print(f"My local rank: {int(os.environ.get('LOCAL_RANK', -1))}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
else:
    print(f"Not running distributed training ❌")

cpu_count = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
print(f"Detected {cpu_count} CPU cores from SLURM allocation.")


# ---------------------------- Hugging Face Login ----------------------------

load_dotenv(dotenv_path= f"{REPO_PATH}/secrets/.env")
hugging_face_token = os.getenv("HUGGING_FACE_TOKEN")
huggingface_hub_login(hugging_face_token)

# ---------------------------- BitsAndBytesConfig ----------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# ---------------------------- LoRA config ----------------------------

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type='SEQ_CLS'
)


# ---------------------------- Load Dataset ----------------------------

dataset = load_dataset("json", data_files=f"{REPO_PATH}/datasets/dataset.jsonl", split="train")
split_ratio = 0.8
split_index = int(len(dataset) * split_ratio)

train_dataset = dataset.select(range(0, split_index))
eval_dataset = dataset.select(range(split_index, len(dataset)))

# Debug: reduce to 200 samples each
if DEBUG:
    train_dataset = train_dataset.select(range(200))
    eval_dataset = eval_dataset.select(range(200))

# Shuffle train set only
train_dataset = train_dataset.shuffle(seed=42)

# ---------------------------- Format for Classification ----------------------------

def format_for_classification(example):
    return {
        "text": example['prompt'],
        "label": int(example["response"])
    }

train_formatted_dataset = train_dataset.map(format_for_classification)
eval_formatted_dataset = eval_dataset.map(format_for_classification)

# ---------------------------- Balance Classes ----------------------------

if args.use_weighted_loss:
    print("🔁 Computing class weights for CrossEntropyLoss...")

    label_counts = Counter(train_formatted_dataset['label'])
    total = sum(label_counts.values())
    class_weights = torch.tensor(
        [total / label_counts[i] for i in sorted(label_counts.keys())],
        dtype=torch.float32
    )

    train_dataset_final = train_formatted_dataset

    print(f"Class weights: {class_weights}")
else:
    # Apply minority oversampling
    majority_class = train_formatted_dataset.filter(lambda example: example['label'] == 0)
    minority_class = train_formatted_dataset.filter(lambda example: example['label'] == 1)

    n_majority = len(majority_class)
    n_minority = len(minority_class)
    RESAMPLING_SIZE = 1

    minority_upsampled_size = int(n_majority / RESAMPLING_SIZE)
    minority_upsampled = minority_class.shuffle(seed=42).select(
        [random.randint(0, n_minority - 1) for _ in range(minority_upsampled_size)]
    )

    balanced_dataset = concatenate_datasets([majority_class, minority_upsampled])
    train_balanced_formatted_dataset = balanced_dataset.shuffle(seed=42)
    print("Label distribution after upsampling:", Counter(balanced_dataset['label']))

    train_dataset_final = train_balanced_formatted_dataset

# ---------------------------- loss function ----------------------------

class LlamaForWeightedClassification(LlamaForSequenceClassification):
    def __init__(self, config, class_weights=None):
        super().__init__(config)
        self.class_weights = class_weights
        self.loss_fn = CrossEntropyLoss(weight=class_weights)

    def forward(self, *args, labels=None, **kwargs):
        output = super().forward(*args, labels=labels, **kwargs)
        if labels is not None and self.class_weights is not None:
            loss = self.loss_fn(output.logits, labels)
            return output.__class__(loss=loss, logits=output.logits, **output)
        else:
            return output

def instatiate_model(model_path):
    if args.use_weighted_loss:
        return LlamaForWeightedClassification.from_pretrained(
            model_path,
            num_labels=2,
            device_map="auto",
            quantization_config=bnb_config,
            class_weights=class_weights.to("cuda" if torch.cuda.is_available() else "cpu")
        )
    else:
        return AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            device_map="auto",
            quantization_config=bnb_config
        )

# ---------------------------- Model & Tokenizer ----------------------------

if args.eval and args.model_path:
    # ---------------------------- Load fine-tuned model + adapter for eval ----------------------------
    from peft import PeftModel

    print(f"📦 Loading fine-tuned model from {args.model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, add_prefix_space=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model_path = f"{REPO_PATH}/LLMs/pretrained/llama3-8b"
    base_model = instatiate_model(base_model_path)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1

    model = PeftModel.from_pretrained(base_model, args.model_path)
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

else:
    # ---------------------------- Model & Tokenizer for training ----------------------------
    MODEL_PATH = f"{REPO_PATH}/LLMs/pretrained/llama3-8b"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, add_prefix_space=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = instatiate_model(MODEL_PATH)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

# ---------------------------- Tokenization ----------------------------

def tokenize_data(example):
    encoding = tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
    encoding["labels"] = example["label"]
    return encoding

dataset = DatasetDict({
    'train': train_dataset_final,
    'eval': eval_formatted_dataset
})

tokenized_dataset = dataset.map(tokenize_data, batched=True, remove_columns=['text', 'label'])
tokenized_dataset.set_format("torch")

print("Tokenized dataset features:")
print(tokenized_dataset['train'].features)

# ---------------------------- Metrics ----------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    print("logits shape:", logits.shape)
    probs = softmax(logits, axis=1)
    probs_class1 = probs[:, 1]

    if np.isnan(probs_class1).any():
        raise ValueError("NaNs found in predicted probabilities")

    THRESHOLD = 0.3
    preds = (probs_class1 >= THRESHOLD).astype(int)

    fpr, tpr, _ = roc_curve(labels, probs_class1)
    auc_score = auc(fpr, tpr)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="binary"),
        "recall": recall_score(labels, preds, average="binary"),
        "f1": f1_score(labels, preds, average="binary"),
        "auc": auc_score
        # "fpr": fpr.tolist(),
        # "tpr": tpr.tolist()
    }

class SaveMetricsCallback(TrainerCallback):
    def __init__(self, output_path):
        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            with open(self.output_path, "a") as f:
                json.dump({
                    "step": state.global_step,
                    "metrics": metrics
                }, f)
                f.write("\n")

callbacks = []

if args.live_metrics:
    callbacks.append(SaveMetricsCallback(live_metrics_path))
    print(f"📊 Live metrics will be saved to: {live_metrics_path}")
else:
    print("📊 Live metrics logging disabled.")



# ---------------------------- Training Arguments ----------------------------

training_args = TrainingArguments(
    output_dir=output_dir,
    logging_dir=tensorboard_dir,
    report_to="tensorboard",
    per_device_train_batch_size=2 if DEBUG else 32,
    per_device_eval_batch_size=2 if DEBUG else 32,
    learning_rate=2e-5,
    max_grad_norm=1.0,
    bf16=True,
    dataloader_num_workers=cpu_count,
    num_train_epochs=1 if DEBUG else 1,
    max_steps=5 if DEBUG else -1,
    logging_steps=1,
    weight_decay = 0.01,
    save_strategy="no" if DEBUG else "steps",
    eval_strategy="steps",
    eval_steps=1 if DEBUG else 50,
    save_steps=1 if DEBUG else 50,
    save_total_limit=1,
    load_best_model_at_end=False if DEBUG else True,
    metric_for_best_model="recall"
)

# ---------------------------- Train ----------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['eval'],
    tokenizer=tokenizer,
    data_collator = collate_fn,
    compute_metrics=compute_metrics,
    callbacks=callbacks
)

if args.eval:
    print("🧪 Running evaluation only...")
    metrics = trainer.evaluate()
    print(metrics)
else:
    print("🚀 Starting training...")
    trainer.train()

# ---------------------------- Save Metrics and Plot ----------------------------

# Save metrics from trainer
training_metrics = trainer.state.log_history
metrics_save_path = os.path.join(metrics_dir, "metrics.json")

with open(metrics_save_path, "w") as f:
    json.dump(training_metrics, f, indent=4)

print(f"✅ Saved metrics to {metrics_save_path}")

# ---------------------------- Save Final Model ----------------------------

if not DEBUG:
    trainer.save_model(finetuned_model_dir)
    tokenizer.save_pretrained(finetuned_tokenizer_dir)
