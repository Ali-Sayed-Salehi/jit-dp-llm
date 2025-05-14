#!/usr/bin/env python

import argparse
import os
import random
import torch
import torch.distributed as dist
import numpy as np
from collections import Counter
from huggingface_hub import login as huggingface_hub_login
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoConfig,
    TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorWithPadding
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
finetuned_model_dir = f"{REPO_PATH}/llama/training/run_{run_timestamp}/model"
finetuned_tokenizer_dir = f"{REPO_PATH}/llama/training/run_{run_timestamp}/tokenizer"

training_dirs = [output_dir, tensorboard_dir, metrics_dir, finetuned_model_dir, finetuned_tokenizer_dir]

for directory in training_dirs:
    os.makedirs(directory, exist_ok=True)

# ---------------------------- Parse Arguments ----------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
parser.add_argument("--eval", action="store_true", help="Run evaluation only")

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

# ---------------------------- Model & Tokenizer ----------------------------

MODEL_PATH = f"{REPO_PATH}/LLMs/pretrained/llama3-8b"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, add_prefix_space=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# config = AutoConfig.from_pretrained(MODEL_PATH, num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=2,
    device_map="auto",
    quantization_config=bnb_config
    # trust_remote_code=True,
    # ignore_mismatched_sizes=True
)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.config.pretraining_tp = 1

collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

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

print("Label distribution:", Counter(balanced_dataset['label']))

# ---------------------------- Tokenization ----------------------------

def tokenize_class(example):
    encoding = tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
    encoding["labels"] = example["label"]
    return encoding

dataset = DatasetDict({
    'train': train_balanced_formatted_dataset,
    'eval': eval_formatted_dataset
})

tokenized_dataset = dataset.map(tokenize_class, batched=True, remove_columns=['text'])
tokenized_dataset.set_format("torch")

# ---------------------------- Apply LoRA ----------------------------

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type='SEQ_CLS'
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---------------------------- Metrics ----------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    print("logits shape:", logits.shape)
    probs = softmax(logits, axis=1)
    probs_class1 = probs[:, 1]

    if np.isnan(probs_class1).any():
        raise ValueError("NaNs found in predicted probabilities")

    THRESHOLD = 0.5
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
    save_strategy="no" if DEBUG else "epoch",
    eval_strategy="steps",
    eval_steps=1 if DEBUG else 10,
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
    compute_metrics=compute_metrics
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
