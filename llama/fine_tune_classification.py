#!/usr/bin/env python

import argparse
import os
import random
import torch
import numpy as np
from collections import Counter
from huggingface_hub import login as huggingface_hub_login
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoConfig,
    TrainingArguments, Trainer, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.special import softmax

# ---------------------------- constants  ----------------------------

REPO_PATH = "/speed-scratch/a_s87063/repos/perf-pilot"

# ---------------------------- Parse Arguments ----------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
parser.add_argument("--eval", action="store_true", help="Run evaluation only")

args = parser.parse_args()
DEBUG = args.debug

# ---------------------------- GPU Setup ----------------------------

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
config = AutoConfig.from_pretrained(MODEL_PATH, num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    config=config,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
    ignore_mismatched_sizes=True
)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# ---------------------------- Load Dataset ----------------------------

dataset = load_dataset("json", data_files=f"{REPO_PATH}/datasets/dataset.jsonl", split="train")
split_ratio = 0.8
split_index = int(len(dataset) * split_ratio)

train_dataset = dataset.select(range(0, split_index))
eval_dataset = dataset.select(range(split_index, len(dataset)))

# Debug: reduce to 20 samples each
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

train_tokenized_dataset = train_balanced_formatted_dataset.map(tokenize_class, batched=True)
eval_tokenized_dataset = eval_formatted_dataset.map(tokenize_class, batched=True)

# ---------------------------- Apply LoRA ----------------------------

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---------------------------- Metrics ----------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    print("logits shape:", logits.shape)
    probs = softmax(logits, axis=1)
    probs_class1 = probs[:, 1]

    # if np.isnan(probs_class1).any():
    #     raise ValueError("NaNs found in predicted probabilities")

    THRESHOLD = 0.4
    preds = (probs_class1 >= THRESHOLD).astype(int)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="binary"),
        "recall": recall_score(labels, preds, average="binary"),
        "f1": f1_score(labels, preds, average="binary")
        # "auc": roc_auc_score(labels, probs_class1),
    }

# ---------------------------- Training Arguments ----------------------------

output_dir = f"{REPO_PATH}/llama/debug_output" if DEBUG else f"{REPO_PATH}/llama/training/output/llama-classifier"
logging_dir = f"{REPO_PATH}/llama/debug_logs" if DEBUG else f"{REPO_PATH}/llama/training/logs"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1 if DEBUG else 2,
    per_device_eval_batch_size=1 if DEBUG else 2,
    learning_rate=1e-6,
    max_grad_norm=1.0, # gradient clipping
    bf16=True,
    dataloader_num_workers=0,
    num_train_epochs=1 if DEBUG else 3,
    max_steps=10 if DEBUG else -1,
    logging_steps=1,
    save_strategy="no" if DEBUG else "epoch",
    eval_strategy="steps" if DEBUG else "epoch",
    eval_steps=2 if DEBUG else None,
    save_total_limit=1,
    load_best_model_at_end=False if DEBUG else True,
    metric_for_best_model="recall"
)

# ---------------------------- Train ----------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_dataset,
    eval_dataset=eval_tokenized_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

if args.eval:
    print("🧪 Running evaluation only...")
    metrics = trainer.evaluate()
    print(metrics)
else:
    print("🚀 Starting training...")
    trainer.train()

    for name, param in model.named_parameters():
        if param.requires_grad and torch.isnan(param).any():
            print(f"NaN detected in model parameter after training: {name}")

# ---------------------------- Save Final Model ----------------------------

if not DEBUG:
    trainer.save_model(f"{REPO_PATH}/LLMs/finetuned/classification/model/llama3-8B")
    tokenizer.save_pretrained(f"{REPO_PATH}/LLMs/finetuned/classification/tokenizer/llama3-8B")
