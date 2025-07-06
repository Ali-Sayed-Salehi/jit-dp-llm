#!/usr/bin/env python3
"""
Script to fine-tune a local causal LM using Unsloth, PEFT LoRA, and Hugging Face Trainer.

It uses only the 'prompt' part of your dataset for next-token prediction.
LoRA weights are merged before saving the final model.

Requirements:
  pip install unsloth peft transformers datasets
"""

import os
import sys
import argparse
from datetime import datetime

import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)
from unsloth import patch_transformers  # ✅ Patches transformers for speed

# ------------------------------ Unsloth Patch ------------------------------
patch_transformers()

# ------------------------------ CLI Arguments ------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Causal LM with Unsloth and LoRA.")
    parser.add_argument("--model_path", type=str, required=True, help="Local path to pretrained causal LM")
    parser.add_argument("--dataset", type=str, default="imdb", help="Dataset name: 'imdb' or custom")
    parser.add_argument("--output_dir", type=str, default="./finetuned_lm", help="Output directory for saving model")
    parser.add_argument("--lora", action="store_true", help="Enable LoRA fine-tuning")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    return parser.parse_args()

args = parse_args()

# ------------------------------ Load Model & Tokenizer ------------------------------
print(f"✅ Loading model from: {args.model_path}")
tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)

# ------------------------------ Load Dataset ------------------------------
if args.dataset == "imdb":
    dataset = load_dataset("imdb")
    dataset = DatasetDict({
        "train": dataset["train"],
        "test": dataset["test"],
    })
else:
    dataset = load_dataset("json", data_files=args.dataset)

# Use only the 'prompt' field if present, else default to 'text'
def format_for_lm(example):
    if "prompt" in example:
        return {"text": example["prompt"]}
    return {"text": example.get("text", "")}

dataset = dataset.map(format_for_lm, remove_columns=dataset["train"].column_names)

# ------------------------------ Tokenize ------------------------------
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized.set_format("torch")

# ------------------------------ Data Collator ------------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # For causal LM, MLM is False
)

# ------------------------------ Load Model & Apply LoRA ------------------------------
from transformers import AutoModelForCausalLM

print("✅ Loading model...")
model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True)

if args.lora:
    print("✨ Applying LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # typical for LLaMA
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

# ------------------------------ Training Arguments ------------------------------
training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    bf16=True,
    gradient_checkpointing=True,
)

# ------------------------------ Trainer ------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized.get("test"),
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ------------------------------ Train ------------------------------
trainer.train()

# ------------------------------ Merge LoRA Weights & Save ------------------------------
if args.lora:
    print("✅ Merging LoRA weights back to base model...")
    model = model.merge_and_unload()

save_dir = os.path.join(args.output_dir, f"merged-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"✅ Final model saved to: {save_dir}")
