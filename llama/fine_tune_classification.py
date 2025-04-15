#!/usr/bin/env python

from huggingface_hub import login as huggingface_hub_login
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig, AutoConfig, AutoModelForSequenceClassification
import torch
from peft import LoraConfig, get_peft_model, TaskType
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pprint import pprint
from collections import Counter
import random

print(f"Rank {os.environ.get('RANK')}: CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"Rank {os.environ.get('RANK')}: torch.cuda.device_count() = {torch.cuda.device_count()}")


# GPU memory preparations
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# Log into Hugging Face
load_dotenv(dotenv_path="../secrets/.env")
hugging_face_token = os.getenv("HUGGING_FACE_TOKEN")
huggingface_hub_login(hugging_face_token)


# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)


# Load model and tokenizer
MODEL_PATH = "../LLMs/pretrained/llama3-8b"
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


# Split the dataset into training and evaluation
dataset = load_dataset("json", data_files="../datasets/dataset.jsonl", split="train")

split_ratio = 0.8
split_index = int(len(dataset) * split_ratio)

train_dataset = dataset.select(range(0, split_index))
eval_dataset = dataset.select(range(split_index, len(dataset)))


# Shuffle ONLY the training set
train_dataset = train_dataset.shuffle(seed=42)

train_dataset.to_json("../datasets/train.jsonl", orient="records", lines=True)
eval_dataset.to_json("../datasets/eval.jsonl", orient="records", lines=True)


# Format datasets for classification
def format_for_classification(example):
    return {
        "text": example['prompt'],
        "label": int(example["response"])
    }

train_formatted_dataset = train_dataset.map(format_for_classification)
eval_formatted_dataset = eval_dataset.map(format_for_classification)


# Upsample the minority class to address class imbalance
majority_class = train_formatted_dataset.filter(lambda example: example['label'] == 0)
minority_class = train_formatted_dataset.filter(lambda example: example['label'] == 1)

n_majority = len(majority_class)
n_minority = len(minority_class)

RESAMPLING_SIZE = 5     # creates 1 to 5 class imbalance
minority_upsampled_size = int(n_majority / RESAMPLING_SIZE)

minority_upsampled = minority_class.shuffle(seed=42).select(
    [random.randint(0, n_minority - 1) for _ in range(minority_upsampled_size)]
)

balanced_dataset = concatenate_datasets([majority_class, minority_upsampled])

train_balanced_formatted_dataset = balanced_dataset.shuffle(seed=42)

print(Counter(balanced_dataset['label']))


# Tokenize datasets
def tokenize_class(example):
    encoding = tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
    encoding["labels"] = example["label"]
    return encoding

train_tokenized_dataset = train_balanced_formatted_dataset.map(tokenize_class, batched=True)
eval_tokenized_dataset = eval_formatted_dataset.map(tokenize_class, batched=True)



# Apply LoRA with PEFT
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


# Define evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    probs_class1 = probs[:, 1]
    THRESHOLD = 0.4
    preds = (probs_class1 >= threshold).astype(int)
    
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="binary"), # Of all samples predicted as class 1, how many were actually 1?
        "recall": recall_score(labels, preds, average="binary"), # Of all actual class 1 samples, how many did the model correctly predict?
        "f1": f1_score(labels, preds, average="binary"),
        "auc": roc_auc_score(labels, probs_class1),
    }


# Set Up TrainingArguments and Trainer
training_args = TrainingArguments(
    output_dir="./training/output/llama-classifier",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    bf16=True,  # Use mixed-precision training to reduce memory usage
    gradient_accumulation_steps=4,  # simulates a larger batch size without needing more memory
    dataloader_num_workers=4,
    num_train_epochs=2,
    learning_rate=2e-4,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./training/logs",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="recall"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_dataset,
    eval_dataset=eval_tokenized_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


# Train the model
trainer.train()


# Save the fine-tuned model
trainer.save_model("../LLMs/finetuned/classification/model/llama3-8B")
tokenizer.save_pretrained("../LLMs/finetuned/classification/tokenizer/llama3-8B")


# Test the fine-tuned model
# predictions = trainer.predict(eval_tokenized_dataset)
# logits = predictions.predictions
# probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)

# print("Probabilities for class 1:", probs[:, 1])

