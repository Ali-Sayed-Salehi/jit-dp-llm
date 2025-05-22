from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import pipeline
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
from dotenv import load_dotenv
import random

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
parser.add_argument("--live_metrics", action="store_true", help="Enable saving evaluation metrics after each eval step")
parser.add_argument("--perf_data", action="store_true", help="Use performance dataset with oversampling instead of IMDb")
parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Name of the model to use")

args = parser.parse_args()
DEBUG = args.debug

# ------------------------- Local model path -------------------------

MODEL_PATH = os.path.join(REPO_PATH, "LLMs", "pretrained", args.model_name)
print(f"🧠 Using model: {args.model_name} from {MODEL_PATH}")

# ------------------------- HF login -------------------------

load_dotenv(dotenv_path= f"{REPO_PATH}/secrets/.env")
hugging_face_token = os.getenv("HUGGING_FACE_TOKEN")
huggingface_hub_login(hugging_face_token)

# ------------------------- Load dataset -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

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

    # Perform minority oversampling
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
    print("📂 Using IMDb dataset...")
    dataset = load_dataset("imdb")


def tokenize_data(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_dataset = dataset.map(tokenize_data, batched=True)
tokenized_dataset.set_format("torch")

print("Tokenized dataset features:")
print(tokenized_dataset['train'].features)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ------------------------- define metrics -------------------------
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "precision": precision.compute(predictions=predictions, references=labels, average="binary")["precision"],
        "recall": recall.compute(predictions=predictions, references=labels, average="binary")["recall"],
        "f1": f1.compute(predictions=predictions, references=labels, average="binary")["f1"]
    }

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

# ------------------------- Load model locally -------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

# ------------------------- Training arguments -------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1 if DEBUG else 2,
    max_steps=2 if DEBUG else -1,
    weight_decay=0.01,
    logging_strategy="steps",
    logging_steps=1 if DEBUG else 10,
    save_strategy="no" if DEBUG else "steps",
    eval_strategy="steps",
    eval_steps=1 if DEBUG else 50,
    save_steps=1 if DEBUG else 50,
    save_total_limit=1,
    load_best_model_at_end=False if DEBUG else True,
    metric_for_best_model="recall"
)

# ------------------------- Trainer -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=callbacks
)

trainer.train()

# ---------------------------- Save Metrics and Plot ----------------------------

training_metrics = trainer.state.log_history
metrics_save_path = os.path.join(metrics_dir, "metrics.json")

with open(metrics_save_path, "w") as f:
    json.dump(training_metrics, f, indent=4)

print(f"✅ Saved metrics to {metrics_save_path}")
