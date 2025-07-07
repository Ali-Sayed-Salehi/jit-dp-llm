import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import random
import json
from datetime import datetime

import torch
import numpy as np
from torch.nn import functional as F
from dotenv import load_dotenv
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    DataCollatorWithPadding,
    BitsAndBytesConfig
)

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)

from sequence_classification_utils import (
    parse_training_args,
    setup_training_directories,
    login_to_huggingface,
    load_and_split_dataset,
    compute_class_distribution,
    apply_class_imbalance_strategy,
    estimate_max_sequence_length,
    FocalLoss,
    CustomTrainer,
    SaveMetricsCallback,
    compute_custom_metrics,
    recall_at_top_k,
    run_final_inference,
    evaluate_and_save_best_model,
    save_training_metrics,
    save_training_config,
    setup_live_metrics,
    register_custom_llama_if_needed
    )

# ---------------------------- Parse Arguments ----------------------------
args = parse_training_args()

DEBUG = args.debug
LLAMA = "llama" in args.model_path.lower()
FL_ALPHA = 2
FL_GAMMA = 5
# what percentile of sequence lengths from the data we use as cut-off limit for tokenizer
SEQ_LEN_PERCENTILE = 100
RECALL_AT_TOP_K_PERCENTAGES = [0.05, 0.1, 0.3]
trainer_callbacks = []

# ---------------------------- handle directories  ----------------------------

REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f"✅ Detected REPO_PATH: {REPO_PATH}")

paths = setup_training_directories(REPO_PATH, args.continue_from_dir)

output_dir = paths["output_dir"]
run_timestamp = paths["run_timestamp"]
metrics_dir = paths["metrics_dir"]
tensorboard_dir = paths["tensorboard_dir"]
config_path = paths["config_path"]
live_metrics_path = paths["live_metrics_path"]
finetuned_model_dir = paths["model_dir"]
finetuned_tokenizer_dir = paths["tokenizer_dir"]

# ------------------------- Local model path -------------------------
MODEL_PATH = args.model_path
print(f"✅ Using provided MODEL_PATH: {MODEL_PATH}")

# ------------------------- HF login -------------------------
login_to_huggingface(REPO_PATH)

# ------------------------- Load dataset and fix class imbalance -------------------------

dataset = load_and_split_dataset(
    dataset_name=args.dataset,
    repo_path=REPO_PATH,
    debug=DEBUG
)

imbalance_fix = args.class_imbalance_fix

# Compute class distribution before imbalance fix
original_class_distribution = compute_class_distribution(dataset['train']["label"])

dataset, class_weights, focal_loss_dict = apply_class_imbalance_strategy(
    dataset=dataset,
    strategy=imbalance_fix,
    seed=42,
    alpha=FL_ALPHA,
    gamma=FL_GAMMA
)


# Prepare loss function if needed
focal_loss_fct = None
if imbalance_fix == "focal_loss":
    focal_loss_fct = FocalLoss(**focal_loss_dict)

# Compute class distribution after imbalance fix
class_distribution = compute_class_distribution(dataset["train"]["label"])

# ------------------------- tokenize -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)

if LLAMA:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

MAX_SEQ_LENGTH = estimate_max_sequence_length(
    dataset=dataset,
    tokenizer=tokenizer,
    config=config,
    percentile=SEQ_LEN_PERCENTILE,
    override_max_seq_length=args.max_seq_length
)

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
def custom_metrics(eval_pred):
    return compute_custom_metrics(eval_pred, threshold=args.threshold, percentages=RECALL_AT_TOP_K_PERCENTAGES)

trainer_callbacks.extend(
    setup_live_metrics(args.live_metrics, live_metrics_path)
)

# ------------------------- Load model and quantization-------------------------
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

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
        device_map="auto",
        local_files_only=True
    )
else:
    print("🧠 Loading model without quantization...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        local_files_only=True
    )

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
if LLAMA:
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
    num_train_epochs=1 if DEBUG else 5,
    max_steps=2 if DEBUG else -1,
    weight_decay=0.01,
    logging_strategy="steps",
    logging_steps=1 if DEBUG else 25,
    save_strategy="steps",
    eval_strategy="steps",
    eval_steps=1 if DEBUG else 25,
    save_steps=1 if DEBUG else 25,
    save_total_limit=2,
    load_best_model_at_end= True,
    metric_for_best_model=args.selection_metric,
    greater_is_better=True,
    label_names=["labels"],
    max_grad_norm=1.0,
    bf16=args.bf16,
    gradient_checkpointing=args.gradient_checkpointing
)

# ------------------------- Save Config to File -------------------------
save_training_config(
    config_path=config_path,
    run_timestamp=run_timestamp,
    args=args,
    training_args=training_args,
    class_distribution=class_distribution,
    original_class_distribution=original_class_distribution,
    MAX_SEQ_LENGTH=MAX_SEQ_LENGTH,
    SEQ_LEN_PERCENTILE=SEQ_LEN_PERCENTILE,
    DEBUG=DEBUG,
    dataset=dataset,
    RECALL_AT_TOP_K_PERCENTAGES=RECALL_AT_TOP_K_PERCENTAGES,
    FL_GAMMA=FL_GAMMA,
    FL_ALPHA=FL_ALPHA,
    model_config=model.config.to_dict()
)

# ------------------------- Trainer -------------------------
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    compute_metrics=custom_metrics,
    callbacks=trainer_callbacks,
    class_weights=class_weights if args.class_imbalance_fix == "weighted_loss" else None,
    focal_loss_fct=focal_loss_fct if args.class_imbalance_fix == "focal_loss" else None
)

torch.cuda.empty_cache()

trainer.train(resume_from_checkpoint= True if args.continue_from_dir else False)

# ---------------------------- Evaluate Best Model and Save ----------------------------
evaluate_and_save_best_model(
    trainer=trainer,
    training_args=training_args,
    metrics_dir=metrics_dir,
    adapter_dir=finetuned_model_dir,
    tokenizer_dir=finetuned_tokenizer_dir,
)

# ---------------------------- Save Metrics ----------------------------
save_training_metrics(trainer, metrics_dir, filename="metrics.json")

# ---------------------------- Run inference on held-out test set ----------------------------
run_final_inference(
    trainer=trainer,
    test_dataset=tokenized_dataset["final_test"],
    metrics_dir=metrics_dir,
    percentages=RECALL_AT_TOP_K_PERCENTAGES,
    threshold=args.threshold,
)
