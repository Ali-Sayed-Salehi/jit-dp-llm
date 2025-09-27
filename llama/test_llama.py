#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_model,
    LoraConfig,
    PeftType,
    TaskType,
    prepare_model_for_kbit_training,
)
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
)
from tqdm import tqdm

print("Setting configs")
batch_size = 8
model_name_or_path = "/home/alis/links/scratch/repos/jit-dp-llm/LLMs/snapshots/meta-llama/Llama-3.1-8B"
task = "mrpc"
peft_type = PeftType.LORA
device = "cuda"
num_epochs = 1
max_train_steps = 10
lr = 1e-4

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["v_proj", "down_proj", "up_proj", "q_proj", "gate_proj", "k_proj", "o_proj"],
)

padding_side = "left" if any(k in model_name_or_path for k in ("gpt", "opt", "bloom", "llama")) else "right"

print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, padding_side=padding_side, local_files_only=True
)
if getattr(tokenizer, "pad_token_id", None) is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("Loading dataset (offline from disk)")
datasets = load_from_disk("/home/alis/links/scratch/repos/jit-dp-llm/datasets/glue-mrpc")

def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)

print("Tokenizing dataset")
tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "sentence1", "sentence2"],
)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")

# Dataloaders
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
)

q_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("Loading model")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path,
    return_dict=True,
    quantization_config=q_config,
    device_map="auto",
    local_files_only=True,
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

optimizer = AdamW(params=model.parameters(), lr=lr)

# Scheduler for exactly two steps; no warmup
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer, num_warmup_steps=0, num_training_steps=max_train_steps
)

model.config.pad_token_id = tokenizer.pad_token_id

print("Moving model to device")
model.to(device)

print("Training …")
model.train()
eval_iter = iter(eval_dataloader)
steps_done = 0

for step, batch in enumerate(tqdm(train_dataloader, desc="train")):
    if steps_done >= max_train_steps:
        break

    # ---- Train step ----
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    train_loss = outputs.loss

    train_loss.backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    # ---- Eval step (whole validation set) ----
    model.eval()
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for ebatch in eval_dataloader:
            ebatch = {k: v.to(device) for k, v in ebatch.items()}
            eout = model(**ebatch)              # eout.loss is mean loss over the batch
            bs = ebatch["labels"].size(0)
            total_loss += eout.loss.item() * bs # accumulate sum of per-example loss
            total_examples += bs

    eval_loss = total_loss / max(total_examples, 1)
    print(f"step {steps_done+1}: train_loss={train_loss.item():.6f}  eval_loss={eval_loss:.6f}")
    model.train()

    steps_done += 1

print("Done.")
