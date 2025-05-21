from huggingface_hub import notebook_login
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import pipeline
from dotenv import load_dotenv
import os

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

args = parser.parse_args()
DEBUG = args.debug

# ------------------------- Local model path -------------------------

MODEL_PATH = f"{REPO_PATH}/LLMs/pretrained/distilbert-base-uncased"

# ------------------------- HF login -------------------------

load_dotenv(dotenv_path= f"{REPO_PATH}/secrets/.env")
hugging_face_token = os.getenv("HUGGING_FACE_TOKEN")
huggingface_hub_login(hugging_face_token)

# ------------------------- Load dataset -------------------------
imdb = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_imdb = imdb.map(preprocess_function, batched=True)

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
    max_steps=5 if DEBUG else -1,
    weight_decay=0.01,
    save_strategy="no" if DEBUG else "steps",
    eval_strategy="steps",
    eval_steps=1 if DEBUG else 50,
    save_steps=1 if DEBUG else 50,
    save_total_limit=1,
    load_best_model_at_end=True,
    push_to_hub=False
)

# ------------------------- Trainer -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# ---------------------------- Save Metrics and Plot ----------------------------

training_metrics = trainer.state.log_history
metrics_save_path = os.path.join(metrics_dir, "metrics.json")

with open(metrics_save_path, "w") as f:
    json.dump(training_metrics, f, indent=4)

print(f"✅ Saved metrics to {metrics_save_path}")
