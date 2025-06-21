from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

model_id = "meta-llama/Llama-3.1-8B"

REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f"✅ Detected REPO_PATH: {REPO_PATH}")
save_path = os.path.join(REPO_PATH, "LLMs", "pretrained", model_id)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(save_path)

model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2
    )
model.save_pretrained(save_path)

print(f"✅ model and tokenizer saved to {save_path}")