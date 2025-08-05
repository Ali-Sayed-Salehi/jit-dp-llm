import os
import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig
)

from utils import (
    login_to_huggingface
)

# ----------------- Argument parser -----------------
parser = argparse.ArgumentParser(
    description="Download and save a model + tokenizer (Causal LM or Sequence Classification)."
)
parser.add_argument(
    "--model_id",
    type=str,
    default="meta-llama/Llama-3.1-8B",
    help="Hugging Face model ID",
)
parser.add_argument(
    "--model_head",
    choices=["causal-lm", "sequence-classification"],
    default="causal-lm",
    help="Which model head to download: 'causal-lm' or 'sequence-classification'",
)

args = parser.parse_args()

# ----------------- Paths -----------------
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
save_path = os.path.join(REPO_PATH, "LLMs", "pretrained", args.model_head, args.model_id)

print(f"✅ REPO_PATH: {REPO_PATH}")
print(f"✅ Model will be saved to: {save_path}")

# ------------------------- HF login -------------------------
login_to_huggingface(REPO_PATH)

# ----------------- Tokenizer -----------------
print("✨ Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
tokenizer.save_pretrained(save_path)
print("✅ Tokenizer saved.")

# ----------------- Model -----------------
print(f"✨ Downloading {args.model_head} model...")
# config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)

print("🔢 Using 4-bit quantization...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16,
)

if args.model_head == "causal-lm":
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        # quantization_config=quant_config
    )
else:
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        # quantization_config=quant_config
    )

os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
print(f"✅ Model saved to {save_path}")
