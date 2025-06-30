#!/usr/bin/env python3

import os
import argparse
import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    LlamaModel,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

# ----------------- Custom Llama4 Sequence Classification -----------------
class Llama4ForSequenceClassification(PreTrainedModel):
    """
    LLaMA 4 Model transformer with a sequence classification head on top (linear layer).

    This implementation follows the pattern used by Hugging Face for LLaMA 2/3.
    It pools the last non-padding token from the final hidden states and passes
    it through a linear layer to get logits.
    """

    config_class = AutoConfig
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]
        logits = self.score(hidden_states)         # [batch, seq_len, num_labels]

        batch_size = input_ids.shape[0]

        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        else:
            non_pad_mask = (input_ids != self.config.pad_token_id).int()
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# ----------------- Argument parser -----------------
parser = argparse.ArgumentParser(description="Download and save model + tokenizer for sequence classification.")
parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B", help="Hugging Face model ID")
parser.add_argument("--add_classification_head", action="store_true", help="Add custom classification head (e.g. for LLaMA 4)")

args = parser.parse_args()

# ----------------- Paths -----------------
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
save_path = os.path.join(REPO_PATH, "LLMs", "pretrained", args.model_id)

print(f"✅ Detected REPO_PATH: {REPO_PATH}")
print(f"✅ Saving to: {save_path}")

# ----------------- Tokenizer -----------------
tokenizer = AutoTokenizer.from_pretrained(args.model_id)
tokenizer.save_pretrained(save_path)
print("✅ Tokenizer saved.")

# ----------------- Model -----------------
if args.add_classification_head:
    print("✨ Adding custom classification head...")
    config = AutoConfig.from_pretrained(args.model_id)
    config.num_labels = 2  # Or make configurable

    base_model = LlamaModel.from_pretrained(args.model_id, config=config)
    model = Llama4ForSequenceClassification(config)
    model.model.load_state_dict(base_model.state_dict(), strict=False)
else:
    print("🔗 Using standard AutoModelForSequenceClassification...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=2
    )

model.save_pretrained(save_path)
print(f"✅ Model saved to {save_path}")
