#!/usr/bin/env python3

import os
import argparse
import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutputWithPast


class CustomLlamaForSequenceClassification(PreTrainedModel):
    """
    Custom LLaMA model with a sequence classification head.
    To run:
        python attach_llama_classification_head.py \
        --base_lm_path /models/meta-llama/Llama-3.1-8B \
        --save_path repo_paths/LLMs/pretrained/sequence-classification/custom
    """

    config_class = AutoConfig
    base_model_prefix = "model"
    model_type = "llama-sequence-classification"

    def __init__(self, config, transformer_backbone):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = transformer_backbone
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")

        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            non_pad_mask = (input_ids != self.config.pad_token_id).int()
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(pooled_logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Attach a sequence classification head to a LLaMA Causal LM model."
    )
    parser.add_argument(
        "--base_lm_path",
        type=str,
        required=True,
        help="Path to the base LLaMA Causal LM model."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save the new model with the classification head."
    )
    args = parser.parse_args()

    # ---------- Defaults ----------
    num_labels = 2
    model_type = "llama-sequence-classification"
    architecture_name = "CustomLlamaForSequenceClassification"

    # Get model_id as last two path segments
    norm_base = os.path.normpath(args.base_lm_path)
    path_parts = norm_base.split(os.sep)
    if len(path_parts) >= 2:
        model_id = os.path.join(path_parts[-2], path_parts[-1])
    else:
        model_id = path_parts[-1]

    SAVE_PATH = os.path.join(args.save_path, model_id)
    os.makedirs(SAVE_PATH, exist_ok=True)
    print(f"Using save path: {SAVE_PATH}")

    # ---------- Register custom class ----------
    AutoConfig.register(model_type, AutoConfig)
    AutoModelForSequenceClassification.register(AutoConfig, CustomLlamaForSequenceClassification)
    print(f"✅ Registered CustomLlamaForSequenceClassification with model_type '{model_type}'.")

    # ---------- Load base LM ----------
    config = AutoConfig.from_pretrained(args.base_lm_path)
    config.num_labels = num_labels
    config.model_type = model_type
    config.architectures = [architecture_name]

    causal_lm = AutoModelForCausalLM.from_pretrained(args.base_lm_path, config=config)
    transformer_backbone = causal_lm.model

    # ---------- Attach head ----------
    final_model = CustomLlamaForSequenceClassification(config, transformer_backbone)

    # ---------- Save model ----------
    final_model.save_pretrained(SAVE_PATH)
    config.save_pretrained(SAVE_PATH)

    # ---------- Save tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(args.base_lm_path)
    tokenizer.save_pretrained(SAVE_PATH)
    print(f"✅ Tokenizer saved to: {SAVE_PATH}")

    print(f"✅ New model with classification head saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()
