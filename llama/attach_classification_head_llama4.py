import os
import argparse
import torch
from torch import nn
from functools import wraps

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    Llama4TextConfig,
    Llama4TextModel,
    Llama4PreTrainedModel
)
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

# ✅ Local version of 🤗's can_return_tuple
def can_return_tuple(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return_dict = self.config.return_dict if hasattr(self, "config") else True
        return_dict_passed = kwargs.pop("return_dict", return_dict)
        if return_dict_passed is not None:
            return_dict = return_dict_passed
        output = func(self, *args, **kwargs)
        if not return_dict and not isinstance(output, tuple):
            output = output.to_tuple()
        return output
    return wrapper

class CustomLlama4TextConfig(Llama4TextConfig):
    model_type = "custom-llama4-classification"

class CustomLlama4ForSequenceClassification(Llama4PreTrainedModel):
    config_class = CustomLlama4TextConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Llama4TextModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
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

        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]

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

        # loss is not calculated because it is overridden by compute_loss in CustomTrainer class

        return SequenceClassifierOutputWithPast(
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_lm_path", type=str, required=True, help="Path to base LLaMA Causal LM")
    parser.add_argument("--save_path", type=str, required=True, help="Root path to save the new classification model")
    args = parser.parse_args()

    path_parts = os.path.normpath(args.base_lm_path).split(os.sep)
    model_name = path_parts[-1]
    org_name = path_parts[-2]

    SAVE_PATH = os.path.join(args.save_path, org_name, model_name)

    # ✅ Register the custom config + model class
    AutoConfig.register(CustomLlama4TextConfig.model_type, CustomLlama4TextConfig)
    AutoModelForSequenceClassification.register(CustomLlama4TextConfig, CustomLlama4ForSequenceClassification)

    # ✅ Load base Causal LM and get its backbone weights
    print(f"🔍 Loading base Causal LM: {args.base_lm_path}")
    causal_lm = AutoModelForCausalLM.from_pretrained(args.base_lm_path, local_files_only=True)
    backbone = causal_lm.model  # type: Llama4Model

    # ✅ Build your custom config
    base_config = AutoConfig.from_pretrained(args.base_lm_path, local_files_only=True)
    config = CustomLlama4TextConfig.from_dict(base_config.to_dict())
    config.model_type = CustomLlama4TextConfig.model_type
    config.architectures = ["CustomLlama4ForSequenceClassification"]
    config.num_labels = 2
    config.pad_token_id = base_config.pad_token_id

    # ✅ Create your classification model
    final_model = CustomLlama4ForSequenceClassification(config)

    # ✅ COPY the backbone weights
    final_model.model.load_state_dict(backbone.state_dict())
    print(f"✅ Backbone weights copied from Causal LM!")

    # ✅ Save the final model + config
    os.makedirs(SAVE_PATH, exist_ok=True)
    final_model.save_pretrained(SAVE_PATH)
    config.save_pretrained(SAVE_PATH)

    # ✅ Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_lm_path)
    tokenizer.save_pretrained(SAVE_PATH)

    print(f"✅ New classification model + tokenizer saved to: {SAVE_PATH}")

if __name__ == "__main__":
    main()
