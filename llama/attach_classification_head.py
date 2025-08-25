import os
import argparse
import torch
from torch import nn
from functools import wraps

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Llama4TextConfig,
    Llama4TextModel,
    Llama4PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.modeling_utils import load_sharded_checkpoint


# ---- HF-compatible helper ----
def can_return_tuple(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return_dict = getattr(self.config, "return_dict", True)
        return_dict_passed = kwargs.pop("return_dict", return_dict)
        if return_dict_passed is not None:
            return_dict = return_dict_passed
        output = func(self, *args, **kwargs)
        if not return_dict and not isinstance(output, tuple):
            output = output.to_tuple()
        return output
    return wrapper


# ---- Custom LLaMA4 classifier head ----
class CustomLlama4TextConfig(Llama4TextConfig):
    model_type = "custom-llama4-classification"


class CustomLlama4ForSequenceClassification(Llama4PreTrainedModel):
    config_class = CustomLlama4TextConfig

    base_model_prefix = "model"
    _no_split_modules = ["Llama4TextDecoderLayer"]
    _tied_weights_keys = ["model.embed_tokens.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2  # always binary
        self.model = Llama4TextModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        nn.init.normal_(self.score.weight, mean=0.0, std=0.02)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

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

        if input_ids is not None:
            batch_size = input_ids.size(0)
        else:
            batch_size = inputs_embeds.size(0)

        pad_id = self.config.pad_token_id
        if pad_id is None and batch_size != 1:
            raise ValueError("Batch > 1 requires pad_token_id to be set.")

        if pad_id is None:
            pooled_logits = logits[:, -1, :]
        else:
            non_pad_mask = (input_ids != pad_id).int()
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
            pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        return SequenceClassifierOutputWithPast(
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


def _derive_seq_cls_config_from_snapshot(snapshot_dir: str, is_llama4: bool):
    if not os.path.isdir(snapshot_dir):
        raise FileNotFoundError(f"seq_cls_config_path (snapshot) not found: {snapshot_dir}")

    try:
        if is_llama4:
            base_cfg = Llama4TextConfig.from_pretrained(snapshot_dir, local_files_only=True)
            class_cfg = CustomLlama4TextConfig(**base_cfg.to_dict())
            class_cfg.architectures = ["CustomLlama4ForSequenceClassification"]
            class_cfg.model_type = CustomLlama4TextConfig.model_type
            class_cfg.num_labels = 2
            return class_cfg, "llama4"
        else:
            base_cfg = AutoConfig.from_pretrained(snapshot_dir, local_files_only=True)
            base_cfg.num_labels = 2
            return base_cfg, "auto"
    except Exception as e:
        raise RuntimeError(
            f"Failed to derive a sequence-classification config from snapshot at {snapshot_dir}: {e}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_lm_path", type=str, required=True,
                        help="Path to base Causal LM (folder with config/model/tokenizer).")
    parser.add_argument("--seq_cls_config_path", type=str, required=True,
                        help="Path to a *CAUSAL LM snapshot directory*. We'll derive a seq-cls config from this.")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Root path to save the new classification model.")
    parser.add_argument("--llama4", action="store_true",
                        help="Set if backbone is LLaMA4.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    path_parts = os.path.normpath(args.base_lm_path).split(os.sep)
    model_name = path_parts[-1]
    org_name = path_parts[-2] if len(path_parts) >= 2 else "unknown"
    SAVE_PATH = os.path.join(args.save_path, org_name, model_name)
    os.makedirs(SAVE_PATH, exist_ok=True)

    base_tokenizer = AutoTokenizer.from_pretrained(args.base_lm_path, local_files_only=True)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
    target_vocab_size = len(base_tokenizer)

    class_config, mode = _derive_seq_cls_config_from_snapshot(
        args.seq_cls_config_path, is_llama4=args.llama4
    )

    if mode == "llama4":
        final_model = CustomLlama4ForSequenceClassification(class_config)
    else:
        final_model = AutoModelForSequenceClassification.from_config(class_config)

    final_model.config.pad_token_id = base_tokenizer.pad_token_id
    if getattr(base_tokenizer, "eos_token_id", None) is not None:
        final_model.config.eos_token_id = base_tokenizer.eos_token_id
    if getattr(base_tokenizer, "bos_token_id", None) is not None:
        final_model.config.bos_token_id = base_tokenizer.bos_token_id
    if getattr(base_tokenizer, "unk_token_id", None) is not None:
        final_model.config.unk_token_id = base_tokenizer.unk_token_id

    if final_model.get_input_embeddings().weight.size(0) != target_vocab_size:
        final_model.resize_token_embeddings(target_vocab_size)
    final_model.config.vocab_size = target_vocab_size

    load_sharded_checkpoint(final_model, args.base_lm_path, strict=False)

    if args.dtype == "bf16":
        final_model.to(torch.bfloat16)
        final_model.config.torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        final_model.to(torch.float16)
        final_model.config.torch_dtype = torch.float16
    else:
        final_model.to(torch.float32)
        final_model.config.torch_dtype = torch.float32

    final_model.save_pretrained(SAVE_PATH)
    base_tokenizer.save_pretrained(SAVE_PATH)

    print(f"✅ New classification model + tokenizer saved to: {SAVE_PATH}")
    print(f"   vocab_size={final_model.config.vocab_size}, pad_token_id={final_model.config.pad_token_id}, dtype={final_model.config.torch_dtype}")


if __name__ == "__main__":
    main()
