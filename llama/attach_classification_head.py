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
    AutoModelForCausalLM
)
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.modeling_utils import load_sharded_checkpoint

import math
import random


# ---- Utils ----
def _tensor_allclose(a: torch.Tensor, b: torch.Tensor, rtol=1e-6, atol=1e-6):
    if a.shape != b.shape:
        return False, float("inf"), float("inf")
    # Compare in fp32 to avoid dtype noise
    a32 = a.detach().to(torch.float32).cpu()
    b32 = b.detach().to(torch.float32).cpu()
    diff = (a32 - b32).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    # relative diff per element; avoid div-by-zero by adding tiny eps
    denom = torch.maximum(a32.abs(), b32.abs())
    rel = torch.where(denom > 0, diff / denom, torch.zeros_like(diff))
    max_rel = float(rel.max().item()) if rel.numel() else 0.0
    ok = torch.allclose(a32, b32, rtol=rtol, atol=atol)
    return bool(ok), max_abs, max_rel


def _pick_random_shared_param_name(base_sd_keys, cls_sd_keys, seed=0):
    """Pick a random parameter name present in both models, excluding embeddings and heads."""
    blacklist_substrings = [
        "embed_tokens",   # input embeddings
        "lm_head",        # LM head
        "score",          # your seq-cls head
        "classifier",     # generic heads
    ]
    shared = [k for k in base_sd_keys if k in cls_sd_keys and not any(s in k for s in blacklist_substrings)]
    if not shared:
        return None
    random.seed(seed)
    return random.choice(shared)


def verify_shared_weights(base_lm_path: str, cls_model: torch.nn.Module, rtol=1e-6, atol=1e-6, seed=0):
    """
    Verifies that (1) input embeddings and (2) one random shared layer
    in the classification model match the base LM weights.
    """
    print("Loading base causal LM for weight verification ...")
    # Load in the same dtype to avoid extra rounding noise
    base_model = AutoModelForCausalLM.from_pretrained(
        base_lm_path, local_files_only=True, torch_dtype=getattr(cls_model, "dtype", None)
    )

    # ---- Embedding check ----
    base_emb = base_model.get_input_embeddings().weight
    cls_emb = cls_model.get_input_embeddings().weight
    print(f"[VERIFY] Embedding shapes: base={tuple(base_emb.shape)}, cls={tuple(cls_emb.shape)}; "
          f"dtypes: base={base_emb.dtype}, cls={cls_emb.dtype}")
    ok, max_abs, max_rel = _tensor_allclose(base_emb, cls_emb, rtol=rtol, atol=atol)
    if ok:
        print(f"[VERIFY] ✅ Embeddings match within tolerances (rtol={rtol}, atol={atol}). "
              f"max_abs_diff={max_abs:.3e}, max_rel_diff={max_rel:.3e}")
    else:
        print(f"[VERIFY] ❌ Embeddings differ. max_abs_diff={max_abs:.3e}, max_rel_diff={max_rel:.3e}")

    # ---- Random shared layer check ----
    base_sd = base_model.state_dict()
    cls_sd = cls_model.state_dict()

    pick = _pick_random_shared_param_name(base_sd.keys(), cls_sd.keys(), seed=seed)
    if pick is None:
        print("[VERIFY] ⚠️ No suitable shared parameter (besides embeddings/heads) found to compare.")
        return

    a = base_sd[pick]
    b = cls_sd[pick]
    print(f"[VERIFY] Random shared param: {pick}")
    print(f"[VERIFY] Shapes: base={tuple(a.shape)}, cls={tuple(b.shape)}; dtypes: base={a.dtype}, cls={b.dtype}")
    ok2, max_abs2, max_rel2 = _tensor_allclose(a, b, rtol=rtol, atol=atol)
    if ok2:
        print(f"[VERIFY] ✅ Random layer matches within tolerances (rtol={rtol}, atol={atol}). "
              f"max_abs_diff={max_abs2:.3e}, max_rel_diff={max_rel2:.3e}")
    else:
        print(f"[VERIFY] ❌ Random layer differs. max_abs_diff={max_abs2:.3e}, max_rel_diff={max_rel2:.3e}")

# ---- Utils ----



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

    SAVE_PATH = args.save_path
    os.makedirs(SAVE_PATH, exist_ok=True)

    base_tokenizer = AutoTokenizer.from_pretrained(args.base_lm_path, local_files_only=True)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
    target_vocab_size = len(base_tokenizer)

    class_config, mode = _derive_seq_cls_config_from_snapshot(
        args.seq_cls_config_path, is_llama4=args.llama4
    )

    print("Creating empty seq cls model from config ...")
    if mode == "llama4":
        final_model = CustomLlama4ForSequenceClassification(class_config)
    else:
        final_model = AutoModelForSequenceClassification.from_config(class_config)

    print("Adapting tokenizer and resizing the embeding layer in seq cls model for new tokens ...", flush=True)
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

    print("Loading checkpoint shards from clm model into seq cls model ...")
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

    print("Verifying shared weights against base LM ...")
    verify_shared_weights(args.base_lm_path, final_model, rtol=1e-6, atol=1e-6, seed=0)

    print("Saving model and tokenizer ...")
    final_model.save_pretrained(SAVE_PATH)
    base_tokenizer.save_pretrained(SAVE_PATH)

    print(f"✅ New classification model + tokenizer saved to: {SAVE_PATH}")
    print(f"   vocab_size={final_model.config.vocab_size}, pad_token_id={final_model.config.pad_token_id}, dtype={final_model.config.torch_dtype}")


if __name__ == "__main__":
    main()
