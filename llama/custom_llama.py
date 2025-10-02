import torch
import torch.nn as nn
from typing import Optional
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaPreTrainedModel,
    LlamaConfig,
)
from transformers import AutoModel
from transformers.utils import logging

logger = logging.get_logger(__name__)

# --- This mirrors Transformers' pattern: a Generic class + the Llama mixin ---

class GenericForSequenceClassificationWithPooling(object):
    """
    Same spirit as HF's GenericForSequenceClassification but with selectable pooling:
    - "last": match HF default (last non-pad token)
    - "mean": masked mean over time
    - "max": masked max over time
    Set via `config.pooling = "last"|"mean"|"max"`, and you can also override per call:
    forward(..., pooling="mean")
    """
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        setattr(self, self.base_model_prefix, AutoModel.from_config(config))
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.post_init()

    # ---- pooling helpers ----
    def _build_mask(
        self,
        input_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        device: torch.device,
        seq_len: int,
        pad_token_id: Optional[int],
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        """
        Return a [B, T] mask of 1=keep, 0=pad, or None if unknown.
        Prefers attention_mask; otherwise derive from input_ids + pad_token_id if possible.
        """
        if attention_mask is not None:
            return attention_mask
        if input_ids is not None and pad_token_id is not None:
            return (input_ids != pad_token_id).to(device=device, dtype=torch.long)
        # Unknown padding — return None to pool over all positions
        if batch_size != 1 and pad_token_id is None:
            # Mirror HF's guard for batches>1 with no pad token
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        return None  # treat all positions as valid

    def _masked_mean_pool(self, hidden: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # hidden: [B, T, H], mask: [B, T] {0/1} or None
        if mask is None:
            return hidden.mean(dim=1)
        mask = mask.to(hidden.dtype).unsqueeze(-1)          # [B, T, 1]
        summed = (hidden * mask).sum(dim=1)                 # [B, H]
        denom = mask.sum(dim=1).clamp(min=1.0)              # [B, 1]
        return summed / denom

    def _masked_max_pool(self, hidden: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return hidden.max(dim=1).values
        mask = mask.bool().unsqueeze(-1)                    # [B, T, 1]
        neg_inf = torch.finfo(hidden.dtype).min
        hidden_masked = torch.where(mask, hidden, torch.full_like(hidden, neg_inf))
        return hidden_masked.max(dim=1).values

    def _last_non_pad_index(
        self,
        input_ids: Optional[torch.LongTensor],
        pad_token_id: Optional[int],
        device: torch.device,
        batch_size: int,
        seq_len: int,
        using_inputs_embeds: bool,
    ) -> torch.Tensor:
        """
        HF-compatible last-non-pad selection. Returns [B] indices.
        """
        if pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")

        if pad_token_id is None:
            # batch==1 case allowed by HF: use -1 (last position)
            return torch.tensor([-1], device=device, dtype=torch.int64).repeat(batch_size)

        if input_ids is not None:
            non_pad_mask = (input_ids != pad_token_id).to(device=device, dtype=torch.int32)  # [B, T]
            token_indices = torch.arange(seq_len, device=device, dtype=torch.int32)          # [T]
            # rightmost non-pad via argmax over idx*mask
            last_non_pad = (token_indices * non_pad_mask).argmax(-1)                         # [B]
            return last_non_pad.to(torch.int64)

        # inputs_embeds without input_ids: replicate HF warning and use last pos
        if using_inputs_embeds:
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. "
                "Results may be unexpected if using padding tokens in conjunction with `inputs_embeds`."
            )
        return torch.full((batch_size,), seq_len - 1, device=device, dtype=torch.int64)

    # ---- forward ----
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional["Cache"] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        # override config.pooling at call-site if desired
        pooling: Optional[str] = None,
        **kwargs,
    ) -> SequenceClassifierOutputWithPast:

        transformer_outputs = getattr(self, self.base_model_prefix)(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = transformer_outputs.last_hidden_state       # [B, T, H]
        device = hidden_states.device
        batch_size = (input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0])
        seq_len = hidden_states.shape[1]
        pad_token_id = getattr(self.config, "pad_token_id", None)

        # token-wise logits (kept for compatibility with HF loss_function signature)
        token_logits = self.score(hidden_states)                    # [B, T, num_labels]

        # choose pooling mode
        mode = (pooling or getattr(self.config, "pooling", None) or "last").lower()
        if mode not in ("last", "mean", "max"):
            raise ValueError(f"Unsupported pooling '{mode}'. Use 'last' | 'mean' | 'max'.")

        if mode == "last":
            # HF-style last-non-pad selection
            last_idx = self._last_non_pad_index(
                input_ids=input_ids,
                pad_token_id=pad_token_id,
                device=device,
                batch_size=batch_size,
                seq_len=seq_len,
                using_inputs_embeds=(input_ids is None and inputs_embeds is not None),
            )                                                     # [B]
            pooled_logits = token_logits[torch.arange(batch_size, device=device), last_idx]  # [B, C]
        else:
            # Build mask then pool H, then classify once
            mask = self._build_mask(
                input_ids=input_ids,
                attention_mask=attention_mask,
                device=device,
                seq_len=seq_len,
                pad_token_id=pad_token_id,
                batch_size=batch_size,
            )                                                     # [B, T] or None

            if mode == "mean":
                pooled_hidden = self._masked_mean_pool(hidden_states, mask)                  # [B, H]
            else:  # "max"
                pooled_hidden = self._masked_max_pool(hidden_states, mask)                   # [B, H]

            pooled_logits = self.score(pooled_hidden)                                        # [B, C]

        # ----- loss (match HF Generic loss_function call) -----
        loss = None
        if labels is not None:
            # The HF Generic uses a helper; we mirror its interface:
            loss = self.loss_function(logits=token_logits, labels=labels,
                                      pooled_logits=pooled_logits, config=self.config)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,                                # like HF: return pooled (per-example) logits
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class LlamaForSequenceClassificationMaxPoolMeanLast(
    GenericForSequenceClassificationWithPooling,
    LlamaPreTrainedModel,
):
    """
    Drop-in replacement for LlamaForSequenceClassification with selectable pooling.
    """
    config: LlamaConfig
    pass
