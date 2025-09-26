from collections import defaultdict

from utils import *

def _n_params(t): return int(t.numel())

def _is_lora_param_name(name: str) -> bool:
    # PEFT names like: <...>.lora_A.weight / <...>.lora_B.weight
    return (".lora_A." in name) or (".lora_B." in name) or (".lora_embedding_A." in name) or (".lora_embedding_B." in name)

def _is_embed_token_adapter(name: str) -> bool:
    # When using trainable_token_indices, PEFT makes a token adapter under embed_tokens
    return ("embed_tokens" in name) and ("token_adapter" in name) and name.endswith(".weight")

def _is_lm_head(name: str) -> bool:
    tail = name.split(".")[-2:]
    return ("lm_head" in name) and (tail[-1] in {"weight","bias"})

def _is_cls_head(name: str) -> bool:
    return (("score." in name) or ("classifier." in name)) and name.split(".")[-1] in {"weight","bias"}

def count_trainable_params(model, tokenizer=None, task="clm", added_token_ids=None, verbose=True):
    """
    Returns a dict with:
      - actual_total: sum of p.numel() for p.requires_grad
      - buckets: lora_total, lm_head, seq_head, embed_token_adapter (nominal),
                 embed_token_effective (added_rows * hidden_size, if tokenizer provided)
      - theoretical: derived from LoRA A/B shapes + optional heads + effective embed rows
    """
    buckets = defaultdict(int)
    # 1) Actual counts from requires_grad
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = _n_params(p)
        if _is_lora_param_name(name):
            buckets["lora_total"] += n
        elif _is_embed_token_adapter(name):
            buckets["embed_token_adapter_nominal"] += n  # This is usually full VxD, even if only rows are "active"
        elif _is_lm_head(name):
            buckets["lm_head"] += n
        elif _is_cls_head(name):
            buckets["seq_head"] += n
        else:
            buckets["other"] += n

    actual_total = sum(buckets.values())

    # 2) Theoretical LoRA count from A/B shapes (r*(in+out) per layer)
    # Group A/B by the same base prefix
    lora_pairs = {}  # prefix -> {"A": (out, r), "B": (r, in)}
    for name, p in model.named_parameters():
        if not _is_lora_param_name(name): 
            continue
        base = name.rsplit(".lora_", 1)[0]  # strip ".lora_A." or ".lora_B."
        if base not in lora_pairs: lora_pairs[base] = {}
        if ".lora_A." in name:
            # A is [out, r]
            out, r = p.shape[0], p.shape[1]
            lora_pairs[base]["A"] = (out, r)
        elif ".lora_B." in name:
            # B is [r, in]
            r, in_f = p.shape[0], p.shape[1]
            lora_pairs[base]["B"] = (r, in_f)

    theoretical_lora = 0
    for base, pair in lora_pairs.items():
        if "A" in pair and "B" in pair:
            out, rA = pair["A"]
            rB, in_f = pair["B"]
            r = rA  # should equal rB
            theoretical_lora += r * (in_f + out)

    # 3) Theoretical heads
    theoretical_lm_head = 0
    theoretical_seq_head = 0

    # If the head params are trainable, just sum their shapes (same as "actual", but we compute fresh)
    # (We prefer to derive by shape so it's robust if some submodule holds the params.)
    if task == "clm":
        head = getattr(model, "get_output_embeddings", lambda: None)()
        if head is not None and hasattr(head, "weight") and head.weight.requires_grad:
            theoretical_lm_head += _n_params(head.weight)
            if getattr(head, "bias", None) is not None and head.bias.requires_grad:
                theoretical_lm_head += _n_params(head.bias)
    else:  # seq_cls
        head = getattr(model, "score", None)
        if head is None:
            head = getattr(model, "classifier", None)
            if hasattr(head, "out_proj"): head = head.out_proj
        if head is not None and hasattr(head, "weight") and head.weight.requires_grad:
            theoretical_seq_head += _n_params(head.weight)
            if getattr(head, "bias", None) is not None and head.bias.requires_grad:
                theoretical_seq_head += _n_params(head.bias)

    # 4) Effective embed rows (only the added ids) for a more meaningful number
    theoretical_embed_effective = 0
    if tokenizer is not None:
        if added_token_ids is None:
            try:
                added_token_ids = sorted(getattr(tokenizer, "get_added_vocab", lambda: {})().values())
            except Exception:
                added_token_ids = []
        if added_token_ids:
            emb = model.get_input_embeddings()
            hidden = emb.weight.shape[1]
            theoretical_embed_effective = len(added_token_ids) * hidden

    theoretical_total = theoretical_lora + theoretical_lm_head + theoretical_seq_head + theoretical_embed_effective

    if verbose:
        print("\n=== Trainable parameter summary ===")
        print(f"Actual trainable (requires_grad): {actual_total:,}")
        print("Breakdown (actual):")
        for k in ["lora_total","lm_head","seq_head","embed_token_adapter_nominal","other"]:
            if k in buckets:
                print(f"  • {k:28s} {buckets[k]:,}")
        if tokenizer is not None and theoretical_embed_effective:
            print(f"\nEffective added-token rows (V_add * d_model): {theoretical_embed_effective:,}  (more meaningful than nominal V*d for token_adapter)")
        print("\nTheoretical (sanity check):")
        print(f"  • LoRA (sum over layers r*(in+out))     : {theoretical_lora:,}")
        if task == "clm": print(f"  • lm_head (if trainable)                 : {theoretical_lm_head:,}")
        if task != "clm": print(f"  • seq head (if trainable)                : {theoretical_seq_head:,}")
        if theoretical_embed_effective: print(f"  • embed added-rows effective             : {theoretical_embed_effective:,}")
        print(f"≈ Expected total (effective view)         : {theoretical_total:,}")
        print("====================================\n")

    return {
        "actual_total": actual_total,
        "actual_breakdown": dict(buckets),
        "theoretical": {
            "lora": theoretical_lora,
            "lm_head": theoretical_lm_head,
            "seq_head": theoretical_seq_head,
            "embed_effective": theoretical_embed_effective,
            "expected_total_effective": theoretical_total,
        },
    }
