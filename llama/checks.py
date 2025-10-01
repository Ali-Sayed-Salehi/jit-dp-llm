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



from typing import Dict, List, Mapping, Optional, Any, Union
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

def _check_one_dataset(
    ds: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    *,
    label_key: str = "label",
    drs_token: str = "[/drs]",
    zero_token: str = " 0",
    one_token: str = " 1",
    strict_single_token: bool = True,
    max_errors: int = 20,
    tail_tokens_to_show: int = 8,
    check_pad_consistency: bool = True,
) -> Dict[str, Any]:
    """
    Validates that each example looks like:
      input_ids = [..., drs_id, label_id, (pads...)]
      attention_mask marks both appended tokens as 1
      labels == -100 everywhere except the final label position (last non-pad)
      (optional) ex[label_key] ∈ {0,1} and matches label_id

    Returns a dict with summary stats and up to `max_errors` detailed errors.
    """
    # --- resolve ids (no caching) ---
    drs_id = tokenizer.convert_tokens_to_ids(drs_token)
    if drs_id == tokenizer.unk_token_id:
        raise ValueError(
            f"{drs_token!r} is not in the tokenizer vocab. "
            f"Add it before mapping (tokenizer.add_tokens(['{drs_token}'])) and reload."
        )

    ids0 = tokenizer.encode(zero_token, add_special_tokens=False)
    ids1 = tokenizer.encode(one_token, add_special_tokens=False)
    if strict_single_token and not (len(ids0) == 1 and len(ids1) == 1):
        raise ValueError(
            f"Expected single-token labels for {zero_token!r}/{one_token!r}, "
            "but they split into multiple ids. Add dedicated tokens (e.g., '<zero>', '<one>') "
            "and pass those via zero_token/one_token, or disable strict_single_token."
        )
    ID0 = ids0[0] if len(ids0) >= 1 else tokenizer.unk_token_id
    ID1 = ids1[0] if len(ids1) >= 1 else tokenizer.unk_token_id
    label_id_map = {0: ID0, 1: ID1}

    n = len(ds)
    errors: List[Dict[str, Any]] = []
    num_ok = 0
    label_counts = {0: 0, 1: 0, "other": 0}
    last_label_id_counts: Dict[int, int] = {}

    pad_id = tokenizer.pad_token_id

    for idx in range(n):
        ex = ds[idx]
        ids: List[int] = list(ex["input_ids"])
        attn: List[int] = list(ex.get("attention_mask", [1] * len(ids)))
        lbls: Optional[List[int]] = ex.get("labels", None)

        ok = True
        notes: List[str] = []

        # Basic shape checks
        if len(ids) != len(attn):
            ok = False
            notes.append(f"length mismatch: len(input_ids)={len(ids)} len(attention_mask)={len(attn)}")

        # Last non-pad index
        nonpad_len = sum(attn)
        if nonpad_len < 2:
            ok = False
            notes.append(f"nonpad_len={nonpad_len} (need at least 2 to hold [/drs] and label)")
            last_idx = None
        else:
            last_idx = nonpad_len - 1  # index of final non-pad token
            drs_pos = last_idx - 1     # position expected for [/drs]

            # Bounds
            if last_idx >= len(ids) or drs_pos < 0:
                ok = False
                notes.append(f"indices out of bounds: last_idx={last_idx}, drs_pos={drs_pos}, len={len(ids)}")
            else:
                # Check the two appended tokens
                if ids[drs_pos] != drs_id:
                    ok = False
                    notes.append(f"expected input_ids[{drs_pos}] == drs_id ({drs_id}), got {ids[drs_pos]}")
                # label_id to expect (if we have label_key and it's binary)
                expected_label_id = None
                if label_key in ex:
                    try:
                        y = int(ex[label_key])
                        if y in (0, 1):
                            expected_label_id = label_id_map[y]
                            label_counts[y] += 1
                        else:
                            label_counts["other"] += 1
                            notes.append(f"{label_key} not in {{0,1}}: {ex[label_key]!r}")
                    except Exception as e:
                        label_counts["other"] += 1
                        notes.append(f"{label_key} parse error: {e!r}")
                # If we know the expected label id, verify it
                if expected_label_id is not None and ids[last_idx] != expected_label_id:
                    ok = False
                    notes.append(
                        f"expected input_ids[{last_idx}] == label_id ({expected_label_id}), got {ids[last_idx]}"
                    )

                # attention_mask must mark both as 1
                if attn[drs_pos] != 1 or attn[last_idx] != 1:
                    ok = False
                    notes.append(f"attention_mask at drs/label not 1: attn[{drs_pos}]={attn[drs_pos]}, attn[{last_idx}]={attn[last_idx]}")

                # Labels checks
                if lbls is not None:
                    if len(lbls) != len(ids):
                        ok = False
                        notes.append(f"labels length mismatch: len(labels)={len(lbls)} len(input_ids)={len(ids)}")
                    else:
                        not_ignored = [i for i, v in enumerate(lbls) if v != -100]
                        if len(not_ignored) != 1 or not_ignored[0] != last_idx:
                            ok = False
                            notes.append(f"labels should supervise exactly and only last_idx={last_idx}, got positions {not_ignored}")
                        elif lbls[last_idx] != ids[last_idx]:
                            ok = False
                            notes.append(f"labels[last_idx] != input_ids[last_idx]: {lbls[last_idx]} vs {ids[last_idx]}")

                # Count which vocab id is at the final position (diagnostics)
                last_label_id_counts[ids[last_idx]] = last_label_id_counts.get(ids[last_idx], 0) + 1

        # Optional: pad consistency (positions with mask==0 should be pad_token_id, if defined)
        if check_pad_consistency and pad_id is not None and len(ids) == len(attn):
            for j, m in enumerate(attn):
                if m == 0 and ids[j] != pad_id:
                    ok = False
                    notes.append(f"pad inconsistency at pos {j}: attention_mask=0 but input_ids[{j}]={ids[j]} != pad_id={pad_id}")
                    break

        if ok:
            num_ok += 1
        else:
            # Collect debug tail
            tail_start = max(0, (len(ids) - tail_tokens_to_show))
            tail_ids = ids[tail_start:]
            tail_attn = attn[tail_start:]
            tail_lbls = lbls[tail_start:] if isinstance(lbls, list) else None
            # Try to decode tail safely (ignore decoding errors)
            try:
                decoded_tail = tokenizer.decode(ids[max(0, (nonpad_len - tail_tokens_to_show)): nonpad_len], skip_special_tokens=False)
            except Exception:
                decoded_tail = ""
            errors.append({
                "index": idx,
                "notes": notes,
                "last_idx": last_idx,
                "tail_input_ids": tail_ids,
                "tail_attention_mask": tail_attn,
                "tail_labels": tail_lbls,
                "decoded_nonpad_tail": decoded_tail,
            })
            if len(errors) >= max_errors:
                break

    return {
        "num_examples": n,
        "num_ok": num_ok,
        "num_errors": len(errors),
        "error_rate": 0.0 if n == 0 else (n - num_ok) / n,
        "label_counts": label_counts,
        "final_token_id_histogram": last_label_id_counts,
        "drs_token_id": drs_id,
        "label_token_ids": {"0": ID0, "1": ID1},
        "pad_token_id": pad_id,
        "errors": errors,
    }


def check_drs_append(
    data: Union[Dataset, DatasetDict],
    tokenizer: PreTrainedTokenizerBase,
    **kwargs,
) -> Dict[str, Any]:
    """
    Wrapper that accepts either a Dataset or a DatasetDict (with splits).
    Returns a summary (and per-split summaries for DatasetDict).
    """
    if isinstance(data, DatasetDict):
        out = {"splits": {}}
        for name, split in data.items():
            out["splits"][name] = _check_one_dataset(split, tokenizer, **kwargs)
        # aggregate simple totals
        total = sum(v["num_examples"] for v in out["splits"].values())
        ok = sum(v["num_ok"] for v in out["splits"].values())
        err = sum(v["num_errors"] for v in out["splits"].values())
        out.update({"num_examples": total, "num_ok": ok, "num_errors": err,
                    "error_rate": 0.0 if total == 0 else (total - ok) / total})
        return out
    else:
        return _check_one_dataset(data, tokenizer, **kwargs)
