#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

# ----------------------------
# Core helpers
# ----------------------------

def load_results_json(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expected schema (subset):
    {
      "samples": [
        {"true_label": 0, "prediction": 1, "confidence": 0.64},
        ...
      ]
    }
    'confidence' is P(prediction). We convert to P(y=1).
    """
    data = json.loads(Path(path).read_text())
    samples = data.get("samples", [])
    if not samples:
        raise ValueError("No 'samples' found in the JSON.")

    y_true_list, y_posprob_list = [], []
    for s in samples:
        y_true = int(s["true_label"])
        pred   = int(s["prediction"])
        conf   = float(s["confidence"])
        if y_true not in (0, 1) or pred not in (0, 1):
            raise ValueError(f"Labels must be 0/1, got true={y_true}, pred={pred}")
        if not (0.0 <= conf <= 1.0):
            raise ValueError(f"'confidence' must be in [0,1], got {conf}")

        # Convert P(predicted class) -> P(y=1)
        p_pos = conf if pred == 1 else (1.0 - conf)

        y_true_list.append(y_true)
        y_posprob_list.append(p_pos)

    y_true = np.array(y_true_list, dtype=int)
    y_score = np.clip(np.array(y_posprob_list, dtype=float), 0.0, 1.0)
    return y_true, y_score

def confusion_at_threshold(y_true: np.ndarray, y_score: np.ndarray, t: float) -> Tuple[int,int,int,int]:
    y_pred = (y_score >= t).astype(int)
    TP = int(((y_true==1) & (y_pred==1)).sum())
    FP = int(((y_true==0) & (y_pred==1)).sum())
    FN = int(((y_true==1) & (y_pred==0)).sum())
    TN = int(((y_true==0) & (y_pred==0)).sum())
    return TP, FP, FN, TN

def reweight_to_prevalence(TP:int, FP:int, FN:int, TN:int, pi_obs:float, pi_tgt:float) -> Dict[str, float]:
    """
    Importance-weight the confusion matrix to simulate a target prevalence.
    Recall stays the same; precision/F1/accuracy change under reweighting.
    """
    if not (0.0 < pi_obs < 1.0):
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan"), "accuracy": float("nan")}
    w_pos = pi_tgt / pi_obs
    w_neg = (1.0 - pi_tgt) / (1.0 - pi_obs)
    TPp, FNp, FPp, TNp = TP*w_pos, FN*w_pos, FP*w_neg, TN*w_neg

    P = TPp/(TPp+FPp) if (TPp+FPp)>0 else 0.0
    R = TPp/(TPp+FNp) if (TPp+FNp)>0 else 0.0  # equals original recall numerically
    F1 = 2*P*R/(P+R) if (P+R)>0 else 0.0
    Acc = (TPp+TNp)/(TPp+FPp+FNp+TNp) if (TPp+FPp+FNp+TNp)>0 else 0.0
    return {"precision": P, "recall": R, "f1": F1, "accuracy": Acc}

def thresholds_from_scores(y_score: np.ndarray) -> np.ndarray:
    """Midpoints between unique sorted scores + {0,1}."""
    uniq = np.unique(y_score)
    if uniq.size <= 1:
        return np.array([0.0, 1.0], dtype=float)
    mids = (uniq[:-1] + uniq[1:]) / 2.0
    return np.concatenate(([0.0], mids, [1.0]))

def best_f1_for_prevalence(y_true: np.ndarray, y_score: np.ndarray, pi_tgt: float) -> Tuple[float, Dict[str, float]]:
    """Find threshold that maximizes F1 under the target prevalence."""
    pi_obs = float(np.mean(y_true)) if y_true.size else float("nan")
    thr_grid = thresholds_from_scores(y_score)
    best_f1, best_t = -1.0, 0.5
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    for t in thr_grid:
        TP, FP, FN, TN = confusion_at_threshold(y_true, y_score, t)
        m = reweight_to_prevalence(TP, FP, FN, TN, pi_obs, pi_tgt)
        f1 = m["f1"] if np.isfinite(m["f1"]) else -1.0
        if f1 > best_f1:
            best_f1, best_t, best_metrics = f1, float(t), m
    return best_t, best_metrics

def threshold_for_recall(y_true: np.ndarray, y_score: np.ndarray, recall_target: float) -> Tuple[float, float, Tuple[int,int,int,int]]:
    """
    Find threshold whose observed recall is closest to recall_target.
    Tie-break by: higher recall, then higher threshold.
    Returns (chosen_threshold, achieved_recall, (TP,FP,FN,TN)).
    """
    thr_grid = thresholds_from_scores(y_score)
    best = None  # (abs_diff, -recall, -threshold) for sorting

    chosen_t, chosen_rec, chosen_conf = 0.5, 0.0, (0,0,0,0)
    for t in thr_grid:
        TP, FP, FN, TN = confusion_at_threshold(y_true, y_score, t)
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        key = (abs(rec - recall_target), -rec, -float(t))
        if (best is None) or (key < best):
            best = key
            chosen_t, chosen_rec, chosen_conf = float(t), rec, (TP, FP, FN, TN)
    return chosen_t, chosen_rec, chosen_conf

def clean_metric_block(threshold: float, metrics: Dict[str, float]) -> Dict[str, float]:
    return {
        "threshold": float(threshold),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "accuracy": float(metrics["accuracy"]),
    }

def _dedupe_preserve_order(values: List[float], tol: float = 1e-12) -> List[float]:
    out = []
    for v in values:
        if not any(abs(v - u) <= tol for u in out):
            out.append(v)
    return out

def parse_pi_list_with_observed(pi_args: List[float], pi_obs: float) -> List[float]:
    """
    If no pi_args are provided, default to [0.192, 0.0966, pi_obs].
    If provided, use user list PLUS observed prevalence. Deduped.
    Accepts percentages (>1.0) or fractions.
    """
    if not pi_args:
        return _dedupe_preserve_order([0.192, 0.0966, pi_obs])
    provided = [x/100.0 if x > 1.0 else x for x in pi_args]
    return _dedupe_preserve_order(provided + [pi_obs])

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Compute metrics at a fixed threshold, best-F1 per prevalence, and F1 at a target recall (reweighted per prevalence)."
    )
    ap.add_argument("--json", required=True, type=Path, help="Path to results JSON.")
    ap.add_argument("--out", required=True, type=Path, help="Where to write the clean JSON.")
    ap.add_argument("--prevalence", "-p", type=float, nargs="*", default=None,
                    help="Target prevalences as percentages or fractions. If omitted, uses 19.2%, 9.66%, and observed prevalence.")
    ap.add_argument("--threshold", "-t", type=float, default=0.5,
                    help="Fixed decision threshold on P(y=1) in [0,1] (default: 0.5).")
    ap.add_argument("--recall-target", "-r", type=float, default=0.83,
                    help="Target recall to match (default: 0.83).")
    args = ap.parse_args()

    if not (0.0 <= args.threshold <= 1.0):
        raise ValueError(f"--threshold must be in [0,1], got {args.threshold}")
    if not (0.0 <= args.recall_target <= 1.0):
        raise ValueError(f"--recall-target must be in [0,1], got {args.recall_target}")

    y_true, y_score = load_results_json(args.json)
    pi_obs = float(np.mean(y_true)) if y_true.size else float("nan")
    pis = parse_pi_list_with_observed(args.prevalence if args.prevalence is not None else [], pi_obs)

    # ---------- metrics at fixed threshold ----------
    TP_t, FP_t, FN_t, TN_t = confusion_at_threshold(y_true, y_score, args.threshold)

    # ---------- best-F1 per prevalence ----------
    best_f1_block = {}
    for pi_t in pis:
        t_star, m_star = best_f1_for_prevalence(y_true, y_score, pi_t)
        best_f1_block[f"{pi_t}"] = clean_metric_block(t_star, m_star)

    # ---------- F1 at target recall (choose threshold by observed recall) ----------
    t_rec, rec_achieved, conf = threshold_for_recall(y_true, y_score, args.recall_target)
    TP_r, FP_r, FN_r, TN_r = conf
    recall_target_block = {}
    for pi_t in pis:
        m = reweight_to_prevalence(TP_r, FP_r, FN_r, TN_r, pi_obs, pi_t)
        recall_target_block[f"{pi_t}"] = clean_metric_block(t_rec, m)

    output = {
        "prevalences": pis,                   # includes observed prevalence
        "observed_prevalence": pi_obs,
        "used_threshold": float(args.threshold),
        "threshold_fixed": {},                # metrics at --threshold, reweighted per prevalence
        "best_f1": best_f1_block,             # best-F1 threshold per prevalence
        "recall_target": {
            "target_recall": float(args.recall_target),
            "achieved_recall": float(rec_achieved),  # same across prevalences
            "per_prevalence": recall_target_block
        }
    }

    for pi_t in pis:
        m = reweight_to_prevalence(TP_t, FP_t, FN_t, TN_t, pi_obs, pi_t)
        output["threshold_fixed"][f"{pi_t}"] = clean_metric_block(args.threshold, m)

    args.out.write_text(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
