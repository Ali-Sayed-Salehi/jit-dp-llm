#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

# ----------------------------
# Core helpers
# ----------------------------

def load_results_json(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = json.loads(Path(path).read_text())
    samples = data.get("samples", [])
    if not samples:
        raise ValueError("No 'samples' found in the JSON.")
    y_true  = np.array([int(s["true_label"]) for s in samples], dtype=int)
    y_score = np.array([float(s["confidence"]) for s in samples], dtype=float)  # expected: P(y=1)
    # clip to [0,1] just in case
    y_score = np.clip(y_score, 0.0, 1.0)
    return y_true, y_score

def confusion_at_threshold(y_true: np.ndarray, y_score: np.ndarray, t: float) -> Tuple[int,int,int,int]:
    y_pred = (y_score >= t).astype(int)
    TP = int(((y_true==1) & (y_pred==1)).sum())
    FP = int(((y_true==0) & (y_pred==1)).sum())
    FN = int(((y_true==1) & (y_pred==0)).sum())
    TN = int(((y_true==0) & (y_pred==0)).sum())
    return TP, FP, FN, TN

def metrics_from_confusion(TP:int, FP:int, FN:int, TN:int) -> Dict[str, float]:
    P = TP/(TP+FP) if (TP+FP)>0 else 0.0
    R = TP/(TP+FN) if (TP+FN)>0 else 0.0
    F1 = 2*P*R/(P+R) if (P+R)>0 else 0.0
    Acc = (TP+TN)/max(1, (TP+FP+FN+TN))
    return {"precision": P, "recall": R, "f1": F1, "accuracy": Acc}

def reweight_to_prevalence(TP:int, FP:int, FN:int, TN:int, pi_obs:float, pi_tgt:float) -> Dict[str, float]:
    """
    Importance-weight the confusion matrix to simulate a target prevalence.
    Recall stays the same; precision/F1/accuracy change under reweighting.
    """
    if not (0.0 < pi_obs < 1.0):
        # Degenerate case (all positives or all negatives in data)
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
    """Dense, data-driven threshold grid: midpoints between unique sorted scores + {0,1}."""
    uniq = np.unique(y_score)
    if uniq.size <= 1:
        return np.array([0.0, 1.0], dtype=float)
    mids = (uniq[:-1] + uniq[1:]) / 2.0
    return np.concatenate(([0.0], mids, [1.0]))

def best_f1_for_prevalence(
    y_true: np.ndarray, y_score: np.ndarray, pi_tgt: float
) -> Tuple[float, Dict[str, float]]:
    """
    Find threshold that maximizes F1 *under the target prevalence*.
    Returns (threshold, metrics_at_that_threshold_reweighted).
    """
    pi_obs = float(np.mean(y_true)) if y_true.size else float("nan")
    thr_grid = thresholds_from_scores(y_score)

    best_f1 = -1.0
    best_t = 0.5
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    for t in thr_grid:
        TP, FP, FN, TN = confusion_at_threshold(y_true, y_score, t)
        m = reweight_to_prevalence(TP, FP, FN, TN, pi_obs, pi_tgt)
        f1 = m["f1"] if np.isfinite(m["f1"]) else -1.0
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
            best_metrics = m

    return best_t, best_metrics

def clean_metric_block(threshold: float, metrics: Dict[str, float]) -> Dict[str, float]:
    return {
        "threshold": float(threshold),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "accuracy": float(metrics["accuracy"]),
    }

def parse_pi_list(pi_args: List[float]) -> List[float]:
    # Accept percentages or fractions; e.g., 19.2 -> 0.192 ; 0.192 stays 0.192
    if not pi_args:
        return [0.192, 0.0966]
    out = []
    for x in pi_args:
        out.append(x/100.0 if x > 1.0 else x)
    return out

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Compute minimal metrics JSON at threshold=0.5 and at best-F1 thresholds per target prevalence."
    )
    ap.add_argument("--json", required=True, type=Path, help="Path to results JSON (with 'samples' list and 'confidence').")
    ap.add_argument("--out", required=True, type=Path, help="Where to write the clean JSON.")
    ap.add_argument("--prevalence", "-p", type=float, nargs="*", default=None,
                    help="Target prevalences as percentages or fractions (default: 19.2 9.66).")
    args = ap.parse_args()

    y_true, y_score = load_results_json(args.json)
    pi_obs = float(np.mean(y_true)) if y_true.size else float("nan")
    pis = parse_pi_list(args.prevalence if args.prevalence is not None else [])

    # Compute once for threshold = 0.5 (then reweight to each target prevalence)
    TP05, FP05, FN05, TN05 = confusion_at_threshold(y_true, y_score, 0.5)

    output = {
        # You asked for only the requested info. We still add the list of target prevalences used.
        "prevalences": pis,  # fractions (e.g., 0.192)
        "threshold_0_5": {},
        "best_f1": {},
    }

    # Fill metrics at t=0.5 for each target prevalence
    for pi_t in pis:
        m = reweight_to_prevalence(TP05, FP05, FN05, TN05, pi_obs, pi_t)
        output["threshold_0_5"][f"{pi_t}"] = clean_metric_block(0.5, m)

    # For each target prevalence, find best-F1 threshold (under that prevalence) and metrics
    for pi_t in pis:
        t_star, m_star = best_f1_for_prevalence(y_true, y_score, pi_t)
        output["best_f1"][f"{pi_t}"] = clean_metric_block(t_star, m_star)

    # Write clean JSON
    args.out.write_text(json.dumps(output, indent=2))
    # No prints, no grids—just the JSON file.

if __name__ == "__main__":
    main()
