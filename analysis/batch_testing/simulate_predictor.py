#!/usr/bin/env python3
import os
import json
import random
import math

# ====== CONFIG / CONSTS ======
TARGET_ROC_AUC = 0.7  # pick your target auc here
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT_JSONL = os.path.join(REPO_ROOT, "datasets", "mozilla_perf", "perf_bugs_with_diff.jsonl")
OUT_JSON = os.path.join(REPO_ROOT, "analysis", "batch_testing", "predictor_sim_results.json")

def load_samples(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            # regressor info is in "regression" or "regressor" as "true"/"false"
            label = 1 if str(obj.get("regression") or obj.get("regressor")).lower() == "true" else 0
            commit_id = obj.get("revision") or obj.get("node") or ""
            samples.append({"commit_id": commit_id, "true_label": label})
    return samples

def _clamp01(x):
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def generate_confidence(label, target_auc):
    # keep in a sane range
    target_auc = max(0.5, min(target_auc, 0.995))

    # turn auc into a separation factor in [0, 1]
    # 0.5 -> 0 (no separation), 1.0 -> 1 (max separation)
    sep = (target_auc - 0.5) / 0.5  # in [0,1]

    # base means around the center
    # when sep is small -> means close to 0.5
    # when sep is big   -> 0.25 vs 0.75
    neg_mean = 0.5 - 0.25 * sep
    pos_mean = 0.5 + 0.25 * sep

    # std gets a bit smaller when separation is high
    base_std = 0.15
    std = base_std * (1.1 - 0.6 * sep)  # from ~0.165 down to ~0.09

    if label == 1:
        score = random.gauss(pos_mean, std)
    else:
        score = random.gauss(neg_mean, std)

    return _clamp01(score)


def main():
    random.seed(42)

    samples = load_samples(INPUT_JSONL)
    if not samples:
        print("No samples found. Exiting.")
        return

    out_samples = []
    for s in samples:
        score = generate_confidence(s["true_label"], TARGET_ROC_AUC)
        pred = 1 if score >= 0.5 else 0
        conf = score if pred == 1 else 1.0 - score
        out_samples.append(
            {
                "commit_id": s["commit_id"],
                "true_label": s["true_label"],
                "prediction": pred,
                "confidence": float(conf),
            }
        )

    out_obj = {
        "threshold": None,
        "used_samples": len(samples),
        "samples": out_samples,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)

    print(f"Saved results to {OUT_JSON}")

if __name__ == "__main__":
    main()
