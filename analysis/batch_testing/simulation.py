#!/usr/bin/env python3
import os
import json
import random
from datetime import datetime, timedelta, timezone

# ====== CONFIG (edit here) ======
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

ALL_COMMITS_PATH = os.path.join(REPO_ROOT, "datasets", "mozilla_perf", "all_commits.jsonl")
INPUT_JSON = os.path.join(REPO_ROOT, "analysis", "batch_testing", "predictor_sim_results.json")
OUTPUT_PATH = os.path.join(REPO_ROOT, "analysis", "batch_testing", "simulated_results.json")

BATCH_HOURS = 4
FSB_SIZE = 20
CUTOFF = datetime.fromisoformat("2024-10-10T00:00:00+00:00")
PRED_THRESHOLD = 0.5
RANDOM_SEED = 42
# =================================

random.seed(RANDOM_SEED)

from batch_strats import simulate_twb_with_bisect, simulate_fsb_with_bisect
from bisection_strats import (
    risk_ordered_bisect,
    unordered_bisect,
    time_ordered_bisect,
    TEST_DURATION_MIN,
)


def load_predictions(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    preds = {}
    for s in data.get("samples", []):
        cid = s["commit_id"]
        true_label = int(s["true_label"])
        original_pred = int(s["prediction"])
        original_conf = float(s["confidence"])

        if original_pred == 1:
            p_pos = original_conf
        else:
            p_pos = 1.0 - original_conf

        final_pred = 1 if p_pos >= PRED_THRESHOLD else 0

        preds[cid] = {
            "true_label": true_label,
            "pred_label": final_pred,
            "p_pos": p_pos,
        }
    return preds


def parse_hg_date(date_field):
    if isinstance(date_field, list) and len(date_field) == 2:
        unix_ts, offset_sec = date_field
        dt_utc = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
        dt_local = dt_utc + timedelta(seconds=offset_sec)
        return dt_local
    if isinstance(date_field, str):
        return datetime.fromisoformat(date_field)
    return datetime.now(timezone.utc)


def read_commits_from_all(all_commits_path, pred_map):
    commits = []
    with open(all_commits_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            node = obj.get("node")
            date_field = obj.get("date")
            if not node or date_field is None:
                continue

            ts = parse_hg_date(date_field)
            if ts <= CUTOFF:
                continue

            if node in pred_map:
                info = pred_map[node]
                true_label = info["true_label"]
                pred_label = info["pred_label"]
                p_pos = info["p_pos"]
            else:
                true_label = 0
                pred_label = 0
                p_pos = 0.0

            commits.append({
                "commit_id": node,
                "true_label": bool(true_label),
                "prediction": pred_label,
                "risk": p_pos,
                "ts": ts,
            })

    commits.sort(key=lambda x: x["ts"])
    return commits


def run_exhaustive_testing(commits):
    total_tests_run = 0
    culprit_times = []
    feedback_times = {}

    for c in commits:
        total_tests_run += 1
        finish_time = c["ts"] + timedelta(minutes=TEST_DURATION_MIN)
        fb_min = (finish_time - c["ts"]).total_seconds() / 60
        feedback_times[c["commit_id"]] = fb_min
        if c["true_label"]:
            culprit_times.append(fb_min)

    if feedback_times:
        mean_fb = sum(feedback_times.values()) / len(feedback_times)
    else:
        mean_fb = 0.0

    if culprit_times:
        mean_ttc = sum(culprit_times) / len(culprit_times)
        max_ttc = max(culprit_times)
    else:
        mean_ttc = 0.0
        max_ttc = 0.0

    return {
        "total_tests_run": total_tests_run,
        "mean_feedback_time_min": round(mean_fb, 2),
        "mean_time_to_culprit_min": round(mean_ttc, 2),
        "max_time_to_culprit_min": round(max_ttc, 2),
    }


BATCHING_STRATEGIES = [
    ("TWB-N", simulate_twb_with_bisect, BATCH_HOURS),
    ("FSB-N", simulate_fsb_with_bisect, FSB_SIZE),
]

BISECTION_STRATEGIES = [
    ("ROB", risk_ordered_bisect),
    ("UB", unordered_bisect),
    ("TOB", time_ordered_bisect),
]


def main():
    pred_map = load_predictions(INPUT_JSON)
    commits = read_commits_from_all(ALL_COMMITS_PATH, pred_map)
    if not commits:
        print("No commits found after cutoff; exiting.")
        return

    et_results = run_exhaustive_testing(commits)

    ci_results = {}
    for b_name, b_fn, b_param in BATCHING_STRATEGIES:
        for bis_name, bis_fn in BISECTION_STRATEGIES:
            combo_name = f"{b_name} + {bis_name}"
            res = b_fn(commits, bis_fn, b_param)
            ci_results[combo_name] = res

    best_tests = min(ci_results.items(), key=lambda kv: kv[1]["total_tests_run"])
    best_max_ttc = min(ci_results.items(), key=lambda kv: kv[1]["max_time_to_culprit_min"])
    best_mean_fb = min(ci_results.items(), key=lambda kv: kv[1]["mean_feedback_time_min"])

    out = {
        "Exhaustive Testing (ET)": et_results,
    }
    for name, res in ci_results.items():
        out[name] = res

    out["best_by_total_tests"] = best_tests[0]
    out["best_by_max_ttc"] = best_max_ttc[0]
    out["best_by_mean_feedback_time"] = best_mean_fb[0]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("saved to", OUTPUT_PATH)
    print(f"Best (fewest tests): {best_tests[0]}")
    print(f"Best (lowest max TTC): {best_max_ttc[0]}")
    print(f"Best (lowest mean feedback): {best_mean_fb[0]}")


if __name__ == "__main__":
    main()
