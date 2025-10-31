#!/usr/bin/env python3
import os
import json
import random
from datetime import datetime, timedelta, timezone

# ====== CONFIG (edit here) ======
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

ALL_COMMITS_PATH = os.path.join(REPO_ROOT, "datasets", "mozilla_perf", "all_commits.jsonl")
INPUT_JSON = os.path.join(REPO_ROOT, "analysis", "predictor_results.json")  # model output
OUTPUT_PATH = os.path.join(REPO_ROOT, "analysis", "batch_testing", "simulated_results.json")

BATCH_HOURS = 4        # Time window (TWB-N)
TEST_DURATION_MIN = 10 # Duration per test
CUTOFF = datetime.fromisoformat("2024-10-10T00:00:00+00:00")
RANDOM_SEED = 42
# =================================

random.seed(RANDOM_SEED)


def load_predictions(path):
    """commit_id -> {true_label, prediction, confidence}"""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    preds = {}
    for s in data.get("samples", []):
        cid = s["commit_id"]
        preds[cid] = {
            "true_label": int(s["true_label"]),
            "prediction": int(s["prediction"]),
            "confidence": float(s["confidence"]),
        }
    return preds


def parse_hg_date(date_field):
    """
    Mercurial-style date: [unix_ts, offset_seconds]
    Example: [1742116411, -7200]
    """
    if isinstance(date_field, list) and len(date_field) == 2:
        unix_ts, offset_sec = date_field
        dt_utc = datetime.utcfromtimestamp(unix_ts).replace(tzinfo=timezone.utc)
        dt_local = dt_utc + timedelta(seconds=offset_sec)
        return dt_local
    if isinstance(date_field, str):
        return datetime.fromisoformat(date_field)
    return datetime.now(timezone.utc)


def read_commits_from_all(all_commits_path, pred_map):
    """
    Read all commits from all_commits.jsonl (node, date as [ts, offset]).
    If commit not in pred_map -> predicted clean with conf=1.0
    """
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
                pred = info["prediction"]
                conf = info["confidence"]
            else:
                true_label = 0
                pred = 0
                conf = 1.0

            # risk = P(positive class)
            risk = conf if pred == 1 else 1.0 - conf

            commits.append({
                "commit_id": node,
                "true_label": bool(true_label),
                "prediction": pred,
                "confidence": conf,
                "risk": risk,
                "ts": ts,
            })

    commits.sort(key=lambda x: x["ts"])
    return commits


def batch_has_regressor(batch):
    return any(c["true_label"] for c in batch)


# ===== BASELINE MODE: EXHAUSTIVE TESTING (ET) =====
def run_exhaustive_testing(commits):
    """
    Test every commit individually.
    No batching, no bisection.
    """
    total_tests_run = 0
    culprit_times = []
    feedback_times = {}  # commit_id -> minutes

    for c in commits:
        total_tests_run += 1
        finish_time = c["ts"] + timedelta(minutes=TEST_DURATION_MIN)
        fb_min = (finish_time - c["ts"]).total_seconds() / 60
        feedback_times[c["commit_id"]] = fb_min
        if c["true_label"]:
            culprit_times.append(fb_min)

    mean_fb = sum(feedback_times.values()) / len(feedback_times) if feedback_times else 0
    mean_ttc = sum(culprit_times) / len(culprit_times) if culprit_times else 0
    max_ttc = max(culprit_times) if culprit_times else 0

    return {
        "total_tests_run": total_tests_run,
        "mean_feedback_time_min": round(mean_fb, 2),
        "mean_time_to_culprit_min": round(mean_ttc, 2),
        "max_time_to_culprit_min": round(max_ttc, 2),
    }


# ===== BISECTION STRATEGY: ROB =====
def risk_ordered_bisect(batch, start_time, total_tests_run, culprit_times, feedback_times):
    """
    Risk-Ordered Bisection. 
    Orders commits by risk score before each split and prioritizes testing the riskier half first.
    """
    total_tests_run += 1
    finish_time = start_time + timedelta(minutes=TEST_DURATION_MIN)

    if not batch_has_regressor(batch):
        # whole batch passes -> feedback for all here
        for c in batch:
            cid = c["commit_id"]
            if cid not in feedback_times:
                fb_min = (finish_time - c["ts"]).total_seconds() / 60
                feedback_times[cid] = fb_min
        return total_tests_run, culprit_times, feedback_times

    if len(batch) == 1:
        c = batch[0]
        fb_min = (finish_time - c["ts"]).total_seconds() / 60
        feedback_times[c["commit_id"]] = fb_min
        if c["true_label"]:
            culprit_times.append(fb_min)
        return total_tests_run, culprit_times, feedback_times

    # sort by risk and split
    batch.sort(key=lambda x: x["risk"], reverse=True)
    mid = len(batch) // 2
    risky_half = batch[:mid]
    safe_half = batch[mid:]

    total_tests_run, culprit_times, feedback_times = risk_ordered_bisect(
        risky_half, finish_time, total_tests_run, culprit_times, feedback_times
    )
    total_tests_run, culprit_times, feedback_times = risk_ordered_bisect(
        safe_half, finish_time, total_tests_run, culprit_times, feedback_times
    )
    return total_tests_run, culprit_times, feedback_times


# ===== BISECTION STRATEGY: UB =====
def unordered_bisect(batch, start_time, total_tests_run, culprit_times, feedback_times):
    """
    Unordered Bisection (UB): plain bisection, recurse halves in random order,
    and fill feedback for passing sub-batches.
    """
    total_tests_run += 1
    finish_time = start_time + timedelta(minutes=TEST_DURATION_MIN)

    if not batch_has_regressor(batch):
        for c in batch:
            cid = c["commit_id"]
            if cid not in feedback_times:
                fb_min = (finish_time - c["ts"]).total_seconds() / 60
                feedback_times[cid] = fb_min
        return total_tests_run, culprit_times, feedback_times

    if len(batch) == 1:
        c = batch[0]
        fb_min = (finish_time - c["ts"]).total_seconds() / 60
        feedback_times[c["commit_id"]] = fb_min
        if c["true_label"]:
            culprit_times.append(fb_min)
        return total_tests_run, culprit_times, feedback_times

    mid = len(batch) // 2
    left = batch[:mid]
    right = batch[mid:]
    halves = [left, right]
    random.shuffle(halves)

    for h in halves:
        total_tests_run, culprit_times, feedback_times = unordered_bisect(
            h, finish_time, total_tests_run, culprit_times, feedback_times
        )

    return total_tests_run, culprit_times, feedback_times


# ===== BATCHING STRATEGY: TIME-WINDOW (TWB-N) =====
def simulate_twb_with_bisect(commits, bisect_fn, batch_hours):
    """
    Batches all commits submitted within the last N hours into a single test run.
    """
    total_tests_run = 0
    culprit_times = []
    feedback_times = {}  # commit_id -> minutes

    batch_start = commits[0]["ts"]
    batch_end = batch_start + timedelta(hours=batch_hours)
    current_batch = []

    for c in commits:
        if c["ts"] < batch_end:
            current_batch.append(c)
        else:
            total_tests_run, culprit_times, feedback_times = bisect_fn(
                current_batch, batch_end, total_tests_run, culprit_times, feedback_times
            )
            batch_start = c["ts"]
            batch_end = batch_start + timedelta(hours=batch_hours)
            current_batch = [c]

    if current_batch:
        total_tests_run, culprit_times, feedback_times = bisect_fn(
            current_batch, batch_end, total_tests_run, culprit_times, feedback_times
        )

    # compute metrics
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


def main():
    pred_map = load_predictions(INPUT_JSON)
    commits = read_commits_from_all(ALL_COMMITS_PATH, pred_map)
    if not commits:
        print("No commits found after cutoff; exiting.")
        return

    # 1) Baseline
    et_results = run_exhaustive_testing(commits)

    # 2) CI-style strategies
    twb_rob_results = simulate_twb_with_bisect(commits, risk_ordered_bisect, BATCH_HOURS)
    twb_ub_results = simulate_twb_with_bisect(commits, unordered_bisect, BATCH_HOURS)

    ci_results = {
        "TWB-N + ROB": twb_rob_results,
        "TWB-N + UB": twb_ub_results,
    }

    # choose bests among CI-like ones
    best_tests = min(ci_results.items(), key=lambda kv: kv[1]["total_tests_run"])
    best_max_ttc = min(ci_results.items(), key=lambda kv: kv[1]["max_time_to_culprit_min"])

    out = {
        "Exhaustive Testing (ET)": et_results,
        "TWB-N + ROB": twb_rob_results,
        "TWB-N + UB": twb_ub_results,
        "best_by_total_tests": best_tests[0],
        "best_by_max_ttc": best_max_ttc[0],
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("saved to", OUTPUT_PATH)
    print(f"Best (fewest tests): {best_tests[0]}")
    print(f"Best (lowest max TTC): {best_max_ttc[0]}")


if __name__ == "__main__":
    main()
