#!/usr/bin/env python3
import os
import json
from datetime import datetime, timedelta

# ====== CONSTS (edit here) ======
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

ALL_COMMITS_PATH = os.path.join(
    REPO_ROOT, "datasets", "mozilla_perf", "all_commits.jsonl"
)
PERF_COMMITS_PATH = os.path.join(
    REPO_ROOT, "datasets", "mozilla_perf", "perf_bugs_with_diff.jsonl"
)
OUTPUT_PATH = os.path.join(
    REPO_ROOT, "analysis", "batch_testing", "simulated_results.json"
)

BATCH_HOURS = 4
TEST_DURATION_MIN = 10
CUTOFF = datetime.fromisoformat("2024-10-10T00:00:00+00:00")  # filter commits after this
# ================================


def load_perf_regressor(path):
    """
    Build a map: revision/node -> is_regressor (True/False)
    perf_bugs_with_diff.jsonl has "regressor": "true"/"false"
    We will match all_commits.jsonl["node"] to this "revision".
    """
    m = {}
    if not os.path.exists(path):
        return m
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rev = obj.get("revision") or obj.get("node")
            if not rev:
                continue
            reg_val = str(obj.get("regressor", "false")).strip().lower()
            is_reg = reg_val == "true"
            m[rev] = is_reg
    return m


def read_commits(all_commits_path, perf_map):
    commits = []
    with open(all_commits_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # all_commits.jsonl fields: node, description, date
            ts = datetime.fromisoformat(obj["date"])
            if ts <= CUTOFF:
                continue
            node = obj["node"]
            is_reg = bool(perf_map.get(node, False))
            commits.append({"ts": ts, "regressor": is_reg})
    commits.sort(key=lambda x: x["ts"])
    return commits


def batch_has_regressor(batch):
    return any(c["regressor"] for c in batch)


def simulate_batch_bisect(batch, start_time, total_tests_run, culprit_times):
    # 1) test whole batch
    total_tests_run += 1
    finish_time = start_time + timedelta(minutes=TEST_DURATION_MIN)

    if not batch_has_regressor(batch):
        return (
            total_tests_run,
            culprit_times,
            {id(c): (finish_time - c["ts"]).total_seconds() / 60 for c in batch},
        )

    # single failing commit
    if len(batch) == 1:
        c = batch[0]
        ttc = (finish_time - c["ts"]).total_seconds() / 60
        culprit_times.append(ttc)
        return total_tests_run, culprit_times, {id(c): ttc}

    # split and recurse
    mid = len(batch) // 2
    left = batch[:mid]
    right = batch[mid:]

    left_total, culprit_times, left_fb = simulate_batch_bisect(
        left, finish_time, total_tests_run, culprit_times
    )
    right_total, culprit_times, right_fb = simulate_batch_bisect(
        right, finish_time, left_total, culprit_times
    )

    left_fb.update(right_fb)
    return right_total, culprit_times, left_fb


def simulate(commits):

    total_tests_run = 0
    all_feedback = []
    all_culprit_times = []

    batch_start_time = commits[0]["ts"]
    batch_end_time = batch_start_time + timedelta(hours=BATCH_HOURS)
    current_batch = []

    def flush(current_batch, batch_end_time, total_tests_run, all_culprit_times, all_feedback):
        if not current_batch:
            return total_tests_run, all_culprit_times, all_feedback
        total_tests_run, all_culprit_times, fb_dict = simulate_batch_bisect(
            current_batch, batch_end_time, total_tests_run, all_culprit_times
        )
        all_feedback.extend(fb_dict.values())
        return total_tests_run, all_culprit_times, all_feedback

    for c in commits:
        if c["ts"] < batch_end_time:
            current_batch.append(c)
        else:
            total_tests_run, all_culprit_times, all_feedback = flush(
                current_batch,
                batch_end_time,
                total_tests_run,
                all_culprit_times,
                all_feedback,
            )
            batch_start_time = c["ts"]
            batch_end_time = batch_start_time + timedelta(hours=BATCH_HOURS)
            current_batch = [c]

    # flush last batch
    total_tests_run, all_culprit_times, all_feedback = flush(
        current_batch,
        batch_end_time,
        total_tests_run,
        all_culprit_times,
        all_feedback,
    )

    mean_fb = sum(all_feedback) / len(all_feedback) if all_feedback else 0
    mean_ttc = sum(all_culprit_times) / len(all_culprit_times) if all_culprit_times else 0
    # max_ttc = max_fb
    max_ttc = max(all_culprit_times) if all_culprit_times else 0 
    
    return {
        "total_tests_run": total_tests_run,
        "mean_feedback_time_min": round(mean_fb, 2),
        "mean_time_to_culprit_min": round(mean_ttc, 2),
        "max_time_to_culprit_min": round(max_ttc, 2),
    }


def main():
    perf_map = load_perf_regressor(PERF_COMMITS_PATH)
    commits = read_commits(ALL_COMMITS_PATH, perf_map)
    results = simulate(commits)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("saved to", OUTPUT_PATH)


if __name__ == "__main__":
    main()
