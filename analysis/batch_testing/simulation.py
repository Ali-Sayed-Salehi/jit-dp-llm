#!/usr/bin/env python3
import os
import json
import random
from datetime import datetime, timedelta, timezone

from batch_strats import (
    simulate_twb_with_bisect,
    simulate_fsb_with_bisect,
    simulate_rasb_t_with_bisect,
    simulate_hrab_t_n_with_bisect,
    simulate_dlrtwb_t_ab_with_bisect,
    simulate_rapb_t_a_with_bisect,
)
from bisection_strats import (
    risk_ordered_bisect,
    unordered_bisect,
    time_ordered_bisect,
    risk_weighted_adaptive_bisect,
    top_k_risk_first_bisect,  # if you're keeping TKRB
    TEST_DURATION_MIN,
)

# ====== CONFIG (edit here) ======
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

ALL_COMMITS_PATH = os.path.join(REPO_ROOT, "datasets", "mozilla_perf", "all_commits.jsonl")
INPUT_JSON = os.path.join(REPO_ROOT, "analysis", "batch_testing", "predictor_sim_results.json")
OUTPUT_PATH = os.path.join(REPO_ROOT, "analysis", "batch_testing", "simulated_results.json")

BATCH_HOURS = 4
FSB_SIZE = 20
RASB_THRESHOLD = 0.5
HRAB_RISK_THRESHOLD = 0.8
HRAB_WINDOW_HOURS = 4
DLR_RISK_THRESHOLD = 0.6
DLR_HIGH_HOURS = 1
DLR_LOW_HOURS = 4
RAPB_THRESHOLD = 0.35
RAPB_AGING_PER_HOUR = 0.05

DEFAULT_CUTOFF = datetime.fromisoformat("2024-10-10T00:00:00+00:00")
PRED_THRESHOLD = 0.7
RANDOM_SEED = 42

BATCHING_STRATEGIES = [
    ("TWB-N", simulate_twb_with_bisect, BATCH_HOURS),
    ("FSB-N", simulate_fsb_with_bisect, FSB_SIZE),
    ("RASB-T", simulate_rasb_t_with_bisect, RASB_THRESHOLD),
    ("HRAB-T-N", simulate_hrab_t_n_with_bisect, (HRAB_RISK_THRESHOLD, HRAB_WINDOW_HOURS)),
    ("DLRTWB-T-a-b", simulate_dlrtwb_t_ab_with_bisect, (DLR_RISK_THRESHOLD, DLR_HIGH_HOURS, DLR_LOW_HOURS)),
    ("RAPB-T-a", simulate_rapb_t_a_with_bisect, (RAPB_THRESHOLD, RAPB_AGING_PER_HOUR)),
]

BISECTION_STRATEGIES = [
    ("ROB", risk_ordered_bisect),
    ("UB", unordered_bisect),
    ("TOB", time_ordered_bisect),
    ("RWAB", risk_weighted_adaptive_bisect),
    ("TKRB-K", top_k_risk_first_bisect),
]

# ---- PREDICTION THRESHOLD SWEEP ----
PRED_THRESHOLD_CANDIDATES = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

# ---- PARAM SWEEPS FOR BATCHING STRATEGIES ----
PARAM_SWEEPS = {
    "TWB-N": [
        {"BATCH_HOURS": v}
        for v in [0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 18, 24]
    ],
    "FSB-N": [
        {"FSB_SIZE": v}
        for v in [5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60, 75, 100]
    ],
    "RASB-T": [
        {"RASB_THRESHOLD": v}
        for v in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]
    ],
    "HRAB-T-N": [
        {"HRAB_RISK_THRESHOLD": t, "HRAB_WINDOW_HOURS": n}
        for t in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        for n in [1, 2, 3, 4, 6, 8, 10, 12]
    ],
    "DLRTWB-T-a-b": [
        {"DLR_RISK_THRESHOLD": t, "DLR_HIGH_HOURS": a, "DLR_LOW_HOURS": b}
        for t in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for a in [0.25, 0.5, 1, 1.5, 2, 3]
        for b in [2, 3, 4, 6, 8, 10, 12]
    ],
    "RAPB-T-a": [
        {"RAPB_THRESHOLD": t, "RAPB_AGING_PER_HOUR": a}
        for t in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
        for a in [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.12, 0.15]
    ],
}

# =================================

random.seed(RANDOM_SEED)


def load_predictions(path, pred_threshold):
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

        final_pred = 1 if p_pos >= pred_threshold else 0

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


def get_cutoff_from_input(all_commits_path, pred_map):
    """Find the oldest and newest commit (by ts) that is present in INPUT_JSON."""
    oldest = None
    newest = None
    if not pred_map:
        return None, None
    with open(all_commits_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            node = obj.get("node")
            if node not in pred_map:
                continue
            date_field = obj.get("date")
            if date_field is None:
                continue
            ts = parse_hg_date(date_field)
            if oldest is None or ts < oldest:
                oldest = ts
            if newest is None or ts > newest:
                newest = ts
    return oldest, newest



def read_commits_from_all(all_commits_path, pred_map, lower_cutoff, upper_cutoff=None):
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

            # lower bound: must be AFTER this
            if ts <= lower_cutoff:
                continue

            # upper bound: must be BEFORE / EQUAL this
            if upper_cutoff is not None and ts > upper_cutoff:
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


def main():
    # 1) figure out the time window from INPUT_JSON (oldest + newest commit)
    # we just use the first threshold candidate to identify which commits appear in INPUT_JSON
    tmp_pred_map = load_predictions(
        INPUT_JSON, pred_threshold=PRED_THRESHOLD_CANDIDATES[0]
    )
    dynamic_oldest, dynamic_newest = get_cutoff_from_input(ALL_COMMITS_PATH, tmp_pred_map)

    lower_cutoff = dynamic_oldest or DEFAULT_CUTOFF
    upper_cutoff = dynamic_newest  # can be None if INPUT_JSON commits weren't found

    # 2) build commits once for baseline/exhaustive testing
    base_commits = read_commits_from_all(
        ALL_COMMITS_PATH,
        tmp_pred_map,
        lower_cutoff,
        upper_cutoff,
    )
    if not base_commits:
        print("No commits found in cutoff window; exiting.")
        return

    # Exhaustive testing baseline (doesn't depend on threshold)
    et_results = run_exhaustive_testing(base_commits)

    # Baseline: TWB-N + UB, with fixed BATCH_HOURS=4
    baseline = simulate_twb_with_bisect(base_commits, unordered_bisect, BATCH_HOURS)
    baseline_fb = baseline["mean_feedback_time_min"]
    baseline_mean_ttc = baseline["mean_time_to_culprit_min"]
    baseline_max_ttc = baseline["max_time_to_culprit_min"]
    baseline_tests = baseline["total_tests_run"]

    def pct_diff(val, base):
        if base and base > 0:
            return round((val - base) / base * 100.0, 2)
        return 0.0

    # we will keep: best result per (batching + bisection) across ALL PRED_THRESHOLDs
    ci_results = {}

    # 3) sweep over prediction thresholds
    for pred_thr in PRED_THRESHOLD_CANDIDATES:
        print(f"\n🔎 Sweeping with PRED_THRESHOLD = {pred_thr}")
        pred_map = load_predictions(INPUT_JSON, pred_threshold=pred_thr)
        commits = read_commits_from_all(
            ALL_COMMITS_PATH,
            pred_map,
            lower_cutoff,
            upper_cutoff,
        )
        if not commits:
            print("  (no commits for this threshold in window, skipping)")
            continue

        # 4) for each batching strategy
        for b_name, b_fn, b_param in BATCHING_STRATEGIES:
            # get pre-defined sweep for this strategy (extracted at top of file)
            candidates = PARAM_SWEEPS.get(b_name, [b_param])

            # 5) for each bisection strategy
            for bis_name, bis_fn in BISECTION_STRATEGIES:
                combo_name = f"{b_name} + {bis_name}"

                # best (feasible, i.e. not violating baseline on max_ttc)
                best_res = None
                best_params = None
                best_tests = float("inf")

                # fallback: lowest max_ttc even if it violates baseline
                fb_res = None
                fb_params = None
                fb_best_max_ttc = float("inf")

                for param_dict in candidates:
                    # unpack params according to batching strategy
                    if b_name == "HRAB-T-N":
                        param = (
                            param_dict["HRAB_RISK_THRESHOLD"],
                            param_dict["HRAB_WINDOW_HOURS"],
                        )
                    elif b_name == "DLRTWB-T-a-b":
                        param = (
                            param_dict["DLR_RISK_THRESHOLD"],
                            param_dict["DLR_HIGH_HOURS"],
                            param_dict["DLR_LOW_HOURS"],
                        )
                    elif b_name == "RAPB-T-a":
                        param = (
                            param_dict["RAPB_THRESHOLD"],
                            param_dict["RAPB_AGING_PER_HOUR"],
                        )
                    else:
                        # single-key dict
                        param = list(param_dict.values())[0]

                    res = b_fn(commits, bis_fn, param)

                    # track fallback (the one with lowest max_ttc overall)
                    if res["max_time_to_culprit_min"] < fb_best_max_ttc:
                        fb_best_max_ttc = res["max_time_to_culprit_min"]
                        fb_res = res
                        fb_params = param_dict

                    # feasible: must not exceed baseline max TTC
                    if res["max_time_to_culprit_min"] <= baseline_max_ttc:
                        if res["total_tests_run"] < best_tests:
                            best_res = res
                            best_params = param_dict
                            best_tests = res["total_tests_run"]

                # build final candidate for this (combo, threshold)
                if best_res is not None:
                    # doesn't violate baseline
                    if baseline_tests > 0:
                        saved_pct = round(
                            (baseline_tests - best_res["total_tests_run"]) / baseline_tests * 100.0,
                            2,
                        )
                    else:
                        saved_pct = 0.0

                    best_res["best_params"] = best_params
                    best_res["tests_saved_pct_vs_baseline"] = saved_pct
                    best_res["violates_baseline"] = False
                    best_res["pred_threshold_used"] = pred_thr
                    best_res["mean_feedback_time_pct_vs_baseline"] = pct_diff(
                        best_res["mean_feedback_time_min"], baseline_fb
                    )
                    best_res["mean_time_to_culprit_pct_vs_baseline"] = pct_diff(
                        best_res["mean_time_to_culprit_min"], baseline_mean_ttc
                    )
                    best_res["max_time_to_culprit_pct_vs_baseline"] = pct_diff(
                        best_res["max_time_to_culprit_min"], baseline_max_ttc
                    )
                    candidate_final = best_res
                else:
                    # no feasible config -> use fallback
                    if fb_res is None:
                        # nothing to record for this combo+threshold
                        continue

                    if baseline_tests > 0:
                        saved_pct = round(
                            (baseline_tests - fb_res["total_tests_run"]) / baseline_tests * 100.0,
                            2,
                        )
                    else:
                        saved_pct = 0.0

                    fb_res["best_params"] = fb_params
                    fb_res["tests_saved_pct_vs_baseline"] = saved_pct
                    fb_res["violates_baseline"] = True
                    fb_res["pred_threshold_used"] = pred_thr
                    fb_res["mean_feedback_time_pct_vs_baseline"] = pct_diff(
                        fb_res["mean_feedback_time_min"], baseline_fb
                    )
                    fb_res["mean_time_to_culprit_pct_vs_baseline"] = pct_diff(
                        fb_res["mean_time_to_culprit_min"], baseline_mean_ttc
                    )
                    fb_res["max_time_to_culprit_pct_vs_baseline"] = pct_diff(
                        fb_res["max_time_to_culprit_min"], baseline_max_ttc
                    )
                    candidate_final = fb_res

                # compare with what we already stored for this combo (from previous thresholds)
                existing = ci_results.get(combo_name)

                def is_better(new, old):
                    if old is None:
                        return True
                    # prefer non-violating over violating
                    if not new["violates_baseline"] and old["violates_baseline"]:
                        return True
                    if new["violates_baseline"] and not old["violates_baseline"]:
                        return False
                    # both same violation status -> prefer fewer tests
                    return new["total_tests_run"] < old["total_tests_run"]

                if is_better(candidate_final, existing):
                    ci_results[combo_name] = candidate_final
                    print(
                        f"✅ {combo_name} improved with PRED_THRESHOLD={pred_thr}: "
                        f"tests={candidate_final['total_tests_run']}, "
                        f"violates_baseline={candidate_final['violates_baseline']}"
                    )

    # 6) summarize best overall
    if ci_results:
        best_tests_combo = min(
            ci_results.items(), key=lambda kv: kv[1]["total_tests_run"]
        )
        best_max_ttc_combo = min(
            ci_results.items(), key=lambda kv: kv[1]["max_time_to_culprit_min"]
        )
        best_mean_fb_combo = min(
            ci_results.items(), key=lambda kv: kv[1]["mean_feedback_time_min"]
        )
    else:
        best_tests_combo = ("-", {})
        best_max_ttc_combo = ("-", {})
        best_mean_fb_combo = ("-", {})

    out = {
        "Exhaustive Testing (ET)": et_results,
        "Baseline (TWB-N + UB, BATCH_HOURS=4)": baseline,
    }
    out.update(ci_results)
    out["best_by_total_tests"] = best_tests_combo[0]
    out["best_by_max_ttc"] = best_max_ttc_combo[0]
    out["best_by_mean_feedback_time"] = best_mean_fb_combo[0]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Saved to", OUTPUT_PATH)
    print(f"Best (fewest tests): {best_tests_combo[0]}")
    print(f"Best (lowest max TTC): {best_max_ttc_combo[0]}")
    print(f"Best (lowest mean feedback): {best_mean_fb_combo[0]}")


if __name__ == "__main__":
    main()
