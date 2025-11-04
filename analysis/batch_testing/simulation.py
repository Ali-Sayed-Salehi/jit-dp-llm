#!/usr/bin/env python3
import os
import json
import random
from datetime import datetime, timedelta, timezone
import argparse

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


# ---- PARAM SWEEPS (multi-resolution; uses existing frange/irange) ----
def frange(start, stop, step, round_to=4):
    """Inclusive-ish float range."""
    vals = []
    v = start
    while v <= stop + 1e-9:
        vals.append(round(v, round_to))
        v += step
    return vals

def irange(start, stop, step=1):
    return list(range(start, stop + 1, step))

# Denser where you found wins (~0.5–0.6, and high-confidence tail), but broad overall
PRED_THRESHOLD_CANDIDATES = (
    frange(0.30, 0.70, 0.01)    # fine around mid-range
    + frange(0.70, 0.90, 0.02)  # moderate
    + frange(0.90, 0.99, 0.005) # fine tail
    + [0.995]
)

# TWB: very fine for short windows, moderate mid, coarse long
_TWB_ALL = [{"BATCH_HOURS": v} for v in (
    frange(0.25, 6.0, 0.25)   # 15-min steps to 6h
    + frange(6.0, 12.0, 0.5)  # 30-min steps 6–12h
    + frange(12.0, 24.0, 1.0) # 1h steps 12–24h
)]

# FSB: fine around 10–16, then broader
_FSB_ALL = [{"FSB_SIZE": v} for v in (
    irange(4, 20, 1)      # fine 4..20
    + irange(22, 50, 2)   # 2-step 22..50
    + irange(60, 200, 10) # coarse checkpoints
)]

# RASB: fine sweep where bests appeared (0.35–0.55), plus broad context
_RASB_ALL = [{"RASB_THRESHOLD": v} for v in (
    frange(0.10, 0.30, 0.05)
    + frange(0.30, 0.60, 0.01)   # fine band
    + frange(0.60, 0.95, 0.05)
)]

# HRAB: windows focused 3–6h (+ 1,2,8); risks dense 0.65–0.80 with broader neighbors
_HRAB_RISKS = (
    frange(0.50, 0.65, 0.05)
    + frange(0.65, 0.80, 0.01)   # fine band
    + frange(0.80, 0.99, 0.05)
)
_HRAB_WINDOWS = (
    irange(1, 8, 1) + irange(10, 24, 2)  # 1..8 by 1, then 10..24 by 2
)
_HRAB_ALL = [
    {"HRAB_RISK_THRESHOLD": t, "HRAB_WINDOW_HOURS": n}
    for t in _HRAB_RISKS
    for n in _HRAB_WINDOWS
]

# DLRTWB: focus t∈[0.60,0.90] (fine), with neighbors; HIGH fine 0.25–3h; LOW fine 3–8h then coarser
_DLRTWB_THRESH = (
    frange(0.40, 0.60, 0.05)
    + frange(0.60, 0.90, 0.01)   # fine band
    + frange(0.90, 0.95, 0.02)
)
_DLRTWB_HIGH_HOURS = (
    frange(0.25, 3.0, 0.25) + frange(3.0, 6.0, 0.5)
)
_DLRTWB_LOW_HOURS = (
    frange(3.0, 8.0, 1.0) + frange(8.0, 24.0, 2.0)
)
_DLRTWB_ALL = [
    {"DLR_RISK_THRESHOLD": t, "DLR_HIGH_HOURS": a, "DLR_LOW_HOURS": b}
    for t in _DLRTWB_THRESH
    for a in _DLRTWB_HIGH_HOURS
    for b in _DLRTWB_LOW_HOURS
]

# RAPB: very fine near bests (T≈0.50–0.60, aging≈0.01), plus broader tails
_RAPB_THRESH = (
    frange(0.30, 0.45, 0.05)
    + frange(0.45, 0.65, 0.01)   # fine band
    + frange(0.65, 0.80, 0.05)
)
_RAPB_AGING = (
    frange(0.005, 0.020, 0.005)  # very fine around 0.01
    + frange(0.020, 0.050, 0.005)
    + frange(0.050, 0.200, 0.01)
)
_RAPB_ALL = [
    {"RAPB_THRESHOLD": t, "RAPB_AGING_PER_HOUR": a}
    for t in _RAPB_THRESH
    for a in _RAPB_AGING
]

PARAM_SWEEPS = {
    "TWB-N": _TWB_ALL,
    "FSB-N": _FSB_ALL,
    "RASB-T": _RASB_ALL,
    "HRAB-T-N": _HRAB_ALL,
    "DLRTWB-T-a-b": _DLRTWB_ALL,
    "RAPB-T-a": _RAPB_ALL,
}

# (Optional) If runtime explodes, cap any giant grid temporarily, e.g.:
# MAX_SWEEP = 5000
# PARAM_SWEEPS["DLRTWB-T-a-b"] = PARAM_SWEEPS["DLRTWB-T-a-b"][:MAX_SWEEP]
# PARAM_SWEEPS["RAPB-T-a"] = PARAM_SWEEPS["RAPB-T-a"][:MAX_SWEEP]

random.seed(RANDOM_SEED)
# =================================

def get_args():
    parser = argparse.ArgumentParser(
        description="Batch-testing simulation"
    )
    parser.add_argument(
        "--input-json-eval",
        default=os.path.join(
            REPO_ROOT, "analysis", "batch_testing", "final_test_results_perf_mbert_eval.json"
        ),
        help="Path to EVAL predictions json"
    )
    parser.add_argument(
        "--input-json-final",
        default=os.path.join(
            REPO_ROOT, "analysis", "batch_testing", "final_test_results_perf_mbert_final_test.json"
        ),
        help="Path to FINAL predictions json"
    )
    parser.add_argument(
        "--output-eval",
        default=os.path.join(
            REPO_ROOT, "analysis", "batch_testing", "batch_sim_results_eval.json"
        ),
        help="Where to write EVAL sim results"
    )
    parser.add_argument(
        "--output-final",
        default=os.path.join(
            REPO_ROOT, "analysis", "batch_testing", "batch_sim_results_final_test.json"
        ),
        help="Where to write FINAL sim results"
    )
    return parser.parse_args()


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
    """Find the oldest and newest commit (by ts) that is present in the given INPUT_JSON."""
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

            if ts <= lower_cutoff:
                continue
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
        mean_fb_hr = round((sum(feedback_times.values()) / len(feedback_times)) / 60.0, 2)
    else:
        mean_fb_hr = 0.0

    if culprit_times:
        mean_ttc_hr = round((sum(culprit_times) / len(culprit_times)) / 60.0, 2)
        max_ttc_hr = round(max(culprit_times) / 60.0, 2)
    else:
        mean_ttc_hr = 0.0
        max_ttc_hr = 0.0

    return {
        "total_tests_run": total_tests_run,
        "mean_feedback_time_hr": mean_fb_hr,
        "mean_time_to_culprit_hr": mean_ttc_hr,
        "max_time_to_culprit_hr": max_ttc_hr,
    }


def convert_result_minutes_to_hours(res):
    """Mutates and returns res: converts *_min to *_hr and drops *_min."""
    if "mean_feedback_time_min" in res:
        res["mean_feedback_time_hr"] = round(res["mean_feedback_time_min"] / 60.0, 2)
        del res["mean_feedback_time_min"]
    if "mean_time_to_culprit_min" in res:
        res["mean_time_to_culprit_hr"] = round(res["mean_time_to_culprit_min"] / 60.0, 2)
        del res["mean_time_to_culprit_min"]
    if "max_time_to_culprit_min" in res:
        res["max_time_to_culprit_hr"] = round(res["max_time_to_culprit_min"] / 60.0, 2)
        del res["max_time_to_culprit_min"]
    return res


def lookup_batching(name):
    for n, fn, default_param in BATCHING_STRATEGIES:
        if n == name:
            return fn, default_param
    return None, None


def lookup_bisection(name):
    for n, fn in BISECTION_STRATEGIES:
        if n == name:
            return fn
    return None


def run_evaluation():
    # figure out window from EVAL input
    tmp_pred_map = load_predictions(
        INPUT_JSON_EVAL, pred_threshold=PRED_THRESHOLD_CANDIDATES[0]
    )
    dynamic_oldest, dynamic_newest = get_cutoff_from_input(ALL_COMMITS_PATH, tmp_pred_map)
    lower_cutoff = dynamic_oldest or DEFAULT_CUTOFF
    upper_cutoff = dynamic_newest

    # build commits for ET/baseline
    base_commits = read_commits_from_all(
        ALL_COMMITS_PATH, tmp_pred_map, lower_cutoff, upper_cutoff
    )
    if not base_commits:
        print("No commits found in EVAL window; exiting eval.")
        return None

    et_results = run_exhaustive_testing(base_commits)
    baseline = simulate_twb_with_bisect(base_commits, unordered_bisect, BATCH_HOURS)
    baseline = convert_result_minutes_to_hours(baseline)
    baseline_fb = baseline["mean_feedback_time_hr"]
    baseline_mean_ttc = baseline["mean_time_to_culprit_hr"]
    baseline_max_ttc = baseline["max_time_to_culprit_hr"]
    baseline_tests = baseline["total_tests_run"]

    def time_saved_pct(base, val):
        if base and base > 0:
            return round((base - val) / base * 100.0, 2)
        return 0.0

    ci_results = {}

    # sweep over prediction thresholds (EVAL PHASE)
    for pred_thr in PRED_THRESHOLD_CANDIDATES:
        pred_map = load_predictions(INPUT_JSON_EVAL, pred_threshold=pred_thr)
        commits = read_commits_from_all(
            ALL_COMMITS_PATH, pred_map, lower_cutoff, upper_cutoff
        )
        if not commits:
            print("  (no commits for this EVAL threshold in window, skipping)")
            continue

        for b_name, b_fn, b_param in BATCHING_STRATEGIES:
            candidates = PARAM_SWEEPS.get(b_name, [b_param])

            for bis_name, bis_fn in BISECTION_STRATEGIES:
                combo_name = f"{b_name} + {bis_name}"

                best_res = None
                best_params = None
                best_tests = float("inf")

                fb_res = None
                fb_params = None
                fb_best_max_ttc = float("inf")

                for param_dict in candidates:
                    # unpack
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
                        param = list(param_dict.values())[0]

                    res = b_fn(commits, bis_fn, param)
                    res = convert_result_minutes_to_hours(res)

                    # fallback
                    if res["max_time_to_culprit_hr"] < fb_best_max_ttc:
                        fb_best_max_ttc = res["max_time_to_culprit_hr"]
                        fb_res = res
                        fb_params = param_dict

                    # feasible (doesn't violate baseline)
                    if res["max_time_to_culprit_hr"] <= baseline_max_ttc:
                        if res["total_tests_run"] < best_tests:
                            best_res = res
                            best_params = param_dict
                            best_tests = res["total_tests_run"]

                if best_res is not None:
                    if baseline_tests > 0:
                        saved_pct = round(
                            (baseline_tests - best_res["total_tests_run"]) / baseline_tests * 100.0,
                            2,
                        )
                    else:
                        saved_pct = 0.0

                    best_res["best_params"] = best_params
                    best_res["tests_saved_vs_baseline_pct"] = saved_pct
                    best_res["violates_baseline"] = False
                    best_res["pred_threshold_used"] = pred_thr
                    best_res["mean_feedback_time_saved_vs_baseline_pct"] = time_saved_pct(
                        baseline_fb, best_res["mean_feedback_time_hr"]
                    )
                    best_res["mean_time_to_culprit_saved_vs_baseline_pct"] = time_saved_pct(
                        baseline_mean_ttc, best_res["mean_time_to_culprit_hr"]
                    )
                    best_res["max_time_to_culprit_saved_vs_baseline_pct"] = time_saved_pct(
                        baseline_max_ttc, best_res["max_time_to_culprit_hr"]
                    )
                    candidate_final = best_res
                else:
                    if fb_res is None:
                        continue

                    if baseline_tests > 0:
                        saved_pct = round(
                            (baseline_tests - fb_res["total_tests_run"]) / baseline_tests * 100.0,
                            2,
                        )
                    else:
                        saved_pct = 0.0

                    fb_res["best_params"] = fb_params
                    fb_res["tests_saved_vs_baseline_pct"] = saved_pct
                    fb_res["violates_baseline"] = True
                    fb_res["pred_threshold_used"] = pred_thr
                    fb_res["mean_feedback_time_saved_vs_baseline_pct"] = time_saved_pct(
                        baseline_fb, fb_res["mean_feedback_time_hr"]
                    )
                    fb_res["mean_time_to_culprit_saved_vs_baseline_pct"] = time_saved_pct(
                        baseline_mean_ttc, fb_res["mean_time_to_culprit_hr"]
                    )
                    fb_res["max_time_to_culprit_saved_vs_baseline_pct"] = time_saved_pct(
                        baseline_max_ttc, fb_res["max_time_to_culprit_hr"]
                    )
                    candidate_final = fb_res

                existing = ci_results.get(combo_name)

                def is_better(new, old):
                    if old is None:
                        return True
                    if not new["violates_baseline"] and old["violates_baseline"]:
                        return True
                    if new["violates_baseline"] and not old["violates_baseline"]:
                        return False
                    return new["total_tests_run"] < old["total_tests_run"]

                if is_better(candidate_final, existing):
                    ci_results[combo_name] = candidate_final

    # summarize eval
    if ci_results:
        best_tests_combo = min(ci_results.items(), key=lambda kv: kv[1]["total_tests_run"])
        best_max_ttc_combo = min(ci_results.items(), key=lambda kv: kv[1]["max_time_to_culprit_hr"])
        best_mean_fb_combo = min(ci_results.items(), key=lambda kv: kv[1]["mean_feedback_time_hr"])
    else:
        best_tests_combo = ("-", {})
        best_max_ttc_combo = ("-", {})
        best_mean_fb_combo = ("-", {})

    out_eval = {
        "Exhaustive Testing (ET)": et_results,
        "Baseline (TWB-N + UB, BATCH_HOURS=4)": baseline,
    }
    out_eval.update(ci_results)
    out_eval["best_by_total_tests"] = best_tests_combo[0]
    out_eval["best_by_max_ttc"] = best_max_ttc_combo[0]
    out_eval["best_by_mean_feedback_time"] = best_mean_fb_combo[0]

    return {
        "eval_output": out_eval,
        "eval_lower_cutoff": lower_cutoff.isoformat(),
        "eval_upper_cutoff": upper_cutoff.isoformat() if upper_cutoff else None,
    }


def run_final_test(eval_payload):
    """
    Use the best params from EVAL and run on the FINAL input json (no sweep).
    """
    # window from FINAL input
    tmp_pred_map_final = load_predictions(
        INPUT_JSON_FINAL, pred_threshold=PRED_THRESHOLD_CANDIDATES[0]
    )
    final_oldest, final_newest = get_cutoff_from_input(ALL_COMMITS_PATH, tmp_pred_map_final)
    final_lower = final_oldest or DEFAULT_CUTOFF
    final_upper = final_newest

    # ET + baseline for final
    base_commits_final = read_commits_from_all(
        ALL_COMMITS_PATH, tmp_pred_map_final, final_lower, final_upper
    )
    if not base_commits_final:
        raise RuntimeError("No commits found in FINAL window; exiting final.")

    et_results_final = run_exhaustive_testing(base_commits_final)
    baseline_final = simulate_twb_with_bisect(base_commits_final, unordered_bisect, BATCH_HOURS)
    baseline_final = convert_result_minutes_to_hours(baseline_final)

    # baseline metrics (for % diffs and violation check)
    baseline_fb = baseline_final["mean_feedback_time_hr"]
    baseline_mean_ttc = baseline_final["mean_time_to_culprit_hr"]
    baseline_max_ttc = baseline_final["max_time_to_culprit_hr"]
    baseline_tests = baseline_final["total_tests_run"]

    def time_saved_pct(base, val):
        if base and base > 0:
            return round((base - val) / base * 100.0, 2)
        return 0.0

    final_results = {
        "Exhaustive Testing (ET)": et_results_final,
        "Baseline (TWB-N + UB, BATCH_HOURS=4)": baseline_final,
    }

    # replay best configs from EVAL on FINAL
    eval_out = eval_payload["eval_output"]

    for key, val in eval_out.items():
        if key in (
            "Exhaustive Testing (ET)",
            "Baseline (TWB-N + UB, BATCH_HOURS=4)",
            "best_by_total_tests",
            "best_by_max_ttc",
            "best_by_mean_feedback_time",
        ):
            continue

        combo_name = key
        eval_res = val
        pred_thr = eval_res.get("pred_threshold_used", PRED_THRESHOLD)

        # get batching/bisection names
        if " + " not in combo_name:
            raise RuntimeError(f"Invalid combo name (no ' + '): {combo_name}")
        b_name, bis_name = combo_name.split(" + ", 1)
        b_fn, _ = lookup_batching(b_name)
        bis_fn = lookup_bisection(bis_name)
        if b_fn is None or bis_fn is None:
            raise RuntimeError(f"Unknown batching or bisection strategy in combo: {combo_name}")

        # load preds for FINAL
        pred_map_final = load_predictions(INPUT_JSON_FINAL, pred_threshold=pred_thr)
        commits_final = read_commits_from_all(
            ALL_COMMITS_PATH, pred_map_final, final_lower, final_upper
        )
        if not commits_final:
            raise RuntimeError(
                f"No commits found in FINAL window for combo {combo_name} and threshold {pred_thr}"
            )

        best_params = eval_res.get("best_params", None)
        if best_params is None:
            continue

        # unpack params like in eval
        if b_name == "HRAB-T-N":
            param = (
                best_params["HRAB_RISK_THRESHOLD"],
                best_params["HRAB_WINDOW_HOURS"],
            )
        elif b_name == "DLRTWB-T-a-b":
            param = (
                best_params["DLR_RISK_THRESHOLD"],
                best_params["DLR_HIGH_HOURS"],
                best_params["DLR_LOW_HOURS"],
            )
        elif b_name == "RAPB-T-a":
            param = (
                best_params["RAPB_THRESHOLD"],
                best_params["RAPB_AGING_PER_HOUR"],
            )
        else:
            param = list(best_params.values())[0]

        res_final = b_fn(commits_final, bis_fn, param)
        res_final = convert_result_minutes_to_hours(res_final)

        violates = res_final["max_time_to_culprit_hr"] > baseline_max_ttc
        if baseline_tests > 0:
            saved_pct = round(
                (baseline_tests - res_final["total_tests_run"]) / baseline_tests * 100.0,
                2,
            )
        else:
            saved_pct = 0.0

        res_final["violates_baseline"] = violates
        res_final["tests_saved_vs_baseline_pct"] = saved_pct
        res_final["mean_feedback_time_saved_vs_baseline_pct"] = time_saved_pct(
            baseline_fb, res_final["mean_feedback_time_hr"]
        )
        res_final["mean_time_to_culprit_saved_vs_baseline_pct"] = time_saved_pct(
            baseline_mean_ttc, res_final["mean_time_to_culprit_hr"]
        )
        res_final["max_time_to_culprit_saved_vs_baseline_pct"] = time_saved_pct(
            baseline_max_ttc, res_final["max_time_to_culprit_hr"]
        )

        res_final["best_params_from_eval"] = best_params
        res_final["pred_threshold_used_from_eval"] = pred_thr

        final_results[combo_name] = res_final

    # summarize best on FINAL
    combo_entries = [
        (k, v)
        for k, v in final_results.items()
        if k not in (
            "Exhaustive Testing (ET)",
            "Baseline (TWB-N + UB, BATCH_HOURS=4)",
        )
    ]
    if not combo_entries:
        raise RuntimeError(
            f"No combo entries found in final results. final_results.items(): {final_results.items()}"
        )

    best_tests_combo = min(combo_entries, key=lambda kv: kv[1]["total_tests_run"])
    best_max_ttc_combo = min(combo_entries, key=lambda kv: kv[1]["max_time_to_culprit_hr"])
    best_mean_fb_combo = min(combo_entries, key=lambda kv: kv[1]["mean_feedback_time_hr"])
    final_results["best_by_total_tests"] = best_tests_combo[0]
    final_results["best_by_max_ttc"] = best_max_ttc_combo[0]
    final_results["best_by_mean_feedback_time"] = best_mean_fb_combo[0]

    return final_results



def main():

    args = get_args()

    global INPUT_JSON_EVAL, INPUT_JSON_FINAL, OUTPUT_PATH_EVAL, OUTPUT_PATH_FINAL
    INPUT_JSON_EVAL = args.input_json_eval
    INPUT_JSON_FINAL = args.input_json_final
    OUTPUT_PATH_EVAL = args.output_eval
    OUTPUT_PATH_FINAL = args.output_final

    # ---- 1) EVALUATION PHASE ----
    eval_payload = run_evaluation()
    if eval_payload is None:
        raise RuntimeError("Evaluation was unseccussful. No eval payload present")

    os.makedirs(os.path.dirname(OUTPUT_PATH_EVAL), exist_ok=True)
    with open(OUTPUT_PATH_EVAL, "w", encoding="utf-8") as f:
        json.dump(eval_payload["eval_output"], f, indent=2)
    print("✅ Saved EVAL results to", OUTPUT_PATH_EVAL)

    # ---- 2) FINAL TEST PHASE (no sweep; reuse best from eval) ----
    final_results = run_final_test(eval_payload)
    if final_results is not None:
        os.makedirs(os.path.dirname(OUTPUT_PATH_FINAL), exist_ok=True)
        with open(OUTPUT_PATH_FINAL, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2)
        print("✅ Saved FINAL TEST results to", OUTPUT_PATH_FINAL)


if __name__ == "__main__":
    main()
