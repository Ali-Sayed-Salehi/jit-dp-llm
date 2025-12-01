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
    simulate_rapb_t_a_with_bisect,
    simulate_rrbb_with_bisect
)
from bisection_strats import (
    time_ordered_bisect,
    time_ordered_linear,
    time_ordered_parallel,
    risk_weighted_adaptive_bisect,
    TEST_DURATION_MIN,
    TESTS_PER_RUN,
    TestExecutor,
    run_test_suite,
)

# ====== CONFIG (edit here) ======
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

ALL_COMMITS_PATH = os.path.join(REPO_ROOT, "datasets", "mozilla_perf", "all_commits.jsonl")

BATCH_HOURS = 4
FSB_SIZE = 20
RASB_THRESHOLD = 0.5
RAPB_THRESHOLD = 0.35
RAPB_AGING_PER_HOUR = 0.05

DEFAULT_CUTOFF = datetime.fromisoformat("2024-10-10T00:00:00+00:00")
PRED_THRESHOLD = 0.7
RANDOM_SEED = 42

NUM_TEST_WORKERS = 100  # <--- central test machine capacity (K)

BATCHING_STRATEGIES = [
    ("TWB-N", simulate_twb_with_bisect, BATCH_HOURS),
    ("FSB-N", simulate_fsb_with_bisect, FSB_SIZE),
    ("RASB-T", simulate_rasb_t_with_bisect, RASB_THRESHOLD),
    ("RAPB-T-a", simulate_rapb_t_a_with_bisect, (RAPB_THRESHOLD, RAPB_AGING_PER_HOUR)),
    ("RRBB-T", simulate_rrbb_with_bisect, 1.0),
]

BISECTION_STRATEGIES = [
    ("TOB", time_ordered_bisect),
    ("SEQ", time_ordered_linear),
    ("PAR", time_ordered_parallel),
    ("RWAB", risk_weighted_adaptive_bisect),
]

# ---- PARAM SWEEPS (kept for --mode sweep; uses existing frange/irange) ----
def frange(start, stop, step, round_to=4):
    vals = []
    v = start
    while v <= stop + 1e-9:
        vals.append(round(v, round_to))
        v += step
    return vals

def irange(start, stop, step=1):
    return list(range(start, stop + 1, step))

PRED_THRESHOLD_CANDIDATES = (
    frange(0.30, 0.70, 0.01)
    + frange(0.70, 0.90, 0.02)
    + frange(0.90, 0.99, 0.005)
    + [0.995]
)

_TWB_ALL = [{"BATCH_HOURS": v} for v in (
    frange(0.25, 6.0, 0.25)
    + frange(6.0, 12.0, 0.5)
    + frange(12.0, 24.0, 1.0)
)]
_FSB_ALL = [{"FSB_SIZE": v} for v in (
    irange(4, 20, 1) + irange(22, 50, 2) + irange(60, 200, 10)
)]
_RASB_ALL = [{"RASB_THRESHOLD": v} for v in (
    frange(0.10, 0.30, 0.05)
    + frange(0.30, 0.60, 0.01)
    + frange(0.60, 0.95, 0.05)
)]

_RAPB_THRESH = (
    frange(0.30, 0.45, 0.05)
    + frange(0.45, 0.65, 0.01)
    + frange(0.65, 0.80, 0.05)
)
_RAPB_AGING = (
    frange(0.005, 0.020, 0.005)
    + frange(0.020, 0.050, 0.005)
    + frange(0.050, 0.200, 0.01)
)
_RAPB_ALL = [{"RAPB_THRESHOLD": t, "RAPB_AGING_PER_HOUR": a} for t in _RAPB_THRESH for a in _RAPB_AGING]

_RRBB_ALL = [{"RRBB_BUDGET": v} for v in (
    frange(0.25, 1.0, 0.05)   # fine-grained for low budgets
    + frange(1.0, 3.0, 0.10)  # medium range
    + frange(3.0, 6.0, 0.25)  # coarser for high budgets
)]

PARAM_SWEEPS = {
    "TWB-N": _TWB_ALL,
    "FSB-N": _FSB_ALL,
    "RASB-T": _RASB_ALL,
    "RAPB-T-a": _RAPB_ALL,
    "RRBB-T": _RRBB_ALL
}

random.seed(RANDOM_SEED)

# =================================
def get_args():
    parser = argparse.ArgumentParser(description="Batch-testing simulation")
    parser.add_argument("--mode", choices=["sweep", "mopt"], default="sweep",
                        help="sweep: grid/param sweep. mopt: multi-objective optimization with Optuna (continuous search).")
    parser.add_argument("--mopt-trials", type=int, default=200,
                        help="Number of Optuna trials per (batch,bisect) combo in mopt mode.")
    parser.add_argument(
        "--input-json-eval",
        default=os.path.join(REPO_ROOT, "analysis", "batch_testing", "final_test_results_perf_mbert_eval.json"),
        help="Path to EVAL predictions json"
    )
    parser.add_argument(
        "--input-json-final",
        default=os.path.join(REPO_ROOT, "analysis", "batch_testing", "final_test_results_perf_mbert_final_test.json"),
        help="Path to FINAL predictions json"
    )
    parser.add_argument(
        "--output-eval",
        default=os.path.join(REPO_ROOT, "analysis", "batch_testing", "batch_sim_results_eval.json"),
        help="Where to write EVAL sim results (and where --final-only will read from)"
    )
    parser.add_argument(
        "--output-final",
        default=os.path.join(REPO_ROOT, "analysis", "batch_testing", "batch_sim_results_final_test.json"),
        help="Where to write FINAL sim results"
    )
    # skip eval and only run FINAL using precomputed eval results
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Skip running eval; load eval results from --output-eval and run only the FINAL replay."
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
    """
    Exhaustive Testing (ET) with a central executor:
    - Every commit gets its own full perf run.
    - Tests are submitted at the commit timestamp.
    - A central machine with NUM_TEST_WORKERS workers runs them, each taking TEST_DURATION_MIN.
    - Queue wait time is reflected in feedback metrics.
    """
    if not commits:
        return {
            "total_tests_run": 0,
            "mean_feedback_time_hr": 0.0,
            "mean_time_to_culprit_hr": 0.0,
            "max_time_to_culprit_hr": 0.0,
        }

    total_tests_run = 0
    culprit_times = []
    feedback_times = {}

    # Central executor for this ET run
    executor = TestExecutor(NUM_TEST_WORKERS, TEST_DURATION_MIN)

    for c in commits:
        submit_time = c["ts"]  # test becomes available when the commit lands
        finish_time = run_test_suite(executor, submit_time, TESTS_PER_RUN)

        # Count the test cost just like before
        total_tests_run += TESTS_PER_RUN

        fb_min = (finish_time - c["ts"]).total_seconds() / 60.0
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


# ------------------- BEST-OVERALL LOGIC -------------------
BEST_OVERALL_FIELDS = [
    "tests_saved_vs_baseline_pct",
    "mean_feedback_time_saved_vs_baseline_pct",
    "mean_time_to_culprit_saved_vs_baseline_pct",
    "max_time_to_culprit_saved_vs_baseline_pct",
]


def _overall_improvement_score(entry):
    """
    Overall improvement score:
        tests_saved_vs_baseline_pct
        + (mean_feedback_time_saved_vs_baseline_pct
           + mean_time_to_culprit_saved_vs_baseline_pct
           + max_time_to_culprit_saved_vs_baseline_pct) / 3
    """
    tests_saved = entry.get("tests_saved_vs_baseline_pct", 0.0)
    mean_fb = entry.get("mean_feedback_time_saved_vs_baseline_pct", 0.0)
    mean_ttc = entry.get("mean_time_to_culprit_saved_vs_baseline_pct", 0.0)
    max_ttc = entry.get("max_time_to_culprit_saved_vs_baseline_pct", 0.0)
    return tests_saved + (mean_fb + mean_ttc + max_ttc) / 3.0


def _is_better_or_equal_to_baseline(entry):
    """
    Returns True if this combo is at least as good as baseline
    on all tracked metrics (i.e., no negative improvements).
    """
    return all(entry.get(f, 0.0) >= 0.0 for f in BEST_OVERALL_FIELDS)


def choose_best_overall_from_items(items):
    """
    items: list of (name, entry_dict)

    Logic:
      1. Prefer combos that are >= baseline on all metrics
         (no negative saved_vs_baseline_pct*).
      2. If none, fall back to all combos.
      3. Within the pool, pick the one with the highest overall improvement score.
      4. If the best score <= 0, return "NA".
    """
    if not items:
        return "NA"

    # First, separate "all-metrics-non-worse" combos
    non_worse = [(name, v) for name, v in items if _is_better_or_equal_to_baseline(v)]
    candidate_pool = non_worse if non_worse else items

    best_name, best_entry = max(candidate_pool, key=lambda kv: _overall_improvement_score(kv[1]))
    if _overall_improvement_score(best_entry) <= 0.0:
        return "NA"
    return best_name


# ------------------- SWEEP PIPELINE -------------------
def run_evaluation_sweep(INPUT_JSON_EVAL):
    # window from EVAL input
    tmp_pred_map = load_predictions(INPUT_JSON_EVAL, pred_threshold=PRED_THRESHOLD_CANDIDATES[0])
    dynamic_oldest, dynamic_newest = get_cutoff_from_input(ALL_COMMITS_PATH, tmp_pred_map)
    lower_cutoff = dynamic_oldest or DEFAULT_CUTOFF
    upper_cutoff = dynamic_newest

    # ET & baseline
    base_commits = read_commits_from_all(ALL_COMMITS_PATH, tmp_pred_map, lower_cutoff, upper_cutoff)
    if not base_commits:
        return None

    et_results = run_exhaustive_testing(base_commits)
    baseline = simulate_twb_with_bisect(base_commits, time_ordered_bisect, BATCH_HOURS, NUM_TEST_WORKERS)
    baseline = convert_result_minutes_to_hours(baseline)

    baseline_fb = baseline["mean_feedback_time_hr"]
    baseline_mean_ttc = baseline["mean_time_to_culprit_hr"]
    baseline_max_ttc = baseline["max_time_to_culprit_hr"]
    baseline_tests = baseline["total_tests_run"]

    def time_saved_pct(base, val):
        return round((base - val) / base * 100.0, 2) if base and base > 0 else 0.0

    ci_results = {}
    for pred_thr in PRED_THRESHOLD_CANDIDATES:
        pred_map = load_predictions(INPUT_JSON_EVAL, pred_threshold=pred_thr)
        commits = read_commits_from_all(ALL_COMMITS_PATH, pred_map, lower_cutoff, upper_cutoff)
        if not commits:
            continue

        for b_name, b_fn, _b_default in BATCHING_STRATEGIES:
            candidates = PARAM_SWEEPS.get(b_name, [])
            for bis_name, bis_fn in BISECTION_STRATEGIES:
                combo_name = f"{b_name} + {bis_name}"

                best_res = None
                best_params = None
                best_tests = float("inf")
                fb_res = None
                fb_params = None
                fb_best_max_ttc = float("inf")

                for param_dict in candidates:
                    if b_name == "RAPB-T-a":
                        param = (param_dict["RAPB_THRESHOLD"], param_dict["RAPB_AGING_PER_HOUR"])
                    else:
                        param = list(param_dict.values())[0]

                    res = b_fn(commits, bis_fn, param, NUM_TEST_WORKERS)
                    res = convert_result_minutes_to_hours(res)

                    if res["max_time_to_culprit_hr"] < fb_best_max_ttc:
                        fb_best_max_ttc = res["max_time_to_culprit_hr"]
                        fb_res = res
                        fb_params = param_dict

                    if res["max_time_to_culprit_hr"] <= baseline_max_ttc:
                        if res["total_tests_run"] < best_tests:
                            best_res = res
                            best_params = param_dict
                            best_tests = res["total_tests_run"]

                candidate_final = None
                if best_res is not None:
                    saved_pct = time_saved_pct(baseline_tests, best_res["total_tests_run"])
                    best_res.update({
                        "best_params": best_params,
                        "tests_saved_vs_baseline_pct": saved_pct,
                        "violates_baseline": False,
                        "pred_threshold_used": pred_thr,
                        "mean_feedback_time_saved_vs_baseline_pct": time_saved_pct(baseline_fb, best_res["mean_feedback_time_hr"]),
                        "mean_time_to_culprit_saved_vs_baseline_pct": time_saved_pct(baseline_mean_ttc, best_res["mean_time_to_culprit_hr"]),
                        "max_time_to_culprit_saved_vs_baseline_pct": time_saved_pct(baseline_max_ttc, best_res["max_time_to_culprit_hr"]),
                    })
                    candidate_final = best_res
                else:
                    if fb_res is None:
                        continue
                    saved_pct = time_saved_pct(baseline_tests, fb_res["total_tests_run"])
                    fb_res.update({
                        "best_params": fb_params,
                        "tests_saved_vs_baseline_pct": saved_pct,
                        "violates_baseline": True,
                        "pred_threshold_used": pred_thr,
                        "mean_feedback_time_saved_vs_baseline_pct": time_saved_pct(baseline_fb, fb_res["mean_feedback_time_hr"]),
                        "mean_time_to_culprit_saved_vs_baseline_pct": time_saved_pct(baseline_mean_ttc, fb_res["mean_time_to_culprit_hr"]),
                        "max_time_to_culprit_saved_vs_baseline_pct": time_saved_pct(baseline_max_ttc, fb_res["max_time_to_culprit_hr"]),
                    })
                    candidate_final = fb_res

                existing = ci_results.get(combo_name)
                def is_better(new, old):
                    if old is None: return True
                    if not new["violates_baseline"] and old["violates_baseline"]: return True
                    if new["violates_baseline"] and not old["violates_baseline"]: return False
                    return new["total_tests_run"] < old["total_tests_run"]
                if is_better(candidate_final, existing):
                    ci_results[combo_name] = candidate_final

    # summarize
    out_eval = {
        "Exhaustive Testing (ET)": et_results,
        "Baseline (TWB-N + TOB, BATCH_HOURS=4)": baseline,
        "num_test_workers": NUM_TEST_WORKERS,
    }
    out_eval.update(ci_results)

    # Per-metric bests
    out_eval["best_by_total_tests"] = (
        min(ci_results.items(), key=lambda kv: kv[1]["total_tests_run"])[0]
        if ci_results else "-"
    )
    out_eval["best_by_max_ttc"] = (
        min(ci_results.items(), key=lambda kv: kv[1]["max_time_to_culprit_hr"])[0]
        if ci_results else "-"
    )
    out_eval["best_by_mean_feedback_time"] = (
        min(ci_results.items(), key=lambda kv: kv[1]["mean_feedback_time_hr"])[0]
        if ci_results else "-"
    )

    # Best overall vs baseline (centralized logic)
    items = list(ci_results.items())
    out_eval["bet_overall_improvement_over_baseline"] = choose_best_overall_from_items(items)

    return {
        "eval_output": out_eval,
        "eval_lower_cutoff": lower_cutoff.isoformat(),
        "eval_upper_cutoff": upper_cutoff.isoformat() if upper_cutoff else None,
        "mode": "sweep"
    }


# ------------------- MOPT (Optuna, continuous search) -------------------
def run_evaluation_mopt(INPUT_JSON_EVAL, n_trials):
    try:
        import optuna
    except ImportError as e:
        raise RuntimeError("Optuna is required for --mode mopt. Install with `pip install optuna`.") from e

    tmp_pred_map = load_predictions(INPUT_JSON_EVAL, pred_threshold=PRED_THRESHOLD_CANDIDATES[0])
    dynamic_oldest, dynamic_newest = get_cutoff_from_input(ALL_COMMITS_PATH, tmp_pred_map)
    lower_cutoff = dynamic_oldest or DEFAULT_CUTOFF
    upper_cutoff = dynamic_newest

    base_commits_for_context = read_commits_from_all(ALL_COMMITS_PATH, tmp_pred_map, lower_cutoff, upper_cutoff)
    et_results = run_exhaustive_testing(base_commits_for_context) if base_commits_for_context else {}
    baseline = {}
    if base_commits_for_context:
        baseline = simulate_twb_with_bisect(base_commits_for_context, time_ordered_bisect, BATCH_HOURS, NUM_TEST_WORKERS)
        baseline = convert_result_minutes_to_hours(baseline)
    baseline_max_ttc = baseline.get("max_time_to_culprit_hr", None)

    def pick_best(pareto, baseline_max_ttc_local):
        if not pareto:
            return None
        feas = [r for r in pareto if baseline_max_ttc_local is None or r["max_time_to_culprit_hr"] <= baseline_max_ttc_local]
        if feas:
            return min(feas, key=lambda r: (r["total_tests_run"], r["max_time_to_culprit_hr"]))
        return min(pareto, key=lambda r: (r["max_time_to_culprit_hr"], r["total_tests_run"]))

    # unified output (same shape as sweep)
    out_eval = {
        "Exhaustive Testing (ET)": et_results,
        "Baseline (TWB-N + TOB, BATCH_HOURS=4)": baseline,
        "num_test_workers": NUM_TEST_WORKERS,
    }

    # Per combo study with continuous/int ranges
    for b_name, b_fn, _ in BATCHING_STRATEGIES:
        for bis_name, bis_fn in BISECTION_STRATEGIES:
            combo_key = f"{b_name} + {bis_name}"

            def objective(trial):
                # Shared continuous pred threshold
                pred_thr = trial.suggest_float("pred_threshold", 0.30, 0.995)
                # Strategy-specific continuous/int params
                if b_name == "TWB-N":
                    param = trial.suggest_float("BATCH_HOURS", 0.25, 24.0)
                elif b_name == "FSB-N":
                    param = trial.suggest_int("FSB_SIZE", 4, 200)
                elif b_name == "RASB-T":
                    param = trial.suggest_float("RASB_THRESHOLD", 0.10, 0.95)
                elif b_name == "RAPB-T-a":
                    T = trial.suggest_float("RAPB_THRESHOLD", 0.30, 0.80)
                    a = trial.suggest_float("RAPB_AGING_PER_HOUR", 0.005, 0.200)
                    param = (T, a)
                else:
                    return (float("inf"), float("inf"))

                pred_map = load_predictions(INPUT_JSON_EVAL, pred_threshold=pred_thr)
                commits = read_commits_from_all(ALL_COMMITS_PATH, pred_map, lower_cutoff, upper_cutoff)
                if not commits:
                    return (float("inf"), float("inf"))

                res = b_fn(commits, bis_fn, param, NUM_TEST_WORKERS)
                res = convert_result_minutes_to_hours(res)
                return (res.get("total_tests_run", float("inf")), res.get("max_time_to_culprit_hr", float("inf")))

            study = optuna.create_study(directions=["minimize", "minimize"])
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

            # Build Pareto & choose a single best for unified eval_output
            pareto = []
            for t in study.best_trials:
                params = t.params
                pred_thr = params.get("pred_threshold", PRED_THRESHOLD)

                # unpack for re-eval
                if b_name == "TWB-N":
                    param = params["BATCH_HOURS"]
                elif b_name == "FSB-N":
                    param = int(params["FSB_SIZE"])
                elif b_name == "RASB-T":
                    param = params["RASB_THRESHOLD"]
                elif b_name == "RAPB-T-a":
                    param = (params["RAPB_THRESHOLD"], params["RAPB_AGING_PER_HOUR"])
                else:
                    continue

                pred_map = load_predictions(INPUT_JSON_EVAL, pred_threshold=pred_thr)
                commits = read_commits_from_all(ALL_COMMITS_PATH, pred_map, lower_cutoff, upper_cutoff)
                if not commits:
                    continue
                res = b_fn(commits, bis_fn, param, NUM_TEST_WORKERS)
                res = convert_result_minutes_to_hours(res)
                pareto.append({
                    "pred_threshold_used": pred_thr,
                    "best_params": (
                        {"BATCH_HOURS": param} if b_name == "TWB-N" else
                        {"FSB_SIZE": param} if b_name == "FSB-N" else
                        {"RASB_THRESHOLD": param} if b_name == "RASB-T" else
                        {"RAPB_THRESHOLD": param[0], "RAPB_AGING_PER_HOUR": param[1]}
                    ),
                    "total_tests_run": res.get("total_tests_run"),
                    "mean_feedback_time_hr": res.get("mean_feedback_time_hr"),
                    "mean_time_to_culprit_hr": res.get("mean_time_to_culprit_hr"),
                    "max_time_to_culprit_hr": res.get("max_time_to_culprit_hr"),
                    "violates_baseline": (
                        baseline_max_ttc is not None and
                        res.get("max_time_to_culprit_hr", 0) > baseline_max_ttc
                    )
                })

            selected = pick_best(pareto, baseline_max_ttc)
            if selected is None:
                continue

            # compute deltas vs baseline for the selected
            baseline_fb = baseline.get("mean_feedback_time_hr", None)
            baseline_mean_ttc = baseline.get("mean_time_to_culprit_hr", None)
            baseline_tests = baseline.get("total_tests_run", None)

            def time_saved_pct(base, val):
                return round((base - val) / base * 100.0, 2) if base and base > 0 else 0.0

            result_entry = {
                "total_tests_run": selected["total_tests_run"],
                "mean_feedback_time_hr": selected["mean_feedback_time_hr"],
                "mean_time_to_culprit_hr": selected["mean_time_to_culprit_hr"],
                "max_time_to_culprit_hr": selected["max_time_to_culprit_hr"],
                "violates_baseline": selected["violates_baseline"],
                "best_params": selected["best_params"],
                "pred_threshold_used": selected["pred_threshold_used"],
                "tests_saved_vs_baseline_pct": time_saved_pct(baseline_tests, selected["total_tests_run"]) if baseline_tests else 0.0,
                "mean_feedback_time_saved_vs_baseline_pct": time_saved_pct(baseline_fb, selected["mean_feedback_time_hr"]) if baseline_fb is not None else 0.0,
                "mean_time_to_culprit_saved_vs_baseline_pct": time_saved_pct(baseline_mean_ttc, selected["mean_time_to_culprit_hr"]) if baseline_mean_ttc is not None else 0.0,
                "max_time_to_culprit_saved_vs_baseline_pct": time_saved_pct(baseline_max_ttc, selected["max_time_to_culprit_hr"]) if baseline_max_ttc is not None else 0.0,
                "_mopt_pareto_sample_size": len(pareto)
            }
            out_eval[combo_key] = result_entry

    # summary (unified)
    combo_items = [
        (k, v)
        for k, v in out_eval.items()
        if k not in (
            "Exhaustive Testing (ET)",
            "Baseline (TWB-N + TOB, BATCH_HOURS=4)",
            "num_test_workers",
        )
    ]

    # Per-metric bests
    out_eval["best_by_total_tests"] = (
        min(combo_items, key=lambda kv: kv[1]["total_tests_run"])[0]
        if combo_items else "-"
    )
    out_eval["best_by_max_ttc"] = (
        min(combo_items, key=lambda kv: kv[1]["max_time_to_culprit_hr"])[0]
        if combo_items else "-"
    )
    out_eval["best_by_mean_feedback_time"] = (
        min(combo_items, key=lambda kv: kv[1]["mean_feedback_time_hr"])[0]
        if combo_items else "-"
    )

    # Best overall vs baseline (centralized logic)
    out_eval["bet_overall_improvement_over_baseline"] = choose_best_overall_from_items(combo_items)

    return {
        "eval_output": out_eval,
        "eval_lower_cutoff": lower_cutoff.isoformat(),
        "eval_upper_cutoff": upper_cutoff.isoformat() if upper_cutoff else None,
        "mode": "mopt"
    }


# ------------------- FINAL REPLAY (unified) -------------------
def run_final_test_unified(eval_payload, INPUT_JSON_FINAL, OUTPUT_PATH_FINAL):
    # Build FINAL window from FINAL predictions
    tmp_pred_map_final = load_predictions(INPUT_JSON_FINAL, pred_threshold=PRED_THRESHOLD_CANDIDATES[0])
    final_oldest, final_newest = get_cutoff_from_input(ALL_COMMITS_PATH, tmp_pred_map_final)
    final_lower = final_oldest or DEFAULT_CUTOFF
    final_upper = final_newest

    # ET + Baseline on FINAL window
    base_commits_final = read_commits_from_all(ALL_COMMITS_PATH, tmp_pred_map_final, final_lower, final_upper)
    if not base_commits_final:
        raise RuntimeError("No commits found in FINAL window; exiting final.")

    et_results_final = run_exhaustive_testing(base_commits_final)
    baseline_final = simulate_twb_with_bisect(base_commits_final, time_ordered_bisect, BATCH_HOURS, NUM_TEST_WORKERS)
    baseline_final = convert_result_minutes_to_hours(baseline_final)

    baseline_fb = baseline_final["mean_feedback_time_hr"]
    baseline_mean_ttc = baseline_final["mean_time_to_culprit_hr"]
    baseline_max_ttc = baseline_final["max_time_to_culprit_hr"]
    baseline_tests = baseline_final["total_tests_run"]

    def time_saved_pct(base, val):
        return round((base - val) / base * 100.0, 2) if base and base > 0 else 0.0

    final_results = {
        "Exhaustive Testing (ET)": et_results_final,
        "Baseline (TWB-N + TOB, BATCH_HOURS=4)": baseline_final,
        "final_window": {"lower": final_lower.isoformat(), "upper": final_upper.isoformat() if final_upper else None},
        "num_test_workers": NUM_TEST_WORKERS,
    }

    eval_out = eval_payload["eval_output"]

    # Replay every combo found in eval_out (except ET/Baseline/summary keys)
    for combo_name, val in eval_out.items():
        if combo_name in (
            "Exhaustive Testing (ET)",
            "Baseline (TWB-N + TOB, BATCH_HOURS=4)",
            "best_by_total_tests",
            "best_by_max_ttc",
            "best_by_mean_feedback_time",
            "bet_overall_improvement_over_baseline",
            "num_test_workers",
        ):
            continue

        # Expect "Batch + Bisect" naming
        if " + " not in combo_name:
            continue
        b_name, bis_name = combo_name.split(" + ", 1)
        b_fn, _ = lookup_batching(b_name)
        bis_fn = lookup_bisection(bis_name)
        if b_fn is None or bis_fn is None:
            continue

        best_params = val.get("best_params")
        if not isinstance(best_params, dict):
            # Skip malformed entries
            continue

        pred_thr = val.get("pred_threshold_used", PRED_THRESHOLD)

        # Unpack params for this strategy
        if b_name == "RAPB-T-a":
            param = (best_params["RAPB_THRESHOLD"], best_params["RAPB_AGING_PER_HOUR"])
        else:
            # single-valued dict e.g., {"BATCH_HOURS": x} or {"FSB_SIZE": n} or {"RASB_THRESHOLD": t}
            # take first (and only) value
            try:
                param = list(best_params.values())[0]
            except Exception:
                continue

        # Build FINAL commits using FINAL predictions and the chosen threshold
        pred_map_final = load_predictions(INPUT_JSON_FINAL, pred_threshold=pred_thr)
        commits_final = read_commits_from_all(ALL_COMMITS_PATH, pred_map_final, final_lower, final_upper)
        if not commits_final:
            continue

        res_final = b_fn(commits_final, bis_fn, param, NUM_TEST_WORKERS)
        res_final = convert_result_minutes_to_hours(res_final)

        violates = res_final.get("max_time_to_culprit_hr", float("inf")) > baseline_max_ttc
        saved_pct = time_saved_pct(baseline_tests, res_final.get("total_tests_run", baseline_tests))

        res_final.update({
            "violates_baseline": violates,
            "tests_saved_vs_baseline_pct": saved_pct,
            "mean_feedback_time_saved_vs_baseline_pct": time_saved_pct(baseline_fb, res_final.get("mean_feedback_time_hr", baseline_fb)),
            "mean_time_to_culprit_saved_vs_baseline_pct": time_saved_pct(baseline_mean_ttc, res_final.get("mean_time_to_culprit_hr", baseline_mean_ttc)),
            "max_time_to_culprit_saved_vs_baseline_pct": time_saved_pct(baseline_max_ttc, res_final.get("max_time_to_culprit_hr", baseline_max_ttc)),
            "best_params_from_eval": best_params,
            "pred_threshold_used_from_eval": pred_thr,
        })

        final_results[combo_name] = res_final

    # ----- Robust summary winners on FINAL (avoid KeyError) -----
    eligible = []
    for k, v in final_results.items():
        if k in (
            "Exhaustive Testing (ET)",
            "Baseline (TWB-N + TOB, BATCH_HOURS=4)",
            "final_window",
            "best_by_total_tests",
            "best_by_max_ttc",
            "best_by_mean_feedback_time",
            "bet_overall_improvement_over_baseline",
            "num_test_workers",
        ):
            continue
        if isinstance(v, dict) and all(m in v for m in ("total_tests_run", "max_time_to_culprit_hr", "mean_feedback_time_hr")):
            eligible.append((k, v))

    if eligible:
        final_results["best_by_total_tests"] = min(eligible, key=lambda kv: kv[1]["total_tests_run"])[0]
        final_results["best_by_max_ttc"] = min(eligible, key=lambda kv: kv[1]["max_time_to_culprit_hr"])[0]
        final_results["best_by_mean_feedback_time"] = min(eligible, key=lambda kv: kv[1]["mean_feedback_time_hr"])[0]
        # Best overall vs baseline on FINAL window (centralized logic)
        final_results["bet_overall_improvement_over_baseline"] = choose_best_overall_from_items(eligible)
    else:
        final_results["best_by_total_tests"] = "-"
        final_results["best_by_max_ttc"] = "-"
        final_results["best_by_mean_feedback_time"] = "-"
        final_results["bet_overall_improvement_over_baseline"] = "NA"

    # Save FINAL
    os.makedirs(os.path.dirname(OUTPUT_PATH_FINAL), exist_ok=True)
    with open(OUTPUT_PATH_FINAL, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    print("✅ Saved FINAL replay to", OUTPUT_PATH_FINAL)
    return final_results


def main():
    args = get_args()

    global INPUT_JSON_EVAL, INPUT_JSON_FINAL, OUTPUT_PATH_EVAL, OUTPUT_PATH_FINAL
    INPUT_JSON_EVAL = args.input_json_eval
    INPUT_JSON_FINAL = args.input_json_final
    OUTPUT_PATH_EVAL = args.output_eval
    OUTPUT_PATH_FINAL = args.output_final

    if args.final_only:
        # Reuse existing eval results
        if not os.path.exists(OUTPUT_PATH_EVAL):
            raise FileNotFoundError(
                f"--final-only was set but eval results file not found at: {OUTPUT_PATH_EVAL}"
            )
        with open(OUTPUT_PATH_EVAL, "r", encoding="utf-8") as f:
            reused_eval_output = json.load(f)

        # Shape expected by run_final_test_unified
        eval_payload = {"eval_output": reused_eval_output}
        print(f"🔁 Reusing eval results from {OUTPUT_PATH_EVAL}; running FINAL only...")
        run_final_test_unified(eval_payload, INPUT_JSON_FINAL, OUTPUT_PATH_FINAL)
        return

    # Normal flow: run eval (sweep or mopt), save eval output, then FINAL replay
    if args.mode == "mopt":
        eval_payload = run_evaluation_mopt(INPUT_JSON_EVAL, n_trials=args.mopt_trials)
    else:
        eval_payload = run_evaluation_sweep(INPUT_JSON_EVAL)

    if eval_payload is None:
        raise RuntimeError("Evaluation was unsuccessful. No eval payload present")

    os.makedirs(os.path.dirname(OUTPUT_PATH_EVAL), exist_ok=True)
    with open(OUTPUT_PATH_EVAL, "w", encoding="utf-8") as f:
        json.dump(eval_payload["eval_output"], f, indent=2)
    print("✅ Saved EVAL results to", OUTPUT_PATH_EVAL)

    # Unified FINAL replay for both modes
    run_final_test_unified(eval_payload, INPUT_JSON_FINAL, OUTPUT_PATH_FINAL)


if __name__ == "__main__":
    main()
