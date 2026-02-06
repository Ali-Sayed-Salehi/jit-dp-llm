#!/usr/bin/env python3

"""
Batch-testing simulation driver.

This script evaluates combinations of:
  - Batching strategies (when to trigger a batch test), and
  - Bisection strategies (how to locate culprit commits within a failing batch),
under a shared time/cost model of limited test capacity.

Key references:
  - `analysis/batch_testing/README.md` for a detailed explanation of each
    strategy and the shared execution model.
  - `analysis/batch_testing/batch_strats.py` for batching implementations.
  - `analysis/batch_testing/bisection_strats.py` for bisection implementations,
    metadata loading, and the central test executor.

The Optuna mode (`run_evaluation_mopt`) tunes batching parameters. For TKRB it
also tunes `bisection_strats.TKRB_TOP_K`.
"""

import os
import json
import random
import math
from datetime import datetime, timedelta, timezone
import argparse
import csv
import logging

import bisection_strats as bisection_mod

from batch_strats import (
    simulate_twb_with_bisect,
    simulate_twb_s_with_bisect,
    simulate_fsb_with_bisect,
    simulate_fsb_s_with_bisect,
    simulate_rasb_with_bisect,
    simulate_rasb_s_with_bisect,
    simulate_rasb_la_with_bisect,
    simulate_rasb_la_s_with_bisect,
    simulate_rapb_with_bisect,
    simulate_rapb_s_with_bisect,
    simulate_rapb_la_with_bisect,
    simulate_rapb_la_s_with_bisect,
    simulate_ratb_with_bisect,
    simulate_ratb_s_with_bisect,
    simulate_twsb_with_bisect,
)

from bisection_strats import (
    time_ordered_bisect,
    exhaustive_parallel,
    risk_weighted_adaptive_bisect,
    risk_weighted_adaptive_bisect_log_survival,
    topk_risk_first_bisect,
    sequential_walk_backward_bisect,
    sequential_walk_forward_bisect,
    TestExecutor,
    run_test_suite,
    configure_bisection_defaults,
    validate_failing_signatures_coverage,
    configure_full_suite_signatures_union,
)

logger = logging.getLogger(__name__)

# ====== CONFIG ======
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

ALL_COMMITS_PATH = os.path.join(REPO_ROOT, "datasets", "mozilla_perf", "all_commits.jsonl")
SIG_GROUP_JOB_DURATIONS_CSV = os.path.join(
    REPO_ROOT, "datasets", "mozilla_perf", "sig_group_job_durations.csv"
)

BATCH_HOURS = 4
FSB_SIZE = 20
RASB_THRESHOLD = 0.5
RAPB_THRESHOLD = 0.35
RAPB_AGING_PER_HOUR = 0.05
LINEAR_RISK_BUDGET = 1.0
RATB_THRESHOLD = 0.5
RATB_TIME_WINDOW_HOURS = 4

DEFAULT_CUTOFF = datetime.fromisoformat("2024-10-10T00:00:00+00:00")
RANDOM_SEED = 42

DEFAULT_TEST_DURATION_MIN = 20.0
# Constant build-time overhead applied once per suite run (batch tests + bisection steps).
BUILD_TIME_HOURS = 1.5
BUILD_TIME_MINUTES = BUILD_TIME_HOURS * 60.0

# Central test machine capacity, expressed as per-platform worker pools.
# Each signature-group job is routed to a pool based on `machine_platform`
# metadata (see datasets/mozilla_perf/all_signatures.jsonl).
WORKER_POOLS = dict(bisection_mod.DEFAULT_WORKER_POOLS)

# Number of perf signature-groups to run for each "full suite" batch test step.
# If this is -1 or larger than the available groups, all groups are used.
FULL_SUITE_SIGNATURES_PER_RUN = 200

# Global toggle for lightweight, debugging-oriented runs.
DRY_RUN = False

BATCHING_STRATEGIES = [
    ("TWB",  simulate_twb_with_bisect,      BATCH_HOURS),
    ("TWB-s",  simulate_twb_s_with_bisect,      BATCH_HOURS),
    ("FSB",  simulate_fsb_with_bisect,      FSB_SIZE),
    ("FSB-s",  simulate_fsb_s_with_bisect,      FSB_SIZE),
    ("RASB", simulate_rasb_with_bisect,   RASB_THRESHOLD),
    ("RASB-s", simulate_rasb_s_with_bisect,   RASB_THRESHOLD),
    ("RASB-la", simulate_rasb_la_with_bisect,   LINEAR_RISK_BUDGET),
    ("RASB-la-s", simulate_rasb_la_s_with_bisect,   LINEAR_RISK_BUDGET),
    ("RAPB", simulate_rapb_with_bisect, (RAPB_THRESHOLD, RAPB_AGING_PER_HOUR)),
    ("RAPB-s", simulate_rapb_s_with_bisect, (RAPB_THRESHOLD, RAPB_AGING_PER_HOUR)),
    ("RAPB-la", simulate_rapb_la_with_bisect, (LINEAR_RISK_BUDGET, RAPB_AGING_PER_HOUR)),
    ("RAPB-la-s", simulate_rapb_la_s_with_bisect, (LINEAR_RISK_BUDGET, RAPB_AGING_PER_HOUR)),
    ("RATB", simulate_ratb_with_bisect,     (RATB_THRESHOLD, RATB_TIME_WINDOW_HOURS)),
    ("RATB-s", simulate_ratb_s_with_bisect,     (RATB_THRESHOLD, RATB_TIME_WINDOW_HOURS)),
]

BISECTION_STRATEGIES = [
    ("TOB",  time_ordered_bisect),
    ("PAR",  exhaustive_parallel),
    ("RWAB", risk_weighted_adaptive_bisect),
    ("RWAB-LS", risk_weighted_adaptive_bisect_log_survival),
    ("TKRB", topk_risk_first_bisect),
    ("SWB",  sequential_walk_backward_bisect),
    ("SWF",  sequential_walk_forward_bisect),
]

# Seed will be finalized in main(), but we also provide a
# deterministic default at import time.
random.seed(RANDOM_SEED)

# =================================

def _worker_pools_for_output(pools: dict) -> dict:
    """
    Produce a stable, human-friendly worker pool mapping for JSON output.

    - In single-pool mode, emits {"default": N}
    - In multi-pool mode, emits android/windows/linux/mac first (when present),
      then any extra pools in sorted order.
    """
    if not isinstance(pools, dict):
        return {"default": int(pools)}

    if set(pools.keys()) == {"default"}:
        return {"default": int(pools["default"])}

    preferred = ["android", "windows", "linux", "mac"]
    out = {}
    for k in preferred:
        if k in pools:
            out[k] = int(pools[k])
    for k in sorted(pools.keys()):
        if k in out:
            continue
        out[k] = int(pools[k])
    return out


def _load_batch_signature_durations(path):
    """
    Load ET "full suite" jobs from sig_group_job_durations.csv.

    Returns:
      list[(signature_group_id, duration_minutes)]
    """
    suite = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sig_group = row.get("signature_group_id")
                dur = row.get("duration_minutes")
                if not sig_group or not dur:
                    continue
                try:
                    sig_group_id = int(sig_group)
                    duration = float(dur)
                except ValueError:
                    continue
                suite.append((sig_group_id, duration))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"ET full-suite durations CSV not found at: {path}"
        ) from exc

    if not suite:
        raise RuntimeError(
            f"No valid (signature_group_id, duration_minutes) entries found in CSV at: {path}"
        )

    return suite


# Used by run_exhaustive_testing (loaded lazily so `--help` works without datasets present).
BATCH_SIGNATURE_DURATIONS_ET = None


def get_args():
    """
    Parse CLI arguments for evaluation/final runs.

    This script is designed to run "evaluation" (Optuna tuning) on one set of
    predictions and then optionally replay the chosen configurations on a
    "final test" prediction set.
    """
    parser = argparse.ArgumentParser(description="Batch-testing simulation (Optuna-only mopt).")

    parser.add_argument(
        "--mopt-trials",
        type=int,
        default=200,
        help="Number of Optuna trials per (batch,bisect) combo.",
    )
    parser.add_argument(
        "--input-json-eval",
        default=os.path.join(
            REPO_ROOT,
            "analysis",
            "batch_testing",
            "final_test_results_perf_mbert_eval.json",
        ),
        help="Path to EVAL predictions json",
    )
    parser.add_argument(
        "--input-json-final",
        default=os.path.join(
            REPO_ROOT,
            "analysis",
            "batch_testing",
            "final_test_results_perf_mbert_final_test.json",
        ),
        help="Path to FINAL predictions json",
    )
    parser.add_argument(
        "--output-eval",
        default=os.path.join(
            REPO_ROOT,
            "analysis",
            "batch_testing",
            "batch_sim_results_eval.json",
        ),
        help="Where to write EVAL sim results (and where --final-only will read from)",
    )
    parser.add_argument(
        "--output-final",
        default=os.path.join(
            REPO_ROOT,
            "analysis",
            "batch_testing",
            "batch_sim_results_final_test.json",
        ),
        help="Where to write FINAL sim results",
    )
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Skip running eval; load eval results from --output-eval and run only the FINAL replay.",
    )
    parser.add_argument(
        "--num-test-workers",
        type=int,
        default=None,
        help=(
            "(Deprecated) If set, use a single shared worker pool with this many "
            "workers (ignores per-platform pools). If omitted, uses per-platform "
            "worker pools."
        ),
    )
    parser.add_argument(
        "--build-time-minutes",
        type=float,
        default=BUILD_TIME_MINUTES,
        help=(
            "Build-time overhead (minutes) added once per suite run "
            f"(default: {BUILD_TIME_MINUTES:.1f})."
        ),
    )
    parser.add_argument(
        "--workers-android",
        type=int,
        default=bisection_mod.ANDROID_WORKERS,
        help="Android worker pool size (signature-groups routed by machine_platform).",
    )
    parser.add_argument(
        "--workers-windows",
        type=int,
        default=bisection_mod.WINDOWS_WORKERS,
        help="Windows worker pool size (signature-groups routed by machine_platform).",
    )
    parser.add_argument(
        "--workers-linux",
        type=int,
        default=bisection_mod.LINUX_WORKERS,
        help="Linux worker pool size (signature-groups routed by machine_platform).",
    )
    parser.add_argument(
        "--workers-mac",
        type=int,
        default=bisection_mod.MAC_WORKERS,
        help=(
            "Mac worker pool size (signature-groups routed by machine_platform; "
            "includes iOS jobs)."
        ),
    )
    parser.add_argument(
        "--unknown-platform-pool",
        type=str,
        default=bisection_mod.DEFAULT_UNKNOWN_PLATFORM_POOL,
        help=(
            "Worker pool key to use when a signature-group/job cannot be mapped to a "
            "platform (missing signature-group id, missing signature metadata, or "
            "unrecognized machine_platform)."
        ),
    )
    parser.add_argument(
        "--full-suite-sigs-per-run",
        type=int,
        default=FULL_SUITE_SIGNATURES_PER_RUN,
        help=(
            "Number of perf signature-groups to run per 'full suite' batch test step "
            "(-1 means use all groups; positive values cap via random subset)."
        ),
    )
    parser.add_argument(
        "--dont-use-all-tests-per-batch",
        action="store_true",
        help=(
            "If set, each initial batch test run executes only a random subset "
            "of perf signature-groups, with size equal to --full-suite-sigs-per-run. "
            "By default (flag not set), all signature-groups that appear at least "
            "once within the cutoff window are executed."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Lightweight mode: run the baseline plus at most two random "
            "(batching, bisection) combinations when no explicit strategy "
            "filters are provided."
        ),
    )
    parser.add_argument(
        "--batching",
        default="all",
        help=(
            "Comma-separated batching strategy names to simulate (default: all). "
            "Examples: 'TWB,RATB-s' or 'all' or 'none'."
        ),
    )
    parser.add_argument(
        "--bisection",
        default="all",
        help=(
            "Comma-separated bisection strategy names to simulate (default: all). "
            "Examples: 'TOB,PAR' or 'all' or 'none'."
        ),
    )
    parser.add_argument(
        "--skip-exhaustive-testing",
        action="store_true",
        help="If set, skip the Exhaustive Testing (ET) baseline (always run TWSB+PAR baseline).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for the simulation.",
    )
    return parser.parse_args()


def _parse_csv_strategy_list(raw: str):
    if raw is None:
        return []
    raw = str(raw).strip()
    if not raw:
        return []
    parts = []
    for token in raw.split(","):
        tok = token.strip()
        if tok:
            parts.append(tok)
    return parts


def _resolve_strategy_filter(raw: str, all_names, kind: str):
    """
    Turn a user-provided CSV strategy list into a set of selected strategy names.

    raw:
      - "all" (default) => all_names
      - "none" => empty set
      - otherwise => exact match against all_names
    """
    tokens = _parse_csv_strategy_list(raw)
    if not tokens:
        return set(all_names)

    lowered = [t.lower() for t in tokens]
    if len(lowered) == 1 and lowered[0] == "all":
        return set(all_names)
    if len(lowered) == 1 and lowered[0] == "none":
        return set()

    unknown = sorted(set(tokens) - set(all_names))
    if unknown:
        known = ", ".join(all_names)
        bad = ", ".join(unknown)
        raise ValueError(
            f"Unknown {kind} strategy name(s): {bad}. Known {kind} strategies: {known}"
        )
    return set(tokens)


def load_predictions_raw(path):
    """
    Load model predictions and compute per-commit p_pos.

    Returns:
      dict[commit_id] -> {"true_label": int, "p_pos": float}
    """
    logger.debug("Loading raw predictions from %s", path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prediction file does not exist: {path}")
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

        preds[cid] = {"true_label": true_label, "p_pos": p_pos}
    return preds


def build_commits_from_all_with_raw_preds(
    all_commits_path, preds_raw, lower_cutoff, upper_cutoff=None
):
    """
    Build the simulation commit stream from `all_commits.jsonl` using `preds_raw`.

    This avoids per-trial re-reading of the prediction JSON during Optuna runs.

    Returns:
      (commits_sorted, predicted_indices)
    where predicted_indices are indices in commits_sorted for commits that
    have a non-zero predicted risk.
    """
    logger.debug(
        "Building commits from %s with lower_cutoff=%s upper_cutoff=%s (preds_raw size=%d)",
        all_commits_path,
        lower_cutoff,
        upper_cutoff,
        len(preds_raw),
    )
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

            info = preds_raw.get(node)
            if info is not None:
                true_label = bool(int(info["true_label"]))
                p_pos = float(info["p_pos"])
            else:
                true_label = False
                p_pos = 0.0

            commits.append(
                {
                    "commit_id": node,
                    "true_label": true_label,
                    "risk": p_pos,
                    "ts": ts,
                }
            )

    commits.sort(key=lambda x: x["ts"])
    predicted_indices = [i for i, c in enumerate(commits) if c.get("risk", 0.0) > 0.0]
    logger.debug(
        "Finished building commit list: %d commits within window (%d predicted)",
        len(commits),
        len(predicted_indices),
    )
    return commits, predicted_indices


def parse_hg_date(date_field):
    """
    Parse the `date` field from the mozilla perf `all_commits.jsonl` dataset.

    Supports:
      - `[unix_ts, offset_seconds]` pairs (Mercurial-style), and
      - ISO-8601 datetime strings.
    """
    if isinstance(date_field, list) and len(date_field) == 2:
        unix_ts, offset_seconds_west = date_field
        # Mercurial stores tz offset as seconds west of UTC (positive values mean UTC-<hours>).
        # Python's timezone offset is seconds east of UTC, so we negate.
        tz = timezone(timedelta(seconds=-int(offset_seconds_west)))
        return datetime.fromtimestamp(float(unix_ts), tz=tz)
    if isinstance(date_field, str):
        return datetime.fromisoformat(date_field)
    raise TypeError(
        f"Unsupported hg date_field type {type(date_field)!r}; expected list[ts, offset] or ISO string."
    )


def get_cutoff_from_input(all_commits_path, pred_map):
    """Find the oldest and newest commit (by ts) that is present in the given INPUT_JSON."""
    oldest = None
    newest = None
    if not pred_map:
        raise ValueError("Prediction map is empty in get_cutoff_from_input")
    logger.debug(
        "Scanning %s to determine cutoff window for %d predicted commits",
        all_commits_path,
        len(pred_map),
    )
    with open(all_commits_path, "r", encoding="utf-8") as f:
        scanned = 0
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
            scanned += 1
            if scanned % 50000 == 0:
                logger.debug("Processed %d lines while computing cutoff window...", scanned)
    logger.debug(
        "Computed cutoff window: oldest=%s newest=%s (scanned %d lines)",
        oldest,
        newest,
        scanned,
    )
    return oldest, newest


def run_exhaustive_testing(commits):
    """
    Exhaustive Testing baseline (ET).

    Models running a (possibly capped) full suite for every commit in time
    order. This provides a reference point for total cost and feedback times.
    """
    logger.info("Starting exhaustive testing over %d commits", len(commits))
    if not commits:
        logger.info("No commits provided to run_exhaustive_testing; returning zeros.")
        return {
            "total_tests_run": 0,
            "mean_feedback_time_hr": 0.0,
            "mean_time_to_culprit_hr": 0.0,
            "max_time_to_culprit_hr": 0.0,
            "p90_time_to_culprit_hr": 0.0,
            "p95_time_to_culprit_hr": 0.0,
            "p99_time_to_culprit_hr": 0.0,
            "total_cpu_time_hr": 0.0,
            "num_regressors_total": 0,
            "num_regressors_found": 0,
            "found_all_regressors": True,
        }

    global BATCH_SIGNATURE_DURATIONS_ET
    if BATCH_SIGNATURE_DURATIONS_ET is None:
        BATCH_SIGNATURE_DURATIONS_ET = _load_batch_signature_durations(SIG_GROUP_JOB_DURATIONS_CSV)

    total_tests_run = 0
    culprit_times = []
    feedback_times = {}
    num_regressors_total = sum(1 for c in commits if c.get("true_label"))

    executor = TestExecutor(WORKER_POOLS)
    logger.debug(
        "Created TestExecutor with worker pools for exhaustive testing: %s",
        WORKER_POOLS,
    )

    for idx, c in enumerate(commits, start=1):
        submit_time = c["ts"]

        # Full suite, but capped to a fixed random subset of signature-groups
        # if a non-negative cap is configured. A negative value (e.g., -1)
        # means "use all signature-groups".
        limit = FULL_SUITE_SIGNATURES_PER_RUN
        if limit is None:
            raise RuntimeError(
                "FULL_SUITE_SIGNATURES_PER_RUN must not be None in "
                "run_exhaustive_testing; use -1 to indicate 'all signature-groups'."
            )
        if isinstance(limit, int) and limit >= 0 and len(BATCH_SIGNATURE_DURATIONS_ET) > limit:
            durations = random.sample(BATCH_SIGNATURE_DURATIONS_ET, limit)
        else:
            durations = BATCH_SIGNATURE_DURATIONS_ET
        finish_time = run_test_suite(executor, submit_time, durations)

        total_tests_run += len(durations)

        fb_min = (finish_time - c["ts"]).total_seconds() / 60.0
        feedback_times[c["commit_id"]] = fb_min
        if c["true_label"]:
            culprit_times.append(fb_min)

    if feedback_times:
        mean_fb_min = sum(feedback_times.values()) / len(feedback_times)
    else:
        mean_fb_min = 0.0

    if culprit_times:
        culprit_times_sorted = sorted(float(x) for x in culprit_times)
        mean_ttc_min = sum(culprit_times_sorted) / len(culprit_times_sorted)
        max_ttc_min = culprit_times_sorted[-1]

        def _percentile_linear(sorted_values, p: float) -> float:
            if not sorted_values:
                return 0.0
            p = float(p)
            if p <= 0.0:
                return float(sorted_values[0])
            if p >= 100.0:
                return float(sorted_values[-1])
            n = len(sorted_values)
            pos = (n - 1) * (p / 100.0)
            lo = int(math.floor(pos))
            hi = int(math.ceil(pos))
            if lo == hi:
                return float(sorted_values[lo])
            frac = pos - lo
            return float(
                sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac
            )

        p90_ttc_min = _percentile_linear(culprit_times_sorted, 90)
        p95_ttc_min = _percentile_linear(culprit_times_sorted, 95)
        p99_ttc_min = _percentile_linear(culprit_times_sorted, 99)
    else:
        mean_ttc_min = 0.0
        max_ttc_min = 0.0
        p90_ttc_min = 0.0
        p95_ttc_min = 0.0
        p99_ttc_min = 0.0

    total_cpu_time_min = getattr(executor, "total_cpu_minutes", 0.0)

    res = {
        "total_tests_run": total_tests_run,
        "mean_feedback_time_min": round(mean_fb_min, 2),
        "mean_time_to_culprit_min": round(mean_ttc_min, 2),
        "max_time_to_culprit_min": round(max_ttc_min, 2),
        "p90_time_to_culprit_min": round(p90_ttc_min, 2),
        "p95_time_to_culprit_min": round(p95_ttc_min, 2),
        "p99_time_to_culprit_min": round(p99_ttc_min, 2),
        "total_cpu_time_min": round(float(total_cpu_time_min), 2),
        "num_regressors_total": int(num_regressors_total),
        "num_regressors_found": int(num_regressors_total),
        "found_all_regressors": True,
    }

    # Reuse common conversion helper for consistency
    res = convert_result_minutes_to_hours(res)
    return res


def convert_result_minutes_to_hours(res):
    """
    Convert common result fields from minutes to hours, in-place.

    This keeps JSON outputs easier to interpret when aggregating across long
    time windows.
    """
    if "mean_feedback_time_min" in res:
        res["mean_feedback_time_hr"] = round(res["mean_feedback_time_min"] / 60.0, 2)
        del res["mean_feedback_time_min"]
    if "mean_time_to_culprit_min" in res:
        res["mean_time_to_culprit_hr"] = round(
            res["mean_time_to_culprit_min"] / 60.0, 2
        )
        del res["mean_time_to_culprit_min"]
    if "max_time_to_culprit_min" in res:
        res["max_time_to_culprit_hr"] = round(
            res["max_time_to_culprit_min"] / 60.0, 2
        )
        del res["max_time_to_culprit_min"]
    if "p90_time_to_culprit_min" in res:
        res["p90_time_to_culprit_hr"] = round(
            res["p90_time_to_culprit_min"] / 60.0, 2
        )
        del res["p90_time_to_culprit_min"]
    if "p95_time_to_culprit_min" in res:
        res["p95_time_to_culprit_hr"] = round(
            res["p95_time_to_culprit_min"] / 60.0, 2
        )
        del res["p95_time_to_culprit_min"]
    if "p99_time_to_culprit_min" in res:
        res["p99_time_to_culprit_hr"] = round(
            res["p99_time_to_culprit_min"] / 60.0, 2
        )
        del res["p99_time_to_culprit_min"]
    if "total_cpu_time_min" in res:
        res["total_cpu_time_hr"] = round(
            res["total_cpu_time_min"] / 60.0, 2
        )
        del res["total_cpu_time_min"]
    return res


def lookup_batching(name):
    """
    Map a batching strategy name to its `(fn, default_param)` pair.

    Used by the "final replay" stage to re-run the selected configurations.
    """
    # Special-case TWSB so it can appear in eval/final combos without being
    # part of the Optuna-tuned BATCHING_STRATEGIES grid.
    if name == "TWSB":
        return simulate_twsb_with_bisect, None
    for n, fn, default_param in BATCHING_STRATEGIES:
        if n == name:
            return fn, default_param
    return None, None


def lookup_bisection(name):
    """
    Map a bisection strategy name to its function.
    """
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
    "cpu_time_saved_vs_baseline_pct",
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
    # CPU time is tracked separately but not included in this aggregate score.
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

    best_name, best_entry = max(
        candidate_pool, key=lambda kv: _overall_improvement_score(kv[1])
    )
    if _overall_improvement_score(best_entry) <= 0.0:
        return "NA"
    return best_name


# ------------------- MOPT (Optuna, continuous search) -------------------
def run_evaluation_mopt(
    INPUT_JSON_EVAL,
    n_trials,
    selected_batching=None,
    selected_bisection=None,
    run_exhaustive_testing_et=True,
    dry_run=False,
    preds_raw=None,
    base_commits_for_context=None,
    predicted_indices=None,
    lower_cutoff=None,
    upper_cutoff=None,
):
    """
    Run the Optuna-based evaluation ("mopt") stage on the EVAL prediction set.

    For each (batching, bisection) pair in the configured grids, this function:
      - Tunes the batching strategy parameter(s) for that batching policy.
      - For the `TKRB` bisection strategy, also tunes `TKRB_TOP_K` by mutating
        `bisection_strats.TKRB_TOP_K` prior to each trial run.

    Outputs a unified JSON payload containing:
      - ET (exhaustive) baseline, TWSB+PAR baseline,
      - one selected Pareto-optimal configuration per combo, and
      - summary “best-by-metric” fields for quick comparison.
    """
    try:
        import optuna
    except ImportError as e:
        raise RuntimeError(
            "Optuna is required for this script. Install with `pip install optuna`."
        ) from e

    logger.info(
        "Starting Optuna evaluation (mopt) with INPUT_JSON_EVAL=%s, n_trials=%d",
        INPUT_JSON_EVAL,
        n_trials,
    )

    selected_batching_set = (
        set(selected_batching) if selected_batching is not None else None
    )
    selected_bisection_set = (
        set(selected_bisection) if selected_bisection is not None else None
    )
    baseline_selected = (
        (selected_batching_set is None or "TWSB" in selected_batching_set)
        and (selected_bisection_set is None or "PAR" in selected_bisection_set)
    )

    # Allow main() to pass precomputed/cached inputs to avoid re-reading
    # INPUT_JSON_EVAL and scanning all_commits.jsonl again.
    if preds_raw is None:
        preds_raw = load_predictions_raw(INPUT_JSON_EVAL)
    if (
        base_commits_for_context is None
        or predicted_indices is None
        or lower_cutoff is None
    ):
        dynamic_oldest, dynamic_newest = get_cutoff_from_input(
            ALL_COMMITS_PATH, preds_raw
        )
        lower_cutoff = dynamic_oldest or DEFAULT_CUTOFF
        upper_cutoff = dynamic_newest
        base_commits_for_context, predicted_indices = build_commits_from_all_with_raw_preds(
            ALL_COMMITS_PATH,
            preds_raw,
            lower_cutoff,
            upper_cutoff,
        )
    if run_exhaustive_testing_et and base_commits_for_context:
        et_results = run_exhaustive_testing(base_commits_for_context)
    else:
        et_results = {}
    baseline = {}
    if base_commits_for_context and baseline_selected:
        logger.info(
            "Running baseline TWSB + PAR simulation for context on %d commits",
            len(base_commits_for_context),
        )
        baseline = simulate_twsb_with_bisect(
            base_commits_for_context,
            exhaustive_parallel,
            None,
            WORKER_POOLS,
        )
        baseline = convert_result_minutes_to_hours(baseline)
    baseline_max_ttc = baseline.get("max_time_to_culprit_hr", None)
    baseline_cpu = baseline.get("total_cpu_time_hr", None)

    baseline_fb = baseline.get("mean_feedback_time_hr", None)
    baseline_mean_ttc = baseline.get("mean_time_to_culprit_hr", None)
    baseline_tests = baseline.get("total_tests_run", None)

    def time_saved_pct(base, val):
        return (
            round((base - val) / base * 100.0, 2)
            if base and base > 0
            else 0.0
        )

    def pick_best(pareto, baseline_max_ttc_local):
        if not pareto:
            return None
        feas = [
            r
            for r in pareto
            if (
                baseline_max_ttc_local is None
                or r["max_time_to_culprit_hr"] <= baseline_max_ttc_local
            )
        ]
        if feas:
            return min(
                feas, key=lambda r: (r["total_tests_run"], r["max_time_to_culprit_hr"])
            )
        return min(
            pareto, key=lambda r: (r["max_time_to_culprit_hr"], r["total_tests_run"])
        )

    # unified output (same shape as before)
    out_eval = {
        "Exhaustive Testing (ET)": et_results,
        "worker_pools": _worker_pools_for_output(WORKER_POOLS),
        "num_test_workers": sum(int(v) for v in WORKER_POOLS.values()),
    }
    if baseline_selected:
        out_eval["Baseline (TWSB + PAR)"] = baseline

    # Per combo study with continuous/int ranges
    logger.info(
        "Beginning Optuna studies over %d batching strategies x %d bisection strategies",
        len(BATCHING_STRATEGIES),
        len(BISECTION_STRATEGIES),
    )

    if selected_batching is not None:
        selected_batching = set(selected_batching)
    if selected_bisection is not None:
        selected_bisection = set(selected_bisection)

    # In dry-run mode, restrict ourselves to a tiny subset of combinations
    # to keep turnaround fast, but only when the user did not provide explicit
    # batching/bisection filters.
    all_combo_names = [
        (b_name, bis_name)
        for b_name, _, _ in BATCHING_STRATEGIES
        for bis_name, _ in BISECTION_STRATEGIES
        if (selected_batching is None or b_name in selected_batching)
        and (selected_bisection is None or bis_name in selected_bisection)
    ]
    if dry_run and (selected_batching is None) and (selected_bisection is None):
        if len(all_combo_names) <= 2:
            selected_combos = set(all_combo_names)
        else:
            selected_combos = set(random.sample(all_combo_names, 2))
        logger.info(
            "Dry run mode: limiting Optuna studies to two random combos: %s",
            ", ".join(f"{b} + {bis}" for b, bis in selected_combos),
        )
    else:
        selected_combos = None

    for b_name, b_fn, _ in BATCHING_STRATEGIES:
        if selected_batching is not None and b_name not in selected_batching:
            continue
        for bis_name, bis_fn in BISECTION_STRATEGIES:
            if selected_bisection is not None and bis_name not in selected_bisection:
                continue
            if dry_run and selected_combos is not None and (b_name, bis_name) not in selected_combos:
                continue
            combo_key = f"{b_name} + {bis_name}"
            logger.info(
                "Creating Optuna study for combo %s with %d trials",
                combo_key,
                n_trials,
            )

            def objective(trial):
                if trial.number % 10 == 0:
                    logger.info(
                        "Optuna trial %d/%d for combo %s",
                        trial.number,
                        n_trials,
                        combo_key,
                    )
                # Bisection-specific tunables (only for TKRB).
                if bis_name == "TKRB":
                    bisection_mod.TKRB_TOP_K = trial.suggest_int("TKRB_TOP_K", 1, 10)

                def normalize_batch_name(name: str) -> str:
                    return name[:-2] if name.endswith("-s") else name

                # Strategy-specific continuous/int params.
                #
                # Note: "-s" batching variants share the same parameter space
                # as their non-"-s" counterparts; they differ only in how the
                # initial batch test suite is constructed.
                normalized_batch_name = normalize_batch_name(b_name)
                if normalized_batch_name == "TWB":
                    param = trial.suggest_float("BATCH_HOURS", 0.25, 24.0)
                elif normalized_batch_name == "FSB":
                    param = trial.suggest_int("FSB_SIZE", 4, 200)
                elif normalized_batch_name == "RASB":
                    param = trial.suggest_float("RASB_THRESHOLD", 0.10, 0.95)
                elif normalized_batch_name == "RASB-la":
                    param = trial.suggest_float("RASB_LA_BUDGET", 0.25, 6.0)
                elif normalized_batch_name == "RAPB":
                    T = trial.suggest_float("RAPB_THRESHOLD", 0.30, 0.80)
                    a = trial.suggest_float("RAPB_AGING_PER_HOUR", 0.005, 0.200)
                    param = (T, a)
                elif normalized_batch_name == "RAPB-la":
                    T = trial.suggest_float("RAPB_LA_BUDGET", 0.25, 6.0)
                    a = trial.suggest_float(
                        "RAPB_LA_AGING_PER_HOUR", 0.005, 0.200
                    )
                    param = (T, a)
                elif normalized_batch_name == "RATB":
                    thr = trial.suggest_float("RATB_THRESHOLD", 0.10, 0.95)
                    tw = trial.suggest_float("RATB_TIME_WINDOW_HOURS", 0.25, 24.0)
                    param = (thr, tw)
                else:
                    logger.warning(
                        "Unknown batching strategy name %s (normalized=%s); returning inf",
                        b_name,
                        normalized_batch_name,
                    )
                    return (float("inf"), float("inf"))


                if not base_commits_for_context:
                    logger.warning(
                        "No commits returned for combo %s at trial %d",
                        combo_key,
                        trial.number,
                    )
                    return (float("inf"), float("inf"))

                res = b_fn(base_commits_for_context, bis_fn, param, WORKER_POOLS)
                res = convert_result_minutes_to_hours(res)
                return (
                    res.get("total_tests_run", float("inf")),
                    res.get("max_time_to_culprit_hr", float("inf")),
                )

            study = optuna.create_study(directions=["minimize", "minimize"])
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            logger.info(
                "Completed Optuna study for combo %s; best trial count=%d",
                combo_key,
                len(study.best_trials),
            )

            # Build Pareto & choose a single best for unified eval_output
            pareto = []
            for t in study.best_trials:
                params = t.params

                if bis_name == "TKRB":
                    bisection_mod.TKRB_TOP_K = int(params.get("TKRB_TOP_K", 1))

                # unpack for re-eval (same mapping as objective; "-s" variants use
                # the same underlying parameter(s)).
                normalized_batch_name = (
                    b_name[:-2] if b_name.endswith("-s") else b_name
                )
                if normalized_batch_name == "TWB":
                    param = params["BATCH_HOURS"]
                elif normalized_batch_name == "FSB":
                    param = int(params["FSB_SIZE"])
                elif normalized_batch_name == "RASB":
                    param = params["RASB_THRESHOLD"]
                elif normalized_batch_name == "RASB-la":
                    param = params["RASB_LA_BUDGET"]
                elif normalized_batch_name == "RAPB":
                    param = (
                        params["RAPB_THRESHOLD"],
                        params["RAPB_AGING_PER_HOUR"],
                    )
                elif normalized_batch_name == "RAPB-la":
                    param = (
                        params["RAPB_LA_BUDGET"],
                        params["RAPB_LA_AGING_PER_HOUR"],
                    )
                elif normalized_batch_name == "RATB":
                    param = (
                        params["RATB_THRESHOLD"],
                        params["RATB_TIME_WINDOW_HOURS"],
                    )
                else:
                    logger.warning(
                        "Unknown batching strategy name %s (normalized=%s); skipping pareto entry",
                        b_name,
                        normalized_batch_name,
                    )
                    continue

                if not base_commits_for_context:
                    continue

                res = b_fn(base_commits_for_context, bis_fn, param, WORKER_POOLS)
                res = convert_result_minutes_to_hours(res)

                if normalized_batch_name == "TWB":
                    best_param_dict = {"BATCH_HOURS": param}
                elif normalized_batch_name == "FSB":
                    best_param_dict = {"FSB_SIZE": param}
                elif normalized_batch_name == "RASB":
                    best_param_dict = {"RASB_THRESHOLD": param}
                elif normalized_batch_name == "RASB-la":
                    best_param_dict = {"RASB_LA_BUDGET": param}
                elif normalized_batch_name == "RAPB":
                    best_param_dict = {
                        "RAPB_THRESHOLD": param[0],
                        "RAPB_AGING_PER_HOUR": param[1],
                    }
                elif normalized_batch_name == "RAPB-la":
                    best_param_dict = {
                        "RAPB_LA_BUDGET": param[0],
                        "RAPB_LA_AGING_PER_HOUR": param[1],
                    }
                elif normalized_batch_name == "RATB":
                    best_param_dict = {
                        "RATB_THRESHOLD": param[0],
                        "RATB_TIME_WINDOW_HOURS": param[1],
                    }
                else:
                    continue
                if bis_name == "TKRB":
                    best_param_dict["TKRB_TOP_K"] = int(bisection_mod.TKRB_TOP_K)

                pareto.append(
                    {
                        "best_params": best_param_dict,
                        "total_tests_run": res.get("total_tests_run"),
                        "mean_feedback_time_hr": res.get("mean_feedback_time_hr"),
                        "mean_time_to_culprit_hr": res.get(
                            "mean_time_to_culprit_hr"
                        ),
                        "max_time_to_culprit_hr": res.get(
                            "max_time_to_culprit_hr"
                        ),
                        "p90_time_to_culprit_hr": res.get(
                            "p90_time_to_culprit_hr"
                        ),
                        "p95_time_to_culprit_hr": res.get(
                            "p95_time_to_culprit_hr"
                        ),
                        "p99_time_to_culprit_hr": res.get(
                            "p99_time_to_culprit_hr"
                        ),
                        "total_cpu_time_hr": res.get("total_cpu_time_hr"),
                        "num_regressors_total": res.get("num_regressors_total", 0),
                        "num_regressors_found": res.get("num_regressors_found", 0),
                        "found_all_regressors": bool(
                            res.get("found_all_regressors", False)
                        ),
                    }
                )

            selected = pick_best(pareto, baseline_max_ttc)
            if selected is None:
                logger.warning(
                    "No Pareto-optimal configuration selected for combo %s; skipping",
                    combo_key,
                )
                continue

            # compute deltas vs baseline for the selected
            result_entry = {
                "total_tests_run": selected["total_tests_run"],
                "mean_feedback_time_hr": selected["mean_feedback_time_hr"],
                "mean_time_to_culprit_hr": selected["mean_time_to_culprit_hr"],
                "max_time_to_culprit_hr": selected["max_time_to_culprit_hr"],
                "p90_time_to_culprit_hr": selected.get("p90_time_to_culprit_hr"),
                "p95_time_to_culprit_hr": selected.get("p95_time_to_culprit_hr"),
                "p99_time_to_culprit_hr": selected.get("p99_time_to_culprit_hr"),
                "total_cpu_time_hr": selected.get("total_cpu_time_hr"),
                "num_regressors_total": selected.get("num_regressors_total", 0),
                "num_regressors_found": selected.get("num_regressors_found", 0),
                "found_all_regressors": bool(
                    selected.get("found_all_regressors", False)
                ),
                "best_params": selected["best_params"],
                "tests_saved_vs_baseline_pct": time_saved_pct(
                    baseline_tests, selected["total_tests_run"]
                )
                if baseline_tests
                else 0.0,
                "mean_feedback_time_saved_vs_baseline_pct": time_saved_pct(
                    baseline_fb, selected["mean_feedback_time_hr"]
                )
                if baseline_fb is not None
                else 0.0,
                "mean_time_to_culprit_saved_vs_baseline_pct": time_saved_pct(
                    baseline_mean_ttc,
                    selected["mean_time_to_culprit_hr"],
                )
                if baseline_mean_ttc is not None
                else 0.0,
                "max_time_to_culprit_saved_vs_baseline_pct": time_saved_pct(
                    baseline_max_ttc,
                    selected["max_time_to_culprit_hr"],
                )
                if baseline_max_ttc is not None
                else 0.0,
                "_mopt_pareto_sample_size": len(pareto),
            }
            out_eval[combo_key] = result_entry

    # Also evaluate fixed-parameter TWSB with all bisection strategies on the
    # same base commit set used for the baseline. Skip this in dry-run mode
    # to keep the number of evaluated combinations small.
    if (
        base_commits_for_context
        and not dry_run
        and (selected_batching is None or "TWSB" in selected_batching)
    ):
        for bis_name, bis_fn in BISECTION_STRATEGIES:
            if selected_bisection is not None and bis_name not in selected_bisection:
                continue
            combo_key = f"TWSB + {bis_name}"
            logger.info(
                "Running fixed TWSB combo %s on %d commits",
                combo_key,
                len(base_commits_for_context),
            )
            res = simulate_twsb_with_bisect(
                base_commits_for_context,
                bis_fn,
                None,
                WORKER_POOLS,
            )
            res = convert_result_minutes_to_hours(res)

            result_entry = {
                "total_tests_run": res.get("total_tests_run"),
                "mean_feedback_time_hr": res.get("mean_feedback_time_hr"),
                "mean_time_to_culprit_hr": res.get("mean_time_to_culprit_hr"),
                "max_time_to_culprit_hr": res.get("max_time_to_culprit_hr"),
                "p90_time_to_culprit_hr": res.get("p90_time_to_culprit_hr"),
                "p95_time_to_culprit_hr": res.get("p95_time_to_culprit_hr"),
                "p99_time_to_culprit_hr": res.get("p99_time_to_culprit_hr"),
                "total_cpu_time_hr": res.get("total_cpu_time_hr"),
                "num_regressors_total": res.get("num_regressors_total", 0),
                "num_regressors_found": res.get("num_regressors_found", 0),
                "found_all_regressors": bool(
                    res.get("found_all_regressors", False)
                ),
                "best_params": {},  # no tunable params for TWSB
                "tests_saved_vs_baseline_pct": time_saved_pct(
                    baseline_tests,
                    res.get("total_tests_run", baseline_tests),
                )
                if baseline_tests is not None
                else 0.0,
                "mean_feedback_time_saved_vs_baseline_pct": time_saved_pct(
                    baseline_fb,
                    res.get("mean_feedback_time_hr", baseline_fb),
                )
                if baseline_fb is not None
                else 0.0,
                "mean_time_to_culprit_saved_vs_baseline_pct": time_saved_pct(
                    baseline_mean_ttc,
                    res.get("mean_time_to_culprit_hr", baseline_mean_ttc),
                )
                if baseline_mean_ttc is not None
                else 0.0,
                "max_time_to_culprit_saved_vs_baseline_pct": time_saved_pct(
                    baseline_max_ttc,
                    res.get("max_time_to_culprit_hr", baseline_max_ttc),
                )
                if baseline_max_ttc is not None
                else 0.0,
                "_mopt_pareto_sample_size": 1,
            }
            out_eval[combo_key] = result_entry

    # summary (unified)
    # Only consider actual (strategy + bisection) result entries; `out_eval`
    # also contains metadata keys like worker pool sizing.
    combo_items = []
    for k, v in out_eval.items():
        if k in (
            "Exhaustive Testing (ET)",
            "Baseline (TWSB + PAR)",
            "num_test_workers",
            "worker_pools",
        ):
            continue
        if not isinstance(v, dict):
            continue
        if not all(
            metric in v
            for metric in (
                "total_tests_run",
                "max_time_to_culprit_hr",
                "mean_feedback_time_hr",
            )
        ):
            continue
        combo_items.append((k, v))

    # Per-metric bests
    out_eval["best_by_total_tests"] = (
        min(combo_items, key=lambda kv: kv[1]["total_tests_run"])[0]
        if combo_items
        else "-"
    )
    out_eval["best_by_max_ttc"] = (
        min(combo_items, key=lambda kv: kv[1]["max_time_to_culprit_hr"])[0]
        if combo_items
        else "-"
    )
    out_eval["best_by_mean_feedback_time"] = (
        min(combo_items, key=lambda kv: kv[1]["mean_feedback_time_hr"])[0]
        if combo_items
        else "-"
    )

    # Best overall vs baseline (centralized logic)
    out_eval["bet_overall_improvement_over_baseline"] = choose_best_overall_from_items(
        combo_items
    )

    return {
        "eval_output": out_eval,
        "eval_lower_cutoff": lower_cutoff.isoformat(),
        "eval_upper_cutoff": upper_cutoff.isoformat() if upper_cutoff else None,
        "mode": "mopt",
    }


# ------------------- FINAL REPLAY (unified) -------------------
def run_final_test_unified(
    eval_payload,
    INPUT_JSON_FINAL,
    OUTPUT_PATH_FINAL,
    selected_batching=None,
    selected_bisection=None,
    run_exhaustive_testing_et=True,
    preds_raw_final=None,
    base_commits_final=None,
    predicted_indices_final=None,
    final_lower=None,
    final_upper=None,
):
    """
    Replay selected configurations on the FINAL prediction set.

    This takes the chosen configurations from the EVAL stage (`eval_payload`),
    then re-runs the same (batching, bisection) combos on `INPUT_JSON_FINAL`.

    For each combo it:
      - Reconstructs the tuned parameters (tuned batching params; for TKRB,
        `TKRB_TOP_K`).
      - Runs the simulation.
      - Computes deltas vs the baseline from the EVAL payload.
      - Writes a unified JSON result file to `OUTPUT_PATH_FINAL`.
    """
    logger.info(
        "Starting FINAL replay with INPUT_JSON_FINAL=%s, OUTPUT_PATH_FINAL=%s",
        INPUT_JSON_FINAL,
        OUTPUT_PATH_FINAL,
    )

    # Allow main() to pass precomputed/cached inputs to avoid re-reading
    # INPUT_JSON_FINAL and scanning all_commits.jsonl again.
    if preds_raw_final is None:
        preds_raw_final = load_predictions_raw(INPUT_JSON_FINAL)
    if (
        base_commits_final is None
        or predicted_indices_final is None
        or final_lower is None
    ):
        final_oldest, final_newest = get_cutoff_from_input(
            ALL_COMMITS_PATH, preds_raw_final
        )
        final_lower = final_oldest or DEFAULT_CUTOFF
        final_upper = final_newest

        # ET + Baseline on FINAL window
        base_commits_final, predicted_indices_final = build_commits_from_all_with_raw_preds(
            ALL_COMMITS_PATH,
            preds_raw_final,
            final_lower,
            final_upper,
        )
    if not base_commits_final:
        raise RuntimeError("No commits found in FINAL window; exiting final.")

    selected_batching_set = (
        set(selected_batching) if selected_batching is not None else None
    )
    selected_bisection_set = (
        set(selected_bisection) if selected_bisection is not None else None
    )
    baseline_selected = (
        (selected_batching_set is None or "TWSB" in selected_batching_set)
        and (selected_bisection_set is None or "PAR" in selected_bisection_set)
    )

    if run_exhaustive_testing_et:
        et_results_final = run_exhaustive_testing(base_commits_final)
    else:
        et_results_final = {}
    baseline_final = {}
    if baseline_selected:
        baseline_final = simulate_twsb_with_bisect(
            base_commits_final, exhaustive_parallel, None, WORKER_POOLS
        )
        baseline_final = convert_result_minutes_to_hours(baseline_final)

    baseline_fb = baseline_final.get("mean_feedback_time_hr")
    baseline_mean_ttc = baseline_final.get("mean_time_to_culprit_hr")
    baseline_max_ttc = baseline_final.get("max_time_to_culprit_hr")
    baseline_tests = baseline_final.get("total_tests_run")

    def time_saved_pct(base, val):
        return (
            round((base - val) / base * 100.0, 2)
            if base and base > 0
            else 0.0
        )

    final_results = {
        "Exhaustive Testing (ET)": et_results_final,
        "final_window": {
            "lower": final_lower.isoformat(),
            "upper": final_upper.isoformat() if final_upper else None,
        },
        "worker_pools": _worker_pools_for_output(WORKER_POOLS),
        "num_test_workers": sum(int(v) for v in WORKER_POOLS.values()),
    }
    if baseline_selected:
        final_results["Baseline (TWSB + PAR)"] = baseline_final

    eval_out = eval_payload["eval_output"]
    if selected_batching is not None:
        selected_batching = set(selected_batching)
    if selected_bisection is not None:
        selected_bisection = set(selected_bisection)

    # Replay every combo found in eval_out (except ET/Baseline/summary keys)
    for combo_name, val in eval_out.items():
        if combo_name in (
            "Exhaustive Testing (ET)",
            "Baseline (TWSB + PAR)",
            "best_by_total_tests",
            "best_by_max_ttc",
            "best_by_mean_feedback_time",
            "bet_overall_improvement_over_baseline",
            "num_test_workers",
            "worker_pools",
        ):
            continue

        # Expect "Batch + Bisect" naming
        if " + " not in combo_name:
            continue
        b_name, bis_name = combo_name.split(" + ", 1)
        if selected_batching is not None and b_name not in selected_batching:
            continue
        if selected_bisection is not None and bis_name not in selected_bisection:
            continue
        b_fn, _ = lookup_batching(b_name)
        bis_fn = lookup_bisection(bis_name)
        if b_fn is None or bis_fn is None:
            continue

        # Unpack params for this strategy
        best_params = val.get("best_params")
        normalized_batch_name = b_name[:-2] if b_name.endswith("-s") else b_name
        if normalized_batch_name == "RAPB":
            if not isinstance(best_params, dict):
                continue
            param = (
                best_params["RAPB_THRESHOLD"],
                best_params["RAPB_AGING_PER_HOUR"],
            )
        elif normalized_batch_name == "RAPB-la":
            if not isinstance(best_params, dict):
                continue
            param = (
                best_params["RAPB_LA_BUDGET"],
                best_params["RAPB_LA_AGING_PER_HOUR"],
            )
        elif normalized_batch_name == "RATB":
            if not isinstance(best_params, dict):
                continue
            param = (
                best_params["RATB_THRESHOLD"],
                best_params["RATB_TIME_WINDOW_HOURS"],
            )
        elif normalized_batch_name == "TWSB":
            # No tunable params for TWSB
            param = None
        else:
            # single-valued dict e.g., {"BATCH_HOURS": x} 
            # or {"FSB_SIZE": n} or {"RASB_THRESHOLD": t} 
            # or {"RASB_LA_BUDGET": b}
            try:
                if not isinstance(best_params, dict):
                    continue
                param = list(best_params.values())[0]
            except Exception:
                continue

        res_final = b_fn(base_commits_final, bis_fn, param, WORKER_POOLS)
        res_final = convert_result_minutes_to_hours(res_final)

        saved_pct = time_saved_pct(
            baseline_tests, res_final.get("total_tests_run", baseline_tests)
        )

        res_final.update(
            {
                "tests_saved_vs_baseline_pct": saved_pct,
                "mean_feedback_time_saved_vs_baseline_pct": time_saved_pct(
                    baseline_fb, res_final.get("mean_feedback_time_hr", baseline_fb)
                ),
                "mean_time_to_culprit_saved_vs_baseline_pct": time_saved_pct(
                    baseline_mean_ttc,
                    res_final.get("mean_time_to_culprit_hr", baseline_mean_ttc),
                ),
                "max_time_to_culprit_saved_vs_baseline_pct": time_saved_pct(
                    baseline_max_ttc,
                    res_final.get("max_time_to_culprit_hr", baseline_max_ttc),
                ),
                "best_params_from_eval": best_params,
            }
        )

        final_results[combo_name] = res_final

    # ----- Robust summary winners on FINAL (avoid KeyError) -----
    eligible = []
    for k, v in final_results.items():
        if k in (
            "Exhaustive Testing (ET)",
            "Baseline (TWSB + PAR)",
            "final_window",
            "best_by_total_tests",
            "best_by_max_ttc",
            "best_by_mean_feedback_time",
            "bet_overall_improvement_over_baseline",
            "num_test_workers",
        ):
            continue
        if isinstance(v, dict) and all(
            m in v
            for m in (
                "total_tests_run",
                "max_time_to_culprit_hr",
                "mean_feedback_time_hr",
            )
        ):
            eligible.append((k, v))

    if eligible:
        final_results["best_by_total_tests"] = min(
            eligible, key=lambda kv: kv[1]["total_tests_run"]
        )[0]
        final_results["best_by_max_ttc"] = min(
            eligible, key=lambda kv: kv[1]["max_time_to_culprit_hr"]
        )[0]
        final_results["best_by_mean_feedback_time"] = min(
            eligible, key=lambda kv: kv[1]["mean_feedback_time_hr"]
        )[0]
        # Best overall vs baseline on FINAL window (centralized logic)
        final_results["bet_overall_improvement_over_baseline"] = (
            choose_best_overall_from_items(eligible)
        )
    else:
        final_results["best_by_total_tests"] = "-"
        final_results["best_by_max_ttc"] = "-"
        final_results["best_by_mean_feedback_time"] = "-"
        final_results["bet_overall_improvement_over_baseline"] = "NA"

    # Save FINAL
    os.makedirs(os.path.dirname(OUTPUT_PATH_FINAL), exist_ok=True)
    with open(OUTPUT_PATH_FINAL, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    logger.info("Saved FINAL replay to %s", OUTPUT_PATH_FINAL)
    return final_results


def main():
    """
    CLI entrypoint.

    Runs EVAL Optuna tuning by default, then optionally runs FINAL replay
    using the selected EVAL configurations.
    """
    args = get_args()

    log_level_name = getattr(args, "log_level", "INFO")
    log_level = getattr(logging, log_level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    # Apply random seed override as early as possible for reproducibility
    global RANDOM_SEED
    RANDOM_SEED = int(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    logger.info("Starting batch-testing simulation CLI")
    logger.info(
        "Parsed CLI args: mopt_trials=%d, final_only=%s, num_test_workers=%s, "
        "workers(android/windows/linux/mac)=(%d/%d/%d/%d), "
        "unknown_platform_pool=%s, "
        "build_time_minutes=%s, "
        "full_suite_sigs_per_run=%s, dont_use_all_tests_per_batch=%s, "
        "batching=%s, bisection=%s, skip_exhaustive_testing=%s, dry_run=%s, "
        "log_level=%s, random_seed=%d",
        args.mopt_trials,
        args.final_only,
        str(args.num_test_workers),
        int(getattr(args, "workers_android", 0)),
        int(getattr(args, "workers_windows", 0)),
        int(getattr(args, "workers_linux", 0)),
        int(getattr(args, "workers_mac", 0)),
        str(getattr(args, "unknown_platform_pool", "")),
        str(getattr(args, "build_time_minutes", None)),
        str(args.full_suite_sigs_per_run),
        str(args.dont_use_all_tests_per_batch),
        str(getattr(args, "batching", "all")),
        str(getattr(args, "bisection", "all")),
        str(bool(getattr(args, "skip_exhaustive_testing", False))),
        str(bool(getattr(args, "dry_run", False))),
        log_level_name,
        RANDOM_SEED,
    )

    # Apply CLI overrides to global WORKER_POOLS and FULL_SUITE_SIGNATURES_PER_RUN
    global WORKER_POOLS, FULL_SUITE_SIGNATURES_PER_RUN, DRY_RUN
    if args.num_test_workers is not None:
        # Single shared pool (legacy mode)
        WORKER_POOLS = {"default": int(args.num_test_workers)}
    else:
        WORKER_POOLS = {
            "android": int(getattr(args, "workers_android", bisection_mod.ANDROID_WORKERS)),
            "windows": int(getattr(args, "workers_windows", bisection_mod.WINDOWS_WORKERS)),
            "linux": int(getattr(args, "workers_linux", bisection_mod.LINUX_WORKERS)),
            "mac": int(getattr(args, "workers_mac", bisection_mod.MAC_WORKERS)),
        }

    unknown_platform_pool = str(getattr(args, "unknown_platform_pool", "")).strip().lower()
    if args.num_test_workers is None and unknown_platform_pool not in WORKER_POOLS:
        raise ValueError(
            f"--unknown-platform-pool must be one of {sorted(WORKER_POOLS.keys())}; "
            f"got {unknown_platform_pool!r}"
        )

    # Basic validation: pool sizes must be positive integers.
    bad_pools = {k: v for k, v in WORKER_POOLS.items() if int(v) <= 0}
    if bad_pools:
        raise ValueError(f"All worker pool sizes must be positive; got: {bad_pools}")
    # By default, each initial batch test run executes all signature-groups
    # that appear at least once within the cutoff window. If the user
    # passes --dont-use-all-tests-per-batch, we instead cap each run
    # to a random subset whose size is --full-suite-sigs-per-run.
    if args.dont_use_all_tests_per_batch:
        if args.full_suite_sigs_per_run is not None:
            val = int(args.full_suite_sigs_per_run)
            if val < 0:
                FULL_SUITE_SIGNATURES_PER_RUN = -1
            elif val == 0:
                raise ValueError(
                    "--full-suite-sigs-per-run must be a positive integer "
                    "or -1 to indicate 'use all signature-groups'; got 0."
                )
            else:
                FULL_SUITE_SIGNATURES_PER_RUN = val
        else:
            # Fall back to default cap (may itself be -1 or a positive cap).
            FULL_SUITE_SIGNATURES_PER_RUN = FULL_SUITE_SIGNATURES_PER_RUN
    else:
        # Default: no cap; use all signature-groups from the cutoff-window union.
        FULL_SUITE_SIGNATURES_PER_RUN = -1

    DRY_RUN = bool(getattr(args, "dry_run", False))

    # Resolve build-time overhead (minutes) from CLI.
    build_time_minutes = float(getattr(args, "build_time_minutes", BUILD_TIME_MINUTES))

    # Propagate defaults/knobs to bisection_strats
    configure_bisection_defaults(
        default_test_duration_min=DEFAULT_TEST_DURATION_MIN,
        full_suite_signatures_per_run=FULL_SUITE_SIGNATURES_PER_RUN,
        build_time_minutes=build_time_minutes,
        unknown_platform_pool=unknown_platform_pool,
    )
    global INPUT_JSON_EVAL, INPUT_JSON_FINAL, OUTPUT_PATH_EVAL, OUTPUT_PATH_FINAL
    INPUT_JSON_EVAL = args.input_json_eval
    INPUT_JSON_FINAL = args.input_json_final
    OUTPUT_PATH_EVAL = args.output_eval
    OUTPUT_PATH_FINAL = args.output_final

    # Resolve strategy filters (default is "all", meaning no filtering).
    batching_names = [n for n, _, _ in BATCHING_STRATEGIES] + ["TWSB"]
    bisection_names = [n for n, _ in BISECTION_STRATEGIES]
    selected_batching_set = _resolve_strategy_filter(
        getattr(args, "batching", "all"), batching_names, kind="batching"
    )
    selected_bisection_set = _resolve_strategy_filter(
        getattr(args, "bisection", "all"), bisection_names, kind="bisection"
    )
    selected_batching = (
        None if selected_batching_set == set(batching_names) else selected_batching_set
    )
    selected_bisection = (
        None
        if selected_bisection_set == set(bisection_names)
        else selected_bisection_set
    )
    run_et = not bool(getattr(args, "skip_exhaustive_testing", False))

    # Determine the set of failing (buggy) revisions that fall within
    # the cutoff windows used for EVAL and FINAL, and restrict the
    # sanity-check coverage validation to those revisions only.
    logger.info(
        "Computing failing revisions within EVAL and FINAL cutoff windows "
        "for perf signature coverage validation."
    )

    # EVAL window
    eval_preds_raw = load_predictions_raw(INPUT_JSON_EVAL)
    eval_oldest, eval_newest = get_cutoff_from_input(ALL_COMMITS_PATH, eval_preds_raw)
    eval_lower = eval_oldest or DEFAULT_CUTOFF
    eval_upper = eval_newest
    eval_commits, eval_predicted_indices = build_commits_from_all_with_raw_preds(
        ALL_COMMITS_PATH,
        eval_preds_raw,
        eval_lower,
        eval_upper,
    )
    failing_revs_eval = {
        c["commit_id"] for c in eval_commits if c.get("true_label")
    }

    # FINAL window
    final_preds_raw = load_predictions_raw(INPUT_JSON_FINAL)
    final_oldest, final_newest = get_cutoff_from_input(
        ALL_COMMITS_PATH, final_preds_raw
    )
    final_lower = final_oldest or DEFAULT_CUTOFF
    final_upper = final_newest
    final_commits, final_predicted_indices = build_commits_from_all_with_raw_preds(
        ALL_COMMITS_PATH,
        final_preds_raw,
        final_lower,
        final_upper,
    )
    failing_revs_final = {
        c["commit_id"] for c in final_commits if c.get("true_label")
    }

    failing_revisions = failing_revs_eval.union(failing_revs_final)
    logger.info(
        "Identified %d failing revisions within EVAL/FINAL windows for "
        "perf coverage validation.",
        len(failing_revisions),
    )

    # Restrict the "full suite" batch runs to the union of signature-groups
    # that actually appear at least once within the cutoff windows used
    # for EVAL and FINAL. Whether we run all of them or a random subset
    # is controlled by FULL_SUITE_SIGNATURES_PER_RUN.
    cutoff_revs = {
        c["commit_id"] for c in eval_commits
    }.union(
        c["commit_id"] for c in final_commits
    )
    configure_full_suite_signatures_union(cutoff_revs)

    # Sanity check: ensure that all failing perf signature-groups for the
    # commits we will simulate (within the EVAL and FINAL windows) are
    # actually present in perf_jobs_per_revision_details_rectified.jsonl. If not,
    # the simulation would be unable to exercise all relevant failing
    # signature-groups and results would be misleading.
    validate_failing_signatures_coverage(failing_revisions=failing_revisions)

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
        logger.info(
            "Reusing eval results from %s; running FINAL only...",
            OUTPUT_PATH_EVAL,
        )
        run_final_test_unified(
            eval_payload,
            INPUT_JSON_FINAL,
            OUTPUT_PATH_FINAL,
            selected_batching=selected_batching,
            selected_bisection=selected_bisection,
            run_exhaustive_testing_et=run_et,
            preds_raw_final=final_preds_raw,
            base_commits_final=final_commits,
            predicted_indices_final=final_predicted_indices,
            final_lower=final_lower,
            final_upper=final_upper,
        )
        return

    # Normal flow: run Optuna evaluation (mopt), save eval output, then FINAL replay
    eval_payload = run_evaluation_mopt(
        INPUT_JSON_EVAL,
        n_trials=args.mopt_trials,
        selected_batching=selected_batching,
        selected_bisection=selected_bisection,
        run_exhaustive_testing_et=run_et,
        dry_run=DRY_RUN,
        preds_raw=eval_preds_raw,
        base_commits_for_context=eval_commits,
        predicted_indices=eval_predicted_indices,
        lower_cutoff=eval_lower,
        upper_cutoff=eval_upper,
    )

    if eval_payload is None:
        raise RuntimeError("Evaluation was unsuccessful. No eval payload present")

    os.makedirs(os.path.dirname(OUTPUT_PATH_EVAL), exist_ok=True)
    with open(OUTPUT_PATH_EVAL, "w", encoding="utf-8") as f:
        json.dump(eval_payload["eval_output"], f, indent=2)
    logger.info("Saved EVAL results to %s", OUTPUT_PATH_EVAL)

    # Unified FINAL replay
    run_final_test_unified(
        eval_payload,
        INPUT_JSON_FINAL,
        OUTPUT_PATH_FINAL,
        selected_batching=selected_batching,
        selected_bisection=selected_bisection,
        run_exhaustive_testing_et=run_et,
        preds_raw_final=final_preds_raw,
        base_commits_final=final_commits,
        predicted_indices_final=final_predicted_indices,
        final_lower=final_lower,
        final_upper=final_upper,
    )


if __name__ == "__main__":
    main()
