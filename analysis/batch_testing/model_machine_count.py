#!/usr/bin/env python3
"""
Machine-count sweep for batch-testing simulation strategies.

For each worker-pool multiplier, this script:
  1) runs Optuna tuning on the EVAL set (like simulation.py),
  2) replays the chosen configuration(s) on the FINAL set, and
  3) records FINAL latency metrics for plotting.

It then writes one plot per (batching × bisection) combo showing:
  - mean_feedback_time_hr
  - mean_time_to_culprit_hr
  - max_time_to_culprit_hr
vs worker-pool multiplier.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import tempfile
from decimal import Decimal, InvalidOperation


logger = logging.getLogger(__name__)


def _parse_decimal(raw: str) -> Decimal:
    txt = str(raw).strip().lower()
    if txt.startswith("x"):
        txt = txt[1:].strip()
    try:
        return Decimal(txt)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid decimal value: {raw!r}") from exc


def default_multipliers() -> list[float]:
    """
    Default sweep schedule:
      - start at 0.01
      - increase by 0.05 until reaching 1 (include 1.0 explicitly)
      - then increase by 0.5 up to 5.0 (inclusive)
    """
    vals: list[float] = []

    cur = Decimal("0.01")
    step_small = Decimal("0.05")
    one = Decimal("1.0")

    while cur < one:
        vals.append(float(cur))
        cur += step_small

    if not vals or vals[-1] != 1.0:
        vals.append(1.0)

    cur = Decimal("1.5")
    step_large = Decimal("0.5")
    hi = Decimal("5.0")
    while cur <= hi + Decimal("1e-18"):
        vals.append(float(cur))
        cur += step_large

    # Ensure stable ordering + uniqueness (guarding against future edits).
    return sorted(set(vals))


def generate_multipliers(multiplier_list: str | None) -> list[float]:
    if multiplier_list:
        multipliers: list[float] = []
        for token in str(multiplier_list).split(","):
            tok = token.strip()
            if not tok:
                continue
            multipliers.append(float(_parse_decimal(tok)))
        if not multipliers:
            raise ValueError("--multiplier-list was provided but no values were parsed.")
        return sorted(multipliers)

    return default_multipliers()


def scaled_worker_pools(
    *,
    base_pools: dict[str, int],
    multiplier: float,
    min_workers_per_pool: int = 1,
) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, base in base_pools.items():
        scaled = int(round(float(base) * float(multiplier)))
        if scaled < int(min_workers_per_pool):
            scaled = int(min_workers_per_pool)
        out[name] = scaled
    return out


def sanitize_filename(text: str) -> str:
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch in (" ", "+", ".", "/", "\\", ":", "|"):
            keep.append("_")
        else:
            keep.append("_")
    slug = "".join(keep)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "plot"


def build_split_inputs(*, sim, input_json: str):
    preds_raw = sim.load_predictions_raw(input_json)
    oldest, newest = sim.get_cutoff_from_input(sim.ALL_COMMITS_PATH, preds_raw)
    lower = oldest or sim.DEFAULT_CUTOFF
    upper = newest
    commits = sim.build_commits_from_all_with_raw_preds(
        sim.ALL_COMMITS_PATH,
        preds_raw,
        lower,
        upper,
    )
    return preds_raw, commits, lower, upper


def _list_strategies_and_exit(*, sim) -> None:
    batching_names = [n for n, _, _ in sim.BATCHING_STRATEGIES] + ["TWSB"]
    bisection_names = [n for n, _ in sim.BISECTION_STRATEGIES]
    print("Batching strategies:")
    for name in batching_names:
        print(f"  - {name}")
    print("\nBisection strategies:")
    for name in bisection_names:
        print(f"  - {name}")
    raise SystemExit(0)


def get_args(*, sim, bisection_mod) -> argparse.Namespace:
    default_eval = os.path.join(
        sim.REPO_ROOT,
        "analysis",
        "batch_testing",
        "final_test_results_perf_codebert_eval.json",
    )
    default_final = os.path.join(
        sim.REPO_ROOT,
        "analysis",
        "batch_testing",
        "final_test_results_perf_codebert_final_test.json",
    )
    default_plots_dir = os.path.join(
        sim.REPO_ROOT, "analysis", "batch_testing", "results", "plots"
    )
    default_out_json = os.path.join(
        sim.REPO_ROOT, "analysis", "batch_testing", "results", "machine_count_sweep.json"
    )

    ap = argparse.ArgumentParser(
        description=(
            "Sweep worker pool multipliers and plot FINAL latency metrics per combo.\n\n"
            "Example:\n"
            "  python analysis/batch_testing/model_machine_count.py \\\n"
            "    --batching TWB,RATB \\\n"
            "    --bisection PAR,TOB \\\n"
            "    --mopt-trials 50 \\\n"
            "    --skip-exhaustive-testing\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    ap.add_argument(
        "--list-strategies",
        action="store_true",
        help="Print available batching/bisection strategy names and exit.",
    )
    ap.add_argument("--input-json-eval", default=default_eval, help="EVAL predictions JSON.")
    ap.add_argument(
        "--input-json-final", default=default_final, help="FINAL predictions JSON."
    )

    ap.add_argument(
        "--mopt-trials",
        type=int,
        default=200,
        help="Base Optuna trials per tunable parameter (same meaning as simulation.py).",
    )
    ap.add_argument(
        "--optuna-seed",
        type=int,
        default=sim.RANDOM_SEED,
        help="Optuna sampler seed (for reproducibility).",
    )
    ap.add_argument(
        "--optimize-for-timeliness-metric",
        default=sim.DEFAULT_OPTIMIZE_FOR_TIMELINESS_METRIC,
        help="Timeliness metric for Optuna objective + Pareto selection (e.g., max_ttc, mean_ttc, p95_ttc, mft).",
    )
    ap.add_argument(
        "--baseline-opt-metric-multplier",
        type=float,
        default=1.0,
        help="Baseline timeliness multiplier used when selecting a single Pareto point per combo.",
    )

    ap.add_argument(
        "--batching",
        default="all",
        help=(
            "Comma-separated batching strategy names to simulate (default: all).\n"
            "Examples: 'TWB,RATB-s' or 'all' or 'none'."
        ),
    )
    ap.add_argument(
        "--bisection",
        default="all",
        help=(
            "Comma-separated bisection strategy names to simulate (default: all).\n"
            "Examples: 'TOB,PAR' or 'all' or 'none'."
        ),
    )

    ap.add_argument(
        "--multiplier-list",
        default=None,
        help=(
            "Optional CSV list of multipliers overriding the default schedule.\n"
            "Example: '0.01,0.06,0.11,0.16,1,1.5,2,2.5,3,3.5,4,4.5,5'"
        ),
    )

    # Base worker pool counts (these are the values being multiplied).
    ap.add_argument(
        "--base-workers-android",
        type=int,
        default=bisection_mod.ANDROID_WORKERS,
        help="Base android worker pool size.",
    )
    ap.add_argument(
        "--base-workers-windows",
        type=int,
        default=bisection_mod.WINDOWS_WORKERS,
        help="Base windows worker pool size.",
    )
    ap.add_argument(
        "--base-workers-linux",
        type=int,
        default=bisection_mod.LINUX_WORKERS,
        help="Base linux worker pool size.",
    )
    ap.add_argument(
        "--base-workers-mac",
        type=int,
        default=bisection_mod.MAC_WORKERS,
        help="Base mac worker pool size.",
    )
    ap.add_argument(
        "--unknown-platform-pool",
        type=str,
        default=bisection_mod.DEFAULT_UNKNOWN_PLATFORM_POOL,
        help="Pool key used when a signature-group cannot be mapped to a platform.",
    )
    ap.add_argument(
        "--build-time-minutes",
        type=float,
        default=sim.BUILD_TIME_MINUTES,
        help="Build-time overhead (minutes) added once per suite run.",
    )
    ap.add_argument(
        "--default-test-duration-min",
        type=float,
        default=sim.DEFAULT_TEST_DURATION_MIN,
        help="Fallback test duration (minutes) for missing signature-group durations.",
    )

    ap.add_argument(
        "--skip-exhaustive-testing",
        action="store_true",
        help="Skip the expensive Exhaustive Testing (ET) baseline (recommended for sweeps).",
    )

    ap.add_argument(
        "--plots-dir",
        default=default_plots_dir,
        help="Directory to write per-combo plots.",
    )
    ap.add_argument(
        "--out-json",
        default=default_out_json,
        help="Where to write the aggregated sweep results JSON.",
    )
    ap.add_argument(
        "--skip-plots",
        action="store_true",
        help="If set, skip generating plots (still writes --out-json).",
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    ap.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for any non-Optuna randomness.",
    )

    return ap.parse_args()


def main() -> None:
    try:
        import simulation as sim
        import bisection_strats as bisection_mod
    except ImportError as exc:
        raise RuntimeError(
            "Could not import batch-testing simulator modules. Run this script as:\n"
            "  python analysis/batch_testing/model_machine_count.py ..."
        ) from exc

    args = get_args(sim=sim, bisection_mod=bisection_mod)

    if args.list_strategies:
        _list_strategies_and_exit(sim=sim)

    # Fail fast on required dependency.
    try:
        import optuna  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Optuna is required to run this sweep. Install with `pip install optuna`."
        ) from exc

    log_level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    random.seed(int(args.random_seed))
    sim.DRY_RUN = False

    batching_names = [n for n, _, _ in sim.BATCHING_STRATEGIES] + ["TWSB"]
    bisection_names = [n for n, _ in sim.BISECTION_STRATEGIES]
    selected_batching_set = sim._resolve_strategy_filter(
        getattr(args, "batching", "all"), batching_names, kind="batching"
    )
    selected_bisection_set = sim._resolve_strategy_filter(
        getattr(args, "bisection", "all"), bisection_names, kind="bisection"
    )
    if not selected_batching_set:
        raise ValueError("--batching resolved to an empty set (no batching strategies selected).")
    if not selected_bisection_set:
        raise ValueError("--bisection resolved to an empty set (no bisection strategies selected).")

    selected_batching = None if selected_batching_set == set(batching_names) else selected_batching_set
    selected_bisection = None if selected_bisection_set == set(bisection_names) else selected_bisection_set

    multiplier_values = generate_multipliers(args.multiplier_list)

    base_worker_pools = {
        "android": int(args.base_workers_android),
        "windows": int(args.base_workers_windows),
        "linux": int(args.base_workers_linux),
        "mac": int(args.base_workers_mac),
    }
    bad_pools = {k: v for k, v in base_worker_pools.items() if int(v) <= 0}
    if bad_pools:
        raise ValueError(f"All base worker pools must be positive; got: {bad_pools}")

    unknown_platform_pool = str(args.unknown_platform_pool).strip().lower()
    if unknown_platform_pool not in base_worker_pools:
        raise ValueError(
            f"--unknown-platform-pool must be one of {sorted(base_worker_pools.keys())}; got {unknown_platform_pool!r}"
        )

    logger.info(
        "Running sweep: batching=%s bisection=%s multipliers=%d base_worker_pools=%s",
        getattr(args, "batching", "all"),
        getattr(args, "bisection", "all"),
        len(multiplier_values),
        base_worker_pools,
    )

    # Configure shared simulator knobs (same as simulation.py main()).
    sim.configure_bisection_defaults(
        default_test_duration_min=float(args.default_test_duration_min),
        build_time_minutes=float(args.build_time_minutes),
        unknown_platform_pool=unknown_platform_pool,
    )

    # Build and cache split inputs once; worker pools only affect scheduling.
    logger.info("Loading EVAL split from %s", args.input_json_eval)
    eval_preds_raw, eval_commits, eval_lower, eval_upper = build_split_inputs(
        sim=sim, input_json=args.input_json_eval
    )
    logger.info("Loading FINAL split from %s", args.input_json_final)
    final_preds_raw, final_commits, final_lower, final_upper = build_split_inputs(
        sim=sim, input_json=args.input_json_final
    )

    # Match simulation.py behavior: restrict the "full suite" to the union of
    # signature-groups that appear at least once in EVAL+FINAL windows, and
    # validate failing signature coverage.
    cutoff_revs = {c["commit_id"] for c in eval_commits}.union(
        {c["commit_id"] for c in final_commits}
    )
    sim.configure_full_suite_signatures_union(cutoff_revs)

    failing_revisions = {
        c["commit_id"] for c in eval_commits if c.get("true_label")
    }.union({c["commit_id"] for c in final_commits if c.get("true_label")})
    sim.validate_failing_signatures_coverage(failing_revisions=failing_revisions)

    timeliness_metric_key = sim.resolve_timeliness_metric_key(
        getattr(args, "optimize_for_timeliness_metric", sim.DEFAULT_OPTIMIZE_FOR_TIMELINESS_METRIC)
    )
    baseline_opt_metric_multplier = float(args.baseline_opt_metric_multplier)
    if baseline_opt_metric_multplier <= 0.0:
        raise ValueError(
            f"--baseline-opt-metric-multplier must be > 0; got {baseline_opt_metric_multplier!r}"
        )

    os.makedirs(str(args.plots_dir), exist_ok=True)
    out_json_dir = os.path.dirname(str(args.out_json))
    if out_json_dir:
        os.makedirs(out_json_dir, exist_ok=True)

    batching_list = sorted(selected_batching_set)
    bisection_list = sorted(selected_bisection_set)
    combo_keys = [f"{b} + {bis}" for b in batching_list for bis in bisection_list]
    sweep_results: dict[str, list[dict]] = {k: [] for k in combo_keys}

    run_et = not bool(getattr(args, "skip_exhaustive_testing", False))

    with tempfile.TemporaryDirectory(prefix="batch_testing_machine_count_sweep_") as tmp_dir:
        tmp_final_path = os.path.join(tmp_dir, "final_replay.json")

        for idx, mult in enumerate(multiplier_values, start=1):
            worker_pools = scaled_worker_pools(base_pools=base_worker_pools, multiplier=mult)
            sim.WORKER_POOLS = dict(worker_pools)

            logger.info(
                "[%d/%d] multiplier=%.6f worker_pools=%s",
                idx,
                len(multiplier_values),
                float(mult),
                worker_pools,
            )

            eval_payload = sim.run_evaluation_mopt(
                args.input_json_eval,
                n_trials=int(args.mopt_trials),
                optuna_seed=int(args.optuna_seed),
                optimize_for_timeliness_metric=timeliness_metric_key,
                baseline_opt_metric_multplier=baseline_opt_metric_multplier,
                selected_batching=selected_batching,
                selected_bisection=selected_bisection,
                run_exhaustive_testing_et=run_et,
                dry_run=False,
                preds_raw=eval_preds_raw,
                base_commits_for_context=eval_commits,
                lower_cutoff=eval_lower,
                upper_cutoff=eval_upper,
            )

            final_results = sim.run_final_test_unified(
                eval_payload,
                args.input_json_final,
                tmp_final_path,
                selected_batching=selected_batching,
                selected_bisection=selected_bisection,
                run_exhaustive_testing_et=run_et,
                preds_raw_final=final_preds_raw,
                base_commits_final=final_commits,
                final_lower=final_lower,
                final_upper=final_upper,
            )

            for combo_key in combo_keys:
                if combo_key not in final_results:
                    raise KeyError(
                        f"Missing combo {combo_key!r} in FINAL results for multiplier={float(mult):.6f}. "
                        "This typically means the eval stage did not produce a selected configuration for it."
                    )
                entry = final_results[combo_key]
                if not isinstance(entry, dict):
                    raise TypeError(
                        f"FINAL result for {combo_key!r} was not a dict (got {type(entry)})."
                    )

                try:
                    mft = float(entry["mean_feedback_time_hr"])
                    mean_ttc = float(entry["mean_time_to_culprit_hr"])
                    max_ttc = float(entry["max_time_to_culprit_hr"])
                except KeyError as exc:
                    raise KeyError(
                        f"Missing expected metric {exc!s} for combo {combo_key!r} at multiplier={float(mult):.6f}."
                    ) from exc
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Non-numeric metric value for combo {combo_key!r} at multiplier={float(mult):.6f}."
                    ) from exc

                sweep_results[combo_key].append(
                    {
                        "multiplier": float(mult),
                        "worker_pools": worker_pools,
                        "mean_feedback_time_hr": mft,
                        "mean_time_to_culprit_hr": mean_ttc,
                        "max_time_to_culprit_hr": max_ttc,
                        "total_tests_run": float(entry.get("total_tests_run", 0.0)),
                        "total_cpu_time_hr": float(entry.get("total_cpu_time_hr", 0.0)),
                        "found_all_regressors": bool(entry.get("found_all_regressors", False)),
                    }
                )

    payload = {
        "meta": {
            "input_json_eval": str(args.input_json_eval),
            "input_json_final": str(args.input_json_final),
            "batching_raw": str(getattr(args, "batching", "all")),
            "bisection_raw": str(getattr(args, "bisection", "all")),
            "batching_selected": sorted(selected_batching_set),
            "bisection_selected": sorted(selected_bisection_set),
            "multipliers": multiplier_values,
            "base_worker_pools": base_worker_pools,
            "mopt_trials": int(args.mopt_trials),
            "optuna_seed": int(args.optuna_seed),
            "optimize_for_timeliness_metric": timeliness_metric_key,
            "baseline_opt_metric_multplier": baseline_opt_metric_multplier,
            "skip_exhaustive_testing": bool(getattr(args, "skip_exhaustive_testing", False)),
        },
        "results": sweep_results,
    }

    with open(str(args.out_json), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote sweep results JSON to %s", str(args.out_json))

    if args.skip_plots:
        logger.info("--skip-plots was set; skipping plot generation.")
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plot generation. Install with `pip install matplotlib` "
            "(or run with --skip-plots)."
        ) from exc

    for combo_key in combo_keys:
        rows = sweep_results.get(combo_key, [])
        xs = [r["multiplier"] for r in rows]
        mft = [r["mean_feedback_time_hr"] for r in rows]
        mean_ttc = [r["mean_time_to_culprit_hr"] for r in rows]
        max_ttc = [r["max_time_to_culprit_hr"] for r in rows]

        plt.figure(figsize=(11, 6.5))
        plt.plot(xs, mft, label="mean_feedback_time_hr", linewidth=2)
        plt.plot(xs, mean_ttc, label="mean_time_to_culprit_hr", linewidth=2)
        plt.plot(xs, max_ttc, label="max_time_to_culprit_hr", linewidth=2)
        plt.axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6)

        plt.xlabel("Worker pool multiplier (x)")
        plt.ylabel("Hours (lower is better)")
        plt.title(f"{combo_key}: FINAL latency vs worker capacity")
        plt.grid(True, linestyle=":", linewidth=0.6)
        plt.legend()
        plt.tight_layout()

        out_name = f"{sanitize_filename(combo_key)}__machine_count_sweep.png"
        out_path = os.path.join(str(args.plots_dir), out_name)
        plt.savefig(out_path, dpi=200)
        plt.close()
        logger.info("Wrote plot to %s", out_path)


if __name__ == "__main__":
    main()
