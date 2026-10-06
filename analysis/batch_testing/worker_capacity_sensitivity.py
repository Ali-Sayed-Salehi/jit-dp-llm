#!/usr/bin/env python3
"""Vary worker capacity using the paper's simulation and frozen parameters."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import time

import bisection_strats as bis
import simulation as sim
from model_machine_count import scaled_worker_pools
from worker_diagnostics import WorkerDiagnostics, observe_workers


HERE = Path(__file__).resolve().parent
STRATEGIES = ("TWSB", "RAPB-la", "RASB-la", "FSB")
METRIC_KEYS = (
    "total_tests_run", "mean_feedback_time_hr", "mean_time_to_culprit_hr",
    "max_time_to_culprit_hr", "p90_time_to_culprit_hr", "p95_time_to_culprit_hr",
    "p99_time_to_culprit_hr", "total_cpu_time_hr", "num_regressors_total",
    "num_regressors_found", "found_all_regressors",
)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def load_split(path, split, seed):
    predictions = sim.load_predictions_raw(str(path))
    oldest, newest = sim.get_cutoff_from_input(sim.ALL_COMMITS_PATH, predictions)
    commits = sim.build_commits_from_all_with_raw_preds(
        sim.ALL_COMMITS_PATH, predictions, oldest or sim.DEFAULT_CUTOFF, newest,
        risk_score_mode="learned", risk_seed=seed, split_name=split,
    )
    return commits


def frozen_parameters(saved_eval):
    parameters = {"TWSB": (None, {})}
    for strategy in STRATEGIES[1:]:
        saved = saved_eval[strategy + " + PAR"]["best_params"]
        if strategy == "FSB":
            value = int(saved["FSB_SIZE"])
        elif strategy == "RASB-la":
            value = float(saved["RASB_LA_BUDGET"])
        else:
            value = (float(saved["RAPB_LA_BUDGET"]), float(saved["RAPB_LA_AGING_PER_HOUR"]))
        parameters[strategy] = (value, saved)
    return parameters


def verify_paper_metrics(metrics, saved):
    for key in METRIC_KEYS:
        if saved.get(key) != metrics.get(key):
            raise AssertionError(f"Paper reference verification failed for {key}")


def replay(strategy, commits, parameters, pools, *, observe=True):
    sim.WORKER_POOLS = dict(pools)

    def run():
        if strategy == "ET":
            return sim.run_exhaustive_testing(commits)
        function, _ = sim.lookup_batching(strategy)
        return sim.convert_result_minutes_to_hours(function(
            commits, sim.lookup_bisection("PAR"), parameters[strategy][0], pools,
        ))

    if not observe:
        return {"metrics": run()}
    recorder = WorkerDiagnostics(commits[0]["ts"], commits[-1]["ts"])
    with observe_workers(recorder):
        metrics = run()
    telemetry, daily = recorder.finish()
    if telemetry["aggregate"]["total_jobs"] != metrics["total_tests_run"]:
        raise AssertionError("Observed jobs differ from strategy test count")
    for pool in telemetry["per_pool"].values():
        if not -1e-8 <= pool["worker_utilization_pct"] <= 100.0 + 1e-8:
            raise AssertionError("Utilization exceeds available worker time")
    return {"metrics": metrics, "workers": telemetry, "daily": daily}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-eval", type=Path, default=HERE / "results/50t_paper_reproduction/batch_eval_mopt.json")
    parser.add_argument("--reference-final", type=Path, default=HERE / "results/50t_paper_reproduction/batch_test_mopt.json")
    parser.add_argument("--input-eval", type=Path, default=HERE / "final_test_results_perf_codebert_eval.json")
    parser.add_argument("--input-final", type=Path, default=HERE / "final_test_results_perf_codebert_final_test.json")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results/worker_capacity_backlog")
    parser.add_argument("--multipliers", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25, 1.5])
    parser.add_argument("--build-time-minutes", type=float, default=98.7)
    parser.add_argument("--skip-exhaustive-testing", action="store_true")
    args = parser.parse_args()
    if any(m <= 0 for m in args.multipliers) or len(set(args.multipliers)) != len(args.multipliers):
        parser.error("Capacity multipliers must be distinct positive values")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    start = time.monotonic()
    saved_eval = json.loads(args.saved_eval.read_text())
    reference = json.loads(args.reference_final.read_text())
    if saved_eval.get("risk_score_mode") != "learned":
        raise ValueError("This experiment requires saved learned-risk parameters")
    seed = int(saved_eval["risk_score_seed"])
    parameters = frozen_parameters(saved_eval)
    base_pools = saved_eval["worker_pools"]
    bis.configure_bisection_defaults(build_time_minutes=args.build_time_minutes)
    eval_commits = load_split(args.input_eval, "eval", seed)
    commits = load_split(args.input_final, "final", seed)
    if not commits or not eval_commits:
        raise ValueError("Both paper input windows must be nonempty")
    all_commits = eval_commits + commits
    bis.configure_full_suite_signatures_union({c["commit_id"] for c in all_commits})
    bis.validate_failing_signatures_coverage(failing_revisions={
        c["commit_id"] for c in all_commits if c.get("true_label")
    })

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance_paths = [
        args.saved_eval, args.reference_final, args.input_eval, args.input_final,
        Path(sim.ALL_COMMITS_PATH), Path(bis.SIG_GROUP_JOB_DURATIONS_CSV),
        Path(bis.ALERT_FAIL_SIGS_CSV), Path(bis.PERF_JOBS_PER_REV_JSON),
        Path(bis.SIG_GROUPS_JSONL), Path(bis.ALL_SIGNATURES_JSONL),
        HERE / "worker_capacity_sensitivity.py", HERE / "worker_diagnostics.py",
        HERE / "model_machine_count.py",
        HERE / "batch_strats.py", HERE / "bisection_strats.py", HERE / "simulation.py",
    ]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
        "inputs_and_code": [file_identity(path) for path in provenance_paths],
        "capacity_multipliers": args.multipliers,
        "base_worker_pools": base_pools,
        "scaling": "Per-pool round(base * multiplier), Python round; minimum one worker",
        "bisection": "PAR", "risk_score_mode": "learned", "risk_score_seed": seed,
        "build_time_minutes": args.build_time_minutes,
        "parameters": {name: p[1] for name, p in parameters.items()},
        "retuned": False,
        "final_window": {"start": commits[0]["ts"].isoformat(), "end": commits[-1]["ts"].isoformat(), "commits": len(commits)},
        "full_suite_signature_groups": len(bis.get_batch_signature_durations()),
        "experiment_scope": "Only worker capacity varies; the paper's simulation and batching parameters are fixed",
        "queue_wait": "Actual test start minus readiness after build, job-weighted across root and diagnosis tests",
        "backlog": "Ready jobs that have not started; excludes running jobs and builds",
        "utilization": "Test busy time clipped to first-to-last FINAL commit / capacity in that same window; includes idle pools",
        "end_of_input": "Counts at the last input timestamp, after events at that timestamp",
        "completion": "All submitted tests and diagnosis events drain; TTC includes post-input completion",
        "cost_scope": "Busy and idle worker-hours reported; no cloud tariff or build-worker pricing assumption added",
    }
    write_json(output / "manifest.json", manifest)
    snapshot = output / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for entry in manifest["inputs_and_code"]:
        source = Path(entry["path"])
        if source.suffix == ".py":
            (snapshot / source.name).write_bytes(source.read_bytes())

    # Verify recording leaves the original simulator and saved paper metrics intact.
    reference_runs = {}
    for strategy in STRATEGIES:
        print(f"Checking original 100% replay: {strategy}", flush=True)
        observed = replay(strategy, commits, parameters, base_pools)
        plain = replay(strategy, commits, parameters, base_pools, observe=False)
        key = "Baseline (TWSB + PAR)" if strategy == "TWSB" else strategy + " + PAR"
        verify_paper_metrics(observed["metrics"], reference[key])
        if observed["metrics"] != plain["metrics"]:
            raise AssertionError(f"Worker recording verification failed: {strategy}")
        observed.pop("daily")
        observed["matches_saved_paper_metrics"] = True
        observed["recording_preserves_original_metrics"] = True
        reference_runs[strategy] = observed
    write_json(output / "original_100pct_reproduction.json", reference_runs)

    results, summary, pool_rows = [], [], []
    strategies = list(STRATEGIES) + ([] if args.skip_exhaustive_testing else ["ET"])
    for multiplier in sorted(args.multipliers):
        pools = scaled_worker_pools(base_pools=base_pools, multiplier=multiplier)
        directory = output / f"capacity_{multiplier:.2f}"
        for strategy in strategies:
            print(f"Running {multiplier:.0%} capacity: {strategy} ({sum(pools.values())} workers)", flush=True)
            replay_start = time.monotonic()
            result = replay(strategy, commits, parameters, pools)
            if multiplier == 1.0 and strategy != "ET":
                key = "Baseline (TWSB + PAR)" if strategy == "TWSB" else strategy + " + PAR"
                verify_paper_metrics(result["metrics"], reference[key])
            daily = result.pop("daily")
            result.update({"strategy": strategy, "capacity_multiplier": multiplier, "worker_pools": pools,
                           "parameters": parameters[strategy][1] if strategy != "ET" else {},
                           "runtime_seconds": time.monotonic() - replay_start})
            write_json(directory / (strategy + ".json"), result)
            write_csv(directory / (strategy + "_daily.csv"), daily)
            results.append(result)
            summary.append({"capacity_multiplier": multiplier, "strategy": strategy,
                            "num_workers": sum(pools.values()), **result["metrics"], **result["workers"]["aggregate"]})
            for pool, metrics in result["workers"]["per_pool"].items():
                pool_rows.append({"capacity_multiplier": multiplier, "strategy": strategy, "pool": pool, **metrics})
            # Persist completed runs incrementally, including before long ET runs.
            write_csv(output / "summary.csv", summary)
            write_csv(output / "per_pool.csv", pool_rows)
            print(f"  max TTC {result['metrics']['max_time_to_culprit_hr']:.2f} h; "
                  f"mean queue wait {result['workers']['aggregate']['mean_job_queue_wait_hr']:.3f} h", flush=True)

    write_json(output / "results.json", {"manifest": manifest, "runs": results})
    write_json(output / "completion.json", {"runs": len(results), "runtime_seconds": time.monotonic() - start,
                                           "all_regressors_found_in_all_runs": all(r["metrics"]["found_all_regressors"] for r in results)})
    print(f"Completed {len(results)} runs. Results: {output}", flush=True)


if __name__ == "__main__":
    main()
