#!/usr/bin/env python3
"""Sweep performance-bisect test runner counts and plot final-test metrics."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import simulation
except ImportError:  # pragma: no cover - supports direct script execution.
    import simulation  # type: ignore


DEFAULT_PLOTS_DIR = simulation.DEFAULT_OUTPUT_DIR / "plots"
DEFAULT_SWEEP_OUTPUT_JSON = (
    simulation.DEFAULT_OUTPUT_DIR / "machine_count_sweep_final_test.json"
)
MAIN_METRICS = (
    "mean_elapsed_hours",
    "max_elapsed_hours",
    "mean_test_runs",
    "max_test_runs",
    "success_rate_percent",
)
METRIC_LABELS = {
    "mean_elapsed_hours": "Mean elapsed time (hours)",
    "max_elapsed_hours": "Max elapsed time (hours)",
    "mean_test_runs": "Mean test runs",
    "max_test_runs": "Max test runs",
    "success_rate_percent": "Success rate (%)",
}
METRIC_TITLES = {
    "mean_elapsed_hours": "Final-Test Mean Elapsed Time",
    "max_elapsed_hours": "Final-Test Max Elapsed Time",
    "mean_test_runs": "Final-Test Mean Test Runs",
    "max_test_runs": "Final-Test Max Test Runs",
    "success_rate_percent": "Final-Test Success Rate",
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the worker-count sweep and write final-test plots."""

    sweep_args, simulation_argv = parse_args(argv)
    simulation_args = simulation.parse_args(simulation_argv)

    if simulation_args.dataset != "all":
        raise ValueError(
            "plot_for_machine_counts.py always runs eval and final_test; "
            "--dataset must be omitted or set to all."
        )
    if len(simulation_args.oracles) != 1:
        raise ValueError(
            "plot_for_machine_counts.py expects exactly one selected test oracle "
            "because sweep outputs are keyed by localizer."
        )

    output_dir = simulation_args.output_dir
    plots_dir = sweep_args.plots_dir or output_dir / "plots"
    sweep_output_json = (
        sweep_args.sweep_output_json
        or output_dir / DEFAULT_SWEEP_OUTPUT_JSON.name
    )

    plots_dir.mkdir(parents=True, exist_ok=True)
    sweep_output_json.parent.mkdir(parents=True, exist_ok=True)
    plt = None if sweep_args.skip_plots else load_matplotlib()

    worker_counts = parse_worker_counts(sweep_args.worker_counts)
    results_by_localizer: dict[str, list[dict[str, Any]]] = {}

    with tempfile.TemporaryDirectory(
        prefix="perf_bisect_worker_count_sweep_",
    ) as tmp_dir:
        tmp_root = Path(tmp_dir)
        for index, worker_count in enumerate(worker_counts, start=1):
            print(
                f"[{index}/{len(worker_counts)}] running simulation "
                f"with test_runner_count={worker_count}"
            )
            final_summary = run_simulation_for_worker_count(
                simulation_argv=simulation_argv,
                worker_count=worker_count,
                output_dir=tmp_root / f"workers_{worker_count}",
            )
            add_final_summary_rows(
                results_by_localizer=results_by_localizer,
                final_summary=final_summary,
                worker_count=worker_count,
            )

    normalize_result_order(results_by_localizer)
    payload = build_sweep_payload(
        worker_counts=worker_counts,
        localizers=simulation_args.localizers,
        results_by_localizer=results_by_localizer,
    )
    sweep_output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    print(f"wrote {sweep_output_json}")

    if sweep_args.skip_plots:
        print("--skip-plots was set; skipping plot generation")
        return 0

    write_metric_plots(
        plt=plt,
        results_by_localizer=results_by_localizer,
        plots_dir=plots_dir,
    )
    return 0


def parse_args(
    argv: Sequence[str] | None,
) -> tuple[argparse.Namespace, list[str]]:
    """Parse sweep options and leave simulation.py options for simulation.parse_args."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--worker-counts",
        nargs="+",
        required=True,
        help=(
            "Test runner counts to sweep. Accepts space-separated values, "
            "comma-separated values, or both, e.g. '1 2 4' or '1,2,4'."
        ),
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help=(
            "Directory for metric plots. Defaults to <simulation --output-dir>/plots "
            f"({DEFAULT_PLOTS_DIR})."
        ),
    )
    parser.add_argument(
        "--sweep-output-json",
        type=Path,
        default=None,
        help=(
            "Path for the aggregated final-test sweep results JSON. Defaults to "
            f"<simulation --output-dir>/{DEFAULT_SWEEP_OUTPUT_JSON.name}."
        ),
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write the sweep JSON but do not generate plots.",
    )

    sweep_args, simulation_argv = parser.parse_known_args(argv)
    if simulation_argv and simulation_argv[0] == "--":
        simulation_argv = simulation_argv[1:]
    if option_was_provided(simulation_argv, "--workers"):
        parser.error("use --worker-counts for this script, not simulation.py --workers")
    return sweep_args, simulation_argv


def option_was_provided(argv: Sequence[str], option: str) -> bool:
    """Return whether an option appears in argv."""

    return any(token == option or token.startswith(f"{option}=") for token in argv)


def parse_worker_counts(raw_values: Sequence[str]) -> list[int]:
    """Parse, validate, de-duplicate, and sort worker counts."""

    worker_counts: list[int] = []
    for raw_value in raw_values:
        for token in str(raw_value).split(","):
            stripped = token.strip()
            if not stripped:
                continue
            try:
                worker_count = int(stripped)
            except ValueError as exc:
                raise ValueError(
                    f"invalid --worker-counts value {stripped!r}"
                ) from exc
            if worker_count < 1:
                raise ValueError("--worker-counts values must be at least 1")
            worker_counts.append(worker_count)

    if not worker_counts:
        raise ValueError("--worker-counts did not contain any values")
    return sorted(set(worker_counts))


def run_simulation_for_worker_count(
    *,
    simulation_argv: Sequence[str],
    worker_count: int,
    output_dir: Path,
) -> Mapping[str, Any]:
    """Run simulation.py for one worker count and return final-test summary JSON."""

    output_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        *simulation_argv,
        "--dataset",
        "all",
        "--workers",
        str(worker_count),
        "--output-dir",
        str(output_dir),
    ]
    exit_code = simulation.main(argv)
    if exit_code != 0:
        raise RuntimeError(
            f"simulation.py failed for worker_count={worker_count} "
            f"with exit code {exit_code}"
        )

    final_summary_path = output_dir / "per_bisect_results_final_test.json"
    if not final_summary_path.exists():
        raise FileNotFoundError(
            f"simulation.py did not write expected final-test summary: "
            f"{final_summary_path}"
        )
    with final_summary_path.open() as fh:
        return json.load(fh)


def add_final_summary_rows(
    *,
    results_by_localizer: dict[str, list[dict[str, Any]]],
    final_summary: Mapping[str, Any],
    worker_count: int,
) -> None:
    """Append localizer metric rows from one final-test summary."""

    runs = final_summary.get("runs")
    if not isinstance(runs, list):
        raise TypeError("final-test summary must contain a list field named 'runs'")

    seen_localizers: set[str] = set()
    for run in runs:
        if not isinstance(run, Mapping):
            raise TypeError("each final-test run must be a JSON object")

        localizer = str(run.get("localizer") or "")
        if not localizer:
            raise ValueError("final-test run is missing localizer")
        if localizer in seen_localizers:
            raise ValueError(
                f"duplicate localizer {localizer!r} for worker_count={worker_count}"
            )
        seen_localizers.add(localizer)

        raw_metrics = run.get("metrics")
        if not isinstance(raw_metrics, Mapping):
            raise TypeError(
                f"final-test run for {localizer!r} is missing metrics"
            )
        metrics = extract_main_metrics(raw_metrics, localizer=localizer)

        row: dict[str, Any] = {
            "worker_count": worker_count,
            "metrics": metrics,
        }
        optimized_parameters = run.get("optuna_optimized_parameters")
        if isinstance(optimized_parameters, Mapping):
            row["optuna_optimized_parameters"] = dict(optimized_parameters)

        results_by_localizer.setdefault(localizer, []).append(row)


def extract_main_metrics(
    raw_metrics: Mapping[str, Any],
    *,
    localizer: str,
) -> dict[str, float | int | None]:
    """Return only the main scalar metrics used by this sweep."""

    metrics: dict[str, float | int | None] = {}
    for metric_name in MAIN_METRICS:
        raw_value = raw_metrics.get(metric_name)
        if raw_value is None:
            metrics[metric_name] = None
            continue
        if not isinstance(raw_value, (int, float)):
            raise TypeError(
                f"metric {metric_name!r} for {localizer!r} must be numeric or null"
            )
        value = float(raw_value)
        if not math.isfinite(value):
            metrics[metric_name] = None
        elif isinstance(raw_value, int):
            metrics[metric_name] = int(raw_value)
        else:
            metrics[metric_name] = value
    return metrics


def normalize_result_order(
    results_by_localizer: dict[str, list[dict[str, Any]]],
) -> None:
    """Sort every localizer series by worker count."""

    for rows in results_by_localizer.values():
        rows.sort(key=lambda row: int(row["worker_count"]))


def build_sweep_payload(
    *,
    worker_counts: Sequence[int],
    localizers: Sequence[str],
    results_by_localizer: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Build the JSON payload written by this script."""

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "split": "final_test",
        "worker_counts": list(worker_counts),
        "metrics": list(MAIN_METRICS),
        "localizers": list(localizers),
        "results": {
            localizer: results_by_localizer.get(localizer, [])
            for localizer in localizers
        },
    }


def write_metric_plots(
    *,
    plt: Any,
    results_by_localizer: Mapping[str, list[dict[str, Any]]],
    plots_dir: Path,
) -> None:
    """Write one plot per main metric, with localizer series on each plot."""

    plots_dir.mkdir(parents=True, exist_ok=True)
    for metric_name in MAIN_METRICS:
        fig, ax = plt.subplots(figsize=(12, 7))
        plotted = False
        for localizer, rows in sorted(results_by_localizer.items()):
            xy_values = [
                (
                    int(row["worker_count"]),
                    row["metrics"].get(metric_name),
                )
                for row in rows
                if isinstance(row.get("metrics"), Mapping)
            ]
            xy_values = [
                (worker_count, value)
                for worker_count, value in xy_values
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            ]
            if not xy_values:
                continue

            xs = [worker_count for worker_count, _ in xy_values]
            ys = [float(value) for _, value in xy_values]
            ax.plot(xs, ys, marker="o", linewidth=2, label=localizer)
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.set_xlabel("Test runner count")
        ax.set_ylabel(METRIC_LABELS[metric_name])
        ax.set_title(f"{METRIC_TITLES[metric_name]} by Test Runner Count")
        ax.grid(True, linestyle=":", linewidth=0.7)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        fig.tight_layout()

        output_path = plots_dir / f"{metric_name}_by_worker_count.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {output_path}")


def load_matplotlib() -> Any:
    """Import matplotlib before running the expensive sweep."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plot generation. Install matplotlib "
            "or rerun with --skip-plots."
        ) from exc
    return plt


if __name__ == "__main__":
    raise SystemExit(main())
