"""Run culprit-localization simulations on Mozilla performance regressions."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

try:
    from .localization import LOCALIZERS, CulpritLocalizer
    from .test_oracle import (
        RevisionPerfIndex,
        RevisionRecord,
        SignatureInfo,
        SignatureInfoIndex,
        SummaryComparison,
        TestExecutor,
        TestOracle,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from localization import LOCALIZERS, CulpritLocalizer
    from test_oracle import (
        RevisionPerfIndex,
        RevisionRecord,
        SignatureInfo,
        SignatureInfoIndex,
        SummaryComparison,
        TestExecutor,
        TestOracle,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BISECT_DATA_DIR = REPO_ROOT / "datasets" / "mozilla_perf_bisect"
PROMPT_REGRESSION_DATA_DIR = REPO_ROOT / "datasets" / "mozilla_perf"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "perf_bisect" / "results"
DEFAULT_ORACLE_METRICS = (
    REPO_ROOT / "analysis" / "perf_bisect" / "per_regression_oracle_metrics.jsonl"
)

DATASETS = {
    "eval": "perf_bisect_regressions_eval.jsonl",
    "final_test": "perf_bisect_regressions_final_test.jsonl",
}
DEFAULT_BACKFILL_RETRIGGER_COUNT_MIN = 0
DEFAULT_BACKFILL_RETRIGGER_COUNT_MAX = 5
OBJECTIVE_FAILURE_PENALTY = 1_000_000_000.0


@dataclass(frozen=True)
class SimulationParameters:
    """Tunable algorithm parameters used by one simulation run."""

    backfill_retrigger_count: int

    def to_json(self) -> dict[str, Any]:
        """Serialize parameter values into JSON-compatible primitives."""

        return {
            "backfill_retrigger_count": self.backfill_retrigger_count,
        }


@dataclass(frozen=True)
class OracleMetrics:
    """Per-regression noisy-oracle parameters."""

    regression_id: int
    summary_oracle_accuracy: float


@dataclass(frozen=True)
class OracleSpec:
    """Factory descriptor for a registered test oracle implementation."""

    name: str

    def build(
        self,
        *,
        signature_info: SignatureInfoIndex,
        oracle_metrics: OracleMetrics,
        workers: int,
        random_seed: int | None,
    ) -> TestOracle:
        """Construct the concrete oracle for one independent regression run."""

        if self.name == SummaryComparison.name:
            return SummaryComparison(
                signature_info=signature_info,
                executor=TestExecutor(workers=workers),
                oracle_accuracy=oracle_metrics.summary_oracle_accuracy,
                random_seed=random_seed,
            )
        raise ValueError(f"unknown oracle: {self.name}")


ORACLES: dict[str, OracleSpec] = {
    SummaryComparison.name: OracleSpec(SummaryComparison.name),
}


def main(argv: Sequence[str] | None = None) -> int:
    """Load inputs, run requested simulations, and write summary/detail outputs."""

    args = parse_args(argv)
    dataset_names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    default_parameters = SimulationParameters(
        backfill_retrigger_count=args.backfill_retrigger_count,
    )

    signature_info = load_signature_info(args.signature_info)
    revision_perf = load_revision_perf(args.revision_data)
    oracle_metrics = load_oracle_metrics(args.oracle_metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    required_dataset_names = set(dataset_names)
    if args.optuna_trials > 0:
        required_dataset_names.add("eval")

    regressions_by_dataset: dict[str, list[dict[str, Any]]] = {}
    regressions_path_by_dataset: dict[str, Path] = {}
    for dataset_name in sorted(required_dataset_names):
        regressions_path = resolve_regression_path(
            args.regression_dir,
            DATASETS[dataset_name],
        )
        regressions_path_by_dataset[dataset_name] = regressions_path
        regressions_by_dataset[dataset_name] = load_jsonl(regressions_path)

    combo_parameters: dict[tuple[str, str], SimulationParameters] | None = None
    combo_optuna: dict[tuple[str, str], dict[str, Any]] | None = None
    optimization_settings = None
    if args.optuna_trials > 0:
        optimization_settings = {
            "enabled": True,
            "tuning_dataset": "eval",
            "optuna_trials": args.optuna_trials,
            "optuna_seed": args.optuna_seed,
            "backfill_retrigger_count_min": args.backfill_retrigger_count_min,
            "backfill_retrigger_count_max": args.backfill_retrigger_count_max,
            "objectives": [
                {"metric": "mean_trtc_hours", "direction": "minimize"},
                {"metric": "mean_test_runs", "direction": "minimize"},
                {"metric": "success_rate_percent", "direction": "maximize"},
            ],
            "selection": (
                "highest success_rate_percent on the Pareto frontier; ties "
                "minimize mean_test_runs, then mean_trtc_hours"
            ),
        }
        combo_parameters, combo_optuna = optimize_parameters_on_eval(
            regressions=regressions_by_dataset["eval"],
            signature_info=signature_info,
            revision_perf=revision_perf,
            oracle_metrics=oracle_metrics,
            oracle_names=args.oracles,
            localizer_names=args.localizers,
            workers=args.workers,
            default_parameters=default_parameters,
            backfill_retrigger_count_min=args.backfill_retrigger_count_min,
            backfill_retrigger_count_max=args.backfill_retrigger_count_max,
            optuna_trials=args.optuna_trials,
            optuna_seed=args.optuna_seed,
            random_seed=args.random_seed,
        )

    for dataset_name in dataset_names:
        summary_output, details_output = run_dataset(
            dataset_name=dataset_name,
            regressions=regressions_by_dataset[dataset_name],
            regressions_path=regressions_path_by_dataset[dataset_name],
            signature_info_path=args.signature_info,
            revision_data_path=args.revision_data,
            oracle_metrics_path=args.oracle_metrics,
            signature_info=signature_info,
            revision_perf=revision_perf,
            oracle_metrics=oracle_metrics,
            oracle_names=args.oracles,
            localizer_names=args.localizers,
            workers=args.workers,
            default_parameters=default_parameters,
            combo_parameters=combo_parameters,
            combo_optuna=combo_optuna,
            optimization_settings=optimization_settings,
            random_seed=args.random_seed,
        )
        output_path = args.output_dir / f"per_bisect_results_{dataset_name}.json"
        details_output_path = (
            args.output_dir / f"per_bisect_results_{dataset_name}_details.json"
        )
        output_path.write_text(json.dumps(summary_output, indent=2, sort_keys=True) + "\n")
        details_output_path.write_text(
            json.dumps(details_output, indent=2, sort_keys=True) + "\n"
        )
        print(f"wrote {output_path}")
        print(f"wrote {details_output_path}")

    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for dataset, executor, oracle, and localizer setup."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["all", *DATASETS.keys()],
        default="all",
        help="Regression split to simulate.",
    )
    parser.add_argument(
        "--regression-dir",
        type=Path,
        default=DEFAULT_BISECT_DATA_DIR,
        help="Directory containing perf_bisect_regressions_*.jsonl.",
    )
    parser.add_argument(
        "--signature-info",
        type=Path,
        default=DEFAULT_BISECT_DATA_DIR / "per_sig_perf_data_info.jsonl",
        help="Path to per_sig_perf_data_info.jsonl.",
    )
    parser.add_argument(
        "--revision-data",
        type=Path,
        default=DEFAULT_BISECT_DATA_DIR / "per_revision_perf_data.jsonl",
        help=(
            "Path to per_revision_perf_data.jsonl; only revision graph fields "
            "are loaded."
        ),
    )
    parser.add_argument(
        "--oracle-metrics",
        type=Path,
        default=DEFAULT_ORACLE_METRICS,
        help=(
            "Path to per_regression_oracle_metrics.jsonl. SummaryComparison "
            "uses summary_oracle_accuracy as its noisy-oracle accuracy."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for per_bisect_results_*.json outputs. Defaults to "
            "analysis/perf_bisect/results under the repo root."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of simulated test executor workers.",
    )
    parser.add_argument(
        "--oracles",
        nargs="+",
        choices=sorted(ORACLES),
        default=sorted(ORACLES),
        help="Test oracle implementations to run.",
    )
    parser.add_argument(
        "--localizers",
        nargs="+",
        choices=sorted(LOCALIZERS),
        default=sorted(LOCALIZERS),
        help="Culprit localization algorithms to run.",
    )
    parser.add_argument(
        "--backfill-retrigger-count",
        type=int,
        default=2,
        help=(
            "Number of suspicious Backfill decision sets to retrigger before "
            "leaving the localization undefined. Suspicious sets include "
            "adjacent non-monotonic intervals and all-clean sequences. Used "
            "directly when Optuna is disabled, and as a fallback default for "
            "combos without tuned parameters."
        ),
    )
    parser.add_argument(
        "--optuna-trials",
        "--mopt-trials",
        dest="optuna_trials",
        type=int,
        default=0,
        help=(
            "Number of Optuna trials per localizer/oracle combo. When positive, "
            "parameters are optimized on the eval split and replayed on the "
            "requested split(s)."
        ),
    )
    parser.add_argument(
        "--optuna-seed",
        type=int,
        default=None,
        help=(
            "Seed for Optuna samplers, used to make optimization reproducible. "
            "Defaults to --random-seed."
        ),
    )
    parser.add_argument(
        "--backfill-retrigger-count-min",
        type=int,
        default=DEFAULT_BACKFILL_RETRIGGER_COUNT_MIN,
        help="Minimum Backfill retrigger count sampled by Optuna.",
    )
    parser.add_argument(
        "--backfill-retrigger-count-max",
        type=int,
        default=DEFAULT_BACKFILL_RETRIGGER_COUNT_MAX,
        help="Maximum Backfill retrigger count sampled by Optuna.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Seed for noisy oracle verdict draws.",
    )
    args = parser.parse_args(argv)
    if args.backfill_retrigger_count < 0:
        parser.error("--backfill-retrigger-count must be non-negative")
    if args.optuna_trials < 0:
        parser.error("--optuna-trials must be non-negative")
    if args.backfill_retrigger_count_min < 0:
        parser.error("--backfill-retrigger-count-min must be non-negative")
    if args.backfill_retrigger_count_max < args.backfill_retrigger_count_min:
        parser.error(
            "--backfill-retrigger-count-max must be greater than or equal to "
            "--backfill-retrigger-count-min"
        )
    if args.optuna_seed is None:
        args.optuna_seed = args.random_seed
    elif args.optuna_trials > 0 and args.optuna_seed != args.random_seed:
        parser.error("--optuna-seed must match --random-seed when Optuna is enabled")
    return args


def resolve_regression_path(regression_dir: Path, filename: str) -> Path:
    """Resolve a regression split path, including the prompt-described fallback dir."""

    path = regression_dir / filename
    if path.exists():
        return path

    prompt_path = PROMPT_REGRESSION_DATA_DIR / filename
    if prompt_path.exists():
        return prompt_path

    raise FileNotFoundError(
        f"could not find {filename} in {regression_dir} or {PROMPT_REGRESSION_DATA_DIR}"
    )


def run_dataset(
    *,
    dataset_name: str,
    regressions: Sequence[Mapping[str, Any]],
    regressions_path: Path,
    signature_info_path: Path,
    revision_data_path: Path,
    oracle_metrics_path: Path,
    signature_info: SignatureInfoIndex,
    revision_perf: RevisionPerfIndex,
    oracle_metrics: Mapping[int, OracleMetrics],
    oracle_names: Sequence[str],
    localizer_names: Sequence[str],
    workers: int,
    default_parameters: SimulationParameters,
    combo_parameters: Mapping[tuple[str, str], SimulationParameters] | None = None,
    combo_optuna: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    optimization_settings: Mapping[str, Any] | None = None,
    random_seed: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run every requested localizer/oracle combination for one regression split."""

    summary_runs = []
    detail_runs = []

    for localizer_name in localizer_names:
        for oracle_name in oracle_names:
            combo_key = (localizer_name, oracle_name)
            parameters = (
                combo_parameters.get(combo_key, default_parameters)
                if combo_parameters is not None
                else default_parameters
            )
            results, metrics = run_combo(
                regressions=regressions,
                localizer_name=localizer_name,
                oracle_name=oracle_name,
                signature_info=signature_info,
                revision_perf=revision_perf,
                oracle_metrics=oracle_metrics,
                workers=workers,
                parameters=parameters,
                random_seed=random_seed,
            )
            summary_run = {
                "localizer": localizer_name,
                "test_oracle": oracle_name,
                "parameters": parameters.to_json(),
                "metrics": metrics,
            }
            if combo_optuna is not None and combo_key in combo_optuna:
                summary_run["optuna"] = dict(combo_optuna[combo_key])
            summary_runs.append(summary_run)
            print_undefined_localizations(
                dataset_name=dataset_name,
                localizer_name=localizer_name,
                oracle_name=oracle_name,
                results=results,
            )
            detail_run = {
                "localizer": localizer_name,
                "test_oracle": oracle_name,
                "parameters": parameters.to_json(),
                "results": [result.to_json() for result in results],
            }
            if combo_optuna is not None and combo_key in combo_optuna:
                detail_run["optuna"] = dict(combo_optuna[combo_key])
            detail_runs.append(detail_run)

    base_output = {
        "dataset": dataset_name,
        "generated_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "regressions": str(regressions_path),
            "signature_info": str(signature_info_path),
            "revision_data": str(revision_data_path),
            "oracle_metrics": str(oracle_metrics_path),
        },
        "settings": {
            "workers": workers,
            "job_duration_source": "per-signature job_duration minutes",
            "default_parameters": default_parameters.to_json(),
            "backfill_retrigger_count": default_parameters.backfill_retrigger_count,
            "random_seed": random_seed,
            "random_seed_derivation": "random_seed + regression_id - 1",
        },
    }
    if optimization_settings is not None:
        base_output["settings"]["optimization"] = dict(optimization_settings)
    return (
        {**base_output, "runs": summary_runs},
        {**base_output, "runs": detail_runs},
    )


def run_combo(
    *,
    regressions: Sequence[Mapping[str, Any]],
    localizer_name: str,
    oracle_name: str,
    signature_info: SignatureInfoIndex,
    revision_perf: RevisionPerfIndex,
    oracle_metrics: Mapping[int, OracleMetrics],
    workers: int,
    parameters: SimulationParameters,
    random_seed: int | None,
) -> tuple[list[Any], dict[str, Any]]:
    """Run one localizer/oracle pair and return per-regression and summary metrics."""

    localizer = build_localizer(
        localizer_name,
        parameters=parameters,
    )
    results = [
        run_one_regression(
            regression_index=regression_index,
            regression=regression,
            localizer=localizer,
            oracle_spec=ORACLES[oracle_name],
            signature_info=signature_info,
            revision_perf=revision_perf,
            oracle_metrics=oracle_metrics,
            workers=workers,
            random_seed=random_seed,
        )
        for regression_index, regression in enumerate(regressions)
    ]
    return results, compute_metrics(results)


def build_localizer(
    localizer_name: str,
    *,
    parameters: SimulationParameters,
) -> CulpritLocalizer:
    """Construct a localizer from the registry."""

    localizer_cls = LOCALIZERS[localizer_name]
    if localizer_name == "Backfill":
        return localizer_cls(
            backfill_retrigger_count=parameters.backfill_retrigger_count,
        )
    return localizer_cls()


def optimize_parameters_on_eval(
    *,
    regressions: Sequence[Mapping[str, Any]],
    signature_info: SignatureInfoIndex,
    revision_perf: RevisionPerfIndex,
    oracle_metrics: Mapping[int, OracleMetrics],
    oracle_names: Sequence[str],
    localizer_names: Sequence[str],
    workers: int,
    default_parameters: SimulationParameters,
    backfill_retrigger_count_min: int,
    backfill_retrigger_count_max: int,
    optuna_trials: int,
    optuna_seed: int,
    random_seed: int | None,
) -> tuple[dict[tuple[str, str], SimulationParameters], dict[tuple[str, str], dict[str, Any]]]:
    """Tune algorithm parameters on the eval split for every selected combo."""

    combo_parameters: dict[tuple[str, str], SimulationParameters] = {}
    combo_optuna: dict[tuple[str, str], dict[str, Any]] = {}

    for localizer_name in localizer_names:
        for oracle_name in oracle_names:
            combo_key = (localizer_name, oracle_name)
            parameters, optuna_meta = optimize_combo_on_eval(
                regressions=regressions,
                localizer_name=localizer_name,
                oracle_name=oracle_name,
                signature_info=signature_info,
                revision_perf=revision_perf,
                oracle_metrics=oracle_metrics,
                workers=workers,
                default_parameters=default_parameters,
                backfill_retrigger_count_min=backfill_retrigger_count_min,
                backfill_retrigger_count_max=backfill_retrigger_count_max,
                optuna_trials=optuna_trials,
                optuna_seed=optuna_seed,
                random_seed=random_seed,
            )
            combo_parameters[combo_key] = parameters
            combo_optuna[combo_key] = optuna_meta
            print(
                "optuna selected "
                f"{localizer_name}/{oracle_name}: {parameters.to_json()}"
            )

    return combo_parameters, combo_optuna


def optimize_combo_on_eval(
    *,
    regressions: Sequence[Mapping[str, Any]],
    localizer_name: str,
    oracle_name: str,
    signature_info: SignatureInfoIndex,
    revision_perf: RevisionPerfIndex,
    oracle_metrics: Mapping[int, OracleMetrics],
    workers: int,
    default_parameters: SimulationParameters,
    backfill_retrigger_count_min: int,
    backfill_retrigger_count_max: int,
    optuna_trials: int,
    optuna_seed: int,
    random_seed: int | None,
) -> tuple[SimulationParameters, dict[str, Any]]:
    """Run one multi-objective Optuna study on eval for one localizer/oracle pair."""

    if not has_tunable_parameters(localizer_name=localizer_name, oracle_name=oracle_name):
        return (
            default_parameters,
            {
                "skipped": True,
                "reason": "no_tunable_parameters",
                "n_trials": 0,
                "seed": optuna_seed,
            },
        )

    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna is required. Install with `pip install optuna`.") from exc

    try:
        sampler = optuna.samplers.NSGAIISampler(seed=int(optuna_seed))
    except AttributeError:
        sampler = optuna.samplers.RandomSampler(seed=int(optuna_seed))

    study = optuna.create_study(
        directions=["minimize", "minimize", "maximize"],
        sampler=sampler,
    )

    def objective(trial: Any) -> tuple[float, float, float]:
        parameters = suggest_parameters(
            trial,
            localizer_name=localizer_name,
            oracle_name=oracle_name,
            default_parameters=default_parameters,
            backfill_retrigger_count_min=backfill_retrigger_count_min,
            backfill_retrigger_count_max=backfill_retrigger_count_max,
        )
        _, metrics = run_combo(
            regressions=regressions,
            localizer_name=localizer_name,
            oracle_name=oracle_name,
            signature_info=signature_info,
            revision_perf=revision_perf,
            oracle_metrics=oracle_metrics,
            workers=workers,
            parameters=parameters,
            random_seed=random_seed,
        )
        trial.set_user_attr("parameters", parameters.to_json())
        trial.set_user_attr("metrics", metrics)
        return objective_values(metrics)

    study.optimize(objective, n_trials=int(optuna_trials), show_progress_bar=False)
    if not study.best_trials:
        raise RuntimeError(
            f"Optuna produced no Pareto-optimal trials for {localizer_name}/{oracle_name}."
        )

    selected_trial = select_pareto_trial(study.best_trials)
    selected_parameters = parameters_from_trial(selected_trial)
    optuna_meta = {
        "skipped": False,
        "n_trials": int(optuna_trials),
        "seed": int(optuna_seed),
        "sampler": sampler.__class__.__name__,
        "directions": ["minimize", "minimize", "maximize"],
        "objectives": [
            "mean_trtc_hours",
            "mean_test_runs",
            "success_rate_percent",
        ],
        "selection": (
            "highest success_rate_percent on the Pareto frontier; ties minimize "
            "mean_test_runs, then mean_trtc_hours"
        ),
        "pareto_front_trial_count": len(study.best_trials),
        "selected_trial_number": int(selected_trial.number),
        "selected_values": objective_values_to_json(selected_trial.values),
        "selected_metrics": selected_trial.user_attrs["metrics"],
        "selected_parameters": selected_parameters.to_json(),
        "best_trial_params_raw": dict(selected_trial.params),
        "pareto_front": [
            trial_to_pareto_json(trial)
            for trial in sorted(study.best_trials, key=lambda item: item.number)
        ],
    }
    return selected_parameters, optuna_meta


def has_tunable_parameters(*, localizer_name: str, oracle_name: str) -> bool:
    """Return whether the selected combo currently exposes Optuna parameters."""

    del oracle_name
    return localizer_name == "Backfill"


def suggest_parameters(
    trial: Any,
    *,
    localizer_name: str,
    oracle_name: str,
    default_parameters: SimulationParameters,
    backfill_retrigger_count_min: int,
    backfill_retrigger_count_max: int,
) -> SimulationParameters:
    """Ask Optuna for one parameter set for the selected algorithm combo."""

    del oracle_name
    backfill_retrigger_count = default_parameters.backfill_retrigger_count
    if localizer_name == "Backfill":
        backfill_retrigger_count = trial.suggest_int(
            "Backfill_backfill_retrigger_count",
            int(backfill_retrigger_count_min),
            int(backfill_retrigger_count_max),
        )
    return SimulationParameters(
        backfill_retrigger_count=int(backfill_retrigger_count),
    )


def objective_values(metrics: Mapping[str, Any]) -> tuple[float, float, float]:
    """Return Optuna objective values from aggregate simulation metrics."""

    return (
        finite_objective_value(metrics.get("mean_trtc_hours")),
        finite_objective_value(metrics.get("mean_test_runs")),
        float(metrics.get("success_rate_percent") or 0.0),
    )


def finite_objective_value(value: Any) -> float:
    """Convert missing or non-finite objective inputs into a finite penalty."""

    if value is None:
        return OBJECTIVE_FAILURE_PENALTY
    objective_value = float(value)
    if not math.isfinite(objective_value):
        return OBJECTIVE_FAILURE_PENALTY
    return objective_value


def select_pareto_trial(trials: Sequence[Any]) -> Any:
    """Choose the best Pareto trial by success rate, then cost tie-breakers."""

    return max(
        trials,
        key=lambda trial: (
            float(trial.values[2]),
            -float(trial.values[1]),
            -float(trial.values[0]),
            -int(trial.number),
        ),
    )


def parameters_from_trial(trial: Any) -> SimulationParameters:
    """Read serialized simulation parameters from an Optuna trial."""

    raw_parameters = trial.user_attrs["parameters"]
    return SimulationParameters(
        backfill_retrigger_count=int(raw_parameters["backfill_retrigger_count"]),
    )


def objective_values_to_json(values: Sequence[float] | None) -> dict[str, float] | None:
    """Serialize Optuna objective values with metric names."""

    if values is None:
        return None
    return {
        "mean_trtc_hours": float(values[0]),
        "mean_test_runs": float(values[1]),
        "success_rate_percent": float(values[2]),
    }


def trial_to_pareto_json(trial: Any) -> dict[str, Any]:
    """Serialize one Pareto-front Optuna trial."""

    return {
        "trial_number": int(trial.number),
        "values": objective_values_to_json(trial.values),
        "params": dict(trial.params),
        "parameters": trial.user_attrs.get("parameters"),
        "metrics": trial.user_attrs.get("metrics"),
    }


def run_one_regression(
    *,
    regression_index: int,
    regression: Mapping[str, Any],
    localizer: CulpritLocalizer,
    oracle_spec: OracleSpec,
    signature_info: SignatureInfoIndex,
    revision_perf: RevisionPerfIndex,
    oracle_metrics: Mapping[int, OracleMetrics],
    workers: int,
    random_seed: int | None,
):
    """Run one regression with a fresh oracle and test executor."""

    regression_id = require_regression_id(
        regression,
        context=f"split row index {regression_index}",
    )
    regression_seed = (
        None
        if random_seed is None
        else random_seed + regression_id - 1
    )
    metrics = oracle_metrics.get(regression_id)
    if metrics is None:
        raise ValueError(f"missing oracle metrics for regression_id {regression_id}")

    oracle = oracle_spec.build(
        signature_info=signature_info,
        oracle_metrics=metrics,
        workers=workers,
        random_seed=regression_seed,
    )
    return localizer.localize(
        regression,
        revision_perf=revision_perf,
        oracle=oracle,
    )


def compute_metrics(results: Sequence[Any]) -> dict[str, Any]:
    """Compute aggregate success, TRTC, and test-run metrics for one run."""

    total = len(results)
    successes = [result for result in results if result.success]
    trtc_values = [
        result.trtc_hours
        for result in successes
        if result.trtc_hours is not None
    ]
    test_runs = [result.test_runs for result in results]
    undefined_causes = Counter(
        result.undefined_reason or "unknown"
        for result in results
        if not result.success
    )

    return {
        "total_regressions": total,
        "successful_localizations": len(successes),
        "undefined_localizations": total - len(successes),
        "undefined_causes": dict(sorted(undefined_causes.items())),
        "success_rate_percent": (
            round(len(successes) / total * 100.0, 1) if total else 0.0
        ),
        "mean_trtc_hours": round(mean(trtc_values), 2) if trtc_values else None,
        "max_trtc_hours": round(max(trtc_values), 2) if trtc_values else None,
        "mean_test_runs": round(mean(test_runs), 1) if test_runs else None,
        "max_test_runs": max(test_runs) if test_runs else None,
    }


def print_undefined_localizations(
    *,
    dataset_name: str,
    localizer_name: str,
    oracle_name: str,
    results: Sequence[Any],
) -> None:
    """Print undefined localization regression IDs grouped by undefined cause."""

    undefined_by_cause: dict[str, list[int | None]] = {}
    for result in results:
        if result.success:
            continue
        cause = result.undefined_reason or "unknown"
        undefined_by_cause.setdefault(cause, []).append(result.regression_id)

    run_label = (
        f"{dataset_name} / localizer={localizer_name} / test_oracle={oracle_name}"
    )
    if not undefined_by_cause:
        print(f"{run_label}: no undefined localizations")
        return

    print(f"{run_label}: undefined localizations by cause")
    for cause, regression_ids in sorted(undefined_by_cause.items()):
        print(f"  {cause} ({len(regression_ids)})")
        for regression_id in sorted(
            regression_ids,
            key=lambda item: (item is None, item if item is not None else 0),
        ):
            formatted_id = regression_id if regression_id is not None else "unknown"
            print(f"    regression_id={formatted_id}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load newline-delimited JSON records from disk."""

    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_oracle_metrics(path: Path) -> dict[int, OracleMetrics]:
    """Load per-regression summary oracle accuracies."""

    metrics_by_regression_id: dict[int, OracleMetrics] = {}
    for raw in load_jsonl(path):
        regression_id = int(raw["regression_id"])
        if regression_id < 1:
            raise ValueError(
                f"oracle metrics regression_id must be positive: {regression_id!r}"
            )
        if regression_id in metrics_by_regression_id:
            raise ValueError(
                f"duplicate oracle metrics for regression_id {regression_id}"
            )

        summary_accuracy = parse_oracle_accuracy(
            raw.get("summary_oracle_accuracy"),
            context=f"regression_id {regression_id} summary_oracle_accuracy",
        )
        metrics_by_regression_id[regression_id] = OracleMetrics(
            regression_id=regression_id,
            summary_oracle_accuracy=summary_accuracy,
        )

    return metrics_by_regression_id


def parse_oracle_accuracy(value: Any, *, context: str) -> float:
    """Parse a required probability used by the noisy oracle."""

    if value is None:
        raise ValueError(f"missing oracle accuracy for {context}")
    try:
        accuracy = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid oracle accuracy for {context}: {value!r}") from exc
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError(
            f"oracle accuracy must be between 0 and 1 for {context}: {value!r}"
        )
    return accuracy


def require_regression_id(
    regression: Mapping[str, Any],
    *,
    context: str,
) -> int:
    """Parse the required positive regression_id from one regression row."""

    raw_id = regression.get("regression_id")
    if raw_id is None:
        raise ValueError(f"missing regression_id at {context}")
    try:
        regression_id = int(raw_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid regression_id at {context}: {raw_id!r}") from exc
    if regression_id < 1:
        raise ValueError(f"regression_id must be positive at {context}: {raw_id!r}")
    return regression_id


def load_signature_info(path: Path) -> SignatureInfoIndex:
    """Load per-signature job durations."""

    infos = []
    for raw in load_jsonl(path):
        raw_job_duration = raw.get("job_duration")
        if raw_job_duration is None:
            raise ValueError(
                "missing job_duration for "
                f"signature_id={raw.get('signature_id')!r}"
            )
        job_duration_minutes = float(raw_job_duration)
        if job_duration_minutes <= 0.0:
            raise ValueError(
                "job_duration must be positive for "
                f"signature_id={raw.get('signature_id')!r}: {raw_job_duration!r}"
            )
        infos.append(
            SignatureInfo(
                signature_id=int(raw["signature_id"]),
                job_duration_minutes=job_duration_minutes,
                platform=raw.get("platform"),
            )
        )
    return SignatureInfoIndex(infos)


def load_revision_perf(path: Path) -> RevisionPerfIndex:
    """Load revision graph nodes needed for good-to-bad paths."""

    records = []
    for raw in load_jsonl(path):
        records.append(
            RevisionRecord(
                node=str(raw["node"]),
                parents=[str(parent) for parent in raw.get("parents", [])],
                date=raw.get("date"),
            )
        )
    return RevisionPerfIndex(records)


if __name__ == "__main__":
    raise SystemExit(main())
