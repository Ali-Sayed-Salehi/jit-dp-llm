"""Run culprit-localization simulations on Mozilla performance regressions."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
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
DEFAULT_DIST_PLOTS_DIR = DEFAULT_OUTPUT_DIR / "dist_plots"
DEFAULT_ORACLE_METRICS = (
    REPO_ROOT / "analysis" / "perf_bisect" / "per_regression_oracle_metrics.jsonl"
)
DEFAULT_RISK_SCORES = DEFAULT_BISECT_DATA_DIR / "per_commit_risk_scores.jsonl"

DATASETS = {
    "eval": "perf_bisect_regressions_eval.jsonl",
    "final_test": "perf_bisect_regressions_final_test.jsonl",
}
DEFAULT_BACKFILL_RETRIGGER_COUNT_MIN = 0
DEFAULT_BACKFILL_RETRIGGER_COUNT_MAX = 5
DEFAULT_PROBE_REPEAT_COUNT_MIN = 1
DEFAULT_PROBE_REPEAT_COUNT_MAX = 5
DEFAULT_MIDPOINT_RETRIGGER_COUNT_MIN = 0
DEFAULT_MIDPOINT_RETRIGGER_COUNT_MAX = 5
DEFAULT_MULTISECTION_SECTION_COUNT = 4
DEFAULT_MULTISECTION_SECTION_COUNT_MIN = 2
DEFAULT_MULTISECTION_SECTION_COUNT_MAX = 16
DEFAULT_MULTISECTION_RETRIGGER_COUNT = 0
DEFAULT_MULTISECTION_RETRIGGER_COUNT_MIN = 0
DEFAULT_MULTISECTION_RETRIGGER_COUNT_MAX = 5
DEFAULT_PBA_CONFIDENCE_THRESHOLD = 0.9
DEFAULT_PBA_CONFIDENCE_THRESHOLD_MIN = 0.6
DEFAULT_PBA_CONFIDENCE_THRESHOLD_MAX = 0.99
DEFAULT_PBA_REPEAT_COUNT = 1
DEFAULT_PBA_REPEAT_COUNT_MIN = 1
DEFAULT_PBA_REPEAT_COUNT_MAX = 5
DEFAULT_PBA_MAX_TEST_RUNS = 100
DEFAULT_PBA_MAX_TEST_RUNS_MIN = 1
DEFAULT_PBA_MAX_TEST_RUNS_MAX = 200
DEFAULT_PBA_RISK_PRIOR_UNIFORM_WEIGHT = 0.05
DEFAULT_PBA_RISK_PRIOR_UNIFORM_WEIGHT_MIN = 0.0
DEFAULT_PBA_RISK_PRIOR_UNIFORM_WEIGHT_MAX = 0.5
OBJECTIVE_FAILURE_PENALTY = 1_000_000_000.0
TUNABLE_PARAMETER_FIELDS_BY_LOCALIZER = {
    "Backfill": ("backfill_retrigger_count",),
    "BackfillWithRepeat": ("backfill_retrigger_count", "probe_repeat_count"),
    "ProbabilisticBisection_PosteriorMedian_RiskAwarePrior": (
        "pba_confidence_threshold",
        "pba_repeat_count",
        "pba_max_test_runs",
        "pba_risk_prior_uniform_weight",
    ),
    "ProbabilisticBisection_PosteriorMedian_UniformPrior": (
        "pba_confidence_threshold",
        "pba_repeat_count",
        "pba_max_test_runs",
    ),
    "ProbabilisticMultiSection_PosteriorQuantile_UniformPrior": (
        "multisection_section_count",
        "pba_confidence_threshold",
        "pba_repeat_count",
        "pba_max_test_runs",
    ),
    "RiskWeightedBisection": ("midpoint_retrigger_count",),
    "RiskWeightedMultisection": (
        "multisection_section_count",
        "multisection_retrigger_count",
    ),
    "StandardMidpointBisection": ("midpoint_retrigger_count",),
    "StandardMidpointMultisection": (
        "multisection_section_count",
        "multisection_retrigger_count",
    ),
}
BEST_COMBO_METRIC_SPECS = (
    {
        "field": "best_combo_by_success_rate",
        "metric": "success_rate_percent",
        "direction": "maximize",
        "vote_weight": 4,
    },
    {
        "field": "best_combo_by_mean_elapsed",
        "metric": "mean_elapsed_hours",
        "direction": "minimize",
        "vote_weight": 1,
    },
    {
        "field": "best_combo_by_mean_test_runs",
        "metric": "mean_test_runs",
        "direction": "minimize",
        "vote_weight": 1,
    },
    {
        "field": "best_combo_by_max_elapsed",
        "metric": "max_elapsed_hours",
        "direction": "minimize",
        "vote_weight": 1,
    },
    {
        "field": "best_combo_by_max_test_runs",
        "metric": "max_test_runs",
        "direction": "minimize",
        "vote_weight": 1,
    },
)


@dataclass(frozen=True)
class SimulationParameters:
    """Tunable algorithm parameters used by one simulation run."""

    backfill_retrigger_count: int
    probe_repeat_count: int
    midpoint_retrigger_count: int
    multisection_section_count: int
    multisection_retrigger_count: int
    pba_confidence_threshold: float
    pba_repeat_count: int
    pba_max_test_runs: int
    pba_risk_prior_uniform_weight: float

    def to_json(self) -> dict[str, Any]:
        """Serialize parameter values into JSON-compatible primitives."""

        return {
            "backfill_retrigger_count": self.backfill_retrigger_count,
            "probe_repeat_count": self.probe_repeat_count,
            "midpoint_retrigger_count": self.midpoint_retrigger_count,
            "multisection_section_count": self.multisection_section_count,
            "multisection_retrigger_count": self.multisection_retrigger_count,
            "pba_confidence_threshold": self.pba_confidence_threshold,
            "pba_repeat_count": self.pba_repeat_count,
            "pba_max_test_runs": self.pba_max_test_runs,
            "pba_risk_prior_uniform_weight": (
                self.pba_risk_prior_uniform_weight
            ),
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
    plt = load_matplotlib() if args.draw_dist_plots else None
    dataset_names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    default_parameters = SimulationParameters(
        backfill_retrigger_count=args.backfill_retrigger_count,
        probe_repeat_count=args.probe_repeat_count,
        midpoint_retrigger_count=args.midpoint_retrigger_count,
        multisection_section_count=args.multisection_section_count,
        multisection_retrigger_count=args.multisection_retrigger_count,
        pba_confidence_threshold=args.pba_confidence_threshold,
        pba_repeat_count=args.pba_repeat_count,
        pba_max_test_runs=args.pba_max_test_runs,
        pba_risk_prior_uniform_weight=args.pba_risk_prior_uniform_weight,
    )

    signature_info = load_signature_info(args.signature_info)
    revision_perf = load_revision_perf(args.revision_data)
    oracle_metrics = load_oracle_metrics(args.oracle_metrics)
    risk_scores_required = uses_risk_scores(args.localizers)
    risk_scores = load_risk_scores(args.risk_scores) if risk_scores_required else {}
    risk_scores_path = args.risk_scores if risk_scores_required else None
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
    optimization_settings = None
    if args.optuna_trials > 0:
        optimization_settings = {
            "enabled": True,
            "tuning_dataset": "eval",
            "optuna_trials": args.optuna_trials,
            "optuna_trials_scale_by_parameter_count": True,
            "optuna_seed": args.optuna_seed,
            "backfill_retrigger_count_min": args.backfill_retrigger_count_min,
            "backfill_retrigger_count_max": args.backfill_retrigger_count_max,
            "probe_repeat_count_min": args.probe_repeat_count_min,
            "probe_repeat_count_max": args.probe_repeat_count_max,
            "midpoint_retrigger_count_min": args.midpoint_retrigger_count_min,
            "midpoint_retrigger_count_max": args.midpoint_retrigger_count_max,
            "multisection_section_count_min": args.multisection_section_count_min,
            "multisection_section_count_max": args.multisection_section_count_max,
            "multisection_retrigger_count_min": (
                args.multisection_retrigger_count_min
            ),
            "multisection_retrigger_count_max": (
                args.multisection_retrigger_count_max
            ),
            "pba_confidence_threshold_min": args.pba_confidence_threshold_min,
            "pba_confidence_threshold_max": args.pba_confidence_threshold_max,
            "pba_repeat_count_min": args.pba_repeat_count_min,
            "pba_repeat_count_max": args.pba_repeat_count_max,
            "pba_max_test_runs_min": args.pba_max_test_runs_min,
            "pba_max_test_runs_max": args.pba_max_test_runs_max,
            "pba_risk_prior_uniform_weight_min": (
                args.pba_risk_prior_uniform_weight_min
            ),
            "pba_risk_prior_uniform_weight_max": (
                args.pba_risk_prior_uniform_weight_max
            ),
            "objectives": [
                {"metric": "mean_elapsed_hours", "direction": "minimize"},
                {"metric": "mean_test_runs", "direction": "minimize"},
                {"metric": "success_rate_percent", "direction": "maximize"},
            ],
            "selection": (
                "highest success_rate_percent on the Pareto frontier; ties "
                "minimize mean_test_runs, then mean_elapsed_hours"
            ),
        }
        combo_parameters = optimize_parameters_on_eval(
            regressions=regressions_by_dataset["eval"],
            signature_info=signature_info,
            revision_perf=revision_perf,
            oracle_metrics=oracle_metrics,
            risk_scores=risk_scores,
            oracle_names=args.oracles,
            localizer_names=args.localizers,
            workers=args.workers,
            default_parameters=default_parameters,
            backfill_retrigger_count_min=args.backfill_retrigger_count_min,
            backfill_retrigger_count_max=args.backfill_retrigger_count_max,
            probe_repeat_count_min=args.probe_repeat_count_min,
            probe_repeat_count_max=args.probe_repeat_count_max,
            midpoint_retrigger_count_min=args.midpoint_retrigger_count_min,
            midpoint_retrigger_count_max=args.midpoint_retrigger_count_max,
            multisection_section_count_min=args.multisection_section_count_min,
            multisection_section_count_max=args.multisection_section_count_max,
            multisection_retrigger_count_min=args.multisection_retrigger_count_min,
            multisection_retrigger_count_max=args.multisection_retrigger_count_max,
            pba_confidence_threshold_min=args.pba_confidence_threshold_min,
            pba_confidence_threshold_max=args.pba_confidence_threshold_max,
            pba_repeat_count_min=args.pba_repeat_count_min,
            pba_repeat_count_max=args.pba_repeat_count_max,
            pba_max_test_runs_min=args.pba_max_test_runs_min,
            pba_max_test_runs_max=args.pba_max_test_runs_max,
            pba_risk_prior_uniform_weight_min=(
                args.pba_risk_prior_uniform_weight_min
            ),
            pba_risk_prior_uniform_weight_max=(
                args.pba_risk_prior_uniform_weight_max
            ),
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
            risk_scores_path=risk_scores_path,
            signature_info=signature_info,
            revision_perf=revision_perf,
            oracle_metrics=oracle_metrics,
            risk_scores=risk_scores,
            oracle_names=args.oracles,
            localizer_names=args.localizers,
            workers=args.workers,
            default_parameters=default_parameters,
            combo_parameters=combo_parameters,
            optimization_settings=optimization_settings,
            random_seed=args.random_seed,
            distribution_plots_dir=(
                args.dist_plots_dir if args.draw_dist_plots else None
            ),
            plt=plt,
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
        "--risk-scores",
        type=Path,
        default=DEFAULT_RISK_SCORES,
        help=(
            "Path to per_commit_risk_scores.jsonl. Risk-aware and "
            "risk-weighted localizers use risk_score values from this file."
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
        "--draw-dist-plots",
        action="store_true",
        help=(
            "Draw per-localizer final-test distribution plots for test runs "
            "and elapsed time."
        ),
    )
    parser.add_argument(
        "--dist-plots-dir",
        type=Path,
        default=DEFAULT_DIST_PLOTS_DIR,
        help=(
            "Directory for final-test distribution plots written when "
            "--draw-dist-plots is set."
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
        "--probe-repeat-count",
        type=int,
        default=1,
        help=(
            "Number of first-submission test attempts per candidate revision "
            "for BackfillWithRepeat. Used directly when Optuna is disabled, "
            "and as a fallback default for combos without tuned parameters."
        ),
    )
    parser.add_argument(
        "--midpoint-retrigger-count",
        type=int,
        default=0,
        help=(
            "Number of additional midpoint decisions to retrigger for "
            "StandardMidpointBisection before choosing the bisection branch. "
            "Used directly when Optuna is disabled, and as a fallback default "
            "for combos without tuned parameters."
        ),
    )
    parser.add_argument(
        "--multisection-section-count",
        type=int,
        default=DEFAULT_MULTISECTION_SECTION_COUNT,
        help=(
            "Number of sections for multisection localizers. Each round tests "
            "section cut boundaries in parallel. Used directly when Optuna is "
            "disabled, and as a fallback default for combos without tuned "
            "parameters."
        ),
    )
    parser.add_argument(
        "--multisection-retrigger-count",
        type=int,
        default=DEFAULT_MULTISECTION_RETRIGGER_COUNT,
        help=(
            "Number of additional boundary-decision batches to retrigger for "
            "multisection localizers before selecting the next interval. Used "
            "directly when Optuna is disabled, and as a fallback default for "
            "combos without tuned parameters."
        ),
    )
    parser.add_argument(
        "--pba-confidence-threshold",
        type=float,
        default=DEFAULT_PBA_CONFIDENCE_THRESHOLD,
        help=(
            "Posterior probability required for "
            "ProbabilisticBisection_PosteriorMedian_UniformPrior to accept a "
            "single culprit. Used directly when Optuna is disabled, and as a "
            "fallback default for combos without tuned parameters."
        ),
    )
    parser.add_argument(
        "--pba-repeat-count",
        type=int,
        default=DEFAULT_PBA_REPEAT_COUNT,
        help=(
            "Number of repeated observations for each posterior-median PBA "
            "probe. Used directly when Optuna is disabled, and as a fallback "
            "default for combos without tuned parameters."
        ),
    )
    parser.add_argument(
        "--pba-max-test-runs",
        type=int,
        default=DEFAULT_PBA_MAX_TEST_RUNS,
        help=(
            "Maximum test jobs for one PBA regression before returning the MAP "
            "guess as a low-confidence or ambiguous localization."
        ),
    )
    parser.add_argument(
        "--pba-risk-prior-uniform-weight",
        type=float,
        default=DEFAULT_PBA_RISK_PRIOR_UNIFORM_WEIGHT,
        help=(
            "Uniform-prior mixture weight for "
            "ProbabilisticBisection_PosteriorMedian_RiskAwarePrior. A larger "
            "value trusts per-commit risk scores less."
        ),
    )
    parser.add_argument(
        "--optuna-trials",
        "--mopt-trials",
        dest="optuna_trials",
        type=int,
        default=0,
        help=(
            "Base number of Optuna trials per localizer/oracle combo per "
            "tunable parameter. When positive, parameters are optimized on the "
            "eval split and replayed on the requested split(s)."
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
        "--probe-repeat-count-min",
        type=int,
        default=DEFAULT_PROBE_REPEAT_COUNT_MIN,
        help="Minimum BackfillWithRepeat first-submission repeat count sampled by Optuna.",
    )
    parser.add_argument(
        "--probe-repeat-count-max",
        type=int,
        default=DEFAULT_PROBE_REPEAT_COUNT_MAX,
        help="Maximum BackfillWithRepeat first-submission repeat count sampled by Optuna.",
    )
    parser.add_argument(
        "--midpoint-retrigger-count-min",
        type=int,
        default=DEFAULT_MIDPOINT_RETRIGGER_COUNT_MIN,
        help="Minimum StandardMidpointBisection retrigger count sampled by Optuna.",
    )
    parser.add_argument(
        "--midpoint-retrigger-count-max",
        type=int,
        default=DEFAULT_MIDPOINT_RETRIGGER_COUNT_MAX,
        help="Maximum StandardMidpointBisection retrigger count sampled by Optuna.",
    )
    parser.add_argument(
        "--multisection-section-count-min",
        type=int,
        default=DEFAULT_MULTISECTION_SECTION_COUNT_MIN,
        help="Minimum multisection section count sampled by Optuna.",
    )
    parser.add_argument(
        "--multisection-section-count-max",
        type=int,
        default=DEFAULT_MULTISECTION_SECTION_COUNT_MAX,
        help="Maximum multisection section count sampled by Optuna.",
    )
    parser.add_argument(
        "--multisection-retrigger-count-min",
        type=int,
        default=DEFAULT_MULTISECTION_RETRIGGER_COUNT_MIN,
        help="Minimum multisection retrigger count sampled by Optuna.",
    )
    parser.add_argument(
        "--multisection-retrigger-count-max",
        type=int,
        default=DEFAULT_MULTISECTION_RETRIGGER_COUNT_MAX,
        help="Maximum multisection retrigger count sampled by Optuna.",
    )
    parser.add_argument(
        "--pba-confidence-threshold-min",
        type=float,
        default=DEFAULT_PBA_CONFIDENCE_THRESHOLD_MIN,
        help="Minimum PBA confidence threshold sampled by Optuna.",
    )
    parser.add_argument(
        "--pba-confidence-threshold-max",
        type=float,
        default=DEFAULT_PBA_CONFIDENCE_THRESHOLD_MAX,
        help="Maximum PBA confidence threshold sampled by Optuna.",
    )
    parser.add_argument(
        "--pba-repeat-count-min",
        type=int,
        default=DEFAULT_PBA_REPEAT_COUNT_MIN,
        help="Minimum PBA repeat count sampled by Optuna.",
    )
    parser.add_argument(
        "--pba-repeat-count-max",
        type=int,
        default=DEFAULT_PBA_REPEAT_COUNT_MAX,
        help="Maximum PBA repeat count sampled by Optuna.",
    )
    parser.add_argument(
        "--pba-max-test-runs-min",
        type=int,
        default=DEFAULT_PBA_MAX_TEST_RUNS_MIN,
        help="Minimum PBA max-test-runs value sampled by Optuna.",
    )
    parser.add_argument(
        "--pba-max-test-runs-max",
        type=int,
        default=DEFAULT_PBA_MAX_TEST_RUNS_MAX,
        help="Maximum PBA max-test-runs value sampled by Optuna.",
    )
    parser.add_argument(
        "--pba-risk-prior-uniform-weight-min",
        type=float,
        default=DEFAULT_PBA_RISK_PRIOR_UNIFORM_WEIGHT_MIN,
        help="Minimum risk-aware PBA uniform-prior mixture weight sampled by Optuna.",
    )
    parser.add_argument(
        "--pba-risk-prior-uniform-weight-max",
        type=float,
        default=DEFAULT_PBA_RISK_PRIOR_UNIFORM_WEIGHT_MAX,
        help="Maximum risk-aware PBA uniform-prior mixture weight sampled by Optuna.",
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
    if args.probe_repeat_count < 1:
        parser.error("--probe-repeat-count must be at least 1")
    if args.midpoint_retrigger_count < 0:
        parser.error("--midpoint-retrigger-count must be non-negative")
    if args.multisection_section_count < 2:
        parser.error("--multisection-section-count must be at least 2")
    if args.multisection_retrigger_count < 0:
        parser.error("--multisection-retrigger-count must be non-negative")
    if not 0.0 < args.pba_confidence_threshold <= 1.0:
        parser.error("--pba-confidence-threshold must be in (0, 1]")
    if args.pba_repeat_count < 1:
        parser.error("--pba-repeat-count must be at least 1")
    if args.pba_max_test_runs < 1:
        parser.error("--pba-max-test-runs must be at least 1")
    if not 0.0 <= args.pba_risk_prior_uniform_weight <= 1.0:
        parser.error("--pba-risk-prior-uniform-weight must be in [0, 1]")
    if args.optuna_trials < 0:
        parser.error("--optuna-trials must be non-negative")
    if args.backfill_retrigger_count_min < 0:
        parser.error("--backfill-retrigger-count-min must be non-negative")
    if args.backfill_retrigger_count_max < args.backfill_retrigger_count_min:
        parser.error(
            "--backfill-retrigger-count-max must be greater than or equal to "
            "--backfill-retrigger-count-min"
        )
    if args.probe_repeat_count_min < 1:
        parser.error("--probe-repeat-count-min must be at least 1")
    if args.probe_repeat_count_max < args.probe_repeat_count_min:
        parser.error(
            "--probe-repeat-count-max must be greater than or equal to "
            "--probe-repeat-count-min"
        )
    if args.midpoint_retrigger_count_min < 0:
        parser.error("--midpoint-retrigger-count-min must be non-negative")
    if args.midpoint_retrigger_count_max < args.midpoint_retrigger_count_min:
        parser.error(
            "--midpoint-retrigger-count-max must be greater than or equal to "
            "--midpoint-retrigger-count-min"
        )
    if args.multisection_section_count_min < 2:
        parser.error("--multisection-section-count-min must be at least 2")
    if args.multisection_section_count_max < args.multisection_section_count_min:
        parser.error(
            "--multisection-section-count-max must be greater than or equal to "
            "--multisection-section-count-min"
        )
    if args.multisection_retrigger_count_min < 0:
        parser.error("--multisection-retrigger-count-min must be non-negative")
    if args.multisection_retrigger_count_max < args.multisection_retrigger_count_min:
        parser.error(
            "--multisection-retrigger-count-max must be greater than or equal to "
            "--multisection-retrigger-count-min"
        )
    if not 0.0 < args.pba_confidence_threshold_min <= 1.0:
        parser.error("--pba-confidence-threshold-min must be in (0, 1]")
    if not 0.0 < args.pba_confidence_threshold_max <= 1.0:
        parser.error("--pba-confidence-threshold-max must be in (0, 1]")
    if args.pba_confidence_threshold_max < args.pba_confidence_threshold_min:
        parser.error(
            "--pba-confidence-threshold-max must be greater than or equal to "
            "--pba-confidence-threshold-min"
        )
    if args.pba_repeat_count_min < 1:
        parser.error("--pba-repeat-count-min must be at least 1")
    if args.pba_repeat_count_max < args.pba_repeat_count_min:
        parser.error(
            "--pba-repeat-count-max must be greater than or equal to "
            "--pba-repeat-count-min"
        )
    if args.pba_max_test_runs_min < 1:
        parser.error("--pba-max-test-runs-min must be at least 1")
    if args.pba_max_test_runs_max < args.pba_max_test_runs_min:
        parser.error(
            "--pba-max-test-runs-max must be greater than or equal to "
            "--pba-max-test-runs-min"
        )
    if not 0.0 <= args.pba_risk_prior_uniform_weight_min <= 1.0:
        parser.error("--pba-risk-prior-uniform-weight-min must be in [0, 1]")
    if not 0.0 <= args.pba_risk_prior_uniform_weight_max <= 1.0:
        parser.error("--pba-risk-prior-uniform-weight-max must be in [0, 1]")
    if (
        args.pba_risk_prior_uniform_weight_max
        < args.pba_risk_prior_uniform_weight_min
    ):
        parser.error(
            "--pba-risk-prior-uniform-weight-max must be greater than or "
            "equal to --pba-risk-prior-uniform-weight-min"
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
    risk_scores_path: Path | None,
    signature_info: SignatureInfoIndex,
    revision_perf: RevisionPerfIndex,
    oracle_metrics: Mapping[int, OracleMetrics],
    risk_scores: Mapping[str, float],
    oracle_names: Sequence[str],
    localizer_names: Sequence[str],
    workers: int,
    default_parameters: SimulationParameters,
    combo_parameters: Mapping[tuple[str, str], SimulationParameters] | None = None,
    optimization_settings: Mapping[str, Any] | None = None,
    random_seed: int | None,
    distribution_plots_dir: Path | None = None,
    plt: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run every requested localizer/oracle combination for one regression split."""

    summary_runs = []
    detail_runs = []
    if distribution_plots_dir is not None:
        distribution_plots_dir.mkdir(parents=True, exist_ok=True)

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
                risk_scores=risk_scores,
                workers=workers,
                parameters=parameters,
                random_seed=random_seed,
            )
            summary_run = {
                "localizer": localizer_name,
                "test_oracle": oracle_name,
                "metrics": metrics,
            }
            if combo_parameters is not None and combo_key in combo_parameters:
                summary_run["optuna_optimized_parameters"] = (
                    optimized_parameter_values(
                        localizer_name=localizer_name,
                        parameters=parameters,
                    )
                )
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
                "results": [result.to_json() for result in results],
            }
            if combo_parameters is not None and combo_key in combo_parameters:
                detail_run["optuna_optimized_parameters"] = (
                    optimized_parameter_values(
                        localizer_name=localizer_name,
                        parameters=parameters,
                    )
                )
            detail_runs.append(detail_run)
            if (
                distribution_plots_dir is not None
                and plt is not None
                and dataset_name == "final_test"
            ):
                write_distribution_plots(
                    dataset_name=dataset_name,
                    localizer_name=localizer_name,
                    oracle_name=oracle_name,
                    results=results,
                    plots_dir=distribution_plots_dir,
                    plt=plt,
                )

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
            "probe_repeat_count": default_parameters.probe_repeat_count,
            "midpoint_retrigger_count": default_parameters.midpoint_retrigger_count,
            "multisection_section_count": (
                default_parameters.multisection_section_count
            ),
            "multisection_retrigger_count": (
                default_parameters.multisection_retrigger_count
            ),
            "pba_batch_size": 1,
            "pba_query_strategy": "posterior_median",
            "pba_confidence_threshold": (
                default_parameters.pba_confidence_threshold
            ),
            "pba_repeat_count": default_parameters.pba_repeat_count,
            "pba_max_test_runs": default_parameters.pba_max_test_runs,
            "pba_risk_prior_uniform_weight": (
                default_parameters.pba_risk_prior_uniform_weight
            ),
            "random_seed": random_seed,
            "random_seed_derivation": "random_seed + regression_id - 1",
        },
    }
    if risk_scores_path is not None:
        base_output["inputs"]["risk_scores"] = str(risk_scores_path)
    if optimization_settings is not None:
        base_output["settings"]["optimization"] = dict(optimization_settings)
    best_combo_fields = compute_best_combo_fields(summary_runs)
    return (
        {**base_output, **best_combo_fields, "runs": summary_runs},
        {**base_output, **best_combo_fields, "runs": detail_runs},
    )


def optimized_parameter_values(
    *,
    localizer_name: str,
    parameters: SimulationParameters,
) -> dict[str, Any]:
    """Return only the Optuna-tunable parameter values for one localizer."""

    serialized_parameters = parameters.to_json()
    return {
        parameter_name: serialized_parameters[parameter_name]
        for parameter_name in TUNABLE_PARAMETER_FIELDS_BY_LOCALIZER.get(
            localizer_name,
            (),
        )
    }


def compute_best_combo_fields(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute top-level best-combo selections from aggregate run metrics."""

    if not runs:
        return {}

    best_fields: dict[str, Any] = {
        "best_combo_vote_weights": {
            str(spec["metric"]): int(spec["vote_weight"])
            for spec in BEST_COMBO_METRIC_SPECS
        },
    }
    votes_by_combo: Counter[tuple[str, str]] = Counter()
    metric_votes_by_combo: dict[tuple[str, str], dict[str, int]] = {}

    for spec in BEST_COMBO_METRIC_SPECS:
        metric = str(spec["metric"])
        best_run = best_run_for_metric(
            runs,
            metric=metric,
            direction=str(spec["direction"]),
        )
        vote_weight = int(spec["vote_weight"])
        combo_key = run_combo_key(best_run)
        votes_by_combo[combo_key] += vote_weight
        metric_votes_by_combo.setdefault(combo_key, {})[metric] = vote_weight
        best_fields[str(spec["field"])] = combo_selection_to_json(best_run)

    best_overall = min(
        runs,
        key=lambda run: (
            -votes_by_combo[run_combo_key(run)],
            *run_preference_sort_key(run),
        ),
    )
    best_overall_key = run_combo_key(best_overall)
    best_fields["best_combo_overall"] = combo_selection_to_json(
        best_overall,
        metric_votes=metric_votes_by_combo.get(best_overall_key, {}),
    )
    return best_fields


def best_run_for_metric(
    runs: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    direction: str,
) -> Mapping[str, Any]:
    """Return the best run for one metric with deterministic tie-breaking."""

    return min(
        runs,
        key=lambda run: (
            primary_metric_sort_value(run, metric=metric, direction=direction),
            *run_preference_sort_key(run),
        ),
    )


def primary_metric_sort_value(
    run: Mapping[str, Any],
    *,
    metric: str,
    direction: str,
) -> float:
    """Return a sortable value where lower means better for the target metric."""

    value = numeric_metric_value(run, metric)
    if value is None:
        return math.inf
    if direction == "maximize":
        return -value
    if direction == "minimize":
        return value
    raise ValueError(f"unknown metric direction: {direction!r}")


def run_preference_sort_key(run: Mapping[str, Any]) -> tuple[Any, ...]:
    """Tie-break combo selection by success, cost metrics, then stable names."""

    return (
        primary_metric_sort_value(
            run,
            metric="success_rate_percent",
            direction="maximize",
        ),
        primary_metric_sort_value(
            run,
            metric="mean_elapsed_hours",
            direction="minimize",
        ),
        primary_metric_sort_value(
            run,
            metric="mean_test_runs",
            direction="minimize",
        ),
        primary_metric_sort_value(
            run,
            metric="max_elapsed_hours",
            direction="minimize",
        ),
        primary_metric_sort_value(
            run,
            metric="max_test_runs",
            direction="minimize",
        ),
        str(run.get("localizer", "")),
        str(run.get("test_oracle", "")),
        json.dumps(run.get("parameters", {}), sort_keys=True),
    )


def numeric_metric_value(run: Mapping[str, Any], metric: str) -> float | None:
    """Read a finite numeric metric value from a serialized run block."""

    raw_value = run.get("metrics", {}).get(metric)
    if raw_value is None:
        return None
    value = float(raw_value)
    return value if math.isfinite(value) else None


def run_combo_key(run: Mapping[str, Any]) -> tuple[str, str]:
    """Return the stable identity for one localizer/oracle combo block."""

    return (str(run.get("localizer", "")), str(run.get("test_oracle", "")))


def combo_selection_to_json(
    run: Mapping[str, Any],
    *,
    metric_votes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Serialize one selected combo as a compact localizer/oracle identifier."""

    output = {
        "localizer": run.get("localizer"),
        "test_oracle": run.get("test_oracle"),
    }
    if metric_votes is not None:
        output["metric_votes"] = dict(sorted(metric_votes.items()))
    return output


def run_combo(
    *,
    regressions: Sequence[Mapping[str, Any]],
    localizer_name: str,
    oracle_name: str,
    signature_info: SignatureInfoIndex,
    revision_perf: RevisionPerfIndex,
    oracle_metrics: Mapping[int, OracleMetrics],
    risk_scores: Mapping[str, float],
    workers: int,
    parameters: SimulationParameters,
    random_seed: int | None,
) -> tuple[list[Any], dict[str, Any]]:
    """Run one localizer/oracle pair and return per-regression and summary metrics."""

    localizer = build_localizer(
        localizer_name,
        parameters=parameters,
        risk_scores=risk_scores,
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
    risk_scores: Mapping[str, float],
) -> CulpritLocalizer:
    """Construct a localizer from the registry."""

    localizer_cls = LOCALIZERS[localizer_name]
    if localizer_name == "Backfill":
        return localizer_cls(
            backfill_retrigger_count=parameters.backfill_retrigger_count,
        )
    if localizer_name == "BackfillWithRepeat":
        return localizer_cls(
            backfill_retrigger_count=parameters.backfill_retrigger_count,
            probe_repeat_count=parameters.probe_repeat_count,
        )
    if localizer_name == "StandardMidpointBisection":
        return localizer_cls(
            midpoint_retrigger_count=parameters.midpoint_retrigger_count,
        )
    if localizer_name == "StandardMidpointMultisection":
        return localizer_cls(
            multisection_section_count=parameters.multisection_section_count,
            multisection_retrigger_count=parameters.multisection_retrigger_count,
        )
    if localizer_name == "RiskWeightedMultisection":
        return localizer_cls(
            multisection_section_count=parameters.multisection_section_count,
            multisection_retrigger_count=parameters.multisection_retrigger_count,
            risk_scores=risk_scores,
        )
    if localizer_name == "ProbabilisticBisection_PosteriorMedian_UniformPrior":
        return localizer_cls(
            pba_confidence_threshold=parameters.pba_confidence_threshold,
            pba_repeat_count=parameters.pba_repeat_count,
            pba_max_test_runs=parameters.pba_max_test_runs,
        )
    if localizer_name == "ProbabilisticMultiSection_PosteriorQuantile_UniformPrior":
        return localizer_cls(
            multisection_section_count=parameters.multisection_section_count,
            pba_confidence_threshold=parameters.pba_confidence_threshold,
            pba_repeat_count=parameters.pba_repeat_count,
            pba_max_test_runs=parameters.pba_max_test_runs,
        )
    if localizer_name == "ProbabilisticBisection_PosteriorMedian_RiskAwarePrior":
        return localizer_cls(
            pba_confidence_threshold=parameters.pba_confidence_threshold,
            pba_repeat_count=parameters.pba_repeat_count,
            pba_max_test_runs=parameters.pba_max_test_runs,
            pba_risk_prior_uniform_weight=(
                parameters.pba_risk_prior_uniform_weight
            ),
            risk_scores=risk_scores,
        )
    if localizer_name == "RiskWeightedBisection":
        return localizer_cls(
            midpoint_retrigger_count=parameters.midpoint_retrigger_count,
            risk_scores=risk_scores,
        )
    return localizer_cls()


def optimize_parameters_on_eval(
    *,
    regressions: Sequence[Mapping[str, Any]],
    signature_info: SignatureInfoIndex,
    revision_perf: RevisionPerfIndex,
    oracle_metrics: Mapping[int, OracleMetrics],
    risk_scores: Mapping[str, float],
    oracle_names: Sequence[str],
    localizer_names: Sequence[str],
    workers: int,
    default_parameters: SimulationParameters,
    backfill_retrigger_count_min: int,
    backfill_retrigger_count_max: int,
    probe_repeat_count_min: int,
    probe_repeat_count_max: int,
    midpoint_retrigger_count_min: int,
    midpoint_retrigger_count_max: int,
    multisection_section_count_min: int,
    multisection_section_count_max: int,
    multisection_retrigger_count_min: int,
    multisection_retrigger_count_max: int,
    pba_confidence_threshold_min: float,
    pba_confidence_threshold_max: float,
    pba_repeat_count_min: int,
    pba_repeat_count_max: int,
    pba_max_test_runs_min: int,
    pba_max_test_runs_max: int,
    pba_risk_prior_uniform_weight_min: float,
    pba_risk_prior_uniform_weight_max: float,
    optuna_trials: int,
    optuna_seed: int,
    random_seed: int | None,
) -> dict[tuple[str, str], SimulationParameters]:
    """Tune algorithm parameters on the eval split for every selected combo."""

    combo_parameters: dict[tuple[str, str], SimulationParameters] = {}

    for localizer_name in localizer_names:
        for oracle_name in oracle_names:
            combo_key = (localizer_name, oracle_name)
            parameters = optimize_combo_on_eval(
                regressions=regressions,
                localizer_name=localizer_name,
                oracle_name=oracle_name,
                signature_info=signature_info,
                revision_perf=revision_perf,
                oracle_metrics=oracle_metrics,
                risk_scores=risk_scores,
                workers=workers,
                default_parameters=default_parameters,
                backfill_retrigger_count_min=backfill_retrigger_count_min,
                backfill_retrigger_count_max=backfill_retrigger_count_max,
                probe_repeat_count_min=probe_repeat_count_min,
                probe_repeat_count_max=probe_repeat_count_max,
                midpoint_retrigger_count_min=midpoint_retrigger_count_min,
                midpoint_retrigger_count_max=midpoint_retrigger_count_max,
                multisection_section_count_min=multisection_section_count_min,
                multisection_section_count_max=multisection_section_count_max,
                multisection_retrigger_count_min=multisection_retrigger_count_min,
                multisection_retrigger_count_max=multisection_retrigger_count_max,
                pba_confidence_threshold_min=pba_confidence_threshold_min,
                pba_confidence_threshold_max=pba_confidence_threshold_max,
                pba_repeat_count_min=pba_repeat_count_min,
                pba_repeat_count_max=pba_repeat_count_max,
                pba_max_test_runs_min=pba_max_test_runs_min,
                pba_max_test_runs_max=pba_max_test_runs_max,
                pba_risk_prior_uniform_weight_min=(
                    pba_risk_prior_uniform_weight_min
                ),
                pba_risk_prior_uniform_weight_max=(
                    pba_risk_prior_uniform_weight_max
                ),
                optuna_trials=optuna_trials,
                optuna_seed=optuna_seed,
                random_seed=random_seed,
            )
            combo_parameters[combo_key] = parameters
            print(
                "optuna selected "
                f"{localizer_name}/{oracle_name}: {parameters.to_json()}"
            )

    return combo_parameters


def optimize_combo_on_eval(
    *,
    regressions: Sequence[Mapping[str, Any]],
    localizer_name: str,
    oracle_name: str,
    signature_info: SignatureInfoIndex,
    revision_perf: RevisionPerfIndex,
    oracle_metrics: Mapping[int, OracleMetrics],
    risk_scores: Mapping[str, float],
    workers: int,
    default_parameters: SimulationParameters,
    backfill_retrigger_count_min: int,
    backfill_retrigger_count_max: int,
    probe_repeat_count_min: int,
    probe_repeat_count_max: int,
    midpoint_retrigger_count_min: int,
    midpoint_retrigger_count_max: int,
    multisection_section_count_min: int,
    multisection_section_count_max: int,
    multisection_retrigger_count_min: int,
    multisection_retrigger_count_max: int,
    pba_confidence_threshold_min: float,
    pba_confidence_threshold_max: float,
    pba_repeat_count_min: int,
    pba_repeat_count_max: int,
    pba_max_test_runs_min: int,
    pba_max_test_runs_max: int,
    pba_risk_prior_uniform_weight_min: float,
    pba_risk_prior_uniform_weight_max: float,
    optuna_trials: int,
    optuna_seed: int,
    random_seed: int | None,
) -> SimulationParameters:
    """Run one multi-objective Optuna study on eval for one localizer/oracle pair."""

    if not has_tunable_parameters(
        localizer_name=localizer_name,
        oracle_name=oracle_name,
    ):
        return default_parameters

    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna is required. Install with `pip install optuna`."
        ) from exc

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
            probe_repeat_count_min=probe_repeat_count_min,
            probe_repeat_count_max=probe_repeat_count_max,
            midpoint_retrigger_count_min=midpoint_retrigger_count_min,
            midpoint_retrigger_count_max=midpoint_retrigger_count_max,
            multisection_section_count_min=multisection_section_count_min,
            multisection_section_count_max=multisection_section_count_max,
            multisection_retrigger_count_min=multisection_retrigger_count_min,
            multisection_retrigger_count_max=multisection_retrigger_count_max,
            pba_confidence_threshold_min=pba_confidence_threshold_min,
            pba_confidence_threshold_max=pba_confidence_threshold_max,
            pba_repeat_count_min=pba_repeat_count_min,
            pba_repeat_count_max=pba_repeat_count_max,
            pba_max_test_runs_min=pba_max_test_runs_min,
            pba_max_test_runs_max=pba_max_test_runs_max,
            pba_risk_prior_uniform_weight_min=pba_risk_prior_uniform_weight_min,
            pba_risk_prior_uniform_weight_max=pba_risk_prior_uniform_weight_max,
        )
        _, metrics = run_combo(
            regressions=regressions,
            localizer_name=localizer_name,
            oracle_name=oracle_name,
            signature_info=signature_info,
            revision_perf=revision_perf,
            oracle_metrics=oracle_metrics,
            risk_scores=risk_scores,
            workers=workers,
            parameters=parameters,
            random_seed=random_seed,
        )
        trial.set_user_attr("parameters", parameters.to_json())
        trial.set_user_attr("metrics", metrics)
        return objective_values(metrics)

    effective_trials = effective_optuna_trials(
        localizer_name=localizer_name,
        oracle_name=oracle_name,
        optuna_trials=optuna_trials,
    )
    study.optimize(objective, n_trials=effective_trials, show_progress_bar=False)
    if not study.best_trials:
        raise RuntimeError(
            f"Optuna produced no Pareto-optimal trials for {localizer_name}/{oracle_name}."
        )

    selected_trial = select_pareto_trial(study.best_trials)
    selected_parameters = parameters_from_trial(selected_trial)
    return selected_parameters


def has_tunable_parameters(*, localizer_name: str, oracle_name: str) -> bool:
    """Return whether the selected combo currently exposes Optuna parameters."""

    return (
        tunable_parameter_count(
            localizer_name=localizer_name,
            oracle_name=oracle_name,
        )
        > 0
    )


def tunable_parameter_count(*, localizer_name: str, oracle_name: str) -> int:
    """Return the number of Optuna-tunable parameters for one combo."""

    del oracle_name
    return len(TUNABLE_PARAMETER_FIELDS_BY_LOCALIZER.get(localizer_name, ()))


def effective_optuna_trials(
    *,
    localizer_name: str,
    oracle_name: str,
    optuna_trials: int,
) -> int:
    """Scale the configured base trial count by the combo's parameter count."""

    return int(optuna_trials) * tunable_parameter_count(
        localizer_name=localizer_name,
        oracle_name=oracle_name,
    )


def suggest_parameters(
    trial: Any,
    *,
    localizer_name: str,
    oracle_name: str,
    default_parameters: SimulationParameters,
    backfill_retrigger_count_min: int,
    backfill_retrigger_count_max: int,
    probe_repeat_count_min: int,
    probe_repeat_count_max: int,
    midpoint_retrigger_count_min: int,
    midpoint_retrigger_count_max: int,
    multisection_section_count_min: int,
    multisection_section_count_max: int,
    multisection_retrigger_count_min: int,
    multisection_retrigger_count_max: int,
    pba_confidence_threshold_min: float,
    pba_confidence_threshold_max: float,
    pba_repeat_count_min: int,
    pba_repeat_count_max: int,
    pba_max_test_runs_min: int,
    pba_max_test_runs_max: int,
    pba_risk_prior_uniform_weight_min: float,
    pba_risk_prior_uniform_weight_max: float,
) -> SimulationParameters:
    """Ask Optuna for one parameter set for the selected algorithm combo."""

    del oracle_name
    backfill_retrigger_count = default_parameters.backfill_retrigger_count
    probe_repeat_count = default_parameters.probe_repeat_count
    midpoint_retrigger_count = default_parameters.midpoint_retrigger_count
    multisection_section_count = default_parameters.multisection_section_count
    multisection_retrigger_count = default_parameters.multisection_retrigger_count
    pba_confidence_threshold = default_parameters.pba_confidence_threshold
    pba_repeat_count = default_parameters.pba_repeat_count
    pba_max_test_runs = default_parameters.pba_max_test_runs
    pba_risk_prior_uniform_weight = (
        default_parameters.pba_risk_prior_uniform_weight
    )
    if localizer_name == "Backfill":
        backfill_retrigger_count = trial.suggest_int(
            "Backfill_backfill_retrigger_count",
            int(backfill_retrigger_count_min),
            int(backfill_retrigger_count_max),
        )
    elif localizer_name == "BackfillWithRepeat":
        backfill_retrigger_count = trial.suggest_int(
            "BackfillWithRepeat_backfill_retrigger_count",
            int(backfill_retrigger_count_min),
            int(backfill_retrigger_count_max),
        )
        probe_repeat_count = trial.suggest_int(
            "BackfillWithRepeat_probe_repeat_count",
            int(probe_repeat_count_min),
            int(probe_repeat_count_max),
        )
    elif localizer_name == "StandardMidpointBisection":
        midpoint_retrigger_count = trial.suggest_int(
            "StandardMidpointBisection_midpoint_retrigger_count",
            int(midpoint_retrigger_count_min),
            int(midpoint_retrigger_count_max),
        )
    elif localizer_name == "RiskWeightedBisection":
        midpoint_retrigger_count = trial.suggest_int(
            "RiskWeightedBisection_midpoint_retrigger_count",
            int(midpoint_retrigger_count_min),
            int(midpoint_retrigger_count_max),
        )
    elif localizer_name == "StandardMidpointMultisection":
        multisection_section_count = trial.suggest_int(
            "StandardMidpointMultisection_multisection_section_count",
            int(multisection_section_count_min),
            int(multisection_section_count_max),
        )
        multisection_retrigger_count = trial.suggest_int(
            "StandardMidpointMultisection_multisection_retrigger_count",
            int(multisection_retrigger_count_min),
            int(multisection_retrigger_count_max),
        )
    elif localizer_name == "RiskWeightedMultisection":
        multisection_section_count = trial.suggest_int(
            "RiskWeightedMultisection_multisection_section_count",
            int(multisection_section_count_min),
            int(multisection_section_count_max),
        )
        multisection_retrigger_count = trial.suggest_int(
            "RiskWeightedMultisection_multisection_retrigger_count",
            int(multisection_retrigger_count_min),
            int(multisection_retrigger_count_max),
        )
    elif localizer_name == "ProbabilisticBisection_PosteriorMedian_UniformPrior":
        pba_confidence_threshold = trial.suggest_float(
            (
                "ProbabilisticBisection_PosteriorMedian_UniformPrior_"
                "pba_confidence_threshold"
            ),
            float(pba_confidence_threshold_min),
            float(pba_confidence_threshold_max),
        )
        pba_repeat_count = trial.suggest_int(
            "ProbabilisticBisection_PosteriorMedian_UniformPrior_pba_repeat_count",
            int(pba_repeat_count_min),
            int(pba_repeat_count_max),
        )
        pba_max_test_runs = trial.suggest_int(
            "ProbabilisticBisection_PosteriorMedian_UniformPrior_pba_max_test_runs",
            int(pba_max_test_runs_min),
            int(pba_max_test_runs_max),
        )
    elif localizer_name == "ProbabilisticMultiSection_PosteriorQuantile_UniformPrior":
        multisection_section_count = trial.suggest_int(
            (
                "ProbabilisticMultiSection_PosteriorQuantile_UniformPrior_"
                "multisection_section_count"
            ),
            int(multisection_section_count_min),
            int(multisection_section_count_max),
        )
        pba_confidence_threshold = trial.suggest_float(
            (
                "ProbabilisticMultiSection_PosteriorQuantile_UniformPrior_"
                "pba_confidence_threshold"
            ),
            float(pba_confidence_threshold_min),
            float(pba_confidence_threshold_max),
        )
        pba_repeat_count = trial.suggest_int(
            (
                "ProbabilisticMultiSection_PosteriorQuantile_UniformPrior_"
                "pba_repeat_count"
            ),
            int(pba_repeat_count_min),
            int(pba_repeat_count_max),
        )
        pba_max_test_runs = trial.suggest_int(
            (
                "ProbabilisticMultiSection_PosteriorQuantile_UniformPrior_"
                "pba_max_test_runs"
            ),
            int(pba_max_test_runs_min),
            int(pba_max_test_runs_max),
        )
    elif localizer_name == "ProbabilisticBisection_PosteriorMedian_RiskAwarePrior":
        pba_confidence_threshold = trial.suggest_float(
            (
                "ProbabilisticBisection_PosteriorMedian_RiskAwarePrior_"
                "pba_confidence_threshold"
            ),
            float(pba_confidence_threshold_min),
            float(pba_confidence_threshold_max),
        )
        pba_repeat_count = trial.suggest_int(
            "ProbabilisticBisection_PosteriorMedian_RiskAwarePrior_pba_repeat_count",
            int(pba_repeat_count_min),
            int(pba_repeat_count_max),
        )
        pba_max_test_runs = trial.suggest_int(
            "ProbabilisticBisection_PosteriorMedian_RiskAwarePrior_pba_max_test_runs",
            int(pba_max_test_runs_min),
            int(pba_max_test_runs_max),
        )
        pba_risk_prior_uniform_weight = trial.suggest_float(
            (
                "ProbabilisticBisection_PosteriorMedian_RiskAwarePrior_"
                "pba_risk_prior_uniform_weight"
            ),
            float(pba_risk_prior_uniform_weight_min),
            float(pba_risk_prior_uniform_weight_max),
        )
    return SimulationParameters(
        backfill_retrigger_count=int(backfill_retrigger_count),
        probe_repeat_count=int(probe_repeat_count),
        midpoint_retrigger_count=int(midpoint_retrigger_count),
        multisection_section_count=int(multisection_section_count),
        multisection_retrigger_count=int(multisection_retrigger_count),
        pba_confidence_threshold=float(pba_confidence_threshold),
        pba_repeat_count=int(pba_repeat_count),
        pba_max_test_runs=int(pba_max_test_runs),
        pba_risk_prior_uniform_weight=float(pba_risk_prior_uniform_weight),
    )


def objective_values(metrics: Mapping[str, Any]) -> tuple[float, float, float]:
    """Return Optuna objective values from aggregate simulation metrics."""

    return (
        finite_objective_value(metrics.get("mean_elapsed_hours")),
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
        probe_repeat_count=int(raw_parameters["probe_repeat_count"]),
        midpoint_retrigger_count=int(raw_parameters["midpoint_retrigger_count"]),
        multisection_section_count=int(
            raw_parameters["multisection_section_count"]
        ),
        multisection_retrigger_count=int(
            raw_parameters["multisection_retrigger_count"]
        ),
        pba_confidence_threshold=float(raw_parameters["pba_confidence_threshold"]),
        pba_repeat_count=int(raw_parameters["pba_repeat_count"]),
        pba_max_test_runs=int(raw_parameters["pba_max_test_runs"]),
        pba_risk_prior_uniform_weight=float(
            raw_parameters.get(
                "pba_risk_prior_uniform_weight",
                DEFAULT_PBA_RISK_PRIOR_UNIFORM_WEIGHT,
            )
        ),
    )


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
    """Compute aggregate success, elapsed-time, and test-run metrics for one run."""

    total = len(results)
    successes = [result for result in results if result.success]
    elapsed_values = [result.elapsed_hours for result in results]
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
        "mean_elapsed_hours": round(mean(elapsed_values), 2)
        if elapsed_values
        else None,
        "max_elapsed_hours": round(max(elapsed_values), 2)
        if elapsed_values
        else None,
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


def write_distribution_plots(
    *,
    dataset_name: str,
    localizer_name: str,
    oracle_name: str,
    results: Sequence[Any],
    plots_dir: Path,
    plt: Any,
) -> None:
    """Write histogram plots for elapsed-time and test-run distributions."""

    elapsed_values = [result.elapsed_hours for result in results]
    test_run_values = [result.test_runs for result in results]
    if not elapsed_values and not test_run_values:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_distribution_histogram(
        ax=axes[0],
        values=test_run_values,
        title="Test Runs",
        xlabel="Test runs",
        color="#1f77b4",
    )
    plot_distribution_histogram(
        ax=axes[1],
        values=elapsed_values,
        title="Elapsed Time",
        xlabel="Elapsed time (hours)",
        color="#ff7f0e",
    )
    fig.suptitle(f"{localizer_name} / {dataset_name}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    output_path = plots_dir / (
        f"{slugify_path_component(dataset_name)}__"
        f"{slugify_path_component(localizer_name)}__"
        f"{slugify_path_component(oracle_name)}_distributions.png"
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


def plot_distribution_histogram(
    *,
    ax: Any,
    values: Sequence[float | int],
    title: str,
    xlabel: str,
    color: str,
) -> None:
    """Draw one histogram with stable bins and summary guide lines."""

    numeric_values = [float(value) for value in values if math.isfinite(float(value))]
    if not numeric_values:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    value_mean = mean(numeric_values)
    value_median = median(numeric_values)
    ax.hist(
        numeric_values,
        bins=histogram_bin_count(numeric_values),
        color=color,
        edgecolor="white",
        alpha=0.85,
        linewidth=0.35,
    )
    mean_line = ax.axvline(
        value_mean,
        color="#222222",
        linestyle="--",
        linewidth=1.5,
    )
    median_line = ax.axvline(
        value_median,
        color="#666666",
        linestyle=":",
        linewidth=1.5,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.7)
    ax.legend(
        [mean_line, median_line],
        [f"mean={value_mean:.2f}", f"median={value_median:.2f}"],
        frameon=False,
    )


def histogram_bin_count(values: Sequence[float]) -> int:
    """Return a small, stable histogram bin count for per-regression metrics."""

    if len(values) <= 1 or min(values) == max(values):
        return 1
    return max(5, min(20, int(math.sqrt(len(values)))))


def slugify_path_component(value: str) -> str:
    """Convert an identifier into a filesystem-safe lowercase file component."""

    cleaned = "".join(
        character.lower() if character.isalnum() else "_"
        for character in value.strip()
    )
    return cleaned.strip("_") or "value"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load newline-delimited JSON records from disk."""

    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def uses_risk_scores(localizer_names: Sequence[str]) -> bool:
    """Return whether the requested localizers need per-commit risk scores."""

    return any(
        localizer_name
        in {
            "ProbabilisticBisection_PosteriorMedian_RiskAwarePrior",
            "RiskWeightedBisection",
            "RiskWeightedMultisection",
        }
        for localizer_name in localizer_names
    )


def load_risk_scores(path: Path) -> dict[str, float]:
    """Load validated per-commit risk scores keyed by revision hash."""

    scores: dict[str, float] = {}
    for row_index, raw in enumerate(load_jsonl(path), start=1):
        commit_id = raw.get("commit_id")
        if not isinstance(commit_id, str) or not commit_id:
            raise ValueError(f"{path}: row {row_index} has invalid commit_id")
        if commit_id in scores:
            raise ValueError(f"{path}: duplicate commit_id {commit_id!r}")

        raw_score = raw.get("risk_score")
        try:
            risk_score = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}: row {row_index} has invalid risk_score {raw_score!r}"
            ) from exc
        if not math.isfinite(risk_score) or not 0.0 <= risk_score <= 1.0:
            raise ValueError(
                f"{path}: row {row_index} risk_score must be finite and in [0, 1]: "
                f"{raw_score!r}"
            )
        scores[commit_id] = risk_score

    return scores


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


def load_matplotlib() -> Any:
    """Import matplotlib lazily and configure a non-interactive backend."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for distribution plot generation. "
            "Install matplotlib or rerun without --draw-dist-plots."
        ) from exc
    return plt


if __name__ == "__main__":
    raise SystemExit(main())
