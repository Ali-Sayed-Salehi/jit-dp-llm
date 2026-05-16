"""Run culprit-localization simulations on Mozilla performance regressions."""

from __future__ import annotations

import argparse
import json
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
        test_duration_minutes: float,
        random_seed: int | None,
    ) -> TestOracle:
        """Construct the concrete oracle for one independent regression run."""

        if self.name == SummaryComparison.name:
            return SummaryComparison(
                signature_info=signature_info,
                executor=TestExecutor(
                    workers=workers,
                    test_duration_minutes=test_duration_minutes,
                ),
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

    signature_info = load_signature_info(args.signature_info)
    revision_perf = load_revision_perf(args.revision_data)
    oracle_metrics = load_oracle_metrics(args.oracle_metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in dataset_names:
        regressions_path = resolve_regression_path(
            args.regression_dir,
            DATASETS[dataset_name],
        )
        regressions = load_jsonl(regressions_path)
        summary_output, details_output = run_dataset(
            dataset_name=dataset_name,
            regressions=regressions,
            regressions_path=regressions_path,
            signature_info_path=args.signature_info,
            revision_data_path=args.revision_data,
            oracle_metrics_path=args.oracle_metrics,
            signature_info=signature_info,
            revision_perf=revision_perf,
            oracle_metrics=oracle_metrics,
            oracle_names=args.oracles,
            localizer_names=args.localizers,
            workers=args.workers,
            test_duration_minutes=args.test_duration_minutes,
            backfill_retrigger_count=args.backfill_retrigger_count,
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
        "--test-duration-minutes",
        type=float,
        default=1.0,
        help="Simulated duration of one performance test run.",
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
            "adjacent non-monotonic intervals and all-clean sequences."
        ),
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
    test_duration_minutes: float,
    backfill_retrigger_count: int,
    random_seed: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run every requested localizer/oracle combination for one regression split."""

    summary_runs = []
    detail_runs = []

    for localizer_name in localizer_names:
        localizer = build_localizer(
            localizer_name,
            backfill_retrigger_count=backfill_retrigger_count,
        )
        for oracle_name in oracle_names:
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
                    test_duration_minutes=test_duration_minutes,
                    random_seed=random_seed,
                )
                for regression_index, regression in enumerate(regressions)
            ]
            summary_runs.append(
                {
                    "localizer": localizer_name,
                    "test_oracle": oracle_name,
                    "metrics": compute_metrics(results),
                }
            )
            print_undefined_localizations(
                dataset_name=dataset_name,
                localizer_name=localizer_name,
                oracle_name=oracle_name,
                results=results,
            )
            detail_runs.append(
                {
                    "localizer": localizer_name,
                    "test_oracle": oracle_name,
                    "results": [result.to_json() for result in results],
                }
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
            "test_duration_minutes": test_duration_minutes,
            "backfill_retrigger_count": backfill_retrigger_count,
            "random_seed": random_seed,
            "random_seed_derivation": "random_seed + regression_id - 1",
        },
    }
    return (
        {**base_output, "runs": summary_runs},
        {**base_output, "runs": detail_runs},
    )


def build_localizer(
    localizer_name: str,
    *,
    backfill_retrigger_count: int,
) -> CulpritLocalizer:
    """Construct a localizer from the registry."""

    localizer_cls = LOCALIZERS[localizer_name]
    if localizer_name == "Backfill":
        return localizer_cls(backfill_retrigger_count=backfill_retrigger_count)
    return localizer_cls()


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
    test_duration_minutes: float,
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
        test_duration_minutes=test_duration_minutes,
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
        result.trtc_minutes
        for result in successes
        if result.trtc_minutes is not None
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
        "mean_trtc_minutes": round(mean(trtc_values), 1) if trtc_values else None,
        "max_trtc_minutes": round(max(trtc_values), 1) if trtc_values else None,
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
    """Load per-signature replicate counts."""

    infos = []
    for raw in load_jsonl(path):
        infos.append(
            SignatureInfo(
                signature_id=int(raw["signature_id"]),
                replicate_count=int(raw["replicate_counts"]),
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
