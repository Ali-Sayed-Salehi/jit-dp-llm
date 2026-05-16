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

DATASETS = {
    "eval": "perf_bisect_regressions_eval.jsonl",
    "final_test": "perf_bisect_regressions_final_test.jsonl",
}


@dataclass(frozen=True)
class OracleSpec:
    """Factory descriptor for a registered test oracle implementation."""

    name: str

    def build(
        self,
        *,
        signature_info: SignatureInfoIndex,
        revision_perf: RevisionPerfIndex,
        workers: int,
        test_duration_minutes: float,
        random_seed: int | None,
    ) -> TestOracle:
        """Construct the concrete oracle for one independent regression run."""

        if self.name == SummaryComparison.name:
            return SummaryComparison(
                signature_info=signature_info,
                revision_perf=revision_perf,
                executor=TestExecutor(
                    workers=workers,
                    test_duration_minutes=test_duration_minutes,
                ),
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
            signature_info=signature_info,
            revision_perf=revision_perf,
            oracle_names=args.oracles,
            localizer_names=args.localizers,
            workers=args.workers,
            test_duration_minutes=args.test_duration_minutes,
            backfill_non_monotonic_retrigger_count=(
                args.backfill_non_monotonic_retrigger_count
            ),
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
        help="Path to per_revision_perf_data.jsonl.",
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
        "--backfill-non-monotonic-retrigger-count",
        type=int,
        default=2,
        help=(
            "Number of adjacent non-monotonic Backfill intervals to retrigger "
            "before leaving the localization undefined."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Seed for random same-side fallback draws.",
    )
    args = parser.parse_args(argv)
    if args.backfill_non_monotonic_retrigger_count < 0:
        parser.error("--backfill-non-monotonic-retrigger-count must be non-negative")
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
    signature_info: SignatureInfoIndex,
    revision_perf: RevisionPerfIndex,
    oracle_names: Sequence[str],
    localizer_names: Sequence[str],
    workers: int,
    test_duration_minutes: float,
    backfill_non_monotonic_retrigger_count: int,
    random_seed: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run every requested localizer/oracle combination for one regression split."""

    summary_runs = []
    detail_runs = []

    for localizer_name in localizer_names:
        localizer = build_localizer(
            localizer_name,
            backfill_non_monotonic_retrigger_count=(
                backfill_non_monotonic_retrigger_count
            ),
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
        },
        "settings": {
            "workers": workers,
            "test_duration_minutes": test_duration_minutes,
            "backfill_non_monotonic_retrigger_count": (
                backfill_non_monotonic_retrigger_count
            ),
            "random_seed": random_seed,
            "random_seed_derivation": (
                "random_seed + regression_id - 1 when available; "
                "otherwise random_seed + split row index"
            ),
        },
    }
    return (
        {**base_output, "runs": summary_runs},
        {**base_output, "runs": detail_runs},
    )


def build_localizer(
    localizer_name: str,
    *,
    backfill_non_monotonic_retrigger_count: int,
) -> CulpritLocalizer:
    """Construct a localizer from the registry."""

    localizer_cls = LOCALIZERS[localizer_name]
    if localizer_name == "Backfill":
        return localizer_cls(
            backfill_non_monotonic_retrigger_count=(
                backfill_non_monotonic_retrigger_count
            )
        )
    return localizer_cls()


def run_one_regression(
    *,
    regression_index: int,
    regression: Mapping[str, Any],
    localizer: CulpritLocalizer,
    oracle_spec: OracleSpec,
    signature_info: SignatureInfoIndex,
    revision_perf: RevisionPerfIndex,
    workers: int,
    test_duration_minutes: float,
    random_seed: int | None,
):
    """Run one regression with a fresh oracle and test executor."""

    regression_seed_offset = regression_seed_index(
        regression,
        fallback_index=regression_index,
    )
    regression_seed = (
        None
        if random_seed is None
        else random_seed + regression_seed_offset
    )
    oracle = oracle_spec.build(
        signature_info=signature_info,
        revision_perf=revision_perf,
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load newline-delimited JSON records from disk."""

    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def regression_seed_index(
    regression: Mapping[str, Any],
    *,
    fallback_index: int,
) -> int:
    """Return the stable per-regression seed offset."""

    regression_id = parse_regression_id(regression, context="regression seed")
    if regression_id is None:
        return fallback_index
    return regression_id - 1


def parse_regression_id(
    regression: Mapping[str, Any],
    *,
    context: str,
) -> int | None:
    """Parse an optional positive regression_id from one regression row."""

    raw_id = regression.get("regression_id")
    if raw_id is None:
        return None
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
    """Load revision graph nodes plus summary and replicate measurements."""

    records = []
    for raw in load_jsonl(path):
        summary_values: dict[int, list[float]] = {}
        replicate_values: dict[int, list[float]] = {}
        for measurement in raw.get("perf_measurement_data", []):
            signature_id = int(measurement["signature_id"])
            value = float(measurement["value"])
            if measurement.get("replicate") is True:
                replicate_values.setdefault(signature_id, []).append(value)
            elif measurement.get("replicate") is False:
                summary_values.setdefault(signature_id, []).append(value)

        records.append(
            RevisionRecord(
                node=str(raw["node"]),
                parents=[str(parent) for parent in raw.get("parents", [])],
                date=raw.get("date"),
                summary_values=summary_values,
                replicate_values=replicate_values,
            )
        )
    return RevisionPerfIndex(records)


if __name__ == "__main__":
    raise SystemExit(main())
