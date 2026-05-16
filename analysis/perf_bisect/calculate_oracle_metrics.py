#!/usr/bin/env python3
"""Calculate real-oracle measurement accuracy for perf-bisect regressions.

The output contains one row per regression from the eval and final-test
perf-bisect datasets:

  - regression_id
  - summary_oracle_accuracy
  - replicate_oracle_accuracy

Candidate revisions are found from the Mercurial parent graph in
all_commits.jsonl. The known-good revision is excluded, and the known-bad
revision is included because it can be the culprit revision.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "datasets" / "mozilla_perf_bisect"
DEFAULT_REVISION_DATA = DEFAULT_DATA_DIR / "per_revision_perf_data.jsonl"
DEFAULT_COMMITS = DEFAULT_DATA_DIR / "all_commits.jsonl"
DEFAULT_REGRESSION_INPUTS = (
    DEFAULT_DATA_DIR / "perf_bisect_regressions_eval.jsonl",
    DEFAULT_DATA_DIR / "perf_bisect_regressions_final_test.jsonl",
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "analysis" / "perf_bisect" / "per_regression_oracle_metrics.jsonl"
)
DEFAULT_SMOOTHING_ALPHA = 0.5
MINIMUM_ORACLE_ACCURACY = 0.51
NULL_NODE = "0000000000000000000000000000000000000000"


@dataclass(frozen=True)
class Regression:
    """A single perf-bisect regression row."""

    regression_id: int
    good_revision: str
    bad_revision: str
    culprit_revision: str
    signature_id: int
    good_value: float
    bad_value: float
    num_candidate_revisions: int | None = None

    @property
    def baseline(self) -> float:
        """Return the midpoint between the known-good and known-bad values."""

        return (self.good_value + self.bad_value) / 2.0


@dataclass
class MeasurementValues:
    """Summary and replicate values for one revision/signature pair."""

    summary: list[float] = field(default_factory=list)
    replicate: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class CommitGraph:
    """Mercurial parent graph indexed by node."""

    index_by_node: dict[str, int]
    parents_by_node: dict[str, list[str]]

    def path_between(self, good_revision: str, bad_revision: str) -> list[str] | None:
        """Return a parent-link path from good_revision to bad_revision."""

        good_index = self.index_by_node.get(good_revision)
        bad_index = self.index_by_node.get(bad_revision)
        if good_index is None or bad_index is None:
            return None
        if good_index > bad_index:
            return None
        if good_revision == bad_revision:
            return [good_revision]

        queue: deque[str] = deque([bad_revision])
        next_child_by_parent: dict[str, str | None] = {bad_revision: None}

        while queue:
            revision = queue.popleft()
            if revision == good_revision:
                break
            for parent in self.parents_by_node.get(revision, []):
                parent_index = self.index_by_node.get(parent)
                if parent_index is None or parent_index < good_index:
                    continue
                if parent not in next_child_by_parent:
                    next_child_by_parent[parent] = revision
                    queue.append(parent)

        if good_revision not in next_child_by_parent:
            return None

        path = []
        revision: str | None = good_revision
        while revision is not None:
            path.append(revision)
            revision = next_child_by_parent[revision]
        return path


def main(argv: Sequence[str] | None = None) -> int:
    """Load inputs, calculate per-regression accuracies, and write JSONL."""

    args = parse_args(argv)
    regressions = load_regressions(args.regressions)
    commit_graph = load_commit_graph(args.commits)
    paths_by_regression_id, path_stats = build_candidate_paths(
        regressions,
        commit_graph,
        exclude_bad_revision=args.exclude_bad_revision,
    )

    needed_revisions = {
        revision
        for path in paths_by_regression_id.values()
        for revision in path.candidate_revisions
    }
    needed_signatures = {regression.signature_id for regression in regressions}
    measurements = load_measurements(
        args.revision_data,
        needed_revisions=needed_revisions,
        needed_signatures=needed_signatures,
    )

    output_rows = [
        calculate_regression_metrics(
            regression,
            candidate_path=paths_by_regression_id.get(regression.regression_id),
            measurements=measurements,
            smoothing_alpha=args.smoothing_alpha,
        )
        for regression in regressions
    ]
    output_rows, excluded_rows = exclude_low_accuracy_rows(
        output_rows,
        minimum_accuracy=MINIMUM_ORACLE_ACCURACY,
    )
    write_jsonl(args.output, output_rows)

    metric_stats = summarize_output_rows(output_rows)
    print(f"loaded_regressions={len(regressions)}")
    print(f"candidate_revisions={len(needed_revisions)}")
    print(f"measurement_keys={len(measurements)}")
    for key, value in sorted(path_stats.items()):
        print(f"{key}={value}")
    print(f"oracle_metric_rows_excluded_low_accuracy={len(excluded_rows)}")
    for key, value in sorted(metric_stats.items()):
        print(f"{key}={value}")
    print(f"wrote {len(output_rows)} rows to {args.output}")
    return 0


@dataclass(frozen=True)
class CandidatePath:
    """Candidate revisions and the culprit boundary within the full path."""

    revisions_from_good_to_bad: list[str]
    candidate_revisions: list[str]
    culprit_index: int

    def is_expected_bad(self, revision: str) -> bool:
        """Return whether a candidate revision should measure above baseline."""

        return self.revisions_from_good_to_bad.index(revision) >= self.culprit_index


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision-data",
        type=Path,
        default=DEFAULT_REVISION_DATA,
        help="Path to per_revision_perf_data.jsonl.",
    )
    parser.add_argument(
        "--commits",
        type=Path,
        default=DEFAULT_COMMITS,
        help="Path to all_commits.jsonl.",
    )
    parser.add_argument(
        "--regressions",
        type=Path,
        nargs="+",
        default=list(DEFAULT_REGRESSION_INPUTS),
        help="Regression JSONL files to score, in output order.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--exclude-bad-revision",
        action="store_true",
        help=(
            "Exclude the known-bad endpoint from candidate measurements. By "
            "default it is included because bad_revision can be the culprit."
        ),
    )
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=DEFAULT_SMOOTHING_ALPHA,
        help=(
            "Positive additive smoothing prior for oracle accuracies. The "
            "default 0.5 uses Jeffreys smoothing: "
            "(correct + alpha) / (total + 2 * alpha)."
        ),
    )
    args = parser.parse_args(argv)
    if args.smoothing_alpha <= 0:
        parser.error("--smoothing-alpha must be positive")
    return args


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield JSON objects from a JSONL file with line numbers."""

    with path.open(encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_num}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"expected JSON object at {path}:{line_num}, "
                    f"got {type(record).__name__}"
                )
            yield line_num, record


def load_regressions(paths: Sequence[Path]) -> list[Regression]:
    """Load regression rows from one or more JSONL files."""

    regressions = []
    seen_ids: set[int] = set()
    for path in paths:
        for line_num, raw in iter_jsonl(path):
            regression = parse_regression(raw, path=path, line_num=line_num)
            if regression.regression_id in seen_ids:
                raise ValueError(
                    f"duplicate regression_id {regression.regression_id} "
                    f"at {path}:{line_num}"
                )
            seen_ids.add(regression.regression_id)
            regressions.append(regression)
    return regressions


def parse_regression(raw: Mapping[str, Any], *, path: Path, line_num: int) -> Regression:
    """Parse and validate the fields needed from one regression row."""

    failing_sig = raw.get("failing_sig")
    if not isinstance(failing_sig, Mapping):
        raise ValueError(f"missing failing_sig object at {path}:{line_num}")

    try:
        return Regression(
            regression_id=int(raw["regression_id"]),
            good_revision=str(raw["good_revision"]),
            bad_revision=str(raw["bad_revision"]),
            culprit_revision=str(raw["culprit_revision"]),
            signature_id=int(failing_sig["signature_id"]),
            good_value=float(failing_sig["Good_value"]),
            bad_value=float(failing_sig["bad_value"]),
            num_candidate_revisions=optional_int(raw.get("num_candidate_revisions")),
        )
    except KeyError as exc:
        raise ValueError(f"missing required field {exc} at {path}:{line_num}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid regression row at {path}:{line_num}: {exc}") from exc


def optional_int(value: Any) -> int | None:
    """Return an int for present values, otherwise None."""

    if value is None:
        return None
    return int(value)


def load_commit_graph(path: Path) -> CommitGraph:
    """Load Mercurial commit parent links from all_commits.jsonl."""

    index_by_node: dict[str, int] = {}
    parents_by_node: dict[str, list[str]] = {}
    for _, raw in iter_jsonl(path):
        node = raw.get("node")
        if not isinstance(node, str) or not node:
            continue
        if node in index_by_node:
            raise ValueError(f"duplicate commit node in {path}: {node}")

        index_by_node[node] = len(index_by_node)
        parents = [
            parent
            for parent in raw.get("parents", [])
            if isinstance(parent, str) and parent and parent != NULL_NODE
        ]
        parents_by_node[node] = parents
    return CommitGraph(index_by_node=index_by_node, parents_by_node=parents_by_node)


def build_candidate_paths(
    regressions: Sequence[Regression],
    commit_graph: CommitGraph,
    *,
    exclude_bad_revision: bool,
) -> tuple[dict[int, CandidatePath], Counter[str]]:
    """Build candidate revision paths for all regressions."""

    paths: dict[int, CandidatePath] = {}
    stats: Counter[str] = Counter()
    for regression in regressions:
        full_path = commit_graph.path_between(
            regression.good_revision,
            regression.bad_revision,
        )
        if full_path is None or len(full_path) < 2:
            stats["path_missing"] += 1
            continue
        if regression.culprit_revision not in full_path:
            stats["culprit_not_in_path"] += 1
            continue

        candidates = full_path[1:-1] if exclude_bad_revision else full_path[1:]
        if regression.num_candidate_revisions is not None:
            interior_count = max(len(full_path) - 2, 0)
            if regression.num_candidate_revisions != interior_count:
                stats["stored_candidate_count_mismatch"] += 1

        paths[regression.regression_id] = CandidatePath(
            revisions_from_good_to_bad=full_path,
            candidate_revisions=candidates,
            culprit_index=full_path.index(regression.culprit_revision),
        )
    return paths, stats


def load_measurements(
    path: Path,
    *,
    needed_revisions: set[str],
    needed_signatures: set[int],
) -> dict[tuple[str, int], MeasurementValues]:
    """Load summary and replicate values for needed revision/signature pairs."""

    measurements: dict[tuple[str, int], MeasurementValues] = {}
    if not needed_revisions or not needed_signatures:
        return measurements

    for _, raw in iter_jsonl(path):
        node = raw.get("node")
        if node not in needed_revisions:
            continue

        for measurement in raw.get("perf_measurement_data", []):
            if not isinstance(measurement, Mapping):
                continue
            try:
                signature_id = int(measurement["signature_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if signature_id not in needed_signatures:
                continue

            try:
                value = float(measurement["value"])
            except (KeyError, TypeError, ValueError):
                continue

            values = measurements.setdefault((str(node), signature_id), MeasurementValues())
            if measurement.get("replicate") is True:
                values.replicate.append(value)
            elif measurement.get("replicate") is False:
                values.summary.append(value)

    return measurements


def calculate_regression_metrics(
    regression: Regression,
    *,
    candidate_path: CandidatePath | None,
    measurements: Mapping[tuple[str, int], MeasurementValues],
    smoothing_alpha: float,
) -> dict[str, int | float | None]:
    """Calculate summary and replicate oracle accuracies for one regression."""

    summary_correct = 0
    summary_total = 0
    replicate_correct = 0
    replicate_total = 0

    if candidate_path is not None:
        for revision in candidate_path.candidate_revisions:
            values = measurements.get((revision, regression.signature_id))
            if values is None:
                continue
            expected_bad = candidate_path.is_expected_bad(revision)
            correct_summary, total_summary = score_values(
                values.summary,
                baseline=regression.baseline,
                expected_bad=expected_bad,
            )
            correct_replicate, total_replicate = score_values(
                values.replicate,
                baseline=regression.baseline,
                expected_bad=expected_bad,
            )
            summary_correct += correct_summary
            summary_total += total_summary
            replicate_correct += correct_replicate
            replicate_total += total_replicate

    return {
        "regression_id": regression.regression_id,
        "summary_oracle_accuracy": accuracy(
            summary_correct,
            summary_total,
            smoothing_alpha=smoothing_alpha,
        ),
        "replicate_oracle_accuracy": accuracy(
            replicate_correct,
            replicate_total,
            smoothing_alpha=smoothing_alpha,
        ),
    }


def score_values(
    values: Sequence[float],
    *,
    baseline: float,
    expected_bad: bool,
) -> tuple[int, int]:
    """Return correct and total counts for values compared to the baseline."""

    correct = 0
    for value in values:
        if expected_bad:
            correct += value > baseline
        else:
            correct += value < baseline
    return correct, len(values)


def accuracy(correct: int, total: int, *, smoothing_alpha: float) -> float | None:
    """Return smoothed correct / total, or None when no values were available."""

    if total == 0:
        return None
    return (correct + smoothing_alpha) / (total + 2 * smoothing_alpha)


def exclude_low_accuracy_rows(
    rows: Sequence[dict[str, int | float | None]],
    *,
    minimum_accuracy: float,
) -> tuple[list[dict[str, int | float | None]], list[dict[str, int | float | None]]]:
    """Drop rows where either available oracle accuracy is below the threshold."""

    kept_rows: list[dict[str, int | float | None]] = []
    excluded_rows: list[dict[str, int | float | None]] = []
    for row in rows:
        summary_accuracy = row["summary_oracle_accuracy"]
        replicate_accuracy = row["replicate_oracle_accuracy"]
        should_exclude = (
            (summary_accuracy is not None and summary_accuracy < minimum_accuracy)
            or (
                replicate_accuracy is not None
                and replicate_accuracy < minimum_accuracy
            )
        )
        if should_exclude:
            excluded_rows.append(row)
            print(
                "[WARN] Excluding oracle metric row for "
                f"regression_id={row['regression_id']}: "
                f"summary_oracle_accuracy={summary_accuracy}, "
                f"replicate_oracle_accuracy={replicate_accuracy}, "
                f"minimum_accuracy={minimum_accuracy}."
            )
        else:
            kept_rows.append(row)
    return kept_rows, excluded_rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write rows to a JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row))
            fh.write("\n")


def summarize_output_rows(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    """Summarize nullable metric counts for logging."""

    stats: Counter[str] = Counter()
    for row in rows:
        if row["summary_oracle_accuracy"] is None:
            stats["summary_accuracy_null"] += 1
        else:
            stats["summary_accuracy_present"] += 1
        if row["replicate_oracle_accuracy"] is None:
            stats["replicate_accuracy_null"] += 1
        else:
            stats["replicate_accuracy_present"] += 1
    return stats


if __name__ == "__main__":
    raise SystemExit(main())
