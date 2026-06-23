#!/usr/bin/env python3
"""Calculate real-oracle summary measurement accuracy for perf-bisect regressions.

The output contains one row per scorable regression from the eval and
final-test perf-bisect datasets:

  - regression_id
  - summary_oracle_accuracy

Rows with no available summary measurements are excluded instead of being
written with a null accuracy.

By default, the script also writes a smoothed distribution plot of the oracle
accuracy distribution beside the JSONL output.

Candidate revisions are found from the Mercurial parent graph in the reduced
per_revision_perf_data.jsonl by default. The known-good revision is excluded,
and the known-bad revision is included because it can be the culprit revision.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass, field
import json
import logging
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "datasets" / "mozilla_perf_bisect_v2" / "reduced"
DEFAULT_REVISION_DATA = DEFAULT_DATA_DIR / "per_revision_perf_data.jsonl"
DEFAULT_COMMITS = DEFAULT_REVISION_DATA
DEFAULT_REGRESSION_INPUTS = (
    DEFAULT_DATA_DIR / "perf_bisect_regressions_eval.jsonl",
    DEFAULT_DATA_DIR / "perf_bisect_regressions_final_test.jsonl",
)
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "per_regression_oracle_metrics_v2.jsonl"
DEFAULT_DISTRIBUTION_PLOT_DPI = 200
DEFAULT_DISTRIBUTION_PLOT_FIGSIZE = (6.0, 3.6)
DEFAULT_DISTRIBUTION_CURVE_POINTS = 320
DEFAULT_DISTRIBUTION_REFERENCE_BINS = 20
DEFAULT_SMOOTHING_ALPHA = 0.5
MINIMUM_ORACLE_ACCURACY = 0.51
NULL_NODE = "0000000000000000000000000000000000000000"
PLOT_ORACLE_ACCURACY_FIELD = "summary_oracle_accuracy"
PLOT_TITLE_FONT_SIZE = 16
PLOT_LABEL_FONT_SIZE = 13
PLOT_TICK_FONT_SIZE = 11
PLOT_STATS_FONT_SIZE = 12


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
    """Summary values for one revision/signature pair."""

    summary: list[float] = field(default_factory=list)


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
    configure_logging(args.log_level)
    logger.info("Starting perf-bisect oracle metric calculation.")
    logger.info("Regression inputs: %s", ", ".join(str(p) for p in args.regressions))
    logger.info("Revision data: %s", args.revision_data)
    logger.info("Commit graph: %s", args.commits)
    logger.info("Output: %s", args.output)

    if args.skip_plot:
        logger.info("Step 1/7: Plot generation disabled.")
        plt = None
    else:
        logger.info("Step 1/7: Loading matplotlib for distribution plot.")
        plt = load_matplotlib()

    logger.info("Step 2/7: Loading regression rows.")
    regressions = load_regressions(args.regressions)
    needed_signatures = {regression.signature_id for regression in regressions}

    logger.info("Step 3/7: Loading commit graph.")
    preloaded_measurements = None
    if same_path(args.commits, args.revision_data):
        logger.info(
            "Revision data is also the commit graph input; scanning it once "
            "for graph fields and needed summary measurements."
        )
        commit_graph, preloaded_measurements = (
            load_commit_graph_and_measurements(
                args.revision_data,
                needed_signatures=needed_signatures,
            )
        )
    else:
        commit_graph = load_commit_graph(args.commits)

    logger.info("Step 4/7: Building candidate revision paths.")
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
    logger.info(
        "Candidate path result: regressions=%d, paths=%d, "
        "candidate_revisions=%d, signatures=%d",
        len(regressions),
        len(paths_by_regression_id),
        len(needed_revisions),
        len(needed_signatures),
    )
    if path_stats:
        logger.info("Candidate path counters: %s", dict(sorted(path_stats.items())))

    logger.info("Step 5/7: Loading summary measurements for candidate paths.")
    if preloaded_measurements is not None:
        logger.info(
            "Using summary measurements preloaded during the graph scan; "
            "not rescanning revision data."
        )
        measurements = filter_measurements_by_revisions(
            preloaded_measurements,
            needed_revisions=needed_revisions,
        )
    else:
        measurements = load_measurements(
            args.revision_data,
            needed_revisions=needed_revisions,
            needed_signatures=needed_signatures,
        )

    logger.info("Step 6/7: Calculating and filtering oracle metrics.")
    output_rows = [
        calculate_regression_metrics(
            regression,
            candidate_path=paths_by_regression_id.get(regression.regression_id),
            measurements=measurements,
            smoothing_alpha=args.smoothing_alpha,
        )
        for regression in regressions
    ]
    output_rows, excluded_missing_rows, excluded_low_accuracy_rows = (
        exclude_unusable_or_low_accuracy_rows(
            output_rows,
            minimum_accuracy=MINIMUM_ORACLE_ACCURACY,
        )
    )
    logger.info(
        "Oracle metrics calculated: kept_rows=%d, "
        "excluded_missing_accuracy=%d, excluded_low_accuracy=%d",
        len(output_rows),
        len(excluded_missing_rows),
        len(excluded_low_accuracy_rows),
    )

    logger.info("Step 7/7: Writing oracle metric outputs.")
    write_jsonl(args.output, output_rows)
    if plt is not None:
        plot_output = args.plot_output or default_distribution_plot_path(args.output)
        logger.info("Writing oracle accuracy distribution plot: %s", plot_output)
        write_accuracy_distribution_plot(plt=plt, path=plot_output, rows=output_rows)

    metric_stats = summarize_output_rows(output_rows)
    print(f"loaded_regressions={len(regressions)}")
    print(f"candidate_revisions={len(needed_revisions)}")
    print(f"measurement_keys={len(measurements)}")
    for key, value in sorted(path_stats.items()):
        print(f"{key}={value}")
    print(
        "oracle_metric_rows_excluded_missing_accuracy="
        f"{len(excluded_missing_rows)}"
    )
    print(
        "oracle_metric_rows_excluded_low_accuracy="
        f"{len(excluded_low_accuracy_rows)}"
    )
    for key, value in sorted(metric_stats.items()):
        print(f"{key}={value}")
    print(f"wrote {len(output_rows)} rows to {args.output}")
    logger.info("Finished perf-bisect oracle metric calculation.")
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
        help=(
            "Path to a JSONL parent graph. Defaults to the reduced "
            "per_revision_perf_data.jsonl."
        ),
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
        "--plot-output",
        type=Path,
        default=None,
        help=(
            "Output PNG path for the oracle accuracy distribution plot. "
            "Defaults to '<output stem>_distribution.png' beside --output."
        ),
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Write the JSONL output but do not generate a distribution plot.",
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
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Python logging level.",
    )
    args = parser.parse_args(argv)
    if args.smoothing_alpha <= 0:
        parser.error("--smoothing-alpha must be positive")
    return args


def configure_logging(log_level: str) -> None:
    """Configure script logging."""

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def same_path(left: Path, right: Path) -> bool:
    """Return whether two path arguments refer to the same filesystem path."""

    return left.resolve() == right.resolve()


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
        before_count = len(regressions)
        for line_num, raw in iter_jsonl(path):
            regression = parse_regression(raw, path=path, line_num=line_num)
            if regression.regression_id in seen_ids:
                raise ValueError(
                    f"duplicate regression_id {regression.regression_id} "
                    f"at {path}:{line_num}"
                )
            seen_ids.add(regression.regression_id)
            regressions.append(regression)
        logger.info(
            "Loaded regressions: rows=%d, path=%s",
            len(regressions) - before_count,
            path,
        )
    logger.info("Loaded total regression rows: %d", len(regressions))
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
    """Load Mercurial commit parent links from a graph JSONL file."""

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
    logger.info(
        "Loaded commit graph: revisions=%d, parent_entries=%d, source=%s",
        len(index_by_node),
        sum(len(parents) for parents in parents_by_node.values()),
        path,
    )
    return CommitGraph(index_by_node=index_by_node, parents_by_node=parents_by_node)


def load_commit_graph_and_measurements(
    path: Path,
    *,
    needed_signatures: set[int],
) -> tuple[CommitGraph, dict[tuple[str, int], MeasurementValues]]:
    """Load graph fields and needed summary measurements in one file pass."""

    index_by_node: dict[str, int] = {}
    parents_by_node: dict[str, list[str]] = {}
    measurements: dict[tuple[str, int], MeasurementValues] = {}
    stats: Counter[str] = Counter()

    for _, raw in iter_jsonl(path):
        stats["revision_rows_scanned"] += 1
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

        for measurement in raw.get("perf_measurement_data", []):
            add_summary_measurement(
                measurements,
                node=str(node),
                measurement=measurement,
                needed_signatures=needed_signatures,
                stats=stats,
            )

    logger.info(
        "Loaded commit graph and summary measurements: revisions=%d, "
        "parent_entries=%d, measurement_keys=%d, source=%s",
        len(index_by_node),
        sum(len(parents) for parents in parents_by_node.values()),
        len(measurements),
        path,
    )
    logger.info("Combined graph/measurement scan counters: %s", dict(stats))
    return (
        CommitGraph(index_by_node=index_by_node, parents_by_node=parents_by_node),
        measurements,
    )


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
    logger.info(
        "Built candidate paths: paths=%d, missing=%d, culprit_not_in_path=%d",
        len(paths),
        stats["path_missing"],
        stats["culprit_not_in_path"],
    )
    return paths, stats


def filter_measurements_by_revisions(
    measurements: Mapping[tuple[str, int], MeasurementValues],
    *,
    needed_revisions: set[str],
) -> dict[tuple[str, int], MeasurementValues]:
    """Keep preloaded measurements only for candidate revisions."""

    filtered = {
        key: values
        for key, values in measurements.items()
        if key[0] in needed_revisions
    }
    logger.info(
        "Filtered preloaded measurements to candidate revisions: "
        "before_keys=%d, after_keys=%d, after_values=%d",
        len(measurements),
        len(filtered),
        sum(len(values.summary) for values in filtered.values()),
    )
    return filtered


def load_measurements(
    path: Path,
    *,
    needed_revisions: set[str],
    needed_signatures: set[int],
) -> dict[tuple[str, int], MeasurementValues]:
    """Load summary values for needed revision/signature pairs."""

    measurements: dict[tuple[str, int], MeasurementValues] = {}
    if not needed_revisions or not needed_signatures:
        logger.info("No revisions or signatures requested; skipping measurement load.")
        return measurements

    stats: Counter[str] = Counter()
    for _, raw in iter_jsonl(path):
        stats["revision_rows_scanned"] += 1
        node = raw.get("node")
        if node not in needed_revisions:
            continue
        stats["candidate_revision_rows"] += 1

        for measurement in raw.get("perf_measurement_data", []):
            add_summary_measurement(
                measurements,
                node=str(node),
                measurement=measurement,
                needed_signatures=needed_signatures,
                stats=stats,
            )

    logger.info(
        "Loaded measurements: keys=%d, stats=%s",
        len(measurements),
        dict(sorted(stats.items())),
    )
    return measurements


def add_summary_measurement(
    measurements: dict[tuple[str, int], MeasurementValues],
    *,
    node: str,
    measurement: Any,
    needed_signatures: set[int],
    stats: Counter[str],
) -> None:
    """Add one summary measurement if it belongs to a needed signature."""

    stats["measurements_seen"] += 1
    if not isinstance(measurement, Mapping):
        stats["measurement_not_object"] += 1
        return
    if measurement.get("replicate") is not False:
        stats["measurement_not_summary"] += 1
        return
    try:
        signature_id = int(measurement["signature_id"])
    except (KeyError, TypeError, ValueError):
        stats["measurement_bad_signature"] += 1
        return
    if signature_id not in needed_signatures:
        stats["measurement_unneeded_signature"] += 1
        return

    try:
        value = float(measurement["value"])
    except (KeyError, TypeError, ValueError):
        stats["measurement_bad_value"] += 1
        return

    values = measurements.setdefault((node, signature_id), MeasurementValues())
    values.summary.append(value)
    stats["summary_values_loaded"] += 1


def calculate_regression_metrics(
    regression: Regression,
    *,
    candidate_path: CandidatePath | None,
    measurements: Mapping[tuple[str, int], MeasurementValues],
    smoothing_alpha: float,
) -> dict[str, int | float | None]:
    """Calculate summary oracle accuracy for one regression."""

    summary_correct = 0
    summary_total = 0

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
            summary_correct += correct_summary
            summary_total += total_summary

    return {
        "regression_id": regression.regression_id,
        "summary_oracle_accuracy": accuracy(
            summary_correct,
            summary_total,
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


def exclude_unusable_or_low_accuracy_rows(
    rows: Sequence[dict[str, int | float | None]],
    *,
    minimum_accuracy: float,
) -> tuple[
    list[dict[str, int | float | None]],
    list[dict[str, int | float | None]],
    list[dict[str, int | float | None]],
]:
    """Drop rows without accuracy or below the accuracy threshold."""

    kept_rows: list[dict[str, int | float | None]] = []
    excluded_missing_rows: list[dict[str, int | float | None]] = []
    excluded_low_accuracy_rows: list[dict[str, int | float | None]] = []
    for row in rows:
        summary_accuracy = row["summary_oracle_accuracy"]
        if summary_accuracy is None:
            excluded_missing_rows.append(row)
            logger.warning(
                "Excluding oracle metric row for regression_id=%s: "
                "no summary measurements available.",
                row["regression_id"],
            )
        elif summary_accuracy < minimum_accuracy:
            excluded_low_accuracy_rows.append(row)
            logger.warning(
                "Excluding oracle metric row for regression_id=%s: "
                "summary_oracle_accuracy=%s, minimum_accuracy=%s.",
                row["regression_id"],
                summary_accuracy,
                minimum_accuracy,
            )
        else:
            kept_rows.append(row)
    return kept_rows, excluded_missing_rows, excluded_low_accuracy_rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write rows to a JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row))
            fh.write("\n")
    logger.info("Wrote oracle metrics JSONL: path=%s", path)


def default_distribution_plot_path(output_path: Path) -> Path:
    """Return the default distribution plot path for one metrics output path."""

    return output_path.with_name(f"{output_path.stem}_distribution.png")


def write_accuracy_distribution_plot(
    *,
    plt: Any,
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write a smoothed curve showing the oracle accuracy distribution."""

    values = oracle_accuracy_values(rows, PLOT_ORACLE_ACCURACY_FIELD)
    if not values:
        print("no oracle accuracy values available for distribution plot")
        return

    min_accuracy = min(values)
    median_accuracy = median(values)
    max_accuracy = max(values)
    lower_bound = min(0.5, min_accuracy)
    upper_bound = max(1.0, max_accuracy)
    if math.isclose(lower_bound, upper_bound):
        lower_bound = max(0.0, lower_bound - 0.05)
        upper_bound = min(1.0, upper_bound + 0.05)

    fig, ax = plt.subplots(figsize=DEFAULT_DISTRIBUTION_PLOT_FIGSIZE)
    color = "#2563eb"
    x_values, y_values = smoothed_histogram_curve(
        values,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        points=DEFAULT_DISTRIBUTION_CURVE_POINTS,
        reference_bins=DEFAULT_DISTRIBUTION_REFERENCE_BINS,
    )
    ax.plot(x_values, y_values, linewidth=2.4, color=color)
    ax.fill_between(x_values, y_values, color=color, alpha=0.10)
    ax.text(
        0.03,
        0.95,
        (
            f"Min: {min_accuracy:.3f}\n"
            f"Median: {median_accuracy:.3f}\n"
            f"Max: {max_accuracy:.3f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PLOT_STATS_FONT_SIZE,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#d1d5db",
            "alpha": 0.9,
        },
    )

    ax.set_xlabel("Oracle accuracy", fontsize=PLOT_LABEL_FONT_SIZE)
    ax.set_ylabel("Smoothed regression count", fontsize=PLOT_LABEL_FONT_SIZE)
    ax.set_title("Oracle Accuracy Distribution", fontsize=PLOT_TITLE_FONT_SIZE)
    ax.set_xlim(lower_bound, upper_bound)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="both", labelsize=PLOT_TICK_FONT_SIZE)
    ax.grid(True, linestyle=":", linewidth=0.7)
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DEFAULT_DISTRIBUTION_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote oracle accuracy distribution plot: %s", path)
    print(f"wrote {path}")


def smoothed_histogram_curve(
    values: Sequence[float],
    *,
    lower_bound: float,
    upper_bound: float,
    points: int,
    reference_bins: int,
) -> tuple[list[float], list[float]]:
    """Return a Gaussian-smoothed histogram curve scaled to bin counts."""

    if points < 2:
        raise ValueError("points must be at least 2")
    if reference_bins < 1:
        raise ValueError("reference_bins must be positive")

    domain_width = upper_bound - lower_bound
    if domain_width <= 0:
        return [lower_bound], [float(len(values))]

    bandwidth = estimate_distribution_bandwidth(values, domain_width)
    bin_width = domain_width / reference_bins
    kernel_scale = bin_width / (bandwidth * math.sqrt(2.0 * math.pi))
    x_values = [
        lower_bound + domain_width * i / (points - 1)
        for i in range(points)
    ]
    y_values = [
        sum(
            reflected_gaussian_kernel(
                x,
                value,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                bandwidth=bandwidth,
            )
            for value in values
        )
        * kernel_scale
        for x in x_values
    ]
    return x_values, y_values


def reflected_gaussian_kernel(
    x_value: float,
    sample: float,
    *,
    lower_bound: float,
    upper_bound: float,
    bandwidth: float,
) -> float:
    """Return a boundary-reflected Gaussian kernel value."""

    reflected_below = 2.0 * lower_bound - sample
    reflected_above = 2.0 * upper_bound - sample
    return sum(
        math.exp(-0.5 * ((x_value - reflected_sample) / bandwidth) ** 2)
        for reflected_sample in (sample, reflected_below, reflected_above)
    )


def estimate_distribution_bandwidth(
    values: Sequence[float],
    domain_width: float,
) -> float:
    """Estimate a KDE bandwidth that stays smooth for narrow accuracy ranges."""

    minimum_bandwidth = domain_width / 80.0
    maximum_bandwidth = domain_width / 5.0
    if len(values) < 2:
        return minimum_bandwidth

    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (
        len(values) - 1
    )
    standard_deviation = math.sqrt(max(variance, 0.0))
    if standard_deviation <= 0:
        return minimum_bandwidth

    bandwidth = 1.06 * standard_deviation * len(values) ** -0.2
    return min(max(bandwidth, minimum_bandwidth), maximum_bandwidth)


def oracle_accuracy_values(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> list[float]:
    """Return finite numeric oracle accuracy values for one output field."""

    values = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        value = float(value)
        if math.isfinite(value):
            values.append(value)
    return values


def load_matplotlib() -> Any:
    """Import matplotlib for headless plot generation."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plot generation. Install matplotlib "
            "or rerun with --skip-plot."
        ) from exc
    return plt


def summarize_output_rows(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    """Summarize metric counts for logging."""

    stats: Counter[str] = Counter()
    for row in rows:
        if row["summary_oracle_accuracy"] is None:
            stats["summary_accuracy_null"] += 1
            continue
        stats["summary_accuracy_present"] += 1
    return stats


if __name__ == "__main__":
    raise SystemExit(main())
