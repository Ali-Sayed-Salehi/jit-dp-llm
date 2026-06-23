#!/usr/bin/env python3
"""Count summary performance test jobs across each perf-bisect interval.

For every regression row, this script reconstructs the Mercurial parent-link
path from the known-good revision to the known-bad revision, excludes both
endpoints, and counts cached Treeherder `performance/summary` measurements on
those interior revisions where `replicate` is exactly `false` and `signature_id`
matches the regression's failing performance signature.

The default inputs are the Mozilla perf-bisect v2 regression files and the
reduced v2 per-revision performance cache. Output rows are filtered to
regressions present in the reduced v2 oracle-metrics JSONL. The default output
is:

  datasets/mozilla_perf_bisect_v2/per_regression_test_job_counts.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "datasets" / "mozilla_perf_bisect_v2"
DEFAULT_REDUCED_DATA_DIR = DEFAULT_DATA_DIR / "reduced"
DEFAULT_REVISION_DATA = DEFAULT_REDUCED_DATA_DIR / "per_revision_perf_data.jsonl"
DEFAULT_COMMITS = DEFAULT_REVISION_DATA
DEFAULT_ORACLE_METRICS = DEFAULT_REDUCED_DATA_DIR / "per_regression_oracle_metrics_v2.jsonl"
DEFAULT_REGRESSION_INPUTS = (
    DEFAULT_DATA_DIR / "perf_bisect_regressions_eval.jsonl",
    DEFAULT_DATA_DIR / "perf_bisect_regressions_final_test.jsonl",
)
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "per_regression_test_job_counts.jsonl"
NULL_NODE = "0000000000000000000000000000000000000000"


@dataclass(frozen=True)
class Regression:
    """A single perf-bisect regression row."""

    regression_id: int
    good_revision: str
    bad_revision: str
    culprit_revision: str
    signature_id: int
    alert_summary_id: int | None = None
    num_candidate_revisions: int | None = None


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
    """Load inputs, count per-regression jobs, and write JSONL."""

    args = parse_args(argv)
    configure_logging(args.log_level)

    logger.info("Starting per-regression summary job count calculation.")
    logger.info("Regression inputs: %s", ", ".join(str(p) for p in args.regressions))
    logger.info("Revision data: %s", args.revision_data)
    logger.info("Commit graph: %s", args.commits)
    logger.info("Oracle metrics allowlist: %s", args.oracle_metrics)
    logger.info("Output: %s", args.output)

    logger.info("Step 1/6: Loading oracle-metric regression allowlist.")
    oracle_regression_ids = load_oracle_regression_ids(args.oracle_metrics)

    logger.info("Step 2/6: Loading regression rows.")
    regressions = load_regressions(args.regressions)
    regressions = filter_regressions_to_oracle_metrics(
        regressions,
        oracle_regression_ids=oracle_regression_ids,
    )
    needed_signature_ids = {regression.signature_id for regression in regressions}
    logger.info(
        "Loaded regressed signatures needed for oracle-filtered rows: %d",
        len(needed_signature_ids),
    )

    logger.info("Step 3/6: Loading commit graph and summary job counts by signature.")
    if same_path(args.commits, args.revision_data):
        commit_graph, summary_counts_by_revision_signature, scan_stats = (
            load_commit_graph_and_summary_counts(
                args.revision_data,
                needed_signature_ids=needed_signature_ids,
            )
        )
    else:
        commit_graph = load_commit_graph(args.commits)
        summary_counts_by_revision_signature, scan_stats = load_summary_counts(
            args.revision_data,
            needed_signature_ids=needed_signature_ids,
        )
    logger.info("Revision data scan counters: %s", dict(sorted(scan_stats.items())))

    logger.info("Step 4/6: Building regression interval rows.")
    output_rows, row_stats = build_output_rows(
        regressions,
        commit_graph=commit_graph,
        summary_counts_by_revision_signature=summary_counts_by_revision_signature,
    )
    logger.info("Regression interval counters: %s", dict(sorted(row_stats.items())))

    logger.info("Step 5/6: Writing output JSONL.")
    write_jsonl(args.output, output_rows)

    logger.info("Step 6/6: Calculating aggregate statistics.")
    count_stats = summarize_output_rows(output_rows)
    for key, value in sorted(count_stats.items()):
        logger.info("%s=%s", key, value)
        print(f"{key}={value}")
    print(f"wrote {len(output_rows)} rows to {args.output}")

    logger.info("Finished per-regression summary job count calculation.")
    return 0


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
        help="Regression JSONL files to count, in output order.",
    )
    parser.add_argument(
        "--oracle-metrics",
        type=Path,
        default=DEFAULT_ORACLE_METRICS,
        help=(
            "Oracle metrics JSONL whose regression_id values are used as an "
            "output allowlist."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Python logging level.",
    )
    return parser.parse_args(argv)


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


def load_oracle_regression_ids(path: Path) -> set[int]:
    """Load regression IDs present in the oracle-metrics dataset."""

    regression_ids: set[int] = set()
    for line_num, raw in iter_jsonl(path):
        try:
            regression_id = int(raw["regression_id"])
        except KeyError as exc:
            raise ValueError(
                f"missing required field {exc} at {path}:{line_num}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid regression_id at {path}:{line_num}: {exc}"
            ) from exc
        if regression_id in regression_ids:
            logger.warning(
                "Duplicate regression_id=%s in oracle metrics at %s:%d.",
                regression_id,
                path,
                line_num,
            )
        regression_ids.add(regression_id)

    logger.info(
        "Loaded oracle-metric regression allowlist: rows=%d, source=%s",
        len(regression_ids),
        path,
    )
    return regression_ids


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


def filter_regressions_to_oracle_metrics(
    regressions: Sequence[Regression],
    *,
    oracle_regression_ids: set[int],
) -> list[Regression]:
    """Keep only regressions whose ID is present in oracle metrics."""

    filtered_regressions = [
        regression
        for regression in regressions
        if regression.regression_id in oracle_regression_ids
    ]
    missing_from_regressions = oracle_regression_ids - {
        regression.regression_id for regression in regressions
    }
    logger.info(
        "Filtered regressions to oracle metrics allowlist: before=%d, after=%d, "
        "allowlist_ids=%d, oracle_ids_missing_from_regressions=%d",
        len(regressions),
        len(filtered_regressions),
        len(oracle_regression_ids),
        len(missing_from_regressions),
    )
    if missing_from_regressions:
        logger.warning(
            "Oracle metric regression IDs missing from regression inputs: %s",
            sorted(missing_from_regressions)[:20],
        )
    return filtered_regressions


def parse_regression(raw: Mapping[str, Any], *, path: Path, line_num: int) -> Regression:
    """Parse and validate the fields needed from one regression row."""

    try:
        return Regression(
            regression_id=int(raw["regression_id"]),
            alert_summary_id=optional_int(raw.get("alert_summary_id")),
            good_revision=str(raw["good_revision"]),
            bad_revision=str(raw["bad_revision"]),
            culprit_revision=str(raw["culprit_revision"]),
            signature_id=parse_failing_signature_id(raw, path=path, line_num=line_num),
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


def parse_failing_signature_id(
    raw: Mapping[str, Any],
    *,
    path: Path,
    line_num: int,
) -> int:
    """Parse the single regressed performance signature for a regression row."""

    failing_sig = raw.get("failing_sig")
    if not isinstance(failing_sig, Mapping):
        raise ValueError(f"missing failing_sig object at {path}:{line_num}")
    try:
        return int(failing_sig["signature_id"])
    except KeyError as exc:
        raise ValueError(
            f"missing failing_sig.signature_id at {path}:{line_num}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid failing_sig.signature_id at {path}:{line_num}: {exc}"
        ) from exc


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
        parents_by_node[node] = clean_parent_nodes(raw.get("parents"))

    logger.info(
        "Loaded commit graph: revisions=%d, parent_entries=%d, source=%s",
        len(index_by_node),
        sum(len(parents) for parents in parents_by_node.values()),
        path,
    )
    return CommitGraph(index_by_node=index_by_node, parents_by_node=parents_by_node)


def load_commit_graph_and_summary_counts(
    path: Path,
    *,
    needed_signature_ids: set[int],
) -> tuple[CommitGraph, dict[str, Counter[int]], Counter[str]]:
    """Load graph fields and per-signature summary counts in one pass."""

    index_by_node: dict[str, int] = {}
    parents_by_node: dict[str, list[str]] = {}
    summary_counts_by_revision_signature: dict[str, Counter[int]] = {}
    stats: Counter[str] = Counter()

    for _, raw in iter_jsonl(path):
        stats["revision_rows_scanned"] += 1
        node = raw.get("node")
        if not isinstance(node, str) or not node:
            stats["revision_rows_without_node"] += 1
            continue
        if node in index_by_node:
            raise ValueError(f"duplicate commit node in {path}: {node}")

        index_by_node[node] = len(index_by_node)
        parents_by_node[node] = clean_parent_nodes(raw.get("parents"))
        summary_counts_by_revision_signature[node] = count_summary_measurements(
            raw,
            needed_signature_ids=needed_signature_ids,
            stats=stats,
        )

    logger.info(
        "Loaded commit graph and summary counts: revisions=%d, parent_entries=%d, "
        "revisions_with_matching_summary_jobs=%d, matching_summary_jobs=%d, "
        "source=%s",
        len(index_by_node),
        sum(len(parents) for parents in parents_by_node.values()),
        sum(bool(counts) for counts in summary_counts_by_revision_signature.values()),
        sum(sum(counts.values()) for counts in summary_counts_by_revision_signature.values()),
        path,
    )
    return (
        CommitGraph(index_by_node=index_by_node, parents_by_node=parents_by_node),
        summary_counts_by_revision_signature,
        stats,
    )


def load_summary_counts(
    path: Path,
    *,
    needed_signature_ids: set[int],
) -> tuple[dict[str, Counter[int]], Counter[str]]:
    """Load summary measurement counts keyed by revision and signature."""

    summary_counts_by_revision_signature: dict[str, Counter[int]] = {}
    stats: Counter[str] = Counter()

    for _, raw in iter_jsonl(path):
        stats["revision_rows_scanned"] += 1
        node = raw.get("node")
        if not isinstance(node, str) or not node:
            stats["revision_rows_without_node"] += 1
            continue
        summary_counts_by_revision_signature[node] = count_summary_measurements(
            raw,
            needed_signature_ids=needed_signature_ids,
            stats=stats,
        )

    logger.info(
        "Loaded summary counts: revisions=%d, revisions_with_matching_summary_jobs=%d, "
        "matching_summary_jobs=%d, source=%s",
        len(summary_counts_by_revision_signature),
        sum(bool(counts) for counts in summary_counts_by_revision_signature.values()),
        sum(sum(counts.values()) for counts in summary_counts_by_revision_signature.values()),
        path,
    )
    return summary_counts_by_revision_signature, stats


def clean_parent_nodes(raw_parents: Any) -> list[str]:
    """Return non-null parent revision nodes from a raw parents field."""

    if not isinstance(raw_parents, list):
        return []
    return [
        parent
        for parent in raw_parents
        if isinstance(parent, str) and parent and parent != NULL_NODE
    ]


def count_summary_measurements(
    revision_row: Mapping[str, Any],
    *,
    needed_signature_ids: set[int],
    stats: Counter[str],
) -> Counter[int]:
    """Count cached summary measurements by signature for one revision row."""

    measurements = revision_row.get("perf_measurement_data", [])
    if not isinstance(measurements, list):
        stats["perf_measurement_data_not_list"] += 1
        return Counter()

    counts: Counter[int] = Counter()
    for measurement in measurements:
        stats["measurements_seen"] += 1
        if not isinstance(measurement, Mapping):
            stats["measurement_not_object"] += 1
            continue
        if measurement.get("replicate") is not False:
            stats["non_summary_measurements_seen"] += 1
            continue

        stats["summary_measurements_seen"] += 1
        try:
            signature_id = int(measurement["signature_id"])
        except (KeyError, TypeError, ValueError):
            stats["summary_measurement_bad_signature"] += 1
            continue
        if signature_id not in needed_signature_ids:
            stats["summary_measurement_unneeded_signature"] += 1
            continue

        counts[signature_id] += 1
        stats["matching_summary_measurements_seen"] += 1
    return counts


def build_output_rows(
    regressions: Sequence[Regression],
    *,
    commit_graph: CommitGraph,
    summary_counts_by_revision_signature: Mapping[str, Counter[int]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Create one output row per regression."""

    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()

    for regression in regressions:
        full_path = commit_graph.path_between(
            regression.good_revision,
            regression.bad_revision,
        )
        if full_path is None or len(full_path) < 2:
            stats["path_missing"] += 1
            rows.append(
                {
                    "regression_id": regression.regression_id,
                    "alert_summary_id": regression.alert_summary_id,
                    "good_revision": regression.good_revision,
                    "bad_revision": regression.bad_revision,
                    "culprit_revision": regression.culprit_revision,
                    "signature_id": regression.signature_id,
                    "num_candidate_revisions": regression.num_candidate_revisions,
                    "path_found": False,
                    "interior_revision_count": None,
                    "summary_test_job_count": None,
                    "revision_summary_test_job_counts": [],
                }
            )
            continue

        interior_revisions = full_path[1:-1]
        if regression.num_candidate_revisions is not None:
            expected = max(len(full_path) - 2, 0)
            if regression.num_candidate_revisions != expected:
                stats["stored_candidate_count_mismatch"] += 1

        revision_counts = [
            {
                "revision": revision,
                "summary_test_jobs": int(
                    summary_counts_by_revision_signature.get(
                        revision,
                        Counter(),
                    )[regression.signature_id]
                ),
            }
            for revision in interior_revisions
        ]
        total_count = sum(item["summary_test_jobs"] for item in revision_counts)

        stats["path_found"] += 1
        stats["interior_revisions"] += len(interior_revisions)
        stats["summary_test_jobs"] += total_count
        if total_count == 0:
            stats["regressions_with_zero_summary_jobs"] += 1

        rows.append(
            {
                "regression_id": regression.regression_id,
                "alert_summary_id": regression.alert_summary_id,
                "good_revision": regression.good_revision,
                "bad_revision": regression.bad_revision,
                "culprit_revision": regression.culprit_revision,
                "signature_id": regression.signature_id,
                "num_candidate_revisions": regression.num_candidate_revisions,
                "path_found": True,
                "interior_revision_count": len(interior_revisions),
                "summary_test_job_count": total_count,
                "revision_summary_test_job_counts": revision_counts,
            }
        )

    return rows, stats


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write rows to a JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row))
            fh.write("\n")
    logger.info("Wrote JSONL: path=%s", path)


def summarize_output_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize count distribution for valid rows."""

    counts = []
    for row in rows:
        value = row.get("summary_test_job_count")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        value = float(value)
        if math.isfinite(value):
            counts.append(value)

    if not counts:
        return {
            "regression_rows": len(rows),
            "regression_rows_with_counts": 0,
            "summary_test_job_count_mean": None,
            "summary_test_job_count_max": None,
        }

    return {
        "regression_rows": len(rows),
        "regression_rows_with_counts": len(counts),
        "summary_test_job_count_mean": round(mean(counts), 4),
        "summary_test_job_count_max": int(max(counts)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
