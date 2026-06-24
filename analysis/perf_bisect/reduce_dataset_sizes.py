#!/usr/bin/env python3
"""Build the reduced Mozilla perf-bisect v2 dataset.

The reduced dataset is intentionally limited to inputs needed by
`simulation.py` and `calculate_oracle_metrics.py`:

  - perf_bisect_regressions_eval.jsonl
  - perf_bisect_regressions_final_test.jsonl
  - per_sig_perf_data_info.jsonl
  - per_revision_perf_data.jsonl
  - per_regression_oracle_metrics_v2.jsonl, when present in the source
  - per_commit_risk_scores.jsonl, when present in the source

Raw alert summaries, full commit exports, per-signature summary caches, and all
replicate measurements are deliberately omitted.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = REPO_ROOT / "datasets" / "mozilla_perf_bisect_v2"
DEFAULT_OUTPUT_DIR = DEFAULT_SOURCE_DIR / "reduced"
REGRESSION_FILENAMES = (
    "perf_bisect_regressions_eval.jsonl",
    "perf_bisect_regressions_final_test.jsonl",
)
SIGNATURE_INFO_FILENAME = "per_sig_perf_data_info.jsonl"
REVISION_DATA_FILENAME = "per_revision_perf_data.jsonl"
SUMMARY_DATA_FILENAME = "per_sig_perf_data_summary.jsonl"
COMMITS_FILENAME = "all_commits.jsonl"
ORACLE_METRICS_FILENAME = "per_regression_oracle_metrics_v2.jsonl"
RISK_SCORES_FILENAME = "per_commit_risk_scores.jsonl"
REDUCED_FILENAMES = (
    *REGRESSION_FILENAMES,
    SIGNATURE_INFO_FILENAME,
    REVISION_DATA_FILENAME,
    ORACLE_METRICS_FILENAME,
    RISK_SCORES_FILENAME,
)
NULL_NODE = "0000000000000000000000000000000000000000"


@dataclass(frozen=True)
class CommitRecord:
    """Compact commit graph record."""

    node: str
    parents: list[str]
    date: Any = None


@dataclass(frozen=True)
class CommitGraph:
    """Parent graph in source-file order."""

    records: list[CommitRecord]
    index_by_node: dict[str, int]
    parents_by_node: dict[str, list[str]]

    def path_between(self, good_revision: str, bad_revision: str) -> list[str] | None:
        """Return the parent-link path from good_revision to bad_revision."""

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

        path: list[str] = []
        revision: str | None = good_revision
        while revision is not None:
            path.append(revision)
            revision = next_child_by_parent[revision]
        return path


def main(argv: Sequence[str] | None = None) -> int:
    """Write the reduced dataset."""

    args = parse_args(argv)
    configure_logging(args.log_level)
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()

    logger.info("Starting reduced Mozilla perf-bisect v2 dataset build.")
    logger.info("Source dataset: %s", source_dir)
    logger.info("Reduced output: %s", output_dir)

    logger.info("Step 1/7: Loading and reducing regression splits.")
    regression_rows_by_filename = load_reduced_regression_splits(source_dir)
    all_regressions = [
        regression
        for rows in regression_rows_by_filename.values()
        for regression in rows
    ]
    needed_signature_ids = collect_needed_signature_ids(all_regressions)

    logger.info("Step 2/7: Loading source revision graph.")
    graph_input = choose_graph_input(source_dir)
    preloaded_summary_samples = None
    if graph_input == source_dir / REVISION_DATA_FILENAME:
        logger.info(
            "Source per-revision data is also the graph input; scanning it "
            "once for graph fields and needed summary measurements."
        )
        commit_graph, preloaded_summary_samples = (
            load_commit_graph_and_summary_samples_from_revision_data(
                graph_input,
                needed_signature_ids=needed_signature_ids,
            )
        )
    else:
        commit_graph = load_commit_graph(graph_input)

    logger.info("Step 3/7: Reconstructing regression candidate paths.")
    path_nodes, candidate_nodes, path_stats = collect_path_nodes(
        all_regressions,
        commit_graph=commit_graph,
    )

    logger.info("Step 4/7: Checking reduced output directory.")
    output_paths = [
        output_dir / filename
        for filename in (
            *REGRESSION_FILENAMES,
            SIGNATURE_INFO_FILENAME,
            REVISION_DATA_FILENAME,
        )
    ]
    if (source_dir / ORACLE_METRICS_FILENAME).exists():
        output_paths.append(output_dir / ORACLE_METRICS_FILENAME)
    if (source_dir / RISK_SCORES_FILENAME).exists():
        output_paths.append(output_dir / RISK_SCORES_FILENAME)
    ensure_no_unmanaged_outputs(
        output_dir,
        managed_paths=[output_dir / filename for filename in REDUCED_FILENAMES],
    )
    ensure_can_write(output_paths, overwrite=args.overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Step 5/7: Writing reduced regression splits.")
    for filename, rows in regression_rows_by_filename.items():
        write_jsonl(output_dir / filename, rows)

    logger.info("Step 6/7: Writing reduced signature and revision data.")
    write_reduced_signature_info(
        source_dir / SIGNATURE_INFO_FILENAME,
        output_dir / SIGNATURE_INFO_FILENAME,
        needed_signature_ids=needed_signature_ids,
    )
    write_reduced_revision_data(
        source_dir=source_dir,
        output_path=output_dir / REVISION_DATA_FILENAME,
        commit_graph=commit_graph,
        path_nodes=path_nodes,
        candidate_nodes=candidate_nodes,
        needed_signature_ids=needed_signature_ids,
        preloaded_summary_samples=preloaded_summary_samples,
    )

    logger.info("Step 7/7: Writing optional compact metrics and risk scores.")
    maybe_write_reduced_oracle_metrics(
        source_dir / ORACLE_METRICS_FILENAME,
        output_dir / ORACLE_METRICS_FILENAME,
        regression_ids={int(row["regression_id"]) for row in all_regressions},
    )
    maybe_write_reduced_risk_scores(
        source_dir / RISK_SCORES_FILENAME,
        output_dir / RISK_SCORES_FILENAME,
        candidate_nodes=candidate_nodes,
    )

    print(f"source_dir={source_dir}")
    print(f"output_dir={output_dir}")
    print(f"graph_input={graph_input}")
    print(f"regression_rows={len(all_regressions)}")
    print(f"needed_signatures={len(needed_signature_ids)}")
    print(f"path_nodes={len(path_nodes)}")
    print(f"candidate_nodes={len(candidate_nodes)}")
    for key, value in sorted(path_stats.items()):
        print(f"{key}={value}")
    print("wrote reduced dataset")
    logger.info("Finished reduced Mozilla perf-bisect v2 dataset build.")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing the full Mozilla perf-bisect v2 dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the reduced dataset will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing reduced dataset files.",
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


def ensure_can_write(paths: Sequence[Path], *, overwrite: bool) -> None:
    """Fail before writing if any output already exists and overwrite is off."""

    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        logger.info("Found existing managed outputs and overwrite is disabled.")
        formatted = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing reduced dataset files. "
            "Pass --overwrite to replace them:\n"
            f"{formatted}"
        )


def ensure_no_unmanaged_outputs(
    output_dir: Path,
    *,
    managed_paths: Sequence[Path],
) -> None:
    """Refuse to mix reduced outputs with files this script does not manage."""

    if not output_dir.exists():
        logger.info("Reduced output directory does not exist yet.")
        return
    managed = {path.resolve() for path in managed_paths}
    unmanaged = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.resolve() not in managed
    ]
    if unmanaged:
        logger.info("Found unmanaged files in reduced output directory.")
        formatted = "\n".join(f"  {path}" for path in unmanaged)
        raise FileExistsError(
            "The reduced dataset directory contains files that are not managed "
            "by this script. Move or remove them before reducing so the output "
            f"contains only required datasets:\n{formatted}"
        )


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield JSON objects from a JSONL file."""

    with path.open(encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_num}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_num}")
            yield line_num, value


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Write compact JSONL records and return the row count."""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":"), sort_keys=True))
            fh.write("\n")
            count += 1
    logger.info("Wrote JSONL: rows=%d, path=%s", count, path)
    return count


def load_reduced_regression_splits(
    source_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Load and reduce eval/final-test regression rows."""

    rows_by_filename: dict[str, list[dict[str, Any]]] = {}
    seen_regression_ids: set[int] = set()
    for filename in REGRESSION_FILENAMES:
        path = source_dir / filename
        rows = []
        for line_num, raw in iter_jsonl(path):
            row = reduce_regression_row(raw, path=path, line_num=line_num)
            regression_id = int(row["regression_id"])
            if regression_id in seen_regression_ids:
                raise ValueError(f"duplicate regression_id {regression_id} in {path}")
            seen_regression_ids.add(regression_id)
            rows.append(row)
        rows_by_filename[filename] = rows
        logger.info("Loaded regression split: rows=%d, path=%s", len(rows), path)
    return rows_by_filename


def reduce_regression_row(
    raw: Mapping[str, Any],
    *,
    path: Path,
    line_num: int,
) -> dict[str, Any]:
    """Keep only fields needed by simulation and oracle-metric generation."""

    failing_sig = raw.get("failing_sig")
    if not isinstance(failing_sig, Mapping):
        raise ValueError(f"missing failing_sig object at {path}:{line_num}")

    reduced: dict[str, Any] = {
        "regression_id": parse_positive_int(
            raw.get("regression_id"),
            context=f"{path}:{line_num} regression_id",
        ),
        "good_revision": parse_revision(
            raw.get("good_revision"),
            context=f"{path}:{line_num} good_revision",
        ),
        "bad_revision": parse_revision(
            raw.get("bad_revision"),
            context=f"{path}:{line_num} bad_revision",
        ),
        "culprit_revision": parse_revision(
            raw.get("culprit_revision"),
            context=f"{path}:{line_num} culprit_revision",
        ),
        "failing_sig": {
            "signature_id": parse_positive_int(
                failing_sig.get("signature_id"),
                context=f"{path}:{line_num} failing_sig.signature_id",
            ),
            "Good_value": parse_finite_number(
                failing_sig.get("Good_value"),
                context=f"{path}:{line_num} failing_sig.Good_value",
            ),
            "bad_value": parse_finite_number(
                failing_sig.get("bad_value"),
                context=f"{path}:{line_num} failing_sig.bad_value",
            ),
        },
    }
    if raw.get("alert_summary_id") is not None:
        reduced["alert_summary_id"] = parse_positive_int(
            raw.get("alert_summary_id"),
            context=f"{path}:{line_num} alert_summary_id",
        )
    if raw.get("num_candidate_revisions") is not None:
        reduced["num_candidate_revisions"] = parse_non_negative_int(
            raw.get("num_candidate_revisions"),
            context=f"{path}:{line_num} num_candidate_revisions",
        )
    platform = failing_sig.get("platform")
    if platform is not None and str(platform):
        reduced["failing_sig"]["platform"] = str(platform)
    return reduced


def parse_revision(value: Any, *, context: str) -> str:
    """Parse a required revision hash."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"missing or invalid revision for {context}: {value!r}")
    return value


def parse_positive_int(value: Any, *, context: str) -> int:
    """Parse a positive integer."""

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer for {context}: {value!r}") from exc
    if parsed < 1:
        raise ValueError(f"expected positive integer for {context}: {value!r}")
    return parsed


def parse_non_negative_int(value: Any, *, context: str) -> int:
    """Parse a non-negative integer."""

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer for {context}: {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"expected non-negative integer for {context}: {value!r}")
    return parsed


def parse_finite_number(value: Any, *, context: str) -> float:
    """Parse a finite float."""

    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid number for {context}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"expected finite number for {context}: {value!r}")
    return parsed


def collect_needed_signature_ids(regressions: Sequence[Mapping[str, Any]]) -> set[int]:
    """Return failing signature IDs required by the reduced dataset."""

    return {
        int(regression["failing_sig"]["signature_id"])
        for regression in regressions
    }


def choose_graph_input(source_dir: Path) -> Path:
    """Choose the full source graph used to reconstruct candidate paths."""

    per_revision = source_dir / REVISION_DATA_FILENAME
    if per_revision.exists():
        logger.info("Using source per-revision graph: %s", per_revision)
        return per_revision
    commits = source_dir / COMMITS_FILENAME
    if commits.exists():
        logger.info("Using source full commit graph: %s", commits)
        return commits
    raise FileNotFoundError(
        f"expected either {per_revision} or {commits} to build revision paths"
    )


def load_commit_graph(path: Path) -> CommitGraph:
    """Load compact parent graph fields from a source graph JSONL."""

    records: list[CommitRecord] = []
    index_by_node: dict[str, int] = {}
    parents_by_node: dict[str, list[str]] = {}
    for line_num, raw in iter_jsonl(path):
        node = raw.get("node")
        if not isinstance(node, str) or not node:
            raise ValueError(f"missing node at {path}:{line_num}")
        if node in index_by_node:
            raise ValueError(f"duplicate node {node!r} at {path}:{line_num}")
        parents = normalized_parents(raw.get("parents"))
        record = CommitRecord(node=node, parents=parents, date=raw.get("date"))
        index_by_node[node] = len(records)
        parents_by_node[node] = parents
        records.append(record)
    logger.info("Loaded commit graph: revisions=%d, source=%s", len(records), path)
    return CommitGraph(
        records=records,
        index_by_node=index_by_node,
        parents_by_node=parents_by_node,
    )


def load_commit_graph_and_summary_samples_from_revision_data(
    path: Path,
    *,
    needed_signature_ids: set[int],
) -> tuple[CommitGraph, dict[str, list[dict[str, Any]]]]:
    """Load graph fields and summary samples from per-revision data in one pass."""

    records: list[CommitRecord] = []
    index_by_node: dict[str, int] = {}
    parents_by_node: dict[str, list[str]] = {}
    samples_by_revision: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stats: Counter[str] = Counter()

    for line_num, raw in iter_jsonl(path):
        stats["revision_rows_scanned"] += 1
        node = raw.get("node")
        if not isinstance(node, str) or not node:
            raise ValueError(f"missing node at {path}:{line_num}")
        if node in index_by_node:
            raise ValueError(f"duplicate node {node!r} at {path}:{line_num}")

        parents = normalized_parents(raw.get("parents"))
        record = CommitRecord(node=node, parents=parents, date=raw.get("date"))
        index_by_node[node] = len(records)
        parents_by_node[node] = parents
        records.append(record)

        for measurement in raw.get("perf_measurement_data", []):
            stats["measurements_seen"] += 1
            sample = reduce_measurement_sample(
                measurement,
                needed_signature_ids=needed_signature_ids,
                require_summary_flag=True,
            )
            if sample is not None:
                samples_by_revision[node].append(sample)
                stats["summary_samples_preloaded"] += 1

    logger.info(
        "Loaded commit graph and summary samples: revisions=%d, "
        "revisions_with_samples=%d, source=%s",
        len(records),
        len(samples_by_revision),
        path,
    )
    logger.info("Combined per-revision scan counters: %s", dict(stats))
    return (
        CommitGraph(
            records=records,
            index_by_node=index_by_node,
            parents_by_node=parents_by_node,
        ),
        dict(samples_by_revision),
    )


def normalized_parents(raw_parents: Any) -> list[str]:
    """Return non-null parent revisions."""

    if not isinstance(raw_parents, list):
        return []
    return [
        parent
        for parent in raw_parents
        if isinstance(parent, str) and parent and parent != NULL_NODE
    ]


def collect_path_nodes(
    regressions: Sequence[Mapping[str, Any]],
    *,
    commit_graph: CommitGraph,
) -> tuple[set[str], set[str], Counter[str]]:
    """Collect graph nodes needed by every regression path."""

    path_nodes: set[str] = set()
    candidate_nodes: set[str] = set()
    stats: Counter[str] = Counter()
    missing_paths: list[int] = []
    for regression in regressions:
        regression_id = int(regression["regression_id"])
        path = commit_graph.path_between(
            str(regression["good_revision"]),
            str(regression["bad_revision"]),
        )
        if path is None or len(path) < 2:
            stats["path_missing"] += 1
            missing_paths.append(regression_id)
            continue
        if str(regression["culprit_revision"]) not in path:
            stats["culprit_not_in_path"] += 1
        path_nodes.update(path)
        candidate_nodes.update(path[1:])

    if missing_paths:
        preview = ", ".join(str(regression_id) for regression_id in missing_paths[:10])
        raise ValueError(
            "cannot build a reproducible reduced dataset because candidate paths "
            f"are missing for {len(missing_paths)} regressions. First ids: {preview}"
        )
    logger.info(
        "Collected path nodes: regressions=%d, path_nodes=%d, "
        "candidate_nodes=%d",
        len(regressions),
        len(path_nodes),
        len(candidate_nodes),
    )
    if stats:
        logger.info("Path collection counters: %s", dict(sorted(stats.items())))
    return path_nodes, candidate_nodes, stats


def write_reduced_signature_info(
    source_path: Path,
    output_path: Path,
    *,
    needed_signature_ids: set[int],
) -> None:
    """Write compact signature runtime metadata."""

    info_by_signature: dict[int, dict[str, Any]] = {}
    for line_num, raw in iter_jsonl(source_path):
        signature_id = parse_positive_int(
            raw.get("signature_id"),
            context=f"{source_path}:{line_num} signature_id",
        )
        if signature_id not in needed_signature_ids:
            continue
        job_duration = parse_finite_number(
            raw.get("job_duration"),
            context=f"{source_path}:{line_num} job_duration",
        )
        if job_duration <= 0:
            raise ValueError(
                f"job_duration must be positive at {source_path}:{line_num}: "
                f"{job_duration!r}"
            )
        record: dict[str, Any] = {
            "signature_id": signature_id,
            "job_duration": job_duration,
        }
        platform = raw.get("platform")
        if platform is not None and str(platform):
            record["platform"] = str(platform)
        info_by_signature[signature_id] = record

    missing = sorted(needed_signature_ids - set(info_by_signature))
    if missing:
        preview = ", ".join(str(signature_id) for signature_id in missing[:10])
        raise ValueError(
            f"missing signature info for {len(missing)} signatures. "
            f"First ids: {preview}"
        )

    write_jsonl(
        output_path,
        (info_by_signature[signature_id] for signature_id in sorted(info_by_signature)),
    )
    logger.info(
        "Wrote reduced signature info: signatures=%d, path=%s",
        len(info_by_signature),
        output_path,
    )


def write_reduced_revision_data(
    *,
    source_dir: Path,
    output_path: Path,
    commit_graph: CommitGraph,
    path_nodes: set[str],
    candidate_nodes: set[str],
    needed_signature_ids: set[int],
    preloaded_summary_samples: Mapping[str, list[dict[str, Any]]] | None,
) -> None:
    """Write path-subset graph rows with summary measurements only."""

    source_revision_data = source_dir / REVISION_DATA_FILENAME
    if preloaded_summary_samples is not None:
        logger.info(
            "Using summary measurements preloaded during the graph scan; "
            "not rescanning source per-revision data."
        )
        samples_by_revision = filter_samples_to_candidate_nodes(
            preloaded_summary_samples,
            candidate_nodes=candidate_nodes,
        )
    elif source_revision_data.exists():
        logger.info(
            "Loading summary measurements from source per-revision data: %s",
            source_revision_data,
        )
        samples_by_revision = load_summary_samples_from_revision_data(
            source_revision_data,
            candidate_nodes=candidate_nodes,
            needed_signature_ids=needed_signature_ids,
        )
    else:
        logger.info(
            "Source per-revision data missing; loading summary measurements "
            "from per-signature cache."
        )
        samples_by_revision = load_summary_samples_from_signature_data(
            source_dir / SUMMARY_DATA_FILENAME,
            candidate_nodes=candidate_nodes,
            needed_signature_ids=needed_signature_ids,
        )

    rows = (
        reduced_revision_record(record, samples_by_revision=samples_by_revision)
        for record in commit_graph.records
        if record.node in path_nodes
    )
    write_jsonl(output_path, rows)
    logger.info(
        "Wrote reduced revision data: path_nodes=%d, revisions_with_samples=%d, "
        "summary_samples=%d, path=%s",
        len(path_nodes),
        len(samples_by_revision),
        sum(len(samples) for samples in samples_by_revision.values()),
        output_path,
    )


def filter_samples_to_candidate_nodes(
    samples_by_revision: Mapping[str, list[dict[str, Any]]],
    *,
    candidate_nodes: set[str],
) -> dict[str, list[dict[str, Any]]]:
    """Keep preloaded samples only for candidate revisions."""

    filtered = {
        revision: samples
        for revision, samples in samples_by_revision.items()
        if revision in candidate_nodes
    }
    logger.info(
        "Filtered preloaded summary samples to candidates: before_revisions=%d, "
        "after_revisions=%d, after_samples=%d",
        len(samples_by_revision),
        len(filtered),
        sum(len(samples) for samples in filtered.values()),
    )
    return filtered


def load_summary_samples_from_revision_data(
    path: Path,
    *,
    candidate_nodes: set[str],
    needed_signature_ids: set[int],
) -> dict[str, list[dict[str, Any]]]:
    """Read summary measurements from a full per-revision dataset."""

    samples_by_revision: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stats: Counter[str] = Counter()
    for _, raw in iter_jsonl(path):
        stats["revision_rows_scanned"] += 1
        node = raw.get("node")
        if not isinstance(node, str) or node not in candidate_nodes:
            continue
        stats["candidate_revision_rows"] += 1
        for measurement in raw.get("perf_measurement_data", []):
            stats["measurements_seen"] += 1
            sample = reduce_measurement_sample(
                measurement,
                needed_signature_ids=needed_signature_ids,
                require_summary_flag=True,
            )
            if sample is not None:
                samples_by_revision[node].append(sample)
                stats["summary_samples_kept"] += 1
    logger.info("Loaded summary samples from revision data: %s", dict(stats))
    return dict(samples_by_revision)


def load_summary_samples_from_signature_data(
    path: Path,
    *,
    candidate_nodes: set[str],
    needed_signature_ids: set[int],
) -> dict[str, list[dict[str, Any]]]:
    """Read summary measurements from a per-signature summary cache."""

    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; cannot build reduced {REVISION_DATA_FILENAME}"
        )

    samples_by_revision: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stats: Counter[str] = Counter()
    for line_num, raw in iter_jsonl(path):
        stats["signature_rows_scanned"] += 1
        signature_id = parse_positive_int(
            raw.get("signature_id"),
            context=f"{path}:{line_num} signature_id",
        )
        if signature_id not in needed_signature_ids:
            continue
        stats["needed_signature_rows"] += 1
        for measurement in raw.get("perf_measurement_data", []):
            stats["measurements_seen"] += 1
            if not isinstance(measurement, Mapping):
                continue
            revision = measurement.get("revision")
            if not isinstance(revision, str) or revision not in candidate_nodes:
                continue
            sample = reduce_measurement_sample(
                {**measurement, "signature_id": signature_id, "replicate": False},
                needed_signature_ids=needed_signature_ids,
                require_summary_flag=True,
            )
            if sample is not None:
                samples_by_revision[revision].append(sample)
                stats["summary_samples_kept"] += 1
    logger.info("Loaded summary samples from signature data: %s", dict(stats))
    return dict(samples_by_revision)


def reduce_measurement_sample(
    measurement: Any,
    *,
    needed_signature_ids: set[int],
    require_summary_flag: bool,
) -> dict[str, Any] | None:
    """Return the compact summary measurement shape, or None if not needed."""

    if not isinstance(measurement, Mapping):
        return None
    if require_summary_flag and measurement.get("replicate") is not False:
        return None
    try:
        signature_id = int(measurement["signature_id"])
    except (KeyError, TypeError, ValueError):
        return None
    if signature_id not in needed_signature_ids:
        return None
    try:
        value = float(measurement["value"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return {
        "signature_id": signature_id,
        "value": value,
        "replicate": False,
    }


def reduced_revision_record(
    record: CommitRecord,
    *,
    samples_by_revision: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Serialize one reduced per-revision graph row."""

    return {
        "node": record.node,
        "parents": record.parents,
        "date": record.date,
        "perf_measurement_data": samples_by_revision.get(record.node, []),
    }


def maybe_write_reduced_oracle_metrics(
    source_path: Path,
    output_path: Path,
    *,
    regression_ids: set[int],
) -> None:
    """Copy compact oracle metrics if they already exist in the source dataset."""

    if not source_path.exists():
        logger.warning(
            "Oracle metrics source is missing: %s. Run "
            "calculate_oracle_metrics.py after reduction to create it.",
            source_path,
        )
        return

    rows = []
    for line_num, raw in iter_jsonl(source_path):
        regression_id = parse_positive_int(
            raw.get("regression_id"),
            context=f"{source_path}:{line_num} regression_id",
        )
        if regression_id not in regression_ids:
            continue
        rows.append(
            {
                "regression_id": regression_id,
                "summary_oracle_accuracy": parse_probability(
                    raw.get("summary_oracle_accuracy"),
                    context=f"{source_path}:{line_num} summary_oracle_accuracy",
                ),
            }
        )
    write_jsonl(output_path, rows)
    logger.info(
        "Wrote reduced oracle metrics: rows=%d, source=%s, path=%s",
        len(rows),
        source_path,
        output_path,
    )


def parse_probability(value: Any, *, context: str) -> float:
    """Parse a finite probability."""

    parsed = parse_finite_number(value, context=context)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"expected probability for {context}: {value!r}")
    return parsed


def maybe_write_reduced_risk_scores(
    source_path: Path,
    output_path: Path,
    *,
    candidate_nodes: set[str],
) -> None:
    """Copy candidate-only risk scores if source risk scores exist."""

    if not source_path.exists():
        logger.warning(
            "Risk-score source is missing: %s. Risk localizers will still "
            "require --ignore-risk or a risk-score file.",
            source_path,
        )
        return

    rows = []
    scored_nodes: set[str] = set()
    for line_num, raw in iter_jsonl(source_path):
        commit_id = raw.get("commit_id")
        if not isinstance(commit_id, str) or commit_id not in candidate_nodes:
            continue
        rows.append(
            {
                "commit_id": commit_id,
                "risk_score": parse_probability(
                    raw.get("risk_score"),
                    context=f"{source_path}:{line_num} risk_score",
                ),
            }
        )
        scored_nodes.add(commit_id)
    write_jsonl(output_path, rows)
    logger.info(
        "Wrote reduced risk scores: rows=%d, source=%s, path=%s",
        len(rows),
        source_path,
        output_path,
    )

    missing = candidate_nodes - scored_nodes
    if missing:
        logger.warning(
            "Risk scores are missing for %d candidate revisions; use "
            "--ignore-risk unless these scores are supplied.",
            len(missing),
        )


if __name__ == "__main__":
    raise SystemExit(main())
