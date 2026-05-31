#!/usr/bin/env python3
"""
Assign CodeBERT risk scores to perf-bisect candidate commits.

This script reads:

    datasets/mozilla_perf_bisect/perf_bisect_regressions_eval.jsonl
    datasets/mozilla_perf_bisect/perf_bisect_regressions_final_test.jsonl
    datasets/mozilla_perf_bisect/all_commits.jsonl
    datasets/mozilla_perf_bisect/final_test_results_perf_codebert_eval.json
    datasets/mozilla_perf_bisect/final_test_results_perf_codebert_final_test.json

It writes:

    datasets/mozilla_perf_bisect/per_commit_risk_scores.jsonl

The prediction JSON files only contain scores for commits that were examples in
the risk model's eval/final-test splits. For candidate commits missing from
those files, the script first copies scores backward across contiguous
same-``Bug <id>`` commit blocks, matching the behavior of
data_extraction/conduit/get_commit_risk_scores.py. Remaining missing candidates
receive the lowest observed model risk score so they remain valid probability
predictions without using the perf-bisect labels as model output.

The output contains one row per unique candidate commit, ordered by
all_commits.jsonl:

    commit_id, risk_score, bug_inducing, score_source, source_commit_id,
    ticket_id
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "datasets" / "mozilla_perf_bisect"

DEFAULT_COMMITS_JSONL = DATASET_DIR / "all_commits.jsonl"
DEFAULT_EVAL_REGRESSIONS_JSONL = (
    DATASET_DIR / "perf_bisect_regressions_eval.jsonl"
)
DEFAULT_FINAL_TEST_REGRESSIONS_JSONL = (
    DATASET_DIR / "perf_bisect_regressions_final_test.jsonl"
)
DEFAULT_REGRESSION_JSONLS = (
    DEFAULT_EVAL_REGRESSIONS_JSONL,
    DEFAULT_FINAL_TEST_REGRESSIONS_JSONL,
)
DEFAULT_EVAL_PREDICTIONS_JSON = (
    DATASET_DIR / "final_test_results_perf_codebert_eval.json"
)
DEFAULT_FINAL_TEST_PREDICTIONS_JSON = (
    DATASET_DIR / "final_test_results_perf_codebert_final_test.json"
)
DEFAULT_OUTPUT_JSONL = DATASET_DIR / "per_commit_risk_scores.jsonl"

NULL_NODE = "0000000000000000000000000000000000000000"
BUG_RE = re.compile(r"^\s*Bug\s+(\d+)\s*[-:]?\s*", re.IGNORECASE)
TICKET_RE = re.compile(r"\bBug\s+(\d+)\b", re.IGNORECASE)
REVERT_RE = re.compile(r"^\s*(revert|backed out)\b", re.IGNORECASE)


@dataclass(frozen=True)
class CommitRecord:
    node: str
    desc: str
    parents: tuple[str, ...]


@dataclass(frozen=True)
class CommitGraph:
    commits: tuple[CommitRecord, ...]
    node_to_index: dict[str, int]
    parents_by_node: dict[str, tuple[str, ...]]

    def path_between(self, good_revision: str, bad_revision: str) -> list[str] | None:
        """Return a good-to-bad parent path, if one exists."""

        good_index = self.node_to_index.get(good_revision)
        bad_index = self.node_to_index.get(bad_revision)
        if good_index is None or bad_index is None:
            return None
        if good_index > bad_index:
            return None
        if good_revision == bad_revision:
            return [good_revision]

        queue = [bad_revision]
        next_child_by_parent: dict[str, str | None] = {bad_revision: None}

        for revision in queue:
            if revision == good_revision:
                break
            for parent in self.parents_by_node.get(revision, ()):
                parent_index = self.node_to_index.get(parent)
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


@dataclass
class CandidateInfo:
    candidate_regression_count: int = 0
    culprit_regression_count: int = 0

    @property
    def bug_inducing(self) -> int:
        return 1 if self.culprit_regression_count > 0 else 0


@dataclass(frozen=True)
class ScoredCandidate:
    risk_score: float
    score_source: str
    source_commit_id: str | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--commits-jsonl",
        type=Path,
        default=DEFAULT_COMMITS_JSONL,
        help="Path to all_commits.jsonl.",
    )
    parser.add_argument(
        "--regressions-jsonl",
        type=Path,
        nargs="+",
        default=list(DEFAULT_REGRESSION_JSONLS),
        help=(
            "One or more perf-bisect regression JSONL paths. Defaults to the "
            "eval and final-test split files."
        ),
    )
    parser.add_argument(
        "--eval-predictions-json",
        type=Path,
        default=DEFAULT_EVAL_PREDICTIONS_JSON,
        help="Path to eval prediction JSON.",
    )
    parser.add_argument(
        "--final-test-predictions-json",
        type=Path,
        default=DEFAULT_FINAL_TEST_PREDICTIONS_JSON,
        help="Path to final-test prediction JSON.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_OUTPUT_JSONL,
        help="Path to write per-commit risk scores.",
    )
    return parser.parse_args(argv)


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as input_file:
        for line_num, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected object at {path}:{line_num}")
            yield line_num, record


def parse_binary_label(value: Any, *, path: Path, row_index: int, field: str) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int) and value in (0, 1):
        return value
    if isinstance(value, float) and value.is_integer() and int(value) in (0, 1):
        return int(value)
    raise ValueError(f"{path}: row {row_index} has invalid {field}: {value!r}")


def parse_probability(value: Any, *, path: Path, row_index: int, field: str) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}: row {row_index} has invalid {field}: {value!r}"
        ) from exc
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"{path}: row {row_index} has {field} outside [0, 1]: {probability}"
        )
    return probability


def risk_score_from_predicted_class(
    prediction: Any,
    confidence: Any,
    *,
    path: Path,
    row_index: int,
) -> float:
    pred = parse_binary_label(
        prediction,
        path=path,
        row_index=row_index,
        field="prediction",
    )
    conf = parse_probability(
        confidence,
        path=path,
        row_index=row_index,
        field="confidence",
    )
    return conf if pred == 1 else 1.0 - conf


def risk_score_from_positive_confidence(
    confidence: Any,
    *,
    path: Path,
    row_index: int,
) -> float:
    return parse_probability(
        confidence,
        path=path,
        row_index=row_index,
        field="confidence",
    )


def load_prediction_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    with path.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)

    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")

    rows = data.get("samples")
    if isinstance(rows, list):
        return rows, "predicted_class"

    rows = data.get("results")
    if isinstance(rows, list):
        return rows, "positive_class"

    raise ValueError(f"{path} must contain a list field named samples or results")


def load_commit_risk_scores(path: Path) -> dict[str, float]:
    rows, confidence_mode = load_prediction_rows(path)

    scores: dict[str, float] = {}
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: row {row_index} is not an object")
        commit_id = row.get("commit_id")
        if not isinstance(commit_id, str) or not commit_id:
            raise ValueError(f"{path}: row {row_index} has invalid commit_id")
        if commit_id in scores:
            raise ValueError(f"{path}: duplicate commit_id {commit_id!r}")

        confidence = row.get("confidence")
        if confidence is None:
            raise ValueError(f"{path}: row {row_index} is missing confidence")

        if confidence_mode == "predicted_class":
            risk_score = risk_score_from_predicted_class(
                row.get("prediction"),
                confidence,
                path=path,
                row_index=row_index,
            )
        else:
            risk_score = risk_score_from_positive_confidence(
                confidence,
                path=path,
                row_index=row_index,
            )

        scores[commit_id] = risk_score

    if not scores:
        raise ValueError(f"{path} contains no prediction rows")
    return scores


def load_combined_risk_scores(paths: list[Path]) -> dict[str, float]:
    combined: dict[str, float] = {}
    for path in paths:
        scores = load_commit_risk_scores(path)
        for commit_id, risk_score in scores.items():
            existing = combined.get(commit_id)
            if existing is not None and existing != risk_score:
                raise ValueError(
                    f"{commit_id} appears in multiple prediction files with "
                    f"different risk scores: {existing} vs {risk_score}"
                )
            combined[commit_id] = risk_score
    return combined


def load_commit_graph(path: Path) -> CommitGraph:
    commits: list[CommitRecord] = []
    node_to_index: dict[str, int] = {}
    parents_by_node: dict[str, tuple[str, ...]] = {}

    for _, record in iter_jsonl(path):
        node = record.get("node")
        if not isinstance(node, str) or not node:
            raise ValueError(f"Commit is missing a valid node: {record!r}")
        if node in node_to_index:
            raise ValueError(f"Duplicate commit node encountered: {node}")

        current_index = len(commits)
        raw_parents = record.get("parents", [])
        if not isinstance(raw_parents, list):
            raise ValueError(f"Commit {node} has non-list parents: {raw_parents!r}")

        parents: list[str] = []
        for parent in raw_parents:
            if parent == NULL_NODE:
                continue
            if not isinstance(parent, str):
                raise ValueError(f"Commit {node} has non-string parent: {parent!r}")
            parent_index = node_to_index.get(parent)
            if parent_index is None:
                raise ValueError(f"Commit {node} references missing parent {parent}.")
            if parent_index >= current_index:
                raise ValueError(
                    "all_commits.jsonl is not parent-before-child ordered: "
                    f"commit {node} at index {current_index} has parent {parent} "
                    f"at index {parent_index}."
                )
            parents.append(parent)

        desc = record.get("desc", "")
        if desc is None:
            desc = ""
        if not isinstance(desc, str):
            raise ValueError(f"Commit {node} has non-string desc: {desc!r}")

        node_to_index[node] = current_index
        parents_tuple = tuple(parents)
        parents_by_node[node] = parents_tuple
        commits.append(CommitRecord(node=node, desc=desc, parents=parents_tuple))

    return CommitGraph(
        commits=tuple(commits),
        node_to_index=node_to_index,
        parents_by_node=parents_by_node,
    )


def bug_id_from_desc(desc: str | None) -> str | None:
    text = desc or ""
    if REVERT_RE.match(text):
        return None
    match = BUG_RE.match(text)
    return match.group(1) if match else None


def ticket_id_from_desc(desc: str | None) -> int | None:
    match = TICKET_RE.search(desc or "")
    return int(match.group(1)) if match else None


def expand_scores_to_contiguous_bug_blocks(
    *,
    graph: CommitGraph,
    direct_scores: dict[str, float],
) -> tuple[dict[str, float], dict[str, str]]:
    expanded: dict[str, float] = dict(direct_scores)
    inherited_from: dict[str, str] = {}

    for scored_commit_id, risk_score in direct_scores.items():
        index = graph.node_to_index.get(scored_commit_id)
        if index is None:
            continue

        bug_id = bug_id_from_desc(graph.commits[index].desc)
        if bug_id is None:
            continue

        cursor = index
        while cursor >= 0 and bug_id_from_desc(graph.commits[cursor].desc) == bug_id:
            commit_id = graph.commits[cursor].node
            existing = expanded.get(commit_id)
            if existing is not None and existing != risk_score:
                raise ValueError(
                    f"Conflicting inherited risk scores for {commit_id}: "
                    f"{existing} vs {risk_score} from {scored_commit_id}"
                )

            expanded[commit_id] = risk_score
            if commit_id != scored_commit_id:
                inherited_from[commit_id] = scored_commit_id
            cursor -= 1

    return expanded, inherited_from


def build_candidate_infos(
    *,
    graph: CommitGraph,
    regressions_jsonls: list[Path],
) -> tuple[dict[str, CandidateInfo], Counter[str]]:
    candidate_infos: dict[str, CandidateInfo] = {}
    stats: Counter[str] = Counter()

    for regressions_jsonl in regressions_jsonls:
        stats["regression_files_seen"] += 1
        for line_num, regression in iter_jsonl(regressions_jsonl):
            good_revision = regression.get("good_revision")
            bad_revision = regression.get("bad_revision")
            culprit_revision = regression.get("culprit_revision")
            if not isinstance(good_revision, str) or not good_revision:
                raise ValueError(
                    f"{regressions_jsonl}:{line_num} invalid good_revision"
                )
            if not isinstance(bad_revision, str) or not bad_revision:
                raise ValueError(
                    f"{regressions_jsonl}:{line_num} invalid bad_revision"
                )
            if not isinstance(culprit_revision, str) or not culprit_revision:
                raise ValueError(
                    f"{regressions_jsonl}:{line_num} invalid culprit_revision"
                )

            stats["regressions_seen"] += 1
            path = graph.path_between(good_revision, bad_revision)
            if path is None or len(path) < 2:
                stats["regressions_skipped_no_revision_path"] += 1
                continue

            candidate_revisions = path[1:]
            if culprit_revision not in candidate_revisions:
                stats["regressions_with_culprit_outside_candidate_path"] += 1

            expected_interior_count = regression.get("num_candidate_revisions")
            if isinstance(expected_interior_count, int):
                interior_count = max(len(path) - 2, 0)
                if expected_interior_count != interior_count:
                    stats["regressions_with_candidate_count_mismatch"] += 1

            for commit_id in candidate_revisions:
                info = candidate_infos.setdefault(commit_id, CandidateInfo())
                info.candidate_regression_count += 1
                if commit_id == culprit_revision:
                    info.culprit_regression_count += 1
                    stats["positive_candidate_contexts"] += 1
                else:
                    stats["negative_candidate_contexts"] += 1
                stats["candidate_contexts"] += 1

    stats["unique_candidate_commits"] = len(candidate_infos)
    stats["unique_positive_commits"] = sum(
        1 for info in candidate_infos.values() if info.bug_inducing == 1
    )
    return candidate_infos, stats


def score_candidate(
    *,
    commit_id: str,
    direct_scores: dict[str, float],
    expanded_scores: dict[str, float],
    inherited_from: dict[str, str],
    missing_floor_score: float,
) -> ScoredCandidate:
    if commit_id in direct_scores:
        return ScoredCandidate(
            risk_score=direct_scores[commit_id],
            score_source="direct",
            source_commit_id=commit_id,
        )
    if commit_id in expanded_scores:
        return ScoredCandidate(
            risk_score=expanded_scores[commit_id],
            score_source="inherited_same_bug",
            source_commit_id=inherited_from.get(commit_id),
        )
    return ScoredCandidate(
        risk_score=missing_floor_score,
        score_source="missing_floor",
        source_commit_id=None,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    for path in (
        args.commits_jsonl,
        *args.regressions_jsonl,
        args.eval_predictions_json,
        args.final_test_predictions_json,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    graph = load_commit_graph(args.commits_jsonl)
    direct_scores = load_combined_risk_scores(
        [args.eval_predictions_json, args.final_test_predictions_json]
    )
    expanded_scores, inherited_from = expand_scores_to_contiguous_bug_blocks(
        graph=graph,
        direct_scores=direct_scores,
    )
    missing_floor_score = min(expanded_scores.values())

    candidate_infos, stats = build_candidate_infos(
        graph=graph,
        regressions_jsonls=args.regressions_jsonl,
    )

    output_stats: Counter[str] = Counter()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    ordered_candidate_ids = sorted(
        candidate_infos,
        key=lambda commit_id: graph.node_to_index[commit_id],
    )

    with args.output_jsonl.open("w", encoding="utf-8") as output_file:
        for commit_id in ordered_candidate_ids:
            info = candidate_infos[commit_id]
            scored = score_candidate(
                commit_id=commit_id,
                direct_scores=direct_scores,
                expanded_scores=expanded_scores,
                inherited_from=inherited_from,
                missing_floor_score=missing_floor_score,
            )
            output_stats[f"score_source_{scored.score_source}"] += 1

            record = graph.commits[graph.node_to_index[commit_id]]
            output_file.write(
                json.dumps(
                    {
                        "commit_id": commit_id,
                        "risk_score": scored.risk_score,
                        "bug_inducing": info.bug_inducing,
                        "score_source": scored.score_source,
                        "source_commit_id": scored.source_commit_id,
                        "ticket_id": ticket_id_from_desc(record.desc),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            output_stats["rows_written"] += 1

    print(f"Wrote {output_stats['rows_written']} rows to {args.output_jsonl}.", file=sys.stderr)
    print("Candidate stats:", file=sys.stderr)
    for key, value in sorted(stats.items()):
        print(f"  {key}={value}", file=sys.stderr)
    print("Score stats:", file=sys.stderr)
    print(f"  direct_prediction_scores={len(direct_scores)}", file=sys.stderr)
    print(f"  expanded_scores={len(expanded_scores)}", file=sys.stderr)
    print(f"  missing_floor_score={missing_floor_score}", file=sys.stderr)
    for key, value in sorted(output_stats.items()):
        print(f"  {key}={value}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
