#!/usr/bin/env python3
"""
Expand Mozilla code-review risk predictions to contiguous same-bug commits.

This script reads:

    datasets/mozilla_code_review/all_commits.jsonl
    datasets/mozilla_code_review/risk_predictions_eval.json
    datasets/mozilla_code_review/risk_predictions_final_test.json

The prediction files contain scores for the commits that represent rows in the
JIT risk-scoring dataset. That dataset collapses each contiguous block of
``Bug <id>`` commits into one net diff and uses the newest commit in the block
as the dataset row's commit id. To recover per-commit scores, this script walks
backward from every scored commit while previous commits have the same bug id,
mirroring the block selection in data_extraction/mercurial/link_bug_diffs.py.

It writes:

    datasets/mozilla_code_review/per_commit_risk_scores.jsonl

Each output row contains:

    commit_id, risk_score, desc

Only commits in the valid interval from the eval split's first Mercurial commit
through the final-test split's last Mercurial commit are considered. Commits in
that interval with neither a direct prediction score nor an inherited contiguous
same-bug score are skipped.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "datasets" / "mozilla_code_review"

DEFAULT_INPUT_JSONL = DATASET_DIR / "all_commits.jsonl"
DEFAULT_OUTPUT_JSONL = DATASET_DIR / "per_commit_risk_scores.jsonl"
DEFAULT_EVAL_PREDICTIONS_JSON = DATASET_DIR / "risk_predictions_eval.json"
DEFAULT_FINAL_TEST_PREDICTIONS_JSON = DATASET_DIR / "risk_predictions_final_test.json"

NULL_NODE = "0000000000000000000000000000000000000000"
BUG_RE = re.compile(r"^\s*Bug\s+(\d+)\s*[-:]?\s*", re.IGNORECASE)
REVERT_RE = re.compile(r"^\s*(revert|backed out)\b", re.IGNORECASE)


@dataclass(frozen=True)
class SplitBoundary:
    name: str
    start_index: int
    end_index: int
    start_commit_id: str
    end_commit_id: str
    sample_commit_ids: frozenset[str]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        default=str(DEFAULT_INPUT_JSONL),
        help="Path to all_commits.jsonl.",
    )
    parser.add_argument(
        "--eval-predictions-json",
        default=str(DEFAULT_EVAL_PREDICTIONS_JSON),
        help="Path to risk_predictions_eval.json.",
    )
    parser.add_argument(
        "--final-test-predictions-json",
        default=str(DEFAULT_FINAL_TEST_PREDICTIONS_JSON),
        help="Path to risk_predictions_final_test.json.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(DEFAULT_OUTPUT_JSONL),
        help="Path to write per_commit_risk_scores.jsonl.",
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


def load_commits(path: Path) -> list[dict[str, Any]]:
    commits: list[dict[str, Any]] = []
    node_to_index: dict[str, int] = {}

    for _, record in iter_jsonl(path):
        node = record.get("node")
        if not isinstance(node, str) or not node:
            raise ValueError(f"Commit is missing a valid node: {record!r}")
        if node in node_to_index:
            raise ValueError(f"Duplicate commit node encountered: {node}")

        current_index = len(commits)
        parents = record.get("parents", [])
        if not isinstance(parents, list):
            raise ValueError(f"Commit {node} has non-list parents: {parents!r}")

        for parent in parents:
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

        node_to_index[node] = current_index
        commits.append(record)

    return commits


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

    rows = data.get("samples")
    if isinstance(rows, list):
        return rows, "predicted_class"

    rows = data.get("results")
    if isinstance(rows, list):
        return rows, "positive_class"

    raise ValueError(f"{path} must contain a list field named samples or results")


def load_prediction_commit_ids(path: Path) -> list[str]:
    rows, _ = load_prediction_rows(path)

    commit_ids: list[str] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: row {row_index} is not an object")
        commit_id = row.get("commit_id")
        if not isinstance(commit_id, str) or not commit_id:
            raise ValueError(f"{path}: row {row_index} has invalid commit_id")
        commit_ids.append(commit_id)

    if not commit_ids:
        raise ValueError(f"{path} contains no prediction commit ids")
    return commit_ids


def load_risk_scores(path: Path) -> dict[str, float]:
    rows, confidence_mode = load_prediction_rows(path)

    risk_scores: dict[str, float] = {}
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: row {row_index} is not an object")
        commit_id = row.get("commit_id")
        if not isinstance(commit_id, str) or not commit_id:
            raise ValueError(f"{path}: row {row_index} has invalid commit_id")
        if commit_id in risk_scores:
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
        risk_scores[commit_id] = risk_score

    if not risk_scores:
        raise ValueError(f"{path} contains no prediction rows")
    return risk_scores


def load_combined_risk_scores(paths: list[Path]) -> dict[str, float]:
    combined: dict[str, float] = {}
    for path in paths:
        risk_scores = load_risk_scores(path)
        for commit_id, risk_score in risk_scores.items():
            existing = combined.get(commit_id)
            if existing is not None and existing != risk_score:
                raise ValueError(
                    f"{commit_id} appears in multiple prediction files with "
                    f"different risk scores: {existing} vs {risk_score}"
                )
            combined[commit_id] = risk_score
    return combined


def build_split_boundary(
    *,
    name: str,
    predictions_json: Path,
    node_to_index: dict[str, int],
) -> SplitBoundary:
    commit_ids = load_prediction_commit_ids(predictions_json)
    missing_commit_ids = [
        commit_id for commit_id in commit_ids if commit_id not in node_to_index
    ]
    if missing_commit_ids:
        preview = ", ".join(missing_commit_ids[:5])
        raise ValueError(
            f"{predictions_json} contains {len(missing_commit_ids)} commit ids "
            f"that are not in all_commits.jsonl. First missing ids: {preview}"
        )

    indexed_commit_ids = sorted(
        (node_to_index[commit_id], commit_id) for commit_id in commit_ids
    )
    start_index, start_commit_id = indexed_commit_ids[0]
    end_index, end_commit_id = indexed_commit_ids[-1]

    return SplitBoundary(
        name=name,
        start_index=start_index,
        end_index=end_index,
        start_commit_id=start_commit_id,
        end_commit_id=end_commit_id,
        sample_commit_ids=frozenset(commit_ids),
    )


def bug_id_from_desc(desc: str | None) -> str | None:
    text = desc or ""
    if REVERT_RE.match(text):
        return None
    match = BUG_RE.match(text)
    return match.group(1) if match else None


def expand_scores_to_contiguous_bug_blocks(
    *,
    commits: list[dict[str, Any]],
    node_to_index: dict[str, int],
    direct_risk_scores: dict[str, float],
) -> tuple[dict[str, float], dict[str, str]]:
    expanded: dict[str, float] = dict(direct_risk_scores)
    inherited_from: dict[str, str] = {}

    for scored_commit_id, risk_score in direct_risk_scores.items():
        index = node_to_index.get(scored_commit_id)
        if index is None:
            continue

        bug_id = bug_id_from_desc(commits[index].get("desc"))
        if bug_id is None:
            continue

        cursor = index
        while cursor >= 0 and bug_id_from_desc(commits[cursor].get("desc")) == bug_id:
            commit_id = commits[cursor]["node"]
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


def describe_boundary(boundary: SplitBoundary) -> str:
    return (
        f"{boundary.name}: {boundary.start_commit_id} "
        f"(index {boundary.start_index}) -> {boundary.end_commit_id} "
        f"(index {boundary.end_index}), samples={len(boundary.sample_commit_ids)}"
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_jsonl = Path(args.input_jsonl)
    eval_predictions_json = Path(args.eval_predictions_json)
    final_test_predictions_json = Path(args.final_test_predictions_json)
    output_jsonl = Path(args.output_jsonl)

    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")
    if not eval_predictions_json.exists():
        raise FileNotFoundError(
            f"Eval predictions JSON not found: {eval_predictions_json}"
        )
    if not final_test_predictions_json.exists():
        raise FileNotFoundError(
            f"Final-test predictions JSON not found: {final_test_predictions_json}"
        )

    commits = load_commits(input_jsonl)
    node_to_index = {commit["node"]: index for index, commit in enumerate(commits)}

    eval_boundary = build_split_boundary(
        name="eval",
        predictions_json=eval_predictions_json,
        node_to_index=node_to_index,
    )
    final_test_boundary = build_split_boundary(
        name="final test",
        predictions_json=final_test_predictions_json,
        node_to_index=node_to_index,
    )
    print(describe_boundary(eval_boundary), file=sys.stderr)
    print(describe_boundary(final_test_boundary), file=sys.stderr)

    start_index = eval_boundary.start_index
    end_index = final_test_boundary.end_index
    if start_index > end_index:
        raise ValueError(
            "Eval start boundary comes after final-test end boundary: "
            f"{eval_boundary.start_commit_id} > {final_test_boundary.end_commit_id}"
        )

    direct_risk_scores = load_combined_risk_scores(
        [eval_predictions_json, final_test_predictions_json]
    )
    expanded_risk_scores, inherited_from = expand_scores_to_contiguous_bug_blocks(
        commits=commits,
        node_to_index=node_to_index,
        direct_risk_scores=direct_risk_scores,
    )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "commits_in_valid_interval": 0,
        "rows_written": 0,
        "direct_scores_written": 0,
        "inherited_scores_written": 0,
        "commits_skipped_without_score": 0,
    }

    with output_jsonl.open("w", encoding="utf-8") as output_file:
        for index in range(start_index, end_index + 1):
            stats["commits_in_valid_interval"] += 1
            commit_id = commits[index]["node"]
            risk_score = expanded_risk_scores.get(commit_id)
            if risk_score is None:
                stats["commits_skipped_without_score"] += 1
                continue

            if commit_id in direct_risk_scores:
                stats["direct_scores_written"] += 1
            elif commit_id in inherited_from:
                stats["inherited_scores_written"] += 1

            output_file.write(
                json.dumps(
                    {
                        "commit_id": commit_id,
                        "risk_score": risk_score,
                        "desc": commits[index].get("desc", ""),
                    }
                )
                + "\n"
            )
            stats["rows_written"] += 1

    print(f"Wrote {stats['rows_written']} rows to {output_jsonl}.", file=sys.stderr)
    print("Stats:", file=sys.stderr)
    for key, value in stats.items():
        print(f"  {key}={value}", file=sys.stderr)


if __name__ == "__main__":
    main()
