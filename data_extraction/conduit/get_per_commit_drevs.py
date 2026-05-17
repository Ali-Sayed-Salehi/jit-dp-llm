#!/usr/bin/env python3
"""
Fetch Mozilla Phabricator Differential Revisions referenced by commits.

This script reads datasets/mozilla_code_review/all_commits.jsonl. For each
commit, it looks for a trailing Mozilla Phabricator Differential Revision URL
in the commit description, for example:

    https://phabricator.services.mozilla.com/D206595

Commits without a matching trailing URL are skipped. Matching DREVs are fetched
through Conduit and only published, closed revisions are written to:

    datasets/mozilla_code_review/per_commit_drevs.jsonl

Each output row contains `commit_id` with the commit hash, `dataset_split`
(`eval` or `final test`), `risk_score` with the model's probability that the
commit is buggy, and `drev` with the raw revision object returned by
`differential.revision.search`.

The script uses risk_predictions_eval.json and risk_predictions_final_test.json
to derive Mercurial-order split boundaries. It only considers commits from the
start of the eval boundary through the end of the final-test boundary.

Use `--debug` to process only the last 10 eligible commits by first-parent graph
order. The debug subset is selected before extracting DREV URLs or initializing
the Phabricator client, so skipped commits do not cause API calls.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from dataclasses import dataclass
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_JSONL = (
    REPO_ROOT / "datasets" / "mozilla_code_review" / "all_commits.jsonl"
)
DEFAULT_OUTPUT_JSONL = (
    REPO_ROOT / "datasets" / "mozilla_code_review" / "per_commit_drevs.jsonl"
)
DEFAULT_EVAL_PREDICTIONS_JSON = (
    REPO_ROOT / "datasets" / "mozilla_code_review" / "risk_predictions_eval.json"
)
DEFAULT_FINAL_TEST_PREDICTIONS_JSON = (
    REPO_ROOT
    / "datasets"
    / "mozilla_code_review"
    / "risk_predictions_final_test.json"
)
DEFAULT_PHABRICATOR_API_URL = "https://phabricator.services.mozilla.com/api/"

NULL_NODE = "0000000000000000000000000000000000000000"
DREV_URL_RE = re.compile(
    r"https://phabricator\.services\.mozilla\.com/D(\d+)\s*\Z",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SplitBoundary:
    name: str
    start_index: int
    end_index: int
    start_commit_id: str
    end_commit_id: str
    sample_commit_ids: frozenset[str]
    sample_indices: tuple[int, ...]


@dataclass(frozen=True)
class CommitWorkItem:
    commit: dict[str, Any]
    dataset_split: str
    risk_score: float = 0.0
    has_risk_score: bool = False


class ConduitCaller:
    def __init__(
        self,
        *,
        min_interval_seconds: float,
        max_retries: int,
        retry_base_sleep_seconds: float,
    ) -> None:
        self.min_interval_seconds = min_interval_seconds
        self.max_retries = max_retries
        self.retry_base_sleep_seconds = retry_base_sleep_seconds
        self.last_call_time = 0.0

    def call(self, method: Any, **kwargs: Any) -> Any:
        for attempt in range(self.max_retries):
            now = time.time()
            elapsed = now - self.last_call_time
            if elapsed < self.min_interval_seconds:
                time.sleep(self.min_interval_seconds - elapsed)

            try:
                result = method(**kwargs)
                self.last_call_time = time.time()
                return result
            except Exception as exc:
                message = str(exc)
                if "429" not in message and "Too Many Requests" not in message:
                    raise

                wait_seconds = self.retry_base_sleep_seconds * (attempt + 1)
                print(
                    "Received 429 from Phabricator "
                    f"(attempt {attempt + 1}/{self.max_retries}); "
                    f"sleeping {wait_seconds:g}s before retry.",
                    file=sys.stderr,
                )
                time.sleep(wait_seconds)

        raise RuntimeError(
            f"Exceeded {self.max_retries} retries due to repeated 429 responses."
        )


class PhabricatorClient:
    def __init__(self, *, api_url: str, caller: ConduitCaller) -> None:
        self.api_url = api_url
        self.caller = caller
        self._phab: Any | None = None

    @property
    def phab(self) -> Any:
        if self._phab is None:
            try:
                from phabricator import Phabricator
            except ImportError as exc:
                raise RuntimeError(
                    "The `phabricator` package is required. Install python-phabricator "
                    "in the environment used to run this script."
                ) from exc

            self._phab = Phabricator(host=self.api_url, token=load_api_token())
            self.caller.call(self._phab.update_interfaces)
            self._phab.timeout = 30
        return self._phab

    def get_revision_by_id(self, drev_id: int) -> dict[str, Any] | None:
        result = self.caller.call(
            self.phab.differential.revision.search,
            constraints={"ids": [drev_id]},
            limit=1,
        )
        data = result.get("data", [])
        return data[0] if data else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        default=str(DEFAULT_INPUT_JSONL),
        help="Path to all_commits.jsonl.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(DEFAULT_OUTPUT_JSONL),
        help="Path to write per-commit DREV JSONL.",
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
        "--debug",
        action="store_true",
        help=(
            "Process only the last debug-count eligible commits by first-parent "
            "graph order."
        ),
    )
    parser.add_argument(
        "--debug-count",
        type=int,
        default=10,
        help="Number of graph-tail commits to process in debug mode.",
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("PHABRICATOR_API_URL", DEFAULT_PHABRICATOR_API_URL),
        help="Phabricator Conduit API URL.",
    )
    parser.add_argument(
        "--rate-limit-min-interval",
        type=float,
        default=0.5,
        help="Minimum seconds between Conduit calls.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retries for Conduit 429 responses.",
    )
    parser.add_argument(
        "--retry-base-sleep",
        type=float,
        default=5.0,
        help="Base sleep seconds for Conduit 429 backoff.",
    )
    return parser.parse_args(argv)


def load_api_token() -> str:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError(
            "The `python-dotenv` package is required to load secrets/.env."
        ) from exc

    load_dotenv(dotenv_path=REPO_ROOT / "secrets" / ".env")
    token = os.getenv("CONDUIT_API_TOKEN")
    if not token:
        raise RuntimeError("CONDUIT_API_TOKEN is not set in environment or secrets/.env")
    return token


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


def extract_drev_id(desc: str | None) -> int | None:
    if not desc:
        return None
    match = DREV_URL_RE.search(desc)
    if not match:
        return None
    return int(match.group(1))


def is_published_and_closed(revision: dict[str, Any]) -> bool:
    fields = revision.get("fields", {})
    if not isinstance(fields, dict):
        return False
    status = fields.get("status", {})
    if not isinstance(status, dict):
        return False

    is_published = status.get("value") == "published"
    is_closed = bool(
        status.get("closed", False)
        or fields.get("closed", False)
        or revision.get("closed", False)
    )
    return is_published and is_closed


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
        sample_indices=tuple(index for index, _ in indexed_commit_ids),
    )


def nearest_sample_distance(sample_indices: tuple[int, ...], index: int) -> int:
    position = bisect_left(sample_indices, index)
    distances: list[int] = []
    if position < len(sample_indices):
        distances.append(abs(sample_indices[position] - index))
    if position > 0:
        distances.append(abs(sample_indices[position - 1] - index))
    if not distances:
        raise ValueError("Cannot compute distance for an empty sample index list")
    return min(distances)


def classify_dataset_split(
    *,
    commit_id: str,
    index: int,
    eval_boundary: SplitBoundary,
    final_test_boundary: SplitBoundary,
) -> str:
    in_eval_boundary = eval_boundary.start_index <= index <= eval_boundary.end_index
    in_final_test_boundary = (
        final_test_boundary.start_index <= index <= final_test_boundary.end_index
    )

    if commit_id in eval_boundary.sample_commit_ids:
        return eval_boundary.name
    if commit_id in final_test_boundary.sample_commit_ids:
        return final_test_boundary.name

    if in_eval_boundary and not in_final_test_boundary:
        return eval_boundary.name
    if in_final_test_boundary and not in_eval_boundary:
        return final_test_boundary.name

    eval_distance = nearest_sample_distance(eval_boundary.sample_indices, index)
    final_test_distance = nearest_sample_distance(
        final_test_boundary.sample_indices,
        index,
    )
    if eval_distance < final_test_distance:
        return eval_boundary.name
    if final_test_distance < eval_distance:
        return final_test_boundary.name

    return (
        final_test_boundary.name
        if index >= final_test_boundary.start_index
        else eval_boundary.name
    )


def build_work_items(
    *,
    commits: list[dict[str, Any]],
    node_to_index: dict[str, int],
    eval_boundary: SplitBoundary,
    final_test_boundary: SplitBoundary,
    risk_scores: dict[str, float] | None = None,
) -> list[CommitWorkItem]:
    start_index = eval_boundary.start_index
    end_index = final_test_boundary.end_index
    if start_index > end_index:
        raise ValueError(
            "Eval start boundary comes after final-test end boundary: "
            f"{eval_boundary.start_commit_id} > {final_test_boundary.end_commit_id}"
        )

    work_items: list[CommitWorkItem] = []
    risk_scores = risk_scores or {}
    for index in range(start_index, end_index + 1):
        commit = commits[index]
        commit_id = commit["node"]
        dataset_split = classify_dataset_split(
            commit_id=commit_id,
            index=node_to_index[commit_id],
            eval_boundary=eval_boundary,
            final_test_boundary=final_test_boundary,
        )
        risk_score = risk_scores.get(commit_id)
        work_items.append(
            CommitWorkItem(
                commit=commit,
                dataset_split=dataset_split,
                risk_score=risk_score if risk_score is not None else 0.0,
                has_risk_score=risk_score is not None,
            )
        )

    return work_items


def describe_boundary(boundary: SplitBoundary) -> str:
    return (
        f"{boundary.name}: {boundary.start_commit_id} "
        f"(index {boundary.start_index}) -> {boundary.end_commit_id} "
        f"(index {boundary.end_index}), samples={len(boundary.sample_indices)}"
    )


def select_last_commits_by_first_parent(
    commits: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    if count <= 0 or not commits:
        return []

    node_to_commit: dict[str, dict[str, Any]] = {}
    node_to_index: dict[str, int] = {}
    children_by_node: dict[str, list[str]] = {}

    for index, commit in enumerate(commits):
        node = commit["node"]
        node_to_commit[node] = commit
        node_to_index[node] = index
        children_by_node.setdefault(node, [])

    for commit in commits:
        node = commit["node"]
        parents = commit.get("parents", [])
        for parent in parents:
            if parent in node_to_commit:
                children_by_node.setdefault(parent, []).append(node)

    heads = [
        commit["node"]
        for commit in commits
        if not children_by_node.get(commit["node"])
    ]
    if not heads:
        return commits[-count:]

    head = max(heads, key=lambda node: node_to_index[node])
    if len(heads) > 1:
        print(
            f"Found {len(heads)} graph heads; using latest head by file order: {head}",
            file=sys.stderr,
        )

    selected_newest_first: list[dict[str, Any]] = []
    seen: set[str] = set()
    current_node: str | None = head

    while current_node and len(selected_newest_first) < count:
        if current_node in seen:
            raise ValueError(f"Cycle detected while walking parents at {current_node}")
        seen.add(current_node)

        commit = node_to_commit[current_node]
        selected_newest_first.append(commit)

        parent_nodes = [
            parent
            for parent in commit.get("parents", [])
            if parent != NULL_NODE and parent in node_to_commit
        ]
        current_node = parent_nodes[0] if parent_nodes else None

    return list(reversed(selected_newest_first))


def select_last_work_items_by_first_parent(
    work_items: list[CommitWorkItem],
    count: int,
) -> list[CommitWorkItem]:
    if count <= 0 or not work_items:
        return []

    selected_commits = select_last_commits_by_first_parent(
        [item.commit for item in work_items],
        count,
    )
    item_by_node = {item.commit["node"]: item for item in work_items}
    return [item_by_node[commit["node"]] for commit in selected_commits]


def get_work_items_to_process(
    *,
    all_work_items: list[CommitWorkItem],
    debug: bool,
    debug_count: int,
) -> list[CommitWorkItem]:
    if not debug:
        return all_work_items
    if debug_count <= 0:
        print(
            "DEBUG: selected 0 eligible commits from first-parent graph tail.",
            file=sys.stderr,
        )
        return []

    selected = select_last_work_items_by_first_parent(all_work_items, debug_count)
    print(
        f"DEBUG: selected {len(selected)} eligible commits from "
        "first-parent graph tail.",
        file=sys.stderr,
    )
    return selected


def process_commits(
    *,
    work_items: Iterator[CommitWorkItem],
    output_jsonl: Path,
    client: PhabricatorClient,
) -> dict[str, int]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    revision_cache: dict[int, dict[str, Any] | None] = {}
    stats = {
        "commits_seen": 0,
        "commits_with_trailing_drev_url": 0,
        "unique_drevs_fetched": 0,
        "published_closed_drevs_written": 0,
        "skipped_unpublished_or_open": 0,
        "missing_risk_scores": 0,
        "fetch_errors": 0,
    }

    with output_jsonl.open("w", encoding="utf-8") as output_file:
        for work_item in work_items:
            commit = work_item.commit
            stats["commits_seen"] += 1
            commit_id = commit.get("node")
            if not isinstance(commit_id, str) or not commit_id:
                print(f"[WARN] Skipping commit without valid node: {commit!r}", file=sys.stderr)
                continue

            drev_id = extract_drev_id(commit.get("desc"))
            if drev_id is None:
                continue
            stats["commits_with_trailing_drev_url"] += 1

            if drev_id not in revision_cache:
                try:
                    revision_cache[drev_id] = client.get_revision_by_id(drev_id)
                    stats["unique_drevs_fetched"] += 1
                except Exception as exc:
                    revision_cache[drev_id] = None
                    stats["fetch_errors"] += 1
                    print(f"[WARN] Failed to fetch D{drev_id}: {exc}", file=sys.stderr)

            revision = revision_cache[drev_id]
            if revision is None:
                continue
            if not is_published_and_closed(revision):
                stats["skipped_unpublished_or_open"] += 1
                continue

            if not work_item.has_risk_score:
                stats["missing_risk_scores"] += 1
                print(
                    f"[WARN] Missing risk score for valid commit {commit_id}; "
                    "using 0.0.",
                    file=sys.stderr,
                )

            output_row = {
                "commit_id": commit_id,
                "dataset_split": work_item.dataset_split,
                "risk_score": work_item.risk_score,
                "drev": revision,
            }
            output_file.write(json.dumps(output_row) + "\n")
            stats["published_closed_drevs_written"] += 1

    return stats


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_jsonl = Path(args.input_jsonl)
    output_jsonl = Path(args.output_jsonl)
    eval_predictions_json = Path(args.eval_predictions_json)
    final_test_predictions_json = Path(args.final_test_predictions_json)

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
    node_to_index = {
        commit["node"]: index for index, commit in enumerate(commits)
    }

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
    risk_scores = load_combined_risk_scores(
        [eval_predictions_json, final_test_predictions_json]
    )

    overlap_start = max(eval_boundary.start_index, final_test_boundary.start_index)
    overlap_end = min(eval_boundary.end_index, final_test_boundary.end_index)
    if overlap_start <= overlap_end:
        print(
            "Split boundary windows overlap for "
            f"{overlap_end - overlap_start + 1} commits; explicit prediction "
            "membership is used first, then nearest split sample.",
            file=sys.stderr,
        )

    all_work_items = build_work_items(
        commits=commits,
        node_to_index=node_to_index,
        eval_boundary=eval_boundary,
        final_test_boundary=final_test_boundary,
        risk_scores=risk_scores,
    )
    work_items = get_work_items_to_process(
        all_work_items=all_work_items,
        debug=args.debug,
        debug_count=args.debug_count,
    )

    caller = ConduitCaller(
        min_interval_seconds=args.rate_limit_min_interval,
        max_retries=args.max_retries,
        retry_base_sleep_seconds=args.retry_base_sleep,
    )
    client = PhabricatorClient(api_url=args.api_url, caller=caller)

    stats = process_commits(
        work_items=iter(work_items),
        output_jsonl=output_jsonl,
        client=client,
    )

    print(f"Wrote {stats['published_closed_drevs_written']} rows to {output_jsonl}")
    print(json.dumps(stats, sort_keys=True), file=sys.stderr)


if __name__ == "__main__":
    main()
