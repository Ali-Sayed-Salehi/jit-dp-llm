#!/usr/bin/env python3
"""
Build a perf-bisect regression dataset from Treeherder alert summaries.

Inputs:
  - `datasets/mozilla_perf_bisect/alert_summaries.csv`
  - `datasets/mozilla_perf_bisect/all_commits.jsonl`
  - `datasets/mozilla_perf_bisect/alert_summary_fail_perf_sigs_no_fw_2_6_18.csv`
  - Optional: `datasets/mozilla_perf_bisect/per_sig_perf_data_info.jsonl`

Output:
  - `datasets/mozilla_perf_bisect/perf_bisect_regressions_eval.jsonl`
  - `datasets/mozilla_perf_bisect/perf_bisect_regressions_final_test.jsonl`

Each output JSONL row contains:
  - `alert_summary_id`
  - `good_revision`
  - `bad_revision`
  - `num_candidate_revisions`
  - `culprit_revision`
  - `failing_sig`

`failing_sig` contains exactly one failing signature:
  - `signature_id`
  - `Good_value`
  - `bad_value`
  - `alert_threshold`
  - `platform`

Implementation notes:
  - The `alerts` CSV field is a Python literal string, so it is parsed with
    `ast.literal_eval`, not `json.loads`.
  - `num_candidate_revisions` is computed from commit indices in
    `all_commits.jsonl`, which is validated to be parent-before-child ordered.
    This avoids relying on Mercurial timestamps.
  - Rows are excluded if the culprit revision is not inside the Mercurial DAG
    range `(good_revision, bad_revision]`.
  - Eval and final-test split boundaries are computed from the shuffled
    `samples` arrays in the prediction JSON files by locating those commits in
    the parent-before-child `all_commits.jsonl` order.
  - If `per_sig_perf_data_info.jsonl` is present, its `alert_threshold` and
    `platform` fields are used. Otherwise `alert_threshold` remains `null`,
    and `platform` falls back to the alert payload's
    `series_signature.machine_platform`.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from dataclasses import dataclass
import json
import os
import sys
from typing import Any


csv.field_size_limit(sys.maxsize)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

DEFAULT_ALERT_SUMMARIES_CSV = os.path.join(
    REPO_ROOT,
    "datasets",
    "mozilla_perf_bisect",
    "alert_summaries.csv",
)
DEFAULT_COMMITS_JSONL = os.path.join(
    REPO_ROOT,
    "datasets",
    "mozilla_perf_bisect",
    "all_commits.jsonl",
)
DEFAULT_FILTER_CSV = os.path.join(
    REPO_ROOT,
    "datasets",
    "mozilla_perf_bisect",
    "alert_summary_fail_perf_sigs_no_fw_2_6_18.csv",
)
DEFAULT_SIG_INFO_JSONL = os.path.join(
    REPO_ROOT,
    "datasets",
    "mozilla_perf_bisect",
    "per_sig_perf_data_info.jsonl",
)
DEFAULT_EVAL_PREDS_JSON = os.path.join(
    REPO_ROOT,
    "datasets",
    "mozilla_perf_bisect",
    "final_test_results_perf_codebert_eval.json",
)
DEFAULT_FINAL_TEST_PREDS_JSON = os.path.join(
    REPO_ROOT,
    "datasets",
    "mozilla_perf_bisect",
    "final_test_results_perf_codebert_final_test.json",
)
DEFAULT_EVAL_OUTPUT_JSONL = os.path.join(
    REPO_ROOT,
    "datasets",
    "mozilla_perf_bisect",
    "perf_bisect_regressions_eval.jsonl",
)
DEFAULT_FINAL_TEST_OUTPUT_JSONL = os.path.join(
    REPO_ROOT,
    "datasets",
    "mozilla_perf_bisect",
    "perf_bisect_regressions_final_test.jsonl",
)

NULL_NODE = "0000000000000000000000000000000000000000"


@dataclass(frozen=True)
class SplitBoundary:
    name: str
    start_index: int
    end_index: int
    start_revision: str
    end_revision: str
    sample_count: int
    unique_commit_count: int
    known_commit_count: int
    missing_commit_count: int


@dataclass
class CommitGraph:
    node_to_index: dict[str, int]
    parents_by_node: dict[str, list[str]]

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        if ancestor == descendant:
            return ancestor in self.node_to_index

        ancestor_index = self.node_to_index.get(ancestor)
        descendant_index = self.node_to_index.get(descendant)
        if ancestor_index is None or descendant_index is None:
            return False
        if ancestor_index > descendant_index:
            return False

        stack = [descendant]
        seen: set[str] = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            if node == ancestor:
                return True
            for parent in self.parents_by_node.get(node, []):
                parent_index = self.node_to_index.get(parent)
                if parent_index is not None and parent_index >= ancestor_index:
                    stack.append(parent)

        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create eval and final-test perf bisect regression JSONL files "
            "from Treeherder alert summaries and Mercurial commit history."
        )
    )
    parser.add_argument(
        "--alert-summaries-csv",
        default=DEFAULT_ALERT_SUMMARIES_CSV,
        help="Path to alert_summaries.csv.",
    )
    parser.add_argument(
        "--commits-jsonl",
        default=DEFAULT_COMMITS_JSONL,
        help="Path to all_commits.jsonl.",
    )
    parser.add_argument(
        "--filter-csv",
        default=DEFAULT_FILTER_CSV,
        help=(
            "CSV containing the alert_summary_id allowlist used to filter "
            "alert_summaries.csv rows."
        ),
    )
    parser.add_argument(
        "--sig-info-jsonl",
        default=DEFAULT_SIG_INFO_JSONL,
        help=(
            "Optional JSONL with per-signature metadata such as alert_threshold "
            "and platform. Missing files are tolerated."
        ),
    )
    parser.add_argument(
        "--eval-preds-json",
        default=DEFAULT_EVAL_PREDS_JSON,
        help=(
            "Prediction JSON whose samples define the eval split boundary."
        ),
    )
    parser.add_argument(
        "--final-test-preds-json",
        default=DEFAULT_FINAL_TEST_PREDS_JSON,
        help=(
            "Prediction JSON whose samples define the final-test split "
            "boundary."
        ),
    )
    parser.add_argument(
        "--eval-output",
        default=DEFAULT_EVAL_OUTPUT_JSONL,
        help="Eval output JSONL path.",
    )
    parser.add_argument(
        "--final-test-output",
        default=DEFAULT_FINAL_TEST_OUTPUT_JSONL,
        help="Final-test output JSONL path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, stop after writing the first N total output rows.",
    )
    return parser.parse_args()


def ensure_file_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")


def ensure_parent_dir(path: str) -> None:
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_num, json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {e}") from e


def load_commit_graph(commits_jsonl: str) -> CommitGraph:
    node_to_index: dict[str, int] = {}
    parents_by_node: dict[str, list[str]] = {}
    total_commits = 0

    for _, record in iter_jsonl(commits_jsonl):
        if not isinstance(record, dict):
            continue

        node = record.get("node")
        if not isinstance(node, str) or not node:
            continue

        if node in node_to_index:
            raise ValueError(f"Duplicate commit node encountered: {node}")

        current_index = len(node_to_index)
        parents = [
            parent
            for parent in record.get("parents", [])
            if isinstance(parent, str) and parent
        ]

        for parent in parents:
            if parent == NULL_NODE:
                continue
            parent_index = node_to_index.get(parent)
            if parent_index is None:
                raise ValueError(
                    f"Commit {node} references missing parent {parent}."
                )
            if parent_index >= current_index:
                raise ValueError(
                    "all_commits.jsonl is not parent-before-child ordered: "
                    f"commit {node} at index {current_index} has parent {parent} "
                    f"at index {parent_index}."
                )

        node_to_index[node] = current_index
        parents_by_node[node] = [parent for parent in parents if parent != NULL_NODE]
        total_commits += 1

    print(f"Loaded {total_commits} commits from {commits_jsonl}.")
    return CommitGraph(node_to_index=node_to_index, parents_by_node=parents_by_node)


def classify_culprit_boundary(
    *,
    commit_graph: CommitGraph,
    good_revision: str,
    bad_revision: str,
    culprit_revision: str,
) -> str | None:
    if not commit_graph.is_ancestor(good_revision, bad_revision):
        return "good_revision_is_not_an_ancestor_of_bad_revision"
    if culprit_revision == good_revision:
        return "culprit_revision_equals_good_revision"

    good_ancestor_culprit = commit_graph.is_ancestor(
        good_revision,
        culprit_revision,
    )
    culprit_ancestor_bad = commit_graph.is_ancestor(
        culprit_revision,
        bad_revision,
    )
    if good_ancestor_culprit and culprit_ancestor_bad:
        return None
    if not good_ancestor_culprit and culprit_ancestor_bad:
        return "culprit_revision_is_not_after_good_revision"
    if good_ancestor_culprit and not culprit_ancestor_bad:
        return "culprit_revision_is_not_at_or_before_bad_revision"
    return "culprit_revision_is_not_on_good_to_bad_ancestry_range"


def load_prediction_sample_commit_ids(predictions_json: str) -> tuple[set[str], int]:
    ensure_file_exists(predictions_json)

    try:
        with open(predictions_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {predictions_json}: {e}") from e

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected top-level object in {predictions_json}, "
            f"got {type(payload).__name__}."
        )

    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError(
            f"Expected {predictions_json} to contain a list field named "
            "'samples'."
        )

    commit_ids: set[str] = set()
    duplicate_commit_ids = 0
    invalid_samples = 0

    for sample in samples:
        if not isinstance(sample, dict):
            invalid_samples += 1
            continue

        commit_id = sample.get("commit_id")
        if not isinstance(commit_id, str) or not commit_id.strip():
            invalid_samples += 1
            continue

        commit_id = commit_id.strip()
        if commit_id in commit_ids:
            duplicate_commit_ids += 1
        commit_ids.add(commit_id)

    if not commit_ids:
        raise ValueError(
            f"No valid sample commit_id values found in {predictions_json}."
        )

    print(
        f"Loaded {len(commit_ids)} unique sample commit ids from "
        f"{len(samples)} samples in {predictions_json}."
    )
    if duplicate_commit_ids:
        print(
            f"[WARN] Ignored {duplicate_commit_ids} duplicate sample commit ids "
            f"in {predictions_json}."
        )
    if invalid_samples:
        print(
            f"[WARN] Ignored {invalid_samples} samples without a valid commit_id "
            f"in {predictions_json}."
        )

    return commit_ids, len(samples)


def build_split_boundary(
    *,
    name: str,
    predictions_json: str,
    node_to_index: dict[str, int],
) -> SplitBoundary:
    commit_ids, sample_count = load_prediction_sample_commit_ids(predictions_json)

    known_commits: list[tuple[int, str]] = []
    missing_commit_count = 0
    for commit_id in commit_ids:
        index = node_to_index.get(commit_id)
        if index is None:
            missing_commit_count += 1
            continue
        known_commits.append((index, commit_id))

    if not known_commits:
        raise ValueError(
            f"None of the sample commit ids in {predictions_json} were found "
            "in all_commits.jsonl."
        )

    start_index, start_revision = min(known_commits)
    end_index, end_revision = max(known_commits)

    if missing_commit_count:
        print(
            f"[WARN] {missing_commit_count} unique sample commit ids from "
            f"{predictions_json} were not found in all_commits.jsonl."
        )

    print(
        f"{name} boundary: {start_revision} (index {start_index}) -> "
        f"{end_revision} (index {end_index}); "
        f"{len(known_commits)} known unique commits."
    )

    return SplitBoundary(
        name=name,
        start_index=start_index,
        end_index=end_index,
        start_revision=start_revision,
        end_revision=end_revision,
        sample_count=sample_count,
        unique_commit_count=len(commit_ids),
        known_commit_count=len(known_commits),
        missing_commit_count=missing_commit_count,
    )


def revisions_within_boundary(
    good_index: int,
    bad_index: int,
    boundary: SplitBoundary,
) -> bool:
    return (
        boundary.start_index <= good_index <= boundary.end_index
        and boundary.start_index <= bad_index <= boundary.end_index
    )


def load_signature_metadata(sig_info_jsonl: str) -> dict[int, dict[str, Any]]:
    if not sig_info_jsonl or not os.path.exists(sig_info_jsonl):
        print(
            "Optional per-signature metadata cache not found; "
            "alert_threshold will remain null when not present in alerts."
        )
        return {}

    metadata_by_signature: dict[int, dict[str, Any]] = {}
    for line_num, record in iter_jsonl(sig_info_jsonl):
        if not isinstance(record, dict):
            continue

        raw_signature_id = record.get("signature_id")
        try:
            signature_id = int(raw_signature_id)
        except Exception:
            print(
                f"[WARN] Skipping invalid signature_id in {sig_info_jsonl}:{line_num}: "
                f"{raw_signature_id!r}"
            )
            continue

        metadata_by_signature[signature_id] = {
            "alert_threshold": record.get("alert_threshold"),
            "platform": record.get("platform"),
        }

    print(
        f"Loaded optional metadata for {len(metadata_by_signature)} signatures "
        f"from {sig_info_jsonl}."
    )
    return metadata_by_signature


def load_allowed_summary_ids(filter_csv: str) -> set[int]:
    ensure_file_exists(filter_csv)

    allowed_summary_ids: set[int] = set()
    with open(filter_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):
            raw_summary_id = row.get("alert_summary_id")
            try:
                summary_id = int(str(raw_summary_id).strip())
            except Exception:
                print(
                    f"[WARN] Skipping invalid alert_summary_id in "
                    f"{filter_csv}:{row_num}: {raw_summary_id!r}"
                )
                continue
            allowed_summary_ids.add(summary_id)

    print(
        f"Loaded {len(allowed_summary_ids)} allowed alert summary ids from "
        f"{filter_csv}."
    )
    return allowed_summary_ids


def parse_alerts(raw_alerts: str, *, row_num: int) -> list[dict[str, Any]]:
    raw_alerts = (raw_alerts or "").strip()
    if not raw_alerts:
        return []

    try:
        parsed = ast.literal_eval(raw_alerts)
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Invalid alerts payload at CSV row {row_num}: {e}") from e

    if not isinstance(parsed, list):
        raise ValueError(
            f"Expected alerts to parse to a list at CSV row {row_num}, "
            f"got {type(parsed).__name__}."
        )

    return [alert for alert in parsed if isinstance(alert, dict)]


def build_failing_sig_records(
    alerts: list[dict[str, Any]],
    *,
    signature_metadata: dict[int, dict[str, Any]],
    stats: Counter[str],
) -> list[dict[str, Any]]:
    failing_sig_records: list[dict[str, Any]] = []
    seen_signature_ids: set[int] = set()

    for alert in alerts:
        if not alert.get("is_regression"):
            continue

        series_signature = alert.get("series_signature")
        if not isinstance(series_signature, dict):
            stats["alerts_missing_series_signature"] += 1
            continue

        raw_signature_id = series_signature.get("id")
        try:
            signature_id = int(raw_signature_id)
        except Exception:
            stats["alerts_missing_signature_id"] += 1
            continue

        if signature_id in seen_signature_ids:
            stats["duplicate_signature_ids_within_summary"] += 1
            continue
        seen_signature_ids.add(signature_id)

        metadata = signature_metadata.get(signature_id, {})
        alert_threshold = metadata.get("alert_threshold")
        if alert_threshold is None:
            alert_threshold = alert.get("alert_threshold")
            if alert_threshold is None:
                stats["failing_sigs_missing_alert_threshold"] += 1

        platform = metadata.get("platform")
        if platform is None:
            platform = alert.get("platform")
        if platform is None:
            platform = series_signature.get("machine_platform")
        if platform is None:
            stats["failing_sigs_missing_platform"] += 1

        failing_sig_records.append(
            {
                "signature_id": signature_id,
                "Good_value": alert.get("prev_value"),
                "bad_value": alert.get("new_value"),
                "alert_threshold": alert_threshold,
                "platform": platform,
            }
        )

    return failing_sig_records


def write_output_rows(
    dst,
    *,
    split_name: str,
    summary_id: int,
    good_revision: str,
    bad_revision: str,
    num_candidate_revisions: int,
    culprit_revision: str,
    failing_sig_records: list[dict[str, Any]],
    stats: Counter[str],
    limit: int,
) -> bool:
    for failing_sig in failing_sig_records:
        if limit > 0 and stats["rows_written"] >= limit:
            return True

        output_record = {
            "alert_summary_id": summary_id,
            "good_revision": good_revision,
            "bad_revision": bad_revision,
            "num_candidate_revisions": num_candidate_revisions,
            "culprit_revision": culprit_revision,
            "failing_sig": failing_sig,
        }
        dst.write(json.dumps(output_record))
        dst.write("\n")
        stats["rows_written"] += 1
        stats[f"rows_written_{split_name}"] += 1

    return False


def create_dataset(
    *,
    alert_summaries_csv: str,
    commits_jsonl: str,
    filter_csv: str,
    sig_info_jsonl: str,
    eval_preds_json: str,
    final_test_preds_json: str,
    eval_output_jsonl: str,
    final_test_output_jsonl: str,
    limit: int,
) -> None:
    ensure_file_exists(alert_summaries_csv)
    ensure_file_exists(commits_jsonl)
    ensure_parent_dir(eval_output_jsonl)
    ensure_parent_dir(final_test_output_jsonl)

    commit_graph = load_commit_graph(commits_jsonl)
    node_to_index = commit_graph.node_to_index
    eval_boundary = build_split_boundary(
        name="eval",
        predictions_json=eval_preds_json,
        node_to_index=node_to_index,
    )
    final_test_boundary = build_split_boundary(
        name="final_test",
        predictions_json=final_test_preds_json,
        node_to_index=node_to_index,
    )
    if (
        eval_boundary.start_index <= final_test_boundary.end_index
        and final_test_boundary.start_index <= eval_boundary.end_index
    ):
        print(
            "[WARN] Eval and final-test split boundaries overlap; rows whose "
            "good_revision and bad_revision fit both ranges will be written "
            "to the eval output only."
        )

    allowed_summary_ids = load_allowed_summary_ids(filter_csv)
    signature_metadata = load_signature_metadata(sig_info_jsonl)
    stats: Counter[str] = Counter()
    seen_summary_ids: set[int] = set()

    with open(alert_summaries_csv, "r", newline="", encoding="utf-8") as src, open(
        eval_output_jsonl, "w", encoding="utf-8"
    ) as eval_dst, open(
        final_test_output_jsonl, "w", encoding="utf-8"
    ) as final_test_dst:
        reader = csv.DictReader(src)
        for row_num, row in enumerate(reader, start=2):
            if limit > 0 and stats["rows_written"] >= limit:
                break

            stats["rows_total"] += 1

            raw_summary_id = row.get("id")
            try:
                summary_id = int(str(raw_summary_id).strip())
            except Exception:
                stats["rows_skipped_invalid_summary_id"] += 1
                continue

            if summary_id not in allowed_summary_ids:
                stats["rows_skipped_not_in_filter"] += 1
                continue

            if summary_id in seen_summary_ids:
                stats["rows_skipped_duplicate_summary_id"] += 1
                continue
            seen_summary_ids.add(summary_id)

            good_revision = (row.get("original_prev_push_revision") or "").strip()
            bad_revision = (row.get("original_revision") or "").strip()
            culprit_revision = (row.get("revision") or "").strip()

            if not good_revision or not bad_revision or not culprit_revision:
                stats["rows_skipped_missing_revision_fields"] += 1
                continue

            good_index = node_to_index.get(good_revision)
            bad_index = node_to_index.get(bad_revision)
            culprit_index = node_to_index.get(culprit_revision)
            if good_index is None or bad_index is None or culprit_index is None:
                stats["rows_skipped_missing_commit_nodes"] += 1
                continue

            if bad_index <= good_index:
                stats["rows_skipped_non_forward_revision_range"] += 1
                continue

            matches_eval = revisions_within_boundary(
                good_index,
                bad_index,
                eval_boundary,
            )
            matches_final_test = revisions_within_boundary(
                good_index,
                bad_index,
                final_test_boundary,
            )

            if matches_eval:
                matching_outputs = [(eval_boundary.name, eval_dst)]
                if matches_final_test:
                    stats["rows_matching_multiple_split_boundaries"] += 1
            elif matches_final_test:
                matching_outputs = [(final_test_boundary.name, final_test_dst)]
            else:
                stats["rows_skipped_outside_split_boundaries"] += 1
                continue

            num_candidate_revisions = bad_index - good_index - 1
            if num_candidate_revisions <= 1:
                stats["rows_skipped_too_few_candidate_revisions"] += 1
                continue

            culprit_boundary_reason = classify_culprit_boundary(
                commit_graph=commit_graph,
                good_revision=good_revision,
                bad_revision=bad_revision,
                culprit_revision=culprit_revision,
            )
            if culprit_boundary_reason is not None:
                stats["rows_skipped_culprit_outside_revision_range"] += 1
                stats[f"rows_skipped_{culprit_boundary_reason}"] += 1
                print(
                    "[WARN] Excluding alert summary "
                    f"{summary_id} from CSV row {row_num}: "
                    f"{culprit_boundary_reason}; "
                    f"good_revision={good_revision}, "
                    f"culprit_revision={culprit_revision}, "
                    f"bad_revision={bad_revision}."
                )
                continue

            try:
                alerts = parse_alerts(row.get("alerts") or "", row_num=row_num)
            except ValueError as e:
                print(f"[WARN] {e}")
                stats["rows_skipped_invalid_alerts"] += 1
                continue

            failing_sig_records = build_failing_sig_records(
                alerts,
                signature_metadata=signature_metadata,
                stats=stats,
            )
            if not failing_sig_records:
                stats["rows_skipped_no_valid_failing_sigs"] += 1
                continue

            for split_name, dst in matching_outputs:
                limit_reached = write_output_rows(
                    dst,
                    split_name=split_name,
                    summary_id=summary_id,
                    good_revision=good_revision,
                    bad_revision=bad_revision,
                    num_candidate_revisions=num_candidate_revisions,
                    culprit_revision=culprit_revision,
                    failing_sig_records=failing_sig_records,
                    stats=stats,
                    limit=limit,
                )
                if limit_reached:
                    break
            if limit_reached:
                break

    print(f"Wrote {stats['rows_written_eval']} rows to {eval_output_jsonl}.")
    print(
        f"Wrote {stats['rows_written_final_test']} rows to "
        f"{final_test_output_jsonl}."
    )
    print(f"Wrote {stats['rows_written']} rows total.")
    print("Summary:")
    print(f"  rows_total={stats['rows_total']}")
    print(f"  rows_written={stats['rows_written']}")
    print(f"  rows_written_eval={stats['rows_written_eval']}")
    print(f"  rows_written_final_test={stats['rows_written_final_test']}")
    print(
        "  rows_skipped_missing_revision_fields="
        f"{stats['rows_skipped_missing_revision_fields']}"
    )
    print(
        "  rows_skipped_not_in_filter="
        f"{stats['rows_skipped_not_in_filter']}"
    )
    print(
        "  rows_skipped_duplicate_summary_id="
        f"{stats['rows_skipped_duplicate_summary_id']}"
    )
    print(
        "  rows_skipped_missing_commit_nodes="
        f"{stats['rows_skipped_missing_commit_nodes']}"
    )
    print(
        "  rows_skipped_outside_split_boundaries="
        f"{stats['rows_skipped_outside_split_boundaries']}"
    )
    print(
        "  rows_matching_multiple_split_boundaries="
        f"{stats['rows_matching_multiple_split_boundaries']}"
    )
    print(
        "  rows_skipped_invalid_alerts="
        f"{stats['rows_skipped_invalid_alerts']}"
    )
    print(
        "  rows_skipped_no_valid_failing_sigs="
        f"{stats['rows_skipped_no_valid_failing_sigs']}"
    )
    print(
        "  rows_skipped_non_forward_revision_range="
        f"{stats['rows_skipped_non_forward_revision_range']}"
    )
    print(
        "  rows_skipped_culprit_outside_revision_range="
        f"{stats['rows_skipped_culprit_outside_revision_range']}"
    )
    print(
        "  rows_skipped_too_few_candidate_revisions="
        f"{stats['rows_skipped_too_few_candidate_revisions']}"
    )
    print(
        "  failing_sigs_missing_alert_threshold="
        f"{stats['failing_sigs_missing_alert_threshold']}"
    )
    print(
        "  failing_sigs_missing_platform="
        f"{stats['failing_sigs_missing_platform']}"
    )


def main() -> None:
    args = parse_args()
    create_dataset(
        alert_summaries_csv=args.alert_summaries_csv,
        commits_jsonl=args.commits_jsonl,
        filter_csv=args.filter_csv,
        sig_info_jsonl=args.sig_info_jsonl,
        eval_preds_json=args.eval_preds_json,
        final_test_preds_json=args.final_test_preds_json,
        eval_output_jsonl=args.eval_output,
        final_test_output_jsonl=args.final_test_output,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
