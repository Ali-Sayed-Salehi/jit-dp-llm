#!/usr/bin/env python3
"""
Build a perf-bisect regression dataset from Treeherder alert summaries.

Inputs:
  - `datasets/mozilla_perf_bisect/alert_summaries.csv`
  - `datasets/mozilla_perf_bisect/all_commits.jsonl`
  - `datasets/mozilla_perf_bisect/alert_summary_fail_perf_sigs_no_fw_2_6_18.csv`
  - Optional: `datasets/mozilla_perf_bisect/per_sig_perf_data_info.jsonl`

Output:
  - `datasets/mozilla_perf/perf_bisect_regressions.jsonl`

Each output JSONL row contains:
  - `alert_summary_id`
  - `good_revision`
  - `bad_revision`
  - `num_candidate_revisions`
  - `culprit_revision`
  - `failing_sigs`

`failing_sigs` is a dict keyed by signature id string. Each value contains:
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
DEFAULT_OUTPUT_JSONL = os.path.join(
    REPO_ROOT,
    "datasets",
    "mozilla_perf",
    "perf_bisect_regressions.jsonl",
)

NULL_NODE = "0000000000000000000000000000000000000000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create perf_bisect_regressions.jsonl from Treeherder alert "
            "summaries and Mercurial commit history."
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
        "--output",
        default=DEFAULT_OUTPUT_JSONL,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, stop after writing the first N output rows.",
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


def load_commit_indices(commits_jsonl: str) -> dict[str, int]:
    node_to_index: dict[str, int] = {}
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
        total_commits += 1

    print(f"Loaded {total_commits} commits from {commits_jsonl}.")
    return node_to_index


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


def build_failing_sigs(
    alerts: list[dict[str, Any]],
    *,
    signature_metadata: dict[int, dict[str, Any]],
    stats: Counter[str],
) -> dict[str, dict[str, Any]]:
    failing_sigs: dict[str, dict[str, Any]] = {}

    for alert in alerts:
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

        signature_key = str(signature_id)
        if signature_key in failing_sigs:
            stats["duplicate_signature_ids_within_summary"] += 1
            continue

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

        failing_sigs[signature_key] = {
            "Good_value": alert.get("prev_value"),
            "bad_value": alert.get("new_value"),
            "alert_threshold": alert_threshold,
            "platform": platform,
        }

    return failing_sigs


def create_dataset(
    *,
    alert_summaries_csv: str,
    commits_jsonl: str,
    filter_csv: str,
    sig_info_jsonl: str,
    output_jsonl: str,
    limit: int,
) -> None:
    ensure_file_exists(alert_summaries_csv)
    ensure_file_exists(commits_jsonl)
    ensure_parent_dir(output_jsonl)

    node_to_index = load_commit_indices(commits_jsonl)
    allowed_summary_ids = load_allowed_summary_ids(filter_csv)
    signature_metadata = load_signature_metadata(sig_info_jsonl)
    stats: Counter[str] = Counter()
    seen_summary_ids: set[int] = set()

    with open(alert_summaries_csv, "r", newline="", encoding="utf-8") as src, open(
        output_jsonl, "w", encoding="utf-8"
    ) as dst:
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
            if good_index is None or bad_index is None:
                stats["rows_skipped_missing_commit_nodes"] += 1
                continue

            if bad_index <= good_index:
                stats["rows_skipped_non_forward_revision_range"] += 1
                continue

            try:
                alerts = parse_alerts(row.get("alerts") or "", row_num=row_num)
            except ValueError as e:
                print(f"[WARN] {e}")
                stats["rows_skipped_invalid_alerts"] += 1
                continue

            failing_sigs = build_failing_sigs(
                alerts,
                signature_metadata=signature_metadata,
                stats=stats,
            )
            if not failing_sigs:
                stats["rows_skipped_no_valid_failing_sigs"] += 1
                continue

            output_record = {
                "alert_summary_id": summary_id,
                "good_revision": good_revision,
                "bad_revision": bad_revision,
                "num_candidate_revisions": bad_index - good_index - 1,
                "culprit_revision": culprit_revision,
                "failing_sigs": failing_sigs,
            }
            dst.write(json.dumps(output_record))
            dst.write("\n")
            stats["rows_written"] += 1

    print(f"Wrote {stats['rows_written']} rows to {output_jsonl}.")
    print("Summary:")
    print(f"  rows_total={stats['rows_total']}")
    print(f"  rows_written={stats['rows_written']}")
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
        output_jsonl=args.output,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
