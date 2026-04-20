#!/usr/bin/env python3
"""
Build per-revision performance measurement data from per-signature JSONL caches.

Inputs:
  - `datasets/mozilla_perf_bisect/all_commits.jsonl`
  - `datasets/mozilla_perf_bisect/per_sig_perf_data_replicates.jsonl`
  - `datasets/mozilla_perf_bisect/per_sig_perf_data_summary.jsonl`

Output:
  - `datasets/mozilla_perf_bisect/per_revision_perf_data.jsonl`

Each output line contains one commit from `all_commits.jsonl`, plus a new
`perf_measurement_data` field. That field is a list of matching measurement
samples whose `revision` equals the commit's `node`. Every matched sample is
augmented with:
  - `signature_id`
  - `replicate` (`True` for the replicates JSONL, `False` for the summary JSONL)

Time window:
  - Inclusive start: 2025-06-01T00:00:00+00:00
  - Exclusive end: 2025-11-01T00:00:00+00:00

Windowing note:
  - Commit timestamps can be noisy, so the script does not filter every commit
    by timestamp. Instead, it uses timestamps only to identify the start/end
    boundary cuts and checks two commits on each side of each boundary to avoid
    choosing an obviously out-of-place timestamp. Once those cut positions are
    chosen, the script keeps the topological commit slice between them, relying
    on parent-before-child ordering in `all_commits.jsonl`.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from typing import Any


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATASET_DIR = os.path.join(REPO_ROOT, "datasets", "mozilla_perf_bisect")

DEFAULT_COMMITS_JSONL = os.path.join(DATASET_DIR, "all_commits.jsonl")
DEFAULT_REPLICATES_JSONL = os.path.join(
    DATASET_DIR,
    "per_sig_perf_data_replicates.jsonl",
)
DEFAULT_SUMMARY_JSONL = os.path.join(
    DATASET_DIR,
    "per_sig_perf_data_summary.jsonl",
)
DEFAULT_OUTPUT_JSONL = os.path.join(
    DATASET_DIR,
    "per_revision_perf_data.jsonl",
)

START_DATE = datetime(2025, 6, 1, tzinfo=UTC)
END_DATE_EXCLUSIVE = datetime(2025, 11, 1, tzinfo=UTC)
BOUNDARY_NEIGHBORHOOD = 2


@dataclass(frozen=True)
class CommitEntry:
    index: int
    node: str
    parents: tuple[str, ...]
    timestamp_utc: datetime | None
    record: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join commits in the target revision window with per-signature perf "
            "samples and write one JSONL row per revision."
        )
    )
    parser.add_argument(
        "--commits-jsonl",
        default=DEFAULT_COMMITS_JSONL,
        help="Input JSONL path containing Mercurial commit metadata.",
    )
    parser.add_argument(
        "--replicates-jsonl",
        default=DEFAULT_REPLICATES_JSONL,
        help="Input JSONL path containing replicate perf measurement data.",
    )
    parser.add_argument(
        "--summary-jsonl",
        default=DEFAULT_SUMMARY_JSONL,
        help="Input JSONL path containing summary perf measurement data.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_JSONL,
        help="Output JSONL path for per-revision perf measurement data.",
    )
    parser.add_argument(
        "--limit-commits",
        type=int,
        default=0,
        help="If > 0, keep only the first N commits after the boundary cuts.",
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
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping invalid JSON at {path}:{line_num}: {e}")


def parse_hg_commit_datetime_utc(record: dict[str, Any]) -> datetime | None:
    raw_date = record.get("date")
    if not isinstance(raw_date, list) or not raw_date:
        return None

    try:
        timestamp = float(raw_date[0])
    except Exception:
        return None

    return datetime.fromtimestamp(timestamp, tz=UTC)


def load_all_commits(commits_jsonl: str) -> list[CommitEntry]:
    entries: list[CommitEntry] = []
    node_to_index: dict[str, int] = {}

    for raw_index, record in enumerate(iter_jsonl(commits_jsonl)):
        if not isinstance(record, dict):
            continue

        node = record.get("node")
        if not isinstance(node, str) or not node:
            continue

        if node in node_to_index:
            raise ValueError(f"Duplicate commit node encountered: {node}")

        parents = tuple(
            parent
            for parent in record.get("parents", [])
            if isinstance(parent, str) and parent
        )
        timestamp_utc = parse_hg_commit_datetime_utc(record)
        entries.append(
            CommitEntry(
                index=len(entries),
                node=node,
                parents=parents,
                timestamp_utc=timestamp_utc,
                record=record,
            )
        )
        node_to_index[node] = len(entries) - 1

    for entry in entries:
        for parent in entry.parents:
            if parent == "0000000000000000000000000000000000000000":
                continue

            parent_index = node_to_index.get(parent)
            if parent_index is None:
                raise ValueError(
                    f"Commit {entry.node} references missing parent {parent}."
                )
            if parent_index >= entry.index:
                raise ValueError(
                    "all_commits.jsonl is not parent-before-child ordered: "
                    f"commit {entry.node} at index {entry.index} has parent {parent} "
                    f"at index {parent_index}."
                )

    return entries


def find_raw_boundary_cut_index(
    entries: list[CommitEntry],
    boundary_time: datetime,
) -> int:
    for entry in entries:
        if entry.timestamp_utc is not None and entry.timestamp_utc >= boundary_time:
            return entry.index
    return len(entries)


def score_boundary_cut(
    entries: list[CommitEntry],
    boundary_time: datetime,
    cut_index: int,
) -> tuple[int, int]:
    left_start = max(0, cut_index - BOUNDARY_NEIGHBORHOOD)
    right_end = min(len(entries), cut_index + BOUNDARY_NEIGHBORHOOD + 1)

    good_left = 0
    for idx in range(left_start, cut_index):
        timestamp_utc = entries[idx].timestamp_utc
        if timestamp_utc is not None and timestamp_utc < boundary_time:
            good_left += 1

    good_right = 0
    for idx in range(cut_index, right_end):
        timestamp_utc = entries[idx].timestamp_utc
        if timestamp_utc is not None and timestamp_utc >= boundary_time:
            good_right += 1

    return good_left + good_right, good_right


def print_boundary_context(
    entries: list[CommitEntry],
    *,
    label: str,
    boundary_time: datetime,
    raw_cut_index: int,
    selected_cut_index: int,
) -> None:
    print(
        f"{label} boundary at {boundary_time.isoformat()}: "
        f"raw_cut_index={raw_cut_index}, selected_cut_index={selected_cut_index}"
    )

    context_start = max(0, selected_cut_index - BOUNDARY_NEIGHBORHOOD - 1)
    context_end = min(len(entries), selected_cut_index + BOUNDARY_NEIGHBORHOOD + 2)
    for idx in range(context_start, context_end):
        entry = entries[idx]
        timestamp_text = (
            entry.timestamp_utc.isoformat() if entry.timestamp_utc is not None else "None"
        )
        relation = ">= " if (
            entry.timestamp_utc is not None and entry.timestamp_utc >= boundary_time
        ) else "<  "
        marker = "|CUT|" if idx == selected_cut_index else "     "
        print(
            f"  {marker} idx={idx} {relation} ts={timestamp_text} "
            f"node={entry.node}"
        )

    if selected_cut_index == len(entries):
        print("  |CUT| at dataset end (no commit on or after boundary).")


def choose_boundary_cut_index(
    entries: list[CommitEntry],
    *,
    boundary_time: datetime,
    label: str,
) -> int:
    raw_cut_index = find_raw_boundary_cut_index(entries, boundary_time)
    if raw_cut_index == len(entries):
        selected_cut_index = len(entries)
        print_boundary_context(
            entries,
            label=label,
            boundary_time=boundary_time,
            raw_cut_index=raw_cut_index,
            selected_cut_index=selected_cut_index,
        )
        return selected_cut_index

    candidate_start = max(0, raw_cut_index - BOUNDARY_NEIGHBORHOOD)
    candidate_end = min(len(entries), raw_cut_index + BOUNDARY_NEIGHBORHOOD + 1)

    best_cut_index = raw_cut_index
    best_score = (-1, -1)
    best_distance = float("inf")

    for cut_index in range(candidate_start, candidate_end + 1):
        score = score_boundary_cut(entries, boundary_time, cut_index)
        distance = abs(cut_index - raw_cut_index)
        if (
            score > best_score
            or (score == best_score and distance < best_distance)
            or (
                score == best_score
                and distance == best_distance
                and cut_index < best_cut_index
            )
        ):
            best_cut_index = cut_index
            best_score = score
            best_distance = distance

    print_boundary_context(
        entries,
        label=label,
        boundary_time=boundary_time,
        raw_cut_index=raw_cut_index,
        selected_cut_index=best_cut_index,
    )
    return best_cut_index


def select_commit_window(
    entries: list[CommitEntry],
    *,
    limit_commits: int,
) -> list[CommitEntry]:
    start_cut_index = choose_boundary_cut_index(
        entries,
        boundary_time=START_DATE,
        label="Start",
    )
    end_cut_index = choose_boundary_cut_index(
        entries,
        boundary_time=END_DATE_EXCLUSIVE,
        label="End",
    )

    if end_cut_index < start_cut_index:
        raise ValueError(
            f"End cut {end_cut_index} is earlier than start cut {start_cut_index}."
        )

    selected_entries = entries[start_cut_index:end_cut_index]
    if limit_commits > 0:
        selected_entries = selected_entries[:limit_commits]

    return selected_entries


def normalize_signature_id(raw_signature_id: Any) -> Any:
    try:
        return int(raw_signature_id)
    except Exception:
        return raw_signature_id


def load_samples_by_revision(
    input_jsonl: str,
    *,
    replicate: bool,
    target_revisions: set[str],
    samples_by_revision: dict[str, list[dict[str, Any]]],
) -> int:
    signature_rows = 0
    total_samples = 0
    matched_samples = 0

    for record in iter_jsonl(input_jsonl):
        if not isinstance(record, dict):
            continue

        signature_rows += 1
        signature_id = normalize_signature_id(record.get("signature_id"))
        perf_measurement_data = record.get("perf_measurement_data")
        if not isinstance(perf_measurement_data, list):
            continue

        for sample in perf_measurement_data:
            if not isinstance(sample, dict):
                continue

            total_samples += 1
            revision = sample.get("revision")
            if not isinstance(revision, str) or revision not in target_revisions:
                continue

            augmented_sample = dict(sample)
            augmented_sample["signature_id"] = signature_id
            augmented_sample["replicate"] = replicate
            samples_by_revision[revision].append(augmented_sample)
            matched_samples += 1

    print(
        f"Loaded {matched_samples} matching samples from {input_jsonl} "
        f"(replicate={replicate}, signature_rows={signature_rows}, "
        f"total_samples_scanned={total_samples})."
    )
    return matched_samples


def write_output_jsonl(
    output_jsonl: str,
    selected_entries: list[CommitEntry],
    samples_by_revision: dict[str, list[dict[str, Any]]],
) -> tuple[int, int]:
    ensure_parent_dir(output_jsonl)

    commits_with_samples = 0
    total_samples_written = 0
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for entry in selected_entries:
            if "perf_measurement_data" in entry.record:
                raise ValueError(
                    "Commit record already contains `perf_measurement_data`; "
                    "refusing to overwrite it."
                )

            perf_measurement_data = samples_by_revision.get(entry.node, [])
            if perf_measurement_data:
                commits_with_samples += 1
                total_samples_written += len(perf_measurement_data)

            output_record = dict(entry.record)
            output_record.pop("desc", None)
            output_record["perf_measurement_data"] = perf_measurement_data
            f.write(json.dumps(output_record))
            f.write("\n")

    return commits_with_samples, total_samples_written


def main() -> None:
    args = parse_args()

    ensure_file_exists(args.commits_jsonl)
    ensure_file_exists(args.replicates_jsonl)
    ensure_file_exists(args.summary_jsonl)

    all_entries = load_all_commits(args.commits_jsonl)
    print(f"Loaded {len(all_entries)} total commits from {args.commits_jsonl}.")

    selected_entries = select_commit_window(
        all_entries,
        limit_commits=args.limit_commits,
    )
    print(
        f"Selected {len(selected_entries)} commits after boundary-cut windowing "
        f"between {START_DATE.isoformat()} and {END_DATE_EXCLUSIVE.isoformat()}."
    )

    if not selected_entries:
        ensure_parent_dir(args.output)
        with open(args.output, "w", encoding="utf-8"):
            pass
        print(f"No commits matched the selected window. Wrote empty file to {args.output}.")
        return

    target_revisions = {entry.node for entry in selected_entries}
    samples_by_revision: dict[str, list[dict[str, Any]]] = defaultdict(list)

    total_matched_samples = 0
    total_matched_samples += load_samples_by_revision(
        args.replicates_jsonl,
        replicate=True,
        target_revisions=target_revisions,
        samples_by_revision=samples_by_revision,
    )
    total_matched_samples += load_samples_by_revision(
        args.summary_jsonl,
        replicate=False,
        target_revisions=target_revisions,
        samples_by_revision=samples_by_revision,
    )

    commits_with_samples, total_samples_written = write_output_jsonl(
        args.output,
        selected_entries,
        samples_by_revision,
    )
    print(
        f"Wrote {len(selected_entries)} commit rows to {args.output}. "
        f"Commits with samples: {commits_with_samples}. "
        f"Matched samples written: {total_samples_written} "
        f"(matched during load: {total_matched_samples})."
    )


if __name__ == "__main__":
    main()
