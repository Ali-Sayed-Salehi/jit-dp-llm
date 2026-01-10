#!/usr/bin/env python3
"""
Create signature groups from `sigs_by_job_id_detailed.jsonl`.

A "signature group" is the set of signatures that always co-occur: two
signatures are in the same group iff they appear in exactly the same set of
jobs.

Input (JSONL):
  - `datasets/mozilla_perf/sigs_by_job_id_detailed.jsonl`
      Each line: {"job_id": <int>, "signature_ids": [<int>, ...], ...}

Output (JSONL):
  - `datasets/mozilla_perf/sig_groups.jsonl`
      Each line: {"sig_group_id": <int>, "signatures": [<int>, ...]}
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Iterable


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATASET_DIR = os.path.join(REPO_ROOT, "datasets", "mozilla_perf")

DEFAULT_INPUT_JSONL = os.path.join(DATASET_DIR, "sigs_by_job_id_detailed.jsonl")
DEFAULT_OUTPUT_JSONL = os.path.join(DATASET_DIR, "sig_groups.jsonl")


def iter_jsonl(path: str) -> Iterable[Any]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_jobs(
    input_jsonl: str,
) -> tuple[dict[int, set[int]], set[int], Counter[str], Counter[str]]:
    """
    Returns:
      - job_to_sigs: job_id -> set(sig_id)
      - all_sig_ids: all parsed signature IDs across all jobs
      - dropped_sig_values: counter of non-int signature_id representations
      - dropped_job_records: counter of job-level parse issues
    """
    job_to_sigs: dict[int, set[int]] = {}
    all_sig_ids: set[int] = set()
    dropped_sig_values: Counter[str] = Counter()
    dropped_job_records: Counter[str] = Counter()

    for record in iter_jsonl(input_jsonl):
        if not isinstance(record, dict):
            dropped_job_records["non_dict_record"] += 1
            continue

        job_id = record.get("job_id")
        if job_id is None:
            dropped_job_records["missing_job_id"] += 1
            continue
        try:
            job_id_int = int(job_id)
        except Exception:
            dropped_job_records["non_int_job_id"] += 1
            continue

        sig_ids = record.get("signature_ids")
        if not isinstance(sig_ids, list):
            dropped_job_records["signature_ids_not_list"] += 1
            continue

        parsed: set[int] = set()
        for raw in sig_ids:
            try:
                sig_id_int = int(raw)
            except Exception:
                dropped_sig_values[repr(raw)] += 1
                continue
            parsed.add(sig_id_int)
            all_sig_ids.add(sig_id_int)

        job_to_sigs[job_id_int] = parsed

    return job_to_sigs, all_sig_ids, dropped_sig_values, dropped_job_records


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Group signatures that always appear together in the same jobs, "
            "writing `sig_groups.jsonl`."
        )
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_JSONL,
        help="Path to sigs_by_job_id_detailed.jsonl.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_JSONL,
        help="Path to sig_groups.jsonl output.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(
            f"Input JSONL not found at {args.input}. "
            "Run data_extraction/treeherder/get_sigs_per_job.py first."
        )

    job_to_sigs, all_sig_ids, dropped_sig_values, dropped_job_records = load_jobs(
        args.input
    )

    sig_to_jobs: dict[int, set[int]] = defaultdict(set)
    for job_id, sig_ids in job_to_sigs.items():
        for sig_id in sig_ids:
            sig_to_jobs[sig_id].add(job_id)

    jobs_to_sigs: dict[frozenset[int], list[int]] = defaultdict(list)
    for sig_id, jobs in sig_to_jobs.items():
        jobs_to_sigs[frozenset(jobs)].append(sig_id)

    groups: list[list[int]] = []
    for sigs in jobs_to_sigs.values():
        sigs_sorted = sorted(sigs)
        groups.append(sigs_sorted)

    groups.sort(key=lambda sigs: (-len(sigs), sigs[0]))

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    written_sig_ids: set[int] = set()
    with open(args.output, "w", encoding="utf-8") as f:
        for group_id, sigs in enumerate(groups):
            written_sig_ids.update(sigs)
            f.write(
                json.dumps(
                    {"Sig_group_id": group_id, "signatures": sigs}, sort_keys=True
                )
                + "\n"
            )

    missing_sig_ids = sorted(all_sig_ids - written_sig_ids)
    if missing_sig_ids:
        print(
            f"[WARN] {len(missing_sig_ids)} signatures from {args.input} are missing in {args.output}."
        )
        print(
            "Reason: signatures were dropped during parsing (e.g., non-integer IDs) or "
            "their containing job records were skipped."
        )
        print("Missing signature IDs (parsed ints):", missing_sig_ids[:50])
        if len(missing_sig_ids) > 50:
            print(f"... and {len(missing_sig_ids) - 50} more")
    else:
        print(
            f"All {len(all_sig_ids)} parsed signature IDs from {args.input} are present in {args.output}."
        )

    if dropped_job_records:
        print("[INFO] Dropped job records:", dict(dropped_job_records))
    if dropped_sig_values:
        most_common = dropped_sig_values.most_common(10)
        print("[INFO] Dropped non-integer signature_ids (top 10):", most_common)

    print(f"Wrote {len(groups)} signature groups to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
