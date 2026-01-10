#!/usr/bin/env python3
"""
Invert the per-signature Treeherder cache into a per-job mapping.

Input (JSONL):
  `datasets/mozilla_perf/perf_jobs_by_signature.jsonl`
  Each line:
    {"signature_id": <int>, "jobs": [<job dict>, ...]}
  Where each job dict typically contains a `job_id`.

Output (JSONL):
  `datasets/mozilla_perf/sigs_by_job_id.jsonl`
  Each line:
    {"job_id": <int>, "signature_ids": [<int>, ...]}

Usage:
  python data_extraction/treeherder/get_sigs_per_job.py
  python data_extraction/treeherder/get_sigs_per_job.py --input ... --output ...
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATASET_DIR = os.path.join(REPO_ROOT, "datasets", "mozilla_perf")

DEFAULT_INPUT = os.path.join(DATASET_DIR, "perf_jobs_by_signature.jsonl")
DEFAULT_OUTPUT = os.path.join(DATASET_DIR, "sigs_by_job_id.jsonl")


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def build_sigs_by_job_id(signature_jobs_jsonl: str) -> dict[int, set[int]]:
    job_to_sigs: dict[int, set[int]] = defaultdict(set)

    for record in iter_jsonl(signature_jobs_jsonl):
        signature_id = record.get("signature_id")
        if signature_id is None:
            continue
        try:
            signature_id_int = int(signature_id)
        except Exception:
            continue

        jobs = record.get("jobs") or []
        if not isinstance(jobs, list):
            continue

        for job in jobs:
            if not isinstance(job, dict):
                continue
            job_id = job.get("job_id", job.get("id"))
            if job_id is None:
                continue
            try:
                job_id_int = int(job_id)
            except Exception:
                continue
            job_to_sigs[job_id_int].add(signature_id_int)

    return job_to_sigs


def write_sigs_by_job_id(job_to_sigs: dict[int, set[int]], out_jsonl: str) -> None:
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for job_id in sorted(job_to_sigs):
            record = {
                "job_id": job_id,
                "signature_ids": sorted(job_to_sigs[job_id]),
            }
            f.write(json.dumps(record) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a per-job signature list from perf_jobs_by_signature.jsonl."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to input JSONL.")
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT, help="Path to output JSONL."
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(
            f"Input JSONL not found at {args.input}. "
            "Run get_num_perf_tests.py first to generate perf_jobs_by_signature.jsonl."
        )

    job_to_sigs = build_sigs_by_job_id(args.input)
    write_sigs_by_job_id(job_to_sigs, args.output)

    print(f"Wrote {len(job_to_sigs)} jobs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

