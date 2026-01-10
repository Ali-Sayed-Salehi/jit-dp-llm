#!/usr/bin/env python3
"""
Create per-job signature lists from Treeherder caches and produce per-job
metadata summaries.

Inputs (JSONL):
  - `datasets/mozilla_perf/perf_jobs_by_signature.jsonl` (optional)
      Each line: {"signature_id": <int>, "jobs": [<job dict>, ...]}
      Used to create `sigs_by_job_id.jsonl` if it doesn't already exist.
  - `datasets/mozilla_perf/sigs_by_job_id.jsonl`
      Each line: {"job_id": <int>, "signature_ids": [<int>, ...]}
  - `datasets/mozilla_perf/all_signatures.jsonl`
      Each line: a signature metadata object with at least an `id` field.

Outputs (JSONL):
  - `datasets/mozilla_perf/sigs_by_job_id.jsonl`
      Each line: {"job_id": <int>, "signature_ids": [<int>, ...]}
  - `datasets/mozilla_perf/sigs_by_job_id_detailed.jsonl`
      Each line:
        {"job_id": <int>, "signature_ids": [...], "common_sig_features": [...]}
      `common_sig_features` is the list of signature metadata field names whose
      value is identical across *all* signatures in that job. If even one
      signature is missing the field or has a different value, the field is
      excluded.

Conclusion:
  Prints which signature fields are always shared within every job, i.e. the
  intersection of per-job `common_sig_features`.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATASET_DIR = os.path.join(REPO_ROOT, "datasets", "mozilla_perf")

DEFAULT_SIGNATURE_JOBS_JSONL = os.path.join(DATASET_DIR, "perf_jobs_by_signature.jsonl")
DEFAULT_SIGS_BY_JOB_JSONL = os.path.join(DATASET_DIR, "sigs_by_job_id.jsonl")
DEFAULT_ALL_SIGNATURES_JSONL = os.path.join(DATASET_DIR, "all_signatures.jsonl")
DEFAULT_DETAILED_OUTPUT_JSONL = os.path.join(DATASET_DIR, "sigs_by_job_id_detailed.jsonl")


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


def load_sigs_by_job_id(path: str) -> dict[int, list[int]]:
    job_to_sigs: dict[int, list[int]] = {}
    for record in iter_jsonl(path):
        if not isinstance(record, dict):
            continue
        job_id = record.get("job_id")
        sig_ids = record.get("signature_ids")
        if job_id is None or not isinstance(sig_ids, list):
            continue
        try:
            job_id_int = int(job_id)
        except Exception:
            continue

        parsed_sigs: list[int] = []
        for sig_id in sig_ids:
            try:
                parsed_sigs.append(int(sig_id))
            except Exception:
                continue

        job_to_sigs[job_id_int] = sorted(set(parsed_sigs))

    return job_to_sigs


def load_all_signatures(path: str) -> dict[int, dict[str, Any]]:
    sigs: dict[int, dict[str, Any]] = {}
    for record in iter_jsonl(path):
        if not isinstance(record, dict):
            continue
        sig_id = record.get("id")
        if sig_id is None:
            continue
        try:
            sig_id_int = int(sig_id)
        except Exception:
            continue
        sigs[sig_id_int] = record
    return sigs


def common_fields_across_signatures(
    signature_records: list[dict[str, Any]],
    *,
    exclude_fields: set[str],
) -> list[str]:
    if not signature_records:
        return []

    common_keys = set(signature_records[0].keys())
    for rec in signature_records[1:]:
        common_keys &= set(rec.keys())

    common_keys -= exclude_fields

    common: list[str] = []
    for key in sorted(common_keys):
        first_val = signature_records[0].get(key)
        if all(rec.get(key) == first_val for rec in signature_records[1:]):
            common.append(key)
    return common


def write_sigs_by_job_id_detailed(
    *,
    sigs_by_job_id_jsonl: str,
    all_signatures_jsonl: str,
    out_jsonl: str,
    exclude_fields: set[str],
) -> list[set[str]]:
    job_to_sigs = load_sigs_by_job_id(sigs_by_job_id_jsonl)
    sig_details = load_all_signatures(all_signatures_jsonl)

    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    per_job_common_keys: list[set[str]] = []

    missing_sig_ids = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for job_id in sorted(job_to_sigs):
            sig_ids = job_to_sigs[job_id]
            signature_records: list[dict[str, Any]] = []
            for sig_id in sig_ids:
                rec = sig_details.get(sig_id)
                if rec is None:
                    missing_sig_ids += 1
                    signature_records.append({})
                else:
                    signature_records.append(rec)

            common_sig_features = common_fields_across_signatures(
                signature_records, exclude_fields=exclude_fields
            )
            per_job_common_keys.append(set(common_sig_features))

            out_record = {
                "job_id": job_id,
                "signature_ids": sig_ids,
                "common_sig_features": common_sig_features,
            }
            f.write(json.dumps(out_record) + "\n")

    if missing_sig_ids:
        print(
            f"[WARN] Missing metadata for {missing_sig_ids} signature IDs while building {out_jsonl}."
        )

    return per_job_common_keys


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create per-job signature lists and a detailed per-job JSONL describing "
            "which signature metadata fields are shared within each job."
        )
    )
    parser.add_argument(
        "--signature-jobs-jsonl",
        default=DEFAULT_SIGNATURE_JOBS_JSONL,
        help="Path to perf_jobs_by_signature.jsonl (used only if sigs_by_job_id.jsonl is missing).",
    )
    parser.add_argument(
        "--sigs-by-job-jsonl",
        default=DEFAULT_SIGS_BY_JOB_JSONL,
        help="Path to sigs_by_job_id.jsonl (input for detailed output; created if missing).",
    )
    parser.add_argument(
        "--all-signatures-jsonl",
        default=DEFAULT_ALL_SIGNATURES_JSONL,
        help="Path to all_signatures.jsonl produced by get_perf_sigs.py.",
    )
    parser.add_argument(
        "--detailed-output",
        default=DEFAULT_DETAILED_OUTPUT_JSONL,
        help="Path to sigs_by_job_id_detailed.jsonl output.",
    )
    parser.add_argument(
        "--exclude-field",
        action="append",
        default=["id"],
        help="Signature metadata field to exclude from common_sig_features (repeatable). Default: id",
    )
    args = parser.parse_args()

    exclude_fields = set(args.exclude_field or [])

    if not os.path.exists(args.sigs_by_job_jsonl):
        if not os.path.exists(args.signature_jobs_jsonl):
            raise FileNotFoundError(
                f"Neither {args.sigs_by_job_jsonl} nor {args.signature_jobs_jsonl} exists. "
                "Create sigs_by_job_id.jsonl first, or run get_num_perf_tests.py to generate "
                "perf_jobs_by_signature.jsonl."
            )
        job_to_sigs = build_sigs_by_job_id(args.signature_jobs_jsonl)
        write_sigs_by_job_id(job_to_sigs, args.sigs_by_job_jsonl)
        print(f"Wrote {len(job_to_sigs)} jobs to {args.sigs_by_job_jsonl}")

    if not os.path.exists(args.all_signatures_jsonl):
        raise FileNotFoundError(
            f"Signature metadata JSONL not found at {args.all_signatures_jsonl}. "
            "Run get_perf_sigs.py to generate all_signatures.jsonl."
        )

    per_job_common_keys = write_sigs_by_job_id_detailed(
        sigs_by_job_id_jsonl=args.sigs_by_job_jsonl,
        all_signatures_jsonl=args.all_signatures_jsonl,
        out_jsonl=args.detailed_output,
        exclude_fields=exclude_fields,
    )

    if per_job_common_keys:
        always_common = set.intersection(*per_job_common_keys)
    else:
        always_common = set()

    if always_common:
        print(
            "Conclusion: fields always shared across all signatures within every job: "
            + ", ".join(sorted(always_common))
        )
    else:
        print(
            "Conclusion: no signature metadata field is shared across all signatures within every job."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
