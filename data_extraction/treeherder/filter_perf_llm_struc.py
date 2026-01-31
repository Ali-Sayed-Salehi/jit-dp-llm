#!/usr/bin/env python3
"""
Relabel `perf_llm_struc.jsonl` by excluding failing signatures in specific Treeherder frameworks.

Problem this solves
  `datasets/mozilla_perf/perf_llm_struc.jsonl` is an LLM-ready dataset with records:
    {"commit_id": "...", "prompt": "...", "response": "0"|"1"}

  The file does NOT contain signature IDs. If you want to ignore some performance signatures
  (e.g., certain Treeherder frameworks) when determining whether a commit is a perf regressor,
  you must relabel affected rows.

Definition used here
  For a regressor commit (`response == "1"`), look up the revision's failing perf signature IDs
  from `alert_summary_fail_perf_sigs.csv`. Drop any failing signatures whose `framework_id` is in
  the excluded set. If the revision had failing signatures originally but has ZERO failing
  signatures after filtering, flip its label to clean (`response = "0"`).

Flow
  1. Load excluded signature IDs by reading `all_signatures.jsonl` and selecting signatures whose
     `framework_id` is in `--exclude-framework-ids`.
  2. Load failing signature IDs per revision from `alert_summary_fail_perf_sigs.csv` and union
     duplicates by `revision`.
  3. Stream `perf_llm_struc.jsonl` and rewrite it to a new JSONL:
     - Leave non-regressor rows unchanged.
     - For regressor rows, flip `response` to "0" iff all failing signatures are excluded.
  4. Print sanity-check statistics and optionally write a JSON report.

Inputs (defaults)
  - `datasets/mozilla_perf/perf_llm_struc.jsonl`
  - `datasets/mozilla_perf/alert_summary_fail_perf_sigs.csv`
  - `datasets/mozilla_perf/all_signatures.jsonl`

Outputs (defaults)
  - `datasets/mozilla_perf/perf_llm_struc_no_fw_2_6_18.jsonl`
  - (optional) `--report-json` with summary counters

Usage
  python data_extraction/treeherder/filter_perf_llm_struc.py
  python data_extraction/treeherder/filter_perf_llm_struc.py --exclude-framework-ids 2 6 18 --report-json report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Iterable


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATASET_DIR = os.path.join(REPO_ROOT, "datasets", "mozilla_perf")

DEFAULT_IN_JSONL = os.path.join(DATASET_DIR, "perf_llm_struc.jsonl")
DEFAULT_FAIL_SIGS_CSV = os.path.join(DATASET_DIR, "alert_summary_fail_perf_sigs.csv")
DEFAULT_ALL_SIGNATURES_JSONL = os.path.join(DATASET_DIR, "all_signatures.jsonl")
DEFAULT_OUT_JSONL = os.path.join(DATASET_DIR, "perf_llm_struc_no_fw_2_6_18.jsonl")


def iter_jsonl(path: str) -> Iterable[tuple[int, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_num, json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}: {e}") from e


def load_excluded_signature_ids(
    all_signatures_jsonl: str,
    excluded_framework_ids: set[int],
) -> tuple[set[int], set[int], Counter[str], Counter[int]]:
    """
    Returns:
      - excluded_sig_ids: signature IDs whose framework_id is excluded
      - known_sig_ids: all signature IDs seen in all_signatures.jsonl
      - stats: parsing counters
      - excluded_framework_counts: how many excluded signatures per framework_id
    """
    excluded_sig_ids: set[int] = set()
    known_sig_ids: set[int] = set()
    stats: Counter[str] = Counter()
    excluded_framework_counts: Counter[int] = Counter()

    for line_num, record in iter_jsonl(all_signatures_jsonl):
        stats["lines_total"] += 1
        if not isinstance(record, dict):
            stats["non_dict_records"] += 1
            continue

        sig_id = record.get("id")
        if sig_id is None:
            stats["missing_id"] += 1
            continue
        try:
            sig_id_int = int(sig_id)
        except Exception:
            stats["non_int_id"] += 1
            continue

        known_sig_ids.add(sig_id_int)

        fw = record.get("framework_id")
        if fw is None:
            stats["missing_framework_id"] += 1
            continue
        try:
            fw_int = int(fw)
        except Exception:
            stats["non_int_framework_id"] += 1
            continue

        if fw_int in excluded_framework_ids:
            excluded_sig_ids.add(sig_id_int)
            excluded_framework_counts[fw_int] += 1

    stats["unique_signature_ids"] = len(known_sig_ids)
    stats["excluded_signature_ids"] = len(excluded_sig_ids)
    return excluded_sig_ids, known_sig_ids, stats, excluded_framework_counts


def load_failing_sigs_by_revision(
    fail_sigs_csv: str,
) -> tuple[dict[str, set[int]], set[int], Counter[str]]:
    """
    Load failing perf signature IDs per revision from alert_summary_fail_perf_sigs.csv.

    The CSV is expected to include:
      - revision (str)
      - fail_perf_sig_ids (JSON string of list[int])

    Returns:
      - revision_to_fail_sig_ids: revision -> set(signature_id)
      - all_fail_sig_ids: union of all failing signature IDs across all revisions
      - stats: parsing counters
    """
    revision_to_fail_sig_ids: dict[str, set[int]] = defaultdict(set)
    all_fail_sig_ids: set[int] = set()
    stats: Counter[str] = Counter()

    with open(fail_sigs_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["rows_total"] += 1
            revision = (row.get("revision") or "").strip()
            if not revision:
                stats["rows_missing_revision"] += 1
                continue

            raw = row.get("fail_perf_sig_ids")
            if raw is None:
                stats["rows_missing_fail_perf_sig_ids"] += 1
                continue

            raw = raw.strip()
            try:
                parsed = json.loads(raw) if raw else []
            except json.JSONDecodeError:
                stats["rows_fail_sig_json_parse_error"] += 1
                continue

            if not isinstance(parsed, list):
                stats["rows_fail_sig_not_list"] += 1
                continue

            if not parsed:
                stats["rows_empty_fail_sig_list"] += 1

            for value in parsed:
                stats["fail_sig_values_total"] += 1
                try:
                    sig_id = int(value)
                except Exception:
                    stats["fail_sig_values_non_int"] += 1
                    continue
                revision_to_fail_sig_ids[revision].add(sig_id)
                all_fail_sig_ids.add(sig_id)

    stats["unique_revisions"] = len(revision_to_fail_sig_ids)
    stats["unique_failing_signature_ids"] = len(all_fail_sig_ids)
    return revision_to_fail_sig_ids, all_fail_sig_ids, stats


def rewrite_perf_llm_dataset(
    *,
    in_jsonl: str,
    out_jsonl: str,
    revision_to_fail_sig_ids: dict[str, set[int]],
    excluded_sig_ids: set[int],
    limit: int = 0,
) -> tuple[Counter[str], list[str]]:
    """
    Rewrite perf_llm_struc.jsonl with relabeled `response` values.

    Returns:
      - stats counters
      - flipped_commit_ids_preview (up to 20 ids)
    """
    stats: Counter[str] = Counter()
    flipped_preview: list[str] = []

    out_dir = os.path.dirname(os.path.abspath(out_jsonl))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(in_jsonl, "r", encoding="utf-8") as in_f, open(
        out_jsonl, "w", encoding="utf-8"
    ) as out_f:
        for line_num, line in enumerate(in_f, start=1):
            raw = line.strip()
            if not raw:
                continue

            stats["input_lines_total"] += 1
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} of {in_jsonl}: {e}") from e

            if not isinstance(record, dict):
                stats["non_dict_records"] += 1
                continue

            commit_id = (record.get("commit_id") or "").strip()
            response = record.get("response")
            response_str = "" if response is None else str(response).strip()

            if response_str == "1":
                stats["regressor_records_total"] += 1
                if not commit_id:
                    stats["regressor_missing_commit_id"] += 1
                fail_sigs = revision_to_fail_sig_ids.get(commit_id)
                if fail_sigs is None:
                    stats["regressor_missing_fail_sig_mapping"] += 1
                else:
                    stats["regressor_with_fail_sig_mapping"] += 1
                    orig_cnt = len(fail_sigs)
                    filtered_cnt = len(fail_sigs - excluded_sig_ids)
                    stats["regressor_fail_sig_count_sum"] += orig_cnt
                    stats["regressor_fail_sig_removed_sum"] += orig_cnt - filtered_cnt

                    if orig_cnt > 0 and filtered_cnt == 0:
                        record["response"] = "0"
                        stats["regressor_flipped_to_clean"] += 1
                        if commit_id and len(flipped_preview) < 20:
                            flipped_preview.append(commit_id)
                    else:
                        stats["regressor_kept_as_regressor"] += 1
            else:
                stats["non_regressor_records_total"] += 1

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["output_records_written"] += 1

            if limit and stats["output_records_written"] >= limit:
                stats["stopped_due_to_limit"] += 1
                break

    return stats, flipped_preview


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Relabel perf_llm_struc.jsonl by excluding failing signatures in specific framework_ids."
        )
    )
    parser.add_argument("--in", dest="in_jsonl", default=DEFAULT_IN_JSONL, help="Input JSONL.")
    parser.add_argument(
        "--out",
        dest="out_jsonl",
        default=DEFAULT_OUT_JSONL,
        help="Output JSONL (rewritten labels).",
    )
    parser.add_argument(
        "--fail-sigs-csv",
        default=DEFAULT_FAIL_SIGS_CSV,
        help="alert_summary_fail_perf_sigs.csv path.",
    )
    parser.add_argument(
        "--all-signatures",
        default=DEFAULT_ALL_SIGNATURES_JSONL,
        help="all_signatures.jsonl path (for framework_id lookup).",
    )
    parser.add_argument(
        "--exclude-framework-ids",
        nargs="+",
        type=int,
        default=[2, 6, 18],
        help="Framework IDs to exclude (default: 2 6 18).",
    )
    parser.add_argument(
        "--report-json",
        default="",
        help="Optional path to write a JSON summary report.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If set, only rewrite the first N records (for quick experimentation).",
    )
    args = parser.parse_args(argv)

    in_jsonl = os.path.abspath(args.in_jsonl)
    out_jsonl = os.path.abspath(args.out_jsonl)
    fail_sigs_csv = os.path.abspath(args.fail_sigs_csv)
    all_signatures_jsonl = os.path.abspath(args.all_signatures)
    excluded_framework_ids = set(args.exclude_framework_ids or [])

    if not os.path.exists(in_jsonl):
        print(f"Input not found: {in_jsonl}", file=sys.stderr)
        return 2
    if not os.path.exists(fail_sigs_csv):
        print(
            f"Failing-signatures CSV not found: {fail_sigs_csv}\n"
            "Run `python data_extraction/treeherder/get_failing_perf_sigs.py` first.",
            file=sys.stderr,
        )
        return 2
    if not os.path.exists(all_signatures_jsonl):
        print(
            f"Signature metadata JSONL not found: {all_signatures_jsonl}\n"
            "Run `python data_extraction/treeherder/get_perf_sigs.py` first.",
            file=sys.stderr,
        )
        return 2

    excluded_sig_ids, known_sig_ids, sig_stats, excluded_fw_counts = load_excluded_signature_ids(
        all_signatures_jsonl,
        excluded_framework_ids,
    )
    revision_to_fail_sig_ids, all_fail_sig_ids, fail_stats = load_failing_sigs_by_revision(
        fail_sigs_csv
    )

    missing_fail_sig_meta = sorted(all_fail_sig_ids - known_sig_ids)
    excluded_fail_sigs = all_fail_sig_ids & excluded_sig_ids

    rewrite_stats, flipped_preview = rewrite_perf_llm_dataset(
        in_jsonl=in_jsonl,
        out_jsonl=out_jsonl,
        revision_to_fail_sig_ids=revision_to_fail_sig_ids,
        excluded_sig_ids=excluded_sig_ids,
        limit=max(0, int(args.limit or 0)),
    )

    # -------------------------
    # Sanity-check reporting
    # -------------------------
    print("=== Framework exclusion ===")
    print("Excluded framework_ids:", sorted(excluded_framework_ids))
    print("all_signatures.jsonl:", all_signatures_jsonl)
    print("Known signature IDs:", sig_stats.get("unique_signature_ids", 0))
    print("Excluded signature IDs:", sig_stats.get("excluded_signature_ids", 0))
    if excluded_fw_counts:
        print(
            "Excluded signatures by framework_id:",
            {k: excluded_fw_counts[k] for k in sorted(excluded_fw_counts)},
        )

    print("\n=== Failing signatures CSV ===")
    print("alert_summary_fail_perf_sigs.csv:", fail_sigs_csv)
    print("Rows total:", fail_stats.get("rows_total", 0))
    print("Unique revisions:", fail_stats.get("unique_revisions", 0))
    print("Unique failing signature IDs:", fail_stats.get("unique_failing_signature_ids", 0))
    print("Fail-sig values total:", fail_stats.get("fail_sig_values_total", 0))
    print("Fail-sig values non-int:", fail_stats.get("fail_sig_values_non_int", 0))
    print("Failing signature IDs excluded:", len(excluded_fail_sigs))
    print("Failing signature IDs missing metadata:", len(missing_fail_sig_meta))
    if missing_fail_sig_meta:
        preview = ", ".join(str(x) for x in missing_fail_sig_meta[:20])
        more = "" if len(missing_fail_sig_meta) <= 20 else f" (+{len(missing_fail_sig_meta) - 20} more)"
        print("Missing-metadata signature_id preview:", f"{preview}{more}")

    print("\n=== perf_llm_struc relabel ===")
    print("Input:", in_jsonl)
    print("Output:", out_jsonl)
    print("Records written:", rewrite_stats.get("output_records_written", 0))
    if rewrite_stats.get("stopped_due_to_limit"):
        print("Stopped early due to --limit.")
    print("Regressor records (response==1):", rewrite_stats.get("regressor_records_total", 0))
    print(
        "Regressor records with fail-sig mapping:",
        rewrite_stats.get("regressor_with_fail_sig_mapping", 0),
    )
    print(
        "Regressor records missing fail-sig mapping:",
        rewrite_stats.get("regressor_missing_fail_sig_mapping", 0),
    )
    print("Regressor records flipped to clean:", rewrite_stats.get("regressor_flipped_to_clean", 0))
    print("Regressor records kept:", rewrite_stats.get("regressor_kept_as_regressor", 0))
    if flipped_preview:
        print("Flipped commit_id examples:", ", ".join(flipped_preview))

    # Optional JSON report
    if args.report_json:
        report_path = os.path.abspath(args.report_json)
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        report = {
            "inputs": {
                "perf_llm_struc_jsonl": in_jsonl,
                "alert_summary_fail_perf_sigs_csv": fail_sigs_csv,
                "all_signatures_jsonl": all_signatures_jsonl,
                "excluded_framework_ids": sorted(excluded_framework_ids),
            },
            "excluded_signatures": {
                "known_signature_ids": sig_stats.get("unique_signature_ids", 0),
                "excluded_signature_ids": sig_stats.get("excluded_signature_ids", 0),
                "excluded_signatures_by_framework_id": {
                    str(k): excluded_fw_counts[k] for k in sorted(excluded_fw_counts)
                },
            },
            "failing_signatures": {
                "csv_rows_total": fail_stats.get("rows_total", 0),
                "unique_revisions": fail_stats.get("unique_revisions", 0),
                "unique_failing_signature_ids": fail_stats.get("unique_failing_signature_ids", 0),
                "unique_failing_signature_ids_excluded": len(excluded_fail_sigs),
                "unique_failing_signature_ids_missing_metadata": len(missing_fail_sig_meta),
            },
            "relabel": dict(rewrite_stats),
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
        print("\nWrote report JSON:", report_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

