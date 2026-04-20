#!/usr/bin/env python3
"""
Fetch Treeherder performance measurement data for failing perf signatures.

Input:
  - `datasets/mozilla_perf_bisect/alert_summary_fail_perf_sigs_no_fw_2_6_18.csv`

Outputs:
  - `datasets/mozilla_perf_bisect/per_sig_perf_data_summary.jsonl`
      Measurements fetched with `replicates=False`
  - `datasets/mozilla_perf_bisect/per_sig_perf_data_replicates.jsonl`
      Measurements fetched with `replicates=True`

Each JSONL line contains only:
  - `signature_id`
  - `filter_stats`
  - `perf_measurement_data`

The script preserves that minimal output shape while still supporting restartable
writes by inferring already-processed signatures from line order when explicit
signature metadata is absent.

Time window:
  - Inclusive start: 2025-06-01T00:00:00+00:00
  - Exclusive end: 2025-11-01T00:00:00+00:00

Notes:
  - Treeherder's `interval` is a relative lookback from "now", so we fetch a
    lookback large enough to cover the start date, then filter measurements by
    `submit_time` into the target UTC window. Measurements with missing, blank,
    or unparseable `submit_time` are retained.
  - Per-signature replicate counts are derived separately by
    `get_perf_data_info.py` from `per_sig_perf_data_replicates.jsonl`.
  - `--debug` restricts processing to 2 signatures for both output files.
  - Outputs are restartable: if a JSONL already exists, signatures already
    present in that file are skipped unless `--overwrite` is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import UTC, datetime
from typing import Any

from requests.exceptions import RequestException, Timeout


csv.field_size_limit(sys.maxsize)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATASET_DIR = os.path.join(REPO_ROOT, "datasets", "mozilla_perf_bisect")

DEFAULT_INPUT_CSV = os.path.join(
    DATASET_DIR,
    "alert_summary_fail_perf_sigs_no_fw_2_6_18.csv",
)
DEFAULT_SUMMARY_OUTPUT = os.path.join(
    DATASET_DIR,
    "per_sig_perf_data_summary.jsonl",
)
DEFAULT_REPLICATES_OUTPUT = os.path.join(
    DATASET_DIR,
    "per_sig_perf_data_replicates.jsonl",
)

REPOSITORY = "autoland"
START_DATE = datetime(2025, 6, 1, tzinfo=UTC)
END_DATE_EXCLUSIVE = datetime(2025, 11, 1, tzinfo=UTC)
DEBUG_SIGNATURE_LIMIT = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Treeherder performance measurement data for failing perf "
            "signatures and write JSONL outputs for summary and replicate views."
        )
    )
    parser.add_argument(
        "--input-csv",
        default=DEFAULT_INPUT_CSV,
        help="CSV containing alert summaries with fail_perf_sig_ids.",
    )
    parser.add_argument(
        "--summary-output",
        default=DEFAULT_SUMMARY_OUTPUT,
        help="JSONL output path for `replicates=False` results.",
    )
    parser.add_argument(
        "--replicates-output",
        default=DEFAULT_REPLICATES_OUTPUT,
        help="JSONL output path for `replicates=True` results.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, process only the first N signature IDs after sorting.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Debug mode: process only the first 2 sorted signature IDs for both "
            "output files."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output JSONL files instead of resuming.",
    )
    return parser.parse_args()


def parse_iso_datetime(value: str) -> datetime | None:
    value = (value or "").strip()
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def compute_interval_seconds() -> int:
    """
    Use a lookback interval large enough to reach START_DATE from the current UTC
    time, with a small buffer to match the pattern used elsewhere in this repo.
    """
    timeframe_days = max(1, (datetime.now(UTC) - START_DATE).days + 2)
    return timeframe_days * 24 * 60 * 60


def load_signature_ids(input_csv: str) -> list[int]:
    signature_ids: set[int] = set()

    with open(input_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):
            raw_sig_ids = (row.get("fail_perf_sig_ids") or "").strip()
            if not raw_sig_ids:
                continue

            try:
                parsed = json.loads(raw_sig_ids)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in fail_perf_sig_ids at {input_csv}:{row_num}: {e}"
                ) from e

            if not isinstance(parsed, list):
                raise ValueError(
                    f"Expected fail_perf_sig_ids to be a list at {input_csv}:{row_num}"
                )

            for value in parsed:
                try:
                    signature_ids.add(int(value))
                except Exception as e:
                    raise ValueError(
                        f"Non-integer signature ID in fail_perf_sig_ids at "
                        f"{input_csv}:{row_num}: {value!r}"
                    ) from e

    return sorted(signature_ids)


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


def load_processed_signature_ids(path: str, signature_ids: list[int]) -> set[int]:
    if not os.path.exists(path):
        return set()

    processed: set[int] = set()
    inferred_processed_count = 0
    for record in iter_jsonl(path):
        if not isinstance(record, dict):
            continue
        signature_id = record.get("signature_id")
        if signature_id is None:
            if inferred_processed_count < len(signature_ids):
                processed.add(signature_ids[inferred_processed_count])
                inferred_processed_count += 1
            continue
        try:
            processed.add(int(signature_id))
        except Exception:
            continue
    return processed


def fetch_perf_measurement_data(
    client: Any,
    signature_id: int,
    *,
    interval_seconds: int,
    replicates: bool,
) -> list[dict[str, Any]]:
    params = {
        "repository": REPOSITORY,
        "signature": signature_id,
        "interval": interval_seconds,
        "all_data": True,
        "replicates": replicates,
    }

    try:
        data_list = client._get_json("performance/summary", **params)
    except Timeout as e:
        print(
            f"[WARN] Timeout fetching signature {signature_id} "
            f"(replicates={replicates}): {e}"
        )
        return []
    except RequestException as e:
        print(
            f"[WARN] Request error fetching signature {signature_id} "
            f"(replicates={replicates}): {e}"
        )
        return []
    except Exception as e:
        print(
            f"[WARN] Unexpected error fetching signature {signature_id} "
            f"(replicates={replicates}): {e}"
        )
        return []

    if not data_list:
        return []

    first_item = data_list[0]
    if not isinstance(first_item, dict):
        return []

    data = first_item.get("data", [])
    if not isinstance(data, list):
        return []

    return [row for row in data if isinstance(row, dict)]


def filter_measurements_to_window(
    measurements: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    kept: list[dict[str, Any]] = []
    stats = {
        "raw_measurements": len(measurements),
        "kept_measurements": 0,
        "kept_missing_submit_time": 0,
        "kept_unparseable_submit_time": 0,
        "filtered_on_or_after_end": 0,
    }

    for row in measurements:
        submit_time = row.get("submit_time")
        submit_time_value = "" if submit_time is None else str(submit_time).strip()
        if not submit_time_value:
            kept.append(row)
            stats["kept_missing_submit_time"] += 1
            continue

        submit_dt = parse_iso_datetime(submit_time_value)
        if submit_dt is None:
            kept.append(row)
            stats["kept_unparseable_submit_time"] += 1
            continue

        if submit_dt < START_DATE:
            continue

        if submit_dt >= END_DATE_EXCLUSIVE:
            stats["filtered_on_or_after_end"] += 1
            continue

        kept.append(row)

    stats["kept_measurements"] = len(kept)
    return kept, stats


def append_jsonl(path: str, record: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def truncate_if_requested(path: str, overwrite: bool) -> None:
    if overwrite and os.path.exists(path):
        os.remove(path)


def process_signatures(
    client: Any,
    signature_ids: list[int],
    *,
    output_path: str,
    replicates: bool,
    overwrite: bool,
    interval_seconds: int,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    truncate_if_requested(output_path, overwrite)

    processed = load_processed_signature_ids(output_path, signature_ids)
    total = len(signature_ids)

    print(
        f"Processing {total} signatures for {output_path} "
        f"(replicates={replicates}, already_have={len(processed)})."
    )

    for idx, signature_id in enumerate(signature_ids, start=1):
        if signature_id in processed:
            continue

        print(f"[{idx}/{total}] Fetching signature {signature_id} (replicates={replicates})...")
        raw_measurements = fetch_perf_measurement_data(
            client,
            signature_id,
            interval_seconds=interval_seconds,
            replicates=replicates,
        )
        filtered_measurements, filter_stats = filter_measurements_to_window(
            raw_measurements
        )

        record = {
            "signature_id": signature_id,
            "filter_stats": filter_stats,
            "perf_measurement_data": filtered_measurements,
        }
        append_jsonl(output_path, record)
        processed.add(signature_id)

    print(f"Done with {output_path}. Wrote/kept {len(processed)} signatures.")


def main() -> int:
    args = parse_args()

    try:
        from thclient import TreeherderClient  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'thclient'. Install the Treeherder client "
            "dependency in your Python environment and re-run."
        ) from e

    signature_ids = load_signature_ids(args.input_csv)
    if not signature_ids:
        print(f"No signature IDs found in {args.input_csv}.")
        return 0

    if args.debug:
        signature_ids = signature_ids[:DEBUG_SIGNATURE_LIMIT]
        print(
            f"Debug mode enabled: restricting processing to "
            f"{len(signature_ids)} signatures."
        )
    elif args.limit and args.limit > 0:
        signature_ids = signature_ids[: args.limit]

    interval_seconds = compute_interval_seconds()
    print(
        f"Loaded {len(signature_ids)} unique signatures from {args.input_csv}. "
        f"Using interval={interval_seconds} seconds."
    )

    client = TreeherderClient()

    process_signatures(
        client,
        signature_ids,
        output_path=args.summary_output,
        replicates=False,
        overwrite=args.overwrite,
        interval_seconds=interval_seconds,
    )
    process_signatures(
        client,
        signature_ids,
        output_path=args.replicates_output,
        replicates=True,
        overwrite=args.overwrite,
        interval_seconds=interval_seconds,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
