#!/usr/bin/env python3
"""
Extract per-signature perf metadata from performance measurement data.

Input:
  - `datasets/mozilla_perf_bisect/per_sig_perf_data_replicates.jsonl`
      Output of `get_perf_test_data_per_sig.py` fetched with `replicates=True`
  - `datasets/mozilla_perf_bisect/sig_groups.jsonl`
      Maps signature IDs to `Sig_group_id`
  - `datasets/mozilla_perf_bisect/sig_group_job_durations.csv`
      Maps `signature_group_id` to `duration_minutes`

Output:
  - `datasets/mozilla_perf_bisect/per_sig_perf_data_info.jsonl`

Each JSONL line contains only:
  - `signature_id`
  - `replicate_counts`
  - `job_duration`
  - `lower_is_better`
  - `alert_threshold`
  - `platform`

`replicate_counts` is inferred by sorting the input record's
`perf_measurement_data` chronologically, taking the newest row's `id`, and
counting how many measurements in that same signature share that `id`.
Validation then walks backward to the next older row with a different `id` and
compares its replicate count against the newest sample's count. Mismatches emit
warnings and keep the larger of the two replicate counts.

Notes:
  - `lower_is_better`, `alert_threshold`, and `platform` are fetched per
    signature from Treeherder's `performance/summary` endpoint using a fixed
    one-month lookback interval.
  - `--debug` restricts processing to 2 signatures.
  - Outputs are restartable: if the output JSONL already exists, signatures
    already present in that file with the full output schema are skipped unless
    `--overwrite` is used.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from typing import Any

from requests.exceptions import RequestException, Timeout


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATASET_DIR = os.path.join(REPO_ROOT, "datasets", "mozilla_perf_bisect")

DEFAULT_INPUT_JSONL = os.path.join(
    DATASET_DIR,
    "per_sig_perf_data_replicates.jsonl",
)
DEFAULT_OUTPUT_JSONL = os.path.join(
    DATASET_DIR,
    "per_sig_perf_data_info.jsonl",
)
DEFAULT_SIG_GROUPS_JSONL = os.path.join(
    DATASET_DIR,
    "sig_groups.jsonl",
)
DEFAULT_SIG_GROUP_DURATIONS_CSV = os.path.join(
    DATASET_DIR,
    "sig_group_job_durations.csv",
)

DEBUG_SIGNATURE_LIMIT = 2
REPOSITORY = "autoland"
SUMMARY_METADATA_FIELDS = (
    "lower_is_better",
    "alert_threshold",
    "platform",
)
REQUIRED_OUTPUT_FIELDS = (
    "replicate_counts",
    "job_duration",
    *SUMMARY_METADATA_FIELDS,
)
SUMMARY_INTERVAL_SECONDS = 30 * 24 * 60 * 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-signature replicate counts plus Treeherder summary "
            "metadata from replicate measurement JSONL produced by "
            "get_perf_test_data_per_sig.py."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        default=DEFAULT_INPUT_JSONL,
        help="Input JSONL path containing per-signature replicate measurements.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_JSONL,
        help="Output JSONL path for per-signature perf metadata.",
    )
    parser.add_argument(
        "--sig-groups-jsonl",
        default=DEFAULT_SIG_GROUPS_JSONL,
        help="JSONL path mapping signatures to Sig_group_id.",
    )
    parser.add_argument(
        "--sig-group-durations-csv",
        default=DEFAULT_SIG_GROUP_DURATIONS_CSV,
        help="CSV path mapping signature_group_id to duration_minutes.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, process only the first N signature records from the input.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: process only the first 2 signature records.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the existing output JSONL instead of resuming.",
    )
    return parser.parse_args()


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


def append_jsonl(path: str, record: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def truncate_if_requested(path: str, overwrite: bool) -> None:
    if overwrite and os.path.exists(path):
        os.remove(path)


def record_has_required_fields(record: dict[str, Any]) -> bool:
    return all(field in record for field in REQUIRED_OUTPUT_FIELDS)


def normalize_existing_output(path: str) -> None:
    if not os.path.exists(path):
        return

    records_by_signature_id: dict[int, dict[str, Any]] = {}
    changed = False

    for record in iter_jsonl(path):
        if not isinstance(record, dict):
            changed = True
            continue

        raw_signature_id = record.get("signature_id")
        try:
            signature_id = int(raw_signature_id)
        except Exception:
            changed = True
            continue

        normalized_record = dict(record)
        normalized_record["signature_id"] = signature_id
        if not record_has_required_fields(normalized_record):
            changed = True
            continue

        if signature_id in records_by_signature_id:
            changed = True
        records_by_signature_id[signature_id] = normalized_record

    if not changed:
        return

    with open(path, "w", encoding="utf-8") as f:
        for record in records_by_signature_id.values():
            f.write(json.dumps(record) + "\n")

    print(
        f"Normalized existing output at {path}; kept "
        f"{len(records_by_signature_id)} complete signature records."
    )


def load_processed_signature_ids(path: str) -> set[int]:
    if not os.path.exists(path):
        return set()

    processed: set[int] = set()
    for record in iter_jsonl(path):
        if not isinstance(record, dict):
            continue

        signature_id = record.get("signature_id")
        if not record_has_required_fields(record):
            continue
        try:
            processed.add(int(signature_id))
        except Exception:
            continue
    return processed


def load_signature_to_group_id(sig_groups_jsonl: str) -> dict[int, int]:
    if not os.path.exists(sig_groups_jsonl):
        raise FileNotFoundError(f"sig_groups.jsonl not found at {sig_groups_jsonl}")

    signature_to_group: dict[int, int] = {}
    for record in iter_jsonl(sig_groups_jsonl):
        if not isinstance(record, dict):
            continue

        raw_group_id = record.get("Sig_group_id")
        signatures = record.get("signatures", [])
        try:
            group_id = int(raw_group_id)
        except Exception:
            print(f"[WARN] Skipping sig group row with invalid Sig_group_id: {raw_group_id!r}")
            continue

        if not isinstance(signatures, list):
            print(
                f"[WARN] Sig group {group_id} has non-list `signatures`; skipping its members."
            )
            continue

        for raw_signature_id in signatures:
            try:
                signature_id = int(raw_signature_id)
            except Exception:
                print(
                    f"[WARN] Sig group {group_id} contains invalid signature id: "
                    f"{raw_signature_id!r}"
                )
                continue

            previous_group_id = signature_to_group.get(signature_id)
            if previous_group_id is not None and previous_group_id != group_id:
                print(
                    f"[WARN] Signature {signature_id} appears in multiple sig groups: "
                    f"{previous_group_id} and {group_id}. Keeping the first."
                )
                continue

            signature_to_group[signature_id] = group_id

    return signature_to_group


def load_group_to_duration_minutes(durations_csv: str) -> dict[int, float]:
    if not os.path.exists(durations_csv):
        raise FileNotFoundError(
            f"sig_group_job_durations.csv not found at {durations_csv}"
        )

    group_to_duration: dict[int, float] = {}
    with open(durations_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):
            raw_group_id = row.get("signature_group_id")
            raw_duration = row.get("duration_minutes")
            try:
                group_id = int(str(raw_group_id).strip())
            except Exception:
                print(
                    f"[WARN] Skipping duration row {row_num} with invalid "
                    f"signature_group_id: {raw_group_id!r}"
                )
                continue

            try:
                duration = float(str(raw_duration).strip())
            except Exception:
                print(
                    f"[WARN] Skipping duration row {row_num} for group {group_id} "
                    f"with invalid duration_minutes: {raw_duration!r}"
                )
                continue

            group_to_duration[group_id] = duration

    return group_to_duration


def measurement_sort_key(
    measurement: dict[str, Any],
    *,
    original_index: int,
) -> tuple[str, int, int]:
    push_timestamp = measurement.get("push_timestamp")
    submit_time = measurement.get("submit_time")
    push_id = measurement.get("push_id")

    timestamp_key = ""
    if isinstance(push_timestamp, str) and push_timestamp.strip():
        timestamp_key = push_timestamp.strip()
    elif isinstance(submit_time, str) and submit_time.strip():
        timestamp_key = submit_time.strip()

    try:
        push_id_key = int(push_id)
    except Exception:
        push_id_key = -1

    return (timestamp_key, push_id_key, original_index)


def infer_replicate_counts(
    measurements: list[dict[str, Any]],
    *,
    signature_id: int,
) -> int:
    ordered_measurements = [
        row
        for _, row in sorted(
            (
                (
                    measurement_sort_key(row, original_index=original_index),
                    row,
                )
                for original_index, row in enumerate(measurements)
                if isinstance(row, dict) and row.get("id") is not None
            ),
            key=lambda item: item[0],
        )
    ]
    if not ordered_measurements:
        return 0

    measurement_ids = [row["id"] for row in ordered_measurements]
    replicate_counts_by_id = Counter(measurement_ids)
    newest_measurement = ordered_measurements[-1]
    newest_id = newest_measurement["id"]
    newest_count = replicate_counts_by_id[newest_id]

    validation_measurement = None
    for measurement in reversed(ordered_measurements[:-1]):
        if measurement["id"] != newest_id:
            validation_measurement = measurement
            break

    if validation_measurement is None:
        return newest_count

    validation_id = validation_measurement["id"]
    validation_count = replicate_counts_by_id[validation_id]
    if validation_count != newest_count:
        selected_count = max(newest_count, validation_count)
        print(
            f"[WARN] Signature {signature_id} replicate count mismatch between "
            f"newest sample id {newest_id!r} "
            f"(push_timestamp={newest_measurement.get('push_timestamp')!r}) -> "
            f"{newest_count} and older validation sample id {validation_id!r} "
            f"(push_timestamp={validation_measurement.get('push_timestamp')!r}) "
            f"-> {validation_count}. Keeping the larger count: {selected_count}."
        )
        return selected_count
    return newest_count


def fetch_signature_summary_metadata(
    client: Any,
    signature_id: int,
    *,
    interval_seconds: int,
) -> dict[str, Any]:
    params = {
        "repository": REPOSITORY,
        "signature": signature_id,
        "interval": interval_seconds,
        "all_data": True,
        "replicates": False,
    }

    try:
        summary_list = client._get_json("performance/summary", **params)
    except Timeout as e:
        print(f"[WARN] Timeout fetching summary metadata for signature {signature_id}: {e}")
        return {field: None for field in SUMMARY_METADATA_FIELDS}
    except RequestException as e:
        print(
            f"[WARN] Request error fetching summary metadata for signature "
            f"{signature_id}: {e}"
        )
        return {field: None for field in SUMMARY_METADATA_FIELDS}
    except Exception as e:
        print(
            f"[WARN] Unexpected error fetching summary metadata for signature "
            f"{signature_id}: {e}"
        )
        return {field: None for field in SUMMARY_METADATA_FIELDS}

    if not summary_list:
        print(f"[WARN] No Treeherder summary returned for signature {signature_id}.")
        return {field: None for field in SUMMARY_METADATA_FIELDS}

    first_summary = summary_list[0]
    if not isinstance(first_summary, dict):
        print(
            f"[WARN] Treeherder summary for signature {signature_id} had an "
            "unexpected shape."
        )
        return {field: None for field in SUMMARY_METADATA_FIELDS}

    return {field: first_summary.get(field) for field in SUMMARY_METADATA_FIELDS}


def process_signatures(
    client: Any,
    input_jsonl: str,
    output_path: str,
    signature_to_group_id: dict[int, int],
    group_to_duration_minutes: dict[int, float],
    *,
    interval_seconds: int,
    limit: int,
    overwrite: bool,
) -> None:
    if not os.path.exists(input_jsonl):
        raise FileNotFoundError(
            f"Input JSONL not found at {input_jsonl}. Run "
            "get_perf_test_data_per_sig.py first to produce replicate data."
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    truncate_if_requested(output_path, overwrite)
    normalize_existing_output(output_path)

    processed = load_processed_signature_ids(output_path)
    scoped_total = 0

    print(
        f"Processing signatures from {input_jsonl} into {output_path} "
        f"(already_have={len(processed)}, limit={limit or 'all'}, "
        f"summary_interval={interval_seconds}s)."
    )

    for record in iter_jsonl(input_jsonl):
        if not isinstance(record, dict):
            continue

        signature_id = record.get("signature_id")
        try:
            signature_id = int(signature_id)
        except Exception:
            print(f"[WARN] Skipping record with invalid signature_id: {signature_id!r}")
            continue

        scoped_total += 1
        if limit > 0 and scoped_total > limit:
            break

        if signature_id in processed:
            continue

        measurements = record.get("perf_measurement_data", [])
        if not isinstance(measurements, list):
            print(
                f"[WARN] Signature {signature_id} has non-list "
                "`perf_measurement_data`; treating it as empty."
            )
            measurements = []

        replicate_counts = infer_replicate_counts(
            measurements,
            signature_id=signature_id,
        )
        group_id = signature_to_group_id.get(signature_id)
        if group_id is None:
            print(f"[WARN] No Sig_group_id found for signature {signature_id}.")
            job_duration = None
        else:
            job_duration = group_to_duration_minutes.get(group_id)
            if job_duration is None:
                print(
                    f"[WARN] No duration_minutes found for Sig_group_id {group_id} "
                    f"(signature {signature_id})."
                )
        summary_metadata = fetch_signature_summary_metadata(
            client,
            signature_id,
            interval_seconds=interval_seconds,
        )

        append_jsonl(
            output_path,
            {
                "signature_id": signature_id,
                "replicate_counts": replicate_counts,
                "job_duration": job_duration,
                **summary_metadata,
            },
        )
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

    limit = DEBUG_SIGNATURE_LIMIT if args.debug else args.limit
    signature_to_group_id = load_signature_to_group_id(args.sig_groups_jsonl)
    group_to_duration_minutes = load_group_to_duration_minutes(
        args.sig_group_durations_csv
    )
    client = TreeherderClient()
    process_signatures(
        client,
        args.input_jsonl,
        args.output,
        signature_to_group_id,
        group_to_duration_minutes,
        interval_seconds=SUMMARY_INTERVAL_SECONDS,
        limit=limit,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
