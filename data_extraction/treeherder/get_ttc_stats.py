#!/usr/bin/env python3
"""
Compute "time to culprit" (TTC) stats from alert summaries.

TTC is defined as:
  created_time - push_timestamp

Where:
  - created is an ISO timestamp like: 2025-10-23T14:17:17.853020
  - push_timestamp is a Unix epoch seconds timestamp like: 1760754049

Outputs summary statistics to a JSON file.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import statistics as stats
import sys
from typing import Any, Dict, List, Optional


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATASET_DIR = os.path.join(REPO_ROOT, "datasets", "mozilla_perf")

DEFAULT_INPUT_CSV = os.path.join(DATASET_DIR, "alert_summaries.csv")
DEFAULT_OUTPUT_JSON = os.path.join(DATASET_DIR, "time_to_culprit_stats.json")


def _parse_created_iso(value: str) -> Optional[dt.datetime]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        created = dt.datetime.fromisoformat(value)
    except ValueError:
        return None

    if created.tzinfo is None:
        return created.replace(tzinfo=dt.timezone.utc)
    return created.astimezone(dt.timezone.utc)


def _parse_push_timestamp(value: str) -> Optional[int]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _percentile(sorted_values: List[float], p: float) -> Optional[float]:
    if not sorted_values:
        return None
    if p <= 0:
        return float(sorted_values[0])
    if p >= 1:
        return float(sorted_values[-1])
    idx = int(round((len(sorted_values) - 1) * p))
    idx = max(0, min(len(sorted_values) - 1, idx))
    return float(sorted_values[idx])


def _basic_stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min_seconds": None,
            "max_seconds": None,
            "mean_seconds": None,
            "median_seconds": None,
            "stdev_seconds": None,
            "p90_seconds": None,
            "p95_seconds": None,
            "p99_seconds": None,
        }

    values_sorted = sorted(values)
    mean_seconds = float(stats.fmean(values_sorted))
    median_seconds = float(stats.median(values_sorted))
    stdev_seconds = float(stats.pstdev(values_sorted)) if len(values_sorted) > 1 else 0.0

    return {
        "count": len(values_sorted),
        "min_seconds": float(values_sorted[0]),
        "max_seconds": float(values_sorted[-1]),
        "mean_seconds": mean_seconds,
        "median_seconds": median_seconds,
        "stdev_seconds": stdev_seconds,
        "p90_seconds": _percentile(values_sorted, 0.90),
        "p95_seconds": _percentile(values_sorted, 0.95),
        "p99_seconds": _percentile(values_sorted, 0.99),
        "mean_hours": mean_seconds / 3600.0,
        "median_hours": median_seconds / 3600.0,
        "max_hours": float(values_sorted[-1]) / 3600.0,
    }


def compute_stats(csv_path: str) -> Dict[str, Any]:
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)

    total_rows = 0
    missing_or_invalid_rows = 0
    negative_deltas = 0
    deltas: List[float] = []

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            created = _parse_created_iso(row.get("created", ""))
            push_ts = _parse_push_timestamp(row.get("push_timestamp", ""))
            if created is None or push_ts is None:
                missing_or_invalid_rows += 1
                continue

            push_time = dt.datetime.fromtimestamp(push_ts, tz=dt.timezone.utc)
            delta_seconds = (created - push_time).total_seconds()
            if delta_seconds < 0:
                negative_deltas += 1
            deltas.append(delta_seconds)

    summary = _basic_stats(deltas)
    summary.update(
        {
            "total_rows": total_rows,
            "missing_or_invalid_rows": missing_or_invalid_rows,
            "negative_delta_rows": negative_deltas,
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute time-to-culprit statistics from alert_summaries.csv"
    )
    parser.add_argument(
        "--input-csv",
        default=DEFAULT_INPUT_CSV,
        help="Path to alert_summaries.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--output-json",
        default=DEFAULT_OUTPUT_JSON,
        help="Where to write stats JSON (default: %(default)s)",
    )
    args = parser.parse_args()

    input_csv = os.path.abspath(args.input_csv)
    output_json = os.path.abspath(args.output_json)

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    payload: Dict[str, Any] = {
        "input_csv": input_csv,
        "generated_at_utc": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "definition": {
            "ttc_seconds": "created(UTC) - push_timestamp(UTC epoch seconds)"
        },
        "stats": compute_stats(input_csv),
    }

    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote TTC stats to: {output_json}")


if __name__ == "__main__":
    main()
