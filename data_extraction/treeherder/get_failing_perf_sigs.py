#!/usr/bin/env python3
import csv
import sys

# Allow extremely large fields in CSV
csv.field_size_limit(sys.maxsize)

import ast
import json
import os
from typing import Dict, Set, List, Any



def repo_root() -> str:
    # This file is expected at: repo-root/data_extraction/treeherder/get_failing_perf_sigs.py
    # So repo root is two levels up from this file.
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, os.pardir, os.pardir))


def default_paths():
    root = repo_root()
    alerts_with_bug = os.path.join(
        root, "datasets", "mozilla_perf", "alerts_with_bug_and_test_info.csv"
    )
    alert_summaries = os.path.join(
        root, "datasets", "mozilla_perf", "alert_summaries.csv"
    )
    output = os.path.join(
        root, "datasets", "mozilla_perf", "alert_summary_fail_perf_sigs.csv"
    )
    return alerts_with_bug, alert_summaries, output


def load_needed_summary_ids(alerts_with_bug_path: str) -> Set[str]:
    """Collect unique perf_reg_alert_summary_id values from the first CSV."""
    ids: Set[str] = set()
    with open(alerts_with_bug_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        total_rows = 0
        missing_id_rows = 0
        ids: Set[str] = set()

        for row in reader:
            total_rows += 1
            sid = row.get("perf_reg_alert_summary_id")
            if not sid:
                missing_id_rows += 1
                continue
            ids.add(sid.strip())

    print("total_rows:", total_rows)
    print("unique IDs:", len(ids))
    print("rows missing perf_reg_alert_summary_id:", missing_id_rows)

    return ids


def load_alert_summaries(alert_summaries_path: str, needed_ids: Set[str]) -> Dict[str, str]:
    """
    Load alerts field from alert_summaries.csv for the summary IDs we care about.
    Returns a mapping: summary_id -> alerts_string
    """
    summaries: Dict[str, str] = {}
    with open(alert_summaries_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("id")
            if sid and sid in needed_ids:
                summaries[sid] = row.get("alerts", "") or ""
    return summaries


def parse_alerts(alerts_raw: str, summary_id: str = "") -> List[Any]:
    alerts_raw = alerts_raw.strip()
    if not alerts_raw:
        return []

    try:
        value = ast.literal_eval(alerts_raw)
    except Exception as e:
        # DEBUG: show why parsing failed
        print(f"DEBUG: Failed to parse alerts for summary {summary_id}: {e}")
        print("DEBUG: alerts_raw snippet:")
        print(alerts_raw[:500])
        return []

    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    return []


def get_failing_sig_ids_from_alerts(alerts_raw: str, summary_id: str = "") -> List[int]:
    alerts = parse_alerts(alerts_raw, summary_id=summary_id)
    sig_ids: Set[int] = set()

    for alert in alerts:
        if not isinstance(alert, dict):
            continue

        # Only count signatures that represent regressions
        if not alert.get("is_regression"):
            continue

        series_sig = alert.get("series_signature") or {}
        if not isinstance(series_sig, dict):
            continue

        sig_id = series_sig.get("id")
        if sig_id is not None:
            try:
                sig_ids.add(int(sig_id))
            except (TypeError, ValueError):
                continue

    return sorted(sig_ids)


def main(alerts_with_bug_path: str, alert_summaries_path: str, output_path: str) -> None:
    needed_ids = load_needed_summary_ids(alerts_with_bug_path)
    summaries = load_alert_summaries(alert_summaries_path, needed_ids)

    # Prepare rows for output
    rows = []
    # Sort IDs numerically when possible, otherwise lexicographically
    def sort_key(x: str):
        try:
            return int(x)
        except ValueError:
            return x

    for sid in sorted(needed_ids, key=sort_key):
        alerts_raw = summaries.get(sid, "")
        fail_sig_ids = get_failing_sig_ids_from_alerts(alerts_raw, summary_id=sid)
        rows.append(
            {
                "alert_summary_id": sid,
                "fail_perf_sig_ids": json.dumps(fail_sig_ids),
                "num_fail_perf_sig_ids": len(fail_sig_ids),
            }
        )

    # Write output CSV
    fieldnames = ["alert_summary_id", "fail_perf_sig_ids", "num_fail_perf_sig_ids"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    # Usage:
    #   python get_failing_perf_sigs.py
    #   python get_failing_perf_sigs.py <alerts_with_bug_csv> <alert_summaries_csv> <output_csv>
    if len(sys.argv) >= 4:
        alerts_with_bug_csv = sys.argv[1]
        alert_summaries_csv = sys.argv[2]
        output_csv = sys.argv[3]
    else:
        alerts_with_bug_csv, alert_summaries_csv, output_csv = default_paths()

    main(alerts_with_bug_csv, alert_summaries_csv, output_csv)
