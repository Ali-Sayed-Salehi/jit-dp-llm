#!/usr/bin/env python3
"""
Filter failing perf signature alert summaries by framework and creation time.

Inputs (under `datasets/mozilla_perf_bisect/`):
  - `alert_summary_fail_perf_sigs.csv`
  - `alert_summaries.csv`

Rows are removed from `alert_summary_fail_perf_sigs.csv` when the matching
`alert_summaries.csv` row:
  - has `framework` in {2, 6, 18}, or
  - has `created` before 2025-06-19T00:00:00Z.

Output:
  - `alert_summary_fail_perf_sigs_no_fw_2_6_18.csv`

The script also prints how many rows were removed and their alert summary IDs.
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import UTC, datetime


# Allow reading very large CSV fields (notably the `alerts` column).
csv.field_size_limit(sys.maxsize)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATASET_DIR = os.path.join(REPO_ROOT, "datasets", "mozilla_perf_bisect")

INPUT_FAIL_SIGS_CSV = os.path.join(DATASET_DIR, "alert_summary_fail_perf_sigs.csv")
INPUT_ALERT_SUMMARIES_CSV = os.path.join(DATASET_DIR, "alert_summaries.csv")
OUTPUT_CSV = os.path.join(
    DATASET_DIR,
    "alert_summary_fail_perf_sigs_no_fw_2_6_18.csv",
)

EXCLUDED_FRAMEWORKS = {2, 6, 18}
CREATED_CUTOFF = datetime(2025, 6, 19, tzinfo=UTC)


def parse_created(created_raw: str) -> datetime | None:
    """Parse a created timestamp and normalize it to UTC."""
    created_raw = (created_raw or "").strip()
    if not created_raw:
        return None

    try:
        created_dt = datetime.fromisoformat(created_raw)
    except ValueError:
        return None

    # Treeherder timestamps in this CSV are naive ISO strings; treat them as UTC.
    if created_dt.tzinfo is None:
        return created_dt.replace(tzinfo=UTC)
    return created_dt.astimezone(UTC)


def load_alert_summary_metadata(path: str) -> dict[str, tuple[int | None, datetime | None]]:
    """
    Return:
      alert_summary_id -> (framework_int_or_none, created_dt_or_none)
    """
    metadata: dict[str, tuple[int | None, datetime | None]] = {}

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            alert_summary_id = (row.get("id") or "").strip()
            if not alert_summary_id:
                continue

            framework_raw = (row.get("framework") or "").strip()
            try:
                framework = int(framework_raw) if framework_raw else None
            except ValueError:
                framework = None

            created_dt = parse_created(row.get("created") or "")
            metadata[alert_summary_id] = (framework, created_dt)

    return metadata


def main() -> int:
    alert_summary_metadata = load_alert_summary_metadata(INPUT_ALERT_SUMMARIES_CSV)

    kept_rows: list[dict[str, str]] = []
    removed_ids: list[str] = []
    missing_in_alert_summaries: list[str] = []

    with open(INPUT_FAIL_SIGS_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {INPUT_FAIL_SIGS_CSV}")
        fieldnames = reader.fieldnames

        for row in reader:
            alert_summary_id = (row.get("alert_summary_id") or "").strip()

            metadata = alert_summary_metadata.get(alert_summary_id)
            if metadata is None:
                missing_in_alert_summaries.append(alert_summary_id)
                kept_rows.append(row)
                continue

            framework, created_dt = metadata

            remove_for_framework = framework in EXCLUDED_FRAMEWORKS
            remove_for_created = created_dt is not None and created_dt < CREATED_CUTOFF

            if remove_for_framework or remove_for_created:
                removed_ids.append(alert_summary_id)
                continue

            kept_rows.append(row)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    print(f"Input rows: {len(kept_rows) + len(removed_ids)}")
    print(f"Removed rows: {len(removed_ids)}")
    print(f"Removed alert_summary_id values: {removed_ids}")
    print(f"Kept rows: {len(kept_rows)}")
    if missing_in_alert_summaries:
        print(
            f"Warning: {len(missing_in_alert_summaries)} alert_summary_id values were not found "
            "in alert_summaries.csv and were kept as-is."
        )
        print(f"Missing alert_summary_id values: {missing_in_alert_summaries}")
    print(f"Wrote filtered CSV to: {OUTPUT_CSV}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
