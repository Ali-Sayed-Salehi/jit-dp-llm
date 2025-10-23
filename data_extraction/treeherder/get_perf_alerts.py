#!/usr/bin/env python3
"""
Mozilla Perfherder – Regression Bugs & Tests Collector (Incremental & Resumable)
================================================================================

What this script produces (incrementally, with resume)
------------------------------------------------------
It crawls Perfherder **performance alert summaries** (paged) going back ~1 year.
For each alert summary it:
  1) **Appends** the raw row to:
       datasets/mozilla_perf/alert_summaries.csv
     (skips if this alert summary `id` already exists, so runs are resumable)

  2) If the alert summary represents an **addressed regression** (statuses:
     fixed / wontfix / backedout) and has a `bug_number`, it extracts the
     regression **perf tests** and **appends** a compact row to:
       datasets/mozilla_perf/alerts_with_bug_and_test_info.csv
     (also skips if this summary `id` already exists there)

If the script is interrupted, simply run it again — it will **continue** from where
it left off by checking existing CSV IDs and skipping duplicates.

Debug mode
----------
Use `--debug` to stop after **10 newly-saved rows** (to the raw CSV). This is
counted only for *new* rows appended during this run.

Example:
    python collect_regressions_incremental.py --debug
"""

import argparse
import ast
import os
import time
from typing import List, Dict, Set, Optional
from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
from thclient import TreeherderClient

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_PATH, "datasets", "mozilla_perf")

RAW_CSV = os.path.join(OUT_DIR, "alert_summaries.csv")
DERIVED_CSV = os.path.join(OUT_DIR, "alerts_with_bug_and_test_info.csv")

TIMESPAN_IN_YEARS = 10
TIMESPAN_IN_DAYS = TIMESPAN_IN_YEARS * 365
DEBUG_MAX_NEW_ITEMS = 10

# Perfherder status codes
ALERT_SUMMARY_STATUS_DICT = {
    "untriaged": 0,
    "downstream": 1,
    "reassigned": 2,
    "invalid": 3,
    "improvement": 4,
    "investigating": 5,
    "wontfix": 6,
    "fixed": 7,
    "backedout": 8,
}

INCLUDED_ALERT_SUMMARY_STATUSES = {
    ALERT_SUMMARY_STATUS_DICT["wontfix"],
    ALERT_SUMMARY_STATUS_DICT["fixed"],
    ALERT_SUMMARY_STATUS_DICT["backedout"],
}

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)

def file_exists_nonempty(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0

def load_existing_ids(csv_path: str, id_col: str = "id") -> Set[int]:
    """Load existing IDs from a CSV (if present). Reads only the id column for speed."""
    if not file_exists_nonempty(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=[id_col])
        # Convert to ints where possible
        return set(int(x) for x in df[id_col].dropna().tolist())
    except Exception:
        return set()

def load_existing_ids_generic(csv_path: str, id_col: str) -> Set[int]:
    """Same as load_existing_ids, but allows arbitrary id column names."""
    if not file_exists_nonempty(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=[id_col])
        return set(int(x) for x in df[id_col].dropna().tolist())
    except Exception:
        return set()

def append_rows_to_csv(rows: List[Dict], csv_path: str):
    """Append a list of dict rows to CSV, creating header only on first write."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not file_exists_nonempty(csv_path)
    df.to_csv(csv_path, mode="a", header=header, index=False)

def parse_next_page(next_url: Optional[str]) -> Optional[int]:
    """Extract next page number from a 'next' URL (e.g., '?page=3')."""
    if not next_url:
        return None
    try:
        parts = next_url.split("page=", 1)
        if len(parts) < 2:
            return None
        tail = parts[1]
        if "&" in tail:
            tail = tail.split("&", 1)[0]
        return int(tail)
    except Exception:
        return None

def is_addressed_regression(alert_summary: Dict) -> bool:
    status = alert_summary.get("status")
    return status in INCLUDED_ALERT_SUMMARY_STATUSES

def extract_regression_tests_signatures(alert_summary: Dict) -> List[Dict]:
    """
    From `alerts` and `related_alerts`, collect `series_signature.test`
    where `is_regression` is True.
    """
    signatures_set = set()
    for key in ("alerts", "related_alerts"):
        items = alert_summary.get(key, [])
        if isinstance(items, str):
            # If the raw CSV was reloaded at some point as strings
            try:
                items = ast.literal_eval(items)
            except Exception:
                items = []
        for alert in items or []:
            if not isinstance(alert, dict):
                continue
            if not alert.get("is_regression"):
                continue
            sig = alert.get("series_signature") or {}
            test_name = sig.get("test")
            platform_name = sig.get("machine_platform")

            if test_name or platform_name:
                signatures_set.add((test_name, platform_name))  

    return [{"test": t, "platform": p} for t, p in signatures_set]

def build_compact_row(alert_summary: Dict) -> Optional[Dict]:
    """
    Build a single compact row for the DERIVED_CSV if applicable.
    Returns None if we should not include this summary (e.g., not an addressed regression).
    """
    if not is_addressed_regression(alert_summary):
        return None

    bug = alert_summary.get("bug_number")
    if pd.isna(bug) or bug in (None, ""):
        return None

    sigs_list = extract_regression_tests_signatures(alert_summary)
    return {
        "regression_bug_id": int(bug),
        "reg_perf_tests_sigs_list": sigs_list,
        "perf_reg_alert_summary_id": alert_summary.get("id"),
        "id": alert_summary.get("id"),  # store the id as well to make skipping easy
    }

# --------------------------------------------------------------------------------------
# Core incremental fetch + save
# --------------------------------------------------------------------------------------

def incremental_fetch_and_save(client: TreeherderClient, debug: bool):
    existing_raw_ids = load_existing_ids(RAW_CSV, "id")
    existing_derived_ids = load_existing_ids_generic(DERIVED_CSV, "perf_reg_alert_summary_id")

    if debug:
        remaining_quota = max(0, DEBUG_MAX_NEW_ITEMS - len(existing_raw_ids))
        print(f"🔧 Debug mode ON: need at most {remaining_quota} more raw rows to reach {DEBUG_MAX_NEW_ITEMS} total.")
        if remaining_quota == 0:
            print("✅ Already have the debug sample on disk; skipping fetch.")
            return
    else:
        remaining_quota = float("inf")

    print(f"Resuming with {len(existing_raw_ids)} raw rows already on disk, "
          f"{len(existing_derived_ids)} derived rows already on disk.")

    now = datetime.now()
    threshold_time = now - relativedelta(days=TIMESPAN_IN_DAYS)

    uri = "performance/alertsummary"
    params = {"page": 1}

    new_raw_saved = 0
    alert_push_time = now

    while alert_push_time >= threshold_time and new_raw_saved < remaining_quota:
        resp = client._get_json(uri, **params)
        results = resp.get("results", [])
        if not results:
            break

        rows_to_append_raw = []
        rows_to_append_derived = []

        for row in results:
            if new_raw_saved >= remaining_quota:
                break

            row_id = row.get("id")
            if row_id is None:
                continue

            if int(row_id) in existing_raw_ids:
                # If raw exists but derived missing, still try to backfill derived
                if int(row_id) not in existing_derived_ids:
                    compact = build_compact_row(row)
                    if compact:
                        rows_to_append_derived.append({k: v for k, v in compact.items() if k != "id"})
                        existing_derived_ids.add(int(row_id))
                continue

            # New raw row
            rows_to_append_raw.append(row)
            existing_raw_ids.add(int(row_id))
            new_raw_saved += 1

            compact = build_compact_row(row)
            if compact and int(row_id) not in existing_derived_ids:
                rows_to_append_derived.append({k: v for k, v in compact.items() if k != "id"})
                existing_derived_ids.add(int(row_id))

        if rows_to_append_raw:
            append_rows_to_csv(rows_to_append_raw, RAW_CSV)
            print(f"⬇️  Appended {len(rows_to_append_raw)} raw rows "
                  f"(added this run: {new_raw_saved}, total on disk: {len(existing_raw_ids)}).")

        if rows_to_append_derived:
            append_rows_to_csv(rows_to_append_derived, DERIVED_CSV)
            print(f"🧩 Appended {len(rows_to_append_derived)} derived rows.")

        if new_raw_saved >= remaining_quota:
            print("🔚 Debug total quota reached. Stopping.")
            break

        next_page = parse_next_page(resp.get("next"))
        if not next_page:
            break
        params["page"] = next_page

        last_ts = results[-1].get("push_timestamp")
        if last_ts is None:
            break
        alert_push_time = datetime.fromtimestamp(last_ts)

        time.sleep(0.1 if debug else 0.5)


def reconcile_missing_derived_from_raw():
    if not file_exists_nonempty(RAW_CSV):
        return

    print("🔎 Reconciling: filling any missing derived rows from existing raw CSV...")
    raw_df = pd.read_csv(RAW_CSV)
    if "id" not in raw_df.columns:
        print("⚠️  Raw CSV missing 'id' column; skipping reconcile.")
        return

    derived_ids = load_existing_ids_generic(DERIVED_CSV, "perf_reg_alert_summary_id")
    need_df = raw_df[~raw_df["id"].astype(int).isin(derived_ids)].copy()

    if need_df.empty:
        print("✅ Nothing to reconcile; derived CSV is up-to-date.")
        return

    # Safely literal-eval list-like columns
    for col in ("alerts", "related_alerts"):
        if col in need_df.columns:
            need_df.loc[:, col] = need_df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

    rows_to_append = []
    for _, row in need_df.iterrows():
        row_dict = row.to_dict()
        compact = build_compact_row(row_dict)
        if compact:
            compact.pop("id", None)
            rows_to_append.append(compact)

    if rows_to_append:
        append_rows_to_csv(rows_to_append, DERIVED_CSV)
        print(f"🧩 Reconciled and appended {len(rows_to_append)} missing derived rows.")
    else:
        print("ℹ️  No eligible rows to reconcile into derived CSV.")

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Incremental, resumable collector of regression bugs & tests from Perfherder alert summaries.")
    parser.add_argument("--debug", action="store_true", help=f"Fetch only {DEBUG_MAX_NEW_ITEMS} NEW raw rows this run.")
    args = parser.parse_args()

    ensure_out_dir()
    client = TreeherderClient()

    incremental_fetch_and_save(client, debug=args.debug)
    reconcile_missing_derived_from_raw()

    print(f"✅ Done.\n- Raw CSV:     {RAW_CSV}\n- Derived CSV: {DERIVED_CSV}")

if __name__ == "__main__":
    main()
