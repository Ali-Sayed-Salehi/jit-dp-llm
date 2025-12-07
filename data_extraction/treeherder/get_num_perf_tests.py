#!/usr/bin/env python3
"""
Compute how many performance jobs run per revision on autoland.

This script:
- Loads perf test signatures from job_durations.csv.
- Uses all_commits.jsonl as the canonical set of autoland revisions,
  restricted to a configurable timeframe and excluding very recent submissions.
- For each relevant revision, counts jobs whose submit_time is within
  five minutes after the push timestamp.
- Writes a CSV with revision, submission timestamp, and total job count,
  plus summary statistics and a distribution plot.
"""

import os
import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta
from statistics import mean, median, geometric_mean, quantiles
import matplotlib.pyplot as plt
from requests.exceptions import Timeout, RequestException

from thclient import TreeherderClient

# ---------------------------------------------------------
# Paths (repo root inferred from this script's location)
# ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

DATASET_DIR = os.path.join(REPO_ROOT, "datasets", "mozilla_perf")
JOB_DURATIONS_CSV = os.path.join(DATASET_DIR, "job_durations.csv")

# Output CSV: counts of perf jobs per revision
REVISION_COUNTS_CSV = os.path.join(DATASET_DIR, "perf_jobs_per_revision.csv")

# Input JSONL: all commits on the repository (Mercurial autoland)
ALL_COMMITS_JSONL = os.path.join(DATASET_DIR, "all_commits.jsonl")

# NEW: paths for statistics JSON and distribution plot
STATS_JSON = os.path.join(DATASET_DIR, "perf_jobs_stats.json")
DIST_PLOT_PNG = os.path.join(DATASET_DIR, "perf_jobs_per_revision_dist.png")

os.makedirs(DATASET_DIR, exist_ok=True)

# ---------------------------------------------------------
# Parameters
# ---------------------------------------------------------
TIMEFRAME_DAYS = 90
# how many days back counts as "recent" for commit submission
RECENT_SUBMISSION_DAYS = 60

REPOSITORY = "autoland"

client = TreeherderClient()

def load_revisions_from_all_commits(jsonl_path: str):
    """
    Load revisions from all_commits.jsonl that:
      - fall within the TIMEFRAME_DAYS window, and
      - are older than RECENT_SUBMISSION_DAYS (i.e., ignore very recent submissions).

    Returns a dict: {revision_hash -> submission_datetime_utc}.
    """
    revisions = {}

    now = datetime.utcnow()
    timeframe_cutoff = now - timedelta(days=TIMEFRAME_DAYS)
    recent_cutoff = now - timedelta(days=RECENT_SUBMISSION_DAYS)

    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"all_commits.jsonl not found at {jsonl_path}")

    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            node = record.get("node")
            date_field = record.get("date")
            if not node or not isinstance(date_field, list) or not date_field:
                continue

            # date is [unix_timestamp, offset]; use the unix timestamp in UTC
            try:
                commit_dt = datetime.utcfromtimestamp(date_field[0])
            except Exception:
                continue

            # Filter by timeframe first
            if commit_dt < timeframe_cutoff:
                continue

            # Ignore revisions that were submitted in the last RECENT_SUBMISSION_DAYS days
            if commit_dt >= recent_cutoff:
                continue

            revisions[node] = commit_dt

    print(
        f"Loaded {len(revisions)} revisions from all_commits.jsonl "
        f"within timeframe and older than {RECENT_SUBMISSION_DAYS} days."
    )
    return revisions


def load_signatures_from_csv(csv_path: str):
    """Load all signature IDs from job_durations.csv."""
    signatures = set()
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            try:
                signatures.add(int(row[0]))
            except Exception:
                pass
    print(f"Loaded {len(signatures)} signatures.")
    return signatures


def fetch_jobs_for_signature(signature_id: int):
    """Fetch recent jobs for a signature from Treeherder."""
    timeframe_seconds = TIMEFRAME_DAYS * 24 * 60 * 60
    params = {
        "repository": REPOSITORY,
        "signature": signature_id,
        "interval": timeframe_seconds,
        "all_data": True,
        "replicates": False,
    }

    try:
        data_list = client._get_json("performance/summary", **params)
    except Timeout as e:
        # This covers ReadTimeout and other timeout variants
        print(
            f"[WARN] Timeout when fetching jobs for signature {signature_id}: {e}. "
            "Skipping this signature."
        )
        return []
    except RequestException as e:
        # Any other requests-related network error
        print(
            f"[WARN] Request error when fetching jobs for signature {signature_id}: {e}. "
            "Skipping this signature."
        )
        return []
    except Exception as e:
        # Last-resort guard so a weird error doesn't kill the whole script
        print(
            f"[WARN] Unexpected error when fetching jobs for signature {signature_id}: {e}. "
            "Skipping this signature."
        )
        return []

    if not data_list:
        return []

    return data_list[0].get("data", []) or []


def aggregate_revision_counts(signatures, allowed_revisions):
    """
    Accumulate revision -> total number of jobs across all signatures.

    Rules:
      - Only count jobs where submit_time is within 5 minutes AFTER push_timestamp.
      - Only consider revisions present in allowed_revisions, which are:
          * within the TIMEFRAME_DAYS window, and
          * older than RECENT_SUBMISSION_DAYS days (recent submissions are ignored).
    """
    # For counting jobs that pass the 5-minute filter
    revision_jobs = defaultdict(int)

    dt_format = "%Y-%m-%dT%H:%M:%S"
    max_diff = timedelta(minutes=5)

    total = len(signatures)
    for idx, sig_id in enumerate(signatures, start=1):
        print(f"[{idx}/{total}] Fetching jobs for signature {sig_id}...")
        jobs = fetch_jobs_for_signature(sig_id)

        for job in jobs:
            rev = job.get("revision")
            if not rev or rev not in allowed_revisions:
                continue

            push_ts_str = job.get("push_timestamp")
            submit_time_str = job.get("submit_time")

            # Only count jobs that pass the 5-minute condition
            if not push_ts_str or not submit_time_str:
                continue

            try:
                push_ts = datetime.strptime(push_ts_str, dt_format)
                submit_time = datetime.strptime(submit_time_str, dt_format)
            except ValueError:
                # Skip jobs with malformed timestamps for counting
                continue

            diff = submit_time - push_ts
            # Only consider jobs where submit_time is in [push_ts, push_ts + 5 min]
            if diff < timedelta(0) or diff > max_diff:
                continue

            # Tentatively count; we'll drop revisions in recent_revisions later
            revision_jobs[rev] += 1

    # Ensure all allowed revisions are present, even if they had zero jobs
    for rev in allowed_revisions:
        revision_jobs.setdefault(rev, 0)

    print(f"Collected job counts for {len(revision_jobs)} revisions.")
    return dict(revision_jobs)


def write_revision_counts_csv(revision_counts, revision_timestamps, out_path: str):
    """
    Write revision -> (submit_time_iso, job count) to CSV.

    Rows are sorted by submission timestamp ascending.
    """
    # Build rows with timestamps; drop revisions we don't have timestamps for
    rows = []
    for rev, count in revision_counts.items():
        ts = revision_timestamps.get(rev)
        if ts is None:
            continue
        rows.append((rev, ts, count))

    # Sort by timestamp
    rows.sort(key=lambda x: x[1])

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["revision", "submit_time_iso", "total_jobs"])

        for rev, ts, count in rows:
            writer.writerow([rev, ts.isoformat(), count])

    print(f"Wrote revision counts CSV to {out_path}")


# NEW: utilities for stats & plotting, using the saved CSV
def load_counts_from_csv(csv_path: str):
    """Read job counts per revision from perf_jobs_per_revision.csv."""
    counts = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            try:
                # current format: [revision, submit_time_iso, total_jobs]
                # be robust and always read the last column as the count
                cnt = int(row[-1])
                counts.append(cnt)
            except Exception:
                continue
    return counts


def compute_statistics(counts):
    """
    Compute statistics on job counts per revision.

    Returns a dict suitable for JSON serialization.
    """
    if not counts:
        return {}

    n = len(counts)
    sorted_counts = sorted(counts)

    # Basic stats
    mean_val = mean(sorted_counts)
    median_val = median(sorted_counts)
    min_val = sorted_counts[0]
    max_val = sorted_counts[-1]

    # Percentiles using statistics.quantiles (n=100 -> 1..99 percentiles)
    qs = quantiles(sorted_counts, n=100, method="inclusive")
    # Index i corresponds to (i+1)th percentile
    p25 = qs[24]   # 25th
    p75 = qs[74]   # 75th
    p90 = qs[89]   # 90th
    p95 = qs[94]   # 95th
    p99 = qs[98]   # 99th

    # 10% trimmed mean: drop lowest and highest 10% of observations
    k = int(0.10 * n)
    if k > 0 and (n - 2 * k) > 0:
        trimmed = sorted_counts[k:-k]
    else:
        # Not enough data to trim meaningfully; fall back to full data
        trimmed = sorted_counts
    trimmed_mean_val = mean(trimmed)

    # Geometric mean (requires all counts > 0; that's true here)
    geo_mean_val = geometric_mean(sorted_counts)

    # Round a bit for nicer JSON output (you can adjust precision)
    def r(x):
        return round(x, 2)

    stats = {
        "observations": n,
        "mean_jobs": r(mean_val),
        "median_jobs": r(median_val),
        "min_jobs": int(min_val),
        "max_jobs": int(max_val),
        "percentiles": {
            "25": r(p25),
            "75": r(p75),
            "90": r(p90),
            "95": r(p95),
            "99": r(p99),
        },
        "trimmed_mean_10pct": r(trimmed_mean_val),
        "geometric_mean": r(geo_mean_val),
    }

    return stats


def write_stats_json(counts, out_path: str):
    """Compute statistics and write them to a JSON file."""
    stats = compute_statistics(counts)
    if not stats:
        print("No counts available to compute statistics.")
        return

    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote statistics JSON to {out_path}")


def plot_distribution(counts, out_path: str):
    """Plot a linear-scale histogram of job counts per revision."""
    if not counts:
        print("No counts available to plot distribution.")
        return

    plt.figure()
    plt.hist(counts, bins=50)  # linear scale by default
    plt.xlabel("Jobs per revision")
    plt.ylabel("Number of revisions")
    plt.title("Distribution of perf jobs per revision")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"Wrote distribution plot to {out_path}")


def main():
    # 1) Load signatures from job_durations.csv
    signatures = load_signatures_from_csv(JOB_DURATIONS_CSV)

    # 2) Load revisions from all_commits.jsonl within timeframe and not recent
    #    This returns {revision_hash -> submission_datetime_utc}
    revision_timestamps = load_revisions_from_all_commits(ALL_COMMITS_JSONL)

    # 3) Fetch all jobs and aggregate revision counts (with filters)
    revision_counts = aggregate_revision_counts(signatures, revision_timestamps)

    # 4) Save summary CSV (sorted by submission time, with timestamp column)
    write_revision_counts_csv(revision_counts, revision_timestamps, REVISION_COUNTS_CSV)

    # 5) Reload counts from CSV and compute stats + plot
    counts = load_counts_from_csv(REVISION_COUNTS_CSV)
    write_stats_json(counts, STATS_JSON)
    plot_distribution(counts, DIST_PLOT_PNG)


if __name__ == "__main__":
    main()
