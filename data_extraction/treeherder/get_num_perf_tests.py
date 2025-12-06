#!/usr/bin/env python3

import os
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta
from statistics import mean, median, geometric_mean, quantiles
import matplotlib.pyplot as plt
import requests
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

# NEW: paths for statistics JSON and distribution plot
STATS_JSON = os.path.join(DATASET_DIR, "perf_jobs_stats.json")
DIST_PLOT_PNG = os.path.join(DATASET_DIR, "perf_jobs_per_revision_dist.png")

os.makedirs(DATASET_DIR, exist_ok=True)

# ---------------------------------------------------------
# Parameters
# ---------------------------------------------------------
TIMEFRAME = 90 * 24 * 60 * 60
REPOSITORY = "autoland"

# how many days back counts as "recent" for push_timestamp
RECENT_PUSH_DAYS = 60

client = TreeherderClient()


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
    params = {
        "repository": REPOSITORY,
        "signature": signature_id,
        "interval": TIMEFRAME,
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


def aggregate_revision_counts(signatures):
    """
    Accumulate revision -> total number of jobs across all signatures.

    Rules:
      - Only count jobs where submit_time is within 5 minutes AFTER push_timestamp.
      - Completely ignore any revision that has at least one job whose
        push_timestamp is within the last RECENT_PUSH_DAYS days.
    """
    # For counting jobs that pass the 5-minute filter
    revision_jobs = defaultdict(int)

    # For tracking revisions that should be excluded due to recent push_timestamp
    recent_revisions = set()

    dt_format = "%Y-%m-%dT%H:%M:%S"
    max_diff = timedelta(minutes=5)

    now = datetime.utcnow()
    recent_cutoff = now - timedelta(days=RECENT_PUSH_DAYS)

    total = len(signatures)
    for idx, sig_id in enumerate(signatures, start=1):
        print(f"[{idx}/{total}] Fetching jobs for signature {sig_id}...")
        jobs = fetch_jobs_for_signature(sig_id)

        for job in jobs:
            rev = job.get("revision")
            if not rev:
                continue

            push_ts_str = job.get("push_timestamp")
            submit_time_str = job.get("submit_time")

            # --- 1) Mark revisions with a recent push_timestamp for exclusion ---
            push_ts = None
            if push_ts_str:
                try:
                    push_ts = datetime.strptime(push_ts_str, dt_format)
                    if push_ts >= recent_cutoff:
                        recent_revisions.add(rev)
                except ValueError:
                    # Malformed timestamp; don't use it for recent detection
                    push_ts = None

            # --- 2) Only count jobs that pass the 5-minute condition ---
            if not push_ts_str or not submit_time_str:
                continue

            try:
                # reuse parsed push_ts when possible
                if push_ts is None:
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

    # --- 3) Build final counts excluding recent revisions completely ---
    revision_counts = {
        rev: count
        for rev, count in revision_jobs.items()
        if rev not in recent_revisions
    }

    print(
        f"Collected job counts for {len(revision_counts)} revisions "
        f"(excluded {len(recent_revisions)} revisions with recent pushes)."
    )
    return revision_counts


def write_revision_counts_csv(revision_counts, out_path: str):
    """Write revision -> job count to CSV."""
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["revision", "total_jobs"])

        for rev, count in revision_counts.items():
            writer.writerow([rev, count])

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
                # row = [revision, total_jobs]
                cnt = int(row[1])
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

    # 2) Fetch all jobs and aggregate revision counts (with filters)
    revision_counts = aggregate_revision_counts(signatures)

    # 3) Save summary CSV
    write_revision_counts_csv(revision_counts, REVISION_COUNTS_CSV)

    # 4) Reload counts from CSV (per your requirement) and compute stats + plot
    counts = load_counts_from_csv(REVISION_COUNTS_CSV)
    write_stats_json(counts, STATS_JSON)
    plot_distribution(counts, DIST_PLOT_PNG)


if __name__ == "__main__":
    main()
