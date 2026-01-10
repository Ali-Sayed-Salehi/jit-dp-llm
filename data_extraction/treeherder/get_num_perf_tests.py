#!/usr/bin/env python3
"""
Compute how many performance jobs run per revision on Mozilla autoland.

Goal
  Produce a per-revision "how many perf tests ran?" signal over a fixed UTC
  commit date window. This is useful for understanding perf coverage over time
  (e.g., before doing downstream analysis or simulation).

What gets counted
  - The unit of counting is a Treeherder performance *signature* (i.e. a perf
    test signature id), not individual job instances:
      *For each revision, count the number of unique signature IDs that have at
      least one job associated with that revision.*
  - A signature contributes at most 1 count per revision even if multiple jobs
    exist.
  - Jobs are associated to a revision when they pass a timing sanity check:
      - If the job has no `submit_time` (older Treeherder records), it is
        accepted without the timing filter.
      - Otherwise, require `push_timestamp` and only accept jobs whose
        `submit_time` is in the interval:
          [push_timestamp, push_timestamp + 10 minutes]

Data sources
  - Signature IDs are fetched from Treeherder via `performance/signatures`.
  - Jobs per signature are fetched from Treeherder via `performance/summary`
    using a lookback interval derived from `REVISION_START_DATE`.
  - The canonical set of revisions comes from:
      `datasets/mozilla_perf/all_commits.jsonl`
    (one JSON object per line containing `node` and `date`).

Caching / resumability
  Treeherder calls are the expensive step. To make the process restartable,
  per-signature job data is cached in JSONL:
    `datasets/mozilla_perf/perf_jobs_by_signature.jsonl`
  Each line is:
    {"signature_id": <int>, "jobs": [<job dict>, ...]}
  On subsequent runs, signatures already present in the cache are not refetched.

Outputs (written under `datasets/mozilla_perf/`)
  - `perf_jobs_per_revision.csv`: revision, submit_time_iso (commit time), count
  - `perf_jobs_per_revision_details.jsonl`: same + signature_ids list
  - `perf_jobs_stats.json`: summary statistics over counts
  - `perf_jobs_per_revision_dist.png`: histogram of counts

Configuration / usage
  - Edit `REVISION_START_DATE` / `REVISION_END_DATE` to change the revision
    window.
  - Run: `python data_extraction/treeherder/get_num_perf_tests.py`
  - Debug mode (`--debug`) samples up to 20 signatures to keep runtime short.
"""

import os
import csv
import json
import random
import argparse
from collections import defaultdict
from datetime import datetime, timedelta, UTC
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

# Output CSV: counts of perf jobs per revision
REVISION_COUNTS_CSV = os.path.join(DATASET_DIR, "perf_jobs_per_revision.csv")

# Output JSONL: per-revision details including signature IDs
REVISION_DETAILS_JSONL = os.path.join(DATASET_DIR, "perf_jobs_per_revision_details.jsonl")

# Input JSONL: all commits on the repository (Mercurial autoland)
ALL_COMMITS_JSONL = os.path.join(DATASET_DIR, "all_commits.jsonl")

# NEW: intermediate cache: per-signature jobs
# One line per signature:
#   {"signature_id": <int>, "jobs": [<job dicts from Treeherder>]}
SIGNATURE_JOBS_JSONL = os.path.join(DATASET_DIR, "perf_jobs_by_signature.jsonl")

# NEW: paths for statistics JSON and distribution plot
STATS_JSON = os.path.join(DATASET_DIR, "perf_jobs_stats.json")
DIST_PLOT_PNG = os.path.join(DATASET_DIR, "perf_jobs_per_revision_dist.png")

os.makedirs(DATASET_DIR, exist_ok=True)

# ---------------------------------------------------------
# Parameters
# ---------------------------------------------------------
# Inclusive date range (UTC) for revisions to include in the output
REVISION_START_DATE = datetime(2025, 6, 19, tzinfo=UTC)
REVISION_END_DATE = datetime(2025, 10, 25, tzinfo=UTC)

# How far back to fetch jobs from Treeherder (in days), chosen so that
# the interval reaches at least REVISION_START_DATE.
TIMEFRAME_DAYS = max(1, (datetime.now(UTC) - REVISION_START_DATE).days + 2)

REPOSITORY = "autoland"

client = TreeherderClient()


def load_revisions_from_all_commits(jsonl_path: str):
    """
    Load revisions from all_commits.jsonl that:
      - have commit timestamps between REVISION_START_DATE and REVISION_END_DATE (inclusive).

    Returns a dict: {revision_hash -> submission_datetime_utc}.
    """
    revisions = {}

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
                commit_dt = datetime.fromtimestamp(date_field[0], UTC)
            except Exception:
                continue

            # Keep only revisions whose commit time is inside the configured window
            if commit_dt < REVISION_START_DATE or commit_dt > REVISION_END_DATE:
                continue

            revisions[node] = commit_dt

    print(
        f"Loaded {len(revisions)} revisions from all_commits.jsonl "
        f"between {REVISION_START_DATE.date()} and {REVISION_END_DATE.date()}."
    )
    return revisions


def load_signatures_from_api(repository: str):
    """
    Load all performance signature IDs from Treeherder for the given repository.

    This mirrors the behavior in get_job_duration.py, but only returns the
    integer signature IDs. Some signatures may not have any jobs in the
    requested timeframe; those are handled later when fetching jobs.
    """
    try:
        signatures = client._get_json("performance/signatures", repository)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch signatures from Treeherder: {e}")

    if not signatures:
        print("Warning: no signatures returned from Treeherder.")
        return set()

    sig_ids = {int(sig["id"]) for sig in signatures.values() if "id" in sig}
    print(f"Loaded {len(sig_ids)} signatures from Treeherder for '{repository}'.")
    return sig_ids


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


# -------------------------------------------------------------------
# NEW: Intermediate cache utilities for per-signature job data
# -------------------------------------------------------------------
def load_signature_jobs_cache(jsonl_path: str):
    """
    Load per-signature jobs from a JSONL cache file.

    Each line is expected to be:
      {"signature_id": <int>, "jobs": [<job dicts>]}
    Returns:
      dict[int, list[dict]] mapping signature_id -> jobs list.
    """
    cache = {}
    if not os.path.exists(jsonl_path):
        print(f"No existing signature jobs cache at {jsonl_path}.")
        return cache

    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            sig_id = record.get("signature_id")
            jobs = record.get("jobs")
            if sig_id is None or jobs is None:
                continue
            try:
                sig_id_int = int(sig_id)
            except Exception:
                continue

            cache[sig_id_int] = jobs

    print(f"Loaded cached jobs for {len(cache)} signatures from {jsonl_path}.")
    return cache


def append_signature_jobs(jsonl_path: str, signature_id: int, jobs):
    """
    Append a single row for a signature to the JSONL cache.

    This writes one line at a time so that if the script is interrupted,
    previously written signatures remain usable and we can resume.
    """
    record = {
        "signature_id": signature_id,
        "jobs": jobs,
    }
    # Write one signature per line for robustness
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def build_or_load_signature_jobs(signatures, jsonl_path: str):
    """
    Ensure we have job data for all signatures, using a JSONL cache.

    - Load any existing signatures from jsonl_path.
    - For signatures not in the cache, fetch from Treeherder and append
      one JSON line per newly fetched signature.
    - Return a dict[int, list[dict]]: signature_id -> jobs.
    """
    cache = load_signature_jobs_cache(jsonl_path)

    signatures = set(signatures)
    missing = signatures - set(cache.keys())
    total = len(signatures)

    if missing:
        print(
            f"{len(missing)} signatures missing from cache; "
            "fetching and appending to JSONL."
        )
    else:
        print("All signatures already present in cache; no API calls needed.")

    # Iterate in a stable order for nicer logs
    for idx, sig_id in enumerate(sorted(signatures), start=1):
        if sig_id in cache:
            print(f"[{idx}/{total}] Using cached jobs for signature {sig_id}.")
            continue

        print(f"[{idx}/{total}] Fetching jobs for signature {sig_id}...")
        jobs = fetch_jobs_for_signature(sig_id)

        # Store in cache dict and persist as a single JSONL line.
        cache[sig_id] = jobs
        append_signature_jobs(jsonl_path, sig_id, jobs)

    return cache


# -------------------------------------------------------------------
# Aggregation using cached per-signature job data
# -------------------------------------------------------------------
def aggregate_revision_counts(signature_jobs, allowed_revisions):
    """
    Accumulate revision -> total number of UNIQUE signatures with at least one job.

    Inputs:
      - signature_jobs: dict[int, list[dict]] mapping signature_id -> jobs list.
        (Jobs are taken from the per-signature cache JSONL, not from live API.)
      - allowed_revisions: dict[revision_hash -> commit_datetime_utc] used to
        filter jobs to only those revisions in our commit window.

    Rules:
      - A signature is counted at most once per revision, even if it has
        multiple jobs.
      - Only count jobs where submit_time is within 10 minutes AFTER push_timestamp.
      - Only consider revisions present in allowed_revisions, which are:
          * whose commit timestamps fall within the configured revision
            date window [REVISION_START_DATE, REVISION_END_DATE].
    """
    revision_signatures = defaultdict(set)

    dt_format = "%Y-%m-%dT%H:%M:%S"
    max_diff = timedelta(minutes=10)

    total = len(signature_jobs)
    for idx, (sig_id, jobs) in enumerate(signature_jobs.items(), start=1):
        print(f"[{idx}/{total}] Aggregating jobs for signature {sig_id}...")
        for job in jobs:
            rev = job.get("revision")
            if not rev or rev not in allowed_revisions:
                continue

            push_ts_str = job.get("push_timestamp")
            submit_time_str = job.get("submit_time")

            # If submit_time is missing (jobs older than 2025-08-08), accept the job without applying
            # the submit_time/push_timestamp timing condition.
            if not submit_time_str:
                revision_signatures[rev].add(sig_id)
                continue

            # For jobs with a submit_time, we require push_timestamp and
            # enforce the 5-minute window.
            if not push_ts_str:
                continue

            try:
                push_ts = datetime.strptime(push_ts_str, dt_format)
                submit_time = datetime.strptime(submit_time_str, dt_format)
            except ValueError:
                # Skip jobs with malformed timestamps for counting
                continue

            diff = submit_time - push_ts
            # Only consider jobs where submit_time is in [push_ts, push_ts + 10 min]
            if diff < timedelta(0) or diff > max_diff:
                continue

            revision_signatures[rev].add(sig_id)

    # Ensure all allowed revisions are present, even if they had zero jobs
    for rev in allowed_revisions:
        revision_signatures.setdefault(rev, set())

    # Derive job counts from the number of unique signatures per revision
    revision_jobs = {rev: len(sigs) for rev, sigs in revision_signatures.items()}

    print(
        f"Collected job counts for {len(revision_jobs)} revisions "
        "(unique signatures per revision)."
    )
    # convert sets to sorted lists for JSON-serializability
    return revision_jobs, {rev: sorted(sigs) for rev, sigs in revision_signatures.items()}


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


def write_revision_details_jsonl(
    revision_counts, revision_signatures, revision_timestamps, out_path: str
):
    """
    Write per-revision details to JSONL:
      {"revision", "submit_time_iso", "total_jobs", "signature_ids"}.

    Lines are sorted by submission timestamp ascending.
    """
    rows = []
    for rev, count in revision_counts.items():
        ts = revision_timestamps.get(rev)
        if ts is None:
            continue
        sigs = revision_signatures.get(rev, [])
        rows.append((rev, ts, sigs, count))

    rows.sort(key=lambda x: x[1])

    with open(out_path, "w") as f:
        for rev, ts, sigs, count in rows:
            record = {
                "revision": rev,
                "submit_time_iso": ts.isoformat(),
                "total_jobs": count,
                "signature_ids": sigs,
            }
            f.write(json.dumps(record) + "\n")

    print(f"Wrote revision details JSONL to {out_path}")


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

    # Geometric mean: only defined for strictly positive values.
    positive_counts = [c for c in sorted_counts if c > 0]
    geo_mean_val = geometric_mean(positive_counts) if positive_counts else None

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
        "geometric_mean": (r(geo_mean_val) if geo_mean_val is not None else None),
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


def main(debug: bool = False):
    # 1) Load signatures from Treeherder via the performance/signatures API.
    #    Some signatures may not have any jobs; those will simply contribute
    #    zero jobs when we fetch and aggregate.
    signatures = load_signatures_from_api(REPOSITORY)

    if debug:
        # In debug mode, only use a random subset of 20 signatures
        num_to_sample = min(20, len(signatures))
        sampled = set(random.sample(list(signatures), num_to_sample))
        print(
            f"[DEBUG] Using a random subset of {num_to_sample} signatures "
            f"out of {len(signatures)} total."
        )
        signatures = sampled

    # 2) Load revisions from all_commits.jsonl within timeframe
    #    This returns {revision_hash -> submission_datetime_utc}
    revision_timestamps = load_revisions_from_all_commits(ALL_COMMITS_JSONL)

    # 3) Build or extend the per-signature jobs JSONL cache, then use it
    #    for all subsequent computation instead of hitting the API again.
    signature_jobs = build_or_load_signature_jobs(signatures, SIGNATURE_JOBS_JSONL)

    # 4) Aggregate revision counts using cached signature jobs
    revision_counts, revision_signatures = aggregate_revision_counts(
        signature_jobs,
        revision_timestamps,
    )

    # 5) Save summary CSV (sorted by submission time, with timestamp column only)
    write_revision_counts_csv(revision_counts, revision_timestamps, REVISION_COUNTS_CSV)

    # 6) Save detailed JSONL with signature_ids included
    write_revision_details_jsonl(
        revision_counts,
        revision_signatures,
        revision_timestamps,
        REVISION_DETAILS_JSONL,
    )

    # 7) Reload counts from CSV and compute stats + plot
    counts = load_counts_from_csv(REVISION_COUNTS_CSV)
    write_stats_json(counts, STATS_JSON)
    plot_distribution(counts, DIST_PLOT_PNG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compute how many performance jobs run per revision on autoland. "
            "Uses a per-signature JSONL cache so interrupted runs can resume. "
            "In debug mode, only fetch results for 20 randomly selected "
            "signatures."
        )
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Only fetch results for 20 randomly selected signatures.",
    )
    args = parser.parse_args()

    main(debug=args.debug)
