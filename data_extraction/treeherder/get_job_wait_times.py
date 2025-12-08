#!/usr/bin/env python3
"""
Compute job wait times (submit → start) for the revision with
the highest number of perf jobs on autoland.

This script:
  - Loads perf job counts per revision from perf_jobs_per_revision.csv.
  - Identifies the revision with the largest total_jobs value.
  - Loads per-signature job summaries from perf_jobs_by_signature.jsonl.
  - Collects all job_ids whose summary job's revision matches the target revision.
  - Fetches full job objects from Treeherder for those job_ids.
  - Saves all detailed jobs to a JSONL file.
  - Reloads that JSONL and computes wait time statistics based on
    (start_timestamp - submit_timestamp), which are Unix epoch seconds.
  - Writes the statistics to a JSON file.
"""

import argparse
import csv
import json
import os
import statistics as stats
from typing import Dict, List, Tuple, Any

from thclient import TreeherderClient

# -------------------------------------------------------------------
# Paths (repo root inferred from this script's location)
# -------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

DATASET_DIR = os.path.join(REPO_ROOT, "datasets", "mozilla_perf")

# Inputs
REVISION_COUNTS_CSV = os.path.join(DATASET_DIR, "perf_jobs_per_revision.csv")
SIGNATURE_JOBS_JSONL = os.path.join(DATASET_DIR, "perf_jobs_by_signature.jsonl")

# Outputs
MAX_REVISION_JOBS_JSONL = os.path.join(DATASET_DIR, "max_revision_jobs.jsonl")
WAIT_TIME_STATS_JSON = os.path.join(DATASET_DIR, "job_wait_time_stats.json")

REPOSITORY = "autoland"

os.makedirs(DATASET_DIR, exist_ok=True)

client = TreeherderClient()


# -------------------------------------------------------------------
# Helpers for loading inputs
# -------------------------------------------------------------------
def load_max_revision(csv_path: str) -> Tuple[str, int]:
    """
    Load perf_jobs_per_revision.csv and return (revision, total_jobs)
    for the row with the highest total_jobs value.

    CSV format:
      revision,submit_time_iso,total_jobs
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Revision counts CSV not found at {csv_path}. "
            "Run get_num_perf_tests.py first to generate it."
        )

    max_revision = None
    max_jobs = -1

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            try:
                # Be robust: take last column as total_jobs
                total_jobs = int(row[-1])
            except Exception:
                continue

            rev = row[0]
            if total_jobs > max_jobs:
                max_jobs = total_jobs
                max_revision = rev

    if max_revision is None:
        raise ValueError(f"No valid rows found in {csv_path}.")

    print(
        f"Max-jobs revision: {max_revision} with {max_jobs} perf jobs "
        f"(from {os.path.basename(csv_path)})."
    )
    return max_revision, max_jobs


def load_signature_jobs(jsonl_path: str) -> Dict[int, List[dict]]:
    """
    Load per-signature jobs from perf_jobs_by_signature.jsonl.

    Each line is:
      {"signature_id": <int>, "jobs": [<job dicts>]}
    Returns dict[int, list[dict]] mapping signature_id -> jobs list.
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(
            f"Signature jobs JSONL not found at {jsonl_path}. "
            "Run get_num_perf_tests.py first to generate it."
        )

    signature_jobs: Dict[int, List[dict]] = {}

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
            jobs = record.get("jobs", [])
            if sig_id is None:
                continue
            try:
                sig_id_int = int(sig_id)
            except Exception:
                continue

            signature_jobs[sig_id_int] = jobs

    print(
        f"Loaded jobs for {len(signature_jobs)} signatures "
        f"from {os.path.basename(jsonl_path)}."
    )
    return signature_jobs


def collect_job_ids_for_revision(
    signature_jobs: Dict[int, List[dict]],
    revision: str,
) -> Dict[int, dict]:
    """
    From the per-signature job summaries, collect unique job_ids whose
    job['revision'] equals the target revision.

    Returns a dict: {job_id -> summary_job_dict}.
    """
    job_by_id: Dict[int, dict] = {}

    total_signatures = len(signature_jobs)
    for idx, (sig_id, jobs) in enumerate(signature_jobs.items(), start=1):
        print(
            f"[{idx}/{total_signatures}] Scanning signature {sig_id} for revision {revision}..."
        )
        if not jobs:
            continue

        for job in jobs:
            if not isinstance(job, dict):
                continue

            if job.get("revision") != revision:
                continue

            job_id = job.get("job_id")
            if job_id is None:
                continue

            try:
                job_id_int = int(job_id)
            except Exception:
                continue

            # Deduplicate by job_id
            job_by_id[job_id_int] = job

    print(
        f"Found {len(job_by_id)} unique job_ids on revision {revision} "
        f"across {total_signatures} signatures."
    )
    return job_by_id


# -------------------------------------------------------------------
# Treeherder job fetching and JSONL writing
# -------------------------------------------------------------------
def load_existing_job_ids(jsonl_path: str, revision: str) -> List[int]:
    """
    Load job IDs already present in max_revision_jobs.jsonl for the
    given revision, so we don't refetch them.
    """
    if not os.path.exists(jsonl_path):
        return []

    job_ids: List[int] = []
    revisions_in_file = set()

    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                job = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(job, dict):
                continue

            rev = job.get("revision")
            if rev is not None:
                revisions_in_file.add(rev)

            # Treeherder jobs usually have "id" as the job id; fall back to "job_id".
            job_id = job.get("id", job.get("job_id"))
            if job_id is None:
                continue
            try:
                job_ids.append(int(job_id))
            except Exception:
                continue

    if revisions_in_file and (revisions_in_file != {revision}):
        print(
            f"[WARN] Existing {os.path.basename(jsonl_path)} contains jobs for "
            f"revisions {sorted(revisions_in_file)}; expected only {revision}. "
            "Previously stored jobs will still be used for de-duplication."
        )

    print(
        f"Found {len(job_ids)} existing detailed jobs for de-duplication "
        f"in {jsonl_path}."
    )
    return job_ids


def fetch_and_append_jobs(job_ids: List[int], out_path: str) -> int:
    """
    Fetch full job objects from Treeherder for each job_id in job_ids and
    append them to the JSONL file one-by-one.

    Returns the number of jobs successfully fetched and written.
    """
    if not job_ids:
        print("No new job_ids to fetch; skipping Treeherder calls.")
        return 0

    fetched_count = 0
    total = len(job_ids)

    # Append mode so we can resume runs without losing previous data.
    with open(out_path, "a") as f:
        for idx, job_id in enumerate(job_ids, start=1):
            print(f"[{idx}/{total}] Fetching job_id={job_id} from Treeherder...")
            try:
                jobs = client.get_jobs(REPOSITORY, id=job_id)
            except Exception as e:
                print(f"[WARN] Error fetching job_id={job_id}: {e}")
                continue

            if not jobs:
                print(f"[WARN] No jobs returned for job_id={job_id}")
                continue

            job = jobs[0]
            f.write(json.dumps(job) + "\n")
            f.flush()
            fetched_count += 1

    print(f"Fetched and wrote {fetched_count} new detailed jobs out of {total} job_ids.")
    return fetched_count


# -------------------------------------------------------------------
# Wait time computation and statistics
# -------------------------------------------------------------------
def load_jobs_from_jsonl(jsonl_path: str) -> List[dict]:
    """Load job dicts from a JSONL file."""
    jobs: List[dict] = []
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Jobs JSONL not found at {jsonl_path}")

    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                job = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(job, dict):
                jobs.append(job)

    print(f"Loaded {len(jobs)} jobs from {jsonl_path}")
    return jobs


def compute_wait_times_seconds(jobs: List[dict]) -> List[float]:
    """
    For each job, compute wait time in seconds as:
        start_timestamp - submit_timestamp

    Both fields are expected to be Unix epoch times (seconds).
    Jobs with missing or invalid timestamps, or negative waits,
    are skipped.
    """
    waits: List[float] = []
    skipped_missing = 0
    skipped_negative = 0

    for job in jobs:
        submit_ts = job.get("submit_timestamp")
        start_ts = job.get("start_timestamp")

        if submit_ts is None or start_ts is None:
            skipped_missing += 1
            continue

        try:
            submit_f = float(submit_ts)
            start_f = float(start_ts)
        except (TypeError, ValueError):
            skipped_missing += 1
            continue

        wait = start_f - submit_f
        if wait < 0:
            skipped_negative += 1
            continue

        waits.append(wait)

    print(
        f"Computed wait times for {len(waits)} jobs "
        f"(skipped {skipped_missing} missing and {skipped_negative} negative)."
    )
    return waits


def percentile(data: List[float], p: float) -> float:
    """
    Compute the p-th percentile (0-100) using linear interpolation
    between closest ranks.
    """
    if not data:
        return float("nan")
    if p <= 0:
        return data[0]
    if p >= 100:
        return data[-1]

    k = (len(data) - 1) * (p / 100.0)
    low = int(k)
    high = min(low + 1, len(data) - 1)
    if low == high:
        return data[low]
    frac = k - low
    return data[low] * (1.0 - frac) + data[high] * frac


def compute_wait_time_stats(wait_seconds: List[float]) -> dict:
    """Compute summary statistics for a list of wait times in seconds."""
    if not wait_seconds:
        return {}

    data = sorted(wait_seconds)
    n = len(data)

    mean_val = stats.mean(data)
    median_val = stats.median(data)
    min_val = data[0]
    max_val = data[-1]

    stats_json = {
        "observations": n,
        "seconds": {
            "mean": mean_val,
            "median": median_val,
            "min": min_val,
            "max": max_val,
            "p25": percentile(data, 25),
            "p75": percentile(data, 75),
            "p90": percentile(data, 90),
            "p95": percentile(data, 95),
            "p99": percentile(data, 99),
        },
        "minutes": {
            "mean": mean_val / 60.0,
            "median": median_val / 60.0,
            "min": min_val / 60.0,
            "max": max_val / 60.0,
            "p25": percentile(data, 25) / 60.0,
            "p75": percentile(data, 75) / 60.0,
            "p90": percentile(data, 90) / 60.0,
            "p95": percentile(data, 95) / 60.0,
            "p99": percentile(data, 99) / 60.0,
        },
    }

    return stats_json


def write_wait_time_stats(stats_dict: dict, out_path: str) -> None:
    """Write wait time statistics to a JSON file."""
    if not stats_dict:
        print("No wait time statistics to write (empty input).")
        return

    with open(out_path, "w") as f:
        json.dump(stats_dict, f, indent=2)

    print(f"Wrote wait time statistics to {out_path}")


# -------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------
def main(dry_run: bool = False) -> None:
    # 1) Identify the revision with the highest total_jobs.
    target_revision, total_jobs = load_max_revision(REVISION_COUNTS_CSV)

    # 2) Load all per-signature job summaries.
    signature_jobs = load_signature_jobs(SIGNATURE_JOBS_JSONL)

    # 3) Collect all unique job_ids whose summary job has the target revision.
    jobs_for_revision = collect_job_ids_for_revision(signature_jobs, target_revision)
    if not jobs_for_revision:
        print(f"No jobs found for revision {target_revision}; nothing to do.")
        return

    print(
        f"Target revision {target_revision} has {len(jobs_for_revision)} "
        f"unique job_ids in the summary cache (CSV reported {total_jobs} jobs)."
    )

    # 4) Determine which jobs are already present in the JSONL file, so we
    #    can resume without re-fetching them.
    existing_job_ids = set(load_existing_job_ids(MAX_REVISION_JOBS_JSONL, target_revision))
    all_job_ids = set(jobs_for_revision.keys())
    job_ids_to_fetch = sorted(all_job_ids - existing_job_ids)

    print(
        f"{len(existing_job_ids)} job_ids already have detailed entries; "
        f"{len(job_ids_to_fetch)} remain to fetch."
    )

    if dry_run:
        # In dry-run mode, only fetch a small subset for quick experimentation.
        original_count = len(job_ids_to_fetch)
        job_ids_to_fetch = job_ids_to_fetch[:10]
        print(
            f"[DRY RUN] Limiting fetch to {len(job_ids_to_fetch)} job_ids "
            f"(out of {original_count} remaining)."
        )

    # Fetch missing jobs and append them row-by-row so the run can be resumed.
    fetched_this_run = fetch_and_append_jobs(job_ids_to_fetch, MAX_REVISION_JOBS_JSONL)
    if not fetched_this_run and not existing_job_ids:
        print("No detailed jobs available (none fetched and none previously stored).")
        return

    # 5) Reload JSONL and compute wait times.
    jobs_loaded = load_jobs_from_jsonl(MAX_REVISION_JOBS_JSONL)
    wait_seconds = compute_wait_times_seconds(jobs_loaded)

    if not wait_seconds:
        print("No valid wait times computed; not writing stats JSON.")
        return

    stats_dict = compute_wait_time_stats(wait_seconds)
    stats_dict["revision"] = target_revision
    stats_dict["total_jobs_from_csv"] = total_jobs
    stats_dict["unique_job_ids_in_cache"] = len(jobs_for_revision)
    stats_dict["detailed_jobs_in_jsonl"] = len(jobs_loaded)
    stats_dict["detailed_jobs_fetched_this_run"] = fetched_this_run

    # 6) Write wait time statistics to JSON.
    write_wait_time_stats(stats_dict, WAIT_TIME_STATS_JSON)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compute wait time statistics for jobs on the busiest autoland revision. "
            "Uses max_revision_jobs.jsonl as an append-only cache so interrupted "
            "runs can be resumed."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only fetch up to 10 new jobs for quick experimentation.",
    )
    args = parser.parse_args()

    main(dry_run=args.dry_run)
