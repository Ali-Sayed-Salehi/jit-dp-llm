"""
Fetch Treeherder performance signatures and sample job durations
over a configurable time window, saving a CSV, statistics JSON,
and a histogram plot of job durations.

This version reads pre-fetched jobs from perf_jobs_by_signature.jsonl
produced by get_num_perf_tests.py, but fetches per-job timing details
from Treeherder when computing durations.
"""

import math
import os
import csv
import matplotlib.pyplot as plt
import statistics as stats
import json
from thclient import TreeherderClient

# Determine repo root dynamically:
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

# Output paths
DATASET_DIR = os.path.join(REPO_ROOT, "datasets", "mozilla_perf")
CSV_PATH = os.path.join(DATASET_DIR, "job_durations.csv")
PLOT_PATH = os.path.join(DATASET_DIR, "job_durations.png")

# Input: per-signature jobs cache produced by get_num_perf_tests.py
SIGNATURE_JOBS_JSONL = os.path.join(DATASET_DIR, "perf_jobs_by_signature.jsonl")

REPOSITORY = "autoland"
client = TreeherderClient()

os.makedirs(DATASET_DIR, exist_ok=True)


def load_signature_jobs(jsonl_path: str):
    """
    Load per-signature jobs from perf_jobs_by_signature.jsonl.

    Each line is:
      {"signature_id": <int>, "jobs": [<job dicts>]}
    Returns dict[int, list[dict]] mapping signature_id -> jobs list.
    """
    signature_jobs = {}

    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(
            f"Signature jobs JSONL not found at {jsonl_path}. "
            "Run get_num_perf_tests.py first to generate it."
        )

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
        f"from {jsonl_path}."
    )
    return signature_jobs


def job_duration_minutes(job: dict):
    """
    Fetch per-job details from Treeherder and compute duration in minutes.

    Expects a summary job dict that contains a \"job_id\" field. The
    duration is computed as:
        (end_timestamp - start_timestamp) / 60
    If anything is missing or invalid, returns None.
    """
    if not isinstance(job, dict):
        return None

    job_id = job.get("job_id")
    if job_id is None:
        return None

    try:
        jobs = client.get_jobs(REPOSITORY, id=job_id)
        if not jobs:
            return None

        job_info = jobs[0]
        start_ts = job_info.get("start_timestamp")
        end_ts = job_info.get("end_timestamp")
        if start_ts is None or end_ts is None:
            return None

        dur_seconds = float(end_ts) - float(start_ts)
        return dur_seconds / 60.0
    except Exception:
        return None

# -------------------------
# Load already-processed signatures
# -------------------------
processed_signatures = set()
csv_exists = os.path.exists(CSV_PATH)
if csv_exists:
    with open(CSV_PATH, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip header if present
        for row in reader:
            try:
                processed_signatures.add(int(row[0]))
            except:
                pass

print(f"Loaded {len(processed_signatures)} processed signatures from previous runs.")

# -------------------------
# Load all signature jobs from cache
# -------------------------
signature_jobs = load_signature_jobs(SIGNATURE_JOBS_JSONL)
print(f"Total signatures in cache: {len(signature_jobs)}")

# -------------------------
# Write CSV row-by-row
# -------------------------
csv_mode = "a" if csv_exists else "w"

with open(CSV_PATH, csv_mode, newline="") as csvfile:

    writer = csv.writer(csvfile)

    if not csv_exists:
        writer.writerow(["signature_id", "duration_minutes"])

    for signature_id, jobs_list in signature_jobs.items():
        # Skip signatures already processed
        if signature_id in processed_signatures:
            continue

        # Require at least 3 jobs in the cache; otherwise skip this signature
        if not jobs_list or len(jobs_list) < 3:
            continue

        # Pick first, middle, last jobs
        ids_to_fetch = [
            jobs_list[0],
            jobs_list[len(jobs_list) // 2],
            jobs_list[-1],
        ]

        durations = []
        for job in ids_to_fetch:
            dur = job_duration_minutes(job)
            if dur is not None:
                durations.append(dur)

        # If we couldn't compute at least 1 durations, skip this signature
        if len(durations) < 1:
            continue

        # Mean of sampled durations
        duration_mean = sum(durations) / len(durations)

        # Format to two decimal places
        formatted_duration = f"{duration_mean:.2f}"

        # Append to CSV
        writer.writerow([signature_id, formatted_duration])
        csvfile.flush()

print(f"CSV written row-by-row to {CSV_PATH}")

# ============================================================
# 4) PLOT USING THE CSV FILE
# ============================================================

plot_durations = []

# Load CSV values back for plotting
with open(CSV_PATH, "r", newline="") as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # header
    for row in reader:
        try:
            # duration saved as string, must convert to float
            plot_durations.append(float(row[1]))
        except:
            pass

# === Added helper functions for statistics ===

def percentile(data, p):
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
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data[int(k)]
    d0 = data[f] * (c - k)
    d1 = data[c] * (k - f)
    return d0 + d1

def trimmed_mean(data, trim_fraction=0.10):
    """
    10% trimmed mean by default: drop lowest and highest
    trim_fraction of the data.
    """
    if not data:
        return float("nan")
    n = len(data)
    k = int(n * trim_fraction)
    if 2 * k >= n:
        # Not enough data to trim; fall back to regular mean
        return stats.mean(data)
    trimmed = data[k:n - k]
    return stats.mean(trimmed)

def geometric_mean(data):
    """
    Geometric mean of positive values in data.
    Ignores non-positive values (0 or negative); if none remain,
    returns NaN.
    """
    positive = [x for x in data if x > 0]
    if not positive:
        return float("nan")
    log_sum = sum(math.log(x) for x in positive)
    return math.exp(log_sum / len(positive))

# Plot + stats
if plot_durations:
    # ------- Statistics section -------
    sorted_data = sorted(plot_durations)
    n = len(sorted_data)

    mean_val = stats.mean(sorted_data)
    median_val = stats.median(sorted_data)
    min_val = sorted_data[0]
    max_val = sorted_data[-1]

    p25 = percentile(sorted_data, 25)
    p75 = percentile(sorted_data, 75)
    p90 = percentile(sorted_data, 90)
    p95 = percentile(sorted_data, 95)
    p99 = percentile(sorted_data, 99)

    tmean_10 = trimmed_mean(sorted_data, 0.10)
    gmean = geometric_mean(sorted_data)

    stats_json = {
        "observations": n,
        "mean": mean_val,
        "median": median_val,
        "min": min_val,
        "max": max_val,
        "percentiles": {
            "25": p25,
            "75": p75,
            "90": p90,
            "95": p95,
            "99": p99
        },
        "trimmed_mean_10pct": tmean_10,
        "geometric_mean": gmean
    }

    # Print stats to console
    print("\n=== Job Duration Statistics (minutes) ===")
    print(json.dumps(stats_json, indent=4))
    print("========================================\n")

    # Save stats to JSON file
    STATS_PATH = os.path.join(DATASET_DIR, "job_duration_stats.json")
    with open(STATS_PATH, "w") as f:
        json.dump(stats_json, f, indent=4)

    print(f"Saved statistics JSON to {STATS_PATH}\n")

    # ------- Plot section (unchanged) -------
    plt.hist(plot_durations, bins=50)
    plt.xlabel("Job duration (minutes)")
    plt.ylabel("Number of performance tests")
    plt.title("Distribution of performance test job durations")
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    print(f"Saved plot to {PLOT_PATH}")
    plt.show()
else:
    print("No durations found in CSV; plot not generated.")
