"""
Fetch Treeherder performance signatures and sample job durations
over a configurable time window, saving a CSV, statistics JSON,
and a histogram plot of job durations.
"""

from math import ceil
import math
import os
import csv
import matplotlib.pyplot as plt
from thclient import TreeherderClient
from pprint import pprint
import statistics as stats
import json

# Determine repo root dynamically:
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

# Output paths
DATASET_DIR = os.path.join(REPO_ROOT, "datasets", "mozilla_perf")
CSV_PATH = os.path.join(DATASET_DIR, "job_durations.csv")
PLOT_PATH = os.path.join(DATASET_DIR, "job_durations.png")
NO_DATA_SIG_PATH = os.path.join(DATASET_DIR, "no_data_signatures.txt")

# Timeframe (in days). Adjust this value as needed.
TIMEFRAME_DAYS = 365
TIMEFRAME = TIMEFRAME_DAYS * 24 * 60 * 60

client = TreeherderClient()

os.makedirs(DATASET_DIR, exist_ok=True)

# -------------------------
# Load no-data signatures
# -------------------------
no_data_signatures = set()
if os.path.exists(NO_DATA_SIG_PATH):
    with open(NO_DATA_SIG_PATH, "r") as f:
        for line in f:
            try:
                no_data_signatures.add(int(line.strip()))
            except:
                pass

print(f"Loaded {len(no_data_signatures)} signatures with no jobs from previous runs.")

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
# Fetch all signatures
# -------------------------
signatures = client._get_json("performance/signatures", "autoland")
print(f"Signatures size: {len(signatures)}")

# -------------------------
# Write CSV row-by-row
# -------------------------
csv_mode = "a" if csv_exists else "w"

with open(CSV_PATH, csv_mode, newline="") as csvfile, \
     open(NO_DATA_SIG_PATH, "a") as no_data_file:

    writer = csv.writer(csvfile)

    if not csv_exists:
        writer.writerow(["signature_id", "duration_minutes"])

    for sig in signatures.values():
        signature_id = sig["id"]

        # Skip signatures already processed or known to have no jobs
        if signature_id in processed_signatures or signature_id in no_data_signatures:
            continue

        performance_summary_params = {
            "repository": "autoland",
            "signature": signature_id,
            "interval": TIMEFRAME,
            "all_data": True,
            "replicates": False,
        }

        data_list = client._get_json("performance/summary", **performance_summary_params)
        if not data_list:
            continue

        jobs_list = data_list[0].get("data", [])
        if not jobs_list:
            # Save only if summary exists but no jobs
            if signature_id not in no_data_signatures:
                no_data_signatures.add(signature_id)
                no_data_file.write(f"{signature_id}\n")
                no_data_file.flush()
            continue

        # Pick first, middle, last jobs
        ids_to_fetch = [
            jobs_list[0]["job_id"],
            jobs_list[len(jobs_list) // 2]["job_id"],
            jobs_list[-1]["job_id"],
        ]

        durations = []
        for job_id in ids_to_fetch:
            job_info = client.get_jobs("autoland", id=job_id)[0]
            dur_s = job_info["end_timestamp"] - job_info["start_timestamp"]
            durations.append(dur_s / 60.0)

        # Mean of sampled durations
        duration_mean = sum(durations) / len(durations)

        # Format to two decimal places
        formatted_duration = f"{duration_mean:.2f}"

        # Append to CSV
        writer.writerow([signature_id, formatted_duration])
        csvfile.flush()

print(f"CSV written row-by-row to {CSV_PATH}")
print(f"Updated no-jobs signatures saved to {NO_DATA_SIG_PATH}")

# ============================================================
# 4) PLOT USING THE CSV FILE (NOT IN-MEMORY DURATIONS)
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
