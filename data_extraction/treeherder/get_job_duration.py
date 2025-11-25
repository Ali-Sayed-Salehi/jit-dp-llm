from math import ceil
import os
import csv
import matplotlib.pyplot as plt
from thclient import TreeherderClient
from pprint import pprint

# Determine repo root dynamically:
# script is at: repo-root/data_extraction/treeherder/get_job_duration.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

# Output paths under repo-root/datasets/mozilla_perf/
DATASET_DIR = os.path.join(REPO_ROOT, "datasets", "mozilla_perf")
CSV_PATH = os.path.join(DATASET_DIR, "job_durations.csv")
PLOT_PATH = os.path.join(DATASET_DIR, "job_durations.png")
NO_DATA_SIG_PATH = os.path.join(DATASET_DIR, "no_data_signatures.txt")

# How far back we look
TIMEFRAME = 60 * 24 * 60 * 60  # 60 days

client = TreeherderClient()

# Ensure output directory exists
os.makedirs(DATASET_DIR, exist_ok=True)

# --------------------------------------------------
# Load signatures that previously had no jobs
# --------------------------------------------------
no_data_signatures = set()
if os.path.exists(NO_DATA_SIG_PATH):
    with open(NO_DATA_SIG_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                no_data_signatures.add(int(line))
            except ValueError:
                continue

print(f"Loaded {len(no_data_signatures)} signatures with no jobs from previous runs.")

# --------------------------------------------------
# Load signatures already processed in CSV
# --------------------------------------------------
processed_signatures = set()
csv_exists = os.path.exists(CSV_PATH)

if csv_exists:
    with open(CSV_PATH, mode="r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip header
        for row in reader:
            if not row:
                continue
            try:
                processed_signatures.add(int(row[0]))
            except ValueError:
                continue

print(f"Loaded {len(processed_signatures)} processed signatures from previous runs.")

# --------------------------------------------------
# 1) Get all performance test signatures.
# --------------------------------------------------
signature_summary_params = {}
signatures = client._get_json("performance/signatures", "autoland", **signature_summary_params)
print(f"Signatures size: {len(signatures)}")

durations_minutes = []

# --------------------------------------------------
# 2) Write CSV row-by-row
# --------------------------------------------------
csv_mode = "a" if csv_exists else "w"

with open(CSV_PATH, mode=csv_mode, newline="") as csvfile, \
     open(NO_DATA_SIG_PATH, mode="a") as no_data_file:

    writer = csv.writer(csvfile)

    if not csv_exists:
        writer.writerow(["signature_id", "duration_minutes"])

    # Loop over signatures
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
            print(f"No performance summary found for signature: {signature_id}")
            continue

        jobs_list = data_list[0].get("data", [])
        if not jobs_list:
            # print(f"No jobs found for signature (has summary but no jobs): {signature_id}")
            if signature_id not in no_data_signatures:
                no_data_signatures.add(signature_id)
                no_data_file.write(f"{signature_id}\n")
                no_data_file.flush()
            continue

        # --------------------------------------------------
        # mean of first, middle, and last job durations
        # --------------------------------------------------
        idx_first = 0
        idx_last = len(jobs_list) - 1
        idx_middle = len(jobs_list) // 2

        job_ids_to_sample = [
            jobs_list[idx_first]["job_id"],
            jobs_list[idx_middle]["job_id"],
            jobs_list[idx_last]["job_id"]
        ]

        durations = []
        for job_id in job_ids_to_sample:
            job_info = client.get_jobs("autoland", id=job_id)[0]
            dur_s = job_info["end_timestamp"] - job_info["start_timestamp"]
            durations.append(dur_s / 60.0)  # convert to minutes

        duration_mean = sum(durations) / len(durations)

        # Record to histogram + CSV
        durations_minutes.append(duration_mean)
        writer.writerow([signature_id, duration_mean])
        csvfile.flush()

print(f"CSV written row-by-row to {CSV_PATH}")
print(f"Updated no-jobs signatures saved to {NO_DATA_SIG_PATH}")

# --------------------------------------------------
# 4) Plot the distribution and save figure
# --------------------------------------------------
if durations_minutes:
    plt.hist(durations_minutes, bins=2)
    plt.xlabel("Job duration (minutes)")
    plt.ylabel("Number of performance tests")
    plt.title("Distribution of performance test job durations (last timeframe)")
    plt.tight_layout()

    plt.savefig(PLOT_PATH)
    print(f"Saved plot to {PLOT_PATH}")

    plt.show()
else:
    print("No performance test jobs found in the timeframe; plot not generated.")
