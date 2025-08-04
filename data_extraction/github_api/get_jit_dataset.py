import pandas as pd
from pprint import pprint
import sys
import os
import time
import json
import csv
import argparse
from fetch_commit_data import get_commit_diff, get_commit_message
from javalang_structured_diff import extract_structured_diff

# ---------------------------- Parse Args ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Run in debug mode with only a few commits")
parser.add_argument("--struc", action="store_true", help="get structred diffs")
parser.add_argument("--ast", action="store_true", help="get structred diffs with ast paths in each change block")
parser.add_argument("--small", action="store_true", help="Use the apachejit small dataset instead of the complete dataset.")
parser.add_argument("--small", action="store_true", help="Use the apachejit small dataset instead of the complete dataset.")
parser.add_argument("--dataset_name", type=str, help="name of the dataset, either apachejit or jit_defects4j.")
args = parser.parse_args()

if args.ast and not args.struc:
    raise ValueError("ast flag can only be set with structured diffs")

size = "small" if args.small else "total"
DEBUG = args.debug

if args.dataset_name == "apachejit":
    dataset_name = "apachejit"
elif args.dataset_name == "jit_defects4j":
    dataset_name = "apacjit_defects4jhejit"


# ---------------------------- Setup Paths ----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "..", "datasets", dataset_name, f"{dataset_name}_{size}.csv")

if args.ast:
    success_path = os.path.join(script_dir, "..", "..", "datasets", dataset_name, f"{dataset_name}_{size}_with_struc_ast_diff.jsonl")
    fail_path = os.path.join(script_dir, "..", "..", "datasets", dataset_name, f"{dataset_name}_{size}_failed_struc_ast.csv")
elif args.struc:
    success_path = os.path.join(script_dir, "..", "..", "datasets", dataset_name, f"{dataset_name}_{size}_with_struc_diff.jsonl")
    fail_path = os.path.join(script_dir, "..", "..", "datasets", dataset_name, f"{dataset_name}_{size}_failed_struc.csv")
else:
    success_path = os.path.join(script_dir, "..", "..", "datasets", dataset_name, f"{dataset_name}_{size}_with_diff.jsonl")
    fail_path = os.path.join(script_dir, "..", "..", "datasets", dataset_name, f"{dataset_name}_{size}_failed.csv")

apachejit_df = pd.read_csv(csv_path)
apachejit_list = apachejit_df.to_dict(orient='records')

# ---------------------------- Load Already Processed ----------------------------
processed_commit_ids = set()
if os.path.exists(success_path):
    with open(success_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                processed_commit_ids.add(data.get("commit_id"))
            except json.JSONDecodeError:
                continue

if DEBUG:
    apachejit_list = apachejit_list[:10]
    print("🐞 DEBUG mode is ON: Only processing first 10 commits")


OWNER = "apache"
# Track if fail file exists to decide on writing header
fail_file_exists = os.path.exists(fail_path)

if args.struc:
    for commit in apachejit_list:
        commit_hash = commit.get('commit_id')
        if commit_hash in processed_commit_ids:
            print(f"⏩ Skipping already processed commit {commit_hash}")
            continue

        project = commit.get('project')
        project_parts = project.split("/")
        repo = project_parts[1]

        try:
            if args.ast:
                structured_diff = extract_structured_diff(OWNER, repo, commit_hash, True)
            else:
                structured_diff = extract_structured_diff(OWNER, repo, commit_hash, False)
        except Exception as e:
            print(f"❌ Skipping commit {commit_hash} due to exception: {e}")
            write_header = not fail_file_exists
            with open(fail_path, "a", newline='', encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=commit.keys())
                if write_header:
                    writer.writeheader()
                    fail_file_exists = True
                writer.writerow(commit)
            continue

        commit_with_diff = commit
        commit_with_diff['diff'] = structured_diff

        with open(success_path, "a", encoding="utf-8") as f:
            json.dump(commit_with_diff, f)
            f.write("\n")


else:
    MAX_RETRIES = 2
    RETRY_DELAY = 1  # seconds
    REQUEST_DELAY = 1.4  # GitHub API limit ~5000 req/hour

    for commit in apachejit_list:
        commit_hash = commit.get('commit_id')
        if commit_hash in processed_commit_ids:
            print(f"⏩ Skipping already processed commit {commit_hash}")
            continue

        project = commit.get('project')
        project_parts = project.split("/")
        repo = project_parts[1]

        retries = 0
        success = False

        while retries < MAX_RETRIES and not success:
            try:
                message = get_commit_message(OWNER, repo, commit_hash)
                diff = get_commit_diff(OWNER, repo, commit_hash)
                success = True
                if retries > 0:
                    print(f"✅ Success after {retries} retries for {commit_hash}")
                time.sleep(REQUEST_DELAY)
            except Exception as e:
                retries += 1
                if retries < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"❌ Skipping commit {commit_hash} after {MAX_RETRIES} failed attempts.")
                    # Append failed commit immediately
                    write_header = not fail_file_exists
                    with open(fail_path, "a", newline='', encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=commit.keys())
                        if write_header:
                            writer.writeheader()
                            fail_file_exists = True
                        writer.writerow(commit)
                    break

        if not success:
            continue

        commit_with_diff = commit
        commit_with_diff['commit_message'] = message
        commit_with_diff['diff'] = diff

        with open(success_path, "a", encoding="utf-8") as f:
            json.dump(commit_with_diff, f)
            f.write("\n")

print("✅ Finished. All results written incrementally to output files.")
