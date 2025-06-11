import pandas as pd
from pprint import pprint
import sys
import os
import time
from fetch_commit_data import get_commit_diff, get_commit_message

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "datasets", "jit_dp", "apachejit_small.csv")

apachejit_df = pd.read_csv(csv_path)
apachejit_list = apachejit_df.to_dict(orient='records')

jit_with_diff_list = []
failed_commits = []

OWNER = "apache"
MAX_RETRIES = 1
RETRY_DELAY = 1  # seconds
REQUEST_DELAY = 1.4  # rate limit is 5000 req/hour

for commit in apachejit_list:
    project = commit.get('project')
    project_parts = project.split("/")
    repo = project_parts[1]
    commit_hash = commit.get('commit_id')

    retries = 0
    success = False

    while retries < MAX_RETRIES and not success:
        try:
            message = get_commit_message(OWNER, repo, commit_hash)
            diff = get_commit_diff(OWNER, repo, commit_hash)
            success = True
            if retries > 0:
                print(f"✅ Success after {retries} retries")
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            retries += 1
            if retries < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                print(f"❌ Skipping commit {commit_hash} after {MAX_RETRIES} failed attempts.")
                failed_commits.append(commit)
                break

    if not success:
        continue

    commit_with_diff = commit
    commit_with_diff['commit_message'] = message
    commit_with_diff['diff'] = diff
    jit_with_diff_list.append(commit_with_diff)

# Save successful commits to JSONL
jit_with_diff_df = pd.DataFrame(jit_with_diff_list)
jit_with_diff_path = os.path.join(script_dir, "..", "datasets", "jit_dp", "apachejit_with_diff_small.jsonl")
jit_with_diff_df.to_json(jit_with_diff_path, orient="records", lines=True)
print(f"✅ Saved {len(jit_with_diff_list)} successful commits to {jit_with_diff_path}")

# Save failed commits to CSV
if failed_commits:
    failed_df = pd.DataFrame(failed_commits)
    failed_path = os.path.join(script_dir, "..", "datasets", "jit_dp", "apachejit_failed_small.csv")
    failed_df.to_csv(failed_path, index=False)
    print(f"⚠️ Saved {len(failed_commits)} failed commits to {failed_path}")
else:
    print("✅ All commits processed successfully!")
