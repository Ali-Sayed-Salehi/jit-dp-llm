import requests
import time
import json
from unidiff import PatchSet
from phabricator import Phabricator
from datetime import datetime, timedelta
from pprint import pprint
import pandas as pd
import ast
from dotenv import load_dotenv
import os
import sys
import argparse
import re

def remove_bug_prefix(text):
    return re.sub(r"^Bug\s*\d+[^\w]*\s*", "", text, flags=re.IGNORECASE)

parser = argparse.ArgumentParser(description="Choose which dataset to load")
parser.add_argument("--debug", action="store_true", help="Process only a small portion of the data")
args = parser.parse_args()
DEBUG = args.debug

REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
secrets_path = os.path.join(REPO_PATH, "secrets", ".env")
load_dotenv(dotenv_path=secrets_path)

API_TOKEN = os.getenv("CONDUIT_API_TOKEN")
BASE_URL = "https://phabricator.services.mozilla.com/api/"

phab = Phabricator(token=API_TOKEN, host=BASE_URL)
phab.update_interfaces()
phab.user.whoami()

bugs_path = os.path.join(REPO_PATH, "datasets", "mozilla_perf", "perf_bugs_with_drevs.csv")
output_path = os.path.join(REPO_PATH, "datasets", "mozilla_perf", "perf_bugs_with_diff.jsonl")

# Load already-processed bug_ids from output file
processed_bug_ids = set()
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                processed_bug_ids.add(json.loads(line)["bug_id"])
            except:
                continue

bugs_df = pd.read_csv(bugs_path)
columns_to_parse = ['drev_ids', 'diff_ids', 'titles', 'summaries']
for col in columns_to_parse:
    bugs_df[col] = bugs_df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

bugs_list = bugs_df.to_dict(orient='records')
if DEBUG:
    bugs_list = bugs_list[:200]

with open(output_path, "a", encoding="utf-8") as fout:
    for bug in bugs_list:
        if bug["bug_id"] in processed_bug_ids:
            print(f"⏩ Skipping bug {bug['bug_id']} (already processed)")
            continue

        bug_diff_ids_list = bug.get('diff_ids', [])
        if not bug_diff_ids_list:
            continue

        diffs = []
        for diff_id in bug_diff_ids_list:
            try:
                raw_diff = phab.differential.getrawdiff(diffID=str(diff_id)).response
                diffs.append(raw_diff)
            except Exception as e:
                print(f"⚠️ Failed to fetch diff {diff_id} for bug {bug['bug_id']}: {e}")
                continue

        bug['raw_diff'] = "\n".join(diffs)

        titles = bug.get('titles', [])
        summaries = bug.get('summaries', [])
        bug['titles'] = "\n".join(remove_bug_prefix(t) for t in titles)
        bug['summaries'] = "\n".join(summaries)

        # Write single line
        fout.write(json.dumps(bug) + "\n")
        fout.flush()

print(f"\n🎉 All bugs processed and saved to {output_path}")
