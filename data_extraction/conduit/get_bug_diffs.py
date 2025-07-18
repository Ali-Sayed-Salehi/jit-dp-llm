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
from utils import remove_bug_prefix

# - For each DREV, get its diff and attach it as a new column to its corresponding bug id from the perf_bugs.csv

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
bugs_df = pd.read_csv(bugs_path)
columns_to_parse = ['drev_ids', 'diff_ids', 'titles', 'summaries']
for col in columns_to_parse:
    bugs_df[col] = bugs_df[col].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) else []
    )

bugs_list = bugs_df.to_dict(orient='records')

if DEBUG:
    bugs_list = bugs_list[:5]

bugs_with_diff_list = []

for bug in bugs_list:
    bug_with_diff = bug
    bug_diff_ids_list = bug.get('diff_ids', [])
    if not bug_diff_ids_list:
        continue

    full_diff = ""
    if bug_diff_ids_list:
        for diff_id in bug_diff_ids_list:
            raw_diff = phab.differential.getrawdiff(diffID = str(diff_id)).response
            new_diff = "\n" + raw_diff
            full_diff += new_diff

    bug['raw_diff'] = full_diff

    bug_drev_titles_list = bug.get('titles', [])
    full_title = ""
    for title in bug_drev_titles_list:
        title = remove_bug_prefix(title)
        new_title = "\n" + title
        full_title += new_title

    bug['drev_title'] = full_title

    bug_drev_summaries_list = bug.get('summaries', [])
    full_summary = ""
    for summary in bug_drev_titles_list:
        new_summary = "\n" + summary
        full_summary += new_summary

    bug['drev_summary'] = full_summary

    bugs_with_diff_list.append(bug_with_diff)


bugs_with_diff_df = pd.DataFrame(bugs_with_diff_list)
perf_bugs_with_diff_path = os.path.join(REPO_PATH, "datasets", "mozilla_perf", "perf_bugs_with_diff.jsonl")
bugs_with_diff_df.to_json(perf_bugs_with_diff_path, orient="records", lines=True)

print(f"✅ Saved all perf bugs with added raw diff to {perf_bugs_with_diff_path}")