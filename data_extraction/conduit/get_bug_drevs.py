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

# - For each bug id from perf_bugs.csv fetch its corresponding differential revision
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

all_diff_revisions_path = os.path.join(REPO_PATH, "datasets", "mozilla_perf", "all_differential_revisions.csv")

all_revisions_df = pd.read_csv(all_diff_revisions_path)
all_revisions_df['fields'] = all_revisions_df['fields'].apply(ast.literal_eval)
all_revisions_list = all_revisions_df.to_dict(orient='records')

perf_bugs_path = os.path.join(REPO_PATH, "datasets", "mozilla_perf", "perf_bugs.csv")
bugs_df = pd.read_csv(perf_bugs_path)
bugs_df['regressed_perf_tests'] = bugs_df['regressed_perf_tests'].apply(ast.literal_eval)
bugs_list = bugs_df.to_dict(orient='records')

bugs_with_drevs_list = []

for bug in bugs_list:
    bug_with_drevs = bug
    bugzilla_bug_id = bug['bug_id']
    drevs_for_bug = []

    for drev in all_revisions_list:
        if str(drev['fields']['bugzilla.bug-id']).strip() == str(bugzilla_bug_id).strip():
            if(drev['fields']['status']['value'] == "published"):
                drevs_for_bug.append(drev)

    if len(drevs_for_bug) > 1:
        print(f"🛑 Bug with bugzilla id {bugzilla_bug_id} has multiple drevs.")

    bug_with_drevs['drev_ids'] = []
    bug_with_drevs['diff_ids'] = []
    bug_with_drevs['titles'] = []
    bug_with_drevs['summaries'] = []

    if drevs_for_bug:
        for drev in drevs_for_bug:
            bug_with_drevs['drev_ids'].append(drev['id'])
            bug_with_drevs['diff_ids'].append(drev['fields'].get('diffID', ''))
            bug_with_drevs['titles'].append(drev['fields'].get('title', ''))
            bug_with_drevs['summaries'].append(drev['fields'].get('summary', ''))

    bugs_with_drevs_list.append(bug_with_drevs)

bugs_with_drevs_df = pd.DataFrame(bugs_with_drevs_list)
perf_bugs_with_drevs_path = os.path.join(REPO_PATH, "datasets", "mozilla_perf", "perf_bugs_with_drevs.csv")
bugs_with_drevs_df.to_csv(perf_bugs_with_drevs_path, index=False)

print(f"✅ Saved all perf bugs with added drev id and diff id to {perf_bugs_with_drevs_path}")

