"""
Export recent Mozilla Phabricator Differential Revisions for one repository.

This script connects to Mozilla Phabricator through the Conduit API and retrieves
all Differential Revisions created within the last `YEARS_BACK` years for the
repository selected by `TARGET_REPO_KEY`. The target repository must be present
in `REPO_PHIDS`; the checked-in default targets `autoland`.

At runtime the script:
  - loads `CONDUIT_API_TOKEN` from `secrets/.env`;
  - initializes a Phabricator client for
    `https://phabricator.services.mozilla.com/api/`;
  - converts `YEARS_BACK` into a `createdStart` epoch timestamp;
  - repeatedly calls `differential.revision.search` with the selected repository
    PHID, `order="oldest"`, and the API pagination cursor;
  - accumulates every returned revision, then sorts the complete list by
    `fields.dateCreated`.

The final revision list is converted directly to a pandas DataFrame and written
as CSV under `datasets/mozilla_perf/` using this filename pattern:

    all_differential_revisions_{TARGET_REPO_KEY}_{YEARS_BACK}y.csv

The CSV preserves the raw Conduit revision objects as DataFrame columns, so
nested fields such as `fields` are serialized by pandas rather than flattened.
Adjust `YEARS_BACK` and `TARGET_REPO_KEY` near the top of the file before
running when a different time window or repository is needed.
"""

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

# ==========================
# CONFIG
# ==========================

# How many years back to fetch DREVs from (float or int).
YEARS_BACK = 1  # e.g., 0.5 for ~6 months, 2 for ~2 years

# Which repository to query: "mozilla-central" or "autoland"
TARGET_REPO_KEY = "autoland"

# Fill these with the actual PHIDs:
REPO_PHIDS = {
    "mozilla-central": "PHID-REPO-saax4qdxlbbhahhp2kg5",
    "autoland":        "PHID-REPO-wxrrnneqyw2v3wcqbkfj",
}

# ==========================
# SETUP
# ==========================

REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
secrets_path = os.path.join(REPO_PATH, "secrets", ".env")
load_dotenv(dotenv_path=secrets_path)

API_TOKEN = os.getenv("CONDUIT_API_TOKEN")

BASE_URL = "https://phabricator.services.mozilla.com/api/"

# Initialize Phabricator client
phab = Phabricator(token=API_TOKEN, host=BASE_URL)
phab.update_interfaces()
phab.user.whoami()

if TARGET_REPO_KEY not in REPO_PHIDS:
    raise ValueError(f"Unknown TARGET_REPO_KEY: {TARGET_REPO_KEY}")

TARGET_REPO_PHID = REPO_PHIDS[TARGET_REPO_KEY]
print(f"Using repo: {TARGET_REPO_KEY} (PHID: {TARGET_REPO_PHID})")

LIMIT = 100  # page size

# Time range: last YEARS_BACK years
now = datetime.now()
days_back = int(365 * YEARS_BACK)
start_time = now - timedelta(days=days_back)
start_epoch = int(start_time.timestamp())

print(f"Fetching DREVs from {start_time.isoformat()} (≈{YEARS_BACK} years back) to now")

# ==========================
# FETCH REVISIONS
# ==========================

all_revisions_list = []
after_cursor = None

while True:
    response = phab.differential.revision.search(
        constraints={
            "createdStart": start_epoch,
            "repositoryPHIDs": [TARGET_REPO_PHID],
        },
        order="oldest",
        limit=LIMIT,
        after=after_cursor,
    )

    data = response["data"]
    all_revisions_list.extend(data)

    cursor = response["cursor"]
    if cursor["after"] is None:
        break
    after_cursor = cursor["after"]

# ==========================
# SORT BY CREATION DATE
# ==========================

# Each revision has fields.dateCreated (epoch seconds)
all_revisions_list.sort(key=lambda rev: rev["fields"]["dateCreated"])

# Optional: debug
# for rev in all_revisions_list:
#     created_time = datetime.fromtimestamp(rev['fields']['dateCreated']).strftime('%Y-%m-%d %H:%M:%S')
#     print(f"ID: {rev['id']}, PHID: {rev['phid']}, Title: {rev['fields']['title']}, Created: {created_time}")

# ==========================
# SAVE TO CSV
# ==========================

all_revisions_df = pd.DataFrame(all_revisions_list)
all_diff_revisions_path = os.path.join(
    REPO_PATH,
    "datasets",
    "mozilla_perf",
    f"all_differential_revisions_{TARGET_REPO_KEY}_{YEARS_BACK}y.csv",
)
all_revisions_df.to_csv(all_diff_revisions_path, index=False)

print(f"✅ All {TARGET_REPO_KEY} differential revisions saved to {all_diff_revisions_path}")
