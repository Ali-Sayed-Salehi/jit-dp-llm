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

# - Get all the differential revisions (DREV) from last year until now

REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
secrets_path = os.path.join(REPO_PATH, "secrets", ".env")
load_dotenv(dotenv_path=secrets_path)

API_TOKEN = os.getenv("CONDUIT_API_TOKEN")

BASE_URL = "https://phabricator.services.mozilla.com/api/"

# Initialize Phabricator client
phab = Phabricator(token=API_TOKEN, host=BASE_URL)
phab.update_interfaces()
phab.user.whoami()

MOZILLA_CENTRAL_PHID = "PHID-REPO-saax4qdxlbbhahhp2kg5"
LIMIT = 100 # page size

# Time range: last 7 days
now = datetime.now()
start_time = now - timedelta(days=365)
start_epoch = int(start_time.timestamp())

all_revisions_list = []
after_cursor = None

while True:
    response = phab.differential.revision.search(
        constraints={
            'createdStart':start_epoch,
            'repositoryPHIDs':[MOZILLA_CENTRAL_PHID]
        },
        order= 'oldest',
        limit= LIMIT,
        after= after_cursor
    )

    data = response['data']
    all_revisions_list.extend(data)

    cursor = response['cursor']
    if cursor['after'] is None:
        break
    after_cursor = cursor['after']

# for rev in all_revisions_list:
#     created_time = datetime.fromtimestamp(rev['fields']['dateCreated']).strftime('%Y-%m-%d %H:%M:%S')
#     print(f"ID: {rev['id']}, PHID: {rev['phid']}, Title: {rev['fields']['title']}, Created: {created_time}")

all_revisions_df = pd.DataFrame(all_revisions_list)
all_diff_revisions_path = os.path.join(REPO_PATH, "datasets", "mozilla_perf", "all_differential_revisions.csv")
all_revisions_df.to_csv(all_diff_revisions_path, index=False)

print(f"✅ All differential revisions saved to {all_diff_revisions_path}")
