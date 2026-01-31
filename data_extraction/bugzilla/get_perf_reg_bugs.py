#!/usr/bin/env python3
"""
Extract Treeherder-filed performance regression bugs from Bugzilla.

This script targets bugs created by the "treeherder Bug Filer" account and attempts to parse a
regressor revision from the filing comment text (a 40-hex SHA1-like string).

Flow:
  1. Query Bugzilla for bugs where `creator == "treeherder Bug Filer"` (paged).
  2. For each bug id, fetch comments and locate the Treeherder regression template comment.
  3. Extract the first 40-hex SHA1 from the qualifying comment text.
  4. Write one CSV row per bug where a regressor revision is found.

Inputs:
  - Bugzilla REST API:
    - `https://bugzilla.mozilla.org/rest/bug`
    - `https://bugzilla.mozilla.org/rest/bug/<id>/comment`
  - `secrets/.env` (optional): `BUGZILLA_API_KEY`

Outputs (CSV, default):
  - `datasets/mozilla_perf/perf_reg_bugs.csv`
    Columns: `regression_bug_id`, `regressor_push_head_revision`, `creation_time`

Notes:
  - The script has an in-code `DEBUG` flag that limits fetching to a single page.
"""

import os
import csv
import re
import requests
from dotenv import load_dotenv

# ---------- Config ----------
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_CSV = os.path.join(REPO_PATH, "datasets", "mozilla_perf", "perf_reg_bugs.csv")

# Credentials and base URL
secrets_path = os.path.join(REPO_PATH, "secrets", ".env")
load_dotenv(dotenv_path=secrets_path)
API_KEY = os.getenv("BUGZILLA_API_KEY")
BUGZILLA_API_URL = "https://bugzilla.mozilla.org/rest"

# Debug mode: only fetch 10 rows
DEBUG = True

# Filter settings
CREATOR_REALNAME = "treeherder Bug Filer"
PERF_REGR_PREFIX = "Perfherder has detected a browsertime performance regression from push "
SHA1_RE = re.compile(r"\b[0-9a-f]{40}\b")

# ---------- Helpers ----------
def get(url, params=None):
    params = dict(params or {})
    if API_KEY:
        params["api_key"] = API_KEY
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def iter_th_bugs():
    """Yield all bugs filed by Treeherder Bug Filer."""
    limit = 50 if DEBUG else 150
    offset = 0
    fields = "id,creation_time"
    url = f"{BUGZILLA_API_URL}/bug"
    while True:
        data = get(url, params={
            "creator": CREATOR_REALNAME,
            "include_fields": fields,
            "limit": limit,
            "offset": offset
        })
        bugs = data.get("bugs", [])
        if not bugs:
            break
        for b in bugs:
            yield b
        offset += len(bugs)
        if len(bugs) < limit or DEBUG:  # stop after one page in debug mode
            break

def extract_regressor_sha(bug_id: int):
    """Extract SHA1 of regressor from qualifying comment."""
    data = get(f"{BUGZILLA_API_URL}/bug/{bug_id}/comment")
    comments = data.get("bugs", {}).get(str(bug_id), {}).get("comments", [])
    for c in comments:
        if c.get("creator_detail", {}).get("real_name") != CREATOR_REALNAME:
            continue
        text = c.get("text", "")
        if not text.startswith(PERF_REGR_PREFIX):
            continue
        if "performance regression" not in text.lower():
            continue
        m = SHA1_RE.search(text)
        if m:
            return m.group(0)
    return None

# ---------- Main ----------
def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
                "regression_bug_id", 
                "regressor_push_head_revision", 
                "creation_time"
            ])

        for bug in iter_th_bugs():
            bug_id = bug["id"]
            created = bug.get("creation_time")
            sha = extract_regressor_sha(bug_id)
            if sha:
                w.writerow([
                        bug_id, 
                        sha,
                        created
                    ])

if __name__ == "__main__":
    main()
