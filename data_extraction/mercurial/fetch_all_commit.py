#!/usr/bin/env python3
"""Fetch autoland commit metadata and export it as JSONL.

Flow:
  1. Ensure a local clone of Mozilla’s `autoland` Mercurial repo exists under
     `data_extraction/mercurial/repos/autoland` (clone if missing; otherwise `hg pull -u`).
  2. Export all changesets via `hg log -Tjson -r all()` and write one JSON object per line.

Inputs:
  - Mercurial remote: `https://hg-edge.mozilla.org/integration/autoland`
  - Local checkout: `data_extraction/mercurial/repos/autoland`

Outputs:
  - `datasets/mozilla_perf/all_commits.jsonl`
    Each line contains (at minimum): `node`, `desc`, `date`, `parents`.

Notes:
  - Output ordering follows whatever `hg log -r all()` returns (not explicitly sorted by date).
"""
import os
import sys
import csv
import subprocess
import json

# ======= CONSTANTS (edit these if needed) =======
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
AUTOLAND_URL = "https://hg-edge.mozilla.org/integration/autoland"
DEST_ROOT = os.path.join(REPO_ROOT, "data_extraction", "mercurial", "repos")
DEST_REPO = os.path.join(DEST_ROOT, "autoland")
OUT_JSONL = os.path.join(REPO_ROOT, "datasets", "mozilla_perf", "all_commits.jsonl")
# ================================================

def run(cmd, cwd=None, stdout=None):
    try:
        subprocess.run(cmd, cwd=cwd, check=True, stdout=stdout)
    except FileNotFoundError:
        print("Error: 'hg' not found. Install Mercurial and ensure it's on PATH.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(e.returncode)

def main():

    # Ensure destination directory exists
    os.makedirs(DEST_ROOT, exist_ok=True)

    # Clone or update autoland
    if not os.path.isdir(DEST_REPO):
        print(f"Cloning Autoland into: {DEST_REPO}")
        run(["hg", "clone", AUTOLAND_URL, DEST_REPO])
    else:
        print(f"Repo exists at {DEST_REPO}; pulling updates.")
        run(["hg", "pull", "-u"], cwd=DEST_REPO)

    # Export all commits (hash + message) as JSONL
    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    print(f"Writing all commits (JSONL) to {OUT_JSONL} ...")

    # Get structured JSON from Mercurial and convert to JSONL
    result = subprocess.run(
        ["hg", "log", "-Tjson", "-r", "all()"],
        cwd=DEST_REPO,
        check=True,
        stdout=subprocess.PIPE,
    )
    changesets = json.loads(result.stdout.decode("utf-8"))

    with open(OUT_JSONL, "w", encoding="utf-8") as out:
        for cs in changesets:
            json.dump({
                "node": cs.get("node"), 
                "desc": cs.get("desc", ""),
                "date": cs.get("date", ""),
                "parents": cs.get("parents", "")
                }, out, ensure_ascii=False)
            out.write("\n")

    print("Done.")

if __name__ == "__main__":
    main()
