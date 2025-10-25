#!/usr/bin/env python3
import os
import sys
import csv
import subprocess
import json  # added

# ======= CONSTANTS (edit these if needed) =======
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH = os.path.join(REPO_ROOT, "datasets", "mozilla_perf", "perf_bugs.csv")
AUTOLAND_URL = "https://hg-edge.mozilla.org/integration/autoland"
DEST_ROOT = os.path.join(REPO_ROOT, "data_extraction", "mercurial", "repos")
DEST_REPO = os.path.join(DEST_ROOT, "autoland")
OUT_JSONL = os.path.join(REPO_ROOT, "datasets", "mozilla_perf", "autoland_commits.jsonl")
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
    # Minimal CSV read (just to ensure it exists and is readable)
    if not os.path.isfile(CSV_PATH):
        print(f"Error: CSV not found at {CSV_PATH}", file=sys.stderr)
        sys.exit(1)
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        _ = csv.DictReader(f)
    print("Loaded perf_bugs.csv")

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
            json.dump({"hash": cs.get("node"), "message": cs.get("desc", "")}, out, ensure_ascii=False)
            out.write("\n")

    print("Done.")

if __name__ == "__main__":
    main()
