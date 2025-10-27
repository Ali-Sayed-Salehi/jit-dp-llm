#!/usr/bin/env python3
import csv
import json
import os
import re
import subprocess
import sys
from typing import List, Dict, Optional
from datetime import datetime, timezone, timedelta

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COMMITS_JSONL = os.path.join(REPO_ROOT, "datasets", "mozilla_perf", "all_commits.jsonl")
PERF_BUGS_CSV = os.path.join(REPO_ROOT, "datasets", "mozilla_perf", "perf_bugs.csv")
HG_REPO = os.path.join(REPO_ROOT, "data_extraction", "mercurial", "repos", "autoland")
OUT_JSONL = os.path.join(REPO_ROOT, "datasets", "mozilla_perf", "perf_bugs_with_diff.jsonl")

BUG_RE = re.compile(r"^\s*Bug\s+(\d+)\s*[-:]?\s*", re.IGNORECASE)
REVERT_RE = re.compile(r"^\s*(revert|backed out)\b", re.IGNORECASE)
DIFFREV_LINE_RE = re.compile(r"^\s*Differential\s+Revision:\s*\S+\s*$", re.IGNORECASE)

def load_commits(path: str) -> List[Dict]:
    commits: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                commits.append(json.loads(line))
            except Exception:
                pass
    return commits

def bug_id_from_desc(desc: Optional[str]) -> Optional[str]:
    s = desc or ""
    if REVERT_RE.match(s):
        return None
    m = BUG_RE.match(s)
    return m.group(1) if m else None

def clean_desc(desc: str) -> str:
    """Remove 'Bug <id>' prefix and 'Differential Revision: ...' lines from commit message."""
    text = BUG_RE.sub("", desc or "").strip()
    lines = [ln for ln in text.splitlines() if not DIFFREV_LINE_RE.match(ln)]
    return "\n".join(lines).strip()

def run(*args: str, cwd: Optional[str] = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), cwd=cwd, check=check,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def unified_range_diff(oldest_node: str, newest_node: str, node_to_p1: Dict[str, Optional[str]]) -> str:
    """
    Compute unified diff from the FIRST PARENT of oldest_node to newest_node.
    Prefer parent hash from all_commits.jsonl; fall back to Mercurial's p1() if missing.
    """
    p1 = node_to_p1.get(oldest_node)
    if p1:
        cp = run("hg", "diff", "-r", p1, "-r", newest_node, cwd=HG_REPO, check=False)
    else:
        cp = run("hg", "diff", "-r", f"p1({oldest_node})", "-r", newest_node, cwd=HG_REPO, check=False)
    return cp.stdout.decode("utf-8", errors="replace") if cp.stdout else ""

def contiguous_prev_same_bug(commits: List[Dict], start_idx: int, bug_id: str) -> List[str]:
    """From start_idx (newest match), walk backward collecting contiguous commits for the same bug."""
    nodes: List[str] = []
    if bug_id_from_desc(commits[start_idx].get("desc", "")) == bug_id:
        nodes.append(commits[start_idx]["node"])
    j = start_idx - 1
    while j >= 0:
        if bug_id_from_desc(commits[j].get("desc", "")) == bug_id:
            nodes.append(commits[j]["node"])
            j -= 1
        else:
            break
    return nodes  # order: newest -> older ... oldest

def hgdate_to_iso(date_field) -> str:
    """Convert Mercurial 'date' array [epoch_seconds, tz_offset_seconds] to ISO-8601."""
    if not isinstance(date_field, (list, tuple)) or len(date_field) != 2:
        return ""
    epoch, tzoff = date_field
    try:
        tz = timezone(timedelta(seconds=int(tzoff)))
        return datetime.fromtimestamp(float(epoch), tz).isoformat(timespec="seconds")
    except Exception:
        try:
            return datetime.utcfromtimestamp(float(epoch)).replace(tzinfo=timezone.utc).isoformat(timespec="seconds")
        except Exception:
            return ""

def main():
    if not os.path.isdir(HG_REPO):
        print("Mercurial repo missing.", file=sys.stderr)
        sys.exit(1)

    # Ensure output directory exists and truncate output file at start.
    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)

    commits = load_commits(COMMITS_JSONL)
    node_to_idx = {c["node"]: i for i, c in enumerate(commits)}
    node_to_desc = {c["node"]: c.get("desc", "") for c in commits}
    node_to_date = {c["node"]: c.get("date") for c in commits}  # [epoch, tzoff]

    # Build first-parent map from all_commits.jsonl
    node_to_p1: Dict[str, Optional[str]] = {c["node"]: (c.get("parents") or [None])[0] for c in commits}

    # Open once and write each result as it is processed.
    with open(OUT_JSONL, "w", encoding="utf-8") as f_out, open(PERF_BUGS_CSV, "r", encoding="utf-8") as f_in:
        for row in csv.DictReader(f_in):
            bug_id = (row.get("bug_id") or "").strip()
            if not bug_id:
                continue

            is_regressor = str(row.get("bug_is_perf_regressor", "")).lower() == "true"

            newest_idx = None
            if is_regressor:
                regrev = (row.get("regressor_revision") or "").strip()
                if not regrev or regrev not in node_to_idx:
                    print(f"regressor revision {regrev} not found in autoland commits.")
                    continue
                newest_idx = node_to_idx[regrev]
            else:
                idxs = [i for i, c in enumerate(commits) if bug_id_from_desc(c.get("desc", "")) == bug_id]
                if not idxs:
                    continue
                newest_idx = max(idxs)

            seq = contiguous_prev_same_bug(commits, newest_idx, bug_id)
            if not seq:
                continue
            newest, oldest = seq[0], seq[-1]
            block_nodes = list(reversed(seq))  # oldest -> newest

            # unified diff for the contiguous block using first-parent from JSONL
            diff_text = unified_range_diff(oldest, newest, node_to_p1)
            if not diff_text.strip():
                continue

            messages = [clean_desc(node_to_desc.get(n, "")) for n in block_nodes]
            combined_msg = "\n".join(m for m in messages if m).strip()

            last_commit_date_iso = hgdate_to_iso(node_to_date.get(newest))

            record = {
                "bug_id": bug_id,
                "diff": diff_text,
                "commit_message": combined_msg,
                "revision": newest,
                "last_commit_date": last_commit_date_iso,
            }

            f_out.write(json.dumps(record) + "\n")
            f_out.flush()

if __name__ == "__main__":
    main()
