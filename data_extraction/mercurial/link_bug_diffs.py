#!/usr/bin/env python3
"""Link Bugzilla bugs to Autoland code changes (net diff per bug).

This script produces a dataset that joins:
1) Bugzilla bug metadata (including dependency relations), and
2) The actual patch content that landed in Mozilla's Autoland Mercurial repo.

It is intentionally modeled after `data_extraction/mercurial/get_bug_diffs.py`, but targets the
JIT dataset inputs (`mozilla_jit/*`) and uses only the "newest contiguous block of commits for a
bug" rule (no perf-regressor-specific logic).

--------
Inputs
--------
All inputs have CLI overrides; defaults are:

- `--commits-jsonl`
  `/speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_jit/all_commits.jsonl`

  JSONL of Autoland commits. Each line is a dict like:
  `{"node": "<rev>", "desc": "<commit message>", "date": [<epoch>, <tzoff>], "parents": ["<p1>", ...]}`

  Notes:
  - The file order is assumed to match Mercurial history order as exported by the project.
  - We treat `parents[0]` as the "first parent" (p1) for stable diff base selection.

- `--bugs-jsonl`
  `/speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_jit/all_bugs.jsonl`

  JSONL of Bugzilla bugs. Each line is a dict like:
  `{"id": 123, "regressed_by": [..], "regressions": [..], "creation_time": "...", ...}`

- `--hg-repo`
  `data_extraction/mercurial/repos/autoland`

  Local Mercurial checkout used for `hg diff` calls.

---------
Linking
---------
For each bug id from `all_bugs.jsonl`, we locate the newest contiguous block of commits in
`all_commits.jsonl` whose commit message starts with `Bug <id>` (case-insensitive).

- "Contiguous block" means: starting from the newest matching commit, walk backwards in the
  commit list while commits continue to match the same bug id; stop at the first non-matching
  commit. This yields a block `[oldest, ..., newest]`.
- Backouts/reverts are ignored: commit messages starting with `Revert` or `Backed out` are
  skipped entirely and never considered bug-linked.
- Bugs that have *more than one* such contiguous block anywhere in history are ignored (skipped),
  and their bug id is logged to stderr.

This produces exactly one diff per bug (if any matching commit block exists).

--------------
Diff strategy
--------------
For a block with `oldest` and `newest` commits, we compute the net code change introduced by the
block as a single unified diff between:

- Base: the *first parent* of `oldest` (`parents[0]` from the commits JSONL when present; else
  Mercurial expression `p1(<oldest>)`), and
- Head: `newest`.

Command:
  `hg diff -r <base> -r <newest>`

This avoids concatenating per-commit diffs and instead captures the net tree change for the block.

-------
Output
-------
Writes JSONL to `--out-jsonl` (default: `datasets/mozilla_jit/mozilla_jit.jsonl`), one record per
bug with a non-empty computed diff:

- `bug_id` (string)
- `regressed_by` (list[str]) and `regressions` (list[str]) copied from Bugzilla, but filtered to
  contain only ids that exist in `all_bugs.jsonl`
- `diff` (string): unified diff text
- `commit_message` (string): concatenation of the block's commit messages with the leading
  `Bug <id>` removed and `Differential Revision:` lines stripped
- `revision` (string): the newest commit node in the block
- `last_commit_date` (string): ISO timestamp derived from Mercurial `date` field of `revision`
- `bug_creation_time` (string): Bugzilla `creation_time` normalized so `Z` becomes `+00:00`
- `regression` (bool): true iff `regressions` is non-empty
- `regressor` (bool): true iff `regressions` is non-empty

---------
Controls
---------
- `--dry-run`: stop after writing 100 output records
- `--limit N`: stop after writing N output records (overrides `--dry-run`)
- `--cutoff-date`: skip bugs created before this ISO date/datetime (e.g. `2023-01-01` or
  `2023-01-01T00:00:00+00:00`)

---------
Notes
---------
- Since some buigs might be skipped due to various reasons, 
  the regression and regressed_by fields in the output might have dangling references. 
  For historical simulations, make sure to not rely on regression and regressor boolean fields and double check those values. 
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DEFAULT_COMMITS_JSONL = os.path.join(REPO_ROOT, "datasets", "mozilla_jit", "all_commits.jsonl")
DEFAULT_BUGS_JSONL = os.path.join(REPO_ROOT, "datasets", "mozilla_jit", "all_bugs.jsonl")
DEFAULT_HG_REPO = os.path.join(REPO_ROOT, "data_extraction", "mercurial", "repos", "autoland")
DEFAULT_OUT_JSONL = os.path.join(REPO_ROOT, "datasets", "mozilla_jit", "mozilla_jit.jsonl")

BUG_RE = re.compile(r"^\s*Bug\s+(\d+)\s*[-:]?\s*", re.IGNORECASE)
REVERT_RE = re.compile(r"^\s*(revert|backed out)\b", re.IGNORECASE)
DIFFREV_LINE_RE = re.compile(r"^\s*Differential\s+Revision:\s*\S+\s*$", re.IGNORECASE)


def run(*args: str, cwd: Optional[str] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess command and capture its output.

    Args:
        *args: Command and arguments, passed directly to `subprocess.run`.
        cwd: Optional working directory for the command.
        check: If True, raise `CalledProcessError` for non-zero exit codes.

    Returns:
        The `subprocess.CompletedProcess` result with `stdout`/`stderr` captured as bytes.
    """
    return subprocess.run(
        list(args),
        cwd=cwd,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file into a list of dicts.

    Lines that are empty, invalid JSON, or not JSON objects are skipped.
    """
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def bug_id_from_desc(desc: Optional[str]) -> Optional[str]:
    """Extract a Bugzilla bug id from the start of a commit message.

    Returns None for backouts/reverts (messages starting with "Revert" or "Backed out") and for
    messages that don't begin with a `Bug <id>` prefix.
    """
    s = desc or ""
    if REVERT_RE.match(s):
        return None
    m = BUG_RE.match(s)
    return m.group(1) if m else None


def clean_desc(desc: str) -> str:
    """Normalize a commit message for the dataset.

    Removes the leading `Bug <id>` prefix and drops any `Differential Revision:` lines.
    """
    text = BUG_RE.sub("", desc or "").strip()
    lines = [ln for ln in text.splitlines() if not DIFFREV_LINE_RE.match(ln)]
    return "\n".join(lines).strip()


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
            return (
                datetime.utcfromtimestamp(float(epoch))
                .replace(tzinfo=timezone.utc)
                .isoformat(timespec="seconds")
            )
        except Exception:
            return ""


def normalize_bugzilla_time(s: Optional[str]) -> str:
    """Normalize Bugzilla timestamps into an ISO string with timezone when possible."""
    if not s:
        return ""
    text = str(s).strip()
    if not text:
        return ""
    if text.endswith("Z"):
        return text[:-1] + "+00:00"
    return text


def parse_iso_datetime(s: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 date/datetime string into a timezone-aware datetime.

    Accepts either:
    - Date: `YYYY-MM-DD` (interpreted as midnight UTC), or
    - Datetime: `YYYY-MM-DDTHH:MM:SS[.ffffff][±HH:MM]` (timezone assumed UTC if missing).

    Returns None if parsing fails.
    """
    if not s:
        return None
    text = str(s).strip()
    if not text:
        return None

    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        try:
            dt = datetime.fromisoformat(text + "T00:00:00")
        except Exception:
            return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def contiguous_prev_same_bug(commits: Sequence[Dict], start_idx: int, bug_id: str) -> List[str]:
    """From start_idx (newest match), walk backward collecting contiguous commits for the same bug."""
    nodes: List[str] = []
    if bug_id_from_desc(commits[start_idx].get("desc", "")) != bug_id:
        return nodes
    nodes.append(commits[start_idx]["node"])
    j = start_idx - 1
    while j >= 0:
        if bug_id_from_desc(commits[j].get("desc", "")) == bug_id:
            nodes.append(commits[j]["node"])
            j -= 1
        else:
            break
    return nodes  # newest -> ... -> oldest


def unified_range_diff(
    hg_repo: str, oldest_node: str, newest_node: str, node_to_p1: Dict[str, Optional[str]]
) -> str:
    """
    Compute unified diff from the FIRST PARENT of oldest_node to newest_node.
    Prefer parent hash from all_commits.jsonl; fall back to Mercurial's p1() if missing.
    """
    p1 = node_to_p1.get(oldest_node)
    if p1:
        cp = run("hg", "diff", "-r", p1, "-r", newest_node, cwd=hg_repo, check=False)
    else:
        cp = run("hg", "diff", "-r", f"p1({oldest_node})", "-r", newest_node, cwd=hg_repo, check=False)
    return cp.stdout.decode("utf-8", errors="replace") if cp.stdout else ""


def build_bug_to_newest_idx(commits: Sequence[Dict]) -> Dict[str, int]:
    """Build a mapping from bug id to its newest matching commit index."""
    bug_to_newest_idx: Dict[str, int] = {}
    for i, c in enumerate(commits):
        bug_id = bug_id_from_desc(c.get("desc", ""))
        if bug_id:
            bug_to_newest_idx[bug_id] = i
    return bug_to_newest_idx


def build_bug_to_earliest_idx(commits: Sequence[Dict]) -> Dict[str, int]:
    """Build a mapping from bug id to its earliest matching commit index."""
    bug_to_earliest_idx: Dict[str, int] = {}
    for i, c in enumerate(commits):
        bug_id = bug_id_from_desc(c.get("desc", ""))
        if bug_id and bug_id not in bug_to_earliest_idx:
            bug_to_earliest_idx[bug_id] = i
    return bug_to_earliest_idx


def filter_known_bug_ids(ids: Sequence, known_bug_ids: set[str]) -> List[str]:
    """Filter/normalize bug id values to known bug ids.

    Coerces each entry to `str`, trims whitespace, and keeps only ids present in `known_bug_ids`.
    """
    out: List[str] = []
    for x in ids or []:
        s = str(x).strip()
        if s and s in known_bug_ids:
            out.append(s)
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional argv sequence (defaults to `sys.argv[1:]` when None).
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--commits-jsonl", default=DEFAULT_COMMITS_JSONL, help="Path to all_commits.jsonl")
    p.add_argument("--bugs-jsonl", default=DEFAULT_BUGS_JSONL, help="Path to all_bugs.jsonl")
    p.add_argument("--hg-repo", default=DEFAULT_HG_REPO, help="Path to local Mercurial repo (Autoland)")
    p.add_argument("--out-jsonl", default=DEFAULT_OUT_JSONL, help="Output JSONL path")
    p.add_argument("--dry-run", action="store_true", help="Write only 100 output rows")
    p.add_argument("--limit", type=int, default=0, help="Write only N rows (overrides --dry-run)")
    p.add_argument(
        "--cutoff-date",
        default="",
        help="Skip bugs created before this ISO date/datetime (e.g. 2023-01-01 or 2023-01-01T00:00:00+00:00)",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Link bugs to diffs and write the joined JSONL dataset.

    Loads commits and bugs from JSONL, selects at most one newest contiguous commit block per bug,
    computes a net `hg diff` for that block, and writes one output JSON object per bug.
    """
    args = parse_args(argv)

    if not os.path.isdir(args.hg_repo):
        print(f"Mercurial repo missing: {args.hg_repo}", file=sys.stderr)
        return 1

    cutoff_dt = None
    if args.cutoff_date:
        cutoff_dt = parse_iso_datetime(normalize_bugzilla_time(args.cutoff_date))
        if cutoff_dt is None:
            print(f"Invalid --cutoff-date: {args.cutoff_date}", file=sys.stderr)
            return 2

    commits = load_jsonl(args.commits_jsonl)
    bugs = load_jsonl(args.bugs_jsonl)
    if not commits:
        print(f"No commits loaded from: {args.commits_jsonl}", file=sys.stderr)
        return 1
    if not bugs:
        print(f"No bugs loaded from: {args.bugs_jsonl}", file=sys.stderr)
        return 1

    node_to_desc = {c.get("node"): c.get("desc", "") for c in commits if c.get("node")}
    node_to_date = {c.get("node"): c.get("date") for c in commits if c.get("node")}
    node_to_p1: Dict[str, Optional[str]] = {
        c["node"]: (c.get("parents") or [None])[0] for c in commits if c.get("node")
    }

    known_bug_ids = {str(b.get("id")).strip() for b in bugs if str(b.get("id", "")).strip()}
    bug_to_newest_idx = build_bug_to_newest_idx(commits)
    bug_to_earliest_idx = build_bug_to_earliest_idx(commits)

    limit = args.limit if args.limit and args.limit > 0 else (100 if args.dry_run else 0)

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    written = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as f_out:
        for bug in bugs:
            bug_id = str(bug.get("id", "")).strip()
            if not bug_id:
                continue

            if cutoff_dt is not None:
                created = parse_iso_datetime(normalize_bugzilla_time(bug.get("creation_time")))
                if created is not None and created < cutoff_dt:
                    continue

            newest_idx = bug_to_newest_idx.get(bug_id)
            if newest_idx is None:
                continue

            seq = contiguous_prev_same_bug(commits, newest_idx, bug_id)
            if not seq:
                continue
            oldest_idx = newest_idx - (len(seq) - 1)
            earliest_idx = bug_to_earliest_idx.get(bug_id)
            # if earliest_idx is not None and earliest_idx < oldest_idx:
            #     if bug.get("regressions", []):
            #         print(f"Skipping regressor bug {bug_id}: multiple contiguous commit blocks", file=sys.stderr)
            #     continue
            newest_node, oldest_node = seq[0], seq[-1]
            block_nodes = list(reversed(seq))  # oldest -> newest

            diff_text = unified_range_diff(args.hg_repo, oldest_node, newest_node, node_to_p1)
            if not diff_text.strip():
                continue

            messages = [clean_desc(node_to_desc.get(n, "")) for n in block_nodes]
            combined_msg = "\n".join(m for m in messages if m).strip()

            regressed_by = filter_known_bug_ids(bug.get("regressed_by", []) or [], known_bug_ids)
            regressions = filter_known_bug_ids(bug.get("regressions", []) or [], known_bug_ids)

            record = {
                "bug_id": bug_id,
                "regressed_by": regressed_by,
                "regressions": regressions,
                "diff": diff_text,
                "commit_message": combined_msg,
                "revision": newest_node,
                "last_commit_date": hgdate_to_iso(node_to_date.get(newest_node)),
                "bug_creation_time": normalize_bugzilla_time(bug.get("creation_time")),
                "regression": bool(regressed_by),
                "regressor": bool(regressions),
            }

            f_out.write(json.dumps(record) + "\n")
            written += 1
            if limit and written >= limit:
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
