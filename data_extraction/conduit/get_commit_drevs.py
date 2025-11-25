#!/usr/bin/env python3
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from phabricator import Phabricator

# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------
DEBUG_MODE = True
DEBUG_MAX_COMMITS = 1
MAX_COMMITS_TO_PROCESS = 100_000  # newest commits to consider from all_commits.jsonl

# Rate limiting: minimum time (in seconds) between any two Conduit calls.
# 0.5 => ~2 calls/second. Increase for more conservative behavior.
RATE_LIMIT_MIN_INTERVAL = 0.5

# Global timestamp of last Conduit call (used for simple rate limiting)
_last_conduit_call_time: float = 0.0

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_COMMITS = REPO_ROOT / "datasets" / "mozilla_perf" / "all_commits.jsonl"
OUTPUT_FILE = REPO_ROOT / "datasets" / "mozilla_perf" / "commits_with_drev.csv"

secrets_path = os.path.join(REPO_ROOT, "secrets", ".env")
load_dotenv(dotenv_path=secrets_path)

PHABRICATOR_API_URL = os.getenv(
    "PHABRICATOR_API_URL",
    "https://phabricator.services.mozilla.com/api/",
)
PHABRICATOR_API_TOKEN = os.getenv("CONDUIT_API_TOKEN")

# Matches: ... https://phabricator.services.mozilla.com/D206595 at the *end* of desc
DREV_URL_RE = re.compile(
    r"https://phabricator\.services\.mozilla\.com/D(\d+)\s*\Z",
    re.IGNORECASE,
)

# Rate limiting / retry settings
MAX_RETRIES = 5
RETRY_BASE_SLEEP_SECONDS = 5

# --------------------------------------------------------------------
# Phabricator client
# --------------------------------------------------------------------

_PHAB: Optional[Phabricator] = None


def get_phab() -> Phabricator:
    global _PHAB
    if _PHAB is None:
        if not PHABRICATOR_API_TOKEN:
            raise RuntimeError("CONDUIT_API_TOKEN is not set in environment or .env")
        # phabricator client expects the API endpoint as host
        _PHAB = Phabricator(host=PHABRICATOR_API_URL, token=PHABRICATOR_API_TOKEN)
        _PHAB.update_interfaces()
        _PHAB.user.whoami()
        _PHAB.timeout = 30
    return _PHAB


# --------------------------------------------------------------------
# Generic retry wrapper for Conduit calls (handles 429)
# --------------------------------------------------------------------

def call_with_retry(method, **kwargs) -> Any:
    """
    Call a Phabricator method with:
      - a simple global rate limit (min interval between calls)
      - basic retry-on-429 behavior.
    """
    global _last_conduit_call_time

    for attempt in range(MAX_RETRIES):
        # Enforce minimum interval between calls
        now = time.time()
        elapsed = now - _last_conduit_call_time
        if elapsed < RATE_LIMIT_MIN_INTERVAL:
            time.sleep(RATE_LIMIT_MIN_INTERVAL - elapsed)

        try:
            result = method(**kwargs)
            _last_conduit_call_time = time.time()
            return result
        except Exception as e:
            msg = str(e)
            # crude but effective check for rate limiting
            if "429" in msg or "Too Many Requests" in msg:
                wait = RETRY_BASE_SLEEP_SECONDS * (attempt + 1)
                print(
                    f"Received 429 from Phabricator (attempt {attempt + 1}/{MAX_RETRIES}), "
                    f"sleeping {wait} seconds before retry...",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            # other errors: re-raise immediately
            raise

    raise RuntimeError(
        f"Exceeded max retries ({MAX_RETRIES}) due to repeated 429 errors for method {method}."
    )



# --------------------------------------------------------------------
# Conduit helpers (via phabricator client)
# --------------------------------------------------------------------

def get_revision_by_id(drev_id: int) -> Optional[Dict[str, Any]]:
    phab = get_phab()
    result = call_with_retry(
        phab.differential.revision.search,
        constraints={"ids": [drev_id]},
        limit=1,
    )
    data = result.get("data", [])
    return data[0] if data else None


def get_diffs_for_revision(revision_phid: str) -> List[Dict[str, Any]]:
    phab = get_phab()
    result = call_with_retry(
        phab.differential.diff.search,
        constraints={"revisionPHIDs": [revision_phid]},
        limit=100,
    )
    diffs = result.get("data", [])
    if not diffs:
        raise RuntimeError(f"Revision {revision_phid} has no diffs.")
    return diffs


def get_transactions_for_revision(revision_phid: str) -> List[Dict[str, Any]]:
    """
    Fetch all transactions for a revision. Raises if none are returned.
    """
    phab = get_phab()
    result = call_with_retry(
        phab.transaction.search,
        objectIdentifier=revision_phid,
        limit=100,
    )
    transactions = result.get("data", [])

    if not transactions:
        raise RuntimeError(
            f"transaction.search returned no transactions for revision {revision_phid}."
        )

    return transactions


def extract_comments_from_transactions(transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter the given transactions to comment-like ones.
    """
    comment_types = {"comment", "inline"}
    comments = [
        t for t in transactions
        if t.get("type") in comment_types
    ]
    return comments


def get_first_close_timestamp_from_transactions(
    transactions: List[Dict[str, Any]],
    revision_phid: str,
) -> int:
    """
    Return the timestamp (dateCreated) of the *first* status change
    that moves the revision into a closed/published state.

    Raises RuntimeError if no such status change is found.
    """
    close_timestamps: List[int] = []

    for t in transactions:
        if t.get("type") != "status":
            continue

        fields = t.get("fields", {})
        new_status = fields.get("new")

        # Heuristic: treat transition to "closed" or "published" as closure
        if new_status in {"closed", "published"}:
            ts = int(t.get("dateCreated", 0))
            if ts:
                close_timestamps.append(ts)

    if not close_timestamps:
        raise RuntimeError(f"No close_timestamps found for revision {revision_phid}.")

    return min(close_timestamps)


# --------------------------------------------------------------------
# Logic for has_change_inducing_review and PR lead time
# --------------------------------------------------------------------

def compute_has_change_inducing_review(
    diffs: List[Dict[str, Any]],
    comments: List[Dict[str, Any]],
) -> bool:
    """
    Return True if there exists any non-final diff whose timestamp is later than the
    *earliest* review comment timestamp.
    """
    if not diffs:
        raise RuntimeError("diffs cannot be empty in compute_has_change_inducing_review.")
    if not comments:
        return False

    # Extract timestamps
    def get_diff_ts(d: Dict[str, Any]) -> int:
        return int(d.get("fields", {}).get("dateCreated", 0))

    def get_comment_ts(c: Dict[str, Any]) -> int:
        return int(c.get("dateCreated", 0))

    # Final diff = the one with the latest dateCreated
    diffs_sorted = sorted(diffs, key=get_diff_ts)
    final_diff = diffs_sorted[-1]
    final_diff_id = final_diff.get("id")

    # Use *earliest* comment timestamp
    earliest_comment_ts = min(get_comment_ts(c) for c in comments)

    # Excluding the final diff, check if any diff comes after the earliest review comment
    for d in diffs_sorted:
        if d.get("id") == final_diff_id:
            continue
        if get_diff_ts(d) > earliest_comment_ts:
            return True

    return False


def compute_pr_lead_time_hours(revision: Dict[str, Any], diffs: List[Dict[str, Any]]) -> Optional[float]:
    if not revision or not diffs:
        raise RuntimeError("revision and diffs cannot be None.")

    rev_fields = revision.get("fields", {})
    rev_created = int(rev_fields.get("dateCreated", 0))
    if rev_created == 0:
        raise RuntimeError("revision dateCreated cannot be zero.")

    # Final diff = diff with the latest dateCreated (assumed to be the landed one)
    def get_diff_ts(d: Dict[str, Any]) -> int:
        return int(d.get("fields", {}).get("dateCreated", 0))

    final_diff = max(diffs, key=get_diff_ts)
    landed_ts = get_diff_ts(final_diff)
    if landed_ts <= 0:
        raise RuntimeError("landed_ts <= 0.")

    # Convert seconds to hours
    delta_sec = max(0, landed_ts - rev_created)
    return delta_sec / 3600.0


# --------------------------------------------------------------------
# Main processing
# --------------------------------------------------------------------

def extract_drev_id_from_desc(desc: str) -> Optional[int]:
    if not desc:
        raise RuntimeError("Commit desc cannot be empty.")

    # Look for the full Differential Revision URL and capture the numeric ID
    m = DREV_URL_RE.search(desc)
    if not m:
        return None

    drev_id_str = m.group(1)  # numeric part after /D
    try:
        return int(drev_id_str)
    except ValueError:
        raise RuntimeError(f"Could not parse drev id from URL in desc: {desc!r}")

def load_commits_sorted_by_date() -> List[Dict[str, Any]]:
    """
    Load all commits from INPUT_COMMITS, sort them by their Mercurial timestamp
    (date[0]), and return the list sorted from oldest to newest.
    Commits with missing/invalid dates are treated as having timestamp 0.
    """
    commits_with_ts: List[tuple[int, Dict[str, Any]]] = []

    with INPUT_COMMITS.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                commit = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line[:80]}...", file=sys.stderr)
                continue

            # Extract Mercurial timestamp: date[0]
            ts = 0
            date = commit.get("date")
            if isinstance(date, list) and date:
                try:
                    ts = int(date[0])
                except (TypeError, ValueError):
                    ts = 0  # fall back to 0 for bad/missing timestamps

            commits_with_ts.append((ts, commit))

    # Sort by timestamp ascending (oldest first)
    commits_with_ts.sort(key=lambda x: x[0])

    # Drop the timestamp, keep only commit dicts
    sorted_commits = [c for _, c in commits_with_ts]
    return sorted_commits


def process_commit(commit: Dict[str, Any]) -> Dict[str, Any]:
    node = commit.get("node")
    desc = commit.get("desc")
    date = commit.get("date")

    drev_id = extract_drev_id_from_desc(desc)
    has_change_inducing_review = None
    pr_lead_time_hours = None

    if drev_id is None:
        return None

    try:
        revision = get_revision_by_id(drev_id)
    except Exception as e:
        print(f"Error fetching revision for D{drev_id}: {e}", file=sys.stderr)
        revision = None

    if revision is not None:
        fields = revision.get("fields", {})
        status = fields.get("status", {})
        status_value = status.get("value")
        status_closed = bool(status.get("closed", False))

        if status_value == "published" and status_closed:
            rev_phid = revision.get("phid")
            try:
                diffs = get_diffs_for_revision(rev_phid)

                # Single transaction.search call:
                transactions = get_transactions_for_revision(rev_phid)
                comments = extract_comments_from_transactions(transactions)
                first_close_ts = get_first_close_timestamp_from_transactions(
                    transactions, rev_phid
                )

                # Restrict diffs and comments to what happened *on or before* first closure.
                def diff_ts(d: Dict[str, Any]) -> int:
                    return int(d.get("fields", {}).get("dateCreated", 0))

                diffs_before_close = [d for d in diffs if diff_ts(d) <= first_close_ts]
                if diffs_before_close:
                    diffs = diffs_before_close
                else:
                    # Fallback: if filtering somehow removed all diffs, keep them all
                    print(
                        f"Warning: no diffs before first closure for D{drev_id}, "
                        "using all diffs instead.",
                        file=sys.stderr,
                    )

                comments = [
                    c for c in comments
                    if int(c.get("dateCreated", 0)) <= first_close_ts
                ]

                has_change_inducing_review = compute_has_change_inducing_review(diffs, comments)
                pr_lead_time_hours = compute_pr_lead_time_hours(revision, diffs)

            except Exception as e:
                print(f"Error fetching/processing diffs/transactions for D{drev_id}: {e}", file=sys.stderr)

    result_row = {
        "node": node,
        "desc": desc,
        "date": date,
        "drev_id": drev_id,
        "has_change_inducing_review": has_change_inducing_review,
        "pr_lead_time_hours": pr_lead_time_hours,
    }
    return result_row


def main() -> None:
    if not INPUT_COMMITS.exists():
        print(f"Input file not found: {INPUT_COMMITS}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load all commits and sort by date[0] (Mercurial timestamp)
    all_commits_sorted = load_commits_sorted_by_date()
    total_commits = len(all_commits_sorted)

    # Keep only the newest MAX_COMMITS_TO_PROCESS commits (by date)
    if MAX_COMMITS_TO_PROCESS is not None:
        if total_commits > MAX_COMMITS_TO_PROCESS:
            commits_to_process = all_commits_sorted[-MAX_COMMITS_TO_PROCESS:]
        else:
            commits_to_process = all_commits_sorted
    else:
        commits_to_process = all_commits_sorted

    print(
        f"Total commits in file: {total_commits}. "
        f"Processing {len(commits_to_process)} newest commits by date.",
        file=sys.stderr,
    )

    processed_commits = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as fout:
        for commit in commits_to_process:
            processed_commits += 1
            if DEBUG_MODE and processed_commits > DEBUG_MAX_COMMITS:
                print(
                    f"DEBUG_MODE is ON: reached {DEBUG_MAX_COMMITS} commits, stopping.",
                    file=sys.stderr,
                )
                break

            row = process_commit(commit)
            if row is not None:
                fout.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
