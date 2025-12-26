#!/usr/bin/env python3
"""
Fetches bugs from Bugzilla (bugzilla.mozilla.org) via the REST API and writes them to a JSONL file
(`datasets/mozilla_jit/all_bugs.jsonl`) with a fixed set of fields. Supports pagination and a
`--dry-run` mode to fetch only a single page. The API query is configured to only fetch bugs with
resolution in {FIXED, WONTFIX} and exclude bugs with classification "Graveyard".
"""

import argparse
import json
import os
import random
import time
from typing import Any

import requests
from dotenv import load_dotenv
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_JSONL = os.path.join(REPO_PATH, "datasets", "mozilla_jit", "all_bugs.jsonl")

secrets_path = os.path.join(REPO_PATH, "secrets", ".env")
load_dotenv(dotenv_path=secrets_path)

API_KEY = os.getenv("BUGZILLA_API_KEY")
BUGZILLA_API_URL = "https://bugzilla.mozilla.org/rest"
BUGZILLA_API = f"{BUGZILLA_API_URL}/bug"

FIELDS = [
    "id",
    "type",
    "regressed_by",
    "regressions",
    "resolution",
    "product",
    "component",
    "creation_time",
]

ALLOWED_RESOLUTIONS = ["FIXED", "WONTFIX"]
EXCLUDED_CLASSIFICATION = "Graveyard"


def _build_session(retries: int, backoff: float) -> Session:
    retry_kwargs: dict[str, Any] = {
        "total": retries,
        "connect": retries,
        "read": retries,
        "status": retries,
        "backoff_factor": backoff,
        "status_forcelist": (408, 425, 429, 500, 502, 503, 504),
        "raise_on_status": False,
        "respect_retry_after_header": True,
    }
    try:
        retry = Retry(allowed_methods=frozenset({"GET"}), **retry_kwargs)
    except TypeError:
        retry = Retry(method_whitelist=frozenset({"GET"}), **retry_kwargs)
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _get_json(session: Session, url: str, params: dict, timeout_s: float) -> dict[str, Any]:
    params = dict(params)
    if API_KEY:
        params["api_key"] = API_KEY
    r = session.get(url, params=params, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def _normalize_row(bug: dict) -> dict:
    row = {k: bug.get(k) for k in FIELDS}
    return row


def _count_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch all Bugzilla bugs to a JSONL file.")
    parser.add_argument(
        "--out",
        default=OUT_JSONL,
        help="Output JSONL path (default: datasets/mozilla_jit/all_bugs.jsonl)",
    )
    parser.add_argument("--limit", type=int, default=150, help="Page size (default: 150)")
    parser.add_argument(
        "--timeout",
        type=float,
        default=60,
        help="Request timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=8,
        help="Retry count for transient errors (default: 8)",
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=1.0,
        help="Exponential backoff factor for retries (default: 1.0)",
    )
    parser.add_argument(
        "--since",
        default="1990-01-01T00:00:00Z",
        help="Only fetch bugs created after this ISO timestamp (default: 1990-01-01T00:00:00Z)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch only one page of bugs and exit.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to existing output and continue from its line count as offset.",
    )
    parser.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="Start pagination at this offset (default: 0)",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    session = _build_session(retries=args.retries, backoff=args.backoff)

    offset = args.start_offset
    if args.resume and os.path.exists(args.out):
        offset = _count_lines(args.out)

    total = offset if offset else 0
    file_mode = "a" if (args.resume and os.path.exists(args.out)) or offset > 0 else "w"
    with open(args.out, file_mode, encoding="utf-8") as f:

        while True:
            params = {
                "include_fields": ",".join(FIELDS),
                "f1": "creation_ts",
                "o1": "greaterthan",
                "v1": args.since,
                "f2": "resolution",
                "o2": "anyexact",
                "v2": ",".join(ALLOWED_RESOLUTIONS),
                "f3": "classification",
                "o3": "notequals",
                "v3": EXCLUDED_CLASSIFICATION,
                "limit": args.limit,
                "offset": offset,
            }
            try:
                data = _get_json(session, BUGZILLA_API, params=params, timeout_s=args.timeout)
            except requests.RequestException as e:
                sleep_s = min(60.0, 2.0 + random.random() * 2.0)
                print(f"Request failed at offset={offset} (sleep {sleep_s:.1f}s): {e}")
                time.sleep(sleep_s)
                print(
                    "Giving up after retries. Re-run with `--resume` (or `--start-offset`) to continue."
                )
                return 1
            bugs = data.get("bugs", [])
            if not bugs:
                break

            for bug in bugs:
                f.write(json.dumps(_normalize_row(bug), ensure_ascii=False) + "\n")
            f.flush()

            total += len(bugs)
            # print(f"Fetched {len(bugs)} bugs (total={total}, offset={offset})")

            offset += len(bugs)
            if len(bugs) < args.limit or args.dry_run:
                break

    print(f"✅ Saved Bugzilla bugs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
