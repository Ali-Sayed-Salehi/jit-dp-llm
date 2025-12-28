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

import requests
from dotenv import load_dotenv

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


def _get_json(url: str, params: dict) -> dict:
    params = dict(params)
    if API_KEY:
        params["api_key"] = API_KEY
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def _normalize_row(bug: dict) -> dict:
    row = {k: bug.get(k) for k in FIELDS}
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch all Bugzilla bugs to a JSONL file.")
    parser.add_argument(
        "--out",
        default=OUT_JSONL,
        help="Output JSONL path (default: datasets/mozilla_jit/all_bugs.jsonl)",
    )
    parser.add_argument("--limit", type=int, default=150, help="Page size (default: 150)")
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
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    offset = 0
    total = 0
    with open(args.out, "w", encoding="utf-8") as f:

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
            data = _get_json(BUGZILLA_API, params=params)
            bugs = data.get("bugs", [])
            if not bugs:
                break

            for bug in bugs:
                f.write(json.dumps(_normalize_row(bug), ensure_ascii=False) + "\n")

            total += len(bugs)
            print(f"Fetched {len(bugs)} bugs (total={total}, offset={offset})")

            offset += len(bugs)
            if len(bugs) < args.limit or args.dry_run:
                break

    print(f"✅ Saved Bugzilla bugs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
