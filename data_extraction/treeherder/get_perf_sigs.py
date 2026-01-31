#!/usr/bin/env python3
"""
Fetch full performance signature metadata from Treeherder and write JSONL.

Flow:
  1. Fetch the set of signature IDs via `performance/signatures`.
  2. For each signature id, fetch the full record by querying:

    client._get_json("performance/signatures", "<repo>", id=<signature_id>)

Inputs:
  - Treeherder API:
    - `performance/signatures` (enumerate ids + fetch details per id)

Output (JSONL):
  `datasets/mozilla_perf/all_signatures.jsonl`
  One signature record per line (a JSON object containing fields like
  id, signature_hash, framework_id, suite, etc.).

Usage:
  python data_extraction/treeherder/get_perf_sigs.py
  python data_extraction/treeherder/get_perf_sigs.py --repository autoland --debug 25
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any

from requests.exceptions import Timeout, RequestException

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATASET_DIR = os.path.join(REPO_ROOT, "datasets", "mozilla_perf")

DEFAULT_OUTPUT = os.path.join(DATASET_DIR, "all_signatures.jsonl")
DEFAULT_REPOSITORY = "autoland"


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_processed_signature_ids(out_jsonl: str) -> set[int]:
    if not os.path.exists(out_jsonl):
        return set()

    processed: set[int] = set()
    for record in iter_jsonl(out_jsonl):
        if not isinstance(record, dict):
            continue
        sig_id = record.get("id")
        if sig_id is None:
            continue
        try:
            processed.add(int(sig_id))
        except Exception:
            continue
    return processed


def load_signature_ids(client: TreeherderClient, repository: str) -> list[int]:
    signatures = client._get_json("performance/signatures", repository)
    if not signatures:
        return []
    sig_ids = []
    for sig in signatures.values():
        if not isinstance(sig, dict) or "id" not in sig:
            continue
        try:
            sig_ids.append(int(sig["id"]))
        except Exception:
            continue
    return sorted(set(sig_ids))


def normalize_signature_details(signature_id: int, data: Any) -> dict | None:
    """
    Treeherder may return either:
      - {<id>: {...details...}}  (common for `id=<id>` queries)
      - {...details...}
    Normalize into a single dict with an `id` field.
    """
    if isinstance(data, dict):
        if "id" in data:
            details = dict(data)
        else:
            # Often: {"307933": {...}}
            if signature_id in data:
                details = data.get(signature_id)
            elif str(signature_id) in data:
                details = data.get(str(signature_id))
            elif len(data) == 1:
                details = next(iter(data.values()))
            else:
                details = None

        if isinstance(details, dict):
            if "id" not in details:
                details["id"] = signature_id
            return details
    return None


def fetch_signature_details(
    client: TreeherderClient, repository: str, signature_id: int
) -> dict | None:
    try:
        data = client._get_json(
            "performance/signatures", repository, **{"id": signature_id}
        )
    except Timeout as e:
        print(f"[WARN] Timeout fetching signature {signature_id}: {e}")
        return None
    except RequestException as e:
        print(f"[WARN] Request error fetching signature {signature_id}: {e}")
        return None
    except Exception as e:
        print(f"[WARN] Unexpected error fetching signature {signature_id}: {e}")
        return None

    details = normalize_signature_details(signature_id, data)
    if details is None:
        print(f"[WARN] Unrecognized response for signature {signature_id}: {type(data)}")
    return details


def append_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Treeherder performance signature metadata and write JSONL."
    )
    parser.add_argument(
        "--repository", default=DEFAULT_REPOSITORY, help="Treeherder repository name."
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output JSONL path (one signature record per line).",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        help="If set, sample up to N signatures instead of fetching all.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle signature processing order (useful with --debug sampling).",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    try:
        from thclient import TreeherderClient  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'thclient'. Install the Treeherder client "
            "(see `requirements.txt` entry: treeherder-client) in your Python environment "
            "and re-run."
        ) from e

    client = TreeherderClient()
    sig_ids = load_signature_ids(client, args.repository)
    if not sig_ids:
        print(f"No signatures returned for repository '{args.repository}'.")
        return 0

    if args.shuffle:
        random.shuffle(sig_ids)

    if args.debug and args.debug > 0:
        sig_ids = sig_ids[: args.debug]

    processed = load_processed_signature_ids(args.output)
    total = len(sig_ids)
    print(
        f"Found {total} signatures for '{args.repository}'. "
        f"Output: {args.output} (already have {len(processed)})."
    )

    for idx, sig_id in enumerate(sig_ids, start=1):
        if sig_id in processed:
            continue
        print(f"[{idx}/{total}] Fetching signature {sig_id}...")
        details = fetch_signature_details(client, args.repository, sig_id)
        if details is None:
            continue
        append_jsonl(args.output, details)
        processed.add(sig_id)

    print(f"Done. Wrote/kept {len(processed)} signature records in {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
