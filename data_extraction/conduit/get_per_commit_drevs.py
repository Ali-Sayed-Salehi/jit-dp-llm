#!/usr/bin/env python3
"""
Fetch Mozilla Phabricator Differential Revisions referenced by commits.

This script reads datasets/mozilla_code_review/all_commits.jsonl. For each
commit, it looks for a trailing Mozilla Phabricator Differential Revision URL
in the commit description, for example:

    https://phabricator.services.mozilla.com/D206595

Commits without a matching trailing URL are skipped. Matching DREVs are fetched
through Conduit and only published, closed revisions are written to:

    datasets/mozilla_code_review/per_commit_drevs.jsonl

Each output row contains `commit_id` with the commit hash, followed by `drev`
with the raw revision object returned by `differential.revision.search`.

Use `--debug` to process only the last 10 commits by first-parent graph order.
The debug subset is selected before extracting DREV URLs or initializing the
Phabricator client, so skipped commits do not cause API calls.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_JSONL = (
    REPO_ROOT / "datasets" / "mozilla_code_review" / "all_commits.jsonl"
)
DEFAULT_OUTPUT_JSONL = (
    REPO_ROOT / "datasets" / "mozilla_code_review" / "per_commit_drevs.jsonl"
)
DEFAULT_PHABRICATOR_API_URL = "https://phabricator.services.mozilla.com/api/"

NULL_NODE = "0000000000000000000000000000000000000000"
DREV_URL_RE = re.compile(
    r"https://phabricator\.services\.mozilla\.com/D(\d+)\s*\Z",
    re.IGNORECASE,
)


class ConduitCaller:
    def __init__(
        self,
        *,
        min_interval_seconds: float,
        max_retries: int,
        retry_base_sleep_seconds: float,
    ) -> None:
        self.min_interval_seconds = min_interval_seconds
        self.max_retries = max_retries
        self.retry_base_sleep_seconds = retry_base_sleep_seconds
        self.last_call_time = 0.0

    def call(self, method: Any, **kwargs: Any) -> Any:
        for attempt in range(self.max_retries):
            now = time.time()
            elapsed = now - self.last_call_time
            if elapsed < self.min_interval_seconds:
                time.sleep(self.min_interval_seconds - elapsed)

            try:
                result = method(**kwargs)
                self.last_call_time = time.time()
                return result
            except Exception as exc:
                message = str(exc)
                if "429" not in message and "Too Many Requests" not in message:
                    raise

                wait_seconds = self.retry_base_sleep_seconds * (attempt + 1)
                print(
                    "Received 429 from Phabricator "
                    f"(attempt {attempt + 1}/{self.max_retries}); "
                    f"sleeping {wait_seconds:g}s before retry.",
                    file=sys.stderr,
                )
                time.sleep(wait_seconds)

        raise RuntimeError(
            f"Exceeded {self.max_retries} retries due to repeated 429 responses."
        )


class PhabricatorClient:
    def __init__(self, *, api_url: str, caller: ConduitCaller) -> None:
        self.api_url = api_url
        self.caller = caller
        self._phab: Any | None = None

    @property
    def phab(self) -> Any:
        if self._phab is None:
            try:
                from phabricator import Phabricator
            except ImportError as exc:
                raise RuntimeError(
                    "The `phabricator` package is required. Install python-phabricator "
                    "in the environment used to run this script."
                ) from exc

            self._phab = Phabricator(host=self.api_url, token=load_api_token())
            self.caller.call(self._phab.update_interfaces)
            self._phab.timeout = 30
        return self._phab

    def get_revision_by_id(self, drev_id: int) -> dict[str, Any] | None:
        result = self.caller.call(
            self.phab.differential.revision.search,
            constraints={"ids": [drev_id]},
            limit=1,
        )
        data = result.get("data", [])
        return data[0] if data else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        default=str(DEFAULT_INPUT_JSONL),
        help="Path to all_commits.jsonl.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(DEFAULT_OUTPUT_JSONL),
        help="Path to write per-commit DREV JSONL.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Process only the last debug-count commits by first-parent graph order.",
    )
    parser.add_argument(
        "--debug-count",
        type=int,
        default=10,
        help="Number of graph-tail commits to process in debug mode.",
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("PHABRICATOR_API_URL", DEFAULT_PHABRICATOR_API_URL),
        help="Phabricator Conduit API URL.",
    )
    parser.add_argument(
        "--rate-limit-min-interval",
        type=float,
        default=0.5,
        help="Minimum seconds between Conduit calls.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retries for Conduit 429 responses.",
    )
    parser.add_argument(
        "--retry-base-sleep",
        type=float,
        default=5.0,
        help="Base sleep seconds for Conduit 429 backoff.",
    )
    return parser.parse_args(argv)


def load_api_token() -> str:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError(
            "The `python-dotenv` package is required to load secrets/.env."
        ) from exc

    load_dotenv(dotenv_path=REPO_ROOT / "secrets" / ".env")
    token = os.getenv("CONDUIT_API_TOKEN")
    if not token:
        raise RuntimeError("CONDUIT_API_TOKEN is not set in environment or secrets/.env")
    return token


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as input_file:
        for line_num, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected object at {path}:{line_num}")
            yield line_num, record


def extract_drev_id(desc: str | None) -> int | None:
    if not desc:
        return None
    match = DREV_URL_RE.search(desc)
    if not match:
        return None
    return int(match.group(1))


def is_published_and_closed(revision: dict[str, Any]) -> bool:
    fields = revision.get("fields", {})
    if not isinstance(fields, dict):
        return False
    status = fields.get("status", {})
    if not isinstance(status, dict):
        return False

    is_published = status.get("value") == "published"
    is_closed = bool(
        status.get("closed", False)
        or fields.get("closed", False)
        or revision.get("closed", False)
    )
    return is_published and is_closed


def load_commits_for_debug(path: Path) -> list[dict[str, Any]]:
    commits: list[dict[str, Any]] = []
    node_to_index: dict[str, int] = {}

    for _, record in iter_jsonl(path):
        node = record.get("node")
        if not isinstance(node, str) or not node:
            raise ValueError(f"Commit is missing a valid node: {record!r}")
        if node in node_to_index:
            raise ValueError(f"Duplicate commit node encountered: {node}")

        current_index = len(commits)
        parents = record.get("parents", [])
        if not isinstance(parents, list):
            raise ValueError(f"Commit {node} has non-list parents: {parents!r}")

        for parent in parents:
            if parent == NULL_NODE:
                continue
            if not isinstance(parent, str):
                raise ValueError(f"Commit {node} has non-string parent: {parent!r}")
            parent_index = node_to_index.get(parent)
            if parent_index is None:
                raise ValueError(f"Commit {node} references missing parent {parent}.")
            if parent_index >= current_index:
                raise ValueError(
                    "all_commits.jsonl is not parent-before-child ordered: "
                    f"commit {node} at index {current_index} has parent {parent} "
                    f"at index {parent_index}."
                )

        node_to_index[node] = current_index
        commits.append(record)

    return commits


def select_last_commits_by_first_parent(
    commits: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    if count <= 0 or not commits:
        return []

    node_to_commit: dict[str, dict[str, Any]] = {}
    node_to_index: dict[str, int] = {}
    children_by_node: dict[str, list[str]] = {}

    for index, commit in enumerate(commits):
        node = commit["node"]
        node_to_commit[node] = commit
        node_to_index[node] = index
        children_by_node.setdefault(node, [])

    for commit in commits:
        node = commit["node"]
        parents = commit.get("parents", [])
        for parent in parents:
            if parent in node_to_commit:
                children_by_node.setdefault(parent, []).append(node)

    heads = [
        commit["node"]
        for commit in commits
        if not children_by_node.get(commit["node"])
    ]
    if not heads:
        return commits[-count:]

    head = max(heads, key=lambda node: node_to_index[node])
    if len(heads) > 1:
        print(
            f"Found {len(heads)} graph heads; using latest head by file order: {head}",
            file=sys.stderr,
        )

    selected_newest_first: list[dict[str, Any]] = []
    seen: set[str] = set()
    current_node: str | None = head

    while current_node and len(selected_newest_first) < count:
        if current_node in seen:
            raise ValueError(f"Cycle detected while walking parents at {current_node}")
        seen.add(current_node)

        commit = node_to_commit[current_node]
        selected_newest_first.append(commit)

        parent_nodes = [
            parent
            for parent in commit.get("parents", [])
            if parent != NULL_NODE and parent in node_to_commit
        ]
        current_node = parent_nodes[0] if parent_nodes else None

    return list(reversed(selected_newest_first))


def get_commits_to_process(
    path: Path,
    debug: bool,
    debug_count: int,
) -> list[dict[str, Any]] | None:
    if not debug:
        return None
    if debug_count <= 0:
        print("DEBUG: selected 0 commits from first-parent graph tail.", file=sys.stderr)
        return []

    commits = load_commits_for_debug(path)
    selected = select_last_commits_by_first_parent(commits, debug_count)
    print(
        f"DEBUG: selected {len(selected)} commits from first-parent graph tail.",
        file=sys.stderr,
    )
    return selected


def iter_commits_to_process(
    path: Path,
    debug_commits: list[dict[str, Any]] | None,
) -> Iterator[dict[str, Any]]:
    if debug_commits is not None:
        yield from debug_commits
        return

    for _, record in iter_jsonl(path):
        yield record


def process_commits(
    *,
    commits: Iterator[dict[str, Any]],
    output_jsonl: Path,
    client: PhabricatorClient,
) -> dict[str, int]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    revision_cache: dict[int, dict[str, Any] | None] = {}
    stats = {
        "commits_seen": 0,
        "commits_with_trailing_drev_url": 0,
        "unique_drevs_fetched": 0,
        "published_closed_drevs_written": 0,
        "skipped_unpublished_or_open": 0,
        "fetch_errors": 0,
    }

    with output_jsonl.open("w", encoding="utf-8") as output_file:
        for commit in commits:
            stats["commits_seen"] += 1
            commit_id = commit.get("node")
            if not isinstance(commit_id, str) or not commit_id:
                print(f"[WARN] Skipping commit without valid node: {commit!r}", file=sys.stderr)
                continue

            drev_id = extract_drev_id(commit.get("desc"))
            if drev_id is None:
                continue
            stats["commits_with_trailing_drev_url"] += 1

            if drev_id not in revision_cache:
                try:
                    revision_cache[drev_id] = client.get_revision_by_id(drev_id)
                    stats["unique_drevs_fetched"] += 1
                except Exception as exc:
                    revision_cache[drev_id] = None
                    stats["fetch_errors"] += 1
                    print(f"[WARN] Failed to fetch D{drev_id}: {exc}", file=sys.stderr)

            revision = revision_cache[drev_id]
            if revision is None:
                continue
            if not is_published_and_closed(revision):
                stats["skipped_unpublished_or_open"] += 1
                continue

            output_row = {
                "commit_id": commit_id,
                "drev": revision,
            }
            output_file.write(json.dumps(output_row) + "\n")
            stats["published_closed_drevs_written"] += 1

    return stats


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_jsonl = Path(args.input_jsonl)
    output_jsonl = Path(args.output_jsonl)

    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")

    debug_commits = get_commits_to_process(
        input_jsonl,
        debug=args.debug,
        debug_count=args.debug_count,
    )
    commits = iter_commits_to_process(input_jsonl, debug_commits)

    caller = ConduitCaller(
        min_interval_seconds=args.rate_limit_min_interval,
        max_retries=args.max_retries,
        retry_base_sleep_seconds=args.retry_base_sleep,
    )
    client = PhabricatorClient(api_url=args.api_url, caller=caller)

    stats = process_commits(
        commits=commits,
        output_jsonl=output_jsonl,
        client=client,
    )

    print(f"Wrote {stats['published_closed_drevs_written']} rows to {output_jsonl}")
    print(json.dumps(stats, sort_keys=True), file=sys.stderr)


if __name__ == "__main__":
    main()
