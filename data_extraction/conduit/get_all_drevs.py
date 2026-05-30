#!/usr/bin/env python3
"""
Fetch all Mozilla Phabricator Differential Revisions for the dataset window.

This script reads:

    datasets/mozilla_code_review/all_commits.jsonl
    datasets/mozilla_code_review/risk_predictions_eval.json
    datasets/mozilla_code_review/risk_predictions_final_test.json

It derives split boundaries the same way as the other current Conduit scripts:
prediction commit IDs are mapped into the Mercurial ordering from
all_commits.jsonl, then the first eval commit and last final-test commit define
the dataset interval. The Conduit query window is padded by one calendar month
on both sides:

    eval start boundary - 1 month
    final-test end boundary + 1 month

All Differential Revisions created in that padded window for the selected
repositories are fetched with `differential.revision.search` and appended to:

    datasets/mozilla_code_review/all_drevs.jsonl

Each output line is one raw revision object returned by Conduit. If the output
file already exists, DREV IDs already present in the file are skipped. The same
output file is also used for per-repository timestamp-based resume: the script
scans existing rows, finds the maximum persisted `fields.dateCreated` for each
selected repository, and re-queries from that timestamp inclusively. Inclusive
resume plus DREV-ID deduplication avoids losing rows when a page boundary or
interrupted write falls in the middle of several revisions with the same
timestamp.

After the repository creation-window queries finish, the script also scans
in-boundary commit messages for trailing DREV URLs and fetches any referenced
DREV IDs still missing from the output. This covers reviews created before the
padded DREV creation window but landed by commits inside the dataset interval.
"""

from __future__ import annotations

import argparse
import calendar
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "datasets" / "mozilla_code_review"

DEFAULT_INPUT_JSONL = DATASET_DIR / "all_commits.jsonl"
DEFAULT_OUTPUT_JSONL = DATASET_DIR / "all_drevs.jsonl"
DEFAULT_EVAL_PREDICTIONS_JSON = DATASET_DIR / "risk_predictions_eval.json"
DEFAULT_FINAL_TEST_PREDICTIONS_JSON = (
    DATASET_DIR / "risk_predictions_final_test.json"
)
DEFAULT_PHABRICATOR_API_URL = "https://phabricator.services.mozilla.com/api/"

NULL_NODE = "0000000000000000000000000000000000000000"
DEFAULT_REPOSITORY_KEYS = ("autoland", "mozilla-central")
REPO_PHIDS = {
    "mozilla-central": "PHID-REPO-saax4qdxlbbhahhp2kg5",
    "autoland": "PHID-REPO-wxrrnneqyw2v3wcqbkfj",
}
DREV_URL_RE = re.compile(
    r"https://phabricator\.services\.mozilla\.com/D(\d+)\s*\Z",
    re.IGNORECASE,
)
MISSING_BACKFILL_WARNING_LIMIT = 20


@dataclass(frozen=True)
class SplitBoundary:
    name: str
    start_index: int
    end_index: int
    start_commit_id: str
    end_commit_id: str
    sample_commit_ids: frozenset[str]


@dataclass(frozen=True)
class DrevWindow:
    start_epoch: int
    end_epoch: int
    start_datetime: datetime
    end_datetime: datetime


@dataclass
class ExistingRepositoryOutputState:
    row_count: int
    max_date_created: int | None
    can_resume_from_timestamp: bool


@dataclass
class ExistingOutputState:
    drev_ids: set[int]
    repositories: dict[str, ExistingRepositoryOutputState]


class ConduitCaller:
    def __init__(
        self,
        *,
        min_interval_seconds: float,
    ) -> None:
        self.min_interval_seconds = min_interval_seconds
        self.last_call_time = 0.0

    def call(self, method: Any, **kwargs: Any) -> Any:
        now = time.time()
        elapsed = now - self.last_call_time
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)

        try:
            return method(**kwargs)
        finally:
            self.last_call_time = time.time()


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

    def search_revisions(
        self,
        *,
        constraints: dict[str, Any],
        page_limit: int,
        after_cursor: str | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "constraints": constraints,
            "order": "oldest",
            "limit": page_limit,
        }
        if after_cursor is not None:
            kwargs["after"] = after_cursor
        return self.caller.call(self.phab.differential.revision.search, **kwargs)

    def search_revisions_by_ids(self, drev_ids: list[int]) -> dict[str, Any]:
        return self.caller.call(
            self.phab.differential.revision.search,
            constraints={"ids": drev_ids},
            limit=len(drev_ids),
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        default=str(DEFAULT_INPUT_JSONL),
        help="Path to all_commits.jsonl.",
    )
    parser.add_argument(
        "--eval-predictions-json",
        default=str(DEFAULT_EVAL_PREDICTIONS_JSON),
        help="Path to risk_predictions_eval.json.",
    )
    parser.add_argument(
        "--final-test-predictions-json",
        default=str(DEFAULT_FINAL_TEST_PREDICTIONS_JSON),
        help="Path to risk_predictions_final_test.json.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(DEFAULT_OUTPUT_JSONL),
        help="Path to write all DREV JSONL.",
    )
    parser.add_argument(
        "--repository-key",
        choices=sorted(REPO_PHIDS),
        action="append",
        default=None,
        help=(
            "Known Mozilla repository to query. May be specified more than "
            "once. Defaults to autoland and mozilla-central."
        ),
    )
    parser.add_argument(
        "--repository-phid",
        action="append",
        default=None,
        help=(
            "Explicit repository PHID to query. May be specified more than "
            "once. Overrides --repository-key when provided."
        ),
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("PHABRICATOR_API_URL", DEFAULT_PHABRICATOR_API_URL),
        help="Phabricator Conduit API URL.",
    )
    parser.add_argument(
        "--page-limit",
        type=int,
        default=100,
        help="Conduit page size for differential.revision.search.",
    )
    parser.add_argument(
        "--boundary-padding-months",
        type=int,
        default=1,
        help="Calendar months to pad before eval start and after final-test end.",
    )
    parser.add_argument(
        "--rate-limit-min-interval",
        type=float,
        default=0.5,
        help="Minimum seconds between Conduit calls.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help=(
            "Maximum number of Conduit pages to fetch. Intended for debugging; "
            "when set, output may stop before --createdEnd."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the existing output JSONL instead of appending missing DREVs.",
    )
    parser.add_argument(
        "--skip-referenced-drev-backfill",
        action="store_true",
        help=(
            "Skip the final by-ID backfill for DREVs referenced by in-boundary "
            "commits but not found by the repository creation-window queries."
        ),
    )
    return parser.parse_args(argv)


def resolve_repositories(args: argparse.Namespace) -> list[tuple[str, str]]:
    phid_to_key = {phid: key for key, phid in REPO_PHIDS.items()}
    repositories: list[tuple[str, str]] = []
    seen_phids: set[str] = set()

    if args.repository_phid:
        for index, repository_phid in enumerate(args.repository_phid, start=1):
            if not isinstance(repository_phid, str) or not repository_phid:
                raise ValueError("--repository-phid values must be non-empty")
            if repository_phid in seen_phids:
                continue
            repository_label = phid_to_key.get(repository_phid, f"custom-{index}")
            repositories.append((repository_label, repository_phid))
            seen_phids.add(repository_phid)
        return repositories

    repository_keys = args.repository_key or list(DEFAULT_REPOSITORY_KEYS)
    for repository_key in repository_keys:
        repository_phid = REPO_PHIDS[repository_key]
        if repository_phid in seen_phids:
            continue
        repositories.append((repository_key, repository_phid))
        seen_phids.add(repository_phid)

    return repositories


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
        raise RuntimeError(
            "CONDUIT_API_TOKEN is not set in environment or secrets/.env"
        )
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


def load_commits(path: Path) -> list[dict[str, Any]]:
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


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)

    rows = data.get("samples")
    if isinstance(rows, list):
        return rows

    rows = data.get("results")
    if isinstance(rows, list):
        return rows

    raise ValueError(f"{path} must contain a list field named samples or results")


def load_prediction_commit_ids(path: Path) -> list[str]:
    rows = load_prediction_rows(path)

    commit_ids: list[str] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: row {row_index} is not an object")
        commit_id = row.get("commit_id")
        if not isinstance(commit_id, str) or not commit_id:
            raise ValueError(f"{path}: row {row_index} has invalid commit_id")
        commit_ids.append(commit_id)

    if not commit_ids:
        raise ValueError(f"{path} contains no prediction commit ids")
    return commit_ids


def extract_drev_id(desc: str | None) -> int | None:
    if not desc:
        return None
    match = DREV_URL_RE.search(desc)
    if not match:
        return None
    return int(match.group(1))


def collect_referenced_drev_ids(
    *,
    commits: list[dict[str, Any]],
    eval_boundary: SplitBoundary,
    final_test_boundary: SplitBoundary,
) -> set[int]:
    if eval_boundary.start_index > final_test_boundary.end_index:
        raise ValueError(
            "Eval start boundary comes after final-test end boundary: "
            f"{eval_boundary.start_commit_id} > {final_test_boundary.end_commit_id}"
        )

    drev_ids: set[int] = set()
    for index in range(eval_boundary.start_index, final_test_boundary.end_index + 1):
        drev_id = extract_drev_id(commits[index].get("desc"))
        if drev_id is not None:
            drev_ids.add(drev_id)
    return drev_ids


def build_split_boundary(
    *,
    name: str,
    predictions_json: Path,
    node_to_index: dict[str, int],
) -> SplitBoundary:
    commit_ids = load_prediction_commit_ids(predictions_json)
    missing_commit_ids = [
        commit_id for commit_id in commit_ids if commit_id not in node_to_index
    ]
    if missing_commit_ids:
        preview = ", ".join(missing_commit_ids[:5])
        raise ValueError(
            f"{predictions_json} contains {len(missing_commit_ids)} commit ids "
            f"that are not in all_commits.jsonl. First missing ids: {preview}"
        )

    indexed_commit_ids = sorted(
        (node_to_index[commit_id], commit_id) for commit_id in commit_ids
    )
    start_index, start_commit_id = indexed_commit_ids[0]
    end_index, end_commit_id = indexed_commit_ids[-1]

    return SplitBoundary(
        name=name,
        start_index=start_index,
        end_index=end_index,
        start_commit_id=start_commit_id,
        end_commit_id=end_commit_id,
        sample_commit_ids=frozenset(commit_ids),
    )


def describe_boundary(boundary: SplitBoundary) -> str:
    return (
        f"{boundary.name}: {boundary.start_commit_id} "
        f"(index {boundary.start_index}) -> {boundary.end_commit_id} "
        f"(index {boundary.end_index}), samples={len(boundary.sample_commit_ids)}"
    )


def parse_commit_datetime(commit: dict[str, Any], *, label: str) -> datetime:
    date_value = commit.get("date")
    if not isinstance(date_value, list) or not date_value:
        raise ValueError(f"{label} commit is missing a valid date: {commit!r}")

    try:
        timestamp = int(date_value[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} commit has invalid date timestamp: {date_value!r}"
        ) from exc

    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def add_calendar_months(value: datetime, months: int) -> datetime:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def build_drev_window(
    *,
    eval_boundary: SplitBoundary,
    final_test_boundary: SplitBoundary,
    commits: list[dict[str, Any]],
    padding_months: int,
) -> DrevWindow:
    if padding_months < 0:
        raise ValueError("--boundary-padding-months must be non-negative")
    if eval_boundary.start_index > final_test_boundary.end_index:
        raise ValueError(
            "Eval start boundary comes after final-test end boundary: "
            f"{eval_boundary.start_commit_id} > {final_test_boundary.end_commit_id}"
        )

    eval_start = parse_commit_datetime(
        commits[eval_boundary.start_index],
        label="eval start boundary",
    )
    final_test_end = parse_commit_datetime(
        commits[final_test_boundary.end_index],
        label="final-test end boundary",
    )
    start_datetime = add_calendar_months(eval_start, -padding_months)
    end_datetime = add_calendar_months(final_test_end, padding_months)
    if start_datetime > end_datetime:
        raise ValueError(
            f"Computed DREV window is empty: {start_datetime.isoformat()} > "
            f"{end_datetime.isoformat()}"
        )

    return DrevWindow(
        start_epoch=int(start_datetime.timestamp()),
        end_epoch=int(end_datetime.timestamp()),
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )


def revision_date_created(revision: dict[str, Any]) -> int:
    fields = revision.get("fields", {})
    if not isinstance(fields, dict):
        raise ValueError(f"Revision row is missing fields: {revision!r}")
    date_created = fields.get("dateCreated")
    if isinstance(date_created, bool):
        raise ValueError(f"Revision row has invalid dateCreated: {revision!r}")
    try:
        timestamp = int(date_created)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Revision row has invalid dateCreated: {revision!r}"
        ) from exc
    if timestamp < 0:
        raise ValueError(f"Revision row has negative dateCreated: {revision!r}")
    return timestamp


def revision_repository_phid(revision: dict[str, Any]) -> str | None:
    fields = revision.get("fields", {})
    if not isinstance(fields, dict):
        return None
    repository_phid = fields.get("repositoryPHID")
    return repository_phid if isinstance(repository_phid, str) else None


def load_existing_output_state(
    *,
    output_jsonl: Path,
    repository_phids: set[str],
    window: DrevWindow,
) -> ExistingOutputState:
    repository_work_state: dict[str, dict[str, Any]] = {
        repository_phid: {
            "row_count": 0,
            "max_date_created": None,
            "previous_date_created": None,
            "can_resume_from_timestamp": True,
        }
        for repository_phid in repository_phids
    }
    if not output_jsonl.exists():
        return ExistingOutputState(
            drev_ids=set(),
            repositories={
                repository_phid: ExistingRepositoryOutputState(
                    row_count=0,
                    max_date_created=None,
                    can_resume_from_timestamp=False,
                )
                for repository_phid in repository_phids
            },
        )

    existing_drev_ids: set[int] = set()
    duplicate_count = 0
    other_repository_count = 0
    missing_repository_count = 0

    for line_num, record in iter_jsonl(output_jsonl):
        drev = record.get("drev") if isinstance(record.get("drev"), dict) else record
        if not isinstance(drev, dict):
            raise ValueError(f"{output_jsonl}:{line_num}: missing DREV object")

        drev_id = drev.get("id")
        if not isinstance(drev_id, int):
            raise ValueError(f"{output_jsonl}:{line_num}: missing integer DREV id")

        repository = revision_repository_phid(drev)
        if repository is None:
            missing_repository_count += 1
        elif repository not in repository_work_state:
            other_repository_count += 1

        date_created = revision_date_created(drev)
        if repository in repository_work_state:
            state = repository_work_state[repository]
            state["row_count"] += 1
            if not window.start_epoch <= date_created <= window.end_epoch:
                print(
                    f"[WARN] {output_jsonl}:{line_num}: D{drev_id} has "
                    f"dateCreated={date_created} outside the current query "
                    f"window for {repository}; timestamp resume disabled for "
                    "that repository.",
                    file=sys.stderr,
                )
                state["can_resume_from_timestamp"] = False

            previous_date_created = state["previous_date_created"]
            if (
                previous_date_created is not None
                and date_created < previous_date_created
            ):
                print(
                    f"[WARN] {output_jsonl}:{line_num}: dateCreated for "
                    f"{repository} went backward from {previous_date_created} "
                    f"to {date_created}; timestamp resume disabled for that "
                    "repository.",
                    file=sys.stderr,
                )
                state["can_resume_from_timestamp"] = False
            state["previous_date_created"] = date_created
            state["max_date_created"] = (
                date_created
                if state["max_date_created"] is None
                else max(state["max_date_created"], date_created)
            )

        if drev_id in existing_drev_ids:
            duplicate_count += 1
            print(
                f"[WARN] Duplicate existing output row for D{drev_id} "
                f"at {output_jsonl}:{line_num}",
                file=sys.stderr,
            )
            continue
        existing_drev_ids.add(drev_id)

    repository_states = {
        repository_phid: ExistingRepositoryOutputState(
            row_count=int(state["row_count"]),
            max_date_created=state["max_date_created"],
            can_resume_from_timestamp=bool(state["can_resume_from_timestamp"])
            if state["max_date_created"] is not None
            else False,
        )
        for repository_phid, state in repository_work_state.items()
    }

    if existing_drev_ids:
        print(
            f"Found {len(existing_drev_ids)} existing DREV rows in {output_jsonl}; "
            "matching DREVs will be skipped.",
            file=sys.stderr,
        )
    for repository_phid, state in repository_states.items():
        if state.row_count:
            print(
                f"Found {state.row_count} existing rows for repository "
                f"{repository_phid}.",
                file=sys.stderr,
            )
    if duplicate_count:
        print(
            f"[WARN] Found {duplicate_count} duplicate existing output rows in "
            f"{output_jsonl}.",
            file=sys.stderr,
        )
    if other_repository_count:
        print(
            f"Found {other_repository_count} rows for repositories outside the "
            "current query set; they will still be used for DREV-ID dedupe.",
            file=sys.stderr,
        )
    if missing_repository_count:
        print(
            f"[WARN] Found {missing_repository_count} rows without "
            "fields.repositoryPHID; they will still be used for DREV-ID dedupe.",
            file=sys.stderr,
        )

    return ExistingOutputState(
        drev_ids=existing_drev_ids,
        repositories=repository_states,
    )


def fetch_and_write_drevs(
    *,
    client: PhabricatorClient,
    output_jsonl: Path,
    existing_drev_ids: set[int],
    repository_label: str,
    repository_phid: str,
    window: DrevWindow,
    resume_created_start_epoch: int,
    page_limit: int,
    max_pages: int | None,
) -> dict[str, int]:
    if page_limit <= 0:
        raise ValueError("--page-limit must be positive")
    if max_pages is not None and max_pages <= 0:
        raise ValueError("--max-pages must be positive when provided")

    constraints = {
        "createdStart": resume_created_start_epoch,
        "createdEnd": window.end_epoch,
        "repositoryPHIDs": [repository_phid],
    }
    stats = {
        "pages_fetched": 0,
        "drevs_seen": 0,
        "existing_rows_skipped": 0,
        "drevs_written": 0,
        "stopped_after_max_pages": 0,
    }
    after_cursor: str | None = None

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("a", encoding="utf-8") as output_file:
        while True:
            result = client.search_revisions(
                constraints=constraints,
                page_limit=page_limit,
                after_cursor=after_cursor,
            )
            stats["pages_fetched"] += 1

            data = result.get("data", [])
            if not isinstance(data, list):
                raise ValueError(
                    "Unexpected differential.revision.search response: "
                    f"`data` is {type(data).__name__}"
                )

            page_written_count = 0
            for revision in data:
                if not isinstance(revision, dict):
                    raise ValueError(
                        "Unexpected differential.revision.search response: "
                        f"revision row is {type(revision).__name__}"
                    )

                stats["drevs_seen"] += 1
                drev_id = revision.get("id")
                if not isinstance(drev_id, int):
                    raise ValueError(
                        f"Revision row is missing integer id: {revision!r}"
                    )
                revision_date_created(revision)

                if drev_id in existing_drev_ids:
                    stats["existing_rows_skipped"] += 1
                    continue

                output_file.write(json.dumps(revision) + "\n")
                existing_drev_ids.add(drev_id)
                stats["drevs_written"] += 1
                page_written_count += 1

            if page_written_count:
                output_file.flush()
                os.fsync(output_file.fileno())

            cursor = result.get("cursor", {})
            after_cursor = cursor.get("after") if isinstance(cursor, dict) else None
            print(
                f"[{repository_label}] Fetched page {stats['pages_fetched']}: "
                f"{stats['drevs_seen']} DREVs seen, "
                f"{stats['drevs_written']} written.",
                file=sys.stderr,
            )
            if after_cursor is None:
                return stats
            if max_pages is not None and stats["pages_fetched"] >= max_pages:
                stats["stopped_after_max_pages"] = 1
                print(
                    "[WARN] Stopped after "
                    f"--max-pages={max_pages} with pagination cursor "
                    f"after={after_cursor!r}; output is intentionally "
                    "incomplete. Re-run without --max-pages to fetch through "
                    "the full DREV creation window.",
                    file=sys.stderr,
                )
                return stats


def fetch_and_write_drevs_by_id(
    *,
    client: PhabricatorClient,
    output_jsonl: Path,
    existing_drev_ids: set[int],
    drev_ids: set[int],
    batch_size: int,
) -> dict[str, int]:
    if batch_size <= 0:
        raise ValueError("--page-limit must be positive")

    ids_to_fetch = sorted(
        drev_id for drev_id in drev_ids if drev_id not in existing_drev_ids
    )
    stats = {
        "backfill_drevs_requested": len(ids_to_fetch),
        "backfill_batches_fetched": 0,
        "backfill_drevs_seen": 0,
        "backfill_drevs_written": 0,
        "backfill_drevs_missing": 0,
    }
    if not ids_to_fetch:
        print(
            "Referenced-DREV backfill: no missing referenced DREV IDs.",
            file=sys.stderr,
        )
        return stats

    print(
        f"Referenced-DREV backfill: fetching {len(ids_to_fetch)} missing "
        "commit-referenced DREV IDs.",
        file=sys.stderr,
    )

    missing_ids: set[int] = set()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("a", encoding="utf-8") as output_file:
        for start in range(0, len(ids_to_fetch), batch_size):
            batch_ids = ids_to_fetch[start : start + batch_size]
            result = client.search_revisions_by_ids(batch_ids)
            stats["backfill_batches_fetched"] += 1

            data = result.get("data", [])
            if not isinstance(data, list):
                raise ValueError(
                    "Unexpected differential.revision.search response: "
                    f"`data` is {type(data).__name__}"
                )

            returned_ids: set[int] = set()
            batch_written_count = 0
            for revision in data:
                if not isinstance(revision, dict):
                    raise ValueError(
                        "Unexpected differential.revision.search response: "
                        f"revision row is {type(revision).__name__}"
                    )
                drev_id = revision.get("id")
                if not isinstance(drev_id, int):
                    raise ValueError(
                        f"Revision row is missing integer id: {revision!r}"
                    )
                returned_ids.add(drev_id)
                stats["backfill_drevs_seen"] += 1
                revision_date_created(revision)

                if drev_id in existing_drev_ids:
                    continue
                output_file.write(json.dumps(revision) + "\n")
                existing_drev_ids.add(drev_id)
                stats["backfill_drevs_written"] += 1
                batch_written_count += 1

            missing_ids.update(set(batch_ids) - returned_ids)
            if batch_written_count:
                output_file.flush()
                os.fsync(output_file.fileno())

            print(
                "Referenced-DREV backfill batch "
                f"{stats['backfill_batches_fetched']}: "
                f"{stats['backfill_drevs_seen']} DREVs seen, "
                f"{stats['backfill_drevs_written']} written.",
                file=sys.stderr,
            )

    stats["backfill_drevs_missing"] = len(missing_ids)
    if missing_ids:
        preview = ", ".join(
            f"D{drev_id}"
            for drev_id in sorted(missing_ids)[:MISSING_BACKFILL_WARNING_LIMIT]
        )
        print(
            f"[WARN] {len(missing_ids)} referenced DREV IDs were not returned "
            f"by Conduit. First missing IDs: {preview}",
            file=sys.stderr,
        )

    return stats


def empty_stats() -> dict[str, int]:
    return {
        "pages_fetched": 0,
        "drevs_seen": 0,
        "existing_rows_skipped": 0,
        "drevs_written": 0,
        "stopped_after_max_pages": 0,
    }


def add_stats(total: dict[str, int], increment: dict[str, int]) -> None:
    for key, value in increment.items():
        total[key] = total.get(key, 0) + value


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_jsonl = Path(args.input_jsonl)
    eval_predictions_json = Path(args.eval_predictions_json)
    final_test_predictions_json = Path(args.final_test_predictions_json)
    output_jsonl = Path(args.output_jsonl)

    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")
    if not eval_predictions_json.exists():
        raise FileNotFoundError(
            f"Eval predictions JSON not found: {eval_predictions_json}"
        )
    if not final_test_predictions_json.exists():
        raise FileNotFoundError(
            f"Final-test predictions JSON not found: {final_test_predictions_json}"
        )
    if args.overwrite and output_jsonl.exists():
        output_jsonl.unlink()

    repositories = resolve_repositories(args)
    if not repositories:
        raise ValueError("At least one repository must be selected")

    commits = load_commits(input_jsonl)
    node_to_index = {commit["node"]: index for index, commit in enumerate(commits)}
    eval_boundary = build_split_boundary(
        name="eval",
        predictions_json=eval_predictions_json,
        node_to_index=node_to_index,
    )
    final_test_boundary = build_split_boundary(
        name="final test",
        predictions_json=final_test_predictions_json,
        node_to_index=node_to_index,
    )
    window = build_drev_window(
        eval_boundary=eval_boundary,
        final_test_boundary=final_test_boundary,
        commits=commits,
        padding_months=args.boundary_padding_months,
    )

    print(describe_boundary(eval_boundary), file=sys.stderr)
    print(describe_boundary(final_test_boundary), file=sys.stderr)
    print(
        "DREV creation window: "
        f"{window.start_datetime.isoformat()} -> {window.end_datetime.isoformat()} "
        f"({window.start_epoch} -> {window.end_epoch})",
        file=sys.stderr,
    )
    print(
        "Repository PHIDs: "
        + ", ".join(f"{label}={phid}" for label, phid in repositories)
        + f" (output={output_jsonl})",
        file=sys.stderr,
    )
    if args.max_pages is not None:
        print(
            "[WARN] --max-pages is set; this is a bounded debug run and may "
            "not reach the DREV creation window end.",
            file=sys.stderr,
        )

    caller = ConduitCaller(min_interval_seconds=args.rate_limit_min_interval)
    client = PhabricatorClient(api_url=args.api_url, caller=caller)
    existing_output = load_existing_output_state(
        output_jsonl=output_jsonl,
        repository_phids={repository_phid for _, repository_phid in repositories},
        window=window,
    )
    total_stats = empty_stats()
    for repository_label, repository_phid in repositories:
        repository_state = existing_output.repositories[repository_phid]
        resume_created_start_epoch = (
            repository_state.max_date_created
            if (
                repository_state.can_resume_from_timestamp
                and repository_state.max_date_created is not None
            )
            else window.start_epoch
        )
        if (
            repository_state.can_resume_from_timestamp
            and repository_state.max_date_created is not None
        ):
            print(
                f"[{repository_label}] Resuming from existing output timestamp: "
                f"createdStart={repository_state.max_date_created}.",
                file=sys.stderr,
            )
        elif repository_state.row_count:
            print(
                f"[{repository_label}] [WARN] Timestamp resume disabled; "
                "scanning the full query window and deduping against existing "
                "output.",
                file=sys.stderr,
            )
        else:
            print(
                f"[{repository_label}] No existing rows for this repository; "
                "scanning the full query window.",
                file=sys.stderr,
            )

        stats = fetch_and_write_drevs(
            client=client,
            output_jsonl=output_jsonl,
            existing_drev_ids=existing_output.drev_ids,
            repository_label=repository_label,
            repository_phid=repository_phid,
            window=window,
            resume_created_start_epoch=resume_created_start_epoch,
            page_limit=args.page_limit,
            max_pages=args.max_pages,
        )
        add_stats(total_stats, stats)
        print(
            json.dumps(
                {
                    "repository": repository_label,
                    "repository_phid": repository_phid,
                    **stats,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )

    if args.skip_referenced_drev_backfill:
        print(
            "Referenced-DREV backfill skipped by "
            "--skip-referenced-drev-backfill.",
            file=sys.stderr,
        )
    elif args.max_pages is not None:
        print(
            "Referenced-DREV backfill skipped because --max-pages is set for "
            "a bounded debug run.",
            file=sys.stderr,
        )
    else:
        referenced_drev_ids = collect_referenced_drev_ids(
            commits=commits,
            eval_boundary=eval_boundary,
            final_test_boundary=final_test_boundary,
        )
        backfill_stats = fetch_and_write_drevs_by_id(
            client=client,
            output_jsonl=output_jsonl,
            existing_drev_ids=existing_output.drev_ids,
            drev_ids=referenced_drev_ids,
            batch_size=args.page_limit,
        )
        add_stats(total_stats, backfill_stats)
        print(json.dumps(backfill_stats, sort_keys=True), file=sys.stderr)

    total_written = total_stats.get("drevs_written", 0) + total_stats.get(
        "backfill_drevs_written",
        0,
    )
    print(f"Wrote {total_written} rows to {output_jsonl}.")
    print(json.dumps(total_stats, sort_keys=True), file=sys.stderr)


if __name__ == "__main__":
    main()
