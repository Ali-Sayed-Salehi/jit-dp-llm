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
repository are fetched with `differential.revision.search` and appended to:

    datasets/mozilla_code_review/all_drevs.jsonl

Each output line is one raw revision object returned by Conduit. If the output
file already exists, DREV IDs already present in the file are skipped. The same
output file is also used for timestamp-based resume: the script scans existing
rows, finds the maximum persisted `fields.dateCreated`, and re-queries from
that timestamp inclusively. Inclusive resume plus DREV-ID deduplication avoids
losing rows when a page boundary or interrupted write falls in the middle of
several revisions with the same timestamp.
"""

from __future__ import annotations

import argparse
import calendar
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
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
DEFAULT_REPOSITORY_KEY = "autoland"
REPO_PHIDS = {
    "mozilla-central": "PHID-REPO-saax4qdxlbbhahhp2kg5",
    "autoland": "PHID-REPO-wxrrnneqyw2v3wcqbkfj",
}


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
class ExistingOutputState:
    drev_ids: set[int]
    max_date_created: int | None
    can_resume_from_timestamp: bool


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
        default=DEFAULT_REPOSITORY_KEY,
        help="Known Mozilla repository to query.",
    )
    parser.add_argument(
        "--repository-phid",
        default=None,
        help=(
            "Explicit repository PHID to query. Overrides --repository-key when "
            "provided."
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
        help="Maximum number of Conduit pages to fetch. Intended for debugging.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the existing output JSONL instead of appending missing DREVs.",
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
    repository_phid: str,
    window: DrevWindow,
) -> ExistingOutputState:
    if not output_jsonl.exists():
        return ExistingOutputState(
            drev_ids=set(),
            max_date_created=None,
            can_resume_from_timestamp=False,
        )

    existing_drev_ids: set[int] = set()
    duplicate_count = 0
    max_date_created: int | None = None
    previous_date_created: int | None = None
    can_resume_from_timestamp = True

    for line_num, record in iter_jsonl(output_jsonl):
        drev = record.get("drev") if isinstance(record.get("drev"), dict) else record
        if not isinstance(drev, dict):
            raise ValueError(f"{output_jsonl}:{line_num}: missing DREV object")

        drev_id = drev.get("id")
        if not isinstance(drev_id, int):
            raise ValueError(f"{output_jsonl}:{line_num}: missing integer DREV id")

        repository = revision_repository_phid(drev)
        if repository is not None and repository != repository_phid:
            print(
                f"[WARN] {output_jsonl}:{line_num}: D{drev_id} belongs to "
                f"{repository}, not {repository_phid}; timestamp resume disabled.",
                file=sys.stderr,
            )
            can_resume_from_timestamp = False

        date_created = revision_date_created(drev)
        if not window.start_epoch <= date_created <= window.end_epoch:
            print(
                f"[WARN] {output_jsonl}:{line_num}: D{drev_id} has "
                f"dateCreated={date_created} outside the current query window; "
                "timestamp resume disabled.",
                file=sys.stderr,
            )
            can_resume_from_timestamp = False

        if (
            previous_date_created is not None
            and date_created < previous_date_created
        ):
            print(
                f"[WARN] {output_jsonl}:{line_num}: dateCreated went backward "
                f"from {previous_date_created} to {date_created}; timestamp "
                "resume disabled.",
                file=sys.stderr,
            )
            can_resume_from_timestamp = False
        previous_date_created = date_created
        max_date_created = (
            date_created
            if max_date_created is None
            else max(max_date_created, date_created)
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

    if existing_drev_ids:
        print(
            f"Found {len(existing_drev_ids)} existing DREV rows in {output_jsonl}; "
            "matching DREVs will be skipped.",
            file=sys.stderr,
        )
    if duplicate_count:
        print(
            f"[WARN] Found {duplicate_count} duplicate existing output rows in "
            f"{output_jsonl}.",
            file=sys.stderr,
        )
    if can_resume_from_timestamp and max_date_created is not None:
        print(
            f"Resuming from existing output timestamp: "
            f"createdStart={max_date_created}.",
            file=sys.stderr,
        )
    elif existing_drev_ids:
        print(
            "[WARN] Timestamp resume disabled; scanning the full query window and "
            "deduping against existing output.",
            file=sys.stderr,
        )

    return ExistingOutputState(
        drev_ids=existing_drev_ids,
        max_date_created=max_date_created,
        can_resume_from_timestamp=can_resume_from_timestamp,
    )


def fetch_and_write_drevs(
    *,
    client: PhabricatorClient,
    output_jsonl: Path,
    existing_drev_ids: set[int],
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
                f"Fetched page {stats['pages_fetched']}: "
                f"{stats['drevs_seen']} DREVs seen, "
                f"{stats['drevs_written']} written.",
                file=sys.stderr,
            )
            if after_cursor is None:
                return stats
            if max_pages is not None and stats["pages_fetched"] >= max_pages:
                return stats


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

    repository_phid = args.repository_phid or REPO_PHIDS[args.repository_key]

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
        f"Repository PHID: {repository_phid} "
        f"(key={args.repository_key!r}, output={output_jsonl})",
        file=sys.stderr,
    )

    caller = ConduitCaller(min_interval_seconds=args.rate_limit_min_interval)
    client = PhabricatorClient(api_url=args.api_url, caller=caller)
    existing_output = load_existing_output_state(
        output_jsonl=output_jsonl,
        repository_phid=repository_phid,
        window=window,
    )
    resume_created_start_epoch = (
        existing_output.max_date_created
        if (
            existing_output.can_resume_from_timestamp
            and existing_output.max_date_created is not None
        )
        else window.start_epoch
    )
    stats = fetch_and_write_drevs(
        client=client,
        output_jsonl=output_jsonl,
        existing_drev_ids=existing_output.drev_ids,
        repository_phid=repository_phid,
        window=window,
        resume_created_start_epoch=resume_created_start_epoch,
        page_limit=args.page_limit,
        max_pages=args.max_pages,
    )

    print(f"Wrote {stats['drevs_written']} rows to {output_jsonl}.")
    print(json.dumps(stats, sort_keys=True), file=sys.stderr)


if __name__ == "__main__":
    main()
