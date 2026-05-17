#!/usr/bin/env python3
"""
Create a per-commit Mozilla code-review dataset.

This script reads:

    datasets/mozilla_code_review/per_commit_drev_transactions.jsonl

It writes one JSON object per per-commit DREV row to:

    datasets/mozilla_code_review/drev_review_data.jsonl

Each output row contains:

    commit_id, dataset_split, risk_score, drev_submission_date, drev_author,
    files_changed, reviews

`files_changed` is extracted from a local Mercurial clone of Mozilla's autoland
repository at:

    data_extraction/mercurial/repos/autoland

The autoland repository is cloned if missing and updated with `hg pull -u` if it
already exists. Use `--debug` to process only the last 10 rows from
per_commit_drev_transactions.jsonl.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "datasets" / "mozilla_code_review"

DEFAULT_DREV_TRANSACTIONS_JSONL = DATASET_DIR / "per_commit_drev_transactions.jsonl"
DEFAULT_OUTPUT_JSONL = DATASET_DIR / "drev_review_data.jsonl"
DEFAULT_AUTOLAND_REPO = (
    REPO_ROOT / "data_extraction" / "mercurial" / "repos" / "autoland"
)
DEFAULT_AUTOLAND_URL = "https://hg.mozilla.org/integration/autoland"


@dataclass(frozen=True)
class JsonlRecord:
    line_num: int
    record: dict[str, Any]


@dataclass(frozen=True)
class InputRow:
    line_num: int
    commit_id: str
    dataset_split: str
    risk_score: float
    transactions: list[dict[str, Any]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drev-transactions-jsonl",
        default=str(DEFAULT_DREV_TRANSACTIONS_JSONL),
        help="Path to per_commit_drev_transactions.jsonl.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(DEFAULT_OUTPUT_JSONL),
        help="Path to write drev_review_data.jsonl.",
    )
    parser.add_argument(
        "--autoland-repo",
        default=str(DEFAULT_AUTOLAND_REPO),
        help="Local path for the Mozilla autoland Mercurial repository.",
    )
    parser.add_argument(
        "--autoland-url",
        default=DEFAULT_AUTOLAND_URL,
        help="Mercurial URL used when cloning autoland.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Process only the last debug-count rows from "
            "per_commit_drev_transactions.jsonl."
        ),
    )
    parser.add_argument(
        "--debug-count",
        type=int,
        default=10,
        help="Number of trailing transaction rows to process in debug mode.",
    )
    parser.add_argument(
        "--include-app-comments",
        action="store_true",
        help="Include comments authored by Phabricator applications.",
    )
    parser.add_argument(
        "--skip-repo-update",
        action="store_true",
        help=(
            "Do not clone or pull autoland. The local repository must already "
            "exist. This is intended for local smoke tests."
        ),
    )
    return parser.parse_args(argv)


def iter_jsonl(path: Path) -> Iterator[JsonlRecord]:
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
            yield JsonlRecord(line_num=line_num, record=record)


def load_jsonl_records(path: Path, *, tail_count: int | None = None) -> list[JsonlRecord]:
    if tail_count is not None and tail_count <= 0:
        return []

    records: list[JsonlRecord] | deque[JsonlRecord]
    records = deque(maxlen=tail_count) if tail_count is not None else []

    for item in iter_jsonl(path):
        records.append(item)
    return list(records)


def parse_risk_score(value: Any, *, line_label: str) -> float:
    try:
        risk_score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{line_label}: invalid risk_score {value!r}") from exc
    if not 0.0 <= risk_score <= 1.0:
        raise ValueError(f"{line_label}: risk_score outside [0, 1]: {risk_score}")
    return risk_score


def parse_input_row(item: JsonlRecord, *, path: Path) -> InputRow:
    record = item.record
    line_label = f"{path}:{item.line_num}"

    commit_id = record.get("commit_id")
    if not isinstance(commit_id, str) or not commit_id:
        raise ValueError(f"{line_label}: missing commit_id")

    dataset_split = record.get("dataset_split")
    if not isinstance(dataset_split, str) or not dataset_split:
        raise ValueError(f"{line_label}: missing dataset_split")

    transactions = record.get("transactions")
    if not isinstance(transactions, list):
        raise ValueError(f"{line_label}: missing transactions list")
    typed_transactions: list[dict[str, Any]] = []
    for index, transaction in enumerate(transactions):
        if not isinstance(transaction, dict):
            raise ValueError(f"{line_label}: transactions[{index}] is not an object")
        typed_transactions.append(transaction)

    if "risk_score" not in record:
        raise ValueError(f"{line_label}: missing risk_score")
    risk_score = parse_risk_score(record.get("risk_score"), line_label=line_label)

    return InputRow(
        line_num=item.line_num,
        commit_id=commit_id,
        dataset_split=dataset_split,
        risk_score=risk_score,
        transactions=typed_transactions,
    )


def load_input_rows(path: Path, *, tail_count: int | None = None) -> list[InputRow]:
    return [
        parse_input_row(item, path=path)
        for item in load_jsonl_records(path, tail_count=tail_count)
    ]


def ensure_autoland_repo(
    *,
    repo_path: Path,
    repo_url: str,
    skip_update: bool,
) -> None:
    if shutil.which("hg") is None:
        raise RuntimeError("Mercurial CLI `hg` was not found on PATH")

    hg_dir = repo_path / ".hg"
    if skip_update:
        if not hg_dir.is_dir():
            raise FileNotFoundError(
                f"--skip-repo-update was used, but no Mercurial repo exists at {repo_path}"
            )
        return

    if hg_dir.is_dir():
        print(f"Updating autoland repository at {repo_path}", file=sys.stderr)
        subprocess.run(["hg", "pull", "-u"], cwd=repo_path, check=True)
        return

    if repo_path.exists():
        try:
            next(repo_path.iterdir())
        except StopIteration:
            repo_path.rmdir()
        else:
            raise RuntimeError(
                f"{repo_path} exists but is not a Mercurial repository. Move it "
                "aside or pass a different --autoland-repo path."
            )

    repo_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning autoland repository into {repo_path}", file=sys.stderr)
    subprocess.run(["hg", "clone", repo_url, str(repo_path)], check=True)


def run_hg_capture(repo_path: Path, args: list[str]) -> str:
    result = subprocess.run(
        ["hg", *args],
        cwd=repo_path,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        command = " ".join(["hg", *args])
        raise RuntimeError(
            f"`{command}` failed in {repo_path} with exit code {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    return result.stdout


def get_changed_files(repo_path: Path, commit_id: str) -> list[str]:
    output = run_hg_capture(repo_path, ["status", "--change", commit_id])
    files: list[str] = []
    for line in output.splitlines():
        if not line:
            continue
        if len(line) >= 3 and line[1] == " ":
            files.append(line[2:])
        else:
            files.append(line)
    return files


def unix_timestamp_to_iso(value: Any) -> str | None:
    if value is None:
        return None
    try:
        timestamp = int(value)
    except (TypeError, ValueError):
        return None
    if timestamp <= 0:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace(
        "+00:00",
        "Z",
    )


def transaction_sort_key(transaction: dict[str, Any]) -> tuple[int, int]:
    date_created = transaction.get("dateCreated")
    transaction_id = transaction.get("id")
    try:
        timestamp = int(date_created)
    except (TypeError, ValueError):
        timestamp = sys.maxsize
    try:
        numeric_id = int(transaction_id)
    except (TypeError, ValueError):
        numeric_id = sys.maxsize
    return timestamp, numeric_id


def find_create_transaction(transactions: list[dict[str, Any]]) -> dict[str, Any] | None:
    create_transactions = [
        transaction
        for transaction in transactions
        if transaction.get("type") == "create"
    ]
    if create_transactions:
        return min(create_transactions, key=transaction_sort_key)
    if transactions:
        return min(transactions, key=transaction_sort_key)
    return None


def is_app_author(author: str | None) -> bool:
    return isinstance(author, str) and author.startswith("PHID-APPS-")


def extract_reviews(
    transactions: list[dict[str, Any]],
    *,
    include_app_comments: bool,
) -> list[dict[str, Any]]:
    review_items: list[tuple[int, int, dict[str, Any]]] = []

    for transaction in transactions:
        comments = transaction.get("comments")
        if not isinstance(comments, list):
            continue

        for comment in comments:
            if not isinstance(comment, dict):
                continue
            if comment.get("removed") is True:
                continue

            content = comment.get("content")
            raw_comment = content.get("raw") if isinstance(content, dict) else None
            if not isinstance(raw_comment, str) or not raw_comment.strip():
                continue

            author = comment.get("authorPHID") or transaction.get("authorPHID")
            if not isinstance(author, str) or not author:
                author = None
            if not include_app_comments and is_app_author(author):
                continue

            submission_date_value = comment.get("dateCreated", transaction.get("dateCreated"))
            submission_date = unix_timestamp_to_iso(submission_date_value)
            try:
                sort_timestamp = int(submission_date_value)
            except (TypeError, ValueError):
                sort_timestamp = sys.maxsize
            try:
                comment_id = int(comment.get("id"))
            except (TypeError, ValueError):
                comment_id = sys.maxsize

            review_items.append(
                (
                    sort_timestamp,
                    comment_id,
                    {
                        "author": author,
                        "submission_date": submission_date,
                        "comment": raw_comment,
                    },
                )
            )

    review_items.sort(key=lambda item: (item[0], item[1]))
    return [review for _, _, review in review_items]


def build_output_row(
    *,
    input_row: InputRow,
    changed_files_by_commit: dict[str, list[str]],
    autoland_repo: Path,
    include_app_comments: bool,
) -> dict[str, Any]:
    commit_id = input_row.commit_id
    if commit_id not in changed_files_by_commit:
        changed_files_by_commit[commit_id] = get_changed_files(autoland_repo, commit_id)

    create_transaction = find_create_transaction(input_row.transactions)
    drev_author = None
    drev_submission_date = None
    if create_transaction is not None:
        author = create_transaction.get("authorPHID")
        drev_author = author if isinstance(author, str) and author else None
        drev_submission_date = unix_timestamp_to_iso(create_transaction.get("dateCreated"))

    return {
        "commit_id": commit_id,
        "dataset_split": input_row.dataset_split,
        "risk_score": input_row.risk_score,
        "drev_submission_date": drev_submission_date,
        "drev_author": drev_author,
        "files_changed": changed_files_by_commit[commit_id],
        "reviews": extract_reviews(
            input_row.transactions,
            include_app_comments=include_app_comments,
        ),
    }


def write_dataset(
    *,
    rows: list[InputRow],
    autoland_repo: Path,
    output_jsonl: Path,
    include_app_comments: bool,
) -> dict[str, int]:
    stats = {
        "rows_seen": len(rows),
        "rows_written": 0,
        "reviews_written": 0,
    }
    changed_files_by_commit: dict[str, list[str]] = {}

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as output_file:
        for input_row in rows:
            output_row = build_output_row(
                input_row=input_row,
                changed_files_by_commit=changed_files_by_commit,
                autoland_repo=autoland_repo,
                include_app_comments=include_app_comments,
            )

            output_file.write(json.dumps(output_row) + "\n")
            stats["rows_written"] += 1
            stats["reviews_written"] += len(output_row["reviews"])

    return stats


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    drev_transactions_jsonl = Path(args.drev_transactions_jsonl)
    output_jsonl = Path(args.output_jsonl)
    autoland_repo = Path(args.autoland_repo)

    if not drev_transactions_jsonl.exists():
        raise FileNotFoundError(f"DREV transaction JSONL not found: {drev_transactions_jsonl}")

    rows = load_input_rows(
        drev_transactions_jsonl,
        tail_count=args.debug_count if args.debug else None,
    )
    if args.debug:
        print(f"DEBUG: selected {len(rows)} trailing DREV transaction rows.", file=sys.stderr)

    ensure_autoland_repo(
        repo_path=autoland_repo,
        repo_url=args.autoland_url,
        skip_update=args.skip_repo_update,
    )

    stats = write_dataset(
        rows=rows,
        autoland_repo=autoland_repo,
        output_jsonl=output_jsonl,
        include_app_comments=args.include_app_comments,
    )
    print(json.dumps(stats, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
