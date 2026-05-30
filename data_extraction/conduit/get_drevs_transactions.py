#!/usr/bin/env python3
"""
Fetch transactions for per-commit Mozilla Phabricator Differential Revisions.

This script reads:

    datasets/mozilla_code_review/per_commit_drevs.jsonl

For each DREV row, it fetches all transactions with Conduit's
`transaction.search` and appends one JSON object per commit/DREV to:

    datasets/mozilla_code_review/per_commit_drev_transactions.jsonl

Each output row contains `commit_id`, `drev_id`, `drev_phid`, `dataset_split`,
and `transactions`, where `transactions` is the list of raw transaction objects
returned by Conduit.

If the output file already exists, rows with commit IDs already present in the
file are skipped before any Conduit API call is made. Existing output rows are
also used as a DREV-level transaction cache keyed by DREV PHID, so missing
commit rows for an already-fetched DREV can be written without refetching from
Conduit. Rate-limit errors abort the run immediately, preserving rows already
written so the run can be resumed later.

Use `--debug` to process only 10 eval DREV rows and 10 final-test DREV rows
from per_commit_drevs.jsonl. The debug subset is selected before initializing
the Phabricator client. Transactions are fetched once per unique DREV PHID in
the selected rows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_JSONL = (
    REPO_ROOT / "datasets" / "mozilla_code_review" / "per_commit_drevs.jsonl"
)
DEFAULT_OUTPUT_JSONL = (
    REPO_ROOT
    / "datasets"
    / "mozilla_code_review"
    / "per_commit_drev_transactions.jsonl"
)
DEFAULT_PHABRICATOR_API_URL = "https://phabricator.services.mozilla.com/api/"
DEBUG_DATASET_SPLITS = ("eval", "final test")


@dataclass(frozen=True)
class DrevRef:
    commit_id: str
    drev_id: int
    dataset_split: str
    phid: str


@dataclass
class ExistingOutputState:
    commit_ids: set[str]
    transactions_by_phid: dict[str, list[dict[str, Any]]]


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

    def get_transactions(
        self,
        revision_phid: str,
        *,
        page_limit: int,
    ) -> list[dict[str, Any]]:
        transactions: list[dict[str, Any]] = []
        after_cursor: str | None = None

        while True:
            kwargs: dict[str, Any] = {
                "objectIdentifier": revision_phid,
                "limit": page_limit,
            }
            if after_cursor is not None:
                kwargs["after"] = after_cursor

            result = self.caller.call(self.phab.transaction.search, **kwargs)
            data = result.get("data", [])
            transactions.extend(data)

            cursor = result.get("cursor", {})
            after_cursor = cursor.get("after") if isinstance(cursor, dict) else None
            if after_cursor is None:
                return transactions


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        default=str(DEFAULT_INPUT_JSONL),
        help="Path to per_commit_drevs.jsonl.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(DEFAULT_OUTPUT_JSONL),
        help="Path to write per-commit DREV transaction JSONL.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Process only debug-count eval rows and debug-count final-test rows "
            "from per_commit_drevs.jsonl."
        ),
    )
    parser.add_argument(
        "--debug-count",
        type=int,
        default=10,
        help="Number of rows per split to process in debug mode.",
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
        help="Conduit page size for transaction.search.",
    )
    parser.add_argument(
        "--rate-limit-min-interval",
        type=float,
        default=0.5,
        help="Minimum seconds between Conduit calls.",
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


def load_existing_output_state(path: Path) -> ExistingOutputState:
    if not path.exists():
        return ExistingOutputState(commit_ids=set(), transactions_by_phid={})

    existing_commit_ids: set[str] = set()
    transactions_by_phid: dict[str, list[dict[str, Any]]] = {}
    duplicate_count = 0
    for line_num, record in iter_jsonl(path):
        commit_id = record.get("commit_id")
        if not isinstance(commit_id, str) or not commit_id:
            raise ValueError(f"{path}:{line_num}: missing commit_id")
        if commit_id in existing_commit_ids:
            duplicate_count += 1
            print(
                f"[WARN] Duplicate existing output row for commit {commit_id} "
                f"at {path}:{line_num}",
                file=sys.stderr,
            )
            continue
        existing_commit_ids.add(commit_id)

        transactions = record.get("transactions")
        if not isinstance(transactions, list):
            print(
                f"[WARN] {path}:{line_num}: missing transactions list; row "
                "cannot be used as a DREV transaction cache entry.",
                file=sys.stderr,
            )
            continue

        typed_transactions: list[dict[str, Any]] = []
        for transaction_index, transaction in enumerate(transactions):
            if not isinstance(transaction, dict):
                print(
                    f"[WARN] {path}:{line_num}: transactions[{transaction_index}] "
                    "is not an object; row cannot be used as a DREV "
                    "transaction cache entry.",
                    file=sys.stderr,
                )
                typed_transactions = []
                break
            typed_transactions.append(transaction)

        drev_phid = record.get("drev_phid")
        if not isinstance(drev_phid, str) or not drev_phid:
            continue
        if drev_phid not in transactions_by_phid:
            transactions_by_phid[drev_phid] = typed_transactions

    if existing_commit_ids:
        print(
            f"Found {len(existing_commit_ids)} existing rows in {path}; "
            "matching commits will be skipped.",
            file=sys.stderr,
        )
    if duplicate_count:
        print(
            f"[WARN] Found {duplicate_count} duplicate existing output rows in {path}.",
            file=sys.stderr,
        )
    if transactions_by_phid:
        print(
            f"Loaded cached transactions for {len(transactions_by_phid)} DREVs "
            f"from {path}.",
            file=sys.stderr,
        )
    return ExistingOutputState(
        commit_ids=existing_commit_ids,
        transactions_by_phid=transactions_by_phid,
    )


def parse_drev_ref(record: dict[str, Any], *, line_label: str) -> DrevRef | None:
    commit_id = record.get("commit_id")
    dataset_split = record.get("dataset_split")
    drev = record.get("drev")
    if not isinstance(drev, dict):
        # Support flat DREV records as long as the required top-level metadata
        # fields are present.
        drev = record

    drev_id = drev.get("id")
    phid = drev.get("phid")

    if not isinstance(commit_id, str) or not commit_id:
        print(f"[WARN] Skipping {line_label}: missing commit_id.", file=sys.stderr)
        return None
    if dataset_split not in {"eval", "final test"}:
        print(
            f"[WARN] Skipping {line_label}: invalid dataset_split "
            f"{dataset_split!r}.",
            file=sys.stderr,
        )
        return None
    if not isinstance(phid, str) or not phid:
        print(f"[WARN] Skipping {line_label}: missing DREV phid.", file=sys.stderr)
        return None

    try:
        drev_id_int = int(drev_id)
    except (TypeError, ValueError):
        print(f"[WARN] Skipping {line_label}: invalid DREV id {drev_id!r}.", file=sys.stderr)
        return None

    return DrevRef(
        commit_id=commit_id,
        drev_id=drev_id_int,
        dataset_split=dataset_split,
        phid=phid,
    )


def load_drev_refs(path: Path, *, per_split_count: int | None = None) -> list[DrevRef]:
    if per_split_count is not None and per_split_count <= 0:
        return []

    refs: list[DrevRef] = []
    selected_counts = (
        {split: 0 for split in DEBUG_DATASET_SPLITS}
        if per_split_count is not None
        else None
    )

    for line_num, record in iter_jsonl(path):
        ref = parse_drev_ref(record, line_label=f"{path}:{line_num}")
        if ref is not None:
            if selected_counts is not None:
                if ref.dataset_split not in selected_counts:
                    continue
                if selected_counts[ref.dataset_split] >= per_split_count:
                    continue

                selected_counts[ref.dataset_split] += 1
            refs.append(ref)
            if selected_counts is not None and all(
                split_count >= per_split_count
                for split_count in selected_counts.values()
            ):
                break
    return list(refs)


def describe_debug_split_counts(refs: list[DrevRef]) -> str:
    counts = {
        split: sum(ref.dataset_split == split for ref in refs)
        for split in DEBUG_DATASET_SPLITS
    }
    return ", ".join(f"{counts[split]} {split}" for split in DEBUG_DATASET_SPLITS)


def is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "429" in message or "rate limit" in message or "rate-limit" in message


def write_transactions(
    *,
    refs: list[DrevRef],
    output_jsonl: Path,
    client: PhabricatorClient,
    page_limit: int,
    existing_state: ExistingOutputState,
) -> dict[str, int]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    refs_by_phid: dict[str, list[DrevRef]] = {}
    stats = {
        "drev_rows_seen": len(refs),
        "existing_rows_skipped": 0,
        "unique_drevs_fetched": 0,
        "cached_drevs_reused": 0,
        "drev_rows_written": 0,
        "transaction_records_written": 0,
        "fetch_errors": 0,
    }

    for ref in refs:
        if ref.commit_id in existing_state.commit_ids:
            stats["existing_rows_skipped"] += 1
            continue
        refs_by_phid.setdefault(ref.phid, []).append(ref)

    with output_jsonl.open("a", encoding="utf-8") as output_file:
        for phid, phid_refs in refs_by_phid.items():
            first_ref = phid_refs[0]
            transactions = existing_state.transactions_by_phid.get(phid)
            if transactions is not None:
                stats["cached_drevs_reused"] += 1
            else:
                try:
                    transactions = client.get_transactions(phid, page_limit=page_limit)
                    existing_state.transactions_by_phid[phid] = transactions
                    stats["unique_drevs_fetched"] += 1
                except Exception as exc:
                    if is_rate_limit_error(exc):
                        print(
                            f"[ERROR] Rate limit while fetching transactions "
                            f"for D{first_ref.drev_id} ({phid}); aborting so "
                            "the run can be resumed later without refetching "
                            "already-written rows.",
                            file=sys.stderr,
                        )
                        raise
                    stats["fetch_errors"] += 1
                    print(
                        f"[WARN] Failed to fetch transactions for D{first_ref.drev_id} "
                        f"({phid}): {exc}",
                        file=sys.stderr,
                    )
                    continue

            for ref in phid_refs:
                output_row = {
                    "commit_id": ref.commit_id,
                    "drev_id": ref.drev_id,
                    "drev_phid": ref.phid,
                    "dataset_split": ref.dataset_split,
                    "transactions": transactions,
                }
                output_file.write(json.dumps(output_row) + "\n")
                output_file.flush()
                os.fsync(output_file.fileno())
                existing_state.commit_ids.add(ref.commit_id)
                stats["drev_rows_written"] += 1
                stats["transaction_records_written"] += len(transactions)

    return stats


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_jsonl = Path(args.input_jsonl)
    output_jsonl = Path(args.output_jsonl)

    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")

    refs = load_drev_refs(
        input_jsonl,
        per_split_count=args.debug_count if args.debug else None,
    )
    if args.debug:
        print(
            f"DEBUG: selected {len(refs)} DREV rows "
            f"({describe_debug_split_counts(refs)}).",
            file=sys.stderr,
        )

    existing_state = load_existing_output_state(output_jsonl)
    caller = ConduitCaller(
        min_interval_seconds=args.rate_limit_min_interval,
    )
    client = PhabricatorClient(api_url=args.api_url, caller=caller)

    stats = write_transactions(
        refs=refs,
        output_jsonl=output_jsonl,
        client=client,
        page_limit=args.page_limit,
        existing_state=existing_state,
    )

    print(f"Wrote {stats['drev_rows_written']} rows to {output_jsonl}")
    print(json.dumps(stats, sort_keys=True), file=sys.stderr)


if __name__ == "__main__":
    main()
