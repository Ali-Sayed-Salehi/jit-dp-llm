#!/usr/bin/env python3
"""
Fetch transactions for per-commit Mozilla Phabricator Differential Revisions.

This script reads:

    datasets/mozilla_code_review/per_commit_drevs.jsonl

For each DREV row, it fetches all transactions with Conduit's
`transaction.search` and writes one JSON object per commit/DREV to:

    datasets/mozilla_code_review/per_commit_drev_transactions.jsonl

Each output row contains `commit_id`, `drev_id`, `dataset_split`, `risk_score`,
and `transactions`, where `transactions` is the list of raw transaction objects
returned by Conduit.

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
    risk_score: float
    phid: str


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


def parse_risk_score(value: Any, *, line_label: str) -> float:
    try:
        risk_score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{line_label}: invalid risk_score {value!r}."
        ) from exc
    if not 0.0 <= risk_score <= 1.0:
        raise ValueError(
            f"{line_label}: risk_score outside [0, 1]: {risk_score}."
        )
    return risk_score


def parse_drev_ref(record: dict[str, Any], *, line_label: str) -> DrevRef | None:
    commit_id = record.get("commit_id")
    dataset_split = record.get("dataset_split")
    risk_score_value = record.get("risk_score")
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

    if "risk_score" not in record:
        raise ValueError(
            f"{line_label}: missing risk_score. Regenerate per_commit_drevs.jsonl "
            "with get_per_commit_drevs.py before fetching transactions."
        )
    risk_score = parse_risk_score(risk_score_value, line_label=line_label)

    return DrevRef(
        commit_id=commit_id,
        drev_id=drev_id_int,
        dataset_split=dataset_split,
        risk_score=risk_score,
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


def write_transactions(
    *,
    refs: list[DrevRef],
    output_jsonl: Path,
    client: PhabricatorClient,
    page_limit: int,
) -> dict[str, int]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    refs_by_phid: dict[str, list[DrevRef]] = {}
    for ref in refs:
        refs_by_phid.setdefault(ref.phid, []).append(ref)

    stats = {
        "drev_rows_seen": len(refs),
        "unique_drevs_fetched": 0,
        "drev_rows_written": 0,
        "transaction_records_written": 0,
        "fetch_errors": 0,
    }

    with output_jsonl.open("w", encoding="utf-8") as output_file:
        for phid, phid_refs in refs_by_phid.items():
            first_ref = phid_refs[0]
            try:
                transactions = client.get_transactions(phid, page_limit=page_limit)
                stats["unique_drevs_fetched"] += 1
            except Exception as exc:
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
                    "dataset_split": ref.dataset_split,
                    "risk_score": ref.risk_score,
                    "transactions": transactions,
                }
                output_file.write(json.dumps(output_row) + "\n")
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

    caller = ConduitCaller(
        min_interval_seconds=args.rate_limit_min_interval,
        max_retries=args.max_retries,
        retry_base_sleep_seconds=args.retry_base_sleep,
    )
    client = PhabricatorClient(api_url=args.api_url, caller=caller)

    stats = write_transactions(
        refs=refs,
        output_jsonl=output_jsonl,
        client=client,
        page_limit=args.page_limit,
    )

    print(f"Wrote {stats['drev_rows_written']} rows to {output_jsonl}")
    print(json.dumps(stats, sort_keys=True), file=sys.stderr)


if __name__ == "__main__":
    main()
