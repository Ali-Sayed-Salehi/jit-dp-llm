#!/usr/bin/env python3
"""
Create an expanded Mozilla perf-bisect dataset under `datasets/mozilla_perf_bisect_v2`.

This script intentionally does not read the CodeBERT/risk-score prediction JSONs
that define the original eval/final-test boundaries. Instead it:

  1. copies reusable non-risk artifacts from `datasets/mozilla_perf_bisect`;
  2. fetches fresh Treeherder alert summaries into v2;
  3. exports a fresh `all_commits.jsonl` from the existing local Autoland
     checkout without pulling it by default;
  4. merges those fresh artifacts with the existing v1 artifacts by stable ids;
  5. derives failing-signature CSVs from the merged alert summaries without a
     created-date cutoff;
  6. fetches current per-signature performance measurements and merges them with
     existing v1 per-signature measurement caches;
  7. builds `per_sig_perf_data_info.jsonl` and `per_revision_perf_data.jsonl`;
  8. creates regression rows from all merged alert summaries; and
  9. writes the first 30% of chronological regression rows to eval and the last
     70% to final_test.

The old `datasets/mozilla_perf_bisect` directory is never modified.
"""

from __future__ import annotations

import argparse
import ast
import csv
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qs, urlparse

from requests.exceptions import RequestException, Timeout


csv.field_size_limit(sys.maxsize)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = REPO_ROOT / "datasets" / "mozilla_perf_bisect"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "datasets" / "mozilla_perf_bisect_v2"
DEFAULT_AUTOLAND_REPO = REPO_ROOT / "data_extraction" / "mercurial" / "repos" / "autoland"
AUTOLAND_URL = "https://hg-edge.mozilla.org/integration/autoland"
REPOSITORY = "autoland"
NULL_NODE = "0000000000000000000000000000000000000000"

ALERT_SUMMARY_STATUS = {
    "untriaged": 0,
    "downstream": 1,
    "reassigned": 2,
    "invalid": 3,
    "improvement": 4,
    "investigating": 5,
    "wontfix": 6,
    "fixed": 7,
    "backedout": 8,
}
DEFAULT_EXCLUDED_ALERT_SUMMARY_IDS = {46805}
DEFAULT_ALERT_THRESHOLD = 2
DEFAULT_JOB_DURATION_MINUTES = 10.0

STATIC_COPY_FILES = (
    "sig_groups.jsonl",
    "sig_group_job_durations.csv",
)


@dataclass(frozen=True)
class CommitGraph:
    records: list[dict[str, Any]]
    node_to_index: dict[str, int]
    parents_by_node: dict[str, list[str]]

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        if ancestor == descendant:
            return ancestor in self.node_to_index

        ancestor_index = self.node_to_index.get(ancestor)
        descendant_index = self.node_to_index.get(descendant)
        if ancestor_index is None or descendant_index is None:
            return False
        if ancestor_index > descendant_index:
            return False

        stack = [descendant]
        seen: set[str] = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            if node == ancestor:
                return True
            for parent in self.parents_by_node.get(node, []):
                parent_index = self.node_to_index.get(parent)
                if parent_index is not None and parent_index >= ancestor_index:
                    stack.append(parent)
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Existing perf-bisect dataset directory to augment from.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the v2 dataset will be written.",
    )
    parser.add_argument(
        "--autoland-repo",
        type=Path,
        default=DEFAULT_AUTOLAND_REPO,
        help=(
            "Local Mercurial Autoland checkout used for commit export. The "
            "checkout is not pulled unless --pull-commits is set."
        ),
    )
    parser.add_argument(
        "--fresh-lookback-days",
        type=int,
        default=370,
        help=(
            "Treeherder performance/summary lookback for fresh measurements. "
            "Old local measurement caches are merged in separately."
        ),
    )
    parser.add_argument(
        "--fresh-alert-lookback-days",
        type=int,
        default=370,
        help=(
            "Stop fetching alert-summary pages once the oldest row on a page is "
            "older than this many days. Use 0 to page until Treeherder stops."
        ),
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.30,
        help="Fraction of chronological regression rows to write to eval.",
    )
    parser.add_argument(
        "--exclude-framework-ids",
        type=parse_int_csv,
        default=[],
        help=(
            "Comma-separated alert-summary framework ids to exclude from the "
            "regression files. Default is empty so v2 uses all merged alert data. "
            "Pass '2,6,18' to match the old no_fw_2_6_18 convention."
        ),
    )
    parser.add_argument(
        "--included-statuses",
        type=parse_status_csv,
        default=None,
        help=(
            "Optional comma-separated alert summary statuses to keep, e.g. "
            "'fixed,backedout,wontfix'. Default keeps any summary with a "
            "regression alert."
        ),
    )
    parser.add_argument(
        "--exclude-alert-summary-ids",
        type=parse_int_csv,
        default=sorted(DEFAULT_EXCLUDED_ALERT_SUMMARY_IDS),
        help=(
            "Comma-separated alert summary ids to exclude. Defaults to known "
            "inconsistent v1 exclusions."
        ),
    )
    parser.add_argument(
        "--max-alert-pages",
        type=int,
        default=0,
        help="If > 0, fetch at most this many alert-summary pages.",
    )
    parser.add_argument(
        "--limit-signatures",
        type=int,
        default=0,
        help="If > 0, fetch fresh measurement data for only the first N signatures.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help=(
            "Refetch or regenerate v2 output files even when they already "
            "exist. By default, existing v2 artifacts are reused."
        ),
    )
    parser.add_argument(
        "--skip-fetch-alerts",
        action="store_true",
        help="Do not fetch fresh Treeherder alert summaries; only use source-dir data.",
    )
    parser.add_argument(
        "--fetch-commits",
        action="store_true",
        help=(
            "Compatibility flag. Exporting all_commits.jsonl from the local "
            "Autoland checkout is now the default and this flag does not pull."
        ),
    )
    parser.add_argument(
        "--pull-commits",
        action="store_true",
        help=(
            "Run `hg pull -u` on the local Autoland checkout before exporting "
            "all_commits.jsonl. Off by default."
        ),
    )
    parser.add_argument(
        "--skip-fetch-commits",
        action="store_true",
        help=(
            "Do not export from the local Autoland checkout; copy source-dir "
            "all_commits.jsonl instead."
        ),
    )
    parser.add_argument(
        "--skip-fetch-measurements",
        action="store_true",
        help="Do not fetch fresh per-signature measurements; only use source-dir caches.",
    )
    parser.add_argument(
        "--skip-fetch-signature-metadata",
        action="store_true",
        help="Do not fetch missing signature metadata from Treeherder.",
    )
    parser.add_argument(
        "--skip-fetch-job-durations",
        action="store_true",
        help=(
            "Do not fetch missing per-signature job runtimes from Treeherder. "
            "Missing durations will fall back to signature-group or median values."
        ),
    )
    parser.add_argument(
        "--job-duration-samples",
        type=int,
        default=3,
        help=(
            "Maximum number of Treeherder jobs to sample per signature when "
            "fetching missing job runtimes."
        ),
    )
    parser.add_argument(
        "--keep-desc-in-revision-data",
        action="store_true",
        help="Keep commit descriptions in per_revision_perf_data.jsonl.",
    )
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Python logging level.",
    )
    return parser.parse_args()


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )


def log_counter(title: str, stats: Counter[str]) -> None:
    if not stats:
        logger.info("%s: no counters recorded.", title)
        return
    logger.info("%s:", title)
    for key, value in sorted(stats.items()):
        logger.info("  %s=%s", key, value)


def parse_int_csv(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [int(part) for part in value]


def parse_status_csv(value: str | None) -> set[int] | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None

    statuses: set[int] = set()
    for part in value.split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name.isdigit():
            statuses.add(int(name))
            continue
        if name not in ALERT_SUMMARY_STATUS:
            allowed = ", ".join(sorted(ALERT_SUMMARY_STATUS))
            raise argparse.ArgumentTypeError(
                f"unknown alert status {name!r}; expected one of: {allowed}"
            )
        statuses.add(ALERT_SUMMARY_STATUS[name])
    return statuses or None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSON at %s:%d: %s", path, line_num, exc)
                continue
            if isinstance(value, dict):
                yield value


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record))
            f.write("\n")
            count += 1
    return count


def parse_alerts(raw_alerts: Any) -> list[dict[str, Any]]:
    if isinstance(raw_alerts, list):
        return [alert for alert in raw_alerts if isinstance(alert, dict)]
    if isinstance(raw_alerts, dict):
        return [raw_alerts]

    raw_text = "" if raw_alerts is None else str(raw_alerts).strip()
    if not raw_text:
        return []
    try:
        parsed = ast.literal_eval(raw_text)
    except (SyntaxError, ValueError):
        return []
    if isinstance(parsed, list):
        return [alert for alert in parsed if isinstance(alert, dict)]
    if isinstance(parsed, dict):
        return [parsed]
    return []


def parse_datetime(value: Any) -> datetime | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def parse_hg_datetime(record: Mapping[str, Any]) -> datetime | None:
    raw_date = record.get("date")
    if not isinstance(raw_date, list) or not raw_date:
        return None
    try:
        return datetime.fromtimestamp(float(raw_date[0]), tz=UTC)
    except Exception:
        return None


def json_dumps_compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def load_csv_by_id(path: Path, id_field: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    if not path.exists():
        return rows

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = (row.get(id_field) or "").strip()
            if not row_id:
                continue
            rows[row_id] = dict(row)
    return rows


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def use_existing(path: Path, *, overwrite_existing: bool) -> bool:
    return path.exists() and not overwrite_existing


def write_csv(path: Path, rows: list[dict[str, Any]], preferred_fields: list[str]) -> None:
    ensure_dir(path.parent)
    fields = list(preferred_fields)
    seen = set(fields)
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_cell(row.get(field, "")) for field in fields})


def csv_cell(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return repr(value)
    if value is None:
        return ""
    return value


def copy_static_artifacts(
    source_dir: Path,
    output_dir: Path,
    *,
    overwrite_existing: bool,
) -> None:
    for name in STATIC_COPY_FILES:
        src = source_dir / name
        dst = output_dir / name
        if use_existing(dst, overwrite_existing=overwrite_existing):
            logger.info("Using existing static artifact: %s", dst)
            continue
        if src.exists():
            shutil.copy2(src, dst)
            logger.info("Copied static artifact: %s -> %s", src, dst)
        else:
            logger.warning("Static artifact missing in source dataset: %s", src)


def fetch_treeherder_client() -> Any:
    try:
        from thclient import TreeherderClient  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'thclient'. Install treeherder-client or rerun "
            "with the relevant --skip-fetch-* flag."
        ) from exc
    return TreeherderClient()


def next_page_from_url(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    page_values = parse_qs(parsed.query).get("page")
    if not page_values:
        return None
    return page_values[0]


def fetch_alert_summaries(
    output_path: Path,
    *,
    lookback_days: int,
    max_pages: int,
) -> list[dict[str, Any]]:
    logger.info(
        "Fetching Treeherder alert summaries: lookback_days=%d, max_pages=%d, output=%s",
        lookback_days,
        max_pages,
        output_path,
    )
    client = fetch_treeherder_client()
    params: dict[str, Any] = {
        "page": 1,
        "hide_related_and_invalid": True,
    }
    threshold = (
        datetime.now(UTC).timestamp() - lookback_days * 24 * 60 * 60
        if lookback_days > 0
        else None
    )
    rows: list[dict[str, Any]] = []
    pages = 0

    while True:
        pages += 1
        if max_pages > 0 and pages > max_pages:
            logger.info("Reached alert-summary page limit (%d); stopping fetch.", max_pages)
            break
        payload = client._get_json("performance/alertsummary", **params)
        page_rows = payload.get("results", []) if isinstance(payload, dict) else []
        page_rows = [row for row in page_rows if isinstance(row, dict)]
        if not page_rows:
            logger.info("Alert-summary page %s returned no rows; stopping fetch.", params["page"])
            break
        rows.extend(page_rows)
        logger.info(
            "Fetched alert-summary page %s: page_rows=%d, total_rows=%d",
            params["page"],
            len(page_rows),
            len(rows),
        )

        if threshold is not None:
            last_push_timestamp = page_rows[-1].get("push_timestamp")
            try:
                if float(last_push_timestamp) < threshold:
                    logger.info(
                        "Alert-summary lookback reached on page %s: oldest_push_timestamp=%s, "
                        "threshold=%s; stopping fetch.",
                        params["page"],
                        last_push_timestamp,
                        threshold,
                    )
                    break
            except Exception:
                logger.debug(
                    "Could not parse last alert-summary push_timestamp=%r on page %s.",
                    last_push_timestamp,
                    params["page"],
                )
                pass

        next_page = next_page_from_url(payload.get("next"))
        if next_page is None:
            logger.info("Treeherder returned no next alert-summary page; stopping fetch.")
            break
        params["page"] = next_page

    preferred_fields = [
        "id",
        "push_id",
        "prev_push_id",
        "original_revision",
        "original_prev_push_revision",
        "revision",
        "created",
        "push_timestamp",
        "repository",
        "framework",
        "alerts",
        "related_alerts",
        "status",
        "bug_number",
    ]
    write_csv(output_path, rows, preferred_fields)
    logger.info(
        "Finished alert-summary fetch: pages=%d, rows=%d, output=%s",
        min(pages, max_pages) if max_pages > 0 else pages,
        len(rows),
        output_path,
    )
    return rows


def merge_alert_summaries(source_dir: Path, output_dir: Path, fresh_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    source_rows = load_csv_by_id(source_dir / "alert_summaries.csv", "id")
    merged: dict[str, dict[str, Any]] = dict(source_rows)
    for row in fresh_rows:
        row_id = str(row.get("id") or "").strip()
        if row_id:
            merged[row_id] = dict(row)

    rows = list(merged.values())
    rows.sort(key=alert_sort_key)
    preferred_fields = [
        "id",
        "push_id",
        "prev_push_id",
        "original_revision",
        "original_prev_push_revision",
        "revision",
        "created",
        "first_triaged",
        "triage_due_date",
        "repository",
        "framework",
        "alerts",
        "related_alerts",
        "status",
        "bug_number",
        "push_timestamp",
    ]
    write_csv(output_dir / "alert_summaries.csv", rows, preferred_fields)
    logger.info(
        "Merged alert summaries: source_rows=%d, fresh_rows=%d, merged_rows=%d, output=%s",
        len(source_rows),
        len(fresh_rows),
        len(rows),
        output_dir / "alert_summaries.csv",
    )
    return [stringify_csv_row(row) for row in rows]


def stringify_csv_row(row: Mapping[str, Any]) -> dict[str, str]:
    return {str(key): str(csv_cell(value)) for key, value in row.items()}


def alert_sort_key(row: Mapping[str, Any]) -> tuple[datetime, int]:
    created = parse_datetime(row.get("created"))
    if created is None:
        try:
            created = datetime.fromtimestamp(float(row.get("push_timestamp")), tz=UTC)
        except Exception:
            created = datetime.max.replace(tzinfo=UTC)
    try:
        row_id = int(str(row.get("id") or "0"))
    except ValueError:
        row_id = 0
    return created, row_id


def maybe_export_commits(
    output_dir: Path,
    autoland_repo: Path,
    *,
    skip_export: bool,
    pull_first: bool,
    source_dir: Path,
    overwrite_existing: bool,
) -> None:
    output_path = output_dir / "all_commits.jsonl"
    if use_existing(output_path, overwrite_existing=overwrite_existing):
        logger.info("Using existing commit export: %s; skipping commit export.", output_path)
        return

    if skip_export:
        source_path = source_dir / "all_commits.jsonl"
        if not source_path.exists():
            raise FileNotFoundError(f"Cannot skip commit export; missing {source_path}")
        shutil.copy2(source_path, output_path)
        logger.info("Copied commit export from source dataset: %s -> %s", source_path, output_path)
        return

    ensure_dir(autoland_repo.parent)
    if not autoland_repo.exists():
        raise FileNotFoundError(
            f"Local Autoland checkout not found at {autoland_repo}. "
            "Create it first, or run the older fetch_all_commit.py once, or "
            "pass --skip-fetch-commits to copy source-dir all_commits.jsonl."
        )
    if pull_first:
        logger.info("Pulling Autoland checkout before commit export: %s", autoland_repo)
        run(["hg", "pull", "-u"], cwd=autoland_repo)

    logger.info("Exporting Autoland commits from local checkout: %s", autoland_repo)
    result = subprocess.run(
        ["hg", "log", "-Tjson", "-r", "all()"],
        cwd=autoland_repo,
        check=True,
        stdout=subprocess.PIPE,
    )
    changesets = json.loads(result.stdout.decode("utf-8"))
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as f:
        for cs in changesets:
            f.write(
                json.dumps(
                    {
                        "node": cs.get("node"),
                        "desc": cs.get("desc", ""),
                        "date": cs.get("date", ""),
                        "parents": cs.get("parents", ""),
                    },
                    ensure_ascii=False,
                )
            )
            f.write("\n")
    logger.info("Exported Autoland commits: count=%d, output=%s", len(changesets), output_path)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    try:
        logger.info("Running command: %s%s", " ".join(cmd), f" (cwd={cwd})" if cwd else "")
        subprocess.run(cmd, cwd=cwd, check=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError("Missing required executable: hg") from exc


def load_commit_graph(path: Path) -> CommitGraph:
    records: list[dict[str, Any]] = []
    node_to_index: dict[str, int] = {}
    parents_by_node: dict[str, list[str]] = {}

    for record in iter_jsonl(path):
        node = record.get("node")
        if not isinstance(node, str) or not node:
            continue
        if node in node_to_index:
            raise ValueError(f"Duplicate commit node in {path}: {node}")

        current_index = len(records)
        parents = [
            parent
            for parent in record.get("parents", [])
            if isinstance(parent, str) and parent and parent != NULL_NODE
        ]
        for parent in parents:
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
        parents_by_node[node] = parents
        records.append(record)

    logger.info("Loaded commit graph: commits=%d, source=%s", len(records), path)
    return CommitGraph(records=records, node_to_index=node_to_index, parents_by_node=parents_by_node)


def derive_failing_signature_csvs(
    alert_rows: list[dict[str, str]],
    output_dir: Path,
    *,
    included_statuses: set[int] | None,
    excluded_framework_ids: set[int],
    overwrite_existing: bool,
) -> tuple[set[int], dict[int, dict[str, Any]], set[int]]:
    all_output = output_dir / "alert_summary_fail_perf_sigs.csv"
    filtered_output = output_dir / "alert_summary_fail_perf_sigs_no_fw_2_6_18.csv"
    logger.info(
        "Deriving failing-signature CSVs: alert_rows=%d, included_statuses=%s, "
        "excluded_framework_ids=%s",
        len(alert_rows),
        sorted(included_statuses) if included_statuses is not None else "any",
        sorted(excluded_framework_ids),
    )
    if (
        use_existing(all_output, overwrite_existing=overwrite_existing)
        and use_existing(filtered_output, overwrite_existing=overwrite_existing)
    ):
        signature_ids = load_signature_ids_from_fail_sigs_csv(all_output)
        signature_alert_metadata = collect_signature_alert_metadata(alert_rows)
        logger.info(
            "Using existing failing-signature CSVs: all=%s, filtered=%s, "
            "unique_failing_signatures=%d",
            all_output,
            filtered_output,
            len(signature_ids),
        )
        return signature_ids, signature_alert_metadata, set()

    all_rows: list[dict[str, Any]] = []
    filtered_rows: list[dict[str, Any]] = []
    signature_alert_metadata: dict[int, dict[str, Any]] = {}
    all_signature_ids: set[int] = set()
    framework_filtered_ids: set[int] = set()
    stats: Counter[str] = Counter()

    for row in alert_rows:
        stats["alert_rows"] += 1
        summary_id_text = (row.get("id") or "").strip()
        if not summary_id_text:
            stats["missing_summary_id"] += 1
            continue

        try:
            status = int(str(row.get("status") or "").strip())
        except ValueError:
            status = None
        if included_statuses is not None and status not in included_statuses:
            stats["status_filtered"] += 1
            continue

        fail_signature_ids: set[int] = set()
        for alert in parse_alerts(row.get("alerts")):
            if not alert.get("is_regression"):
                continue
            series_signature = alert.get("series_signature") or {}
            if not isinstance(series_signature, dict):
                continue
            try:
                signature_id = int(series_signature["id"])
            except Exception:
                continue
            fail_signature_ids.add(signature_id)
            all_signature_ids.add(signature_id)
            metadata = signature_alert_metadata.setdefault(signature_id, {})
            if metadata.get("platform") is None:
                metadata["platform"] = (
                    alert.get("platform")
                    or series_signature.get("machine_platform")
                )
            if metadata.get("alert_threshold") is None:
                metadata["alert_threshold"] = alert.get("alert_threshold")
            if metadata.get("lower_is_better") is None:
                metadata["lower_is_better"] = alert.get("lower_is_better")

        if not fail_signature_ids:
            stats["no_regression_signatures"] += 1
            continue

        output_row = {
            "alert_summary_id": summary_id_text,
            "revision": row.get("revision", ""),
            "fail_perf_sig_ids": json.dumps(sorted(fail_signature_ids)),
            "num_fail_perf_sig_ids": len(fail_signature_ids),
        }
        all_rows.append(output_row)

        framework = parse_optional_int(row.get("framework"))
        if framework in excluded_framework_ids:
            framework_filtered_ids.add(int(summary_id_text))
            continue
        filtered_rows.append(output_row)

    fields = [
        "alert_summary_id",
        "revision",
        "fail_perf_sig_ids",
        "num_fail_perf_sig_ids",
    ]
    write_csv(all_output, all_rows, fields)
    write_csv(filtered_output, filtered_rows, fields)
    logger.info(
        "Derived failing-signature CSVs: all_rows=%d, framework_filtered_rows=%d, "
        "unique_failing_signatures=%d, all_output=%s, filtered_output=%s",
        len(all_rows),
        len(filtered_rows),
        len(all_signature_ids),
        all_output,
        filtered_output,
    )
    log_counter("Failing-signature derivation counters", stats)
    return all_signature_ids, signature_alert_metadata, framework_filtered_ids


def load_signature_ids_from_fail_sigs_csv(path: Path) -> set[int]:
    signature_ids: set[int] = set()
    for row_num, row in enumerate(load_csv_rows(path), start=2):
        raw_sig_ids = (row.get("fail_perf_sig_ids") or "").strip()
        if not raw_sig_ids:
            continue
        try:
            parsed = json.loads(raw_sig_ids)
        except json.JSONDecodeError as exc:
            logger.warning("Invalid fail_perf_sig_ids at %s:%d: %s", path, row_num, exc)
            continue
        if not isinstance(parsed, list):
            continue
        for value in parsed:
            signature_id = parse_optional_int(value)
            if signature_id is not None:
                signature_ids.add(signature_id)
    return signature_ids


def collect_signature_alert_metadata(
    alert_rows: list[dict[str, str]],
) -> dict[int, dict[str, Any]]:
    metadata_by_signature: dict[int, dict[str, Any]] = {}
    for row in alert_rows:
        for alert in parse_alerts(row.get("alerts")):
            if not alert.get("is_regression"):
                continue
            series_signature = alert.get("series_signature") or {}
            if not isinstance(series_signature, dict):
                continue
            signature_id = parse_optional_int(series_signature.get("id"))
            if signature_id is None:
                continue
            metadata = metadata_by_signature.setdefault(signature_id, {})
            if metadata.get("platform") is None:
                metadata["platform"] = (
                    alert.get("platform")
                    or series_signature.get("machine_platform")
                )
            if metadata.get("alert_threshold") is None:
                metadata["alert_threshold"] = alert.get("alert_threshold")
            if metadata.get("lower_is_better") is None:
                metadata["lower_is_better"] = alert.get("lower_is_better")
    return metadata_by_signature


def parse_optional_int(value: Any) -> int | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def load_per_signature_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for record in iter_jsonl(path):
        try:
            signature_id = int(record["signature_id"])
        except Exception:
            continue
        records[signature_id] = {
            "signature_id": signature_id,
            "filter_stats": dict(record.get("filter_stats") or {}),
            "perf_measurement_data": [
                row
                for row in record.get("perf_measurement_data", [])
                if isinstance(row, dict)
            ],
        }
    return records


def fetch_measurements_for_signatures(
    signature_ids: list[int],
    *,
    lookback_days: int,
    limit_signatures: int,
    fetch_summary: bool,
    fetch_replicates: bool,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    if limit_signatures > 0:
        logger.info(
            "Limiting fresh measurement fetch: requested_signatures=%d, limit_signatures=%d",
            len(signature_ids),
            limit_signatures,
        )
        signature_ids = signature_ids[:limit_signatures]
    if not fetch_summary and not fetch_replicates:
        logger.info("Fresh measurement fetch not needed; both output caches already exist.")
        return {}, {}

    logger.info(
        "Fetching fresh Treeherder measurements: signatures=%d, lookback_days=%d, "
        "fetch_summary=%s, fetch_replicates=%s",
        len(signature_ids),
        lookback_days,
        fetch_summary,
        fetch_replicates,
    )
    client = fetch_treeherder_client()
    interval_seconds = max(1, lookback_days) * 24 * 60 * 60
    summary: dict[int, dict[str, Any]] = {}
    replicates: dict[int, dict[str, Any]] = {}

    for idx, signature_id in enumerate(signature_ids, start=1):
        logger.info(
            "Fetching measurements for signature %d (%d/%d).",
            signature_id,
            idx,
            len(signature_ids),
        )
        if fetch_summary:
            summary[signature_id] = fetch_one_signature_measurements(
                client,
                signature_id,
                interval_seconds=interval_seconds,
                replicates=False,
            )
        if fetch_replicates:
            replicates[signature_id] = fetch_one_signature_measurements(
                client,
                signature_id,
                interval_seconds=interval_seconds,
                replicates=True,
            )
    logger.info(
        "Finished fresh measurement fetch: summary_signatures=%d, summary_measurements=%d, "
        "replicate_signatures=%d, replicate_measurements=%d",
        len(summary),
        count_measurements(summary),
        len(replicates),
        count_measurements(replicates),
    )
    return summary, replicates


def count_measurements(records: Mapping[int, Mapping[str, Any]]) -> int:
    return sum(
        len(record.get("perf_measurement_data", []))
        for record in records.values()
    )


def fetch_one_signature_measurements(
    client: Any,
    signature_id: int,
    *,
    interval_seconds: int,
    replicates: bool,
) -> dict[str, Any]:
    params = {
        "repository": REPOSITORY,
        "signature": signature_id,
        "interval": interval_seconds,
        "all_data": True,
        "replicates": replicates,
    }
    try:
        payload = client._get_json("performance/summary", **params)
    except (Timeout, RequestException) as exc:
        logger.warning(
            "Request failed for signature %d (replicates=%s): %s",
            signature_id,
            replicates,
            exc,
        )
        payload = []
    except Exception as exc:
        logger.warning(
            "Unexpected failure for signature %d (replicates=%s): %s",
            signature_id,
            replicates,
            exc,
        )
        payload = []

    measurements: list[dict[str, Any]] = []
    if payload and isinstance(payload, list) and isinstance(payload[0], dict):
        data = payload[0].get("data", [])
        if isinstance(data, list):
            measurements = [row for row in data if isinstance(row, dict)]

    return {
        "signature_id": signature_id,
        "filter_stats": {
            "raw_measurements": len(measurements),
            "kept_measurements": len(measurements),
            "fresh_fetch": True,
        },
        "perf_measurement_data": measurements,
    }


def merge_signature_measurements(
    old_records: dict[int, dict[str, Any]],
    fresh_records: dict[int, dict[str, Any]],
    output_path: Path,
) -> dict[int, dict[str, Any]]:
    all_signature_ids = sorted(set(old_records) | set(fresh_records))
    merged: dict[int, dict[str, Any]] = {}
    logger.info(
        "Merging per-signature measurements: old_signatures=%d, fresh_signatures=%d, "
        "merged_signatures=%d, output=%s",
        len(old_records),
        len(fresh_records),
        len(all_signature_ids),
        output_path,
    )

    for signature_id in all_signature_ids:
        measurements: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        for source in (old_records.get(signature_id), fresh_records.get(signature_id)):
            if not source:
                continue
            for measurement in source.get("perf_measurement_data", []):
                if not isinstance(measurement, dict):
                    continue
                key = measurement_dedupe_key(measurement)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                measurements.append(measurement)

        measurements.sort(key=measurement_sort_key)
        merged[signature_id] = {
            "signature_id": signature_id,
            "filter_stats": {
                "old_measurements": len(old_records.get(signature_id, {}).get("perf_measurement_data", [])),
                "fresh_measurements": len(fresh_records.get(signature_id, {}).get("perf_measurement_data", [])),
                "merged_measurements": len(measurements),
            },
            "perf_measurement_data": measurements,
        }

    write_jsonl(output_path, (merged[sig] for sig in all_signature_ids))
    logger.info(
        "Wrote merged per-signature measurements: signatures=%d, measurements=%d, output=%s",
        len(merged),
        count_measurements(merged),
        output_path,
    )
    return merged


def measurement_dedupe_key(measurement: Mapping[str, Any]) -> str:
    key_fields = {
        "id": measurement.get("id"),
        "job_id": measurement.get("job_id"),
        "revision": measurement.get("revision"),
        "push_timestamp": measurement.get("push_timestamp"),
        "submit_time": measurement.get("submit_time"),
        "value": measurement.get("value"),
    }
    return json_dumps_compact(key_fields)


def measurement_sort_key(measurement: Mapping[str, Any]) -> tuple[datetime, str]:
    submit_time = parse_datetime(measurement.get("submit_time"))
    if submit_time is None:
        submit_time = parse_datetime(measurement.get("push_timestamp"))
    if submit_time is None:
        submit_time = datetime.min.replace(tzinfo=UTC)
    return submit_time, json_dumps_compact(measurement)


def load_signature_to_group_id(path: Path) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for record in iter_jsonl(path):
        group_id = parse_optional_int(record.get("Sig_group_id"))
        signatures = record.get("signatures")
        if group_id is None or not isinstance(signatures, list):
            continue
        for raw_signature_id in signatures:
            signature_id = parse_optional_int(raw_signature_id)
            if signature_id is not None:
                mapping[signature_id] = group_id
    return mapping


def load_group_durations(path: Path) -> dict[int, float]:
    durations: dict[int, float] = {}
    if not path.exists():
        return durations
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            group_id = parse_optional_int(row.get("signature_group_id"))
            if group_id is None:
                group_id = parse_optional_int(row.get("Sig_group_id"))
            try:
                duration = float(row.get("duration_minutes", ""))
            except ValueError:
                continue
            if group_id is not None and math.isfinite(duration) and duration > 0:
                durations[group_id] = duration
    return durations


def load_existing_signature_info(path: Path) -> dict[int, dict[str, Any]]:
    info: dict[int, dict[str, Any]] = {}
    for record in iter_jsonl(path):
        signature_id = parse_optional_int(record.get("signature_id"))
        if signature_id is not None:
            info[signature_id] = dict(record)
    return info


def infer_replicate_count(signature_id: int, record: dict[str, Any]) -> int:
    measurements = [
        row
        for row in record.get("perf_measurement_data", [])
        if isinstance(row, dict) and row.get("id") is not None
    ]
    if not measurements:
        return 1
    measurements.sort(key=measurement_sort_key)
    newest = measurements[-1]
    newest_id = newest.get("id")
    newest_count = sum(1 for row in measurements if row.get("id") == newest_id)
    return max(1, newest_count)


def fetch_signature_metadata(
    signature_ids: list[int],
    *,
    skip_fetch: bool,
) -> dict[int, dict[str, Any]]:
    if skip_fetch:
        logger.info(
            "Skipping missing signature metadata fetch by request: signatures=%d",
            len(signature_ids),
        )
        return {}
    if not signature_ids:
        logger.info("No missing signature metadata to fetch.")
        return {}
    logger.info("Fetching missing signature metadata: signatures=%d", len(signature_ids))
    client = fetch_treeherder_client()
    metadata: dict[int, dict[str, Any]] = {}
    interval_seconds = 30 * 24 * 60 * 60
    for idx, signature_id in enumerate(signature_ids, start=1):
        logger.info(
            "Fetching metadata for signature %d (%d/%d).",
            signature_id,
            idx,
            len(signature_ids),
        )
        metadata[signature_id] = fetch_one_signature_metadata(
            client,
            signature_id,
            interval_seconds=interval_seconds,
        )
    logger.info("Finished missing signature metadata fetch: signatures=%d", len(metadata))
    return metadata


def fetch_one_signature_metadata(
    client: Any,
    signature_id: int,
    *,
    interval_seconds: int,
) -> dict[str, Any]:
    params = {
        "repository": REPOSITORY,
        "signature": signature_id,
        "interval": interval_seconds,
        "all_data": True,
        "replicates": False,
    }
    empty = {
        "lower_is_better": None,
        "alert_threshold": None,
        "platform": None,
    }
    try:
        payload = client._get_json("performance/summary", **params)
    except (Timeout, RequestException, Exception) as exc:
        logger.warning("Failed to fetch metadata for signature %d: %s", signature_id, exc)
        return dict(empty)
    if not payload or not isinstance(payload, list) or not isinstance(payload[0], dict):
        return dict(empty)
    return {
        "lower_is_better": payload[0].get("lower_is_better"),
        "alert_threshold": payload[0].get("alert_threshold"),
        "platform": payload[0].get("platform"),
    }


def load_candidate_job_ids(
    signature_id: int,
    *records: dict[int, dict[str, Any]],
    max_jobs: int,
) -> list[int]:
    job_ids: list[int] = []
    seen: set[int] = set()
    measurements: list[dict[str, Any]] = []

    for records_by_signature in records:
        record = records_by_signature.get(signature_id)
        if not record:
            continue
        for measurement in record.get("perf_measurement_data", []):
            if isinstance(measurement, dict):
                measurements.append(measurement)

    measurements.sort(key=measurement_sort_key, reverse=True)
    for measurement in measurements:
        job_id = parse_optional_int(measurement.get("job_id"))
        if job_id is None or job_id in seen:
            continue
        seen.add(job_id)
        job_ids.append(job_id)
        if len(job_ids) >= max_jobs:
            break
    return job_ids


def fetch_recent_job_ids_for_signature(
    client: Any,
    signature_id: int,
    *,
    interval_seconds: int,
    max_jobs: int,
) -> list[int]:
    params = {
        "repository": REPOSITORY,
        "signature": signature_id,
        "interval": interval_seconds,
        "all_data": True,
        "replicates": False,
    }
    try:
        payload = client._get_json("performance/summary", **params)
    except (Timeout, RequestException) as exc:
        logger.warning("Failed to fetch jobs for signature %d: %s", signature_id, exc)
        return []
    except Exception as exc:
        logger.warning("Unexpected job fetch failure for signature %d: %s", signature_id, exc)
        return []

    if not payload or not isinstance(payload, list) or not isinstance(payload[0], dict):
        return []

    data = payload[0].get("data", [])
    if not isinstance(data, list):
        return []

    seen: set[int] = set()
    job_ids: list[int] = []
    measurements = [row for row in data if isinstance(row, dict)]
    measurements.sort(key=measurement_sort_key, reverse=True)
    for measurement in measurements:
        job_id = parse_optional_int(measurement.get("job_id"))
        if job_id is None or job_id in seen:
            continue
        seen.add(job_id)
        job_ids.append(job_id)
        if len(job_ids) >= max_jobs:
            break
    return job_ids


def fetch_job_duration_minutes(client: Any, job_id: int) -> float | None:
    try:
        jobs = client.get_jobs(REPOSITORY, id=job_id)
    except Exception as exc:
        logger.warning("Failed to fetch job details for job_id=%d: %s", job_id, exc)
        return None

    if not jobs or not isinstance(jobs, list) or not isinstance(jobs[0], dict):
        return None

    job = jobs[0]
    try:
        start_timestamp = float(job["start_timestamp"])
        end_timestamp = float(job["end_timestamp"])
    except (KeyError, TypeError, ValueError):
        return None

    duration_minutes = (end_timestamp - start_timestamp) / 60.0
    if math.isfinite(duration_minutes) and duration_minutes > 0:
        return duration_minutes
    return None


def fetch_missing_job_durations(
    signature_ids: list[int],
    *,
    summary_records: dict[int, dict[str, Any]],
    replicate_records: dict[int, dict[str, Any]],
    lookback_days: int,
    sample_count: int,
    skip_fetch: bool,
) -> dict[int, float]:
    if skip_fetch:
        logger.info(
            "Skipping missing job duration fetch by request: signatures=%d",
            len(signature_ids),
        )
        return {}
    if not signature_ids:
        logger.info("No missing job durations to fetch.")
        return {}

    logger.info(
        "Fetching missing job durations: signatures=%d, lookback_days=%d, samples_per_signature=%d",
        len(signature_ids),
        lookback_days,
        sample_count,
    )
    client = fetch_treeherder_client()
    interval_seconds = max(1, lookback_days) * 24 * 60 * 60
    sample_count = max(1, sample_count)
    durations: dict[int, float] = {}

    for idx, signature_id in enumerate(signature_ids, start=1):
        logger.info(
            "Fetching job runtime for signature %d (%d/%d).",
            signature_id,
            idx,
            len(signature_ids),
        )
        job_ids = load_candidate_job_ids(
            signature_id,
            summary_records,
            replicate_records,
            max_jobs=sample_count,
        )
        if not job_ids:
            job_ids = fetch_recent_job_ids_for_signature(
                client,
                signature_id,
                interval_seconds=interval_seconds,
                max_jobs=sample_count,
            )

        sampled_durations = [
            duration
            for job_id in job_ids
            if (duration := fetch_job_duration_minutes(client, job_id)) is not None
        ]
        if sampled_durations:
            durations[signature_id] = statistics.mean(sampled_durations)
        else:
            logger.warning(
                "Could not compute job runtime for signature %d; will use fallback duration.",
                signature_id,
            )

    logger.info(
        "Finished missing job duration fetch: fetched_durations=%d, requested_signatures=%d",
        len(durations),
        len(signature_ids),
    )
    return durations


def build_signature_info(
    *,
    output_dir: Path,
    source_dir: Path,
    needed_signature_ids: set[int],
    summary_records: dict[int, dict[str, Any]],
    replicate_records: dict[int, dict[str, Any]],
    alert_metadata: dict[int, dict[str, Any]],
    skip_fetch_metadata: bool,
    skip_fetch_job_durations: bool,
    job_duration_samples: int,
    job_duration_lookback_days: int,
) -> dict[int, dict[str, Any]]:
    logger.info(
        "Building per-signature info: needed_signatures=%d, summary_records=%d, "
        "replicate_records=%d",
        len(needed_signature_ids),
        len(summary_records),
        len(replicate_records),
    )
    existing = load_existing_signature_info(source_dir / "per_sig_perf_data_info.jsonl")
    signature_to_group = load_signature_to_group_id(output_dir / "sig_groups.jsonl")
    group_durations = load_group_durations(output_dir / "sig_group_job_durations.csv")

    duration_values = [
        float(record["job_duration"])
        for record in existing.values()
        if is_positive_number(record.get("job_duration"))
    ]
    duration_values.extend(group_durations.values())
    default_duration = (
        statistics.median(duration_values)
        if duration_values
        else DEFAULT_JOB_DURATION_MINUTES
    )
    logger.info(
        "Loaded signature info inputs: existing_info=%d, signature_group_mappings=%d, "
        "group_durations=%d, default_job_duration=%.3f",
        len(existing),
        len(signature_to_group),
        len(group_durations),
        default_duration,
    )

    metadata_missing_ids = sorted(
        sig for sig in needed_signature_ids if sig not in existing
    )
    logger.info("Signature metadata missing count: %d", len(metadata_missing_ids))
    fetched_metadata = fetch_signature_metadata(
        metadata_missing_ids,
        skip_fetch=skip_fetch_metadata,
    )
    duration_missing_ids = sorted(
        signature_id
        for signature_id in needed_signature_ids
        if not (
            existing.get(signature_id)
            and is_positive_number(existing[signature_id].get("job_duration"))
        )
        and not (
            (group_id := signature_to_group.get(signature_id)) is not None
            and group_id in group_durations
        )
    )
    logger.info("Signature job duration missing count: %d", len(duration_missing_ids))
    fetched_job_durations = fetch_missing_job_durations(
        duration_missing_ids,
        summary_records=summary_records,
        replicate_records=replicate_records,
        lookback_days=job_duration_lookback_days,
        sample_count=job_duration_samples,
        skip_fetch=skip_fetch_job_durations,
    )

    output_records: list[dict[str, Any]] = []
    result: dict[int, dict[str, Any]] = {}
    stats: Counter[str] = Counter()
    for signature_id in sorted(needed_signature_ids):
        existing_record = existing.get(signature_id)
        alert_record = alert_metadata.get(signature_id, {})
        fetched_record = fetched_metadata.get(signature_id, {})

        group_id = signature_to_group.get(signature_id)
        job_duration = None
        if existing_record and is_positive_number(existing_record.get("job_duration")):
            job_duration = float(existing_record["job_duration"])
            stats["duration_existing"] += 1
        elif signature_id in fetched_job_durations:
            job_duration = fetched_job_durations[signature_id]
            stats["duration_fetched"] += 1
        elif group_id is not None and group_id in group_durations:
            job_duration = group_durations[group_id]
            stats["duration_group"] += 1
        else:
            job_duration = float(default_duration)
            stats["duration_default"] += 1

        replicate_count = None
        if existing_record and parse_optional_int(existing_record.get("replicate_counts")):
            replicate_count = parse_optional_int(existing_record.get("replicate_counts"))
            stats["replicate_count_existing"] += 1
        else:
            replicate_count = infer_replicate_count(
                signature_id,
                replicate_records.get(signature_id, {"perf_measurement_data": []}),
            )
            stats["replicate_count_inferred"] += 1

        lower_is_better = first_present(
            existing_record,
            fetched_record,
            alert_record,
            key="lower_is_better",
        )
        alert_threshold = first_present(
            existing_record,
            fetched_record,
            alert_record,
            key="alert_threshold",
        )
        if alert_threshold is None:
            alert_threshold = DEFAULT_ALERT_THRESHOLD
            stats["alert_threshold_default"] += 1
        platform = first_present(
            existing_record,
            fetched_record,
            alert_record,
            key="platform",
        )
        if platform is None:
            stats["platform_missing"] += 1

        record = {
            "signature_id": signature_id,
            "replicate_counts": replicate_count or 1,
            "job_duration": job_duration,
            "lower_is_better": lower_is_better,
            "alert_threshold": alert_threshold,
            "platform": platform,
        }
        output_records.append(record)
        result[signature_id] = record

    output_path = output_dir / "per_sig_perf_data_info.jsonl"
    write_jsonl(output_path, output_records)
    logger.info(
        "Wrote per-signature info: signatures=%d, output=%s",
        len(output_records),
        output_path,
    )
    log_counter("Per-signature info counters", stats)
    return result


def is_positive_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def first_present(*records: Mapping[str, Any] | None, key: str) -> Any:
    for record in records:
        if not record:
            continue
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def write_per_revision_perf_data(
    *,
    output_dir: Path,
    commit_graph: CommitGraph,
    summary_records: dict[int, dict[str, Any]],
    replicate_records: dict[int, dict[str, Any]],
    keep_desc: bool,
) -> None:
    logger.info(
        "Building per-revision performance data: commits=%d, summary_signatures=%d, "
        "replicate_signatures=%d, keep_desc=%s",
        len(commit_graph.records),
        len(summary_records),
        len(replicate_records),
        keep_desc,
    )
    samples_by_revision: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for replicate, records in ((False, summary_records), (True, replicate_records)):
        for signature_id, record in records.items():
            for sample in record.get("perf_measurement_data", []):
                if not isinstance(sample, dict):
                    continue
                revision = sample.get("revision")
                if not isinstance(revision, str):
                    continue
                augmented = dict(sample)
                augmented["signature_id"] = signature_id
                augmented["replicate"] = replicate
                samples_by_revision[revision].append(augmented)

    output_path = output_dir / "per_revision_perf_data.jsonl"
    commits_with_samples = 0
    total_samples = 0
    with output_path.open("w", encoding="utf-8") as f:
        for commit in commit_graph.records:
            node = commit.get("node")
            if not isinstance(node, str):
                continue
            samples = samples_by_revision.get(node, [])
            if samples:
                commits_with_samples += 1
                total_samples += len(samples)
            output_record = dict(commit)
            if not keep_desc:
                output_record.pop("desc", None)
            output_record["perf_measurement_data"] = samples
            f.write(json.dumps(output_record))
            f.write("\n")
    logger.info(
        "Wrote per-revision performance data: revision_rows=%d, commits_with_samples=%d, "
        "samples=%d, output=%s",
        len(commit_graph.records),
        commits_with_samples,
        total_samples,
        output_path,
    )


def classify_culprit_boundary(
    *,
    commit_graph: CommitGraph,
    good_revision: str,
    bad_revision: str,
    culprit_revision: str,
) -> str | None:
    if not commit_graph.is_ancestor(good_revision, bad_revision):
        return "good_revision_is_not_an_ancestor_of_bad_revision"
    if culprit_revision == good_revision:
        return "culprit_revision_equals_good_revision"

    good_ancestor_culprit = commit_graph.is_ancestor(good_revision, culprit_revision)
    culprit_ancestor_bad = commit_graph.is_ancestor(culprit_revision, bad_revision)
    if good_ancestor_culprit and culprit_ancestor_bad:
        return None
    if not good_ancestor_culprit and culprit_ancestor_bad:
        return "culprit_revision_is_not_after_good_revision"
    if good_ancestor_culprit and not culprit_ancestor_bad:
        return "culprit_revision_is_not_at_or_before_bad_revision"
    return "culprit_revision_is_not_on_good_to_bad_ancestry_range"


def build_regression_rows(
    *,
    alert_rows: list[dict[str, str]],
    commit_graph: CommitGraph,
    signature_info: dict[int, dict[str, Any]],
    included_statuses: set[int] | None,
    excluded_framework_ids: set[int],
    excluded_alert_summary_ids: set[int],
) -> list[dict[str, Any]]:
    logger.info(
        "Building regression rows: alert_rows=%d, included_statuses=%s, "
        "excluded_framework_ids=%s, excluded_alert_summary_ids=%s",
        len(alert_rows),
        sorted(included_statuses) if included_statuses is not None else "any",
        sorted(excluded_framework_ids),
        sorted(excluded_alert_summary_ids),
    )
    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    seen_summary_ids: set[int] = set()

    for row in alert_rows:
        stats["alert_rows_total"] += 1
        summary_id = parse_optional_int(row.get("id"))
        if summary_id is None:
            stats["invalid_summary_id"] += 1
            continue
        if summary_id in excluded_alert_summary_ids:
            stats["excluded_summary_id"] += 1
            continue
        if summary_id in seen_summary_ids:
            stats["duplicate_summary_id"] += 1
            continue
        seen_summary_ids.add(summary_id)

        status = parse_optional_int(row.get("status"))
        if included_statuses is not None and status not in included_statuses:
            stats["status_filtered"] += 1
            continue

        framework = parse_optional_int(row.get("framework"))
        if framework in excluded_framework_ids:
            stats["framework_filtered"] += 1
            continue

        good_revision = (row.get("original_prev_push_revision") or "").strip()
        bad_revision = (row.get("original_revision") or "").strip()
        culprit_revision = (row.get("revision") or "").strip()
        if not good_revision or not bad_revision or not culprit_revision:
            stats["missing_revision_fields"] += 1
            continue

        good_index = commit_graph.node_to_index.get(good_revision)
        bad_index = commit_graph.node_to_index.get(bad_revision)
        culprit_index = commit_graph.node_to_index.get(culprit_revision)
        if good_index is None or bad_index is None or culprit_index is None:
            stats["missing_commit_nodes"] += 1
            continue
        if bad_index <= good_index:
            stats["non_forward_revision_range"] += 1
            continue

        num_candidate_revisions = bad_index - good_index - 1
        if num_candidate_revisions <= 1:
            stats["too_few_candidate_revisions"] += 1
            continue

        culprit_boundary_reason = classify_culprit_boundary(
            commit_graph=commit_graph,
            good_revision=good_revision,
            bad_revision=bad_revision,
            culprit_revision=culprit_revision,
        )
        if culprit_boundary_reason is not None:
            stats["culprit_outside_revision_range"] += 1
            stats[f"culprit_{culprit_boundary_reason}"] += 1
            continue

        failing_sig_records = failing_sig_records_from_alerts(
            parse_alerts(row.get("alerts")),
            signature_info=signature_info,
            stats=stats,
        )
        if not failing_sig_records:
            stats["no_valid_failing_sigs"] += 1
            continue

        created = parse_datetime(row.get("created"))
        for failing_sig in failing_sig_records:
            rows.append(
                {
                    "_created_sort": created,
                    "_bad_index_sort": bad_index,
                    "_summary_id_sort": summary_id,
                    "alert_summary_id": summary_id,
                    "good_revision": good_revision,
                    "bad_revision": bad_revision,
                    "num_candidate_revisions": num_candidate_revisions,
                    "culprit_revision": culprit_revision,
                    "failing_sig": failing_sig,
                }
            )

    rows.sort(key=regression_sort_key)
    for row in rows:
        row.pop("_created_sort", None)
        row.pop("_bad_index_sort", None)
        row.pop("_summary_id_sort", None)

    logger.info("Built regression rows before split: rows=%d", len(rows))
    log_counter("Regression row build counters", stats)
    return rows


def failing_sig_records_from_alerts(
    alerts: list[dict[str, Any]],
    *,
    signature_info: dict[int, dict[str, Any]],
    stats: Counter[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for alert in alerts:
        if not alert.get("is_regression"):
            continue
        series_signature = alert.get("series_signature") or {}
        if not isinstance(series_signature, dict):
            continue
        signature_id = parse_optional_int(series_signature.get("id"))
        if signature_id is None or signature_id in seen:
            continue
        seen.add(signature_id)

        good_value = alert.get("prev_value")
        bad_value = alert.get("new_value")
        if not is_finite_number(good_value) or not is_finite_number(bad_value):
            stats["failing_sigs_missing_numeric_values"] += 1
            continue

        info = signature_info.get(signature_id, {})
        alert_threshold = first_present(info, alert, key="alert_threshold")
        if alert_threshold is None:
            alert_threshold = DEFAULT_ALERT_THRESHOLD
            stats["failing_sigs_alert_threshold_default"] += 1

        platform = first_present(info, alert, series_signature, key="platform")
        if platform is None:
            platform = series_signature.get("machine_platform")
        if platform is None:
            stats["failing_sigs_missing_platform"] += 1

        records.append(
            {
                "signature_id": signature_id,
                "Good_value": good_value,
                "bad_value": bad_value,
                "alert_threshold": alert_threshold,
                "platform": platform,
            }
        )
    return records


def is_finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def regression_sort_key(row: Mapping[str, Any]) -> tuple[datetime, int, int]:
    created = row.get("_created_sort")
    if not isinstance(created, datetime):
        created = datetime.max.replace(tzinfo=UTC)
    return (
        created,
        int(row.get("_bad_index_sort") or 0),
        int(row.get("_summary_id_sort") or 0),
    )


def write_regression_splits(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    eval_fraction: float,
    overwrite_existing: bool,
) -> None:
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError(f"--eval-fraction must be in (0, 1), got {eval_fraction}")

    split_index = int(len(rows) * eval_fraction)
    if rows:
        split_index = max(1, min(len(rows) - 1, split_index))

    eval_rows = rows[:split_index]
    final_rows = rows[split_index:]

    eval_path = output_dir / "perf_bisect_regressions_eval.jsonl"
    final_path = output_dir / "perf_bisect_regressions_final_test.jsonl"
    if use_existing(eval_path, overwrite_existing=overwrite_existing):
        logger.info("Using existing eval regression split: %s; skipping write.", eval_path)
    else:
        write_regression_file(eval_path, eval_rows, starting_regression_id=1)
        logger.info(
            "Wrote eval regression split: rows=%d, output=%s",
            len(eval_rows),
            eval_path,
        )

    if use_existing(final_path, overwrite_existing=overwrite_existing):
        logger.info("Using existing final-test regression split: %s; skipping write.", final_path)
    else:
        write_regression_file(
            final_path,
            final_rows,
            starting_regression_id=len(eval_rows) + 1,
        )
        logger.info(
            "Wrote final-test regression split: rows=%d, output=%s",
            len(final_rows),
            final_path,
        )
    logger.info(
        "Regression split result: total_rows=%d, eval_rows=%d, final_test_rows=%d, "
        "eval_fraction=%.4f",
        len(rows),
        len(eval_rows),
        len(final_rows),
        eval_fraction,
    )


def write_regression_file(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    starting_regression_id: int,
) -> None:
    def with_ids() -> Iterable[dict[str, Any]]:
        for offset, row in enumerate(rows):
            yield {
                "regression_id": starting_regression_id + offset,
                **row,
            }

    write_jsonl(path, with_ids())


def summarize_v2_directory(output_dir: Path) -> None:
    logger.info("Created v2 dataset files:")
    for name in (
        "alert_summaries.csv",
        "alert_summary_fail_perf_sigs.csv",
        "alert_summary_fail_perf_sigs_no_fw_2_6_18.csv",
        "all_commits.jsonl",
        "per_sig_perf_data_summary.jsonl",
        "per_sig_perf_data_replicates.jsonl",
        "per_sig_perf_data_info.jsonl",
        "per_revision_perf_data.jsonl",
        "perf_bisect_regressions_eval.jsonl",
        "perf_bisect_regressions_final_test.jsonl",
    ):
        path = output_dir / name
        if path.exists():
            logger.info("  %s (%s bytes)", path, f"{path.stat().st_size:,}")


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)

    logger.info("Starting Mozilla perf-bisect v2 dataset build.")
    logger.info("Source dataset: %s", source_dir)
    logger.info("Output dataset: %s", output_dir)
    logger.info(
        "Run options: overwrite_existing=%s, skip_fetch_alerts=%s, skip_fetch_commits=%s, "
        "pull_commits=%s, skip_fetch_measurements=%s, skip_fetch_signature_metadata=%s, "
        "skip_fetch_job_durations=%s",
        args.overwrite_existing,
        args.skip_fetch_alerts,
        args.skip_fetch_commits,
        args.pull_commits,
        args.skip_fetch_measurements,
        args.skip_fetch_signature_metadata,
        args.skip_fetch_job_durations,
    )

    logger.info("Step 1/8: Copying reusable static artifacts.")
    copy_static_artifacts(
        source_dir,
        output_dir,
        overwrite_existing=args.overwrite_existing,
    )

    logger.info("Step 2/8: Fetching and merging alert summaries.")
    merged_alerts_path = output_dir / "alert_summaries.csv"
    fresh_alerts_path = output_dir / "alert_summaries_fresh.csv"
    if use_existing(merged_alerts_path, overwrite_existing=args.overwrite_existing):
        alert_rows = load_csv_rows(merged_alerts_path)
        logger.info(
            "Using existing merged alert summaries: rows=%d, path=%s; skipping alert fetch/merge.",
            len(alert_rows),
            merged_alerts_path,
        )
    else:
        fresh_alert_rows: list[dict[str, Any]] = []
        if args.skip_fetch_alerts:
            logger.info("Skipping fresh alert fetch by request.")
        elif use_existing(fresh_alerts_path, overwrite_existing=args.overwrite_existing):
            fresh_alert_rows = load_csv_rows(fresh_alerts_path)
            logger.info(
                "Using existing fresh alert summaries: rows=%d, path=%s; skipping alert fetch.",
                len(fresh_alert_rows),
                fresh_alerts_path,
            )
        else:
            fresh_alert_rows = fetch_alert_summaries(
                fresh_alerts_path,
                lookback_days=args.fresh_alert_lookback_days,
                max_pages=args.max_alert_pages,
            )

        alert_rows = merge_alert_summaries(source_dir, output_dir, fresh_alert_rows)

    logger.info("Step 3/8: Deriving failing performance signatures.")
    needed_signature_ids, alert_signature_metadata, _framework_filtered_ids = (
        derive_failing_signature_csvs(
            alert_rows,
            output_dir,
            included_statuses=args.included_statuses,
            excluded_framework_ids=set(args.exclude_framework_ids),
            overwrite_existing=args.overwrite_existing,
        )
    )
    logger.info(
        "Needed failing signatures for downstream data: signatures=%d, metadata_signatures=%d",
        len(needed_signature_ids),
        len(alert_signature_metadata),
    )

    logger.info("Step 4/8: Preparing commit graph.")
    maybe_export_commits(
        output_dir,
        args.autoland_repo.resolve(),
        skip_export=args.skip_fetch_commits,
        pull_first=args.pull_commits,
        source_dir=source_dir,
        overwrite_existing=args.overwrite_existing,
    )
    commit_graph = load_commit_graph(output_dir / "all_commits.jsonl")

    logger.info("Step 5/8: Fetching and merging per-signature performance measurements.")
    summary_output = output_dir / "per_sig_perf_data_summary.jsonl"
    replicates_output = output_dir / "per_sig_perf_data_replicates.jsonl"
    summary_exists = use_existing(
        summary_output,
        overwrite_existing=args.overwrite_existing,
    )
    replicates_exists = use_existing(
        replicates_output,
        overwrite_existing=args.overwrite_existing,
    )
    if summary_exists and replicates_exists:
        merged_summary = load_per_signature_jsonl(summary_output)
        merged_replicates = load_per_signature_jsonl(replicates_output)
        logger.info(
            "Using existing per-signature measurement caches: summary_signatures=%d, "
            "summary_measurements=%d, replicate_signatures=%d, replicate_measurements=%d, "
            "summary_path=%s, replicate_path=%s",
            len(merged_summary),
            count_measurements(merged_summary),
            len(merged_replicates),
            count_measurements(merged_replicates),
            summary_output,
            replicates_output,
        )
    else:
        old_summary = load_per_signature_jsonl(source_dir / "per_sig_perf_data_summary.jsonl")
        old_replicates = load_per_signature_jsonl(source_dir / "per_sig_perf_data_replicates.jsonl")
        logger.info(
            "Loaded source measurement caches: old_summary_signatures=%d, old_summary_measurements=%d, "
            "old_replicate_signatures=%d, old_replicate_measurements=%d",
            len(old_summary),
            count_measurements(old_summary),
            len(old_replicates),
            count_measurements(old_replicates),
        )
        fresh_summary: dict[int, dict[str, Any]] = {}
        fresh_replicates: dict[int, dict[str, Any]] = {}
        if args.skip_fetch_measurements:
            logger.info("Skipping fresh measurement fetch by request.")
        else:
            fresh_summary, fresh_replicates = fetch_measurements_for_signatures(
                sorted(needed_signature_ids),
                lookback_days=args.fresh_lookback_days,
                limit_signatures=args.limit_signatures,
                fetch_summary=not summary_exists,
                fetch_replicates=not replicates_exists,
            )

        if summary_exists:
            merged_summary = load_per_signature_jsonl(summary_output)
            logger.info(
                "Using existing summary measurement cache: signatures=%d, measurements=%d, path=%s; "
                "skipping summary cache write.",
                len(merged_summary),
                count_measurements(merged_summary),
                summary_output,
            )
        else:
            merged_summary = merge_signature_measurements(
                old_summary,
                fresh_summary,
                summary_output,
            )

        if replicates_exists:
            merged_replicates = load_per_signature_jsonl(replicates_output)
            logger.info(
                "Using existing replicate measurement cache: signatures=%d, measurements=%d, path=%s; "
                "skipping replicate cache write.",
                len(merged_replicates),
                count_measurements(merged_replicates),
                replicates_output,
            )
        else:
            merged_replicates = merge_signature_measurements(
                old_replicates,
                fresh_replicates,
                replicates_output,
            )

    logger.info("Step 6/8: Building per-signature metadata and runtime info.")
    sig_info_output = output_dir / "per_sig_perf_data_info.jsonl"
    if use_existing(sig_info_output, overwrite_existing=args.overwrite_existing):
        signature_info = load_existing_signature_info(sig_info_output)
        logger.info(
            "Using existing per-signature info: signatures=%d, path=%s; "
            "skipping signature metadata/runtime fetch.",
            len(signature_info),
            sig_info_output,
        )
    else:
        signature_info = build_signature_info(
            output_dir=output_dir,
            source_dir=source_dir,
            needed_signature_ids=needed_signature_ids,
            summary_records=merged_summary,
            replicate_records=merged_replicates,
            alert_metadata=alert_signature_metadata,
            skip_fetch_metadata=args.skip_fetch_signature_metadata,
            skip_fetch_job_durations=args.skip_fetch_job_durations,
            job_duration_samples=args.job_duration_samples,
            job_duration_lookback_days=args.fresh_lookback_days,
        )

    logger.info("Step 7/8: Writing per-revision performance data.")
    per_revision_output = output_dir / "per_revision_perf_data.jsonl"
    if use_existing(per_revision_output, overwrite_existing=args.overwrite_existing):
        logger.info(
            "Using existing per-revision performance data: %s; skipping write.",
            per_revision_output,
        )
    else:
        write_per_revision_perf_data(
            output_dir=output_dir,
            commit_graph=commit_graph,
            summary_records=merged_summary,
            replicate_records=merged_replicates,
            keep_desc=args.keep_desc_in_revision_data,
        )

    logger.info("Step 8/8: Building regression rows and split files.")
    regression_rows = build_regression_rows(
        alert_rows=alert_rows,
        commit_graph=commit_graph,
        signature_info=signature_info,
        included_statuses=args.included_statuses,
        excluded_framework_ids=set(args.exclude_framework_ids),
        excluded_alert_summary_ids=set(args.exclude_alert_summary_ids),
    )
    write_regression_splits(
        regression_rows,
        output_dir,
        eval_fraction=args.eval_fraction,
        overwrite_existing=args.overwrite_existing,
    )
    summarize_v2_directory(output_dir)
    logger.info("Finished Mozilla perf-bisect v2 dataset build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
