#!/usr/bin/env python3
"""
Build file rename timelines for the Mozilla code-review dataset window.

This script derives the same Mercurial-order split boundaries used by
create_code_review_dataset.py from:

    datasets/mozilla_code_review/all_commits.jsonl
    datasets/mozilla_code_review/risk_predictions_eval.json
    datasets/mozilla_code_review/risk_predictions_final_test.json

It scans the local autoland Mercurial repository from the eval split's starting
boundary through the final-test split's ending boundary, records true rename
chains from Mercurial copy metadata, falls back to conservative content
similarity for same-commit delete/add pairs without metadata, and writes the
timelines to:

    datasets/mozilla_code_review/file_path_timeline.jsonl

Use `--debug` to scan only the first `--debug-count` Mercurial revisions in the
boundary window.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "datasets" / "mozilla_code_review"

DEFAULT_INPUT_JSONL = DATASET_DIR / "all_commits.jsonl"
DEFAULT_OUTPUT_JSONL = DATASET_DIR / "file_path_timeline.jsonl"
DEFAULT_EVAL_PREDICTIONS_JSON = DATASET_DIR / "risk_predictions_eval.json"
DEFAULT_FINAL_TEST_PREDICTIONS_JSON = DATASET_DIR / "risk_predictions_final_test.json"
DEFAULT_AUTOLAND_REPO = (
    REPO_ROOT / "data_extraction" / "mercurial" / "repos" / "autoland"
)
DEFAULT_AUTOLAND_URL = "https://hg-edge.mozilla.org/integration/autoland"
DEFAULT_HG_PULL_MAX_ATTEMPTS = 5
DEFAULT_HG_PULL_RETRY_BASE_SLEEP_SECONDS = 5.0
DEFAULT_SIMILARITY_THRESHOLD = 0.90
DEFAULT_SIMILARITY_MARGIN = 0.05
DEFAULT_SIMILARITY_MAX_FILE_BYTES = 1_000_000
DEFAULT_SIMILARITY_MAX_PAIRS = 2_500
NULL_NODE = "0000000000000000000000000000000000000000"


@dataclass(frozen=True)
class JsonlRecord:
    line_num: int
    record: dict[str, Any]


@dataclass(frozen=True)
class SplitBoundary:
    name: str
    start_index: int
    end_index: int
    start_commit_id: str
    end_commit_id: str
    sample_commit_ids: frozenset[str]


@dataclass(frozen=True)
class RenameEvent:
    rev: int
    commit_index: int
    commit_id: str
    source_path: str
    destination_path: str
    detection_method: str
    similarity: float | None = None


@dataclass(frozen=True)
class SimilarityRename:
    source_path: str
    destination_path: str
    similarity: float


@dataclass
class FilePathTimeline:
    timeline_id: int
    start_path: str
    start_rev: int
    start_index: int
    events: list[RenameEvent] = field(default_factory=list)
    deleted_rev: int | None = None
    deleted_index: int | None = None
    deleted_commit_id: str | None = None

    @property
    def canonical_path(self) -> str:
        return self.events[-1].destination_path if self.events else self.start_path


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
        "--output-file-path-timeline-jsonl",
        default=str(DEFAULT_OUTPUT_JSONL),
        dest="output_jsonl",
        help="Path to write file_path_timeline.jsonl.",
    )
    parser.add_argument(
        "--autoland-repo",
        default=str(DEFAULT_AUTOLAND_REPO),
        help="Local path for the Mozilla autoland Mercurial repository.",
    )
    parser.add_argument(
        "--autoland-url",
        default=DEFAULT_AUTOLAND_URL,
        help="Mercurial URL used when cloning or pulling autoland.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Scan only the first debug-count Mercurial revisions in the window.",
    )
    parser.add_argument(
        "--debug-count",
        type=int,
        default=100,
        help="Number of Mercurial revisions to scan in debug mode.",
    )
    parser.add_argument(
        "--skip-repo-update",
        action="store_true",
        help=(
            "Do not clone or pull autoland. The local repository must already "
            "exist. This is intended for local smoke tests."
        ),
    )
    parser.add_argument(
        "--hg-pull-max-attempts",
        type=int,
        default=DEFAULT_HG_PULL_MAX_ATTEMPTS,
        help="Maximum attempts for `hg pull -u` before failing.",
    )
    parser.add_argument(
        "--hg-pull-retry-base-sleep",
        type=float,
        default=DEFAULT_HG_PULL_RETRY_BASE_SLEEP_SECONDS,
        help="Base sleep seconds for retrying failed `hg pull -u` commands.",
    )
    parser.add_argument(
        "--disable-similarity-inference",
        action="store_true",
        help="Only use Mercurial copy metadata; do not infer manual delete/add renames.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="Minimum line similarity for inferring a manual rename.",
    )
    parser.add_argument(
        "--similarity-margin",
        type=float,
        default=DEFAULT_SIMILARITY_MARGIN,
        help="Required score gap between the best and second-best match.",
    )
    parser.add_argument(
        "--similarity-max-file-bytes",
        type=int,
        default=DEFAULT_SIMILARITY_MAX_FILE_BYTES,
        help="Skip content-similarity checks for files larger than this many bytes.",
    )
    parser.add_argument(
        "--similarity-max-pairs",
        type=int,
        default=DEFAULT_SIMILARITY_MAX_PAIRS,
        help=(
            "Skip content-similarity inference for a changeset if it has more "
            "extension-compatible added/deleted file pairs than this."
        ),
    )
    args = parser.parse_args(argv)
    if args.debug_count < 0:
        parser.error("--debug-count must be non-negative")
    if args.hg_pull_max_attempts < 1:
        parser.error("--hg-pull-max-attempts must be at least 1")
    if args.hg_pull_retry_base_sleep < 0:
        parser.error("--hg-pull-retry-base-sleep must be non-negative")
    if not 0.0 <= args.similarity_threshold <= 1.0:
        parser.error("--similarity-threshold must be between 0 and 1")
    if args.similarity_margin < 0:
        parser.error("--similarity-margin must be non-negative")
    if args.similarity_max_file_bytes < 0:
        parser.error("--similarity-max-file-bytes must be non-negative")
    if args.similarity_max_pairs < 0:
        parser.error("--similarity-max-pairs must be non-negative")
    return args


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


def load_commits(path: Path) -> list[dict[str, Any]]:
    commits: list[dict[str, Any]] = []
    node_to_index: dict[str, int] = {}

    for item in iter_jsonl(path):
        record = item.record
        node = record.get("node")
        if not isinstance(node, str) or not node:
            raise ValueError(f"{path}:{item.line_num}: commit is missing a valid node")
        if node in node_to_index:
            raise ValueError(f"{path}:{item.line_num}: duplicate commit node {node}")

        current_index = len(commits)
        parents = record.get("parents", [])
        if not isinstance(parents, list):
            raise ValueError(f"{path}:{item.line_num}: commit {node} has non-list parents")

        for parent in parents:
            if parent == NULL_NODE:
                continue
            if not isinstance(parent, str):
                raise ValueError(
                    f"{path}:{item.line_num}: commit {node} has non-string parent"
                )
            parent_index = node_to_index.get(parent)
            if parent_index is None:
                raise ValueError(
                    f"{path}:{item.line_num}: commit {node} references missing "
                    f"parent {parent}."
                )
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


def run_hg_pull_with_retries(
    *,
    repo_path: Path,
    repo_url: str,
    max_attempts: int,
    retry_base_sleep_seconds: float,
) -> None:
    command = ["hg", "pull", "-u", repo_url]
    last_error: subprocess.CalledProcessError | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            subprocess.run(command, cwd=repo_path, check=True)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt == max_attempts:
                break

            wait_seconds = retry_base_sleep_seconds * attempt
            print(
                f"`{' '.join(command)}` failed with exit code {exc.returncode} "
                f"(attempt {attempt}/{max_attempts}); retrying in "
                f"{wait_seconds:g}s.",
                file=sys.stderr,
            )
            if wait_seconds > 0:
                time.sleep(wait_seconds)

    raise RuntimeError(
        f"`{' '.join(command)}` failed in {repo_path} after {max_attempts} attempts."
    ) from last_error


def ensure_autoland_repo(
    *,
    repo_path: Path,
    repo_url: str,
    skip_update: bool,
    hg_pull_max_attempts: int,
    hg_pull_retry_base_sleep_seconds: float,
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
        run_hg_pull_with_retries(
            repo_path=repo_path,
            repo_url=repo_url,
            max_attempts=hg_pull_max_attempts,
            retry_base_sleep_seconds=hg_pull_retry_base_sleep_seconds,
        )
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


def get_hg_rev(repo_path: Path, commit_id: str) -> int:
    output = run_hg_capture(repo_path, ["log", "-r", commit_id, "-T", "{rev}\n"])
    value = output.strip()
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid Mercurial revision for {commit_id}: {value!r}") from exc


def iter_hg_log_records(
    *,
    repo_path: Path,
    start_rev: int,
    end_rev: int,
) -> Iterator[dict[str, Any]]:
    template = (
        "{dict(rev, node, file_adds, file_dels, file_copies, p1rev=p1.rev)|json}\n"
    )
    command = [
        "hg",
        "log",
        "-r",
        f"{start_rev}:{end_rev}",
        "--copies",
        "-T",
        template,
    ]
    process = subprocess.Popen(
        command,
        cwd=repo_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None

    for line_num, line in enumerate(process.stdout, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            process.kill()
            raise ValueError(
                f"Invalid JSON from `{' '.join(command)}` line {line_num}: {exc}"
            ) from exc
        if not isinstance(record, dict):
            process.kill()
            raise ValueError(
                f"Expected object from `{' '.join(command)}` line {line_num}"
            )
        yield record

    stderr = process.stderr.read()
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(
            f"`{' '.join(command)}` failed in {repo_path} with exit code "
            f"{returncode}: {stderr.strip()}"
        )


def parse_int_field(record: dict[str, Any], field_name: str) -> int:
    value = record.get(field_name)
    if not isinstance(value, int):
        raise ValueError(f"Mercurial log record has invalid {field_name}: {value!r}")
    return value


def parse_string_field(record: dict[str, Any], field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Mercurial log record has invalid {field_name}: {value!r}")
    return value


def parse_file_list(record: dict[str, Any], field_name: str) -> list[str]:
    value = record.get(field_name)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Mercurial log record has invalid {field_name}: {value!r}")
    return value


def parse_file_copies(record: dict[str, Any]) -> dict[str, str]:
    value = record.get("file_copies")
    if not isinstance(value, dict):
        raise ValueError(f"Mercurial log record has invalid file_copies: {value!r}")

    copies: dict[str, str] = {}
    for destination, source in value.items():
        if not isinstance(destination, str) or not isinstance(source, str):
            raise ValueError(
                f"Mercurial log record has invalid file_copies entry: "
                f"{destination!r}: {source!r}"
            )
        copies[destination] = source
    return copies


def path_suffix(path: str) -> str:
    return Path(path).suffix.lower()


def has_compatible_suffix(source_path: str, destination_path: str) -> bool:
    source_suffix = path_suffix(source_path)
    destination_suffix = path_suffix(destination_path)
    return (
        not source_suffix
        or not destination_suffix
        or source_suffix == destination_suffix
    )


def get_hg_file_size(repo_path: Path, *, rev: int, path: str) -> int | None:
    result = subprocess.run(
        ["hg", "files", "-r", str(rev), "-T", "{size}\n", "--", path],
        cwd=repo_path,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return None

    value = result.stdout.strip()
    if not value:
        return None
    try:
        return int(value.splitlines()[0])
    except ValueError:
        return None


def get_hg_file_bytes(
    repo_path: Path,
    *,
    rev: int,
    path: str,
    max_file_bytes: int,
) -> bytes | None:
    file_size = get_hg_file_size(repo_path, rev=rev, path=path)
    if file_size is None or file_size > max_file_bytes:
        return None

    result = subprocess.run(
        ["hg", "cat", "-r", str(rev), "--", path],
        cwd=repo_path,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return None
    if len(result.stdout) > max_file_bytes:
        return None
    return result.stdout


def line_similarity(old_content: bytes, new_content: bytes) -> float | None:
    if b"\0" in old_content or b"\0" in new_content:
        return None

    old_lines = old_content.splitlines()
    new_lines = new_content.splitlines()
    if not old_lines or not new_lines:
        return None

    return SequenceMatcher(None, old_lines, new_lines, autojunk=False).ratio()


def count_compatible_pairs(
    *,
    source_paths: set[str],
    destination_paths: set[str],
) -> int:
    return sum(
        1
        for source_path in source_paths
        for destination_path in destination_paths
        if has_compatible_suffix(source_path, destination_path)
    )


def is_clear_best_match(
    scores: list[tuple[float, str]],
    *,
    similarity_margin: float,
) -> bool:
    if len(scores) == 1:
        return True
    return scores[0][0] - scores[1][0] >= similarity_margin


def infer_similarity_renames(
    *,
    repo_path: Path,
    parent_rev: int,
    rev: int,
    source_paths: set[str],
    destination_paths: set[str],
    similarity_threshold: float,
    similarity_margin: float,
    similarity_max_file_bytes: int,
    similarity_max_pairs: int,
    stats: dict[str, int],
) -> dict[str, SimilarityRename]:
    if parent_rev < 0 or not source_paths or not destination_paths:
        return {}

    compatible_pair_count = count_compatible_pairs(
        source_paths=source_paths,
        destination_paths=destination_paths,
    )
    if compatible_pair_count == 0:
        return {}
    if compatible_pair_count > similarity_max_pairs:
        stats["similarity_skipped_large_changesets"] += 1
        return {}

    stats["similarity_candidate_changesets"] += 1
    old_content_by_path: dict[str, bytes | None] = {}
    new_content_by_path: dict[str, bytes | None] = {}
    selected_by_destination: dict[str, list[SimilarityRename]] = {}

    for source_path in sorted(source_paths):
        old_content = old_content_by_path.get(source_path)
        if source_path not in old_content_by_path:
            old_content = get_hg_file_bytes(
                repo_path,
                rev=parent_rev,
                path=source_path,
                max_file_bytes=similarity_max_file_bytes,
            )
            old_content_by_path[source_path] = old_content
        if old_content is None:
            stats["similarity_source_files_skipped"] += 1
            continue

        destination_scores: list[tuple[float, str]] = []
        for destination_path in sorted(destination_paths):
            if not has_compatible_suffix(source_path, destination_path):
                continue

            new_content = new_content_by_path.get(destination_path)
            if destination_path not in new_content_by_path:
                new_content = get_hg_file_bytes(
                    repo_path,
                    rev=rev,
                    path=destination_path,
                    max_file_bytes=similarity_max_file_bytes,
                )
                new_content_by_path[destination_path] = new_content
            if new_content is None:
                continue

            score = line_similarity(old_content, new_content)
            if score is None:
                continue
            stats["similarity_pairs_compared"] += 1
            destination_scores.append((score, destination_path))

        if not destination_scores:
            continue

        destination_scores.sort(key=lambda item: (-item[0], item[1]))
        if destination_scores[0][0] < similarity_threshold:
            continue
        if not is_clear_best_match(
            destination_scores,
            similarity_margin=similarity_margin,
        ):
            stats["similarity_ambiguous_sources"] += 1
            continue

        score, destination_path = destination_scores[0]
        selected_by_destination.setdefault(destination_path, []).append(
            SimilarityRename(
                source_path=source_path,
                destination_path=destination_path,
                similarity=score,
            )
        )

    inferred_renames: dict[str, SimilarityRename] = {}
    for destination_path, matches in selected_by_destination.items():
        if len(matches) > 1:
            stats["similarity_destination_conflicts"] += 1
            continue

        match = matches[0]
        inferred_renames[destination_path] = match

    return inferred_renames


def build_file_path_timelines(
    *,
    repo_path: Path,
    node_to_index: dict[str, int],
    boundary_start_rev: int,
    boundary_start_index: int,
    boundary_end_rev: int,
    infer_similarity: bool,
    similarity_threshold: float,
    similarity_margin: float,
    similarity_max_file_bytes: int,
    similarity_max_pairs: int,
) -> tuple[list[FilePathTimeline], dict[str, int]]:
    active_path_start_rev: dict[str, int] = {}
    active_path_start_index: dict[str, int] = {}
    active_path_to_timeline: dict[str, FilePathTimeline] = {}
    timelines: list[FilePathTimeline] = []
    stats = {
        "changesets_scanned": 0,
        "copy_events_seen": 0,
        "metadata_rename_events_written": 0,
        "similarity_candidate_changesets": 0,
        "similarity_pairs_compared": 0,
        "similarity_source_files_skipped": 0,
        "similarity_skipped_large_changesets": 0,
        "similarity_ambiguous_sources": 0,
        "similarity_destination_conflicts": 0,
        "similarity_rename_events_written": 0,
        "rename_events_written": 0,
        "timelines_written": 0,
    }

    for record in iter_hg_log_records(
        repo_path=repo_path,
        start_rev=boundary_start_rev,
        end_rev=boundary_end_rev,
    ):
        stats["changesets_scanned"] += 1
        rev = parse_int_field(record, "rev")
        parent_rev = parse_int_field(record, "p1rev")
        commit_id = parse_string_field(record, "node")
        try:
            commit_index = node_to_index[commit_id]
        except KeyError as exc:
            raise ValueError(
                f"Mercurial log returned commit {commit_id}, but it is missing "
                "from all_commits.jsonl."
            ) from exc

        file_adds = set(parse_file_list(record, "file_adds"))
        file_dels = set(parse_file_list(record, "file_dels"))
        file_copies = parse_file_copies(record)
        stats["copy_events_seen"] += len(file_copies)

        metadata_rename_dest_to_source = {
            destination: source
            for destination, source in file_copies.items()
            if destination in file_adds and source in file_dels
        }
        metadata_rename_sources = set(metadata_rename_dest_to_source.values())
        metadata_rename_destinations = set(metadata_rename_dest_to_source)

        similarity_renames: dict[str, SimilarityRename] = {}
        if infer_similarity:
            similarity_renames = infer_similarity_renames(
                repo_path=repo_path,
                parent_rev=parent_rev,
                rev=rev,
                source_paths=file_dels - metadata_rename_sources,
                destination_paths=file_adds - metadata_rename_destinations,
                similarity_threshold=similarity_threshold,
                similarity_margin=similarity_margin,
                similarity_max_file_bytes=similarity_max_file_bytes,
                similarity_max_pairs=similarity_max_pairs,
                stats=stats,
            )

        rename_dest_to_source = dict(metadata_rename_dest_to_source)
        for destination, match in similarity_renames.items():
            rename_dest_to_source[destination] = match.source_path

        rename_sources = set(rename_dest_to_source.values())
        rename_destinations = set(rename_dest_to_source)

        for destination, source in rename_dest_to_source.items():
            timeline = active_path_to_timeline.pop(source, None)
            if timeline is None:
                timeline = FilePathTimeline(
                    timeline_id=len(timelines),
                    start_path=source,
                    start_rev=active_path_start_rev.get(source, boundary_start_rev),
                    start_index=active_path_start_index.get(
                        source,
                        boundary_start_index,
                    ),
                )
                timelines.append(timeline)

            active_path_start_rev.pop(source, None)
            active_path_start_index.pop(source, None)
            similarity = None
            detection_method = "hg_metadata"
            if destination in similarity_renames:
                detection_method = "content_similarity"
                similarity = similarity_renames[destination].similarity
            timeline.events.append(
                RenameEvent(
                    rev=rev,
                    commit_index=commit_index,
                    commit_id=commit_id,
                    source_path=source,
                    destination_path=destination,
                    detection_method=detection_method,
                    similarity=similarity,
                )
            )
            timeline.deleted_rev = None
            timeline.deleted_index = None
            timeline.deleted_commit_id = None
            active_path_to_timeline[destination] = timeline
            active_path_start_rev[destination] = rev
            active_path_start_index[destination] = commit_index
            if detection_method == "content_similarity":
                stats["similarity_rename_events_written"] += 1
            else:
                stats["metadata_rename_events_written"] += 1
            stats["rename_events_written"] += 1

        for path in file_dels:
            if path in rename_sources:
                continue

            active_path_start_rev.pop(path, None)
            active_path_start_index.pop(path, None)
            timeline = active_path_to_timeline.pop(path, None)
            if timeline is not None:
                timeline.deleted_rev = rev
                timeline.deleted_index = commit_index
                timeline.deleted_commit_id = commit_id

        for path in file_adds:
            if path in rename_destinations:
                continue

            replaced_timeline = active_path_to_timeline.pop(path, None)
            if replaced_timeline is not None:
                replaced_timeline.deleted_rev = rev
                replaced_timeline.deleted_index = commit_index
                replaced_timeline.deleted_commit_id = commit_id

            active_path_start_rev[path] = rev
            active_path_start_index[path] = commit_index

    stats["timelines_written"] = len(timelines)
    return timelines, stats


def timeline_to_record(timeline: FilePathTimeline) -> dict[str, Any]:
    return {
        "record_type": "timeline",
        "timeline_id": timeline.timeline_id,
        "start_path": timeline.start_path,
        "start_rev": timeline.start_rev,
        "start_index": timeline.start_index,
        "canonical_path": timeline.canonical_path,
        "deleted_rev": timeline.deleted_rev,
        "deleted_index": timeline.deleted_index,
        "deleted_commit_id": timeline.deleted_commit_id,
        "events": [
            {
                "rev": event.rev,
                "commit_index": event.commit_index,
                "commit_id": event.commit_id,
                "source_path": event.source_path,
                "destination_path": event.destination_path,
                "detection_method": event.detection_method,
                "similarity": event.similarity,
            }
            for event in timeline.events
        ],
    }


def write_file_path_timeline(
    *,
    output_jsonl: Path,
    timelines: list[FilePathTimeline],
    metadata: dict[str, Any],
) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as output_file:
        output_file.write(json.dumps(metadata) + "\n")
        for timeline in timelines:
            output_file.write(json.dumps(timeline_to_record(timeline)) + "\n")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_jsonl = Path(args.input_jsonl)
    eval_predictions_json = Path(args.eval_predictions_json)
    final_test_predictions_json = Path(args.final_test_predictions_json)
    output_jsonl = Path(args.output_jsonl)
    autoland_repo = Path(args.autoland_repo)

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
    print(describe_boundary(eval_boundary), file=sys.stderr)
    print(describe_boundary(final_test_boundary), file=sys.stderr)

    if eval_boundary.start_index > final_test_boundary.end_index:
        raise ValueError(
            "Eval start boundary comes after final-test end boundary: "
            f"{eval_boundary.start_commit_id} > {final_test_boundary.end_commit_id}"
        )

    ensure_autoland_repo(
        repo_path=autoland_repo,
        repo_url=args.autoland_url,
        skip_update=args.skip_repo_update,
        hg_pull_max_attempts=args.hg_pull_max_attempts,
        hg_pull_retry_base_sleep_seconds=args.hg_pull_retry_base_sleep,
    )

    boundary_start_rev = get_hg_rev(autoland_repo, eval_boundary.start_commit_id)
    boundary_end_rev = get_hg_rev(autoland_repo, final_test_boundary.end_commit_id)
    if boundary_start_rev > boundary_end_rev:
        raise ValueError(
            "Eval start Mercurial revision comes after final-test end revision: "
            f"{boundary_start_rev} > {boundary_end_rev}"
        )

    scan_end_rev = boundary_end_rev
    scan_end_index = final_test_boundary.end_index
    scan_end_commit_id = final_test_boundary.end_commit_id
    if args.debug:
        if args.debug_count == 0:
            scan_end_rev = boundary_start_rev - 1
            scan_end_index = eval_boundary.start_index - 1
            scan_end_commit_id = eval_boundary.start_commit_id
        else:
            scan_end_rev = min(
                boundary_end_rev,
                boundary_start_rev + args.debug_count - 1,
            )
            scan_end_commit_id = run_hg_capture(
                autoland_repo,
                ["log", "-r", str(scan_end_rev), "-T", "{node}\n"],
            ).strip()
            scan_end_index = node_to_index.get(scan_end_commit_id, scan_end_rev)
        print(
            f"DEBUG: scanning Mercurial revisions {boundary_start_rev}:{scan_end_rev}.",
            file=sys.stderr,
        )

    timelines: list[FilePathTimeline]
    stats: dict[str, int]
    if scan_end_rev < boundary_start_rev:
        timelines = []
        stats = {
            "changesets_scanned": 0,
            "copy_events_seen": 0,
            "metadata_rename_events_written": 0,
            "similarity_candidate_changesets": 0,
            "similarity_pairs_compared": 0,
            "similarity_source_files_skipped": 0,
            "similarity_skipped_large_changesets": 0,
            "similarity_ambiguous_sources": 0,
            "similarity_destination_conflicts": 0,
            "similarity_rename_events_written": 0,
            "rename_events_written": 0,
            "timelines_written": 0,
        }
    else:
        timelines, stats = build_file_path_timelines(
            repo_path=autoland_repo,
            node_to_index=node_to_index,
            boundary_start_rev=boundary_start_rev,
            boundary_start_index=eval_boundary.start_index,
            boundary_end_rev=scan_end_rev,
            infer_similarity=not args.disable_similarity_inference,
            similarity_threshold=args.similarity_threshold,
            similarity_margin=args.similarity_margin,
            similarity_max_file_bytes=args.similarity_max_file_bytes,
            similarity_max_pairs=args.similarity_max_pairs,
        )

    metadata = {
        "record_type": "metadata",
        "format_version": 1,
        "boundary_start_commit_id": eval_boundary.start_commit_id,
        "boundary_start_index": eval_boundary.start_index,
        "boundary_start_rev": boundary_start_rev,
        "boundary_end_commit_id": final_test_boundary.end_commit_id,
        "boundary_end_index": final_test_boundary.end_index,
        "boundary_end_rev": boundary_end_rev,
        "scan_end_commit_id": scan_end_commit_id,
        "scan_end_index": scan_end_index,
        "scan_end_rev": scan_end_rev,
        "debug": bool(args.debug),
        "similarity_inference_enabled": not args.disable_similarity_inference,
        "similarity_threshold": args.similarity_threshold,
        "similarity_margin": args.similarity_margin,
        "similarity_max_file_bytes": args.similarity_max_file_bytes,
        "similarity_max_pairs": args.similarity_max_pairs,
    }
    write_file_path_timeline(
        output_jsonl=output_jsonl,
        timelines=timelines,
        metadata=metadata,
    )

    print(f"Wrote {len(timelines)} file path timelines to {output_jsonl}.", file=sys.stderr)
    print(json.dumps(stats, sort_keys=True), file=sys.stderr)


if __name__ == "__main__":
    main()
