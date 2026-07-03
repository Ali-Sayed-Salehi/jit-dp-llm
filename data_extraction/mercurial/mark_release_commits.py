#!/usr/bin/env python3
"""Mark release-train commits in the Mozilla JIT Autoland commit JSONL.

The script reads release metadata from the local Mozilla Central Mercurial repo
and adds a boolean `release` field to each row in `all_commits.jsonl`.

By default this rewrites `datasets/mozilla_jit/all_commits.jsonl` atomically.
It never clones, pulls, or updates the Mozilla Central checkout.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOZILLA_CENTRAL_REPO = (
    REPO_ROOT / "data_extraction" / "mercurial" / "repos" / "mozilla-central"
)
DEFAULT_COMMITS_PATH = REPO_ROOT / "datasets" / "mozilla_jit" / "all_commits.jsonl"

ZERO_NODE = "0" * 40
DEFAULT_RELEASE_TAG_PATTERNS: Tuple[str, ...] = (
    # Modern Mozilla Central release-train branch points.
    r"^FIREFOX_BETA_[0-9]+_BASE$",
    # Legacy tags present in older Mozilla Central history.
    r"^FIREFOX_[0-9A-Za-z._]+_RELEASE$",
)


def _run_hg_log_for_tag_pattern(repo_path: Path, tag_pattern: str) -> str:
    """Return `hg log` output for revisions whose tag matches `tag_pattern`."""
    if '"' in tag_pattern:
        raise ValueError(f"Tag pattern may not contain double quotes: {tag_pattern!r}")

    revset = f'tag("re:{tag_pattern}")'
    try:
        result = subprocess.run(
            ["hg", "log", "-r", revset, "-T", "{node}\t{tags}\n"],
            cwd=repo_path,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Mercurial executable `hg` was not found on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(
            f"Failed to read release tags from {repo_path} with revset {revset!r}: {stderr}"
        ) from exc

    return result.stdout


def load_release_nodes(repo_path: Path, tag_patterns: Sequence[str]) -> Set[str]:
    """Load full commit nodes for active release tags in the local Mercurial repo."""
    repo_path = repo_path.resolve()
    if not (repo_path / ".hg").is_dir():
        raise FileNotFoundError(f"Mercurial repo not found: {repo_path}")

    release_nodes: Set[str] = set()
    for pattern in tag_patterns:
        for line in _run_hg_log_for_tag_pattern(repo_path, str(pattern)).splitlines():
            if not line.strip():
                continue
            node = line.split("\t", 1)[0].strip()
            if node and node != ZERO_NODE:
                release_nodes.add(node)
    return release_nodes


def _read_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object row in {path}:{line_no}")
            yield line_no, obj


def mark_release_commits(
    *,
    commits_path: Path,
    output_path: Path,
    release_nodes: Set[str],
    dry_run: bool = False,
) -> Dict[str, int]:
    """Write commit rows with `release` set from `release_nodes`."""
    commits_path = commits_path.resolve()
    output_path = output_path.resolve()
    if not commits_path.is_file():
        raise FileNotFoundError(f"Commit JSONL not found: {commits_path}")

    total = 0
    marked = 0
    changed = 0

    out_file = None
    temp_path: Path | None = None
    try:
        if not dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path == commits_path:
                fd, temp_name = tempfile.mkstemp(
                    prefix=f".{commits_path.name}.", suffix=".tmp", dir=commits_path.parent
                )
                temp_path = Path(temp_name)
                out_file = os.fdopen(fd, "w", encoding="utf-8")
            else:
                out_file = output_path.open("w", encoding="utf-8")

        for _line_no, commit in _read_jsonl(commits_path):
            total += 1
            node = commit.get("node")
            is_release = isinstance(node, str) and node in release_nodes
            if is_release:
                marked += 1
            if commit.get("release") is not is_release:
                changed += 1
            commit["release"] = bool(is_release)

            if out_file is not None:
                json.dump(commit, out_file, ensure_ascii=False)
                out_file.write("\n")

        if out_file is not None:
            out_file.flush()
            os.fsync(out_file.fileno())
            out_file.close()
            out_file = None
            if temp_path is not None:
                os.replace(temp_path, commits_path)
                temp_path = None
    finally:
        if out_file is not None:
            out_file.close()
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

    return {
        "total_commits": total,
        "release_commits": marked,
        "changed_rows": changed,
    }


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add a boolean `release` marker to Mozilla JIT Autoland commits."
    )
    parser.add_argument(
        "--mozilla-central-repo",
        default=str(DEFAULT_MOZILLA_CENTRAL_REPO),
        help="Local Mozilla Central Mercurial checkout. The script reads it but does not pull/update it.",
    )
    parser.add_argument(
        "--commits-path",
        default=str(DEFAULT_COMMITS_PATH),
        help="Input Autoland commits JSONL.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL. Defaults to rewriting --commits-path atomically.",
    )
    parser.add_argument(
        "--tag-pattern",
        action="append",
        default=None,
        help=(
            "Mercurial tag regex for release commits. May be passed multiple times. "
            "Defaults to Firefox beta-base release-train tags plus legacy Firefox *_RELEASE tags."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute counts without writing any output.",
    )
    return parser.parse_args()


def main() -> int:
    args = get_args()
    repo_path = Path(args.mozilla_central_repo).expanduser()
    commits_path = Path(args.commits_path).expanduser()
    output_path = Path(args.output).expanduser() if args.output else commits_path
    tag_patterns: List[str] = list(args.tag_pattern or DEFAULT_RELEASE_TAG_PATTERNS)

    release_nodes = load_release_nodes(repo_path=repo_path, tag_patterns=tag_patterns)
    stats = mark_release_commits(
        commits_path=commits_path,
        output_path=output_path,
        release_nodes=release_nodes,
        dry_run=bool(args.dry_run),
    )
    summary = {
        **stats,
        "release_nodes_from_mozilla_central": len(release_nodes),
        "mozilla_central_repo": str(repo_path.resolve()),
        "commits_path": str(commits_path.resolve()),
        "output_path": None if args.dry_run else str(output_path.resolve()),
        "tag_patterns": tag_patterns,
        "dry_run": bool(args.dry_run),
    }
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
