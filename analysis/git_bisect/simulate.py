#!/usr/bin/env python3

"""
Git-bisect simulation helpers for the Mozilla JIT dataset.

This script simulates a simplified bisection workflow over a linear commit
history window:

Inputs:
  - Bug dataset (`datasets/mozilla_jit/mozilla_jit.jsonl`)
  - Ordered commits (`datasets/mozilla_jit/all_commits.jsonl`) matching Autoland history
  - Risk predictions (`analysis/git_bisect/risk_predictions_*.json`) used to define the
    simulation commit window (the min/max commit ids present in predictions)

Core assumptions (by design for this simulation):
  - Linear history: a commit at index i contains a regressor at index r iff i >= r.
  - Single culprit per regression: if a bug has multiple "available regressors",
    only the latest regressor commit is treated as the true culprit, stored as
    `available_regressor`.

Simulation outline for each regression bug:
  1) Determine `culprit_index` from `available_regressor`.
  2) Choose a "good" index via the configured lookback strategy, starting from
     the "bad" observation point.
  3) Choose a "bad" index as the last commit at or before `bug_creation_time`.
  4) Run the configured bisection strategy to locate the single culprit and count tests.

Run:
  - `python analysis/git_bisect/simulate.py --dry-run`
  - Results are written as a JSON summary to `--output`.
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from bisection import BisectionStrategy, GitBisectBaseline
from lookback import FixedStrideLookback, LookbackStrategy


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

BUGS_PATH = os.path.join(REPO_ROOT, "datasets", "mozilla_jit", "mozilla_jit.jsonl")
COMMITS_PATH = os.path.join(REPO_ROOT, "datasets", "mozilla_jit", "all_commits.jsonl")

RISK_FINAL_PATH = os.path.join(
    REPO_ROOT, "analysis", "git_bisect", "risk_predictions_final_test.json"
)

logger = logging.getLogger(__name__)


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a JSONL file (one object per line)."""
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object rows in {path}:{line_no}, got {type(obj).__name__}")
            yield obj


def load_bugs_with_available_regressors(
    path: str,
    *,
    node_to_index: Dict[str, int],
    window_start: int,
    window_end: int,
) -> List[Dict[str, Any]]:
    """
    Load bugs from `mozilla_jit.jsonl` and add a new field `available_regressor`.

    For each bug, if `regressed_by` is non-empty, each referenced bug id is
    checked against the full dataset, and also required to fall within the
    simulation cutoff window.

    A regressor is considered "available" if:
      - The regressor bug_id exists in the dataset, AND
      - The regressor bug has a `revision` that exists in `node_to_index`, AND
      - That revision's commit index is within [window_start, window_end].

    If multiple regressors are available for a single bug, only the latest
    (highest commit index) is retained in `available_regressor`.
    """
    bugs: List[Dict[str, Any]] = list(_read_jsonl(path))

    bugs_by_id = build_bug_id_index(bugs)

    for bug in bugs:
        latest: Optional[Tuple[int, str]] = None
        seen = set()
        regressed_by = bug.get("regressed_by") or []
        if not isinstance(regressed_by, list):
            raise ValueError(
                f"Expected `regressed_by` to be a list for bug_id={bug.get('bug_id')}, "
                f"got {type(regressed_by).__name__}"
            )

        for candidate in regressed_by:
            if candidate is None:
                continue
            candidate_id = str(candidate)
            if candidate_id in seen:
                continue

            reg_bug = bugs_by_id.get(candidate_id)
            if not reg_bug:
                continue
            reg_rev = reg_bug.get("revision")
            if not reg_rev:
                continue
            reg_idx = node_to_index.get(str(reg_rev))
            if reg_idx is None:
                continue
            if reg_idx < window_start or reg_idx > window_end:
                continue

            if latest is None or reg_idx > latest[0]:
                latest = (reg_idx, candidate_id)
            seen.add(candidate_id)

        bug["available_regressor"] = latest[1] if latest is not None else None

    logger.debug(
        "Loaded %d bugs; computed `available_regressor` within window [%d,%d].",
        len(bugs),
        window_start,
        window_end,
    )
    return bugs


def load_commits(path: str) -> List[Dict[str, Any]]:
    """
    Load ordered commits from `all_commits.jsonl`.

    Rows correspond to Mercurial commits; ordering matches Autoland history.
    """
    return list(_read_jsonl(path))


def _commit_datetime_utc(commit: Dict[str, Any]) -> datetime:
    """Convert a commit row's Mercurial `date` field into a UTC datetime."""
    date = commit.get("date")
    if (
        not isinstance(date, list)
        or len(date) < 1
        or not isinstance(date[0], (int, float))
    ):
        raise ValueError(f"Invalid commit date format for node={commit.get('node')}")
    return datetime.fromtimestamp(float(date[0]), tz=timezone.utc)


def build_node_to_index(commits: List[Dict[str, Any]]) -> Dict[str, int]:
    """Build a mapping from commit `node` (hash) to its index in `commits`."""
    node_to_index: Dict[str, int] = {}
    for idx, commit in enumerate(commits):
        node = commit.get("node")
        if not node:
            raise ValueError(
                f"Missing node for commit at index {idx}"
            )
        node_to_index[str(node)] = idx
    return node_to_index


def build_bug_id_index(bugs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build a mapping from `bug_id` (as string) to the corresponding bug object."""
    bugs_by_id: Dict[str, Dict[str, Any]] = {}
    for bug in bugs:
        bug_id = bug.get("bug_id")
        if bug_id is not None:
            bugs_by_id[str(bug_id)] = bug
    return bugs_by_id


def _positive_probability_from_predicted_class(
    prediction: int, confidence: float, label_order: List[str]
) -> float:
    """
    Convert (predicted class, confidence) into P(POSITIVE).

    The predictions files store `confidence` as the probability of the predicted
    class, not necessarily P(POSITIVE).
    """
    if len(label_order) != 2:
        raise ValueError(f"Expected binary label_order, got {label_order}")
    if label_order[0] == "NEGATIVE" and label_order[1] == "POSITIVE":
        return float(confidence) if int(prediction) == 1 else 1.0 - float(confidence)
    if label_order[0] == "POSITIVE" and label_order[1] == "NEGATIVE":
        return float(confidence) if int(prediction) == 0 else 1.0 - float(confidence)
    raise ValueError(f"Unexpected label_order={label_order}; expected POSITIVE/NEGATIVE")


def load_risk_predictions(path: str) -> Dict[str, float]:
    """
    Load a `risk_predictions_*.json` and return a mapping of:
      commit_id -> risk score (P(POSITIVE))
    """
    with open(path, "r", encoding="utf-8") as f:
        blob = json.load(f)

    samples = blob.get("results")
    if not samples:
        samples = blob.get("samples")
    if not samples:
        raise ValueError(
            f"No prediction rows found in {path}: expected a non-empty `results`, `samples`, or `sample` list."
        )
    label_order = blob.get("label_order") or ["NEGATIVE", "POSITIVE"]
    if not isinstance(samples, list):
        raise ValueError(f"Expected `results`/`samples` list in {path}")
    if not isinstance(label_order, list):
        raise ValueError(f"Expected `label_order` list in {path}")

    risk_by_commit: Dict[str, float] = {}
    for i, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(
                f"Invalid sample at index {i} in {path}: expected object, got {type(sample).__name__}"
            )
        commit_id = sample.get("commit_id")
        if not commit_id:
            raise ValueError(f"Invalid sample at index {i} in {path}: missing `commit_id`")
        prediction = sample.get("prediction")
        confidence = sample.get("confidence")
        if prediction is None or confidence is None:
            raise ValueError(
                f"Invalid sample at index {i} in {path}: missing `prediction` and/or `confidence`"
            )

        risk_by_commit[str(commit_id)] = _positive_probability_from_predicted_class(
            int(prediction), float(confidence), label_order
        )

    return risk_by_commit


def _risk_window_from_predictions(
    commits: List[Dict[str, Any]], risk_by_commit: Dict[str, float]
) -> Tuple[int, int]:
    """
    Return (start_index, end_index) inclusive for the commit window covered by the
    given predictions, based on `commit_id` matching `all_commits.jsonl`'s `node`.
    """
    node_to_index = build_node_to_index(commits)
    indices: List[int] = []
    for commit_id in risk_by_commit.keys():
        idx = node_to_index.get(str(commit_id))
        if idx is not None:
            indices.append(idx)

    if not indices:
        raise RuntimeError("No prediction commit_ids matched any commit `node` values.")

    return min(indices), max(indices)


def build_risk_by_index(
    *,
    commits: List[Dict[str, Any]],
    risk_by_commit: Dict[str, float],
    window_start: int,
    window_end: int,
) -> List[Optional[float]]:
    """
    Build a per-commit risk list aligned to `commits` order.

    Within the simulation cutoff window [window_start, window_end], any commit
    that is missing from `risk_by_commit` is treated as clean (risk=0.0).
    Outside the window, values are set to None.
    """
    risk_by_index: List[Optional[float]] = [None] * len(commits)
    for i in range(window_start, window_end + 1):
        risk_by_index[i] = 0.0

    node_to_index = build_node_to_index(commits)
    for commit_id, risk in risk_by_commit.items():
        idx = node_to_index.get(str(commit_id))
        if idx is None:
            continue
        if idx < window_start or idx > window_end:
            continue
        risk_by_index[idx] = float(risk)

    return risk_by_index


def simulate_strategy_combo(
    *,
    bugs: List[Dict[str, Any]],
    bugs_by_id: Dict[str, Dict[str, Any]],
    node_to_index: Dict[str, int],
    nodes_by_index: List[Optional[str]],
    sorted_times_utc: List[datetime],
    sorted_time_indices: List[int],
    window_start: int,
    window_end: int,
    risk_by_index: List[Optional[float]],
    lookback_code: str,
    lookback: LookbackStrategy,
    bisection_code: str,
    bisection: BisectionStrategy,
) -> Dict[str, Any]:
    """Run simulation for a single (lookback, bisection) strategy pair."""
    def _fmt(idx: int) -> str:
        node = nodes_by_index[idx] if 0 <= idx < len(nodes_by_index) else None
        return f"{idx} ({node})" if node else str(idx)

    total_tests = 0
    total_lookback_tests = 0
    total_bisection_tests = 0
    total_culprits_found = 0

    skipped = {
        "not_regression": 0,
        "no_available_regressors": 0,
        "bad_not_in_window": 0,
        "good_not_in_window": 0,
        "culprit_after_bad": 0,
        "no_regressors_in_range": 0,
    }

    processed = 0
    logger.info("Simulating combo %s+%s over %d bugs", lookback_code, bisection_code, len(bugs))
    for bug in bugs:
        if not bug.get("regression", False):
            skipped["not_regression"] += 1
            continue

        available_regressor = bug.get("available_regressor")
        if not available_regressor:
            skipped["no_available_regressors"] += 1
            continue

        bug_time = _parse_bug_time(bug.get("bug_creation_time"))
        bad_index = _find_last_commit_before_or_at(
            sorted_times_utc, sorted_time_indices, bug_time
        )
        if bad_index is None:
            raise RuntimeError(
                f"Could not determine bad commit for bug_id={bug.get('bug_id')} "
                f"bug_creation_time={bug.get('bug_creation_time')!r}"
            )
        if bad_index > window_end:
            skipped["bad_not_in_window"] += 1
            continue

        reg_bug_id = str(available_regressor)
        reg_bug = bugs_by_id.get(reg_bug_id)
        if not reg_bug:
            raise KeyError(
                f"Bug {bug.get('bug_id')} references regressor bug_id={reg_bug_id}, "
                "but that bug_id is not present in the loaded dataset."
            )
        reg_rev = reg_bug.get("revision")
        if not reg_rev:
            raise ValueError(f"Regressor bug_id={reg_bug_id} is missing `revision`.")
        culprit_index = node_to_index.get(str(reg_rev))
        if culprit_index is None:
            raise KeyError(
                f"Regressor bug_id={reg_bug_id} revision={reg_rev} not found in commits list."
            )

        # Invariant for this simulation: the regression is observed at `bad_index`,
        # so `bad_index` must be at/after the regressor commit index. If
        # `culprit_index > bad_index`, then the "bad" commit is actually still
        # "good" under the linear-history test model, indicating inconsistent
        # timestamps/labels for this bug.
        if culprit_index > bad_index:
            skipped["culprit_after_bad"] += 1
            continue

        lookback_outcome = lookback.find_good_index(
            start_index=bad_index, culprit_index=culprit_index
        )
        if lookback_outcome.good_index is None:
            raise KeyError(
                "lookback did not find a clean commit for "
                f"bug_id={bug.get('bug_id')} start_index={_fmt(bad_index)} culprit_index={_fmt(culprit_index)}"
            )

        good_index = lookback_outcome.good_index
        lookback_tests = lookback_outcome.steps

        if good_index < window_start:
            skipped["good_not_in_window"] += 1
            continue
        if good_index >= bad_index:
            raise RuntimeError(
                f"Invalid good/bad ordering for bug_id={bug.get('bug_id')}: "
                f"good_index={_fmt(good_index)} >= bad_index={_fmt(bad_index)}"
            )

        if not (good_index < culprit_index <= bad_index):
            raise RuntimeError(
                f"Invalid culprit range for bug_id={bug.get('bug_id')}: "
                f"good_index={_fmt(good_index)}, bad_index={_fmt(bad_index)}, culprit_index={_fmt(culprit_index)}"
            )

        bisect_outcome = bisection.run(
            good_index=good_index,
            bad_index=bad_index,
            culprit_index=culprit_index,
            risk_by_index=risk_by_index,
        )

        total_culprits_found += 1 if bisect_outcome.found_index is not None else 0
        total_lookback_tests += lookback_tests
        total_bisection_tests += bisect_outcome.tests
        total_tests += lookback_tests + bisect_outcome.tests
        processed += 1

    logger.info(
        "Finished combo %s+%s: processed=%d total_tests=%d culprits_found=%d skipped=%s",
        lookback_code,
        bisection_code,
        processed,
        total_tests,
        total_culprits_found,
        skipped,
    )
    return {
        "combo": f"{lookback_code}+{bisection_code}",
        "total_tests": total_tests,
        "total_lookback_tests": total_lookback_tests,
        "total_bisection_tests": total_bisection_tests,
        "total_culprits_found": total_culprits_found,
        "bugs": {"processed": processed, "skipped": skipped},
    }


def _parse_bug_time(value: Any) -> datetime:
    """Parse bug creation time into a timezone-aware UTC datetime."""
    if not isinstance(value, str) or not value:
        raise ValueError("bug_creation_time must be a non-empty ISO datetime string")
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _find_last_commit_before_or_at(
    sorted_times_utc: List[datetime], sorted_indices: List[int], t: datetime
) -> Optional[int]:
    """
    Return the commit index of the commit with the largest timestamp <= t.
    """
    pos = bisect.bisect_right(sorted_times_utc, t) - 1
    return sorted_indices[pos] if pos >= 0 else None


def  _build_commit_time_search(commits: List[Dict[str, Any]]) -> Tuple[List[datetime], List[int]]:
    """Build time-sorted commit timestamps and their corresponding commit indices."""
    pairs: List[Tuple[datetime, int]] = []
    for idx, c in enumerate(commits):
        pairs.append((_commit_datetime_utc(c), idx))
    pairs.sort(key=lambda x: x[0])
    return [t for t, _ in pairs], [i for _, i in pairs]


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Load and normalize data for git-bisect simulation.")
    parser.add_argument(
        "--bugs-path",
        default=BUGS_PATH,
        help="Path to `mozilla_jit.jsonl` (defaults to the repo's `datasets/mozilla_jit/`).",
    )
    parser.add_argument(
        "--commits-path",
        default=COMMITS_PATH,
        help="Path to `all_commits.jsonl` (defaults to the repo's `datasets/mozilla_jit/`).",
    )
    parser.add_argument("--risk-final", default=RISK_FINAL_PATH)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Python logging level.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(REPO_ROOT, "analysis", "git_bisect", "simulation_baseline_final_test.json"),
        help="Where to write the JSON summary output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only simulate the first 1k bug rows (for quick iteration).",
    )
    return parser.parse_args()


def main() -> int:
    """Run the historical simulation and write a JSON summary."""
    args = get_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("Using bugs_path=%s", args.bugs_path)
    logger.info("Using commits_path=%s", args.commits_path)
    logger.info("Using risk_final=%s", args.risk_final)

    for p in (args.bugs_path, args.commits_path, args.risk_final):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Required input not found: {p}\n"
                "If you use DVC, ensure datasets are pulled into `datasets/`."
            )

    commits = load_commits(args.commits_path)
    final_risk_by_commit = load_risk_predictions(args.risk_final)
    nodes_by_index: List[Optional[str]] = [
        str(c.get("node")) if c.get("node") else None for c in commits
    ]

    node_to_index = build_node_to_index(commits)
    sorted_times_utc, sorted_time_indices = _build_commit_time_search(commits)

    window_start, window_end = _risk_window_from_predictions(commits, final_risk_by_commit)
    window_start_node = commits[window_start].get("node")
    window_end_node = commits[window_end].get("node")
    logger.info(
        "Risk window: [%d,%d] (%s..%s) over %d commits",
        window_start,
        window_end,
        window_start_node,
        window_end_node,
        window_end - window_start + 1,
    )
    risk_by_index = build_risk_by_index(
        commits=commits,
        risk_by_commit=final_risk_by_commit,
        window_start=window_start,
        window_end=window_end,
    )

    all_bugs = load_bugs_with_available_regressors(
        args.bugs_path,
        node_to_index=node_to_index,
        window_start=window_start,
        window_end=window_end,
    )
    bugs = all_bugs[:1000] if args.dry_run else all_bugs
    bugs_by_id = build_bug_id_index(all_bugs)
    logger.info("Loaded bugs=%d (simulating=%d dry_run=%s)", len(all_bugs), len(bugs), bool(args.dry_run))

    lookback_strategies: List[Tuple[str, LookbackStrategy]] = [
        ("FSL", FixedStrideLookback(stride=20)),
    ]
    bisection_strategies: List[Tuple[str, BisectionStrategy]] = [
        ("GB", GitBisectBaseline()),
    ]

    results: List[Dict[str, Any]] = []
    for lookback_code, lookback in lookback_strategies:
        for bisection_code, bisection in bisection_strategies:
            results.append(
                simulate_strategy_combo(
                    bugs=bugs,
                    bugs_by_id=bugs_by_id,
                    node_to_index=node_to_index,
                    nodes_by_index=nodes_by_index,
                    sorted_times_utc=sorted_times_utc,
                    sorted_time_indices=sorted_time_indices,
                    window_start=window_start,
                    window_end=window_end,
                    risk_by_index=risk_by_index,
                    lookback_code=lookback_code,
                    lookback=lookback,
                    bisection_code=bisection_code,
                    bisection=bisection,
                )
            )

    baseline_combo = f"{lookback_strategies[0][0]}+{bisection_strategies[0][0]}"
    summary: Dict[str, Any] = {
        "dataset": "final_test",
        "dry_run": bool(args.dry_run),
        "commit_window": {
            "start_index": window_start,
            "end_index": window_end,
            "start_node": window_start_node,
            "end_node": window_end_node,
            "num_commits": window_end - window_start + 1,
        },
        "bugs": {
            "loaded": len(all_bugs),
            "simulated": len(bugs),
        },
        "risk_predictions": {
            "path": os.path.relpath(args.risk_final, REPO_ROOT),
            "num_commits_with_risk": len(final_risk_by_commit),
        },
        "baseline": baseline_combo,
        "results": results
    }

    logger.info("Writing summary to %s", args.output)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
