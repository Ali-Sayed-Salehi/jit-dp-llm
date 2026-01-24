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
  - Evaluation (Optuna tuning): `python analysis/git_bisect/simulate.py --mopt-trials 200`
  - Replay tuned params on final test: `python analysis/git_bisect/simulate.py --final-only`
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from bisection import (
    BisectionStrategy,
    GitBisectBaseline,
    RiskSeries,
    SequentialWalkBackwardBisection,
    SequentialWalkForwardBisection,
    TopKRiskFirstBisection,
    RiskWeightedBisectionLogSurvival,
    RiskWeightedBisectionSum,
)
from lookback import (
    AdaptiveTimeWindowLookback,
    AdaptiveFixedStrideLookback,
    FixedStrideLookback,
    LookbackStrategy,
    NightlyBuildLookback,
    NoLookback,
    RiskAwareTriggerLookback,
    RiskWeightedLookbackLogSurvival,
    RiskWeightedLookbackSum,
    TimeWindowLookback,
)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

BUGS_PATH = os.path.join(REPO_ROOT, "datasets", "mozilla_jit", "mozilla_jit.jsonl")
COMMITS_PATH = os.path.join(REPO_ROOT, "datasets", "mozilla_jit", "all_commits.jsonl")

RISK_EVAL_PATH = os.path.join(REPO_ROOT, "analysis", "git_bisect", "risk_predictions_eval.json")
RISK_FINAL_PATH = os.path.join(REPO_ROOT, "analysis", "git_bisect", "risk_predictions_final_test.json")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrategySpec:
    """
    Declarative description of a strategy family for simulation/tuning.

    The `suggest_params` callable should use Optuna parameter names prefixed
    with `<code>_` (e.g. `FSLB_stride`) to avoid collisions across strategies.
    """
    code: str
    name: str
    default_params: Dict[str, Any]
    build: Callable[["PreparedInputs", Dict[str, Any]], Any]
    suggest_params: Optional[Callable[[Any], Dict[str, Any]]] = None  # optuna.Trial -> params


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
    max_tests_per_search = 0

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
            start_index=bad_index, culprit_index=culprit_index, start_time_utc=bug_time
        )
        if lookback_outcome.good_index is None:
            skipped["no_regressors_in_range"] += 1
            continue

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

        tests_per_search = int(lookback_tests) + int(bisect_outcome.tests)
        if tests_per_search > max_tests_per_search:
            max_tests_per_search = tests_per_search

        total_culprits_found += 1 if bisect_outcome.found_index is not None else 0
        total_lookback_tests += lookback_tests
        total_bisection_tests += bisect_outcome.tests
        total_tests += lookback_tests + bisect_outcome.tests
        processed += 1

    mean_tests_per_search: Optional[float]
    max_tests_per_search_out: Optional[int]
    if processed <= 0:
        mean_tests_per_search = None
        max_tests_per_search_out = None
    else:
        mean_tests_per_search = float(total_tests) / float(processed)
        max_tests_per_search_out = int(max_tests_per_search)

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
        "mean_tests_per_search": mean_tests_per_search,
        "max_tests_per_search": max_tests_per_search_out,
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


@dataclass(frozen=True)
class PreparedInputs:
    """Preloaded, pre-indexed inputs required to run a simulation on one dataset."""
    dataset: str
    bugs: List[Dict[str, Any]]
    bugs_by_id: Dict[str, Dict[str, Any]]
    node_to_index: Dict[str, int]
    nodes_by_index: List[Optional[str]]
    sorted_times_utc: List[datetime]
    sorted_time_indices: List[int]
    window_start: int
    window_end: int
    window_start_node: Optional[str]
    window_end_node: Optional[str]
    risk_by_index: Sequence[Optional[float]]
    risk_predictions_path: str
    num_commits_with_risk: int
    num_bugs_loaded: int


def prepare_inputs(
    *,
    dataset: str,
    bugs_path: str,
    commits_path: str,
    risk_path: str,
    dry_run: bool,
) -> PreparedInputs:
    """
    Load commits, predictions, and bugs; build indices and the commit window.

    This does all expensive IO and preprocessing once so that Optuna trials can
    reuse the same prepared state.
    """
    commits = load_commits(commits_path)
    risk_by_commit = load_risk_predictions(risk_path)
    nodes_by_index: List[Optional[str]] = [
        str(c.get("node")) if c.get("node") else None for c in commits
    ]

    node_to_index = build_node_to_index(commits)
    sorted_times_utc, sorted_time_indices = _build_commit_time_search(commits)

    window_start, window_end = _risk_window_from_predictions(commits, risk_by_commit)
    window_start_node = commits[window_start].get("node")
    window_end_node = commits[window_end].get("node")

    risk_by_index = build_risk_by_index(
        commits=commits,
        risk_by_commit=risk_by_commit,
        window_start=window_start,
        window_end=window_end,
    )
    risk_by_index = RiskSeries(risk_by_index)

    all_bugs = load_bugs_with_available_regressors(
        bugs_path,
        node_to_index=node_to_index,
        window_start=window_start,
        window_end=window_end,
    )
    bugs = all_bugs[:1000] if dry_run else all_bugs
    bugs_by_id = build_bug_id_index(all_bugs)

    return PreparedInputs(
        dataset=dataset,
        bugs=bugs,
        bugs_by_id=bugs_by_id,
        node_to_index=node_to_index,
        nodes_by_index=nodes_by_index,
        sorted_times_utc=sorted_times_utc,
        sorted_time_indices=sorted_time_indices,
        window_start=window_start,
        window_end=window_end,
        window_start_node=str(window_start_node) if window_start_node else None,
        window_end_node=str(window_end_node) if window_end_node else None,
        risk_by_index=risk_by_index,
        risk_predictions_path=risk_path,
        num_commits_with_risk=len(risk_by_commit),
        num_bugs_loaded=len(all_bugs),
    )


def run_combo(
    *,
    inputs: PreparedInputs,
    lookback_spec: StrategySpec,
    lookback_params: Dict[str, Any],
    bisection_spec: StrategySpec,
    bisection_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build concrete strategy instances and run a single simulation combo.

    Returns the `simulate_strategy_combo` metrics with an added `params` field.
    """
    lookback: LookbackStrategy = lookback_spec.build(inputs, lookback_params)
    bisection: BisectionStrategy = bisection_spec.build(inputs, bisection_params)

    res = simulate_strategy_combo(
        bugs=inputs.bugs,
        bugs_by_id=inputs.bugs_by_id,
        node_to_index=inputs.node_to_index,
        nodes_by_index=inputs.nodes_by_index,
        sorted_times_utc=inputs.sorted_times_utc,
        sorted_time_indices=inputs.sorted_time_indices,
        window_start=inputs.window_start,
        window_end=inputs.window_end,
        risk_by_index=inputs.risk_by_index,
        lookback_code=lookback_spec.code,
        lookback=lookback,
        bisection_code=bisection_spec.code,
        bisection=bisection,
    )
    res["params"] = {
        "lookback": {"code": lookback_spec.code, "name": lookback_spec.name, **lookback_params},
        "bisection": {"code": bisection_spec.code, "name": bisection_spec.name, **bisection_params},
    }
    return res


def _flatten_params_for_output(*, lookback_params: Dict[str, Any], bisection_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a compact, output-friendly parameter mapping.

    The final JSON output omits strategy metadata and includes only the actually
    used parameter values with stable, namespaced keys (e.g. `Lookback_stride`).
    """
    flattened: Dict[str, Any] = {}
    for key, value in lookback_params.items():
        flattened[f"Lookback_{key}"] = value
    for key, value in bisection_params.items():
        flattened[f"Bisection_{key}"] = value
    return flattened


def optimize_combo_params(
    *,
    inputs: PreparedInputs,
    lookback_spec: StrategySpec,
    bisection_spec: StrategySpec,
    n_trials: int,
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Tune a (lookback, bisection) combo on the given prepared dataset via Optuna.

    Objective:
      - Minimize `max_tests_per_search`.
      - Minimize `mean_tests_per_search`.
      - Treat trials as infeasible if any processed bug fails to identify a culprit
        (i.e., `total_culprits_found < processed`), returning `(inf, inf)`.

    Returns (best_lookback_params, best_bisection_params, payload) where payload
    includes `metrics` (re-run at best params) and `optuna` metadata.
    """
    if lookback_spec.suggest_params is None and bisection_spec.suggest_params is None:
        # Nothing to optimize: run once at defaults and return.
        best_lookback_params = dict(lookback_spec.default_params)
        best_bisection_params = dict(bisection_spec.default_params)
        best_res = run_combo(
            inputs=inputs,
            lookback_spec=lookback_spec,
            lookback_params=best_lookback_params,
            bisection_spec=bisection_spec,
            bisection_params=best_bisection_params,
        )
        optuna_meta = {
            "skipped": True,
            "reason": "no_suggest_params",
            "n_trials": 0,
            "seed": int(seed),
        }
        return best_lookback_params, best_bisection_params, {"metrics": best_res, "optuna": optuna_meta}

    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna is required. Install with `pip install optuna`.") from exc

    try:
        sampler = optuna.samplers.NSGAIISampler(seed=int(seed))
    except AttributeError:
        # Fallback for older Optuna versions where NSGAIISampler may not exist.
        sampler = optuna.samplers.RandomSampler(seed=int(seed))
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)

    def objective(trial: Any) -> Tuple[float, float]:
        lookback_params = (
            lookback_spec.suggest_params(trial)
            if lookback_spec.suggest_params is not None
            else dict(lookback_spec.default_params)
        )
        bisection_params = (
            bisection_spec.suggest_params(trial)
            if bisection_spec.suggest_params is not None
            else dict(bisection_spec.default_params)
        )

        res = run_combo(
            inputs=inputs,
            lookback_spec=lookback_spec,
            lookback_params=lookback_params,
            bisection_spec=bisection_spec,
            bisection_params=bisection_params,
        )

        processed = int(res["bugs"]["processed"])
        found = int(res["total_culprits_found"])
        if processed <= 0 or found < processed:
            return (float("inf"), float("inf"))

        max_tests_per_search = res.get("max_tests_per_search")
        mean_tests_per_search = res.get("mean_tests_per_search")
        if max_tests_per_search is None or mean_tests_per_search is None:
            return (float("inf"), float("inf"))
        return (float(max_tests_per_search), float(mean_tests_per_search))

    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)

    # Select a single point from the Pareto front. We primarily minimize the
    # worst-case search cost, and break ties by mean cost.
    if not study.best_trials:
        raise RuntimeError("Optuna produced no Pareto-optimal trials.")
    selected_trial = min(study.best_trials, key=lambda t: (float(t.values[0]), float(t.values[1])))
    best_trial_params_raw = dict(selected_trial.params)
    for key in best_trial_params_raw.keys():
        if not (
            key.startswith(f"{lookback_spec.code}_")
            or key.startswith(f"{bisection_spec.code}_")
        ):
            raise ValueError(
                "Optuna parameter names must be prefixed with strategy code "
                f"(expected {lookback_spec.code}_* or {bisection_spec.code}_*), got: {key}"
            )

    best_lookback_params = dict(lookback_spec.default_params)
    best_bisection_params = dict(bisection_spec.default_params)
    if lookback_spec.suggest_params is not None or bisection_spec.suggest_params is not None:
        # Only keys produced by suggest_* are present; keep defaults for other params.
        for k, v in best_trial_params_raw.items():
            if k.startswith(f"{lookback_spec.code}_"):
                best_lookback_params[k[len(f"{lookback_spec.code}_") :]] = v
            if k.startswith(f"{bisection_spec.code}_"):
                best_bisection_params[k[len(f"{bisection_spec.code}_") :]] = v

    # Re-run once at the chosen point to record metrics.
    best_res = run_combo(
        inputs=inputs,
        lookback_spec=lookback_spec,
        lookback_params=best_lookback_params,
        bisection_spec=bisection_spec,
        bisection_params=best_bisection_params,
    )
    optuna_meta = {
        "n_trials": int(n_trials),
        "seed": int(seed),
        "pareto_best_trial_count": int(len(study.best_trials)),
        "selected_trial_number": int(selected_trial.number),
        "selection": "lexicographic (min max_tests_per_search, then min mean_tests_per_search)",
        "selected_values": {
            "max_tests_per_search": float(selected_trial.values[0]),
            "mean_tests_per_search": float(selected_trial.values[1]),
        },
        "best_trial_params_raw": best_trial_params_raw,
    }
    return best_lookback_params, best_bisection_params, {"metrics": best_res, "optuna": optuna_meta}


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
    parser.add_argument("--risk-eval", default=RISK_EVAL_PATH)
    parser.add_argument("--risk-final", default=RISK_FINAL_PATH)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Python logging level.",
    )
    parser.add_argument(
        "--output-eval",
        default=os.path.join(REPO_ROOT, "analysis", "git_bisect", "results", "simulation_optuna_eval.json"),
        help="Where to write the EVAL Optuna tuning output JSON.",
    )
    parser.add_argument(
        "--output-final",
        default=os.path.join(REPO_ROOT, "analysis", "git_bisect", "results", "simulation_optuna_final_test.json"),
        help="Where to write the FINAL replay output JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only simulate the first 1k bug rows (for quick iteration).",
    )
    parser.add_argument(
        "--mopt-trials",
        type=int,
        default=200,
        help="Number of Optuna trials per (lookback,bisection) combo on EVAL.",
    )
    parser.add_argument(
        "--optuna-seed",
        type=int,
        default=42,
        help="Random seed for Optuna samplers.",
    )
    parser.add_argument(
        "--lookback",
        default="all",
        help=(
            "Comma-separated lookback strategy codes or names to simulate (default: all). "
            "Examples: 'NLB,FSLB,AFSLB' or 'no_lookback,fixed_stride'."
        ),
    )
    parser.add_argument(
        "--bisection",
        default="all",
        help=(
            "Comma-separated bisection strategy codes or names to simulate (default: all). "
            "Examples: 'GB,RWBS,RWBLS' or 'git_bisect,rwb-s'."
        ),
    )
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Skip eval tuning; read best params from --output-eval and run FINAL replay only.",
    )
    return parser.parse_args()


def _parse_csv_strategy_list(raw: str) -> List[str]:
    if raw is None:
        return []
    raw = str(raw).strip()
    if not raw:
        return []
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def _filter_strategy_specs(raw: str, *, specs: Sequence[StrategySpec], kind: str) -> List[StrategySpec]:
    """
    Filter a list of StrategySpec objects based on a user-provided CSV list.

    - "all" (default) selects all specs.
    - Tokens may be either `spec.code` or `spec.name`.
    """
    tokens = _parse_csv_strategy_list(raw)
    if not tokens:
        return list(specs)

    lowered = [t.lower() for t in tokens]
    if len(lowered) == 1 and lowered[0] == "all":
        return list(specs)

    by_code = {s.code: s for s in specs}
    by_name = {s.name: s for s in specs}

    selected_codes: List[str] = []
    unknown: List[str] = []
    for tok in tokens:
        if tok in by_code:
            selected_codes.append(tok)
        elif tok in by_name:
            selected_codes.append(by_name[tok].code)
        else:
            unknown.append(tok)

    if unknown:
        known_codes = ", ".join(s.code for s in specs)
        known_names = ", ".join(s.name for s in specs)
        bad = ", ".join(unknown)
        raise ValueError(
            f"Unknown {kind} strategy(s): {bad}. "
            f"Known {kind} codes: {known_codes}. "
            f"Known {kind} names: {known_names}."
        )

    selected = set(selected_codes)
    return [s for s in specs if s.code in selected]


def main() -> int:
    """Run the historical simulation and write a JSON summary."""
    args = get_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("Using bugs_path=%s", args.bugs_path)
    logger.info("Using commits_path=%s", args.commits_path)
    logger.info("Using risk_eval=%s", args.risk_eval)
    logger.info("Using risk_final=%s", args.risk_final)

    for p in (args.bugs_path, args.commits_path, args.risk_eval, args.risk_final):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Required input not found: {p}\n"
                "If you use DVC, ensure datasets are pulled into `datasets/`."
            )

    lookback_specs: List[StrategySpec] = [
        StrategySpec(
            code="NBLB",
            name="nightly_builds",
            default_params={},
            build=lambda inputs, _p: NightlyBuildLookback(
                sorted_times_utc=inputs.sorted_times_utc,
                sorted_time_indices=inputs.sorted_time_indices,
                window_start=inputs.window_start,
            ),
            suggest_params=None,
        ),
        StrategySpec(
            code="NLB",
            name="no_lookback",
            default_params={},
            build=lambda inputs, _p: NoLookback(window_start=inputs.window_start),
            suggest_params=None,
        ),
        StrategySpec(
            code="FSLB",
            name="fixed_stride",
            default_params={"stride": 20},
            build=lambda inputs, p: FixedStrideLookback(
                stride=int(p["stride"]),
                window_start=inputs.window_start,
            ),
            suggest_params=lambda trial: {
                "stride": trial.suggest_int("FSLB_stride", 1, 500, log=True)
            },
        ),
        StrategySpec(
            code="AFSLB",
            name="adaptive_fixed_stride",
            default_params={"stride": 20, "alpha": 0.5},
            build=lambda inputs, p: AdaptiveFixedStrideLookback(
                stride=int(p["stride"]),
                alpha=float(p["alpha"]),
                window_start=inputs.window_start,
            ),
            suggest_params=lambda trial: {
                "stride": trial.suggest_int("AFSLB_stride", 1, 500, log=True),
                "alpha": trial.suggest_float("AFSLB_alpha", 0.0, 1.0),
            },
        ),
        StrategySpec(
            code="RATLB",
            name="risk_aware_trigger",
            default_params={"threshold": 0.5},
            build=lambda inputs, p: RiskAwareTriggerLookback(
                risk_by_index=inputs.risk_by_index,
                threshold=float(p["threshold"]),
                window_start=inputs.window_start,
            ),
            suggest_params=lambda trial: {
                "threshold": trial.suggest_float("RATLB_threshold", 0.0, 1.0)
            },
        ),
        StrategySpec(
            code="RWLBS",
            name="rwlb-s",
            default_params={"threshold": 0.5},
            build=lambda inputs, p: RiskWeightedLookbackSum(
                risk_by_index=inputs.risk_by_index,
                threshold=float(p["threshold"]),
                window_start=inputs.window_start,
            ),
            suggest_params=lambda trial: {
                "threshold": trial.suggest_float("RWLBS_threshold", 0.0, 1.0)
            },
        ),
        StrategySpec(
            code="RWLBLS",
            name="rwlb-ls",
            default_params={"threshold": 0.5},
            build=lambda inputs, p: RiskWeightedLookbackLogSurvival(
                risk_by_index=inputs.risk_by_index,
                threshold=float(p["threshold"]),
                window_start=inputs.window_start,
            ),
            suggest_params=lambda trial: {
                "threshold": trial.suggest_float("RWLBLS_threshold", 0.0, 1.0)
            },
        ),
        StrategySpec(
            code="TWLB",
            name="time_window",
            default_params={"hours": 24},
            build=lambda inputs, p: TimeWindowLookback(
                sorted_times_utc=inputs.sorted_times_utc,
                sorted_time_indices=inputs.sorted_time_indices,
                hours=float(p["hours"]),
                window_start=inputs.window_start,
            ),
            suggest_params=lambda trial: {
                "hours": trial.suggest_int("TWLB_hours", 1, 24 * 30, log=True)
            },
        ),
        StrategySpec(
            code="ATWLB",
            name="adaptive_time_window",
            default_params={"hours": 24, "alpha": 0.5},
            build=lambda inputs, p: AdaptiveTimeWindowLookback(
                sorted_times_utc=inputs.sorted_times_utc,
                sorted_time_indices=inputs.sorted_time_indices,
                hours=float(p["hours"]),
                alpha=float(p["alpha"]),
                window_start=inputs.window_start,
            ),
            suggest_params=lambda trial: {
                "hours": trial.suggest_int("ATWLB_hours", 1, 24 * 30, log=True),
                "alpha": trial.suggest_float("ATWLB_alpha", 0.0, 1.0),
            },
        ),
    ]
    bisection_specs: List[StrategySpec] = [
        StrategySpec(
            code="GB",
            name="git_bisect",
            default_params={},
            build=lambda _inputs, _p: GitBisectBaseline(),
            suggest_params=None,
        ),
        StrategySpec(
            code="SWBB",
            name="swbb",
            default_params={},
            build=lambda _inputs, _p: SequentialWalkBackwardBisection(),
            suggest_params=None,
        ),
        StrategySpec(
            code="SWFB",
            name="swfb",
            default_params={},
            build=lambda _inputs, _p: SequentialWalkForwardBisection(),
            suggest_params=None,
        ),
        StrategySpec(
            code="TKRB",
            name="tkrb-k",
            default_params={"k": 10},
            build=lambda _inputs, p: TopKRiskFirstBisection(k=int(p["k"])),
            suggest_params=lambda trial: {
                "k": trial.suggest_int("TKRB_k", 1, 500, log=True)
            },
        ),
        StrategySpec(
            code="RWBS",
            name="rwb-s",
            default_params={},
            build=lambda _inputs, _p: RiskWeightedBisectionSum(),
            suggest_params=None,
        ),
        StrategySpec(
            code="RWBLS",
            name="rwb-ls",
            default_params={},
            build=lambda _inputs, _p: RiskWeightedBisectionLogSurvival(),
            suggest_params=None,
        ),
    ]

    lookback_specs = _filter_strategy_specs(args.lookback, specs=lookback_specs, kind="lookback")
    bisection_specs = _filter_strategy_specs(args.bisection, specs=bisection_specs, kind="bisection")
    if not lookback_specs:
        raise ValueError("No lookback strategies selected. Use --lookback all or provide at least one code/name.")
    if not bisection_specs:
        raise ValueError("No bisection strategies selected. Use --bisection all or provide at least one code/name.")
    logger.info("Selected lookback strategies: %s", ",".join(s.code for s in lookback_specs))
    logger.info("Selected bisection strategies: %s", ",".join(s.code for s in bisection_specs))

    tuned_params_by_combo: Dict[str, Dict[str, Dict[str, Any]]] = {}
    eval_summary: Optional[Dict[str, Any]] = None
    baseline_combo = "NLB+GB"

    def _as_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _pct_vs_baseline(*, value: Optional[float], baseline_value: Optional[float]) -> Optional[float]:
        if baseline_value is None or baseline_value <= 0.0:
            return None
        if value is None:
            return None
        return 100.0 * (float(baseline_value) - float(value)) / float(baseline_value)

    def _annotate_results_with_baseline(
        *,
        results: List[Dict[str, Any]],
        metric_specs: Sequence[Tuple[Any, Any]],
        get_total_tests: Any,
        get_mean_tests_per_search: Any,
        get_max_tests_per_search: Any,
    ) -> Dict[str, Any]:
        baseline_row = next((r for r in results if str(r.get("combo")) == baseline_combo), None)
        if baseline_row is None:
            logger.warning(
                "Baseline combo %r not found in results (likely filtered out); skipping baseline-relative metrics.",
                baseline_combo,
            )
        else:
            try:
                baseline_processed = int(
                    (((baseline_row.get("metrics") or {}).get("bugs") or {}).get("processed") or 0)
                )
            except (TypeError, ValueError):
                baseline_processed = None

            if baseline_processed is not None:
                mismatched = False
                for row in results:
                    try:
                        processed = int(
                            ((((row.get("metrics") or {}).get("bugs") or {}).get("processed") or 0))
                        )
                    except (TypeError, ValueError):
                        continue
                    if processed != baseline_processed:
                        mismatched = True
                        break
                if mismatched:
                    logger.warning(
                        "Some combos processed a different number of bugs than the baseline "
                        "(baseline=%s processed=%d); comparisons may be unfair.",
                        baseline_combo,
                        baseline_processed,
                    )

            for get_value, set_pct in metric_specs:
                baseline_value = _as_float(get_value(baseline_row))
                for row in results:
                    pct = _pct_vs_baseline(value=_as_float(get_value(row)), baseline_value=baseline_value)
                    set_pct(row, pct)

        def _best_combo(get_value: Any) -> Optional[str]:
            best_combo: Optional[str] = None
            best_value: Optional[float] = None
            for row in results:
                v = _as_float(get_value(row))
                if v is None:
                    continue
                if best_value is None or v < best_value:
                    best_value = float(v)
                    best_combo = str(row.get("combo"))
            return best_combo

        best_row = min(results, key=lambda r: int(get_total_tests(r)))
        return {
            "best_combo_by_total_tests": str(best_row.get("combo")),
            "best_combo_by_mean_tests_per_search": _best_combo(get_mean_tests_per_search),
            "best_combo_by_max_tests_per_search": _best_combo(get_max_tests_per_search),
        }

    if not args.final_only:
        eval_inputs = prepare_inputs(
            dataset="eval",
            bugs_path=args.bugs_path,
            commits_path=args.commits_path,
            risk_path=args.risk_eval,
            dry_run=bool(args.dry_run),
        )
        logger.info(
            "EVAL risk window: [%d,%d] (%s..%s) over %d commits",
            eval_inputs.window_start,
            eval_inputs.window_end,
            eval_inputs.window_start_node,
            eval_inputs.window_end_node,
            eval_inputs.window_end - eval_inputs.window_start + 1,
        )
        logger.info(
            "Loaded bugs=%d (simulating=%d dry_run=%s) for EVAL",
            eval_inputs.num_bugs_loaded,
            len(eval_inputs.bugs),
            bool(args.dry_run),
        )

        eval_results: List[Dict[str, Any]] = []
        for lookback_spec in lookback_specs:
            for bisection_spec in bisection_specs:
                combo_key = f"{lookback_spec.code}+{bisection_spec.code}"
                is_baseline = (
                    lookback_spec.name == NightlyBuildLookback.name
                    and bisection_spec.name == GitBisectBaseline.name
                )
                logger.info("Optuna tuning combo=%s trials=%d", combo_key, int(args.mopt_trials))
                lookback_params, bisection_params, payload = optimize_combo_params(
                    inputs=eval_inputs,
                    lookback_spec=lookback_spec,
                    bisection_spec=bisection_spec,
                    n_trials=int(args.mopt_trials),
                    seed=int(args.optuna_seed),
                )
                tuned_params_by_combo[combo_key] = {
                    "lookback": lookback_params,
                    "bisection": bisection_params,
                }
                eval_results.append(
                    {
                        "combo": combo_key,
                        "lookback": {"code": lookback_spec.code, "name": lookback_spec.name},
                        "bisection": {"code": bisection_spec.code, "name": bisection_spec.name},
                        "best_params": tuned_params_by_combo[combo_key],
                        **payload,
                    }
                )

        eval_comparison = _annotate_results_with_baseline(
            results=eval_results,
            metric_specs=[
                (
                    lambda r: (r.get("metrics") or {}).get("total_tests"),
                    lambda r, pct: r.setdefault("metrics", {}).update({"total_tests_saved_vs_baseline_pct": pct}),
                ),
                (
                    lambda r: (r.get("metrics") or {}).get("mean_tests_per_search"),
                    lambda r, pct: r.setdefault("metrics", {}).update(
                        {"mean_tests_per_search_saved_vs_baseline_pct": pct}
                    ),
                ),
                (
                    lambda r: (r.get("metrics") or {}).get("max_tests_per_search"),
                    lambda r, pct: r.setdefault("metrics", {}).update({"max_tests_per_search_saved_vs_baseline_pct": pct}),
                ),
            ],
            get_total_tests=lambda r: (r.get("metrics") or {}).get("total_tests", 0),
            get_mean_tests_per_search=lambda r: (r.get("metrics") or {}).get("mean_tests_per_search"),
            get_max_tests_per_search=lambda r: (r.get("metrics") or {}).get("max_tests_per_search"),
        )
        eval_summary = {
            "dataset": "eval",
            "dry_run": bool(args.dry_run),
            **eval_comparison,
            "commit_window": {
                "start_index": eval_inputs.window_start,
                "end_index": eval_inputs.window_end,
                "start_node": eval_inputs.window_start_node,
                "end_node": eval_inputs.window_end_node,
                "num_commits": eval_inputs.window_end - eval_inputs.window_start + 1,
            },
            "bugs": {"loaded": eval_inputs.num_bugs_loaded, "simulated": len(eval_inputs.bugs)},
            "risk_predictions": {
                "path": os.path.relpath(eval_inputs.risk_predictions_path, REPO_ROOT),
                "num_commits_with_risk": eval_inputs.num_commits_with_risk,
            },
            "optimization": {
                "mopt_trials_per_combo": int(args.mopt_trials),
                "optuna_seed": int(args.optuna_seed),
                "objective": (
                    "minimize max_tests_per_search and mean_tests_per_search; "
                    "infeasible if culprits_found < processed"
                ),
            },
            "results": eval_results,
        }

        logger.info("Writing EVAL summary to %s", args.output_eval)
        os.makedirs(os.path.dirname(args.output_eval), exist_ok=True)
        with open(args.output_eval, "w", encoding="utf-8") as f:
            json.dump(eval_summary, f, indent=2, sort_keys=True)
            f.write("\n")
        print(args.output_eval)
    else:
        logger.info("Loading tuned params from %s", args.output_eval)
        with open(args.output_eval, "r", encoding="utf-8") as f:
            eval_summary = json.load(f)
        for row in eval_summary.get("results", []):
            combo_key = row.get("combo")
            best_params = row.get("best_params")
            if combo_key and isinstance(best_params, dict):
                tuned_params_by_combo[str(combo_key)] = best_params

    final_inputs = prepare_inputs(
        dataset="final_test",
        bugs_path=args.bugs_path,
        commits_path=args.commits_path,
        risk_path=args.risk_final,
        dry_run=bool(args.dry_run),
    )
    logger.info(
        "FINAL risk window: [%d,%d] (%s..%s) over %d commits",
        final_inputs.window_start,
        final_inputs.window_end,
        final_inputs.window_start_node,
        final_inputs.window_end_node,
        final_inputs.window_end - final_inputs.window_start + 1,
    )
    logger.info(
        "Loaded bugs=%d (simulating=%d dry_run=%s) for FINAL",
        final_inputs.num_bugs_loaded,
        len(final_inputs.bugs),
        bool(args.dry_run),
    )

    final_results: List[Dict[str, Any]] = []
    for lookback_spec in lookback_specs:
        for bisection_spec in bisection_specs:
            combo_key = f"{lookback_spec.code}+{bisection_spec.code}"
            params = tuned_params_by_combo.get(combo_key) or {
                "lookback": dict(lookback_spec.default_params),
                "bisection": dict(bisection_spec.default_params),
            }
            lookback_params = dict(params.get("lookback") or {})
            bisection_params = dict(params.get("bisection") or {})
            final_results.append(
                run_combo(
                    inputs=final_inputs,
                    lookback_spec=lookback_spec,
                    lookback_params=lookback_params,
                    bisection_spec=bisection_spec,
                    bisection_params=bisection_params,
                )
            )
            final_results[-1].pop("bugs", None)
            final_results[-1]["params"] = _flatten_params_for_output(
                lookback_params=lookback_params,
                bisection_params=bisection_params,
            )

    final_comparison = _annotate_results_with_baseline(
        results=final_results,
        metric_specs=[
            (
                lambda r: r.get("total_tests"),
                lambda r, pct: r.__setitem__("total_tests_saved_vs_baseline_pct", pct),
            ),
            (
                lambda r: r.get("mean_tests_per_search"),
                lambda r, pct: r.__setitem__("mean_tests_per_search_saved_vs_baseline_pct", pct),
            ),
            (
                lambda r: r.get("max_tests_per_search"),
                lambda r, pct: r.__setitem__("max_tests_per_search_saved_vs_baseline_pct", pct),
            ),
        ],
        get_total_tests=lambda r: r.get("total_tests", 0),
        get_mean_tests_per_search=lambda r: r.get("mean_tests_per_search"),
        get_max_tests_per_search=lambda r: r.get("max_tests_per_search"),
    )
    final_summary: Dict[str, Any] = {
        "dataset": "final_test",
        "dry_run": bool(args.dry_run),
        **final_comparison,
        "commit_window": {
            "start_index": final_inputs.window_start,
            "end_index": final_inputs.window_end,
            "start_node": final_inputs.window_start_node,
            "end_node": final_inputs.window_end_node,
            "num_commits": final_inputs.window_end - final_inputs.window_start + 1,
        },
        "bugs": {"loaded": final_inputs.num_bugs_loaded, "simulated": len(final_inputs.bugs)},
        "risk_predictions": {
            "path": os.path.relpath(final_inputs.risk_predictions_path, REPO_ROOT),
            "num_commits_with_risk": final_inputs.num_commits_with_risk,
        },
        "tuned_from_eval": {
            "path": os.path.relpath(args.output_eval, REPO_ROOT),
            "present": bool(eval_summary),
        },
        "results": final_results,
    }

    logger.info("Writing FINAL summary to %s", args.output_final)
    os.makedirs(os.path.dirname(args.output_final), exist_ok=True)
    with open(args.output_final, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, sort_keys=True)
        f.write("\n")

    print(args.output_final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
