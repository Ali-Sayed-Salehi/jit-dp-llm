from __future__ import annotations

"""
Lookback strategies for selecting a known-good commit.

Given a regression observation at some `start_index` and the regressor commit(s),
lookback strategies model how a developer might search backwards in history to
find a commit that does not yet contain the regression.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Protocol, Sequence

import bisect

from bisection import RiskSeries

logger = logging.getLogger(__name__)

def _forced_fallback_outcome(
    *,
    steps: int,
    window_start: int,
    culprit_index: int,
    known_results: Optional[dict[int, bool]] = None,
) -> "LookbackOutcome":
    """
    Return a forced-fallback outcome using `window_start` as the good boundary.

    Some strategies support a `max_trials` limit. When it is hit, we stop
    searching and return `window_start` (if it is strictly before the culprit)
    so bisection can proceed.
    """
    good = int(window_start) if int(window_start) < int(culprit_index) else None
    return LookbackOutcome(
        good_index=good,
        steps=int(steps),
        known_results={} if known_results is None else dict(known_results),
    )


@dataclass(frozen=True)
class LookbackOutcome:
    """Result of selecting a known-good commit for a single bug/regression."""
    good_index: Optional[int]
    steps: int
    known_results: dict[int, bool] = field(default_factory=dict)


class LookbackStrategy(Protocol):
    name: str

    def find_good_index(
        self, *, start_index: int, culprit_index: int, start_time_utc: Optional[datetime] = None
    ) -> LookbackOutcome:
        """Return the chosen good commit index for a bug, plus any bookkeeping."""
        ...


class NoLookback:
    """
    No Lookback (NLB).

    Do not search for a known-good commit via additional tests. Instead, use
    a fixed clean baseline commit (`window_start`, index 0 by default) as the
    known-good boundary and let bisection search within (window_start, start_index].
    """

    name = "no_lookback"

    def __init__(self, *, window_start: int = 0) -> None:
        if window_start < 0:
            raise ValueError("window_start must be non-negative")
        self.window_start = int(window_start)

    def find_good_index(
        self, *, start_index: int, culprit_index: int, start_time_utc: Optional[datetime] = None
    ) -> LookbackOutcome:
        _ = start_time_utc  # not used by this strategy

        start_index = int(start_index)
        culprit_index = int(culprit_index)

        if start_index <= self.window_start:
            # There is no commit < start_index within the simulation window.
            return LookbackOutcome(good_index=None, steps=0)

        good = self.window_start if self.window_start < culprit_index else None
        return LookbackOutcome(good_index=good, steps=0)


class FixedStrideLookback:
    """
    Baseline lookback:
      - Starting from `start_index`, step backwards by `stride` commits.
      - Return the first commit index that does not contain the culprit.

    In a linear history, a commit at index i "contains" the culprit at index c
    if i >= c. Therefore, "does not contain" is equivalent to i < c.
    """

    name = "fixed_stride"

    def __init__(self, stride: int = 20, *, window_start: int = 0, max_trials: Optional[int] = None) -> None:
        """Create a fixed-stride lookback strategy."""
        if stride <= 0:
            raise ValueError("stride must be positive")
        if window_start < 0:
            raise ValueError("window_start must be non-negative")
        if max_trials is not None and int(max_trials) <= 0:
            raise ValueError("max_trials must be positive")
        self.stride = stride
        self.window_start = int(window_start)
        self.max_trials = int(max_trials) if max_trials is not None else None

    def find_good_index(
        self, *, start_index: int, culprit_index: int, start_time_utc: Optional[datetime] = None
    ) -> LookbackOutcome:
        """Find a "good" commit by stepping back in fixed increments."""
        _ = start_time_utc  # not used by this strategy

        culprit_index = int(culprit_index)
        idx = int(start_index)
        steps = 0
        known_results: dict[int, bool] = {}
        while True:
            if self.max_trials is not None and steps >= self.max_trials:
                return _forced_fallback_outcome(
                    steps=steps,
                    window_start=self.window_start,
                    culprit_index=culprit_index,
                    known_results=known_results,
                )

            candidate = idx - int(self.stride)
            if candidate < self.window_start:
                candidate = self.window_start
            candidate = int(candidate)

            # No further progress possible: there is no commit < start_index within the window.
            if candidate >= idx:
                logger.debug(
                    "Lookback failed to find in-window good_index (start=%d culprit=%d stride=%d window_start=%d steps=%d)",
                    start_index,
                    culprit_index,
                    self.stride,
                    self.window_start,
                    steps,
                )
                return LookbackOutcome(good_index=None, steps=steps, known_results=known_results)

            steps += 1
            known_results[candidate] = bool(candidate >= culprit_index)
            if candidate < culprit_index:
                logger.debug(
                    "Lookback found good_index=%d after steps=%d (start=%d culprit=%d stride=%d window_start=%d)",
                    candidate,
                    steps,
                    start_index,
                    culprit_index,
                    self.stride,
                    self.window_start,
                )
                return LookbackOutcome(good_index=candidate, steps=steps, known_results=known_results)

            if candidate == self.window_start:
                logger.debug(
                    "Lookback failed to find in-window good_index (start=%d culprit=%d stride=%d window_start=%d steps=%d)",
                    start_index,
                    culprit_index,
                    self.stride,
                    self.window_start,
                    steps,
                )
                return LookbackOutcome(good_index=None, steps=steps, known_results=known_results)

            idx = candidate


class FixedStrideLookbackAdaptiveDecrease:
    """
    Fixed-stride lookback with adaptive decrease (FSLB-AD).

    Like FixedStrideLookback (FSLB), but reduces the stride after each failed
    test.

    Policy:
      - Iteration 1: step back by `stride` commits and test that commit.
      - Iteration k: step back by `stride * (alpha ** (k-1))` commits and test.
      - On failure, the tested commit becomes the new known-bad boundary.
      - Return the first tested commit that passes.

    `steps` counts only tested commits (the known-bad `start_index` is not
    re-tested).
    """

    name = "fixed_stride-ad"

    def __init__(
        self, stride: int = 20, *, alpha: float = 0.5, window_start: int = 0, max_trials: Optional[int] = None
    ) -> None:
        if stride <= 0:
            raise ValueError("stride must be positive")
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("alpha must be in [0,1]")
        if window_start < 0:
            raise ValueError("window_start must be non-negative")
        if max_trials is not None and int(max_trials) <= 0:
            raise ValueError("max_trials must be positive")
        self.stride = int(stride)
        self.alpha = float(alpha)
        self.window_start = int(window_start)
        self.max_trials = int(max_trials) if max_trials is not None else None

    def _stride_for_step(self, steps_executed: int) -> int:
        # Use ceil so the stride never drops to 0, which would stall progress.
        return max(1, int(math.ceil(float(self.stride) * float(self.alpha ** int(steps_executed)))))

    def find_good_index(
        self, *, start_index: int, culprit_index: int, start_time_utc: Optional[datetime] = None
    ) -> LookbackOutcome:
        _ = start_time_utc  # not used by this strategy

        start_index = int(start_index)
        culprit_index = int(culprit_index)

        if start_index <= self.window_start:
            # There is no commit < start_index within the simulation window.
            return LookbackOutcome(good_index=None, steps=0)

        idx = int(start_index)
        steps = 0
        known_results: dict[int, bool] = {}
        while idx > self.window_start:
            if self.max_trials is not None and steps >= self.max_trials:
                return _forced_fallback_outcome(
                    steps=steps,
                    window_start=self.window_start,
                    culprit_index=culprit_index,
                    known_results=known_results,
                )

            stride = self._stride_for_step(steps)
            candidate = idx - int(stride)
            if candidate < self.window_start:
                candidate = self.window_start
            if candidate >= idx:
                candidate = idx - 1
            candidate = int(candidate)

            steps += 1
            known_results[candidate] = bool(candidate >= culprit_index)
            if candidate < culprit_index:
                logger.debug(
                    "FSLB-AD found good_index=%d after steps=%d (start=%d culprit=%d stride=%d alpha=%s window_start=%d)",
                    candidate,
                    steps,
                    start_index,
                    culprit_index,
                    self.stride,
                    self.alpha,
                    self.window_start,
                )
                return LookbackOutcome(good_index=int(candidate), steps=steps, known_results=known_results)

            if candidate == self.window_start:
                return LookbackOutcome(good_index=None, steps=steps, known_results=known_results)

            idx = int(candidate)

        return LookbackOutcome(good_index=None, steps=steps, known_results=known_results)


class FixedStrideLookbackAdaptiveIncrease:
    """
    Fixed-stride lookback with adaptive increase (FSLB-AI).

    Like FixedStrideLookback (FSLB), but increases the stride after each failed
    test.

    Policy:
      - Iteration 1: step back by `stride` commits and test that commit.
      - Iteration k: step back by `stride * (alpha ** (k-1))` commits and test,
        with `alpha > 1`.
      - On failure, the tested commit becomes the new known-bad boundary.
      - Return the first tested commit that passes.

    `steps` counts only tested commits (the known-bad `start_index` is not
    re-tested).
    """

    name = "fixed_stride-ai"

    def __init__(
        self, stride: int = 20, *, alpha: float = 2.0, window_start: int = 0, max_trials: Optional[int] = None
    ) -> None:
        if stride <= 0:
            raise ValueError("stride must be positive")
        if alpha <= 1.0:
            raise ValueError("alpha must be > 1")
        if window_start < 0:
            raise ValueError("window_start must be non-negative")
        if max_trials is not None and int(max_trials) <= 0:
            raise ValueError("max_trials must be positive")
        self.stride = int(stride)
        self.alpha = float(alpha)
        self.window_start = int(window_start)
        self.max_trials = int(max_trials) if max_trials is not None else None

    def _stride_for_step(self, steps_executed: int) -> int:
        return max(1, int(math.ceil(float(self.stride) * float(self.alpha ** int(steps_executed)))))

    def find_good_index(
        self, *, start_index: int, culprit_index: int, start_time_utc: Optional[datetime] = None
    ) -> LookbackOutcome:
        _ = start_time_utc  # not used by this strategy

        start_index = int(start_index)
        culprit_index = int(culprit_index)

        if start_index <= self.window_start:
            # There is no commit < start_index within the simulation window.
            return LookbackOutcome(good_index=None, steps=0)

        idx = int(start_index)
        steps = 0
        known_results: dict[int, bool] = {}
        while idx > self.window_start:
            if self.max_trials is not None and steps >= self.max_trials:
                return _forced_fallback_outcome(
                    steps=steps,
                    window_start=self.window_start,
                    culprit_index=culprit_index,
                    known_results=known_results,
                )

            stride = self._stride_for_step(steps)
            candidate = idx - int(stride)
            if candidate < self.window_start:
                candidate = self.window_start
            if candidate >= idx:
                candidate = idx - 1
            candidate = int(candidate)

            steps += 1
            known_results[candidate] = bool(candidate >= culprit_index)
            if candidate < culprit_index:
                logger.debug(
                    "FSLB-AI found good_index=%d after steps=%d (start=%d culprit=%d stride=%d alpha=%s window_start=%d)",
                    candidate,
                    steps,
                    start_index,
                    culprit_index,
                    self.stride,
                    self.alpha,
                    self.window_start,
                )
                return LookbackOutcome(good_index=int(candidate), steps=steps, known_results=known_results)

            if candidate == self.window_start:
                return LookbackOutcome(good_index=None, steps=steps, known_results=known_results)

            idx = int(candidate)

        return LookbackOutcome(good_index=None, steps=steps, known_results=known_results)


class NightlyBuildLookback:
    """
    Nightly-build lookback:
      - From the regression observation time, define "today" as that date in UTC.
      - Test the nightly builds going backwards day by day.
      - A nightly build for day D is the last commit with timestamp strictly before
        00:00 UTC on day D+1 (i.e., the last commit "submitted the day before").

    Selection rule:
      - Return the first nightly build commit index that does not contain the culprit
        (i.e. is strictly before `culprit_index` under the linear history model).
    """

    name = "nightly_builds"

    def __init__(
        self,
        *,
        sorted_times_utc: Sequence[datetime],
        sorted_time_indices: Sequence[int],
        window_start: int = 0,
    ) -> None:
        if len(sorted_times_utc) != len(sorted_time_indices):
            raise ValueError("sorted_times_utc and sorted_time_indices must have the same length")
        if not sorted_times_utc:
            raise ValueError("sorted_times_utc must be non-empty")
        if window_start < 0:
            raise ValueError("window_start must be non-negative")
        self.sorted_times_utc = list(sorted_times_utc)
        self.sorted_time_indices = list(sorted_time_indices)
        self.window_start = int(window_start)

    def _last_commit_strictly_before(self, t: datetime) -> Optional[int]:
        import bisect

        pos = bisect.bisect_left(self.sorted_times_utc, t) - 1
        return self.sorted_time_indices[pos] if pos >= 0 else None

    def find_good_index(
        self, *, start_index: int, culprit_index: int, start_time_utc: Optional[datetime] = None
    ) -> LookbackOutcome:
        if start_time_utc is None:
            raise ValueError("NightlyBuildLookback requires start_time_utc to be provided")
        if start_time_utc.tzinfo is None:
            start_time_utc = start_time_utc.replace(tzinfo=timezone.utc)
        start_time_utc = start_time_utc.astimezone(timezone.utc)

        culprit_index = int(culprit_index)
        start_index = int(start_index)

        if start_index <= self.window_start:
            # There is no commit < start_index within the simulation window.
            return LookbackOutcome(good_index=None, steps=0)

        midnight_today = datetime(
            start_time_utc.year,
            start_time_utc.month,
            start_time_utc.day,
            tzinfo=timezone.utc,
        )

        steps = 0
        known_results: dict[int, bool] = {}
        cutoff = midnight_today
        while True:
            nightly_idx = self._last_commit_strictly_before(cutoff)
            if nightly_idx is None:
                raise RuntimeError(
                    "Nightly lookback failed: no commits before "
                    f"cutoff={cutoff.isoformat()} (start_index={start_index}, culprit_index={culprit_index}, steps={steps})"
                )

            if nightly_idx < self.window_start:
                nightly_idx = self.window_start

            # If the nightly build resolves to the already-known "bad" observation commit
            # (possible when the bug is observed soon after midnight with no intervening commits),
            # skip it without counting a new test.
            if nightly_idx >= start_index:
                cutoff = cutoff - timedelta(days=1)
                continue

            steps += 1
            known_results[int(nightly_idx)] = bool(int(nightly_idx) >= culprit_index)
            if nightly_idx < culprit_index:
                logger.debug(
                    "Nightly lookback found good_index=%d after steps=%d (start=%d culprit=%d)",
                    nightly_idx,
                    steps,
                    start_index,
                    culprit_index,
                )
                return LookbackOutcome(good_index=nightly_idx, steps=steps, known_results=known_results)

            if nightly_idx == self.window_start:
                # No earlier in-window commit can be tested.
                return LookbackOutcome(good_index=None, steps=steps, known_results=known_results)

            cutoff = cutoff - timedelta(days=1)


class RiskAwareTriggerLookback:
    """
    Risk-Aware Trigger Lookback (RATLB).

    Starting from a known-bad commit index (`start_index`), walk backwards in the
    commit history looking for "trigger" commits whose predicted risk exceeds a
    threshold `T`. For each trigger commit at index i, test the commit
    immediately before it (i-1). The first tested commit that passes is returned
    as the known-good commit.

    Notes:
      - The scan through commits is free; `steps` counts only the number of
        tested commits.
      - `risk_by_index` values of None are treated as 0.0 risk.
      - If no triggers are found, the strategy falls back to testing the first
        commit in the simulation window (`window_start`).
    """

    name = "risk_aware_trigger"

    def __init__(
        self,
        *,
        risk_by_index: Sequence[Optional[float]],
        threshold: float = 0.5,
        window_start: int = 0,
        max_trials: Optional[int] = None,
    ) -> None:
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("threshold must be in [0,1]")
        if window_start < 0:
            raise ValueError("window_start must be non-negative")
        if max_trials is not None and int(max_trials) <= 0:
            raise ValueError("max_trials must be positive")
        self.risk_by_index = risk_by_index
        self.threshold = float(threshold)
        self.window_start = int(window_start)
        self.max_trials = int(max_trials) if max_trials is not None else None

    def _risk(self, idx: int) -> float:
        v = self.risk_by_index[idx]
        return 0.0 if v is None else float(v)

    def find_good_index(
        self, *, start_index: int, culprit_index: int, start_time_utc: Optional[datetime] = None
    ) -> LookbackOutcome:
        _ = start_time_utc  # not used by this strategy

        start_index = int(start_index)
        culprit_index = int(culprit_index)

        # Defensive clamp: `risk_by_index` is aligned to commit indices.
        max_idx = len(self.risk_by_index) - 1
        search_idx = min(start_index, max_idx)

        steps = 0
        known_results: dict[int, bool] = {}
        min_trigger_idx = max(self.window_start + 1, 1)

        while True:
            if self.max_trials is not None and steps >= self.max_trials:
                return _forced_fallback_outcome(
                    steps=steps,
                    window_start=self.window_start,
                    culprit_index=culprit_index,
                    known_results=known_results,
                )

            trigger_idx: Optional[int] = None
            i = search_idx
            # We need i-1 to exist and we do not test outside the simulation window,
            # so only consider triggers at i >= window_start+1.
            while i >= min_trigger_idx:
                if self._risk(i) > self.threshold:
                    trigger_idx = i
                    break
                i -= 1

            if trigger_idx is None:
                # No triggers found: fall back to testing the first commit in the window.
                steps += 1
                known_results[int(self.window_start)] = bool(int(self.window_start) >= culprit_index)
                good = self.window_start if self.window_start < culprit_index else None
                return LookbackOutcome(good_index=good, steps=steps, known_results=known_results)

            candidate = int(trigger_idx - 1)
            steps += 1
            known_results[candidate] = bool(candidate >= culprit_index)
            if candidate < culprit_index:
                logger.debug(
                    "RATLB found good_index=%d after steps=%d (start=%d culprit=%d threshold=%s trigger=%d)",
                    candidate,
                    steps,
                    start_index,
                    culprit_index,
                    self.threshold,
                    trigger_idx,
                )
                return LookbackOutcome(good_index=candidate, steps=steps, known_results=known_results)

            if candidate <= self.window_start:
                # No earlier in-window candidate can be tested.
                return LookbackOutcome(good_index=None, steps=steps, known_results=known_results)

            # Candidate still contains the culprit; continue scanning earlier history.
            search_idx = candidate


class RiskWeightedLookbackSum:
    """
    Risk-Weighted Lookback (RWLB-S).

    Starting from a known-bad commit index (`start_index`), move backwards in
    history until the cumulative predicted risk reaches a threshold `T`, then
    test that commit. If it still fails, treat it as the new known-bad boundary
    and repeat. The first passing tested commit is returned as the known-good
    commit.

    RWLB-S computes cumulative risk as the *sum* of per-commit probabilities.

    Notes:
      - The scan through commits is free; `steps` counts only tested commits.
      - `risk_by_index` values of None are treated as 0.0 risk (via RiskSeries).
      - If the threshold is never reached before `window_start`, the strategy
        tests `window_start` as a fallback.
    """

    name = "rwlb-s"

    def __init__(
        self,
        *,
        risk_by_index: Sequence[Optional[float]],
        threshold: float = 0.5,
        window_start: int = 0,
        max_trials: Optional[int] = None,
    ) -> None:
        if threshold < 0.0:
            raise ValueError("threshold must be non-negative")
        if window_start < 0:
            raise ValueError("window_start must be non-negative")
        if max_trials is not None and int(max_trials) <= 0:
            raise ValueError("max_trials must be positive")
        self.risk_by_index = risk_by_index if isinstance(risk_by_index, RiskSeries) else RiskSeries(risk_by_index)
        self.threshold = float(threshold)
        self.window_start = int(window_start)
        self.max_trials = int(max_trials) if max_trials is not None else None

    def _choose_candidate(self, *, bad_index: int) -> int:
        """
        Choose the next commit to test.

        Given a known-bad commit `bad_index`, find the closest earlier commit
        `candidate < bad_index` such that:

          sum(p_i for i in [candidate, bad_index)) >= threshold

        If no such candidate exists within the window, return `window_start`.
        """
        bad_index = int(bad_index)
        if bad_index <= self.window_start:
            return int(self.window_start)

        prefix = self.risk_by_index.prefix_sums
        needed = float(prefix[bad_index] - self.threshold)
        # Largest candidate in [window_start, bad_index) with prefix[candidate] <= needed.
        pos = bisect.bisect_right(prefix, needed, lo=self.window_start, hi=bad_index)
        candidate = int(pos - 1)
        if candidate < self.window_start:
            candidate = int(self.window_start)
        if candidate >= bad_index:
            candidate = int(bad_index - 1)
        return int(candidate)

    def find_good_index(
        self, *, start_index: int, culprit_index: int, start_time_utc: Optional[datetime] = None
    ) -> LookbackOutcome:
        _ = start_time_utc  # not used by this strategy

        start_index = int(start_index)
        culprit_index = int(culprit_index)

        if start_index <= self.window_start:
            # There is no commit < start_index within the simulation window.
            return LookbackOutcome(good_index=None, steps=0)

        # Defensive clamp: `risk_by_index` is aligned to commit indices.
        max_idx = len(self.risk_by_index) - 1
        bad = min(start_index, max_idx)

        steps = 0
        known_results: dict[int, bool] = {}
        while bad > self.window_start:
            if self.max_trials is not None and steps >= self.max_trials:
                return _forced_fallback_outcome(
                    steps=steps,
                    window_start=self.window_start,
                    culprit_index=culprit_index,
                    known_results=known_results,
                )

            candidate = self._choose_candidate(bad_index=bad)
            if candidate >= bad:
                candidate = bad - 1
            if candidate < self.window_start:
                candidate = self.window_start
            candidate = int(candidate)

            steps += 1
            known_results[candidate] = bool(candidate >= culprit_index)
            if candidate < culprit_index:
                logger.debug(
                    "RWLB-S found good_index=%d after steps=%d (start=%d culprit=%d threshold=%s)",
                    candidate,
                    steps,
                    start_index,
                    culprit_index,
                    self.threshold,
                )
                return LookbackOutcome(good_index=int(candidate), steps=steps, known_results=known_results)

            if candidate <= self.window_start:
                # No earlier in-window candidate can be tested.
                return LookbackOutcome(good_index=None, steps=steps, known_results=known_results)

            # Still failing; move the known-bad boundary backward and try again.
            bad = int(candidate)

        return LookbackOutcome(good_index=None, steps=steps, known_results=known_results)


class RiskWeightedLookbackLogSurvival:
    """
    Risk-Weighted Lookback (RWLB-LS).

    Same policy as RWLB-S, but uses a combined probability mass over a range:

      1 - ∏(1 - p_i)

    to compute the cumulative risk when scanning backwards from the current
    known-bad boundary.
    """

    name = "rwlb-ls"

    def __init__(
        self,
        *,
        risk_by_index: Sequence[Optional[float]],
        threshold: float = 0.5,
        window_start: int = 0,
        max_trials: Optional[int] = None,
    ) -> None:
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("threshold must be in [0,1]")
        if window_start < 0:
            raise ValueError("window_start must be non-negative")
        if max_trials is not None and int(max_trials) <= 0:
            raise ValueError("max_trials must be positive")
        self.risk_by_index = risk_by_index if isinstance(risk_by_index, RiskSeries) else RiskSeries(risk_by_index)
        self.threshold = float(threshold)
        self.window_start = int(window_start)
        self.max_trials = int(max_trials) if max_trials is not None else None

    def _range_combined_probability(self, *, start: int, end: int) -> float:
        # Inline a fast range query to avoid additional indirection in tight loops.
        log_survival = float(self.risk_by_index.prefix_log_survival[end] - self.risk_by_index.prefix_log_survival[start])
        if math.isinf(log_survival) and log_survival < 0:
            return 1.0
        survival = math.exp(log_survival)
        return float(1.0 - survival)

    def _choose_candidate(self, *, bad_index: int) -> int:
        """
        Choose the next commit to test by thresholding combined probability.

        Find the closest earlier commit `candidate < bad_index` such that the
        combined probability over [candidate, bad_index) is >= threshold.
        """
        bad_index = int(bad_index)
        if bad_index <= self.window_start:
            return int(self.window_start)

        # Threshold of 0 tests immediately adjacent history.
        if self.threshold <= 0.0:
            return int(max(self.window_start, bad_index - 1))

        # If even the full in-window interval doesn't reach the threshold, fall
        # back to testing the first commit in the window.
        if self._range_combined_probability(start=self.window_start, end=bad_index) < self.threshold:
            return int(self.window_start)

        lo = int(self.window_start)
        hi = int(bad_index - 1)
        best = int(self.window_start)

        # g(candidate) = combined_prob([candidate, bad_index)) is monotone
        # decreasing in candidate.
        while lo <= hi:
            mid = (lo + hi) // 2
            mass = self._range_combined_probability(start=mid, end=bad_index)
            if mass >= self.threshold:
                best = mid
                lo = mid + 1  # move closer to bad_index
            else:
                hi = mid - 1

        if best >= bad_index:
            best = int(bad_index - 1)
        if best < self.window_start:
            best = int(self.window_start)
        return int(best)

    def find_good_index(
        self, *, start_index: int, culprit_index: int, start_time_utc: Optional[datetime] = None
    ) -> LookbackOutcome:
        _ = start_time_utc  # not used by this strategy

        start_index = int(start_index)
        culprit_index = int(culprit_index)

        if start_index <= self.window_start:
            # There is no commit < start_index within the simulation window.
            return LookbackOutcome(good_index=None, steps=0)

        # Defensive clamp: `risk_by_index` is aligned to commit indices.
        max_idx = len(self.risk_by_index) - 1
        bad = min(start_index, max_idx)

        steps = 0
        known_results: dict[int, bool] = {}
        while bad > self.window_start:
            if self.max_trials is not None and steps >= self.max_trials:
                return _forced_fallback_outcome(
                    steps=steps,
                    window_start=self.window_start,
                    culprit_index=culprit_index,
                    known_results=known_results,
                )

            candidate = self._choose_candidate(bad_index=bad)
            if candidate >= bad:
                candidate = bad - 1
            if candidate < self.window_start:
                candidate = self.window_start
            candidate = int(candidate)

            steps += 1
            known_results[candidate] = bool(candidate >= culprit_index)
            if candidate < culprit_index:
                logger.debug(
                    "RWLB-LS found good_index=%d after steps=%d (start=%d culprit=%d threshold=%s)",
                    candidate,
                    steps,
                    start_index,
                    culprit_index,
                    self.threshold,
                )
                return LookbackOutcome(good_index=int(candidate), steps=steps, known_results=known_results)

            if candidate <= self.window_start:
                return LookbackOutcome(good_index=None, steps=steps, known_results=known_results)

            bad = int(candidate)

        return LookbackOutcome(good_index=None, steps=steps, known_results=known_results)


class TimeWindowLookbackAdaptiveDecrease:
    """
    Time-window lookback with adaptive decrease (TWLB-AD).

    Like Time-Window Lookback (TWLB), but reduces the time window after each
    failed test.

    Policy:
      - Iteration 1: jump back by `hours` in time and test that commit.
      - Iteration k: jump back by `hours * (alpha ** (k-1))` in time and test.
      - On failure, the tested commit becomes the new known-bad boundary.
      - Return the first tested commit that passes.

    `steps` counts only tested commits (the known-bad `start_index` is not
    re-tested).
    """

    name = "time_window-ad"

    def __init__(
        self,
        *,
        sorted_times_utc: Sequence[datetime],
        sorted_time_indices: Sequence[int],
        hours: float = 24.0,
        alpha: float = 0.5,
        window_start: int = 0,
        max_trials: Optional[int] = None,
    ) -> None:
        if len(sorted_times_utc) != len(sorted_time_indices):
            raise ValueError("sorted_times_utc and sorted_time_indices must have the same length")
        if not sorted_times_utc:
            raise ValueError("sorted_times_utc must be non-empty")
        if hours <= 0.0:
            raise ValueError("hours must be positive")
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("alpha must be in [0,1]")
        if window_start < 0:
            raise ValueError("window_start must be non-negative")
        if max_trials is not None and int(max_trials) <= 0:
            raise ValueError("max_trials must be positive")

        self.sorted_times_utc = list(sorted_times_utc)
        self.sorted_time_indices = list(sorted_time_indices)
        self.hours = float(hours)
        self.alpha = float(alpha)
        self.window_start = int(window_start)
        self.max_trials = int(max_trials) if max_trials is not None else None

        # Fast index->time mapping for repeated lookups.
        time_by_index: List[Optional[datetime]] = [None] * len(self.sorted_time_indices)
        for t, idx in zip(self.sorted_times_utc, self.sorted_time_indices):
            if idx < 0 or idx >= len(time_by_index):
                raise ValueError(f"Invalid commit index {idx} in sorted_time_indices")
            time_by_index[idx] = t
        if any(t is None for t in time_by_index):
            raise ValueError("sorted_time_indices must contain every commit index exactly once")
        self.time_by_index = time_by_index

    def _last_commit_at_or_before(self, t: datetime) -> Optional[int]:
        pos = bisect.bisect_right(self.sorted_times_utc, t) - 1
        return self.sorted_time_indices[pos] if pos >= 0 else None

    def find_good_index(
        self, *, start_index: int, culprit_index: int, start_time_utc: Optional[datetime] = None
    ) -> LookbackOutcome:
        _ = start_time_utc  # not used by this strategy

        start_index = int(start_index)
        culprit_index = int(culprit_index)

        if start_index <= self.window_start:
            # There is no commit < start_index within the simulation window.
            return LookbackOutcome(good_index=None, steps=0)

        # Defensive clamp.
        cur_idx = min(start_index, len(self.time_by_index) - 1)

        steps = 0
        known_results: dict[int, bool] = {}
        while cur_idx > self.window_start:
            if self.max_trials is not None and steps >= self.max_trials:
                return _forced_fallback_outcome(
                    steps=steps,
                    window_start=self.window_start,
                    culprit_index=culprit_index,
                    known_results=known_results,
                )

            cur_time = self.time_by_index[cur_idx]
            if cur_time is None:
                raise RuntimeError(f"Missing commit time for index {cur_idx}")

            # First test uses `hours`; subsequent tests shrink by `alpha`.
            cur_hours = float(self.hours) * float(self.alpha ** steps)
            target = cur_time - timedelta(hours=cur_hours)

            candidate = self._last_commit_at_or_before(target)
            if candidate is None:
                break
            if candidate >= cur_idx:
                candidate = cur_idx - 1
            if candidate < self.window_start:
                candidate = self.window_start
            candidate = int(candidate)

            steps += 1
            known_results[candidate] = bool(candidate >= culprit_index)
            if candidate < culprit_index:
                logger.debug(
                    "TWLB-AD found good_index=%d after steps=%d (start=%d culprit=%d hours=%s alpha=%s)",
                    candidate,
                    steps,
                    start_index,
                    culprit_index,
                    self.hours,
                    self.alpha,
                )
                return LookbackOutcome(good_index=int(candidate), steps=steps, known_results=known_results)

            if candidate == self.window_start:
                return LookbackOutcome(good_index=None, steps=steps, known_results=known_results)

            cur_idx = int(candidate)

        # If we ran out of history/time-window targets, fall back to testing the first commit in the window.
        steps += 1
        known_results[int(self.window_start)] = bool(int(self.window_start) >= culprit_index)
        good = self.window_start if self.window_start < culprit_index else None
        return LookbackOutcome(good_index=good, steps=steps, known_results=known_results)


class TimeWindowLookbackAdaptiveIncrease:
    """
    Time-window lookback with adaptive increase (TWLB-AI).

    Like Time-Window Lookback (TWLB), but increases the time window after each
    failed test.

    Policy:
      - Iteration 1: jump back by `hours` in time and test that commit.
      - Iteration k: jump back by `hours * (alpha ** (k-1))` in time and test,
        with `alpha > 1`.
      - On failure, the tested commit becomes the new known-bad boundary.
      - Return the first tested commit that passes.

    `steps` counts only tested commits (the known-bad `start_index` is not
    re-tested).
    """

    name = "time_window-ai"

    def __init__(
        self,
        *,
        sorted_times_utc: Sequence[datetime],
        sorted_time_indices: Sequence[int],
        hours: float = 24.0,
        alpha: float = 2.0,
        window_start: int = 0,
        max_trials: Optional[int] = None,
    ) -> None:
        if len(sorted_times_utc) != len(sorted_time_indices):
            raise ValueError("sorted_times_utc and sorted_time_indices must have the same length")
        if not sorted_times_utc:
            raise ValueError("sorted_times_utc must be non-empty")
        if hours <= 0.0:
            raise ValueError("hours must be positive")
        if alpha <= 1.0:
            raise ValueError("alpha must be > 1")
        if window_start < 0:
            raise ValueError("window_start must be non-negative")
        if max_trials is not None and int(max_trials) <= 0:
            raise ValueError("max_trials must be positive")

        self.sorted_times_utc = list(sorted_times_utc)
        self.sorted_time_indices = list(sorted_time_indices)
        self.hours = float(hours)
        self.alpha = float(alpha)
        self.window_start = int(window_start)
        self.max_trials = int(max_trials) if max_trials is not None else None

        # Fast index->time mapping for repeated lookups.
        time_by_index: List[Optional[datetime]] = [None] * len(self.sorted_time_indices)
        for t, idx in zip(self.sorted_times_utc, self.sorted_time_indices):
            if idx < 0 or idx >= len(time_by_index):
                raise ValueError(f"Invalid commit index {idx} in sorted_time_indices")
            time_by_index[idx] = t
        if any(t is None for t in time_by_index):
            raise ValueError("sorted_time_indices must contain every commit index exactly once")
        self.time_by_index = time_by_index

    def _last_commit_at_or_before(self, t: datetime) -> Optional[int]:
        pos = bisect.bisect_right(self.sorted_times_utc, t) - 1
        return self.sorted_time_indices[pos] if pos >= 0 else None

    def find_good_index(
        self, *, start_index: int, culprit_index: int, start_time_utc: Optional[datetime] = None
    ) -> LookbackOutcome:
        _ = start_time_utc  # not used by this strategy

        start_index = int(start_index)
        culprit_index = int(culprit_index)

        if start_index <= self.window_start:
            return LookbackOutcome(good_index=None, steps=0)

        cur_idx = min(start_index, len(self.time_by_index) - 1)

        steps = 0
        known_results: dict[int, bool] = {}
        while cur_idx > self.window_start:
            if self.max_trials is not None and steps >= self.max_trials:
                return _forced_fallback_outcome(
                    steps=steps,
                    window_start=self.window_start,
                    culprit_index=culprit_index,
                    known_results=known_results,
                )

            cur_time = self.time_by_index[cur_idx]
            if cur_time is None:
                raise RuntimeError(f"Missing commit time for index {cur_idx}")

            cur_hours = float(self.hours) * float(self.alpha ** steps)
            target = cur_time - timedelta(hours=cur_hours)

            candidate = self._last_commit_at_or_before(target)
            if candidate is None:
                break
            if candidate >= cur_idx:
                candidate = cur_idx - 1
            if candidate < self.window_start:
                candidate = self.window_start
            candidate = int(candidate)

            steps += 1
            known_results[candidate] = bool(candidate >= culprit_index)
            if candidate < culprit_index:
                logger.debug(
                    "TWLB-AI found good_index=%d after steps=%d (start=%d culprit=%d hours=%s alpha=%s)",
                    candidate,
                    steps,
                    start_index,
                    culprit_index,
                    self.hours,
                    self.alpha,
                )
                return LookbackOutcome(good_index=int(candidate), steps=steps, known_results=known_results)

            if candidate == self.window_start:
                return LookbackOutcome(good_index=None, steps=steps, known_results=known_results)

            cur_idx = int(candidate)

        # If we ran out of history/time-window targets, fall back to testing the first commit in the window.
        steps += 1
        known_results[int(self.window_start)] = bool(int(self.window_start) >= culprit_index)
        good = self.window_start if self.window_start < culprit_index else None
        return LookbackOutcome(good_index=good, steps=steps, known_results=known_results)


class TimeWindowLookback:
    """
    Time-Window Lookback (TWLB).

    Like fixed-stride lookback, but uses a fixed time window instead of a fixed
    number of commits:

      - Start from a known-bad commit index (`start_index`).
      - Repeatedly jump back by `hours` in time, find the most recent commit at
        or before that time, and test it.
      - Return the first tested commit that is clean.

    `steps` counts only tested commits (the known-bad `start_index` is not
    re-tested).
    """

    name = "time_window"

    def __init__(
        self,
        *,
        sorted_times_utc: Sequence[datetime],
        sorted_time_indices: Sequence[int],
        hours: float = 24.0,
        window_start: int = 0,
        max_trials: Optional[int] = None,
    ) -> None:
        if len(sorted_times_utc) != len(sorted_time_indices):
            raise ValueError("sorted_times_utc and sorted_time_indices must have the same length")
        if not sorted_times_utc:
            raise ValueError("sorted_times_utc must be non-empty")
        if hours <= 0.0:
            raise ValueError("hours must be positive")
        if window_start < 0:
            raise ValueError("window_start must be non-negative")
        if max_trials is not None and int(max_trials) <= 0:
            raise ValueError("max_trials must be positive")

        self.sorted_times_utc = list(sorted_times_utc)
        self.sorted_time_indices = list(sorted_time_indices)
        self.hours = float(hours)
        self.window_start = int(window_start)
        self.max_trials = int(max_trials) if max_trials is not None else None

        # Fast index->time mapping for repeated lookups.
        time_by_index: List[Optional[datetime]] = [None] * len(self.sorted_time_indices)
        for t, idx in zip(self.sorted_times_utc, self.sorted_time_indices):
            if idx < 0 or idx >= len(time_by_index):
                raise ValueError(f"Invalid commit index {idx} in sorted_time_indices")
            time_by_index[idx] = t
        if any(t is None for t in time_by_index):
            raise ValueError("sorted_time_indices must contain every commit index exactly once")
        self.time_by_index = time_by_index

    def _last_commit_at_or_before(self, t: datetime) -> Optional[int]:
        import bisect

        pos = bisect.bisect_right(self.sorted_times_utc, t) - 1
        return self.sorted_time_indices[pos] if pos >= 0 else None

    def find_good_index(
        self, *, start_index: int, culprit_index: int, start_time_utc: Optional[datetime] = None
    ) -> LookbackOutcome:
        _ = start_time_utc  # not used by this strategy

        start_index = int(start_index)
        culprit_index = int(culprit_index)

        if start_index <= self.window_start:
            # There is no commit < start_index within the simulation window.
            return LookbackOutcome(good_index=None, steps=0)

        # Defensive clamp.
        cur_idx = min(start_index, len(self.time_by_index) - 1)

        steps = 0
        known_results: dict[int, bool] = {}
        while cur_idx > self.window_start:
            if self.max_trials is not None and steps >= self.max_trials:
                return _forced_fallback_outcome(
                    steps=steps,
                    window_start=self.window_start,
                    culprit_index=culprit_index,
                    known_results=known_results,
                )

            cur_time = self.time_by_index[cur_idx]
            if cur_time is None:
                raise RuntimeError(f"Missing commit time for index {cur_idx}")

            target = cur_time - timedelta(hours=self.hours)
            candidate = self._last_commit_at_or_before(target)
            if candidate is None:
                break
            if candidate >= cur_idx:
                candidate = cur_idx - 1
            if candidate < self.window_start:
                candidate = self.window_start
            candidate = int(candidate)

            steps += 1
            known_results[candidate] = bool(candidate >= culprit_index)
            if candidate < culprit_index:
                logger.debug(
                    "TWLB found good_index=%d after steps=%d (start=%d culprit=%d hours=%s)",
                    candidate,
                    steps,
                    start_index,
                    culprit_index,
                    self.hours,
                )
                return LookbackOutcome(good_index=candidate, steps=steps, known_results=known_results)

            if candidate == self.window_start:
                return LookbackOutcome(good_index=None, steps=steps, known_results=known_results)

            cur_idx = candidate

        # If we ran out of history/time-window targets, fall back to testing the first commit in the window.
        steps += 1
        known_results[int(self.window_start)] = bool(int(self.window_start) >= culprit_index)
        good = self.window_start if self.window_start < culprit_index else None
        return LookbackOutcome(good_index=good, steps=steps, known_results=known_results)


class FixedStrideLookbackForcedFallback(FixedStrideLookback):
    """Fixed stride lookback with forced fallback after `max_trials` tests."""

    name = "fixed_stride-ff"


class FixedStrideLookbackAdaptiveDecreaseForcedFallback(FixedStrideLookbackAdaptiveDecrease):
    """Adaptive fixed stride lookback with forced fallback after `max_trials` tests."""

    name = "fixed_stride-ad-ff"


class FixedStrideLookbackAdaptiveIncreaseForcedFallback(FixedStrideLookbackAdaptiveIncrease):
    """Adaptive-increase fixed stride lookback with forced fallback after `max_trials` tests."""

    name = "fixed_stride-ai-ff"


class RiskAwareTriggerLookbackForcedFallback(RiskAwareTriggerLookback):
    """Risk-aware trigger lookback with forced fallback after `max_trials` tests."""

    name = "risk_aware_trigger-ff"


class RiskWeightedLookbackSumForcedFallback(RiskWeightedLookbackSum):
    """Risk-weighted (sum) lookback with forced fallback after `max_trials` tests."""

    name = "rwlb-s-ff"


class RiskWeightedLookbackLogSurvivalForcedFallback(RiskWeightedLookbackLogSurvival):
    """Risk-weighted (log-survival) lookback with forced fallback after `max_trials` tests."""

    name = "rwlb-ls-ff"


class TimeWindowLookbackForcedFallback(TimeWindowLookback):
    """Time-window lookback with forced fallback after `max_trials` tests."""

    name = "time_window-ff"


class TimeWindowLookbackAdaptiveDecreaseForcedFallback(TimeWindowLookbackAdaptiveDecrease):
    """Adaptive time-window lookback with forced fallback after `max_trials` tests."""

    name = "time_window-ad-ff"


class TimeWindowLookbackAdaptiveIncreaseForcedFallback(TimeWindowLookbackAdaptiveIncrease):
    """Adaptive-increase time-window lookback with forced fallback after `max_trials` tests."""

    name = "time_window-ai-ff"


LOOKBACK_STRATEGIES = {
    NoLookback.name: NoLookback,
    FixedStrideLookback.name: FixedStrideLookback,
    FixedStrideLookbackForcedFallback.name: FixedStrideLookbackForcedFallback,
    FixedStrideLookbackAdaptiveDecrease.name: FixedStrideLookbackAdaptiveDecrease,
    FixedStrideLookbackAdaptiveDecreaseForcedFallback.name: FixedStrideLookbackAdaptiveDecreaseForcedFallback,
    FixedStrideLookbackAdaptiveIncrease.name: FixedStrideLookbackAdaptiveIncrease,
    FixedStrideLookbackAdaptiveIncreaseForcedFallback.name: FixedStrideLookbackAdaptiveIncreaseForcedFallback,
    NightlyBuildLookback.name: NightlyBuildLookback,
    RiskAwareTriggerLookback.name: RiskAwareTriggerLookback,
    RiskAwareTriggerLookbackForcedFallback.name: RiskAwareTriggerLookbackForcedFallback,
    RiskWeightedLookbackSum.name: RiskWeightedLookbackSum,
    RiskWeightedLookbackSumForcedFallback.name: RiskWeightedLookbackSumForcedFallback,
    RiskWeightedLookbackLogSurvival.name: RiskWeightedLookbackLogSurvival,
    RiskWeightedLookbackLogSurvivalForcedFallback.name: RiskWeightedLookbackLogSurvivalForcedFallback,
    TimeWindowLookbackAdaptiveDecrease.name: TimeWindowLookbackAdaptiveDecrease,
    TimeWindowLookbackAdaptiveDecreaseForcedFallback.name: TimeWindowLookbackAdaptiveDecreaseForcedFallback,
    TimeWindowLookbackAdaptiveIncrease.name: TimeWindowLookbackAdaptiveIncrease,
    TimeWindowLookbackAdaptiveIncreaseForcedFallback.name: TimeWindowLookbackAdaptiveIncreaseForcedFallback,
    TimeWindowLookback.name: TimeWindowLookback,
    TimeWindowLookbackForcedFallback.name: TimeWindowLookbackForcedFallback,
}
