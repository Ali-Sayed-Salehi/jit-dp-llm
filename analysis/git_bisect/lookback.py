from __future__ import annotations

"""
Lookback strategies for selecting a known-good commit.

Given a regression observation at some `start_index` and the regressor commit(s),
lookback strategies model how a developer might search backwards in history to
find a commit that does not yet contain the regression.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LookbackOutcome:
    """Result of selecting a known-good commit for a single bug/regression."""
    good_index: Optional[int]
    steps: int


class LookbackStrategy(Protocol):
    name: str

    def find_good_index(
        self, *, start_index: int, culprit_index: int, start_time_utc: Optional[datetime] = None
    ) -> LookbackOutcome:
        """Return the chosen good commit index for a bug, plus any bookkeeping."""
        ...


class FixedStrideLookback:
    """
    Baseline lookback:
      - Starting from `start_index`, step backwards by `stride` commits.
      - Return the first commit index that does not contain the culprit.

    In a linear history, a commit at index i "contains" the culprit at index c
    if i >= c. Therefore, "does not contain" is equivalent to i < c.
    """

    name = "fixed_stride"

    def __init__(self, stride: int = 20) -> None:
        """Create a fixed-stride lookback strategy."""
        if stride <= 0:
            raise ValueError("stride must be positive")
        self.stride = stride

    def find_good_index(
        self, *, start_index: int, culprit_index: int, start_time_utc: Optional[datetime] = None
    ) -> LookbackOutcome:
        """Find a "good" commit by stepping back in fixed increments."""
        _ = start_time_utc  # not used by this strategy

        culprit_index = int(culprit_index)
        idx = int(start_index)
        steps = 0
        while idx >= 0:
            if idx < culprit_index:
                logger.debug(
                    "Lookback found good_index=%d after steps=%d (start=%d culprit=%d stride=%d)",
                    idx,
                    steps,
                    start_index,
                    culprit_index,
                    self.stride,
                )
                return LookbackOutcome(good_index=idx, steps=steps)
            idx -= self.stride
            steps += 1

        logger.debug(
            "Lookback failed to find good_index (start=%d culprit=%d stride=%d steps=%d)",
            start_index,
            culprit_index,
            self.stride,
            steps,
        )
        return LookbackOutcome(good_index=None, steps=steps)


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
    ) -> None:
        if len(sorted_times_utc) != len(sorted_time_indices):
            raise ValueError("sorted_times_utc and sorted_time_indices must have the same length")
        if not sorted_times_utc:
            raise ValueError("sorted_times_utc must be non-empty")
        self.sorted_times_utc = list(sorted_times_utc)
        self.sorted_time_indices = list(sorted_time_indices)

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

        midnight_today = datetime(
            start_time_utc.year,
            start_time_utc.month,
            start_time_utc.day,
            tzinfo=timezone.utc,
        )

        steps = 0
        cutoff = midnight_today
        while True:
            nightly_idx = self._last_commit_strictly_before(cutoff)
            if nightly_idx is None:
                raise RuntimeError(
                    "Nightly lookback failed: no commits before "
                    f"cutoff={cutoff.isoformat()} (start_index={start_index}, culprit_index={culprit_index}, steps={steps})"
                )

            # If the nightly build resolves to the already-known "bad" observation commit
            # (possible when the bug is observed soon after midnight with no intervening commits),
            # skip it without counting a new test.
            if nightly_idx >= start_index:
                cutoff = cutoff - timedelta(days=1)
                continue

            steps += 1
            if nightly_idx < culprit_index:
                logger.debug(
                    "Nightly lookback found good_index=%d after steps=%d (start=%d culprit=%d)",
                    nightly_idx,
                    steps,
                    start_index,
                    culprit_index,
                )
                return LookbackOutcome(good_index=nightly_idx, steps=steps)

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
      - If no triggers are found, the strategy falls back to testing commit 0.
    """

    name = "risk_aware_trigger"

    def __init__(self, *, risk_by_index: Sequence[Optional[float]], threshold: float = 0.5) -> None:
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("threshold must be in [0,1]")
        self.risk_by_index = risk_by_index
        self.threshold = float(threshold)

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

        while True:
            trigger_idx: Optional[int] = None
            i = search_idx
            # We need i-1 to exist, so only consider triggers at i >= 1.
            while i >= 1:
                if self._risk(i) > self.threshold:
                    trigger_idx = i
                    break
                i -= 1

            if trigger_idx is None:
                # No triggers found: fall back to testing the first commit.
                steps += 1
                good = 0 if 0 < culprit_index else None
                return LookbackOutcome(good_index=good, steps=steps)

            candidate = trigger_idx - 1
            steps += 1
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
                return LookbackOutcome(good_index=candidate, steps=steps)

            # Candidate still contains the culprit; continue scanning earlier history.
            search_idx = candidate


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
    ) -> None:
        if len(sorted_times_utc) != len(sorted_time_indices):
            raise ValueError("sorted_times_utc and sorted_time_indices must have the same length")
        if not sorted_times_utc:
            raise ValueError("sorted_times_utc must be non-empty")
        if hours <= 0.0:
            raise ValueError("hours must be positive")

        self.sorted_times_utc = list(sorted_times_utc)
        self.sorted_time_indices = list(sorted_time_indices)
        self.hours = float(hours)

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

        # Defensive clamp.
        cur_idx = min(start_index, len(self.time_by_index) - 1)

        steps = 0
        while cur_idx > 0:
            cur_time = self.time_by_index[cur_idx]
            if cur_time is None:
                raise RuntimeError(f"Missing commit time for index {cur_idx}")

            target = cur_time - timedelta(hours=self.hours)
            candidate = self._last_commit_at_or_before(target)
            if candidate is None:
                break
            if candidate >= cur_idx:
                candidate = cur_idx - 1

            steps += 1
            if candidate < culprit_index:
                logger.debug(
                    "TWLB found good_index=%d after steps=%d (start=%d culprit=%d hours=%s)",
                    candidate,
                    steps,
                    start_index,
                    culprit_index,
                    self.hours,
                )
                return LookbackOutcome(good_index=candidate, steps=steps)

            cur_idx = candidate

        # If we ran out of history/time-window targets, fall back to testing the first commit.
        steps += 1
        good = 0 if 0 < culprit_index else None
        return LookbackOutcome(good_index=good, steps=steps)


LOOKBACK_STRATEGIES = {
    FixedStrideLookback.name: FixedStrideLookback,
    NightlyBuildLookback.name: NightlyBuildLookback,
    RiskAwareTriggerLookback.name: RiskAwareTriggerLookback,
    TimeWindowLookback.name: TimeWindowLookback,
}
