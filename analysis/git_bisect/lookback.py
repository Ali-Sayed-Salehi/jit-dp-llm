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
            if nightly_idx > start_index:
                raise RuntimeError(
                    "Nightly lookback invariant violated: computed nightly build index is after start_index "
                    f"(nightly_idx={nightly_idx}, start_index={start_index}, cutoff={cutoff.isoformat()})"
                )
            if nightly_idx == start_index:
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


LOOKBACK_STRATEGIES = {
    FixedStrideLookback.name: FixedStrideLookback,
    NightlyBuildLookback.name: NightlyBuildLookback,
}
