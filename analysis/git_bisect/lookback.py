from __future__ import annotations

"""
Lookback strategies for selecting a known-good commit.

Given a regression observation at some `start_index` and the regressor commit(s),
lookback strategies model how a developer might search backwards in history to
find a commit that does not yet contain the regression.
"""

import logging
from dataclasses import dataclass
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
        self, *, start_index: int, culprit_index: int
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
        self, *, start_index: int, culprit_index: int
    ) -> LookbackOutcome:
        """Find a "good" commit by stepping back in fixed increments."""
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


LOOKBACK_STRATEGIES = {
    FixedStrideLookback.name: FixedStrideLookback,
}
