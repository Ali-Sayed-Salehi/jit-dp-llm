from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence


@dataclass(frozen=True)
class LookbackOutcome:
    good_index: Optional[int]
    steps: int


class LookbackStrategy(Protocol):
    name: str

    def find_good_index(
        self, *, start_index: int, regressor_indices: Sequence[int]
    ) -> LookbackOutcome:
        """Return the chosen good commit index for a bug, plus any bookkeeping."""
        ...


class FixedStrideLookback:
    """
    Baseline lookback:
      - Starting from `start_index`, step backwards by `stride` commits.
      - Return the first commit index that contains none of the regressors.

    In a linear history, a commit at index i "contains" a regressor at index r
    if i >= r. Therefore, "contains none" is equivalent to i < min(regressors).
    """

    name = "fixed_stride"

    def __init__(self, stride: int = 20) -> None:
        """Create a fixed-stride lookback strategy."""
        if stride <= 0:
            raise ValueError("stride must be positive")
        self.stride = stride

    def find_good_index(
        self, *, start_index: int, regressor_indices: Sequence[int]
    ) -> LookbackOutcome:
        """Find a "good" commit by stepping back in fixed increments."""
        if not regressor_indices:
            return LookbackOutcome(good_index=None, steps=0)

        min_regressor = min(int(x) for x in regressor_indices)
        idx = int(start_index)
        steps = 0
        while idx >= 0:
            if idx < min_regressor:
                return LookbackOutcome(good_index=idx, steps=steps)
            idx -= self.stride
            steps += 1

        return LookbackOutcome(good_index=None, steps=steps)


LOOKBACK_STRATEGIES = {
    FixedStrideLookback.name: FixedStrideLookback,
}
