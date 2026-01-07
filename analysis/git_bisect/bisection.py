from __future__ import annotations

"""
Bisection strategy interfaces and baseline implementations.

These classes model a simplified `git bisect` procedure over a linear history.
Strategies take a known-good index, a known-bad index, and a single culprit
index (the first commit at/after which tests fail), then simulate how many test
executions are required to identify the culprit.
"""

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence


@dataclass(frozen=True)
class BisectionOutcome:
    tests: int
    found_index: Optional[int]


class BisectionStrategy(Protocol):
    name: str

    def run(
        self,
        *,
        good_index: int,
        bad_index: int,
        culprit_index: int,
        risk_by_index: Optional[Sequence[Optional[float]]] = None,
    ) -> BisectionOutcome:
        """Run a bisection procedure and return test count plus found culprit."""
        ...


class GitBisectBaseline:
    """
    Baseline bisection simulation: find a single culprit via binary search
    between (good,bad].

    Test model:
      - A test on commit i "fails" if i >= culprit_index.
      - Otherwise it "passes".
    """

    name = "git_bisect"

    def run(
        self,
        *,
        good_index: int,
        bad_index: int,
        culprit_index: int,
        risk_by_index: Optional[Sequence[Optional[float]]] = None,
    ) -> BisectionOutcome:
        """Simulate standard git bisect to find a single culprit and count tests."""
        _ = risk_by_index  # reserved for future strategies

        if good_index >= bad_index:
            return BisectionOutcome(tests=0, found_index=None)

        culprit_index = int(culprit_index)
        if culprit_index <= good_index or culprit_index > bad_index:
            return BisectionOutcome(tests=0, found_index=None)

        low = int(good_index)
        high = int(bad_index)
        tests = 0

        while high - low > 1:
            mid = (low + high) // 2
            tests += 1
            if mid >= culprit_index:
                high = mid
            else:
                low = mid

        return BisectionOutcome(tests=tests, found_index=high)


BISECTION_STRATEGIES = {
    GitBisectBaseline.name: GitBisectBaseline,
}
