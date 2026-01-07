from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence


@dataclass(frozen=True)
class BisectionOutcome:
    tests: int
    found_indices: List[int]


class BisectionStrategy(Protocol):
    name: str

    def run(
        self,
        *,
        good_index: int,
        bad_index: int,
        suspect_indices: Sequence[int],
        risk_by_index: Optional[Sequence[Optional[float]]] = None,
    ) -> BisectionOutcome:
        """Run a bisection procedure and return test count plus found culprits."""
        ...


class GitBisectBaseline:
    """
    Baseline bisection simulation: repeatedly find the earliest remaining culprit
    via binary search between (good,bad], then mark it clean and continue.

    Test model:
      - A test on commit i "fails" if i >= min(remaining_culprits).
      - Otherwise it "passes".
    """

    name = "git_bisect"

    def run(
        self,
        *,
        good_index: int,
        bad_index: int,
        suspect_indices: Sequence[int],
        risk_by_index: Optional[Sequence[Optional[float]]] = None,
    ) -> BisectionOutcome:
        """Simulate standard git bisect to find culprits and count tests."""
        _ = risk_by_index  # reserved for future strategies

        if good_index >= bad_index:
            return BisectionOutcome(tests=0, found_indices=[])

        remaining = sorted(set(int(x) for x in suspect_indices))
        found: List[int] = []
        tests = 0

        while remaining:
            while remaining and remaining[0] <= good_index:
                remaining.pop(0)
            if not remaining:
                break
            if remaining[0] > bad_index:
                break

            target = remaining[0]
            low = int(good_index)
            high = int(bad_index)
            if not (low < target <= high):
                break

            while high - low > 1:
                mid = (low + high) // 2
                tests += 1
                if mid >= target:
                    high = mid
                else:
                    low = mid

            found.append(high)
            good_index = high  # mark culprit clean for next iteration
            remaining = [x for x in remaining if x != high]

        return BisectionOutcome(tests=tests, found_indices=found)


BISECTION_STRATEGIES = {
    GitBisectBaseline.name: GitBisectBaseline,
}
