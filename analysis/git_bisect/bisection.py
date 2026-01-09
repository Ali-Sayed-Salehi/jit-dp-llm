from __future__ import annotations

"""
Bisection strategy interfaces and baseline implementations.

These classes model a simplified `git bisect` procedure over a linear history.
Strategies take a known-good index, a known-bad index, and a single culprit
index (the first commit at/after which tests fail), then simulate how many test
executions are required to identify the culprit.
"""

import logging
from dataclasses import dataclass
import math
from typing import Optional, Protocol, Sequence, overload

import bisect

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BisectionOutcome:
    """Result of a bisection run: tests executed and the located culprit index (if any)."""
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

class RiskSeries(Sequence[Optional[float]]):
    """
    Sequence wrapper that also provides fast range-risk queries.

    Supports two notions of "cumulative risk" over a time-ordered contiguous range:
      - Sum mass: sum(p_i)
      - Noisy-OR mass: 1 - ∏(1 - p_i)

    `None` values are treated as 0.0 risk.

    Fields:
      - `_values`: Per-commit risk probabilities aligned by commit index (original input as a list).
      - `_prefix_sums`: Prefix sums of probabilities where `_prefix_sums[k] == sum(_values[0:k])`
        (treating None as 0.0). Length is `len(_values) + 1` with `_prefix_sums[0] == 0.0`.
      - `_prefix_log_survival`: Prefix sums of log survival where
        `_prefix_log_survival[k] == sum(log(1 - p_i) for i in [0,k))`. Length is `len(_values) + 1`
        with `_prefix_log_survival[0] == 0.0`. This supports computing `1 - ∏(1-p)` over a range
        via subtraction + `exp`.
    """

    def __init__(self, values: Sequence[Optional[float]]) -> None:
        self._values = list(values)
        prefix_sum: list[float] = [0.0]
        prefix_log_survival: list[float] = [0.0]
        running_sum = 0.0
        running_log_survival = 0.0
        for v in self._values:
            if v is None:
                v = 0.0
            p = float(v)
            if p < 0.0 or p > 1.0:
                raise ValueError(f"Risk probabilities must be in [0,1], got {p}")

            running_sum += p
            prefix_sum.append(running_sum)

            # log(∏(1-p)) = Σ log(1-p); use log1p for stability.
            if p >= 1.0:
                running_log_survival = float("-inf")
            elif not math.isinf(running_log_survival):
                running_log_survival += math.log1p(-p)
            prefix_log_survival.append(running_log_survival)

        self._prefix_sums = prefix_sum
        self._prefix_log_survival = prefix_log_survival

    def __len__(self) -> int:  # pragma: no cover
        return len(self._values)

    @overload
    def __getitem__(self, index: int) -> Optional[float]: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Optional[float]]: ...

    def __getitem__(self, index: int | slice) -> Optional[float] | Sequence[Optional[float]]:  # pragma: no cover
        return self._values[index]

    @property
    def prefix_sums(self) -> Sequence[float]:
        """Prefix sums for sum-mass queries; see `_prefix_sums` in the class docstring."""
        return self._prefix_sums

    @property
    def prefix_log_survival(self) -> Sequence[float]:
        """Prefix sums for log-survival queries; see `_prefix_log_survival` in the class docstring."""
        return self._prefix_log_survival

    def range_sum(self, start: int, end: int) -> float:
        """Return sum of risks in the half-open interval [start, end)."""
        if start < 0 or end < 0 or start > end or end > len(self._values):
            raise ValueError(f"Invalid range [{start},{end}) for length={len(self._values)}")
        return float(self._prefix_sums[end] - self._prefix_sums[start])

    def range_combined_probability(self, start: int, end: int) -> float:
        """
        Return 1 - ∏(1-p) over the half-open interval [start, end).

        This is the combined probability of at least one "positive" event in the
        interval under an independence assumption, computed via log-survival
        prefix sums for numerical stability.
        """
        if start < 0 or end < 0 or start > end or end > len(self._values):
            raise ValueError(f"Invalid range [{start},{end}) for length={len(self._values)}")
        log_survival = float(self._prefix_log_survival[end] - self._prefix_log_survival[start])
        if math.isinf(log_survival) and log_survival < 0:
            return 1.0
        survival = math.exp(log_survival)
        return float(1.0 - survival)


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
            logger.debug("Bisect skipped: good_index=%d >= bad_index=%d", good_index, bad_index)
            return BisectionOutcome(tests=0, found_index=None)

        culprit_index = int(culprit_index)
        if culprit_index <= good_index or culprit_index > bad_index:
            logger.debug(
                "Bisect culprit out of range: good=%d bad=%d culprit=%d",
                good_index,
                bad_index,
                culprit_index,
            )
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

        logger.debug(
            "Bisect located culprit=%d in tests=%d (good=%d bad=%d)",
            high,
            tests,
            good_index,
            bad_index,
        )
        return BisectionOutcome(tests=tests, found_index=high)


class RiskWeightedAdaptiveBisectionSum:
    """
    Risk-Weighted Adaptive Bisection (RWAB-SUM).

    Like standard git bisect on a linear history, but chooses each probe index
    by approximately balancing the cumulative predicted risk mass in the two
    contiguous halves of the current search interval.

    For a current interval (low, high], RWAB chooses a mid in (low, high) such
    that:
      Σ p_i over (low, mid] ~= Σ p_i over (mid, high]
    preserving time order by only splitting into two contiguous sub-batches.
    """

    name = "rwab-s"

    def _choose_mid(
        self,
        *,
        low: int,
        high: int,
        risk_by_index: Optional[Sequence[Optional[float]]],
    ) -> int:
        """
        Choose the next probe index by balancing summed risk mass.

        The current search interval is (low, high], meaning low is known-good
        and high is known-bad, and the culprit lies somewhere in (low, high].

        This method chooses a mid in (low, high) such that the cumulative risk
        (sum of per-commit probabilities) is approximately equal on both sides:

          sum(p_i for i in (low, mid])  ~=  sum(p_i for i in (mid, high])
        """
        if not isinstance(risk_by_index, RiskSeries):
            raise ValueError(
                "RWAB-S requires risk_by_index to be a RiskSeries "
                f"(got {type(risk_by_index).__name__})"
            )

        left = low + 1
        right = high + 1  # half-open end
        total = risk_by_index.range_sum(left, right)
        if total < 0.0:
            raise RuntimeError(
                "RWAB-S computed a negative total risk mass for the current interval "
                f"(left={left}, right={right}, total={total})"
            )
        if total == 0.0:
            return (low + high) // 2

        prefix = risk_by_index.prefix_sums
        desired = float(prefix[left] + (total / 2.0))
        k = bisect.bisect_left(prefix, desired, lo=left, hi=right)
        mid = k - 1

        # Ensure progress even when prefix sums are flat or k hits an edge.
        if mid <= low:
            mid = low + 1
        if mid >= high:
            mid = high - 1
        if mid <= low or mid >= high:
            mid = (low + high) // 2
        return int(mid)

    def run(
        self,
        *,
        good_index: int,
        bad_index: int,
        culprit_index: int,
        risk_by_index: Optional[Sequence[Optional[float]]] = None,
    ) -> BisectionOutcome:
        if good_index >= bad_index:
            logger.debug("RWAB-SUM skipped: good_index=%d >= bad_index=%d", good_index, bad_index)
            return BisectionOutcome(tests=0, found_index=None)

        culprit_index = int(culprit_index)
        if culprit_index <= good_index or culprit_index > bad_index:
            logger.debug(
                "RWAB-SUM culprit out of range: good=%d bad=%d culprit=%d",
                good_index,
                bad_index,
                culprit_index,
            )
            return BisectionOutcome(tests=0, found_index=None)

        low = int(good_index)
        high = int(bad_index)
        tests = 0

        while high - low > 1:
            mid = self._choose_mid(low=low, high=high, risk_by_index=risk_by_index)
            tests += 1
            if mid >= culprit_index:
                high = mid
            else:
                low = mid

        logger.debug(
            "RWAB-SUM located culprit=%d in tests=%d (good=%d bad=%d)",
            high,
            tests,
            good_index,
            bad_index,
        )
        return BisectionOutcome(tests=tests, found_index=high)


class RiskWeightedAdaptiveBisectionLogSurvival:
    """
    Risk-Weighted Adaptive Bisection (RWAB-LS).

    Same as RWAB-SUM, but uses a combined probability mass over an interval:
      1 - ∏(1 - p_i)
    to balance the two time-ordered contiguous halves of the current search range.
    """

    name = "rwab-ls"

    def _choose_mid(
        self,
        *,
        low: int,
        high: int,
        risk_by_index: Optional[Sequence[Optional[float]]],
    ) -> int:
        """
        Choose the next probe index by balancing combined probability mass.

        The current search interval is (low, high], meaning low is known-good
        and high is known-bad, and the culprit lies somewhere in (low, high].

        This method chooses a mid in (low, high) such that the combined
        probability mass is approximately equal on both sides, where combined
        mass for a set of commits is:

          1 - ∏(1 - p_i)
        """
        if not isinstance(risk_by_index, RiskSeries):
            raise ValueError(
                "RWAB-LS requires risk_by_index to be a RiskSeries "
                f"(got {type(risk_by_index).__name__})"
            )

        left = low + 1
        right = high + 1  # half-open end
        total = risk_by_index.range_combined_probability(left, right)
        if total < 0.0:
            raise RuntimeError(
                "RWAB-LS computed a negative combined probability for the current interval "
                f"(left={left}, right={right}, total={total})"
            )
        if total == 0.0:
            return (low + high) // 2

        lo = low + 1
        hi = high - 1
        best_mid = (low + high) // 2
        best_gap = float("inf")

        # f(mid) = mass_left - mass_right is monotone increasing in mid for contiguous halves.
        while lo <= hi:
            mid = (lo + hi) // 2
            mass_left = risk_by_index.range_combined_probability(left, mid + 1)  # (low, mid]
            mass_right = risk_by_index.range_combined_probability(mid + 1, right)  # (mid, high]
            gap = abs(mass_left - mass_right)
            if gap < best_gap:
                best_gap = gap
                best_mid = mid

            if mass_left < mass_right:
                lo = mid + 1
            else:
                hi = mid - 1

        if best_mid <= low:
            best_mid = low + 1
        if best_mid >= high:
            best_mid = high - 1
        if best_mid <= low or best_mid >= high:
            best_mid = (low + high) // 2
        return int(best_mid)

    def run(
        self,
        *,
        good_index: int,
        bad_index: int,
        culprit_index: int,
        risk_by_index: Optional[Sequence[Optional[float]]] = None,
    ) -> BisectionOutcome:
        if good_index >= bad_index:
            logger.debug("RWAB-LS skipped: good_index=%d >= bad_index=%d", good_index, bad_index)
            return BisectionOutcome(tests=0, found_index=None)

        culprit_index = int(culprit_index)
        if culprit_index <= good_index or culprit_index > bad_index:
            logger.debug(
                "RWAB-LS culprit out of range: good=%d bad=%d culprit=%d",
                good_index,
                bad_index,
                culprit_index,
            )
            return BisectionOutcome(tests=0, found_index=None)

        low = int(good_index)
        high = int(bad_index)
        tests = 0

        while high - low > 1:
            mid = self._choose_mid(low=low, high=high, risk_by_index=risk_by_index)
            tests += 1
            if mid >= culprit_index:
                high = mid
            else:
                low = mid

        logger.debug(
            "RWAB-LS located culprit=%d in tests=%d (good=%d bad=%d)",
            high,
            tests,
            good_index,
            bad_index,
        )
        return BisectionOutcome(tests=tests, found_index=high)


BISECTION_STRATEGIES = {
    GitBisectBaseline.name: GitBisectBaseline,
    RiskWeightedAdaptiveBisectionSum.name: RiskWeightedAdaptiveBisectionSum,
    RiskWeightedAdaptiveBisectionLogSurvival.name: RiskWeightedAdaptiveBisectionLogSurvival,
}
