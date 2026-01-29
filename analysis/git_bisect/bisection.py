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
import heapq
from typing import Mapping, Optional, Protocol, Sequence, overload

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
        known_results: Optional[Mapping[int, bool]] = None,
    ) -> BisectionOutcome:
        """Run a bisection procedure and return test count plus found culprit."""
        ...


def _init_cache(
    *,
    good_index: int,
    bad_index: int,
    known_results: Optional[Mapping[int, bool]],
) -> dict[int, bool]:
    """Initialize a pass/fail cache from (optional) prior known results."""
    cache: dict[int, bool] = {}
    if known_results:
        for idx, failed in known_results.items():
            cache[int(idx)] = bool(failed)

    cache[int(good_index)] = False
    cache[int(bad_index)] = True
    return cache


def _tighten_bounds(
    *,
    good_index: int,
    bad_index: int,
    cache: Mapping[int, bool],
) -> tuple[int, int]:
    """
    Tighten (good,bad] bounds using any known pass/fail results.

    Returns (low, high) where:
      - low is a known-good index (pass)
      - high is a known-bad index (fail)
    """
    good_index = int(good_index)
    bad_index = int(bad_index)
    low = good_index
    high = bad_index

    best_good: Optional[int] = None
    best_bad: Optional[int] = None
    for idx, failed in cache.items():
        if idx < good_index or idx > bad_index:
            continue
        if failed:
            if best_bad is None or idx < best_bad:
                best_bad = int(idx)
        else:
            if best_good is None or idx > best_good:
                best_good = int(idx)

    if best_good is not None:
        low = max(low, int(best_good))
    if best_bad is not None:
        high = min(high, int(best_bad))

    # If known results are inconsistent, fall back to the original bounds.
    if low >= high:
        return good_index, bad_index
    return low, high


@dataclass
class _BisectionTester:
    """Monotone test simulator with memoization and a cache-miss test counter."""

    cache: dict[int, bool]
    culprit_index: int
    tests: int = 0

    def test(self, idx: int) -> bool:
        """Return True if commit idx fails; increments `tests` when uncached."""
        idx = int(idx)
        if idx in self.cache:
            return self.cache[idx]
        self.tests += 1
        out = bool(idx >= int(self.culprit_index))
        self.cache[idx] = out
        return out


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
        known_results: Optional[Mapping[int, bool]] = None,
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

        cache = _init_cache(good_index=good_index, bad_index=bad_index, known_results=known_results)
        low, high = _tighten_bounds(good_index=good_index, bad_index=bad_index, cache=cache)
        tester = _BisectionTester(cache=cache, culprit_index=culprit_index)

        while high - low > 1:
            mid = (low + high) // 2
            mid_failed = tester.test(mid)
            if mid_failed:
                high = mid
            else:
                low = mid

        logger.debug(
            "Bisect located culprit=%d in tests=%d (good=%d bad=%d)",
            high,
            tester.tests,
            good_index,
            bad_index,
        )
        return BisectionOutcome(tests=tester.tests, found_index=high)


class RiskWeightedBisectionSum:
    """
    Risk-Weighted Bisection (RWB-SUM).

    Like standard git bisect on a linear history, but chooses each probe index
    by approximately balancing the cumulative predicted risk mass in the two
    contiguous halves of the current search interval.

    For a current interval (low, high], RWB chooses a mid in (low, high) such
    that:
      Σ p_i over (low, mid] ~= Σ p_i over (mid, high]
    preserving time order by only splitting into two contiguous sub-batches.
    """

    name = "rwb-s"

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
                "RWB-S requires risk_by_index to be a RiskSeries "
                f"(got {type(risk_by_index).__name__})"
            )

        left = low + 1
        right = high + 1  # half-open end
        total = risk_by_index.range_sum(left, right)
        if total < 0.0:
            raise RuntimeError(
                "RWB-S computed a negative total risk mass for the current interval "
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
        known_results: Optional[Mapping[int, bool]] = None,
    ) -> BisectionOutcome:
        if good_index >= bad_index:
            logger.debug("RWB-SUM skipped: good_index=%d >= bad_index=%d", good_index, bad_index)
            return BisectionOutcome(tests=0, found_index=None)

        culprit_index = int(culprit_index)
        if culprit_index <= good_index or culprit_index > bad_index:
            logger.debug(
                "RWB-SUM culprit out of range: good=%d bad=%d culprit=%d",
                good_index,
                bad_index,
                culprit_index,
            )
            return BisectionOutcome(tests=0, found_index=None)

        cache = _init_cache(good_index=good_index, bad_index=bad_index, known_results=known_results)
        low, high = _tighten_bounds(good_index=good_index, bad_index=bad_index, cache=cache)
        tester = _BisectionTester(cache=cache, culprit_index=culprit_index)

        while high - low > 1:
            mid = self._choose_mid(low=low, high=high, risk_by_index=risk_by_index)
            mid_failed = tester.test(mid)
            if mid_failed:
                high = mid
            else:
                low = mid

        logger.debug(
            "RWB-SUM located culprit=%d in tests=%d (good=%d bad=%d)",
            high,
            tester.tests,
            good_index,
            bad_index,
        )
        return BisectionOutcome(tests=tester.tests, found_index=high)


class RiskWeightedBisectionLogSurvival:
    """
    Risk-Weighted Bisection (RWB-LS).

    Same as RWB-SUM, but uses a combined probability mass over an interval:
      1 - ∏(1 - p_i)
    to balance the two time-ordered contiguous halves of the current search range.
    """

    name = "rwb-ls"

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
                "RWB-LS requires risk_by_index to be a RiskSeries "
                f"(got {type(risk_by_index).__name__})"
            )

        left = low + 1
        right = high + 1  # half-open end
        total = risk_by_index.range_combined_probability(left, right)
        if total < 0.0:
            raise RuntimeError(
                "RWB-LS computed a negative combined probability for the current interval "
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
        known_results: Optional[Mapping[int, bool]] = None,
    ) -> BisectionOutcome:
        if good_index >= bad_index:
            logger.debug("RWB-LS skipped: good_index=%d >= bad_index=%d", good_index, bad_index)
            return BisectionOutcome(tests=0, found_index=None)

        culprit_index = int(culprit_index)
        if culprit_index <= good_index or culprit_index > bad_index:
            logger.debug(
                "RWB-LS culprit out of range: good=%d bad=%d culprit=%d",
                good_index,
                bad_index,
                culprit_index,
            )
            return BisectionOutcome(tests=0, found_index=None)

        cache = _init_cache(good_index=good_index, bad_index=bad_index, known_results=known_results)
        low, high = _tighten_bounds(good_index=good_index, bad_index=bad_index, cache=cache)
        tester = _BisectionTester(cache=cache, culprit_index=culprit_index)

        while high - low > 1:
            mid = self._choose_mid(low=low, high=high, risk_by_index=risk_by_index)
            mid_failed = tester.test(mid)
            if mid_failed:
                high = mid
            else:
                low = mid

        logger.debug(
            "RWB-LS located culprit=%d in tests=%d (good=%d bad=%d)",
            high,
            tester.tests,
            good_index,
            bad_index,
        )
        return BisectionOutcome(tests=tester.tests, found_index=high)


class SequentialWalkBackwardBisection:
    """
    Sequential Walk Backward Bisection (SWBB).

    Model:
      - We start with a known-good commit at `good_index` and a known-bad commit at
        `bad_index`, so the culprit lies in (good_index, bad_index].
      - Treat `bad_index` as the current known-bad boundary `hi`.
      - Sequentially test commits in reverse: hi-1, hi-2, ...
      - Each failing test moves `hi` backward.
      - The first passing test identifies the culprit as the current `hi`
        (the next commit after the passing boundary).

    This corresponds to testing prefixes [lo, hi-1], [lo, hi-2], ... in a linear
    history where testing at commit k is equivalent to testing [0, k].
    """

    name = "swbb"

    def run(
        self,
        *,
        good_index: int,
        bad_index: int,
        culprit_index: int,
        risk_by_index: Optional[Sequence[Optional[float]]] = None,
        known_results: Optional[Mapping[int, bool]] = None,
    ) -> BisectionOutcome:
        _ = risk_by_index  # not used by this strategy

        if good_index >= bad_index:
            logger.debug("SWBB skipped: good_index=%d >= bad_index=%d", good_index, bad_index)
            return BisectionOutcome(tests=0, found_index=None)

        culprit_index = int(culprit_index)
        if culprit_index <= good_index or culprit_index > bad_index:
            logger.debug(
                "SWBB culprit out of range: good=%d bad=%d culprit=%d",
                good_index,
                bad_index,
                culprit_index,
            )
            return BisectionOutcome(tests=0, found_index=None)

        cache = _init_cache(good_index=good_index, bad_index=bad_index, known_results=known_results)
        low, high = _tighten_bounds(good_index=good_index, bad_index=bad_index, cache=cache)
        good_index = int(low)
        hi = int(high)  # known-bad boundary
        tester = _BisectionTester(cache=cache, culprit_index=culprit_index)

        # Probe from (bad-1) down to (good+1).
        for probe in range(int(high) - 1, good_index, -1):
            probe_failed = tester.test(probe)
            if probe_failed:
                # Still failing; move the known-bad boundary backward.
                hi = probe
                continue

            # Passing boundary found; culprit is the current known-bad boundary.
            logger.debug(
                "SWBB located culprit=%d in tests=%d (good=%d bad=%d)",
                hi,
                tester.tests,
                good_index,
                bad_index,
            )
            return BisectionOutcome(tests=tester.tests, found_index=hi)

        # Never observed a pass within (good, bad); culprit must be good+1.
        logger.debug(
            "SWBB located culprit=%d in tests=%d (good=%d bad=%d)",
            hi,
            tester.tests,
            good_index,
            bad_index,
        )
        return BisectionOutcome(tests=tester.tests, found_index=hi)


class SequentialWalkForwardBisection:
    """
    Sequential Walk Forward Bisection (SWFB).

    Model:
      - We start with a known-good commit at `good_index` and a known-bad commit at
        `bad_index`, so the culprit lies in (good_index, bad_index].
      - Sequentially test commits moving forward from the start of the unknown
        region: good+1, good+2, ...
      - The first failing test identifies the culprit index (since the previous
        commit must have been clean).
    """

    name = "swfb"

    def run(
        self,
        *,
        good_index: int,
        bad_index: int,
        culprit_index: int,
        risk_by_index: Optional[Sequence[Optional[float]]] = None,
        known_results: Optional[Mapping[int, bool]] = None,
    ) -> BisectionOutcome:
        _ = risk_by_index  # not used by this strategy

        if good_index >= bad_index:
            logger.debug("SWFB skipped: good_index=%d >= bad_index=%d", good_index, bad_index)
            return BisectionOutcome(tests=0, found_index=None)

        culprit_index = int(culprit_index)
        if culprit_index <= good_index or culprit_index > bad_index:
            logger.debug(
                "SWFB culprit out of range: good=%d bad=%d culprit=%d",
                good_index,
                bad_index,
                culprit_index,
            )
            return BisectionOutcome(tests=0, found_index=None)

        cache = _init_cache(good_index=good_index, bad_index=bad_index, known_results=known_results)
        low, high = _tighten_bounds(good_index=good_index, bad_index=bad_index, cache=cache)
        good_index = int(low)
        bad_index = int(high)
        tester = _BisectionTester(cache=cache, culprit_index=culprit_index)

        for probe in range(good_index + 1, bad_index + 1):
            probe_failed = tester.test(probe)
            if probe_failed:
                logger.debug(
                    "SWFB located culprit=%d in tests=%d (good=%d bad=%d)",
                    probe,
                    tester.tests,
                    good_index,
                    bad_index,
                )
                return BisectionOutcome(tests=tester.tests, found_index=int(probe))

        # Should be unreachable due to the culprit range check above.
        return BisectionOutcome(tests=tester.tests, found_index=None)


class TopKRiskFirstBisection:
    """
    Top-K Risk-First Bisection (TKRB-K).

    Phase 1 (risk-first):
      - Select the top-K highest-risk commits in (good_index, bad_index] using
        `risk_by_index`.
      - For each candidate commit `i`, test `i-1` and `i` (caching results).
      - If `i` fails and `i-1` passes, then `i` is the culprit.

    Phase 2 (fallback):
      - If not found, run a standard git-bisect binary search on (good,bad]
        while reusing any cached test results from phase 1.
    """

    name = "tkrb-k"

    def __init__(self, k: int = 10) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = int(k)

    def _risk(self, idx: int, risk_by_index: Sequence[Optional[float]]) -> float:
        """Return the (possibly missing) per-commit risk score as a float."""
        v = risk_by_index[idx]
        return 0.0 if v is None else float(v)

    def run(
        self,
        *,
        good_index: int,
        bad_index: int,
        culprit_index: int,
        risk_by_index: Optional[Sequence[Optional[float]]] = None,
        known_results: Optional[Mapping[int, bool]] = None,
    ) -> BisectionOutcome:
        if risk_by_index is None:
            raise ValueError("TKRB-K requires risk_by_index to be provided")

        if good_index >= bad_index:
            logger.debug("TKRB-K skipped: good_index=%d >= bad_index=%d", good_index, bad_index)
            return BisectionOutcome(tests=0, found_index=None)

        culprit_index = int(culprit_index)
        if culprit_index <= good_index or culprit_index > bad_index:
            logger.debug(
                "TKRB-K culprit out of range: good=%d bad=%d culprit=%d",
                good_index,
                bad_index,
                culprit_index,
            )
            return BisectionOutcome(tests=0, found_index=None)

        good_index = int(good_index)
        bad_index = int(bad_index)

        cache = _init_cache(good_index=good_index, bad_index=bad_index, known_results=known_results)
        low, high = _tighten_bounds(good_index=good_index, bad_index=bad_index, cache=cache)
        good_index = int(low)
        bad_index = int(high)
        tester = _BisectionTester(cache=cache, culprit_index=culprit_index)

        # ---- Phase 1: risk-first scan over top-K commits. ----
        start = good_index + 1
        end = bad_index + 1  # inclusive end for range()
        if start < end:
            k = min(self.k, end - start)
            top_indices = heapq.nlargest(
                k,
                range(start, end),
                key=lambda i: (self._risk(i, risk_by_index), -int(i)),
            )

            for idx in top_indices:
                prev = idx - 1
                if prev <= good_index:
                    prev_failed = False
                    cache[prev] = False
                else:
                    prev_failed = tester.test(prev)

                idx_failed = tester.test(idx)
                if idx_failed and not prev_failed:
                    logger.debug(
                        "TKRB-K found culprit=%d during risk-first phase (tests=%d good=%d bad=%d k=%d)",
                        idx,
                        tester.tests,
                        good_index,
                        bad_index,
                        self.k,
                    )
                    return BisectionOutcome(tests=tester.tests, found_index=int(idx))

        # Tighten the search interval using any newly discovered pass/fail results.
        low, high = _tighten_bounds(good_index=good_index, bad_index=bad_index, cache=cache)

        # ---- Phase 2: standard git-bisect with memoization. ----
        while high - low > 1:
            mid = (low + high) // 2
            mid_failed = tester.test(mid)
            if mid_failed:
                high = mid
            else:
                low = mid

        logger.debug(
            "TKRB-K located culprit=%d in tests=%d (good=%d bad=%d k=%d)",
            high,
            tester.tests,
            good_index,
            bad_index,
            self.k,
        )
        return BisectionOutcome(tests=tester.tests, found_index=int(high))


BISECTION_STRATEGIES = {
    GitBisectBaseline.name: GitBisectBaseline,
    RiskWeightedBisectionSum.name: RiskWeightedBisectionSum,
    RiskWeightedBisectionLogSurvival.name: RiskWeightedBisectionLogSurvival,
    SequentialWalkBackwardBisection.name: SequentialWalkBackwardBisection,
    SequentialWalkForwardBisection.name: SequentialWalkForwardBisection,
    TopKRiskFirstBisection.name: TopKRiskFirstBisection,
}
