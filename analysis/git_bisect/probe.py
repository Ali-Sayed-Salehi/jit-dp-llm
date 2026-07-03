from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, List, Optional, Tuple


class ProbeOutcome(str, Enum):
    """Tri-state result for a simulated revision probe."""

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True)
class ProbeResult:
    """Outcome returned by a probe function."""

    outcome: ProbeOutcome
    is_new: bool = True


@dataclass(frozen=True)
class ProbeAttempt:
    """A newly charged probe attempt."""

    index: int
    outcome: ProbeOutcome


ProbeFn = Callable[[int], ProbeResult]


def pass_fail_outcome(*, index: int, culprit_index: int) -> ProbeOutcome:
    """Return the monotone pass/fail outcome for an index."""

    return ProbeOutcome.FAIL if int(index) >= int(culprit_index) else ProbeOutcome.PASS


def outcome_to_failed(outcome: ProbeOutcome | bool) -> Optional[bool]:
    """Convert PASS/FAIL to the historical failed bool; return None for SKIP."""

    if isinstance(outcome, bool):
        return bool(outcome)
    probe_outcome = ProbeOutcome(outcome)
    if probe_outcome == ProbeOutcome.FAIL:
        return True
    if probe_outcome == ProbeOutcome.PASS:
        return False
    return None


def nearby_indices(*, target: int, min_index: int, max_index: int) -> Iterable[int]:
    """
    Yield `target`, then alternating lower/higher neighbors within the range.

    The range is inclusive on both ends.
    """

    target = int(target)
    min_index = int(min_index)
    max_index = int(max_index)
    if min_index > max_index:
        return

    if min_index <= target <= max_index:
        yield target

    max_delta = max(abs(target - min_index), abs(max_index - target))
    for delta in range(1, max_delta + 1):
        lower = target - delta
        if lower >= min_index:
            yield lower
        higher = target + delta
        if higher <= max_index:
            yield higher


def probe_nearby_usable(
    *,
    target: int,
    min_index: int,
    max_index: int,
    probe: ProbeFn,
    attempts: List[ProbeAttempt],
) -> Optional[Tuple[int, ProbeOutcome]]:
    """
    Probe around `target` until a PASS/FAIL revision is found.

    Newly charged probe attempts are appended to `attempts`. Cached probe
    results returned with `is_new=False` are reused without additional cost.
    """

    for idx in nearby_indices(target=int(target), min_index=int(min_index), max_index=int(max_index)):
        result = probe(int(idx))
        if result.is_new:
            attempts.append(ProbeAttempt(index=int(idx), outcome=ProbeOutcome(result.outcome)))
        outcome = ProbeOutcome(result.outcome)
        if outcome != ProbeOutcome.SKIP:
            return int(idx), outcome
    return None
