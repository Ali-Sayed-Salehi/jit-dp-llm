"""
Batching strategies for the batch-testing simulator.

Batching strategies decide *when* to trigger a batch test and *which commits*
are included in each batch. When a batch is flushed, the chosen bisection
strategy (`bisect_fn`) is invoked to identify culprit commits within that batch.

See `analysis/batch_testing/README.md` for a detailed conceptual overview of
all batching and bisection strategies and the shared time/cost model.
"""

from datetime import timedelta
import math
import logging
import heapq

import bisection_strats as bisection_mod
from bisection_strats import (
    TestExecutor,
    run_test_suite,
    schedule_test_suite_jobs,
    get_batch_signature_durations,
    get_tested_signatures_for_revision,
    get_signature_durations_for_ids,
    get_failing_signature_groups_for_revision,
)


logger = logging.getLogger(__name__)

def _accumulate_log_survival(log_survival: float, p: float) -> float:
    """
    Update log survival mass:

        log_survival := Σ log(1 - p_i)

    using `log1p` for numerical stability.

    `p` is expected to be in [0, 1]. If p == 1, survival becomes 0 (log=-inf).
    """
    p = float(p)
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Risk probabilities must be in [0,1], got {p}")
    if p >= 1.0:
        return float("-inf")
    if math.isinf(log_survival) and log_survival < 0:
        return float("-inf")
    return float(log_survival + math.log1p(-p))


def _combined_probability_from_log_survival(log_survival: float) -> float:
    """
    Convert log survival mass to combined probability:

        combined = 1 - exp(log_survival)
    """
    if math.isinf(log_survival) and log_survival < 0:
        return 1.0
    return float(1.0 - math.exp(float(log_survival)))


def _percentile_linear(sorted_values, p: float) -> float:
    """
    Compute the p-th percentile (0-100) using linear interpolation.

    Expects `sorted_values` to be a non-decreasing sequence.
    Returns 0.0 for empty inputs.
    """
    if not sorted_values:
        return 0.0

    p = float(p)
    if p <= 0.0:
        return float(sorted_values[0])
    if p >= 100.0:
        return float(sorted_values[-1])

    n = len(sorted_values)
    # Interpolate between ranks over [0, n-1].
    pos = (n - 1) * (p / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])

    frac = pos - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac)


def build_results(
    total_tests_run,
    culprit_times,
    feedback_times,
    total_cpu_time_min,
    num_regressors_total=None,
    num_regressors_found=None,
):
    """
    Build the standard results payload returned by batching simulations.

    Parameters
    ----------
    total_tests_run:
        Total number of perf job executions (signature-group runs, across all suite runs).
    culprit_times:
        List of minutes-to-culprit for regressors that were found.
    feedback_times:
        Mapping `commit_id -> minutes-to-feedback` for commits that received
        feedback during the simulation.
    total_cpu_time_min:
        Cumulative CPU minutes across all scheduled tests (sums durations,
        independent of parallelism).
    num_regressors_total:
        Total number of true regressors present in the simulation window.
    num_regressors_found:
        Number of true regressors actually identified as culprits by the combo.

    Returns
    -------
    dict with metrics in minutes.
    """
    if feedback_times:
        mean_fb = sum(feedback_times.values()) / len(feedback_times)
    else:
        mean_fb = 0.0

    if culprit_times:
        culprit_times_sorted = sorted(float(x) for x in culprit_times)
        mean_ttc = sum(culprit_times_sorted) / len(culprit_times_sorted)
        max_ttc = culprit_times_sorted[-1]
        p90_ttc = _percentile_linear(culprit_times_sorted, 90)
        p95_ttc = _percentile_linear(culprit_times_sorted, 95)
        p99_ttc = _percentile_linear(culprit_times_sorted, 99)
    else:
        mean_ttc = 0.0
        max_ttc = 0.0
        p90_ttc = 0.0
        p95_ttc = 0.0
        p99_ttc = 0.0

    if num_regressors_total is None:
        num_regressors_total = 0
    if num_regressors_found is None:
        num_regressors_found = len(culprit_times) if culprit_times else 0
    num_regressors_total = int(num_regressors_total)
    num_regressors_found = int(num_regressors_found)

    return {
        "total_tests_run": total_tests_run,
        "mean_feedback_time_min": round(mean_fb, 2),
        "mean_time_to_culprit_min": round(mean_ttc, 2),
        "max_time_to_culprit_min": round(max_ttc, 2),
        "p90_time_to_culprit_min": round(p90_ttc, 2),
        "p95_time_to_culprit_min": round(p95_ttc, 2),
        "p99_time_to_culprit_min": round(p99_ttc, 2),
        "total_cpu_time_min": round(float(total_cpu_time_min), 2),
        "num_regressors_total": num_regressors_total,
        "num_regressors_found": num_regressors_found,
        "found_all_regressors": (num_regressors_found == num_regressors_total),
    }

def _union_tested_signature_group_ids_for_commits(commits):
    """
    Compute the union of perf signature-group IDs that Mozilla actually tested across
    the given commits.

    Used by the "-s" batching variants, where each batch's initial suite is the
    union of signature-groups observed in that batch (instead of the global full
    suite). Group membership comes from
    `perf_jobs_per_revision_details_rectified.jsonl` via
    `get_tested_signatures_for_revision`.
    """
    sig_group_ids = set()
    for c in commits:
        rev = c["commit_id"]
        for sig_group in get_tested_signatures_for_revision(rev):
            try:
                sig_group_ids.add(int(sig_group))
            except (TypeError, ValueError):
                continue
    return sig_group_ids


class _StreamingMetrics:
    __slots__ = ("total_tests_run", "culprit_times", "feedback_times", "found_regressors")

    def __init__(self, total_tests_run, culprit_times, feedback_times, found_regressors):
        self.total_tests_run = int(total_tests_run)
        self.culprit_times = culprit_times
        self.feedback_times = feedback_times
        self.found_regressors = found_regressors


def _record_feedback_if_first(metrics: _StreamingMetrics, commit: dict, at_time) -> None:
    cid = commit.get("commit_id")
    if cid is None or cid in metrics.feedback_times:
        return
    fb_min = (at_time - commit["ts"]).total_seconds() / 60.0
    metrics.feedback_times[cid] = fb_min


def _record_regressor_found_if_first(metrics: _StreamingMetrics, commit: dict, at_time) -> None:
    if not commit.get("true_label"):
        return
    cid = commit.get("commit_id")
    if cid is None or cid in metrics.found_regressors:
        return
    metrics.found_regressors.add(cid)
    ttc_min = (at_time - commit["ts"]).total_seconds() / 60.0
    metrics.culprit_times.append(ttc_min)


class _SigGroupBisectProcess:
    """
    Base class for event-driven bisection on a single signature-group.

    This is used by the streaming batch-test model where each failing
    signature-group job triggers its own bisection starting at the time that
    job finishes, rather than waiting for the entire batch suite to complete.
    """

    def __init__(
        self,
        batch,
        defect_indices,
        sig_group_id,
        executor: TestExecutor,
        metrics: _StreamingMetrics,
        push_event,
    ):
        self.batch = batch
        self.defect_indices = set(int(i) for i in (defect_indices or []))
        # `sig_group_id` is typically an int, but can be None in fallback cases
        # where failing signature-groups cannot be derived from metadata.
        self.sig_group_id = sig_group_id
        self.executor = executor
        self.metrics = metrics
        self._push_event = push_event

        self.n = len(batch)
        self.status = ["unknown"] * self.n

        # Each interval test in this process runs only this signature-group.
        self.durations = get_signature_durations_for_ids([self.sig_group_id])

    def _interval_has_defect(self, lo: int, hi: int) -> bool:
        return any(
            (i in self.defect_indices) and (self.status[i] != "defect_found")
            for i in range(lo, hi + 1)
        )

    def _mark_clean_interval(self, lo: int, hi: int, finish_time) -> None:
        for i in range(lo, hi + 1):
            if self.status[i] != "unknown":
                continue
            self.status[i] = "clean"
            _record_feedback_if_first(self.metrics, self.batch[i], finish_time)

    def _mark_defect_found(self, idx: int, finish_time) -> None:
        if idx < 0 or idx >= self.n:
            raise IndexError(f"defect index out of range: {idx} (n={self.n})")
        if self.status[idx] == "defect_found":
            return
        self.status[idx] = "defect_found"
        c = self.batch[idx]
        _record_feedback_if_first(self.metrics, c, finish_time)
        if idx in self.defect_indices:
            _record_regressor_found_if_first(self.metrics, c, finish_time)

    def _submit_interval(self, op: str, lo: int, hi: int, requested_time, **payload_extra):
        if lo > hi:
            return
        has_defect = self._interval_has_defect(lo, hi)
        self.metrics.total_tests_run += len(self.durations)
        finish_time = run_test_suite(self.executor, requested_time, self.durations)
        self._push_event(
            finish_time,
            "bisect_interval_done",
            {
                "proc": self,
                "op": op,
                "lo": int(lo),
                "hi": int(hi),
                "has_defect": bool(has_defect),
                **payload_extra,
            },
        )

    def start(self, start_time):
        raise NotImplementedError

    def on_interval_done(self, payload: dict, finish_time):
        raise NotImplementedError


class _TOBProcess(_SigGroupBisectProcess):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = "need_whole"
        self.lo = None
        self.hi = None
        self.left = None
        self.right = None

    def _schedule_next(self, now):
        while True:
            unknown = [i for i, s in enumerate(self.status) if s == "unknown"]
            if not unknown:
                return
            if self.mode == "need_whole":
                self.lo = unknown[0]
                self.hi = unknown[-1]
                self._submit_interval("whole", self.lo, self.hi, now)
                return

            if self.mode != "binary":
                raise RuntimeError(f"TOBProcess: unknown mode {self.mode!r}")

            if self.left is None or self.right is None:
                raise RuntimeError("TOBProcess: binary mode without bounds")

            if self.left >= self.right:
                self._mark_defect_found(int(self.left), now)
                self.mode = "need_whole"
                self.left = None
                self.right = None
                continue

            mid = (self.left + self.right) // 2
            self._submit_interval("left", int(self.left), int(mid), now, mid=int(mid))
            return

    def start(self, start_time):
        self._schedule_next(start_time)

    def on_interval_done(self, payload: dict, finish_time):
        op = payload.get("op")
        lo = int(payload["lo"])
        hi = int(payload["hi"])
        has_defect = bool(payload["has_defect"])

        if not has_defect and op in ("whole", "left"):
            self._mark_clean_interval(lo, hi, finish_time)

        if op == "whole":
            if not has_defect:
                return
            self.mode = "binary"
            self.left = lo
            self.right = hi
            self._schedule_next(finish_time)
            return

        if op == "left":
            mid = int(payload["mid"])
            if has_defect:
                self.right = mid
            else:
                self.left = mid + 1
            self._schedule_next(finish_time)
            return

        raise RuntimeError(f"TOBProcess: unexpected op {op!r}")


class _RWABProcess(_SigGroupBisectProcess):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = "need_whole"
        self.lo = None
        self.hi = None
        self.left = None
        self.right = None

        risk_prefix = [0.0] * (self.n + 1)
        for i, c in enumerate(self.batch):
            if "risk" not in c or c["risk"] is None:
                raise ValueError(
                    f"Missing or None 'risk' value in batch for RWAB at index {i}: {c!r}"
                )
            risk_prefix[i + 1] = risk_prefix[i] + float(c["risk"])
        self._risk_prefix = risk_prefix

    def _risk_sum(self, lo, hi):
        return self._risk_prefix[hi + 1] - self._risk_prefix[lo]

    def _choose_risk_balanced_split(self, lo, hi):
        if lo >= hi:
            return lo
        total_risk = self._risk_sum(lo, hi)
        if total_risk <= 0.0:
            return (lo + hi) // 2
        target = total_risk / 2.0
        best_idx = lo
        best_diff = float("inf")
        for s in range(lo, hi):
            left_risk = self._risk_sum(lo, s)
            diff = abs(left_risk - target)
            if diff < best_diff:
                best_diff = diff
                best_idx = s
        return best_idx

    def _schedule_next(self, now):
        while True:
            unknown = [i for i, s in enumerate(self.status) if s == "unknown"]
            if not unknown:
                return
            if self.mode == "need_whole":
                self.lo = unknown[0]
                self.hi = unknown[-1]
                self._submit_interval("whole", self.lo, self.hi, now)
                return

            if self.mode != "binary":
                raise RuntimeError(f"RWABProcess: unknown mode {self.mode!r}")

            if self.left is None or self.right is None:
                raise RuntimeError("RWABProcess: binary mode without bounds")

            if self.left >= self.right:
                self._mark_defect_found(int(self.left), now)
                self.mode = "need_whole"
                self.left = None
                self.right = None
                continue

            split = self._choose_risk_balanced_split(int(self.left), int(self.right))
            self._submit_interval("left", int(self.left), int(split), now, split=int(split))
            return

    def start(self, start_time):
        self._schedule_next(start_time)

    def on_interval_done(self, payload: dict, finish_time):
        op = payload.get("op")
        lo = int(payload["lo"])
        hi = int(payload["hi"])
        has_defect = bool(payload["has_defect"])

        if not has_defect and op in ("whole", "left"):
            self._mark_clean_interval(lo, hi, finish_time)

        if op == "whole":
            if not has_defect:
                return
            self.mode = "binary"
            self.left = lo
            self.right = hi
            self._schedule_next(finish_time)
            return

        if op == "left":
            split = int(payload["split"])
            if has_defect:
                self.right = split
            else:
                self.left = split + 1
            self._schedule_next(finish_time)
            return

        raise RuntimeError(f"RWABProcess: unexpected op {op!r}")


class _RWABLSProcess(_SigGroupBisectProcess):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = "need_whole"
        self.lo = None
        self.hi = None
        self.left = None
        self.right = None

        log_survival_prefix = [0.0] * (self.n + 1)
        certain_fail_prefix = [0] * (self.n + 1)
        for i, c in enumerate(self.batch):
            if "risk" not in c or c["risk"] is None:
                raise ValueError(
                    "Missing or None 'risk' value in batch for RWAB-LS "
                    f"at index {i}: {c!r}"
                )
            p = float(c["risk"])
            if p < 0.0 or p > 1.0:
                raise ValueError(f"Risk probabilities must be in [0,1], got {p} at index {i}: {c!r}")

            log_survival_prefix[i + 1] = log_survival_prefix[i]
            certain_fail_prefix[i + 1] = certain_fail_prefix[i]
            if p >= 1.0:
                certain_fail_prefix[i + 1] += 1
            elif p > 0.0:
                log_survival_prefix[i + 1] += math.log1p(-p)

        self._log_survival_prefix = log_survival_prefix
        self._certain_fail_prefix = certain_fail_prefix

    def _combined_probability(self, lo, hi):
        if lo > hi:
            return 0.0
        start = lo
        end = hi + 1
        if self._certain_fail_prefix[end] - self._certain_fail_prefix[start] > 0:
            return 1.0
        log_survival = float(self._log_survival_prefix[end] - self._log_survival_prefix[start])
        return float(1.0 - math.exp(log_survival))

    def _choose_probability_balanced_split(self, lo, hi):
        if lo >= hi:
            return lo
        total_mass = self._combined_probability(lo, hi)
        if total_mass <= 0.0:
            return (lo + hi) // 2

        search_lo = lo
        search_hi = hi - 1
        best_idx = (lo + hi) // 2
        best_gap = float("inf")

        while search_lo <= search_hi:
            mid = (search_lo + search_hi) // 2
            mass_left = self._combined_probability(lo, mid)
            mass_right = self._combined_probability(mid + 1, hi)
            gap = abs(mass_left - mass_right)
            if gap < best_gap:
                best_gap = gap
                best_idx = mid
            if mass_left < mass_right:
                search_lo = mid + 1
            else:
                search_hi = mid - 1

        if best_idx < lo:
            best_idx = lo
        if best_idx >= hi:
            best_idx = hi - 1
        return best_idx

    def _schedule_next(self, now):
        while True:
            unknown = [i for i, s in enumerate(self.status) if s == "unknown"]
            if not unknown:
                return
            if self.mode == "need_whole":
                self.lo = unknown[0]
                self.hi = unknown[-1]
                self._submit_interval("whole", self.lo, self.hi, now)
                return

            if self.mode != "binary":
                raise RuntimeError(f"RWABLSProcess: unknown mode {self.mode!r}")

            if self.left is None or self.right is None:
                raise RuntimeError("RWABLSProcess: binary mode without bounds")

            if self.left >= self.right:
                self._mark_defect_found(int(self.left), now)
                self.mode = "need_whole"
                self.left = None
                self.right = None
                continue

            split = self._choose_probability_balanced_split(int(self.left), int(self.right))
            self._submit_interval("left", int(self.left), int(split), now, split=int(split))
            return

    def start(self, start_time):
        self._schedule_next(start_time)

    def on_interval_done(self, payload: dict, finish_time):
        op = payload.get("op")
        lo = int(payload["lo"])
        hi = int(payload["hi"])
        has_defect = bool(payload["has_defect"])

        if not has_defect and op in ("whole", "left"):
            self._mark_clean_interval(lo, hi, finish_time)

        if op == "whole":
            if not has_defect:
                return
            self.mode = "binary"
            self.left = lo
            self.right = hi
            self._schedule_next(finish_time)
            return

        if op == "left":
            split = int(payload["split"])
            if has_defect:
                self.right = split
            else:
                self.left = split + 1
            self._schedule_next(finish_time)
            return

        raise RuntimeError(f"RWABLSProcess: unexpected op {op!r}")


class _SWBProcess(_SigGroupBisectProcess):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = "need_whole"
        self.walk_lo = None
        self.walk_hi = None
        self.walk_k = None

    def _schedule_next(self, now):
        while True:
            unknown = [i for i, s in enumerate(self.status) if s == "unknown"]
            if not unknown:
                return

            if self.mode == "need_whole":
                lo = unknown[0]
                hi = unknown[-1]
                self._submit_interval("whole", lo, hi, now)
                return

            if self.mode != "walk":
                raise RuntimeError(f"SWBProcess: unknown mode {self.mode!r}")

            if self.walk_lo is None or self.walk_hi is None or self.walk_k is None:
                raise RuntimeError("SWBProcess: walk mode without bounds")

            if self.walk_lo == self.walk_hi:
                self._mark_defect_found(int(self.walk_lo), now)
                self.mode = "need_whole"
                self.walk_lo = None
                self.walk_hi = None
                self.walk_k = None
                continue

            self._submit_interval(
                "walk",
                int(self.walk_lo),
                int(self.walk_k),
                now,
                k=int(self.walk_k),
            )
            return

    def start(self, start_time):
        self._schedule_next(start_time)

    def on_interval_done(self, payload: dict, finish_time):
        op = payload.get("op")
        lo = int(payload["lo"])
        hi = int(payload["hi"])
        has_defect = bool(payload["has_defect"])

        if not has_defect and op in ("whole", "walk"):
            self._mark_clean_interval(lo, hi, finish_time)

        if op == "whole":
            if not has_defect:
                return
            if lo == hi:
                self._mark_defect_found(lo, finish_time)
                self.mode = "need_whole"
                self._schedule_next(finish_time)
                return
            self.mode = "walk"
            self.walk_lo = lo
            self.walk_hi = hi
            self.walk_k = hi - 1
            self._schedule_next(finish_time)
            return

        if op == "walk":
            k = int(payload["k"])
            if not has_defect:
                culprit = k + 1
                self._mark_defect_found(culprit, finish_time)
                self.mode = "need_whole"
                self.walk_lo = None
                self.walk_hi = None
                self.walk_k = None
                self._schedule_next(finish_time)
                return
            if k <= int(self.walk_lo):
                self._mark_defect_found(int(self.walk_lo), finish_time)
                self.mode = "need_whole"
                self.walk_lo = None
                self.walk_hi = None
                self.walk_k = None
                self._schedule_next(finish_time)
                return
            self.walk_k = k - 1
            self._schedule_next(finish_time)
            return

        raise RuntimeError(f"SWBProcess: unexpected op {op!r}")


class _SWFProcess(_SigGroupBisectProcess):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = "need_whole"
        self.walk_lo = None
        self.walk_hi = None
        self.walk_k = None

    def _schedule_next(self, now):
        while True:
            unknown = [i for i, s in enumerate(self.status) if s == "unknown"]
            if not unknown:
                return

            if self.mode == "need_whole":
                lo = unknown[0]
                hi = unknown[-1]
                self._submit_interval("whole", lo, hi, now)
                return

            if self.mode != "walk":
                raise RuntimeError(f"SWFProcess: unknown mode {self.mode!r}")

            if self.walk_lo is None or self.walk_hi is None or self.walk_k is None:
                raise RuntimeError("SWFProcess: walk mode without bounds")

            if self.walk_lo == self.walk_hi:
                self._mark_defect_found(int(self.walk_lo), now)
                self.mode = "need_whole"
                self.walk_lo = None
                self.walk_hi = None
                self.walk_k = None
                continue

            self._submit_interval(
                "walk",
                int(self.walk_lo),
                int(self.walk_k),
                now,
                k=int(self.walk_k),
            )
            return

    def start(self, start_time):
        self._schedule_next(start_time)

    def on_interval_done(self, payload: dict, finish_time):
        op = payload.get("op")
        lo = int(payload["lo"])
        hi = int(payload["hi"])
        has_defect = bool(payload["has_defect"])

        if not has_defect and op in ("whole", "walk"):
            self._mark_clean_interval(lo, hi, finish_time)

        if op == "whole":
            if not has_defect:
                return
            if lo == hi:
                self._mark_defect_found(lo, finish_time)
                self.mode = "need_whole"
                self._schedule_next(finish_time)
                return
            self.mode = "walk"
            self.walk_lo = lo
            self.walk_hi = hi
            self.walk_k = lo
            self._schedule_next(finish_time)
            return

        if op == "walk":
            k = int(payload["k"])
            if has_defect:
                self._mark_defect_found(k, finish_time)
                self.mode = "need_whole"
                self.walk_lo = None
                self.walk_hi = None
                self.walk_k = None
                self._schedule_next(finish_time)
                return
            if k >= int(self.walk_hi):
                # Should not happen if the whole interval was known to fail, but guard anyway.
                self._mark_defect_found(int(self.walk_hi), finish_time)
                self.mode = "need_whole"
                self.walk_lo = None
                self.walk_hi = None
                self.walk_k = None
                self._schedule_next(finish_time)
                return
            self.walk_k = k + 1
            self._schedule_next(finish_time)
            return

        raise RuntimeError(f"SWFProcess: unexpected op {op!r}")


class _PARProcess(_SigGroupBisectProcess):
    def start(self, start_time):
        n = self.n
        if n <= 0:
            return

        if n == 1:
            self._submit_interval("par_single", 0, 0, start_time, idx=0)
            return

        requested = start_time
        finish_times = []
        for idx in range(0, n - 1):
            self._submit_interval("par_commit", idx, idx, requested, idx=idx)
            # Compute the finish time for the suite run we just submitted by looking
            # at the worker pools; it will be delivered via event processing, but we
            # also need the max finish time for the last-commit feedback event.
            # We cannot cheaply recover it here, so we compute it again when we see
            # the events (tracked in `self._par_max_finish`).
        # The last commit's feedback time is when the last intermediate suite finishes.
        # We implement that by tracking a running max over par_commit completions and
        # firing the last-commit update when we've seen all of them.
        self._par_remaining = n - 1
        self._par_last_commit_idx = n - 1
        self._par_max_finish = requested

    def on_interval_done(self, payload: dict, finish_time):
        op = payload.get("op")
        lo = int(payload["lo"])
        hi = int(payload["hi"])
        has_defect = bool(payload["has_defect"])

        # All PAR operations are singletons (lo==hi) except the last-commit
        # synthetic event.
        if op == "par_single":
            idx = int(payload["idx"])
            _record_feedback_if_first(self.metrics, self.batch[idx], finish_time)
            if idx in self.defect_indices and has_defect:
                _record_regressor_found_if_first(self.metrics, self.batch[idx], finish_time)
            return

        if op == "par_commit":
            idx = int(payload["idx"])
            _record_feedback_if_first(self.metrics, self.batch[idx], finish_time)
            if idx in self.defect_indices and has_defect:
                _record_regressor_found_if_first(self.metrics, self.batch[idx], finish_time)

            # Track max finish time to emit last-commit feedback once all
            # intermediate suites have completed.
            self._par_max_finish = max(self._par_max_finish, finish_time)
            self._par_remaining -= 1
            if self._par_remaining == 0:
                last_idx = int(self._par_last_commit_idx)
                last_time = self._par_max_finish
                _record_feedback_if_first(self.metrics, self.batch[last_idx], last_time)
                if last_idx in self.defect_indices:
                    _record_regressor_found_if_first(self.metrics, self.batch[last_idx], last_time)
            return

        raise RuntimeError(f"PARProcess: unexpected op {op!r} (interval [{lo}, {hi}])")


class _TOBSubsearch:
    """
    TOB-style subroutine restricted to a bounded index range.

    Used by TKRB when it falls back to a TOB search over a prefix, and also for
    TKRB's final fallback over remaining unknowns.
    """

    def __init__(self, parent, lo_bound: int, hi_bound: int):
        self.parent = parent
        self.lo_bound = int(lo_bound)
        self.hi_bound = int(hi_bound)
        self.mode = "need_whole"
        self.left = None
        self.right = None

    def _unknown_in_bounds(self):
        p = self.parent
        return [
            i
            for i in range(self.lo_bound, self.hi_bound + 1)
            if p.status[i] == "unknown"
        ]

    def schedule_next(self, now):
        p = self.parent
        while True:
            unknown = self._unknown_in_bounds()
            if not unknown:
                p._on_subsearch_complete(now)
                return

            if self.mode == "need_whole":
                lo = unknown[0]
                hi = unknown[-1]
                p._submit_interval("tkrb_sub_whole", lo, hi, now, sub_lo=lo, sub_hi=hi)
                return

            if self.mode != "binary":
                raise RuntimeError(f"TOBSubsearch: unknown mode {self.mode!r}")

            if self.left is None or self.right is None:
                raise RuntimeError("TOBSubsearch: binary mode without bounds")

            if self.left >= self.right:
                p._mark_defect_found(int(self.left), now)
                self.mode = "need_whole"
                self.left = None
                self.right = None
                continue

            mid = (self.left + self.right) // 2
            p._submit_interval("tkrb_sub_left", int(self.left), int(mid), now, mid=int(mid))
            return

    def on_interval_done(self, payload: dict, finish_time):
        p = self.parent
        op = payload.get("op")
        lo = int(payload["lo"])
        hi = int(payload["hi"])
        has_defect = bool(payload["has_defect"])

        if not has_defect and op in ("tkrb_sub_whole", "tkrb_sub_left"):
            p._mark_clean_interval(lo, hi, finish_time)

        if op == "tkrb_sub_whole":
            if not has_defect:
                p._on_subsearch_complete(finish_time)
                return
            self.mode = "binary"
            self.left = lo
            self.right = hi
            self.schedule_next(finish_time)
            return

        if op == "tkrb_sub_left":
            mid = int(payload["mid"])
            if has_defect:
                self.right = mid
            else:
                self.left = mid + 1
            self.schedule_next(finish_time)
            return

        raise RuntimeError(f"TOBSubsearch: unexpected op {op!r}")


class _TKRBProcess(_SigGroupBisectProcess):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase = "whole"
        self._top_indices = []
        self._probe_pos = 0
        self._probe_idx = None

        self._probe_results = {}
        self._probes_remaining = 0
        self._probe_barrier_time = None

        self._subsearch = None
        self._resume_phase = None

    def _compute_top_indices(self):
        risks = []
        for i, c in enumerate(self.batch):
            if "risk" not in c or c["risk"] is None:
                raise ValueError(
                    f"Missing or None 'risk' value in batch for TKRB at index {i}: {c!r}"
                )
            risks.append((i, float(c["risk"])))
        risks.sort(key=lambda t: t[1], reverse=True)
        k = int(getattr(bisection_mod, "TKRB_TOP_K", 1))
        return [i for (i, _) in risks[:k]]

    def _start_subsearch(self, lo_bound: int, hi_bound: int, now, resume_phase: str):
        self._subsearch = _TOBSubsearch(self, lo_bound, hi_bound)
        self._resume_phase = str(resume_phase)
        self._subsearch.schedule_next(now)

    def _on_subsearch_complete(self, now):
        self._subsearch = None
        resume = self._resume_phase
        self._resume_phase = None
        if resume == "probe_seq":
            self._schedule_next_probe_seq(now)
        elif resume == "probe_par_post":
            self._schedule_next_probe_par_post(now)
        elif resume == "fallback":
            # Fallback TOB finished; nothing else to do.
            return
        else:
            raise RuntimeError(f"TKRBProcess: unknown resume phase {resume!r}")

    def start(self, start_time):
        self._submit_interval("tkrb_whole", 0, self.n - 1, start_time)

    def _start_probing(self, now):
        self._top_indices = self._compute_top_indices()
        if not self._top_indices:
            self._start_subsearch(0, self.n - 1, now, resume_phase="fallback")
            return

        k = int(getattr(bisection_mod, "TKRB_TOP_K", 1))
        if k <= 1 or len(self._top_indices) <= 1:
            self.phase = "probe_seq"
            self._probe_pos = 0
            self._schedule_next_probe_seq(now)
            return

        # Parallel probes with barrier
        self.phase = "probe_par"
        self._probe_results = {}
        self._probes_remaining = 0
        self._probe_barrier_time = now

        for idx in self._top_indices:
            if self.status[idx] != "unknown":
                continue
            curr_fails = self._interval_has_defect(0, idx)
            self._probe_results[int(idx)] = bool(curr_fails)
            self._probes_remaining += 1
            self._submit_interval("tkrb_probe_par", 0, idx, now, probe_idx=int(idx))

        if self._probes_remaining == 0:
            # Nothing to probe; go straight to fallback.
            self.phase = "fallback"
            self._start_subsearch(0, self.n - 1, now, resume_phase="fallback")

    def _schedule_next_probe_seq(self, now):
        while self._probe_pos < len(self._top_indices):
            idx = int(self._top_indices[self._probe_pos])
            if self.status[idx] != "unknown":
                self._probe_pos += 1
                continue
            self._probe_idx = idx
            self._submit_interval("tkrb_probe_curr", 0, idx, now, probe_idx=idx)
            return

        # Finished probing; fall back to TOB over remaining unknowns.
        self.phase = "fallback"
        self._start_subsearch(0, self.n - 1, now, resume_phase="fallback")

    def _schedule_next_probe_par_post(self, now):
        while self._probe_pos < len(self._top_indices):
            idx = int(self._top_indices[self._probe_pos])
            self._probe_pos += 1
            if self.status[idx] != "unknown":
                continue

            curr_fails = bool(self._probe_results.get(idx, False))
            if not curr_fails:
                continue

            if idx == 0:
                self._mark_defect_found(0, now)
                continue

            self._probe_idx = idx
            self._submit_interval("tkrb_probe_prev_post", 0, idx - 1, now, probe_idx=idx)
            return

        self.phase = "fallback"
        self._start_subsearch(0, self.n - 1, now, resume_phase="fallback")

    def on_interval_done(self, payload: dict, finish_time):
        op = payload.get("op")
        lo = int(payload["lo"])
        hi = int(payload["hi"])
        has_defect = bool(payload["has_defect"])

        # Delegate to TOB subsearch when active.
        if self._subsearch is not None and op in ("tkrb_sub_whole", "tkrb_sub_left"):
            self._subsearch.on_interval_done(payload, finish_time)
            return

        # Most interval tests mark clean immediately when they pass.
        if not has_defect and op not in ("tkrb_probe_par",):
            self._mark_clean_interval(lo, hi, finish_time)

        if op == "tkrb_whole":
            if not has_defect:
                return
            self._start_probing(finish_time)
            return

        if op == "tkrb_probe_curr":
            idx = int(payload["probe_idx"])
            if not has_defect:
                self._probe_pos += 1
                self._schedule_next_probe_seq(finish_time)
                return
            # curr_fails
            if idx == 0:
                self._mark_defect_found(0, finish_time)
                self._probe_pos += 1
                self._schedule_next_probe_seq(finish_time)
                return
            self._submit_interval("tkrb_probe_prev", 0, idx - 1, finish_time, probe_idx=idx)
            return

        if op == "tkrb_probe_prev":
            idx = int(payload["probe_idx"])
            if not has_defect:
                self._mark_defect_found(idx, finish_time)
                self._probe_pos += 1
                self._schedule_next_probe_seq(finish_time)
                return
            # prev_fails -> TOB on prefix [0..idx-1]
            self._probe_pos += 1
            self._start_subsearch(0, idx - 1, finish_time, resume_phase="probe_seq")
            return

        if op == "tkrb_probe_par":
            # Barrier-style probes: do not react until all probes complete.
            self._probe_barrier_time = (
                finish_time
                if self._probe_barrier_time is None
                else max(self._probe_barrier_time, finish_time)
            )
            self._probes_remaining -= 1
            if self._probes_remaining > 0:
                return

            barrier_time = self._probe_barrier_time
            if barrier_time is None:
                barrier_time = finish_time

            # Apply clean intervals for probes that were clean at the barrier time.
            for idx, curr_fails in self._probe_results.items():
                if not curr_fails:
                    self._mark_clean_interval(0, int(idx), barrier_time)

            # Process the remaining post-barrier steps sequentially.
            self.phase = "probe_par_post"
            self._probe_pos = 0
            self._schedule_next_probe_par_post(barrier_time)
            return

        if op == "tkrb_probe_prev_post":
            idx = int(payload["probe_idx"])
            if not has_defect:
                self._mark_defect_found(idx, finish_time)
                self._schedule_next_probe_par_post(finish_time)
                return
            # prev_fails -> TOB on prefix [0..idx-1], then resume post-probe.
            self._start_subsearch(0, idx - 1, finish_time, resume_phase="probe_par_post")
            return

        raise RuntimeError(f"TKRBProcess: unexpected op {op!r}")


_CANONICAL_BISECTION_IDS = ("TOB", "PAR", "RWAB", "RWAB-LS", "TKRB", "SWB", "SWF")


def _bisection_strategy_key(bisect_fn) -> str:
    """
    Normalize the user-provided bisection selector.

    The simulator accepts either:
      - a canonical bisection id string (preferred): "TOB", "PAR", "RWAB", "RWAB-LS", ...
      - (legacy) a bisection function object, in which case we key off `__name__`.
    """
    if isinstance(bisect_fn, str):
        return bisect_fn.strip()
    return getattr(bisect_fn, "__name__", str(bisect_fn))


_SIG_GROUP_BISECT_PROCESS_BY_ID = {
    # Canonical ids used by `simulation.py`.
    "TOB": _TOBProcess,
    "PAR": _PARProcess,
    "RWAB": _RWABProcess,
    "RWAB-LS": _RWABLSProcess,
    "TKRB": _TKRBProcess,
    "SWB": _SWBProcess,
    "SWF": _SWFProcess,
    # Backward-compat: older code passed bisection function objects and we keyed
    # off their `__name__` from `bisection_strats.py`.
    "time_ordered_bisect": _TOBProcess,
    "exhaustive_parallel": _PARProcess,
    "risk_weighted_adaptive_bisect": _RWABProcess,
    "risk_weighted_adaptive_bisect_log_survival": _RWABLSProcess,
    "topk_risk_first_bisect": _TKRBProcess,
    "sequential_walk_backward_bisect": _SWBProcess,
    "sequential_walk_forward_bisect": _SWFProcess,
}


def _resolve_sig_group_bisect_process_cls(bisect_fn):
    key = _bisection_strategy_key(bisect_fn)
    proc_cls = _SIG_GROUP_BISECT_PROCESS_BY_ID.get(key)
    if proc_cls is None:
        proc_cls = _SIG_GROUP_BISECT_PROCESS_BY_ID.get(key.upper())
    if proc_cls is None:
        raise ValueError(
            "Unsupported bisect strategy for streaming per-signature bisection: "
            f"{bisect_fn!r} (key={key!r}). Use one of: {', '.join(_CANONICAL_BISECTION_IDS)}."
        )
    return proc_cls


def _run_streaming_suite_and_bisect_per_sig_group(
    commits,
    suite_durations,
    batch_start_idx: int,
    batch_end_idx: int,
    suite_requested_start_time,
    bisect_fn,
    executor: TestExecutor,
    metrics: _StreamingMetrics,
    last_seen_idx_by_sig: dict | None = None,
):
    """
    Run an initial suite (full or subset) for a batch, and trigger per-signature-group
    bisections at the completion time of each failing signature-group job.

    This models "streaming detection": we don't wait for the entire suite to finish
    before starting bisection for signature-groups that fail early.

    When `last_seen_idx_by_sig` is provided (used by the "-s" batching variants),
    a signature-group may be exercised in a later batch than the one where a
    regressor landed; in that case, detection/bisection ranges are anchored on the
    last index where that signature-group was previously exercised.
    """
    if batch_end_idx < batch_start_idx:
        return
    if not suite_durations:
        return

    # Map: sig_group_id -> set[global_commit_idx] of regressors that would cause
    # this suite's signature-group job to fail.
    suite_sig_ids = set()
    for gid, _dur in suite_durations:
        try:
            suite_sig_ids.add(int(gid))
        except (TypeError, ValueError):
            continue

    # In "-s" modes, batches run a per-batch subset suite, and a signature-group
    # may not be exercised in the same batch where a regressor lands. When that
    # signature-group is exercised later, we still need to detect and bisect
    # regressors that occurred since the last time that group was tested. The
    # `last_seen_idx_by_sig` map tracks those last-tested indices.
    scan_start_idx = int(batch_start_idx)
    if last_seen_idx_by_sig is not None and suite_sig_ids:
        # We scan from the earliest "unknown" index among the signature-groups in
        # this suite. This preserves correctness when groups have been absent for
        # multiple batches (i.e., their last_seen is < batch_start_idx).
        scan_start_idx = min(
            int(last_seen_idx_by_sig.get(gid, -1)) + 1 for gid in suite_sig_ids
        )
        scan_start_idx = max(0, min(int(scan_start_idx), int(batch_start_idx)))

    regressors_by_sig = {}
    for idx in range(scan_start_idx, batch_end_idx + 1):
        c = commits[idx]
        if not c.get("true_label"):
            continue
        # Avoid re-counting regressors already found by earlier bisections in
        # this simulation run.
        cid = c.get("commit_id")
        if cid is not None and cid in metrics.found_regressors:
            continue
        fail_sigs = get_failing_signature_groups_for_revision(c["commit_id"])
        if not fail_sigs:
            # Fallback: when failing signature-groups cannot be derived for a regressor,
            # treat it as failing an "unknown" synthetic signature-group. We model its
            # detection time as the suite makespan (since we can't attribute it to a
            # specific job completion).
            regressors_by_sig.setdefault(None, set()).add(idx)
            continue
        for sig in fail_sigs:
            try:
                gid = int(sig)
            except (TypeError, ValueError):
                continue
            if suite_sig_ids and gid not in suite_sig_ids:
                continue
            regressors_by_sig.setdefault(gid, set()).add(idx)

    # Schedule the suite and capture per-job finish times.
    job_finishes, _suite_finish = schedule_test_suite_jobs(
        executor, suite_requested_start_time, suite_durations
    )
    metrics.total_tests_run += len(suite_durations)

    finish_by_gid = {}
    for gid, t in job_finishes:
        # Preserve None (fallback synthetic group) as a valid key.
        if gid is None:
            finish_by_gid[None] = t
            continue
        try:
            finish_by_gid[int(gid)] = t
        except (TypeError, ValueError):
            continue

    # Ensure the synthetic fallback group has a detection time if present.
    if None in regressors_by_sig and None not in finish_by_gid:
        finish_by_gid[None] = _suite_finish

    # For "-s" modes we need to track per-sig last-seen indices (baseline for bisection).
    prev_last_seen = {}
    if last_seen_idx_by_sig is not None:
        for gid in finish_by_gid.keys():
            prev_last_seen[gid] = int(last_seen_idx_by_sig.get(gid, -1))
            last_seen_idx_by_sig[gid] = int(batch_end_idx)

    failing_gids = []
    for gid in regressors_by_sig.keys():
        if gid not in finish_by_gid:
            continue
        if last_seen_idx_by_sig is None:
            start_idx = int(batch_start_idx)
        else:
            start_idx = int(prev_last_seen.get(gid, -1)) + 1
        if any(start_idx <= int(i) <= int(batch_end_idx) for i in regressors_by_sig.get(gid, set())):
            failing_gids.append(gid)

    if not failing_gids:
        # No failing signature-groups observed in this suite => nothing to bisect.
        return

    events = []
    seq = 0

    def push_event(t, kind: str, payload: dict):
        nonlocal seq
        heapq.heappush(events, (t, seq, kind, payload))
        seq += 1

    # Seed events with per-signature-group failure detections.
    def _gid_sort_key(g):
        # Keep ordering stable even when `None` is present.
        if g is None:
            return (1, 0)
        return (0, int(g))

    for gid in sorted(failing_gids, key=_gid_sort_key):
        push_event(
            finish_by_gid[gid],
            "sig_fail_detected",
            {
                "sig_group_id": gid,
                "prev_last_seen": int(prev_last_seen.get(gid, batch_start_idx - 1)),
            },
        )

    # Run event loop: failing signature detection + interleaved bisection steps.
    while events:
        t, _s, kind, payload = heapq.heappop(events)

        if kind == "sig_fail_detected":
            gid = payload["sig_group_id"]
            prev_idx = int(payload["prev_last_seen"])
            start_idx = prev_idx + 1
            # In non "-s" modes we don't maintain `last_seen_idx_by_sig`; limit
            # bisection to the current batch.
            if last_seen_idx_by_sig is None:
                start_idx = int(batch_start_idx)
            start_idx = max(0, min(int(start_idx), int(batch_end_idx)))

            defect_globals = regressors_by_sig.get(gid, set())
            defect_locals = set()
            for gidx in defect_globals:
                if start_idx <= int(gidx) <= int(batch_end_idx):
                    defect_locals.add(int(gidx) - start_idx)

            batch_slice = commits[start_idx : batch_end_idx + 1]
            proc_cls = _resolve_sig_group_bisect_process_cls(bisect_fn)

            proc = proc_cls(batch_slice, defect_locals, gid, executor, metrics, push_event)
            proc.start(t)
            continue

        if kind == "bisect_interval_done":
            proc = payload["proc"]
            proc.on_interval_done(payload, t)
            continue

        raise RuntimeError(f"Unknown event kind {kind!r}")


def _simulate_signature_union_driver(commits, bisect_fn, batches, num_workers):
    """
    Shared implementation for the "-s" batching variants.

    Each "batch test" executes only the union of perf signature-groups that Mozilla
    actually ran for the revisions in that batch (from
    perf_jobs_per_revision_details_rectified.jsonl, interpreted as
    signature-groups). A regressor is detected by a batch run only if that
    union includes at least one of the regressor's failing signature-groups
    (derived from alert_summary_fail_perf_sigs.csv via sig_groups.jsonl).

    When a batch run detects at least one regressor, we trigger a bisection
    between the last clean revision (per failing signature-group) and the end of the
    batch, using `bisect_fn`. The bisection itself uses failing signature-groups
    (derived from failing signatures via bisection_strats), and we treat
    the batch run as the "root"
    failure so bisection is invoked with is_batch_root=False.
    """
    metrics = _StreamingMetrics(0, [], {}, set())

    if not commits:
        return build_results(
            metrics.total_tests_run,
            metrics.culprit_times,
            metrics.feedback_times,
            0.0,
            num_regressors_total=0,
            num_regressors_found=0,
        )

    num_regressors_total = sum(1 for c in commits if c.get("true_label"))

    executor = TestExecutor(num_workers)

    last_seen_batch_end_idx_by_sig = {}

    for batch_start_idx, batch_end_idx, batch_end_time in batches:
        if batch_end_idx < batch_start_idx:
            continue

        batch_commits = commits[batch_start_idx : batch_end_idx + 1]
        suite_sig_ids = _union_tested_signature_group_ids_for_commits(batch_commits)
        suite_sig_ids_sorted = sorted(int(s) for s in suite_sig_ids)
        durations = get_signature_durations_for_ids(suite_sig_ids_sorted) if suite_sig_ids_sorted else []

        _run_streaming_suite_and_bisect_per_sig_group(
            commits,
            durations,
            batch_start_idx,
            batch_end_idx,
            batch_end_time,
            bisect_fn,
            executor,
            metrics,
            last_seen_idx_by_sig=last_seen_batch_end_idx_by_sig,
        )

    return build_results(
        metrics.total_tests_run,
        metrics.culprit_times,
        metrics.feedback_times,
        executor.total_cpu_minutes,
        num_regressors_total=num_regressors_total,
        num_regressors_found=len(metrics.found_regressors),
    )


def simulate_twsb_with_bisect(commits, bisect_fn, _unused_param, num_workers):
    """
    Time-Window Subset Batching (TWSB).

    For each revision, we run only the perf signature-groups that Mozilla actually
    executed for that revision (from perf_jobs_per_revision_details_rectified.jsonl,
    interpreted as signature-groups).
    When this per-revision subset first exercises any failing signature-group for a
    regressor that lies between the last known clean revision and the current
    revision, we trigger a bisection between that last clean revision and the
    current revision, using the provided bisect_fn.

    Each bisection step uses the same failing signature-group suite as other
    bisection strategies (via bisection_strats).
    """
    logger.info(
        "simulate_twsb_with_bisect: %d commits, bisect_fn=%s",
        len(commits),
        getattr(bisect_fn, "__name__", str(bisect_fn)),
    )

    metrics = _StreamingMetrics(0, [], {}, set())
    num_regressors_total = sum(1 for c in commits if c.get("true_label"))

    if not commits:
        logger.info("simulate_twsb_with_bisect: no commits; returning empty results")
        return build_results(
            metrics.total_tests_run,
            metrics.culprit_times,
            metrics.feedback_times,
            0.0,
            num_regressors_total=num_regressors_total,
            num_regressors_found=0,
        )

    executor = TestExecutor(num_workers)

    # Precompute, for each commit, the set of perf signature-groups that actually ran.
    tested_sig_group_sets = []
    for c in commits:
        rev = c["commit_id"]
        tested_sig_group_sets.append(set(get_tested_signatures_for_revision(rev)))

    # Track true regressors by signature-group (global indices).
    regressors_by_sig = {}
    # Track last time each signature-group was exercised (revision index).
    last_seen_index_by_sig = {}

    proc_cls = _resolve_sig_group_bisect_process_cls(bisect_fn)

    for idx, c in enumerate(commits):
        # As regressors land, add them to the mapping used for per-sig bisection.
        if c.get("true_label"):
            fail_sigs = get_failing_signature_groups_for_revision(c["commit_id"])
            if not fail_sigs:
                regressors_by_sig.setdefault(None, set()).add(idx)
            else:
                for sig in fail_sigs:
                    try:
                        gid = int(sig)
                    except (TypeError, ValueError):
                        continue
                    regressors_by_sig.setdefault(gid, set()).add(idx)

        submit_time = c["ts"]
        suite_sig_ids_raw = tested_sig_group_sets[idx]
        suite_sig_ids = []
        for s in suite_sig_ids_raw:
            try:
                suite_sig_ids.append(int(s))
            except (TypeError, ValueError):
                continue

        if not suite_sig_ids:
            continue

        suite_sig_ids_sorted = sorted(set(suite_sig_ids))
        durations = get_signature_durations_for_ids(suite_sig_ids_sorted)
        if not durations:
            continue

        job_finishes, _suite_finish = schedule_test_suite_jobs(
            executor, submit_time, durations
        )
        metrics.total_tests_run += len(durations)

        finish_by_gid = {}
        for gid, t in job_finishes:
            if gid is None:
                finish_by_gid[None] = t
                continue
            try:
                finish_by_gid[int(gid)] = t
            except (TypeError, ValueError):
                continue

        # Ensure the synthetic fallback group has a detection time if present.
        if None in regressors_by_sig and None not in finish_by_gid:
            finish_by_gid[None] = _suite_finish

        # Update last-seen indices for signature-groups tested at this revision,
        # but keep the previous values for bisection start ranges.
        prev_last_seen = {}
        for gid in finish_by_gid.keys():
            prev_last_seen[gid] = int(last_seen_index_by_sig.get(gid, -1))
            last_seen_index_by_sig[gid] = idx

        # Determine which signature-groups fail *newly* in this revision's subset
        # (i.e., regressors that occurred after the last time the group was tested).
        failing_gids = []
        for gid in finish_by_gid.keys():
            start_idx = int(prev_last_seen.get(gid, -1)) + 1
            if any(start_idx <= int(r) <= idx for r in regressors_by_sig.get(gid, set())):
                failing_gids.append(gid)

        if not failing_gids:
            continue

        events = []
        seq = 0

        def push_event(t, kind: str, payload: dict):
            nonlocal seq
            heapq.heappush(events, (t, seq, kind, payload))
            seq += 1

        def _gid_sort_key(g):
            if g is None:
                return (1, 0)
            return (0, int(g))

        for gid in sorted(set(failing_gids), key=_gid_sort_key):
            push_event(
                finish_by_gid[gid],
                "sig_fail_detected",
                {
                    "sig_group_id": gid,
                    "prev_last_seen": int(prev_last_seen.get(gid, -1)),
                },
            )

        while events:
            t, _s, kind, payload = heapq.heappop(events)

            if kind == "sig_fail_detected":
                gid = payload["sig_group_id"]
                prev_idx = int(payload["prev_last_seen"])
                start_idx = max(0, min(prev_idx + 1, idx))

                defect_globals = regressors_by_sig.get(gid, set())
                defect_locals = set()
                for gidx in defect_globals:
                    if start_idx <= int(gidx) <= idx:
                        defect_locals.add(int(gidx) - start_idx)

                if not defect_locals:
                    continue

                batch_slice = commits[start_idx : idx + 1]
                proc = proc_cls(batch_slice, defect_locals, gid, executor, metrics, push_event)
                proc.start(t)
                continue

            if kind == "bisect_interval_done":
                proc = payload["proc"]
                proc.on_interval_done(payload, t)
                continue

            raise RuntimeError(f"Unknown event kind {kind!r}")

    logger.info(
        "simulate_twsb_with_bisect: finished; total_tests_run=%d",
        metrics.total_tests_run,
    )
    return build_results(
        metrics.total_tests_run,
        metrics.culprit_times,
        metrics.feedback_times,
        executor.total_cpu_minutes,
        num_regressors_total=num_regressors_total,
        num_regressors_found=len(metrics.found_regressors),
    )


def simulate_twb_with_bisect(commits, bisect_fn, batch_hours, num_workers):
    """
    Time-Window Batching (TWB).

    Groups commits into fixed wall-clock windows of length `batch_hours` and
    flushes each window as a batch. The batch test is modeled as occurring at
    the window end time; `bisect_fn` is invoked on the commits in that window.
    """
    logger.info(
        "simulate_twb_with_bisect: %d commits, batch_hours=%.2f, bisect_fn=%s",
        len(commits),
        batch_hours,
        getattr(bisect_fn, "__name__", str(bisect_fn)),
    )
    num_regressors_total = sum(1 for c in commits if c.get("true_label"))

    # One central executor per simulation run
    executor = TestExecutor(num_workers)
    metrics = _StreamingMetrics(0, [], {}, set())

    if not commits:
        logger.info("simulate_twb_with_bisect: no commits; returning empty results")
        return build_results(
            metrics.total_tests_run,
            metrics.culprit_times,
            metrics.feedback_times,
            executor.total_cpu_minutes,
            num_regressors_total=num_regressors_total,
            num_regressors_found=0,
        )

    # Construct the same window flush boundaries as TWB-s and process each batch.
    batches = []
    batch_start_idx = 0
    batch_start_time = commits[0]["ts"]
    batch_end_time = batch_start_time + timedelta(hours=batch_hours)

    for idx, c in enumerate(commits):
        if c["ts"] < batch_end_time:
            continue
        batch_end_idx = idx - 1
        if batch_end_idx >= batch_start_idx:
            batches.append((batch_start_idx, batch_end_idx, batch_end_time))
        batch_start_idx = idx
        batch_start_time = c["ts"]
        batch_end_time = batch_start_time + timedelta(hours=batch_hours)

    if batch_start_idx < len(commits):
        batches.append((batch_start_idx, len(commits) - 1, batch_end_time))

    for batch_start_idx, batch_end_idx, flush_time in batches:
        durations = get_batch_signature_durations()
        _run_streaming_suite_and_bisect_per_sig_group(
            commits,
            durations,
            batch_start_idx,
            batch_end_idx,
            flush_time,
            bisect_fn,
            executor,
            metrics,
            last_seen_idx_by_sig=None,
        )

    logger.info(
        "simulate_twb_with_bisect: finished; total_tests_run=%d",
        metrics.total_tests_run,
    )
    return build_results(
        metrics.total_tests_run,
        metrics.culprit_times,
        metrics.feedback_times,
        executor.total_cpu_minutes,
        num_regressors_total=num_regressors_total,
        num_regressors_found=len(metrics.found_regressors),
    )


def simulate_twb_s_with_bisect(commits, bisect_fn, batch_hours, num_workers):
    """
    Time-Window Batching with subset suite detection (TWB-s).

    Same windowing as TWB, but the initial batch test uses a *subset suite*:
    the union of perf signature-groups Mozilla actually ran for revisions in
    that window. A regressor can be detected only if the subset suite
    intersects its failing signature-groups.

    Once a batch run detects regressors, bisection is invoked with
    `is_batch_root=False` because the batch suite has already run.
    """
    if not commits:
        return build_results(0, [], {}, 0.0, num_regressors_total=0, num_regressors_found=0)

    batches = []
    batch_start_idx = 0
    batch_start_time = commits[0]["ts"]
    batch_end_time = batch_start_time + timedelta(hours=batch_hours)

    for idx, c in enumerate(commits):
        if c["ts"] < batch_end_time:
            continue

        batch_end_idx = idx - 1
        if batch_end_idx >= batch_start_idx:
            batches.append((batch_start_idx, batch_end_idx, batch_end_time))

        batch_start_idx = idx
        batch_start_time = c["ts"]
        batch_end_time = batch_start_time + timedelta(hours=batch_hours)

    if batch_start_idx < len(commits):
        batches.append((batch_start_idx, len(commits) - 1, batch_end_time))

    return _simulate_signature_union_driver(commits, bisect_fn, batches, num_workers)


def simulate_fsb_with_bisect(commits, bisect_fn, batch_size, num_workers):
    """
    Fixed-Size Batching (FSB).

    Groups commits into contiguous batches of size `batch_size` (except a final
    partial batch) and flushes each batch at the timestamp of its last commit.
    """
    logger.info(
        "simulate_fsb_with_bisect: %d commits, batch_size=%d, bisect_fn=%s",
        len(commits),
        batch_size,
        getattr(bisect_fn, "__name__", str(bisect_fn)),
    )
    num_regressors_total = sum(1 for c in commits if c.get("true_label"))

    executor = TestExecutor(num_workers)
    metrics = _StreamingMetrics(0, [], {}, set())

    if not commits:
        logger.info("simulate_fsb_with_bisect: no commits; returning empty results")
        return build_results(
            metrics.total_tests_run,
            metrics.culprit_times,
            metrics.feedback_times,
            executor.total_cpu_minutes,
            num_regressors_total=num_regressors_total,
            num_regressors_found=0,
        )

    # Build contiguous fixed-size batches (same boundaries as FSB-s).
    batches = []
    start = 0
    for end in range(len(commits)):
        if (end - start + 1) >= batch_size:
            batches.append((start, end, commits[end]["ts"]))
            start = end + 1

    if start < len(commits):
        batches.append((start, len(commits) - 1, commits[-1]["ts"]))

    for batch_start_idx, batch_end_idx, flush_time in batches:
        durations = get_batch_signature_durations()
        _run_streaming_suite_and_bisect_per_sig_group(
            commits,
            durations,
            batch_start_idx,
            batch_end_idx,
            flush_time,
            bisect_fn,
            executor,
            metrics,
            last_seen_idx_by_sig=None,
        )

    logger.info(
        "simulate_fsb_with_bisect: finished; total_tests_run=%d",
        metrics.total_tests_run,
    )
    return build_results(
        metrics.total_tests_run,
        metrics.culprit_times,
        metrics.feedback_times,
        executor.total_cpu_minutes,
        num_regressors_total=num_regressors_total,
        num_regressors_found=len(metrics.found_regressors),
    )


def simulate_fsb_s_with_bisect(commits, bisect_fn, batch_size, num_workers):
    """
    Fixed-Size Batching with subset suite detection (FSB-s).

    Same grouping as FSB, but each batch's initial run uses the union of
    signature-groups observed within the batch (subset suite), and a regressor
    is detected only if that suite overlaps its failing signature-groups.
    """
    batches = []
    if not commits:
        return build_results(0, [], {}, 0.0, num_regressors_total=0, num_regressors_found=0)

    start = 0
    for end in range(len(commits)):
        if (end - start + 1) >= batch_size:
            batches.append((start, end, commits[end]["ts"]))
            start = end + 1

    if start < len(commits):
        batches.append((start, len(commits) - 1, commits[-1]["ts"]))

    return _simulate_signature_union_driver(commits, bisect_fn, batches, num_workers)


def simulate_rasb_with_bisect(commits, bisect_fn, threshold, num_workers):
    """
    Risk-Adaptive Stream Batching (RASB).

    Accumulates commits into a batch until the estimated probability that the
    batch contains at least one failure reaches `threshold`.

    Probability model:
      - Assume independence between commits.
      - For each commit i, predicted failure probability is `risk_i`.
      - Probability the batch is clean: Π(1 - risk_i)
      - Probability the batch fails: 1 - Π(1 - risk_i)

    Flush happens at the timestamp of the last commit in the batch. `bisect_fn`
    is invoked on the flushed commits.
    """
    logger.info(
        "simulate_rasb_with_bisect: %d commits, threshold=%.4f, bisect_fn=%s",
        len(commits),
        threshold,
        getattr(bisect_fn, "__name__", str(bisect_fn)),
    )
    num_regressors_total = sum(1 for c in commits if c.get("true_label"))

    executor = TestExecutor(num_workers)
    metrics = _StreamingMetrics(0, [], {}, set())

    log_survival = 0.0

    if not commits:
        logger.info("simulate_rasb_with_bisect: no commits; returning empty results")
        return build_results(
            metrics.total_tests_run,
            metrics.culprit_times,
            metrics.feedback_times,
            executor.total_cpu_minutes,
            num_regressors_total=num_regressors_total,
            num_regressors_found=0,
        )

    # Build batches using the same trigger rule as RASB-s, but run the full suite.
    batches = []
    start = 0
    for idx, c in enumerate(commits):
        p = c["risk"]
        log_survival = _accumulate_log_survival(log_survival, p)
        fail_prob = _combined_probability_from_log_survival(log_survival)
        if fail_prob >= threshold:
            batches.append((start, idx, c["ts"]))
            start = idx + 1
            log_survival = 0.0

    if start < len(commits):
        batches.append((start, len(commits) - 1, commits[-1]["ts"]))

    for batch_start_idx, batch_end_idx, flush_time in batches:
        durations = get_batch_signature_durations()
        _run_streaming_suite_and_bisect_per_sig_group(
            commits,
            durations,
            batch_start_idx,
            batch_end_idx,
            flush_time,
            bisect_fn,
            executor,
            metrics,
            last_seen_idx_by_sig=None,
        )

    logger.info(
        "simulate_rasb_with_bisect: finished; total_tests_run=%d", metrics.total_tests_run
    )
    return build_results(
        metrics.total_tests_run,
        metrics.culprit_times,
        metrics.feedback_times,
        executor.total_cpu_minutes,
        num_regressors_total=num_regressors_total,
        num_regressors_found=len(metrics.found_regressors),
    )


def simulate_rasb_s_with_bisect(commits, bisect_fn, threshold, num_workers):
    """
    RASB with subset suite detection (RASB-s).

    Same batching trigger as RASB, but the initial batch run uses a subset
    suite (union of signature-groups observed within the batch). Detection
    depends on overlap between that subset and each regressor's failing
    signature-groups.
    """
    batches = []
    if not commits:
        return build_results(0, [], {}, 0.0, num_regressors_total=0, num_regressors_found=0)

    start = 0
    log_survival = 0.0

    for idx, c in enumerate(commits):
        p = c["risk"]
        log_survival = _accumulate_log_survival(log_survival, p)
        fail_prob = _combined_probability_from_log_survival(log_survival)
        if fail_prob >= threshold:
            batches.append((start, idx, c["ts"]))
            start = idx + 1
            log_survival = 0.0

    if start < len(commits):
        batches.append((start, len(commits) - 1, commits[-1]["ts"]))

    return _simulate_signature_union_driver(commits, bisect_fn, batches, num_workers)


def simulate_rasb_la_with_bisect(commits, bisect_fn, risk_budget, num_workers):
    """
    Risk-Adaptive Stream Batching with linear aggregation (RASB-la).

    Accumulates commits into a batch until the sum of per-commit risk scores
    reaches `risk_budget`:

        risk_sum = Σ risk_i

    Flush happens at the timestamp of the last commit in the batch. `bisect_fn`
    is invoked on the flushed commits.
    """
    logger.info(
        "simulate_rasb_la_with_bisect: %d commits, risk_budget=%.4f, bisect_fn=%s",
        len(commits),
        float(risk_budget),
        getattr(bisect_fn, "__name__", str(bisect_fn)),
    )
    num_regressors_total = sum(1 for c in commits if c.get("true_label"))

    executor = TestExecutor(num_workers)
    metrics = _StreamingMetrics(0, [], {}, set())

    if not commits:
        logger.info("simulate_rasb_la_with_bisect: no commits; returning empty results")
        return build_results(
            metrics.total_tests_run,
            metrics.culprit_times,
            metrics.feedback_times,
            executor.total_cpu_minutes,
            num_regressors_total=num_regressors_total,
            num_regressors_found=0,
        )

    # Build batches using the same trigger rule as RASB-la-s, but run the full suite.
    batches = []
    start = 0
    risk_sum = 0.0

    for idx, c in enumerate(commits):
        if "risk" not in c or c["risk"] is None:
            raise ValueError(
                f"Missing or None 'risk' value for commit at index {idx}: {c!r}"
            )
        risk_sum += float(c["risk"])
        if risk_sum >= risk_budget:
            batches.append((start, idx, c["ts"]))
            start = idx + 1
            risk_sum = 0.0

    if start < len(commits):
        batches.append((start, len(commits) - 1, commits[-1]["ts"]))

    for batch_start_idx, batch_end_idx, flush_time in batches:
        durations = get_batch_signature_durations()
        _run_streaming_suite_and_bisect_per_sig_group(
            commits,
            durations,
            batch_start_idx,
            batch_end_idx,
            flush_time,
            bisect_fn,
            executor,
            metrics,
            last_seen_idx_by_sig=None,
        )

    logger.info(
        "simulate_rasb_la_with_bisect: finished; total_tests_run=%d",
        metrics.total_tests_run,
    )
    return build_results(
        metrics.total_tests_run,
        metrics.culprit_times,
        metrics.feedback_times,
        executor.total_cpu_minutes,
        num_regressors_total=num_regressors_total,
        num_regressors_found=len(metrics.found_regressors),
    )


def simulate_rasb_la_s_with_bisect(commits, bisect_fn, risk_budget, num_workers):
    """
    RASB-la with subset suite detection (RASB-la-s).

    Same batching trigger as RASB-la, but the initial batch run uses a subset
    suite (union of signature-groups observed within the batch). Detection
    depends on overlap between that subset and each regressor's failing
    signature-groups.
    """
    batches = []
    if not commits:
        return build_results(0, [], {}, 0.0, num_regressors_total=0, num_regressors_found=0)

    start = 0
    risk_sum = 0.0

    for idx, c in enumerate(commits):
        risk_sum += float(c["risk"])
        if risk_sum >= risk_budget:
            batches.append((start, idx, c["ts"]))
            start = idx + 1
            risk_sum = 0.0

    if start < len(commits):
        batches.append((start, len(commits) - 1, commits[-1]["ts"]))

    return _simulate_signature_union_driver(commits, bisect_fn, batches, num_workers)


def simulate_rapb_with_bisect(commits, bisect_fn, params, num_workers):
    """
    Risk-Aware Priority Batching (RAPB).

    Parameters
    ----------
    params:
        Tuple `(threshold_T, aging_rate)` where:
          - `threshold_T` is the target batch failure probability threshold.
          - `aging_rate` controls how quickly older commits become more urgent.

    Each commit's risk is "aged" as it waits in the current batch, increasing
    the estimated failure probability over time until the batch is flushed.
    """
    threshold_T, aging_rate = params

    logger.info(
        "simulate_rapb_with_bisect: %d commits, threshold_T=%.4f, aging_rate=%.4f, bisect_fn=%s",
        len(commits),
        threshold_T,
        aging_rate,
        getattr(bisect_fn, "__name__", str(bisect_fn)),
    )
    num_regressors_total = sum(1 for c in commits if c.get("true_label"))

    executor = TestExecutor(num_workers)
    metrics = _StreamingMetrics(0, [], {}, set())

    if not commits:
        logger.info("simulate_rapb_with_bisect: no commits; returning empty results")
        return build_results(
            metrics.total_tests_run,
            metrics.culprit_times,
            metrics.feedback_times,
            executor.total_cpu_minutes,
            num_regressors_total=num_regressors_total,
            num_regressors_found=0,
        )

    # Build batches using the same trigger rule as RAPB-s, but run the full suite.
    batches = []
    start = 0
    entry_times = []
    current_batch_indices = []

    for idx, c in enumerate(commits):
        now_ts = c["ts"]
        current_batch_indices.append(idx)
        entry_times.append(now_ts)

        log_survival = 0.0
        for commit_idx, entered_at in zip(current_batch_indices, entry_times):
            wait_hours = max(0.0, (now_ts - entered_at).total_seconds() / 3600.0)
            base_risk = float(commits[commit_idx]["risk"])
            decay = math.exp(-aging_rate * wait_hours)
            aged_p = 1.0 - (1.0 - base_risk) * decay
            log_survival = _accumulate_log_survival(log_survival, aged_p)

        fail_prob = _combined_probability_from_log_survival(log_survival)
        if fail_prob >= threshold_T:
            batches.append((start, idx, now_ts))
            start = idx + 1
            entry_times = []
            current_batch_indices = []

    if start < len(commits):
        batches.append((start, len(commits) - 1, commits[-1]["ts"]))

    for batch_start_idx, batch_end_idx, flush_time in batches:
        durations = get_batch_signature_durations()
        _run_streaming_suite_and_bisect_per_sig_group(
            commits,
            durations,
            batch_start_idx,
            batch_end_idx,
            flush_time,
            bisect_fn,
            executor,
            metrics,
            last_seen_idx_by_sig=None,
        )

    logger.info(
        "simulate_rapb_with_bisect: finished; total_tests_run=%d",
        metrics.total_tests_run,
    )
    return build_results(
        metrics.total_tests_run,
        metrics.culprit_times,
        metrics.feedback_times,
        executor.total_cpu_minutes,
        num_regressors_total=num_regressors_total,
        num_regressors_found=len(metrics.found_regressors),
    )


def simulate_rapb_s_with_bisect(commits, bisect_fn, params, num_workers):
    """
    RAPB with subset suite detection (RAPB-s).

    Uses the same batching trigger as RAPB, but the batch run uses a subset
    suite (union of signature-groups observed within the batch) and detection
    depends on overlap with failing signature-groups.
    """
    threshold_T, aging_rate = params
    batches = []
    if not commits:
        return build_results(0, [], {}, 0.0, num_regressors_total=0, num_regressors_found=0)

    start = 0
    entry_times = []
    current_batch_indices = []

    for idx, c in enumerate(commits):
        now_ts = c["ts"]
        current_batch_indices.append(idx)
        entry_times.append(now_ts)

        log_survival = 0.0
        for commit_idx, entered_at in zip(current_batch_indices, entry_times):
            wait_hours = max(0.0, (now_ts - entered_at).total_seconds() / 3600.0)
            base_risk = float(commits[commit_idx]["risk"])
            decay = math.exp(-aging_rate * wait_hours)
            aged_p = 1.0 - (1.0 - base_risk) * decay
            log_survival = _accumulate_log_survival(log_survival, aged_p)

        fail_prob = _combined_probability_from_log_survival(log_survival)
        if fail_prob >= threshold_T:
            batches.append((start, idx, now_ts))
            start = idx + 1
            entry_times = []
            current_batch_indices = []

    if start < len(commits):
        batches.append((start, len(commits) - 1, commits[-1]["ts"]))

    return _simulate_signature_union_driver(commits, bisect_fn, batches, num_workers)

def simulate_rapb_la_with_bisect(commits, bisect_fn, params, num_workers):
    """
    Risk-Aware Priority Batching with linear aggregation (RAPB-la).

    Parameters
    ----------
    params:
        Tuple `(risk_budget_T, aging_rate)` where:
          - `risk_budget_T` is the target *sum of aged risks* threshold.
          - `aging_rate` controls how quickly older commits become more urgent.

    Each commit's risk is "aged" as it waits in the current batch, then we
    compute:

        risk_sum = Σ aged_risk_i

    and flush when `risk_sum >= risk_budget_T`.
    """
    risk_budget_T, aging_rate = params

    logger.info(
        "simulate_rapb_la_with_bisect: %d commits, risk_budget_T=%.4f, aging_rate=%.4f, bisect_fn=%s",
        len(commits),
        float(risk_budget_T),
        float(aging_rate),
        getattr(bisect_fn, "__name__", str(bisect_fn)),
    )
    num_regressors_total = sum(1 for c in commits if c.get("true_label"))

    executor = TestExecutor(num_workers)
    metrics = _StreamingMetrics(0, [], {}, set())

    if not commits:
        logger.info("simulate_rapb_la_with_bisect: no commits; returning empty results")
        return build_results(
            metrics.total_tests_run,
            metrics.culprit_times,
            metrics.feedback_times,
            executor.total_cpu_minutes,
            num_regressors_total=num_regressors_total,
            num_regressors_found=0,
        )

    # Build batches using the same trigger rule as RAPB-la-s, but run the full suite.
    batches = []
    start = 0
    entry_times = []
    current_batch_indices = []

    for idx, c in enumerate(commits):
        now_ts = c["ts"]
        current_batch_indices.append(idx)
        entry_times.append(now_ts)

        risk_sum = 0.0
        for commit_idx, entered_at in zip(current_batch_indices, entry_times):
            wait_hours = max(0.0, (now_ts - entered_at).total_seconds() / 3600.0)
            base_risk = float(commits[commit_idx]["risk"])
            decay = math.exp(-aging_rate * wait_hours)
            aged_p = 1.0 - (1.0 - base_risk) * decay
            risk_sum += aged_p

        if risk_sum >= risk_budget_T:
            batches.append((start, idx, now_ts))
            start = idx + 1
            entry_times = []
            current_batch_indices = []

    if start < len(commits):
        batches.append((start, len(commits) - 1, commits[-1]["ts"]))

    for batch_start_idx, batch_end_idx, flush_time in batches:
        durations = get_batch_signature_durations()
        _run_streaming_suite_and_bisect_per_sig_group(
            commits,
            durations,
            batch_start_idx,
            batch_end_idx,
            flush_time,
            bisect_fn,
            executor,
            metrics,
            last_seen_idx_by_sig=None,
        )

    logger.info(
        "simulate_rapb_la_with_bisect: finished; total_tests_run=%d",
        metrics.total_tests_run,
    )
    return build_results(
        metrics.total_tests_run,
        metrics.culprit_times,
        metrics.feedback_times,
        executor.total_cpu_minutes,
        num_regressors_total=num_regressors_total,
        num_regressors_found=len(metrics.found_regressors),
    )


def simulate_rapb_la_s_with_bisect(commits, bisect_fn, params, num_workers):
    """
    RAPB-la with subset suite detection (RAPB-la-s).

    Uses the same batching trigger as RAPB-la, but the batch run uses a subset
    suite (union of signature-groups observed within the batch) and detection
    depends on overlap with failing signature-groups.
    """
    risk_budget_T, aging_rate = params
    batches = []
    if not commits:
        return build_results(0, [], {}, 0.0, num_regressors_total=0, num_regressors_found=0)

    start = 0
    entry_times = []
    current_batch_indices = []

    for idx, c in enumerate(commits):
        now_ts = c["ts"]
        current_batch_indices.append(idx)
        entry_times.append(now_ts)

        risk_sum = 0.0
        for commit_idx, entered_at in zip(current_batch_indices, entry_times):
            wait_hours = max(0.0, (now_ts - entered_at).total_seconds() / 3600.0)
            base_risk = float(commits[commit_idx]["risk"])
            decay = math.exp(-aging_rate * wait_hours)
            aged_p = 1.0 - (1.0 - base_risk) * decay
            risk_sum += aged_p

        if risk_sum >= risk_budget_T:
            batches.append((start, idx, now_ts))
            start = idx + 1
            entry_times = []
            current_batch_indices = []

    if start < len(commits):
        batches.append((start, len(commits) - 1, commits[-1]["ts"]))

    return _simulate_signature_union_driver(commits, bisect_fn, batches, num_workers)


def simulate_ratb_with_bisect(commits, bisect_fn, params, num_workers):
    """
    Risk-Aware Trigger Batching (RATB) with TWB-style time-window fallback.

    params:
      - threshold (T): risk cutoff; any commit with risk >= T triggers an immediate flush.
      - time_window_hours: TWB-like max age of a batch; if the span between the
        first commit in the batch and the current commit exceeds this window,
        we flush the batch even if no high-risk commit appeared.

    Behavior:
      - Stream commits in time order.
      - Maintain a current batch of contiguous commits.
      - For each commit c:
          * Add c to the current batch.
          * If c["risk"] >= threshold, flush the batch (including c).
          * Else, if the current batch duration exceeds time_window_hours,
            flush the previous batch and start a new one with c.
      - At the end, flush any remaining commits.
    """
    # Backward compatibility: allow scalar `params` (just threshold),
    # defaulting the time window to 4h.
    if isinstance(params, tuple):
        threshold, time_window_hours = params
    else:
        threshold = float(params)
        time_window_hours = 4.0

    logger.info(
        "simulate_ratb_with_bisect: %d commits, threshold=%.4f, time_window_hours=%.2f, bisect_fn=%s",
        len(commits),
        threshold,
        time_window_hours,
        getattr(bisect_fn, "__name__", str(bisect_fn)),
    )
    num_regressors_total = sum(1 for c in commits if c.get("true_label"))

    if not commits:
        logger.info("simulate_ratb_with_bisect: no commits; returning empty results")
        metrics = _StreamingMetrics(0, [], {}, set())
        return build_results(
            metrics.total_tests_run,
            metrics.culprit_times,
            metrics.feedback_times,
            0.0,
            num_regressors_total=num_regressors_total,
            num_regressors_found=0,
        )

    executor = TestExecutor(num_workers)
    metrics = _StreamingMetrics(0, [], {}, set())

    current_batch_start_idx = None
    batch_start_time = None
    window_delta = timedelta(hours=time_window_hours)

    for idx, c in enumerate(commits):
        c_ts = c["ts"]
        if "risk" not in c or c["risk"] is None:
            raise ValueError(
                f"Missing or None 'risk' value for commit at index {idx}: {c!r}"
            )
        risk = float(c["risk"])

        # If starting a new batch
        if current_batch_start_idx is None:
            current_batch_start_idx = idx
            batch_start_time = c_ts
            continue

        # First, risk-triggered flush: include c then flush.
        if risk >= threshold:
            durations = get_batch_signature_durations()
            _run_streaming_suite_and_bisect_per_sig_group(
                commits,
                durations,
                current_batch_start_idx,
                idx,
                c_ts,
                bisect_fn,
                executor,
                metrics,
                last_seen_idx_by_sig=None,
            )
            logger.debug(
                "RATB: risk-triggered flush; batch size %d at index %d/%d (risk=%.4f)",
                (idx - current_batch_start_idx + 1),
                idx + 1,
                len(commits),
                risk,
            )
            current_batch_start_idx = None
            batch_start_time = None
            continue

        # Otherwise, apply TWB-style time-window rule.
        batch_end = batch_start_time + window_delta
        if c_ts >= batch_end:
            # Flush the existing batch (without c) at the time-window boundary.
            durations = get_batch_signature_durations()
            _run_streaming_suite_and_bisect_per_sig_group(
                commits,
                durations,
                current_batch_start_idx,
                idx - 1,
                batch_end,
                bisect_fn,
                executor,
                metrics,
                last_seen_idx_by_sig=None,
            )
            # Start a new batch with c.
            current_batch_start_idx = idx
            batch_start_time = c_ts
        else:
            # Still within time window: just extend the batch.
            pass

    # Flush any leftover commits
    if current_batch_start_idx is not None:
        durations = get_batch_signature_durations()
        _run_streaming_suite_and_bisect_per_sig_group(
            commits,
            durations,
            current_batch_start_idx,
            len(commits) - 1,
            commits[-1]["ts"],
            bisect_fn,
            executor,
            metrics,
            last_seen_idx_by_sig=None,
        )
    logger.info(
        "simulate_ratb_with_bisect: finished; total_tests_run=%d",
        metrics.total_tests_run,
    )
    return build_results(
        metrics.total_tests_run,
        metrics.culprit_times,
        metrics.feedback_times,
        executor.total_cpu_minutes,
        num_regressors_total=num_regressors_total,
        num_regressors_found=len(metrics.found_regressors),
    )


def simulate_ratb_s_with_bisect(commits, bisect_fn, params, num_workers):
    """
    RATB with subset suite detection (RATB-s).

    Uses the same batching boundaries as RATB, but models each batch's initial
    run as a subset suite (union of signature-groups observed within the
    batch). Detection depends on overlap with failing signature-groups, and
    bisection is invoked with `is_batch_root=False` because the batch suite is
    assumed to have already run.
    """
    if isinstance(params, tuple):
        threshold, time_window_hours = params
    else:
        threshold = float(params)
        time_window_hours = 4.0

    batches = []
    if not commits:
        return build_results(0, [], {}, 0.0, num_regressors_total=0, num_regressors_found=0)

    start = 0
    batch_start_time = commits[0]["ts"]
    window_delta = timedelta(hours=time_window_hours)

    for idx, c in enumerate(commits):
        c_ts = c["ts"]
        risk = float(c["risk"])

        if idx == start:
            batch_start_time = c_ts
            continue

        if risk >= threshold:
            batches.append((start, idx, c_ts))
            start = idx + 1
            if start < len(commits):
                batch_start_time = commits[start]["ts"]
            continue

        batch_end_time = batch_start_time + window_delta
        if c_ts >= batch_end_time:
            batches.append((start, idx - 1, batch_end_time))
            start = idx
            batch_start_time = c_ts

    if start < len(commits):
        batches.append((start, len(commits) - 1, commits[-1]["ts"]))

    return _simulate_signature_union_driver(commits, bisect_fn, batches, num_workers)
