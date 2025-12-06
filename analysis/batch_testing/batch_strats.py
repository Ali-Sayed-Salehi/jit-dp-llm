# batch_strats.py

from datetime import timedelta
import math
import logging

from bisection_strats import (
    TestExecutor
)


logger = logging.getLogger(__name__)

def build_results(total_tests_run, culprit_times, feedback_times):
    if feedback_times:
        mean_fb = sum(feedback_times.values()) / len(feedback_times)
    else:
        mean_fb = 0.0

    if culprit_times:
        mean_ttc = sum(culprit_times) / len(culprit_times)
        max_ttc = max(culprit_times)
    else:
        mean_ttc = 0.0
        max_ttc = 0.0

    return {
        "total_tests_run": total_tests_run,
        "mean_feedback_time_min": round(mean_fb, 2),
        "mean_time_to_culprit_min": round(mean_ttc, 2),
        "max_time_to_culprit_min": round(max_ttc, 2),
    }


def simulate_twb_with_bisect(commits, bisect_fn, batch_hours, num_workers):
    logger.info(
        "simulate_twb_with_bisect: %d commits, batch_hours=%.2f, num_workers=%d, bisect_fn=%s",
        len(commits),
        batch_hours,
        num_workers,
        getattr(bisect_fn, "__name__", str(bisect_fn)),
    )
    total_tests_run = 0
    culprit_times = []
    feedback_times = {}

    # One central executor per simulation run
    executor = TestExecutor(num_workers)

    if not commits:
        logger.info("simulate_twb_with_bisect: no commits; returning empty results")
        return build_results(total_tests_run, culprit_times, feedback_times)

    batch_start = commits[0]["ts"]
    batch_end = batch_start + timedelta(hours=batch_hours)
    current_batch = []

    for idx, c in enumerate(commits, start=1):
        if c["ts"] < batch_end:
            current_batch.append(c)
        else:
            total_tests_run, culprit_times, feedback_times = bisect_fn(
                current_batch, batch_end, total_tests_run, culprit_times, feedback_times, executor
            )
            logger.debug(
                "TWB: flushing batch with %d commits ending at %s (processed %d/%d commits)",
                len(current_batch),
                batch_end,
                idx,
                len(commits),
            )
            batch_start = c["ts"]
            batch_end = batch_start + timedelta(hours=batch_hours)
            current_batch = [c]

    if current_batch:
        total_tests_run, culprit_times, feedback_times = bisect_fn(
            current_batch, batch_end, total_tests_run, culprit_times, feedback_times, executor
        )

    logger.info(
        "simulate_twb_with_bisect: finished; total_tests_run=%d, batches processed",
        total_tests_run,
    )
    return build_results(total_tests_run, culprit_times, feedback_times)


def simulate_fsb_with_bisect(commits, bisect_fn, batch_size, num_workers):
    logger.info(
        "simulate_fsb_with_bisect: %d commits, batch_size=%d, num_workers=%d, bisect_fn=%s",
        len(commits),
        batch_size,
        num_workers,
        getattr(bisect_fn, "__name__", str(bisect_fn)),
    )
    total_tests_run = 0
    culprit_times = []
    feedback_times = {}

    executor = TestExecutor(num_workers)

    current_batch = []
    current_end_time = None

    for idx, c in enumerate(commits, start=1):
        current_batch.append(c)
        current_end_time = c["ts"]
        if len(current_batch) >= batch_size:
            total_tests_run, culprit_times, feedback_times = bisect_fn(
                current_batch, current_end_time, total_tests_run, culprit_times, feedback_times, executor
            )
            logger.debug(
                "FSB: flushing batch of size %d at commit index %d/%d",
                len(current_batch),
                idx,
                len(commits),
            )
            current_batch = []
            current_end_time = None

    if current_batch:
        total_tests_run, culprit_times, feedback_times = bisect_fn(
            current_batch, current_end_time, total_tests_run, culprit_times, feedback_times, executor
        )

    logger.info(
        "simulate_fsb_with_bisect: finished; total_tests_run=%d", total_tests_run
    )
    return build_results(total_tests_run, culprit_times, feedback_times)


def simulate_rasb_t_with_bisect(commits, bisect_fn, threshold, num_workers):
    """
    Risk-Adaptive Stream Batching (RASB-T)
    Uses a central executor shared by all batch tests in this simulation.
    """
    logger.info(
        "simulate_rasb_t_with_bisect: %d commits, threshold=%.4f, num_workers=%d, bisect_fn=%s",
        len(commits),
        threshold,
        num_workers,
        getattr(bisect_fn, "__name__", str(bisect_fn)),
    )

    total_tests_run = 0
    culprit_times = []
    feedback_times = {}

    executor = TestExecutor(num_workers)

    current_batch = []
    prod_clean = 1.0
    current_end_time = None

    for idx, c in enumerate(commits, start=1):
        p = c["risk"]
        current_batch.append(c)
        current_end_time = c["ts"]

        prod_clean *= (1.0 - p)
        fail_prob = 1.0 - prod_clean

        if fail_prob >= threshold:
            total_tests_run, culprit_times, feedback_times = bisect_fn(
                current_batch, current_end_time, total_tests_run, culprit_times, feedback_times, executor
            )
            logger.debug(
                "RASB: threshold reached; flushed batch of size %d at index %d/%d with fail_prob=%.4f",
                len(current_batch),
                idx,
                len(commits),
                fail_prob,
            )
            current_batch = []
            prod_clean = 1.0
            current_end_time = None

    if current_batch:
        total_tests_run, culprit_times, feedback_times = bisect_fn(
            current_batch, current_end_time, total_tests_run, culprit_times, feedback_times, executor
        )

    logger.info(
        "simulate_rasb_t_with_bisect: finished; total_tests_run=%d", total_tests_run
    )
    return build_results(total_tests_run, culprit_times, feedback_times)


def simulate_rapb_t_a_with_bisect(commits, bisect_fn, params, num_workers):
    """
    RAPB-T-a (bounded) with central executor.
    """
    threshold_T, aging_rate = params

    logger.info(
        "simulate_rapb_t_a_with_bisect: %d commits, threshold_T=%.4f, aging_rate=%.4f, num_workers=%d, bisect_fn=%s",
        len(commits),
        threshold_T,
        aging_rate,
        num_workers,
        getattr(bisect_fn, "__name__", str(bisect_fn)),
    )

    total_tests_run = 0
    culprit_times = []
    feedback_times = {}

    executor = TestExecutor(num_workers)

    current_batch = []
    entry_times = []
    current_end_time = None

    for idx, c in enumerate(commits, start=1):
        now_ts = c["ts"]

        current_batch.append(c)
        entry_times.append(now_ts)
        current_end_time = now_ts

        prod_clean = 1.0
        for commit, entered_at in zip(current_batch, entry_times):
            wait_hours = max(0.0, (now_ts - entered_at).total_seconds() / 3600.0)
            base_risk = commit["risk"]

            decay = math.exp(-aging_rate * wait_hours)
            aged_p = 1.0 - (1.0 - base_risk) * decay

            prod_clean *= (1.0 - aged_p)

        fail_prob = 1.0 - prod_clean

        if fail_prob >= threshold_T:
            total_tests_run, culprit_times, feedback_times = bisect_fn(
                current_batch, current_end_time, total_tests_run, culprit_times, feedback_times, executor
            )
            logger.debug(
                "RAPB: threshold reached; flushed batch of size %d at index %d/%d with fail_prob=%.4f",
                len(current_batch),
                idx,
                len(commits),
                fail_prob,
            )
            current_batch = []
            entry_times = []
            current_end_time = None

    if current_batch:
        total_tests_run, culprit_times, feedback_times = bisect_fn(
            current_batch, current_end_time, total_tests_run, culprit_times, feedback_times, executor
        )

    logger.info(
        "simulate_rapb_t_a_with_bisect: finished; total_tests_run=%d",
        total_tests_run,
    )
    return build_results(total_tests_run, culprit_times, feedback_times)


def simulate_rrbb_with_bisect(commits, bisect_fn, risk_budget, num_workers):
    """
    Rolling Risk Budget batching (RRBB).

    - Stream commits in time order.
    - Maintain a cumulative risk_sum = Σ p_i in the current batch.
    - Once risk_sum >= risk_budget, flush the batch to bisect_fn and start a new one.
    - Always keeps batches contiguous in commit order.

    Param:
      risk_budget: e.g. 1.0 ~ "about one expected failing commit per batch"
    """
    logger.info(
        "simulate_rrbb_with_bisect: %d commits, risk_budget=%.4f, num_workers=%d, bisect_fn=%s",
        len(commits),
        risk_budget,
        num_workers,
        getattr(bisect_fn, "__name__", str(bisect_fn)),
    )

    total_tests_run = 0
    culprit_times = []
    feedback_times = {}

    if not commits:
        logger.info("simulate_rrbb_with_bisect: no commits; returning empty results")
        return build_results(total_tests_run, culprit_times, feedback_times)

    executor = TestExecutor(num_workers)

    current_batch = []
    current_end_time = None
    risk_sum = 0.0

    for idx, c in enumerate(commits, start=1):
        current_batch.append(c)
        current_end_time = c["ts"]
        risk_sum += float(c.get("risk", 0.0) or 0.0)

        if risk_sum >= risk_budget:
            total_tests_run, culprit_times, feedback_times = bisect_fn(
                current_batch, current_end_time, total_tests_run, culprit_times, feedback_times, executor
            )
            logger.debug(
                "RRBB: risk_budget reached; flushed batch of size %d at index %d/%d with risk_sum=%.4f",
                len(current_batch),
                idx,
                len(commits),
                risk_sum,
            )
            current_batch = []
            current_end_time = None
            risk_sum = 0.0

    if current_batch:
        total_tests_run, culprit_times, feedback_times = bisect_fn(
            current_batch, current_end_time, total_tests_run, culprit_times, feedback_times, executor
        )

    logger.info(
        "simulate_rrbb_with_bisect: finished; total_tests_run=%d", total_tests_run
    )
    return build_results(total_tests_run, culprit_times, feedback_times)


from datetime import timedelta

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
        "simulate_ratb_with_bisect: %d commits, threshold=%.4f, time_window_hours=%.2f, num_workers=%d, bisect_fn=%s",
        len(commits),
        threshold,
        time_window_hours,
        num_workers,
        getattr(bisect_fn, "__name__", str(bisect_fn)),
    )

    total_tests_run = 0
    culprit_times = []
    feedback_times = {}

    if not commits:
        logger.info("simulate_ratb_with_bisect: no commits; returning empty results")
        return build_results(total_tests_run, culprit_times, feedback_times)

    executor = TestExecutor(num_workers)

    current_batch = []
    current_end_time = None
    batch_start_time = None
    window_delta = timedelta(hours=time_window_hours)

    for idx, c in enumerate(commits, start=1):
        c_ts = c["ts"]
        risk = float(c.get("risk", 0.0) or 0.0)

        # If starting a new batch
        if not current_batch:
            current_batch = [c]
            batch_start_time = c_ts
            current_end_time = c_ts
            continue

        # First, risk-triggered flush: include c then flush.
        if risk >= threshold:
            current_batch.append(c)
            current_end_time = c_ts

            total_tests_run, culprit_times, feedback_times = bisect_fn(
                current_batch,
                current_end_time,
                total_tests_run,
                culprit_times,
                feedback_times,
                executor,
            )
            logger.debug(
                "RATB: risk-triggered flush; batch size %d at index %d/%d (risk=%.4f)",
                len(current_batch),
                idx,
                len(commits),
                risk,
            )
            current_batch = []
            batch_start_time = None
            current_end_time = None
            continue

        # Otherwise, apply TWB-style time-window rule.
        batch_end = batch_start_time + window_delta
        if c_ts >= batch_end:
            # Flush the existing batch (without c),
            # using the last commit time in that batch as end time.
            prev_end_time = current_batch[-1]["ts"]
            total_tests_run, culprit_times, feedback_times = bisect_fn(
                current_batch,
                prev_end_time,
                total_tests_run,
                culprit_times,
                feedback_times,
                executor,
            )
            # Start a new batch with c.
            current_batch = [c]
            batch_start_time = c_ts
            current_end_time = c_ts
        else:
            # Still within time window: just extend the batch.
            current_batch.append(c)
            current_end_time = c_ts

    # Flush any leftover commits
    if current_batch:
        total_tests_run, culprit_times, feedback_times = bisect_fn(
            current_batch,
            current_end_time,
            total_tests_run,
            culprit_times,
            feedback_times,
            executor,
        )
    logger.info(
        "simulate_ratb_with_bisect: finished; total_tests_run=%d", total_tests_run
    )
    return build_results(total_tests_run, culprit_times, feedback_times)
