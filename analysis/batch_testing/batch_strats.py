from datetime import timedelta
import math

from bisection_strats import (
    TEST_DURATION_MIN,
    TestExecutor,
)

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
    total_tests_run = 0
    culprit_times = []
    feedback_times = {}

    # One central executor per simulation run
    executor = TestExecutor(num_workers, TEST_DURATION_MIN)

    batch_start = commits[0]["ts"]
    batch_end = batch_start + timedelta(hours=batch_hours)
    current_batch = []

    for c in commits:
        if c["ts"] < batch_end:
            current_batch.append(c)
        else:
            total_tests_run, culprit_times, feedback_times = bisect_fn(
                current_batch, batch_end, total_tests_run, culprit_times, feedback_times, executor
            )
            batch_start = c["ts"]
            batch_end = batch_start + timedelta(hours=batch_hours)
            current_batch = [c]

    if current_batch:
        total_tests_run, culprit_times, feedback_times = bisect_fn(
            current_batch, batch_end, total_tests_run, culprit_times, feedback_times, executor
        )

    return build_results(total_tests_run, culprit_times, feedback_times)


def simulate_fsb_with_bisect(commits, bisect_fn, batch_size, num_workers):
    total_tests_run = 0
    culprit_times = []
    feedback_times = {}

    executor = TestExecutor(num_workers, TEST_DURATION_MIN)

    current_batch = []
    current_end_time = None

    for c in commits:
        current_batch.append(c)
        current_end_time = c["ts"]
        if len(current_batch) >= batch_size:
            total_tests_run, culprit_times, feedback_times = bisect_fn(
                current_batch, current_end_time, total_tests_run, culprit_times, feedback_times, executor
            )
            current_batch = []
            current_end_time = None

    if current_batch:
        total_tests_run, culprit_times, feedback_times = bisect_fn(
            current_batch, current_end_time, total_tests_run, culprit_times, feedback_times, executor
        )

    return build_results(total_tests_run, culprit_times, feedback_times)


def simulate_rasb_t_with_bisect(commits, bisect_fn, threshold, num_workers):
    """
    Risk-Adaptive Stream Batching (RASB-T)
    Uses a central executor shared by all batch tests in this simulation.
    """
    total_tests_run = 0
    culprit_times = []
    feedback_times = {}

    executor = TestExecutor(num_workers, TEST_DURATION_MIN)

    current_batch = []
    prod_clean = 1.0
    current_end_time = None

    for c in commits:
        p = c["risk"]
        current_batch.append(c)
        current_end_time = c["ts"]

        prod_clean *= (1.0 - p)
        fail_prob = 1.0 - prod_clean

        if fail_prob >= threshold:
            total_tests_run, culprit_times, feedback_times = bisect_fn(
                current_batch, current_end_time, total_tests_run, culprit_times, feedback_times, executor
            )
            current_batch = []
            prod_clean = 1.0
            current_end_time = None

    if current_batch:
        total_tests_run, culprit_times, feedback_times = bisect_fn(
            current_batch, current_end_time, total_tests_run, culprit_times, feedback_times, executor
        )

    return build_results(total_tests_run, culprit_times, feedback_times)


def simulate_rapb_t_a_with_bisect(commits, bisect_fn, params, num_workers):
    """
    RAPB-T-a (bounded) with central executor.
    """
    threshold_T, aging_rate = params

    total_tests_run = 0
    culprit_times = []
    feedback_times = {}

    executor = TestExecutor(num_workers, TEST_DURATION_MIN)

    current_batch = []
    entry_times = []
    current_end_time = None

    for c in commits:
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
            current_batch = []
            entry_times = []
            current_end_time = None

    if current_batch:
        total_tests_run, culprit_times, feedback_times = bisect_fn(
            current_batch, current_end_time, total_tests_run, culprit_times, feedback_times, executor
        )

    return build_results(total_tests_run, culprit_times, feedback_times)


def simulate_rrbb_with_bisect(commits, bisect_fn, risk_budget, num_workers):
    """
    Rolling Risk Budget (RRB) batching.

    - Stream commits in time order.
    - Maintain a cumulative risk_sum = Σ p_i in the current batch.
    - Once risk_sum >= risk_budget, flush the batch to bisect_fn and start a new one.
    - Always keeps batches contiguous in commit order.

    Param:
      risk_budget: e.g. 1.0 ~ "about one expected failing commit per batch"
    """
    total_tests_run = 0
    culprit_times = []
    feedback_times = {}

    if not commits:
        return build_results(total_tests_run, culprit_times, feedback_times)

    executor = TestExecutor(num_workers, TEST_DURATION_MIN)

    current_batch = []
    current_end_time = None
    risk_sum = 0.0

    for c in commits:
        current_batch.append(c)
        current_end_time = c["ts"]
        risk_sum += float(c.get("risk", 0.0) or 0.0)

        if risk_sum >= risk_budget:
            total_tests_run, culprit_times, feedback_times = bisect_fn(
                current_batch, current_end_time, total_tests_run, culprit_times, feedback_times, executor
            )
            current_batch = []
            current_end_time = None
            risk_sum = 0.0

    if current_batch:
        total_tests_run, culprit_times, feedback_times = bisect_fn(
            current_batch, current_end_time, total_tests_run, culprit_times, feedback_times, executor
        )

    return build_results(total_tests_run, culprit_times, feedback_times)
