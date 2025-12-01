from datetime import timedelta
import math

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


def simulate_twb_with_bisect(commits, bisect_fn, batch_hours):
    total_tests_run = 0
    culprit_times = []
    feedback_times = {}

    batch_start = commits[0]["ts"]
    batch_end = batch_start + timedelta(hours=batch_hours)
    current_batch = []

    for c in commits:
        if c["ts"] < batch_end:
            current_batch.append(c)
        else:
            total_tests_run, culprit_times, feedback_times = bisect_fn(
                current_batch, batch_end, total_tests_run, culprit_times, feedback_times
            )
            batch_start = c["ts"]
            batch_end = batch_start + timedelta(hours=batch_hours)
            current_batch = [c]

    if current_batch:
        total_tests_run, culprit_times, feedback_times = bisect_fn(
            current_batch, batch_end, total_tests_run, culprit_times, feedback_times
        )

    return build_results(total_tests_run, culprit_times, feedback_times)


def simulate_fsb_with_bisect(commits, bisect_fn, batch_size):
    total_tests_run = 0
    culprit_times = []
    feedback_times = {}

    current_batch = []
    current_end_time = None

    for c in commits:
        current_batch.append(c)
        current_end_time = c["ts"]
        if len(current_batch) >= batch_size:
            total_tests_run, culprit_times, feedback_times = bisect_fn(
                current_batch, current_end_time, total_tests_run, culprit_times, feedback_times
            )
            current_batch = []
            current_end_time = None

    if current_batch:
        total_tests_run, culprit_times, feedback_times = bisect_fn(
            current_batch, current_end_time, total_tests_run, culprit_times, feedback_times
        )

    return build_results(total_tests_run, culprit_times, feedback_times)


def simulate_rasb_t_with_bisect(commits, bisect_fn, threshold):
    """
    Risk-Adaptive Stream Batching (RASB-T)
    - stream through commits in time order
    - keep adding commits to the batch
    - compute failure prob = 1 - Π(1 - p_i)
    - once failure prob >= threshold -> test the batch
    - start a new batch
    """
    total_tests_run = 0
    culprit_times = []
    feedback_times = {}

    current_batch = []
    # product of (1 - p_i) over batch; start at 1
    prod_clean = 1.0
    current_end_time = None

    for c in commits:
        p = c["risk"]  # this is p(positive) we already computed in simulation
        current_batch.append(c)
        current_end_time = c["ts"]

        # update product of clean probs
        prod_clean *= (1.0 - p)
        fail_prob = 1.0 - prod_clean

        if fail_prob >= threshold:
            # test this batch now
            total_tests_run, culprit_times, feedback_times = bisect_fn(
                current_batch, current_end_time, total_tests_run, culprit_times, feedback_times
            )
            # reset for next batch
            current_batch = []
            prod_clean = 1.0
            current_end_time = None

    # leftover batch (below threshold) must still be tested
    if current_batch:
        total_tests_run, culprit_times, feedback_times = bisect_fn(
            current_batch, current_end_time, total_tests_run, culprit_times, feedback_times
        )

    return build_results(total_tests_run, culprit_times, feedback_times)


def simulate_rapb_t_a_with_bisect(commits, bisect_fn, params):
    """
    RAPB-T-a (bounded):
    aged_p = 1 - (1 - base_risk) * exp(-a * waited_hours)
    """
    threshold_T, aging_rate = params

    total_tests_run = 0
    culprit_times = []
    feedback_times = {}

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

            # bounded aging
            decay = math.exp(-aging_rate * wait_hours)
            aged_p = 1.0 - (1.0 - base_risk) * decay

            prod_clean *= (1.0 - aged_p)

        fail_prob = 1.0 - prod_clean

        if fail_prob >= threshold_T:
            total_tests_run, culprit_times, feedback_times = bisect_fn(
                current_batch, current_end_time, total_tests_run, culprit_times, feedback_times
            )
            current_batch = []
            entry_times = []
            current_end_time = None

    if current_batch:
        total_tests_run, culprit_times, feedback_times = bisect_fn(
            current_batch, current_end_time, total_tests_run, culprit_times, feedback_times
        )

    return {
        "total_tests_run": total_tests_run,
        "mean_feedback_time_min": round(sum(feedback_times.values()) / len(feedback_times), 2) if feedback_times else 0.0,
        "mean_time_to_culprit_min": round(sum(culprit_times) / len(culprit_times), 2) if culprit_times else 0.0,
        "max_time_to_culprit_min": round(max(culprit_times), 2) if culprit_times else 0.0,
    }
