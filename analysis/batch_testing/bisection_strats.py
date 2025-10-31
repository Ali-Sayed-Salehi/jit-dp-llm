# bisection_strats.py
import random
from datetime import timedelta

# shared across all bisection algos
TEST_DURATION_MIN = 10


def batch_has_regressor(batch):
    return any(c["true_label"] for c in batch)


def risk_ordered_bisect(batch, start_time, total_tests_run, culprit_times, feedback_times):
    total_tests_run += 1
    finish_time = start_time + timedelta(minutes=TEST_DURATION_MIN)

    if not batch_has_regressor(batch):
        for c in batch:
            cid = c["commit_id"]
            if cid not in feedback_times:
                fb_min = (finish_time - c["ts"]).total_seconds() / 60
                feedback_times[cid] = fb_min
        return total_tests_run, culprit_times, feedback_times

    if len(batch) == 1:
        c = batch[0]
        fb_min = (finish_time - c["ts"]).total_seconds() / 60
        feedback_times[c["commit_id"]] = fb_min
        if c["true_label"]:
            culprit_times.append(fb_min)
        return total_tests_run, culprit_times, feedback_times

    # order by risk, test risky half first
    batch.sort(key=lambda x: x["risk"], reverse=True)
    mid = len(batch) // 2
    risky_half = batch[:mid]
    safe_half = batch[mid:]

    total_tests_run, culprit_times, feedback_times = risk_ordered_bisect(
        risky_half, finish_time, total_tests_run, culprit_times, feedback_times
    )
    total_tests_run, culprit_times, feedback_times = risk_ordered_bisect(
        safe_half, finish_time, total_tests_run, culprit_times, feedback_times
    )
    return total_tests_run, culprit_times, feedback_times


def unordered_bisect(batch, start_time, total_tests_run, culprit_times, feedback_times):
    total_tests_run += 1
    finish_time = start_time + timedelta(minutes=TEST_DURATION_MIN)

    # if the whole batch passes, everyone gets feedback at finish_time
    if not batch_has_regressor(batch):
        for c in batch:
            cid = c["commit_id"]
            if cid not in feedback_times:
                fb_min = (finish_time - c["ts"]).total_seconds() / 60
                feedback_times[cid] = fb_min
        return total_tests_run, culprit_times, feedback_times

    # base case: single commit
    if len(batch) == 1:
        c = batch[0]
        fb_min = (finish_time - c["ts"]).total_seconds() / 60
        feedback_times[c["commit_id"]] = fb_min
        if c["true_label"]:
            culprit_times.append(fb_min)
        return total_tests_run, culprit_times, feedback_times

    # shuffle the batch itself so the two halves are random subsets
    batch = list(batch)
    random.shuffle(batch)

    mid = len(batch) // 2
    left = batch[:mid]
    right = batch[mid:]

    # we can still randomize which half to test first
    halves = [left, right]
    random.shuffle(halves)

    for h in halves:
        total_tests_run, culprit_times, feedback_times = unordered_bisect(
            h, finish_time, total_tests_run, culprit_times, feedback_times
        )

    return total_tests_run, culprit_times, feedback_times



def time_ordered_bisect(batch, start_time, total_tests_run, culprit_times, feedback_times):
    """
    Time-Ordered Bisection (TOB):
    - sort by ts (oldest first)
    - split
    - test older half first
    """
    total_tests_run += 1
    finish_time = start_time + timedelta(minutes=TEST_DURATION_MIN)

    if not batch_has_regressor(batch):
        for c in batch:
            cid = c["commit_id"]
            if cid not in feedback_times:
                fb_min = (finish_time - c["ts"]).total_seconds() / 60
                feedback_times[cid] = fb_min
        return total_tests_run, culprit_times, feedback_times

    if len(batch) == 1:
        c = batch[0]
        fb_min = (finish_time - c["ts"]).total_seconds() / 60
        feedback_times[c["commit_id"]] = fb_min
        if c["true_label"]:
            culprit_times.append(fb_min)
        return total_tests_run, culprit_times, feedback_times

    # sort oldest -> newest
    batch.sort(key=lambda x: x["ts"])
    mid = len(batch) // 2
    older_half = batch[:mid]
    newer_half = batch[mid:]

    # test older commits first, then newer
    total_tests_run, culprit_times, feedback_times = time_ordered_bisect(
        older_half, finish_time, total_tests_run, culprit_times, feedback_times
    )
    total_tests_run, culprit_times, feedback_times = time_ordered_bisect(
        newer_half, finish_time, total_tests_run, culprit_times, feedback_times
    )
    return total_tests_run, culprit_times, feedback_times
