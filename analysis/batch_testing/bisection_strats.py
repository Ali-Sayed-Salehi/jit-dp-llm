# bisection_strats.py
import random
from datetime import timedelta

# shared across all bisection algos
TEST_DURATION_MIN = 10
TOP_K_RISK_FIRST = 3  # for TKRB-K

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


def risk_weighted_adaptive_bisect(batch, start_time, total_tests_run, culprit_times, feedback_times):
    """
    Risk-Weighted Adaptive Bisection (RWAB)
    - for failing batches, split into two sub-batches whose *total risk* is as balanced as possible
    - recurse on the higher-risk sub-batch first
    """
    total_tests_run += 1
    finish_time = start_time + timedelta(minutes=TEST_DURATION_MIN)

    # if whole batch passes, everyone gets feedback
    if not batch_has_regressor(batch):
        for c in batch:
            cid = c["commit_id"]
            if cid not in feedback_times:
                fb_min = (finish_time - c["ts"]).total_seconds() / 60
                feedback_times[cid] = fb_min
        return total_tests_run, culprit_times, feedback_times

    # base case
    if len(batch) == 1:
        c = batch[0]
        fb_min = (finish_time - c["ts"]).total_seconds() / 60
        feedback_times[c["commit_id"]] = fb_min
        if c["true_label"]:
            culprit_times.append(fb_min)
        return total_tests_run, culprit_times, feedback_times

    # compute total risk
    total_risk = sum(c["risk"] for c in batch)

    # if there's no risk signal (all zeros), just split by length
    if total_risk <= 0.0:
        mid = len(batch) // 2
        left = batch[:mid]
        right = batch[mid:]
        left_risk = 0.0
        right_risk = 0.0
    else:
        # greedy risk-balancing:
        # sort by risk desc, then assign each commit to the currently lighter side
        sorted_batch = sorted(batch, key=lambda x: x["risk"], reverse=True)
        left = []
        right = []
        left_risk = 0.0
        right_risk = 0.0
        for c in sorted_batch:
            if left_risk <= right_risk:
                left.append(c)
                left_risk += c["risk"]
            else:
                right.append(c)
                right_risk += c["risk"]

    # test the riskier side first
    first, second = (left, right) if left_risk >= right_risk else (right, left)

    total_tests_run, culprit_times, feedback_times = risk_weighted_adaptive_bisect(
        first, finish_time, total_tests_run, culprit_times, feedback_times
    )
    total_tests_run, culprit_times, feedback_times = risk_weighted_adaptive_bisect(
        second, finish_time, total_tests_run, culprit_times, feedback_times
    )
    return total_tests_run, culprit_times, feedback_times


def top_k_risk_first_bisect(batch, start_time, total_tests_run, culprit_times, feedback_times):
    """
    Top-K Risk-First Bisection (TKRB-K)
    - test whole batch (this call)
    - if it fails, test the top-K riskiest commits one-by-one
    - then run standard (risk-ordered) bisection on the remainder
    """
    total_tests_run += 1
    finish_time = start_time + timedelta(minutes=TEST_DURATION_MIN)

    # if whole batch passes, done
    if not batch_has_regressor(batch):
        for c in batch:
            cid = c["commit_id"]
            if cid not in feedback_times:
                fb_min = (finish_time - c["ts"]).total_seconds() / 60
                feedback_times[cid] = fb_min
        return total_tests_run, culprit_times, feedback_times

    # base case
    if len(batch) == 1:
        c = batch[0]
        fb_min = (finish_time - c["ts"]).total_seconds() / 60
        feedback_times[c["commit_id"]] = fb_min
        if c["true_label"]:
            culprit_times.append(fb_min)
        return total_tests_run, culprit_times, feedback_times

    # sort by risk desc
    sorted_batch = sorted(batch, key=lambda x: x["risk"], reverse=True)
    top_k = sorted_batch[:TOP_K_RISK_FIRST]
    remainder = sorted_batch[TOP_K_RISK_FIRST:]

    # now test each top-k commit individually, one after another
    current_time = finish_time
    for c in top_k:
        total_tests_run += 1
        indiv_finish = current_time + timedelta(minutes=TEST_DURATION_MIN)
        fb_min = (indiv_finish - c["ts"]).total_seconds() / 60
        feedback_times[c["commit_id"]] = fb_min
        if c["true_label"]:
            culprit_times.append(fb_min)
        current_time = indiv_finish  # next test starts after this one

    # after testing top-K, if there's remainder:
    if remainder:
        if batch_has_regressor(remainder):
            # continue with standard (risk-ordered) bisection on the rest
            total_tests_run, culprit_times, feedback_times = risk_ordered_bisect(
                remainder, current_time, total_tests_run, culprit_times, feedback_times
            )
        else:
            # all remaining are clean -> give them feedback at current_time
            for c in remainder:
                cid = c["commit_id"]
                if cid not in feedback_times:
                    fb_min = (current_time - c["ts"]).total_seconds() / 60
                    feedback_times[cid] = fb_min

    return total_tests_run, culprit_times, feedback_times