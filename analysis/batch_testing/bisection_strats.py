from datetime import timedelta

TEST_DURATION_MIN = 21

TESTS_PER_BATCH_RUN = 850      # first test per batch
TESTS_PER_BISECTION_RUN = 4  # all follow-up tests during bisection
TESTS_PER_RUN = TESTS_PER_BATCH_RUN  # for run_exhaustive_testing


def batch_has_regressor(batch):
    return any(c["true_label"] for c in batch)


def time_ordered_bisect(batch, start_time, total_tests_run, culprit_times, feedback_times, is_batch_root=True):
    """
    Time-Ordered Bisection (TOB) implementing repeated binary testing:

      Loop:
        - Test the entire remaining unknown region.
        - If it passes -> done (no more bugs in this batch).
        - If it fails -> do a binary search on the older-to-newer interval to find ONE culprit.
        - Mark that culprit as "defect_found" (disabled).
        - Repeat the loop on the remaining unknown commits.

    Costs:
      - The very first test in this batch: TESTS_PER_BATCH_RUN.
      - Every other test (retests + bisection subtests): TESTS_PER_BISECTION_RUN.
    """

    if not batch:
        return total_tests_run, culprit_times, feedback_times

    # Work on a time-ordered view of this batch
    batch_sorted = sorted(batch, key=lambda c: c["ts"])
    n = len(batch_sorted)

    # Track state for each commit in this batch:
    #   "unknown"      -> not yet proven clean or found defective
    #   "clean"        -> proven clean by some test
    #   "defect_found" -> identified as a culprit; future tests ignore it
    status = ["unknown"] * n

    current_time = start_time
    # We want the first test of this batch to count as a batch run
    first_test = bool(is_batch_root)

    def test_interval(lo, hi):
        """
        Perform one test on commits [lo, hi], inclusive, using ground truth.

        - Uses TESTS_PER_BATCH_RUN for the first test in this batch,
          then TESTS_PER_BISECTION_RUN for all subsequent tests.
        - For pass: all unknown commits in [lo, hi] become "clean" and get feedback.
        - For fail: we don't know which commit is bad yet, just that at least
          one *not-yet-disabled* defect is present in the interval.
        """
        nonlocal total_tests_run, current_time, first_test

        if lo > hi:
            return False

        tests_this_run = TESTS_PER_BATCH_RUN if first_test else TESTS_PER_BISECTION_RUN
        first_test = False

        total_tests_run += tests_this_run
        finish_time = current_time + timedelta(minutes=TEST_DURATION_MIN)

        # A test fails if there's at least one active (not-disabled) defect in [lo, hi]
        has_defect = any(
            batch_sorted[i]["true_label"] and status[i] != "defect_found"
            for i in range(lo, hi + 1)
        )

        if not has_defect:
            # Mark all unknown commits here as clean, and give them feedback
            for i in range(lo, hi + 1):
                if status[i] == "unknown":
                    status[i] = "clean"
                    cid = batch_sorted[i]["commit_id"]
                    if cid not in feedback_times:
                        fb_min = (finish_time - batch_sorted[i]["ts"]).total_seconds() / 60
                        feedback_times[cid] = fb_min

        current_time = finish_time
        return has_defect

    # Repeated Binary Testing over the unknown region
    while True:
        # The remaining unknown indices define our current search space
        unknown_indices = [i for i, s in enumerate(status) if s == "unknown"]
        if not unknown_indices:
            # No unknowns left -> we've either found all bugs or proven everyone clean
            break

        lo = unknown_indices[0]
        hi = unknown_indices[-1]

        # Step 1: test the entire remaining unknown region
        still_fails = test_interval(lo, hi)
        if not still_fails:
            # Whole remaining region is clean; loop can stop
            break

        # Step 2: we know there is at least one active defect in [lo, hi].
        # Perform binary search from older to newer commits to find ONE culprit.
        left = lo
        right = hi
        while left < right:
            mid = (left + right) // 2
            # Always test the older half [left, mid] first (dependency order)
            left_fails = test_interval(left, mid)
            if left_fails:
                # The active defect is in the older half
                right = mid
            else:
                # Older half is clean; defect is in [mid+1, right]
                left = mid + 1

        # Now left == right and there's a defect at index left
        idx = left
        c = batch_sorted[idx]
        status[idx] = "defect_found"

        # Give feedback to this culprit at the time of the last test we just ran
        fb_min = (current_time - c["ts"]).total_seconds() / 60
        feedback_times[c["commit_id"]] = fb_min
        if c["true_label"]:
            culprit_times.append(fb_min)

        # Loop will now:
        #  - treat this defect as disabled (ignored in test_interval),
        #  - re-test remaining unknown region to see if further bugs exist.

    return total_tests_run, culprit_times, feedback_times


def time_ordered_linear(batch, start_time, total_tests_run, culprit_times, feedback_times, is_batch_root=True):
    """
    Sequential 'no-bisection' strategy.

    For a given batch, this just tests each commit individually in
    time order (oldest -> newest) instead of doing any binary search.

    Cost model:
      - First test in the batch: TESTS_PER_BATCH_RUN
      - All subsequent tests in the same batch: TESTS_PER_BISECTION_RUN

    Feedback times and culprit times are computed per-commit.
    """
    if not batch:
        return total_tests_run, culprit_times, feedback_times

    # Ensure time-ordered within the batch
    batch_sorted = sorted(batch, key=lambda c: c["ts"])
    current_time = start_time
    first_test = bool(is_batch_root)

    for c in batch_sorted:
        # Cost for this run
        tests_this_run = TESTS_PER_BATCH_RUN if first_test else TESTS_PER_BISECTION_RUN
        first_test = False

        total_tests_run += tests_this_run
        finish_time = current_time + timedelta(minutes=TEST_DURATION_MIN)

        # Feedback time for this commit
        cid = c["commit_id"]
        fb_min = (finish_time - c["ts"]).total_seconds() / 60
        feedback_times[cid] = fb_min

        if c["true_label"]:
            culprit_times.append(fb_min)

        # Next test starts after this one finishes
        current_time = finish_time

    return total_tests_run, culprit_times, feedback_times
