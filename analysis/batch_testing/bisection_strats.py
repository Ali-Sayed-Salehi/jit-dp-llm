from datetime import timedelta

TEST_DURATION_MIN = 21

TESTS_PER_BATCH_RUN = 850      # first test per batch
TESTS_PER_BISECTION_RUN = 4    # follow-up tests
TESTS_PER_RUN = TESTS_PER_BATCH_RUN


class TestExecutor:
    """
    Central test executor with a fixed number of workers.
    Each test takes a fixed duration (TEST_DURATION_MIN).
    When all workers are busy, new tests wait in a FIFO sense, implemented
    by tracking each worker's 'free at' time and choosing the earliest.
    """
    def __init__(self, num_workers, test_duration_min):
        self.num_workers = num_workers
        self.test_duration = timedelta(minutes=test_duration_min)
        # worker_free_times[i] = datetime when worker i becomes free
        self.worker_free_times = [None] * num_workers

    def _ensure_initialized(self, t0):
        # Initialize all workers as free at t0 the first time we schedule.
        for i in range(self.num_workers):
            if self.worker_free_times[i] is None:
                self.worker_free_times[i] = t0

    def schedule(self, requested_start_time):
        """
        Submit a test that becomes 'ready' at requested_start_time.
        Returns the actual finish time given current queue & workers.
        """
        self._ensure_initialized(requested_start_time)

        # Find worker that becomes free earliest
        idx = min(range(self.num_workers), key=lambda i: self.worker_free_times[i])
        earliest_free = self.worker_free_times[idx]

        actual_start = max(requested_start_time, earliest_free)
        finish_time = actual_start + self.test_duration
        self.worker_free_times[idx] = finish_time
        return finish_time


def run_test_suite(executor, requested_start_time, num_tests):
    """
    Run `num_tests` individual tests in parallel as much as the executor allows.
    All of them become ready at `requested_start_time`.

    Returns the time when the *last* of those tests finishes.
    """
    finish_times = []
    for _ in range(num_tests):
        finish_times.append(executor.schedule(requested_start_time))
    return max(finish_times)


def batch_has_regressor(batch):
    return any(c["true_label"] for c in batch)


# ------------------ SHARED INTERVAL TEST HELPER ------------------ #

def _run_interval_and_update(
    batch_sorted,
    status,
    lo,
    hi,
    current_time,
    first_test,
    total_tests_run,
    feedback_times,
    executor,
):
    """
    Shared helper for 'test_interval' used by multiple bisection strategies.

    Parameters
    ----------
    batch_sorted : list[dict]
        Commits in this batch, sorted by "ts".
    status : list[str]
        Per-commit status: "unknown", "clean", or "defect_found".
    lo, hi : int
        Inclusive index range within batch_sorted.
    current_time : datetime
        Logical current time at which this logical test run is scheduled.
    first_test : bool
        True if this is the first test in the batch (controls 850 vs 4 cost).
    total_tests_run : int
        Accumulated test count.
    feedback_times : dict
        commit_id -> feedback time in minutes.
    executor : TestExecutor
        Shared executor for scheduling physical tests.

    Returns
    -------
    has_defect : bool
    current_time : datetime
    first_test : bool
    total_tests_run : int
    """
    if lo > hi:
        return False, current_time, first_test, total_tests_run

    tests_this_run = TESTS_PER_BATCH_RUN if first_test else TESTS_PER_BISECTION_RUN
    first_test = False

    # Cost accounting
    total_tests_run += tests_this_run

    # All physical tests for this logical run start at current_time
    finish_time = run_test_suite(executor, current_time, tests_this_run)

    has_defect = any(
        batch_sorted[i]["true_label"] and status[i] != "defect_found"
        for i in range(lo, hi + 1)
    )

    if not has_defect:
        # Interval is clean; provide feedback for all unknown commits here.
        for i in range(lo, hi + 1):
            if status[i] == "unknown":
                status[i] = "clean"
                cid = batch_sorted[i]["commit_id"]
                if cid not in feedback_times:
                    fb_min = (finish_time - batch_sorted[i]["ts"]).total_seconds() / 60.0
                    feedback_times[cid] = fb_min

    current_time = finish_time
    return has_defect, current_time, first_test, total_tests_run


# ------------------ TOB (unchanged logic, uses helper) ------------------ #

def time_ordered_bisect(batch, start_time, total_tests_run, culprit_times, feedback_times, executor, is_batch_root=True):
    """
    Time-Ordered Bisection (TOB) with central executor.

    Tests are logically sequential (each test's result is needed to pick the next one),
    so we submit them one at a time to the executor.
    """
    if not batch:
        return total_tests_run, culprit_times, feedback_times

    batch_sorted = sorted(batch, key=lambda c: c["ts"])
    n = len(batch_sorted)

    status = ["unknown"] * n
    current_time = start_time
    first_test = bool(is_batch_root)

    while True:
        unknown_indices = [i for i, s in enumerate(status) if s == "unknown"]
        if not unknown_indices:
            break

        lo = unknown_indices[0]
        hi = unknown_indices[-1]

        # Test the entire unknown region
        still_fails, current_time, first_test, total_tests_run = _run_interval_and_update(
            batch_sorted, status, lo, hi, current_time, first_test, total_tests_run, feedback_times, executor
        )
        if not still_fails:
            break

        left = lo
        right = hi
        # Standard midpoint-based binary search
        while left < right:
            mid = (left + right) // 2
            left_fails, current_time, first_test, total_tests_run = _run_interval_and_update(
                batch_sorted, status, left, mid, current_time, first_test, total_tests_run, feedback_times, executor
            )
            if left_fails:
                right = mid
            else:
                left = mid + 1

        idx = left
        c = batch_sorted[idx]
        status[idx] = "defect_found"

        fb_min = (current_time - c["ts"]).total_seconds() / 60.0
        feedback_times[c["commit_id"]] = fb_min
        if c["true_label"]:
            culprit_times.append(fb_min)

    return total_tests_run, culprit_times, feedback_times


# ------------------ Exhaustive variants (unchanged) ------------------ #

def exhaustive_sequential(batch, start_time, total_tests_run, culprit_times, feedback_times, executor, is_batch_root=True):
    """
    Sequential 'no-bisection' strategy with central executor.
    Tests commits one-by-one from oldest to newest.
    """
    if not batch:
        return total_tests_run, culprit_times, feedback_times

    batch_sorted = sorted(batch, key=lambda c: c["ts"])
    current_time = start_time
    first_test = bool(is_batch_root)

    for c in batch_sorted:
        tests_this_run = TESTS_PER_BATCH_RUN if first_test else TESTS_PER_BISECTION_RUN
        first_test = False

        total_tests_run += tests_this_run

        # Fire all `tests_this_run` tests at once for this commit’s run
        finish_time = run_test_suite(executor, current_time, tests_this_run)

        cid = c["commit_id"]
        fb_min = (finish_time - c["ts"]).total_seconds() / 60.0
        feedback_times[cid] = fb_min

        if c["true_label"]:
            culprit_times.append(fb_min)

        current_time = finish_time

    return total_tests_run, culprit_times, feedback_times


def exhaustive_parallel(batch, start_time, total_tests_run, culprit_times, feedback_times, executor, is_batch_root=True):
    """
    Parallel 'no-bisection' strategy.

    Similar to exhaustive_sequential, but instead of waiting for each test to
    finish before submitting the next one, we submit *all* per-commit tests for
    this batch at the same time.
    """
    if not batch:
        return total_tests_run, culprit_times, feedback_times

    batch_sorted = sorted(batch, key=lambda c: c["ts"])

    first_test = bool(is_batch_root)
    requested_time = start_time   # all tests are "ready" at this time

    for c in batch_sorted:
        tests_this_run = TESTS_PER_BATCH_RUN if first_test else TESTS_PER_BISECTION_RUN
        first_test = False
        total_tests_run += tests_this_run

        finish_time = run_test_suite(executor, requested_time, tests_this_run)

        cid = c["commit_id"]
        fb_min = (finish_time - c["ts"]).total_seconds() / 60.0
        feedback_times[cid] = fb_min

        if c["true_label"]:
            culprit_times.append(fb_min)

    return total_tests_run, culprit_times, feedback_times


# ------------------ RWAB using the same helper ------------------ #

def risk_weighted_adaptive_bisect(batch, start_time, total_tests_run, culprit_times, feedback_times, executor, is_batch_root=True):
    """
    Risk-Weighted Adaptive Bisection (RWAB).

    Like time_ordered_bisect, but splits failing intervals so that the sum of
    predicted risk on the left and right halves is approximately balanced.
    """
    if not batch:
        return total_tests_run, culprit_times, feedback_times

    batch_sorted = sorted(batch, key=lambda c: c["ts"])
    n = len(batch_sorted)

    status = ["unknown"] * n
    current_time = start_time
    first_test = bool(is_batch_root)

    # Prefix sums of risk
    risk_prefix = [0.0] * (n + 1)
    for i, c in enumerate(batch_sorted):
        r = float(c.get("risk", 0.0) or 0.0)
        risk_prefix[i + 1] = risk_prefix[i] + r

    def risk_sum(lo, hi):
        return risk_prefix[hi + 1] - risk_prefix[lo]

    def choose_risk_balanced_split(lo, hi):
        if lo >= hi:
            return lo
        total_risk = risk_sum(lo, hi)
        if total_risk <= 0.0:
            return (lo + hi) // 2

        target = total_risk / 2.0
        best_idx = lo
        best_diff = float("inf")

        for s in range(lo, hi):
            left_risk = risk_sum(lo, s)
            diff = abs(left_risk - target)
            if diff < best_diff:
                best_diff = diff
                best_idx = s

        return best_idx

    while True:
        unknown_indices = [i for i, s in enumerate(status) if s == "unknown"]
        if not unknown_indices:
            break

        lo = unknown_indices[0]
        hi = unknown_indices[-1]

        # Test the whole unknown region
        still_fails, current_time, first_test, total_tests_run = _run_interval_and_update(
            batch_sorted, status, lo, hi, current_time, first_test, total_tests_run, feedback_times, executor
        )
        if not still_fails:
            break

        left = lo
        right = hi
        # Risk-weighted "binary" search
        while left < right:
            split = choose_risk_balanced_split(left, right)
            left_fails, current_time, first_test, total_tests_run = _run_interval_and_update(
                batch_sorted, status, left, split, current_time, first_test, total_tests_run, feedback_times, executor
            )
            if left_fails:
                right = split
            else:
                left = split + 1

        idx = left
        c = batch_sorted[idx]
        status[idx] = "defect_found"

        fb_min = (current_time - c["ts"]).total_seconds() / 60.0
        feedback_times[c["commit_id"]] = fb_min
        if c["true_label"]:
            culprit_times.append(fb_min)

    return total_tests_run, culprit_times, feedback_times
