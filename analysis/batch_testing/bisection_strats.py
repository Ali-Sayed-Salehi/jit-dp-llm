from datetime import timedelta

TEST_DURATION_MIN = 21

TESTS_PER_BATCH_RUN = 850      # first test per batch
TESTS_PER_BISECTION_RUN = 4    # follow-up tests
TESTS_PER_RUN = TESTS_PER_BATCH_RUN

TKRB_TOP_K = 1

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

def exhaustive_parallel(batch, start_time, total_tests_run, culprit_times, feedback_times, executor, is_batch_root=True):
    """
    Parallel 'no-bisection' strategy.

    Semantic A:
      - Baseline is the commit *before* the first commit in the batch.
      - All commits in the batch are potential culprits.

    Semantics here:
      - If is_batch_root:
          * Run ONE expensive batch run (850 tests) for the whole batch (conceptually on tip).
          * Then run cheap per-commit suites (4 tests) for all commits EXCEPT the last.
          * The last commit (D) gets its feedback time when the LAST intermediate
            cheap run finishes (i.e., after C for batch [A,B,C,D]).
      - If not is_batch_root:
          * Just run cheap per-commit suites (4 tests) for all commits independently.
    """
    if not batch:
        return total_tests_run, culprit_times, feedback_times

    batch_sorted = sorted(batch, key=lambda c: c["ts"])
    n = len(batch_sorted)

    # === Degenerate case: only one commit in the batch ===
    # Then there's no "intermediate"; the only commit's feedback is at the
    # finish of its own tests (850 if root, 4 otherwise).
    if n == 1:
        if is_batch_root:
            tests_this_run = TESTS_PER_BATCH_RUN
        else:
            tests_this_run = TESTS_PER_BISECTION_RUN
        total_tests_run += tests_this_run
        finish_time = run_test_suite(executor, start_time, tests_this_run)

        c = batch_sorted[0]
        cid = c["commit_id"]
        fb_min = (finish_time - c["ts"]).total_seconds() / 60.0
        feedback_times[cid] = fb_min
        if c["true_label"]:
            culprit_times.append(fb_min)

        return total_tests_run, culprit_times, feedback_times

    # === Normal case: at least 2 commits ===
    if is_batch_root:
        # 1) One "full" batch run: 850 tests.
        total_tests_run += TESTS_PER_BATCH_RUN
        finish_time_batch = run_test_suite(executor, start_time, TESTS_PER_BATCH_RUN)

        # We'll run cheap per-commit tests for all intermediates (indices 0..n-2)
        # starting after the batch run finishes.
        requested_time = finish_time_batch
    else:
        # Non-root usage: no 850 run; just cheap tests from start_time.
        requested_time = start_time

    # 2) Cheap per-commit suites (4 tests) for all commits EXCEPT the last.
    intermediate_finish_times = []
    for idx in range(0, n - 1):
        c = batch_sorted[idx]

        tests_this_run = TESTS_PER_BISECTION_RUN
        total_tests_run += tests_this_run

        finish_time = run_test_suite(executor, requested_time, tests_this_run)
        intermediate_finish_times.append(finish_time)

        cid = c["commit_id"]
        fb_min = (finish_time - c["ts"]).total_seconds() / 60.0
        feedback_times[cid] = fb_min

        if c["true_label"]:
            culprit_times.append(fb_min)

    # 3) Feedback for the last commit (D):
    #    Its status is resolved once the LAST intermediate's cheap run finishes.
    #    (If there were no intermediates, n==1 case above already handled it.)
    last_commit = batch_sorted[-1]
    last_cid = last_commit["commit_id"]

    if intermediate_finish_times:
        last_intermediate_finish = max(intermediate_finish_times)
    else:
        # Should not happen for n>=2, but guard anyway: fall back to batch finish.
        last_intermediate_finish = finish_time_batch if is_batch_root else requested_time

    fb_min_last = (last_intermediate_finish - last_commit["ts"]).total_seconds() / 60.0
    feedback_times[last_cid] = fb_min_last
    if last_commit["true_label"]:
        culprit_times.append(fb_min_last)

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


# ------------------ TKRB-K: Top-K Risk-First Bisection ------------------ #

def topk_risk_first_bisect(
    batch,
    start_time,
    total_tests_run,
    culprit_times,
    feedback_times,
    executor,
    is_batch_root=True,
):
    """
    Top-K Risk-First Bisection (TKRB-K).

    On batch failure, first isolates and tests the top-K riskiest commits:

      For each of the K riskiest commits (by predicted risk p):

        1. Test subbatch [0 .. i]  (up to and including the risky commit)
        2. If [0 .. i] fails, test [0 .. i-1] (if i > 0).
           - If [0 .. i] fails and [0 .. i-1] is clean, i is a culprit.
           - If both [0 .. i] and [0 .. i-1] fail, run a TOB-style bisection
             restricted to [0 .. i-1].

    If no culprit is found this way (or there are remaining unknown regressors),
    we fall back to a TOB-style search over the remaining unknown region.

    We keep results of any subbatches we already tested, and if TOB needs to
    test exactly the same interval again, we reuse that outcome instead of
    re-running tests.
    """
    if not batch:
        return total_tests_run, culprit_times, feedback_times

    batch_sorted = sorted(batch, key=lambda c: c["ts"])
    n = len(batch_sorted)

    status = ["unknown"] * n
    current_time = start_time
    first_test = bool(is_batch_root)

    # Cache: (lo, hi) -> has_defect boolean, so we don't re-run identical intervals
    interval_cache = {}

    def run_interval(lo, hi):
        """
        Wrapper around _run_interval_and_update that:
          - Skips empty intervals.
          - Checks cache first.
          - Otherwise runs the interval test, updates time/cost/status/feedback,
            and stores the has_defect result in the cache.
        """
        nonlocal current_time, first_test, total_tests_run

        if lo > hi:
            return False

        key = (lo, hi)
        if key in interval_cache:
            return interval_cache[key]

        has_defect, current_time, first_test, total_tests_run = _run_interval_and_update(
            batch_sorted,
            status,
            lo,
            hi,
            current_time,
            first_test,
            total_tests_run,
            feedback_times,
            executor,
        )
        interval_cache[key] = has_defect
        return has_defect

    def tob_bisect_on_range(lo_bound, hi_bound):
        """
        Run a TOB-style bisection restricted to [lo_bound .. hi_bound],
        using run_interval (with cache) for physical runs.

        This can discover one or more culprits inside that subrange.
        """
        nonlocal current_time, first_test, total_tests_run

        while True:
            unknown_indices = [
                i for i in range(lo_bound, hi_bound + 1)
                if status[i] == "unknown"
            ]
            if not unknown_indices:
                break

            lo = unknown_indices[0]
            hi = unknown_indices[-1]

            still_fails = run_interval(lo, hi)
            if not still_fails:
                # This unknown region is actually clean; _run_interval_and_update
                # already updated status + feedback for its commits.
                break

            # Standard midpoint-based binary search
            left = lo
            right = hi
            while left < right:
                mid = (left + right) // 2
                left_fails = run_interval(left, mid)
                if left_fails:
                    right = mid
                else:
                    left = mid + 1

            culprit_idx = left
            c = batch_sorted[culprit_idx]
            if status[culprit_idx] != "defect_found":
                status[culprit_idx] = "defect_found"
                fb_min = (current_time - c["ts"]).total_seconds() / 60.0
                feedback_times[c["commit_id"]] = fb_min
                if c["true_label"]:
                    culprit_times.append(fb_min)

    # ---- Step 1: Initial whole-batch test (like TOB) ----
    whole_fails = run_interval(0, n - 1)
    if not whole_fails:
        # Entire batch clean; _run_interval_and_update has already filled feedback/status.
        return total_tests_run, culprit_times, feedback_times

    # ---- Step 2: Top-K risk-first probing ----
    risks = [
        (i, float(batch_sorted[i].get("risk", 0.0) or 0.0))
        for i in range(n)
    ]
    # Sort by risk descending
    risks.sort(key=lambda t: t[1], reverse=True)
    top_indices = [i for (i, _) in risks[:TKRB_TOP_K]]

    for idx in top_indices:
        if status[idx] != "unknown":
            continue

        # Test subbatch [0 .. idx] (including risky commit)
        curr_fails = run_interval(0, idx)

        # Only test [0 .. idx-1] if [0 .. idx] fails.
        prev_fails = False
        if curr_fails and idx > 0:
            prev_fails = run_interval(0, idx - 1)

        # Case 1: [0..idx] fails, [0..idx-1] clean => idx is a culprit.
        if curr_fails and not prev_fails:
            c = batch_sorted[idx]
            status[idx] = "defect_found"

            fb_min = (current_time - c["ts"]).total_seconds() / 60.0
            feedback_times[c["commit_id"]] = fb_min
            if c["true_label"]:
                culprit_times.append(fb_min)

        # Case 2: BOTH [0..idx] and [0..idx-1] fail => do TOB on [0..idx-1].
        elif curr_fails and prev_fails and idx > 0:
            tob_bisect_on_range(0, idx - 1)

    # ---- Step 3: Fall back to TOB-style search over remaining unknowns ----
    while True:
        unknown_indices = [i for i, s in enumerate(status) if s == "unknown"]
        if not unknown_indices:
            break

        lo = unknown_indices[0]
        hi = unknown_indices[-1]

        # Test the entire unknown region
        still_fails = run_interval(lo, hi)
        if not still_fails:
            # No remaining regressors in unknown region; any clean-interval status
            # and feedback was already applied when that interval was first tested.
            break

        # Standard midpoint-based binary search on the unknown region,
        # but using our cached run_interval.
        left = lo
        right = hi
        while left < right:
            mid = (left + right) // 2
            left_fails = run_interval(left, mid)
            if left_fails:
                right = mid
            else:
                left = mid + 1

        culprit_idx = left
        c = batch_sorted[culprit_idx]
        if status[culprit_idx] != "defect_found":
            status[culprit_idx] = "defect_found"
            fb_min = (current_time - c["ts"]).total_seconds() / 60.0
            feedback_times[c["commit_id"]] = fb_min
            if c["true_label"]:
                culprit_times.append(fb_min)

    return total_tests_run, culprit_times, feedback_times
