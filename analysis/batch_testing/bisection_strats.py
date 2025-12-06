from datetime import timedelta
import os
import csv
import json

# ---------- Perf metadata loading ----------

SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

JOB_DURATIONS_CSV = os.path.join(
    REPO_ROOT, "datasets", "mozilla_perf", "job_durations.csv"
)
ALERT_FAIL_SIGS_CSV = os.path.join(
    REPO_ROOT, "datasets", "mozilla_perf", "alert_summary_fail_perf_sigs.csv"
)

# Fallback if CSVs are missing
DEFAULT_TEST_DURATION_MIN = 21.0

# signature_id -> duration_minutes
SIGNATURE_DURATIONS = {}
# revision (commit_id) -> list[int signature_id]
REVISION_FAIL_SIG_IDS = {}
# list[float] of durations for the "full" batch test (all signatures)
BATCH_SIGNATURE_DURATIONS = []


def _load_perf_metadata():
    """
    Load:
      - job_durations.csv => SIGNATURE_DURATIONS, BATCH_SIGNATURE_DURATIONS
      - alert_summary_fail_perf_sigs.csv => REVISION_FAIL_SIG_IDS
    """
    global SIGNATURE_DURATIONS, REVISION_FAIL_SIG_IDS, BATCH_SIGNATURE_DURATIONS

    if SIGNATURE_DURATIONS and BATCH_SIGNATURE_DURATIONS and REVISION_FAIL_SIG_IDS:
        # Already loaded
        return

    # ----- job_durations.csv -----
    sig_durations = {}
    try:
        with open(JOB_DURATIONS_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sig = row.get("signature_id")
                dur = row.get("duration_minutes")
                if not sig or not dur:
                    continue
                try:
                    sig_id = int(sig)
                    duration = float(dur)
                except ValueError:
                    continue
                sig_durations[sig_id] = duration
    except FileNotFoundError:
        sig_durations = {}

    SIGNATURE_DURATIONS = sig_durations
    BATCH_SIGNATURE_DURATIONS = list(SIGNATURE_DURATIONS.values()) or [
        DEFAULT_TEST_DURATION_MIN
    ]

    # ----- alert_summary_fail_perf_sigs.csv -----
    rev_fail = {}
    try:
        with open(ALERT_FAIL_SIGS_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rev = row.get("revision")
                raw = row.get("fail_perf_sig_ids") or ""
                if not rev or not raw:
                    continue

                sig_ids = []
                # Example: "[5094909, 5095143, 5095203]"
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        sig_ids = [int(x) for x in parsed]
                except json.JSONDecodeError:
                    raw_stripped = raw.strip().strip("[]")
                    if raw_stripped:
                        parts = [p.strip() for p in raw_stripped.split(",") if p.strip()]
                        sig_ids = []
                        for p in parts:
                            try:
                                sig_ids.append(int(p))
                            except ValueError:
                                continue

                rev_fail[rev] = sig_ids
    except FileNotFoundError:
        rev_fail = {}

    REVISION_FAIL_SIG_IDS = rev_fail


_load_perf_metadata()


def get_batch_signature_durations():
    """
    Durations (minutes) for a full perf batch run:
    all signatures from job_durations.csv.
    """
    return BATCH_SIGNATURE_DURATIONS or [DEFAULT_TEST_DURATION_MIN]


def get_failing_signature_durations_for_batch(batch_sorted):
    """
    Durations (minutes) for the failing signatures revealed by the initial
    full batch run for this batch.

    We take the union of failing perf signatures for all *regressor* commits
    in this batch (true_label == True) using alert_summary_fail_perf_sigs.csv,
    and map them to durations via job_durations.csv.

    This matches the workflow:
      - First full batch run fails on some signatures.
      - All bisection steps then only run those failing signatures.
    """
    sig_ids = set()

    for c in batch_sorted:
        if not c.get("true_label"):
            continue
        cid = c["commit_id"]
        for sig in REVISION_FAIL_SIG_IDS.get(cid, []):
            if sig in SIGNATURE_DURATIONS:
                sig_ids.add(sig)

    # if not sig_ids:
    #     # No regressors or no metadata: in principle there would be no
    #     # bisection, so this is rarely used. Fall back to full suite
    #     # to avoid undercounting if it does get used.
    #     return get_batch_signature_durations()

    return [SIGNATURE_DURATIONS[s] for s in sig_ids]


TKRB_TOP_K = 1

class TestExecutor:
    """
    Central test executor with a fixed number of workers.

    Each scheduled test provides its own duration in minutes.
    """

    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        # worker_free_times[i] = datetime when worker i becomes free
        self.worker_free_times = [None] * num_workers

    def _ensure_initialized(self, t0):
        # Initialize all workers as free at t0 the first time we schedule.
        for i in range(self.num_workers):
            if self.worker_free_times[i] is None:
                self.worker_free_times[i] = t0

    def schedule(self, requested_start_time, duration_minutes: float):
        """
        Submit a single test that becomes 'ready' at requested_start_time,
        with its own duration (in minutes).

        Returns the actual finish time given current queue & workers.
        """
        self._ensure_initialized(requested_start_time)

        # Find worker that becomes free earliest
        idx = min(range(self.num_workers), key=lambda i: self.worker_free_times[i])
        earliest_free = self.worker_free_times[idx]

        actual_start = max(requested_start_time, earliest_free)
        finish_time = actual_start + timedelta(minutes=duration_minutes)
        self.worker_free_times[idx] = finish_time
        return finish_time


def run_test_suite(executor: TestExecutor, requested_start_time, durations_minutes):
    """
    Run a suite of tests in parallel as much as the executor allows.

    durations_minutes: iterable of per-test durations (in minutes).
    All tests become ready at `requested_start_time`.

    Returns the time when the *last* of those tests finishes.
    """
    durations = list(durations_minutes)
    if not durations:
        return requested_start_time

    finish_times = [
        executor.schedule(requested_start_time, dur) for dur in durations
    ]
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
    batch_fail_durations,
):
    """
    Shared helper for 'test_interval' used by multiple bisection strategies.

    Semantics:

      - For the first test in a batch (first_test == True), we model a *full*
        perf batch run: all signatures from job_durations.csv.

      - For subsequent interval tests (bisection steps), we model a targeted run
        using the failing perf signatures for this batch.

    Cost is counted as the number of signatures run in the suite.
    """
    if lo > hi:
        return False, current_time, first_test, total_tests_run

    if first_test:
        durations = get_batch_signature_durations()
    else:
        durations = batch_fail_durations

    tests_this_run = len(durations)
    first_test = False  # all subsequent calls in this batch are "bisection"

    # Cost accounting
    total_tests_run += tests_this_run

    # All physical tests for this logical run start at current_time
    finish_time = run_test_suite(executor, current_time, durations)

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


# ------------------ TOB ------------------ #

def time_ordered_bisect(
    batch,
    start_time,
    total_tests_run,
    culprit_times,
    feedback_times,
    executor,
    is_batch_root=True,
):
    """
    Time-Ordered Bisection (TOB) with central executor.

    Cost/time model:

      - First call within a batch:
          * Full batch run (all signatures).
      - Subsequent calls (bisection steps):
          * Only the failing signatures for this batch.
    """
    if not batch:
        return total_tests_run, culprit_times, feedback_times

    batch_sorted = sorted(batch, key=lambda c: c["ts"])
    n = len(batch_sorted)

    status = ["unknown"] * n
    current_time = start_time
    first_test = bool(is_batch_root)

    batch_fail_durations = get_failing_signature_durations_for_batch(batch_sorted)

    while True:
        unknown_indices = [i for i, s in enumerate(status) if s == "unknown"]
        if not unknown_indices:
            break

        lo = unknown_indices[0]
        hi = unknown_indices[-1]

        # Test the entire unknown region
        still_fails, current_time, first_test, total_tests_run = _run_interval_and_update(
            batch_sorted,
            status,
            lo,
            hi,
            current_time,
            first_test,
            total_tests_run,
            feedback_times,
            executor,
            batch_fail_durations,
        )
        if not still_fails:
            break

        left = lo
        right = hi
        # Standard midpoint-based binary search
        while left < right:
            mid = (left + right) // 2
            left_fails, current_time, first_test, total_tests_run = _run_interval_and_update(
                batch_sorted,
                status,
                left,
                mid,
                current_time,
                first_test,
                total_tests_run,
                feedback_times,
                executor,
                batch_fail_durations,
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


# ------------------ Exhaustive variants ------------------ #

def exhaustive_parallel(
    batch,
    start_time,
    total_tests_run,
    culprit_times,
    feedback_times,
    executor,
    is_batch_root=True,
):
    """
    Parallel 'no-bisection' strategy with realistic durations.

    New cost/time model (aligned with your intent):

      - For each batch:
          * First test: ONE "full" batch run using ALL signatures
            from job_durations.csv.

          * Subsequent per-commit runs (for all commits EXCEPT the last):
              - Use the SAME failing-signature suite as bisection for
                this batch, i.e., the union of failing perf signatures
                across all regressor revisions in this batch, as derived
                from alert_summary_fail_perf_sigs.csv.

          * The last commit's feedback time is when the last per-commit
            suite finishes (no additional tests for the last one).

      - If is_batch_root is False:
          * Skip the full batch run and just schedule per-commit runs
            using the batch-level failing signatures.
    """
    if not batch:
        return total_tests_run, culprit_times, feedback_times

    batch_sorted = sorted(batch, key=lambda c: c["ts"])
    n = len(batch_sorted)

    # Precompute batch-level failing signatures (same set used everywhere in this batch)
    batch_fail_durations = get_failing_signature_durations_for_batch(batch_sorted)

    # Degenerate: single commit
    if n == 1:
        c = batch_sorted[0]
        cid = c["commit_id"]

        if is_batch_root:
            # Single-commit batch: just the full suite (conceptually same as batch test)
            durations = get_batch_signature_durations()
        else:
            # Non-root usage: use batch-level failing signatures (still meaningful)
            durations = batch_fail_durations

        tests_this_run = len(durations)
        total_tests_run += tests_this_run
        finish_time = run_test_suite(executor, start_time, durations)

        fb_min = (finish_time - c["ts"]).total_seconds() / 60.0
        feedback_times[cid] = fb_min
        if c["true_label"]:
            culprit_times.append(fb_min)

        return total_tests_run, culprit_times, feedback_times

    # Normal case: at least 2 commits
    if is_batch_root:
        # 1) One "full" batch run: all signatures.
        durations_full = get_batch_signature_durations()
        total_tests_run += len(durations_full)
        finish_time_batch = run_test_suite(executor, start_time, durations_full)
        requested_time = finish_time_batch
    else:
        # Non-root usage: no full batch run; per-commit suites start at start_time.
        requested_time = start_time

    # 2) Per-commit suites using batch-level failing signatures
    #    for all commits EXCEPT the last.
    intermediate_finish_times = []
    for idx in range(0, n - 1):
        c = batch_sorted[idx]
        cid = c["commit_id"]

        durations = batch_fail_durations
        tests_this_run = len(durations)
        total_tests_run += tests_this_run

        finish_time = run_test_suite(executor, requested_time, durations)
        intermediate_finish_times.append(finish_time)

        fb_min = (finish_time - c["ts"]).total_seconds() / 60.0
        feedback_times[cid] = fb_min

        if c["true_label"]:
            culprit_times.append(fb_min)

    # 3) Feedback for the last commit comes when the last per-commit suite finishes.
    last_commit = batch_sorted[-1]
    last_cid = last_commit["commit_id"]

    if intermediate_finish_times:
        last_intermediate_finish = max(intermediate_finish_times)
    else:
        # Shouldn't happen for n >= 2, but guard anyway
        last_intermediate_finish = requested_time

    fb_min_last = (
        last_intermediate_finish - last_commit["ts"]
    ).total_seconds() / 60.0
    feedback_times[last_cid] = fb_min_last
    if last_commit["true_label"]:
        culprit_times.append(fb_min_last)

    return total_tests_run, culprit_times, feedback_times


# ------------------ RWAB ------------------ #

def risk_weighted_adaptive_bisect(
    batch,
    start_time,
    total_tests_run,
    culprit_times,
    feedback_times,
    executor,
    is_batch_root=True,
):
    """
    Risk-Weighted Adaptive Bisection (RWAB).

    Like time_ordered_bisect, but splits failing intervals so that the sum of
    predicted risk on the left and right halves is approximately balanced.

    Cost/time model:

      - First batch test uses the full suite.
      - Subsequent bisection tests use only the failing signatures for
        this batch.
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

    batch_fail_durations = get_failing_signature_durations_for_batch(batch_sorted)

    while True:
        unknown_indices = [i for i, s in enumerate(status) if s == "unknown"]
        if not unknown_indices:
            break

        lo = unknown_indices[0]
        hi = unknown_indices[-1]

        # Test the whole unknown region
        still_fails, current_time, first_test, total_tests_run = _run_interval_and_update(
            batch_sorted,
            status,
            lo,
            hi,
            current_time,
            first_test,
            total_tests_run,
            feedback_times,
            executor,
            batch_fail_durations,
        )
        if not still_fails:
            break

        left = lo
        right = hi
        # Risk-weighted "binary" search
        while left < right:
            split = choose_risk_balanced_split(left, right)
            left_fails, current_time, first_test, total_tests_run = _run_interval_and_update(
                batch_sorted,
                status,
                left,
                split,
                current_time,
                first_test,
                total_tests_run,
                feedback_times,
                executor,
                batch_fail_durations,
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

    Cost/time model:

      - First batch test uses the full suite.
      - All subsequent tests (probes and fallback bisection) use only the
        failing signatures for this batch.
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

    batch_fail_durations = get_failing_signature_durations_for_batch(batch_sorted)

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
            batch_fail_durations,
        )
        interval_cache[key] = has_defect
        return has_defect

    def tob_bisect_on_range(lo_bound, hi_bound):
        """
        Run a TOB-style bisection restricted to [lo_bound .. hi_bound],
        using run_interval (with cache) for physical runs.
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
                break

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
        # Entire batch clean
        return total_tests_run, culprit_times, feedback_times

    # ---- Step 2: Top-K risk-first probing ----
    risks = [
        (i, float(batch_sorted[i].get("risk", 0.0) or 0.0))
        for i in range(n)
    ]
    risks.sort(key=lambda t: t[1], reverse=True)
    top_indices = [i for (i, _) in risks[:TKRB_TOP_K]]

    for idx in top_indices:
        if status[idx] != "unknown":
            continue

        curr_fails = run_interval(0, idx)

        prev_fails = False
        if curr_fails and idx > 0:
            prev_fails = run_interval(0, idx - 1)

        if curr_fails and not prev_fails:
            c = batch_sorted[idx]
            if status[idx] != "defect_found":
                status[idx] = "defect_found"

                fb_min = (current_time - c["ts"]).total_seconds() / 60.0
                feedback_times[c["commit_id"]] = fb_min
                if c["true_label"]:
                    culprit_times.append(fb_min)

        elif curr_fails and prev_fails and idx > 0:
            tob_bisect_on_range(0, idx - 1)

    # ---- Step 3: Fall back to TOB-style search over remaining unknowns ----
    while True:
        unknown_indices = [i for i, s in enumerate(status) if s == "unknown"]
        if not unknown_indices:
            break

        lo = unknown_indices[0]
        hi = unknown_indices[-1]

        still_fails = run_interval(lo, hi)
        if not still_fails:
            break

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
