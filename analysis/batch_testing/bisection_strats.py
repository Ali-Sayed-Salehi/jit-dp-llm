"""
Bisection strategies and shared execution/time model for the batch-testing simulator.

This module provides:
  - Perf metadata loading utilities (signature IDs, durations, failing signatures).
  - A central test-capacity simulator (`TestExecutor` + `run_test_suite`).
  - Bisection strategies that operate on batches (TOB, PAR, RWAB, TKRB, SWB, SWF).

See `analysis/batch_testing/README.md` for a detailed conceptual overview.
"""

from datetime import timedelta
import os
import csv
import json
import logging
import random
import heapq

# ---------- Perf metadata loading ----------

SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

JOB_DURATIONS_CSV = os.path.join(
    REPO_ROOT, "datasets", "mozilla_perf", "job_durations.csv"
)
ALERT_FAIL_SIGS_CSV = os.path.join(
    REPO_ROOT, "datasets", "mozilla_perf", "alert_summary_fail_perf_sigs.csv"
)
PERF_JOBS_PER_REV_JSON = os.path.join(
    REPO_ROOT, "datasets", "mozilla_perf", "perf_jobs_per_revision_details.jsonl"
)

# Fallbacks and knobs (configured by the main simulation script).
_default_test_duration_min = 21.0
# -1 means "use all available signatures" (no downsampling).
_full_suite_signatures_per_run = -1

# signature_id -> duration_minutes
SIGNATURE_DURATIONS = {}
# revision (commit_id) -> list[int signature_id]
REVISION_FAIL_SIG_IDS = {}
# revision (commit_id) -> list[int signature_id] actually tested on that revision
REVISION_TESTED_SIG_IDS = {}
# list[float] of durations for the "full" batch test (all signatures)
BATCH_SIGNATURE_DURATIONS = []

TKRB_TOP_K = 1

logger = logging.getLogger(__name__)


def configure_bisection_defaults(
    default_test_duration_min=None,
    full_suite_signatures_per_run=None,
):
    """
    Configure global defaults used by the simulator's time/cost model.

    Parameters
    ----------
    default_test_duration_min:
        Fallback duration (minutes) used when a signature ID has no recorded
        duration in `job_durations.csv`.
    full_suite_signatures_per_run:
        Caps the number of signatures used for a "full suite" run:
          - `-1` => use all available signatures
          - `N>0` => randomly sample N signatures (models partial suite runs)

    These knobs are typically set by `simulation.py` so all strategies share
    the same configuration.
    """
    global _default_test_duration_min, _full_suite_signatures_per_run

    if default_test_duration_min is not None:
        _default_test_duration_min = float(default_test_duration_min)

    if full_suite_signatures_per_run is not None:
        val = int(full_suite_signatures_per_run)
        if val < 0:
            # -1 => use all signatures (no cap)
            _full_suite_signatures_per_run = -1
        elif val == 0:
            raise ValueError(
                "full_suite_signatures_per_run must be a positive integer "
                "or -1 to indicate 'use all signatures'; got 0."
            )
        else:
            _full_suite_signatures_per_run = val

    logger.info(
        "Configured bisection defaults: default_test_duration_min=%.2f, "
        "full_suite_signatures_per_run=%s",
        _default_test_duration_min,
        str(_full_suite_signatures_per_run),
    )


def configure_full_suite_signatures_union(revisions):
    """
    Given an iterable of revision ids that fall within the simulation's
    cutoff windows, compute the union of all perf signatures that were
    actually tested on at least one of those revisions and update the
    full-suite batch durations accordingly.

    This is used when we want each initial batch test run to execute all
    tests that appear at least once within the cutoff window, instead of
    all signatures from job_durations.csv.

    This function mutates the module-level `BATCH_SIGNATURE_DURATIONS`, which
    is consumed by `get_batch_signature_durations()`.
    """
    global BATCH_SIGNATURE_DURATIONS

    _load_perf_metadata()
    _load_perf_jobs_per_revision()

    rev_set = set(revisions or [])
    if not rev_set:
        raise RuntimeError(
            "configure_full_suite_signatures_union: empty revision set; "
            "cannot construct a full-suite signature union."
        )

    sig_ids = set()
    for rev in rev_set:
        for sig in REVISION_TESTED_SIG_IDS.get(rev, []):
            sig_id = int(sig)
            sig_ids.add(sig_id)

    if not sig_ids:
        raise RuntimeError(
            "configure_full_suite_signatures_union: no tested signatures "
            "found for the provided revisions; cannot build full suite."
        )

    # Use recorded durations where available; fall back to the default
    # duration for any signature missing from job_durations.csv.
    BATCH_SIGNATURE_DURATIONS = [
        SIGNATURE_DURATIONS.get(sig_id, _default_test_duration_min)
        for sig_id in sig_ids
    ]


def _load_perf_metadata():
    """
    Load:
      - job_durations.csv => SIGNATURE_DURATIONS, BATCH_SIGNATURE_DURATIONS
      - alert_summary_fail_perf_sigs.csv => REVISION_FAIL_SIG_IDS
    """
    global SIGNATURE_DURATIONS, REVISION_FAIL_SIG_IDS, BATCH_SIGNATURE_DURATIONS

    if SIGNATURE_DURATIONS and BATCH_SIGNATURE_DURATIONS and REVISION_FAIL_SIG_IDS:
        # Already loaded
        logger.debug("Perf metadata already loaded; skipping reload.")
        return

    # ----- job_durations.csv -----
    sig_durations = {}
    try:
        logger.info("Loading job durations from %s", JOB_DURATIONS_CSV)
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
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"job_durations.csv not found at {JOB_DURATIONS_CSV}"
        ) from exc

    SIGNATURE_DURATIONS = sig_durations
    if not SIGNATURE_DURATIONS:
        raise RuntimeError(
            f"No valid job duration rows loaded from {JOB_DURATIONS_CSV}"
        )
    BATCH_SIGNATURE_DURATIONS = list(SIGNATURE_DURATIONS.values())
    logger.info(
        "Loaded %d signature durations; BATCH_SIGNATURE_DURATIONS length=%d",
        len(SIGNATURE_DURATIONS),
        len(BATCH_SIGNATURE_DURATIONS),
    )

    # ----- alert_summary_fail_perf_sigs.csv -----
    rev_fail = {}
    try:
        logger.info("Loading failing perf signatures from %s", ALERT_FAIL_SIGS_CSV)
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
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"alert_summary_fail_perf_sigs.csv not found at {ALERT_FAIL_SIGS_CSV}"
        ) from exc

    REVISION_FAIL_SIG_IDS = rev_fail
    if not REVISION_FAIL_SIG_IDS:
        raise RuntimeError(
            f"No failing perf signatures loaded from {ALERT_FAIL_SIGS_CSV}"
        )
    logger.info(
        "Loaded failing signature mapping for %d revisions",
        len(REVISION_FAIL_SIG_IDS),
    )


def _load_perf_jobs_per_revision():
    """
    Load:
      - perf_jobs_per_revision_details.jsonl => REVISION_TESTED_SIG_IDS

    The JSON file can be either:
      * a JSON-lines file (one JSON object per line), or
      * a single JSON list of objects.
    Each object is expected to have:
      - 'revision': str
      - 'signature_ids': list[int]
    """
    global REVISION_TESTED_SIG_IDS

    if REVISION_TESTED_SIG_IDS:
        logger.debug(
            "Revision->tested-signatures mapping already loaded; skipping reload."
        )
        return

    mapping = {}
    try:
        logger.info(
            "Loading tested perf signatures per revision from %s (JSONL)",
            PERF_JOBS_PER_REV_JSON,
        )
        with open(PERF_JOBS_PER_REV_JSON, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping malformed JSONL line in %s: %s",
                        PERF_JOBS_PER_REV_JSON,
                        line[:200],
                    )
                    continue
                if not isinstance(obj, dict):
                    continue
                rev = obj.get("revision")
                sig_ids = obj.get("signature_ids") or []
                if not rev:
                    continue
                try:
                    sig_ids_int = [int(s) for s in sig_ids]
                except (TypeError, ValueError):
                    sig_ids_int = []
                mapping[rev] = sig_ids_int
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"perf_jobs_per_revision_details.jsonl not found at {PERF_JOBS_PER_REV_JSON}"
        ) from exc

    REVISION_TESTED_SIG_IDS = mapping
    if not REVISION_TESTED_SIG_IDS:
        raise RuntimeError(
            f"No tested perf signatures loaded from {PERF_JOBS_PER_REV_JSON}"
        )
    logger.info(
        "Loaded tested signatures for %d revisions",
        len(REVISION_TESTED_SIG_IDS),
    )


def validate_failing_signatures_coverage(failing_revisions=None):
    """
    Ensure that failing perf signatures are covered by the
    perf_jobs_per_revision_details.jsonl dataset, restricted to a
    specific set of failing revisions.

    Args:
        failing_revisions: iterable of revision ids that are considered
            buggy and must be present in alert_summary_fail_perf_sigs.csv.

    Raises:
        RuntimeError:
            - if failing_revisions is None or empty,
            - if any of the failing_revisions is missing from
              alert_summary_fail_perf_sigs.csv,
            - or if any relevant failing signature is not covered by
              the perf_jobs_per_revision_details.jsonl dataset.
    """
    _load_perf_metadata()
    _load_perf_jobs_per_revision()

    if not REVISION_FAIL_SIG_IDS:
        # Nothing to validate; either there are no failing signatures
        # or the CSV is missing (already logged in _load_perf_metadata).
        logger.warning(
            "No failing perf signatures loaded from %s; "
            "skipping coverage validation.",
            ALERT_FAIL_SIGS_CSV,
        )
        return

    if not REVISION_TESTED_SIG_IDS:
        raise RuntimeError(
            "perf_jobs_per_revision_details.jsonl did not yield any "
            "tested-signature data. Cannot validate that failing perf "
            "signatures are covered. Please regenerate the dataset at: "
            f"{PERF_JOBS_PER_REV_JSON}"
        )

    if failing_revisions is None:
        raise RuntimeError(
            "failing_revisions must be provided to "
            "validate_failing_signatures_coverage; got None."
        )

    failing_revisions = set(failing_revisions)
    if not failing_revisions:
        raise RuntimeError(
            "failing_revisions is empty; cannot validate perf "
            "signature coverage without any failing revisions."
        )

    # Ensure every failing revision is present in the alert CSV.
    alert_revisions = set(REVISION_FAIL_SIG_IDS.keys())
    missing_revisions = sorted(failing_revisions - alert_revisions)
    if missing_revisions:
        sample = ", ".join(missing_revisions[:20])
        extra = (
            "" if len(missing_revisions) <= 20 else f" (and {len(missing_revisions) - 20} more...)"
        )
        raise RuntimeError(
            "Some failing revisions have no failing perf signature "
            "entries in alert_summary_fail_perf_sigs.csv. Every failing "
            "revision must be present in that file.\n"
            f"First missing revisions: {sample}{extra}\n"
            f"Alert CSV: {ALERT_FAIL_SIGS_CSV}"
        )

    # At this point all failing_revisions are present in the alert CSV.
    relevant_revisions = failing_revisions

    # Collect the set of failing signature IDs for the selected revisions.
    failing_sig_ids = set()
    for rev in relevant_revisions:
        for sig in REVISION_FAIL_SIG_IDS.get(rev, []):
            try:
                failing_sig_ids.add(int(sig))
            except (TypeError, ValueError):
                continue

    # Collect the set of all signature IDs that appear anywhere in the
    # perf_jobs_per_revision_details.jsonl dataset.
    tested_sig_ids = set()
    for sig_ids in REVISION_TESTED_SIG_IDS.values():
        for sig in sig_ids:
            try:
                tested_sig_ids.add(int(sig))
            except (TypeError, ValueError):
                continue

    missing = sorted(failing_sig_ids - tested_sig_ids)
    if missing:
        # Limit how many we show in the error message to keep it readable.
        sample = ", ".join(str(sig) for sig in missing[:20])
        extra = "" if len(missing) <= 20 else f" (and {len(missing) - 20} more...)"
        raise RuntimeError(
            "Found failing perf signatures that are not covered by "
            "perf_jobs_per_revision_details.jsonl at all. "
            "Each relevant failing signature (for the revisions under "
            "consideration) should appear at least once somewhere in the "
            "perf jobs dataset. This means the simulation cannot "
            "exercise all required failing signatures.\n"
            f"First missing signature_ids: {sample}{extra}\n"
            f"Alert CSV: {ALERT_FAIL_SIGS_CSV}\n"
            f"Perf jobs JSONL: {PERF_JOBS_PER_REV_JSON}"
        )


def get_batch_signature_durations():
    """
    Durations (minutes) for a "full suite" perf run.

    By default this is all signatures from `job_durations.csv`, but it can be
    capped to a fixed-size random subset via `_full_suite_signatures_per_run`.

    This suite is used for the *first* run of a batch when bisection strategies
    are invoked with `is_batch_root=True`.
    """
    _load_perf_metadata()

    if not BATCH_SIGNATURE_DURATIONS:
        return [_default_test_duration_min]

    limit = _full_suite_signatures_per_run
    # Non-negative limit => cap via random subset when smaller than the
    # available suite size. Negative (e.g., -1) means "use all".
    if isinstance(limit, int) and limit >= 0 and len(BATCH_SIGNATURE_DURATIONS) > limit:
        durations = random.sample(BATCH_SIGNATURE_DURATIONS, limit)
    else:
        durations = BATCH_SIGNATURE_DURATIONS

    return durations


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
    _load_perf_metadata()

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

    durations = [SIGNATURE_DURATIONS[s] for s in sig_ids]
    logger.debug(
        "Computed failing signature durations for batch of size %d: %d signatures",
        len(batch_sorted),
        len(durations),
    )
    return durations


def get_tested_signatures_for_revision(revision):
    """
    Return the list of signature IDs that were actually tested for the given
    revision according to perf_jobs_per_revision_details.jsonl.
    """
    _load_perf_jobs_per_revision()
    return REVISION_TESTED_SIG_IDS.get(revision, [])


def get_signature_durations_for_ids(signature_ids):
    """
    Map a collection of signature IDs to their durations (in minutes) using
    job_durations.csv. For any unknown signature, we fall back to
    _default_test_duration_min.
    """
    _load_perf_metadata()
    durations = []
    for sig in signature_ids:
        try:
            sig_id = int(sig)
        except (TypeError, ValueError):
            continue
        dur = SIGNATURE_DURATIONS.get(sig_id, _default_test_duration_min)
        durations.append(dur)
    if not durations:
        durations = [_default_test_duration_min]
    return durations


def get_failing_signatures_for_revision(revision):
    """
    Return the list of failing signature IDs for a given revision according to
    alert_summary_fail_perf_sigs.csv.
    """
    _load_perf_metadata()
    return REVISION_FAIL_SIG_IDS.get(revision, [])

class TestExecutor:
    """
    Central test executor with a fixed number of workers.

    Each scheduled test provides its own duration in minutes.
    """

    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        # Min-heap of (free_time, worker_index) for all workers.
        # Initialized lazily on first schedule() call.
        self._worker_heap = []
        # Cumulative CPU time in minutes across all scheduled tests
        self.total_cpu_minutes = 0.0
        logger.debug("Created TestExecutor with %d workers", num_workers)

    def _ensure_initialized(self, t0):
        # Initialize all workers as free at t0 the first time we schedule.
        if not self._worker_heap:
            self._worker_heap = [(t0, i) for i in range(self.num_workers)]
            heapq.heapify(self._worker_heap)

    def schedule(self, requested_start_time, duration_minutes: float):
        """
        Submit a single test that becomes 'ready' at requested_start_time,
        with its own duration (in minutes).

        Returns the actual finish time given current queue & workers.
        """
        self._ensure_initialized(requested_start_time)

        # Find worker that becomes free earliest
        earliest_free, idx = heapq.heappop(self._worker_heap)

        actual_start = max(requested_start_time, earliest_free)
        try:
            duration_float = float(duration_minutes)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Invalid duration_minutes value passed to TestExecutor.schedule: {duration_minutes!r}"
            ) from exc

        finish_time = actual_start + timedelta(minutes=duration_float)
        # Push worker back with updated free time
        heapq.heappush(self._worker_heap, (finish_time, idx))
        # Accumulate CPU time regardless of parallelism
        self.total_cpu_minutes += duration_float
        logger.debug(
            "Scheduled test on worker %d: start=%s, duration=%.2f min, finish=%s",
            idx,
            actual_start,
            duration_minutes,
            finish_time,
        )
        return finish_time


def run_test_suite(executor: TestExecutor, requested_start_time, durations_minutes):
    """
    Run a suite of tests in parallel as much as the executor allows.

    durations_minutes: iterable of per-test durations (in minutes).
    All tests become ready at `requested_start_time`.

    Returns the time when the *last* of those tests finishes.
    """
    if not durations_minutes:
        return requested_start_time

    last_finish = requested_start_time
    count = 0
    for dur in durations_minutes:
        count += 1
        finish_time = executor.schedule(requested_start_time, dur)
        if finish_time > last_finish:
            last_finish = finish_time

    logger.debug(
        "run_test_suite: scheduled %d tests from %s",
        count,
        requested_start_time,
    )
    logger.debug(
        "run_test_suite: last test finished at %s (span=%.2f min)",
        last_finish,
        (last_finish - requested_start_time).total_seconds() / 60.0,
    )
    return last_finish


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

    logger.debug(
        "_run_interval_and_update: interval [%d, %d], first_test=%s, tests_this_run=%d, current_time=%s",
        lo,
        hi,
        first_test,
        tests_this_run,
        current_time,
    )

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
    logger.debug(
        "_run_interval_and_update: finished interval [%d, %d], has_defect=%s, total_tests_run=%d",
        lo,
        hi,
        has_defect,
        total_tests_run,
    )
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

    # Batches are constructed in time order by the caller, so we can
    # avoid re-sorting here and treat the incoming list as already
    # sorted by timestamp.
    batch_sorted = batch
    n = len(batch_sorted)
    tests_before = total_tests_run
    culprits_before = len(culprit_times)
    logger.debug("time_ordered_bisect: starting batch_size=%d", n)

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

    tests_added = total_tests_run - tests_before
    tests_batch_root = len(get_batch_signature_durations()) if is_batch_root else 0
    tests_bisection = max(0, tests_added - tests_batch_root)
    culprits_batch = len(culprit_times) - culprits_before
    logger.debug(
        "time_ordered_bisect: batch_size=%d, tests_batch_root=%d, "
        "tests_bisection=%d, culprits_this_batch=%d, culprits_total=%d, "
        "total_tests_run=%d",
        n,
        tests_batch_root,
        tests_bisection,
        culprits_batch,
        len(culprit_times),
        total_tests_run,
    )
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

    """
    if not batch:
        return total_tests_run, culprit_times, feedback_times

    # Caller provides commits in time order; reuse directly to avoid
    # an extra sort per batch.
    batch_sorted = batch
    n = len(batch_sorted)
    tests_before = total_tests_run
    culprits_before = len(culprit_times)
    logger.debug("exhaustive_parallel: starting batch_size=%d", n)

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

    tests_added = total_tests_run - tests_before
    tests_batch_root = len(get_batch_signature_durations()) if is_batch_root else 0
    tests_bisection = max(0, tests_added - tests_batch_root)
    culprits_batch = len(culprit_times) - culprits_before
    logger.debug(
        "exhaustive_parallel: batch_size=%d, tests_batch_root=%d, "
        "tests_bisection=%d, culprits_this_batch=%d, culprits_total=%d, "
        "total_tests_run=%d",
        n,
        tests_batch_root,
        tests_bisection,
        culprits_batch,
        len(culprit_times),
        total_tests_run,
    )
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

    # Batches from higher-level strategies are already time-ordered.
    batch_sorted = batch
    n = len(batch_sorted)
    tests_before = total_tests_run
    culprits_before = len(culprit_times)
    logger.debug("risk_weighted_adaptive_bisect: starting batch_size=%d", n)

    status = ["unknown"] * n
    current_time = start_time
    first_test = bool(is_batch_root)

    # Prefix sums of risk
    risk_prefix = [0.0] * (n + 1)
    for i, c in enumerate(batch_sorted):
        if "risk" not in c or c["risk"] is None:
            raise ValueError(
                f"Missing or None 'risk' value in batch for risk_weighted_adaptive_bisect at index {i}: {c!r}"
            )
        r = float(c["risk"])
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

    tests_added = total_tests_run - tests_before
    tests_batch_root = len(get_batch_signature_durations()) if is_batch_root else 0
    tests_bisection = max(0, tests_added - tests_batch_root)
    culprits_batch = len(culprit_times) - culprits_before
    logger.debug(
        "risk_weighted_adaptive_bisect: batch_size=%d, tests_batch_root=%d, "
        "tests_bisection=%d, culprits_this_batch=%d, culprits_total=%d, "
        "total_tests_run=%d",
        n,
        tests_batch_root,
        tests_bisection,
        culprits_batch,
        len(culprit_times),
        total_tests_run,
    )
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

    Step 2 (Top-K probing):
      - `TKRB_TOP_K` controls how many of the highest-risk commits are probed.
      - For `TKRB_TOP_K == 1`, probing is sequential.
      - For `TKRB_TOP_K > 1`, all top-K probes are submitted in parallel but
        processed with a barrier (PAR-style): follow-up work happens only after
        all probes finish.
    """
    if not batch:
        return total_tests_run, culprit_times, feedback_times

    # Assume batch is already sorted by timestamp, as produced by the
    # batching strategies; avoid redundant sorting here.
    batch_sorted = batch
    n = len(batch_sorted)
    tests_before = total_tests_run
    culprits_before = len(culprit_times)
    logger.debug("topk_risk_first_bisect: starting batch_size=%d", n)

    status = ["unknown"] * n
    current_time = start_time
    first_test = bool(is_batch_root)

    # Cache: (lo, hi) -> has_defect boolean, so we don't re-run identical intervals
    interval_cache = {}

    batch_fail_durations = get_failing_signature_durations_for_batch(batch_sorted)
    logger.debug(
        "topk_risk_first_bisect: TKRB_TOP_K=%d, failing_signatures=%d",
        int(TKRB_TOP_K),
        len(batch_fail_durations),
    )

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
            logger.debug(
                "topk_risk_first_bisect: using cached interval result for [%d, %d] -> %s",
                lo,
                hi,
                interval_cache[key],
            )
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
        logger.debug(
            "topk_risk_first_bisect: interval [%d, %d] executed, has_defect=%s",
            lo,
            hi,
            has_defect,
        )
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
    risks = []
    for i in range(n):
        c = batch_sorted[i]
        if "risk" not in c or c["risk"] is None:
            raise ValueError(
                f"Missing or None 'risk' value in batch for topk_risk_first_bisect at index {i}: {c!r}"
            )
        risks.append((i, float(c["risk"])))
    risks.sort(key=lambda t: t[1], reverse=True)
    top_indices = [i for (i, _) in risks[:TKRB_TOP_K]]

    def mark_culprit(idx, at_time):
        c = batch_sorted[idx]
        if status[idx] == "defect_found":
            return
        status[idx] = "defect_found"
        fb_min = (at_time - c["ts"]).total_seconds() / 60.0
        feedback_times[c["commit_id"]] = fb_min
        if c["true_label"]:
            culprit_times.append(fb_min)

    def interval_has_defect(lo, hi):
        return any(
            batch_sorted[i]["true_label"] and status[i] != "defect_found"
            for i in range(lo, hi + 1)
        )

    def apply_clean_interval(lo, hi, finish_time):
        for i in range(lo, hi + 1):
            if status[i] != "unknown":
                continue
            status[i] = "clean"
            cid = batch_sorted[i]["commit_id"]
            if cid not in feedback_times:
                fb_min = (finish_time - batch_sorted[i]["ts"]).total_seconds() / 60.0
                feedback_times[cid] = fb_min

    if int(TKRB_TOP_K) <= 1 or len(top_indices) <= 1:
        for idx in top_indices:
            if status[idx] != "unknown":
                continue

            curr_fails = run_interval(0, idx)

            prev_fails = False
            if curr_fails and idx > 0:
                prev_fails = run_interval(0, idx - 1)

            if curr_fails and not prev_fails:
                mark_culprit(idx, current_time)
            elif curr_fails and prev_fails and idx > 0:
                tob_bisect_on_range(0, idx - 1)
    else:
        # Parallel top-K probes with a barrier (PAR-style): submit all probe runs
        # at the same requested start time, wait until all of them complete,
        # then process the results.
        probe_start = current_time
        latest_finish = probe_start
        probe_has_defect_by_idx = {}

        for idx in top_indices:
            if status[idx] != "unknown":
                continue

            lo, hi = 0, idx
            key = (lo, hi)

            if key in interval_cache:
                probe_has_defect_by_idx[idx] = interval_cache[key]
                continue

            durations = batch_fail_durations
            total_tests_run += len(durations)
            finish_time = run_test_suite(executor, probe_start, durations)
            latest_finish = max(latest_finish, finish_time)

            has_defect = interval_has_defect(lo, hi)
            interval_cache[key] = has_defect
            probe_has_defect_by_idx[idx] = has_defect

        # Barrier: do not react to individual probe results until all probes finish.
        current_time = max(current_time, latest_finish)

        # Process probe outcomes after the barrier.
        for idx in top_indices:
            if status[idx] != "unknown":
                continue

            curr_fails = bool(probe_has_defect_by_idx.get(idx, False))
            if not curr_fails:
                apply_clean_interval(0, idx, current_time)
                continue

            prev_fails = False
            if idx > 0:
                prev_fails = run_interval(0, idx - 1)

            if curr_fails and not prev_fails:
                mark_culprit(idx, current_time)
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

    tests_added = total_tests_run - tests_before
    tests_batch_root = len(get_batch_signature_durations()) if is_batch_root else 0
    tests_bisection = max(0, tests_added - tests_batch_root)
    culprits_batch = len(culprit_times) - culprits_before
    logger.debug(
        "topk_risk_first_bisect: batch_size=%d, tests_batch_root=%d, "
        "tests_bisection=%d, culprits_this_batch=%d, culprits_total=%d, "
        "cache_entries=%d, total_tests_run=%d",
        n,
        tests_batch_root,
        tests_bisection,
        culprits_batch,
        len(culprit_times),
        len(interval_cache),
        total_tests_run,
    )
    return total_tests_run, culprit_times, feedback_times


def sequential_walk_backward_bisect(
    batch,
    start_time,
    total_tests_run,
    culprit_times,
    feedback_times,
    executor,
    is_batch_root=True,
):
    """
    Sequential Walk-Backward Bisection (SWB).

    When a batch is known to contain at least one defect, SWB finds culprits by
    walking backwards from the end of the currently-unknown region and testing
    each prior boundary sequentially.

    Concretely, for an unknown interval [lo, hi] that fails:
      - Treat 'hi' as a known-buggy boundary (the interval [lo, hi] contains a regressor).
      - Test [lo, hi-1], then [lo, hi-2], ... sequentially (waiting for each result).
      - If [lo, k] is clean and [lo, k+1] is buggy, the culprit is k+1.

    Cost/time model matches other strategies:
      - First run in a batch (is_batch_root=True) uses the configured full suite.
      - Subsequent runs use the failing-signature suite for this batch.
    """
    if not batch:
        return total_tests_run, culprit_times, feedback_times

    batch_sorted = batch
    n = len(batch_sorted)
    logger.debug("sequential_walk_backward_bisect: starting batch_size=%d", n)

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

        # Test the entire unknown region first.
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

        # Walk backwards from hi, testing [lo, k] for k=hi-1..lo.
        if lo == hi:
            culprit_idx = lo
        else:
            culprit_idx = lo
            for k in range(hi - 1, lo - 1, -1):
                fails_k, current_time, first_test, total_tests_run = _run_interval_and_update(
                    batch_sorted,
                    status,
                    lo,
                    k,
                    current_time,
                    first_test,
                    total_tests_run,
                    feedback_times,
                    executor,
                    batch_fail_durations,
                )
                if not fails_k:
                    culprit_idx = k + 1
                    break
                if k == lo:
                    culprit_idx = lo

        c = batch_sorted[culprit_idx]
        status[culprit_idx] = "defect_found"

        fb_min = (current_time - c["ts"]).total_seconds() / 60.0
        feedback_times[c["commit_id"]] = fb_min
        if c["true_label"]:
            culprit_times.append(fb_min)

    logger.debug(
        "sequential_walk_backward_bisect: finished batch_size=%d total_tests_run=%d",
        n,
        total_tests_run,
    )
    return total_tests_run, culprit_times, feedback_times


def sequential_walk_forward_bisect(
    batch,
    start_time,
    total_tests_run,
    culprit_times,
    feedback_times,
    executor,
    is_batch_root=True,
):
    """
    Sequential Walk-Forward Bisection (SWF).

    When a batch is known to contain at least one defect, SWF finds culprits by
    walking forwards in time from the beginning of the currently-unknown
    region, testing increasingly larger prefixes sequentially.

    For an unknown interval [lo, hi] that fails:
      - Test [lo, lo], then [lo, lo+1], ... sequentially (waiting for each result).
      - The first index k for which [lo, k] fails is the culprit (since [lo, k-1]
        would have been clean).

    Cost/time model matches other strategies:
      - First run in a batch (is_batch_root=True) uses the configured full suite.
      - Subsequent runs use the failing-signature suite for this batch.
    """
    if not batch:
        return total_tests_run, culprit_times, feedback_times

    batch_sorted = batch
    n = len(batch_sorted)
    logger.debug("sequential_walk_forward_bisect: starting batch_size=%d", n)

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

        # Test the entire unknown region first.
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

        # Walk forwards from lo, testing [lo, k] for k=lo..hi.
        culprit_idx = hi
        for k in range(lo, hi + 1):
            fails_k, current_time, first_test, total_tests_run = _run_interval_and_update(
                batch_sorted,
                status,
                lo,
                k,
                current_time,
                first_test,
                total_tests_run,
                feedback_times,
                executor,
                batch_fail_durations,
            )
            if fails_k:
                culprit_idx = k
                break

        c = batch_sorted[culprit_idx]
        status[culprit_idx] = "defect_found"

        fb_min = (current_time - c["ts"]).total_seconds() / 60.0
        feedback_times[c["commit_id"]] = fb_min
        if c["true_label"]:
            culprit_times.append(fb_min)

    logger.debug(
        "sequential_walk_forward_bisect: finished batch_size=%d total_tests_run=%d",
        n,
        total_tests_run,
    )
    return total_tests_run, culprit_times, feedback_times
