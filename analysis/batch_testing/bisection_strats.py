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
        # Min-heap of (free_time, worker_index) for all workers. Entries may
        # become stale if we cancel work; we use lazy skipping.
        self._worker_heap = []
        self._worker_free_times = []
        self._worker_task_stacks = []
        self._worker_t0 = None
        self._next_task_id = 1
        # task_id -> dict with scheduling metadata
        self._tasks = {}
        # Cumulative CPU time in minutes across all scheduled tests
        self.total_cpu_minutes = 0.0
        logger.debug("Created TestExecutor with %d workers", num_workers)

    def _ensure_initialized(self, t0):
        # Initialize all workers as free at t0 the first time we schedule.
        if not self._worker_heap:
            self._worker_t0 = t0
            self._worker_free_times = [t0 for _ in range(self.num_workers)]
            self._worker_task_stacks = [[] for _ in range(self.num_workers)]
            self._worker_heap = [(t0, i) for i in range(self.num_workers)]
            heapq.heapify(self._worker_heap)

    def _pop_earliest_worker(self):
        while self._worker_heap:
            free_time, idx = heapq.heappop(self._worker_heap)
            if free_time == self._worker_free_times[idx]:
                return free_time, idx
        raise RuntimeError("TestExecutor: worker heap unexpectedly empty.")

    def schedule_task(self, requested_start_time, duration_minutes: float):
        """
        Submit a single test that becomes 'ready' at requested_start_time,
        with its own duration (in minutes).

        Returns (task_id, finish_time).
        """
        self._ensure_initialized(requested_start_time)

        # Find worker that becomes free earliest
        earliest_free, idx = self._pop_earliest_worker()

        actual_start = max(requested_start_time, earliest_free)
        try:
            duration_float = float(duration_minutes)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Invalid duration_minutes value passed to TestExecutor.schedule: {duration_minutes!r}"
            ) from exc

        finish_time = actual_start + timedelta(minutes=duration_float)
        task_id = self._next_task_id
        self._next_task_id += 1

        self._tasks[task_id] = {
            "worker": idx,
            "requested_start": requested_start_time,
            "start": actual_start,
            "finish": finish_time,
            "duration": duration_float,
            "canceled": False,
        }
        self._worker_task_stacks[idx].append(task_id)

        # Update worker free time; reinsert (stale entries are skipped lazily).
        self._worker_free_times[idx] = finish_time
        heapq.heappush(self._worker_heap, (finish_time, idx))

        # Accumulate CPU time regardless of parallelism
        self.total_cpu_minutes += duration_float
        # logger.debug(
        #     "Scheduled test on worker %d: start=%s, duration=%.2f min, finish=%s",
        #     idx,
        #     actual_start,
        #     duration_minutes,
        #     finish_time,
        # )
        return task_id, finish_time

    def schedule(self, requested_start_time, duration_minutes: float):
        """
        Backwards-compatible wrapper around `schedule_task`.

        Returns the actual finish time given current queue & workers.
        """
        _task_id, finish_time = self.schedule_task(requested_start_time, duration_minutes)
        return finish_time

    def get_task(self, task_id: int):
        """
        Return scheduling metadata for a task, or None if not found.

        The returned dict is owned by the executor; treat as read-only.
        """
        return self._tasks.get(task_id)

    def cancel_task(self, task_id: int, now):
        """
        Best-effort cancellation of a single scheduled test.

        Cancellation is only supported when:
          - The task has not started by `now`, and
          - The task is the *last* scheduled task on its worker.

        Returns True if the task was canceled, else False.
        """
        task = self._tasks.get(task_id)
        if not task or task.get("canceled"):
            return False
        start_time = task["start"]
        if start_time <= now:
            return False

        worker = task["worker"]
        stack = self._worker_task_stacks[worker]
        if not stack or stack[-1] != task_id:
            return False

        stack.pop()
        task["canceled"] = True

        # Rewind worker free time to the previous task's finish (or t0).
        if stack:
            prev_task = self._tasks[stack[-1]]
            new_free = prev_task["finish"]
        else:
            new_free = self._worker_t0

        self._worker_free_times[worker] = new_free
        heapq.heappush(self._worker_heap, (new_free, worker))

        # Undo CPU accounting for work that never ran.
        self.total_cpu_minutes -= float(task["duration"])
        if self.total_cpu_minutes < 0:
            self.total_cpu_minutes = 0.0

        return True


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


def _submit_test_suite_with_task_ids(
    executor: TestExecutor, requested_start_time, durations_minutes
):
    """
    Submit a suite of tests and return (last_finish_time, task_ids).

    This is used by speculative/asynchronous strategies that may need to
    cancel individual tests later.
    """
    if not durations_minutes:
        return requested_start_time, []

    last_finish = requested_start_time
    task_ids = []
    for dur in durations_minutes:
        task_id, finish_time = executor.schedule_task(requested_start_time, dur)
        task_ids.append(task_id)
        if finish_time > last_finish:
            last_finish = finish_time
    return last_finish, task_ids


def _tob_worst_case_chain(lo: int, hi: int):
    """
    Worst-case (latest-culprit) sequence of binary-search interval tests.

    For the TOB inner loop:
      test [left..mid]; if clean => left=mid+1 else right=mid

    The worst-case path that delays culprit discovery as long as possible is
    the one where each tested left-half is clean (so the culprit is in the
    final rightmost segment).
    """
    chain = []
    left = int(lo)
    right = int(hi)
    while left < right:
        mid = (left + right) // 2
        chain.append((left, mid))
        left = mid + 1
    return chain


def _rwab_worst_case_chain(lo: int, hi: int, choose_split_fn):
    """
    Worst-case (max-tests) sequence for RWAB's split-based search.

    RWAB tests [left..split]; if that interval is clean it continues on
    [split+1..right]. The path that delays culprit identification is the one
    where each tested left interval is clean, pushing the search to the
    rightmost segment.
    """
    chain = []
    left = int(lo)
    right = int(hi)
    while left < right:
        split = int(choose_split_fn(left, right))
        chain.append((left, split))
        left = split + 1
    return chain


def _suite_submit(
    executor: TestExecutor,
    requested_start_time,
    durations,
    kind: str,
    lo: int,
    hi: int,
):
    finish, task_ids = _submit_test_suite_with_task_ids(
        executor, requested_start_time, durations
    )
    return {
        "kind": kind,
        "lo": int(lo),
        "hi": int(hi),
        "finish": finish,
        "task_ids": task_ids,
        "done": False,
        "canceled": False,
    }


def _interval_has_defect(batch_sorted, status, lo, hi):
    return any(
        batch_sorted[i]["true_label"] and status[i] != "defect_found"
        for i in range(lo, hi + 1)
    )


def _mark_interval_clean(batch_sorted, status, lo, hi, finish_time, feedback_times):
    for i in range(lo, hi + 1):
        if status[i] != "unknown":
            continue
        status[i] = "clean"
        cid = batch_sorted[i]["commit_id"]
        if cid not in feedback_times:
            fb_min = (finish_time - batch_sorted[i]["ts"]).total_seconds() / 60.0
            feedback_times[cid] = fb_min


def _cancel_suites_best_effort(executor: TestExecutor, suites, now):
    """
    Attempt to cancel (unstarted) tests from suites in reverse submission order.

    Returns the number of individual tests canceled.
    """
    canceled_tests = 0
    # Cancel newest-first so tasks are most likely to be at the end of each
    # worker's scheduled stack (a requirement for `cancel_task`).
    for suite in reversed(suites):
        if suite.get("done") or suite.get("canceled"):
            continue
        suite["canceled"] = True
        task_ids = suite.get("task_ids") or []
        for task_id in reversed(task_ids):
            if executor.cancel_task(task_id, now):
                canceled_tests += 1
    return canceled_tests


def _speculative_tob_find_one_in_bounds(
    batch_sorted,
    status,
    lo_bound: int,
    hi_bound: int,
    start_time,
    first_test: bool,
    total_tests_run: int,
    feedback_times,
    executor: TestExecutor,
    batch_fail_durations,
    interval_finish_cache=None,
):
    """
    Find at most one culprit within [lo_bound..hi_bound] using TOB logic, but
    submit interval suites speculatively (no per-step waiting).

    Returns:
      (found_culprit: bool, decision_time, first_test, total_tests_run, culprit_idx_or_none)
    """
    if lo_bound > hi_bound:
        return False, start_time, first_test, total_tests_run, None

    unknown_indices = [
        i for i in range(lo_bound, hi_bound + 1) if status[i] == "unknown"
    ]
    if not unknown_indices:
        return False, start_time, first_test, total_tests_run, None

    lo = unknown_indices[0]
    hi = unknown_indices[-1]

    suites = []
    suite_events = []

    def cached_finish(lo_i, hi_i):
        if interval_finish_cache is None:
            return None
        return interval_finish_cache.get((int(lo_i), int(hi_i)))

    if first_test:
        durations_region = get_batch_signature_durations()
        total_tests_run += len(durations_region)
        region_suite = _suite_submit(
            executor, start_time, durations_region, "region", lo, hi
        )
        first_test = False
    else:
        fin = cached_finish(lo, hi)
        if fin is not None:
            region_suite = {
                "kind": "region",
                "lo": int(lo),
                "hi": int(hi),
                "finish": fin,
                "task_ids": [],
                "done": False,
                "canceled": False,
                "cached": True,
            }
        else:
            durations_region = batch_fail_durations
            total_tests_run += len(durations_region)
            region_suite = _suite_submit(
                executor, start_time, durations_region, "region", lo, hi
            )

    suites.append(region_suite)
    heapq.heappush(suite_events, (region_suite["finish"], len(suites) - 1))

    chain = _tob_worst_case_chain(lo, hi)
    for l, m in chain:
        fin = cached_finish(l, m)
        if fin is not None:
            suite = {
                "kind": "bin",
                "lo": int(l),
                "hi": int(m),
                "finish": fin,
                "task_ids": [],
                "done": False,
                "canceled": False,
                "cached": True,
            }
        else:
            total_tests_run += len(batch_fail_durations)
            suite = _suite_submit(
                executor, start_time, batch_fail_durations, "bin", l, m
            )
        suites.append(suite)
        heapq.heappush(suite_events, (suite["finish"], len(suites) - 1))

    region_result = None
    left = lo
    right = hi

    while suite_events:
        finish_time, suite_idx = heapq.heappop(suite_events)
        suite = suites[suite_idx]
        if suite.get("done") or suite.get("canceled"):
            continue

        suite["done"] = True
        interval_lo = suite["lo"]
        interval_hi = suite["hi"]
        has_defect = _interval_has_defect(batch_sorted, status, interval_lo, interval_hi)

        if not has_defect:
            _mark_interval_clean(
                batch_sorted, status, interval_lo, interval_hi, finish_time, feedback_times
            )

        if suite["kind"] == "region":
            region_result = has_defect
            if not has_defect:
                effective_now = max(start_time, finish_time)
                canceled = _cancel_suites_best_effort(executor, suites, effective_now)
                total_tests_run -= canceled
                return False, effective_now, first_test, total_tests_run, None

            if left == right:
                idx = left
                c = batch_sorted[idx]
                status[idx] = "defect_found"
                fb_min = (finish_time - c["ts"]).total_seconds() / 60.0
                feedback_times[c["commit_id"]] = fb_min
                effective_now = max(start_time, finish_time)
                canceled = _cancel_suites_best_effort(executor, suites, effective_now)
                total_tests_run -= canceled
                return True, effective_now, first_test, total_tests_run, idx
            continue

        if region_result is not True or left >= right:
            continue

        mid = (left + right) // 2
        if interval_lo != left or interval_hi != mid:
            continue

        if has_defect:
            right = mid
            effective_now = max(start_time, finish_time)
            canceled = _cancel_suites_best_effort(executor, suites, effective_now)
            total_tests_run -= canceled

            if left == right:
                idx = left
                c = batch_sorted[idx]
                status[idx] = "defect_found"
                fb_min = (finish_time - c["ts"]).total_seconds() / 60.0
                feedback_times[c["commit_id"]] = fb_min
                return True, effective_now, first_test, total_tests_run, idx

            chain = _tob_worst_case_chain(left, right)
            for l, m in chain:
                fin = cached_finish(l, m)
                if fin is not None:
                    new_suite = {
                        "kind": "bin",
                        "lo": int(l),
                        "hi": int(m),
                        "finish": fin,
                        "task_ids": [],
                        "done": False,
                        "canceled": False,
                        "cached": True,
                    }
                else:
                    total_tests_run += len(batch_fail_durations)
                    new_suite = _suite_submit(
                        executor, finish_time, batch_fail_durations, "bin", l, m
                    )
                suites.append(new_suite)
                heapq.heappush(
                    suite_events, (new_suite["finish"], len(suites) - 1)
                )
            continue

        # Clean left-half.
        left = mid + 1
        if left == right:
            idx = left
            c = batch_sorted[idx]
            status[idx] = "defect_found"
            fb_min = (finish_time - c["ts"]).total_seconds() / 60.0
            feedback_times[c["commit_id"]] = fb_min
            effective_now = max(start_time, finish_time)
            canceled = _cancel_suites_best_effort(executor, suites, effective_now)
            total_tests_run -= canceled
            return True, effective_now, first_test, total_tests_run, idx

    # Should not happen; treat as no-culprit.
    if region_suite.get("finish"):
        return False, region_suite["finish"], first_test, total_tests_run, None
    return False, start_time, first_test, total_tests_run, None


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

        # ---- Speculative TOB scheduling (no waiting) ----
        #
        # We submit the tests required by TOB without waiting for each suite's
        # results before submitting the next. Concretely:
        #   - Submit the whole-unknown-region test for [lo..hi].
        #   - Speculatively submit the worst-case (latest-culprit) chain of
        #     midpoint tests for the inner binary search.
        # As results come in, we update the search range; if the culprit is
        # identified, we best-effort cancel any unnecessary pending tests.

        suites = []
        suite_events = []

        # 1) Whole unknown region test (full suite only for the first test of the batch).
        if first_test:
            durations_region = get_batch_signature_durations()
        else:
            durations_region = batch_fail_durations

        total_tests_run += len(durations_region)
        region_finish, region_task_ids = _submit_test_suite_with_task_ids(
            executor, current_time, durations_region
        )
        first_test = False
        region_suite = {
            "kind": "region",
            "lo": lo,
            "hi": hi,
            "finish": region_finish,
            "task_ids": region_task_ids,
            "done": False,
            "canceled": False,
        }
        suites.append(region_suite)
        heapq.heappush(suite_events, (region_finish, len(suites) - 1))

        # 2) Speculative "worst-case" chain for the binary search midpoint tests.
        chain = _tob_worst_case_chain(lo, hi)
        for l, m in chain:
            total_tests_run += len(batch_fail_durations)
            finish, task_ids = _submit_test_suite_with_task_ids(
                executor, current_time, batch_fail_durations
            )
            suite = {
                "kind": "bin",
                "lo": l,
                "hi": m,
                "finish": finish,
                "task_ids": task_ids,
                "done": False,
                "canceled": False,
            }
            suites.append(suite)
            heapq.heappush(suite_events, (finish, len(suites) - 1))

        # 3) Consume results in finish-time order, updating the active bisect range.
        region_result = None
        left = lo
        right = hi
        resolved = False

        while suite_events and not resolved:
            finish_time, suite_idx = heapq.heappop(suite_events)
            suite = suites[suite_idx]
            if suite.get("done") or suite.get("canceled"):
                continue

            suite["done"] = True
            interval_lo = suite["lo"]
            interval_hi = suite["hi"]
            has_defect = _interval_has_defect(batch_sorted, status, interval_lo, interval_hi)

            if not has_defect:
                _mark_interval_clean(
                    batch_sorted, status, interval_lo, interval_hi, finish_time, feedback_times
                )

            if suite["kind"] == "region":
                region_result = has_defect

                if not has_defect:
                    # Entire region is clean; cancel remaining speculative work.
                    canceled = _cancel_suites_best_effort(executor, suites, finish_time)
                    total_tests_run -= canceled
                    current_time = finish_time
                    resolved = True
                    break

                # Degenerate: single unknown commit and region fails => culprit found.
                if left == right:
                    idx = left
                    c = batch_sorted[idx]
                    status[idx] = "defect_found"
                    fb_min = (finish_time - c["ts"]).total_seconds() / 60.0
                    feedback_times[c["commit_id"]] = fb_min
                    if c["true_label"]:
                        culprit_times.append(fb_min)

                    canceled = _cancel_suites_best_effort(executor, suites, finish_time)
                    total_tests_run -= canceled
                    current_time = finish_time
                    resolved = True
                    break

                continue

            # Binary-search step results only affect the search after we know the region fails.
            if region_result is not True or left >= right:
                continue

            mid = (left + right) // 2
            if interval_lo != left or interval_hi != mid:
                # Result for a speculative interval that is not on the active path.
                continue

            if has_defect:
                # Culprit lies in the left half; cancel outstanding right-path work and re-plan.
                right = mid
                canceled = _cancel_suites_best_effort(executor, suites, finish_time)
                total_tests_run -= canceled

                if left == right:
                    idx = left
                    c = batch_sorted[idx]
                    status[idx] = "defect_found"
                    fb_min = (finish_time - c["ts"]).total_seconds() / 60.0
                    feedback_times[c["commit_id"]] = fb_min
                    if c["true_label"]:
                        culprit_times.append(fb_min)
                    current_time = finish_time
                    resolved = True
                    break

                # Submit a new worst-case chain for the narrowed interval, without waiting.
                chain = _tob_worst_case_chain(left, right)
                for l, m in chain:
                    total_tests_run += len(batch_fail_durations)
                    finish, task_ids = _submit_test_suite_with_task_ids(
                        executor, finish_time, batch_fail_durations
                    )
                    new_suite = {
                        "kind": "bin",
                        "lo": l,
                        "hi": m,
                        "finish": finish,
                        "task_ids": task_ids,
                        "done": False,
                        "canceled": False,
                    }
                    suites.append(new_suite)
                    heapq.heappush(suite_events, (finish, len(suites) - 1))

                continue

            # Clean left-half: culprit is to the right.
            left = mid + 1
            if left == right:
                idx = left
                c = batch_sorted[idx]
                status[idx] = "defect_found"
                fb_min = (finish_time - c["ts"]).total_seconds() / 60.0
                feedback_times[c["commit_id"]] = fb_min
                if c["true_label"]:
                    culprit_times.append(fb_min)

                canceled = _cancel_suites_best_effort(executor, suites, finish_time)
                total_tests_run -= canceled
                current_time = finish_time
                resolved = True
                break

        if not resolved:
            # Shouldn't happen: we always schedule at least the region suite.
            current_time = max(current_time, region_finish)
            break

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

        suites = []
        suite_events = []

        # ---- Speculative RWAB scheduling (no waiting) ----
        if first_test:
            durations_region = get_batch_signature_durations()
        else:
            durations_region = batch_fail_durations

        total_tests_run += len(durations_region)
        region_suite = _suite_submit(
            executor, current_time, durations_region, "region", lo, hi
        )
        first_test = False
        suites.append(region_suite)
        heapq.heappush(suite_events, (region_suite["finish"], len(suites) - 1))

        chain = _rwab_worst_case_chain(lo, hi, choose_risk_balanced_split)
        for l, s in chain:
            total_tests_run += len(batch_fail_durations)
            suite = _suite_submit(
                executor, current_time, batch_fail_durations, "bin", l, s
            )
            suites.append(suite)
            heapq.heappush(suite_events, (suite["finish"], len(suites) - 1))

        region_result = None
        left = lo
        right = hi
        resolved = False

        while suite_events and not resolved:
            finish_time, suite_idx = heapq.heappop(suite_events)
            suite = suites[suite_idx]
            if suite.get("done") or suite.get("canceled"):
                continue

            suite["done"] = True
            interval_lo = suite["lo"]
            interval_hi = suite["hi"]
            has_defect = _interval_has_defect(batch_sorted, status, interval_lo, interval_hi)

            if not has_defect:
                _mark_interval_clean(
                    batch_sorted, status, interval_lo, interval_hi, finish_time, feedback_times
                )

            if suite["kind"] == "region":
                region_result = has_defect
                if not has_defect:
                    canceled = _cancel_suites_best_effort(executor, suites, finish_time)
                    total_tests_run -= canceled
                    current_time = finish_time
                    resolved = True
                    break

                if left == right:
                    idx = left
                    c = batch_sorted[idx]
                    status[idx] = "defect_found"
                    fb_min = (finish_time - c["ts"]).total_seconds() / 60.0
                    feedback_times[c["commit_id"]] = fb_min
                    if c["true_label"]:
                        culprit_times.append(fb_min)

                    canceled = _cancel_suites_best_effort(executor, suites, finish_time)
                    total_tests_run -= canceled
                    current_time = finish_time
                    resolved = True
                    break

                continue

            if region_result is not True or left >= right:
                continue

            expected_split = choose_risk_balanced_split(left, right)
            if interval_lo != left or interval_hi != expected_split:
                continue

            if has_defect:
                # Need the left side; cancel right-path speculative work and replan.
                right = expected_split
                canceled = _cancel_suites_best_effort(executor, suites, finish_time)
                total_tests_run -= canceled

                if left == right:
                    idx = left
                    c = batch_sorted[idx]
                    status[idx] = "defect_found"
                    fb_min = (finish_time - c["ts"]).total_seconds() / 60.0
                    feedback_times[c["commit_id"]] = fb_min
                    if c["true_label"]:
                        culprit_times.append(fb_min)
                    current_time = finish_time
                    resolved = True
                    break

                chain = _rwab_worst_case_chain(left, right, choose_risk_balanced_split)
                for l, s in chain:
                    total_tests_run += len(batch_fail_durations)
                    new_suite = _suite_submit(
                        executor, finish_time, batch_fail_durations, "bin", l, s
                    )
                    suites.append(new_suite)
                    heapq.heappush(
                        suite_events, (new_suite["finish"], len(suites) - 1)
                    )

                continue

            # Clean left side.
            left = expected_split + 1
            if left == right:
                idx = left
                c = batch_sorted[idx]
                status[idx] = "defect_found"
                fb_min = (finish_time - c["ts"]).total_seconds() / 60.0
                feedback_times[c["commit_id"]] = fb_min
                if c["true_label"]:
                    culprit_times.append(fb_min)

                canceled = _cancel_suites_best_effort(executor, suites, finish_time)
                total_tests_run -= canceled
                current_time = finish_time
                resolved = True
                break

        if region_result is False:
            break

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

    batch_fail_durations = get_failing_signature_durations_for_batch(batch_sorted)
    logger.debug(
        "topk_risk_first_bisect: TKRB_TOP_K=%d, failing_signatures=%d",
        int(TKRB_TOP_K),
        len(batch_fail_durations),
    )

    def mark_culprit(idx, at_time):
        c = batch_sorted[idx]
        if status[idx] == "defect_found":
            return
        status[idx] = "defect_found"
        fb_min = (at_time - c["ts"]).total_seconds() / 60.0
        feedback_times[c["commit_id"]] = fb_min
        if c["true_label"]:
            culprit_times.append(fb_min)

    # ---- Step 1 + Step 2: Submit whole test + probes without waiting ----
    suites = []
    suite_events = []
    probe_records = {}  # (kind, idx) -> (has_defect, finish_time)
    processed_probe_idxs = set()
    interval_finish_cache = {}  # (lo, hi) -> finish_time for completed targeted intervals

    submit_time = start_time

    # Whole-batch suite (full suite for batch-root; otherwise failing-suite).
    if first_test:
        durations_whole = get_batch_signature_durations()
    else:
        durations_whole = batch_fail_durations
    total_tests_run += len(durations_whole)
    whole_suite = _suite_submit(
        executor, submit_time, durations_whole, "whole", 0, n - 1
    )
    first_test = False
    suites.append(whole_suite)
    heapq.heappush(suite_events, (whole_suite["finish"], 0))

    # Compute top-K indices by risk.
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

    # Submit probe suites upfront (curr=[0..idx], prev=[0..idx-1]).
    for idx in top_indices:
        if idx < 0 or idx >= n:
            continue

        total_tests_run += len(batch_fail_durations)
        suite = _suite_submit(
            executor, submit_time, batch_fail_durations, "probe_curr", 0, idx
        )
        suites.append(suite)
        suite_idx = len(suites) - 1
        heapq.heappush(suite_events, (suite["finish"], suite_idx))

        if idx > 0:
            total_tests_run += len(batch_fail_durations)
            suite = _suite_submit(
                executor, submit_time, batch_fail_durations, "probe_prev", 0, idx - 1
            )
            suites.append(suite)
            suite_idx = len(suites) - 1
            heapq.heappush(suite_events, (suite["finish"], suite_idx))

    whole_result = None
    whole_finish = None

    def apply_probe_clean_results_if_any():
        for (kind, idx), (has_defect, finish_time) in list(probe_records.items()):
            if bool(has_defect):
                continue
            if kind == "curr":
                _mark_interval_clean(
                    batch_sorted, status, 0, idx, finish_time, feedback_times
                )
            elif kind == "prev" and idx > 0:
                _mark_interval_clean(
                    batch_sorted, status, 0, idx - 1, finish_time, feedback_times
                )

    def try_resolve_probes():
        nonlocal current_time, total_tests_run
        if whole_result is not True:
            return

        for idx in top_indices:
            if idx in processed_probe_idxs:
                continue
            if status[idx] != "unknown":
                processed_probe_idxs.add(idx)
                continue

            curr = probe_records.get(("curr", idx))
            if curr is None:
                continue
            curr_fails, curr_finish = bool(curr[0]), curr[1]
            if not curr_fails:
                processed_probe_idxs.add(idx)
                continue

            if idx == 0:
                decision_time = max(whole_finish, curr_finish)
                mark_culprit(0, decision_time)
                processed_probe_idxs.add(idx)
                continue

            prev = probe_records.get(("prev", idx))
            if prev is None:
                continue
            prev_fails, prev_finish = bool(prev[0]), prev[1]
            decision_time = max(whole_finish, curr_finish, prev_finish)

            if not prev_fails:
                mark_culprit(idx, decision_time)
                processed_probe_idxs.add(idx)
                continue

            # Both fail => run TOB-style search over [0..idx-1] (in-place on status).
            sub_time = decision_time
            while True:
                found, sub_time, _unused_first, total_tests_run, culprit_idx = (
                    _speculative_tob_find_one_in_bounds(
                        batch_sorted,
                        status,
                        0,
                        idx - 1,
                        sub_time,
                        False,  # first_test=False (whole-batch suite already ran)
                        total_tests_run,
                        feedback_times,
                        executor,
                        batch_fail_durations,
                        interval_finish_cache=interval_finish_cache,
                    )
                )
                if not found:
                    break
                if culprit_idx is not None and batch_sorted[culprit_idx]["true_label"]:
                    cid = batch_sorted[culprit_idx]["commit_id"]
                    fb = feedback_times.get(cid)
                    if fb is not None:
                        culprit_times.append(float(fb))

            current_time = max(current_time, sub_time)
            processed_probe_idxs.add(idx)

    while suite_events:
        finish_time, suite_idx = heapq.heappop(suite_events)
        suite = suites[suite_idx]
        if suite.get("done") or suite.get("canceled"):
            continue

        suite["done"] = True
        interval_lo = suite["lo"]
        interval_hi = suite["hi"]
        has_defect = _interval_has_defect(batch_sorted, status, interval_lo, interval_hi)

        if suite["kind"] == "whole":
            whole_result = has_defect
            whole_finish = finish_time

            if not has_defect:
                _mark_interval_clean(
                    batch_sorted, status, interval_lo, interval_hi, finish_time, feedback_times
                )
                canceled = _cancel_suites_best_effort(executor, suites, finish_time)
                total_tests_run -= canceled
                current_time = finish_time
                break

            current_time = max(current_time, finish_time)
            apply_probe_clean_results_if_any()
            try_resolve_probes()
            continue

        if suite["kind"] == "probe_curr":
            probe_records[("curr", interval_hi)] = (has_defect, finish_time)
            interval_finish_cache[(0, interval_hi)] = finish_time
        elif suite["kind"] == "probe_prev":
            probe_records[("prev", interval_hi + 1)] = (has_defect, finish_time)
            interval_finish_cache[(0, interval_hi)] = finish_time

        if whole_result is not True:
            continue

        apply_probe_clean_results_if_any()
        try_resolve_probes()

    # ---- Step 3: Fall back to TOB-style search over remaining unknowns ----
    if whole_result is True:
        while True:
            unknown_indices = [i for i, s in enumerate(status) if s == "unknown"]
            if not unknown_indices:
                break

            found, current_time, _unused_first, total_tests_run, culprit_idx = (
                _speculative_tob_find_one_in_bounds(
                    batch_sorted,
                    status,
                    0,
                    n - 1,
                    current_time,
                    False,  # no more full-suite runs in this batch
                    total_tests_run,
                    feedback_times,
                    executor,
                    batch_fail_durations,
                    interval_finish_cache=interval_finish_cache,
                )
            )
            if not found:
                break
            if culprit_idx is not None and batch_sorted[culprit_idx]["true_label"]:
                cid = batch_sorted[culprit_idx]["commit_id"]
                fb = feedback_times.get(cid)
                if fb is not None:
                    culprit_times.append(float(fb))

    tests_added = total_tests_run - tests_before
    tests_batch_root = len(get_batch_signature_durations()) if is_batch_root else 0
    tests_bisection = max(0, tests_added - tests_batch_root)
    culprits_batch = len(culprit_times) - culprits_before
    logger.debug(
        "topk_risk_first_bisect: batch_size=%d, tests_batch_root=%d, "
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

        suites = []
        suite_events = []
        results_by_hi = {}
        region_result = None

        # ---- Submit all SWB tests without waiting (worst-case) ----
        if first_test:
            durations_region = get_batch_signature_durations()
        else:
            durations_region = batch_fail_durations

        total_tests_run += len(durations_region)
        region_suite = _suite_submit(
            executor, current_time, durations_region, "region", lo, hi
        )
        first_test = False
        suites.append(region_suite)
        heapq.heappush(suite_events, (region_suite["finish"], len(suites) - 1))

        # Worst-case for SWB (max tests): keep failing until k==lo.
        for k in range(hi - 1, lo - 1, -1):
            total_tests_run += len(batch_fail_durations)
            suite = _suite_submit(
                executor, current_time, batch_fail_durations, "walk", lo, k
            )
            suites.append(suite)
            heapq.heappush(suite_events, (suite["finish"], len(suites) - 1))

        # ---- Consume results as they finish; cancel remaining when resolved ----
        resolved = False
        region_fail = False

        def maybe_resolve_boundary():
            nonlocal resolved, current_time, total_tests_run
            if not region_fail or resolved:
                return

            # Culprit is the smallest index b such that [lo..b] fails and
            # [lo..b-1] is clean. For SWB we detect it via the boundary:
            # clean at k and fail at k+1 => culprit k+1. k==lo-1 is implicit.
            lo_result = results_by_hi.get(lo)
            if lo_result is not None and bool(lo_result[0]):
                decision_time = lo_result[1]
                c = batch_sorted[lo]
                if status[lo] != "defect_found":
                    status[lo] = "defect_found"
                    fb_min = (decision_time - c["ts"]).total_seconds() / 60.0
                    feedback_times[c["commit_id"]] = fb_min
                    if c["true_label"]:
                        culprit_times.append(fb_min)

                canceled = _cancel_suites_best_effort(executor, suites, decision_time)
                total_tests_run -= canceled
                current_time = decision_time
                resolved = True
                return

            for k in range(lo, hi):
                clean_k = results_by_hi.get(k)
                fail_k1 = results_by_hi.get(k + 1)
                if clean_k is None or fail_k1 is None:
                    continue
                if bool(clean_k[0]) is False and bool(fail_k1[0]) is True:
                    culprit_idx = k + 1
                    decision_time = max(clean_k[1], fail_k1[1])
                    c = batch_sorted[culprit_idx]
                    if status[culprit_idx] != "defect_found":
                        status[culprit_idx] = "defect_found"
                        fb_min = (decision_time - c["ts"]).total_seconds() / 60.0
                        feedback_times[c["commit_id"]] = fb_min
                        if c["true_label"]:
                            culprit_times.append(fb_min)

                    canceled = _cancel_suites_best_effort(executor, suites, decision_time)
                    total_tests_run -= canceled
                    current_time = decision_time
                    resolved = True
                    return

        while suite_events and not resolved:
            finish_time, suite_idx = heapq.heappop(suite_events)
            suite = suites[suite_idx]
            if suite.get("done") or suite.get("canceled"):
                continue

            suite["done"] = True
            interval_lo = suite["lo"]
            interval_hi = suite["hi"]
            has_defect = _interval_has_defect(batch_sorted, status, interval_lo, interval_hi)

            if not has_defect:
                _mark_interval_clean(
                    batch_sorted, status, interval_lo, interval_hi, finish_time, feedback_times
                )

            results_by_hi[interval_hi] = (has_defect, finish_time)

            if suite["kind"] == "region":
                region_result = has_defect
                if not has_defect:
                    canceled = _cancel_suites_best_effort(executor, suites, finish_time)
                    total_tests_run -= canceled
                    current_time = finish_time
                    resolved = True
                    break
                region_fail = True

            maybe_resolve_boundary()

        if region_result is False:
            break

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

        suites = []
        suite_events = []
        results_by_hi = {}
        region_result = None

        # ---- Submit all SWF tests without waiting (worst-case) ----
        if first_test:
            durations_region = get_batch_signature_durations()
        else:
            durations_region = batch_fail_durations

        total_tests_run += len(durations_region)
        region_suite = _suite_submit(
            executor, current_time, durations_region, "region", lo, hi
        )
        first_test = False
        suites.append(region_suite)
        heapq.heappush(suite_events, (region_suite["finish"], len(suites) - 1))

        # Worst-case for SWF (max tests): keep clean until k==hi.
        for k in range(lo, hi):
            total_tests_run += len(batch_fail_durations)
            suite = _suite_submit(
                executor, current_time, batch_fail_durations, "walk", lo, k
            )
            suites.append(suite)
            heapq.heappush(suite_events, (suite["finish"], len(suites) - 1))

        resolved = False
        region_fail = False

        def maybe_resolve_boundary():
            nonlocal resolved, current_time, total_tests_run
            if not region_fail or resolved:
                return

            # Culprit is the smallest index b such that [lo..b] fails and
            # [lo..b-1] is clean. For SWF this is the boundary:
            # fail at k and clean at k-1 => culprit k. lo-1 is implicit.
            lo_result = results_by_hi.get(lo)
            if lo_result is not None and bool(lo_result[0]):
                decision_time = lo_result[1]
                c = batch_sorted[lo]
                if status[lo] != "defect_found":
                    status[lo] = "defect_found"
                    fb_min = (decision_time - c["ts"]).total_seconds() / 60.0
                    feedback_times[c["commit_id"]] = fb_min
                    if c["true_label"]:
                        culprit_times.append(fb_min)

                canceled = _cancel_suites_best_effort(executor, suites, decision_time)
                total_tests_run -= canceled
                current_time = decision_time
                resolved = True
                return

            for k in range(lo + 1, hi + 1):
                fail_k = results_by_hi.get(k)
                clean_prev = results_by_hi.get(k - 1)
                if fail_k is None or clean_prev is None:
                    continue
                if bool(fail_k[0]) is True and bool(clean_prev[0]) is False:
                    culprit_idx = k
                    decision_time = max(fail_k[1], clean_prev[1])
                    c = batch_sorted[culprit_idx]
                    if status[culprit_idx] != "defect_found":
                        status[culprit_idx] = "defect_found"
                        fb_min = (decision_time - c["ts"]).total_seconds() / 60.0
                        feedback_times[c["commit_id"]] = fb_min
                        if c["true_label"]:
                            culprit_times.append(fb_min)

                    canceled = _cancel_suites_best_effort(executor, suites, decision_time)
                    total_tests_run -= canceled
                    current_time = decision_time
                    resolved = True
                    return

        while suite_events and not resolved:
            finish_time, suite_idx = heapq.heappop(suite_events)
            suite = suites[suite_idx]
            if suite.get("done") or suite.get("canceled"):
                continue

            suite["done"] = True
            interval_lo = suite["lo"]
            interval_hi = suite["hi"]
            has_defect = _interval_has_defect(batch_sorted, status, interval_lo, interval_hi)

            if not has_defect:
                _mark_interval_clean(
                    batch_sorted, status, interval_lo, interval_hi, finish_time, feedback_times
                )

            results_by_hi[interval_hi] = (has_defect, finish_time)

            if suite["kind"] == "region":
                region_result = has_defect
                if not has_defect:
                    canceled = _cancel_suites_best_effort(executor, suites, finish_time)
                    total_tests_run -= canceled
                    current_time = finish_time
                    resolved = True
                    break
                region_fail = True

            maybe_resolve_boundary()

        if region_result is False:
            break

    logger.debug(
        "sequential_walk_forward_bisect: finished batch_size=%d total_tests_run=%d",
        n,
        total_tests_run,
    )
    return total_tests_run, culprit_times, feedback_times
