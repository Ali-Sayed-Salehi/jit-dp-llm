"""
Batching strategies for the batch-testing simulator.

Batching strategies decide *when* to trigger a batch test and *which commits*
are included in each batch. When a batch is flushed, the chosen bisection
strategy (`bisect_fn`) is invoked to identify culprit commits within that batch.

See `analysis/batch_testing/README.md` for a detailed conceptual overview of
all batching and bisection strategies and the shared time/cost model.
"""

from datetime import timedelta
import math
import logging

from bisection_strats import (
    TestExecutor,
    run_test_suite,
    get_tested_signatures_for_revision,
    get_signature_durations_for_ids,
    get_failing_signature_groups_for_revision,
)


logger = logging.getLogger(__name__)

def _accumulate_log_survival(log_survival: float, p: float) -> float:
    """
    Update log survival mass:

        log_survival := Σ log(1 - p_i)

    using `log1p` for numerical stability.

    `p` is expected to be in [0, 1]. If p == 1, survival becomes 0 (log=-inf).
    """
    p = float(p)
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Risk probabilities must be in [0,1], got {p}")
    if p >= 1.0:
        return float("-inf")
    if math.isinf(log_survival) and log_survival < 0:
        return float("-inf")
    return float(log_survival + math.log1p(-p))


def _combined_probability_from_log_survival(log_survival: float) -> float:
    """
    Convert log survival mass to combined probability:

        combined = 1 - exp(log_survival)
    """
    if math.isinf(log_survival) and log_survival < 0:
        return 1.0
    return float(1.0 - math.exp(float(log_survival)))


def _percentile_linear(sorted_values, p: float) -> float:
    """
    Compute the p-th percentile (0-100) using linear interpolation.

    Expects `sorted_values` to be a non-decreasing sequence.
    Returns 0.0 for empty inputs.
    """
    if not sorted_values:
        return 0.0

    p = float(p)
    if p <= 0.0:
        return float(sorted_values[0])
    if p >= 100.0:
        return float(sorted_values[-1])

    n = len(sorted_values)
    # Interpolate between ranks over [0, n-1].
    pos = (n - 1) * (p / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])

    frac = pos - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac)


def build_results(total_tests_run, culprit_times, feedback_times, total_cpu_time_min):
    """
    Build the standard results payload returned by batching simulations.

    Parameters
    ----------
    total_tests_run:
        Total number of perf job executions (signature-group runs, across all suite runs).
    culprit_times:
        List of minutes-to-culprit for regressors that were found.
    feedback_times:
        Mapping `commit_id -> minutes-to-feedback` for commits that received
        feedback during the simulation.
    total_cpu_time_min:
        Cumulative CPU minutes across all scheduled tests (sums durations,
        independent of parallelism).

    Returns
    -------
    dict with metrics in minutes.
    """
    if feedback_times:
        mean_fb = sum(feedback_times.values()) / len(feedback_times)
    else:
        mean_fb = 0.0

    if culprit_times:
        culprit_times_sorted = sorted(float(x) for x in culprit_times)
        mean_ttc = sum(culprit_times_sorted) / len(culprit_times_sorted)
        max_ttc = culprit_times_sorted[-1]
        p90_ttc = _percentile_linear(culprit_times_sorted, 90)
        p95_ttc = _percentile_linear(culprit_times_sorted, 95)
        p99_ttc = _percentile_linear(culprit_times_sorted, 99)
    else:
        mean_ttc = 0.0
        max_ttc = 0.0
        p90_ttc = 0.0
        p95_ttc = 0.0
        p99_ttc = 0.0

    return {
        "total_tests_run": total_tests_run,
        "mean_feedback_time_min": round(mean_fb, 2),
        "mean_time_to_culprit_min": round(mean_ttc, 2),
        "max_time_to_culprit_min": round(max_ttc, 2),
        "p90_time_to_culprit_min": round(p90_ttc, 2),
        "p95_time_to_culprit_min": round(p95_ttc, 2),
        "p99_time_to_culprit_min": round(p99_ttc, 2),
        "total_cpu_time_min": round(float(total_cpu_time_min), 2),
    }

def _union_tested_signature_group_ids_for_commits(commits):
    """
    Compute the union of perf signature-group IDs that Mozilla actually tested across
    the given commits.

    Used by the "-s" batching variants, where each batch's initial suite is the
    union of signature-groups observed in that batch (instead of the global full
    suite). Group membership comes from
    `perf_jobs_per_revision_details_rectified.jsonl` via
    `get_tested_signatures_for_revision`.
    """
    sig_group_ids = set()
    for c in commits:
        rev = c["commit_id"]
        for sig_group in get_tested_signatures_for_revision(rev):
            try:
                sig_group_ids.add(int(sig_group))
            except (TypeError, ValueError):
                continue
    return sig_group_ids


def _simulate_signature_union_driver(commits, bisect_fn, batches, num_workers):
    """
    Shared implementation for the "-s" batching variants.

    Each "batch test" executes only the union of perf signature-groups that Mozilla
    actually ran for the revisions in that batch (from
    perf_jobs_per_revision_details_rectified.jsonl, interpreted as
    signature-groups). A regressor is detected by a batch run only if that
    union includes at least one of the regressor's failing signature-groups
    (derived from alert_summary_fail_perf_sigs.csv via sig_groups.jsonl).

    When a batch run detects at least one regressor, we trigger a bisection
    between the last clean revision (per failing signature-group) and the end of the
    batch, using `bisect_fn`. The bisection itself uses failing signature-groups
    (derived from failing signatures via bisection_strats), and we treat
    the batch run as the "root"
    failure so bisection is invoked with is_batch_root=False.
    """
    total_tests_run = 0
    culprit_times = []
    feedback_times = {}

    if not commits:
        return build_results(total_tests_run, culprit_times, feedback_times, 0.0)

    executor = TestExecutor(num_workers)

    last_seen_batch_end_idx_by_sig = {}
    regressor_fail_sigs_by_idx = {}
    last_clean_for_regressor_by_idx = {}
    handled_regressor_idxs = set()

    def note_regressor(idx, c):
        if idx in regressor_fail_sigs_by_idx or idx in handled_regressor_idxs:
            return
        if not c.get("true_label"):
            return
        bug_rev = c["commit_id"]
        fail_sigs = set()
        for sig in get_failing_signature_groups_for_revision(bug_rev):
            try:
                fail_sigs.add(int(sig))
            except (TypeError, ValueError):
                continue
        regressor_fail_sigs_by_idx[idx] = fail_sigs
        if not fail_sigs:
            handled_regressor_idxs.add(idx)
            return

        last_clean_idx = -1
        for sig in fail_sigs:
            prev = last_seen_batch_end_idx_by_sig.get(sig, -1)
            if prev > last_clean_idx:
                last_clean_idx = prev
        last_clean_for_regressor_by_idx[idx] = last_clean_idx

    for batch_start_idx, batch_end_idx, batch_end_time in batches:
        if batch_end_idx < batch_start_idx:
            continue
        for idx in range(batch_start_idx, batch_end_idx + 1):
            note_regressor(idx, commits[idx])

        batch_commits = commits[batch_start_idx : batch_end_idx + 1]
        suite_sig_ids = _union_tested_signature_group_ids_for_commits(batch_commits)

        if suite_sig_ids:
            durations = get_signature_durations_for_ids(suite_sig_ids)
            batch_finish_time = run_test_suite(executor, batch_end_time, durations)
            total_tests_run += len(durations)
        else:
            batch_finish_time = batch_end_time

        # Detect newly-revealed regressors at this batch end.
        detected = []
        suite_sig_ids_set = set(suite_sig_ids)
        for reg_idx, fail_sigs in regressor_fail_sigs_by_idx.items():
            if reg_idx in handled_regressor_idxs:
                continue
            if reg_idx > batch_end_idx:
                continue
            if not fail_sigs:
                handled_regressor_idxs.add(reg_idx)
                continue
            if suite_sig_ids_set.intersection(fail_sigs):
                detected.append(reg_idx)

        # Update last-seen signature indices after this run.
        for sig in suite_sig_ids_set:
            last_seen_batch_end_idx_by_sig[sig] = batch_end_idx

        if not detected:
            continue

        # Bisection range: from the earliest relevant last-clean+1 to the end
        # of this batch, and only consider regressors detected by this run.
        start_idx = batch_start_idx
        start_candidates = [
            last_clean_for_regressor_by_idx.get(i, -1) + 1 for i in detected
        ]
        if start_candidates:
            start_idx = max(0, min(start_candidates))
        start_idx = min(start_idx, batch_end_idx)

        detected_set = set(detected)
        slice_commits = commits[start_idx : batch_end_idx + 1]
        batch_for_bisect = []
        for offset, c in enumerate(slice_commits):
            global_idx = start_idx + offset
            batch_for_bisect.append(
                {
                    **c,
                    "true_label": bool(global_idx in detected_set),
                }
            )

        total_tests_run, culprit_times, feedback_times = bisect_fn(
            batch_for_bisect,
            batch_finish_time,
            total_tests_run,
            culprit_times,
            feedback_times,
            executor,
            False,  # is_batch_root=False; batch suite already ran
        )

        handled_regressor_idxs.update(detected_set)

    return build_results(total_tests_run, culprit_times, feedback_times, executor.total_cpu_minutes)


def simulate_twsb_with_bisect(commits, bisect_fn, _unused_param, num_workers):
    """
    Time-Window Subset Batching (TWSB).

    For each revision, we run only the perf signature-groups that Mozilla actually
    executed for that revision (from perf_jobs_per_revision_details_rectified.jsonl,
    interpreted as signature-groups).
    When this per-revision subset first exercises any failing signature-group for a
    regressor that lies between the last known clean revision and the current
    revision, we trigger a bisection between that last clean revision and the
    current revision, using the provided bisect_fn.

    Each bisection step uses the same failing signature-group suite as other
    bisection strategies (via bisection_strats).
    """
    logger.info(
        "simulate_twsb_with_bisect: %d commits, bisect_fn=%s",
        len(commits),
        getattr(bisect_fn, "__name__", str(bisect_fn)),
    )

    total_tests_run = 0
    culprit_times = []
    feedback_times = {}

    if not commits:
        logger.info("simulate_twsb_with_bisect: no commits; returning empty results")
        return build_results(total_tests_run, culprit_times, feedback_times, 0.0)

    executor = TestExecutor(num_workers)

    # Precompute, for each commit, the set of perf signature-groups that actually ran.
    tested_sig_group_sets = []
    for c in commits:
        rev = c["commit_id"]
        tested_sig_group_sets.append(set(get_tested_signatures_for_revision(rev)))

    # For each regressor commit j, precompute:
    #   - fail_sigs[j]: its failing signature-groups (from alert_summary_fail_perf_sigs.csv via sig_groups.jsonl)
    #   - last_clean_for_regressor[j]: index of the most recent commit k < j
    #     that ran at least one of those failing signature-groups (i.e., was clean
    #     when exercising them, since the bug has not landed yet).
    regressor_fail_sigs = {}
    last_clean_for_regressor = {}

    last_seen_index_by_sig = {}
    for idx, c in enumerate(commits):
        if c["true_label"]:
            bug_rev = c["commit_id"]
            fail_sigs = set(get_failing_signature_groups_for_revision(bug_rev))
            regressor_fail_sigs[idx] = fail_sigs

            last_clean_idx_for_j = -1
            for sig in fail_sigs:
                prev_idx = last_seen_index_by_sig.get(sig, -1)
                if prev_idx > last_clean_idx_for_j:
                    last_clean_idx_for_j = prev_idx
            last_clean_for_regressor[idx] = last_clean_idx_for_j

        # After computing last_clean_for_regressor for any regressor at idx,
        # update "last seen" indices for all signature-groups tested at this commit.
        for sig in tested_sig_group_sets[idx]:
            last_seen_index_by_sig[sig] = idx

    handled_regressor_idxs = set()

    for idx, c in enumerate(commits):
        submit_time = c["ts"]
        sig_id_set = tested_sig_group_sets[idx]

        if sig_id_set:
            durations = get_signature_durations_for_ids(sig_id_set)
            per_rev_finish_time = run_test_suite(executor, submit_time, durations)
            total_tests_run += len(durations)
        else:
            per_rev_finish_time = submit_time

        if not sig_id_set:
            continue

        # Check whether this revision is the first one (since each bug's
        # regressor) to exercise any of that bug's failing signature-groups.
        for j, fail_sigs in regressor_fail_sigs.items():
            if j in handled_regressor_idxs:
                continue
            if j > idx:
                # Bug has not landed yet at this point in the stream.
                continue
            if not fail_sigs:
                handled_regressor_idxs.add(j)
                continue

            if not sig_id_set.intersection(fail_sigs):
                continue

            # We iterate idx in time order and mark j as handled immediately
            # after triggering, so this is the first revision >= j whose
            # subset exercises any failing signature-group for bug j.
            last_clean_idx_for_j = last_clean_for_regressor.get(j, -1)
            batch_start_idx = max(0, last_clean_idx_for_j + 1)
            if batch_start_idx > idx:
                # Should not happen, but guard against empty ranges.
                batch_start_idx = idx

            batch = commits[batch_start_idx : idx + 1]
            if not batch:
                raise RuntimeError(
                    f"simulate_twsb_with_bisect: empty batch for regressor index {j} "
                    f"(last_clean_idx_for_j={last_clean_idx_for_j}, "
                    f"batch_start_idx={batch_start_idx}, idx={idx}, "
                    f"num_commits={len(commits)})"
                )

            total_tests_run, culprit_times, feedback_times = bisect_fn(
                batch,
                per_rev_finish_time,
                total_tests_run,
                culprit_times,
                feedback_times,
                executor,
                False,  # is_batch_root=False so we only use failing signature-groups
            )

            handled_regressor_idxs.add(j)

    logger.info(
        "simulate_twsb_with_bisect: finished; total_tests_run=%d",
        total_tests_run,
    )
    return build_results(total_tests_run, culprit_times, feedback_times, executor.total_cpu_minutes)


def simulate_twb_with_bisect(commits, bisect_fn, batch_hours, num_workers):
    """
    Time-Window Batching (TWB).

    Groups commits into fixed wall-clock windows of length `batch_hours` and
    flushes each window as a batch. The batch test is modeled as occurring at
    the window end time; `bisect_fn` is invoked on the commits in that window.
    """
    logger.info(
        "simulate_twb_with_bisect: %d commits, batch_hours=%.2f, bisect_fn=%s",
        len(commits),
        batch_hours,
        getattr(bisect_fn, "__name__", str(bisect_fn)),
    )
    total_tests_run = 0
    culprit_times = []
    feedback_times = {}

    # One central executor per simulation run
    executor = TestExecutor(num_workers)

    if not commits:
        logger.info("simulate_twb_with_bisect: no commits; returning empty results")
        return build_results(total_tests_run, culprit_times, feedback_times, executor.total_cpu_minutes)

    batch_start = commits[0]["ts"]
    batch_end = batch_start + timedelta(hours=batch_hours)
    current_batch = []

    for idx, c in enumerate(commits, start=1):
        if c["ts"] < batch_end:
            current_batch.append(c)
        else:
            total_tests_run, culprit_times, feedback_times = bisect_fn(
                current_batch, batch_end, total_tests_run, culprit_times, feedback_times, executor
            )
            logger.debug(
                "TWB: flushing batch with %d commits ending at %s (processed %d/%d commits)",
                len(current_batch),
                batch_end,
                idx,
                len(commits),
            )
            batch_start = c["ts"]
            batch_end = batch_start + timedelta(hours=batch_hours)
            current_batch = [c]

    if current_batch:
        total_tests_run, culprit_times, feedback_times = bisect_fn(
            current_batch, batch_end, total_tests_run, culprit_times, feedback_times, executor
        )

    logger.info(
        "simulate_twb_with_bisect: finished; total_tests_run=%d",
        total_tests_run,
    )
    return build_results(total_tests_run, culprit_times, feedback_times, executor.total_cpu_minutes)


def simulate_twb_s_with_bisect(commits, bisect_fn, batch_hours, num_workers):
    """
    Time-Window Batching with subset suite detection (TWB-s).

    Same windowing as TWB, but the initial batch test uses a *subset suite*:
    the union of perf signature-groups Mozilla actually ran for revisions in
    that window. A regressor can be detected only if the subset suite
    intersects its failing signature-groups.

    Once a batch run detects regressors, bisection is invoked with
    `is_batch_root=False` because the batch suite has already run.
    """
    if not commits:
        return build_results(0, [], {}, 0.0)

    batches = []
    batch_start_idx = 0
    batch_start_time = commits[0]["ts"]
    batch_end_time = batch_start_time + timedelta(hours=batch_hours)

    for idx, c in enumerate(commits):
        if c["ts"] < batch_end_time:
            continue

        batch_end_idx = idx - 1
        if batch_end_idx >= batch_start_idx:
            batches.append((batch_start_idx, batch_end_idx, batch_end_time))

        batch_start_idx = idx
        batch_start_time = c["ts"]
        batch_end_time = batch_start_time + timedelta(hours=batch_hours)

    if batch_start_idx < len(commits):
        batches.append((batch_start_idx, len(commits) - 1, batch_end_time))

    return _simulate_signature_union_driver(commits, bisect_fn, batches, num_workers)


def simulate_fsb_with_bisect(commits, bisect_fn, batch_size, num_workers):
    """
    Fixed-Size Batching (FSB).

    Groups commits into contiguous batches of size `batch_size` (except a final
    partial batch) and flushes each batch at the timestamp of its last commit.
    """
    logger.info(
        "simulate_fsb_with_bisect: %d commits, batch_size=%d, bisect_fn=%s",
        len(commits),
        batch_size,
        getattr(bisect_fn, "__name__", str(bisect_fn)),
    )
    total_tests_run = 0
    culprit_times = []
    feedback_times = {}

    executor = TestExecutor(num_workers)

    current_batch = []
    current_end_time = None

    for idx, c in enumerate(commits, start=1):
        current_batch.append(c)
        current_end_time = c["ts"]
        if len(current_batch) >= batch_size:
            total_tests_run, culprit_times, feedback_times = bisect_fn(
                current_batch, current_end_time, total_tests_run, culprit_times, feedback_times, executor
            )
            logger.debug(
                "FSB: flushing batch of size %d at commit index %d/%d",
                len(current_batch),
                idx,
                len(commits),
            )
            current_batch = []
            current_end_time = None

    if current_batch:
        total_tests_run, culprit_times, feedback_times = bisect_fn(
            current_batch, current_end_time, total_tests_run, culprit_times, feedback_times, executor
        )

    logger.info(
        "simulate_fsb_with_bisect: finished; total_tests_run=%d", total_tests_run
    )
    return build_results(total_tests_run, culprit_times, feedback_times, executor.total_cpu_minutes)


def simulate_fsb_s_with_bisect(commits, bisect_fn, batch_size, num_workers):
    """
    Fixed-Size Batching with subset suite detection (FSB-s).

    Same grouping as FSB, but each batch's initial run uses the union of
    signature-groups observed within the batch (subset suite), and a regressor
    is detected only if that suite overlaps its failing signature-groups.
    """
    batches = []
    if not commits:
        return build_results(0, [], {}, 0.0)

    start = 0
    for end in range(len(commits)):
        if (end - start + 1) >= batch_size:
            batches.append((start, end, commits[end]["ts"]))
            start = end + 1

    if start < len(commits):
        batches.append((start, len(commits) - 1, commits[-1]["ts"]))

    return _simulate_signature_union_driver(commits, bisect_fn, batches, num_workers)


def simulate_rasb_with_bisect(commits, bisect_fn, threshold, num_workers):
    """
    Risk-Adaptive Stream Batching (RASB).

    Accumulates commits into a batch until the estimated probability that the
    batch contains at least one failure reaches `threshold`.

    Probability model:
      - Assume independence between commits.
      - For each commit i, predicted failure probability is `risk_i`.
      - Probability the batch is clean: Π(1 - risk_i)
      - Probability the batch fails: 1 - Π(1 - risk_i)

    Flush happens at the timestamp of the last commit in the batch. `bisect_fn`
    is invoked on the flushed commits.
    """
    logger.info(
        "simulate_rasb_with_bisect: %d commits, threshold=%.4f, bisect_fn=%s",
        len(commits),
        threshold,
        getattr(bisect_fn, "__name__", str(bisect_fn)),
    )

    total_tests_run = 0
    culprit_times = []
    feedback_times = {}

    executor = TestExecutor(num_workers)

    current_batch = []
    log_survival = 0.0
    current_end_time = None

    for idx, c in enumerate(commits, start=1):
        p = c["risk"]
        current_batch.append(c)
        current_end_time = c["ts"]

        log_survival = _accumulate_log_survival(log_survival, p)
        fail_prob = _combined_probability_from_log_survival(log_survival)

        if fail_prob >= threshold:
            total_tests_run, culprit_times, feedback_times = bisect_fn(
                current_batch, current_end_time, total_tests_run, culprit_times, feedback_times, executor
            )
            logger.debug(
                "RASB: threshold reached; flushed batch of size %d at index %d/%d with fail_prob=%.4f",
                len(current_batch),
                idx,
                len(commits),
                fail_prob,
            )
            current_batch = []
            log_survival = 0.0
            current_end_time = None

    if current_batch:
        total_tests_run, culprit_times, feedback_times = bisect_fn(
            current_batch, current_end_time, total_tests_run, culprit_times, feedback_times, executor
        )

    logger.info(
        "simulate_rasb_with_bisect: finished; total_tests_run=%d", total_tests_run
    )
    return build_results(total_tests_run, culprit_times, feedback_times, executor.total_cpu_minutes)


def simulate_rasb_s_with_bisect(commits, bisect_fn, threshold, num_workers):
    """
    RASB with subset suite detection (RASB-s).

    Same batching trigger as RASB, but the initial batch run uses a subset
    suite (union of signature-groups observed within the batch). Detection
    depends on overlap between that subset and each regressor's failing
    signature-groups.
    """
    batches = []
    if not commits:
        return build_results(0, [], {}, 0.0)

    start = 0
    log_survival = 0.0

    for idx, c in enumerate(commits):
        p = c["risk"]
        log_survival = _accumulate_log_survival(log_survival, p)
        fail_prob = _combined_probability_from_log_survival(log_survival)
        if fail_prob >= threshold:
            batches.append((start, idx, c["ts"]))
            start = idx + 1
            log_survival = 0.0

    if start < len(commits):
        batches.append((start, len(commits) - 1, commits[-1]["ts"]))

    return _simulate_signature_union_driver(commits, bisect_fn, batches, num_workers)


def simulate_rasb_la_with_bisect(commits, bisect_fn, risk_budget, num_workers):
    """
    Risk-Adaptive Stream Batching with linear aggregation (RASB-la).

    Accumulates commits into a batch until the sum of per-commit risk scores
    reaches `risk_budget`:

        risk_sum = Σ risk_i

    Flush happens at the timestamp of the last commit in the batch. `bisect_fn`
    is invoked on the flushed commits.
    """
    logger.info(
        "simulate_rasb_la_with_bisect: %d commits, risk_budget=%.4f, bisect_fn=%s",
        len(commits),
        float(risk_budget),
        getattr(bisect_fn, "__name__", str(bisect_fn)),
    )

    total_tests_run = 0
    culprit_times = []
    feedback_times = {}

    if not commits:
        logger.info("simulate_rasb_la_with_bisect: no commits; returning empty results")
        return build_results(total_tests_run, culprit_times, feedback_times, 0.0)

    executor = TestExecutor(num_workers)

    current_batch = []
    current_end_time = None
    risk_sum = 0.0

    for idx, c in enumerate(commits, start=1):
        current_batch.append(c)
        current_end_time = c["ts"]

        if "risk" not in c or c["risk"] is None:
            raise ValueError(
                f"Missing or None 'risk' value for commit at index {idx}: {c!r}"
            )
        risk_sum += float(c["risk"])

        if risk_sum >= risk_budget:
            total_tests_run, culprit_times, feedback_times = bisect_fn(
                current_batch,
                current_end_time,
                total_tests_run,
                culprit_times,
                feedback_times,
                executor,
            )
            logger.debug(
                "RASB-la: risk_budget reached; flushed batch of size %d at index %d/%d with risk_sum=%.4f",
                len(current_batch),
                idx,
                len(commits),
                risk_sum,
            )
            current_batch = []
            current_end_time = None
            risk_sum = 0.0

    if current_batch:
        total_tests_run, culprit_times, feedback_times = bisect_fn(
            current_batch,
            current_end_time,
            total_tests_run,
            culprit_times,
            feedback_times,
            executor,
        )

    logger.info(
        "simulate_rasb_la_with_bisect: finished; total_tests_run=%d",
        total_tests_run,
    )
    return build_results(total_tests_run, culprit_times, feedback_times, executor.total_cpu_minutes)


def simulate_rasb_la_s_with_bisect(commits, bisect_fn, risk_budget, num_workers):
    """
    RASB-la with subset suite detection (RASB-la-s).

    Same batching trigger as RASB-la, but the initial batch run uses a subset
    suite (union of signature-groups observed within the batch). Detection
    depends on overlap between that subset and each regressor's failing
    signature-groups.
    """
    batches = []
    if not commits:
        return build_results(0, [], {}, 0.0)

    start = 0
    risk_sum = 0.0

    for idx, c in enumerate(commits):
        risk_sum += float(c["risk"])
        if risk_sum >= risk_budget:
            batches.append((start, idx, c["ts"]))
            start = idx + 1
            risk_sum = 0.0

    if start < len(commits):
        batches.append((start, len(commits) - 1, commits[-1]["ts"]))

    return _simulate_signature_union_driver(commits, bisect_fn, batches, num_workers)


def simulate_rapb_with_bisect(commits, bisect_fn, params, num_workers):
    """
    Risk-Aware Priority Batching (RAPB).

    Parameters
    ----------
    params:
        Tuple `(threshold_T, aging_rate)` where:
          - `threshold_T` is the target batch failure probability threshold.
          - `aging_rate` controls how quickly older commits become more urgent.

    Each commit's risk is "aged" as it waits in the current batch, increasing
    the estimated failure probability over time until the batch is flushed.
    """
    threshold_T, aging_rate = params

    logger.info(
        "simulate_rapb_with_bisect: %d commits, threshold_T=%.4f, aging_rate=%.4f, bisect_fn=%s",
        len(commits),
        threshold_T,
        aging_rate,
        getattr(bisect_fn, "__name__", str(bisect_fn)),
    )

    total_tests_run = 0
    culprit_times = []
    feedback_times = {}

    executor = TestExecutor(num_workers)

    current_batch = []
    entry_times = []
    current_end_time = None

    for idx, c in enumerate(commits, start=1):
        now_ts = c["ts"]

        current_batch.append(c)
        entry_times.append(now_ts)
        current_end_time = now_ts

        log_survival = 0.0
        for commit, entered_at in zip(current_batch, entry_times):
            wait_hours = max(0.0, (now_ts - entered_at).total_seconds() / 3600.0)
            base_risk = commit["risk"]

            decay = math.exp(-aging_rate * wait_hours)
            aged_p = 1.0 - (1.0 - base_risk) * decay

            log_survival = _accumulate_log_survival(log_survival, aged_p)

        fail_prob = _combined_probability_from_log_survival(log_survival)

        if fail_prob >= threshold_T:
            total_tests_run, culprit_times, feedback_times = bisect_fn(
                current_batch, current_end_time, total_tests_run, culprit_times, feedback_times, executor
            )
            logger.debug(
                "RAPB: threshold reached; flushed batch of size %d at index %d/%d with fail_prob=%.4f",
                len(current_batch),
                idx,
                len(commits),
                fail_prob,
            )
            current_batch = []
            entry_times = []
            current_end_time = None

    if current_batch:
        total_tests_run, culprit_times, feedback_times = bisect_fn(
            current_batch, current_end_time, total_tests_run, culprit_times, feedback_times, executor
        )

    logger.info(
        "simulate_rapb_with_bisect: finished; total_tests_run=%d",
        total_tests_run,
    )
    return build_results(total_tests_run, culprit_times, feedback_times, executor.total_cpu_minutes)


def simulate_rapb_s_with_bisect(commits, bisect_fn, params, num_workers):
    """
    RAPB with subset suite detection (RAPB-s).

    Uses the same batching trigger as RAPB, but the batch run uses a subset
    suite (union of signature-groups observed within the batch) and detection
    depends on overlap with failing signature-groups.
    """
    threshold_T, aging_rate = params
    batches = []
    if not commits:
        return build_results(0, [], {}, 0.0)

    start = 0
    entry_times = []
    current_batch_indices = []

    for idx, c in enumerate(commits):
        now_ts = c["ts"]
        current_batch_indices.append(idx)
        entry_times.append(now_ts)

        log_survival = 0.0
        for commit_idx, entered_at in zip(current_batch_indices, entry_times):
            wait_hours = max(0.0, (now_ts - entered_at).total_seconds() / 3600.0)
            base_risk = float(commits[commit_idx]["risk"])
            decay = math.exp(-aging_rate * wait_hours)
            aged_p = 1.0 - (1.0 - base_risk) * decay
            log_survival = _accumulate_log_survival(log_survival, aged_p)

        fail_prob = _combined_probability_from_log_survival(log_survival)
        if fail_prob >= threshold_T:
            batches.append((start, idx, now_ts))
            start = idx + 1
            entry_times = []
            current_batch_indices = []

    if start < len(commits):
        batches.append((start, len(commits) - 1, commits[-1]["ts"]))

    return _simulate_signature_union_driver(commits, bisect_fn, batches, num_workers)

def simulate_rapb_la_with_bisect(commits, bisect_fn, params, num_workers):
    """
    Risk-Aware Priority Batching with linear aggregation (RAPB-la).

    Parameters
    ----------
    params:
        Tuple `(risk_budget_T, aging_rate)` where:
          - `risk_budget_T` is the target *sum of aged risks* threshold.
          - `aging_rate` controls how quickly older commits become more urgent.

    Each commit's risk is "aged" as it waits in the current batch, then we
    compute:

        risk_sum = Σ aged_risk_i

    and flush when `risk_sum >= risk_budget_T`.
    """
    risk_budget_T, aging_rate = params

    logger.info(
        "simulate_rapb_la_with_bisect: %d commits, risk_budget_T=%.4f, aging_rate=%.4f, bisect_fn=%s",
        len(commits),
        float(risk_budget_T),
        float(aging_rate),
        getattr(bisect_fn, "__name__", str(bisect_fn)),
    )

    total_tests_run = 0
    culprit_times = []
    feedback_times = {}

    executor = TestExecutor(num_workers)

    current_batch = []
    entry_times = []
    current_end_time = None

    for idx, c in enumerate(commits, start=1):
        now_ts = c["ts"]

        current_batch.append(c)
        entry_times.append(now_ts)
        current_end_time = now_ts

        risk_sum = 0.0
        for commit, entered_at in zip(current_batch, entry_times):
            wait_hours = max(0.0, (now_ts - entered_at).total_seconds() / 3600.0)
            base_risk = commit["risk"]

            decay = math.exp(-aging_rate * wait_hours)
            aged_p = 1.0 - (1.0 - base_risk) * decay

            risk_sum += aged_p

        if risk_sum >= risk_budget_T:
            total_tests_run, culprit_times, feedback_times = bisect_fn(
                current_batch,
                current_end_time,
                total_tests_run,
                culprit_times,
                feedback_times,
                executor,
            )
            logger.debug(
                "RAPB-la: threshold reached; flushed batch of size %d at index %d/%d with risk_sum=%.4f",
                len(current_batch),
                idx,
                len(commits),
                risk_sum,
            )
            current_batch = []
            entry_times = []
            current_end_time = None

    if current_batch:
        total_tests_run, culprit_times, feedback_times = bisect_fn(
            current_batch,
            current_end_time,
            total_tests_run,
            culprit_times,
            feedback_times,
            executor,
        )

    logger.info(
        "simulate_rapb_la_with_bisect: finished; total_tests_run=%d",
        total_tests_run,
    )
    return build_results(total_tests_run, culprit_times, feedback_times, executor.total_cpu_minutes)


def simulate_rapb_la_s_with_bisect(commits, bisect_fn, params, num_workers):
    """
    RAPB-la with subset suite detection (RAPB-la-s).

    Uses the same batching trigger as RAPB-la, but the batch run uses a subset
    suite (union of signature-groups observed within the batch) and detection
    depends on overlap with failing signature-groups.
    """
    risk_budget_T, aging_rate = params
    batches = []
    if not commits:
        return build_results(0, [], {}, 0.0)

    start = 0
    entry_times = []
    current_batch_indices = []

    for idx, c in enumerate(commits):
        now_ts = c["ts"]
        current_batch_indices.append(idx)
        entry_times.append(now_ts)

        risk_sum = 0.0
        for commit_idx, entered_at in zip(current_batch_indices, entry_times):
            wait_hours = max(0.0, (now_ts - entered_at).total_seconds() / 3600.0)
            base_risk = float(commits[commit_idx]["risk"])
            decay = math.exp(-aging_rate * wait_hours)
            aged_p = 1.0 - (1.0 - base_risk) * decay
            risk_sum += aged_p

        if risk_sum >= risk_budget_T:
            batches.append((start, idx, now_ts))
            start = idx + 1
            entry_times = []
            current_batch_indices = []

    if start < len(commits):
        batches.append((start, len(commits) - 1, commits[-1]["ts"]))

    return _simulate_signature_union_driver(commits, bisect_fn, batches, num_workers)


def simulate_ratb_with_bisect(commits, bisect_fn, params, num_workers):
    """
    Risk-Aware Trigger Batching (RATB) with TWB-style time-window fallback.

    params:
      - threshold (T): risk cutoff; any commit with risk >= T triggers an immediate flush.
      - time_window_hours: TWB-like max age of a batch; if the span between the
        first commit in the batch and the current commit exceeds this window,
        we flush the batch even if no high-risk commit appeared.

    Behavior:
      - Stream commits in time order.
      - Maintain a current batch of contiguous commits.
      - For each commit c:
          * Add c to the current batch.
          * If c["risk"] >= threshold, flush the batch (including c).
          * Else, if the current batch duration exceeds time_window_hours,
            flush the previous batch and start a new one with c.
      - At the end, flush any remaining commits.
    """
    # Backward compatibility: allow scalar `params` (just threshold),
    # defaulting the time window to 4h.
    if isinstance(params, tuple):
        threshold, time_window_hours = params
    else:
        threshold = float(params)
        time_window_hours = 4.0

    logger.info(
        "simulate_ratb_with_bisect: %d commits, threshold=%.4f, time_window_hours=%.2f, bisect_fn=%s",
        len(commits),
        threshold,
        time_window_hours,
        getattr(bisect_fn, "__name__", str(bisect_fn)),
    )

    total_tests_run = 0
    culprit_times = []
    feedback_times = {}

    if not commits:
        logger.info("simulate_ratb_with_bisect: no commits; returning empty results")
        return build_results(total_tests_run, culprit_times, feedback_times, 0.0)

    executor = TestExecutor(num_workers)

    current_batch = []
    current_end_time = None
    batch_start_time = None
    window_delta = timedelta(hours=time_window_hours)

    for idx, c in enumerate(commits, start=1):
        c_ts = c["ts"]
        if "risk" not in c or c["risk"] is None:
            raise ValueError(f"Missing or None 'risk' value for commit at index {idx}: {c!r}")
        risk = float(c["risk"])

        # If starting a new batch
        if not current_batch:
            current_batch = [c]
            batch_start_time = c_ts
            current_end_time = c_ts
            continue

        # First, risk-triggered flush: include c then flush.
        if risk >= threshold:
            current_batch.append(c)
            current_end_time = c_ts

            total_tests_run, culprit_times, feedback_times = bisect_fn(
                current_batch,
                current_end_time,
                total_tests_run,
                culprit_times,
                feedback_times,
                executor,
            )
            logger.debug(
                "RATB: risk-triggered flush; batch size %d at index %d/%d (risk=%.4f)",
                len(current_batch),
                idx,
                len(commits),
                risk,
            )
            current_batch = []
            batch_start_time = None
            current_end_time = None
            continue

        # Otherwise, apply TWB-style time-window rule.
        batch_end = batch_start_time + window_delta
        if c_ts >= batch_end:
            # Flush the existing batch (without c) at the time-window boundary.
            total_tests_run, culprit_times, feedback_times = bisect_fn(
                current_batch,
                batch_end,
                total_tests_run,
                culprit_times,
                feedback_times,
                executor,
            )
            # Start a new batch with c.
            current_batch = [c]
            batch_start_time = c_ts
            current_end_time = c_ts
        else:
            # Still within time window: just extend the batch.
            current_batch.append(c)
            current_end_time = c_ts

    # Flush any leftover commits
    if current_batch:
        total_tests_run, culprit_times, feedback_times = bisect_fn(
            current_batch,
            current_end_time,
            total_tests_run,
            culprit_times,
            feedback_times,
            executor,
        )
    logger.info(
        "simulate_ratb_with_bisect: finished; total_tests_run=%d", total_tests_run
    )
    return build_results(total_tests_run, culprit_times, feedback_times, executor.total_cpu_minutes)


def simulate_ratb_s_with_bisect(commits, bisect_fn, params, num_workers):
    """
    RATB with subset suite detection (RATB-s).

    Uses the same batching boundaries as RATB, but models each batch's initial
    run as a subset suite (union of signature-groups observed within the
    batch). Detection depends on overlap with failing signature-groups, and
    bisection is invoked with `is_batch_root=False` because the batch suite is
    assumed to have already run.
    """
    if isinstance(params, tuple):
        threshold, time_window_hours = params
    else:
        threshold = float(params)
        time_window_hours = 4.0

    batches = []
    if not commits:
        return build_results(0, [], {}, 0.0)

    start = 0
    batch_start_time = commits[0]["ts"]
    window_delta = timedelta(hours=time_window_hours)

    for idx, c in enumerate(commits):
        c_ts = c["ts"]
        risk = float(c["risk"])

        if idx == start:
            batch_start_time = c_ts
            continue

        if risk >= threshold:
            batches.append((start, idx, c_ts))
            start = idx + 1
            if start < len(commits):
                batch_start_time = commits[start]["ts"]
            continue

        batch_end_time = batch_start_time + window_delta
        if c_ts >= batch_end_time:
            batches.append((start, idx - 1, batch_end_time))
            start = idx
            batch_start_time = c_ts

    if start < len(commits):
        batches.append((start, len(commits) - 1, commits[-1]["ts"]))

    return _simulate_signature_union_driver(commits, bisect_fn, batches, num_workers)
