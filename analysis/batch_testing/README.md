# Batch Testing Simulation Strategies

This folder contains a simulation of *batch testing* and *bisection* policies over a stream of commits. The key entrypoint is `analysis/batch_testing/simulation.py`, which runs different **batching strategies** (when to trigger a “batch test”) paired with different **bisection strategies** (how to locate culprits once a batch is known to contain defects).

The code intentionally separates:

- **Batching strategies** (`analysis/batch_testing/batch_strats.py`): decide *which commits are grouped together* and *when a batch test run happens*.
- **Bisection strategies** (`analysis/batch_testing/bisection_strats.py`): given a batch (already known to contain a failure), decide *which intervals/prefixes to test* to identify culprit commits.
- **Execution/time model** (`analysis/batch_testing/bisection_strats.py`): `TestExecutor` models per-platform worker pools; `run_test_suite` schedules suites; `_run_interval_and_update` implements “test this interval” semantics shared by multiple strategies.

## Core Objects and Semantics

### Commit record
Throughout the simulator a “commit” is a dict with at least:

- `commit_id`: revision hash/string
- `ts`: a `datetime` timestamp (commit time / arrival time in the stream)
- `risk`: float in `[0, 1]` used by risk-aware strategies
- `true_label`: bool, whether the commit is a true regressor in the simulation

### What a “test run” means
The simulator models two kinds of test runs:

1. **Full suite run** (“batch root”):
   - Represents a batch test run that executes a “full suite” of perf signature-groups.
   - In code: `get_batch_signature_durations()` decides what “full suite” means (all signature-groups or a capped/subsampled set).

2. **Targeted run** (“bisection step”):
   - Represents a follow-up run using only the **failing signature-group suite** for the batch (union of failing signature-groups across regressors in that batch).
   - In code: `get_failing_signature_durations_for_batch(batch)` computes the durations for that suite.

Each **logical run** (full suite or targeted) consumes “CPU cost” equal to the number of signature-groups in its suite, and occupies worker capacity according to the suite’s per-job durations.

Additionally, each logical run includes a constant **build-time overhead** (default: 1.5 hours, configurable via `--build-time-minutes`) that is added once per run on top of the suite job durations.

### Central test executor (parallelism model)
`TestExecutor(worker_pools)` models per-platform worker pools:

- Each test in a suite is scheduled onto the earliest-available worker.
- Suites can be run concurrently: if multiple suites are submitted with the same `requested_start_time`, their jobs interleave within each platform pool.
- Each signature-group job is routed to a pool based on `machine_platform` metadata (via `datasets/mozilla_perf/all_signatures.jsonl` and `datasets/mozilla_perf/sig_groups.jsonl`).
- `run_test_suite(executor, t0, durations)` returns the time when the last test in that suite finishes.

This is how “parallel test capacity K” is modeled for all strategies.

### Interval testing (the shared primitive)
Most bisection strategies repeatedly “test an interval” `[lo..hi]` of commits in a batch using `_run_interval_and_update`:

- **If the run is clean**, it marks commits in `[lo..hi]` as `clean` and records their feedback time.
- **If the run fails**, it means at least one commit in `[lo..hi]` is a regressor that has not yet been marked as `defect_found`.

The first interval test in a batch may be a **full suite** run (`is_batch_root=True`), and subsequent tests in that batch are **targeted** runs.

## Output Metrics

Each simulation run (for a single batching×bisection combo) produces a dict of summary metrics. In the JSON outputs written by `simulation.py`, time-based metrics are reported in **hours** (`*_hr`).

- `total_tests_run`: total number of signature-group jobs executed across all suite runs (batch roots + bisection steps).
- `total_cpu_time_hr`: sum of all scheduled signature-group job durations (independent of parallelism).
- `mean_feedback_time_hr`: mean time from commit timestamp to first “feedback” time across commits that receive feedback.
- `mean_time_to_culprit_hr`: mean time-to-culprit across detected true regressors.
- `max_time_to_culprit_hr`: worst-case time-to-culprit across detected true regressors.
- `p90_time_to_culprit_hr` / `p95_time_to_culprit_hr` / `p99_time_to_culprit_hr`: percentile time-to-culprit values computed over the same time-to-culprit distribution.

## Batching Strategies (`batch_strats.py`)

Batching strategies determine when to “flush” a batch and call a bisection strategy on that batch. They all share:

- A single `TestExecutor` for the whole simulation run.
- The same bisection function signature: `bisect_fn(batch, batch_end_time, ...)`.

### Common mechanics (flush boundaries, suites, and the “-s” variants)
For all strategies, the batching policy defines **batch boundaries**: contiguous slices of the commit stream. When a boundary is reached, the policy schedules an initial “batch test” and (if needed) launches bisection to localize culprits within that slice.

- **What “flush” means**: the strategy decides a `(batch_start_idx, batch_end_idx, batch_end_time)` and considers the commits in that interval as one unit of work.
- **When the initial run starts**:
  - Time-window policies typically start the run at the *window end time* (even if the last commit arrived earlier).
  - Size/risk stream policies typically start the run at the *timestamp of the last commit in the batch*.
- **What the initial run executes**:
  - Non-`-s` variants: the bisection strategy is responsible for the initial “batch root” run (`is_batch_root=True`), which uses `get_batch_signature_durations()` (often “full suite”, optionally capped/downsampled).
  - `-s` variants: the batching strategy itself runs a **subset suite** for the batch and only triggers bisection if that subset would reveal a failure. Implementation detail: the `-s` strategies all route through `_simulate_signature_union_driver(...)`, which:
    - Builds the subset suite as the union of signature-groups Mozilla actually ran for revisions in the batch (`perf_jobs_per_revision_details_rectified.jsonl`).
    - Detects a regressor in that batch only if the subset suite intersects that regressor’s failing signature-groups (`alert_summary_fail_perf_sigs.csv` → `sig_groups.jsonl`).
    - Invokes `bisect_fn(..., is_batch_root=False)` because the “batch root” suite has already been run by the batching policy.

### TWB / TWB-s - Time-Window Batching
- **Trigger rule (TWB)**: partition the stream into fixed wall-clock windows of size `batch_hours` based on commit timestamps.
  - Commits with `ts` in `[window_start, window_end)` are collected into the same batch.
  - When the window ends, we flush and schedule the initial run at `window_end` (not at the last commit time).
- **Initial suite (TWB)**: the bisection strategy runs the batch root suite (typically “full suite”) at the flush time with `is_batch_root=True`.
- **Why this matters**: window end alignment can introduce “waiting time” (extra latency) for commits near the beginning of a window, but it also prevents tiny batches during bursty periods.

- **TWB-s**: same window boundaries, but the initial run is a subset suite:
  - **Subset construction**: union of signature-groups actually tested by Mozilla across the revisions in the window.
  - **Detection semantics**: a regressor in the window is only detected when the subset suite overlaps at least one of its failing signature-groups; otherwise, no bisection is triggered for that regressor at that window boundary.
  - **Bisection call**: when detection happens, bisection is invoked with `is_batch_root=False` (because the subset suite already ran).

### FSB / FSB-s - Fixed-Size Batching
- **Trigger rule (FSB)**: group commits into contiguous chunks of size `batch_size`.
  - Flush as soon as the batch reaches `batch_size` commits (the final batch may be smaller).
  - The initial run is scheduled at the timestamp of the last commit in the batch.
- **Intuition**: unlike TWB, this enforces a predictable “commits-per-batch” load, but the wall-clock time span of a batch can vary depending on commit rate.

- **FSB-s**: same boundaries as FSB, but uses subset suite detection (union of signature-groups actually tested within that batch), and triggers bisection only when the subset overlaps failing signature-groups.

### RASB / RASB-s - Risk-Adaptive Stream Batching
- **Trigger rule (RASB)**: grow a batch until its predicted probability of containing at least one failure crosses a threshold.
  - Assumes independent per-commit failure probabilities (`risk_i`).
  - Computes `fail_prob = 1 - Π(1 - risk_i)`.
  - Implementation detail: uses log-survival mass `log_survival = Σ log(1 - risk_i)` (via `log1p`) and `fail_prob = 1 - exp(log_survival)` for numerical stability.
  - Schedules the initial run at the timestamp of the last commit in the batch.
- **Intuition**: produces smaller batches when risk is high (for faster detection/localization) and larger batches when risk is low (for lower testing overhead).

- **RASB-s**: same boundary rule, but the initial run is a subset suite and detection depends on overlap with failing signature-groups.

### RASB-la / RASB-la-s - Risk-Adaptive Stream Batching (linear aggregation)
- **Trigger rule (RASB-la)**: grow a batch until a linear “risk budget” is exceeded.
  - Maintains `risk_sum = Σ risk_i` across the current batch.
  - Flushes when `risk_sum >= risk_budget`.
  - Flush time is the timestamp of the most recent commit.
- **How to interpret `risk_budget`**: for probabilistic risks, `risk_sum ≈ 1.0` loosely corresponds to “about one expected failing commit per batch” (though it is not the same as the RASB independence model).

- **RASB-la-s**: same boundaries, but uses subset suite detection and overlap-based triggering.

### RAPB / RAPB-s - Risk-Aware Priority Batching (threshold + aging)
- **Parameters**: `params = (threshold_T, aging_rate)`.
- **Trigger rule (RAPB)**: like RASB, but “ages” each commit’s risk the longer it has been waiting in the current batch.
  - For each commit, compute a waiting time in hours since it entered the batch.
  - Convert base risk into an aged risk `aged_p` using an exponential term controlled by `aging_rate`.
  - Recompute `fail_prob = 1 - Π(1 - aged_p)` from the aged risks (computed via log-survival mass), and flush when it exceeds `threshold_T`.
  - Flush time is the timestamp of the most recent commit.
- **Intuition**: prevents very old commits from waiting indefinitely when the stream is mostly low-risk (aging increases urgency over time).

- **RAPB-s**: same boundary rule, but uses subset suite detection and triggers bisection only when overlap occurs.

### RAPB-la / RAPB-la-s - Risk-Aware Priority Batching (linear aggregation)
- **Parameters**: `params = (risk_budget_T, aging_rate)`.
- **Trigger rule (RAPB-la)**: like RAPB, but aggregates aged risks linearly.
  - For each commit, compute a waiting time in hours since it entered the batch.
  - Convert base risk into an aged risk `aged_p` using an exponential term controlled by `aging_rate`.
  - Maintain `risk_sum = Σ aged_p` and flush when `risk_sum >= risk_budget_T`.
  - Flush time is the timestamp of the most recent commit.

- **RAPB-la-s**: same boundary rule, but uses subset suite detection and overlap-based triggering.

### RATB / RATB-s - Risk-Adaptive Time-Window Batching
- **Parameters**: `params = (threshold, time_window_hours)` (or a scalar `threshold` with a default window of 4h).
- **Trigger rule (RATB)**: hybrid of a risk-triggered flush and a TWB-style maximum batch age.
  - Stream commits in time order, maintaining a contiguous current batch.
  - **Risk trigger**: if a commit arrives with `risk >= threshold`, include it and immediately flush the batch.
  - **Time-window fallback**: if no high-risk commit arrives, flush when the current batch’s age exceeds `time_window_hours` (the time span from the first commit in the batch).
  - Flush time:
    - Risk trigger: the timestamp of the triggering commit.
    - Time-window fallback: the time-window boundary (`batch_start + time_window_hours`).
- **Intuition**: ensures “prompt handling” of very risky commits while still bounding worst-case staleness during low-risk periods.

- **RATB-s**: same boundaries, but uses subset suite detection and overlap-based triggering.

### TWSB - Time-Window Subset Batching (per-revision)
- **Per-revision subset**: instead of flushing explicit multi-commit batches, run a subset suite at *every* revision using the signature-groups Mozilla actually ran for that revision.
- **Trigger rule**: for a regressor commit `j`, trigger bisection at the first revision `i >= j` whose tested subset intersects any of `j`’s failing signature-groups.
  - The bisection range starts at the last known clean revision (for those failing signature-groups) and ends at `i`.
  - The “batch end time” passed to bisection is the completion time of the per-revision subset suite for `i`.
- **Intuition**: models a more continuous “test what was run on each revision” world; detection timing is driven by which signature-groups happen to be exercised over time.

## Bisection Strategies (`bisection_strats.py`)

All bisection strategies share:

- A central worker pool (`TestExecutor`).
- A cost model in “number of signature-groups run”.
- A time model in “when the last test in each suite completes”.

### TOB - Time-Ordered Bisection
- Repeatedly tests the full unknown interval.
- If it fails, splits by midpoint and binary-searches to the first failing commit in time order.

### PAR - Exhaustive Parallel
- Not a bisection per se: after the initial full suite run, it submits targeted runs for all commits (except the last) at the same start time, and uses the completion times as feedback.

### RWAB - Risk-Weighted Adaptive Bisection
- Like TOB, but chooses split points so that *sum of predicted risk* is balanced between halves.

### RWAB-LS - Risk-Weighted Adaptive Bisection (log survival)
- Same as RWAB, but balances split points using the combined probability mass `1 - Π(1 - risk_i)` (computed via log-survival for numerical stability).

### TKRB - Top-K Risk-First Bisection (K tunable)
- After the initial full suite run, probes the **top-K** most risky commits first using prefix tests `[0..idx]`.
- If a probe indicates a defect boundary at `idx`, `idx` is marked as a culprit.
- Otherwise it falls back to a TOB-style search on remaining unknown commits.
- For `K > 1`, Step 2 submits all top-K probes concurrently and applies a barrier: follow-up work happens only after all probes complete.

### SWB - Sequential Walk-Backward
- For a failing unknown interval `[lo..hi]`, treats `hi` as buggy and tests `[lo..hi-1]`, then `[lo..hi-2]`, … sequentially until it becomes clean; the first buggy after a clean is the culprit.

### SWF - Sequential Walk-Forward
- For a failing unknown interval `[lo..hi]`, tests `[lo..lo]`, then `[lo..lo+1]`, … sequentially until it fails; the first failing prefix end is the culprit.

## Optuna Optimization

`simulation.py` uses Optuna to tune batching parameters. For the `TKRB` bisection strategy it also tunes `TKRB_TOP_K` and sets the module-level value in `bisection_strats.py` before each simulation run.

The Optuna objective is multi-objective: minimize `(total_tests_run, max_time_to_culprit_hr)`.

After Optuna finishes a study for a given batching×bisection combo, we select a single configuration from the Pareto front:

- Prefer Pareto points whose `max_time_to_culprit_hr` is **<=** the baseline max TTC (baseline is `TWSB + PAR` when included).
- If any such points exist, choose the one with the smallest `total_tests_run` (tie-break by `max_time_to_culprit_hr`).
- If none exist, choose the one with the smallest `max_time_to_culprit_hr` (tie-break by `total_tests_run`).
