# Batch Testing Simulation Strategies

This folder contains a simulation of *batch testing* and *bisection* policies over a stream of commits. The key entrypoint is `analysis/batch_testing/simulation.py`, which runs different **batching strategies** (when to trigger a “batch test”) paired with different **bisection strategies** (how to locate culprits once a batch is known to contain defects).

The code intentionally separates:

- **Batching strategies** (`analysis/batch_testing/batch_strats.py`): decide *which commits are grouped together* and *when a batch test run happens*.
- **Bisection strategies** (`analysis/batch_testing/bisection_strats.py`): given a batch (already known to contain a failure), decide *which intervals/prefixes to test* to identify culprit commits.
- **Execution/time model** (`analysis/batch_testing/bisection_strats.py`): `TestExecutor` models a fixed-size pool of test workers; `run_test_suite` schedules suites; `_run_interval_and_update` implements “test this interval” semantics shared by multiple strategies.

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
   - Represents a batch test run that executes a “full suite” of perf signatures.
   - In code: `get_batch_signature_durations()` decides what “full suite” means (all signatures or a capped/subsampled set).

2. **Targeted run** (“bisection step”):
   - Represents a follow-up run using only the **failing-signature suite** for the batch (union of failing perf signatures across regressors in that batch).
   - In code: `get_failing_signature_durations_for_batch(batch)` computes the durations for that suite.

Each **logical run** (full suite or targeted) consumes “CPU cost” equal to the number of signatures in its suite, and occupies worker capacity according to the suite’s per-test durations.

### Central test executor (parallelism model)
`TestExecutor(num_workers)` models a fixed number of test workers:

- Each test in a suite is scheduled onto the earliest-available worker.
- Suites can be run concurrently: if multiple suites are submitted with the same `requested_start_time`, their tests interleave on the worker pool.
- `run_test_suite(executor, t0, durations)` returns the time when the last test in that suite finishes.

This is how “parallel test capacity K” is modeled for all strategies.

### Interval testing (the shared primitive)
Most bisection strategies repeatedly “test an interval” `[lo..hi]` of commits in a batch using `_run_interval_and_update`:

- **If the run is clean**, it marks commits in `[lo..hi]` as `clean` and records their feedback time.
- **If the run fails**, it means at least one commit in `[lo..hi]` is a regressor that has not yet been marked as `defect_found`.

The first interval test in a batch may be a **full suite** run (`is_batch_root=True`), and subsequent tests in that batch are **targeted** runs.

## Batching Strategies (`batch_strats.py`)

Batching strategies determine when to “flush” a batch and call a bisection strategy on that batch. They all share:

- A single `TestExecutor` for the whole simulation run.
- The same bisection function signature: `bisect_fn(batch, batch_end_time, ...)`.

### TWB / TWB-s - Time-Window Batching
- **TWB**: batches commits into fixed wall-clock windows of size `batch_hours`.
- When the window ends, it triggers a batch test at the *window end time* and bisection runs on the collected commits.
- **TWB-s**: same batching windows, but each batch test runs only the union of perf signatures actually tested in that window (“subset suite”), and a regressor is “detected” only if the subset overlaps its failing signatures.

### FSB / FSB-s - Fixed-Size Batching
- **FSB**: batches commits into contiguous groups of size `batch_size` and flushes when the group is full.
- **FSB-s**: same grouping, but uses subset suite detection (union of signatures actually tested in that batch).

### RASB / RASB-s - Risk-Adaptive Stream Batching
- Maintains `prod_clean = Π(1 - risk_i)` across the current batch.
- Flushes when `fail_prob = 1 - prod_clean >= threshold`.
- **RASB-s** uses subset suite detection for the batch run.

### RAPB / RAPB-s - Risk-Aware Priority Batching (threshold + aging)
- Maintains a risk score per commit with an “aging” term over time, favoring older untested commits.
- Flushes batches based on a threshold and aging parameter, modeling a policy that prioritizes risky and stale work.
- **RAPB-s** uses subset suite detection.

### RRBB / RRBB-s - Risk-Ranked Budget Batching
- Accumulates commits until a fixed “risk budget” is exceeded, then flushes.
- **RRBB-s** uses subset suite detection.

### RATB / RATB-s - Risk-Adaptive Time-Window Batching
- Uses a time window and a risk threshold jointly to decide when to flush.
- **RATB-s** uses subset suite detection.

### TWSB - Time-Window Subset Batching (per-revision)
- Runs per-revision subset suites (the signatures Mozilla actually ran for that revision).
- Triggers bisection when a revision’s subset first exercises any failing signature for a regressor.
- Unlike other batching strategies, this is closer to a continuous per-commit testing stream.

## Bisection Strategies (`bisection_strats.py`)

All bisection strategies share:

- A central worker pool (`TestExecutor`).
- A cost model in “number of signatures run”.
- A time model in “when the last test in each suite completes”.

### TOB - Time-Ordered Bisection
- Repeatedly tests the full unknown interval.
- If it fails, splits by midpoint and binary-searches to the first failing commit in time order.

### PAR - Exhaustive Parallel
- Not a bisection per se: after the initial full suite run, it submits targeted runs for all commits (except the last) at the same start time, and uses the completion times as feedback.

### RWAB - Risk-Weighted Adaptive Bisection
- Like TOB, but chooses split points so that *sum of predicted risk* is balanced between halves.

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

`simulation.py` uses Optuna to tune batching parameters and shared prediction thresholds. For the `TKRB` bisection strategy it also tunes `TKRB_TOP_K` and sets the module-level value in `bisection_strats.py` before each simulation run.
