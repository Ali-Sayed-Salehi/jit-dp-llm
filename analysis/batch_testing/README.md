# Batch Testing Simulation Strategies

This folder contains a simulation of *batch testing* and *bisection* policies over a stream of commits. The key entrypoint is `analysis/batch_testing/simulation.py`, which runs different **batching strategies** (when to trigger a “batch test”) paired with different **bisection strategies** (how to locate culprits once a batch is known to contain defects).

The code intentionally separates:

- **Batching strategies** (`analysis/batch_testing/batch_strats.py`): decide *which commits are grouped together* and *when a batch test run happens*.
- **Bisection strategies** (`analysis/batch_testing/batch_strats.py`): per-failing-signature-group bisection processes that decide *which intervals/prefixes to test* to identify culprit commits.
- **Execution/time model** (`analysis/batch_testing/bisection_strats.py`): perf metadata loading + `TestExecutor` (per-platform worker pools); `run_test_suite` / `schedule_test_suite_jobs` schedule suites and return completion times.

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
   - In code: `get_batch_signature_durations()` defines the “full suite” (all signature-groups, optionally restricted to the cutoff-window union via `configure_full_suite_signatures_union()`).
   - Note: `simulation.py` restricts the full suite by default to the union of signature-groups that appear at least once within the EVAL+FINAL cutoff windows.

2. **Per-signature-group run** (“bisection step”):
   - Represents a follow-up interval/prefix test for a *single failing signature-group* (one job per logical run).
   - In code: bisection steps schedule a single signature-group job via
     `get_signature_durations_for_ids([sig_group_id])` (from `bisection_strats.py`),
     invoked by the per-signature-group bisection processes in `batch_strats.py`.

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
Most bisection strategies repeatedly “test an interval” `[lo..hi]` of commits in a batch, *per failing signature-group*:

- **If the run is clean**, it marks commits in `[lo..hi]` as `clean` and records their feedback time.
- **If the run fails**, it means at least one commit in `[lo..hi]` is a regressor that has not yet been marked as `defect_found`.

Each interval test schedules a single signature-group job and uses its completion time as the step finish time.

## Output Metrics

Each simulation run (for a single batching×bisection combo) produces a dict of summary metrics. In the JSON outputs written by `simulation.py`, time-based metrics are reported in **hours** (`*_hr`).

- `total_tests_run`: total number of signature-group jobs executed across all suite runs (batch roots + bisection steps).
- `total_cpu_time_hr`: sum of all scheduled signature-group job durations (independent of parallelism).
- `mean_feedback_time_hr`: mean time from commit timestamp to first “feedback” time across commits that receive feedback.
- `mean_time_to_culprit_hr`: mean time-to-culprit across detected true regressors.
- `max_time_to_culprit_hr`: worst-case time-to-culprit across detected true regressors.
- `p90_time_to_culprit_hr` / `p95_time_to_culprit_hr` / `p99_time_to_culprit_hr`: percentile time-to-culprit values computed over the same time-to-culprit distribution.
- `num_regressors_total`: number of true regressors in the simulated window.
- `num_regressors_found`: number of true regressors that were identified as culprits by the strategy combo.
- `found_all_regressors`: whether `num_regressors_found == num_regressors_total`.

## Running the Simulator (`simulation.py`)

The main entrypoint is `analysis/batch_testing/simulation.py`. It runs:

- **EVAL (Optuna tuning)** on `--input-json-eval`, then
- **FINAL replay** of the selected configurations on `--input-json-final`.

Useful flags (updated to match the current CLI):

- Outputs:
  - `--output-eval` / `--output-final`: where to write the EVAL and FINAL JSON outputs.
  - `--final-only`: skip EVAL and run FINAL using the existing `--output-eval` file.
- Strategy selection:
  - `--batching` / `--bisection`: filter to specific strategies (CSV list, or `all` / `none`).
- Optuna:
  - `--mopt-trials`: base number of Optuna trials **per tunable parameter** (actual trials per combo are scaled by the number of tunable parameters for that combo).
  - `--optuna-seed`: seed for reproducible tuning runs.
  - `--optimize-for-timeliness-metric`: timeliness metric used as Optuna’s 2nd objective and for Pareto-point selection (defaults to max TTC).
  - `--baseline-opt-metric-multplier`: controls how strict we are vs the baseline timeliness metric when selecting a single Pareto point per combo.
- Execution model:
  - `--build-time-minutes`: build overhead added once per logical run.
  - Worker pools:
    - Per-platform pools: `--workers-android`, `--workers-windows`, `--workers-linux`, `--workers-mac`
    - `--unknown-platform-pool`: fallback pool key when a signature-group cannot be mapped to a platform.
    - Legacy mode: `--num-test-workers` (single shared pool; deprecated).
- Runtime:
  - `--skip-exhaustive-testing`: skip the very expensive ET baseline (useful for quick iterations).
  - `--dry-run`: when no explicit filters are provided, run the baseline plus at most two random combos.

Notes:

- The script performs a sanity check that failing signature-groups are covered by the perf-jobs dataset (`validate_failing_signatures_coverage`) and will raise if coverage is inconsistent.
- The script restricts “full suite” batch runs to the union of signature-groups that appear at least once in the EVAL+FINAL cutoff windows (`configure_full_suite_signatures_union`). This avoids charging for signature-groups that cannot affect the simulated window.
- The EVAL output JSON is annotated with a `splits` payload describing the EVAL/FINAL windows (cutoffs and sizes). This makes results self-describing for downstream aggregation.

Example (quick run of one combo):

```bash
python analysis/batch_testing/simulation.py \
  --input-json-eval analysis/batch_testing/final_test_results_perf_codebert_eval.json \
  --input-json-final analysis/batch_testing/final_test_results_perf_codebert_final_test.json \
  --batching ARAHATS-la \
  --bisection PAR \
  --mopt-trials 5 \
  --skip-exhaustive-testing
```

## Worker-Capacity Sweep (`model_machine_count.py`)

`analysis/batch_testing/model_machine_count.py` answers a different question than `simulation.py`:

- `simulation.py`: “Given a fixed worker pool, what batching×bisection policy is best?”
- `model_machine_count.py`: “How do *latency metrics* and *test cost* change as we scale worker capacity?”

It sweeps a **worker pool multiplier** (e.g. `x=0.5`, `x=1.0`, `x=2.0`) and, for each multiplier:

1. Scales the per-platform worker pools (android/windows/linux/mac).
2. Runs the standard EVAL tuning (`run_evaluation_mopt`) on the EVAL split.
3. Replays the selected configuration(s) on the FINAL split (`run_final_test_unified`).
4. Records FINAL metrics vs capacity for plotting and for a single aggregated JSON output.

Implementation notes:

- You can override the default sweep schedule with `--multiplier-list` (CSV).
- Worker pools are scaled by rounding and clamped to a minimum of 1 worker per pool.
- Changing capacity can change not only latency but also the *policy behavior* for load-aware strategies (e.g., `LAB*`, `LARAB*`), because they use predicted queue pressure as part of the flush decision.

Outputs:

- JSON (`--out-json`, default: `analysis/batch_testing/results/machine_count_sweep.json`): keyed by combo; each row includes:
  - `multiplier` and `worker_pools`
  - latency: `mean_feedback_time_hr`, `mean_time_to_culprit_hr`, `max_time_to_culprit_hr`
  - cost: `total_tests_run`, `total_cpu_time_hr`
  - correctness: `found_all_regressors`
- Plots (`--plots-dir`): one plot per combo vs multiplier, with latency curves (left y-axis) and `total_tests_run` (right y-axis).

Important interpretation detail:

- This sweep **retunes** at each multiplier, so the reported curves represent the *best-found configuration at that capacity*. This is appropriate when you want “how good can we be with K machines?”, but it means parameters may change with capacity.

Example:

```bash
python analysis/batch_testing/model_machine_count.py \
  --batching TWB,RATB,RATB-hats \
  --bisection PAR \
  --mopt-trials 25 \
  --skip-exhaustive-testing
```

## Batching Strategies (`batch_strats.py`)

Batching strategies determine when to “flush” a batch and call a bisection strategy on that batch. They all share:

- A single `TestExecutor` for the whole simulation run.
- The same bisection selector (`bisect_fn`): a string id like `TOB`, `PAR`, `RWAB`, `RWAB-LS`, `TKRB`, `SWB`, `SWF`.

### Implemented strategy identifiers

Batching strategy names (as implemented in `analysis/batch_testing/simulation.py`):

- Baselines:
  - `TWSB` (time-window subset baseline; not Optuna-tuned)
- Full-suite batching families and their variants:
  - `TWB`, `TWB-s`, `TWB-hats`
  - `FSB`, `FSB-s`, `FSB-hats`
  - `RASB`, `RASB-s`, `RASB-hats`
  - `RASB-la`, `RASB-la-s`, `RASB-la-hats`
  - `RAPB`, `RAPB-s`, `RAPB-hats`
  - `RAPB-la`, `RAPB-la-s`, `RAPB-la-hats`
  - `RATB`, `RATB-s`, `RATB-hats`
  - `HATS`, `RAHATS`, `RAHATS-la`, `ARAHATS`, `ARAHATS-la`
  - `LAB`, `LAB-s`, `LAB-hats`
  - `LARAB`, `LARAB-s`, `LARAB-hats`
  - `LARAB-la`, `LARAB-la-s`, `LARAB-la-hats`

Bisection strategy names:

- `TOB`, `PAR`, `RWAB`, `RWAB-LS`, `TKRB`, `SWB`, `SWF`

### Common mechanics (flush boundaries, suites, and the `-s`/`-hats` variants)
For all strategies, the batching policy defines **batch boundaries**: contiguous slices of the commit stream. When a boundary is reached, the policy schedules an initial “batch test” and (if needed) launches bisection to localize culprits within that slice.

- **Note on streaming per-signature-group policies**: `HATS`, `RAHATS`, and `ARAHATS` are included in the batching grid even though they do not form one global batch at a time. Instead, they maintain independent per-signature-group state and “flush” individual signature-groups when their score reaches a shared budget.

- **What “flush” means**: the strategy decides a `(batch_start_idx, batch_end_idx, batch_end_time)` and considers the commits in that interval as one unit of work.
- **When the initial run starts**:
  - Time-window policies typically start the run at the *window end time* (even if the last commit arrived earlier).
  - Size/risk stream policies typically start the run at the *timestamp of the last commit in the batch*.
- **What the initial run executes**:
  - Non-`-s` variants: the batching policy runs the “full suite” for the batch (via `get_batch_signature_durations()`), then triggers bisection per failing signature-group as soon as that signature-group job finishes (“streaming detection”).
  - `-s` variants: the batching policy runs a **subset suite** for the batch and triggers bisection only when that subset would reveal a failure. Implementation detail: the `-s` strategies all route through `_simulate_signature_union_driver(...)`, which:
    - Builds the subset suite as the union of signature-groups Mozilla actually ran for revisions in the batch (`perf_jobs_per_revision_details_rectified.jsonl`).
    - Detects a regressor in that batch only if the subset suite intersects that regressor’s failing signature-groups (`alert_summary_fail_perf_sigs.csv` → `sig_groups.jsonl`).
    - Anchors bisection ranges using the last revision where each signature-group was previously exercised.
  - `-hats` variants: the batching policy keeps the *same* batch boundaries as its base strategy, but replaces the initial full suite with a **history-aware subset suite**:
    - Each signature-group has an “exercise interval” (in commit-count space) derived from its historical repeat count and a threshold `HATS_SIG_RISK_THRESHOLD`.
    - At each batch flush, we include only signature-groups that are **due** (i.e., whose interval says they should be re-run by this batch end index), and reschedule them forward after they are exercised.
    - Detection and bisection are “last-seen anchored” (as in HATS): when a due signature-group fails, bisection is restricted to commits since that signature-group was last exercised.

Naming/variant conventions:

- `-la` means the strategy uses a **linear risk sum** (`Σ p`) instead of log-survival / hazard mass (`1 - Π(1 - p)`) when aggregating predicted per-commit risks.
- `-s` and `-hats` change suite construction (and therefore detectability) without changing the base batching boundaries.
- Because `-s` and `-hats` suites can omit failing signature-groups, a parameter setting may fail to detect some regressors. In Optuna tuning, such trials are treated as **infeasible** (the objective returns ∞) so the tuned configurations prioritize *finding all regressors*.

### TWB / TWB-s / TWB-hats - Time-Window Batching
- **Trigger rule (TWB)**: partition the stream into fixed wall-clock windows of size `batch_hours` based on commit timestamps.
  - Commits with `ts` in `[window_start, window_end)` are collected into the same batch.
  - When the window ends, we flush and schedule the initial run at `window_end` (not at the last commit time).
- **Initial suite (TWB)**: the batching policy runs the “full suite” at the flush time and triggers bisection per failing signature-group job completion.
- **Why this matters**: window end alignment can introduce “waiting time” (extra latency) for commits near the beginning of a window, but it also prevents tiny batches during bursty periods.

- **TWB-s**: same window boundaries, but the initial run is a subset suite:
  - **Subset construction**: union of signature-groups actually tested by Mozilla across the revisions in the window.
  - **Detection semantics**: a regressor in the window is only detected when the subset suite overlaps at least one of its failing signature-groups; otherwise, no bisection is triggered for that regressor at that window boundary.
  - **Bisection trigger**: when detection happens, per-signature-group bisections start from that signature-group job’s completion time.

- **TWB-hats**: same window boundaries, but the initial run is history-aware subset suite selection:
  - Parameters are `(batch_hours, hats_sig_risk_threshold)`.
  - The subset suite is made of “due” signature-groups (see `-hats` description above), which reduces cost but can delay detection for regressors whose failing signature-groups are not due in a given window.

### FSB / FSB-s / FSB-hats - Fixed-Size Batching
- **Trigger rule (FSB)**: group commits into contiguous chunks of size `batch_size`.
  - Flush as soon as the batch reaches `batch_size` commits (the final batch may be smaller).
  - The initial run is scheduled at the timestamp of the last commit in the batch.
- **Intuition**: unlike TWB, this enforces a predictable “commits-per-batch” load, but the wall-clock time span of a batch can vary depending on commit rate.

- **FSB-s**: same boundaries as FSB, but uses subset suite detection (union of signature-groups actually tested within that batch), and triggers bisection only when the subset overlaps failing signature-groups.

- **FSB-hats**: same boundaries as FSB, but the initial run is a history-aware subset suite:
  - Parameters are `(batch_size, hats_sig_risk_threshold)`.

### RASB / RASB-s / RASB-hats - Risk-Adaptive Stream Batching
- **Trigger rule (RASB)**: grow a batch until its predicted probability of containing at least one failure crosses a threshold.
  - Assumes independent per-commit failure probabilities (`risk_i`).
  - Computes `fail_prob = 1 - Π(1 - risk_i)`.
  - Implementation detail: uses log-survival mass `log_survival = Σ log(1 - risk_i)` (via `log1p`) and `fail_prob = 1 - exp(log_survival)` for numerical stability.
  - Schedules the initial run at the timestamp of the last commit in the batch.
- **Intuition**: produces smaller batches when risk is high (for faster detection/localization) and larger batches when risk is low (for lower testing overhead).

- **RASB-s**: same boundary rule, but the initial run is a subset suite and detection depends on overlap with failing signature-groups.

- **RASB-hats**: same boundaries as RASB, but the initial run is a history-aware subset suite:
  - Parameters are `(rasb_threshold, hats_sig_risk_threshold)`.

### RASB-la / RASB-la-s / RASB-la-hats - Risk-Adaptive Stream Batching (linear aggregation)
- **Trigger rule (RASB-la)**: grow a batch until a linear “risk budget” is exceeded.
  - Maintains `risk_sum = Σ risk_i` across the current batch.
  - Flushes when `risk_sum >= risk_budget`.
  - Flush time is the timestamp of the most recent commit.
- **How to interpret `risk_budget`**: for probabilistic risks, `risk_sum ≈ 1.0` loosely corresponds to “about one expected failing commit per batch” (though it is not the same as the RASB independence model).

- **RASB-la-s**: same boundaries, but uses subset suite detection and overlap-based triggering.

- **RASB-la-hats**: same boundaries as RASB-la, but the initial run is a history-aware subset suite:
  - Parameters are `(risk_budget, hats_sig_risk_threshold)`.

### RAPB / RAPB-s / RAPB-hats - Risk-Aware Priority Batching (threshold + aging)
- **Parameters**: `params = (threshold_T, aging_rate)`.
- **Trigger rule (RAPB)**: like RASB, but “ages” each commit’s risk the longer it has been waiting in the current batch.
  - For each commit, compute a waiting time in hours since it entered the batch.
  - Convert base risk into an aged risk `aged_p` using an exponential term controlled by `aging_rate`.
  - Recompute `fail_prob = 1 - Π(1 - aged_p)` from the aged risks (computed via log-survival mass), and flush when it exceeds `threshold_T`.
  - Flush time is the timestamp of the most recent commit.
- **Intuition**: prevents very old commits from waiting indefinitely when the stream is mostly low-risk (aging increases urgency over time).

- **RAPB-s**: same boundary rule, but uses subset suite detection and triggers bisection only when overlap occurs.

- **RAPB-hats**: same boundaries as RAPB, but the initial run is a history-aware subset suite:
  - Parameters are `(threshold_T, aging_rate, hats_sig_risk_threshold)`.

### RAPB-la / RAPB-la-s / RAPB-la-hats - Risk-Aware Priority Batching (linear aggregation)
- **Parameters**: `params = (risk_budget_T, aging_rate)`.
- **Trigger rule (RAPB-la)**: like RAPB, but aggregates aged risks linearly.
  - For each commit, compute a waiting time in hours since it entered the batch.
  - Convert base risk into an aged risk `aged_p` using an exponential term controlled by `aging_rate`.
  - Maintain `risk_sum = Σ aged_p` and flush when `risk_sum >= risk_budget_T`.
  - Flush time is the timestamp of the most recent commit.

- **RAPB-la-s**: same boundary rule, but uses subset suite detection and overlap-based triggering.

- **RAPB-la-hats**: same boundaries as RAPB-la, but the initial run is a history-aware subset suite:
  - Parameters are `(risk_budget_T, aging_rate, hats_sig_risk_threshold)`.

### RATB / RATB-s / RATB-hats - Risk-Adaptive Time-Window Batching
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

- **RATB-hats**: same boundaries as RATB, but the initial run is a history-aware subset suite:
  - Parameters are `(threshold, time_window_hours, hats_sig_risk_threshold)`.

### TWSB - Time-Window Subset Batching (per-revision)
- **Per-revision subset**: instead of flushing explicit multi-commit batches, run a subset suite at *every* revision using the signature-groups Mozilla actually ran for that revision.
- **Trigger rule**: for a regressor commit `j`, trigger bisection at the first revision `i >= j` whose tested subset intersects any of `j`’s failing signature-groups.
  - The bisection range starts at the last known clean revision (for those failing signature-groups) and ends at `i`.
  - The “batch end time” passed to bisection is the completion time of the per-revision subset suite for `i`.
- **Intuition**: models a more continuous “test what was run on each revision” world; detection timing is driven by which signature-groups happen to be exercised over time.

### HATS - History-aware Test Selection
- **Parameters**: `params = (risk_budget, base_risk_per_commit, repeat_risk_scale, repeat_risk_power)`.
- **Idea**: maintain an independent cumulative *historical* risk score per signature-group, derived from its historical repeat count.
- **Trigger rule (per signature-group)**: as commits stream in, the score grows by a fixed per-commit increment `inc(r)` based on the group’s repeat count `r`. Flush (run that signature-group) when the score reaches `risk_budget`.
- **Initial run**: a single signature-group job (not a full suite). If it fails, bisect only the commits since that signature-group was last exercised.

### RAHATS / RAHATS-la - Risk-aware History-aware Test Selection
- **Parameters**: `params = (risk_budget, hist_base_risk_per_commit, hist_repeat_risk_scale, hist_repeat_risk_power, hist_multiplier, commit_multiplier)`.
- **Idea**: extend HATS by adding a commit-risk term derived from the stream’s per-commit predicted risks.
- **Trigger rule (per signature-group)**: flush when a combined score reaches `risk_budget`:
  - `hist_term`: grows with commits since last test at a rate based on the group’s repeat count, scaled by `hist_multiplier`.
  - `commit_term`: grows with cumulative commit-risk mass since last test, scaled by `commit_multiplier`.
- **RAHATS vs RAHATS-la**: RAHATS uses log-survival “hazard” mass for the commit term (`-log(1 - p)`), while RAHATS-la uses a linear risk sum (`Σ p`).

### ARAHATS / ARAHATS-la - *Aged* Risk-aware History-aware Test Selection
- **Parameters**: `params = (risk_budget, hist_base_risk_per_commit, hist_repeat_risk_scale, hist_repeat_risk_power, hist_multiplier, commit_multiplier, aging_per_hour)`.
- **Idea**: extend RAHATS with an additional “age” term so commits become more urgent the longer they’ve been waiting (in wall-clock time).
- **Trigger rule (per signature-group)**: flush when `hist_term + commit_term + age_term >= risk_budget`, where:
  - `age_term` grows with the sum of per-commit waiting times `Σ(t_now - t_commit)` since that signature-group’s last test, scaled by `aging_per_hour`.
- **ARAHATS vs ARAHATS-la**: ARAHATS uses log-survival hazard mass for the commit term; ARAHATS-la uses a linear risk sum.

### LAB / LAB-s / LAB-hats - Load-Aware Batching (per worker pool)
- **Parameters**: `queue_pressure_threshold_min` (minutes).
- **Idea**: maintain an independent contiguous commit batch per worker pool (android/windows/linux/mac). Use predicted queue pressure to decide when to flush each pool.
- **Trigger rule (per pool)**: if `queue_pressure_minutes <= queue_pressure_threshold_min`, flush that pool’s current batch.
- **Initial run**:
  - `LAB`: run the pool’s share of the full suite (all signature-groups routed to that pool).
  - `LAB-s`: run a subset suite (union of signature-groups observed in that pool’s batch, filtered to the pool); detection depends on overlap with failing signature-groups (like other `-s` variants).

- **LAB-hats**: same per-pool flush logic, but the initial run is a per-pool history-aware subset suite:
  - Parameters are `(queue_pressure_threshold_min, hats_sig_risk_threshold)`.
  - Only “due” signature-groups (as routed to that pool) are exercised at each pool flush.

### LARAB / LARAB-la (+ `-s`/`-hats` variants) - Load-Aware Risk-Adaptive Batching (per worker pool)
- **Parameters**: `params = (risk_multiplier, queue_pressure_multiplier)`.
- **Idea**: like RASB, but batching is independent per pool and the flush decision uses a single unified score combining cumulative risk and queue pressure.
- **Trigger rule (per pool)**: flush when:
  - `LARAB`: `risk_multiplier * hazard - queue_pressure_multiplier * load >= 1.0`, where `hazard = -Σ log(1 - p)` and `load = log1p(queue_pressure / scale)`.
  - `LARAB-la`: same, but uses `risk_sum = Σ p` instead of hazard.
- **`-s` variants**: `LARAB-s` and `LARAB-la-s` run subset suites per pool batch (union of observed signature-groups in that pool batch), with overlap-based detection and last-seen anchored bisection ranges.
- **`-hats` variants**: `LARAB-hats` and `LARAB-la-hats` keep the same per-pool flush rule, but replace the pool’s initial suite with a history-aware “due signature-groups” subset.

## Bisection Strategies (`batch_strats.py`)

Bisection strategies are implemented as **per-signature-group** processes. When an initial suite run indicates that signature-group `g` failed for a batch interval, the simulator spawns a bisection process for `(g, batch_interval)` that:

- Schedules *single-signature-group* interval tests (one logical run == one job) on the shared `TestExecutor`.
- Uses the pass/fail outcome to mark commits as `clean` or `defect_found`.
- Records:
  - feedback time: when a commit first transitions away from `unknown`
  - time-to-culprit: when a true regressor is first marked `defect_found`

All bisection strategies share:

- A central worker pool (`TestExecutor`).
- A cost model in “number of signature-groups run”.
- A time model in “when the last test in each suite completes”.

### TOB - Time-Ordered Bisection
- **Mechanics**: repeatedly tests the full remaining unknown interval; if it fails, binary-searches by midpoint until the culprit index is isolated.
- **Intuition**: a strong baseline when you have no reliable prior over where the culprit is; typically uses `O(log n)` interval tests per culprit (per signature-group) in the single-defect case.

### PAR - Exhaustive Parallel
- **Mechanics**: submits singleton tests for every commit except the last (`[i..i]` for `i ∈ [0..n-2]`) at the same start time; the last commit gets “feedback” once all intermediate singletons finish (by elimination).
- **Intuition**: trades test cost for timeliness. If worker capacity is high enough, PAR drives down time-to-culprit by parallelizing post-detection localization, but it uses `O(n)` interval tests per failing signature-group.

### RWAB - Risk-Weighted Adaptive Bisection
- **Mechanics**: like TOB, but chooses split points so that the *sum of predicted risks* is balanced between the left and right halves, then tests the left side.
- **Intuition**: approximates an information-gain objective under a risk prior. When risk is skewed (a few commits are much riskier), RWAB can reduce expected steps vs midpoint splits.

### RWAB-LS - Risk-Weighted Adaptive Bisection (log survival)
- **Mechanics**: same as RWAB, but balances split points using combined failure mass `1 - Π(1 - risk_i)` instead of `Σ risk_i` (implemented via log-survival for numerical stability).
- **Intuition**: better matches the “at least one failure” probability aggregation used by several risk-aware batching policies.

### TKRB - Top-K Risk-First Bisection (K tunable)
- After the initial full suite run, probes the **top-K** most risky commits first using prefix tests `[0..idx]`.
- If a probe indicates a defect boundary at `idx`, `idx` is marked as a culprit.
- Otherwise it falls back to a TOB-style search on remaining unknown commits.
- For `K > 1`, Step 2 submits all top-K probes concurrently and applies a barrier: follow-up work happens only after all probes complete.
- **Intuition**: prioritizes “likely culprits” first. This can reduce time-to-culprit when the risk model is well-calibrated and failures are concentrated in the top-risk commits.

### SWB - Sequential Walk-Backward
- **Mechanics**: for a failing interval `[lo..hi]`, test `[lo..hi-1]`, then `[lo..hi-2]`, … until an interval becomes clean; the culprit is the first commit that reintroduces failure.
- **Intuition**: effective when culprits tend to be near the end of a batch; worst-case `O(n)` tests but can terminate quickly when the defect is late.

### SWF - Sequential Walk-Forward
- **Mechanics**: for a failing interval `[lo..hi]`, test `[lo..lo]`, then `[lo..lo+1]`, … until a prefix fails; the first failing prefix end is the culprit.
- **Intuition**: effective when culprits tend to be near the start of a batch; worst-case `O(n)` tests but can terminate quickly when the defect is early.

## Optuna Optimization

`simulation.py` uses Optuna to tune batching parameters. For the `TKRB` bisection strategy it also tunes `TKRB_TOP_K` and sets the module-level value in `bisection_strats.py` before each simulation run.

The Optuna objective is multi-objective: minimize `(total_tests_run, timeliness_metric)`, where
`timeliness_metric` is configured by `simulation.py --optimize-for-timeliness-metric` and defaults
to `max_time_to_culprit_hr` (i.e., max TTC).

Timeliness metric naming:

- The CLI accepts shorthands like `max_ttc`, `mean_ttc`, `p95_ttc`, `p99_ttc`, and `mft`, and resolves them to concrete JSON keys (e.g., `max_time_to_culprit_hr`, `mean_feedback_time_hr`).
- You can also provide a full result key directly (e.g., `p95_time_to_culprit_hr`).

Latest tuning behavior (as implemented in `run_evaluation_mopt`):

- **Sampler**: NSGA-II (`optuna.samplers.NSGAIISampler`) when available, with a deterministic fallback to `RandomSampler` for older Optuna versions.
- **Trial budgeting**: `--mopt-trials` is a *base* number of trials per tunable parameter; the simulator scales trials per combo by the estimated number of tunables for that combo. This provides more sampling to higher-dimensional strategies (e.g., `ARAHATS`).
- **Infeasible trials**: if a trial’s simulation does **not** find all regressors (`found_all_regressors=False`), the objective returns `(∞, ∞)` so Optuna will not select it on the Pareto front. This makes “find all regressors” a hard constraint during tuning.
- **`-hats` variants**: tuning adds an additional parameter `HATS_SIG_RISK_THRESHOLD` which controls how aggressively signature-groups are included as “due” in the history-aware subset suite.
- **HATS threshold search range**: the Optuna search range for `HATS_SIG_RISK_THRESHOLD` is derived from the commit stream so that even low-repeat-count signature-groups can still become “due” within the tail of the window (after the last regressor), rather than being pushed beyond the simulation horizon.
- **Baselines in EVAL**:
  - Optional ET: `"Exhaustive Testing (ET)"` (skipped by `--skip-exhaustive-testing`).
  - Optional baseline: `"Baseline (TWSB + PAR)"` when those strategies are included in the current strategy filter.
  - Optional fixed-parameter sweep: when not in `--dry-run`, the simulator also evaluates `TWSB` paired with every bisection strategy (no tunable params) to provide additional baselines.

After Optuna finishes a study for a given batching×bisection combo, we select a single configuration from the Pareto front:

- Prefer Pareto points whose chosen timeliness metric is **<=** the baseline value for that metric
  (baseline is `TWSB + PAR` when included), scaled by `--baseline-opt-metric-multplier`.
- If any such points exist, choose the one with the smallest `total_tests_run` (tie-break by the
  chosen timeliness metric).
- If none exist, choose the one with the smallest chosen timeliness metric (tie-break by
  `total_tests_run`).

EVAL vs FINAL outputs (high-level):

- **EVAL JSON** (written to `--output-eval`): contains the baseline(s), worker-pool metadata, and one selected tuned configuration per combo, plus split metadata under `splits`.
- **FINAL JSON** (written to `--output-final`): replays the chosen EVAL parameters on the FINAL window and attaches saved-vs-baseline percentage fields and summary “best-by-metric” fields for easy comparison.

Important output fields (for analysis / paper writing):

- **Per-combo entries** (both EVAL and FINAL): keys like `"TWB + PAR"` or `"ARAHATS-la + RWAB"` map to dicts that include:
  - core metrics: `total_tests_run`, `total_cpu_time_hr`, `mean_feedback_time_hr`, `mean_time_to_culprit_hr`, `max_time_to_culprit_hr`, plus percentiles
  - correctness: `num_regressors_total`, `num_regressors_found`, `found_all_regressors`
  - parameters: `best_params` in EVAL and `best_params_from_eval` in FINAL
  - deltas vs baseline: `*_saved_vs_baseline_pct` (when the baseline is present)
- **EVAL summary keys**: `best_by_total_tests`, `best_by_timeliness_metric`, `best_by_max_ttc`, `best_by_mean_feedback_time`.
- **FINAL summary keys**: `best_by_total_tests`, `best_by_max_ttc`, `best_by_mean_feedback_time`, plus:
  - `best_overall_improvement_over_baseline`: a ranked list of combos (Pareto-efficient first, then the rest) using a combined “tests saved + timeliness saved” score.
  - `pareto-efficient_combos`: a ranked list of only Pareto-efficient combos (improve both tests and timeliness vs baseline).
