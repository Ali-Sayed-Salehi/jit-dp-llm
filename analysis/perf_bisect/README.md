# Mozilla Performance Bisect Simulation

This directory contains a simulator for historical Mozilla performance
regressions. For each regression row, the simulator tries to localize the exact
culprit revision between a known-good revision and a known-bad revision, then
reports aggregate localization metrics.

## Files

- `simulation.py`: CLI entry point, dataset loading, run orchestration, and output
  writing.
- `calculate_oracle_metrics.py`: computes per-regression summary and replicate
  oracle accuracy from the historical measurement data.
- `localization.py`: culprit localization algorithms. The initial algorithm is
  `Backfill`.
- `test_oracle.py`: test-oracle implementations, revision/signature indexes, and
  the simulated test executor. The initial oracle is `SummaryComparison`.
- `per_regression_oracle_metrics.jsonl`: per-regression oracle accuracy output
  written by `calculate_oracle_metrics.py`.
- `results/per_bisect_results_eval.json`: aggregate metrics for the eval split.
- `results/per_bisect_results_eval_details.json`: per-regression eval details.
- `results/per_bisect_results_final_test.json`: aggregate metrics for the
  final-test split.
- `results/per_bisect_results_final_test_details.json`: per-regression
  final-test details.

In the detail files, each per-regression result includes `regression_id`, which
is unique across the eval and final-test splits and can be used to join against
`per_regression_oracle_metrics.jsonl`. Each oracle decision includes
`expected_decision` and `decision_correct` when the known culprit is present in
the candidate range. `expected_decision` is `clean` before the culprit and `bad`
at or after the culprit. If the known culprit is not in the reconstructed
candidate range, these fields are `null` because the simulator cannot label each
probed candidate's side from that path. Raw probe attempts are recorded in
`decisions`; the decisions actually used for localization are recorded in
`final_decisions`. Backfill results also record
`non_monotonic_retrigger_count` and `non_monotonic_retrigger_intervals`.

## Inputs

The default inputs are loaded from `datasets/mozilla_perf_bisect`:

- `perf_bisect_regressions_eval.jsonl`
- `perf_bisect_regressions_final_test.jsonl`
- `per_sig_perf_data_info.jsonl`
- `per_revision_perf_data.jsonl`
- `all_commits.jsonl`

The regression rows provide the globally unique `regression_id`,
`good_revision`, `bad_revision`, `culprit_revision`, and failing signature
values. The signature-info file provides the number of replicate tests to submit
for each signature. The revision-performance file provides the revision graph.
`analysis/perf_bisect/per_regression_oracle_metrics.jsonl` provides the
per-regression oracle accuracies used by the simulator.

`calculate_oracle_metrics.py` uses `all_commits.jsonl` for Mercurial parent
relationships and `per_revision_perf_data.jsonl` for the real summary and
replicate measurement values.

`simulation.py` also has a fallback for regression files under
`datasets/mozilla_perf`, matching the originally described path, but the current
workspace data lives under `datasets/mozilla_perf_bisect`.

## Simulation Flow

Each regression is simulated independently. That means each regression gets a
fresh `TestExecutor` with an empty queue and no carry-over pressure from earlier
regressions.

For one regression:

1. The simulator reconstructs a good-to-bad revision path from
   `per_revision_perf_data.jsonl`.
2. The selected localization algorithm chooses revisions to probe.
3. The selected test oracle submits simulated performance tests through the
   executor.
4. The oracle classifies each tested revision as `clean` or `bad`.
5. The localizer decides whether the culprit was found.
6. The simulator records per-regression details and updates aggregate metrics.

## Test Executor

The executor models worker queueing. The default is one worker and one minute per
test run. If a signature has `replicate_counts = 25`, one oracle probe submits 25
test runs.

Backfill submits the first probe batch for all candidate revisions at the same
simulation timestamp. With one worker, those tests still complete sequentially
because the worker queue is shared. Increasing `--workers` reduces queueing time.
If Backfill retriggers a non-monotonic interval, that retrigger is submitted
after the earlier batch has completed.

## SummaryComparison Oracle

`SummaryComparison` models a simple noisy oracle. It does not draw historical
summary or replicate measurement values, and it does not compare measurements to
a baseline during simulation.

For each regression, the simulator reads `summary_oracle_accuracy` from
`analysis/perf_bisect/per_regression_oracle_metrics.jsonl`. Each probe first
determines the revision's true side of the known culprit:

- revisions before the culprit are `clean`
- the culprit and revisions after it are `bad`

With probability `summary_oracle_accuracy`, the oracle returns the true side.
Otherwise it flips the verdict. Because decisions are sampled from this noisy
oracle instead of from finite measurement lists, repeated probes never run out
of available draws. Each probe still uses the signature's `replicate_counts` as
the simulated test cost.

Noisy verdict randomness is seeded by `--random-seed`, which defaults to `0`.
Each regression gets a derived seed based on `regression_id - 1`, so default
runs are reproducible and stable across split order.

## Backfill Localizer

`Backfill` probes every revision after `good_revision` through `bad_revision`.
It succeeds only when the final oracle decisions are monotonic:

- zero or more initial candidate revisions are `clean`
- every candidate revision after the first `bad` revision is also `bad`

The found culprit is the first `bad` revision in that monotonic sequence. If the
observed clean/bad pattern contains a `bad` decision followed by a later `clean`
decision, Backfill treats the shortest such interval as the adjacent bad-to-clean
pair and retriggers those two commits. After a retrigger, `final_decisions`
reconciles repeated results for each retriggered commit by majority vote over
the drawn clean/bad decisions, breaking ties with the latest draw. It does not
average measurement values. The raw attempts remain in `decisions`.

This process repeats up to `--backfill-non-monotonic-retrigger-count` times
before the localization remains undefined as `non_monotonic_oracle_decisions`.
The default is `2`, which models a small number of human retriggers without
turning a noisy interval into an unbounded retry loop.

## Metrics

Each summary output reports:

- `mean_trtc_minutes`: mean time from regression to culprit for successful
  localizations only.
- `max_trtc_minutes`: max time from regression to culprit for successful
  localizations only.
- `mean_test_runs`: mean number of test runs across all regressions, including
  undefined localizations.
- `max_test_runs`: max number of test runs across all regressions.
- `success_rate_percent`: percentage of regressions where the exact culprit was
  found.
- `successful_localizations`: count of exact culprit matches.
- `undefined_localizations`: count of regressions that did not localize exactly.
- `undefined_causes`: counts of undefined localizations grouped by
  `undefined_reason`.

Undefined results do not affect TRTC metrics, but they do affect success rate and
test-run metrics.

## Oracle Accuracy Dataset

`calculate_oracle_metrics.py` writes one JSONL row per regression to
`analysis/perf_bisect/per_regression_oracle_metrics.jsonl` by default. Each row
contains:

- `regression_id`
- `summary_oracle_accuracy`
- `replicate_oracle_accuracy`

For each regression, the script reconstructs the good-to-bad parent path from
`all_commits.jsonl`, excludes the known-good endpoint, and includes the
known-bad endpoint. It then scores every matching measurement for the failing
signature from `per_revision_perf_data.jsonl`. Measurements before the culprit
are correct when they are below the midpoint baseline; measurements at or after
the culprit are correct when they are above the baseline.

The baseline is:

```text
(Good_value + bad_value) / 2
```

The written accuracies use additive smoothing so exact `0.0` and `1.0` values
are avoided when at least one measurement exists:

```text
(correct + alpha) / (total + 2 * alpha)
```

The default `alpha` is `0.5`.

Generate the oracle accuracy dataset:

```bash
python analysis/perf_bisect/calculate_oracle_metrics.py
```

Change the smoothing strength:

```bash
python analysis/perf_bisect/calculate_oracle_metrics.py --smoothing-alpha 1.0
```

## Running

Run both splits with defaults:

```bash
python analysis/perf_bisect/simulation.py --dataset all
```

Or use the local runner:

```bash
analysis/perf_bisect/run_local.sh
```

By default, result files are written to
`analysis/perf_bisect/results`.

Run one split:

```bash
python analysis/perf_bisect/simulation.py --dataset eval
python analysis/perf_bisect/simulation.py --dataset final_test
```

Change executor capacity or test duration:

```bash
python analysis/perf_bisect/simulation.py --workers 4 --test-duration-minutes 2
```

Change the number of Backfill non-monotonic interval retriggers:

```bash
python analysis/perf_bisect/simulation.py --backfill-non-monotonic-retrigger-count 3
```

Change the noisy-oracle random seed:

```bash
python analysis/perf_bisect/simulation.py --random-seed 42
```

Write outputs to a different repo-local directory:

```bash
python analysis/perf_bisect/simulation.py --output-dir analysis/perf_bisect/results_experiment
```

## Extending

To add a new oracle, implement `TestOracle` in `test_oracle.py`, register it in
`ORACLES` in `simulation.py`, and expose any required CLI configuration there.

To add a new localization algorithm, implement `CulpritLocalizer` in
`localization.py`, register it in `LOCALIZERS`, and update `build_localizer` in
`simulation.py` if the algorithm needs custom parameters.
