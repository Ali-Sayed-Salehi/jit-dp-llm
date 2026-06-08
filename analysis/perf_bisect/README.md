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
- `localization.py`: culprit localization algorithms, including backfill,
  standard midpoint bisection, midpoint multisection, and probabilistic
  bisection variants.
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
`decisions`; the decisions actually used for hard-decision localization are
recorded in `final_decisions`. Backfill results also record `retrigger_count`
and `retrigger_intervals`. Probabilistic-bisection results additionally record
the final posterior and the posterior trace.

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

The executor models worker queueing. The default is one worker. Each oracle
probe submits one test job for the revision being tested, and that job's runtime
comes from the failing signature's `job_duration` value in
`datasets/mozilla_perf_bisect/per_sig_perf_data_info.jsonl`.

Backfill submits the first probe jobs for all candidate revisions at the same
simulation timestamp. With one worker, those jobs still complete sequentially
because the worker queue is shared. Increasing `--workers` reduces queueing time.
If Backfill retriggers a non-monotonic interval, those jobs are submitted after
the earlier jobs have completed.

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
of available draws. Replicate counts are not used by the simulation.

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
pair and retriggers those two commits. If the whole candidate sequence is all
`clean`, Backfill retriggers the full candidate sequence. An all-`bad` sequence
is already monotonic, so Backfill treats the first candidate after
`good_revision` as the found culprit. After a retrigger, `final_decisions`
reconciles repeated results for each retriggered commit by majority vote over
the drawn clean/bad decisions. If clean and bad votes are tied, the simulator
submits one more test for that revision and votes again. It does not average
measurement values. The raw attempts remain in `decisions`.

This process repeats up to `--backfill-retrigger-count` times before the
localization remains undefined. The default is `2`, which models a small number
of human retriggers without turning noisy results into an unbounded retry loop.

## Standard Midpoint Multisection Localizer

`StandardMidpointMultisection` keeps the same known-good/known-bad invariant as
standard midpoint bisection, but divides the current interval into equal
sections each round. It tests the internal right-boundary revisions for those
sections in one parallel `classify_many` batch, then uses all boundary results
together to choose the adjacent clean-to-bad section as the next interval.

`--multisection-section-count` controls the requested number of sections, and
`--multisection-retrigger-count` controls how many additional boundary batches
are retriggered before voting on those boundaries. Optuna samples these from:

- `--multisection-section-count-min` / `--multisection-section-count-max`
- `--multisection-retrigger-count-min` / `--multisection-retrigger-count-max`

## Risk-Weighted Multisection Localizer

`RiskWeightedMultisection` uses the same multisection workflow and
known-good/known-bad invariant, but chooses section boundaries by cumulative
`risk_score` mass instead of equal revision counts. For the current candidate
interval, it sums risk scores over `path[low_idx + 1 : high_idx + 1]`. If the
effective section count is `m`, boundary `s` targets cumulative risk
`total_risk * s / m`; the localizer tests the internal revision whose cumulative
risk is closest to that target while keeping at least one revision in every
section.

If the current interval has no positive risk mass, it falls back to
`StandardMidpointMultisection` equal-count boundaries. Risk scores are loaded
from `--risk-scores`, which defaults to
`datasets/mozilla_perf_bisect/per_commit_risk_scores.jsonl`.

`RiskWeightedMultisection` uses the same Optuna parameters and ranges as
`StandardMidpointMultisection`:

- `--multisection-section-count-min` / `--multisection-section-count-max`
- `--multisection-retrigger-count-min` / `--multisection-retrigger-count-max`

## Probabilistic Bisection Localizer

`ProbabilisticBisection_PosteriorMedian_UniformPrior` keeps a posterior
distribution over every candidate culprit revision in the good-to-bad path. It
uses a uniform prior, so every candidate starts with equal probability.

`ProbabilisticBisection_CumulativeRiskMedian_UniformPrior` uses the same uniform
prior, posterior update, confidence threshold, repeat count, and test-run budget,
but chooses probes from `--risk-scores`. In each round it computes
`posterior_probability * risk_score` for every candidate, then selects the
internal probe that most evenly balances that posterior-weighted risk mass on
the clean side and bad side of the probe. If the current candidate path has no
positive posterior-weighted risk mass, it falls back to the posterior median.

`ProbabilisticMultiSection_CumulativeRiskQuantile_UniformPrior` uses the same
uniform prior and posterior update, but submits multiple risk-guided probes per
round. It computes the same `posterior_probability * risk_score` query weights,
then targets equal cumulative query-weight quantiles for the requested
`--multisection-section-count`. If the current candidate path has no positive
posterior-weighted risk mass, it falls back to posterior-quantile multisection.

`ProbabilisticBisection_PosteriorMedian_RiskAwarePrior` uses the same posterior
update and posterior-median query strategy, but initializes the posterior from
`--risk-scores`. It normalizes `risk_score` values over the current candidate
path, then mixes in a uniform prior controlled by
`--pba-risk-prior-uniform-weight`. This keeps low-risk commits recoverable when
the risk model is overconfident. The risk-aware variant also requires at least
one known oracle observation before accepting a posterior-MAP culprit, unless
there is only one candidate revision.

Posterior-median PBA variants choose the first revision whose cumulative
posterior probability reaches `0.5`. Bisection PBA localizers submit
`--pba-repeat-count` observations for one revision per round. `pba_batch_size` is
fixed at `1`, so these variants are sequential rather than batched.

For each clean/bad observation, the localizer reads the oracle's accuracy for
that probe. With `SummaryComparison`, this is the regression's
`summary_oracle_accuracy`. If the observation matches what would be expected for
a possible culprit, that culprit's posterior probability is multiplied by the
oracle accuracy. If the observation contradicts that possible culprit, its
posterior probability is multiplied by `1 - accuracy`. The posterior is then
renormalized to sum to `1.0`.

The localizer accepts a single culprit only when the highest posterior
probability is at least `--pba-confidence-threshold` and no other candidate is
tied for that highest probability. If there is a tie, it keeps probing until the
tie breaks or the test budget is exhausted.

`--pba-max-test-runs` is a hard budget for one regression. If the budget is
reached before a candidate passes the confidence threshold, the result records
the current MAP revision in `found_revision`, but the localization is still
undefined with `posterior_confidence_below_threshold`. If multiple candidates
remain tied for highest posterior probability, the result is undefined with
`ambiguous_posterior_tie`.

The tunable PBA parameters are:

- `pba_confidence_threshold`
- `pba_repeat_count`
- `pba_max_test_runs`
- `pba_risk_prior_uniform_weight` for
  `ProbabilisticBisection_PosteriorMedian_RiskAwarePrior`

Optuna samples these from:

- `--pba-confidence-threshold-min` / `--pba-confidence-threshold-max`
- `--pba-repeat-count-min` / `--pba-repeat-count-max`
- `--pba-max-test-runs-min` / `--pba-max-test-runs-max`
- `--pba-risk-prior-uniform-weight-min` /
  `--pba-risk-prior-uniform-weight-max`

## Probabilistic Multisection Localizer

Probabilistic multisection localizers use the same uniform prior, posterior
update, confidence threshold, repeat count, and test-run budget as the uniform
probabilistic bisection localizer. Their query strategy differs: each round
divides a query distribution into `--multisection-section-count` sections and
probes the internal boundaries in one batch.

`ProbabilisticMultiSection_PosteriorQuantile_UniformPrior` uses posterior mass
as its query distribution.
`ProbabilisticMultiSection_CumulativeRiskQuantile_UniformPrior` uses
posterior-weighted risk mass from `--risk-scores` as its query distribution,
falling back to posterior mass if the current interval has no positive query
risk.

If multiple quantile targets land on the same revision, the localizer probes
that revision once in the distinct boundary set. Additional observations for
that revision come from `--pba-repeat-count`, so section count controls fan-out
and repeat count controls repeated evidence. The localizer also avoids using
the known-bad endpoint as a quantile boundary because that probe cannot split
the candidate culprit set.

These localizers do not use `--multisection-retrigger-count`; noisy evidence is
handled through Bayesian posterior updates and `--pba-repeat-count`. Optuna
tunes:

- `multisection_section_count`
- `pba_confidence_threshold`
- `pba_repeat_count`
- `pba_max_test_runs`

## Metrics

Each summary output reports:

- `mean_elapsed_hours`: mean wall-clock test elapsed time across all
  regressions, including undefined localizations.
- `max_elapsed_hours`: max wall-clock test elapsed time across all regressions.
- `mean_test_runs`: mean number of submitted test jobs across all regressions,
  including undefined localizations.
- `max_test_runs`: max number of submitted test jobs across all regressions.
- `success_rate_percent`: percentage of regressions where the exact culprit was
  found.
- `successful_localizations`: count of exact culprit matches.
- `undefined_localizations`: count of regressions that did not localize exactly.
- `undefined_causes`: counts of undefined localizations grouped by
  `undefined_reason`.

Undefined results affect elapsed-time, success-rate, and test-run metrics.

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

Change executor capacity:

```bash
python analysis/perf_bisect/simulation.py --workers 4
```

Sweep executor capacity and plot final-test metrics:

```bash
python analysis/perf_bisect/plot_for_machine_counts.py \
  --worker-counts 1 2 4 8 \
  --optuna-trials 50 \
  --random-seed 42
```

The sweep script runs eval and final-test simulation for each worker count,
tuning parameters independently at each count. It writes aggregate sweep results
to `analysis/perf_bisect/results/machine_count_sweep_final_test.json` and plots
to `analysis/perf_bisect/results/plots` by default.

Change the number of Backfill retriggers:

```bash
python analysis/perf_bisect/simulation.py --backfill-retrigger-count 3
```

Run only the posterior-median PBA localizer:

```bash
python analysis/perf_bisect/simulation.py \
  --dataset eval \
  --localizers ProbabilisticBisection_PosteriorMedian_UniformPrior
```

Run only the cumulative-risk-median PBA localizer:

```bash
python analysis/perf_bisect/simulation.py \
  --dataset eval \
  --localizers ProbabilisticBisection_CumulativeRiskMedian_UniformPrior
```

Run only the cumulative-risk-quantile PBA multisection localizer:

```bash
python analysis/perf_bisect/simulation.py \
  --dataset eval \
  --localizers ProbabilisticMultiSection_CumulativeRiskQuantile_UniformPrior
```

Run only the risk-aware posterior-median PBA localizer:

```bash
python analysis/perf_bisect/simulation.py \
  --dataset eval \
  --localizers ProbabilisticBisection_PosteriorMedian_RiskAwarePrior
```

Change PBA defaults:

```bash
python analysis/perf_bisect/simulation.py \
  --pba-confidence-threshold 0.9 \
  --pba-repeat-count 2 \
  --pba-max-test-runs 120 \
  --pba-risk-prior-uniform-weight 0.05
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
