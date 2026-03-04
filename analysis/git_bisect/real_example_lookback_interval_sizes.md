# Real example: final bisection search interval size per lookback strategy

This note picks **one real regression bug** from the Mozilla JIT dataset and measures how each lookback strategy changes the **final interval** that bisection has to search.

## Data sources

- Bugs: `datasets/mozilla_jit/mozilla_jit_2022.jsonl`
- Commits: `datasets/mozilla_jit/all_commits.jsonl`
- Risk predictions (defines simulation window + risk_by_index): `analysis/git_bisect/risk_predictions_eval.json`
- Tuned parameters source: `analysis/git_bisect/results/simulation_optuna_final_test.json`
- Lookback implementations: `analysis/git_bisect/lookback.py`
- Interval-tightening logic mirrored from: `analysis/git_bisect/bisection.py` (`_tighten_bounds` semantics)

## Example regression

The example bug was chosen as a “large gap” case where the observed bad commit is far after the regressor commit:

- `bug_id`: `1945830`
- `bug_creation_time_utc`: `2025-02-04T14:54:45+00:00`
- `available_regressor` bug id: `1886134`
- Culprit commit (regressor revision):
  - `culprit_index`: `758233`
  - `node`: `e8beb6a08c6d1a925f8cbceb69c4b7ff0f13a5c8`
  - `commit_time_utc`: `2024-10-16T13:41:55+00:00`
- Observed bad commit (the “start_index” / “bad_index” used for lookback):
  - `bad_index`: `772326`
  - `node`: `e7f370ca669c9cd4b838b6ad1cde91e8870b235c`
  - `commit_time_utc`: `2025-02-04T14:44:08+00:00`
- Simulation window (derived from risk predictions):
  - `window_start`: `755957` (`9ceee1849b058456a364a522bfc5baa78c46b4c7`, `2024-09-30T16:01:09+00:00`)
  - `window_end`: `773481` (`4aadb751864afd657cddeb74d2853ff2c2497699`, `2025-02-08T16:36:00+00:00`)

## What “final search interval size” means here

For each lookback strategy `S`, we run:

1) `out = S.find_good_index(start_index=bad_index, culprit_index=culprit_index, start_time_utc=bug_creation_time_utc)`
2) We then tighten the initial bounds `(good_index, bad_index]` using `out.known_results`, exactly as bisection does before it starts testing:
   - `low`  = latest known passing commit index
   - `high` = earliest known failing commit index
   - final bisection interval is `(low, high]`
   - interval **size** is `high - low` (number of candidate commits in the interval, under the linear index model)

All strategies below were run using the **Optuna-tuned lookback parameters** recorded in
`analysis/git_bisect/results/simulation_optuna_final_test.json`, using the parameter set associated
with each lookback strategy’s `+GB` (baseline git-bisect) combo.

## Results

Only the “main” variant of each family is shown here (i.e., no `-ad`, `-ai`, or `-ff` variants).

| Strategy | Tuned params (`simulation_optuna_final_test.json`, `+GB`) | steps | good | low | high | size |
|---|---:|---:|---:|---:|---:|---:|
| `no_lookback` |  | 0 | 755957 | 755957 | 772326 | 16369 |
| `fixed_stride` | stride=499 | 29 | 757855 | 757855 | 758354 | 499 |
| `nightly_builds` |  | 112 | 758135 | 758135 | 758328 | 193 |
| `risk_aware_trigger` | threshold=0.9760307109149444 | 17 | 756842 | 756842 | 758455 | 1613 |
| `rwlb-s` | threshold=181.40549461310692 | 13 | 757523 | 757523 | 758657 | 1134 |
| `rwlb-ls` | threshold=0.9999575670495091 | 331 | 758202 | 758202 | 758238 | 36 |
| `time_window` | hours=175 | 16 | 757363 | 757363 | 758444 | 1081 |

## Probe step details (first 2 + last 2)

For each lookback strategy with `steps=N`, “step *k*” below refers to the *k-th tested commit* produced by the strategy (the known-bad `bad_index=772326` is **not** re-tested).

For each step we report:

- `prev`: the reference index (step 1 uses `bad_index`; later steps use the previous probe’s index)
- `idx`: the commit index tested on this step
- `step size`: `|idx - prev|` (number of commits jumped)
- `dir`: whether the probe moved `back` (earlier history) or `forward` (later history)
- `outcome`: `PASS` iff `idx < culprit_index=758233`, else `FAIL`

### `no_lookback` (NLB)

`steps=0` (no probes).

### `fixed_stride` (FSLB, stride=499)

| step | prev | idx | step size | dir | outcome |
|---:|---:|---:|---:|:---:|:---:|
| 1 | 772326 | 771827 | 499 | back | FAIL |
| 2 | 771827 | 771328 | 499 | back | FAIL |
| 28 | 758853 | 758354 | 499 | back | FAIL |
| 29 | 758354 | 757855 | 499 | back | PASS |

### `nightly_builds` (NBLB)

| step | prev | idx | step size | dir | outcome |
|---:|---:|---:|---:|:---:|:---:|
| 1 | 772326 | 772207 | 119 | back | FAIL |
| 2 | 772207 | 771998 | 209 | back | FAIL |
| 111 | 758494 | 758328 | 166 | back | FAIL |
| 112 | 758328 | 758135 | 193 | back | PASS |

Note: `nightly_builds` enforces a monotone-by-index probe sequence (treat each failing nightly as the new upper bound) so that “going back one day” never selects a later commit index, even when commit timestamps are non-monotone.

### `risk_aware_trigger` (RATLB, threshold=0.9760307109149444)

| step | prev | idx | step size | dir | outcome |
|---:|---:|---:|---:|:---:|:---:|
| 1 | 772326 | 771281 | 1045 | back | FAIL |
| 2 | 771281 | 770737 | 544 | back | FAIL |
| 16 | 760290 | 758455 | 1835 | back | FAIL |
| 17 | 758455 | 756842 | 1613 | back | PASS |

### `rwlb-s` (RWLBS, threshold=181.40549461310692)

| step | prev | idx | step size | dir | outcome |
|---:|---:|---:|---:|:---:|:---:|
| 1 | 772326 | 771106 | 1220 | back | FAIL |
| 2 | 771106 | 770109 | 997 | back | FAIL |
| 12 | 759621 | 758657 | 964 | back | FAIL |
| 13 | 758657 | 757523 | 1134 | back | PASS |

### `rwlb-ls` (RWLBLS, threshold=0.9999575670495091)

| step | prev | idx | step size | dir | outcome |
|---:|---:|---:|---:|:---:|:---:|
| 1 | 772326 | 772291 | 35 | back | FAIL |
| 2 | 772291 | 772257 | 34 | back | FAIL |
| 330 | 758255 | 758238 | 17 | back | FAIL |
| 331 | 758238 | 758202 | 36 | back | PASS |

### `time_window` (TWLB, hours=175)

| step | prev | idx | step size | dir | outcome |
|---:|---:|---:|---:|:---:|:---:|
| 1 | 772326 | 771093 | 1233 | back | FAIL |
| 2 | 771093 | 770147 | 946 | back | FAIL |
| 15 | 759533 | 758444 | 1089 | back | FAIL |
| 16 | 758444 | 757363 | 1081 | back | PASS |

## Notes (qualitative)

- These tuned parameters were selected to optimize end-to-end test cost in the simulation (lookback + bisection), not to minimize interval size alone.
- In this example, `rwlb-ls` produced the tightest interval (**size=36**) but required many lookback probes (**steps=331**).
