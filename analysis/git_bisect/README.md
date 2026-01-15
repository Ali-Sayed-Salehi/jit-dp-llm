# Git Bisect Simulation (Mozilla JIT dataset)

This folder contains a lightweight simulator for “git bisect”-style debugging on the Mozilla JIT bug dataset. It evaluates **lookback strategies** (how to pick a known-good commit) paired with **bisection strategies** (how to find the culprit once you have good/bad boundaries), under a simplified linear-history model.

The main entrypoint is `analysis/git_bisect/simulate.py`.

## What the simulator models

### Linear history + single culprit
The simulator assumes commits form a single linear sequence indexed `0..N-1`. For each regression bug we simulate:

- A **culprit index** `c` (the regressor commit).
- A **known-bad index** `b` (the commit at/just before the bug’s `bug_creation_time`).
- A **known-good index** `g` (chosen by the lookback strategy).

**Test model (monotone failures):**
- Testing commit `i` **fails** iff `i >= c`
- Testing commit `i` **passes** iff `i < c`

This makes bisection deterministic and lets us count “how many tests would be run” by each policy.

### Commit window from risk predictions
Risk predictions define the commit window used for simulation:

- The risk files (e.g. `risk_predictions_eval.json`) contain per-commit predictions keyed by `commit_id`.
- The simulator maps those `commit_id`s to commit indices in `datasets/mozilla_jit/all_commits.jsonl`.
- The **min/max matched indices** define `[window_start, window_end]`.

Outside this window, risk is treated as unavailable (`None`) and many bugs are skipped if their good/bad commit would fall outside the window.

### “Available regressor” selection
Mozilla bugs can reference other bugs in `regressed_by`. For this simulation:

- A regressor bug is “available” if its `revision` exists in `all_commits.jsonl` *and* is inside the current risk window.
- If multiple regressors are available, the simulator uses the **latest** one (highest commit index) as the single `culprit_index`.

## Inputs

Defaults in `analysis/git_bisect/simulate.py`:

- Bugs: `datasets/mozilla_jit/mozilla_jit.jsonl`
- Commits: `datasets/mozilla_jit/all_commits.jsonl`
- Risk predictions:
  - Eval: `analysis/git_bisect/risk_predictions_eval.json`
  - Final: `analysis/git_bisect/risk_predictions_final_test.json`

### Risk prediction file format
`simulate.py` expects a JSON file with either `results` or `samples` containing rows like:

```json
{"commit_id": "<node>", "prediction": 0|1, "confidence": 0.0-1.0}
```

`confidence` is the probability of the predicted class. The file may include `label_order` (default: `["NEGATIVE", "POSITIVE"]`) so the simulator can convert each row into `P(POSITIVE)` for risk-aware strategies.

## Strategies

Each combo is `LOOKBACK_CODE + "+" + BISECTION_CODE` (e.g. `NBLB+GB`).

### Lookback strategies (`analysis/git_bisect/lookback.py`)

- **NBLB** (`NightlyBuildLookback`): walks back by UTC days and tests the “nightly build” boundary commit for each day until it finds a passing commit.
- **FSLB** (`FixedStrideLookback`, Optuna: `FSLB_stride`): steps back by a fixed number of commits each time.
- **RATLB** (`RiskAwareTriggerLookback`, Optuna: `RATLB_threshold`): scans backward; whenever a commit’s risk exceeds threshold `T`, it tests the *previous* commit (`i-1`). The first passing candidate is returned.
- **TWLB** (`TimeWindowLookback`, Optuna: `TWLB_hours`): steps back by a fixed time window of `N` hours (in commit time), tests the commit at/before that timestamp, and repeats until it finds a passing commit.

`LookbackOutcome.steps` is treated as the number of lookback tests executed for that bug.

### Bisection strategies (`analysis/git_bisect/bisection.py`)

- **GB** (`GitBisectBaseline`): standard binary search on `(good, bad]`.
- **SWBB** (`SequentialWalkBackwardBisection`): tests `bad-1, bad-2, ...` until it finds a pass; the culprit is the next known-bad boundary.
- **SWFB** (`SequentialWalkForwardBisection`): tests `good+1, good+2, ...` until it finds a fail; that first failing index is the culprit.
- **TKRB** (`TopKRiskFirstBisection`, Optuna: `TKRB_k`): tests the top-K riskiest commits (and their predecessors) first, caching results; if not found, falls back to binary search while reusing cached test outcomes.
- **RWABS** (`RiskWeightedAdaptiveBisectionSum`): chooses midpoints that balance summed risk mass across the two halves of the current search interval.
- **RWABLS** (`RiskWeightedAdaptiveBisectionLogSurvival`): like RWABS, but balances combined probability `1 - ∏(1 - p)` over each half.

## Running

### Full eval + final replay
This tunes each combo on the eval predictions, writes `--output-eval`, then replays the tuned params on the final predictions:

```bash
python analysis/git_bisect/simulate.py --mopt-trials 200
```

### Final-only replay
Load tuned params from the eval output JSON and run only the final predictions:

```bash
python analysis/git_bisect/simulate.py --final-only
```

### Dry run
For fast iteration, simulate only the first 1000 bug rows:

```bash
python analysis/git_bisect/simulate.py --dry-run --final-only
```

## Outputs and metrics

### Eval output (`--output-eval`)
Includes:
- Dataset metadata (commit window, risk file path, bug counts)
- Per-combo Optuna details (best trial params, best value)
- Per-combo metrics (including a `bugs` breakdown of processed/skipped)

### Final output (`--output-final`)
Includes:
- Dataset metadata
- Best-combo keys: `best_combo_by_total_tests`, `best_combo_by_mean_tests_per_search`, `best_combo_by_max_tests_per_search`
- Per-combo metrics, plus:
  - `total_tests_saved_vs_baseline_pct`, `mean_tests_per_search_saved_vs_baseline_pct`, etc.
  - `params`: flattened “used params” like `{"Lookback_stride": 350, "Bisection_k": 20}`

Per-combo metrics fields:
- `total_tests`: lookback tests + bisection tests summed over all processed bugs
- `total_lookback_tests`, `total_bisection_tests`
- `mean_tests_per_search`, `max_tests_per_search`
- `total_culprits_found`

Baseline for comparisons is `NBLB+GB` (nightly builds + standard git bisect). Percent values are computed as:
`100 * (baseline - value) / baseline` (positive is better / fewer tests).

## Adding a new strategy

- Implement the strategy in `analysis/git_bisect/lookback.py` or `analysis/git_bisect/bisection.py`.
- Add a `StrategySpec` entry in `analysis/git_bisect/simulate.py`:
  - Choose a short `code` (used in combo keys).
  - Provide `default_params` and `build`.
  - If tunable, provide `suggest_params` and ensure Optuna parameter names are prefixed with `<CODE>_` (e.g. `TWLB_hours`).
