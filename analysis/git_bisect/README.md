# Git Bisect Simulation (Mozilla JIT dataset)

This folder contains a lightweight simulator for “git bisect”-style debugging on the Mozilla JIT bug dataset. It evaluates **lookback strategies** (how to pick a known-good commit) paired with **bisection strategies** (how to find the culprit once you have good/bad boundaries), under a simplified linear-history model.

The main entrypoint is `analysis/git_bisect/simulate.py`.

## What the simulator models

### Linear history + single culprit
The simulator assumes commits form a single linear sequence indexed `0..N-1`. For each regression bug we simulate:

- A **culprit index** `c` (the regressor commit).
- A **known-bad index** `b` (the commit at/just before the bug’s `bug_creation_time`).
- A **known-good index** `g` (chosen by the lookback strategy).

### Notation + test oracle
All algorithms operate on integer commit indices.

**Test model (monotone failures):**

For a bug with culprit index `c`, the simulated test outcome is a step function:

$$
\text{test}(i) =
\begin{cases}
\text{PASS} & i < c \\\\
\text{FAIL} & i \ge c
\end{cases}
$$

Once a lookback strategy returns a passing commit `g` (so `g < c`) and we already have a failing commit `b` (so `c \le b`), the culprit is guaranteed to lie in:

$$
c \in (g, b]
$$

This makes bisection deterministic and lets us count “how many tests would be run” by each policy.

### What counts as a “test”
The simulator’s **cost** is the number of times we “run a test” on a specific commit index. For each bug:

$$
\text{tests\_per\_bug} = \text{lookback\_tests} + \text{bisection\_tests}
$$

Important modeling choice: the simulator treats `b` as an *already-observed* failing commit (e.g., the regression was first noticed there), so it is **not** counted as a test. Lookback/bisection only count *additional* probes.

### Commit window from risk predictions
Risk predictions define the commit window used for simulation:

- The risk files (e.g. `risk_predictions_eval.json`) contain per-commit predictions keyed by `commit_id`.
- The simulator maps those `commit_id`s to commit indices in `datasets/mozilla_jit/all_commits.jsonl`.
- The **min/max matched indices** define `[window_start, window_end]`.

Outside this window, risk is treated as unavailable (`None`) and many bugs are skipped if their good/bad commit would fall outside the window.

Lookback strategies are also constrained to return a `good_index` within the window; if a policy would otherwise need to test earlier history, it falls back to testing `window_start` as the earliest available commit.

### Risk scores and “risk mass” over an interval
Some strategies use a per-commit **risk score**:

- Let `p_i ∈ [0, 1]` be the model’s score for commit `i`, interpreted as `P(POSITIVE)` / “how likely this commit is a regressor”.
- Inside `[window_start, window_end]`, missing predictions are treated as `p_i = 0.0`.
- Outside the window the simulator sets risk to “unavailable” (`None`). Strategies are constrained to probe no earlier than `window_start`; NLB uses `window_start` as its known-good boundary.

Two common ways to aggregate risk over a contiguous range of commits `[a, b)`:

1) **Sum mass**

$$
R_{\text{sum}}([a,b)) = \sum_{i=a}^{b-1} p_i
$$

2) **Noisy-OR / log-survival mass**

$$
R_{\text{ls}}([a,b)) = 1 - \prod_{i=a}^{b-1} (1 - p_i)
$$

`R_ls` is the probability that **at least one** commit in the interval is “positive” under an (imperfect) independence assumption. For small `p_i`, `R_ls([a,b)) ≈ R_sum([a,b))`.

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

Concretely (for the default `label_order`):

$$
p_i = P(\text{POSITIVE}) =
\begin{cases}
\text{confidence} & \text{prediction}=\text{POSITIVE} \\\\
1-\text{confidence} & \text{prediction}=\text{NEGATIVE}
\end{cases}
$$

With the default `label_order=["NEGATIVE","POSITIVE"]`, this corresponds to `prediction=1 → POSITIVE` and `prediction=0 → NEGATIVE`.

## Strategies

Each combo is `LOOKBACK_CODE + "+" + BISECTION_CODE` (e.g. `NBLB+GB`).

### Lookback strategies (finding a known-good boundary)
Lookback answers: “starting from a known-bad observation `b`, how do we find a passing commit `g` to bound the culprit?”

Conceptually, a lookback strategy generates a decreasing sequence of candidate commits `x_1 > x_2 > …`, tests each candidate, and returns the first passing one:

- If `test(x_k) = FAIL`, set the known-bad boundary `b ← x_k` and keep going.
- If `test(x_k) = PASS`, return `g = x_k` and stop.

In this simulator, **only the tested candidates count** toward `lookback_tests`; scanning commit metadata/risk scores is treated as free.

#### NBLB: Nightly build lookback
Walk backward by **UTC days** and test one “nightly boundary” commit per day until a pass is found.

- Define day boundaries in UTC (midnight-to-midnight).
- For each day `D` going backward, choose the last commit strictly before `00:00` UTC on day `D+1`, and test it.
- Return the first such boundary commit with index `< c`.

This is meant to approximate “bisecting by nightly builds”: cheap to try one build per day, but potentially coarse.

#### NLB: No lookback
Do **no additional tests** to find a good boundary; use the simulation window start commit:

$$
g = \text{window\_start}
$$

The simulator skips bugs where the regression predates the risk window (i.e., `c <= window_start`), since there is no known-good commit available strictly before the culprit within the window. When applicable, bisection searches over `(window_start, b]`.

#### FSLB: Fixed-stride lookback (Optuna: `FSLB_stride`)
Choose a stride length `s` in commits and repeatedly jump back by exactly `s`:

$$
x_k = \max(\text{window\_start},\; b_{k-1} - s)
$$

Stop on the first `x_k < c`. If you ignore the window clamp, the number of lookback tests needed is:

$$
k^\* = \left\lceil \frac{b_0 - (c-1)}{s} \right\rceil
$$

#### AFSLB: Adaptive fixed-stride lookback (Optuna: `AFSLB_stride`, `AFSLB_alpha`)
Like FSLB, but shrink the stride geometrically after each failed test.

- Step `k` uses stride `s_k = ceil(s · α^{k-1})`, with `α ∈ [0,1]`.
- Candidate is `x_k = b_{k-1} - s_k` (clamped to `window_start`).

The total distance moved back after `K` failed steps (ignoring clamps/ceiling) is a geometric sum:

$$
\sum_{k=1}^{K} s\,\alpha^{k-1} = s\,\frac{1-\alpha^K}{1-\alpha}
$$

#### RATLB: Risk-aware trigger lookback (Optuna: `RATLB_threshold`)
Let `T` be a threshold. Scan backward from the current known-bad boundary `b` looking for a **trigger** commit `i` with `p_i > T`. When you find one, test the commit just before it:

$$
x = i - 1
$$

- If `x < c`, it passes and becomes `g`.
- If it fails, treat it as the new known-bad boundary and keep scanning earlier.
- If no triggers exist in-range, fall back to testing `window_start`.

Intuition: “when the model says a commit looks suspicious, check just before it to try to bracket the regression.”

#### RWLBS: Risk-weighted lookback (sum mass) (Optuna: `RWLBS_threshold`)
Let `T ≥ 0` be a threshold. From the current known-bad boundary `b`, pick the **closest** earlier commit `x < b` such that the summed risk between `x` and `b` reaches the threshold:

$$
x = \arg\max_{j < b} \left\{ j \;:\; R_{\text{sum}}([j, b)) \ge T \right\}
$$

Test `x`; on failure set `b ← x` and repeat. If even the full in-window interval can’t reach `T`, fall back to testing `window_start`.

This makes lookback steps “smaller” when the region just before `b` already contains a lot of predicted risk.

#### RWLBLS: Risk-weighted lookback (log-survival mass) (Optuna: `RWLBLS_threshold`)
Same policy as RWLBS, but threshold the noisy-OR mass instead of the sum:

$$
x = \arg\max_{j < b} \left\{ j \;:\; R_{\text{ls}}([j, b)) \ge T \right\}
$$

Here `T ∈ [0,1]`. This treats many small-risk commits differently from one large-risk commit.

#### TWLB: Time-window lookback (Optuna: `TWLB_hours`)
Let commit `i` have timestamp `t_i` (UTC). Choose a fixed time window `H` hours. From the current boundary `b`, jump back by time and test the nearest commit at-or-before the target time:

$$
\text{target\_time} = t_b - H
$$

Test the commit `x` with the largest `x < b` such that `t_x ≤ target_time`. On failure, set `b ← x` and repeat. If the target time predates all in-window history, fall back to testing `window_start`.

#### ATWLB: Adaptive time-window lookback (Optuna: `ATWLB_hours`, `ATWLB_alpha`)
Like TWLB, but shrink the window geometrically after each failure:

$$
H_k = H \cdot \alpha^{k-1}
$$

So the total time you jump back after `K` failed steps (ignoring discretization to commits) is:

$$
\sum_{k=1}^{K} H\,\alpha^{k-1} = H\,\frac{1-\alpha^K}{1-\alpha}
$$

`LookbackOutcome.steps` is treated as the number of lookback tests executed for that bug.

### Bisection strategies (pinpointing the culprit once you have bounds)
After lookback, we have `g < c ≤ b` and want to identify `c` with minimal additional tests.

All bisection strategies maintain the invariant `test(g)=PASS` and `test(b)=FAIL`, then repeatedly choose a probe `m ∈ (g,b)`:

- If `test(m)=FAIL`, update `b ← m`
- If `test(m)=PASS`, update `g ← m`

When `b = g + 1`, the culprit must be `c = b`.

#### GB: Standard git bisect (binary search)
Probe the midpoint by index:

$$
m = \left\lfloor \frac{g+b}{2} \right\rfloor
$$

This ignores risk predictions. In the worst case, the number of probes is:

$$
\left\lceil \log_2(b-g) \right\rceil
$$

#### SWBB: Sequential walk backward
Start near the known-bad boundary and walk back one commit at a time:

$$
m = b-1,\; b-2,\; b-3,\; \dots
$$

Stop on the first passing probe; the culprit is the next known-bad boundary. Under the monotone test model, the cost is linear in how far the culprit is from `b` (best when `c` is very recent).

#### SWFB: Sequential walk forward
Start just after the known-good boundary and walk forward:

$$
m = g+1,\; g+2,\; g+3,\; \dots
$$

Stop on the first failing probe; that failing commit is the culprit. Under the monotone test model:

$$
\text{bisection\_tests} = c - g
$$

#### TKRB: Top-K risk-first bisection (Optuna: `TKRB_k`)
Two phases:

1) **Risk-first phase:** pick the `K` commits with highest risk `p_i` in `(g,b]` and test each candidate `i` along with its predecessor `i-1`. If `test(i)=FAIL` and `test(i-1)=PASS`, then `i` is the culprit.

2) **Fallback phase:** if not found, run standard binary search on `(g,b]`, reusing any cached test outcomes from phase 1 so repeated probes are free.

This is meant to model “check the most suspicious commits first, but don’t give up on completeness.”

#### RWBS: Risk-weighted bisection (sum mass)
Choose probes to (approximately) balance summed risk mass on both sides of the split. For the current interval `(g,b]`, choose `m` so that:

$$
R_{\text{sum}}((g,m]) \approx R_{\text{sum}}((m,b])
$$

Equivalently, `m` behaves like a weighted median of the interval using weights `p_i` while preserving time order (only contiguous halves are allowed).

#### RWBLS: Risk-weighted bisection (log-survival mass)
Same as RWBS, but balance the noisy-OR / log-survival mass instead of the sum:

$$
R_{\text{ls}}((g,m]) \approx R_{\text{ls}}((m,b])
$$

This tends to treat “many small risks” differently than “one big risk” when deciding where to probe next.

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

### Selecting strategies
By default, the simulator evaluates **all** lookback and bisection strategies. To restrict the run, pass comma-separated lists of strategy **codes** (or internal names):

```bash
python analysis/git_bisect/simulate.py --lookback NLB,FSLB,AFSLB --bisection GB,RWBS --mopt-trials 200
```

## Outputs and metrics

### Eval output (`--output-eval`)
Includes:
- Dataset metadata (commit window, risk file path, bug counts)
- Per-combo Optuna details (selected Pareto-optimal trial params, selected objective values)
- Per-combo metrics (including a `bugs` breakdown of processed/skipped)

Optuna tuning is multi-objective: it minimizes `(max_tests_per_search, mean_tests_per_search)`.

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

Baseline for comparisons is `NLB+GB` (no lookback + standard git bisect). Percent values are computed as:
`100 * (baseline - value) / baseline` (positive is better / fewer tests).

## Adding a new strategy

- Implement the strategy in `analysis/git_bisect/lookback.py` or `analysis/git_bisect/bisection.py`.
- Add a `StrategySpec` entry in `analysis/git_bisect/simulate.py`:
  - Choose a short `code` (used in combo keys).
  - Provide `default_params` and `build`.
  - If tunable, provide `suggest_params` and ensure Optuna parameter names are prefixed with `<CODE>_` (e.g. `TWLB_hours`).
