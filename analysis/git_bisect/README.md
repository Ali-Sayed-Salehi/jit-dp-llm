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

### Optional: window-start lookback penalty
Some lookback policies may end up using the **simulation window start** commit as the known-good boundary (`g = window_start`)—either because the strategy is NLB (no lookback), or because it clamps/falls back to the earliest available commit in the window.

To model an additional fixed overhead in that situation, you can enable `--penalize-window-start-lookback`. When enabled, the simulator adds `--window-start-lookback-penalty-tests` (default: `4`) to `bisection_tests` **per processed bug** whenever `good_index == window_start`.

Concretely, for each processed bug:

$$
\text{tests\_per\_bug} =
\text{lookback\_tests} + \text{bisection\_tests} + \text{window\_start\_penalty}
$$

where:

$$
\text{window\_start\_penalty} =
\begin{cases}
\text{window\_start\_lookback\_penalty\_tests} & \text{if penalize is enabled and } g=\text{window\_start} \\\\
0 & \text{otherwise}
\end{cases}
$$

Implementation detail: the penalty is accounted under **bisection tests** (so it affects `total_bisection_tests` and `total_tests`, and therefore `mean_tests_per_search` / `max_tests_per_search`).

Bugs whose regression predates the risk window (i.e., there is no in-window known-good commit strictly before the culprit) are still skipped as before; enabling this penalty does not change which bugs are processed vs. skipped.

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

### Bugs file (`mozilla_jit.jsonl`)
`mozilla_jit.jsonl` is JSONL (one JSON object per line). The simulator expects at least:

- `bug_id`: bug identifier (string or int)
- `regression`: boolean (only rows with `true` are simulated)
- `bug_creation_time`: ISO datetime string (used to select the known-bad commit)
- `regressed_by`: list of regressor bug ids (optional, can be empty/missing)

To map a regressor bug id to a commit, the simulator looks up the referenced bug row and reads:

- `revision`: commit identifier (must match a `node` in `all_commits.jsonl`)

Because bugs can list multiple regressors, the simulator computes an `available_regressor` field during preprocessing:

- A regressor is “available” if its `revision` exists in the commit list *and* its commit index is inside the current risk window.
- If multiple regressors are available, the simulator uses the **latest** one (highest commit index) as the single culprit for that bug.

### Commits file (`all_commits.jsonl`)
`all_commits.jsonl` is JSONL ordered by history (index `0..N-1`). Each row is treated as a Mercurial commit and must include:

- `node`: commit hash / id (string; used as the commit key everywhere)
- `date`: list whose first element is a UNIX timestamp in seconds (used to build commit-time search)

The simulator uses this file to:

- Build `node_to_index` (commit id → linear index)
- Convert `bug_creation_time` into a known-bad index (`bad_index`) by selecting the latest commit with timestamp `≤ bug_creation_time`

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

#### Forced-fallback variants (-ff)
For every lookback strategy **except** NLB and NBLB, the simulator also includes a `-ff` (“forced fallback”) variant that adds an Optuna-tuned `max_trials` parameter.

If the strategy would execute more than `max_trials` lookback tests while searching backward, it stops early and falls back to using `window_start` as the known-good boundary (instead of continuing to search for a closer passing commit).

Why this exists:
- Some lookback policies can be arbitrarily expensive on long histories (many probes before finding a pass).
- `-ff` turns those into “bounded lookback cost” policies: you cap the lookback probes and accept a larger bisection interval.

How it is represented in the code:
- The “forced fallback” is implemented by passing `max_trials` into the same lookback class (the `-ff` variants in `lookback.py` are name-only subclasses).
- In `simulate.py` these show up as separate strategy specs with their own `code` (e.g. `FSLB-FF`) and their own tunable `max_trials`.
- `LookbackOutcome.steps` counts only the probes that were actually executed before stopping; the fallback boundary selection itself does not add an extra lookback step beyond those probes.
- You can select these variants via `--lookback` using either the strategy codes (e.g. `FSLB-FF`) or the internal names from `lookback.py` (e.g. `fixed_stride-ff`).
- If you enable the optional window-start penalty (`--penalize-window-start-lookback`), then any case where the forced fallback produces `g = window_start` will also incur that penalty (because the penalty is keyed on `good_index == window_start`).

Example mapping of base strategies → forced-fallback variants:
- `FSLB` → `FSLB-FF` (Optuna tunes `stride` and `max_trials`)
- `FSLB-AD` → `FSLB-AD-FF` (Optuna tunes `stride`, `alpha`, and `max_trials`)
- `FSLB-AI` → `FSLB-AI-FF` (Optuna tunes `stride`, `alpha`, and `max_trials`)
- `RATLB` → `RATLB-FF` (Optuna tunes `threshold` and `max_trials`)
- `RATLB-AD` → `RATLB-AD-FF` (Optuna tunes `threshold`, `alpha`, and `max_trials`)
- `RATLB-AI` → `RATLB-AI-FF` (Optuna tunes `threshold`, `alpha`, and `max_trials`)
- `RWLBS` → `RWLBS-FF` (Optuna tunes `threshold` and `max_trials`)
- `RWLBS-AD` → `RWLBS-AD-FF` (Optuna tunes `threshold`, `alpha`, and `max_trials`)
- `RWLBS-AI` → `RWLBS-AI-FF` (Optuna tunes `threshold`, `alpha`, and `max_trials`)
- `RWLBLS` → `RWLBLS-FF` (Optuna tunes `threshold` and `max_trials`)
- `RWLBLS-AD` → `RWLBLS-AD-FF` (Optuna tunes `threshold`, `alpha`, and `max_trials`)
- `RWLBLS-AI` → `RWLBLS-AI-FF` (Optuna tunes `threshold`, `alpha`, and `max_trials`)
- `TWLB` → `TWLB-FF` (Optuna tunes `hours` and `max_trials`)
- `TWLB-AD` → `TWLB-AD-FF` (Optuna tunes `hours`, `alpha`, and `max_trials`)
- `TWLB-AI` → `TWLB-AI-FF` (Optuna tunes `hours`, `alpha`, and `max_trials`)

#### FSLB: Fixed-stride lookback (Optuna: `FSLB_stride`)
Choose a stride length `s` in commits and repeatedly jump back by exactly `s`:

$$
x_k = \max(\text{window\_start},\; b_{k-1} - s)
$$

Stop on the first `x_k < c`. If you ignore the window clamp, the number of lookback tests needed is:

$$
k^\* = \left\lceil \frac{b_0 - (c-1)}{s} \right\rceil
$$

#### FSLB-AD: Adaptive-decrease fixed-stride lookback (Optuna: `FSLB-AD_stride`, `FSLB-AD_alpha`)
Like FSLB, but shrink the stride geometrically after each failed test.

- Step `k` uses stride `s_k = ceil(s · α^{k-1})`, with `α ∈ [0,1]`.
- Candidate is `x_k = b_{k-1} - s_k` (clamped to `window_start`).

The total distance moved back after `K` failed steps (ignoring clamps/ceiling) is a geometric sum:

$$
\sum_{k=1}^{K} s\,\alpha^{k-1} = s\,\frac{1-\alpha^K}{1-\alpha}
$$

#### FSLB-AI: Adaptive-increase fixed-stride lookback (Optuna: `FSLB-AI_stride`, `FSLB-AI_alpha`)
Like FSLB, but grow the stride geometrically after each failed test.

- Step `k` uses stride `s_k = ceil(s · α^{k-1})`, with `α > 1`.
- Candidate is `x_k = b_{k-1} - s_k` (clamped to `window_start`).

The total distance moved back after `K` failed steps (ignoring clamps/ceiling) is:

$$
\sum_{k=1}^{K} s\,\alpha^{k-1} = s\,\frac{\alpha^K - 1}{\alpha - 1}
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

#### RATLB-AD: Adaptive-decrease risk-aware trigger lookback (Optuna: `RATLB-AD_threshold`, `RATLB-AD_alpha`)
Like RATLB, but change the trigger threshold after each failed probe:

$$
T_k = T \cdot \alpha^{k-1},\;\;\alpha \in [0,1]
$$

#### RATLB-AI: Adaptive-increase risk-aware trigger lookback (Optuna: `RATLB-AI_threshold`, `RATLB-AI_alpha`)
Same as RATLB-AD, but with `α > 1` so the threshold increases after each failed probe.

#### RWLBS: Risk-weighted lookback (sum mass) (Optuna: `RWLBS_threshold`)
Let `T ≥ 0` be a threshold. From the current known-bad boundary `b`, pick the **closest** earlier commit `x < b` such that the summed risk between `x` and `b` reaches the threshold:

$$
x = \arg\max_{j < b} \left\{ j \;:\; R_{\text{sum}}([j, b)) \ge T \right\}
$$

Test `x`; on failure set `b ← x` and repeat. If even the full in-window interval can’t reach `T`, fall back to testing `window_start`.

This makes lookback steps “smaller” when the region just before `b` already contains a lot of predicted risk.

#### RWLBS-AD: Adaptive-decrease risk-weighted lookback (sum mass) (Optuna: `RWLBS-AD_threshold`, `RWLBS-AD_alpha`)
Like RWLBS, but decay the risk-mass threshold after each failed probe:

$$
T_k = T \cdot \alpha^{k-1},\;\;\alpha \in [0,1]
$$

#### RWLBS-AI: Adaptive-increase risk-weighted lookback (sum mass) (Optuna: `RWLBS-AI_threshold`, `RWLBS-AI_alpha`)
Same as RWLBS-AD, but with `α > 1` so the threshold increases after each failed probe.

#### RWLBLS: Risk-weighted lookback (log-survival mass) (Optuna: `RWLBLS_threshold`)
Same policy as RWLBS, but threshold the noisy-OR mass instead of the sum:

$$
x = \arg\max_{j < b} \left\{ j \;:\; R_{\text{ls}}([j, b)) \ge T \right\}
$$

Here `T ∈ [0,1]`. This treats many small-risk commits differently from one large-risk commit.

#### RWLBLS-AD: Adaptive-decrease risk-weighted lookback (log-survival mass) (Optuna: `RWLBLS-AD_threshold`, `RWLBLS-AD_alpha`)
Like RWLBLS, but decay the threshold after each failed probe:

$$
T_k = T \cdot \alpha^{k-1},\;\;\alpha \in [0,1]
$$

#### RWLBLS-AI: Adaptive-increase risk-weighted lookback (log-survival mass) (Optuna: `RWLBLS-AI_threshold`, `RWLBLS-AI_alpha`)
Same as RWLBLS-AD, but with `α > 1` so the threshold increases after each failed probe.

#### TWLB: Time-window lookback (Optuna: `TWLB_hours`)
Let commit `i` have timestamp `t_i` (UTC). Choose a fixed time window `H` hours. From the current boundary `b`, jump back by time and test the nearest commit at-or-before the target time:

$$
\text{target\_time} = t_b - H
$$

Test the commit `x` with the largest `x < b` such that `t_x ≤ target_time`. On failure, set `b ← x` and repeat. If the target time predates all in-window history, fall back to testing `window_start`.

#### TWLB-AD: Adaptive-decrease time-window lookback (Optuna: `TWLB-AD_hours`, `TWLB-AD_alpha`)
Like TWLB, but shrink the window geometrically after each failure:

$$
H_k = H \cdot \alpha^{k-1}
$$

So the total time you jump back after `K` failed steps (ignoring discretization to commits) is:

$$
\sum_{k=1}^{K} H\,\alpha^{k-1} = H\,\frac{1-\alpha^K}{1-\alpha}
$$

#### TWLB-AI: Adaptive-increase time-window lookback (Optuna: `TWLB-AI_hours`, `TWLB-AI_alpha`)
Like TWLB, but grow the window geometrically after each failure:

$$
H_k = H \cdot \alpha^{k-1}
$$

So the total time you jump back after `K` failed steps (ignoring discretization to commits) is:

$$
\sum_{k=1}^{K} H\,\alpha^{k-1} = H\,\frac{\alpha^K - 1}{\alpha - 1}
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

### Optuna optimization details
When `--final-only` is **not** set, the script runs a full tuning pass on the **eval** risk predictions before the final replay.

For each selected `(lookback, bisection)` combo:

1) A single Optuna study is created with two objectives:
   - **Minimize** `max_tests_per_search`
   - **Minimize** `mean_tests_per_search`

2) The simulator is executed for each trial and the objective values are computed from the resulting metrics.

3) Trials are treated as **infeasible** (returning `(inf, inf)`) when:
   - No bugs were processed (`processed == 0`), or
   - Any processed bug failed to locate a culprit (`total_culprits_found < processed`)

4) Optuna uses `NSGAIISampler` when available (multi-objective evolutionary search). If that sampler is not available in your Optuna version, the script falls back to `RandomSampler`.

5) After optimization, the script selects a single point from the Pareto front by a simple rule:
   - Choose the Pareto-optimal trial with the smallest `max_tests_per_search`
   - Break ties by the smallest `mean_tests_per_search`

What gets written to `--output-eval`:
- `results[].best_params`: the selected best parameter values for the combo (stored without the Optuna prefix, e.g. `{"stride": 100}` rather than `{"FSLB_stride": 100}`).
- `results[].optuna`: metadata about the run (trial counts, chosen trial number, selected objective values, and the raw Optuna param dict).
- `results[].metrics`: the metrics from a re-run at the selected params (not from an arbitrary trial).

Parameter naming convention:
- Suggested parameters must be prefixed with the strategy `code` to avoid collisions across combos (e.g. `FSLB_stride`, `TWLB-FF_max_trials`, `TKRB_k`).
- `simulate.py` validates that every Optuna parameter name starts with either the selected lookback code or the selected bisection code.

Combos without tunable parameters:
- If both strategies in the combo have no `suggest_params`, Optuna is skipped for that combo and the default parameters are used.

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
python analysis/git_bisect/simulate.py --lookback NLB,FSLB,FSLB-AD --bisection GB,RWBS --mopt-trials 200
```

### Window-start penalty
Enable (or disable) the “window-start lookback” penalty and set its magnitude:

```bash
python analysis/git_bisect/simulate.py --penalize-window-start-lookback --window-start-lookback-penalty-tests 4
```

## Outputs and metrics

### Per-bug cost accounting
For a processed bug, the simulator counts:

- `lookback_tests`: probes performed while trying to find a known-good boundary `g`
- `bisection_tests`: probes performed after having bounds `(g,b]` to locate the culprit
- Optional `window_start` penalty: an extra fixed cost (default `4`) when `g = window_start` and `--penalize-window-start-lookback` is enabled

The per-bug cost is:

$$
\text{tests\_per\_bug} = \text{lookback\_tests} + \text{bisection\_tests}
$$

where `bisection_tests` may already include the optional window-start penalty.

### Per-combo metric definitions
Let `processed` be the number of bugs successfully simulated for a combo. For each processed bug `j`, let:

- `L_j`: lookback tests
- `B_j`: bisection tests (including the optional window-start penalty, if enabled)

Then:

$$
\text{total\_tests} = \sum_{j=1}^{processed} (L_j + B_j)
$$

$$
\text{total\_lookback\_tests} = \sum_{j=1}^{processed} L_j,\;\;\;\;
\text{total\_bisection\_tests} = \sum_{j=1}^{processed} B_j
$$

$$
\text{mean\_tests\_per\_search} = \frac{\text{total\_tests}}{processed}
$$

$$
\text{max\_tests\_per\_search} = \max_{j \in [1,processed]} (L_j + B_j)
$$

If `processed == 0`, `mean_tests_per_search` and `max_tests_per_search` are reported as `null` in the JSON output.

### Eval output (`--output-eval`)
Includes:
- Dataset metadata (commit window, risk file path, bug counts)
- Per-combo Optuna details (selected Pareto-optimal trial params, selected objective values)
- Per-combo metrics (including a `bugs` breakdown of processed/skipped)

Optuna tuning is multi-objective: it minimizes `(max_tests_per_search, mean_tests_per_search)`.

Eval output structure (high level):

```json
{
  "dataset": "eval",
  "dry_run": false,
  "best_combo_by_total_tests": "...",
  "best_combo_by_mean_tests_per_search": "...",
  "best_combo_by_max_tests_per_search": "...",
  "commit_window": { "...": "..." },
  "bugs": { "loaded": 0, "simulated": 0 },
  "risk_predictions": { "...": "..." },
  "optimization": { "...": "..." },
  "results": [
    {
      "combo": "FSLB+GB",
      "lookback": { "code": "FSLB", "name": "fixed_stride" },
      "bisection": { "code": "GB", "name": "git_bisect" },
      "best_params": { "lookback": { "...": "..." }, "bisection": { "...": "..." } },
      "metrics": { "...": "..." },
      "optuna": { "...": "..." }
    }
  ]
}
```

### Final output (`--output-final`)
Includes:
- Dataset metadata
- Best-combo keys: `best_combo_by_total_tests`, `best_combo_by_mean_tests_per_search`, `best_combo_by_max_tests_per_search`
- Per-combo metrics, plus:
  - `total_tests_saved_vs_baseline_pct`, `mean_tests_per_search_saved_vs_baseline_pct`, etc.
  - `params`: flattened “used params” like `{"Lookback_stride": 350, "Bisection_k": 20}`

Final output structure (high level):

```json
{
  "dataset": "final_test",
  "dry_run": false,
  "best_combo_by_total_tests": "...",
  "best_combo_by_mean_tests_per_search": "...",
  "best_combo_by_max_tests_per_search": "...",
  "commit_window": { "...": "..." },
  "bugs": { "loaded": 0, "simulated": 0 },
  "risk_predictions": { "...": "..." },
  "tuned_from_eval": { "path": "...", "present": true },
  "results": [
    {
      "combo": "FSLB+GB",
      "total_tests": 0,
      "total_lookback_tests": 0,
      "total_bisection_tests": 0,
      "mean_tests_per_search": 0.0,
      "max_tests_per_search": 0,
      "total_culprits_found": 0,
      "total_tests_saved_vs_baseline_pct": 0.0,
      "mean_tests_per_search_saved_vs_baseline_pct": 0.0,
      "max_tests_per_search_saved_vs_baseline_pct": 0.0,
      "params": { "Lookback_stride": 20 }
    }
  ]
}
```

Per-combo metrics fields:
- `total_tests`: lookback tests + bisection tests summed over all processed bugs
- `total_lookback_tests`, `total_bisection_tests`
- `mean_tests_per_search`, `max_tests_per_search`
- `total_culprits_found`

In eval output, per-combo metrics are nested under `row["metrics"]` (because the row also includes Optuna metadata). In final output, per-combo metrics are top-level fields in each `results[]` row.

### Baselines and comparability
The baseline for comparisons is `NLB+GB` (no lookback + standard git bisect).

The script also checks whether different combos processed different numbers of bugs (e.g., due to different lookback behaviors leading to different skip patterns). If the processed counts differ from the baseline, the script logs a warning because “tests saved vs baseline” may not be directly comparable.

### Skip reasons (eval output only)
The eval output includes a detailed `bugs` breakdown per combo:

- `bugs.processed`: number of regression bugs successfully simulated for that combo
- `bugs.skipped`: counters explaining why a bug was excluded

Skip keys and meanings:
- `not_regression`: bug row has `regression=false`
- `no_available_regressors`: no regressor bug inside the risk window could be mapped to a commit
- `bad_not_in_window`: the known-bad commit at/just before `bug_creation_time` falls after `window_end`
- `good_not_in_window`: the chosen known-good commit falls before `window_start`
- `culprit_after_bad`: inconsistent timestamps/labels where the regressor commit is after the known-bad commit (`culprit_index > bad_index`)
- `no_regressors_in_range`: lookback could not find an in-window passing commit strictly before the culprit (often meaning the regression predates the risk window)

Percent values are computed as: `100 * (baseline - value) / baseline` (positive is better / fewer tests). In eval output these percent fields are stored under `metrics` (e.g. `metrics.total_tests_saved_vs_baseline_pct`), while in final output they are stored at the top level of each result row (e.g. `total_tests_saved_vs_baseline_pct`).

## Adding a new strategy

- Implement the strategy in `analysis/git_bisect/lookback.py` or `analysis/git_bisect/bisection.py`.
- Add a `StrategySpec` entry in `analysis/git_bisect/simulate.py`:
  - Choose a short `code` (used in combo keys).
  - Provide `default_params` and `build`.
  - If tunable, provide `suggest_params` and ensure Optuna parameter names are prefixed with `<CODE>_` (e.g. `TWLB_hours`).
