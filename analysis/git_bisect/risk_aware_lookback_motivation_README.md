# Risk-Aware Lookback Motivation Diagram

This document explains [risk_aware_lookback_motivation.drawio](./risk_aware_lookback_motivation.drawio).

The current version of the figure focuses only on the lookback algorithms themselves. It intentionally removes:

- the old shared commit timeline at the top,
- the `NoLookback` row,
- and the single-line summaries for each method.

Instead, every step of every shown algorithm now has its own local sequence of commits.

## What Changed In The Layout

The page is now organized into three algorithm sections:

1. `FSLB`
2. `RWLB-S`
3. `RATLB`

Each section has three panels:

1. the first lookback decision,
2. the second lookback decision,
3. the handoff interval given to bisection.

Each panel contains its own local commit sequence. That means the reader no longer has to mentally map a small arrow or label back to a single shared timeline somewhere else on the page.

## Shared Synthetic Scenario

All panels use the same synthetic regression setup:

- true culprit: `C17`
- observed bad commit / start index: `C23`
- a test fails iff the tested commit index is `>= 17`

That matches the monotone failure assumption used in the simulator.

Most omitted commits have low risk `0.02`, but the overall history still contains high-risk false positives outside the drawn windows:

- `C6 = 0.89`
- `C9 = 0.94`
- `C12 = 0.91`

Those omitted commits are mentioned in the context box so the figure still communicates that high-confidence risk is spread out rather than perfectly concentrated next to the culprit.

## Legend

- `S` = current bad boundary
- `P` = tested commit passed
- `F` = tested commit failed
- `T` = trigger commit used by `RATLB`

## FSLB Section

Configuration:

- stride `= 4`

### Panel 1

The panel shows a local sequence around the current bad boundary:

- `C19, C20, C21, C22, C23`

The logic shown in the panel is purely positional:

- current bad boundary is `C23`
- subtract the stride: `23 - 4 = 19`
- test `C19`
- outcome: `FAIL`

The panel explicitly says that risk scores are ignored.

### Panel 2

The second panel redraws a new local sequence:

- `C15, C16, C17, C18, C19`

Again the logic is purely positional:

- current bad boundary is now `C19`
- subtract the stride again: `19 - 4 = 15`
- test `C15`
- outcome: `PASS`

At this point `FSLB` has:

- `C15 = PASS`
- `C19 = FAIL`

### Panel 3

The third panel redraws the local bracket:

- `C15, C16, C17, C18, C19`

It highlights the interval:

- `(C15,C19]`

The panel also states the realized bisection path:

- `C17 -> FAIL`
- `C16 -> PASS`

Total shown cost:

- `2` lookback probes
- `2` bisection probes
- total `4`

## RWLB-S Section

Configuration:

- threshold `T = 1.80`

This section now does two things at the same time:

- it redraws the commit sequence for each step,
- and it shows the cumulative-risk arithmetic that decides which commit to test.

### Panel 1

The panel shows the local window:

- `C19 = 0.82`
- `C20 = 0.93`
- `C21 = 0.50`
- `C22 = 0.02`
- `C23 = 0.02`

The text block computes the threshold interaction explicitly:

- `C22: 0.02`
- `C21..22: 0.52`
- `C20..22: 1.45`
- `C19..22: 2.27 >= 1.80`

So the first candidate whose cumulative risk reaches `T` is `C19`, and:

- test `C19`
- outcome: `FAIL`

The panel also shades the commits contributing to the chosen cumulative-risk window.

### Panel 2

The panel redraws a new local window:

- `C16 = 0.86`
- `C17 = 0.95`
- `C18 = 0.84`
- `C19 = 0.82`

It again shows the arithmetic explicitly:

- `C18: 0.84`
- `C17..18: 1.79`
- `C16..18: 2.65 >= 1.80`

So the next candidate is `C16`, and:

- test `C16`
- outcome: `PASS`

At this point `RWLB-S` has:

- `C16 = PASS`
- `C19 = FAIL`

### Panel 3

The third panel redraws the resulting bracket:

- `C16, C17, C18, C19`

It highlights:

- `(C16,C19]`

The panel also makes the realized-vs-worst-case distinction explicit:

- realized bisect path here: midpoint `C17 -> FAIL`
- realized bisection cost here: `1`
- worst-case bisection cost for that same 3-commit interval: `2`

Total shown cost:

- `2` lookback probes
- `1` realized bisection probe
- total `3`

## RATLB Section

Configuration:

- trigger threshold `T = 0.90`

This section also redraws the sequence at every step, but instead of cumulative sums it shows trigger logic:

- scan left until some commit has `risk > T`
- then test the commit immediately before that trigger

### Panel 1

The local window is:

- `C19 = 0.82`
- `C20 = 0.93`
- `C21 = 0.50`
- `C22 = 0.02`
- `C23 = 0.02`

The text shows the threshold comparisons explicitly:

- `C22 = 0.02 < 0.90`
- `C21 = 0.50 < 0.90`
- `C20 = 0.93 > 0.90`

So `C20` becomes the trigger, and the tested commit is the one immediately before it:

- test `C19`
- outcome: `FAIL`

The panel marks `C20` with a trigger symbol `T`.

### Panel 2

The new local window is:

- `C16 = 0.86`
- `C17 = 0.95`
- `C18 = 0.84`
- `C19 = 0.82`

The threshold checks are:

- `C18 = 0.84 < 0.90`
- `C17 = 0.95 > 0.90`

So `C17` becomes the next trigger, and the tested commit is:

- `C16`

with outcome:

- `PASS`

This is why the diagram now looks less contrived than the earlier version: `RATLB` does not succeed by immediately reacting to a perfect trigger sitting right after the culprit on the first move. It first reacts to the false-positive trigger at `C20`.

### Panel 3

The third panel redraws the same resulting bracket:

- `C16, C17, C18, C19`

The interval is:

- `(C16,C19]`

So `RATLB` ends with the same bracket as `RWLB-S`, but the panel text emphasizes that it got there through two trigger decisions instead of cumulative sums.

Total shown cost:

- `2` lookback probes
- `1` realized bisection probe
- total `3`

## Main Point Of The New Design

The new layout makes three things much clearer than the old version:

1. every algorithm step has its own visible local history,
2. the risk-aware variants now show the exact threshold logic inside the panel instead of only in the caption,
3. `RWLB-S` and `RATLB` can end at the same bracket even though one uses cumulative risk mass and the other uses trigger checks.

## Totals Shown In The Figure

| Algorithm | Lookback probes | Final bracket | Realized bisection probes | Total |
|---|---:|---|---:|---:|
| `FSLB` | 2 | `(C15,C19]` | 2 | 4 |
| `RWLB-S` | 2 | `(C16,C19]` | 1 | 3 |
| `RATLB` | 2 | `(C16,C19]` | 1 | 3 |

## Notes On Interpretation

- The figure now focuses on the mechanics of lookback itself, not on a full baseline comparison against `NoLookback`.
- The `1` bisection step shown for `RWLB-S` and `RATLB` is a realized cost for this specific culprit `C17`, not a worst-case bound for every culprit inside `(C16,C19]`.
- Earlier high-risk false positives are still part of the scenario even though they are not drawn in the per-step local windows.
