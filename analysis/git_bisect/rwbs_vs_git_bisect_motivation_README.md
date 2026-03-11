# RWBS Vs Git Bisect Motivation Diagram

This document explains [rwbs_vs_git_bisect_motivation.drawio](./rwbs_vs_git_bisect_motivation.drawio).

This version uses a compact local history with `10` displayed commits:

- `C0` is drawn explicitly as the known-good boundary
- the current known-bad boundary is `C9`
- the true culprit is `C8`

That makes the interval structure visible directly in the figure instead of leaving the good boundary implicit.

## Shared Synthetic Scenario

All three rows use the same local setup:

- known-good boundary: `C0`
- known-bad boundary: `C9`
- search interval: `(C0,C9]`
- true culprit: `C8`

The simulator's monotone test model is:

- `test(i)` fails iff `i >= 8`

So `C8` is the first failing commit.

## Risk Profile

`C0` is only the explicit good boundary, so it is not part of the risk mass. The candidate risks are:

- `C1 = 0.02`
- `C2 = 0.10`
- `C3 = 0.02`
- `C4 = 0.25`
- `C5 = 0.02`
- `C6 = 0.15`
- `C7 = 0.02`
- `C8 = 0.95`
- `C9 = 0.02`

The elevated-risk commits are intentionally scattered:

- `C2`, `C4`, `C6`, and `C8` are spread out
- they are not one contiguous suspicious block
- only `C8` is the true culprit

## Git Bisect Row

The first row shows standard midpoint bisection.

### Step 1

Current interval:

- `(C0,C9]`

Midpoint by commit count:

- `floor((0 + 9) / 2) = 4`

Probe:

- `C4 -> PASS`

### Step 2

New interval:

- `(C4,C9]`

Next midpoint:

- `floor((4 + 9) / 2) = 6`

Probe:

- `C6 -> PASS`

### Step 3

Git bisect still needs:

1. `C7 -> PASS`
2. `C8 -> FAIL`

Total:

- `4` tests

Validated local trace:

- `C4(P), C6(P), C7(P), C8(F)`

## RWBS Row

The second row shows `RiskWeightedBisectionSum` from [bisection.py](./bisection.py).

`RWBS` chooses probes by balancing cumulative risk mass over the current interval.

### Step 1

Current interval:

- `(C0,C9]`

Total risk mass:

- `1.55`

Half of total risk:

- `0.775`

Cumulative risk:

- through `C7`: `0.58`
- through `C8`: `1.53`

So the half-mass crossing occurs at:

- `C8`

Probe:

- `C8 -> FAIL`

### Step 2

New interval:

- `(C0,C8]`

Total remaining risk mass:

- `1.53`

Half:

- `0.765`

The cumulative crossing is still at `C8`, but `C8` is now the bad boundary, so it is not a legal interior probe. The implementation clamps to the nearest legal interior point:

- `C7 -> PASS`

### Step 3

Now RWBS has:

- `C7(P)`
- `C8(F)`

So the culprit is isolated immediately.

Total:

- `2` tests

Validated local trace:

- `C8(F), C7(P)`

## TKRB Row

The third row adds `TopKRiskFirstBisection` with:

- `k = 2`

`TKRB` ranks commits by individual risk and tries the highest-risk candidates directly.

### Step 1

The top-2 candidates in `(C0,C9]` are:

1. `C8 = 0.95`
2. `C4 = 0.25`

### Step 2

`TKRB` tries the highest-risk candidate first:

- candidate: `C8`
- predecessor: `C7`

Observed outcomes:

- `C7 -> PASS`
- `C8 -> FAIL`

That pair is enough to identify the culprit immediately in the risk-first phase.

### Step 3

Because the first candidate already succeeds:

- the second-ranked candidate `C4` is never used
- fallback binary search never runs

Total:

- `2` tests

Validated local trace:

- `C7(P), C8(F)`

## Summary Table

| Strategy | Core idea | Validated local trace | Total tests |
|---|---|---|---:|
| `GitBisectBaseline` | midpoint by commit count | `C4(P), C6(P), C7(P), C8(F)` | 4 |
| `RiskWeightedBisectionSum` (`RWBS`) | split by summed risk mass | `C8(F), C7(P)` | 2 |
| `TopKRiskFirstBisection` (`TKRB`, `k=2`) | test highest-risk commits directly | `C7(P), C8(F)` | 2 |

## Main Takeaway

The diagram now makes the boundary semantics explicit:

- `C0` is the known-good starting point
- `C9` is the current bad boundary
- the actual search interval is `(C0,C9]`

Within that interval, standard midpoint bisection still burns several passing probes, while `RWBS` and `TKRB` exploit the scattered risk signal to reach the culprit in two tests.
