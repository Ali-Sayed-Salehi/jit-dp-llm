#!/usr/bin/env python3
import math
import random
import bisect
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd

# =========================
# Defaults / knobs
# =========================
RANDOM_SEED            = 42
YEAR_DAYS              = 365
PUSH_RATE_PER_HOUR     = 8.0                    # avg pushes/hour (Poisson)
DEFECT_RATE            = 0.01                   # 1% true culprit pushes

TEST_COST              = 1.0                    # cost per scheduled/triggered test
BACKFILL_COST          = 1.0                    # cost per backfill "step"
TEST_EXEC_HOURS        = 0.5                    # runtime of a single perf test run (hours)

# Predictor driven by AUC (Option A); we will TUNE τ per AUC
TARGET_ROC_AUC         = 0.8                   # simulated predictor quality

# Scheduled test frequency for the proposed policy (real time, not windowed)
TEST_INTERVAL_HOURS    = 4                    # can be < 4 now, e.g., 0.5 or 0.25
BASELINE_TEST_INTERVAL = 4
# Threshold sweep to auto-pick best τ (per AUC & frequency)
THRESHOLD_GRID         = np.linspace(0.5, 0.99, 25)

# Backfill cost model overhead per located culprit
BISECT_VERIFY_OVERHEAD = 2
MIN_BISECT_COST        = 0

SIM_START              = pd.Timestamp("2024-01-01 00:00:00Z")
SIM_END                = SIM_START + pd.Timedelta(hours=YEAR_DAYS * 24)

# Risk-bisection smoothing (avoid zero-prob partitions)
RISK_FLOOR_EPS         = 1e-9
# =========================

rng = np.random.default_rng(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ---------- Utility: inverse Normal CDF (Acklam approximation) ----------

def inv_norm_cdf(p: float) -> float:
    if not (0.0 < p < 1.0):
        if p == 0.0: return -np.inf
        if p == 1.0: return np.inf
        raise ValueError("p must be in (0,1)")
    a = [ -3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
           1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00 ]
    b = [ -5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
           6.680131188771972e+01, -1.328068155288572e+01 ]
    c = [ -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
          -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00 ]
    d = [  7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,  3.754408661907416e+00 ]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)

def dprime_from_auc(target_auc: float) -> float:
    eps = 1e-9
    a = min(max(target_auc, eps), 1 - eps)
    return math.sqrt(2.0) * inv_norm_cdf(a)

# ---------- Data classes ----------

@dataclass
class Push:
    global_idx: int
    ts: pd.Timestamp
    is_defect: bool
    files_changed: int
    lines_changed: int
    hour: int
    score: float

# ---------- AUC-driven score generator ----------

def make_scores_from_auc(y_true: np.ndarray, target_auc: float, rng) -> np.ndarray:
    dprime = dprime_from_auc(target_auc)
    z = dprime * y_true + rng.normal(0.0, 1.0, size=y_true.size)
    return 1.0 / (1.0 + np.exp(-z))

def auc_from_scores(y_true, scores) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    n1 = int(y.sum()); n0 = int(y.size - n1)
    if n1 == 0 or n0 == 0:
        return 0.5
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s)+1)
    rank_sum_pos = ranks[y == 1].sum()
    U = rank_sum_pos - n1*(n1+1)/2.0
    return float(U / (n0 * n1))

# ---------- Synthetic data generation (continuous time) ----------

def generate_pushes_for_year() -> Tuple[List[Push], pd.Timestamp]:
    """
    Generate a year's worth of pushes as a Poisson process per hour.
    Return (pushes_sorted_by_time, sim_end_time).
    """
    total_hours = YEAR_DAYS * 24
    pushes: List[Push] = []
    labels, times, files_list, lines_list, hours_list = [], [], [], [], []

    gi = 0
    for h in range(total_hours):
        base = SIM_START + pd.Timedelta(hours=h)
        n_pushes = rng.poisson(PUSH_RATE_PER_HOUR)
        if n_pushes > 0:
            # Uniform offsets within the hour (in seconds)
            offsets = rng.uniform(0, 3600, size=n_pushes)
            offsets.sort()
            for o in offsets:
                ts = base + pd.Timedelta(seconds=float(o))
                hour = int(ts.tz_convert("UTC").hour) if ts.tzinfo else int(ts.hour)
                files_changed = int(max(1, rng.poisson(5)))
                lines_changed = int(max(1, rng.lognormal(mean=4.0, sigma=0.6)))
                is_defect = bool(rng.random() < DEFECT_RATE)

                times.append(ts)
                labels.append(1 if is_defect else 0)
                files_list.append(files_changed)
                lines_list.append(lines_changed)
                hours_list.append(hour)

    y_arr = np.array(labels, dtype=int)
    scores = make_scores_from_auc(y_arr, TARGET_ROC_AUC, rng) if len(y_arr) > 0 else np.array([], dtype=float)

    pushes = []
    for i in range(len(y_arr)):
        pushes.append(Push(
            global_idx=i,
            ts=times[i],
            is_defect=bool(y_arr[i]),
            files_changed=files_list[i],
            lines_changed=lines_list[i],
            hour=hours_list[i],
            score=float(scores[i]),
        ))

    # Ensure chronological order (should already be sorted by construction)
    pushes.sort(key=lambda p: p.ts)
    # Reassign global_idx to match time order
    for i, p in enumerate(pushes):
        p.global_idx = i

    return pushes, SIM_END

# ---------- Helpers ----------

def iter_defect_indices_in_span(pushes: List[Push], a_last_test_idx: int, b_current_idx: int):
    if b_current_idx is None or b_current_idx <= a_last_test_idx:
        return
    for gi in range(a_last_test_idx + 1, b_current_idx + 1):
        if pushes[gi].is_defect:
            yield gi

def last_push_idx_at_or_before(push_ts: List[pd.Timestamp], t: pd.Timestamp) -> int:
    """Return the index of the last push with timestamp <= t, or -1 if none."""
    pos = bisect.bisect_right(push_ts, t) - 1
    return pos if pos >= 0 else -1

def make_schedule_times(every_hours: float, start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    if every_hours <= 0:
        return []
    times = []
    t = start + pd.Timedelta(hours=every_hours)
    while t <= end:
        times.append(t)
        t += pd.Timedelta(hours=every_hours)
    return times

# ---------- Multi-culprit bisection helpers (unchanged) ----------

def uniform_bisect_steps_for_culprit(span_len: int, culprit_pos: int) -> int:
    if span_len <= 1:
        return 0
    l, r = 0, span_len - 1
    steps = 0
    while l < r:
        mid = (l + r) // 2
        steps += 1
        if culprit_pos <= mid:
            r = mid
        else:
            l = mid + 1
    return steps

def multi_culprit_uniform_bisect_steps(
    pushes: List['Push'],
    start_idx: int,
    end_idx: int,
    defect_indices: List[int],
) -> Tuple[int, Dict[int, int]]:
    L = start_idx + 1
    R = end_idx
    if R <= L:
        return 0, {}
    defects = sorted([d for d in defect_indices if L <= d <= R])
    if not defects:
        return 0, {}

    total_steps = 0
    per_defect_steps: Dict[int, int] = {}

    while defects:
        culprit = max(defects)  # latest-first
        span_len = R - L + 1
        culprit_pos = culprit - L

        probes = uniform_bisect_steps_for_culprit(span_len, culprit_pos)
        steps_this = max(MIN_BISECT_COST, probes + BISECT_VERIFY_OVERHEAD)

        total_steps += steps_this
        per_defect_steps[culprit] = total_steps

        R = culprit - 1
        defects = [d for d in defects if d <= R]

    return total_steps, per_defect_steps

def risk_bisect_steps_for_culprit(scores: np.ndarray, culprit_pos: int) -> int:
    n = int(scores.size)
    if n <= 1:
        return 0
    l, r = 0, n - 1
    steps = 0
    eps = RISK_FLOOR_EPS

    while l < r:
        window = scores[l:r+1].astype(float)
        window = window + eps
        window /= window.sum()

        c = np.cumsum(window)
        k_rel = int(np.searchsorted(c, 0.5, side="left"))
        if k_rel >= (r - l):
            k_rel = r - l - 1
        k = l + k_rel

        steps += 1
        if culprit_pos <= k:
            r = k
        else:
            l = k + 1
    return steps

def multi_culprit_risk_bisect_steps(
    pushes: List['Push'],
    start_idx: int,
    end_idx: int,
    defect_indices: List[int]
) -> Tuple[int, Dict[int, int]]:
    L = start_idx + 1
    R = end_idx
    if R <= L:
        return 0, {}

    defects = sorted([d for d in defect_indices if L <= d <= R])
    if not defects:
        return 0, {}

    total_steps = 0
    per_defect_steps: Dict[int, int] = {}

    while defects:
        culprit = max(defects)  # latest-first
        scores = np.array([pushes[i].score for i in range(L, R + 1)], dtype=float)
        culprit_pos = culprit - L

        probes = risk_bisect_steps_for_culprit(scores, culprit_pos)
        steps_this_culprit = max(MIN_BISECT_COST, probes + BISECT_VERIFY_OVERHEAD)

        total_steps += steps_this_culprit
        per_defect_steps[culprit] = total_steps

        R = culprit - 1
        defects = [d for d in defects if d <= R]

    return total_steps, per_defect_steps

# ---------- Baseline (fixed-interval only) with multi-culprit uniform bisection ----------

def simulate_baseline(pushes: List[Push], interval_hours: float) -> Dict[str, float]:
    scheduled_tests = 0
    backfill_tests = 0
    detections = 0

    detection_latencies_hours: List[float] = []
    exact_latencies_hours: List[float] = []

    last_test_idx = -1
    push_ts = [p.ts for p in pushes]
    schedule_times = make_schedule_times(interval_hours, SIM_START, SIM_END)

    for t in schedule_times:
        scheduled_tests += 1
        b_idx = last_push_idx_at_or_before(push_ts, t)
        if b_idx > last_test_idx:
            detection_time = t
            defect_indices = list(iter_defect_indices_in_span(pushes, last_test_idx, b_idx))
            if defect_indices:
                span_steps, per_defect_steps = multi_culprit_uniform_bisect_steps(
                    pushes, last_test_idx, b_idx, defect_indices
                )
                for defect_idx in defect_indices:
                    dt_hours = (detection_time - pushes[defect_idx].ts).total_seconds() / 3600.0
                    if dt_hours >= 0:
                        detection_latencies_hours.append(dt_hours)
                        steps_to_this = per_defect_steps.get(defect_idx, span_steps)
                        exact_latencies_hours.append(dt_hours + steps_to_this * TEST_EXEC_HOURS)
                detections += 1
                backfill_tests += int(span_steps)
            last_test_idx = b_idx

    total_cost = scheduled_tests * TEST_COST + backfill_tests * BACKFILL_COST
    avg_det = float(np.mean(detection_latencies_hours)) if detection_latencies_hours else 0.0
    max_det = float(np.max(detection_latencies_hours)) if detection_latencies_hours else 0.0
    avg_exact = float(np.mean(exact_latencies_hours)) if exact_latencies_hours else 0.0
    max_exact = float(np.max(exact_latencies_hours)) if exact_latencies_hours else 0.0

    return dict(
        policy=f"baseline_{interval_hours}h",
        scheduled_tests=scheduled_tests,
        predictor_triggered_tests=0,
        backfill_tests=backfill_tests,
        total_tests=scheduled_tests + backfill_tests,
        total_cost=total_cost,
        detections=detections,
        delayed_detections=0,
        avg_detect_hours=avg_det,
        max_detect_hours=max_det,
        avg_time_to_exact_hours=avg_exact,
        max_time_to_exact_hours=max_exact,
    )

# ---------- Proposed (scheduled + commit-triggered) with multi-culprit risk bisection ----------

def simulate_proposed_streaming(pushes: List[Push], threshold: float, test_interval_hours: float) -> Dict[str, float]:
    scheduled_tests = 0
    predictor_tests = 0
    backfill_tests = 0
    detections = 0

    detection_latencies_hours: List[float] = []
    exact_latencies_hours: List[float] = []

    last_test_idx = -1  # global push index of last test boundary

    push_ts = [p.ts for p in pushes]
    schedule_times = make_schedule_times(test_interval_hours, SIM_START, SIM_END)
    next_sched_i = 0

    # Iterate pushes in time; fire due scheduled tests before each push; then handle predictor trigger
    for gi, push in enumerate(pushes):
        # Flush any scheduled tests that occur before or at this push
        while next_sched_i < len(schedule_times) and schedule_times[next_sched_i] <= push.ts:
            t = schedule_times[next_sched_i]
            next_sched_i += 1
            scheduled_tests += 1
            b_idx = last_push_idx_at_or_before(push_ts, t)
            if b_idx > last_test_idx:
                detection_time = t
                defect_indices = list(iter_defect_indices_in_span(pushes, last_test_idx, b_idx))
                if defect_indices:
                    span_steps, per_defect_steps = multi_culprit_risk_bisect_steps(
                        pushes, last_test_idx, b_idx, defect_indices
                    )
                    for defect_idx in defect_indices:
                        dt_hours = (detection_time - pushes[defect_idx].ts).total_seconds() / 3600.0
                        if dt_hours >= 0:
                            detection_latencies_hours.append(dt_hours)
                            steps_to_this = per_defect_steps.get(defect_idx, span_steps)
                            exact_latencies_hours.append(dt_hours + steps_to_this * TEST_EXEC_HOURS)
                    detections += 1
                    backfill_tests += int(span_steps)
                last_test_idx = b_idx

        # Predictor-triggered on this push
        if push.score >= threshold:
            predictor_tests += 1
            detection_time = push.ts
            defect_indices = list(iter_defect_indices_in_span(pushes, last_test_idx, gi))
            if defect_indices:
                span_steps, per_defect_steps = multi_culprit_risk_bisect_steps(
                    pushes, last_test_idx, gi, defect_indices
                )
                for defect_idx in defect_indices:
                    dt_hours = (detection_time - pushes[defect_idx].ts).total_seconds() / 3600.0
                    if dt_hours >= 0:
                        detection_latencies_hours.append(dt_hours)
                        steps_to_this = per_defect_steps.get(defect_idx, span_steps)
                        exact_latencies_hours.append(dt_hours + steps_to_this * TEST_EXEC_HOURS)
                detections += 1
                backfill_tests += int(span_steps)
            last_test_idx = gi

    # Flush remaining scheduled tests after last push until SIM_END
    while next_sched_i < len(schedule_times):
        t = schedule_times[next_sched_i]
        next_sched_i += 1
        scheduled_tests += 1
        b_idx = last_push_idx_at_or_before(push_ts, t)
        if b_idx > last_test_idx:
            detection_time = t
            defect_indices = list(iter_defect_indices_in_span(pushes, last_test_idx, b_idx))
            if defect_indices:
                span_steps, per_defect_steps = multi_culprit_risk_bisect_steps(
                    pushes, last_test_idx, b_idx, defect_indices
                )
                for defect_idx in defect_indices:
                    dt_hours = (detection_time - pushes[defect_idx].ts).total_seconds() / 3600.0
                    if dt_hours >= 0:
                        detection_latencies_hours.append(dt_hours)
                        steps_to_this = per_defect_steps.get(defect_idx, span_steps)
                        exact_latencies_hours.append(dt_hours + steps_to_this * TEST_EXEC_HOURS)
                detections += 1
                backfill_tests += int(span_steps)
            last_test_idx = b_idx

    total_scheduled = scheduled_tests + predictor_tests
    total_cost = total_scheduled * TEST_COST + backfill_tests * BACKFILL_COST
    avg_det = float(np.mean(detection_latencies_hours)) if detection_latencies_hours else 0.0
    max_det = float(np.max(detection_latencies_hours)) if detection_latencies_hours else 0.0
    avg_exact = float(np.mean(exact_latencies_hours)) if exact_latencies_hours else 0.0
    max_exact = float(np.max(exact_latencies_hours)) if exact_latencies_hours else 0.0

    return dict(
        policy=f"proposed_stream_{test_interval_hours}h_tau_{threshold}",
        scheduled_tests=total_scheduled,
        fixed_schedule_tests=scheduled_tests,
        predictor_triggered_tests=predictor_tests,
        backfill_tests=backfill_tests,
        total_tests=total_scheduled + backfill_tests,
        total_cost=total_cost,
        detections=detections,
        delayed_detections=0,
        avg_detect_hours=avg_det,
        max_detect_hours=max_det,
        avg_time_to_exact_hours=avg_exact,
        max_time_to_exact_hours=max_exact,
    )

# ---------- Auto-tune τ for given AUC & test interval ----------

def pick_best_threshold(pushes: List[Push], test_interval_hours: float, grid: np.ndarray) -> Dict[str, float]:
    best = None
    for tau in grid:
        res = simulate_proposed_streaming(pushes, threshold=float(tau), test_interval_hours=test_interval_hours)
        if (best is None) or (res["total_tests"] < best["total_tests"]):
            best = {"tau": float(tau), **res}
    return best

# ---------- Main ----------

def main():
    pushes, sim_end = generate_pushes_for_year()

    total_pushes = len(pushes)
    total_defects = sum(1 for p in pushes if p.is_defect)

    # Realized AUC
    if total_pushes > 0:
        y_all = np.array([1 if p.is_defect else 0 for p in pushes], dtype=int)
        s_all = np.array([p.score for p in pushes], dtype=float)
        realized_auc = auc_from_scores(y_all, s_all)
    else:
        realized_auc = 0.5

    print("=== Data summary ===")
    print(f"Simulation: {SIM_START} → {sim_end}  (~{YEAR_DAYS} days)")
    print(f"Total pushes: {total_pushes} | True defects: {total_defects} "
          f"({100.0*total_defects/max(1,total_pushes):.2f}%)")
    print(f"Target ROC AUC: {TARGET_ROC_AUC:.3f} | Realized ROC AUC (synthetic): {realized_auc:.3f}")
    print(f"Scheduled interval: every {TEST_INTERVAL_HOURS}h (no windowing)")
    print(f"Test runtime: {TEST_EXEC_HOURS} hours per run\n")

    # Baseline (e.g., 4h) - multi-culprit uniform bisection, fixed cadence only
    base_interval = BASELINE_TEST_INTERVAL
    base = simulate_baseline(pushes, interval_hours=base_interval)

    # Tune threshold for proposed streaming policy (multi-culprit risk-aware backfill)
    best = pick_best_threshold(pushes, TEST_INTERVAL_HOURS, THRESHOLD_GRID)

    print("=== Results ===")
    def line(d: Dict[str, float]) -> str:
        return (f"{d['policy']:>36} | scheduled={d['scheduled_tests']:>6} "
                f"| (fixed={d.get('fixed_schedule_tests',0):>4}, trig={d.get('predictor_triggered_tests',0):>4}) "
                f"| backfill={d['backfill_tests']:>6} | total_tests={d['total_tests']:>7} "
                f"| cost={d['total_cost']:>9.1f} | detections={d['detections']:>6} "
                f"| avg_detect_h={d.get('avg_detect_hours',0):>5.2f} | max_detect_h={d.get('max_detect_hours',0):>5.2f} "
                f"| avg_exact_h={d.get('avg_time_to_exact_hours',0):>5.2f} | max_exact_h={d.get('max_time_to_exact_hours',0):>5.2f}")

    print(line(base))
    print(f"{'AUTO τ pick':>36} | τ*={best['tau']:.3f}")
    print(line(best))

    saved_tests = base["total_tests"] - best["total_tests"]
    saved_cost  = base["total_cost"]  - best["total_cost"]
    print("\n=== Savings vs baseline (with τ* and interval={}h) ===".format(TEST_INTERVAL_HOURS))
    print(f"Saved tests: {saved_tests:.1f} (positive = fewer tests than baseline)")
    print(f"Saved cost : {saved_cost:.1f}")
    print(f"Avg detect hours: baseline={base['avg_detect_hours']:.2f} vs proposed={best['avg_detect_hours']:.2f}")
    print(f"Avg time to exact: baseline={base['avg_time_to_exact_hours']:.2f} vs proposed={best['avg_time_to_exact_hours']:.2f}")

    # Sweep table for context
    sweep = []
    for tau in THRESHOLD_GRID:
        res = simulate_proposed_streaming(pushes, threshold=float(tau), test_interval_hours=TEST_INTERVAL_HOURS)
        sweep.append({"threshold": float(tau), **res})
    df = pd.DataFrame(sweep).sort_values("threshold")
    print("\n=== Threshold sweep (proposed streaming) ===")
    print(df[["threshold","scheduled_tests","fixed_schedule_tests","predictor_triggered_tests",
              "backfill_tests","total_tests","detections",
              "avg_detect_hours","max_detect_hours","avg_time_to_exact_hours","max_time_to_exact_hours","total_cost"]].to_string(index=False))

if __name__ == "__main__":
    main()
