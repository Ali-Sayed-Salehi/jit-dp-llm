"""Optional worker observations for the paper's capacity experiment.

The observer records job timing without changing simulation decisions.
"""

from array import array
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np


_RECORDER = ContextVar("worker_recorder", default=None)


def active_job_recorder():
    return _RECORDER.get()


@contextmanager
def observe_workers(recorder):
    token = _RECORDER.set(recorder)
    try:
        yield recorder
    finally:
        _RECORDER.reset(token)


def _peak_queue(arrivals, starts, end=None):
    # At ties, starts remove jobs immediately. Jobs starting when they become
    # ready have no queue interval and are not included in these arrays.
    if not len(arrivals):
        return 0
    if end is not None:
        arrivals = arrivals[:np.searchsorted(arrivals, end, side="right")]
    if not len(arrivals):
        return 0
    counts = np.arange(1, len(arrivals) + 1, dtype=np.int64)
    counts -= np.searchsorted(starts, arrivals, side="right")
    return int(counts.max())


class WorkerDiagnostics:
    """Compact queue intervals, plus streaming job and utilization aggregates.

    Only queued jobs need two timestamps in memory, including for exhaustive
    testing. Busy time is clipped to the shared first-to-last input window;
    unused workers and workers in inactive pools remain in the denominator.
    """

    def __init__(self, window_start, window_end):
        self.window_start = window_start
        self.window_end = window_end
        self.window_seconds = (window_end - window_start).total_seconds()
        if self.window_seconds <= 0:
            raise ValueError("Worker observations require a positive input window")
        self.daily_edges = np.append(
            np.arange(0.0, self.window_seconds, 86400.0), self.window_seconds
        )
        self.pools = {}

    def attach(self, pool_sizes):
        if self.pools:
            raise ValueError("Use a fresh worker recorder for each executor")
        for name, workers in pool_sizes.items():
            self.pools[name] = {
                "workers": workers, "jobs": 0, "queued_jobs": 0,
                "wait_seconds": 0.0, "max_wait_seconds": 0.0,
                "busy_seconds": 0.0, "last_finish_seconds": 0.0,
                "backlog_at_input_end": 0, "running_at_input_end": 0,
                "not_ready_at_input_end": 0,
                "arrivals": array("d"), "starts": array("d"),
                "daily_busy": np.zeros(len(self.daily_edges) - 1),
            }

    def record_job(self, pool, ready_time, start_time, finish_time):
        p = self.pools[pool]
        ready = (ready_time - self.window_start).total_seconds()
        start = (start_time - self.window_start).total_seconds()
        finish = (finish_time - self.window_start).total_seconds()
        wait = start - ready
        if wait < 0 or finish < start:
            raise ValueError("Invalid job timing")
        p["jobs"] += 1
        p["wait_seconds"] += wait
        p["max_wait_seconds"] = max(p["max_wait_seconds"], wait)
        p["last_finish_seconds"] = max(p["last_finish_seconds"], finish)
        if wait > 0:
            p["queued_jobs"] += 1
            p["arrivals"].append(ready)
            p["starts"].append(start)
        end = self.window_seconds
        p["backlog_at_input_end"] += int(ready <= end < start)
        p["running_at_input_end"] += int(start <= end < finish)
        p["not_ready_at_input_end"] += int(ready > end)
        lo, hi = max(start, 0.0), min(finish, end)
        if hi > lo:
            p["busy_seconds"] += hi - lo
            first_bin = min(int(lo // 86400), len(p["daily_busy"]) - 1)
            last_bin = min(int(hi // 86400), len(p["daily_busy"]) - 1)
            for index in range(first_bin, last_bin + 1):
                p["daily_busy"][index] += max(
                    0.0, min(hi, self.daily_edges[index + 1])
                    - max(lo, self.daily_edges[index])
                )

    def finish(self):
        per_pool = {}
        arrivals_all, starts_all = [], []
        daily = []
        for index, end in enumerate(self.daily_edges[1:]):
            daily.append({"elapsed_hr": end / 3600.0})
        for name, p in self.pools.items():
            arrivals = np.frombuffer(p["arrivals"], dtype=np.float64)
            starts = np.frombuffer(p["starts"], dtype=np.float64)
            arrivals.sort()
            starts.sort()
            arrivals_all.append(arrivals)
            starts_all.append(starts)
            available = p["workers"] * self.window_seconds
            per_pool[name] = {
                "workers": p["workers"], "total_jobs": p["jobs"],
                "queued_jobs_pct": 100.0 * p["queued_jobs"] / p["jobs"] if p["jobs"] else 0.0,
                "mean_job_queue_wait_hr": p["wait_seconds"] / max(p["jobs"], 1) / 3600.0,
                "max_job_queue_wait_hr": p["max_wait_seconds"] / 3600.0,
                "max_backlog": _peak_queue(arrivals, starts),
                "max_backlog_during_input": _peak_queue(arrivals, starts, self.window_seconds),
                "backlog_at_input_end": p["backlog_at_input_end"],
                "running_at_input_end": p["running_at_input_end"],
                "not_ready_at_input_end": p["not_ready_at_input_end"],
                "worker_utilization_pct": 100.0 * p["busy_seconds"] / available,
                "busy_worker_hours_during_input": p["busy_seconds"] / 3600.0,
                "idle_worker_hours_during_input": (available - p["busy_seconds"]) / 3600.0,
                "drain_time_hr": max(0.0, p["last_finish_seconds"] - self.window_seconds) / 3600.0,
            }
            for index, row in enumerate(daily):
                end = self.daily_edges[index + 1]
                span = end - self.daily_edges[index]
                row[name + "_backlog"] = int(
                    np.searchsorted(arrivals, end, side="right")
                    - np.searchsorted(starts, end, side="right")
                )
                row[name + "_worker_utilization_pct"] = 100.0 * p["daily_busy"][index] / (p["workers"] * span)

        arrivals = np.concatenate(arrivals_all)
        starts = np.concatenate(starts_all)
        arrivals.sort()
        starts.sort()
        jobs = sum(p["jobs"] for p in self.pools.values())
        workers = sum(p["workers"] for p in self.pools.values())
        busy = sum(p["busy_seconds"] for p in self.pools.values())
        available = workers * self.window_seconds
        aggregate = {
            "total_jobs": jobs,
            "mean_job_queue_wait_hr": sum(p["wait_seconds"] for p in self.pools.values()) / max(jobs, 1) / 3600.0,
            "max_job_queue_wait_hr": max(p["max_wait_seconds"] for p in self.pools.values()) / 3600.0,
            "queued_jobs_pct": 100.0 * sum(p["queued_jobs"] for p in self.pools.values()) / max(jobs, 1),
            "max_backlog": _peak_queue(arrivals, starts),
            "max_backlog_during_input": _peak_queue(arrivals, starts, self.window_seconds),
            "backlog_at_input_end": sum(p["backlog_at_input_end"] for p in self.pools.values()),
            "running_at_input_end": sum(p["running_at_input_end"] for p in self.pools.values()),
            "not_ready_at_input_end": sum(p["not_ready_at_input_end"] for p in self.pools.values()),
            "worker_utilization_pct": 100.0 * busy / available,
            "busy_worker_hours_during_input": busy / 3600.0,
            "idle_worker_hours_during_input": (available - busy) / 3600.0,
            "drain_time_hr": max(p["drain_time_hr"] for p in per_pool.values()),
        }
        for row in daily:
            row["backlog"] = sum(row[name + "_backlog"] for name in self.pools)
            row["worker_utilization_pct"] = sum(
                row[name + "_worker_utilization_pct"] * p["workers"]
                for name, p in self.pools.items()
            ) / workers
        return {"aggregate": aggregate, "per_pool": per_pool}, daily
