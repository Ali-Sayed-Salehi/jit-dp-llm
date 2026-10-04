"""Optional observations of batch coverage and diagnosis; no scheduling decisions.

The recorder is scoped to one FINAL replay. Normal simulations have no active
recorder. Times are in hours in the exported CSVs; missing observations remain
null/blank, including coverage when failing signature metadata is unavailable.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean


SUPPORTED_BATCHING = {"TWSB", "TWB", "FSB", "RASB", "RASB-la", "RAPB", "RAPB-la", "RATB"}
SUPPORTED_BATCHING |= {name + "-s" for name in SUPPORTED_BATCHING if name != "TWSB"}
_ACTIVE = ContextVar("subset_diagnostics", default=None)


def active_recorder():
    return _ACTIVE.get()


@contextmanager
def collect(recorder):
    token = _ACTIVE.set(recorder)
    try:
        yield recorder
    finally:
        _ACTIVE.reset(token)


def _hours(later, earlier):
    if later is None or earlier is None:
        return None
    return (later - earlier).total_seconds() / 3600.0


def _iso(value):
    return value.isoformat() if value is not None else None


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class SubsetDiagnostics:
    def __init__(self, combo, commits, failing_groups, full_suite_ids, build_time_minutes):
        self.combo = combo
        self.commits = commits
        self.full_suite_ids = set(full_suite_ids)
        self.build_time_hr = float(build_time_minutes) / 60.0
        self.batches = []
        self.detections = []
        self.current_detection_id = None
        self._durations = {}
        self._boundaries = []
        self._regressors = {}
        for index, commit in enumerate(commits):
            if not commit.get("true_label"):
                continue
            cid = commit["commit_id"]
            self._regressors[cid] = {
                "commit_index": index, "commit_id": cid, "risk": commit["risk"],
                "arrival": commit["ts"],
                "failing_signature_group_ids": sorted(set(failing_groups(cid))),
                "first_batch_id": None, "first_covering_batch_id": None,
                "first_batch_overlap_count": None, "missed_batches_before_coverage": 0,
                "first_detection_id": None, "finding_detection_id": None,
                "found_at": None,
            }

    def record_batch(self, start, end, flush_time, durations):
        # These diagnostics describe disjoint chronological batches, including
        # empty suites. Growing-prefix/HATS policies are intentionally unsupported.
        expected_start = self._boundaries[-1][1] + 1 if self._boundaries else 0
        if start != expected_start or end < start or end >= len(self.commits):
            raise ValueError(f"Diagnostics require disjoint, ordered batches: {self.combo}")
        batch_id = len(self.batches)
        batch = self.commits[start:end + 1]
        groups = {int(gid) for gid, _ in durations if gid is not None}
        self._durations[batch_id] = {gid: float(duration) / 60.0 for gid, duration in durations}
        self._boundaries.append((start, end, _iso(flush_time)))
        self.batches.append({
            "batch_id": batch_id, "start_commit_index": start, "end_commit_index": end,
            "first_commit_id": batch[0]["commit_id"], "last_commit_id": batch[-1]["commit_id"],
            "first_commit_at": _iso(batch[0]["ts"]), "last_commit_at": _iso(batch[-1]["ts"]),
            "flush_at": _iso(flush_time), "batch_size": len(batch),
            "arrival_span_hr": _hours(batch[-1]["ts"], batch[0]["ts"]),
            "oldest_commit_wait_hr": _hours(flush_time, batch[0]["ts"]),
            "risk_sum": sum(float(c["risk"]) for c in batch),
            "max_risk": max(float(c["risk"]) for c in batch),
            "num_regressors_in_batch": sum(bool(c.get("true_label")) for c in batch),
            "suite_size": len(groups), "suite_fraction_of_full": (
                len(groups & self.full_suite_ids) / len(self.full_suite_ids)
                if self.full_suite_ids else None
            ),
            "signature_group_ids": sorted(groups),
        })
        for reg in self._regressors.values():
            if reg["commit_index"] > end:
                continue
            overlap = groups.intersection(reg["failing_signature_group_ids"])
            if reg["first_batch_id"] is None:
                reg["first_batch_id"] = batch_id
                reg["first_batch_overlap_count"] = (
                    len(overlap) if reg["failing_signature_group_ids"] else None
                )
            if reg["first_covering_batch_id"] is None and reg["failing_signature_group_ids"]:
                if overlap:
                    reg["first_covering_batch_id"] = batch_id
                else:
                    reg["missed_batches_before_coverage"] += 1
        return batch_id

    def record_detection(self, batch_id, gid, start, end, defect_indices, at_time):
        detection_id = len(self.detections)
        # Keep every root failure/search, including multiple signature-groups for
        # one regressor. The finding event is recorded separately from the first.
        row = {
            "detection_id": detection_id, "batch_id": batch_id, "signature_group_id": gid,
            "candidate_start_index": start, "candidate_end_index": end,
            "candidate_count": end - start + 1,
            "candidate_arrival_span_hr": _hours(self.commits[end]["ts"], self.commits[start]["ts"]),
            "regressor_commit_ids": [self.commits[i]["commit_id"] for i in sorted(defect_indices)],
            "detected_at": _iso(at_time),
            "root_build_hr": self.build_time_hr,
            "root_test_hr": self._durations[batch_id].get(gid),
        }
        flush_time = self._flush_time(batch_id)
        row["root_queue_hr"] = (
            max(0.0, _hours(at_time, flush_time) - self.build_time_hr - row["root_test_hr"])
            if row["root_test_hr"] is not None else None
        )
        self.detections.append(row)
        for cid in row["regressor_commit_ids"]:
            reg = self._regressors[cid]
            previous = reg["first_detection_id"]
            if previous is None or at_time < datetime.fromisoformat(self.detections[previous]["detected_at"]):
                reg["first_detection_id"] = detection_id
        return detection_id

    def record_found(self, commit_id, at_time):
        reg = self._regressors[commit_id]
        reg["found_at"] = at_time
        reg["finding_detection_id"] = self.current_detection_id

    def _flush_time(self, batch_id):
        # All batch boundaries store the original datetime as an ISO timestamp.
        return datetime.fromisoformat(self.batches[batch_id]["flush_at"])

    def finish(self, metrics, parameters):
        if self.commits and (not self._boundaries or self._boundaries[-1][1] != len(self.commits) - 1):
            raise ValueError(f"Incomplete batch trace: {self.combo}")
        rows = []
        for reg in self._regressors.values():
            first_id = reg["first_batch_id"]
            cover_id = reg["first_covering_batch_id"]
            first = self.batches[first_id] if first_id is not None else {}
            first_flush = self._flush_time(first_id) if first_id is not None else None
            cover_flush = self._flush_time(cover_id) if cover_id is not None else None
            first_event = (self.detections[reg["first_detection_id"]]
                           if reg["first_detection_id"] is not None else {})
            finding_event = (self.detections[reg["finding_detection_id"]]
                             if reg["finding_detection_id"] is not None else {})
            detected_at = (datetime.fromisoformat(first_event["detected_at"])
                           if first_event else None)
            finding_start = (datetime.fromisoformat(finding_event["detected_at"])
                             if finding_event else None)
            fail_count = len(reg["failing_signature_group_ids"])
            row = {key: value for key, value in reg.items() if key not in {"arrival", "found_at"}}
            row.update({
                "arrival_at": _iso(reg["arrival"]), "found": reg["found_at"] is not None,
                "found_at": _iso(reg["found_at"]), "coverage_metadata_known": bool(fail_count),
                "first_batch_covered": (reg["first_batch_overlap_count"] > 0 if fail_count else None),
                "first_batch_failing_group_coverage": (
                    reg["first_batch_overlap_count"] / fail_count if fail_count else None
                ),
                "first_batch_size": first.get("batch_size"),
                "first_batch_suite_size": first.get("suite_size"),
                "first_batch_arrival_span_hr": first.get("arrival_span_hr"),
                "first_batch_flush_at": _iso(first_flush), "first_coverage_flush_at": _iso(cover_flush),
                "first_detected_at": _iso(detected_at),
                "first_detection_candidate_count": first_event.get("candidate_count"),
                "finding_candidate_count": finding_event.get("candidate_count"),
                "finding_signature_group_id": finding_event.get("signature_group_id"),
                "batch_wait_hr": _hours(first_flush, reg["arrival"]),
                "coverage_wait_hr": _hours(cover_flush, first_flush),
                "coverage_wait_lower_bound_hr": (
                    _hours(self._flush_time(len(self.batches) - 1), first_flush)
                    if fail_count and cover_id is None else None
                ),
                "covered_flush_to_detection_hr": _hours(detected_at, cover_flush),
                "detection_to_culprit_hr": _hours(reg["found_at"], detected_at),
                "finding_search_hr": _hours(reg["found_at"], finding_start),
                "time_to_culprit_hr": _hours(reg["found_at"], reg["arrival"]),
                "first_detection_root_build_hr": first_event.get("root_build_hr"),
                "first_detection_root_queue_hr": first_event.get("root_queue_hr"),
                "first_detection_root_test_hr": first_event.get("root_test_hr"),
            })
            if not fail_count:
                row["missed_batches_before_coverage"] = None
            rows.append(row)
        if sum(row["found"] for row in rows) != metrics["num_regressors_found"]:
            raise ValueError(f"Diagnostic/summary regression counts disagree: {self.combo}")
        covered = [r for r in rows if r["coverage_wait_hr"] is not None]
        summary = {
            "num_batches": len(self.batches), "num_regressors": len(rows),
            "num_covered_in_first_batch": sum(r["first_batch_covered"] is True for r in rows),
            "num_missed_in_first_batch": sum(r["first_batch_covered"] is False for r in rows),
            "num_unknown_coverage": sum(not r["coverage_metadata_known"] for r in rows),
            "num_never_covered": sum(r["coverage_metadata_known"] and r["first_covering_batch_id"] is None for r in rows),
            "mean_coverage_wait_hr": mean(r["coverage_wait_hr"] for r in covered) if covered else None,
            "max_coverage_wait_hr": max((r["coverage_wait_hr"] for r in covered), default=None),
            **metrics,
        }
        return {"combo": self.combo, "parameters": parameters, "summary": summary,
                "batch_boundaries_sha256": _digest(self._boundaries),
                "batches": self.batches, "regressors": rows, "detections": self.detections}


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["combo"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def write_diagnostics(directory, records, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ["batches", "regressors", "detections"]:
        _write_csv(directory / f"{name}.csv", [
            {"combo": record["combo"], **row} for record in records for row in record[name]
        ])
    _write_csv(directory / "summary.csv", [
        {"combo": record["combo"], **record["summary"]} for record in records
    ])
    by_combo = {record["combo"]: record for record in records}
    pairs = []
    for subset in records:
        batching, bisection = subset["combo"].split(" + ", 1)
        if not batching.endswith("-s"):
            continue
        full_name = batching[:-2] + " + " + bisection
        full = by_combo.get(full_name)
        pairs.append({
            "full_combo": full_name, "subset_combo": subset["combo"],
            "full_replayed": full is not None,
            "same_parameters": full["parameters"] == subset["parameters"] if full else None,
            "same_batch_boundaries": (full["batch_boundaries_sha256"] == subset["batch_boundaries_sha256"]
                                      if full else None),
        })
    manifest = {"schema_version": 1, **metadata,
                "files": ["summary.csv", "batches.csv", "regressors.csv", "detections.csv"],
                "configurations": [{"combo": r["combo"], "parameters": r["parameters"],
                                    "batch_boundaries_sha256": r["batch_boundaries_sha256"]} for r in records],
                "paired_comparisons": pairs}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
