"""Test oracle implementations for Mozilla performance bisect simulations."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


class OracleDecision(str, Enum):
    """A localized revision classification."""

    CLEAN = "clean"
    BAD = "bad"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FailingSignature:
    """Failing performance signature fields carried by one regression row."""

    signature_id: int
    platform: str | None = None


@dataclass(frozen=True)
class SignatureInfo:
    """Static metadata for one performance signature."""

    signature_id: int
    job_duration_minutes: float
    platform: str | None = None


class SignatureInfoIndex:
    """Lookup table for per-signature metadata."""

    def __init__(self, infos: Sequence[SignatureInfo]) -> None:
        """Build exact platform-aware and signature-only lookup maps."""

        self._by_exact: dict[tuple[int, str | None], SignatureInfo] = {}
        self._by_signature: dict[int, SignatureInfo] = {}
        for info in infos:
            self._by_exact[(info.signature_id, info.platform)] = info
            self._by_signature.setdefault(info.signature_id, info)

    def get(self, signature_id: int, platform: str | None = None) -> SignatureInfo | None:
        """Return metadata for a signature, preferring an exact platform match."""

        if platform is not None:
            exact = self._by_exact.get((signature_id, platform))
            if exact is not None:
                return exact
        return self._by_signature.get(signature_id)


@dataclass
class RevisionRecord:
    """Revision graph node."""

    node: str
    parents: list[str]
    date: Any = None


class RevisionPerfIndex:
    """Revision graph indexed by node hash."""

    def __init__(self, records: Sequence[RevisionRecord]) -> None:
        """Index revision records by node hash."""

        self.records: dict[str, RevisionRecord] = {
            record.node: record for record in records
        }

    def path_between(self, good_revision: str, bad_revision: str) -> list[str] | None:
        """Return a good-to-bad parent path, if one exists in the revision graph."""

        if good_revision not in self.records or bad_revision not in self.records:
            return None
        if good_revision == bad_revision:
            return [good_revision]

        queue = [bad_revision]
        next_child_by_parent: dict[str, str | None] = {bad_revision: None}

        for revision in queue:
            if revision == good_revision:
                break
            record = self.records.get(revision)
            if record is None:
                continue
            for parent in record.parents:
                if parent not in next_child_by_parent:
                    next_child_by_parent[parent] = revision
                    queue.append(parent)

        if good_revision not in next_child_by_parent:
            return None

        path: list[str] = []
        revision: str | None = good_revision
        while revision is not None:
            path.append(revision)
            revision = next_child_by_parent[revision]
        return path


@dataclass
class TestJob:
    """Timing result for one submitted test job."""

    submitted_at_minutes: float
    completed_at_minutes: float
    test_runs: int


class TestExecutor:
    """A small queueing simulator for performance test workers."""

    def __init__(self, workers: int = 1) -> None:
        """Create an empty worker pool for one independent regression simulation."""

        if workers < 1:
            raise ValueError("workers must be at least 1")

        self.workers = workers
        self._worker_available_at = [0.0 for _ in range(workers)]
        self.now_minutes = 0.0
        self.test_runs = 0

    def submit_job_and_wait(self, duration_minutes: float) -> TestJob:
        """Submit one job and advance time until it completes."""

        return self.submit_jobs_and_wait([("default", duration_minutes)])["default"]

    def submit_jobs_and_wait(
        self,
        jobs: Sequence[tuple[str, float]],
    ) -> dict[str, TestJob]:
        """Submit multiple same-time jobs and wait for the full queued workload."""

        if not jobs:
            return {}

        submitted_at = self.now_minutes
        completed_at = submitted_at
        completed_by_key = {key: submitted_at for key, _ in jobs}

        for key, duration_minutes in jobs:
            if duration_minutes <= 0:
                raise ValueError("duration_minutes must be positive")
            worker_idx = min(
                range(self.workers),
                key=lambda idx: self._worker_available_at[idx],
            )
            start_at = max(submitted_at, self._worker_available_at[worker_idx])
            end_at = start_at + duration_minutes
            self._worker_available_at[worker_idx] = end_at
            completed_by_key[key] = end_at
            completed_at = max(completed_at, end_at)

        self.test_runs += len(jobs)
        self.now_minutes = completed_at
        return {
            key: TestJob(submitted_at, completed_by_key[key], 1)
            for key, _ in jobs
        }


@dataclass
class OracleResult:
    """Observed oracle decision and timing metadata for one probe attempt."""

    revision: str
    decision: OracleDecision
    attempt: int
    draw_index: int | None
    measurement_count: int
    test_runs: int
    submitted_at_minutes: float
    completed_at_minutes: float
    value_source: str | None = None
    value_source_revision: str | None = None


class TestOracle:
    """Base class for localization test oracles."""

    name = "TestOracle"

    def classify(
        self,
        revision: str,
        *,
        regression: Mapping[str, Any],
        revision_path: Sequence[str],
    ) -> OracleResult:
        """Classify one revision as clean or bad."""

        raise NotImplementedError

    def classify_many(
        self,
        revisions: Sequence[str],
        *,
        regression: Mapping[str, Any],
        revision_path: Sequence[str],
    ) -> list[OracleResult]:
        """Classify multiple revisions, using repeated single-revision calls."""

        return [
            self.classify(
                revision,
                regression=regression,
                revision_path=revision_path,
            )
            for revision in revisions
        ]

    def accuracy_for(
        self,
        revision: str,
        *,
        regression: Mapping[str, Any],
        revision_path: Sequence[str],
    ) -> float:
        """Return the clean/bad decision accuracy for one revision probe."""

        raise NotImplementedError


class SummaryComparison(TestOracle):
    """Classify revisions with a per-regression noisy summary oracle."""

    name = "SummaryComparison"

    def __init__(
        self,
        *,
        signature_info: SignatureInfoIndex,
        executor: TestExecutor,
        oracle_accuracy: float,
        random_seed: int | None = 0,
    ) -> None:
        """Create a noisy oracle backed by one summary accuracy value."""

        if not 0.0 <= oracle_accuracy <= 1.0:
            raise ValueError(
                f"oracle_accuracy must be between 0 and 1: {oracle_accuracy!r}"
            )

        self.signature_info = signature_info
        self.executor = executor
        self.oracle_accuracy = oracle_accuracy
        self.query_count = 0
        self._attempt_counts: dict[str, int] = {}
        self._rng = random.Random(random_seed)

    def classify(
        self,
        revision: str,
        *,
        regression: Mapping[str, Any],
        revision_path: Sequence[str],
    ) -> OracleResult:
        """Classify one revision by submitting exactly one test job."""

        return self.classify_many(
            [revision],
            regression=regression,
            revision_path=revision_path,
        )[0]

    def classify_many(
        self,
        revisions: Sequence[str],
        *,
        regression: Mapping[str, Any],
        revision_path: Sequence[str],
    ) -> list[OracleResult]:
        """Submit all requested revision jobs at once, then draw noisy verdicts."""

        if not revisions:
            return []

        failing_sig = parse_failing_signature(regression)
        info = self.signature_info.get(failing_sig.signature_id, failing_sig.platform)
        if info is None:
            raise ValueError(
                "missing job_duration for "
                f"signature_id={failing_sig.signature_id} "
                f"platform={failing_sig.platform!r}"
            )

        requests = []
        jobs_to_submit = []
        for idx, revision in enumerate(revisions):
            attempt = self._attempt_counts.get(revision, 0)
            self._attempt_counts[revision] = attempt + 1
            job_key = f"{idx}:{revision}:{attempt}"
            requests.append((revision, attempt, job_key))
            jobs_to_submit.append((job_key, info.job_duration_minutes))

        jobs = self.executor.submit_jobs_and_wait(jobs_to_submit)
        self.query_count += len(revisions)

        return [
            self._classify_with_completed_job(
                revision=revision,
                attempt=attempt,
                job=jobs[job_key],
                regression=regression,
                revision_path=revision_path,
            )
            for revision, attempt, job_key in requests
        ]

    def _classify_with_completed_job(
        self,
        *,
        revision: str,
        attempt: int,
        job: TestJob,
        regression: Mapping[str, Any],
        revision_path: Sequence[str],
    ) -> OracleResult:
        """Turn a completed test job into a noisy clean/bad decision."""

        expected_decision = self._expected_decision(
            revision=revision,
            regression=regression,
            revision_path=revision_path,
        )
        if expected_decision is None:
            decision = OracleDecision.UNKNOWN
            measurement_count = 0
            source = "missing_culprit_boundary"
        else:
            decision = self._draw_noisy_decision(expected_decision)
            measurement_count = 1
            source = "noisy_summary_oracle"

        return OracleResult(
            revision=revision,
            decision=decision,
            attempt=attempt + 1,
            draw_index=None,
            measurement_count=measurement_count,
            test_runs=1,
            submitted_at_minutes=job.submitted_at_minutes,
            completed_at_minutes=job.completed_at_minutes,
            value_source=source,
            value_source_revision=None,
        )

    def _draw_noisy_decision(self, expected_decision: OracleDecision) -> OracleDecision:
        """Return the expected side with configured probability, otherwise flip it."""

        if self._rng.random() < self.oracle_accuracy:
            return expected_decision
        if expected_decision is OracleDecision.BAD:
            return OracleDecision.CLEAN
        return OracleDecision.BAD

    def accuracy_for(
        self,
        revision: str,
        *,
        regression: Mapping[str, Any],
        revision_path: Sequence[str],
    ) -> float:
        """Return the configured per-regression summary-oracle accuracy."""

        del revision, regression, revision_path
        return self.oracle_accuracy

    @staticmethod
    def _expected_decision(
        *,
        revision: str,
        regression: Mapping[str, Any],
        revision_path: Sequence[str],
    ) -> OracleDecision | None:
        """Return the true side of a revision relative to the known culprit."""

        culprit_revision = regression.get("culprit_revision")
        if culprit_revision is None:
            return None
        culprit_revision = str(culprit_revision)
        try:
            revision_idx = revision_path.index(revision)
            culprit_idx = revision_path.index(culprit_revision)
        except ValueError:
            return None

        return (
            OracleDecision.BAD
            if revision_idx >= culprit_idx
            else OracleDecision.CLEAN
        )


def parse_failing_signature(regression: Mapping[str, Any]) -> FailingSignature:
    """Parse both observed singular and prompt-described plural signature shapes."""

    if "failing_sig" in regression:
        raw = regression["failing_sig"]
        signature_id = int(raw["signature_id"])
    elif "failing_sigs" in regression:
        failing_sigs = regression["failing_sigs"]
        if not failing_sigs:
            raise ValueError("regression has empty failing_sigs")
        signature_id_text, raw = next(iter(failing_sigs.items()))
        signature_id = int(raw.get("signature_id", signature_id_text))
    else:
        raise ValueError("regression is missing failing_sig/failing_sigs")

    return FailingSignature(
        signature_id=signature_id,
        platform=raw.get("platform"),
    )
