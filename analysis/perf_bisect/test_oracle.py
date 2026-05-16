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
    replicate_count: int
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
class TestBatch:
    """Timing result for one submitted batch of replicate tests."""

    submitted_at_minutes: float
    completed_at_minutes: float
    test_runs: int


class TestExecutor:
    """A small queueing simulator for performance test workers."""

    def __init__(self, workers: int = 1, test_duration_minutes: float = 1.0) -> None:
        """Create an empty worker pool for one independent regression simulation."""

        if workers < 1:
            raise ValueError("workers must be at least 1")
        if test_duration_minutes <= 0:
            raise ValueError("test_duration_minutes must be positive")

        self.workers = workers
        self.test_duration_minutes = test_duration_minutes
        self._worker_available_at = [0.0 for _ in range(workers)]
        self.now_minutes = 0.0
        self.test_runs = 0

    def submit_batch_and_wait(self, test_runs: int) -> TestBatch:
        """Submit a same-time batch and advance time until all tests complete."""

        if test_runs < 1:
            raise ValueError("test_runs must be at least 1")

        return self.submit_batches_and_wait([("default", test_runs)])["default"]

    def submit_batches_and_wait(
        self,
        batches: Sequence[tuple[str, int]],
    ) -> dict[str, TestBatch]:
        """Submit multiple same-time batches and wait for the full queued workload."""

        if not batches:
            return {}

        submitted_at = self.now_minutes
        completed_at = submitted_at
        completed_by_key = {key: submitted_at for key, _ in batches}

        for key, test_runs in batches:
            if test_runs < 1:
                raise ValueError("test_runs must be at least 1")
            for _ in range(test_runs):
                worker_idx = min(
                    range(self.workers),
                    key=lambda idx: self._worker_available_at[idx],
                )
                start_at = max(submitted_at, self._worker_available_at[worker_idx])
                end_at = start_at + self.test_duration_minutes
                self._worker_available_at[worker_idx] = end_at
                completed_by_key[key] = max(completed_by_key[key], end_at)
                completed_at = max(completed_at, end_at)

        self.test_runs += sum(test_runs for _, test_runs in batches)
        self.now_minutes = completed_at
        return {
            key: TestBatch(submitted_at, completed_by_key[key], test_runs)
            for key, test_runs in batches
        }


@dataclass
class OracleResult:
    """Observed oracle decision and timing metadata for one probe attempt."""

    revision: str
    decision: OracleDecision
    attempt: int
    draw_index: int | None
    replicate_count: int
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
        """Classify one revision by submitting exactly one test batch."""

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
        """Submit all requested revision tests at once, then draw noisy verdicts."""

        if not revisions:
            return []

        failing_sig = parse_failing_signature(regression)
        info = self.signature_info.get(failing_sig.signature_id, failing_sig.platform)
        replicate_count = info.replicate_count if info is not None else 1

        requests = []
        batches_to_submit = []
        for idx, revision in enumerate(revisions):
            attempt = self._attempt_counts.get(revision, 0)
            self._attempt_counts[revision] = attempt + 1
            batch_key = f"{idx}:{revision}:{attempt}"
            requests.append((revision, attempt, batch_key))
            batches_to_submit.append((batch_key, replicate_count))

        batches = self.executor.submit_batches_and_wait(batches_to_submit)
        self.query_count += len(revisions)

        return [
            self._classify_with_completed_batch(
                revision=revision,
                attempt=attempt,
                replicate_count=replicate_count,
                batch=batches[batch_key],
                regression=regression,
                revision_path=revision_path,
            )
            for revision, attempt, batch_key in requests
        ]

    def _classify_with_completed_batch(
        self,
        *,
        revision: str,
        attempt: int,
        replicate_count: int,
        batch: TestBatch,
        regression: Mapping[str, Any],
        revision_path: Sequence[str],
    ) -> OracleResult:
        """Turn a completed test batch into a noisy clean/bad decision."""

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
            replicate_count=replicate_count,
            measurement_count=measurement_count,
            test_runs=replicate_count,
            submitted_at_minutes=batch.submitted_at_minutes,
            completed_at_minutes=batch.completed_at_minutes,
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
