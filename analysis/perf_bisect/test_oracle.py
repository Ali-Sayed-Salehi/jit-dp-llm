"""Test oracle implementations for Mozilla performance bisect simulations."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class OracleDecision(str, Enum):
    """A localized revision classification."""

    CLEAN = "clean"
    BAD = "bad"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FailingSignature:
    """Performance signature fields carried by one regression row."""

    signature_id: int
    good_value: float
    bad_value: float
    platform: str | None = None

    @property
    def baseline(self) -> float:
        """Return the midpoint used to separate clean and bad measurements."""

        return (self.good_value + self.bad_value) / 2.0


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
    """Revision graph node and its performance measurements."""

    node: str
    parents: list[str]
    date: Any = None
    summary_values: dict[int, list[float]] = field(default_factory=dict)
    replicate_values: dict[int, list[float]] = field(default_factory=dict)


class RevisionPerfIndex:
    """Revision graph plus performance measurements."""

    def __init__(self, records: Sequence[RevisionRecord]) -> None:
        """Index revision records by node hash."""

        self.records: dict[str, RevisionRecord] = {record.node: record for record in records}

    def get_summary_values(self, revision: str, signature_id: int) -> list[float]:
        """Return all summary values recorded for a revision/signature pair."""

        record = self.records.get(revision)
        if record is None:
            return []
        return record.summary_values.get(signature_id, [])

    def get_replicate_values(self, revision: str, signature_id: int) -> list[float]:
        """Return all replicate values recorded for a revision/signature pair."""

        record = self.records.get(revision)
        if record is None:
            return []
        return record.replicate_values.get(signature_id, [])

    def get_summary_value(
        self,
        revision: str,
        signature_id: int,
        draw_index: int = 0,
    ) -> float | None:
        """Return one summary draw by index, if the requested draw exists."""

        values = self.get_summary_values(revision, signature_id)
        if not values:
            return None
        if draw_index < len(values):
            return values[draw_index]
        return None

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
    value: float | None
    baseline: float
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
        """Classify multiple revisions, using the single-revision fallback path."""

        return [
            self.classify(
                revision,
                regression=regression,
                revision_path=revision_path,
            )
            for revision in revisions
        ]


class SummaryComparison(TestOracle):
    """Classify revisions by comparing measurement values against the midpoint."""

    name = "SummaryComparison"

    def __init__(
        self,
        *,
        signature_info: SignatureInfoIndex,
        revision_perf: RevisionPerfIndex,
        executor: TestExecutor,
        random_seed: int | None = 0,
    ) -> None:
        """Create a baseline-comparison oracle backed by revision measurements."""

        self.signature_info = signature_info
        self.revision_perf = revision_perf
        self.executor = executor
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
        """Classify one revision by submitting exactly one replicate batch."""

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
        """Submit all requested revision tests at once, then classify each result."""

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
        drawn_values = self._draw_measurement_values_for_requests(
            requests=requests,
            failing_sig=failing_sig,
            regression=regression,
            revision_path=revision_path,
        )

        return [
            self._classify_with_completed_batch(
                revision=revision,
                attempt=attempt,
                failing_sig=failing_sig,
                replicate_count=replicate_count,
                batch=batches[batch_key],
                drawn_value=drawn_values[batch_key],
            )
            for revision, attempt, batch_key in requests
        ]

    def _draw_measurement_values_for_requests(
        self,
        *,
        requests: Sequence[tuple[str, int, str]],
        failing_sig: FailingSignature,
        regression: Mapping[str, Any],
        revision_path: Sequence[str],
    ) -> dict[str, tuple[float | None, str | None, str | None, int | None]]:
        """Draw direct values first, then randomly draw same-side fallbacks."""

        drawn_values: dict[
            str,
            tuple[float | None, str | None, str | None, int | None],
        ] = {}
        fallback_requests = []

        for revision, attempt, batch_key in requests:
            direct_value = self._direct_measurement_value(
                revision=revision,
                signature=failing_sig,
                measurement_kind="summary",
                value_source="direct_summary",
            )
            if direct_value is None:
                direct_value = self._direct_measurement_value(
                    revision=revision,
                    signature=failing_sig,
                    measurement_kind="replicate",
                    value_source="direct_replicate",
                )
            if direct_value is None:
                fallback_requests.append((revision, batch_key))
                continue
            drawn_values[batch_key] = direct_value

        for revision, batch_key in fallback_requests:
            fallback_value = self._same_side_measurement_value(
                revision=revision,
                signature=failing_sig,
                regression=regression,
                revision_path=revision_path,
                measurement_kind="summary",
                value_source="same_side_summary",
            )
            if fallback_value is None:
                fallback_value = self._same_side_measurement_value(
                    revision=revision,
                    signature=failing_sig,
                    regression=regression,
                    revision_path=revision_path,
                    measurement_kind="replicate",
                    value_source="same_side_replicate",
                )
            if fallback_value is None:
                fallback_value = self._boundary_replicate_value(
                    revision=revision,
                    signature=failing_sig,
                    regression=regression,
                    revision_path=revision_path,
                )
            if fallback_value is None:
                fallback_value = (
                    None,
                    "missing_measurement",
                    None,
                    None,
                )
            drawn_values[batch_key] = fallback_value

        return drawn_values

    def _direct_measurement_value(
        self,
        *,
        revision: str,
        signature: FailingSignature,
        measurement_kind: str,
        value_source: str,
    ) -> tuple[float, str | None, str | None, int | None] | None:
        """Randomly draw one direct observation if available."""

        direct_values = self._measurement_values(
            revision=revision,
            signature_id=signature.signature_id,
            measurement_kind=measurement_kind,
        )
        if not direct_values:
            return None

        value_idx, value = self._rng.choice(list(enumerate(direct_values)))
        return value, value_source, revision, value_idx

    def _classify_with_completed_batch(
        self,
        *,
        revision: str,
        attempt: int,
        failing_sig: FailingSignature,
        replicate_count: int,
        batch: TestBatch,
        drawn_value: tuple[float | None, str | None, str | None, int | None],
    ) -> OracleResult:
        """Turn a completed test batch and selected measurement into a decision."""

        value, source, source_revision, draw_index = drawn_value
        decision = (
            OracleDecision.UNKNOWN
            if value is None
            else self._compare(value, failing_sig.baseline)
        )
        return OracleResult(
            revision=revision,
            decision=decision,
            value=value,
            baseline=failing_sig.baseline,
            attempt=attempt + 1,
            draw_index=draw_index,
            replicate_count=replicate_count,
            measurement_count=0 if value is None else 1,
            test_runs=replicate_count,
            submitted_at_minutes=batch.submitted_at_minutes,
            completed_at_minutes=batch.completed_at_minutes,
            value_source=source,
            value_source_revision=source_revision,
        )

    def _same_side_measurement_value(
        self,
        *,
        revision: str,
        signature: FailingSignature,
        regression: Mapping[str, Any],
        revision_path: Sequence[str],
        measurement_kind: str,
        value_source: str,
    ) -> tuple[float, str | None, str | None, int | None] | None:
        """Randomly draw one fallback value from the same culprit side."""

        if revision not in revision_path:
            return None
        if regression.get("culprit_revision") not in revision_path:
            return None

        revision_idx = revision_path.index(revision)
        culprit_idx = revision_path.index(str(regression["culprit_revision"]))
        is_bad_side = revision_idx >= culprit_idx
        boundary_indices = {0, len(revision_path) - 1}

        side_indices = [
            idx
            for idx in range(len(revision_path))
            if (idx >= culprit_idx) == is_bad_side
            and idx != revision_idx
            and idx not in boundary_indices
        ]
        side_indices.sort(key=lambda idx: abs(idx - revision_idx))

        same_side_values = []
        for idx in side_indices:
            source_revision = revision_path[idx]
            for value_idx, value in enumerate(
                self._measurement_values(
                    revision=source_revision,
                    signature_id=signature.signature_id,
                    measurement_kind=measurement_kind,
                )
            ):
                same_side_values.append((value, source_revision, value_idx))

        if not same_side_values:
            return None

        value, source_revision, value_idx = self._rng.choice(same_side_values)
        return value, value_source, source_revision, value_idx

    def _boundary_replicate_value(
        self,
        *,
        revision: str,
        signature: FailingSignature,
        regression: Mapping[str, Any],
        revision_path: Sequence[str],
    ) -> tuple[float, str | None, str | None, int | None] | None:
        """Randomly draw one replicate value from the same-side boundary."""

        if revision not in revision_path:
            return None
        if regression.get("culprit_revision") not in revision_path:
            return None

        revision_idx = revision_path.index(revision)
        culprit_idx = revision_path.index(str(regression["culprit_revision"]))
        is_bad_side = revision_idx >= culprit_idx
        source_revision = revision_path[-1] if is_bad_side else revision_path[0]
        value_source = (
            "bad_boundary_replicate" if is_bad_side else "good_boundary_replicate"
        )

        boundary_values = []
        for value_idx, value in enumerate(
            self.revision_perf.get_replicate_values(
                source_revision,
                signature.signature_id,
            )
        ):
            boundary_values.append((value, value_idx))

        if not boundary_values:
            return None

        value, value_idx = self._rng.choice(boundary_values)
        return value, value_source, source_revision, value_idx

    def _measurement_values(
        self,
        *,
        revision: str,
        signature_id: int,
        measurement_kind: str,
    ) -> list[float]:
        """Return summary or replicate values for one revision/signature pair."""

        if measurement_kind == "summary":
            return self.revision_perf.get_summary_values(revision, signature_id)
        if measurement_kind == "replicate":
            return self.revision_perf.get_replicate_values(revision, signature_id)
        raise ValueError(f"unknown measurement kind: {measurement_kind}")

    @staticmethod
    def _compare(
        value: float,
        baseline: float,
    ) -> OracleDecision:
        """Classify one value using only the baseline."""

        return OracleDecision.BAD if value > baseline else OracleDecision.CLEAN


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
        good_value=float(raw["Good_value"]),
        bad_value=float(raw["bad_value"]),
        platform=raw.get("platform"),
    )
