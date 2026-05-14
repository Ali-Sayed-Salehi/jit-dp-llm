"""Culprit localization algorithms for Mozilla performance bisect simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

try:
    from .test_oracle import OracleDecision, OracleResult, RevisionPerfIndex, TestOracle
except ImportError:  # pragma: no cover - supports direct script execution.
    from test_oracle import OracleDecision, OracleResult, RevisionPerfIndex, TestOracle


@dataclass
class LocalizationResult:
    """Full result for running one localization algorithm on one regression."""

    localizer: str
    oracle: str
    alert_summary_id: int | None
    signature_id: int | None
    good_revision: str
    bad_revision: str
    culprit_revision: str | None
    found_revision: str | None
    success: bool
    undefined_reason: str | None
    trtc_minutes: float | None
    test_runs: int
    oracle_queries: int
    path_length: int
    candidate_revisions_tested: int
    decisions: list[OracleResult] = field(default_factory=list)
    final_decisions: list[OracleResult] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """Serialize the localization result into JSON-compatible primitives."""

        expected_by_revision = self._expected_decisions(
            self.final_decisions or self.decisions,
            self.culprit_revision,
        )
        return {
            "localizer": self.localizer,
            "oracle": self.oracle,
            "alert_summary_id": self.alert_summary_id,
            "signature_id": self.signature_id,
            "good_revision": self.good_revision,
            "bad_revision": self.bad_revision,
            "culprit_revision": self.culprit_revision,
            "found_revision": self.found_revision,
            "success": self.success,
            "undefined_reason": self.undefined_reason,
            "trtc_minutes": self.trtc_minutes,
            "test_runs": self.test_runs,
            "oracle_queries": self.oracle_queries,
            "path_length": self.path_length,
            "candidate_revisions_tested": self.candidate_revisions_tested,
            "decisions": [
                self._decision_to_json(
                    decision,
                    expected_by_revision.get(decision.revision),
                )
                for decision in self.decisions
            ],
            "final_decisions": [
                self._decision_to_json(
                    decision,
                    expected_by_revision.get(decision.revision),
                )
                for decision in self.final_decisions
            ],
        }

    @staticmethod
    def _expected_decisions(
        decisions: list[OracleResult],
        culprit_revision: str | None,
    ) -> dict[str, OracleDecision]:
        """Return expected clean/bad decisions when the culprit is in the candidates."""

        if culprit_revision is None:
            return {}

        revisions = [decision.revision for decision in decisions]
        if culprit_revision not in revisions:
            return {}

        culprit_idx = revisions.index(culprit_revision)
        return {
            revision: OracleDecision.BAD if idx >= culprit_idx else OracleDecision.CLEAN
            for idx, revision in enumerate(revisions)
        }

    @staticmethod
    def _decision_to_json(
        decision: OracleResult,
        expected_decision: OracleDecision | None,
    ) -> dict[str, Any]:
        """Serialize one oracle decision into JSON-compatible primitives."""

        return {
            "revision": decision.revision,
            "decision": decision.decision.value,
            "expected_decision": (
                expected_decision.value if expected_decision is not None else None
            ),
            "decision_correct": (
                decision.decision is expected_decision
                if expected_decision is not None
                else None
            ),
            "value": decision.value,
            "baseline": decision.baseline,
            "attempt": decision.attempt,
            "draw_index": decision.draw_index,
            "replicate_count": decision.replicate_count,
            "test_runs": decision.test_runs,
            "submitted_at_minutes": decision.submitted_at_minutes,
            "completed_at_minutes": decision.completed_at_minutes,
            "value_source": decision.value_source,
            "value_source_revision": decision.value_source_revision,
        }


class CulpritLocalizer:
    """Base class for culprit localization algorithms."""

    name = "CulpritLocalizer"

    def localize(
        self,
        regression: Mapping[str, Any],
        *,
        revision_perf: RevisionPerfIndex,
        oracle: TestOracle,
    ) -> LocalizationResult:
        """Run the localization algorithm for one regression row."""

        raise NotImplementedError


class Backfill(CulpritLocalizer):
    """Probe every revision after the known-good revision and find the first bad one."""

    name = "Backfill"

    def localize(
        self,
        regression: Mapping[str, Any],
        *,
        revision_perf: RevisionPerfIndex,
        oracle: TestOracle,
    ) -> LocalizationResult:
        """Probe every candidate revision and return the first monotonic bad revision."""

        good_revision = str(regression["good_revision"])
        bad_revision = str(regression["bad_revision"])
        culprit_revision = regression.get("culprit_revision")
        culprit_revision = str(culprit_revision) if culprit_revision is not None else None
        signature_id = self._signature_id(regression)

        path = revision_perf.path_between(good_revision, bad_revision)
        if path is None or len(path) < 2:
            return LocalizationResult(
                localizer=self.name,
                oracle=oracle.name,
                alert_summary_id=regression.get("alert_summary_id"),
                signature_id=signature_id,
                good_revision=good_revision,
                bad_revision=bad_revision,
                culprit_revision=culprit_revision,
                found_revision=None,
                success=False,
                undefined_reason="no_revision_path",
                trtc_minutes=None,
                test_runs=getattr(oracle, "executor", None).test_runs
                if hasattr(getattr(oracle, "executor", None), "test_runs")
                else 0,
                oracle_queries=getattr(oracle, "query_count", 0),
                path_length=0,
                candidate_revisions_tested=0,
            )

        revisions_to_probe = path[1:]
        decisions = oracle.classify_many(
            revisions_to_probe,
            regression=regression,
            revision_path=path,
        )
        all_decisions = list(decisions)
        final_decisions = decisions

        found_revision, undefined_reason = self._evaluate_backfill(
            path=path,
            decisions=final_decisions,
            culprit_revision=culprit_revision,
        )
        success = found_revision is not None and found_revision == culprit_revision
        executor = getattr(oracle, "executor", None)
        trtc_minutes = executor.now_minutes if success and executor is not None else None

        return LocalizationResult(
            localizer=self.name,
            oracle=oracle.name,
            alert_summary_id=regression.get("alert_summary_id"),
            signature_id=signature_id,
            good_revision=good_revision,
            bad_revision=bad_revision,
            culprit_revision=culprit_revision,
            found_revision=found_revision,
            success=success,
            undefined_reason=None if success else undefined_reason,
            trtc_minutes=trtc_minutes,
            test_runs=(
                executor.test_runs
                if executor is not None
                else sum(d.test_runs for d in all_decisions)
            ),
            oracle_queries=getattr(oracle, "query_count", len(decisions)),
            path_length=len(path),
            candidate_revisions_tested=len(revisions_to_probe),
            decisions=all_decisions,
            final_decisions=final_decisions,
        )

    @staticmethod
    def _evaluate_backfill(
        *,
        path: list[str],
        decisions: list[OracleResult],
        culprit_revision: str | None,
    ) -> tuple[str | None, str | None]:
        """Validate monotonic decisions and identify the first bad revision."""

        if culprit_revision is None:
            return None, "missing_culprit_revision"
        if culprit_revision not in path[1:]:
            return None, "culprit_not_in_search_range"

        decision_by_revision = {decision.revision: decision.decision for decision in decisions}
        culprit_idx = path.index(culprit_revision)

        for idx, revision in enumerate(path[1:], start=1):
            decision = decision_by_revision.get(revision)
            expected = OracleDecision.BAD if idx >= culprit_idx else OracleDecision.CLEAN
            if decision is None:
                return None, "missing_oracle_decision"
            if decision is not expected:
                return None, "non_monotonic_oracle_decisions"

        first_bad = next(
            (
                revision
                for revision in path[1:]
                if decision_by_revision.get(revision) is OracleDecision.BAD
            ),
            None,
        )
        if first_bad is None:
            return None, "no_bad_revision_found"
        if first_bad != culprit_revision:
            return None, "first_bad_is_not_culprit"
        return first_bad, None

    @staticmethod
    def _signature_id(regression: Mapping[str, Any]) -> int | None:
        """Extract a signature id from either supported regression schema."""

        if "failing_sig" in regression:
            return int(regression["failing_sig"]["signature_id"])
        if "failing_sigs" in regression and regression["failing_sigs"]:
            signature_id, raw = next(iter(regression["failing_sigs"].items()))
            return int(raw.get("signature_id", signature_id))
        return None


LOCALIZERS: dict[str, type[CulpritLocalizer]] = {
    Backfill.name: Backfill,
}
