"""Culprit localization algorithms for Mozilla performance bisect simulations."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

try:
    from .test_oracle import OracleDecision, OracleResult, RevisionPerfIndex, TestOracle
except ImportError:  # pragma: no cover - supports direct script execution.
    from test_oracle import OracleDecision, OracleResult, RevisionPerfIndex, TestOracle


@dataclass
class LocalizationResult:
    """Full result for running one localization algorithm on one regression."""

    localizer: str
    oracle: str
    regression_id: int | None
    alert_summary_id: int | None
    signature_id: int | None
    good_revision: str
    bad_revision: str
    culprit_revision: str | None
    found_revision: str | None
    success: bool
    undefined_reason: str | None
    trtc_hours: float | None
    test_runs: int
    oracle_queries: int
    path_length: int
    candidate_revisions_tested: int
    decisions: list[OracleResult] = field(default_factory=list)
    final_decisions: list[OracleResult] = field(default_factory=list)
    retrigger_intervals: list[list[str]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """Serialize the localization result into JSON-compatible primitives."""

        expected_by_revision = self._expected_decisions(
            self.final_decisions or self.decisions,
            self.culprit_revision,
        )
        return {
            "localizer": self.localizer,
            "oracle": self.oracle,
            "regression_id": self.regression_id,
            "alert_summary_id": self.alert_summary_id,
            "signature_id": self.signature_id,
            "good_revision": self.good_revision,
            "bad_revision": self.bad_revision,
            "culprit_revision": self.culprit_revision,
            "found_revision": self.found_revision,
            "success": self.success,
            "undefined_reason": self.undefined_reason,
            "trtc_hours": (
                round(self.trtc_hours, 4)
                if self.trtc_hours is not None
                else None
            ),
            "test_runs": self.test_runs,
            "oracle_queries": self.oracle_queries,
            "path_length": self.path_length,
            "candidate_revisions_tested": self.candidate_revisions_tested,
            "retrigger_count": len(self.retrigger_intervals),
            "retrigger_intervals": self.retrigger_intervals,
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
            "attempt": decision.attempt,
            "draw_index": decision.draw_index,
            "measurement_count": decision.measurement_count,
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

    @staticmethod
    def _signature_id(regression: Mapping[str, Any]) -> int | None:
        """Extract a signature id from either supported regression schema."""

        if "failing_sig" in regression:
            return int(regression["failing_sig"]["signature_id"])
        if "failing_sigs" in regression and regression["failing_sigs"]:
            signature_id, raw = next(iter(regression["failing_sigs"].items()))
            return int(raw.get("signature_id", signature_id))
        return None

    @staticmethod
    def _regression_id(regression: Mapping[str, Any]) -> int | None:
        """Extract the stable regression row id when available."""

        raw_id = regression.get("regression_id")
        if raw_id is None:
            return None
        return int(raw_id)


class Backfill(CulpritLocalizer):
    """Probe every revision after the known-good revision and find the first bad one."""

    name = "Backfill"

    def __init__(self, *, backfill_retrigger_count: int = 2) -> None:
        """Configure how often suspicious Backfill decisions are retriggered."""

        if backfill_retrigger_count < 0:
            raise ValueError("backfill_retrigger_count must be non-negative")
        self.backfill_retrigger_count = backfill_retrigger_count

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
        regression_id = self._regression_id(regression)
        signature_id = self._signature_id(regression)

        path = revision_perf.path_between(good_revision, bad_revision)
        if path is None or len(path) < 2:
            return LocalizationResult(
                localizer=self.name,
                oracle=oracle.name,
                regression_id=regression_id,
                alert_summary_id=regression.get("alert_summary_id"),
                signature_id=signature_id,
                good_revision=good_revision,
                bad_revision=bad_revision,
                culprit_revision=culprit_revision,
                found_revision=None,
                success=False,
                undefined_reason="no_revision_path",
                trtc_hours=None,
                test_runs=getattr(oracle, "executor", None).test_runs
                if hasattr(getattr(oracle, "executor", None), "test_runs")
                else 0,
                oracle_queries=getattr(oracle, "query_count", 0),
                path_length=0,
                candidate_revisions_tested=0,
            )

        revisions_to_probe = path[1:]
        decisions = oracle.classify_many(
            self._initial_probe_revisions(revisions_to_probe),
            regression=regression,
            revision_path=path,
        )
        all_decisions = list(decisions)
        final_decisions = self._final_decisions_from_draws(
            revisions=revisions_to_probe,
            decisions=all_decisions,
            oracle=oracle,
            regression=regression,
            revision_path=path,
        )

        found_revision, undefined_reason = self._evaluate_backfill(
            path=path,
            decisions=final_decisions,
            culprit_revision=culprit_revision,
        )
        retrigger_intervals = []
        for _ in range(self.backfill_retrigger_count):
            interval = self._retrigger_interval(
                path=path,
                decisions=final_decisions,
            )
            if interval is None:
                break

            retrigger_intervals.append(interval)
            all_decisions.extend(
                oracle.classify_many(
                    interval,
                    regression=regression,
                    revision_path=path,
                )
            )
            final_decisions = self._final_decisions_from_draws(
                revisions=revisions_to_probe,
                decisions=all_decisions,
                oracle=oracle,
                regression=regression,
                revision_path=path,
            )
            found_revision, undefined_reason = self._evaluate_backfill(
                path=path,
                decisions=final_decisions,
                culprit_revision=culprit_revision,
            )

        success = found_revision is not None and found_revision == culprit_revision
        executor = getattr(oracle, "executor", None)
        trtc_hours = (
            executor.now_minutes / 60.0
            if success and executor is not None
            else None
        )

        return LocalizationResult(
            localizer=self.name,
            oracle=oracle.name,
            regression_id=regression_id,
            alert_summary_id=regression.get("alert_summary_id"),
            signature_id=signature_id,
            good_revision=good_revision,
            bad_revision=bad_revision,
            culprit_revision=culprit_revision,
            found_revision=found_revision,
            success=success,
            undefined_reason=None if success else undefined_reason,
            trtc_hours=trtc_hours,
            test_runs=(
                executor.test_runs
                if executor is not None
                else sum(d.test_runs for d in all_decisions)
            ),
            oracle_queries=getattr(oracle, "query_count", len(all_decisions)),
            path_length=len(path),
            candidate_revisions_tested=len(revisions_to_probe),
            decisions=all_decisions,
            final_decisions=final_decisions,
            retrigger_intervals=retrigger_intervals,
        )

    def _initial_probe_revisions(self, revisions_to_probe: Sequence[str]) -> list[str]:
        """Return the revisions to include in Backfill's first submission batch."""

        return list(revisions_to_probe)

    @staticmethod
    def _evaluate_backfill(
        *,
        path: list[str],
        decisions: list[OracleResult],
        culprit_revision: str | None,
    ) -> tuple[str | None, str | None]:
        """Validate observed monotonicity and identify the first bad revision."""

        decision_by_revision = {
            decision.revision: decision.decision for decision in decisions
        }

        for revision in path[1:]:
            decision = decision_by_revision.get(revision)
            if decision is None:
                return None, "missing_oracle_decision"
            if decision is OracleDecision.UNKNOWN:
                return None, "missing_oracle_measurement"

        if Backfill._shortest_non_monotonic_interval(
            path=path,
            decisions=decisions,
        ) is not None:
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
        if culprit_revision is None:
            return first_bad, "missing_culprit_revision"
        if culprit_revision not in path[1:]:
            return first_bad, "culprit_not_in_search_range"
        if first_bad != culprit_revision:
            return first_bad, "first_bad_is_not_culprit"
        return first_bad, None

    @staticmethod
    def _shortest_non_monotonic_interval(
        *,
        path: list[str],
        decisions: list[OracleResult],
    ) -> list[str] | None:
        """Return the first adjacent bad-to-clean inversion in path order."""

        decision_by_revision = {
            decision.revision: decision.decision for decision in decisions
        }
        revisions_to_probe = path[1:]
        for left_revision, right_revision in zip(
            revisions_to_probe,
            revisions_to_probe[1:],
        ):
            if (
                decision_by_revision.get(left_revision) is OracleDecision.BAD
                and decision_by_revision.get(right_revision) is OracleDecision.CLEAN
            ):
                return [left_revision, right_revision]
        return None

    @staticmethod
    def _retrigger_interval(
        *,
        path: list[str],
        decisions: list[OracleResult],
    ) -> list[str] | None:
        """Return the next set of revisions that should be retriggered."""

        interval = Backfill._shortest_non_monotonic_interval(
            path=path,
            decisions=decisions,
        )
        if interval is not None:
            return interval

        decision_by_revision = {
            decision.revision: decision.decision for decision in decisions
        }
        revisions_to_probe = path[1:]
        revision_decisions = [
            decision_by_revision.get(revision) for revision in revisions_to_probe
        ]
        if (
            revision_decisions
            and all(decision is OracleDecision.CLEAN for decision in revision_decisions)
        ):
            return list(revisions_to_probe)
        return None

    def _final_decisions_from_draws(
        self,
        *,
        revisions: Sequence[str],
        decisions: list[OracleResult],
        oracle: TestOracle,
        regression: Mapping[str, Any],
        revision_path: Sequence[str],
    ) -> list[OracleResult]:
        """Select final decisions, probing once more for tied clean/bad votes."""

        decisions_by_revision: dict[str, list[OracleResult]] = {
            revision: [] for revision in revisions
        }
        for decision in decisions:
            if decision.revision in decisions_by_revision:
                decisions_by_revision[decision.revision].append(decision)

        tied_revisions = [
            revision
            for revision in revisions
            if self._has_clean_bad_tie(decisions_by_revision[revision])
        ]
        if tied_revisions:
            tie_break_decisions = oracle.classify_many(
                tied_revisions,
                regression=regression,
                revision_path=revision_path,
            )
            decisions.extend(tie_break_decisions)
            for decision in tie_break_decisions:
                if decision.revision in decisions_by_revision:
                    decisions_by_revision[decision.revision].append(decision)

        return [
            self._select_revision_decision(revision_decisions)
            for revision in revisions
            if (revision_decisions := decisions_by_revision[revision])
        ]

    @staticmethod
    def _has_clean_bad_tie(decisions: list[OracleResult]) -> bool:
        """Return whether known clean/bad votes are tied and need one more probe."""

        clean_count, bad_count = Backfill._clean_bad_counts(decisions)
        return clean_count > 0 and clean_count == bad_count

    @staticmethod
    def _clean_bad_counts(decisions: list[OracleResult]) -> tuple[int, int]:
        """Count known clean and bad votes, ignoring unknown oracle results."""

        clean_count = sum(
            decision.decision is OracleDecision.CLEAN for decision in decisions
        )
        bad_count = sum(
            decision.decision is OracleDecision.BAD for decision in decisions
        )
        return clean_count, bad_count

    @staticmethod
    def _select_revision_decision(
        decisions: list[OracleResult],
    ) -> OracleResult:
        """Choose one oracle draw, using majority decision after extra tie probes."""

        if len(decisions) == 1:
            return decisions[0]

        known_decisions = [
            decision
            for decision in decisions
            if decision.decision is not OracleDecision.UNKNOWN
        ]
        vote_pool = known_decisions or decisions
        clean_count, bad_count = Backfill._clean_bad_counts(known_decisions)
        draw_count = sum(decision.measurement_count for decision in decisions)

        if bad_count > clean_count:
            selected_decision = OracleDecision.BAD
        elif clean_count > bad_count:
            selected_decision = OracleDecision.CLEAN
        else:
            latest_draw = max(
                decisions,
                key=lambda decision: decision.completed_at_minutes,
            )
            return replace(
                latest_draw,
                decision=OracleDecision.UNKNOWN,
                measurement_count=draw_count,
                value_source="ambiguous_oracle_decision",
                value_source_revision=None,
            )

        selected_draw = max(
            (
                decision
                for decision in vote_pool
                if decision.decision is selected_decision
            ),
            key=lambda decision: decision.completed_at_minutes,
        )

        return replace(
            selected_draw,
            measurement_count=draw_count,
        )


class BackfillWithRepeat(Backfill):
    """Backfill with repeated initial test attempts for every candidate revision."""

    name = "BackfillWithRepeat"

    def __init__(
        self,
        *,
        backfill_retrigger_count: int = 2,
        probe_repeat_count: int = 1,
    ) -> None:
        """Configure repeated first-batch probes and suspicious-set retriggers."""

        super().__init__(backfill_retrigger_count=backfill_retrigger_count)
        if probe_repeat_count < 1:
            raise ValueError("probe_repeat_count must be at least 1")
        self.probe_repeat_count = probe_repeat_count

    def _initial_probe_revisions(self, revisions_to_probe: Sequence[str]) -> list[str]:
        """Repeat every candidate revision in the first submitted probe batch."""

        return [
            revision
            for _ in range(self.probe_repeat_count)
            for revision in revisions_to_probe
        ]


class StandardMidpointBisection(CulpritLocalizer):
    """Find the first bad revision using standard midpoint bisection."""

    name = "StandardMidpointBisection"

    def __init__(self, *, midpoint_retrigger_count: int = 0) -> None:
        """Configure how often each midpoint decision should be retriggered."""

        if midpoint_retrigger_count < 0:
            raise ValueError("midpoint_retrigger_count must be non-negative")
        self.midpoint_retrigger_count = midpoint_retrigger_count

    def localize(
        self,
        regression: Mapping[str, Any],
        *,
        revision_perf: RevisionPerfIndex,
        oracle: TestOracle,
    ) -> LocalizationResult:
        """Probe midpoint revisions until the candidate interval is adjacent."""

        good_revision = str(regression["good_revision"])
        bad_revision = str(regression["bad_revision"])
        culprit_revision = regression.get("culprit_revision")
        culprit_revision = str(culprit_revision) if culprit_revision is not None else None
        regression_id = self._regression_id(regression)
        signature_id = self._signature_id(regression)

        path = revision_perf.path_between(good_revision, bad_revision)
        if path is None or len(path) < 2:
            return LocalizationResult(
                localizer=self.name,
                oracle=oracle.name,
                regression_id=regression_id,
                alert_summary_id=regression.get("alert_summary_id"),
                signature_id=signature_id,
                good_revision=good_revision,
                bad_revision=bad_revision,
                culprit_revision=culprit_revision,
                found_revision=None,
                success=False,
                undefined_reason="no_revision_path",
                trtc_hours=None,
                test_runs=getattr(oracle, "executor", None).test_runs
                if hasattr(getattr(oracle, "executor", None), "test_runs")
                else 0,
                oracle_queries=getattr(oracle, "query_count", 0),
                path_length=0,
                candidate_revisions_tested=0,
            )

        low_idx = 0
        high_idx = len(path) - 1
        all_decisions: list[OracleResult] = []
        final_decision_by_revision: dict[str, OracleResult] = {}
        retrigger_intervals: list[list[str]] = []
        found_revision = None
        undefined_reason = None

        while high_idx - low_idx > 1:
            midpoint_idx = (low_idx + high_idx) // 2
            midpoint_revision = path[midpoint_idx]
            midpoint_decisions = oracle.classify_many(
                [midpoint_revision],
                regression=regression,
                revision_path=path,
            )
            all_decisions.extend(midpoint_decisions)

            if self.midpoint_retrigger_count:
                retrigger_revisions = [
                    midpoint_revision,
                ] * self.midpoint_retrigger_count
                retrigger_intervals.extend(
                    [revision] for revision in retrigger_revisions
                )
                retrigger_decisions = oracle.classify_many(
                    retrigger_revisions,
                    regression=regression,
                    revision_path=path,
                )
                midpoint_decisions.extend(retrigger_decisions)
                all_decisions.extend(retrigger_decisions)

            if Backfill._has_clean_bad_tie(midpoint_decisions):
                tie_break_decisions = oracle.classify_many(
                    [midpoint_revision],
                    regression=regression,
                    revision_path=path,
                )
                midpoint_decisions.extend(tie_break_decisions)
                all_decisions.extend(tie_break_decisions)

            selected_decision, undefined_reason = self._select_midpoint_decision(
                midpoint_decisions,
            )
            if selected_decision is None:
                break

            final_decision_by_revision[midpoint_revision] = selected_decision
            if selected_decision.decision is OracleDecision.CLEAN:
                low_idx = midpoint_idx
            else:
                high_idx = midpoint_idx
        else:
            found_revision = path[high_idx]
            undefined_reason = self._undefined_reason_for_found_revision(
                found_revision=found_revision,
                path=path,
                culprit_revision=culprit_revision,
            )

        success = found_revision is not None and undefined_reason is None
        executor = getattr(oracle, "executor", None)
        trtc_hours = (
            executor.now_minutes / 60.0
            if success and executor is not None
            else None
        )
        final_decisions = [
            final_decision_by_revision[revision]
            for revision in path[1:]
            if revision in final_decision_by_revision
        ]

        return LocalizationResult(
            localizer=self.name,
            oracle=oracle.name,
            regression_id=regression_id,
            alert_summary_id=regression.get("alert_summary_id"),
            signature_id=signature_id,
            good_revision=good_revision,
            bad_revision=bad_revision,
            culprit_revision=culprit_revision,
            found_revision=found_revision,
            success=success,
            undefined_reason=None if success else undefined_reason,
            trtc_hours=trtc_hours,
            test_runs=(
                executor.test_runs
                if executor is not None
                else sum(d.test_runs for d in all_decisions)
            ),
            oracle_queries=getattr(oracle, "query_count", len(all_decisions)),
            path_length=len(path),
            candidate_revisions_tested=len(final_decisions),
            decisions=all_decisions,
            final_decisions=final_decisions,
            retrigger_intervals=retrigger_intervals,
        )

    @staticmethod
    def _select_midpoint_decision(
        decisions: list[OracleResult],
    ) -> tuple[OracleResult | None, str | None]:
        """Choose a midpoint decision by majority vote over known verdicts."""

        known_decisions = [
            decision
            for decision in decisions
            if decision.decision is not OracleDecision.UNKNOWN
        ]
        if not known_decisions:
            return None, "missing_oracle_measurement"

        clean_count = sum(
            decision.decision is OracleDecision.CLEAN for decision in known_decisions
        )
        bad_count = sum(
            decision.decision is OracleDecision.BAD for decision in known_decisions
        )
        if clean_count == bad_count:
            return None, "ambiguous_midpoint_decision"

        selected_verdict = (
            OracleDecision.BAD if bad_count > clean_count else OracleDecision.CLEAN
        )
        selected_draw = max(
            (
                decision
                for decision in known_decisions
                if decision.decision is selected_verdict
            ),
            key=lambda decision: decision.completed_at_minutes,
        )
        draw_count = sum(decision.measurement_count for decision in decisions)
        return replace(selected_draw, measurement_count=draw_count), None

    @staticmethod
    def _undefined_reason_for_found_revision(
        *,
        found_revision: str,
        path: list[str],
        culprit_revision: str | None,
    ) -> str | None:
        """Return why a completed bisection result is not a successful localization."""

        if culprit_revision is None:
            return "missing_culprit_revision"
        if culprit_revision not in path[1:]:
            return "culprit_not_in_search_range"
        if found_revision != culprit_revision:
            return "bisect_found_is_not_culprit"
        return None


LOCALIZERS: dict[str, type[CulpritLocalizer]] = {
    Backfill.name: Backfill,
    BackfillWithRepeat.name: BackfillWithRepeat,
    StandardMidpointBisection.name: StandardMidpointBisection,
}
