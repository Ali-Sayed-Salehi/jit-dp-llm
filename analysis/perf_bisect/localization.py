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
    elapsed_hours: float
    test_runs: int
    oracle_queries: int
    path_length: int
    candidate_revisions_tested: int
    decisions: list[OracleResult] = field(default_factory=list)
    final_decisions: list[OracleResult] = field(default_factory=list)
    retrigger_intervals: list[list[str]] = field(default_factory=list)
    revision_path: list[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Serialize the localization result into JSON-compatible primitives."""

        expected_by_revision = self._expected_decisions(
            self.final_decisions or self.decisions,
            self.culprit_revision,
            self.revision_path,
        )
        output = {
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
            "elapsed_hours": round(self.elapsed_hours, 4),
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
        output.update(self.extra)
        return output

    @staticmethod
    def _expected_decisions(
        decisions: list[OracleResult],
        culprit_revision: str | None,
        revision_path: list[str] | None = None,
    ) -> dict[str, OracleDecision]:
        """Return expected clean/bad decisions when the culprit is in the candidates."""

        if culprit_revision is None:
            return {}

        if revision_path is not None and culprit_revision in revision_path:
            culprit_idx = revision_path.index(culprit_revision)
            return {
                revision: (
                    OracleDecision.BAD
                    if idx >= culprit_idx
                    else OracleDecision.CLEAN
                )
                for idx, revision in enumerate(revision_path)
            }

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

    @staticmethod
    def _elapsed_hours(
        oracle: TestOracle,
        decisions: Sequence[OracleResult] = (),
    ) -> float:
        """Return wall-clock test elapsed time for this localization attempt."""

        executor = getattr(oracle, "executor", None)
        if executor is not None:
            return executor.now_minutes / 60.0
        if decisions:
            return max(decision.completed_at_minutes for decision in decisions) / 60.0
        return 0.0


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
                elapsed_hours=self._elapsed_hours(oracle),
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
        elapsed_hours = self._elapsed_hours(oracle, all_decisions)

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
            elapsed_hours=elapsed_hours,
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
                elapsed_hours=self._elapsed_hours(oracle),
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
        elapsed_hours = self._elapsed_hours(oracle, all_decisions)
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
            elapsed_hours=elapsed_hours,
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


class StandardMidpointMultisection(StandardMidpointBisection):
    """Find the first bad revision by testing equal interval cuts in parallel."""

    name = "StandardMidpointMultisection"

    def __init__(
        self,
        *,
        multisection_section_count: int = 4,
        multisection_retrigger_count: int = 0,
    ) -> None:
        """Configure interval fan-out and repeated boundary probes."""

        if multisection_section_count < 2:
            raise ValueError("multisection_section_count must be at least 2")
        if multisection_retrigger_count < 0:
            raise ValueError("multisection_retrigger_count must be non-negative")
        self.multisection_section_count = multisection_section_count
        self.multisection_retrigger_count = multisection_retrigger_count

    def localize(
        self,
        regression: Mapping[str, Any],
        *,
        revision_perf: RevisionPerfIndex,
        oracle: TestOracle,
    ) -> LocalizationResult:
        """Probe equal section boundaries until the candidate interval is adjacent."""

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
                elapsed_hours=self._elapsed_hours(oracle),
                test_runs=getattr(oracle, "executor", None).test_runs
                if hasattr(getattr(oracle, "executor", None), "test_runs")
                else 0,
                oracle_queries=getattr(oracle, "query_count", 0),
                path_length=0,
                candidate_revisions_tested=0,
                extra=self._settings_to_json(),
            )

        low_idx = 0
        high_idx = len(path) - 1
        all_decisions: list[OracleResult] = []
        final_decision_by_revision: dict[str, OracleResult] = {}
        retrigger_intervals: list[list[str]] = []
        found_revision = None
        undefined_reason = None

        while high_idx - low_idx > 1:
            boundary_indices = self._section_boundary_indices(
                low_idx=low_idx,
                high_idx=high_idx,
                section_count=self.multisection_section_count,
            )
            boundary_revisions = [path[idx] for idx in boundary_indices]
            decisions_by_revision: dict[str, list[OracleResult]] = {
                revision: [] for revision in boundary_revisions
            }

            boundary_decisions = oracle.classify_many(
                boundary_revisions,
                regression=regression,
                revision_path=path,
            )
            all_decisions.extend(boundary_decisions)
            for decision in boundary_decisions:
                decisions_by_revision[decision.revision].append(decision)

            if self.multisection_retrigger_count:
                retrigger_revisions = [
                    revision
                    for _ in range(self.multisection_retrigger_count)
                    for revision in boundary_revisions
                ]
                retrigger_intervals.extend(
                    list(boundary_revisions)
                    for _ in range(self.multisection_retrigger_count)
                )
                retrigger_decisions = oracle.classify_many(
                    retrigger_revisions,
                    regression=regression,
                    revision_path=path,
                )
                all_decisions.extend(retrigger_decisions)
                for decision in retrigger_decisions:
                    decisions_by_revision[decision.revision].append(decision)

            tied_revisions = [
                revision
                for revision in boundary_revisions
                if Backfill._has_clean_bad_tie(decisions_by_revision[revision])
            ]
            if tied_revisions:
                tie_break_decisions = oracle.classify_many(
                    tied_revisions,
                    regression=regression,
                    revision_path=path,
                )
                all_decisions.extend(tie_break_decisions)
                for decision in tie_break_decisions:
                    decisions_by_revision[decision.revision].append(decision)

            selected_by_idx: dict[int, OracleResult] = {}
            for boundary_idx, boundary_revision in zip(
                boundary_indices,
                boundary_revisions,
                strict=True,
            ):
                selected_decision, undefined_reason = self._select_midpoint_decision(
                    decisions_by_revision[boundary_revision],
                )
                if selected_decision is None:
                    break
                final_decision_by_revision[boundary_revision] = selected_decision
                selected_by_idx[boundary_idx] = selected_decision
            if undefined_reason is not None:
                break

            next_interval, undefined_reason = self._next_interval_from_boundaries(
                low_idx=low_idx,
                high_idx=high_idx,
                boundary_indices=boundary_indices,
                selected_by_idx=selected_by_idx,
            )
            if next_interval is None:
                break
            low_idx, high_idx = next_interval
        else:
            found_revision = path[high_idx]
            undefined_reason = self._undefined_reason_for_found_revision(
                found_revision=found_revision,
                path=path,
                culprit_revision=culprit_revision,
            )

        success = found_revision is not None and undefined_reason is None
        executor = getattr(oracle, "executor", None)
        elapsed_hours = self._elapsed_hours(oracle, all_decisions)
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
            elapsed_hours=elapsed_hours,
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
            revision_path=path,
            extra=self._settings_to_json(),
        )

    def _settings_to_json(self) -> dict[str, Any]:
        """Return fixed and tunable multisection settings for result serialization."""

        return {
            "multisection_section_count": self.multisection_section_count,
            "multisection_retrigger_count": self.multisection_retrigger_count,
        }

    @staticmethod
    def _section_boundary_indices(
        *,
        low_idx: int,
        high_idx: int,
        section_count: int,
    ) -> list[int]:
        """Return internal right-boundary indices for equal interval sections."""

        candidate_count = high_idx - low_idx
        effective_section_count = min(section_count, candidate_count)
        return [
            low_idx + (candidate_count * section_number) // effective_section_count
            for section_number in range(1, effective_section_count)
        ]

    @staticmethod
    def _next_interval_from_boundaries(
        *,
        low_idx: int,
        high_idx: int,
        boundary_indices: Sequence[int],
        selected_by_idx: Mapping[int, OracleResult],
    ) -> tuple[tuple[int, int] | None, str | None]:
        """Choose the adjacent clean-to-bad boundary pair for the next interval."""

        ordered_boundaries: list[tuple[int, OracleDecision]] = [
            (low_idx, OracleDecision.CLEAN)
        ]
        ordered_boundaries.extend(
            (idx, selected_by_idx[idx].decision) for idx in boundary_indices
        )
        ordered_boundaries.append((high_idx, OracleDecision.BAD))

        previous_decision = ordered_boundaries[0][1]
        for _, decision in ordered_boundaries[1:]:
            if (
                previous_decision is OracleDecision.BAD
                and decision is OracleDecision.CLEAN
            ):
                return None, "non_monotonic_multisection_decisions"
            previous_decision = decision

        for right_position, (_, decision) in enumerate(
            ordered_boundaries[1:],
            start=1,
        ):
            if decision is OracleDecision.BAD:
                next_low = ordered_boundaries[right_position - 1][0]
                next_high = ordered_boundaries[right_position][0]
                if next_low >= next_high:
                    return None, "non_shrinking_multisection_interval"
                return (next_low, next_high), None

        return None, "no_bad_revision_found"

    @staticmethod
    def _undefined_reason_for_found_revision(
        *,
        found_revision: str,
        path: list[str],
        culprit_revision: str | None,
    ) -> str | None:
        """Return why a completed multisection result is not successful."""

        if culprit_revision is None:
            return "missing_culprit_revision"
        if culprit_revision not in path[1:]:
            return "culprit_not_in_search_range"
        if found_revision != culprit_revision:
            return "multisection_found_is_not_culprit"
        return None


class ProbabilisticBisectionPosteriorMedianUniformPrior(CulpritLocalizer):
    """Probabilistic bisection using posterior-median probes and a uniform prior."""

    name = "ProbabilisticBisection_PosteriorMedian_UniformPrior"
    pba_batch_size = 1
    _POSTERIOR_TIE_EPSILON = 1e-12

    def __init__(
        self,
        *,
        pba_confidence_threshold: float = 0.9,
        pba_repeat_count: int = 1,
        pba_max_test_runs: int = 100,
    ) -> None:
        """Configure posterior confidence, repeated probes, and test-run budget."""

        if not 0.0 < pba_confidence_threshold <= 1.0:
            raise ValueError("pba_confidence_threshold must be in (0, 1]")
        if pba_repeat_count < 1:
            raise ValueError("pba_repeat_count must be at least 1")
        if pba_max_test_runs < 1:
            raise ValueError("pba_max_test_runs must be at least 1")
        self.pba_confidence_threshold = pba_confidence_threshold
        self.pba_repeat_count = pba_repeat_count
        self.pba_max_test_runs = pba_max_test_runs

    def localize(
        self,
        regression: Mapping[str, Any],
        *,
        revision_perf: RevisionPerfIndex,
        oracle: TestOracle,
    ) -> LocalizationResult:
        """Update a culprit posterior until one candidate is confidently selected."""

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
                elapsed_hours=self._elapsed_hours(oracle),
                test_runs=getattr(oracle, "executor", None).test_runs
                if hasattr(getattr(oracle, "executor", None), "test_runs")
                else 0,
                oracle_queries=getattr(oracle, "query_count", 0),
                path_length=0,
                candidate_revisions_tested=0,
                extra=self._settings_to_json(),
            )

        candidate_revisions = path[1:]
        posterior = [1.0 / len(candidate_revisions) for _ in candidate_revisions]
        all_decisions: list[OracleResult] = []
        posterior_trace: list[dict[str, Any]] = []
        found_revision = None
        undefined_reason = None

        while len(all_decisions) < self.pba_max_test_runs:
            best_idx, best_probability, tied_indices = self._posterior_map(posterior)
            if (
                best_probability >= self.pba_confidence_threshold
                and len(tied_indices) == 1
            ):
                found_revision = candidate_revisions[best_idx]
                undefined_reason = self._undefined_reason_for_found_revision(
                    found_revision=found_revision,
                    path=path,
                    culprit_revision=culprit_revision,
                )
                break

            probe_idx = self._posterior_median_index(posterior)
            probe_revision = candidate_revisions[probe_idx]
            remaining_budget = self.pba_max_test_runs - len(all_decisions)
            repeat_count = min(self.pba_repeat_count, remaining_budget)
            decisions = oracle.classify_many(
                [probe_revision] * repeat_count,
                regression=regression,
                revision_path=path,
            )
            all_decisions.extend(decisions)

            probe_accuracy = oracle.accuracy_for(
                probe_revision,
                regression=regression,
                revision_path=path,
            )
            self._validate_accuracy(probe_accuracy)
            known_decisions = [
                decision
                for decision in decisions
                if decision.decision is not OracleDecision.UNKNOWN
            ]
            for decision in known_decisions:
                posterior = self._updated_posterior(
                    posterior=posterior,
                    probe_idx=probe_idx,
                    decision=decision.decision,
                    oracle_accuracy=probe_accuracy,
                )
                if posterior is None:
                    undefined_reason = "posterior_degenerate"
                    break
            if undefined_reason is not None:
                break

            best_idx, best_probability, tied_indices = self._posterior_map(posterior)
            posterior_trace.append(
                {
                    "round": len(posterior_trace) + 1,
                    "probed_revision": probe_revision,
                    "probed_candidate_index": probe_idx,
                    "observations": [decision.decision.value for decision in decisions],
                    "known_observation_count": len(known_decisions),
                    "map_revision": candidate_revisions[best_idx],
                    "map_probability": round(best_probability, 12),
                    "map_tie_count": len(tied_indices),
                }
            )

        if found_revision is None and undefined_reason is None:
            best_idx, best_probability, tied_indices = self._posterior_map(posterior)
            found_revision = candidate_revisions[best_idx]
            if len(tied_indices) > 1:
                undefined_reason = "ambiguous_posterior_tie"
            elif best_probability < self.pba_confidence_threshold:
                undefined_reason = "posterior_confidence_below_threshold"
            else:
                undefined_reason = self._undefined_reason_for_found_revision(
                    found_revision=found_revision,
                    path=path,
                    culprit_revision=culprit_revision,
                )

        best_idx, best_probability, tied_indices = self._posterior_map(posterior)
        if undefined_reason is None:
            undefined_reason = self._undefined_reason_for_found_revision(
                found_revision=found_revision,
                path=path,
                culprit_revision=culprit_revision,
            )
        success = found_revision is not None and undefined_reason is None
        executor = getattr(oracle, "executor", None)
        elapsed_hours = self._elapsed_hours(oracle, all_decisions)
        final_decisions = self._latest_decisions_by_path_order(
            candidate_revisions,
            all_decisions,
        )
        tested_revision_count = len({decision.revision for decision in all_decisions})

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
            elapsed_hours=elapsed_hours,
            test_runs=(
                executor.test_runs
                if executor is not None
                else sum(decision.test_runs for decision in all_decisions)
            ),
            oracle_queries=getattr(oracle, "query_count", len(all_decisions)),
            path_length=len(path),
            candidate_revisions_tested=tested_revision_count,
            decisions=all_decisions,
            final_decisions=final_decisions,
            revision_path=path,
            extra={
                **self._settings_to_json(),
                "pba_prior": "uniform",
                "pba_query_strategy": "posterior_median",
                "pba_found_revision_probability": round(best_probability, 12),
                "pba_map_tie_count": len(tied_indices),
                "pba_map_tied_revisions": [
                    candidate_revisions[index] for index in tied_indices
                ],
                "pba_final_posterior": self._posterior_to_json(
                    candidate_revisions,
                    posterior,
                ),
                "pba_posterior_trace": posterior_trace,
            },
        )

    def _settings_to_json(self) -> dict[str, Any]:
        """Return fixed and tunable PBA settings for result serialization."""

        return {
            "pba_batch_size": self.pba_batch_size,
            "pba_confidence_threshold": self.pba_confidence_threshold,
            "pba_repeat_count": self.pba_repeat_count,
            "pba_max_test_runs": self.pba_max_test_runs,
        }

    @staticmethod
    def _validate_accuracy(accuracy: float) -> None:
        """Validate one oracle accuracy value before using it as a likelihood."""

        if not 0.0 <= accuracy <= 1.0:
            raise ValueError(f"oracle accuracy must be between 0 and 1: {accuracy!r}")

    @classmethod
    def _posterior_map(cls, posterior: Sequence[float]) -> tuple[int, float, list[int]]:
        """Return the MAP index, probability, and all tied MAP indices."""

        best_probability = max(posterior)
        tied_indices = [
            idx
            for idx, probability in enumerate(posterior)
            if abs(probability - best_probability) <= cls._POSTERIOR_TIE_EPSILON
        ]
        return tied_indices[0], best_probability, tied_indices

    @staticmethod
    def _posterior_median_index(posterior: Sequence[float]) -> int:
        """Return the first candidate where posterior CDF reaches one half."""

        cumulative_probability = 0.0
        for idx, probability in enumerate(posterior):
            cumulative_probability += probability
            if cumulative_probability >= 0.5:
                return idx
        return len(posterior) - 1

    @staticmethod
    def _updated_posterior(
        *,
        posterior: Sequence[float],
        probe_idx: int,
        decision: OracleDecision,
        oracle_accuracy: float,
    ) -> list[float] | None:
        """Apply one probabilistic-bisection likelihood update."""

        updated = []
        for candidate_idx, probability in enumerate(posterior):
            expected_bad = candidate_idx <= probe_idx
            if decision is OracleDecision.BAD:
                likelihood = oracle_accuracy if expected_bad else 1.0 - oracle_accuracy
            elif decision is OracleDecision.CLEAN:
                likelihood = oracle_accuracy if not expected_bad else 1.0 - oracle_accuracy
            else:
                likelihood = 1.0
            updated.append(probability * likelihood)

        total_probability = sum(updated)
        if total_probability <= 0.0:
            return None
        return [probability / total_probability for probability in updated]

    @staticmethod
    def _latest_decisions_by_path_order(
        candidate_revisions: Sequence[str],
        decisions: Sequence[OracleResult],
    ) -> list[OracleResult]:
        """Return the latest observed decision for each tested revision in path order."""

        latest_by_revision: dict[str, OracleResult] = {}
        for decision in decisions:
            previous = latest_by_revision.get(decision.revision)
            if (
                previous is None
                or decision.completed_at_minutes >= previous.completed_at_minutes
            ):
                latest_by_revision[decision.revision] = decision
        return [
            latest_by_revision[revision]
            for revision in candidate_revisions
            if revision in latest_by_revision
        ]

    @staticmethod
    def _posterior_to_json(
        candidate_revisions: Sequence[str],
        posterior: Sequence[float],
    ) -> list[dict[str, Any]]:
        """Serialize the final posterior distribution in candidate order."""

        return [
            {
                "revision": revision,
                "probability": round(probability, 12),
            }
            for revision, probability in zip(candidate_revisions, posterior, strict=True)
        ]

    @staticmethod
    def _undefined_reason_for_found_revision(
        *,
        found_revision: str | None,
        path: list[str],
        culprit_revision: str | None,
    ) -> str | None:
        """Return why a selected PBA revision is not a successful localization."""

        if found_revision is None:
            return "missing_found_revision"
        if culprit_revision is None:
            return "missing_culprit_revision"
        if culprit_revision not in path[1:]:
            return "culprit_not_in_search_range"
        if found_revision != culprit_revision:
            return "pba_found_is_not_culprit"
        return None


LOCALIZERS: dict[str, type[CulpritLocalizer]] = {
    Backfill.name: Backfill,
    BackfillWithRepeat.name: BackfillWithRepeat,
    ProbabilisticBisectionPosteriorMedianUniformPrior.name: (
        ProbabilisticBisectionPosteriorMedianUniformPrior
    ),
    StandardMidpointBisection.name: StandardMidpointBisection,
    StandardMidpointMultisection.name: StandardMidpointMultisection,
}
