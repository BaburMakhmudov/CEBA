"""Deterministic validation utilities for CEBA Long-Term Memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from ceba.memory.long_term.schema import (
    ConsolidatedExperience,
    ConsolidationStatus,
    EligibilityDecision,
    EligibilityStatus,
    LongTermMemorySnapshot,
    LongTermTrace,
    LongTermTraceStatus,
    ProvenanceCheck,
    ProvenanceStatus,
    ReplayRecord,
    ReplayStatus,
    ensure_non_empty_string,
)


__all__ = [
    "LongTermValidationIssue",
    "LongTermValidationReport",
    "LongTermValidator",
    "validate_long_term_snapshot",
    "validate_long_term_components",
]


_SEVERITIES = ("INFO", "WARNING", "ERROR")


@dataclass(frozen=True)
class LongTermValidationIssue:
    """Immutable validation issue descriptor."""

    code: str
    message: str
    severity: str
    object_id: str | None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        code = ensure_non_empty_string(self.code, "code")
        message = ensure_non_empty_string(self.message, "message")
        severity = ensure_non_empty_string(self.severity, "severity")
        if severity not in _SEVERITIES:
            raise ValueError("severity must be INFO, WARNING, or ERROR")
        object_id = (
            None
            if self.object_id is None
            else ensure_non_empty_string(self.object_id, "object_id")
        )
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "object_id": self.object_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LongTermValidationReport:
    """Immutable validation report."""

    valid: bool
    issues: tuple[LongTermValidationIssue, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be bool")
        issues = _ensure_issues(self.issues)
        has_error = any(issue.severity == "ERROR" for issue in issues)
        if self.valid and has_error:
            raise ValueError("valid must be False when ERROR issues exist")
        object.__setattr__(self, "issues", issues)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "valid": self.valid,
            "issues": tuple(issue.to_dict() for issue in self.issues),
        }

    def error_count(self) -> int:
        """Return error issue count."""

        return sum(1 for issue in self.issues if issue.severity == "ERROR")

    def warning_count(self) -> int:
        """Return warning issue count."""

        return sum(1 for issue in self.issues if issue.severity == "WARNING")

    def info_count(self) -> int:
        """Return info issue count."""

        return sum(1 for issue in self.issues if issue.severity == "INFO")


class LongTermValidator:
    """Deterministic validator for LTM runtime structures."""

    def __init__(self) -> None:
        pass

    def validate_components(
        self,
        *,
        traces: tuple[LongTermTrace, ...] | list[LongTermTrace] = (),
        experiences: (
            tuple[ConsolidatedExperience, ...]
            | list[ConsolidatedExperience]
        ) = (),
        provenance_checks: (
            tuple[ProvenanceCheck, ...] | list[ProvenanceCheck]
        ) = (),
        replay_records: tuple[ReplayRecord, ...] | list[ReplayRecord] = (),
        eligibility_decisions: (
            tuple[EligibilityDecision, ...] | list[EligibilityDecision]
        ) = (),
    ) -> LongTermValidationReport:
        """Validate long-term memory components."""

        trace_values = _ensure_traces(traces)
        experience_values = _ensure_experiences(experiences)
        provenance_values = _ensure_provenance_checks(provenance_checks)
        replay_values = _ensure_replay_records(replay_records)
        decision_values = _ensure_eligibility_decisions(
            eligibility_decisions,
        )
        issues: list[LongTermValidationIssue] = []
        issues.extend(_collection_info_issues(
            traces=trace_values,
            experiences=experience_values,
            provenance_checks=provenance_values,
            replay_records=replay_values,
            eligibility_decisions=decision_values,
        ))
        issues.extend(_duplicate_issues(trace_values, "trace_id"))
        issues.extend(_duplicate_issues(experience_values, "experience_id"))
        issues.extend(_duplicate_issues(provenance_values, "check_id"))
        issues.extend(_duplicate_issues(replay_values, "replay_id"))
        issues.extend(_duplicate_issues(decision_values, "decision_id"))
        trace_ids = {trace.trace_id for trace in trace_values}
        experience_ids = {
            experience.experience_id for experience in experience_values
        }
        trace_by_id = {trace.trace_id: trace for trace in trace_values}
        issues.extend(_reference_issues(
            experiences=experience_values,
            provenance_checks=provenance_values,
            replay_records=replay_values,
            eligibility_decisions=decision_values,
            trace_ids=trace_ids,
            experience_ids=experience_ids,
        ))
        issues.extend(_trace_warning_issues(trace_values))
        issues.extend(_experience_warning_issues(experience_values))
        issues.extend(_provenance_warning_issues(provenance_values))
        issues.extend(_replay_warning_issues(replay_values))
        issues.extend(_eligibility_warning_issues(
            decisions=decision_values,
            trace_by_id=trace_by_id,
        ))
        valid = not any(issue.severity == "ERROR" for issue in issues)
        return LongTermValidationReport(valid=valid, issues=tuple(issues))

    def validate_snapshot(
        self,
        *,
        snapshot: LongTermMemorySnapshot,
    ) -> LongTermValidationReport:
        """Validate a long-term memory snapshot."""

        if not isinstance(snapshot, LongTermMemorySnapshot):
            raise TypeError("snapshot must be LongTermMemorySnapshot")
        return self.validate_components(
            traces=snapshot.traces,
            experiences=snapshot.experiences,
            provenance_checks=snapshot.provenance_checks,
            replay_records=snapshot.replay_records,
            eligibility_decisions=snapshot.eligibility_decisions,
        )


def validate_long_term_snapshot(
    *,
    snapshot: LongTermMemorySnapshot,
) -> LongTermValidationReport:
    """Validate a long-term memory snapshot."""

    validator = LongTermValidator()
    return validator.validate_snapshot(snapshot=snapshot)


def validate_long_term_components(
    *,
    traces: tuple[LongTermTrace, ...] | list[LongTermTrace] = (),
    experiences: (
        tuple[ConsolidatedExperience, ...] | list[ConsolidatedExperience]
    ) = (),
    provenance_checks: tuple[ProvenanceCheck, ...] | list[ProvenanceCheck] = (),
    replay_records: tuple[ReplayRecord, ...] | list[ReplayRecord] = (),
    eligibility_decisions: (
        tuple[EligibilityDecision, ...] | list[EligibilityDecision]
    ) = (),
) -> LongTermValidationReport:
    """Validate long-term memory components."""

    validator = LongTermValidator()
    return validator.validate_components(
        traces=traces,
        experiences=experiences,
        provenance_checks=provenance_checks,
        replay_records=replay_records,
        eligibility_decisions=eligibility_decisions,
    )


def _collection_info_issues(
    *,
    traces: tuple[LongTermTrace, ...],
    experiences: tuple[ConsolidatedExperience, ...],
    provenance_checks: tuple[ProvenanceCheck, ...],
    replay_records: tuple[ReplayRecord, ...],
    eligibility_decisions: tuple[EligibilityDecision, ...],
) -> tuple[LongTermValidationIssue, ...]:
    issues: list[LongTermValidationIssue] = []
    if not traces:
        issues.append(_issue("empty_traces", "empty traces collection", "INFO"))
    if not experiences:
        issues.append(_issue(
            "empty_experiences",
            "empty experiences collection",
            "INFO",
        ))
    if not provenance_checks:
        issues.append(_issue(
            "empty_provenance_checks",
            "empty provenance_checks collection",
            "INFO",
        ))
    if not replay_records:
        issues.append(_issue(
            "empty_replay_records",
            "empty replay_records collection",
            "INFO",
        ))
    if not eligibility_decisions:
        issues.append(_issue(
            "empty_eligibility_decisions",
            "empty eligibility_decisions collection",
            "INFO",
        ))
    return tuple(issues)


def _reference_issues(
    *,
    experiences: tuple[ConsolidatedExperience, ...],
    provenance_checks: tuple[ProvenanceCheck, ...],
    replay_records: tuple[ReplayRecord, ...],
    eligibility_decisions: tuple[EligibilityDecision, ...],
    trace_ids: set[UUID],
    experience_ids: set[UUID],
) -> tuple[LongTermValidationIssue, ...]:
    issues: list[LongTermValidationIssue] = []
    for experience in experiences:
        for trace_id in experience.trace_ids:
            if trace_id not in trace_ids:
                issues.append(_issue(
                    "experience_trace_missing",
                    "experience trace_id missing from traces",
                    "ERROR",
                    str(experience.experience_id),
                    {"trace_id": str(trace_id)},
                ))
    for check in provenance_checks:
        if check.trace_id not in trace_ids:
            issues.append(_issue(
                "provenance_trace_missing",
                "provenance_check trace_id missing from traces",
                "ERROR",
                str(check.check_id),
                {"trace_id": str(check.trace_id)},
            ))
    for replay in replay_records:
        if replay.trace_id is not None and replay.trace_id not in trace_ids:
            issues.append(_issue(
                "replay_trace_missing",
                "replay_record trace_id missing from traces",
                "ERROR",
                str(replay.replay_id),
                {"trace_id": str(replay.trace_id)},
            ))
        if (
            replay.experience_id is not None
            and replay.experience_id not in experience_ids
        ):
            issues.append(_issue(
                "replay_experience_missing",
                "replay_record experience_id missing from experiences",
                "ERROR",
                str(replay.replay_id),
                {"experience_id": str(replay.experience_id)},
            ))
    for decision in eligibility_decisions:
        if decision.trace_id not in trace_ids:
            issues.append(_issue(
                "eligibility_trace_missing",
                "eligibility_decision trace_id missing from traces",
                "ERROR",
                str(decision.decision_id),
                {"trace_id": str(decision.trace_id)},
            ))
    return tuple(issues)


def _trace_warning_issues(
    traces: tuple[LongTermTrace, ...],
) -> tuple[LongTermValidationIssue, ...]:
    issues: list[LongTermValidationIssue] = []
    for trace in traces:
        object_id = str(trace.trace_id)
        if trace.status is LongTermTraceStatus.INVALID:
            issues.append(_issue(
                "invalid_trace_status",
                "LongTermTrace.status == INVALID",
                "WARNING",
                object_id,
            ))
        if (
            trace.status is LongTermTraceStatus.ACTIVE
            and trace.superseded_at is not None
        ):
            issues.append(_issue(
                "active_trace_superseded_at",
                "active trace with superseded_at present",
                "WARNING",
                object_id,
            ))
        if (
            trace.status is LongTermTraceStatus.SUPERSEDED
            and trace.superseded_at is None
        ):
            issues.append(_issue(
                "superseded_trace_missing_superseded_at",
                "superseded trace without superseded_at",
                "WARNING",
                object_id,
            ))
        if (
            trace.status is LongTermTraceStatus.REPLAYABLE
            and trace.replay_count == 0
        ):
            issues.append(_issue(
                "replayable_trace_without_replay_count",
                "replayable trace with replay_count == 0",
                "WARNING",
                object_id,
            ))
    return tuple(issues)


def _experience_warning_issues(
    experiences: tuple[ConsolidatedExperience, ...],
) -> tuple[LongTermValidationIssue, ...]:
    issues: list[LongTermValidationIssue] = []
    for experience in experiences:
        object_id = str(experience.experience_id)
        if experience.status is ConsolidationStatus.INVALID:
            issues.append(_issue(
                "invalid_experience_status",
                "ConsolidatedExperience.status == INVALID",
                "WARNING",
                object_id,
            ))
        if not experience.trace_ids:
            issues.append(_issue(
                "empty_experience_trace_ids",
                "consolidated experience with empty trace_ids",
                "WARNING",
                object_id,
            ))
    return tuple(issues)


def _provenance_warning_issues(
    checks: tuple[ProvenanceCheck, ...],
) -> tuple[LongTermValidationIssue, ...]:
    issues: list[LongTermValidationIssue] = []
    for check in checks:
        object_id = str(check.check_id)
        if check.status is ProvenanceStatus.INVALID:
            issues.append(_issue(
                "invalid_provenance_status",
                "ProvenanceCheck.status == INVALID",
                "WARNING",
                object_id,
            ))
        if check.status is ProvenanceStatus.MISSING and check.source_ids:
            issues.append(_issue(
                "missing_provenance_with_sources",
                "provenance missing status with non-empty source_ids",
                "WARNING",
                object_id,
            ))
    return tuple(issues)


def _replay_warning_issues(
    records: tuple[ReplayRecord, ...],
) -> tuple[LongTermValidationIssue, ...]:
    issues: list[LongTermValidationIssue] = []
    for record in records:
        object_id = str(record.replay_id)
        if record.status is ReplayStatus.INVALID:
            issues.append(_issue(
                "invalid_replay_status",
                "ReplayRecord.status == INVALID",
                "WARNING",
                object_id,
            ))
        if record.status is ReplayStatus.USED and record.used_at is None:
            issues.append(_issue(
                "used_replay_without_used_at",
                "replay used status without used_at",
                "WARNING",
                object_id,
            ))
    return tuple(issues)


def _eligibility_warning_issues(
    *,
    decisions: tuple[EligibilityDecision, ...],
    trace_by_id: dict[UUID, LongTermTrace],
) -> tuple[LongTermValidationIssue, ...]:
    issues: list[LongTermValidationIssue] = []
    blocked_statuses = (
        LongTermTraceStatus.INVALID,
        LongTermTraceStatus.BLOCKED,
        LongTermTraceStatus.SUPERSEDED,
    )
    for decision in decisions:
        object_id = str(decision.decision_id)
        if decision.status is EligibilityStatus.INVALID:
            issues.append(_issue(
                "invalid_eligibility_status",
                "EligibilityDecision.status == INVALID",
                "WARNING",
                object_id,
            ))
        trace = trace_by_id.get(decision.trace_id)
        if (
            decision.status is EligibilityStatus.ELIGIBLE
            and trace is not None
            and trace.status in blocked_statuses
        ):
            issues.append(_issue(
                "eligible_blocked_trace",
                "eligibility eligible for invalid/blocked/superseded trace",
                "WARNING",
                object_id,
                {"trace_id": str(decision.trace_id)},
            ))
    return tuple(issues)


def _duplicate_issues(
    values: tuple[object, ...],
    field_name: str,
) -> tuple[LongTermValidationIssue, ...]:
    seen: set[object] = set()
    issues: list[LongTermValidationIssue] = []
    for value in values:
        identifier = getattr(value, field_name)
        if identifier in seen:
            issues.append(_issue(
                f"duplicate_{field_name}",
                f"duplicate {field_name}",
                "ERROR",
                str(identifier),
            ))
        seen.add(identifier)
    return tuple(issues)


def _ensure_traces(
    values: tuple[LongTermTrace, ...] | list[LongTermTrace],
) -> tuple[LongTermTrace, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("traces must be tuple or list")
    result = tuple(values)
    for value in result:
        if not isinstance(value, LongTermTrace):
            raise TypeError("traces must contain LongTermTrace")
    return result


def _ensure_experiences(
    values: tuple[ConsolidatedExperience, ...] | list[ConsolidatedExperience],
) -> tuple[ConsolidatedExperience, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("experiences must be tuple or list")
    result = tuple(values)
    for value in result:
        if not isinstance(value, ConsolidatedExperience):
            raise TypeError("experiences must contain ConsolidatedExperience")
    return result


def _ensure_provenance_checks(
    values: tuple[ProvenanceCheck, ...] | list[ProvenanceCheck],
) -> tuple[ProvenanceCheck, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("provenance_checks must be tuple or list")
    result = tuple(values)
    for value in result:
        if not isinstance(value, ProvenanceCheck):
            raise TypeError("provenance_checks must contain ProvenanceCheck")
    return result


def _ensure_replay_records(
    values: tuple[ReplayRecord, ...] | list[ReplayRecord],
) -> tuple[ReplayRecord, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("replay_records must be tuple or list")
    result = tuple(values)
    for value in result:
        if not isinstance(value, ReplayRecord):
            raise TypeError("replay_records must contain ReplayRecord")
    return result


def _ensure_eligibility_decisions(
    values: tuple[EligibilityDecision, ...] | list[EligibilityDecision],
) -> tuple[EligibilityDecision, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("eligibility_decisions must be tuple or list")
    result = tuple(values)
    for value in result:
        if not isinstance(value, EligibilityDecision):
            raise TypeError(
                "eligibility_decisions must contain EligibilityDecision"
            )
    return result


def _ensure_issues(
    values: (
        tuple[LongTermValidationIssue, ...] | list[LongTermValidationIssue]
    ),
) -> tuple[LongTermValidationIssue, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("issues must be tuple or list")
    result = tuple(values)
    for value in result:
        if not isinstance(value, LongTermValidationIssue):
            raise TypeError("issues must contain LongTermValidationIssue")
    return result


def _issue(
    code: str,
    message: str,
    severity: str,
    object_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> LongTermValidationIssue:
    return LongTermValidationIssue(
        code=code,
        message=message,
        severity=severity,
        object_id=object_id,
        metadata={} if metadata is None else metadata,
    )


def _copy_metadata(metadata: dict[str, object]) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(metadata)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
    return copied
