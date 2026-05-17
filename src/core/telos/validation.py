"""Deterministic validation utilities for CEBA Core Telos Processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ceba.core.metrics.schema import (
    ensure_constraint_probability,
    ensure_non_negative_number,
    utc_now,
)
from ceba.core.telos.alignment import AlignmentSummary
from ceba.core.telos.attractor import (
    AttractorScore,
    AttractorSelectionResult,
    AttractorSelectionStatus,
)
from ceba.core.telos.schema import (
    AlignmentStatus,
    AttractorSignature,
    TelosAlignmentResult,
    TelosPriority,
    TelosState,
    TelosStatus,
    TelosVectorRef,
)
from ceba.core.telos.telos_update import (
    TelosUpdatePlan,
    TelosUpdateStatus,
    plan_is_blocked,
)
from ceba.core.telos.telos_vector import TelosVectorPayload


class TelosValidationSeverity(Enum):
    """Severity of one telos validation issue."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class TelosValidationStatus(Enum):
    """Overall status of a telos validation report."""

    VALID = "VALID"
    WARNING = "WARNING"
    INVALID = "INVALID"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class TelosValidationIssue:
    """One deterministic telos validation issue."""

    code: str
    message: str
    severity: TelosValidationSeverity
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        code = _ensure_non_empty_string(self.code, "code")
        message = _ensure_non_empty_string(self.message, "message")
        if not isinstance(self.severity, TelosValidationSeverity):
            raise TypeError("severity must be TelosValidationSeverity")

        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TelosValidationReport:
    """Serializable-friendly validation result."""

    status: TelosValidationStatus
    issues: tuple[TelosValidationIssue, ...]
    checked_object_type: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, TelosValidationStatus):
            raise TypeError("status must be TelosValidationStatus")
        issues = tuple(self.issues)
        for validation_issue in issues:
            if not isinstance(validation_issue, TelosValidationIssue):
                raise TypeError("issues must contain TelosValidationIssue")
        checked_object_type = _ensure_non_empty_string(
            self.checked_object_type,
            "checked_object_type",
        )
        created_at = _ensure_timezone_aware(self.created_at, "created_at")

        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "checked_object_type", checked_object_type)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    @property
    def passed(self) -> bool:
        """Return True when report passed validation."""

        return self.status in (
            TelosValidationStatus.VALID,
            TelosValidationStatus.WARNING,
        )

    @property
    def blocked(self) -> bool:
        """Return True when report is blocked."""

        return self.status is TelosValidationStatus.BLOCKED

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "status": self.status.value,
            "issues": tuple(
                validation_issue.to_dict() for validation_issue in self.issues
            ),
            "checked_object_type": self.checked_object_type,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


def issue(
    *,
    code: str,
    message: str,
    severity: TelosValidationSeverity,
    metadata: dict[str, object] | None = None,
) -> TelosValidationIssue:
    """Create one validation issue."""

    return TelosValidationIssue(
        code=code,
        message=message,
        severity=severity,
        metadata={} if metadata is None else metadata,
    )


def severity_rank(severity: TelosValidationSeverity) -> int:
    """Return deterministic severity rank."""

    if not isinstance(severity, TelosValidationSeverity):
        raise TypeError("severity must be TelosValidationSeverity")
    if severity is TelosValidationSeverity.INFO:
        return 0
    if severity is TelosValidationSeverity.WARNING:
        return 1
    if severity is TelosValidationSeverity.ERROR:
        return 2
    return 3


def status_from_issues(
    issues: tuple[TelosValidationIssue, ...] | list[TelosValidationIssue],
) -> TelosValidationStatus:
    """Compute validation status from issues."""

    values = tuple(issues)
    for validation_issue in values:
        if not isinstance(validation_issue, TelosValidationIssue):
            raise TypeError("issues must contain TelosValidationIssue")
    if not values:
        return TelosValidationStatus.VALID
    severities = tuple(validation_issue.severity for validation_issue in values)
    if TelosValidationSeverity.CRITICAL in severities:
        return TelosValidationStatus.BLOCKED
    if TelosValidationSeverity.ERROR in severities:
        return TelosValidationStatus.INVALID
    if TelosValidationSeverity.WARNING in severities:
        return TelosValidationStatus.WARNING
    return TelosValidationStatus.VALID


def validate_telos_vector_ref(vector: TelosVectorRef) -> TelosValidationReport:
    """Validate a TelosVectorRef."""

    if not isinstance(vector, TelosVectorRef):
        raise TypeError("vector must be TelosVectorRef")
    issues: list[TelosValidationIssue] = []
    if vector.dimension <= 0:
        issues.append(
            issue(
                code="TELOS_VECTOR_DIMENSION_INVALID",
                message="Telos vector dimension must be positive.",
                severity=TelosValidationSeverity.ERROR,
            )
        )
    if not vector.vector_id:
        issues.append(
            issue(
                code="TELOS_VECTOR_ID_EMPTY",
                message="Telos vector id must be non-empty.",
                severity=TelosValidationSeverity.ERROR,
            )
        )
    if not vector.normalized:
        issues.append(
            issue(
                code="TELOS_VECTOR_NOT_NORMALIZED",
                message="Telos vector reference is not normalized.",
                severity=TelosValidationSeverity.WARNING,
            )
        )
    return _report(
        checked_object_type="TelosVectorRef",
        issues=issues,
        metadata={
            "source": "telos_validation",
            "object": "TelosVectorRef",
            "vector_id": vector.vector_id,
            "dimension": vector.dimension,
        },
    )


def validate_telos_vector_payload(
    payload: TelosVectorPayload,
) -> TelosValidationReport:
    """Validate a TelosVectorPayload."""

    if not isinstance(payload, TelosVectorPayload):
        raise TypeError("payload must be TelosVectorPayload")
    issues: list[TelosValidationIssue] = []
    if len(payload.values) != payload.vector_ref.dimension:
        issues.append(
            issue(
                code="TELOS_PAYLOAD_DIMENSION_MISMATCH",
                message="Payload values length must match vector reference dimension.",
                severity=TelosValidationSeverity.ERROR,
            )
        )
    ensure_non_negative_number(payload.norm, field_name="norm")
    if payload.vector_ref.normalized and abs(payload.norm - 1.0) > 1e-9:
        issues.append(
            issue(
                code="TELOS_PAYLOAD_NORMALIZATION_MISMATCH",
                message=(
                    "Normalized vector reference must have approximately unit norm."
                ),
                severity=TelosValidationSeverity.ERROR,
                metadata={"norm": payload.norm},
            )
        )
    return _report(
        checked_object_type="TelosVectorPayload",
        issues=issues,
        metadata={
            "source": "telos_validation",
            "object": "TelosVectorPayload",
            "vector_id": payload.vector_ref.vector_id,
            "dimension": payload.vector_ref.dimension,
        },
    )


def validate_telos_priority(priority: TelosPriority) -> TelosValidationReport:
    """Validate a TelosPriority."""

    if not isinstance(priority, TelosPriority):
        raise TypeError("priority must be TelosPriority")
    issues: list[TelosValidationIssue] = []
    ensure_non_negative_number(priority.weight, field_name="weight")
    _ensure_non_negative_int(priority.rank, "rank")
    if priority.weight == 0.0:
        issues.append(
            issue(
                code="TELOS_PRIORITY_ZERO_WEIGHT",
                message="Telos priority has zero weight.",
                severity=TelosValidationSeverity.WARNING,
            )
        )
    return _report(
        checked_object_type="TelosPriority",
        issues=issues,
        metadata={
            "source": "telos_validation",
            "object": "TelosPriority",
            "level": priority.level.value,
            "rank": priority.rank,
        },
    )


def validate_attractor_signature(
    attractor: AttractorSignature,
) -> TelosValidationReport:
    """Validate an AttractorSignature."""

    if not isinstance(attractor, AttractorSignature):
        raise TypeError("attractor must be AttractorSignature")
    issues: list[TelosValidationIssue] = []
    _ensure_non_empty_string(attractor.name, "name")
    ensure_constraint_probability(attractor.strength, field_name="strength")
    ensure_non_negative_number(
        attractor.stability_weight,
        field_name="stability_weight",
    )
    if attractor.basin_radius is not None:
        ensure_non_negative_number(attractor.basin_radius, field_name="basin_radius")
    if attractor.vector is None:
        issues.append(
            issue(
                code="TELOS_ATTRACTOR_VECTOR_MISSING",
                message="Attractor has no vector reference.",
                severity=TelosValidationSeverity.WARNING,
            )
        )
    if attractor.strength == 0.0:
        issues.append(
            issue(
                code="TELOS_ATTRACTOR_ZERO_STRENGTH",
                message="Attractor has zero strength.",
                severity=TelosValidationSeverity.WARNING,
            )
        )
    return _report(
        checked_object_type="AttractorSignature",
        issues=issues,
        metadata={
            "source": "telos_validation",
            "object": "AttractorSignature",
            "attractor_id": str(attractor.attractor_id),
            "name": attractor.name,
        },
    )


def validate_alignment_result(
    result: TelosAlignmentResult,
) -> TelosValidationReport:
    """Validate a TelosAlignmentResult."""

    if not isinstance(result, TelosAlignmentResult):
        raise TypeError("result must be TelosAlignmentResult")
    issues: list[TelosValidationIssue] = []
    ensure_constraint_probability(result.score, field_name="score")
    ensure_constraint_probability(result.confidence, field_name="confidence")
    if result.status is AlignmentStatus.INVALID:
        issues.append(
            issue(
                code="TELOS_ALIGNMENT_INVALID",
                message="Alignment result status is invalid.",
                severity=TelosValidationSeverity.ERROR,
            )
        )
    if result.status is AlignmentStatus.UNKNOWN:
        issues.append(
            issue(
                code="TELOS_ALIGNMENT_UNKNOWN",
                message="Alignment result status is unknown.",
                severity=TelosValidationSeverity.WARNING,
            )
        )
    if (
        result.source_vector_id is None
        and result.target_vector_id is None
        and result.attractor_id is None
    ):
        issues.append(
            issue(
                code="TELOS_ALIGNMENT_SOURCELESS",
                message=(
                    "Alignment result has no source, target, or attractor reference."
                ),
                severity=TelosValidationSeverity.WARNING,
            )
        )
    return _report(
        checked_object_type="TelosAlignmentResult",
        issues=issues,
        metadata={
            "source": "telos_validation",
            "object": "TelosAlignmentResult",
            "alignment_id": str(result.alignment_id),
            "status": result.status.value,
        },
    )


def validate_attractor_score(score: AttractorScore) -> TelosValidationReport:
    """Validate an AttractorScore."""

    if not isinstance(score, AttractorScore):
        raise TypeError("score must be AttractorScore")
    issues: list[TelosValidationIssue] = []
    ensure_non_negative_number(score.final_score, field_name="final_score")
    if score.alignment_score is not None:
        ensure_constraint_probability(
            score.alignment_score,
            field_name="alignment_score",
        )
    if score.status is AttractorSelectionStatus.BELOW_THRESHOLD:
        issues.append(
            issue(
                code="TELOS_ATTRACTOR_SCORE_BELOW_THRESHOLD",
                message="Attractor score is below threshold.",
                severity=TelosValidationSeverity.WARNING,
            )
        )
    if score.status is AttractorSelectionStatus.INVALID:
        issues.append(
            issue(
                code="TELOS_ATTRACTOR_SCORE_INVALID",
                message="Attractor score status is invalid.",
                severity=TelosValidationSeverity.ERROR,
            )
        )
    return _report(
        checked_object_type="AttractorScore",
        issues=issues,
        metadata={
            "source": "telos_validation",
            "object": "AttractorScore",
            "attractor_id": str(score.attractor.attractor_id),
            "status": score.status.value,
        },
    )


def validate_attractor_selection(
    result: AttractorSelectionResult,
) -> TelosValidationReport:
    """Validate an AttractorSelectionResult."""

    if not isinstance(result, AttractorSelectionResult):
        raise TypeError("result must be AttractorSelectionResult")
    issues: list[TelosValidationIssue] = []
    ensure_constraint_probability(result.threshold, field_name="threshold")
    if result.status in (
        AttractorSelectionStatus.EMPTY,
        AttractorSelectionStatus.BELOW_THRESHOLD,
    ):
        issues.append(
            issue(
                code="TELOS_ATTRACTOR_SELECTION_NOT_SELECTED",
                message="Attractor selection did not select an attractor.",
                severity=TelosValidationSeverity.WARNING,
            )
        )
    if result.status is AttractorSelectionStatus.INVALID:
        issues.append(
            issue(
                code="TELOS_ATTRACTOR_SELECTION_INVALID",
                message="Attractor selection status is invalid.",
                severity=TelosValidationSeverity.ERROR,
            )
        )
    if result.status is AttractorSelectionStatus.SELECTED and result.selected is None:
        issues.append(
            issue(
                code="TELOS_ATTRACTOR_SELECTION_EMPTY_SELECTED",
                message=(
                    "Attractor selection is selected but selected attractor is None."
                ),
                severity=TelosValidationSeverity.ERROR,
            )
        )
    return _report(
        checked_object_type="AttractorSelectionResult",
        issues=issues,
        metadata={
            "source": "telos_validation",
            "object": "AttractorSelectionResult",
            "status": result.status.value,
            "selected_id": result.selected_id,
        },
    )


def validate_alignment_summary(summary: AlignmentSummary) -> TelosValidationReport:
    """Validate an AlignmentSummary."""

    if not isinstance(summary, AlignmentSummary):
        raise TypeError("summary must be AlignmentSummary")
    issues: list[TelosValidationIssue] = []
    _ensure_non_negative_int(summary.count, "count")
    ensure_constraint_probability(summary.mean_score, field_name="mean_score")
    ensure_constraint_probability(
        summary.mean_confidence,
        field_name="mean_confidence",
    )
    if summary.count == 0:
        issues.append(
            issue(
                code="TELOS_ALIGNMENT_SUMMARY_EMPTY",
                message="Alignment summary is empty.",
                severity=TelosValidationSeverity.WARNING,
            )
        )
    return _report(
        checked_object_type="AlignmentSummary",
        issues=issues,
        metadata={
            "source": "telos_validation",
            "object": "AlignmentSummary",
            "count": summary.count,
        },
    )


def validate_telos_state(state: TelosState) -> TelosValidationReport:
    """Validate a TelosState."""

    if not isinstance(state, TelosState):
        raise TypeError("state must be TelosState")
    issues: list[TelosValidationIssue] = []
    ensure_constraint_probability(state.alignment_score, field_name="alignment_score")
    _ensure_non_negative_int(state.update_count, "update_count")
    if state.updated_at < state.created_at:
        issues.append(
            issue(
                code="TELOS_STATE_TIME_ORDER_INVALID",
                message=(
                    "TelosState updated_at must be greater than or equal to created_at."
                ),
                severity=TelosValidationSeverity.ERROR,
            )
        )
    if state.status is TelosStatus.INACTIVE and state.active_attractors:
        issues.append(
            issue(
                code="TELOS_STATE_INACTIVE_WITH_ATTRACTORS",
                message="Inactive TelosState has active attractors.",
                severity=TelosValidationSeverity.WARNING,
            )
        )
    if state.status is TelosStatus.ACTIVE and not state.active_attractors:
        issues.append(
            issue(
                code="TELOS_STATE_ACTIVE_WITHOUT_ATTRACTORS",
                message="Active TelosState has no active attractors.",
                severity=TelosValidationSeverity.WARNING,
            )
        )
    if state.status is TelosStatus.INVALID:
        issues.append(
            issue(
                code="TELOS_STATE_INVALID",
                message="TelosState status is invalid.",
                severity=TelosValidationSeverity.ERROR,
            )
        )
    if state.primary_attractor is not None:
        active_ids = {attractor.attractor_id for attractor in state.active_attractors}
        if state.primary_attractor.attractor_id not in active_ids:
            issues.append(
                issue(
                    code="TELOS_STATE_PRIMARY_NOT_ACTIVE",
                    message="Primary attractor is not present in active_attractors.",
                    severity=TelosValidationSeverity.ERROR,
                )
            )
    return _report(
        checked_object_type="TelosState",
        issues=issues,
        metadata={
            "source": "telos_validation",
            "object": "TelosState",
            "state_id": str(state.state_id),
            "status": state.status.value,
            "active_attractor_count": len(state.active_attractors),
        },
    )


def validate_telos_update_plan(plan: TelosUpdatePlan) -> TelosValidationReport:
    """Validate a TelosUpdatePlan."""

    if not isinstance(plan, TelosUpdatePlan):
        raise TypeError("plan must be TelosUpdatePlan")
    issues: list[TelosValidationIssue] = []
    ensure_constraint_probability(plan.alignment_score, field_name="alignment_score")
    ensure_constraint_probability(plan.confidence, field_name="confidence")
    _ensure_non_empty_string(plan.reason_code, "reason_code")
    if plan.status is TelosUpdateStatus.INVALID:
        issues.append(
            issue(
                code="TELOS_UPDATE_PLAN_INVALID",
                message="Telos update plan status is invalid.",
                severity=TelosValidationSeverity.ERROR,
            )
        )
    if plan_is_blocked(plan):
        issues.append(
            issue(
                code="TELOS_UPDATE_PLAN_BLOCKED",
                message="Telos update plan is blocked.",
                severity=TelosValidationSeverity.CRITICAL,
            )
        )
    if plan.status is TelosUpdateStatus.DEFERRED:
        issues.append(
            issue(
                code="TELOS_UPDATE_PLAN_DEFERRED",
                message="Telos update plan is deferred.",
                severity=TelosValidationSeverity.WARNING,
            )
        )
    if plan.status is TelosUpdateStatus.SKIPPED:
        issues.append(
            issue(
                code="TELOS_UPDATE_PLAN_SKIPPED",
                message="Telos update plan is skipped.",
                severity=TelosValidationSeverity.WARNING,
            )
        )
    return _report(
        checked_object_type="TelosUpdatePlan",
        issues=issues,
        metadata={
            "source": "telos_validation",
            "object": "TelosUpdatePlan",
            "state_id": plan.state_id,
            "status": plan.status.value,
            "action": plan.action.value,
        },
    )


def combine_validation_reports(
    reports: tuple[TelosValidationReport, ...] | list[TelosValidationReport],
    *,
    checked_object_type: str = "combined",
    metadata: dict[str, object] | None = None,
) -> TelosValidationReport:
    """Combine validation reports into one report."""

    values = tuple(reports)
    for report in values:
        if not isinstance(report, TelosValidationReport):
            raise TypeError("reports must contain TelosValidationReport")
    checked_object_type = _ensure_non_empty_string(
        checked_object_type,
        "checked_object_type",
    )
    issues: list[TelosValidationIssue] = []
    for report in values:
        issues.extend(report.issues)

    default_metadata: dict[str, object] = {
        "source": "telos_validation",
        "object": checked_object_type,
        "report_count": len(values),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "telos_validation"

    return TelosValidationReport(
        status=status_from_issues(issues),
        issues=tuple(issues),
        checked_object_type=checked_object_type,
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def validation_report_passed(report: TelosValidationReport) -> bool:
    """Return True when validation report passed."""

    if not isinstance(report, TelosValidationReport):
        raise TypeError("report must be TelosValidationReport")
    return report.passed


def validation_report_blocked(report: TelosValidationReport) -> bool:
    """Return True when validation report is blocked."""

    if not isinstance(report, TelosValidationReport):
        raise TypeError("report must be TelosValidationReport")
    return report.blocked


def _report(
    *,
    checked_object_type: str,
    issues: tuple[TelosValidationIssue, ...] | list[TelosValidationIssue],
    metadata: dict[str, object],
) -> TelosValidationReport:
    return TelosValidationReport(
        status=status_from_issues(issues),
        issues=tuple(issues),
        checked_object_type=checked_object_type,
        created_at=utc_now(),
        metadata=metadata,
    )


def _ensure_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    if value == "":
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _ensure_non_negative_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _ensure_timezone_aware(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _copy_metadata(metadata: dict[str, object]) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(metadata)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
    return copied


def _merge_metadata(
    default_metadata: dict[str, object],
    caller_metadata: dict[str, object] | None,
) -> dict[str, object]:
    merged = _copy_metadata(default_metadata)
    if caller_metadata is not None:
        merged.update(_copy_metadata(caller_metadata))
    return merged


__all__ = [
    "TelosValidationSeverity",
    "TelosValidationStatus",
    "TelosValidationIssue",
    "TelosValidationReport",
    "issue",
    "severity_rank",
    "status_from_issues",
    "validate_telos_vector_ref",
    "validate_telos_vector_payload",
    "validate_telos_priority",
    "validate_attractor_signature",
    "validate_alignment_result",
    "validate_attractor_score",
    "validate_attractor_selection",
    "validate_alignment_summary",
    "validate_telos_state",
    "validate_telos_update_plan",
    "combine_validation_reports",
    "validation_report_passed",
    "validation_report_blocked",
]
