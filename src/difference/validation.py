"""Deterministic validation utilities for CEBA difference processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ois.difference.aggregator import (
    DifferenceAggregateBucket,
    DifferenceAggregateSummary,
    severity_rank,
)
from ois.difference.errors import (
    DifferenceInvariantError,
    DifferenceValidationError,
    EmptyDifferenceSetError,
    build_error_context,
)
from ois.difference.schema import (
    DifferenceCandidate,
    DifferenceClassification,
    DifferenceKind,
    DifferenceSet,
    DifferenceSeverity,
    DifferenceStatus,
    ensure_non_negative,
    ensure_probability,
    ensure_timezone_aware,
)


class ValidationStatus(str, Enum):
    """Stable validation status for difference artifacts."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One deterministic validation issue."""

    code: str
    message: str
    status: ValidationStatus
    difference_id: str | None = None
    field_path: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_non_empty_string(self.code, "code")
        _ensure_non_empty_string(self.message, "message")
        if not isinstance(self.status, ValidationStatus):
            raise TypeError("status must be ValidationStatus")
        if self.difference_id is not None:
            _ensure_non_empty_string(self.difference_id, "difference_id")
        if isinstance(self.field_path, list):
            object.__setattr__(self, "field_path", tuple(self.field_path))
        if not isinstance(self.field_path, tuple):
            raise TypeError("field_path must be a tuple of strings")
        for part in self.field_path:
            _ensure_non_empty_string(part, "field_path")
        _ensure_metadata_dict(self.metadata)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serialization-friendly representation."""

        return {
            "code": self.code,
            "message": self.message,
            "status": self.status.value,
            "difference_id": self.difference_id,
            "field_path": list(self.field_path),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Validation result for one difference artifact or batch."""

    status: ValidationStatus
    issues: tuple[ValidationIssue, ...]
    checked_count: int
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, ValidationStatus):
            raise TypeError("status must be ValidationStatus")
        if not isinstance(self.issues, tuple):
            if isinstance(self.issues, list):
                object.__setattr__(self, "issues", tuple(self.issues))
            else:
                raise TypeError("issues must be a tuple of ValidationIssue")
        for validation_issue in self.issues:
            if not isinstance(validation_issue, ValidationIssue):
                raise TypeError("issues must contain ValidationIssue")
        _ensure_non_negative_int(self.checked_count, "checked_count")
        _ensure_metadata_dict(self.metadata)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def passed(self) -> bool:
        """Return True when validation passed."""

        return self.status is ValidationStatus.PASS

    @property
    def failed(self) -> bool:
        """Return True when validation failed."""

        return self.status is ValidationStatus.FAIL

    def to_dict(self) -> dict[str, object]:
        """Return a serialization-friendly representation."""

        return {
            "status": self.status.value,
            "issues": [validation_issue.to_dict() for validation_issue in self.issues],
            "checked_count": self.checked_count,
            "metadata": dict(self.metadata),
        }


def issue(
    *,
    code: str,
    message: str,
    status: ValidationStatus,
    difference_id: str | None = None,
    field_path: tuple[str, ...] | list[str] = (),
    metadata: dict[str, object] | None = None,
) -> ValidationIssue:
    """Build one validated issue."""

    return ValidationIssue(
        code=code,
        message=message,
        status=status,
        difference_id=difference_id,
        field_path=tuple(field_path),
        metadata={} if metadata is None else metadata,
    )


def report_from_issues(
    issues: tuple[ValidationIssue, ...] | list[ValidationIssue],
    *,
    checked_count: int,
    metadata: dict[str, object] | None = None,
) -> ValidationReport:
    """Build a validation report with dominant status."""

    if not isinstance(issues, (tuple, list)):
        raise TypeError("issues must be a tuple or list of ValidationIssue")
    resolved_issues = tuple(issues)
    for validation_issue in resolved_issues:
        if not isinstance(validation_issue, ValidationIssue):
            raise TypeError("issues must contain ValidationIssue")
    _ensure_non_negative_int(checked_count, "checked_count")

    status = ValidationStatus.PASS
    if any(validation_issue.status is ValidationStatus.FAIL for validation_issue in resolved_issues):
        status = ValidationStatus.FAIL
    elif any(validation_issue.status is ValidationStatus.WARN for validation_issue in resolved_issues):
        status = ValidationStatus.WARN

    return ValidationReport(
        status=status,
        issues=resolved_issues,
        checked_count=checked_count,
        metadata={} if metadata is None else metadata,
    )


def validate_candidate(candidate: DifferenceCandidate) -> ValidationReport:
    """Validate one DifferenceCandidate."""

    difference_id: str | None = None
    try:
        if isinstance(candidate, DifferenceCandidate):
            difference_id = str(candidate.difference_id)
        else:
            raise TypeError("candidate must be DifferenceCandidate")

        issues: list[ValidationIssue] = []
        _append_validation_issue(
            issues,
            difference_id=difference_id,
            field_path=("confidence",),
            code="INVALID_CONFIDENCE",
            message="candidate confidence must be a probability",
            validator=lambda: ensure_probability(candidate.confidence, "confidence"),
        )
        _append_validation_issue(
            issues,
            difference_id=difference_id,
            field_path=("magnitude",),
            code="INVALID_MAGNITUDE",
            message="candidate magnitude must be None or non-negative",
            validator=lambda: ensure_non_negative(candidate.magnitude, "magnitude"),
        )
        _append_validation_issue(
            issues,
            difference_id=difference_id,
            field_path=("created_at",),
            code="INVALID_CREATED_AT",
            message="candidate created_at must be timezone-aware",
            validator=lambda: ensure_timezone_aware(candidate.created_at, "created_at"),
        )

        if candidate.status is DifferenceStatus.CANDIDATE:
            issues.append(
                issue(
                    code="CANDIDATE_STATUS_NOT_DISPATCH_READY",
                    message="candidate status must not be CANDIDATE at dispatch boundary",
                    status=ValidationStatus.FAIL,
                    difference_id=difference_id,
                    field_path=("status",),
                )
            )
        elif candidate.status is DifferenceStatus.REJECTED:
            issues.append(
                issue(
                    code="REJECTED_CANDIDATE",
                    message="rejected candidate is not dispatch-ready",
                    status=ValidationStatus.WARN,
                    difference_id=difference_id,
                    field_path=("status",),
                )
            )

        if candidate.kind is DifferenceKind.UNKNOWN:
            issues.append(
                issue(
                    code="UNKNOWN_DIFFERENCE_KIND",
                    message="candidate kind is UNKNOWN",
                    status=ValidationStatus.WARN,
                    difference_id=difference_id,
                    field_path=("kind",),
                )
            )

        if candidate.left_value is None and candidate.right_value is None:
            issues.append(
                issue(
                    code="MISSING_COMPARABLE_VALUES",
                    message="candidate must include at least one compared value",
                    status=ValidationStatus.FAIL,
                    difference_id=difference_id,
                    field_path=("left_value", "right_value"),
                )
            )

        if not candidate.evidence_refs:
            issues.append(
                issue(
                    code="EMPTY_EVIDENCE_REFS",
                    message="candidate has no evidence references",
                    status=ValidationStatus.WARN,
                    difference_id=difference_id,
                    field_path=("evidence_refs",),
                )
            )

        return report_from_issues(issues, checked_count=1)
    except DifferenceValidationError:
        raise
    except Exception as exc:
        raise DifferenceValidationError(
            message="failed to validate difference candidate",
            context=build_error_context(
                operation="validate_candidate",
                difference_id=difference_id,
            ),
            cause=exc,
        ) from exc


def validate_classification(
    classification: DifferenceClassification,
    candidate: DifferenceCandidate | None = None,
) -> ValidationReport:
    """Validate one DifferenceClassification, optionally against its candidate."""

    difference_id: str | None = None
    try:
        if isinstance(classification, DifferenceClassification):
            difference_id = str(classification.difference_id)
        else:
            raise TypeError("classification must be DifferenceClassification")
        if candidate is not None and not isinstance(candidate, DifferenceCandidate):
            raise TypeError("candidate must be DifferenceCandidate or None")

        issues: list[ValidationIssue] = []
        _append_validation_issue(
            issues,
            difference_id=difference_id,
            field_path=("confidence",),
            code="INVALID_CONFIDENCE",
            message="classification confidence must be a probability",
            validator=lambda: ensure_probability(classification.confidence, "confidence"),
        )

        if not isinstance(classification.severity, DifferenceSeverity):
            issues.append(
                issue(
                    code="INVALID_SEVERITY",
                    message="classification severity must be DifferenceSeverity",
                    status=ValidationStatus.FAIL,
                    difference_id=difference_id,
                    field_path=("severity",),
                )
            )
        if not isinstance(classification.kind, DifferenceKind):
            issues.append(
                issue(
                    code="INVALID_KIND",
                    message="classification kind must be DifferenceKind",
                    status=ValidationStatus.FAIL,
                    difference_id=difference_id,
                    field_path=("kind",),
                )
            )
        if classification.rationale_code is None:
            issues.append(
                issue(
                    code="MISSING_RATIONALE_CODE",
                    message="classification rationale_code is missing",
                    status=ValidationStatus.WARN,
                    difference_id=difference_id,
                    field_path=("rationale_code",),
                )
            )

        if candidate is not None:
            if classification.difference_id != candidate.difference_id:
                issues.append(
                    issue(
                        code="CLASSIFICATION_CANDIDATE_ID_MISMATCH",
                        message="classification difference_id does not match candidate",
                        status=ValidationStatus.FAIL,
                        difference_id=difference_id,
                        field_path=("difference_id",),
                    )
                )
            if classification.kind is not candidate.kind:
                issues.append(
                    issue(
                        code="CLASSIFICATION_CANDIDATE_KIND_MISMATCH",
                        message="classification kind does not match candidate kind",
                        status=ValidationStatus.FAIL,
                        difference_id=difference_id,
                        field_path=("kind",),
                    )
                )

        return report_from_issues(issues, checked_count=1)
    except DifferenceValidationError:
        raise
    except Exception as exc:
        raise DifferenceValidationError(
            message="failed to validate difference classification",
            context=build_error_context(
                operation="validate_classification",
                difference_id=difference_id,
            ),
            cause=exc,
        ) from exc


def validate_difference_set(difference_set: DifferenceSet) -> ValidationReport:
    """Validate a DifferenceSet."""

    observation_id: str | None = None
    try:
        if isinstance(difference_set, DifferenceSet):
            observation_id = difference_set.observation_id
        else:
            raise TypeError("difference_set must be DifferenceSet")
        if not difference_set.candidates:
            raise EmptyDifferenceSetError(message="difference_set candidates must not be empty")

        issues: list[ValidationIssue] = []
        if not difference_set.observation_id:
            issues.append(
                issue(
                    code="EMPTY_OBSERVATION_ID",
                    message="difference_set observation_id must be non-empty",
                    status=ValidationStatus.FAIL,
                    field_path=("observation_id",),
                )
            )
        _append_validation_issue(
            issues,
            difference_id=None,
            field_path=("created_at",),
            code="INVALID_CREATED_AT",
            message="difference_set created_at must be timezone-aware",
            validator=lambda: ensure_timezone_aware(difference_set.created_at, "created_at"),
        )

        for candidate in difference_set.candidates:
            candidate_report = validate_candidate(candidate)
            issues.extend(candidate_report.issues)

        return report_from_issues(issues, checked_count=len(difference_set.candidates))
    except EmptyDifferenceSetError:
        raise
    except DifferenceValidationError:
        raise
    except Exception as exc:
        raise DifferenceValidationError(
            message="failed to validate difference set",
            context=build_error_context(
                operation="validate_difference_set",
                observation_id=observation_id,
            ),
            cause=exc,
        ) from exc


def validate_classifications_for_set(
    difference_set: DifferenceSet,
    classifications: tuple[DifferenceClassification, ...] | list[DifferenceClassification],
) -> ValidationReport:
    """Validate classifications against a DifferenceSet."""

    observation_id: str | None = None
    try:
        if isinstance(difference_set, DifferenceSet):
            observation_id = difference_set.observation_id
        else:
            raise TypeError("difference_set must be DifferenceSet")
        if not isinstance(classifications, (tuple, list)):
            raise TypeError("classifications must be a tuple or list")

        issues: list[ValidationIssue] = []
        candidates_by_id: dict[str, DifferenceCandidate] = {
            str(candidate.difference_id): candidate for candidate in difference_set.candidates
        }
        seen_classification_ids: set[str] = set()

        for classification in classifications:
            if not isinstance(classification, DifferenceClassification):
                raise TypeError("classifications must contain DifferenceClassification")

            difference_id = str(classification.difference_id)
            candidate = candidates_by_id.get(difference_id)
            if difference_id in seen_classification_ids:
                issues.append(
                    issue(
                        code="DUPLICATE_CLASSIFICATION",
                        message="duplicate classification difference_id",
                        status=ValidationStatus.FAIL,
                        difference_id=difference_id,
                        field_path=("difference_id",),
                    )
                )
            seen_classification_ids.add(difference_id)

            if candidate is None:
                issues.append(
                    issue(
                        code="CLASSIFICATION_WITHOUT_CANDIDATE",
                        message="classification has no matching candidate",
                        status=ValidationStatus.FAIL,
                        difference_id=difference_id,
                        field_path=("difference_id",),
                    )
                )
                classification_report = validate_classification(classification)
            else:
                classification_report = validate_classification(classification, candidate)
            issues.extend(classification_report.issues)

        for candidate in difference_set.candidates:
            difference_id = str(candidate.difference_id)
            if difference_id not in seen_classification_ids:
                issues.append(
                    issue(
                        code="CANDIDATE_WITHOUT_CLASSIFICATION",
                        message="candidate has no matching classification",
                        status=ValidationStatus.WARN,
                        difference_id=difference_id,
                        field_path=("classifications",),
                    )
                )

        return report_from_issues(issues, checked_count=len(classifications))
    except DifferenceValidationError:
        raise
    except Exception as exc:
        raise DifferenceValidationError(
            message="failed to validate classifications for set",
            context=build_error_context(
                operation="validate_classifications_for_set",
                observation_id=observation_id,
            ),
            cause=exc,
        ) from exc


def validate_aggregate_summary(summary: DifferenceAggregateSummary) -> ValidationReport:
    """Validate one DifferenceAggregateSummary."""

    observation_id: str | None = None
    try:
        if isinstance(summary, DifferenceAggregateSummary):
            observation_id = summary.observation_id
        else:
            raise TypeError("summary must be DifferenceAggregateSummary")

        issues: list[ValidationIssue] = []
        if not summary.observation_id:
            issues.append(
                issue(
                    code="EMPTY_OBSERVATION_ID",
                    message="summary observation_id must be non-empty",
                    status=ValidationStatus.FAIL,
                    field_path=("observation_id",),
                )
            )
        _append_validation_issue(
            issues,
            difference_id=None,
            field_path=("total_candidates",),
            code="INVALID_TOTAL_CANDIDATES",
            message="summary total_candidates must be non-negative",
            validator=lambda: _ensure_non_negative_int(
                summary.total_candidates,
                "total_candidates",
            ),
        )
        _append_validation_issue(
            issues,
            difference_id=None,
            field_path=("total_classifications",),
            code="INVALID_TOTAL_CLASSIFICATIONS",
            message="summary total_classifications must be non-negative",
            validator=lambda: _ensure_non_negative_int(
                summary.total_classifications,
                "total_classifications",
            ),
        )
        if not isinstance(summary.buckets, tuple):
            issues.append(
                issue(
                    code="INVALID_BUCKETS",
                    message="summary buckets must be a tuple",
                    status=ValidationStatus.FAIL,
                    field_path=("buckets",),
                )
            )
        else:
            for bucket in summary.buckets:
                issues.extend(_validate_bucket_issues(bucket))

        if summary.total_candidates == 0:
            issues.append(
                issue(
                    code="EMPTY_SUMMARY_CANDIDATES",
                    message="summary total_candidates must be greater than zero",
                    status=ValidationStatus.FAIL,
                    field_path=("total_candidates",),
                )
            )
        if summary.total_candidates > 0 and not summary.buckets:
            issues.append(
                issue(
                    code="EMPTY_SUMMARY_BUCKETS",
                    message="summary buckets must not be empty when candidates exist",
                    status=ValidationStatus.FAIL,
                    field_path=("buckets",),
                )
            )
        if summary.max_severity is not None:
            if isinstance(summary.max_severity, DifferenceSeverity):
                severity_rank(summary.max_severity)
            else:
                issues.append(
                    issue(
                        code="INVALID_MAX_SEVERITY",
                        message="summary max_severity must be DifferenceSeverity or None",
                        status=ValidationStatus.FAIL,
                        field_path=("max_severity",),
                    )
                )
        _append_validation_issue(
            issues,
            difference_id=None,
            field_path=("mean_confidence",),
            code="INVALID_MEAN_CONFIDENCE",
            message="summary mean_confidence must be a probability",
            validator=lambda: ensure_probability(summary.mean_confidence, "mean_confidence"),
        )
        _append_validation_issue(
            issues,
            difference_id=None,
            field_path=("max_magnitude",),
            code="INVALID_MAX_MAGNITUDE",
            message="summary max_magnitude must be None or non-negative",
            validator=lambda: ensure_non_negative(summary.max_magnitude, "max_magnitude"),
        )

        return report_from_issues(issues, checked_count=len(summary.buckets))
    except DifferenceValidationError:
        raise
    except Exception as exc:
        raise DifferenceValidationError(
            message="failed to validate aggregate summary",
            context=build_error_context(
                operation="validate_aggregate_summary",
                observation_id=observation_id,
            ),
            cause=exc,
        ) from exc


def validate_dispatch_ready(
    *,
    difference_set: DifferenceSet,
    classifications: tuple[DifferenceClassification, ...] | list[DifferenceClassification],
    summary: DifferenceAggregateSummary,
) -> ValidationReport:
    """Validate all artifacts before dispatch."""

    observation_id: str | None = None
    try:
        if isinstance(difference_set, DifferenceSet):
            observation_id = difference_set.observation_id

        set_report = validate_difference_set(difference_set)
        classification_report = validate_classifications_for_set(difference_set, classifications)
        summary_report = validate_aggregate_summary(summary)

        issues = (
            set_report.issues
            + classification_report.issues
            + summary_report.issues
        )
        checked_count = (
            set_report.checked_count
            + classification_report.checked_count
            + summary_report.checked_count
        )
        return report_from_issues(
            issues,
            checked_count=checked_count,
            metadata={"stage": "dispatch_ready"},
        )
    except EmptyDifferenceSetError:
        raise
    except DifferenceValidationError:
        raise
    except Exception as exc:
        raise DifferenceValidationError(
            message="failed to validate dispatch readiness",
            context=build_error_context(
                operation="validate_dispatch_ready",
                observation_id=observation_id,
            ),
            cause=exc,
        ) from exc


def _append_validation_issue(
    issues: list[ValidationIssue],
    *,
    difference_id: str | None,
    field_path: tuple[str, ...],
    code: str,
    message: str,
    validator: object,
) -> None:
    if not callable(validator):
        raise DifferenceInvariantError(message="validator must be callable")
    try:
        validator()
    except Exception as exc:
        issues.append(
            issue(
                code=code,
                message=message,
                status=ValidationStatus.FAIL,
                difference_id=difference_id,
                field_path=field_path,
                metadata={
                    "cause_type": type(exc).__name__,
                    "cause_message": str(exc),
                },
            )
        )


def _validate_bucket_issues(bucket: object) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    if not isinstance(bucket, DifferenceAggregateBucket):
        return (
            issue(
                code="INVALID_BUCKET",
                message="summary buckets must contain DifferenceAggregateBucket",
                status=ValidationStatus.FAIL,
                field_path=("buckets",),
            ),
        )

    if not bucket.key:
        issues.append(
            issue(
                code="EMPTY_BUCKET_KEY",
                message="bucket key must be non-empty",
                status=ValidationStatus.FAIL,
                field_path=("buckets", "key"),
            )
        )
    _append_validation_issue(
        issues,
        difference_id=None,
        field_path=("buckets", "count"),
        code="INVALID_BUCKET_COUNT",
        message="bucket count must be non-negative",
        validator=lambda: _ensure_non_negative_int(bucket.count, "count"),
    )
    _append_validation_issue(
        issues,
        difference_id=None,
        field_path=("buckets", "mean_confidence"),
        code="INVALID_BUCKET_MEAN_CONFIDENCE",
        message="bucket mean_confidence must be a probability",
        validator=lambda: ensure_probability(bucket.mean_confidence, "mean_confidence"),
    )
    _append_validation_issue(
        issues,
        difference_id=None,
        field_path=("buckets", "max_magnitude"),
        code="INVALID_BUCKET_MAX_MAGNITUDE",
        message="bucket max_magnitude must be None or non-negative",
        validator=lambda: ensure_non_negative(bucket.max_magnitude, "max_magnitude"),
    )
    if not isinstance(bucket.candidate_ids, tuple):
        issues.append(
            issue(
                code="INVALID_BUCKET_CANDIDATE_IDS",
                message="bucket candidate_ids must be a tuple",
                status=ValidationStatus.FAIL,
                field_path=("buckets", "candidate_ids"),
            )
        )
    else:
        for candidate_id in bucket.candidate_ids:
            if not isinstance(candidate_id, str) or not candidate_id:
                issues.append(
                    issue(
                        code="INVALID_BUCKET_CANDIDATE_ID",
                        message="bucket candidate_ids must contain non-empty strings",
                        status=ValidationStatus.FAIL,
                        field_path=("buckets", "candidate_ids"),
                    )
                )
    return tuple(issues)


def _ensure_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(field_name, str) or not field_name:
        raise ValueError("field_name must be a non-empty string")
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _ensure_non_negative_int(value: int, field_name: str) -> int:
    if not isinstance(field_name, str) or not field_name:
        raise ValueError("field_name must be a non-empty string")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{field_name} must be greater than or equal to 0")
    return value


def _ensure_metadata_dict(value: dict[str, object]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("metadata must be a dict")
    if not all(isinstance(key, str) for key in value):
        raise TypeError("metadata keys must be strings")
    return value


__all__ = [
    "ValidationStatus",
    "ValidationIssue",
    "ValidationReport",
    "issue",
    "report_from_issues",
    "validate_candidate",
    "validate_classification",
    "validate_difference_set",
    "validate_classifications_for_set",
    "validate_aggregate_summary",
    "validate_dispatch_ready",
]
