"""Deterministic audit utilities for CEBA difference processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib

from ois.difference.aggregator import DifferenceAggregateSummary
from ois.difference.dispatcher import (
    DifferenceDispatchEnvelope,
    DifferenceDispatchStatus,
    DifferenceDispatchTarget,
)
from ois.difference.errors import DifferenceAuditError, build_error_context
from ois.difference.schema import (
    DifferenceCandidate,
    DifferenceClassification,
    DifferenceKind,
    DifferenceSet,
    DifferenceSeverity,
    DifferenceStatus,
    ensure_timezone_aware,
    utc_now,
)
from ois.difference.validation import ValidationReport, ValidationStatus


class DifferenceAuditEventType(str, Enum):
    """Stable audit event types for the difference pipeline."""

    CANDIDATE_CREATED = "CANDIDATE_CREATED"
    CANDIDATE_NORMALIZED = "CANDIDATE_NORMALIZED"
    CANDIDATE_CLASSIFIED = "CANDIDATE_CLASSIFIED"
    SET_VALIDATED = "SET_VALIDATED"
    SUMMARY_AGGREGATED = "SUMMARY_AGGREGATED"
    DISPATCH_ENVELOPE_CREATED = "DISPATCH_ENVELOPE_CREATED"
    DISPATCH_READY = "DISPATCH_READY"
    DISPATCH_DEAD_LETTER = "DISPATCH_DEAD_LETTER"
    PIPELINE_ERROR = "PIPELINE_ERROR"
    PIPELINE_SUMMARY = "PIPELINE_SUMMARY"


@dataclass(frozen=True, slots=True)
class DifferenceAuditRecord:
    """Immutable audit event for one difference pipeline action."""

    audit_id: str
    event_type: DifferenceAuditEventType
    observation_id: str | None
    difference_id: str | None
    set_id: str | None
    envelope_id: str | None
    status: str | None
    created_at: datetime
    payload: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_non_empty_string(self.audit_id, "audit_id")
        if not isinstance(self.event_type, DifferenceAuditEventType):
            raise TypeError("event_type must be DifferenceAuditEventType")
        _ensure_optional_non_empty_string(self.observation_id, "observation_id")
        _ensure_optional_non_empty_string(self.difference_id, "difference_id")
        _ensure_optional_non_empty_string(self.set_id, "set_id")
        _ensure_optional_non_empty_string(self.envelope_id, "envelope_id")
        _ensure_optional_non_empty_string(self.status, "status")
        ensure_timezone_aware(self.created_at, "created_at")
        _ensure_metadata_dict(self.payload, "payload")
        _ensure_metadata_dict(self.metadata, "metadata")
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serialization-friendly representation."""

        return {
            "audit_id": self.audit_id,
            "event_type": self.event_type.value,
            "observation_id": self.observation_id,
            "difference_id": self.difference_id,
            "set_id": self.set_id,
            "envelope_id": self.envelope_id,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "payload": dict(self.payload),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class DifferenceAuditTrail:
    """Ordered immutable collection of audit records."""

    trail_id: str
    observation_id: str | None
    records: tuple[DifferenceAuditRecord, ...]
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_non_empty_string(self.trail_id, "trail_id")
        _ensure_optional_non_empty_string(self.observation_id, "observation_id")
        if not isinstance(self.records, tuple):
            if isinstance(self.records, list):
                object.__setattr__(self, "records", tuple(self.records))
            else:
                raise TypeError("records must be a tuple of DifferenceAuditRecord")
        for record in self.records:
            if not isinstance(record, DifferenceAuditRecord):
                raise TypeError("records must contain DifferenceAuditRecord")
        ensure_timezone_aware(self.created_at, "created_at")
        _ensure_metadata_dict(self.metadata, "metadata")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def event_count(self) -> int:
        """Return the number of audit records."""

        return len(self.records)

    @property
    def has_errors(self) -> bool:
        """Return True when the trail contains a pipeline error."""

        return any(
            record.event_type is DifferenceAuditEventType.PIPELINE_ERROR
            for record in self.records
        )

    def to_dict(self) -> dict[str, object]:
        """Return a serialization-friendly representation."""

        return {
            "trail_id": self.trail_id,
            "observation_id": self.observation_id,
            "records": [record.to_dict() for record in self.records],
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


def build_audit_id(
    *,
    event_type: DifferenceAuditEventType,
    observation_id: str | None = None,
    difference_id: str | None = None,
    set_id: str | None = None,
    envelope_id: str | None = None,
    status: str | None = None,
    payload: dict[str, object] | None = None,
) -> str:
    """Build a deterministic audit identifier."""

    if not isinstance(event_type, DifferenceAuditEventType):
        raise TypeError("event_type must be DifferenceAuditEventType")
    _ensure_optional_non_empty_string(observation_id, "observation_id")
    _ensure_optional_non_empty_string(difference_id, "difference_id")
    _ensure_optional_non_empty_string(set_id, "set_id")
    _ensure_optional_non_empty_string(envelope_id, "envelope_id")
    _ensure_optional_non_empty_string(status, "status")
    if payload is not None:
        _ensure_metadata_dict(payload, "payload")

    components = (
        event_type.value,
        observation_id or "",
        difference_id or "",
        set_id or "",
        envelope_id or "",
        status or "",
        _stable_payload_repr(payload) if payload is not None else "",
    )
    digest = hashlib.sha256()
    for component in components:
        digest.update(component.encode("utf-8"))
        digest.update(b"\x00")
    return "audit_" + digest.hexdigest()[:32]


def create_audit_record(
    *,
    event_type: DifferenceAuditEventType,
    observation_id: str | None = None,
    difference_id: str | None = None,
    set_id: str | None = None,
    envelope_id: str | None = None,
    status: str | None = None,
    payload: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
) -> DifferenceAuditRecord:
    """Create a validated audit record."""

    try:
        resolved_payload = {} if payload is None else payload
        resolved_metadata = {} if metadata is None else metadata
        audit_id = build_audit_id(
            event_type=event_type,
            observation_id=observation_id,
            difference_id=difference_id,
            set_id=set_id,
            envelope_id=envelope_id,
            status=status,
            payload=resolved_payload,
        )
        return DifferenceAuditRecord(
            audit_id=audit_id,
            event_type=event_type,
            observation_id=observation_id,
            difference_id=difference_id,
            set_id=set_id,
            envelope_id=envelope_id,
            status=status,
            created_at=utc_now(),
            payload=resolved_payload,
            metadata=resolved_metadata,
        )
    except DifferenceAuditError:
        raise
    except Exception as exc:
        raise DifferenceAuditError(
            message="failed to create audit record",
            context=build_error_context(operation="create_audit_record"),
            cause=exc,
        ) from exc


def audit_candidate(candidate: DifferenceCandidate) -> DifferenceAuditRecord:
    """Create an audit record for one DifferenceCandidate."""

    try:
        if not isinstance(candidate, DifferenceCandidate):
            raise TypeError("candidate must be DifferenceCandidate")

        if candidate.status is DifferenceStatus.NORMALIZED:
            event_type = DifferenceAuditEventType.CANDIDATE_NORMALIZED
        elif candidate.status is DifferenceStatus.CLASSIFIED:
            event_type = DifferenceAuditEventType.CANDIDATE_CLASSIFIED
        else:
            event_type = DifferenceAuditEventType.CANDIDATE_CREATED

        return create_audit_record(
            event_type=event_type,
            difference_id=str(candidate.difference_id),
            status=candidate.status.value,
            payload={
                "kind": candidate.kind.value,
                "direction": candidate.direction.value,
                "confidence": candidate.confidence,
                "magnitude": candidate.magnitude,
                "evidence_ref_count": len(candidate.evidence_refs),
                "has_left_value": candidate.left_value is not None,
                "has_right_value": candidate.right_value is not None,
            },
        )
    except DifferenceAuditError:
        raise
    except Exception as exc:
        raise DifferenceAuditError(
            message="failed to audit candidate",
            context=build_error_context(operation="audit_candidate"),
            cause=exc,
        ) from exc


def audit_difference_set(difference_set: DifferenceSet) -> DifferenceAuditRecord:
    """Create an audit record for one DifferenceSet."""

    try:
        if not isinstance(difference_set, DifferenceSet):
            raise TypeError("difference_set must be DifferenceSet")

        return create_audit_record(
            event_type=DifferenceAuditEventType.SET_VALIDATED,
            observation_id=difference_set.observation_id,
            set_id=str(difference_set.set_id),
            status="SET",
            payload={
                "candidate_count": len(difference_set.candidates),
                "candidate_status_counts": _candidate_status_counts(
                    difference_set.candidates,
                ),
                "candidate_kind_counts": _candidate_kind_counts(
                    difference_set.candidates,
                ),
            },
        )
    except DifferenceAuditError:
        raise
    except Exception as exc:
        raise DifferenceAuditError(
            message="failed to audit difference set",
            context=build_error_context(operation="audit_difference_set"),
            cause=exc,
        ) from exc


def audit_classifications(
    classifications: tuple[DifferenceClassification, ...] | list[DifferenceClassification],
    *,
    observation_id: str | None = None,
) -> DifferenceAuditRecord:
    """Create an audit record for classifications."""

    try:
        if not isinstance(classifications, (tuple, list)):
            raise TypeError("classifications must be a tuple or list")
        classifications_tuple = tuple(classifications)
        for classification in classifications_tuple:
            if not isinstance(classification, DifferenceClassification):
                raise TypeError("classifications must contain DifferenceClassification")

        return create_audit_record(
            event_type=DifferenceAuditEventType.CANDIDATE_CLASSIFIED,
            observation_id=observation_id,
            status="CLASSIFIED",
            payload={
                "classification_count": len(classifications_tuple),
                "severity_counts": _classification_severity_counts(classifications_tuple),
                "kind_counts": _classification_kind_counts(classifications_tuple),
            },
        )
    except DifferenceAuditError:
        raise
    except Exception as exc:
        raise DifferenceAuditError(
            message="failed to audit classifications",
            context=build_error_context(
                operation="audit_classifications",
                observation_id=observation_id,
            ),
            cause=exc,
        ) from exc


def audit_aggregate_summary(summary: DifferenceAggregateSummary) -> DifferenceAuditRecord:
    """Create an audit record for aggregate summary."""

    try:
        if not isinstance(summary, DifferenceAggregateSummary):
            raise TypeError("summary must be DifferenceAggregateSummary")

        return create_audit_record(
            event_type=DifferenceAuditEventType.SUMMARY_AGGREGATED,
            observation_id=summary.observation_id,
            status="SUMMARY",
            payload={
                "total_candidates": summary.total_candidates,
                "total_classifications": summary.total_classifications,
                "bucket_count": len(summary.buckets),
                "max_severity": (
                    summary.max_severity.value
                    if summary.max_severity is not None
                    else None
                ),
                "mean_confidence": summary.mean_confidence,
                "max_magnitude": summary.max_magnitude,
            },
        )
    except DifferenceAuditError:
        raise
    except Exception as exc:
        raise DifferenceAuditError(
            message="failed to audit aggregate summary",
            context=build_error_context(operation="audit_aggregate_summary"),
            cause=exc,
        ) from exc


def audit_validation_report(
    report: ValidationReport,
    *,
    observation_id: str | None = None,
    set_id: str | None = None,
) -> DifferenceAuditRecord:
    """Create an audit record for a validation report."""

    try:
        if not isinstance(report, ValidationReport):
            raise TypeError("report must be ValidationReport")

        return create_audit_record(
            event_type=DifferenceAuditEventType.SET_VALIDATED,
            observation_id=observation_id,
            set_id=set_id,
            status=report.status.value,
            payload={
                "checked_count": report.checked_count,
                "issue_count": len(report.issues),
                "failed": report.failed,
                "passed": report.passed,
                "issue_codes": tuple(validation_issue.code for validation_issue in report.issues),
            },
        )
    except DifferenceAuditError:
        raise
    except Exception as exc:
        raise DifferenceAuditError(
            message="failed to audit validation report",
            context=build_error_context(
                operation="audit_validation_report",
                observation_id=observation_id,
            ),
            cause=exc,
        ) from exc


def audit_dispatch_envelope(envelope: DifferenceDispatchEnvelope) -> DifferenceAuditRecord:
    """Create an audit record for a dispatch envelope."""

    try:
        if not isinstance(envelope, DifferenceDispatchEnvelope):
            raise TypeError("envelope must be DifferenceDispatchEnvelope")

        if (
            envelope.status is DifferenceDispatchStatus.DEAD_LETTER
            or envelope.target is DifferenceDispatchTarget.DEAD_LETTER
        ):
            event_type = DifferenceAuditEventType.DISPATCH_DEAD_LETTER
        elif envelope.status is DifferenceDispatchStatus.READY:
            event_type = DifferenceAuditEventType.DISPATCH_READY
        else:
            event_type = DifferenceAuditEventType.DISPATCH_ENVELOPE_CREATED

        return create_audit_record(
            event_type=event_type,
            observation_id=envelope.observation_id,
            set_id=str(envelope.difference_set.set_id),
            envelope_id=envelope.envelope_id,
            status=envelope.status.value,
            payload={
                "target": envelope.target.value,
                "validation_status": envelope.validation_report.status.value,
                "classification_count": len(envelope.classifications),
                "total_candidates": envelope.summary.total_candidates,
                "max_severity": (
                    envelope.summary.max_severity.value
                    if envelope.summary.max_severity is not None
                    else None
                ),
            },
        )
    except DifferenceAuditError:
        raise
    except Exception as exc:
        raise DifferenceAuditError(
            message="failed to audit dispatch envelope",
            context=build_error_context(operation="audit_dispatch_envelope"),
            cause=exc,
        ) from exc


def audit_pipeline_error(
    *,
    operation: str,
    error: BaseException,
    observation_id: str | None = None,
    difference_id: str | None = None,
    set_id: str | None = None,
    envelope_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> DifferenceAuditRecord:
    """Create an audit record for a pipeline error."""

    try:
        _ensure_non_empty_string(operation, "operation")
        if not isinstance(error, BaseException):
            raise TypeError("error must be BaseException")

        return create_audit_record(
            event_type=DifferenceAuditEventType.PIPELINE_ERROR,
            observation_id=observation_id,
            difference_id=difference_id,
            set_id=set_id,
            envelope_id=envelope_id,
            status="ERROR",
            payload={
                "operation": operation,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
            metadata={} if metadata is None else metadata,
        )
    except DifferenceAuditError:
        raise
    except Exception as exc:
        raise DifferenceAuditError(
            message="failed to audit pipeline error",
            context=build_error_context(
                operation="audit_pipeline_error",
                observation_id=observation_id,
                difference_id=difference_id,
            ),
            cause=exc,
        ) from exc


def build_audit_trail(
    *,
    observation_id: str | None = None,
    records: tuple[DifferenceAuditRecord, ...] | list[DifferenceAuditRecord],
    metadata: dict[str, object] | None = None,
) -> DifferenceAuditTrail:
    """Build an ordered immutable audit trail."""

    try:
        _ensure_optional_non_empty_string(observation_id, "observation_id")
        if not isinstance(records, (tuple, list)):
            raise TypeError("records must be a tuple or list")
        records_tuple = tuple(records)
        for record in records_tuple:
            if not isinstance(record, DifferenceAuditRecord):
                raise TypeError("records must contain DifferenceAuditRecord")
        resolved_metadata = {} if metadata is None else metadata
        _ensure_metadata_dict(resolved_metadata, "metadata")

        return DifferenceAuditTrail(
            trail_id=_build_trail_id(
                observation_id=observation_id,
                records=records_tuple,
            ),
            observation_id=observation_id,
            records=records_tuple,
            created_at=utc_now(),
            metadata=resolved_metadata,
        )
    except DifferenceAuditError:
        raise
    except Exception as exc:
        raise DifferenceAuditError(
            message="failed to build audit trail",
            context=build_error_context(
                operation="build_audit_trail",
                observation_id=observation_id,
            ),
            cause=exc,
        ) from exc


def audit_full_dispatch_package(
    *,
    difference_set: DifferenceSet,
    classifications: tuple[DifferenceClassification, ...] | list[DifferenceClassification],
    summary: DifferenceAggregateSummary,
    validation_report: ValidationReport,
    envelope: DifferenceDispatchEnvelope,
) -> DifferenceAuditTrail:
    """Create a full audit trail for a dispatch-ready difference package."""

    try:
        if not isinstance(difference_set, DifferenceSet):
            raise TypeError("difference_set must be DifferenceSet")

        records = (
            audit_difference_set(difference_set),
            audit_classifications(
                classifications,
                observation_id=difference_set.observation_id,
            ),
            audit_aggregate_summary(summary),
            audit_validation_report(
                validation_report,
                observation_id=difference_set.observation_id,
                set_id=str(difference_set.set_id),
            ),
            audit_dispatch_envelope(envelope),
        )
        return build_audit_trail(
            observation_id=difference_set.observation_id,
            records=records,
            metadata={"audit_scope": "full_dispatch_package"},
        )
    except DifferenceAuditError:
        raise
    except Exception as exc:
        raise DifferenceAuditError(
            message="failed to audit full dispatch package",
            context=build_error_context(operation="audit_full_dispatch_package"),
            cause=exc,
        ) from exc


def _build_trail_id(
    *,
    observation_id: str | None,
    records: tuple[DifferenceAuditRecord, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update((observation_id or "").encode("utf-8"))
    digest.update(b"\x00")
    for record in records:
        digest.update(record.audit_id.encode("utf-8"))
        digest.update(b"\x00")
    return "trail_" + digest.hexdigest()[:32]


def _stable_payload_repr(payload: dict[str, object]) -> str:
    return repr(tuple(sorted(payload.items(), key=lambda item: item[0])))


def _candidate_status_counts(
    candidates: tuple[DifferenceCandidate, ...],
) -> dict[str, int]:
    counts = {status.value: 0 for status in DifferenceStatus}
    for candidate in candidates:
        if not isinstance(candidate, DifferenceCandidate):
            raise TypeError("candidates must contain DifferenceCandidate")
        counts[candidate.status.value] = counts.get(candidate.status.value, 0) + 1
    return {key: value for key, value in sorted(counts.items()) if value}


def _candidate_kind_counts(
    candidates: tuple[DifferenceCandidate, ...],
) -> dict[str, int]:
    counts = {kind.value: 0 for kind in DifferenceKind}
    for candidate in candidates:
        if not isinstance(candidate, DifferenceCandidate):
            raise TypeError("candidates must contain DifferenceCandidate")
        counts[candidate.kind.value] = counts.get(candidate.kind.value, 0) + 1
    return {key: value for key, value in sorted(counts.items()) if value}


def _classification_severity_counts(
    classifications: tuple[DifferenceClassification, ...],
) -> dict[str, int]:
    counts = {severity.value: 0 for severity in DifferenceSeverity}
    for classification in classifications:
        if not isinstance(classification, DifferenceClassification):
            raise TypeError("classifications must contain DifferenceClassification")
        counts[classification.severity.value] = (
            counts.get(classification.severity.value, 0) + 1
        )
    return {key: value for key, value in sorted(counts.items()) if value}


def _classification_kind_counts(
    classifications: tuple[DifferenceClassification, ...],
) -> dict[str, int]:
    counts = {kind.value: 0 for kind in DifferenceKind}
    for classification in classifications:
        if not isinstance(classification, DifferenceClassification):
            raise TypeError("classifications must contain DifferenceClassification")
        counts[classification.kind.value] = counts.get(classification.kind.value, 0) + 1
    return {key: value for key, value in sorted(counts.items()) if value}


def _ensure_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(field_name, str) or not field_name:
        raise ValueError("field_name must be a non-empty string")
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _ensure_optional_non_empty_string(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return _ensure_non_empty_string(value, field_name)


def _ensure_metadata_dict(value: dict[str, object], field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a dict")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field_name} keys must be strings")
    return value


__all__ = [
    "DifferenceAuditEventType",
    "DifferenceAuditRecord",
    "DifferenceAuditTrail",
    "build_audit_id",
    "create_audit_record",
    "audit_candidate",
    "audit_difference_set",
    "audit_classifications",
    "audit_aggregate_summary",
    "audit_validation_report",
    "audit_dispatch_envelope",
    "audit_pipeline_error",
    "build_audit_trail",
    "audit_full_dispatch_package",
]
