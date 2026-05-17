"""CEBA Input audit utilities.

Audit records preserve technical traceability across input-layer objects and
errors. This module constructs audit payloads only; it does not persist records
or compute meaning.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import ValidationError

from ois.input.errors import CEBAInputError, AuditWriteError, ensure_input_error
from ois.input.schema import (
    CanonicalObservation,
    InputAuditRecord,
    InputEnvelope,
    InputSourceType,
    ObservationType,
    RawObservation,
    SecurityReport,
    SecurityStatus,
    ValidationReport,
    ValidationStatus,
)


_AUDIT_SCHEMA_VERSION = "1.0.0"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def build_audit_record_from_raw(
    *,
    raw: RawObservation,
    validation_status: ValidationStatus = ValidationStatus.PENDING,
    security_status: SecurityStatus = SecurityStatus.PENDING,
    error_code: str | None = None,
    created_at: datetime | None = None,
) -> InputAuditRecord:
    normalized_raw = _ensure_raw(raw)
    timestamp = _timestamp(created_at)
    try:
        return InputAuditRecord(
            trace_id=normalized_raw.trace_id,
            observation_id=None,
            source_type=normalized_raw.source_type,
            observation_type=None,
            content_hash=None,
            structural_hash=None,
            adapter_name=normalized_raw.adapter_name,
            adapter_version=normalized_raw.adapter_version,
            validation_status=validation_status,
            security_status=security_status,
            received_at=normalized_raw.received_at,
            recorded_at=timestamp,
            error_code=error_code,
            created_at=timestamp,
        )
    except ValidationError as error:
        raise AuditWriteError(
            "failed to build audit record from raw observation",
            trace_id=normalized_raw.trace_id,
            field_path="raw",
            details={"errors": error.errors()},
        ) from error


def build_audit_record_from_envelope(
    *,
    envelope: InputEnvelope,
    validation_status: ValidationStatus | None = None,
    security_status: SecurityStatus | None = None,
    error_code: str | None = None,
    created_at: datetime | None = None,
) -> InputAuditRecord:
    normalized_envelope = _ensure_envelope(envelope)
    timestamp = _timestamp(created_at)
    try:
        return InputAuditRecord(
            trace_id=normalized_envelope.trace_id,
            observation_id=None,
            source_type=normalized_envelope.source.source_type,
            observation_type=None,
            content_hash=None,
            structural_hash=None,
            adapter_name=normalized_envelope.provenance.adapter_name,
            adapter_version=normalized_envelope.provenance.adapter_version,
            validation_status=(
                validation_status
                if validation_status is not None
                else normalized_envelope.validation_status
            ),
            security_status=(
                security_status
                if security_status is not None
                else normalized_envelope.security_status
            ),
            received_at=normalized_envelope.received_at,
            recorded_at=timestamp,
            error_code=error_code,
            created_at=timestamp,
        )
    except ValidationError as error:
        raise AuditWriteError(
            "failed to build audit record from input envelope",
            trace_id=normalized_envelope.trace_id,
            raw_id=normalized_envelope.raw_id,
            field_path="envelope",
            details={"errors": error.errors()},
        ) from error


def build_audit_record_from_observation(
    *,
    observation: CanonicalObservation,
    error_code: str | None = None,
    created_at: datetime | None = None,
) -> InputAuditRecord:
    normalized_observation = _ensure_observation(observation)
    try:
        return InputAuditRecord(
            trace_id=normalized_observation.trace_id,
            observation_id=normalized_observation.observation_id,
            source_type=normalized_observation.source_type,
            observation_type=normalized_observation.observation_type,
            content_hash=normalized_observation.content_hash,
            structural_hash=normalized_observation.structural_hash,
            adapter_name=normalized_observation.provenance.adapter_name,
            adapter_version=normalized_observation.provenance.adapter_version,
            validation_status=normalized_observation.validation_report.status,
            security_status=normalized_observation.security_report.status,
            received_at=normalized_observation.provenance.received_at,
            recorded_at=normalized_observation.recorded_at,
            error_code=error_code,
            created_at=_timestamp(created_at),
        )
    except ValidationError as error:
        raise AuditWriteError(
            "failed to build audit record from canonical observation",
            trace_id=normalized_observation.trace_id,
            observation_id=normalized_observation.observation_id,
            field_path="observation",
            details={"errors": error.errors()},
        ) from error


def build_audit_record_from_error(
    *,
    error: Exception,
    raw: RawObservation | None = None,
    envelope: InputEnvelope | None = None,
    observation: CanonicalObservation | None = None,
    created_at: datetime | None = None,
) -> InputAuditRecord:
    input_error: CEBAInputError = ensure_input_error(error)
    if observation is not None:
        return build_audit_record_from_observation(
            observation=observation,
            error_code=input_error.code,
            created_at=created_at,
        )
    if envelope is not None:
        return build_audit_record_from_envelope(
            envelope=envelope,
            error_code=input_error.code,
            created_at=created_at,
        )
    if raw is not None:
        return build_audit_record_from_raw(
            raw=raw,
            error_code=input_error.code,
            created_at=created_at,
        )

    timestamp = _timestamp(created_at)
    try:
        return InputAuditRecord(
            trace_id=input_error.trace_id or uuid4(),
            observation_id=input_error.observation_id,
            source_type=InputSourceType.SYSTEM,
            observation_type=ObservationType.ERROR_EVENT,
            adapter_name="input_error",
            adapter_version=_AUDIT_SCHEMA_VERSION,
            validation_status=ValidationStatus.FAILED,
            security_status=SecurityStatus.WARNING,
            received_at=timestamp,
            recorded_at=timestamp,
            error_code=input_error.code,
            created_at=timestamp,
        )
    except ValidationError as validation_error:
        raise AuditWriteError(
            "failed to build audit record from input error",
            trace_id=input_error.trace_id,
            observation_id=input_error.observation_id,
            field_path="error",
            details={"errors": validation_error.errors()},
        ) from validation_error


def validation_status_from_report(report: ValidationReport) -> ValidationStatus:
    normalized_report = _ensure_validation_report(report)
    return normalized_report.status


def security_status_from_report(report: SecurityReport) -> SecurityStatus:
    normalized_report = _ensure_security_report(report)
    return normalized_report.status


def audit_record_to_dict(record: InputAuditRecord) -> dict[str, object]:
    normalized_record = _ensure_audit_record(record)
    return normalized_record.model_dump(mode="json")


def _ensure_raw(value: RawObservation) -> RawObservation:
    if not isinstance(value, RawObservation):
        raise AuditWriteError(
            "raw must be RawObservation",
            field_path="raw",
            details={"type": type(value).__name__},
        )
    return value


def _ensure_envelope(value: InputEnvelope) -> InputEnvelope:
    if not isinstance(value, InputEnvelope):
        raise AuditWriteError(
            "envelope must be InputEnvelope",
            field_path="envelope",
            details={"type": type(value).__name__},
        )
    return value


def _ensure_observation(value: CanonicalObservation) -> CanonicalObservation:
    if not isinstance(value, CanonicalObservation):
        raise AuditWriteError(
            "observation must be CanonicalObservation",
            field_path="observation",
            details={"type": type(value).__name__},
        )
    return value


def _ensure_validation_report(value: ValidationReport) -> ValidationReport:
    if not isinstance(value, ValidationReport):
        raise AuditWriteError(
            "report must be ValidationReport",
            field_path="report",
            details={"type": type(value).__name__},
        )
    return value


def _ensure_security_report(value: SecurityReport) -> SecurityReport:
    if not isinstance(value, SecurityReport):
        raise AuditWriteError(
            "report must be SecurityReport",
            field_path="report",
            details={"type": type(value).__name__},
        )
    return value


def _ensure_audit_record(value: InputAuditRecord) -> InputAuditRecord:
    if not isinstance(value, InputAuditRecord):
        raise AuditWriteError(
            "record must be InputAuditRecord",
            field_path="record",
            details={"type": type(value).__name__},
        )
    return value


def _timestamp(value: datetime | None) -> datetime:
    return value if value is not None else utc_now()


__all__ = [
    "audit_record_to_dict",
    "build_audit_record_from_envelope",
    "build_audit_record_from_error",
    "build_audit_record_from_observation",
    "build_audit_record_from_raw",
    "security_status_from_report",
    "utc_now",
    "validation_status_from_report",
]
