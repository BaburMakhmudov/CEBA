"""CEBA Input envelope utilities.

Input envelopes wrap raw observations with source, provenance, and payload
routing details. They preserve traceability without computing meaning.
"""

from __future__ import annotations

from pydantic import JsonValue, ValidationError

from ois.input.errors import SchemaViolationError, ValidationFailedError
from ois.input.provenance import validate_envelope_provenance, validate_provenance
from ois.input.schema import (
    InputEnvelope,
    InputSource,
    PayloadRef,
    PayloadStorageKind,
    Provenance,
    RawObservation,
    SecurityStatus,
    ValidationStatus,
)
from ois.input.source import build_input_source, validate_source_boundary
from ois.input.validation import validate_input_envelope


_INPUT_SCHEMA_VERSION = "1.0.0"


def build_input_envelope(
    *,
    raw: RawObservation,
    provenance: Provenance,
    payload_ref: PayloadRef | None = None,
    inline_payload: JsonValue | None = None,
    source: InputSource | None = None,
    validation_status: ValidationStatus = ValidationStatus.PENDING,
    security_status: SecurityStatus = SecurityStatus.PENDING,
    input_schema_version: str = _INPUT_SCHEMA_VERSION,
) -> InputEnvelope:
    raw = _ensure_raw(raw)
    provenance = _ensure_provenance(provenance)
    validate_provenance(provenance)

    if source is None:
        source = build_input_source(
            source_type=raw.source_type,
            source_name=raw.source_name,
            source_id=None,
            trust_level=raw.trust_level,
            metadata=raw.source_metadata,
        )
    else:
        validate_source_boundary(source)

    if payload_ref is None and inline_payload is None:
        raise SchemaViolationError(
            "payload_ref or inline_payload is required",
            field_path="payload",
        )
    if payload_ref is not None and inline_payload is not None:
        raise SchemaViolationError(
            "payload_ref and inline_payload cannot both be provided",
            field_path="payload",
        )
    if payload_ref is not None:
        payload_ref = _ensure_payload_ref(payload_ref)

    try:
        envelope = InputEnvelope(
            raw_id=raw.raw_id,
            trace_id=raw.trace_id,
            parent_trace_id=raw.parent_trace_id,
            source=source,
            provenance=provenance,
            payload_ref=payload_ref,
            inline_payload=inline_payload,
            received_at=raw.received_at,
            normalized_at=None,
            validation_status=validation_status,
            security_status=security_status,
            input_schema_version=_validate_schema_version(input_schema_version),
        )
    except ValidationError as error:
        raise SchemaViolationError(
            "failed to build input envelope",
            field_path="envelope",
            details={"errors": error.errors()},
        ) from error

    validate_envelope_provenance(envelope)
    report = validate_input_envelope(envelope)
    if not report.is_valid:
        raise ValidationFailedError(
            "input envelope validation failed",
            details={"report": report.model_dump(mode="json")},
        )
    return envelope


def build_inline_envelope(
    *,
    raw: RawObservation,
    provenance: Provenance,
    inline_payload: JsonValue | None = None,
    source: InputSource | None = None,
) -> InputEnvelope:
    raw = _ensure_raw(raw)
    return build_input_envelope(
        raw=raw,
        provenance=provenance,
        inline_payload=raw.raw_payload if inline_payload is None else inline_payload,
        source=source,
    )


def build_referenced_envelope(
    *,
    raw: RawObservation,
    provenance: Provenance,
    payload_ref: PayloadRef,
    source: InputSource | None = None,
) -> InputEnvelope:
    return build_input_envelope(
        raw=raw,
        provenance=provenance,
        payload_ref=_ensure_payload_ref(payload_ref),
        source=source,
    )


def envelope_has_inline_payload(envelope: InputEnvelope) -> bool:
    return _ensure_envelope(envelope).inline_payload is not None


def envelope_has_payload_ref(envelope: InputEnvelope) -> bool:
    return _ensure_envelope(envelope).payload_ref is not None


def require_envelope_payload(envelope: InputEnvelope) -> None:
    envelope = _ensure_envelope(envelope)
    has_inline = envelope.inline_payload is not None
    has_ref = envelope.payload_ref is not None
    if has_inline == has_ref:
        raise SchemaViolationError(
            "exactly one of inline_payload or payload_ref is required",
            field_path="payload",
        )


def require_payload_ref(envelope: InputEnvelope) -> PayloadRef:
    envelope = _ensure_envelope(envelope)
    if envelope.payload_ref is None:
        raise SchemaViolationError(
            "payload_ref is required",
            field_path="payload_ref",
        )
    return envelope.payload_ref


def require_inline_payload(envelope: InputEnvelope) -> JsonValue:
    envelope = _ensure_envelope(envelope)
    if envelope.inline_payload is None:
        raise SchemaViolationError(
            "inline_payload is required",
            field_path="inline_payload",
        )
    return envelope.inline_payload


def with_validation_status(
    envelope: InputEnvelope,
    status: ValidationStatus,
) -> InputEnvelope:
    updated = _ensure_envelope(envelope).model_copy(
        update={"validation_status": status}
    )
    require_valid_envelope(updated)
    return updated


def with_security_status(
    envelope: InputEnvelope,
    status: SecurityStatus,
) -> InputEnvelope:
    updated = _ensure_envelope(envelope).model_copy(update={"security_status": status})
    require_valid_envelope(updated)
    return updated


def mark_envelope_validated(envelope: InputEnvelope) -> InputEnvelope:
    return with_validation_status(envelope, ValidationStatus.PASSED)


def mark_envelope_validation_failed(envelope: InputEnvelope) -> InputEnvelope:
    return with_validation_status(envelope, ValidationStatus.FAILED)


def mark_envelope_security_passed(envelope: InputEnvelope) -> InputEnvelope:
    return with_security_status(envelope, SecurityStatus.PASSED)


def mark_envelope_security_blocked(envelope: InputEnvelope) -> InputEnvelope:
    return with_security_status(envelope, SecurityStatus.BLOCKED)


def envelope_payload_storage_kind(envelope: InputEnvelope) -> PayloadStorageKind:
    envelope = _ensure_envelope(envelope)
    if envelope.payload_ref is not None:
        return envelope.payload_ref.storage_kind
    if envelope.inline_payload is not None:
        return PayloadStorageKind.INLINE
    raise SchemaViolationError(
        "envelope payload is required",
        field_path="payload",
    )


def require_valid_envelope(envelope: InputEnvelope) -> None:
    envelope = _ensure_envelope(envelope)
    require_envelope_payload(envelope)
    validate_envelope_provenance(envelope)
    report = validate_input_envelope(envelope)
    if not report.is_valid:
        raise ValidationFailedError(
            "input envelope validation failed",
            details={"report": report.model_dump(mode="json")},
        )


def _ensure_envelope(value: InputEnvelope) -> InputEnvelope:
    if not isinstance(value, InputEnvelope):
        raise SchemaViolationError(
            "value must be InputEnvelope",
            field_path="envelope",
            details={"type": type(value).__name__},
        )
    return value


def _ensure_raw(value: RawObservation) -> RawObservation:
    if not isinstance(value, RawObservation):
        raise SchemaViolationError(
            "value must be RawObservation",
            field_path="raw",
            details={"type": type(value).__name__},
        )
    return value


def _ensure_provenance(value: Provenance) -> Provenance:
    if not isinstance(value, Provenance):
        raise SchemaViolationError(
            "value must be Provenance",
            field_path="provenance",
            details={"type": type(value).__name__},
        )
    return value


def _ensure_payload_ref(value: PayloadRef) -> PayloadRef:
    if not isinstance(value, PayloadRef):
        raise SchemaViolationError(
            "value must be PayloadRef",
            field_path="payload_ref",
            details={"type": type(value).__name__},
        )
    return value


def _validate_schema_version(value: str) -> str:
    if not isinstance(value, str):
        raise SchemaViolationError(
            "input_schema_version must be a string",
            field_path="input_schema_version",
            details={"type": type(value).__name__},
        )
    normalized = value.strip()
    if not normalized:
        raise SchemaViolationError(
            "input_schema_version must not be empty",
            field_path="input_schema_version",
        )
    return normalized


__all__ = [
    "build_input_envelope",
    "build_inline_envelope",
    "build_referenced_envelope",
    "envelope_has_inline_payload",
    "envelope_has_payload_ref",
    "envelope_payload_storage_kind",
    "mark_envelope_security_blocked",
    "mark_envelope_security_passed",
    "mark_envelope_validated",
    "mark_envelope_validation_failed",
    "require_envelope_payload",
    "require_inline_payload",
    "require_payload_ref",
    "require_valid_envelope",
    "with_security_status",
    "with_validation_status",
]
