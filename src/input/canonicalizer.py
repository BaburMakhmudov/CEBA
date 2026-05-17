"""CEBA Input canonicalization utilities.

Canonicalization creates CanonicalObservation values from validated input
envelopes. It preserves technical input structure and provenance without
computing meaning.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import JsonValue, ValidationError

from ois.input.errors import CanonicalizationError, SchemaViolationError
from ois.input.fingerprint import (
    compute_canonical_content_hash,
    compute_canonical_observation_structural_hash,
)
from ois.input.normalization import (
    build_observation_metadata,
    normalize_datetime,
    normalize_encoding,
    normalize_language_hint,
    normalize_mime_type,
    normalize_text,
)
from ois.input.schema import (
    CanonicalContent,
    CanonicalContentKind,
    CanonicalObservation,
    InputEnvelope,
    ObservationMetadata,
    ObservationType,
    PayloadRef,
    SecurityReport,
    SecurityStatus,
    ValidationReport,
)
from ois.input.validation import (
    passed_report,
    require_valid_canonical_observation,
    require_valid_input_envelope,
)


_SCHEMA_VERSION = "1.0.0"
_DEFAULT_ENCODING = "utf-8"


def canonicalize_envelope(
    *,
    envelope: InputEnvelope,
    observation_type: ObservationType,
    occurred_at: datetime | None = None,
    recorded_at: datetime | None = None,
    language: str | None = None,
    validation_report: ValidationReport | None = None,
    security_report: SecurityReport | None = None,
) -> CanonicalObservation:
    envelope = _ensure_envelope(envelope)
    require_valid_input_envelope(envelope)
    observation_type = _ensure_observation_type(observation_type)

    try:
        normalized_occurred_at = _normalize_event_time(
            occurred_at,
            fallback=envelope.received_at,
            field_path="occurred_at",
        )
        normalized_recorded_at = _normalize_event_time(
            recorded_at,
            fallback=_utc_now(),
            field_path="recorded_at",
        )
        normalized_language = normalize_language_hint(language)
        normalized_encoding = normalize_encoding(_DEFAULT_ENCODING)
        content = build_canonical_content(
            envelope=envelope,
            observation_type=observation_type,
        )
        metadata = build_metadata_from_envelope(
            envelope=envelope,
            content=content,
            language_hint=normalized_language,
        )
        report = validation_report if validation_report is not None else passed_report()
        safety = (
            security_report
            if security_report is not None
            else SecurityReport(status=SecurityStatus.PASSED)
        )

        observation_values: dict[str, Any] = {
            "trace_id": envelope.trace_id,
            "parent_trace_id": envelope.parent_trace_id,
            "source_type": envelope.source.source_type,
            "observation_type": observation_type,
            "content": content,
            "metadata": metadata,
            "provenance": envelope.provenance,
            "occurred_at": normalized_occurred_at,
            "recorded_at": normalized_recorded_at,
            "language": normalized_language,
            "encoding": normalized_encoding,
            "safety_labels": safety.labels,
            "validation_report": report,
            "security_report": safety,
            "schema_version": _SCHEMA_VERSION,
        }
        temp_observation = CanonicalObservation(
            **observation_values,
            content_hash=_placeholder_hash(),
            structural_hash=_placeholder_hash(),
        )
        content_hash = compute_canonical_content_hash(temp_observation)
        second_temp = CanonicalObservation(
            **observation_values,
            observation_id=temp_observation.observation_id,
            content_hash=content_hash,
            structural_hash=_placeholder_hash(),
        )
        structural_hash = compute_canonical_observation_structural_hash(second_temp)
        final_observation = CanonicalObservation(
            **observation_values,
            observation_id=temp_observation.observation_id,
            content_hash=content_hash,
            structural_hash=structural_hash,
        )
    except ValidationError as error:
        raise CanonicalizationError(
            "failed to canonicalize input envelope",
            field_path="canonical_observation",
            details={"errors": error.errors()},
        ) from error

    require_valid_canonical_observation(final_observation)
    return final_observation


def build_canonical_content(
    *,
    envelope: InputEnvelope,
    observation_type: ObservationType,
) -> CanonicalContent:
    envelope = _ensure_envelope(envelope)
    observation_type = _ensure_observation_type(observation_type)

    if envelope.payload_ref is not None:
        return CanonicalContent(
            kind=CanonicalContentKind.BINARY_REF,
            payload_ref=envelope.payload_ref,
            normalized=True,
        )

    inline_payload = envelope.inline_payload
    kind = canonical_content_kind_for_observation(
        observation_type=observation_type,
        has_payload_ref=False,
        inline_payload=inline_payload,
    )

    if isinstance(inline_payload, str):
        return CanonicalContent(
            kind=kind,
            text=normalize_text(inline_payload),
            normalized=True,
        )
    if isinstance(inline_payload, dict):
        return CanonicalContent(
            kind=kind,
            data=inline_payload,
            normalized=True,
        )
    if isinstance(inline_payload, (list, int, float, bool)):
        return CanonicalContent(
            kind=kind,
            data={"value": inline_payload},
            normalized=True,
        )
    if inline_payload is None:
        raise CanonicalizationError(
            "inline_payload is required when payload_ref is absent",
            field_path="inline_payload",
        )
    raise CanonicalizationError(
        "unsupported inline_payload type",
        field_path="inline_payload",
        details={"type": type(inline_payload).__name__},
    )


def build_metadata_from_envelope(
    *,
    envelope: InputEnvelope,
    content: CanonicalContent,
    language_hint: str | None = None,
) -> ObservationMetadata:
    envelope = _ensure_envelope(envelope)
    if not isinstance(content, CanonicalContent):
        raise SchemaViolationError(
            "content must be CanonicalContent",
            field_path="content",
            details={"type": type(content).__name__},
        )

    payload_ref: PayloadRef | None = envelope.payload_ref
    mime_type: str | None
    if payload_ref is not None and payload_ref.mime_type:
        mime_type = payload_ref.mime_type
    elif content.text is not None and content.kind in {
        CanonicalContentKind.TEXT,
        CanonicalContentKind.COMMAND,
        CanonicalContentKind.ERROR,
    }:
        mime_type = "text/plain"
    elif content.data is not None and content.kind in {
        CanonicalContentKind.JSON,
        CanonicalContentKind.EVENT,
    }:
        mime_type = "application/json"
    else:
        mime_type = payload_ref.mime_type if payload_ref is not None else None

    length_chars = len(content.text) if content.text is not None else None
    if payload_ref is not None and payload_ref.size_bytes is not None:
        size_bytes = payload_ref.size_bytes
    elif content.text is not None:
        size_bytes = len(content.text.encode(_DEFAULT_ENCODING))
    else:
        size_bytes = None

    return build_observation_metadata(
        mime_type=normalize_mime_type(mime_type),
        encoding=normalize_encoding(_DEFAULT_ENCODING),
        language_hint=language_hint,
        length_chars=length_chars,
        size_bytes=size_bytes,
        source_metadata=envelope.source.metadata,
        transport_metadata={},
    )


def canonical_content_kind_for_observation(
    *,
    observation_type: ObservationType,
    has_payload_ref: bool,
    inline_payload: JsonValue | None,
) -> CanonicalContentKind:
    observation_type = _ensure_observation_type(observation_type)
    if has_payload_ref:
        return CanonicalContentKind.BINARY_REF
    if isinstance(inline_payload, str):
        if observation_type in {
            ObservationType.COMMAND_INPUT,
            ObservationType.COMMAND_OUTPUT,
        }:
            return CanonicalContentKind.COMMAND
        if observation_type == ObservationType.ERROR_EVENT:
            return CanonicalContentKind.ERROR
        return CanonicalContentKind.TEXT
    if isinstance(inline_payload, dict):
        if observation_type == ObservationType.ERROR_EVENT:
            return CanonicalContentKind.ERROR
        if observation_type in {
            ObservationType.API_EVENT,
            ObservationType.WEBHOOK_EVENT,
            ObservationType.SYSTEM_EVENT,
            ObservationType.LOG_EVENT,
            ObservationType.DATABASE_ROW_CHANGE,
            ObservationType.USER_ACTION,
        }:
            return CanonicalContentKind.EVENT
        return CanonicalContentKind.JSON
    if isinstance(inline_payload, (list, int, float, bool)):
        if observation_type == ObservationType.ERROR_EVENT:
            return CanonicalContentKind.ERROR
        return CanonicalContentKind.JSON
    if inline_payload is None:
        raise CanonicalizationError(
            "inline_payload is required when payload_ref is absent",
            field_path="inline_payload",
        )
    raise CanonicalizationError(
        "unsupported inline_payload type",
        field_path="inline_payload",
        details={"type": type(inline_payload).__name__},
    )


def canonicalize_text_envelope(
    *,
    envelope: InputEnvelope,
    observation_type: ObservationType = ObservationType.TEXT_MESSAGE,
    language: str | None = None,
) -> CanonicalObservation:
    return canonicalize_envelope(
        envelope=envelope,
        observation_type=observation_type,
        language=language,
    )


def canonicalize_error_envelope(
    *,
    envelope: InputEnvelope,
) -> CanonicalObservation:
    return canonicalize_envelope(
        envelope=envelope,
        observation_type=ObservationType.ERROR_EVENT,
    )


def _ensure_envelope(value: InputEnvelope) -> InputEnvelope:
    if not isinstance(value, InputEnvelope):
        raise SchemaViolationError(
            "value must be InputEnvelope",
            field_path="envelope",
            details={"type": type(value).__name__},
        )
    return value


def _ensure_observation_type(value: ObservationType) -> ObservationType:
    if not isinstance(value, ObservationType):
        raise SchemaViolationError(
            "value must be ObservationType",
            field_path="observation_type",
            details={"type": type(value).__name__},
        )
    return value


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_event_time(
    value: datetime | None,
    *,
    fallback: datetime,
    field_path: str,
) -> datetime:
    return normalize_datetime(
        fallback if value is None else value,
        field_path=field_path,
    )


def _placeholder_hash() -> str:
    return "sha256:" + "0" * 64


__all__ = [
    "build_canonical_content",
    "build_metadata_from_envelope",
    "canonical_content_kind_for_observation",
    "canonicalize_envelope",
    "canonicalize_error_envelope",
    "canonicalize_text_envelope",
]
