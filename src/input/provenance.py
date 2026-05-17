"""CEBA Input provenance utilities.

Provenance preserves origin, capture details, and the technical chain that
produced an input observation. It records traceable capture history without
computing meaning.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from ois.input.errors import ProvenanceInvalidError, ProvenanceMissingError
from ois.input.schema import InputEnvelope, Provenance, ProvenanceStep, RawObservation
from ois.input.source import normalize_source_name


_MAX_PROVENANCE_CHAIN_LENGTH = 128
_MAX_STEP_NAME_LENGTH = 128
_MAX_COMPONENT_NAME_LENGTH = 128
_MAX_METADATA_KEYS = 64
_MAX_METADATA_KEY_LENGTH = 128
_MAX_METADATA_STRING_VALUE_LENGTH = 2048

_ALLOWED_METADATA_VALUE_TYPES = (str, int, float, bool, type(None))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def sanitize_provenance_metadata(
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise ProvenanceInvalidError(
            "provenance metadata must be a mapping",
            field_path="metadata",
            details={"type": type(metadata).__name__},
        )
    if len(metadata) > _MAX_METADATA_KEYS:
        raise ProvenanceInvalidError(
            "provenance metadata exceeds maximum key count",
            field_path="metadata",
            details={"max_keys": _MAX_METADATA_KEYS, "key_count": len(metadata)},
        )

    sanitized: dict[str, Any] = {}
    for raw_key, raw_value in metadata.items():
        key = _normalize_metadata_key(raw_key)
        sanitized[key] = _normalize_metadata_value(raw_value, key)
    return sanitized


def build_provenance(
    *,
    raw: RawObservation,
    capture_method: str,
    content_hash: str | None = None,
    envelope_hash: str | None = None,
    actor_id: str | None = None,
    actor_kind: str | None = None,
    origin_uri: str | None = None,
    origin_host: str | None = None,
    step_name: str = "capture",
    component: str | None = None,
    step_metadata: Mapping[str, Any] | None = None,
) -> Provenance:
    normalized_capture_method = _normalize_required_string(
        capture_method,
        field_path="capture_method",
        max_length=_MAX_COMPONENT_NAME_LENGTH,
    )
    raw_source_name = _normalize_required_string(
        raw.source_name,
        field_path="source_name",
        max_length=_MAX_COMPONENT_NAME_LENGTH,
    )
    normalized_source_name = normalize_source_name(raw_source_name)
    normalized_step_name = _normalize_required_string(
        step_name,
        field_path="chain.0.name",
        max_length=_MAX_STEP_NAME_LENGTH,
    )
    normalized_component = _normalize_required_string(
        raw.adapter_name if component is None else component,
        field_path="chain.0.component",
        max_length=_MAX_COMPONENT_NAME_LENGTH,
    )

    try:
        step = ProvenanceStep(
            name=normalized_step_name,
            component=normalized_component,
            occurred_at=raw.received_at,
            metadata=sanitize_provenance_metadata(step_metadata),
        )
        provenance = Provenance(
            source_type=raw.source_type,
            source_name=normalized_source_name,
            actor_id=actor_id,
            actor_kind=actor_kind,
            origin_uri=origin_uri,
            origin_host=origin_host,
            capture_method=normalized_capture_method,
            adapter_name=raw.adapter_name,
            adapter_version=raw.adapter_version,
            received_at=raw.received_at,
            content_hash=content_hash,
            envelope_hash=envelope_hash,
            chain=[step],
        )
    except ValidationError as error:
        raise ProvenanceInvalidError(
            "failed to build provenance",
            details={"errors": error.errors()},
        ) from error

    validate_provenance(provenance)
    return provenance


def append_provenance_step(
    provenance: Provenance,
    *,
    name: str,
    component: str,
    occurred_at: datetime | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Provenance:
    validate_provenance(provenance)
    if len(provenance.chain) >= _MAX_PROVENANCE_CHAIN_LENGTH:
        raise ProvenanceInvalidError(
            "provenance chain exceeds maximum length",
            field_path="chain",
            details={"max_length": _MAX_PROVENANCE_CHAIN_LENGTH},
        )

    step_index = len(provenance.chain)
    try:
        new_step = ProvenanceStep(
            name=_normalize_required_string(
                name,
                field_path=f"chain.{step_index}.name",
                max_length=_MAX_STEP_NAME_LENGTH,
            ),
            component=_normalize_required_string(
                component,
                field_path=f"chain.{step_index}.component",
                max_length=_MAX_COMPONENT_NAME_LENGTH,
            ),
            occurred_at=occurred_at or utc_now(),
            metadata=sanitize_provenance_metadata(metadata),
        )
        updated = provenance.model_copy(
            update={"chain": [*provenance.chain, new_step]}
        )
    except ValidationError as error:
        raise ProvenanceInvalidError(
            "failed to append provenance step",
            details={"errors": error.errors()},
        ) from error

    validate_provenance(updated)
    return updated


def validate_provenance(provenance: Provenance) -> None:
    if provenance is None:
        raise ProvenanceMissingError("provenance is required", field_path="provenance")

    normalized_source_name = _normalize_required_string(
        provenance.source_name,
        field_path="source_name",
        max_length=_MAX_COMPONENT_NAME_LENGTH,
    )
    normalize_source_name(normalized_source_name)
    _normalize_required_string(
        provenance.capture_method,
        field_path="capture_method",
        max_length=_MAX_COMPONENT_NAME_LENGTH,
    )
    _normalize_required_string(
        provenance.adapter_name,
        field_path="adapter_name",
        max_length=_MAX_COMPONENT_NAME_LENGTH,
    )
    _normalize_required_string(
        provenance.adapter_version,
        field_path="adapter_version",
        max_length=_MAX_COMPONENT_NAME_LENGTH,
    )
    _validate_timezone_aware(provenance.received_at, field_path="received_at")

    if not provenance.chain:
        raise ProvenanceMissingError(
            "provenance chain is required",
            field_path="chain",
        )
    if len(provenance.chain) > _MAX_PROVENANCE_CHAIN_LENGTH:
        raise ProvenanceInvalidError(
            "provenance chain exceeds maximum length",
            field_path="chain",
            details={
                "max_length": _MAX_PROVENANCE_CHAIN_LENGTH,
                "length": len(provenance.chain),
            },
        )

    for index, step in enumerate(provenance.chain):
        _normalize_required_string(
            step.name,
            field_path=f"chain.{index}.name",
            max_length=_MAX_STEP_NAME_LENGTH,
        )
        _normalize_required_string(
            step.component,
            field_path=f"chain.{index}.component",
            max_length=_MAX_COMPONENT_NAME_LENGTH,
        )
        _validate_timezone_aware(
            step.occurred_at,
            field_path=f"chain.{index}.occurred_at",
        )
        sanitize_provenance_metadata(step.metadata)


def validate_envelope_provenance(envelope: InputEnvelope) -> None:
    if envelope.provenance is None:
        raise ProvenanceMissingError(
            "envelope provenance is required",
            field_path="provenance",
        )
    if envelope.raw_id is None:
        raise ProvenanceMissingError("envelope raw_id is required", field_path="raw_id")
    if envelope.trace_id is None:
        raise ProvenanceMissingError(
            "envelope trace_id is required",
            field_path="trace_id",
        )
    if envelope.provenance.received_at > envelope.received_at:
        raise ProvenanceInvalidError(
            "provenance received_at must not be later than envelope received_at",
            field_path="provenance.received_at",
            details={
                "provenance_received_at": envelope.provenance.received_at.isoformat(),
                "envelope_received_at": envelope.received_at.isoformat(),
            },
        )
    if envelope.provenance.source_type != envelope.source.source_type:
        raise ProvenanceInvalidError(
            "provenance source_type must match envelope source",
            field_path="provenance.source_type",
            details={
                "provenance_source_type": envelope.provenance.source_type.value,
                "envelope_source_type": envelope.source.source_type.value,
            },
        )
    if envelope.provenance.source_name != envelope.source.source_name:
        raise ProvenanceInvalidError(
            "provenance source_name must match envelope source",
            field_path="provenance.source_name",
            details={
                "provenance_source_name": envelope.provenance.source_name,
                "envelope_source_name": envelope.source.source_name,
            },
        )

    validate_provenance(envelope.provenance)


def require_content_hash(provenance: Provenance) -> str:
    validate_provenance(provenance)
    if provenance.content_hash is None or not provenance.content_hash.strip():
        raise ProvenanceMissingError(
            "provenance content_hash is required",
            field_path="content_hash",
        )
    return provenance.content_hash.strip()


def require_envelope_hash(provenance: Provenance) -> str:
    validate_provenance(provenance)
    if provenance.envelope_hash is None or not provenance.envelope_hash.strip():
        raise ProvenanceMissingError(
            "provenance envelope_hash is required",
            field_path="envelope_hash",
        )
    return provenance.envelope_hash.strip()


def provenance_chain_components(provenance: Provenance) -> tuple[str, ...]:
    validate_provenance(provenance)
    return tuple(step.component for step in provenance.chain)


def _normalize_required_string(
    value: str,
    *,
    field_path: str,
    max_length: int,
) -> str:
    if not isinstance(value, str):
        raise ProvenanceInvalidError(
            "value must be a string",
            field_path=field_path,
            details={"type": type(value).__name__},
        )
    normalized = value.strip()
    if not normalized:
        raise ProvenanceMissingError(
            "value must not be empty",
            field_path=field_path,
        )
    if len(normalized) > max_length:
        raise ProvenanceInvalidError(
            "value exceeds maximum length",
            field_path=field_path,
            details={"max_length": max_length, "length": len(normalized)},
        )
    if _contains_disallowed_control_chars(normalized):
        raise ProvenanceInvalidError(
            "value contains disallowed control characters",
            field_path=field_path,
        )
    return normalized


def _contains_disallowed_control_chars(
    value: str,
    *,
    allow_multiline: bool = False,
) -> bool:
    allowed_multiline_chars = {"\t", "\n", "\r"} if allow_multiline else set()
    return any(
        (ord(char) < 32 or ord(char) == 127) and char not in allowed_multiline_chars
        for char in value
    )


def _normalize_metadata_key(value: Any) -> str:
    if not isinstance(value, str):
        raise ProvenanceInvalidError(
            "provenance metadata keys must be strings",
            field_path="metadata",
            details={"key_type": type(value).__name__},
        )
    key = value.strip()
    if not key:
        raise ProvenanceInvalidError(
            "provenance metadata key must not be empty",
            field_path="metadata",
        )
    if len(key) > _MAX_METADATA_KEY_LENGTH:
        raise ProvenanceInvalidError(
            "provenance metadata key exceeds maximum length",
            field_path=f"metadata.{key}",
            details={"max_length": _MAX_METADATA_KEY_LENGTH, "length": len(key)},
        )
    if _contains_disallowed_control_chars(key):
        raise ProvenanceInvalidError(
            "provenance metadata key contains disallowed control characters",
            field_path=f"metadata.{key}",
        )
    return key


def _normalize_metadata_value(value: Any, key: str) -> Any:
    if not isinstance(value, _ALLOWED_METADATA_VALUE_TYPES):
        raise ProvenanceInvalidError(
            "provenance metadata value has unsupported type",
            field_path=f"metadata.{key}",
            details={"value_type": type(value).__name__},
        )
    if isinstance(value, str):
        normalized = value.strip()
        if len(normalized) > _MAX_METADATA_STRING_VALUE_LENGTH:
            raise ProvenanceInvalidError(
                "provenance metadata string value exceeds maximum length",
                field_path=f"metadata.{key}",
                details={
                    "max_length": _MAX_METADATA_STRING_VALUE_LENGTH,
                    "length": len(normalized),
                },
            )
        if _contains_disallowed_control_chars(normalized, allow_multiline=True):
            raise ProvenanceInvalidError(
                "provenance metadata string value contains disallowed control characters",
                field_path=f"metadata.{key}",
            )
        return normalized
    return value


def _validate_timezone_aware(value: datetime, *, field_path: str) -> None:
    if not isinstance(value, datetime):
        raise ProvenanceInvalidError(
            "datetime value is required",
            field_path=field_path,
            details={"type": type(value).__name__},
        )
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ProvenanceInvalidError(
            "datetime must be timezone-aware",
            field_path=field_path,
        )


__all__ = [
    "append_provenance_step",
    "build_provenance",
    "provenance_chain_components",
    "require_content_hash",
    "require_envelope_hash",
    "sanitize_provenance_metadata",
    "utc_now",
    "validate_envelope_provenance",
    "validate_provenance",
]
