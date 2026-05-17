"""CEBA Input source boundary utilities.

Sources identify where an observation came from and define technical trust
boundaries for input capture. Source handling normalizes identity and metadata
only; it does not compute meaning.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ois.input.errors import SchemaViolationError, UnsupportedInputError
from ois.input.schema import InputSource, InputSourceType, TrustLevel


_ALLOWED_METADATA_VALUE_TYPES = (str, int, float, bool, type(None))

_MAX_SOURCE_NAME_LENGTH = 128
_MAX_SOURCE_ID_LENGTH = 256
_MAX_METADATA_KEYS = 64
_MAX_METADATA_KEY_LENGTH = 128
_MAX_METADATA_STRING_VALUE_LENGTH = 2048


@dataclass(frozen=True, slots=True)
class SourceBoundary:
    source_type: InputSourceType
    default_trust_level: TrustLevel
    allow_external_payloads: bool
    allow_inline_payloads: bool
    requires_actor: bool = False
    metadata_allowlist: frozenset[str] = field(default_factory=frozenset)


SOURCE_BOUNDARIES: dict[InputSourceType, SourceBoundary] = {
    InputSourceType.USER: SourceBoundary(
        source_type=InputSourceType.USER,
        default_trust_level=TrustLevel.AUTHENTICATED,
        allow_external_payloads=False,
        allow_inline_payloads=True,
        requires_actor=False,
        metadata_allowlist=frozenset(
            {"ui_surface", "session_id", "locale", "client_version"}
        ),
    ),
    InputSourceType.API: SourceBoundary(
        source_type=InputSourceType.API,
        default_trust_level=TrustLevel.AUTHENTICATED,
        allow_external_payloads=True,
        allow_inline_payloads=True,
        requires_actor=True,
        metadata_allowlist=frozenset(
            {"method", "path", "request_id", "client_id", "auth_scope"}
        ),
    ),
    InputSourceType.FILE: SourceBoundary(
        source_type=InputSourceType.FILE,
        default_trust_level=TrustLevel.UNTRUSTED,
        allow_external_payloads=False,
        allow_inline_payloads=False,
        requires_actor=False,
        metadata_allowlist=frozenset(
            {"filename", "mime_type", "size_bytes", "upload_id", "storage_uri"}
        ),
    ),
    InputSourceType.SHELL: SourceBoundary(
        source_type=InputSourceType.SHELL,
        default_trust_level=TrustLevel.INTERNAL,
        allow_external_payloads=False,
        allow_inline_payloads=True,
        requires_actor=False,
        metadata_allowlist=frozenset({"cwd", "exit_code", "stream", "command_id"}),
    ),
    InputSourceType.SYSTEM: SourceBoundary(
        source_type=InputSourceType.SYSTEM,
        default_trust_level=TrustLevel.INTERNAL,
        allow_external_payloads=False,
        allow_inline_payloads=True,
        requires_actor=False,
        metadata_allowlist=frozenset(
            {"component", "event_name", "severity", "runtime_id"}
        ),
    ),
    InputSourceType.WEBHOOK: SourceBoundary(
        source_type=InputSourceType.WEBHOOK,
        default_trust_level=TrustLevel.UNTRUSTED,
        allow_external_payloads=True,
        allow_inline_payloads=True,
        requires_actor=False,
        metadata_allowlist=frozenset(
            {"provider", "delivery_id", "signature_validated", "event_type"}
        ),
    ),
    InputSourceType.DATABASE: SourceBoundary(
        source_type=InputSourceType.DATABASE,
        default_trust_level=TrustLevel.INTERNAL,
        allow_external_payloads=False,
        allow_inline_payloads=True,
        requires_actor=False,
        metadata_allowlist=frozenset(
            {"database", "table", "row_id", "operation", "transaction_id"}
        ),
    ),
    InputSourceType.LOG: SourceBoundary(
        source_type=InputSourceType.LOG,
        default_trust_level=TrustLevel.INTERNAL,
        allow_external_payloads=False,
        allow_inline_payloads=True,
        requires_actor=False,
        metadata_allowlist=frozenset(
            {"service", "logger", "level", "line_number", "log_id"}
        ),
    ),
}


def coerce_source_type(value: InputSourceType | str) -> InputSourceType:
    if isinstance(value, InputSourceType):
        return value
    if isinstance(value, str):
        normalized = value.strip()
        for source_type in InputSourceType:
            if normalized == source_type.value or normalized.upper() == source_type.name:
                return source_type
    raise UnsupportedInputError(
        "unsupported source type",
        field_path="source_type",
        details={"value": value},
    )


def coerce_trust_level(
    value: TrustLevel | str | None,
    source_type: InputSourceType,
) -> TrustLevel:
    if value is None:
        return get_source_boundary(source_type).default_trust_level
    if isinstance(value, TrustLevel):
        return value
    if isinstance(value, str):
        normalized = value.strip()
        for trust_level in TrustLevel:
            if normalized == trust_level.value or normalized.upper() == trust_level.name:
                return trust_level
    raise SchemaViolationError(
        "invalid trust level",
        field_path="trust_level",
        details={"value": value, "source_type": source_type.value},
    )


def normalize_source_name(value: str) -> str:
    if not isinstance(value, str):
        raise SchemaViolationError(
            "source_name must be a string",
            field_path="source_name",
            details={"type": type(value).__name__},
        )
    normalized = value.strip()
    if not normalized:
        raise SchemaViolationError(
            "source_name must not be empty",
            field_path="source_name",
        )
    if len(normalized) > _MAX_SOURCE_NAME_LENGTH:
        raise SchemaViolationError(
            "source_name exceeds maximum length",
            field_path="source_name",
            details={
                "max_length": _MAX_SOURCE_NAME_LENGTH,
                "length": len(normalized),
            },
        )
    if _contains_disallowed_control_chars(normalized):
        raise SchemaViolationError(
            "source_name contains disallowed control characters",
            field_path="source_name",
        )
    return normalized


def normalize_source_id(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SchemaViolationError(
            "source_id must be a string",
            field_path="source_id",
            details={"type": type(value).__name__},
        )
    normalized = value.strip()
    if not normalized:
        raise SchemaViolationError(
            "source_id must not be empty",
            field_path="source_id",
        )
    if len(normalized) > _MAX_SOURCE_ID_LENGTH:
        raise SchemaViolationError(
            "source_id exceeds maximum length",
            field_path="source_id",
            details={"max_length": _MAX_SOURCE_ID_LENGTH, "length": len(normalized)},
        )
    if _contains_disallowed_control_chars(normalized):
        raise SchemaViolationError(
            "source_id contains disallowed control characters",
            field_path="source_id",
        )
    return normalized


def sanitize_source_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    source_type: InputSourceType,
) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise SchemaViolationError(
            "metadata must be a mapping",
            field_path="metadata",
            details={"type": type(metadata).__name__},
        )
    if len(metadata) > _MAX_METADATA_KEYS:
        raise SchemaViolationError(
            "metadata exceeds maximum key count",
            field_path="metadata",
            details={"max_keys": _MAX_METADATA_KEYS, "key_count": len(metadata)},
        )

    boundary = get_source_boundary(source_type)
    sanitized: dict[str, Any] = {}
    for raw_key, raw_value in metadata.items():
        key = _normalize_metadata_key(raw_key)
        if key not in boundary.metadata_allowlist:
            continue
        sanitized[key] = _normalize_metadata_value(raw_value, key)
    return sanitized


def build_input_source(
    *,
    source_type: InputSourceType | str,
    source_name: str,
    source_id: str | None = None,
    trust_level: TrustLevel | str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> InputSource:
    coerced_source_type = coerce_source_type(source_type)
    return InputSource(
        source_type=coerced_source_type,
        source_name=normalize_source_name(source_name),
        source_id=normalize_source_id(source_id),
        trust_level=coerce_trust_level(trust_level, coerced_source_type),
        metadata=sanitize_source_metadata(
            metadata,
            source_type=coerced_source_type,
        ),
    )


def get_source_boundary(source_type: InputSourceType | str) -> SourceBoundary:
    coerced_source_type = coerce_source_type(source_type)
    boundary = SOURCE_BOUNDARIES.get(coerced_source_type)
    if boundary is None:
        raise UnsupportedInputError(
            "source type has no registered boundary",
            field_path="source_type",
            details={"source_type": coerced_source_type.value},
        )
    return boundary


def validate_source_boundary(source: InputSource) -> None:
    boundary = get_source_boundary(source.source_type)
    normalize_source_name(source.source_name)
    normalize_source_id(source.source_id)

    if len(source.metadata) > _MAX_METADATA_KEYS:
        raise SchemaViolationError(
            "metadata exceeds maximum key count",
            field_path="metadata",
            details={"max_keys": _MAX_METADATA_KEYS, "key_count": len(source.metadata)},
        )
    for raw_key, raw_value in source.metadata.items():
        key = _normalize_metadata_key(raw_key)
        if key not in boundary.metadata_allowlist:
            raise SchemaViolationError(
                "metadata key is not allowed for source type",
                field_path=f"metadata.{key}",
                details={"source_type": source.source_type.value, "key": key},
            )
        _normalize_metadata_value(raw_value, key)


def source_allows_external_payload(source_type: InputSourceType | str) -> bool:
    return get_source_boundary(source_type).allow_external_payloads


def source_allows_inline_payload(source_type: InputSourceType | str) -> bool:
    return get_source_boundary(source_type).allow_inline_payloads


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
        raise SchemaViolationError(
            "metadata keys must be strings",
            field_path="metadata",
            details={"key_type": type(value).__name__},
        )
    key = value.strip()
    if not key:
        raise SchemaViolationError(
            "metadata key must not be empty",
            field_path="metadata",
        )
    if len(key) > _MAX_METADATA_KEY_LENGTH:
        raise SchemaViolationError(
            "metadata key exceeds maximum length",
            field_path=f"metadata.{key}",
            details={"max_length": _MAX_METADATA_KEY_LENGTH, "length": len(key)},
        )
    if _contains_disallowed_control_chars(key):
        raise SchemaViolationError(
            "metadata key contains disallowed control characters",
            field_path=f"metadata.{key}",
        )
    return key


def _normalize_metadata_value(value: Any, key: str) -> Any:
    if not isinstance(value, _ALLOWED_METADATA_VALUE_TYPES):
        raise SchemaViolationError(
            "metadata value has unsupported type",
            field_path=f"metadata.{key}",
            details={"value_type": type(value).__name__},
        )
    if isinstance(value, str):
        normalized = value.strip()
        if len(normalized) > _MAX_METADATA_STRING_VALUE_LENGTH:
            raise SchemaViolationError(
                "metadata string value exceeds maximum length",
                field_path=f"metadata.{key}",
                details={
                    "max_length": _MAX_METADATA_STRING_VALUE_LENGTH,
                    "length": len(normalized),
                },
            )
        if _contains_disallowed_control_chars(normalized, allow_multiline=True):
            raise SchemaViolationError(
                "metadata string value contains disallowed control characters",
                field_path=f"metadata.{key}",
            )
        return normalized
    return value


__all__ = [
    "SourceBoundary",
    "SOURCE_BOUNDARIES",
    "build_input_source",
    "coerce_source_type",
    "coerce_trust_level",
    "get_source_boundary",
    "normalize_source_id",
    "normalize_source_name",
    "sanitize_source_metadata",
    "source_allows_external_payload",
    "source_allows_inline_payload",
    "validate_source_boundary",
]
