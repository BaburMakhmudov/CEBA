"""Base contract for CEBA Input adapters.

Adapters capture external signals and convert them into RawObservation values.
They preserve technical input boundaries and provenance without computing
meaning, intent, valuation, telos, world model state, deltas, or memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable
from uuid import UUID, uuid4

from pydantic import JsonValue, ValidationError

from ois.input.errors import AdapterCaptureError, SchemaViolationError
from ois.input.schema import InputSourceType, RawObservation, TrustLevel
from ois.input.source import coerce_source_type, coerce_trust_level, normalize_source_name
from ois.input.normalization import normalize_optional_datetime


_ADAPTER_SCHEMA_VERSION = "1.0.0"
_MAX_ADAPTER_NAME_LENGTH = 128
_MAX_ADAPTER_VERSION_LENGTH = 64
_MAX_PERMISSION_LENGTH = 128
_MAX_PERMISSIONS = 64


@dataclass(frozen=True, slots=True)
class AdapterContext:
    trace_id: UUID = field(default_factory=uuid4)
    parent_trace_id: UUID | None = None
    observed_at: datetime | None = None
    received_at: datetime | None = None
    source_name: str | None = None
    source_id: str | None = None
    trust_level: TrustLevel | str | None = None
    permissions: tuple[str, ...] = ()
    transport_metadata: dict[str, Any] = field(default_factory=dict)
    source_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AdapterCapability:
    source_type: InputSourceType
    supports_inline_payload: bool
    supports_payload_ref: bool
    supports_streaming: bool = False
    supports_binary: bool = False


@runtime_checkable
class InputAdapter(Protocol):
    name: str
    version: str
    source_type: InputSourceType
    capability: AdapterCapability

    def supports(self, signal: Any) -> bool:
        ...

    def capture(self, signal: Any, context: AdapterContext) -> RawObservation:
        ...


def validate_adapter_identity(
    *,
    name: str,
    version: str,
    source_type: InputSourceType | str,
) -> tuple[str, str, InputSourceType]:
    if not isinstance(name, str):
        raise SchemaViolationError(
            "adapter name must be a string",
            field_path="adapter_name",
            details={"type": type(name).__name__},
        )
    normalized_name = name.strip()
    if not normalized_name:
        raise SchemaViolationError(
            "adapter name must not be empty",
            field_path="adapter_name",
        )
    if len(normalized_name) > _MAX_ADAPTER_NAME_LENGTH:
        raise SchemaViolationError(
            "adapter name exceeds maximum length",
            field_path="adapter_name",
            details={
                "max_length": _MAX_ADAPTER_NAME_LENGTH,
                "length": len(normalized_name),
            },
        )

    if not isinstance(version, str):
        raise SchemaViolationError(
            "adapter version must be a string",
            field_path="adapter_version",
            details={"type": type(version).__name__},
        )
    normalized_version = version.strip()
    if not normalized_version:
        raise SchemaViolationError(
            "adapter version must not be empty",
            field_path="adapter_version",
        )
    if len(normalized_version) > _MAX_ADAPTER_VERSION_LENGTH:
        raise SchemaViolationError(
            "adapter version exceeds maximum length",
            field_path="adapter_version",
            details={
                "max_length": _MAX_ADAPTER_VERSION_LENGTH,
                "length": len(normalized_version),
            },
        )

    normalized_source_type = coerce_source_type(source_type)
    return normalized_name, normalized_version, normalized_source_type


def validate_permissions(permissions: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if not isinstance(permissions, (tuple, list)):
        raise SchemaViolationError(
            "permissions must be a tuple or list",
            field_path="permissions",
            details={"type": type(permissions).__name__},
        )
    if len(permissions) > _MAX_PERMISSIONS:
        raise SchemaViolationError(
            "permissions exceed maximum count",
            field_path="permissions",
            details={"max_count": _MAX_PERMISSIONS, "count": len(permissions)},
        )

    normalized_permissions: list[str] = []
    seen: set[str] = set()
    for index, permission in enumerate(permissions):
        if not isinstance(permission, str):
            raise SchemaViolationError(
                "permission must be a string",
                field_path=f"permissions.{index}",
                details={"type": type(permission).__name__},
            )
        normalized_permission = permission.strip()
        if not normalized_permission:
            raise SchemaViolationError(
                "permission must not be empty",
                field_path=f"permissions.{index}",
            )
        if len(normalized_permission) > _MAX_PERMISSION_LENGTH:
            raise SchemaViolationError(
                "permission exceeds maximum length",
                field_path=f"permissions.{index}",
                details={
                    "max_length": _MAX_PERMISSION_LENGTH,
                    "length": len(normalized_permission),
                },
            )
        if normalized_permission not in seen:
            normalized_permissions.append(normalized_permission)
            seen.add(normalized_permission)

    return tuple(normalized_permissions)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_adapter_context(
    context: AdapterContext | None,
) -> AdapterContext:
    if context is None:
        return AdapterContext(received_at=utc_now())
    if not isinstance(context, AdapterContext):
        raise SchemaViolationError(
            "context must be AdapterContext",
            field_path="context",
            details={"type": type(context).__name__},
        )

    received_at = context.received_at if context.received_at is not None else utc_now()
    try:
        normalized_observed_at = normalize_optional_datetime(
            context.observed_at,
            field_path="observed_at",
        )
        normalized_received_at = normalize_optional_datetime(
            received_at,
            field_path="received_at",
        )
    except Exception as error:
        raise SchemaViolationError(
            "adapter context datetime normalization failed",
            field_path="context",
            details={"original_type": error.__class__.__name__},
        ) from error
    if normalized_received_at is None:
        raise SchemaViolationError(
            "received_at must not be None",
            field_path="received_at",
        )

    return AdapterContext(
        trace_id=context.trace_id,
        parent_trace_id=context.parent_trace_id,
        observed_at=normalized_observed_at,
        received_at=normalized_received_at,
        source_name=context.source_name,
        source_id=context.source_id,
        trust_level=context.trust_level,
        permissions=validate_permissions(context.permissions),
        transport_metadata=context.transport_metadata,
        source_metadata=context.source_metadata,
    )


def build_raw_observation_from_adapter(
    *,
    adapter_name: str,
    adapter_version: str,
    source_type: InputSourceType | str,
    signal_payload: JsonValue,
    context: AdapterContext | None = None,
    source_name: str | None = None,
    raw_mime_type: str | None = None,
    raw_encoding: str | None = None,
    raw_size_bytes: int | None = None,
) -> RawObservation:
    normalized_name, normalized_version, normalized_source_type = validate_adapter_identity(
        name=adapter_name,
        version=adapter_version,
        source_type=source_type,
    )
    normalized_context = normalize_adapter_context(context)
    raw_payload = _require_json_payload(signal_payload)
    normalized_raw_size_bytes = _validate_raw_size_bytes(raw_size_bytes)
    final_source_name = (
        source_name
        if source_name is not None
        else normalized_context.source_name
        if normalized_context.source_name is not None
        else normalized_name
    )
    normalized_final_source_name = normalize_source_name(final_source_name)
    trust_level = coerce_trust_level(
        normalized_context.trust_level,
        normalized_source_type,
    )

    try:
        return RawObservation(
            trace_id=normalized_context.trace_id,
            parent_trace_id=normalized_context.parent_trace_id,
            source_type=normalized_source_type,
            source_name=normalized_final_source_name,
            received_at=normalized_context.received_at,
            observed_at=normalized_context.observed_at,
            raw_payload=raw_payload,
            raw_size_bytes=normalized_raw_size_bytes,
            raw_mime_type=raw_mime_type,
            raw_encoding=raw_encoding,
            transport_metadata=normalized_context.transport_metadata,
            source_metadata=normalized_context.source_metadata,
            trust_level=trust_level,
            permissions=list(normalized_context.permissions),
            adapter_name=normalized_name,
            adapter_version=normalized_version,
        )
    except ValidationError as error:
        raise AdapterCaptureError(
            "failed to build raw observation from adapter",
            field_path="raw_observation",
            details={"errors": error.errors()},
        ) from error


def capture_with_adapter(
    *,
    adapter: InputAdapter,
    signal: Any,
    context: AdapterContext | None = None,
) -> RawObservation:
    try:
        validate_adapter_identity(
            name=adapter.name,
            version=adapter.version,
            source_type=adapter.source_type,
        )
        if adapter.supports(signal) is False:
            raise AdapterCaptureError(
                "adapter does not support signal",
                field_path="signal",
                details={"adapter": adapter.name},
            )
        normalized_context = normalize_adapter_context(context)
        raw_observation = adapter.capture(signal, normalized_context)
        if not isinstance(raw_observation, RawObservation):
            raise AdapterCaptureError(
                "adapter capture must return RawObservation",
                field_path="adapter.capture",
                details={
                    "adapter": adapter.name,
                    "type": type(raw_observation).__name__,
                },
            )
        return raw_observation
    except (AdapterCaptureError, SchemaViolationError):
        raise
    except Exception as error:
        adapter_name = getattr(adapter, "name", "<unknown>")
        raise AdapterCaptureError(
            "adapter capture failed",
            details={
                "adapter": adapter_name,
                "original_type": error.__class__.__name__,
            },
        ) from error


def _require_json_payload(value: JsonValue) -> JsonValue:
    if value is None:
        raise AdapterCaptureError(
            "signal payload must not be None",
            field_path="signal_payload",
        )
    return value


def _validate_raw_size_bytes(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise SchemaViolationError(
            "raw_size_bytes must be an integer",
            field_path="raw_size_bytes",
            details={"type": type(value).__name__},
        )
    if not isinstance(value, int):
        raise SchemaViolationError(
            "raw_size_bytes must be an integer",
            field_path="raw_size_bytes",
            details={"type": type(value).__name__},
        )
    if value < 0:
        raise SchemaViolationError(
            "raw_size_bytes must be greater than or equal to 0",
            field_path="raw_size_bytes",
            details={"value": value},
        )
    return value


__all__ = [
    "AdapterCapability",
    "AdapterContext",
    "InputAdapter",
    "build_raw_observation_from_adapter",
    "capture_with_adapter",
    "normalize_adapter_context",
    "utc_now",
    "validate_adapter_identity",
    "validate_permissions",
]
