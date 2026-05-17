"""CEBA SystemAdapter.

SystemAdapter captures internal system events as RawObservation values. It
does not inspect runtime state or compute meaning.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import JsonValue

from ois.input.adapters.base import (
    AdapterCapability,
    AdapterContext,
    InputAdapter,
    build_raw_observation_from_adapter,
)
from ois.input.errors import AdapterCaptureError, SchemaViolationError
from ois.input.normalization import normalize_text
from ois.input.schema import InputSourceType, RawObservation


_SYSTEM_ADAPTER_NAME = "system_adapter"
_SYSTEM_ADAPTER_VERSION = "1.0.0"
_DEFAULT_SYSTEM_SOURCE_NAME = "system"
_DEFAULT_JSON_MIME = "application/json"
_DEFAULT_TEXT_MIME = "text/plain"
_DEFAULT_ENCODING = "utf-8"
_MAX_COMPONENT_LENGTH = 256
_MAX_EVENT_NAME_LENGTH = 256
_MAX_RUNTIME_ID_LENGTH = 256
_MAX_SEVERITY_LENGTH = 32
_MAX_TEXT_PAYLOAD_LENGTH = 1_000_000
_MAX_METADATA_KEYS = 64

_ALLOWED_METADATA_KEYS = frozenset({
    "component",
    "event_name",
    "severity",
    "runtime_id",
})


class SystemEventSeverity(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class SystemSignal:
    payload: JsonValue
    component: str
    event_name: str
    severity: SystemEventSeverity | str = SystemEventSeverity.INFO
    runtime_id: str | None = None
    source_name: str | None = None
    raw_mime_type: str | None = None
    raw_encoding: str | None = None
    transport_metadata: dict[str, Any] = field(default_factory=dict)
    source_metadata: dict[str, Any] = field(default_factory=dict)


class SystemAdapter:
    name = _SYSTEM_ADAPTER_NAME
    version = _SYSTEM_ADAPTER_VERSION
    source_type = InputSourceType.SYSTEM
    capability = AdapterCapability(
        source_type=InputSourceType.SYSTEM,
        supports_inline_payload=True,
        supports_payload_ref=False,
        supports_streaming=False,
        supports_binary=False,
    )

    def supports(self, signal: Any) -> bool:
        if isinstance(signal, SystemSignal):
            return True
        if not isinstance(signal, dict):
            return False
        return "payload" in signal and "component" in signal and "event_name" in signal

    def capture(self, signal: Any, context: AdapterContext) -> RawObservation:
        if not isinstance(context, AdapterContext):
            raise SchemaViolationError(
                "context must be AdapterContext",
                field_path="context",
                details={"type": type(context).__name__},
            )

        system_signal = normalize_system_signal(signal)
        system_metadata = build_system_metadata(system_signal)
        source_metadata = merge_system_metadata(
            context_metadata=context.source_metadata,
            signal_metadata=system_signal.source_metadata,
            system_metadata=system_metadata,
        )
        transport_metadata = merge_transport_metadata(
            context_metadata=context.transport_metadata,
            signal_metadata=system_signal.transport_metadata,
        )
        source_name = (
            system_signal.source_name
            if system_signal.source_name is not None
            else context.source_name
            if context.source_name is not None
            else _DEFAULT_SYSTEM_SOURCE_NAME
        )
        raw_mime_type = (
            system_signal.raw_mime_type
            if system_signal.raw_mime_type is not None
            else _DEFAULT_TEXT_MIME
            if isinstance(system_signal.payload, str)
            else _DEFAULT_JSON_MIME
        )
        raw_encoding = (
            system_signal.raw_encoding
            if system_signal.raw_encoding is not None
            else _DEFAULT_ENCODING
        )
        raw_size_bytes = (
            len(system_signal.payload.encode(_DEFAULT_ENCODING))
            if isinstance(system_signal.payload, str)
            else None
        )
        adapter_context = AdapterContext(
            trace_id=context.trace_id,
            parent_trace_id=context.parent_trace_id,
            observed_at=context.observed_at,
            received_at=context.received_at,
            source_name=context.source_name,
            source_id=context.source_id,
            trust_level=context.trust_level,
            permissions=context.permissions,
            transport_metadata=transport_metadata,
            source_metadata=source_metadata,
        )

        return build_raw_observation_from_adapter(
            adapter_name=self.name,
            adapter_version=self.version,
            source_type=InputSourceType.SYSTEM,
            signal_payload=system_signal.payload,
            context=adapter_context,
            source_name=source_name,
            raw_mime_type=raw_mime_type,
            raw_encoding=raw_encoding,
            raw_size_bytes=raw_size_bytes,
        )


def normalize_system_signal(signal: Any) -> SystemSignal:
    if isinstance(signal, SystemSignal):
        return SystemSignal(
            payload=validate_system_payload(signal.payload),
            component=normalize_system_component(signal.component),
            event_name=normalize_system_event_name(signal.event_name),
            severity=normalize_system_severity(signal.severity),
            runtime_id=normalize_runtime_id(signal.runtime_id),
            source_name=_normalize_optional_string(
                signal.source_name,
                field_path="source_name",
                max_length=256,
            ),
            raw_mime_type=_normalize_optional_string(
                signal.raw_mime_type,
                field_path="raw_mime_type",
                max_length=256,
            ),
            raw_encoding=_normalize_optional_string(
                signal.raw_encoding,
                field_path="raw_encoding",
                max_length=256,
            ),
            transport_metadata=merge_transport_metadata(
                signal_metadata=signal.transport_metadata,
            ),
            source_metadata=merge_system_metadata(
                signal_metadata=signal.source_metadata,
            ),
        )
    if isinstance(signal, dict):
        if "payload" not in signal:
            raise AdapterCaptureError(
                "system signal requires payload",
                field_path="signal.payload",
            )
        if "component" not in signal:
            raise AdapterCaptureError(
                "system signal requires component",
                field_path="signal.component",
            )
        if "event_name" not in signal:
            raise AdapterCaptureError(
                "system signal requires event_name",
                field_path="signal.event_name",
            )
        return SystemSignal(
            payload=validate_system_payload(signal["payload"]),
            component=normalize_system_component(signal["component"]),
            event_name=normalize_system_event_name(signal["event_name"]),
            severity=normalize_system_severity(
                signal.get("severity", SystemEventSeverity.INFO)
            ),
            runtime_id=normalize_runtime_id(signal.get("runtime_id")),
            source_name=_normalize_optional_string(
                signal.get("source_name"),
                field_path="source_name",
                max_length=256,
            ),
            raw_mime_type=_normalize_optional_string(
                signal.get("raw_mime_type"),
                field_path="raw_mime_type",
                max_length=256,
            ),
            raw_encoding=_normalize_optional_string(
                signal.get("raw_encoding"),
                field_path="raw_encoding",
                max_length=256,
            ),
            transport_metadata=merge_transport_metadata(
                signal_metadata=signal.get("transport_metadata"),
            ),
            source_metadata=merge_system_metadata(
                signal_metadata=signal.get("source_metadata"),
            ),
        )
    if signal is None:
        raise AdapterCaptureError(
            "system signal must not be None",
            field_path="signal",
        )
    if isinstance(signal, (bytes, bytearray)):
        raise AdapterCaptureError(
            "system signal bytes are not supported",
            field_path="signal",
            details={"type": type(signal).__name__},
        )
    if isinstance(signal, (str, list, int, float, bool)):
        raise AdapterCaptureError(
            "system signal must be SystemSignal or dict with payload, component, and event_name",
            field_path="signal",
            details={"type": type(signal).__name__},
        )
    raise AdapterCaptureError(
        "unsupported system signal type",
        field_path="signal",
        details={"type": type(signal).__name__},
    )


def validate_system_payload(payload: JsonValue) -> JsonValue:
    if payload is None:
        raise AdapterCaptureError(
            "system payload must not be None",
            field_path="payload",
        )
    if isinstance(payload, str):
        normalized = normalize_text(payload)
        if len(normalized) > _MAX_TEXT_PAYLOAD_LENGTH:
            raise AdapterCaptureError(
                "system text payload exceeds maximum length",
                field_path="payload",
                details={
                    "max_length": _MAX_TEXT_PAYLOAD_LENGTH,
                    "length": len(normalized),
                },
            )
        return normalized
    if isinstance(payload, (dict, list, int, float, bool)):
        return payload
    raise AdapterCaptureError(
        "unsupported system payload type",
        field_path="payload",
        details={"type": type(payload).__name__},
    )


def normalize_system_component(value: str) -> str:
    if not isinstance(value, str):
        raise AdapterCaptureError(
            "component must be a string",
            field_path="component",
            details={"type": type(value).__name__},
        )
    normalized = value.strip()
    if not normalized:
        raise AdapterCaptureError(
            "component must not be empty",
            field_path="component",
        )
    if len(normalized) > _MAX_COMPONENT_LENGTH:
        raise AdapterCaptureError(
            "component exceeds maximum length",
            field_path="component",
            details={"max_length": _MAX_COMPONENT_LENGTH, "length": len(normalized)},
        )
    if _contains_disallowed_control_chars(normalized):
        raise AdapterCaptureError(
            "component contains disallowed control characters",
            field_path="component",
        )
    return normalized


def normalize_system_event_name(value: str) -> str:
    if not isinstance(value, str):
        raise AdapterCaptureError(
            "event_name must be a string",
            field_path="event_name",
            details={"type": type(value).__name__},
        )
    normalized = value.strip()
    if not normalized:
        raise AdapterCaptureError(
            "event_name must not be empty",
            field_path="event_name",
        )
    if len(normalized) > _MAX_EVENT_NAME_LENGTH:
        raise AdapterCaptureError(
            "event_name exceeds maximum length",
            field_path="event_name",
            details={"max_length": _MAX_EVENT_NAME_LENGTH, "length": len(normalized)},
        )
    if _contains_disallowed_control_chars(normalized):
        raise AdapterCaptureError(
            "event_name contains disallowed control characters",
            field_path="event_name",
        )
    return normalized


def normalize_system_severity(value: SystemEventSeverity | str) -> SystemEventSeverity:
    if isinstance(value, SystemEventSeverity):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            raise AdapterCaptureError(
                "severity must not be empty",
                field_path="severity",
            )
        if len(normalized) > _MAX_SEVERITY_LENGTH:
            raise AdapterCaptureError(
                "severity exceeds maximum length",
                field_path="severity",
                details={
                    "max_length": _MAX_SEVERITY_LENGTH,
                    "length": len(normalized),
                },
            )
        for severity in SystemEventSeverity:
            if normalized == severity.value:
                return severity
    raise AdapterCaptureError(
        "unsupported system severity",
        field_path="severity",
        details={"value": value},
    )


def normalize_runtime_id(value: str | None) -> str | None:
    return _normalize_optional_string(
        value,
        field_path="runtime_id",
        max_length=_MAX_RUNTIME_ID_LENGTH,
    )


def build_system_metadata(signal: SystemSignal) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "component": signal.component,
        "event_name": signal.event_name,
        "severity": (
            signal.severity.value
            if isinstance(signal.severity, SystemEventSeverity)
            else signal.severity
        ),
    }
    if signal.runtime_id is not None:
        metadata["runtime_id"] = signal.runtime_id
    return metadata


def merge_system_metadata(
    *,
    context_metadata: Mapping[str, Any] | None = None,
    signal_metadata: Mapping[str, Any] | None = None,
    system_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for field_path, metadata in (
        ("context_metadata", context_metadata),
        ("signal_metadata", signal_metadata),
        ("system_metadata", system_metadata),
    ):
        checked_metadata = _ensure_mapping(metadata, field_path=field_path)
        if checked_metadata is None:
            continue
        for key, value in checked_metadata.items():
            if key in _ALLOWED_METADATA_KEYS:
                merged[key] = value
    return merged


def merge_transport_metadata(
    *,
    context_metadata: Mapping[str, Any] | None = None,
    signal_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for field_path, metadata in (
        ("context_metadata", context_metadata),
        ("signal_metadata", signal_metadata),
    ):
        checked_metadata = _ensure_mapping(metadata, field_path=field_path)
        if checked_metadata is not None:
            merged.update(checked_metadata)
    return merged


def _normalize_optional_string(
    value: str | None,
    *,
    field_path: str,
    max_length: int,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise AdapterCaptureError(
            f"{field_path} must be a string",
            field_path=field_path,
            details={"type": type(value).__name__},
        )
    normalized = value.strip()
    if not normalized:
        return None
    if len(normalized) > max_length:
        raise AdapterCaptureError(
            f"{field_path} exceeds maximum length",
            field_path=field_path,
            details={"max_length": max_length, "length": len(normalized)},
        )
    if _contains_disallowed_control_chars(normalized):
        raise AdapterCaptureError(
            f"{field_path} contains disallowed control characters",
            field_path=field_path,
        )
    return normalized


def _contains_disallowed_control_chars(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _ensure_mapping(
    value: Mapping[str, Any] | None,
    *,
    field_path: str,
) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SchemaViolationError(
            f"{field_path} must be a mapping",
            field_path=field_path,
            details={"type": type(value).__name__},
        )
    if len(value) > _MAX_METADATA_KEYS:
        raise SchemaViolationError(
            f"{field_path} exceeds maximum key count",
            field_path=field_path,
            details={"max_keys": _MAX_METADATA_KEYS, "key_count": len(value)},
        )
    return value


def create_system_adapter() -> SystemAdapter:
    return SystemAdapter()


_adapter_protocol_check: InputAdapter = SystemAdapter()


__all__ = [
    "SystemAdapter",
    "SystemEventSeverity",
    "SystemSignal",
    "build_system_metadata",
    "create_system_adapter",
    "merge_system_metadata",
    "merge_transport_metadata",
    "normalize_runtime_id",
    "normalize_system_component",
    "normalize_system_event_name",
    "normalize_system_severity",
    "normalize_system_signal",
    "validate_system_payload",
]
