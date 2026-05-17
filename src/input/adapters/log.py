"""CEBA LogAdapter.

LogAdapter captures already-produced log signals as RawObservation values. It
does not read files, tail logs, or compute meaning.
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


_LOG_ADAPTER_NAME = "log_adapter"
_LOG_ADAPTER_VERSION = "1.0.0"
_DEFAULT_LOG_SOURCE_NAME = "log"
_DEFAULT_JSON_MIME = "application/json"
_DEFAULT_TEXT_MIME = "text/plain"
_DEFAULT_ENCODING = "utf-8"
_MAX_SERVICE_LENGTH = 256
_MAX_LOGGER_LENGTH = 256
_MAX_LEVEL_LENGTH = 32
_MAX_LOG_ID_LENGTH = 512
_MAX_TEXT_PAYLOAD_LENGTH = 2_000_000
_MAX_METADATA_KEYS = 64

_ALLOWED_METADATA_KEYS = frozenset({
    "service",
    "logger",
    "level",
    "line_number",
    "log_id",
})

_ALLOWED_SIGNAL_KEYS = frozenset({
    "payload",
    "service",
    "logger",
    "level",
    "line_number",
    "log_id",
    "source_name",
    "raw_mime_type",
    "raw_encoding",
    "transport_metadata",
    "source_metadata",
})


class LogLevel(str, Enum):
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class LogSignal:
    payload: JsonValue
    service: str
    level: LogLevel | str
    logger: str | None = None
    line_number: int | None = None
    log_id: str | None = None
    source_name: str | None = None
    raw_mime_type: str | None = None
    raw_encoding: str | None = None
    transport_metadata: dict[str, Any] = field(default_factory=dict)
    source_metadata: dict[str, Any] = field(default_factory=dict)


class LogAdapter:
    name = _LOG_ADAPTER_NAME
    version = _LOG_ADAPTER_VERSION
    source_type = InputSourceType.LOG
    capability = AdapterCapability(
        source_type=InputSourceType.LOG,
        supports_inline_payload=True,
        supports_payload_ref=False,
        supports_streaming=False,
        supports_binary=False,
    )

    def supports(self, signal: Any) -> bool:
        if isinstance(signal, LogSignal):
            return True
        if not isinstance(signal, dict):
            return False
        return "payload" in signal and "service" in signal and "level" in signal

    def capture(self, signal: Any, context: AdapterContext) -> RawObservation:
        if not isinstance(context, AdapterContext):
            raise SchemaViolationError(
                "context must be AdapterContext",
                field_path="context",
                details={"type": type(context).__name__},
            )

        log_signal = normalize_log_signal(signal)
        log_metadata = build_log_metadata(log_signal)
        source_metadata = merge_log_metadata(
            context_metadata=context.source_metadata,
            signal_metadata=log_signal.source_metadata,
            log_metadata=log_metadata,
        )
        transport_metadata = merge_transport_metadata(
            context_metadata=context.transport_metadata,
            signal_metadata=log_signal.transport_metadata,
        )
        source_name = (
            log_signal.source_name
            if log_signal.source_name is not None
            else context.source_name
            if context.source_name is not None
            else _DEFAULT_LOG_SOURCE_NAME
        )
        raw_mime_type = (
            log_signal.raw_mime_type
            if log_signal.raw_mime_type is not None
            else _DEFAULT_TEXT_MIME
            if isinstance(log_signal.payload, str)
            else _DEFAULT_JSON_MIME
        )
        raw_encoding = (
            log_signal.raw_encoding
            if log_signal.raw_encoding is not None
            else _DEFAULT_ENCODING
        )
        raw_size_bytes = (
            len(log_signal.payload.encode(_DEFAULT_ENCODING))
            if isinstance(log_signal.payload, str)
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
            source_type=InputSourceType.LOG,
            signal_payload=log_signal.payload,
            context=adapter_context,
            source_name=source_name,
            raw_mime_type=raw_mime_type,
            raw_encoding=raw_encoding,
            raw_size_bytes=raw_size_bytes,
        )


def normalize_log_signal(signal: Any) -> LogSignal:
    if isinstance(signal, LogSignal):
        return LogSignal(
            payload=validate_log_payload(signal.payload),
            service=normalize_log_service(signal.service),
            level=normalize_log_level(signal.level),
            logger=normalize_logger_name(signal.logger),
            line_number=normalize_line_number(signal.line_number),
            log_id=normalize_log_id(signal.log_id),
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
            source_metadata=merge_log_metadata(
                signal_metadata=signal.source_metadata,
            ),
        )
    if isinstance(signal, dict):
        unsupported_keys = set(signal) - _ALLOWED_SIGNAL_KEYS
        if unsupported_keys:
            raise AdapterCaptureError(
                "log signal contains unsupported keys",
                field_path="signal",
                details={"unsupported_keys": sorted(unsupported_keys)},
            )
        if "payload" not in signal:
            raise AdapterCaptureError(
                "log signal requires payload",
                field_path="signal.payload",
            )
        if "service" not in signal:
            raise AdapterCaptureError(
                "log signal requires service",
                field_path="signal.service",
            )
        if "level" not in signal:
            raise AdapterCaptureError(
                "log signal requires level",
                field_path="signal.level",
            )
        return LogSignal(
            payload=validate_log_payload(signal["payload"]),
            service=normalize_log_service(signal["service"]),
            level=normalize_log_level(signal["level"]),
            logger=normalize_logger_name(signal.get("logger")),
            line_number=normalize_line_number(signal.get("line_number")),
            log_id=normalize_log_id(signal.get("log_id")),
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
            source_metadata=merge_log_metadata(
                signal_metadata=signal.get("source_metadata"),
            ),
        )
    if signal is None:
        raise AdapterCaptureError(
            "log signal must not be None",
            field_path="signal",
        )
    if isinstance(signal, (bytes, bytearray)):
        raise AdapterCaptureError(
            "log signal bytes are not supported",
            field_path="signal",
            details={"type": type(signal).__name__},
        )
    if isinstance(signal, (str, list, int, float, bool)):
        raise AdapterCaptureError(
            "log signal must be LogSignal or dict with payload, service, and level",
            field_path="signal",
            details={"type": type(signal).__name__},
        )
    raise AdapterCaptureError(
        "unsupported log signal type",
        field_path="signal",
        details={"type": type(signal).__name__},
    )


def validate_log_payload(payload: JsonValue) -> JsonValue:
    if payload is None:
        raise AdapterCaptureError(
            "log payload must not be None",
            field_path="payload",
        )
    if isinstance(payload, str):
        try:
            normalized = normalize_text(payload)
        except Exception as error:
            raise AdapterCaptureError(
                "log text payload normalization failed",
                field_path="payload",
                details={"original_type": error.__class__.__name__},
            ) from error
        if len(normalized) > _MAX_TEXT_PAYLOAD_LENGTH:
            raise AdapterCaptureError(
                "log text payload exceeds maximum length",
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
        "unsupported log payload type",
        field_path="payload",
        details={"type": type(payload).__name__},
    )


def normalize_log_service(value: str) -> str:
    if not isinstance(value, str):
        raise AdapterCaptureError(
            "service must be a string",
            field_path="service",
            details={"type": type(value).__name__},
        )
    normalized = value.strip()
    if not normalized:
        raise AdapterCaptureError(
            "service must not be empty",
            field_path="service",
        )
    if len(normalized) > _MAX_SERVICE_LENGTH:
        raise AdapterCaptureError(
            "service exceeds maximum length",
            field_path="service",
            details={
                "max_length": _MAX_SERVICE_LENGTH,
                "length": len(normalized),
            },
        )
    if _contains_disallowed_control_chars(normalized):
        raise AdapterCaptureError(
            "service contains disallowed control characters",
            field_path="service",
        )
    return normalized


def normalize_logger_name(value: str | None) -> str | None:
    return _normalize_optional_string(
        value,
        field_path="logger",
        max_length=_MAX_LOGGER_LENGTH,
    )


def normalize_log_level(value: LogLevel | str) -> LogLevel:
    if isinstance(value, LogLevel):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            raise AdapterCaptureError(
                "level must not be empty",
                field_path="level",
            )
        if len(normalized) > _MAX_LEVEL_LENGTH:
            raise AdapterCaptureError(
                "level exceeds maximum length",
                field_path="level",
                details={
                    "max_length": _MAX_LEVEL_LENGTH,
                    "length": len(normalized),
                },
            )
        try:
            return LogLevel(normalized)
        except ValueError as error:
            raise AdapterCaptureError(
                "level is not supported",
                field_path="level",
                details={"level": normalized},
            ) from error
    raise AdapterCaptureError(
        "level must be LogLevel or string",
        field_path="level",
        details={"type": type(value).__name__},
    )


def normalize_line_number(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise AdapterCaptureError(
            "line_number must be an integer",
            field_path="line_number",
            details={"type": type(value).__name__},
        )
    if not isinstance(value, int):
        raise AdapterCaptureError(
            "line_number must be an integer",
            field_path="line_number",
            details={"type": type(value).__name__},
        )
    if value < 0:
        raise AdapterCaptureError(
            "line_number must be greater than or equal to 0",
            field_path="line_number",
            details={"value": value},
        )
    return value


def normalize_log_id(value: str | None) -> str | None:
    return _normalize_optional_string(
        value,
        field_path="log_id",
        max_length=_MAX_LOG_ID_LENGTH,
    )


def build_log_metadata(signal: LogSignal) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "service": signal.service,
        "level": signal.level.value
        if isinstance(signal.level, LogLevel)
        else signal.level,
    }
    if signal.logger is not None:
        metadata["logger"] = signal.logger
    if signal.line_number is not None:
        metadata["line_number"] = signal.line_number
    if signal.log_id is not None:
        metadata["log_id"] = signal.log_id
    return metadata


def merge_log_metadata(
    *,
    context_metadata: Mapping[str, Any] | None = None,
    signal_metadata: Mapping[str, Any] | None = None,
    log_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for field_path, metadata in (
        ("context_metadata", context_metadata),
        ("signal_metadata", signal_metadata),
        ("log_metadata", log_metadata),
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
            details={"max_count": _MAX_METADATA_KEYS, "count": len(value)},
        )
    return value


def create_log_adapter() -> LogAdapter:
    return LogAdapter()


_adapter_protocol_check: InputAdapter = LogAdapter()


__all__ = [
    "LogAdapter",
    "LogLevel",
    "LogSignal",
    "build_log_metadata",
    "create_log_adapter",
    "merge_log_metadata",
    "merge_transport_metadata",
    "normalize_line_number",
    "normalize_log_id",
    "normalize_log_level",
    "normalize_log_service",
    "normalize_log_signal",
    "normalize_logger_name",
    "validate_log_payload",
]
