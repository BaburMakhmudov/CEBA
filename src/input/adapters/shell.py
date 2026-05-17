"""CEBA ShellAdapter.

ShellAdapter captures shell-originated signals as RawObservation values. It
does not execute commands or compute meaning.
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


_SHELL_ADAPTER_NAME = "shell_adapter"
_SHELL_ADAPTER_VERSION = "1.0.0"
_DEFAULT_SHELL_SOURCE_NAME = "shell"
_DEFAULT_TEXT_MIME = "text/plain"
_DEFAULT_JSON_MIME = "application/json"
_DEFAULT_ENCODING = "utf-8"
_MAX_COMMAND_LENGTH = 100_000
_MAX_TEXT_PAYLOAD_LENGTH = 5_000_000
_MAX_CWD_LENGTH = 4096
_MAX_COMMAND_ID_LENGTH = 256
_MAX_STREAM_LENGTH = 32
_MAX_METADATA_KEYS = 64

_ALLOWED_METADATA_KEYS = frozenset({
    "cwd",
    "exit_code",
    "stream",
    "command_id",
})


class ShellStream(str, Enum):
    STDIN = "stdin"
    STDOUT = "stdout"
    STDERR = "stderr"
    STATUS = "status"
    EVENT = "event"


@dataclass(frozen=True, slots=True)
class ShellSignal:
    payload: JsonValue
    stream: ShellStream | str
    command: str | None = None
    cwd: str | None = None
    exit_code: int | None = None
    command_id: str | None = None
    source_name: str | None = None
    raw_mime_type: str | None = None
    raw_encoding: str | None = None
    transport_metadata: dict[str, Any] = field(default_factory=dict)
    source_metadata: dict[str, Any] = field(default_factory=dict)


class ShellAdapter:
    name = _SHELL_ADAPTER_NAME
    version = _SHELL_ADAPTER_VERSION
    source_type = InputSourceType.SHELL
    capability = AdapterCapability(
        source_type=InputSourceType.SHELL,
        supports_inline_payload=True,
        supports_payload_ref=False,
        supports_streaming=False,
        supports_binary=False,
    )

    def supports(self, signal: Any) -> bool:
        if isinstance(signal, ShellSignal):
            return True
        if not isinstance(signal, dict):
            return False
        return "payload" in signal and "stream" in signal

    def capture(self, signal: Any, context: AdapterContext) -> RawObservation:
        if not isinstance(context, AdapterContext):
            raise SchemaViolationError(
                "context must be AdapterContext",
                field_path="context",
                details={"type": type(context).__name__},
            )

        shell_signal = normalize_shell_signal(signal)
        shell_metadata = build_shell_metadata(shell_signal)
        source_metadata = merge_shell_metadata(
            context_metadata=context.source_metadata,
            signal_metadata=shell_signal.source_metadata,
            shell_metadata=shell_metadata,
        )
        transport_metadata = merge_transport_metadata(
            context_metadata=context.transport_metadata,
            signal_metadata=shell_signal.transport_metadata,
        )
        source_name = (
            shell_signal.source_name
            if shell_signal.source_name is not None
            else context.source_name
            if context.source_name is not None
            else _DEFAULT_SHELL_SOURCE_NAME
        )
        raw_mime_type = (
            shell_signal.raw_mime_type
            if shell_signal.raw_mime_type is not None
            else _DEFAULT_TEXT_MIME
            if isinstance(shell_signal.payload, str)
            else _DEFAULT_JSON_MIME
        )
        raw_encoding = (
            shell_signal.raw_encoding
            if shell_signal.raw_encoding is not None
            else _DEFAULT_ENCODING
        )
        raw_size_bytes = (
            len(shell_signal.payload.encode(_DEFAULT_ENCODING))
            if isinstance(shell_signal.payload, str)
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
            source_type=InputSourceType.SHELL,
            signal_payload=shell_signal.payload,
            context=adapter_context,
            source_name=source_name,
            raw_mime_type=raw_mime_type,
            raw_encoding=raw_encoding,
            raw_size_bytes=raw_size_bytes,
        )


def normalize_shell_signal(signal: Any) -> ShellSignal:
    if isinstance(signal, ShellSignal):
        return ShellSignal(
            payload=validate_shell_payload(signal.payload),
            stream=normalize_shell_stream(signal.stream),
            command=normalize_shell_command(signal.command),
            cwd=normalize_cwd(signal.cwd),
            exit_code=normalize_exit_code(signal.exit_code),
            command_id=normalize_command_id(signal.command_id),
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
            source_metadata=merge_shell_metadata(
                signal_metadata=signal.source_metadata,
            ),
        )
    if isinstance(signal, dict):
        if "payload" not in signal:
            raise AdapterCaptureError(
                "shell signal requires payload",
                field_path="signal.payload",
            )
        if "stream" not in signal:
            raise AdapterCaptureError(
                "shell signal requires stream",
                field_path="signal.stream",
            )
        return ShellSignal(
            payload=validate_shell_payload(signal["payload"]),
            stream=normalize_shell_stream(signal["stream"]),
            command=normalize_shell_command(signal.get("command")),
            cwd=normalize_cwd(signal.get("cwd")),
            exit_code=normalize_exit_code(signal.get("exit_code")),
            command_id=normalize_command_id(signal.get("command_id")),
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
            source_metadata=merge_shell_metadata(
                signal_metadata=signal.get("source_metadata"),
            ),
        )
    if signal is None:
        raise AdapterCaptureError(
            "shell signal must not be None",
            field_path="signal",
        )
    if isinstance(signal, (bytes, bytearray)):
        raise AdapterCaptureError(
            "shell signal bytes are not supported",
            field_path="signal",
            details={"type": type(signal).__name__},
        )
    if isinstance(signal, (str, list, int, float, bool)):
        raise AdapterCaptureError(
            "shell signal must be ShellSignal or dict with payload and stream",
            field_path="signal",
            details={"type": type(signal).__name__},
        )
    raise AdapterCaptureError(
        "unsupported shell signal type",
        field_path="signal",
        details={"type": type(signal).__name__},
    )


def validate_shell_payload(payload: JsonValue) -> JsonValue:
    if payload is None:
        raise AdapterCaptureError(
            "shell payload must not be None",
            field_path="payload",
        )
    if isinstance(payload, str):
        normalized = normalize_text(payload)
        if len(normalized) > _MAX_TEXT_PAYLOAD_LENGTH:
            raise AdapterCaptureError(
                "shell text payload exceeds maximum length",
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
        "unsupported shell payload type",
        field_path="payload",
        details={"type": type(payload).__name__},
    )


def normalize_shell_stream(value: ShellStream | str) -> ShellStream:
    if isinstance(value, ShellStream):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            raise AdapterCaptureError(
                "shell stream must not be empty",
                field_path="stream",
            )
        if len(normalized) > _MAX_STREAM_LENGTH:
            raise AdapterCaptureError(
                "shell stream exceeds maximum length",
                field_path="stream",
                details={"max_length": _MAX_STREAM_LENGTH, "length": len(normalized)},
            )
        for stream in ShellStream:
            if normalized == stream.value:
                return stream
    raise AdapterCaptureError(
        "unsupported shell stream",
        field_path="stream",
        details={"value": value},
    )


def normalize_shell_command(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise AdapterCaptureError(
            "command must be a string",
            field_path="command",
            details={"type": type(value).__name__},
        )
    normalized = normalize_text(value)
    if not normalized:
        return None
    if len(normalized) > _MAX_COMMAND_LENGTH:
        raise AdapterCaptureError(
            "command exceeds maximum length",
            field_path="command",
            details={"max_length": _MAX_COMMAND_LENGTH, "length": len(normalized)},
        )
    return normalized


def normalize_cwd(value: str | None) -> str | None:
    return _normalize_optional_string(
        value,
        field_path="cwd",
        max_length=_MAX_CWD_LENGTH,
    )


def normalize_command_id(value: str | None) -> str | None:
    return _normalize_optional_string(
        value,
        field_path="command_id",
        max_length=_MAX_COMMAND_ID_LENGTH,
    )


def normalize_exit_code(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise AdapterCaptureError(
            "exit_code must be an integer",
            field_path="exit_code",
            details={"type": type(value).__name__},
        )
    if not isinstance(value, int):
        raise AdapterCaptureError(
            "exit_code must be an integer",
            field_path="exit_code",
            details={"type": type(value).__name__},
        )
    return value


def build_shell_metadata(signal: ShellSignal) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "stream": signal.stream.value if isinstance(signal.stream, ShellStream) else signal.stream,
    }
    if signal.command_id is not None:
        metadata["command_id"] = signal.command_id
    if signal.cwd is not None:
        metadata["cwd"] = signal.cwd
    if signal.exit_code is not None:
        metadata["exit_code"] = signal.exit_code
    return metadata


def merge_shell_metadata(
    *,
    context_metadata: Mapping[str, Any] | None = None,
    signal_metadata: Mapping[str, Any] | None = None,
    shell_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for field_path, metadata in (
        ("context_metadata", context_metadata),
        ("signal_metadata", signal_metadata),
        ("shell_metadata", shell_metadata),
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


def create_shell_adapter() -> ShellAdapter:
    return ShellAdapter()


_adapter_protocol_check: InputAdapter = ShellAdapter()


__all__ = [
    "ShellAdapter",
    "ShellSignal",
    "ShellStream",
    "build_shell_metadata",
    "create_shell_adapter",
    "merge_shell_metadata",
    "merge_transport_metadata",
    "normalize_command_id",
    "normalize_cwd",
    "normalize_exit_code",
    "normalize_shell_command",
    "normalize_shell_signal",
    "normalize_shell_stream",
    "validate_shell_payload",
]
