"""CEBA UserAdapter.

UserAdapter captures user-originated signals as RawObservation values. It
preserves technical input structure without computing meaning or intent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
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


_USER_ADAPTER_NAME = "user_adapter"
_USER_ADAPTER_VERSION = "1.0.0"
_DEFAULT_USER_SOURCE_NAME = "user"
_DEFAULT_TEXT_MIME = "text/plain"
_DEFAULT_ENCODING = "utf-8"
_MAX_USER_TEXT_LENGTH = 1_000_000
_RESERVED_STRUCTURED_SIGNAL_KEYS = frozenset({
    "method",
    "path",
    "filename",
    "storage_uri",
    "size_bytes",
    "stream",
    "component",
    "event_name",
    "provider",
    "database",
    "table",
    "operation",
    "service",
    "level",
})


@dataclass(frozen=True, slots=True)
class UserSignal:
    payload: JsonValue
    source_name: str | None = None
    raw_mime_type: str | None = None
    raw_encoding: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class UserAdapter:
    name = _USER_ADAPTER_NAME
    version = _USER_ADAPTER_VERSION
    source_type = InputSourceType.USER
    capability = AdapterCapability(
        source_type=InputSourceType.USER,
        supports_inline_payload=True,
        supports_payload_ref=False,
        supports_streaming=False,
        supports_binary=False,
    )

    def supports(self, signal: Any) -> bool:
        if signal is None:
            return False
        if isinstance(signal, (bytes, bytearray)):
            return False
        if isinstance(signal, UserSignal):
            return True
        if isinstance(signal, dict):
            return not any(key in signal for key in _RESERVED_STRUCTURED_SIGNAL_KEYS)
        return isinstance(signal, (str, list, int, float, bool))

    def capture(self, signal: Any, context: AdapterContext) -> RawObservation:
        if not isinstance(context, AdapterContext):
            raise SchemaViolationError(
                "context must be AdapterContext",
                field_path="context",
                details={"type": type(context).__name__},
            )

        user_signal = normalize_user_signal(signal)
        source_name = (
            user_signal.source_name
            if user_signal.source_name is not None
            else context.source_name
            if context.source_name is not None
            else _DEFAULT_USER_SOURCE_NAME
        )
        raw_mime_type = (
            user_signal.raw_mime_type
            if user_signal.raw_mime_type is not None
            else _DEFAULT_TEXT_MIME
            if isinstance(user_signal.payload, str)
            else "application/json"
        )
        raw_encoding = (
            user_signal.raw_encoding
            if user_signal.raw_encoding is not None
            else _DEFAULT_ENCODING
        )
        raw_size_bytes = (
            len(user_signal.payload.encode(_DEFAULT_ENCODING))
            if isinstance(user_signal.payload, str)
            else None
        )
        merged_source_metadata = merge_user_metadata(
            context_metadata=context.source_metadata,
            signal_metadata=user_signal.metadata,
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
            transport_metadata=context.transport_metadata,
            source_metadata=merged_source_metadata,
        )

        return build_raw_observation_from_adapter(
            adapter_name=self.name,
            adapter_version=self.version,
            source_type=InputSourceType.USER,
            signal_payload=user_signal.payload,
            context=adapter_context,
            source_name=source_name,
            raw_mime_type=raw_mime_type,
            raw_encoding=raw_encoding,
            raw_size_bytes=raw_size_bytes,
        )


def normalize_user_signal(signal: Any) -> UserSignal:
    if isinstance(signal, UserSignal):
        payload = validate_user_payload(signal.payload)
        metadata = merge_user_metadata(signal_metadata=signal.metadata)
        return UserSignal(
            payload=payload,
            source_name=signal.source_name,
            raw_mime_type=signal.raw_mime_type,
            raw_encoding=signal.raw_encoding,
            metadata=metadata,
        )
    if isinstance(signal, str):
        payload = validate_user_payload(signal)
        return UserSignal(
            payload=payload,
            raw_mime_type=_DEFAULT_TEXT_MIME,
            raw_encoding=_DEFAULT_ENCODING,
        )
    if isinstance(signal, (dict, list, int, float, bool)):
        return UserSignal(
            payload=signal,
            raw_mime_type="application/json",
            raw_encoding=_DEFAULT_ENCODING,
        )
    if signal is None:
        raise AdapterCaptureError(
            "user signal must not be None",
            field_path="signal",
        )
    if isinstance(signal, (bytes, bytearray)):
        raise AdapterCaptureError(
            "user signal bytes are not supported",
            field_path="signal",
            details={"type": type(signal).__name__},
        )
    raise AdapterCaptureError(
        "unsupported user signal type",
        field_path="signal",
        details={"type": type(signal).__name__},
    )


def validate_user_payload(payload: JsonValue) -> JsonValue:
    if payload is None:
        raise AdapterCaptureError(
            "user payload must not be None",
            field_path="payload",
        )
    if isinstance(payload, str):
        normalized = normalize_text(payload)
        if len(normalized) > _MAX_USER_TEXT_LENGTH:
            raise AdapterCaptureError(
                "user text payload exceeds maximum length",
                field_path="payload",
                details={
                    "max_length": _MAX_USER_TEXT_LENGTH,
                    "length": len(normalized),
                },
            )
        return normalized
    if isinstance(payload, (dict, list, int, float, bool)):
        return payload
    raise AdapterCaptureError(
        "unsupported user payload type",
        field_path="payload",
        details={"type": type(payload).__name__},
    )


def merge_user_metadata(
    *,
    context_metadata: Mapping[str, Any] | None = None,
    signal_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if context_metadata is not None and not isinstance(context_metadata, Mapping):
        raise SchemaViolationError(
            "context_metadata must be a mapping",
            field_path="context_metadata",
            details={"type": type(context_metadata).__name__},
        )
    if signal_metadata is not None and not isinstance(signal_metadata, Mapping):
        raise SchemaViolationError(
            "signal_metadata must be a mapping",
            field_path="signal_metadata",
            details={"type": type(signal_metadata).__name__},
        )

    merged: dict[str, Any] = {}
    if context_metadata is not None:
        merged.update(context_metadata)
    if signal_metadata is not None:
        merged.update(signal_metadata)
    return merged


def create_user_adapter() -> UserAdapter:
    return UserAdapter()


_adapter_protocol_check: InputAdapter = UserAdapter()


__all__ = [
    "UserAdapter",
    "UserSignal",
    "create_user_adapter",
    "merge_user_metadata",
    "normalize_user_signal",
    "validate_user_payload",
]
