"""CEBA ApiAdapter.

ApiAdapter captures API-originated signals as RawObservation values. It
preserves technical request metadata without computing meaning or intent.
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
from ois.input.schema import InputSourceType, RawObservation


_API_ADAPTER_NAME = "api_adapter"
_API_ADAPTER_VERSION = "1.0.0"
_DEFAULT_API_SOURCE_NAME = "api"
_DEFAULT_API_MIME = "application/json"
_DEFAULT_ENCODING = "utf-8"
_MAX_METHOD_LENGTH = 16
_MAX_PATH_LENGTH = 2048
_MAX_METADATA_KEYS = 64

_ALLOWED_METHODS = frozenset({
    "GET",
    "POST",
    "PUT",
    "PATCH",
    "DELETE",
    "HEAD",
    "OPTIONS",
})

_ALLOWED_METADATA_KEYS = frozenset({
    "method",
    "path",
    "request_id",
    "client_id",
    "auth_scope",
})


@dataclass(frozen=True, slots=True)
class ApiSignal:
    payload: JsonValue
    method: str
    path: str
    request_id: str | None = None
    client_id: str | None = None
    auth_scope: str | None = None
    source_name: str | None = None
    raw_mime_type: str | None = None
    raw_encoding: str | None = None
    transport_metadata: dict[str, Any] = field(default_factory=dict)
    source_metadata: dict[str, Any] = field(default_factory=dict)


class ApiAdapter:
    name = _API_ADAPTER_NAME
    version = _API_ADAPTER_VERSION
    source_type = InputSourceType.API
    capability = AdapterCapability(
        source_type=InputSourceType.API,
        supports_inline_payload=True,
        supports_payload_ref=False,
        supports_streaming=False,
        supports_binary=False,
    )

    def supports(self, signal: Any) -> bool:
        if isinstance(signal, ApiSignal):
            return True
        if not isinstance(signal, dict):
            return False
        return "payload" in signal and "method" in signal and "path" in signal

    def capture(self, signal: Any, context: AdapterContext) -> RawObservation:
        if not isinstance(context, AdapterContext):
            raise SchemaViolationError(
                "context must be AdapterContext",
                field_path="context",
                details={"type": type(context).__name__},
            )

        api_signal = normalize_api_signal(signal)
        source_metadata = merge_api_metadata(
            context_metadata=context.source_metadata,
            signal_metadata=api_signal.source_metadata,
            api_metadata=build_api_metadata(api_signal),
        )
        transport_metadata = merge_transport_metadata(
            context_metadata=context.transport_metadata,
            signal_metadata=api_signal.transport_metadata,
        )
        source_name = (
            api_signal.source_name
            if api_signal.source_name is not None
            else context.source_name
            if context.source_name is not None
            else _DEFAULT_API_SOURCE_NAME
        )
        raw_mime_type = (
            api_signal.raw_mime_type
            if api_signal.raw_mime_type is not None
            else _DEFAULT_API_MIME
        )
        raw_encoding = (
            api_signal.raw_encoding
            if api_signal.raw_encoding is not None
            else _DEFAULT_ENCODING
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
            source_type=InputSourceType.API,
            signal_payload=api_signal.payload,
            context=adapter_context,
            source_name=source_name,
            raw_mime_type=raw_mime_type,
            raw_encoding=raw_encoding,
            raw_size_bytes=None,
        )


def normalize_api_signal(signal: Any) -> ApiSignal:
    if isinstance(signal, ApiSignal):
        return ApiSignal(
            payload=validate_api_payload(signal.payload),
            method=normalize_api_method(signal.method),
            path=normalize_api_path(signal.path),
            request_id=_normalize_optional_string(
                signal.request_id,
                field_path="request_id",
            ),
            client_id=_normalize_optional_string(
                signal.client_id,
                field_path="client_id",
            ),
            auth_scope=_normalize_optional_string(
                signal.auth_scope,
                field_path="auth_scope",
            ),
            source_name=_normalize_optional_string(
                signal.source_name,
                field_path="source_name",
            ),
            raw_mime_type=_normalize_optional_string(
                signal.raw_mime_type,
                field_path="raw_mime_type",
            ),
            raw_encoding=_normalize_optional_string(
                signal.raw_encoding,
                field_path="raw_encoding",
            ),
            transport_metadata=merge_transport_metadata(
                signal_metadata=signal.transport_metadata,
            ),
            source_metadata=merge_api_metadata(signal_metadata=signal.source_metadata),
        )
    if isinstance(signal, dict):
        if "payload" not in signal:
            raise AdapterCaptureError(
                "api signal requires payload",
                field_path="signal.payload",
            )
        if "method" not in signal:
            raise AdapterCaptureError(
                "api signal requires method",
                field_path="signal.method",
            )
        if "path" not in signal:
            raise AdapterCaptureError(
                "api signal requires path",
                field_path="signal.path",
            )
        return ApiSignal(
            payload=validate_api_payload(signal["payload"]),
            method=normalize_api_method(signal["method"]),
            path=normalize_api_path(signal["path"]),
            request_id=_normalize_optional_string(
                signal.get("request_id"),
                field_path="request_id",
            ),
            client_id=_normalize_optional_string(
                signal.get("client_id"),
                field_path="client_id",
            ),
            auth_scope=_normalize_optional_string(
                signal.get("auth_scope"),
                field_path="auth_scope",
            ),
            source_name=_normalize_optional_string(
                signal.get("source_name"),
                field_path="source_name",
            ),
            raw_mime_type=_normalize_optional_string(
                signal.get("raw_mime_type"),
                field_path="raw_mime_type",
            ),
            raw_encoding=_normalize_optional_string(
                signal.get("raw_encoding"),
                field_path="raw_encoding",
            ),
            transport_metadata=merge_transport_metadata(
                signal_metadata=signal.get("transport_metadata"),
            ),
            source_metadata=merge_api_metadata(
                signal_metadata=signal.get("source_metadata"),
            ),
        )
    if signal is None:
        raise AdapterCaptureError(
            "api signal must not be None",
            field_path="signal",
        )
    if isinstance(signal, (bytes, bytearray)):
        raise AdapterCaptureError(
            "api signal bytes are not supported",
            field_path="signal",
            details={"type": type(signal).__name__},
        )
    if isinstance(signal, (str, list, int, float, bool)):
        raise AdapterCaptureError(
            "api signal must be ApiSignal or dict with payload, method, and path",
            field_path="signal",
            details={"type": type(signal).__name__},
        )
    raise AdapterCaptureError(
        "unsupported api signal type",
        field_path="signal",
        details={"type": type(signal).__name__},
    )


def validate_api_payload(payload: JsonValue) -> JsonValue:
    if payload is None:
        raise AdapterCaptureError(
            "api payload must not be None",
            field_path="payload",
        )
    if isinstance(payload, (dict, list, str, int, float, bool)):
        return payload
    raise AdapterCaptureError(
        "unsupported api payload type",
        field_path="payload",
        details={"type": type(payload).__name__},
    )


def normalize_api_method(value: str) -> str:
    if not isinstance(value, str):
        raise AdapterCaptureError(
            "api method must be a string",
            field_path="method",
            details={"type": type(value).__name__},
        )
    normalized = value.strip().upper()
    if not normalized:
        raise AdapterCaptureError(
            "api method must not be empty",
            field_path="method",
        )
    if len(normalized) > _MAX_METHOD_LENGTH:
        raise AdapterCaptureError(
            "api method exceeds maximum length",
            field_path="method",
            details={"max_length": _MAX_METHOD_LENGTH, "length": len(normalized)},
        )
    if normalized not in _ALLOWED_METHODS:
        raise AdapterCaptureError(
            "api method is not supported",
            field_path="method",
            details={"method": normalized},
        )
    return normalized


def normalize_api_path(value: str) -> str:
    if not isinstance(value, str):
        raise AdapterCaptureError(
            "api path must be a string",
            field_path="path",
            details={"type": type(value).__name__},
        )
    normalized = value.strip()
    if not normalized:
        raise AdapterCaptureError(
            "api path must not be empty",
            field_path="path",
        )
    if len(normalized) > _MAX_PATH_LENGTH:
        raise AdapterCaptureError(
            "api path exceeds maximum length",
            field_path="path",
            details={"max_length": _MAX_PATH_LENGTH, "length": len(normalized)},
        )
    if not normalized.startswith("/"):
        raise AdapterCaptureError(
            "api path must start with '/'",
            field_path="path",
        )
    if _contains_disallowed_control_chars(normalized):
        raise AdapterCaptureError(
            "api path contains disallowed control characters",
            field_path="path",
        )
    return normalized


def build_api_metadata(signal: ApiSignal) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "method": signal.method,
        "path": signal.path,
    }
    if signal.request_id is not None:
        metadata["request_id"] = signal.request_id
    if signal.client_id is not None:
        metadata["client_id"] = signal.client_id
    if signal.auth_scope is not None:
        metadata["auth_scope"] = signal.auth_scope
    return metadata


def merge_api_metadata(
    *,
    context_metadata: Mapping[str, Any] | None = None,
    signal_metadata: Mapping[str, Any] | None = None,
    api_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for field_path, metadata in (
        ("context_metadata", context_metadata),
        ("signal_metadata", signal_metadata),
        ("api_metadata", api_metadata),
    ):
        if metadata is None:
            continue
        if not isinstance(metadata, Mapping):
            raise SchemaViolationError(
                f"{field_path} must be a mapping",
                field_path=field_path,
                details={"type": type(metadata).__name__},
            )
        if len(metadata) > _MAX_METADATA_KEYS:
            raise SchemaViolationError(
                f"{field_path} exceeds maximum key count",
                field_path=field_path,
                details={"max_keys": _MAX_METADATA_KEYS, "key_count": len(metadata)},
            )
        for key, value in metadata.items():
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
        if metadata is None:
            continue
        if not isinstance(metadata, Mapping):
            raise SchemaViolationError(
                f"{field_path} must be a mapping",
                field_path=field_path,
                details={"type": type(metadata).__name__},
            )
        merged.update(metadata)
    return merged


def _normalize_optional_string(
    value: str | None,
    *,
    field_path: str,
    max_length: int = 256,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SchemaViolationError(
            f"{field_path} must be a string",
            field_path=field_path,
            details={"type": type(value).__name__},
        )
    normalized = value.strip()
    if not normalized:
        return None
    if len(normalized) > max_length:
        raise SchemaViolationError(
            f"{field_path} exceeds maximum length",
            field_path=field_path,
            details={"max_length": max_length, "length": len(normalized)},
        )
    if _contains_disallowed_control_chars(normalized):
        raise SchemaViolationError(
            f"{field_path} contains disallowed control characters",
            field_path=field_path,
        )
    return normalized


def _contains_disallowed_control_chars(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def create_api_adapter() -> ApiAdapter:
    return ApiAdapter()


_adapter_protocol_check: InputAdapter = ApiAdapter()


__all__ = [
    "ApiAdapter",
    "ApiSignal",
    "build_api_metadata",
    "create_api_adapter",
    "merge_api_metadata",
    "merge_transport_metadata",
    "normalize_api_method",
    "normalize_api_path",
    "normalize_api_signal",
    "validate_api_payload",
]
