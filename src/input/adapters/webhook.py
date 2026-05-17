"""CEBA WebhookAdapter.

WebhookAdapter captures webhook deliveries as RawObservation values. It does
not verify signatures or compute meaning.
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


_WEBHOOK_ADAPTER_NAME = "webhook_adapter"
_WEBHOOK_ADAPTER_VERSION = "1.0.0"
_DEFAULT_WEBHOOK_SOURCE_NAME = "webhook"
_DEFAULT_JSON_MIME = "application/json"
_DEFAULT_TEXT_MIME = "text/plain"
_DEFAULT_ENCODING = "utf-8"
_MAX_PROVIDER_LENGTH = 256
_MAX_DELIVERY_ID_LENGTH = 512
_MAX_EVENT_TYPE_LENGTH = 256
_MAX_TEXT_PAYLOAD_LENGTH = 2_000_000
_MAX_METADATA_KEYS = 64

_ALLOWED_METADATA_KEYS = frozenset({
    "provider",
    "delivery_id",
    "signature_validated",
    "event_type",
})


@dataclass(frozen=True, slots=True)
class WebhookSignal:
    payload: JsonValue
    provider: str
    delivery_id: str | None = None
    signature_validated: bool = False
    event_type: str | None = None
    source_name: str | None = None
    raw_mime_type: str | None = None
    raw_encoding: str | None = None
    transport_metadata: dict[str, Any] = field(default_factory=dict)
    source_metadata: dict[str, Any] = field(default_factory=dict)


class WebhookAdapter:
    name = _WEBHOOK_ADAPTER_NAME
    version = _WEBHOOK_ADAPTER_VERSION
    source_type = InputSourceType.WEBHOOK
    capability = AdapterCapability(
        source_type=InputSourceType.WEBHOOK,
        supports_inline_payload=True,
        supports_payload_ref=False,
        supports_streaming=False,
        supports_binary=False,
    )

    def supports(self, signal: Any) -> bool:
        if isinstance(signal, WebhookSignal):
            return True
        if not isinstance(signal, dict):
            return False
        return "payload" in signal and "provider" in signal

    def capture(self, signal: Any, context: AdapterContext) -> RawObservation:
        if not isinstance(context, AdapterContext):
            raise SchemaViolationError(
                "context must be AdapterContext",
                field_path="context",
                details={"type": type(context).__name__},
            )

        webhook_signal = normalize_webhook_signal(signal)
        webhook_metadata = build_webhook_metadata(webhook_signal)
        source_metadata = merge_webhook_metadata(
            context_metadata=context.source_metadata,
            signal_metadata=webhook_signal.source_metadata,
            webhook_metadata=webhook_metadata,
        )
        transport_metadata = merge_transport_metadata(
            context_metadata=context.transport_metadata,
            signal_metadata=webhook_signal.transport_metadata,
        )
        source_name = (
            webhook_signal.source_name
            if webhook_signal.source_name is not None
            else context.source_name
            if context.source_name is not None
            else _DEFAULT_WEBHOOK_SOURCE_NAME
        )
        raw_mime_type = (
            webhook_signal.raw_mime_type
            if webhook_signal.raw_mime_type is not None
            else _DEFAULT_TEXT_MIME
            if isinstance(webhook_signal.payload, str)
            else _DEFAULT_JSON_MIME
        )
        raw_encoding = (
            webhook_signal.raw_encoding
            if webhook_signal.raw_encoding is not None
            else _DEFAULT_ENCODING
        )
        raw_size_bytes = (
            len(webhook_signal.payload.encode(_DEFAULT_ENCODING))
            if isinstance(webhook_signal.payload, str)
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
            source_type=InputSourceType.WEBHOOK,
            signal_payload=webhook_signal.payload,
            context=adapter_context,
            source_name=source_name,
            raw_mime_type=raw_mime_type,
            raw_encoding=raw_encoding,
            raw_size_bytes=raw_size_bytes,
        )


def normalize_webhook_signal(signal: Any) -> WebhookSignal:
    if isinstance(signal, WebhookSignal):
        return WebhookSignal(
            payload=validate_webhook_payload(signal.payload),
            provider=normalize_provider(signal.provider),
            delivery_id=normalize_delivery_id(signal.delivery_id),
            signature_validated=normalize_signature_validated(
                signal.signature_validated
            ),
            event_type=normalize_event_type(signal.event_type),
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
            source_metadata=merge_webhook_metadata(
                signal_metadata=signal.source_metadata,
            ),
        )
    if isinstance(signal, dict):
        if "payload" not in signal:
            raise AdapterCaptureError(
                "webhook signal requires payload",
                field_path="signal.payload",
            )
        if "provider" not in signal:
            raise AdapterCaptureError(
                "webhook signal requires provider",
                field_path="signal.provider",
            )
        return WebhookSignal(
            payload=validate_webhook_payload(signal["payload"]),
            provider=normalize_provider(signal["provider"]),
            delivery_id=normalize_delivery_id(signal.get("delivery_id")),
            signature_validated=normalize_signature_validated(
                signal.get("signature_validated", False)
            ),
            event_type=normalize_event_type(signal.get("event_type")),
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
            source_metadata=merge_webhook_metadata(
                signal_metadata=signal.get("source_metadata"),
            ),
        )
    if signal is None:
        raise AdapterCaptureError(
            "webhook signal must not be None",
            field_path="signal",
        )
    if isinstance(signal, (bytes, bytearray)):
        raise AdapterCaptureError(
            "webhook signal bytes are not supported",
            field_path="signal",
            details={"type": type(signal).__name__},
        )
    if isinstance(signal, (str, list, int, float, bool)):
        raise AdapterCaptureError(
            "webhook signal must be WebhookSignal or dict with payload and provider",
            field_path="signal",
            details={"type": type(signal).__name__},
        )
    raise AdapterCaptureError(
        "unsupported webhook signal type",
        field_path="signal",
        details={"type": type(signal).__name__},
    )


def validate_webhook_payload(payload: JsonValue) -> JsonValue:
    if payload is None:
        raise AdapterCaptureError(
            "webhook payload must not be None",
            field_path="payload",
        )
    if isinstance(payload, str):
        normalized = normalize_text(payload)
        if len(normalized) > _MAX_TEXT_PAYLOAD_LENGTH:
            raise AdapterCaptureError(
                "webhook text payload exceeds maximum length",
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
        "unsupported webhook payload type",
        field_path="payload",
        details={"type": type(payload).__name__},
    )


def normalize_provider(value: str) -> str:
    if not isinstance(value, str):
        raise AdapterCaptureError(
            "provider must be a string",
            field_path="provider",
            details={"type": type(value).__name__},
        )
    normalized = value.strip()
    if not normalized:
        raise AdapterCaptureError(
            "provider must not be empty",
            field_path="provider",
        )
    if len(normalized) > _MAX_PROVIDER_LENGTH:
        raise AdapterCaptureError(
            "provider exceeds maximum length",
            field_path="provider",
            details={"max_length": _MAX_PROVIDER_LENGTH, "length": len(normalized)},
        )
    if _contains_disallowed_control_chars(normalized):
        raise AdapterCaptureError(
            "provider contains disallowed control characters",
            field_path="provider",
        )
    return normalized


def normalize_delivery_id(value: str | None) -> str | None:
    return _normalize_optional_string(
        value,
        field_path="delivery_id",
        max_length=_MAX_DELIVERY_ID_LENGTH,
    )


def normalize_event_type(value: str | None) -> str | None:
    return _normalize_optional_string(
        value,
        field_path="event_type",
        max_length=_MAX_EVENT_TYPE_LENGTH,
    )


def normalize_signature_validated(value: bool) -> bool:
    if not isinstance(value, bool):
        raise AdapterCaptureError(
            "signature_validated must be a boolean",
            field_path="signature_validated",
            details={"type": type(value).__name__},
        )
    return value


def build_webhook_metadata(signal: WebhookSignal) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "provider": signal.provider,
        "signature_validated": signal.signature_validated,
    }
    if signal.delivery_id is not None:
        metadata["delivery_id"] = signal.delivery_id
    if signal.event_type is not None:
        metadata["event_type"] = signal.event_type
    return metadata


def merge_webhook_metadata(
    *,
    context_metadata: Mapping[str, Any] | None = None,
    signal_metadata: Mapping[str, Any] | None = None,
    webhook_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for field_path, metadata in (
        ("context_metadata", context_metadata),
        ("signal_metadata", signal_metadata),
        ("webhook_metadata", webhook_metadata),
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


def create_webhook_adapter() -> WebhookAdapter:
    return WebhookAdapter()


_adapter_protocol_check: InputAdapter = WebhookAdapter()


__all__ = [
    "WebhookAdapter",
    "WebhookSignal",
    "build_webhook_metadata",
    "create_webhook_adapter",
    "merge_transport_metadata",
    "merge_webhook_metadata",
    "normalize_delivery_id",
    "normalize_event_type",
    "normalize_provider",
    "normalize_signature_validated",
    "normalize_webhook_signal",
    "validate_webhook_payload",
]
