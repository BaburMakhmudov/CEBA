"""CEBA FileAdapter.

FileAdapter captures file metadata and reference signals as RawObservation
values. It does not read, execute, or interpret file content.
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
from ois.input.normalization import normalize_mime_type
from ois.input.schema import InputSourceType, RawObservation


_FILE_ADAPTER_NAME = "file_adapter"
_FILE_ADAPTER_VERSION = "1.0.0"
_DEFAULT_FILE_SOURCE_NAME = "file"
_DEFAULT_FILE_MIME = "application/octet-stream"
_DEFAULT_ENCODING = "utf-8"
_MAX_FILENAME_LENGTH = 512
_MAX_STORAGE_URI_LENGTH = 4096
_MAX_UPLOAD_ID_LENGTH = 256
_MAX_METADATA_KEYS = 64

_ALLOWED_METADATA_KEYS = frozenset({
    "filename",
    "mime_type",
    "size_bytes",
    "upload_id",
    "storage_uri",
})


@dataclass(frozen=True, slots=True)
class FileSignal:
    filename: str
    storage_uri: str
    size_bytes: int
    payload: JsonValue | None = None
    mime_type: str | None = None
    upload_id: str | None = None
    source_name: str | None = None
    raw_encoding: str | None = None
    transport_metadata: dict[str, Any] = field(default_factory=dict)
    source_metadata: dict[str, Any] = field(default_factory=dict)


class FileAdapter:
    name = _FILE_ADAPTER_NAME
    version = _FILE_ADAPTER_VERSION
    source_type = InputSourceType.FILE
    capability = AdapterCapability(
        source_type=InputSourceType.FILE,
        supports_inline_payload=True,
        supports_payload_ref=True,
        supports_streaming=False,
        supports_binary=False,
    )

    def supports(self, signal: Any) -> bool:
        if isinstance(signal, FileSignal):
            return True
        if not isinstance(signal, dict):
            return False
        return (
            "filename" in signal
            and "storage_uri" in signal
            and "size_bytes" in signal
        )

    def capture(self, signal: Any, context: AdapterContext) -> RawObservation:
        if not isinstance(context, AdapterContext):
            raise SchemaViolationError(
                "context must be AdapterContext",
                field_path="context",
                details={"type": type(context).__name__},
            )

        file_signal = normalize_file_signal(signal)
        raw_mime_type = (
            file_signal.mime_type
            if file_signal.mime_type is not None
            else normalize_mime_type(None, filename=file_signal.filename)
            or _DEFAULT_FILE_MIME
        )
        file_metadata = build_file_metadata(file_signal)
        source_metadata = merge_file_metadata(
            context_metadata=context.source_metadata,
            signal_metadata=file_signal.source_metadata,
            file_metadata=file_metadata,
        )
        transport_metadata = merge_transport_metadata(
            context_metadata=context.transport_metadata,
            signal_metadata=file_signal.transport_metadata,
        )
        source_name = (
            file_signal.source_name
            if file_signal.source_name is not None
            else context.source_name
            if context.source_name is not None
            else _DEFAULT_FILE_SOURCE_NAME
        )
        raw_encoding = (
            file_signal.raw_encoding
            if file_signal.raw_encoding is not None
            else _DEFAULT_ENCODING
        )
        raw_payload = (
            file_signal.payload if file_signal.payload is not None else file_metadata
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
            source_type=InputSourceType.FILE,
            signal_payload=raw_payload,
            context=adapter_context,
            source_name=source_name,
            raw_mime_type=raw_mime_type,
            raw_encoding=raw_encoding,
            raw_size_bytes=file_signal.size_bytes,
        )


def normalize_file_signal(signal: Any) -> FileSignal:
    if isinstance(signal, FileSignal):
        return FileSignal(
            filename=normalize_filename(signal.filename),
            storage_uri=normalize_storage_uri(signal.storage_uri),
            size_bytes=normalize_file_size(signal.size_bytes),
            payload=validate_file_payload(signal.payload),
            mime_type=normalize_mime_type(signal.mime_type, filename=signal.filename),
            upload_id=normalize_upload_id(signal.upload_id),
            source_name=_normalize_optional_string(
                signal.source_name,
                field_path="source_name",
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
            source_metadata=merge_file_metadata(signal_metadata=signal.source_metadata),
        )
    if isinstance(signal, dict):
        if "filename" not in signal:
            raise AdapterCaptureError(
                "file signal requires filename",
                field_path="signal.filename",
            )
        if "storage_uri" not in signal:
            raise AdapterCaptureError(
                "file signal requires storage_uri",
                field_path="signal.storage_uri",
            )
        if "size_bytes" not in signal:
            raise AdapterCaptureError(
                "file signal requires size_bytes",
                field_path="signal.size_bytes",
            )
        return FileSignal(
            filename=normalize_filename(signal["filename"]),
            storage_uri=normalize_storage_uri(signal["storage_uri"]),
            size_bytes=normalize_file_size(signal["size_bytes"]),
            payload=validate_file_payload(signal.get("payload")),
            mime_type=normalize_mime_type(
                signal.get("mime_type"),
                filename=signal["filename"],
            ),
            upload_id=normalize_upload_id(signal.get("upload_id")),
            source_name=_normalize_optional_string(
                signal.get("source_name"),
                field_path="source_name",
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
            source_metadata=merge_file_metadata(
                signal_metadata=signal.get("source_metadata"),
            ),
        )
    if signal is None:
        raise AdapterCaptureError(
            "file signal must not be None",
            field_path="signal",
        )
    if isinstance(signal, (bytes, bytearray)):
        raise AdapterCaptureError(
            "file signal bytes are not supported",
            field_path="signal",
            details={"type": type(signal).__name__},
        )
    if isinstance(signal, (str, list, int, float, bool)):
        raise AdapterCaptureError(
            "file signal must be FileSignal or dict with filename, storage_uri, and size_bytes",
            field_path="signal",
            details={"type": type(signal).__name__},
        )
    raise AdapterCaptureError(
        "unsupported file signal type",
        field_path="signal",
        details={"type": type(signal).__name__},
    )


def validate_file_payload(payload: JsonValue | None) -> JsonValue | None:
    if payload is None:
        return None
    if isinstance(payload, (dict, list, str, int, float, bool)):
        return payload
    raise AdapterCaptureError(
        "unsupported file payload type",
        field_path="payload",
        details={"type": type(payload).__name__},
    )


def normalize_filename(value: str) -> str:
    if not isinstance(value, str):
        raise AdapterCaptureError(
            "filename must be a string",
            field_path="filename",
            details={"type": type(value).__name__},
        )
    normalized = value.strip()
    if not normalized:
        raise AdapterCaptureError(
            "filename must not be empty",
            field_path="filename",
        )
    if len(normalized) > _MAX_FILENAME_LENGTH:
        raise AdapterCaptureError(
            "filename exceeds maximum length",
            field_path="filename",
            details={"max_length": _MAX_FILENAME_LENGTH, "length": len(normalized)},
        )
    if _contains_disallowed_control_chars(normalized):
        raise AdapterCaptureError(
            "filename contains disallowed control characters",
            field_path="filename",
        )
    if _contains_path_traversal(normalized):
        raise AdapterCaptureError(
            "filename contains path traversal",
            field_path="filename",
        )
    return normalized


def normalize_storage_uri(value: str) -> str:
    if not isinstance(value, str):
        raise AdapterCaptureError(
            "storage_uri must be a string",
            field_path="storage_uri",
            details={"type": type(value).__name__},
        )
    normalized = value.strip()
    if not normalized:
        raise AdapterCaptureError(
            "storage_uri must not be empty",
            field_path="storage_uri",
        )
    if len(normalized) > _MAX_STORAGE_URI_LENGTH:
        raise AdapterCaptureError(
            "storage_uri exceeds maximum length",
            field_path="storage_uri",
            details={
                "max_length": _MAX_STORAGE_URI_LENGTH,
                "length": len(normalized),
            },
        )
    if _contains_disallowed_control_chars(normalized):
        raise AdapterCaptureError(
            "storage_uri contains disallowed control characters",
            field_path="storage_uri",
        )
    if _contains_path_traversal(normalized):
        raise AdapterCaptureError(
            "storage_uri contains path traversal",
            field_path="storage_uri",
        )
    return normalized


def normalize_upload_id(value: str | None) -> str | None:
    return _normalize_optional_string(
        value,
        field_path="upload_id",
        max_length=_MAX_UPLOAD_ID_LENGTH,
    )


def normalize_file_size(value: int) -> int:
    if isinstance(value, bool):
        raise AdapterCaptureError(
            "size_bytes must be an integer",
            field_path="size_bytes",
            details={"type": type(value).__name__},
        )
    if not isinstance(value, int):
        raise AdapterCaptureError(
            "size_bytes must be an integer",
            field_path="size_bytes",
            details={"type": type(value).__name__},
        )
    if value < 0:
        raise AdapterCaptureError(
            "size_bytes must be greater than or equal to 0",
            field_path="size_bytes",
            details={"value": value},
        )
    return value


def build_file_metadata(signal: FileSignal) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "filename": signal.filename,
        "storage_uri": signal.storage_uri,
        "size_bytes": signal.size_bytes,
    }
    if signal.mime_type is not None:
        metadata["mime_type"] = signal.mime_type
    if signal.upload_id is not None:
        metadata["upload_id"] = signal.upload_id
    return metadata


def merge_file_metadata(
    *,
    context_metadata: Mapping[str, Any] | None = None,
    signal_metadata: Mapping[str, Any] | None = None,
    file_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for field_path, metadata in (
        ("context_metadata", context_metadata),
        ("signal_metadata", signal_metadata),
        ("file_metadata", file_metadata),
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
    max_length: int,
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


def _contains_path_traversal(value: str) -> bool:
    normalized = value.lower()
    return "../" in normalized or "..\\" in normalized or "%2e%2e" in normalized


def create_file_adapter() -> FileAdapter:
    return FileAdapter()


_adapter_protocol_check: InputAdapter = FileAdapter()


__all__ = [
    "FileAdapter",
    "FileSignal",
    "build_file_metadata",
    "create_file_adapter",
    "merge_file_metadata",
    "merge_transport_metadata",
    "normalize_file_signal",
    "normalize_file_size",
    "normalize_filename",
    "normalize_storage_uri",
    "normalize_upload_id",
    "validate_file_payload",
]
