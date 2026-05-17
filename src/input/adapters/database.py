"""CEBA DatabaseAdapter.

DatabaseAdapter captures already-observed database change/event signals as
RawObservation values. It does not connect to or query databases, and it does
not compute meaning.
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


_DATABASE_ADAPTER_NAME = "database_adapter"
_DATABASE_ADAPTER_VERSION = "1.0.0"
_DEFAULT_DATABASE_SOURCE_NAME = "database"
_DEFAULT_JSON_MIME = "application/json"
_DEFAULT_TEXT_MIME = "text/plain"
_DEFAULT_ENCODING = "utf-8"
_MAX_DATABASE_NAME_LENGTH = 256
_MAX_TABLE_NAME_LENGTH = 256
_MAX_ROW_ID_LENGTH = 512
_MAX_TRANSACTION_ID_LENGTH = 512
_MAX_OPERATION_LENGTH = 32
_MAX_TEXT_PAYLOAD_LENGTH = 2_000_000
_MAX_METADATA_KEYS = 64

_ALLOWED_METADATA_KEYS = frozenset({
    "database",
    "table",
    "row_id",
    "operation",
    "transaction_id",
})

_ALLOWED_SIGNAL_KEYS = frozenset({
    "payload",
    "database",
    "table",
    "operation",
    "row_id",
    "transaction_id",
    "source_name",
    "raw_mime_type",
    "raw_encoding",
    "transport_metadata",
    "source_metadata",
})


class DatabaseOperation(str, Enum):
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    UPSERT = "upsert"
    SNAPSHOT = "snapshot"
    EVENT = "event"


@dataclass(frozen=True, slots=True)
class DatabaseSignal:
    payload: JsonValue
    database: str
    table: str
    operation: DatabaseOperation | str
    row_id: str | None = None
    transaction_id: str | None = None
    source_name: str | None = None
    raw_mime_type: str | None = None
    raw_encoding: str | None = None
    transport_metadata: dict[str, Any] = field(default_factory=dict)
    source_metadata: dict[str, Any] = field(default_factory=dict)


class DatabaseAdapter:
    name = _DATABASE_ADAPTER_NAME
    version = _DATABASE_ADAPTER_VERSION
    source_type = InputSourceType.DATABASE
    capability = AdapterCapability(
        source_type=InputSourceType.DATABASE,
        supports_inline_payload=True,
        supports_payload_ref=False,
        supports_streaming=False,
        supports_binary=False,
    )

    def supports(self, signal: Any) -> bool:
        if isinstance(signal, DatabaseSignal):
            return True
        if not isinstance(signal, dict):
            return False
        return (
            "payload" in signal
            and "database" in signal
            and "table" in signal
            and "operation" in signal
        )

    def capture(self, signal: Any, context: AdapterContext) -> RawObservation:
        if not isinstance(context, AdapterContext):
            raise SchemaViolationError(
                "context must be AdapterContext",
                field_path="context",
                details={"type": type(context).__name__},
            )

        database_signal = normalize_database_signal(signal)
        database_metadata = build_database_metadata(database_signal)
        source_metadata = merge_database_metadata(
            context_metadata=context.source_metadata,
            signal_metadata=database_signal.source_metadata,
            database_metadata=database_metadata,
        )
        transport_metadata = merge_transport_metadata(
            context_metadata=context.transport_metadata,
            signal_metadata=database_signal.transport_metadata,
        )
        source_name = (
            database_signal.source_name
            if database_signal.source_name is not None
            else context.source_name
            if context.source_name is not None
            else _DEFAULT_DATABASE_SOURCE_NAME
        )
        raw_mime_type = (
            database_signal.raw_mime_type
            if database_signal.raw_mime_type is not None
            else _DEFAULT_TEXT_MIME
            if isinstance(database_signal.payload, str)
            else _DEFAULT_JSON_MIME
        )
        raw_encoding = (
            database_signal.raw_encoding
            if database_signal.raw_encoding is not None
            else _DEFAULT_ENCODING
        )
        raw_size_bytes = (
            len(database_signal.payload.encode(_DEFAULT_ENCODING))
            if isinstance(database_signal.payload, str)
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
            source_type=InputSourceType.DATABASE,
            signal_payload=database_signal.payload,
            context=adapter_context,
            source_name=source_name,
            raw_mime_type=raw_mime_type,
            raw_encoding=raw_encoding,
            raw_size_bytes=raw_size_bytes,
        )


def normalize_database_signal(signal: Any) -> DatabaseSignal:
    if isinstance(signal, DatabaseSignal):
        return DatabaseSignal(
            payload=validate_database_payload(signal.payload),
            database=normalize_database_name(signal.database),
            table=normalize_table_name(signal.table),
            operation=normalize_database_operation(signal.operation),
            row_id=normalize_row_id(signal.row_id),
            transaction_id=normalize_transaction_id(signal.transaction_id),
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
            source_metadata=merge_database_metadata(
                signal_metadata=signal.source_metadata,
            ),
        )
    if isinstance(signal, dict):
        unsupported_keys = set(signal) - _ALLOWED_SIGNAL_KEYS
        if unsupported_keys:
            raise AdapterCaptureError(
                "database signal contains unsupported keys",
                field_path="signal",
                details={"unsupported_keys": sorted(unsupported_keys)},
            )
        if "payload" not in signal:
            raise AdapterCaptureError(
                "database signal requires payload",
                field_path="signal.payload",
            )
        if "database" not in signal:
            raise AdapterCaptureError(
                "database signal requires database",
                field_path="signal.database",
            )
        if "table" not in signal:
            raise AdapterCaptureError(
                "database signal requires table",
                field_path="signal.table",
            )
        if "operation" not in signal:
            raise AdapterCaptureError(
                "database signal requires operation",
                field_path="signal.operation",
            )
        return DatabaseSignal(
            payload=validate_database_payload(signal["payload"]),
            database=normalize_database_name(signal["database"]),
            table=normalize_table_name(signal["table"]),
            operation=normalize_database_operation(signal["operation"]),
            row_id=normalize_row_id(signal.get("row_id")),
            transaction_id=normalize_transaction_id(signal.get("transaction_id")),
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
            source_metadata=merge_database_metadata(
                signal_metadata=signal.get("source_metadata"),
            ),
        )
    if signal is None:
        raise AdapterCaptureError(
            "database signal must not be None",
            field_path="signal",
        )
    if isinstance(signal, (bytes, bytearray)):
        raise AdapterCaptureError(
            "database signal bytes are not supported",
            field_path="signal",
            details={"type": type(signal).__name__},
        )
    if isinstance(signal, (str, list, int, float, bool)):
        raise AdapterCaptureError(
            "database signal must be DatabaseSignal or dict with payload, database, table, and operation",
            field_path="signal",
            details={"type": type(signal).__name__},
        )
    raise AdapterCaptureError(
        "unsupported database signal type",
        field_path="signal",
        details={"type": type(signal).__name__},
    )


def validate_database_payload(payload: JsonValue) -> JsonValue:
    if payload is None:
        raise AdapterCaptureError(
            "database payload must not be None",
            field_path="payload",
        )
    if isinstance(payload, str):
        try:
            normalized = normalize_text(payload)
        except Exception as error:
            raise AdapterCaptureError(
                "database text payload normalization failed",
                field_path="payload",
                details={"original_type": error.__class__.__name__},
            ) from error
        if len(normalized) > _MAX_TEXT_PAYLOAD_LENGTH:
            raise AdapterCaptureError(
                "database text payload exceeds maximum length",
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
        "unsupported database payload type",
        field_path="payload",
        details={"type": type(payload).__name__},
    )


def normalize_database_name(value: str) -> str:
    if not isinstance(value, str):
        raise AdapterCaptureError(
            "database must be a string",
            field_path="database",
            details={"type": type(value).__name__},
        )
    normalized = value.strip()
    if not normalized:
        raise AdapterCaptureError(
            "database must not be empty",
            field_path="database",
        )
    if len(normalized) > _MAX_DATABASE_NAME_LENGTH:
        raise AdapterCaptureError(
            "database exceeds maximum length",
            field_path="database",
            details={
                "max_length": _MAX_DATABASE_NAME_LENGTH,
                "length": len(normalized),
            },
        )
    if _contains_disallowed_control_chars(normalized):
        raise AdapterCaptureError(
            "database contains disallowed control characters",
            field_path="database",
        )
    return normalized


def normalize_table_name(value: str) -> str:
    if not isinstance(value, str):
        raise AdapterCaptureError(
            "table must be a string",
            field_path="table",
            details={"type": type(value).__name__},
        )
    normalized = value.strip()
    if not normalized:
        raise AdapterCaptureError(
            "table must not be empty",
            field_path="table",
        )
    if len(normalized) > _MAX_TABLE_NAME_LENGTH:
        raise AdapterCaptureError(
            "table exceeds maximum length",
            field_path="table",
            details={
                "max_length": _MAX_TABLE_NAME_LENGTH,
                "length": len(normalized),
            },
        )
    if _contains_disallowed_control_chars(normalized):
        raise AdapterCaptureError(
            "table contains disallowed control characters",
            field_path="table",
        )
    return normalized


def normalize_database_operation(value: DatabaseOperation | str) -> DatabaseOperation:
    if isinstance(value, DatabaseOperation):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            raise AdapterCaptureError(
                "operation must not be empty",
                field_path="operation",
            )
        if len(normalized) > _MAX_OPERATION_LENGTH:
            raise AdapterCaptureError(
                "operation exceeds maximum length",
                field_path="operation",
                details={
                    "max_length": _MAX_OPERATION_LENGTH,
                    "length": len(normalized),
                },
            )
        try:
            return DatabaseOperation(normalized)
        except ValueError as error:
            raise AdapterCaptureError(
                "operation is not supported",
                field_path="operation",
                details={"operation": normalized},
            ) from error
    raise AdapterCaptureError(
        "operation must be DatabaseOperation or string",
        field_path="operation",
        details={"type": type(value).__name__},
    )


def normalize_row_id(value: str | None) -> str | None:
    return _normalize_optional_string(
        value,
        field_path="row_id",
        max_length=_MAX_ROW_ID_LENGTH,
    )


def normalize_transaction_id(value: str | None) -> str | None:
    return _normalize_optional_string(
        value,
        field_path="transaction_id",
        max_length=_MAX_TRANSACTION_ID_LENGTH,
    )


def build_database_metadata(signal: DatabaseSignal) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "database": signal.database,
        "table": signal.table,
        "operation": signal.operation.value
        if isinstance(signal.operation, DatabaseOperation)
        else signal.operation,
    }
    if signal.row_id is not None:
        metadata["row_id"] = signal.row_id
    if signal.transaction_id is not None:
        metadata["transaction_id"] = signal.transaction_id
    return metadata


def merge_database_metadata(
    *,
    context_metadata: Mapping[str, Any] | None = None,
    signal_metadata: Mapping[str, Any] | None = None,
    database_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for field_path, metadata in (
        ("context_metadata", context_metadata),
        ("signal_metadata", signal_metadata),
        ("database_metadata", database_metadata),
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


def create_database_adapter() -> DatabaseAdapter:
    return DatabaseAdapter()


_adapter_protocol_check: InputAdapter = DatabaseAdapter()


__all__ = [
    "DatabaseAdapter",
    "DatabaseOperation",
    "DatabaseSignal",
    "build_database_metadata",
    "create_database_adapter",
    "merge_database_metadata",
    "merge_transport_metadata",
    "normalize_database_name",
    "normalize_database_operation",
    "normalize_database_signal",
    "normalize_row_id",
    "normalize_table_name",
    "normalize_transaction_id",
    "validate_database_payload",
]
