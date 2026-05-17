"""Strict schema contract for the CEBA Input Layer.

Input captures external signals as observations only. It does not compute
meaning, intent, emotion, value, telos, world model state, deltas, or memory
updates. CanonicalObservation is the output boundary of Input; parsing starts
after this boundary.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator


class StrictInputModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=False,
        str_strip_whitespace=True,
        use_enum_values=False,
    )


class InputSourceType(str, Enum):
    USER = "user"
    API = "api"
    FILE = "file"
    SHELL = "shell"
    SYSTEM = "system"
    WEBHOOK = "webhook"
    DATABASE = "database"
    LOG = "log"


class ObservationType(str, Enum):
    TEXT_MESSAGE = "text_message"
    COMMAND_INPUT = "command_input"
    COMMAND_OUTPUT = "command_output"
    FILE_UPLOAD = "file_upload"
    FILE_TEXT_EXTRACT = "file_text_extract"
    API_EVENT = "api_event"
    WEBHOOK_EVENT = "webhook_event"
    SYSTEM_EVENT = "system_event"
    LOG_EVENT = "log_event"
    DATABASE_ROW_CHANGE = "database_row_change"
    ERROR_EVENT = "error_event"
    USER_ACTION = "user_action"


class TrustLevel(str, Enum):
    TRUSTED = "trusted"
    INTERNAL = "internal"
    AUTHENTICATED = "authenticated"
    UNTRUSTED = "untrusted"
    UNKNOWN = "unknown"


class ValidationStatus(str, Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


class SecurityStatus(str, Enum):
    PENDING = "pending"
    PASSED = "passed"
    BLOCKED = "blocked"
    REDACTED = "redacted"
    WARNING = "warning"


class PayloadStorageKind(str, Enum):
    INLINE = "inline"
    # Internally managed object reference, such as blob id, S3 key,
    # local storage key, or internal artifact URI.
    REFERENCE = "reference"
    # Externally controlled URI, such as remote URL or third-party object reference.
    EXTERNAL = "external"


class CanonicalContentKind(str, Enum):
    TEXT = "text"
    JSON = "json"
    BINARY_REF = "binary_ref"
    COMMAND = "command"
    EVENT = "event"
    ERROR = "error"


class ErrorSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


def ensure_timezone_aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(timezone.utc)


class InputSource(StrictInputModel):
    source_type: InputSourceType
    source_name: str
    source_id: str | None = None
    trust_level: TrustLevel = TrustLevel.UNKNOWN
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_name")
    @classmethod
    def validate_source_name(cls, value: str) -> str:
        if not value:
            raise ValueError("source_name must not be empty")
        return value


class PayloadRef(StrictInputModel):
    storage_kind: PayloadStorageKind
    uri: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    content_hash: str | None = None

    @field_validator("size_bytes")
    @classmethod
    def validate_size_bytes(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("size_bytes must be greater than or equal to 0")
        return value

    @model_validator(mode="after")
    def validate_storage_location(self) -> PayloadRef:
        if self.storage_kind in {
            PayloadStorageKind.REFERENCE,
            PayloadStorageKind.EXTERNAL,
        } and not self.uri:
            raise ValueError("uri is required for reference or external payloads")
        return self


class TemporalMetadata(StrictInputModel):
    """Reusable timestamp bundle for modules that need grouped temporal metadata."""

    observed_at: datetime | None = None
    received_at: datetime
    normalized_at: datetime | None = None
    recorded_at: datetime | None = None

    @field_validator("received_at")
    @classmethod
    def validate_received_at(cls, value: datetime) -> datetime:
        normalized = ensure_timezone_aware(value)
        if normalized is None:
            raise ValueError("received_at must not be None")
        return normalized

    @field_validator("observed_at", "normalized_at", "recorded_at")
    @classmethod
    def validate_optional_datetimes(cls, value: datetime | None) -> datetime | None:
        return ensure_timezone_aware(value)


class ProvenanceStep(StrictInputModel):
    step_id: UUID = Field(default_factory=uuid4)
    name: str
    component: str
    occurred_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "component")
    @classmethod
    def validate_non_empty_strings(cls, value: str) -> str:
        if not value:
            raise ValueError("value must not be empty")
        return value

    @field_validator("occurred_at")
    @classmethod
    def validate_occurred_at(cls, value: datetime) -> datetime:
        normalized = ensure_timezone_aware(value)
        if normalized is None:
            raise ValueError("occurred_at must not be None")
        return normalized


class Provenance(StrictInputModel):
    source_type: InputSourceType
    source_name: str
    actor_id: str | None = None
    actor_kind: str | None = None
    origin_uri: str | None = None
    origin_host: str | None = None
    capture_method: str
    adapter_name: str
    adapter_version: str
    received_at: datetime
    content_hash: str | None = None
    envelope_hash: str | None = None
    chain: list[ProvenanceStep] = Field(default_factory=list)

    @field_validator("source_name", "capture_method", "adapter_name", "adapter_version")
    @classmethod
    def validate_non_empty_strings(cls, value: str) -> str:
        if not value:
            raise ValueError("value must not be empty")
        return value

    @field_validator("received_at")
    @classmethod
    def validate_received_at(cls, value: datetime) -> datetime:
        normalized = ensure_timezone_aware(value)
        if normalized is None:
            raise ValueError("received_at must not be None")
        return normalized


class ValidationErrorItem(StrictInputModel):
    code: str
    message: str
    field_path: str | None = None
    severity: ErrorSeverity = ErrorSeverity.ERROR


class ValidationWarningItem(StrictInputModel):
    code: str
    message: str
    field_path: str | None = None
    severity: Literal[ErrorSeverity.WARNING] = ErrorSeverity.WARNING


class ValidationReport(StrictInputModel):
    status: ValidationStatus
    is_valid: bool
    errors: list[ValidationErrorItem] = Field(default_factory=list)
    warnings: list[ValidationWarningItem] = Field(default_factory=list)
    schema_version: str = "1.0.0"

    @model_validator(mode="after")
    def validate_status_consistency(self) -> ValidationReport:
        if self.status == ValidationStatus.PASSED:
            if not self.is_valid:
                raise ValueError("is_valid must be True when status is PASSED")
            if self.errors:
                raise ValueError("errors must be empty when status is PASSED")
        if self.status == ValidationStatus.FAILED and self.is_valid:
            raise ValueError("is_valid must be False when status is FAILED")
        return self


class SecurityReport(StrictInputModel):
    status: SecurityStatus
    labels: list[str] = Field(default_factory=list)
    blocked_reason: str | None = None
    redacted_fields: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_blocked_reason(self) -> SecurityReport:
        if self.status == SecurityStatus.BLOCKED and not self.blocked_reason:
            raise ValueError("blocked_reason is required when status is BLOCKED")
        return self


class CanonicalContent(StrictInputModel):
    kind: CanonicalContentKind
    text: str | None = None
    data: dict[str, JsonValue] | None = None
    payload_ref: PayloadRef | None = None
    normalized: bool = False

    @model_validator(mode="after")
    def validate_content_shape(self) -> CanonicalContent:
        if self.kind == CanonicalContentKind.TEXT and self.text is None:
            raise ValueError("text is required for TEXT content")
        if self.kind in {
            CanonicalContentKind.JSON,
            CanonicalContentKind.EVENT,
        } and self.data is None:
            raise ValueError("data is required for JSON or EVENT content")
        if self.kind == CanonicalContentKind.BINARY_REF and self.payload_ref is None:
            raise ValueError("payload_ref is required for BINARY_REF content")
        if self.kind == CanonicalContentKind.COMMAND and self.text is None:
            raise ValueError("text is required for COMMAND content")
        if (
            self.kind == CanonicalContentKind.ERROR
            and self.data is None
            and self.text is None
        ):
            raise ValueError("data or text is required for ERROR content")
        return self


class ObservationMetadata(StrictInputModel):
    mime_type: str | None = None
    encoding: str = "utf-8"
    language_hint: str | None = None
    length_chars: int | None = None
    size_bytes: int | None = None
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    transport_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("length_chars", "size_bytes")
    @classmethod
    def validate_non_negative_ints(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("value must be greater than or equal to 0")
        return value


class RawObservation(StrictInputModel):
    raw_id: UUID = Field(default_factory=uuid4)
    trace_id: UUID = Field(default_factory=uuid4)
    parent_trace_id: UUID | None = None
    source_type: InputSourceType
    source_name: str
    received_at: datetime
    observed_at: datetime | None = None
    raw_payload: JsonValue
    raw_size_bytes: int | None = None
    raw_mime_type: str | None = None
    raw_encoding: str | None = None
    transport_metadata: dict[str, Any] = Field(default_factory=dict)
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    trust_level: TrustLevel = TrustLevel.UNKNOWN
    permissions: list[str] = Field(default_factory=list)
    adapter_name: str
    adapter_version: str

    @field_validator("source_name", "adapter_name", "adapter_version")
    @classmethod
    def validate_non_empty_strings(cls, value: str) -> str:
        if not value:
            raise ValueError("value must not be empty")
        return value

    @field_validator("received_at")
    @classmethod
    def validate_received_at(cls, value: datetime) -> datetime:
        normalized = ensure_timezone_aware(value)
        if normalized is None:
            raise ValueError("received_at must not be None")
        return normalized

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, value: datetime | None) -> datetime | None:
        return ensure_timezone_aware(value)

    @field_validator("raw_size_bytes")
    @classmethod
    def validate_raw_size_bytes(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value < 0:
            raise ValueError("raw_size_bytes must be greater than or equal to 0")
        return value

    @field_validator("raw_payload")
    @classmethod
    def validate_raw_payload(cls, value: JsonValue) -> JsonValue:
        if value is None:
            raise ValueError("raw_payload must not be None")
        return value


class InputEnvelope(StrictInputModel):
    envelope_id: UUID = Field(default_factory=uuid4)
    raw_id: UUID
    trace_id: UUID
    parent_trace_id: UUID | None = None
    source: InputSource
    provenance: Provenance
    payload_ref: PayloadRef | None = None
    inline_payload: JsonValue | None = None
    received_at: datetime
    normalized_at: datetime | None = None
    validation_status: ValidationStatus = ValidationStatus.PENDING
    security_status: SecurityStatus = SecurityStatus.PENDING
    input_schema_version: str = "1.0.0"

    @field_validator("received_at")
    @classmethod
    def validate_received_at(cls, value: datetime) -> datetime:
        normalized = ensure_timezone_aware(value)
        if normalized is None:
            raise ValueError("received_at must not be None")
        return normalized

    @field_validator("normalized_at")
    @classmethod
    def validate_normalized_at(cls, value: datetime | None) -> datetime | None:
        return ensure_timezone_aware(value)

    @model_validator(mode="after")
    def validate_payload_presence(self) -> InputEnvelope:
        if self.payload_ref is None and self.inline_payload is None:
            raise ValueError("payload_ref or inline_payload is required")
        return self


class CanonicalObservation(StrictInputModel):
    observation_id: UUID = Field(default_factory=uuid4)
    trace_id: UUID
    parent_trace_id: UUID | None = None
    source_type: InputSourceType
    observation_type: ObservationType
    content: CanonicalContent
    metadata: ObservationMetadata
    provenance: Provenance
    occurred_at: datetime
    recorded_at: datetime
    language: str | None = None
    encoding: str = "utf-8"
    content_hash: str
    structural_hash: str
    safety_labels: list[str] = Field(default_factory=list)
    validation_report: ValidationReport
    security_report: SecurityReport
    schema_version: str = "1.0.0"

    @field_validator("occurred_at", "recorded_at")
    @classmethod
    def validate_datetimes(cls, value: datetime) -> datetime:
        normalized = ensure_timezone_aware(value)
        if normalized is None:
            raise ValueError("datetime must not be None")
        return normalized

    @field_validator("content_hash", "structural_hash")
    @classmethod
    def validate_non_empty_hashes(cls, value: str) -> str:
        if not value:
            raise ValueError("hash value must not be empty")
        return value

    @model_validator(mode="after")
    def validate_observation_boundary(self) -> CanonicalObservation:
        if not self.validation_report.is_valid:
            raise ValueError(
                "validation_report.is_valid must be True for CanonicalObservation"
            )

        compatible_content_kinds = {
            ObservationType.TEXT_MESSAGE: {CanonicalContentKind.TEXT},
            ObservationType.FILE_TEXT_EXTRACT: {CanonicalContentKind.TEXT},
            ObservationType.COMMAND_INPUT: {
                CanonicalContentKind.COMMAND,
                CanonicalContentKind.TEXT,
            },
            ObservationType.COMMAND_OUTPUT: {
                CanonicalContentKind.COMMAND,
                CanonicalContentKind.TEXT,
            },
            ObservationType.FILE_UPLOAD: {CanonicalContentKind.BINARY_REF},
            ObservationType.API_EVENT: {
                CanonicalContentKind.EVENT,
                CanonicalContentKind.JSON,
            },
            ObservationType.WEBHOOK_EVENT: {
                CanonicalContentKind.EVENT,
                CanonicalContentKind.JSON,
            },
            ObservationType.SYSTEM_EVENT: {
                CanonicalContentKind.EVENT,
                CanonicalContentKind.JSON,
            },
            ObservationType.LOG_EVENT: {
                CanonicalContentKind.EVENT,
                CanonicalContentKind.JSON,
            },
            ObservationType.DATABASE_ROW_CHANGE: {
                CanonicalContentKind.EVENT,
                CanonicalContentKind.JSON,
            },
            ObservationType.USER_ACTION: {
                CanonicalContentKind.EVENT,
                CanonicalContentKind.JSON,
                CanonicalContentKind.TEXT,
            },
            ObservationType.ERROR_EVENT: {CanonicalContentKind.ERROR},
        }
        if self.content.kind not in compatible_content_kinds[self.observation_type]:
            raise ValueError("observation_type is not compatible with content.kind")
        return self


class InputAuditRecord(StrictInputModel):
    audit_id: UUID = Field(default_factory=uuid4)
    trace_id: UUID
    observation_id: UUID | None = None
    source_type: InputSourceType
    observation_type: ObservationType | None = None
    content_hash: str | None = None
    structural_hash: str | None = None
    adapter_name: str
    adapter_version: str
    validation_status: ValidationStatus
    security_status: SecurityStatus
    received_at: datetime
    recorded_at: datetime
    error_code: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("adapter_name", "adapter_version")
    @classmethod
    def validate_non_empty_strings(cls, value: str) -> str:
        if not value:
            raise ValueError("value must not be empty")
        return value

    @field_validator("received_at", "recorded_at", "created_at")
    @classmethod
    def validate_datetimes(cls, value: datetime) -> datetime:
        normalized = ensure_timezone_aware(value)
        if normalized is None:
            raise ValueError("datetime must not be None")
        return normalized


__all__ = [
    "CanonicalContent",
    "CanonicalContentKind",
    "CanonicalObservation",
    "ErrorSeverity",
    "InputAuditRecord",
    "InputEnvelope",
    "InputSource",
    "InputSourceType",
    "JsonValue",
    "ObservationMetadata",
    "ObservationType",
    "PayloadRef",
    "PayloadStorageKind",
    "Provenance",
    "ProvenanceStep",
    "RawObservation",
    "SecurityReport",
    "SecurityStatus",
    "StrictInputModel",
    "TemporalMetadata",
    "TrustLevel",
    "ValidationErrorItem",
    "ValidationReport",
    "ValidationStatus",
    "ValidationWarningItem",
    "ensure_timezone_aware",
]
