"""Typed CEBA Input Layer errors.

These exceptions represent technical boundary failures during input capture,
validation, normalization, security handling, canonicalization, and dispatch.
They do not compute meaning, infer intent, or update memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import UUID


class InputErrorSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class InputErrorCategory(str, Enum):
    ADAPTER = "adapter"
    VALIDATION = "validation"
    SECURITY = "security"
    NORMALIZATION = "normalization"
    CANONICALIZATION = "canonicalization"
    DISPATCH = "dispatch"
    PROVENANCE = "provenance"
    PAYLOAD = "payload"
    ENCODING = "encoding"
    SOURCE = "source"
    FINGERPRINT = "fingerprint"
    AUDIT = "audit"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class InputErrorPayload:
    code: str
    message: str
    category: InputErrorCategory
    severity: InputErrorSeverity = InputErrorSeverity.ERROR
    recoverable: bool = False
    trace_id: UUID | None = None
    raw_id: UUID | None = None
    observation_id: UUID | None = None
    field_path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("code must not be empty")
        if not self.message:
            raise ValueError("message must not be empty")
        if not isinstance(self.details, dict):
            raise TypeError("details must be a dict")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "recoverable": self.recoverable,
            "trace_id": str(self.trace_id) if self.trace_id is not None else None,
            "raw_id": str(self.raw_id) if self.raw_id is not None else None,
            "observation_id": (
                str(self.observation_id)
                if self.observation_id is not None
                else None
            ),
            "field_path": self.field_path,
            "details": self.details,
        }


class CEBAInputError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        category: InputErrorCategory = InputErrorCategory.UNKNOWN,
        severity: InputErrorSeverity = InputErrorSeverity.ERROR,
        recoverable: bool = False,
        trace_id: UUID | None = None,
        raw_id: UUID | None = None,
        observation_id: UUID | None = None,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.payload = InputErrorPayload(
            code=code,
            message=message,
            category=category,
            severity=severity,
            recoverable=recoverable,
            trace_id=trace_id,
            raw_id=raw_id,
            observation_id=observation_id,
            field_path=field_path,
            details=details if details is not None else {},
        )
        super().__init__(message)

    @property
    def code(self) -> str:
        return self.payload.code

    @property
    def category(self) -> InputErrorCategory:
        return self.payload.category

    @property
    def severity(self) -> InputErrorSeverity:
        return self.payload.severity

    @property
    def recoverable(self) -> bool:
        return self.payload.recoverable

    @property
    def trace_id(self) -> UUID | None:
        return self.payload.trace_id

    @property
    def raw_id(self) -> UUID | None:
        return self.payload.raw_id

    @property
    def observation_id(self) -> UUID | None:
        return self.payload.observation_id

    @property
    def field_path(self) -> str | None:
        return self.payload.field_path

    @property
    def details(self) -> dict[str, Any]:
        return self.payload.details

    def to_dict(self) -> dict[str, Any]:
        return self.payload.to_dict()

    def __str__(self) -> str:
        return f"[{self.code}] {self.payload.message}"


class UnsupportedInputError(CEBAInputError):
    def __init__(
        self,
        message: str,
        *,
        trace_id: UUID | None = None,
        raw_id: UUID | None = None,
        observation_id: UUID | None = None,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code="input.unsupported",
            category=InputErrorCategory.SOURCE,
            severity=InputErrorSeverity.ERROR,
            recoverable=False,
            trace_id=trace_id,
            raw_id=raw_id,
            observation_id=observation_id,
            field_path=field_path,
            details=details,
        )


class AdapterResolutionError(CEBAInputError):
    def __init__(
        self,
        message: str,
        *,
        trace_id: UUID | None = None,
        raw_id: UUID | None = None,
        observation_id: UUID | None = None,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code="input.adapter_resolution_failed",
            category=InputErrorCategory.ADAPTER,
            severity=InputErrorSeverity.ERROR,
            recoverable=False,
            trace_id=trace_id,
            raw_id=raw_id,
            observation_id=observation_id,
            field_path=field_path,
            details=details,
        )


class AdapterCaptureError(CEBAInputError):
    def __init__(
        self,
        message: str,
        *,
        trace_id: UUID | None = None,
        raw_id: UUID | None = None,
        observation_id: UUID | None = None,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code="input.adapter_capture_failed",
            category=InputErrorCategory.ADAPTER,
            severity=InputErrorSeverity.ERROR,
            recoverable=True,
            trace_id=trace_id,
            raw_id=raw_id,
            observation_id=observation_id,
            field_path=field_path,
            details=details,
        )


class ValidationFailedError(CEBAInputError):
    def __init__(
        self,
        message: str,
        *,
        trace_id: UUID | None = None,
        raw_id: UUID | None = None,
        observation_id: UUID | None = None,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code="input.validation_failed",
            category=InputErrorCategory.VALIDATION,
            severity=InputErrorSeverity.ERROR,
            recoverable=False,
            trace_id=trace_id,
            raw_id=raw_id,
            observation_id=observation_id,
            field_path=field_path,
            details=details,
        )


class SchemaViolationError(CEBAInputError):
    def __init__(
        self,
        message: str,
        *,
        trace_id: UUID | None = None,
        raw_id: UUID | None = None,
        observation_id: UUID | None = None,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code="input.schema_violation",
            category=InputErrorCategory.VALIDATION,
            severity=InputErrorSeverity.ERROR,
            recoverable=False,
            trace_id=trace_id,
            raw_id=raw_id,
            observation_id=observation_id,
            field_path=field_path,
            details=details,
        )


class SecurityPolicyViolation(CEBAInputError):
    def __init__(
        self,
        message: str,
        *,
        trace_id: UUID | None = None,
        raw_id: UUID | None = None,
        observation_id: UUID | None = None,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code="input.security_policy_violation",
            category=InputErrorCategory.SECURITY,
            severity=InputErrorSeverity.CRITICAL,
            recoverable=False,
            trace_id=trace_id,
            raw_id=raw_id,
            observation_id=observation_id,
            field_path=field_path,
            details=details,
        )


class PayloadTooLargeError(CEBAInputError):
    def __init__(
        self,
        message: str,
        *,
        trace_id: UUID | None = None,
        raw_id: UUID | None = None,
        observation_id: UUID | None = None,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code="input.payload_too_large",
            category=InputErrorCategory.PAYLOAD,
            severity=InputErrorSeverity.ERROR,
            recoverable=True,
            trace_id=trace_id,
            raw_id=raw_id,
            observation_id=observation_id,
            field_path=field_path,
            details=details,
        )


class UnsupportedMimeTypeError(CEBAInputError):
    def __init__(
        self,
        message: str,
        *,
        trace_id: UUID | None = None,
        raw_id: UUID | None = None,
        observation_id: UUID | None = None,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code="input.unsupported_mime_type",
            category=InputErrorCategory.PAYLOAD,
            severity=InputErrorSeverity.ERROR,
            recoverable=False,
            trace_id=trace_id,
            raw_id=raw_id,
            observation_id=observation_id,
            field_path=field_path,
            details=details,
        )


class EncodingError(CEBAInputError):
    def __init__(
        self,
        message: str,
        *,
        trace_id: UUID | None = None,
        raw_id: UUID | None = None,
        observation_id: UUID | None = None,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code="input.encoding_error",
            category=InputErrorCategory.ENCODING,
            severity=InputErrorSeverity.ERROR,
            recoverable=True,
            trace_id=trace_id,
            raw_id=raw_id,
            observation_id=observation_id,
            field_path=field_path,
            details=details,
        )


class NormalizationError(CEBAInputError):
    def __init__(
        self,
        message: str,
        *,
        trace_id: UUID | None = None,
        raw_id: UUID | None = None,
        observation_id: UUID | None = None,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code="input.normalization_failed",
            category=InputErrorCategory.NORMALIZATION,
            severity=InputErrorSeverity.ERROR,
            recoverable=True,
            trace_id=trace_id,
            raw_id=raw_id,
            observation_id=observation_id,
            field_path=field_path,
            details=details,
        )


class CanonicalizationError(CEBAInputError):
    def __init__(
        self,
        message: str,
        *,
        trace_id: UUID | None = None,
        raw_id: UUID | None = None,
        observation_id: UUID | None = None,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code="input.canonicalization_failed",
            category=InputErrorCategory.CANONICALIZATION,
            severity=InputErrorSeverity.ERROR,
            recoverable=False,
            trace_id=trace_id,
            raw_id=raw_id,
            observation_id=observation_id,
            field_path=field_path,
            details=details,
        )


class ProvenanceMissingError(CEBAInputError):
    def __init__(
        self,
        message: str,
        *,
        trace_id: UUID | None = None,
        raw_id: UUID | None = None,
        observation_id: UUID | None = None,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code="input.provenance_missing",
            category=InputErrorCategory.PROVENANCE,
            severity=InputErrorSeverity.ERROR,
            recoverable=False,
            trace_id=trace_id,
            raw_id=raw_id,
            observation_id=observation_id,
            field_path=field_path,
            details=details,
        )


class ProvenanceInvalidError(CEBAInputError):
    def __init__(
        self,
        message: str,
        *,
        trace_id: UUID | None = None,
        raw_id: UUID | None = None,
        observation_id: UUID | None = None,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code="input.provenance_invalid",
            category=InputErrorCategory.PROVENANCE,
            severity=InputErrorSeverity.ERROR,
            recoverable=False,
            trace_id=trace_id,
            raw_id=raw_id,
            observation_id=observation_id,
            field_path=field_path,
            details=details,
        )


class HashMismatchError(CEBAInputError):
    def __init__(
        self,
        message: str,
        *,
        trace_id: UUID | None = None,
        raw_id: UUID | None = None,
        observation_id: UUID | None = None,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code="input.hash_mismatch",
            category=InputErrorCategory.FINGERPRINT,
            severity=InputErrorSeverity.ERROR,
            recoverable=False,
            trace_id=trace_id,
            raw_id=raw_id,
            observation_id=observation_id,
            field_path=field_path,
            details=details,
        )


class DispatchError(CEBAInputError):
    def __init__(
        self,
        message: str,
        *,
        trace_id: UUID | None = None,
        raw_id: UUID | None = None,
        observation_id: UUID | None = None,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code="input.dispatch_failed",
            category=InputErrorCategory.DISPATCH,
            severity=InputErrorSeverity.ERROR,
            recoverable=True,
            trace_id=trace_id,
            raw_id=raw_id,
            observation_id=observation_id,
            field_path=field_path,
            details=details,
        )


class AuditWriteError(CEBAInputError):
    def __init__(
        self,
        message: str,
        *,
        trace_id: UUID | None = None,
        raw_id: UUID | None = None,
        observation_id: UUID | None = None,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code="input.audit_write_failed",
            category=InputErrorCategory.AUDIT,
            severity=InputErrorSeverity.WARNING,
            recoverable=True,
            trace_id=trace_id,
            raw_id=raw_id,
            observation_id=observation_id,
            field_path=field_path,
            details=details,
        )


def ensure_input_error(error: Exception) -> CEBAInputError:
    if isinstance(error, CEBAInputError):
        return error
    return CEBAInputError(
        str(error),
        code="input.unknown",
        category=InputErrorCategory.UNKNOWN,
        severity=InputErrorSeverity.ERROR,
        recoverable=False,
        details={"original_type": error.__class__.__name__},
    )


__all__ = [
    "InputErrorSeverity",
    "InputErrorCategory",
    "InputErrorPayload",
    "CEBAInputError",
    "UnsupportedInputError",
    "AdapterResolutionError",
    "AdapterCaptureError",
    "ValidationFailedError",
    "SchemaViolationError",
    "SecurityPolicyViolation",
    "PayloadTooLargeError",
    "UnsupportedMimeTypeError",
    "EncodingError",
    "NormalizationError",
    "CanonicalizationError",
    "ProvenanceMissingError",
    "ProvenanceInvalidError",
    "HashMismatchError",
    "DispatchError",
    "AuditWriteError",
    "ensure_input_error",
]
