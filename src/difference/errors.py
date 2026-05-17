"""Domain-specific exceptions for CEBA difference processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DifferenceErrorCode(str, Enum):
    """Stable error codes for difference processing failures."""

    DIFFERENCE_ERROR = "DIFFERENCE_ERROR"
    SCHEMA_ERROR = "SCHEMA_ERROR"
    EVIDENCE_ERROR = "EVIDENCE_ERROR"
    NORMALIZATION_ERROR = "NORMALIZATION_ERROR"
    CLASSIFICATION_ERROR = "CLASSIFICATION_ERROR"
    AGGREGATION_ERROR = "AGGREGATION_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    DISPATCH_ERROR = "DISPATCH_ERROR"
    AUDIT_ERROR = "AUDIT_ERROR"
    INVARIANT_ERROR = "INVARIANT_ERROR"
    UNSUPPORTED_DIFFERENCE_KIND = "UNSUPPORTED_DIFFERENCE_KIND"
    INVALID_CONFIDENCE = "INVALID_CONFIDENCE"
    INVALID_MAGNITUDE = "INVALID_MAGNITUDE"
    INVALID_ANCHOR = "INVALID_ANCHOR"
    INVALID_EVIDENCE_REFERENCE = "INVALID_EVIDENCE_REFERENCE"
    EMPTY_DIFFERENCE_SET = "EMPTY_DIFFERENCE_SET"


@dataclass(frozen=True, slots=True)
class DifferenceErrorContext:
    """Serializable-friendly context attached to difference exceptions."""

    operation: str
    difference_id: str | None = None
    observation_id: str | None = None
    field_path: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.operation, str) or not self.operation:
            raise ValueError("operation must be a non-empty string")

        if self.difference_id is not None and not isinstance(self.difference_id, str):
            raise TypeError("difference_id must be a string or None")

        if self.observation_id is not None and not isinstance(self.observation_id, str):
            raise TypeError("observation_id must be a string or None")

        if isinstance(self.field_path, list):
            object.__setattr__(self, "field_path", tuple(self.field_path))

        if not isinstance(self.field_path, tuple):
            raise TypeError("field_path must be a tuple of strings")

        if not all(isinstance(part, str) for part in self.field_path):
            raise TypeError("field_path must contain only strings")

        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")

        if not all(isinstance(key, str) for key in self.metadata):
            raise TypeError("metadata keys must be strings")

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic, serialization-friendly representation."""

        return {
            "operation": self.operation,
            "difference_id": self.difference_id,
            "observation_id": self.observation_id,
            "field_path": list(self.field_path),
            "metadata": dict(self.metadata),
        }


class DifferenceError(Exception):
    """Base exception for all CEBA difference processing failures."""

    default_code: DifferenceErrorCode = DifferenceErrorCode.DIFFERENCE_ERROR

    def __init__(
        self,
        message: str,
        *,
        code: DifferenceErrorCode | None = None,
        context: DifferenceErrorContext | None = None,
        cause: BaseException | None = None,
    ) -> None:
        if not isinstance(message, str) or not message:
            raise ValueError("message must be a non-empty string")

        resolved_code: DifferenceErrorCode = code if code is not None else self.default_code
        if not isinstance(resolved_code, DifferenceErrorCode):
            raise TypeError("code must be a DifferenceErrorCode")

        if context is not None and not isinstance(context, DifferenceErrorContext):
            raise TypeError("context must be a DifferenceErrorContext or None")

        if cause is not None and not isinstance(cause, BaseException):
            raise TypeError("cause must be a BaseException or None")

        self.message: str = message
        self.code: DifferenceErrorCode = resolved_code
        self.context: DifferenceErrorContext | None = context
        self.cause: BaseException | None = cause
        super().__init__(message)

        if cause is not None:
            self.__cause__ = cause

    def __str__(self) -> str:
        return f"{self.code.value}: {self.message}"

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic, serialization-friendly representation."""

        return {
            "error_type": type(self).__name__,
            "code": self.code.value,
            "message": self.message,
            "context": self.context.to_dict() if self.context is not None else None,
            "cause_type": type(self.cause).__name__ if self.cause is not None else None,
            "cause_message": str(self.cause) if self.cause is not None else None,
        }


class DifferenceSchemaError(DifferenceError):
    """Raised when difference schema construction or access fails."""

    default_code: DifferenceErrorCode = DifferenceErrorCode.SCHEMA_ERROR


class DifferenceEvidenceError(DifferenceError):
    """Raised when evidence construction fails."""

    default_code: DifferenceErrorCode = DifferenceErrorCode.EVIDENCE_ERROR


class DifferenceNormalizationError(DifferenceError):
    """Raised when difference normalization fails."""

    default_code: DifferenceErrorCode = DifferenceErrorCode.NORMALIZATION_ERROR


class DifferenceClassificationError(DifferenceError):
    """Raised when difference classification fails."""

    default_code: DifferenceErrorCode = DifferenceErrorCode.CLASSIFICATION_ERROR


class DifferenceAggregationError(DifferenceError):
    """Raised when difference aggregation fails."""

    default_code: DifferenceErrorCode = DifferenceErrorCode.AGGREGATION_ERROR


class DifferenceValidationError(DifferenceError):
    """Raised when difference validation fails."""

    default_code: DifferenceErrorCode = DifferenceErrorCode.VALIDATION_ERROR


class DifferenceDispatchError(DifferenceError):
    """Raised when difference dispatch fails."""

    default_code: DifferenceErrorCode = DifferenceErrorCode.DISPATCH_ERROR


class DifferenceAuditError(DifferenceError):
    """Raised when difference audit handling fails."""

    default_code: DifferenceErrorCode = DifferenceErrorCode.AUDIT_ERROR


class DifferenceInvariantError(DifferenceError):
    """Raised when an internal difference invariant is violated."""

    default_code: DifferenceErrorCode = DifferenceErrorCode.INVARIANT_ERROR


class UnsupportedDifferenceKindError(DifferenceError):
    """Raised when a difference kind is unsupported."""

    default_code: DifferenceErrorCode = DifferenceErrorCode.UNSUPPORTED_DIFFERENCE_KIND


class EmptyDifferenceSetError(DifferenceError):
    """Raised when a required difference set is empty."""

    default_code: DifferenceErrorCode = DifferenceErrorCode.EMPTY_DIFFERENCE_SET


def build_error_context(
    *,
    operation: str,
    difference_id: str | None = None,
    observation_id: str | None = None,
    field_path: tuple[str, ...] | list[str] = (),
    metadata: dict[str, object] | None = None,
) -> DifferenceErrorContext:
    """Build validated context for a difference error."""

    resolved_metadata: dict[str, object] = {} if metadata is None else metadata
    return DifferenceErrorContext(
        operation=operation,
        difference_id=difference_id,
        observation_id=observation_id,
        field_path=tuple(field_path),
        metadata=resolved_metadata,
    )


def wrap_difference_error(
    message: str,
    *,
    error_type: type[DifferenceError],
    context: DifferenceErrorContext | None = None,
    cause: BaseException | None = None,
) -> DifferenceError:
    """Construct a typed difference error while preserving its cause."""

    if not isinstance(error_type, type) or not issubclass(error_type, DifferenceError):
        raise TypeError("error_type must be a subclass of DifferenceError")

    if not isinstance(message, str) or not message:
        raise ValueError("message must be a non-empty string")

    return error_type(message=message, context=context, cause=cause)
