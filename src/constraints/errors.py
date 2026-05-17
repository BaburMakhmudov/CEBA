"""Domain-specific exceptions for CEBA constraint processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ConstraintErrorCode(str, Enum):
    """Stable error codes for constraint processing failures."""

    CONSTRAINT_ERROR = "CONSTRAINT_ERROR"
    SCHEMA_ERROR = "SCHEMA_ERROR"
    REGISTRY_ERROR = "REGISTRY_ERROR"
    MATCHING_ERROR = "MATCHING_ERROR"
    EVALUATION_ERROR = "EVALUATION_ERROR"
    AGGREGATION_ERROR = "AGGREGATION_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    DISPATCH_ERROR = "DISPATCH_ERROR"
    AUDIT_ERROR = "AUDIT_ERROR"
    INVARIANT_ERROR = "INVARIANT_ERROR"
    UNSUPPORTED_CONSTRAINT_KIND = "UNSUPPORTED_CONSTRAINT_KIND"
    UNSUPPORTED_CONSTRAINT_OPERATOR = "UNSUPPORTED_CONSTRAINT_OPERATOR"
    INVALID_CONSTRAINT_TARGET = "INVALID_CONSTRAINT_TARGET"
    INVALID_CONSTRAINT_THRESHOLD = "INVALID_CONSTRAINT_THRESHOLD"
    INVALID_CONSTRAINT_DEFINITION = "INVALID_CONSTRAINT_DEFINITION"
    INVALID_CONSTRAINT_EVALUATION = "INVALID_CONSTRAINT_EVALUATION"
    EMPTY_CONSTRAINT_SET = "EMPTY_CONSTRAINT_SET"
    DUPLICATE_CONSTRAINT_ID = "DUPLICATE_CONSTRAINT_ID"
    CONSTRAINT_NOT_FOUND = "CONSTRAINT_NOT_FOUND"


@dataclass(frozen=True, slots=True)
class ConstraintErrorContext:
    """Serializable-friendly context attached to constraint exceptions."""

    operation: str
    constraint_id: str | None = None
    evaluation_id: str | None = None
    difference_id: str | None = None
    observation_id: str | None = None
    field_path: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_non_empty_string(self.operation, "operation")
        _ensure_optional_non_empty_string(self.constraint_id, "constraint_id")
        _ensure_optional_non_empty_string(self.evaluation_id, "evaluation_id")
        _ensure_optional_non_empty_string(self.difference_id, "difference_id")
        _ensure_optional_non_empty_string(self.observation_id, "observation_id")

        if isinstance(self.field_path, list):
            object.__setattr__(self, "field_path", tuple(self.field_path))
        if not isinstance(self.field_path, tuple):
            raise TypeError("field_path must be a tuple of strings")
        for part in self.field_path:
            _ensure_non_empty_string(part, "field_path")

        _ensure_metadata_dict(self.metadata)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic, serialization-friendly representation."""

        return {
            "operation": self.operation,
            "constraint_id": self.constraint_id,
            "evaluation_id": self.evaluation_id,
            "difference_id": self.difference_id,
            "observation_id": self.observation_id,
            "field_path": list(self.field_path),
            "metadata": dict(self.metadata),
        }


class ConstraintError(Exception):
    """Base exception for all CEBA constraint processing failures."""

    default_code: ConstraintErrorCode = ConstraintErrorCode.CONSTRAINT_ERROR

    def __init__(
        self,
        message: str,
        *,
        code: ConstraintErrorCode | None = None,
        context: ConstraintErrorContext | None = None,
        cause: BaseException | None = None,
    ) -> None:
        _ensure_non_empty_string(message, "message")

        resolved_code = code if code is not None else self.default_code
        if not isinstance(resolved_code, ConstraintErrorCode):
            raise TypeError("code must be ConstraintErrorCode")
        if context is not None and not isinstance(context, ConstraintErrorContext):
            raise TypeError("context must be ConstraintErrorContext or None")
        if cause is not None and not isinstance(cause, BaseException):
            raise TypeError("cause must be BaseException or None")

        self.message: str = message
        self.code: ConstraintErrorCode = resolved_code
        self.context: ConstraintErrorContext | None = context
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


class ConstraintSchemaError(ConstraintError):
    """Raised when constraint schema construction or access fails."""

    default_code: ConstraintErrorCode = ConstraintErrorCode.SCHEMA_ERROR


class ConstraintRegistryError(ConstraintError):
    """Raised when constraint registry handling fails."""

    default_code: ConstraintErrorCode = ConstraintErrorCode.REGISTRY_ERROR


class ConstraintMatchingError(ConstraintError):
    """Raised when constraint matching fails."""

    default_code: ConstraintErrorCode = ConstraintErrorCode.MATCHING_ERROR


class ConstraintEvaluationError(ConstraintError):
    """Raised when constraint evaluation fails."""

    default_code: ConstraintErrorCode = ConstraintErrorCode.EVALUATION_ERROR


class ConstraintAggregationError(ConstraintError):
    """Raised when constraint aggregation fails."""

    default_code: ConstraintErrorCode = ConstraintErrorCode.AGGREGATION_ERROR


class ConstraintValidationError(ConstraintError):
    """Raised when constraint validation fails."""

    default_code: ConstraintErrorCode = ConstraintErrorCode.VALIDATION_ERROR


class ConstraintDispatchError(ConstraintError):
    """Raised when constraint dispatch fails."""

    default_code: ConstraintErrorCode = ConstraintErrorCode.DISPATCH_ERROR


class ConstraintAuditError(ConstraintError):
    """Raised when constraint audit handling fails."""

    default_code: ConstraintErrorCode = ConstraintErrorCode.AUDIT_ERROR


class ConstraintInvariantError(ConstraintError):
    """Raised when an internal constraint invariant is violated."""

    default_code: ConstraintErrorCode = ConstraintErrorCode.INVARIANT_ERROR


class UnsupportedConstraintKindError(ConstraintError):
    """Raised when a constraint kind is unsupported."""

    default_code: ConstraintErrorCode = ConstraintErrorCode.UNSUPPORTED_CONSTRAINT_KIND


class UnsupportedConstraintOperatorError(ConstraintError):
    """Raised when a constraint operator is unsupported."""

    default_code: ConstraintErrorCode = ConstraintErrorCode.UNSUPPORTED_CONSTRAINT_OPERATOR


class InvalidConstraintTargetError(ConstraintError):
    """Raised when a constraint target is invalid."""

    default_code: ConstraintErrorCode = ConstraintErrorCode.INVALID_CONSTRAINT_TARGET


class InvalidConstraintThresholdError(ConstraintError):
    """Raised when a constraint threshold is invalid."""

    default_code: ConstraintErrorCode = ConstraintErrorCode.INVALID_CONSTRAINT_THRESHOLD


class InvalidConstraintDefinitionError(ConstraintError):
    """Raised when a constraint definition is invalid."""

    default_code: ConstraintErrorCode = ConstraintErrorCode.INVALID_CONSTRAINT_DEFINITION


class InvalidConstraintEvaluationError(ConstraintError):
    """Raised when a constraint evaluation is invalid."""

    default_code: ConstraintErrorCode = ConstraintErrorCode.INVALID_CONSTRAINT_EVALUATION


class EmptyConstraintSetError(ConstraintError):
    """Raised when a required constraint set is empty."""

    default_code: ConstraintErrorCode = ConstraintErrorCode.EMPTY_CONSTRAINT_SET


class DuplicateConstraintIdError(ConstraintError):
    """Raised when duplicate constraint identifiers are present."""

    default_code: ConstraintErrorCode = ConstraintErrorCode.DUPLICATE_CONSTRAINT_ID


class ConstraintNotFoundError(ConstraintError):
    """Raised when a requested constraint cannot be found."""

    default_code: ConstraintErrorCode = ConstraintErrorCode.CONSTRAINT_NOT_FOUND


def build_constraint_error_context(
    *,
    operation: str,
    constraint_id: str | None = None,
    evaluation_id: str | None = None,
    difference_id: str | None = None,
    observation_id: str | None = None,
    field_path: tuple[str, ...] | list[str] = (),
    metadata: dict[str, object] | None = None,
) -> ConstraintErrorContext:
    """Build validated context for a constraint error."""

    return ConstraintErrorContext(
        operation=operation,
        constraint_id=constraint_id,
        evaluation_id=evaluation_id,
        difference_id=difference_id,
        observation_id=observation_id,
        field_path=tuple(field_path),
        metadata={} if metadata is None else metadata,
    )


def wrap_constraint_error(
    message: str,
    *,
    error_type: type[ConstraintError],
    context: ConstraintErrorContext | None = None,
    cause: BaseException | None = None,
) -> ConstraintError:
    """Construct a typed constraint error while preserving its cause."""

    if not isinstance(error_type, type) or not issubclass(error_type, ConstraintError):
        raise TypeError("error_type must be a subclass of ConstraintError")
    _ensure_non_empty_string(message, "message")
    return error_type(message=message, context=context, cause=cause)


def _ensure_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(field_name, str) or not field_name:
        raise ValueError("field_name must be a non-empty string")
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _ensure_optional_non_empty_string(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return _ensure_non_empty_string(value, field_name)


def _ensure_metadata_dict(value: dict[str, object]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("metadata must be a dict")
    if not all(isinstance(key, str) for key in value):
        raise TypeError("metadata keys must be strings")
    return value


__all__ = [
    "ConstraintErrorCode",
    "ConstraintErrorContext",
    "ConstraintError",
    "ConstraintSchemaError",
    "ConstraintRegistryError",
    "ConstraintMatchingError",
    "ConstraintEvaluationError",
    "ConstraintAggregationError",
    "ConstraintValidationError",
    "ConstraintDispatchError",
    "ConstraintAuditError",
    "ConstraintInvariantError",
    "UnsupportedConstraintKindError",
    "UnsupportedConstraintOperatorError",
    "InvalidConstraintTargetError",
    "InvalidConstraintThresholdError",
    "InvalidConstraintDefinitionError",
    "InvalidConstraintEvaluationError",
    "EmptyConstraintSetError",
    "DuplicateConstraintIdError",
    "ConstraintNotFoundError",
    "build_constraint_error_context",
    "wrap_constraint_error",
]
