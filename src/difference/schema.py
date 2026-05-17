"""Strict schema objects for CEBA Difference Processing.

This module defines immutable, serializable-friendly structures for representing
differences after parser output has been converted into comparable units. It
does not classify, aggregate, normalize text, call embeddings, infer meaning, or
perform business logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4


class DifferenceKind(str, Enum):
    """High-level technical category for a detected difference."""

    STRUCTURAL = "STRUCTURAL"
    TEMPORAL = "TEMPORAL"
    LEXICAL = "LEXICAL"
    NUMERIC = "NUMERIC"
    RELATIONAL = "RELATIONAL"
    SOURCE = "SOURCE"
    MISSING = "MISSING"
    EXTRA = "EXTRA"
    CONTRADICTION = "CONTRADICTION"
    UNKNOWN = "UNKNOWN"


class DifferenceSeverity(str, Enum):
    """Severity assigned to a difference by a downstream classifier."""

    TRACE = "TRACE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DifferenceStatus(str, Enum):
    """Lifecycle state for a difference artifact."""

    CANDIDATE = "CANDIDATE"
    NORMALIZED = "NORMALIZED"
    CLASSIFIED = "CLASSIFIED"
    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"


class DifferenceDirection(str, Enum):
    """Comparison direction for left/right difference values."""

    EXPECTED_TO_OBSERVED = "EXPECTED_TO_OBSERVED"
    OBSERVED_TO_EXPECTED = "OBSERVED_TO_EXPECTED"
    BIDIRECTIONAL = "BIDIRECTIONAL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class DifferenceAnchor:
    """Local position or unit where a difference appears."""

    anchor_id: str
    source_id: str | None
    field_path: tuple[str, ...]
    span_start: int | None
    span_end: int | None
    label: str | None

    def __post_init__(self) -> None:
        _ensure_non_empty_string(self.anchor_id, "anchor_id")
        _ensure_optional_string(self.source_id, "source_id")
        _ensure_optional_string(self.label, "label")

        if not isinstance(self.field_path, tuple):
            if isinstance(self.field_path, list):
                object.__setattr__(self, "field_path", tuple(self.field_path))
            else:
                raise TypeError("field_path must be a tuple of strings")
        for part in self.field_path:
            _ensure_non_empty_string(part, "field_path")

        if self.span_start is not None:
            _ensure_non_negative_int(self.span_start, "span_start")
        if self.span_end is not None:
            _ensure_non_negative_int(self.span_end, "span_end")
        if (
            self.span_start is not None
            and self.span_end is not None
            and self.span_end < self.span_start
        ):
            raise ValueError("span_end must be greater than or equal to span_start")


@dataclass(frozen=True, slots=True)
class DifferenceValue:
    """Compared value container without interpretation."""

    raw_value: object
    normalized_value: object | None
    value_type: str | None

    def __post_init__(self) -> None:
        _ensure_optional_string(self.value_type, "value_type")


@dataclass(frozen=True, slots=True)
class DifferenceEvidenceRef:
    """Reference to evidence stored or built outside this schema object."""

    evidence_id: str
    source_id: str | None
    confidence: float
    note: str | None

    def __post_init__(self) -> None:
        _ensure_non_empty_string(self.evidence_id, "evidence_id")
        _ensure_optional_string(self.source_id, "source_id")
        ensure_probability(self.confidence, "confidence")
        _ensure_optional_string(self.note, "note")


@dataclass(frozen=True, slots=True)
class DifferenceCandidate:
    """Raw detected difference before full validation."""

    difference_id: UUID
    kind: DifferenceKind
    status: DifferenceStatus
    direction: DifferenceDirection
    left_anchor: DifferenceAnchor | None
    right_anchor: DifferenceAnchor | None
    left_value: DifferenceValue | None
    right_value: DifferenceValue | None
    evidence_refs: tuple[DifferenceEvidenceRef, ...]
    magnitude: float | None
    confidence: float
    created_at: datetime
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        _ensure_uuid(self.difference_id, "difference_id")
        _ensure_enum(self.kind, DifferenceKind, "kind")
        _ensure_enum(self.status, DifferenceStatus, "status")
        _ensure_enum(self.direction, DifferenceDirection, "direction")
        _ensure_optional_instance(self.left_anchor, DifferenceAnchor, "left_anchor")
        _ensure_optional_instance(self.right_anchor, DifferenceAnchor, "right_anchor")
        _ensure_optional_instance(self.left_value, DifferenceValue, "left_value")
        _ensure_optional_instance(self.right_value, DifferenceValue, "right_value")

        if not isinstance(self.evidence_refs, tuple):
            if isinstance(self.evidence_refs, list):
                object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
            else:
                raise TypeError("evidence_refs must be a tuple of DifferenceEvidenceRef")
        for evidence_ref in self.evidence_refs:
            _ensure_instance(
                evidence_ref,
                DifferenceEvidenceRef,
                "evidence_refs",
            )

        ensure_non_negative(self.magnitude, "magnitude")
        ensure_probability(self.confidence, "confidence")
        ensure_timezone_aware(self.created_at, "created_at")
        _ensure_metadata_dict(self.metadata)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class DifferenceClassification:
    """Classifier output associated with a candidate without mutating it."""

    difference_id: UUID
    kind: DifferenceKind
    severity: DifferenceSeverity
    confidence: float
    rationale_code: str | None
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        _ensure_uuid(self.difference_id, "difference_id")
        _ensure_enum(self.kind, DifferenceKind, "kind")
        _ensure_enum(self.severity, DifferenceSeverity, "severity")
        ensure_probability(self.confidence, "confidence")
        _ensure_optional_string(self.rationale_code, "rationale_code")
        _ensure_metadata_dict(self.metadata)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class DifferenceSet:
    """Batch of differences produced from one parsed observation comparison."""

    set_id: UUID
    observation_id: str
    candidates: tuple[DifferenceCandidate, ...]
    created_at: datetime
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        _ensure_uuid(self.set_id, "set_id")
        _ensure_non_empty_string(self.observation_id, "observation_id")

        if not isinstance(self.candidates, tuple):
            if isinstance(self.candidates, list):
                object.__setattr__(self, "candidates", tuple(self.candidates))
            else:
                raise TypeError("candidates must be a tuple of DifferenceCandidate")
        for candidate in self.candidates:
            _ensure_instance(candidate, DifferenceCandidate, "candidates")

        ensure_timezone_aware(self.created_at, "created_at")
        _ensure_metadata_dict(self.metadata)
        object.__setattr__(self, "metadata", dict(self.metadata))


def utc_now() -> datetime:
    """Return the current timezone-aware UTC datetime."""

    return datetime.now(timezone.utc)


def ensure_probability(value: float, field_name: str) -> float:
    """Validate that a value is in the inclusive probability range."""

    _ensure_field_name(field_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a float between 0.0 and 1.0")
    probability = float(value)
    if probability < 0.0 or probability > 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return probability


def ensure_non_negative(value: float | None, field_name: str) -> float | None:
    """Validate that a value is None or non-negative."""

    _ensure_field_name(field_name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a non-negative float or None")
    non_negative = float(value)
    if non_negative < 0.0:
        raise ValueError(f"{field_name} must be greater than or equal to 0.0")
    return non_negative


def ensure_timezone_aware(value: datetime, field_name: str) -> datetime:
    """Validate that a datetime has timezone information."""

    _ensure_field_name(field_name)
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def new_difference_id() -> UUID:
    """Return a new UUID for a difference artifact."""

    return uuid4()


def _ensure_field_name(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("field_name must be a non-empty string")
    return value


def _ensure_non_empty_string(value: str, field_name: str) -> str:
    _ensure_field_name(field_name)
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


def _ensure_optional_string(value: str | None, field_name: str) -> str | None:
    _ensure_field_name(field_name)
    if value is None:
        return None
    return _ensure_non_empty_string(value, field_name)


def _ensure_non_negative_int(value: int, field_name: str) -> int:
    _ensure_field_name(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{field_name} must be greater than or equal to 0")
    return value


def _ensure_uuid(value: UUID, field_name: str) -> UUID:
    _ensure_field_name(field_name)
    if not isinstance(value, UUID):
        raise TypeError(f"{field_name} must be a UUID")
    return value


def _ensure_enum(value: object, enum_type: type[Enum], field_name: str) -> object:
    _ensure_field_name(field_name)
    if not isinstance(value, enum_type):
        raise TypeError(f"{field_name} must be {enum_type.__name__}")
    return value


def _ensure_instance(value: object, expected_type: type[object], field_name: str) -> object:
    _ensure_field_name(field_name)
    if not isinstance(value, expected_type):
        raise TypeError(f"{field_name} must contain {expected_type.__name__}")
    return value


def _ensure_optional_instance(
    value: object | None,
    expected_type: type[object],
    field_name: str,
) -> object | None:
    _ensure_field_name(field_name)
    if value is None:
        return None
    return _ensure_instance(value, expected_type, field_name)


def _ensure_metadata_dict(value: dict[str, object]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("metadata must be a dict")
    for key in value:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
    return value


__all__ = [
    "DifferenceKind",
    "DifferenceSeverity",
    "DifferenceStatus",
    "DifferenceDirection",
    "DifferenceAnchor",
    "DifferenceValue",
    "DifferenceEvidenceRef",
    "DifferenceCandidate",
    "DifferenceClassification",
    "DifferenceSet",
    "utc_now",
    "ensure_probability",
    "ensure_non_negative",
    "ensure_timezone_aware",
    "new_difference_id",
]
