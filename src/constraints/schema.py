"""Strict schema objects for CEBA Constraint Processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from ois.difference.schema import (
    DifferenceKind,
    ensure_non_negative,
    ensure_probability,
    ensure_timezone_aware,
    utc_now,
)


class ConstraintKind(str, Enum):
    """Stable category for a CEBA constraint."""

    STRUCTURAL = "STRUCTURAL"
    TEMPORAL = "TEMPORAL"
    NUMERIC = "NUMERIC"
    RELATIONAL = "RELATIONAL"
    SOURCE = "SOURCE"
    CONSISTENCY = "CONSISTENCY"
    SAFETY = "SAFETY"
    POLICY = "POLICY"
    TELOS = "TELOS"
    UNKNOWN = "UNKNOWN"


class ConstraintScope(str, Enum):
    """Artifact scope where a constraint applies."""

    CANDIDATE = "CANDIDATE"
    DIFFERENCE_SET = "DIFFERENCE_SET"
    CLASSIFICATION = "CLASSIFICATION"
    AGGREGATE = "AGGREGATE"
    DISPATCH = "DISPATCH"
    GLOBAL = "GLOBAL"


class ConstraintStatus(str, Enum):
    """Lifecycle status for a constraint definition."""

    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    DEPRECATED = "DEPRECATED"
    EXPERIMENTAL = "EXPERIMENTAL"


class ConstraintEvaluationStatus(str, Enum):
    """Result status for one constraint evaluation."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


class ConstraintViolationSeverity(str, Enum):
    """Severity of a constraint violation."""

    TRACE = "TRACE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ConstraintOperator(str, Enum):
    """Stable operators for constraint threshold declarations."""

    EXISTS = "EXISTS"
    NOT_EXISTS = "NOT_EXISTS"
    EQUALS = "EQUALS"
    NOT_EQUALS = "NOT_EQUALS"
    GREATER_THAN = "GREATER_THAN"
    GREATER_THAN_OR_EQUAL = "GREATER_THAN_OR_EQUAL"
    LESS_THAN = "LESS_THAN"
    LESS_THAN_OR_EQUAL = "LESS_THAN_OR_EQUAL"
    BETWEEN = "BETWEEN"
    IN_SET = "IN_SET"
    NOT_IN_SET = "NOT_IN_SET"
    CUSTOM = "CUSTOM"


@dataclass(frozen=True, slots=True)
class ConstraintTarget:
    """Identifies what artifact field a constraint applies to."""

    target_id: str
    scope: ConstraintScope
    field_path: tuple[str, ...]
    difference_kind: DifferenceKind | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ensure_non_empty_string(self.target_id, "target_id")
        _ensure_enum(self.scope, ConstraintScope, "scope")
        if isinstance(self.field_path, list):
            object.__setattr__(self, "field_path", tuple(self.field_path))
        if not isinstance(self.field_path, tuple):
            raise TypeError("field_path must be a tuple of strings")
        for part in self.field_path:
            ensure_non_empty_string(part, "field_path")
        if self.difference_kind is not None:
            _ensure_enum(self.difference_kind, DifferenceKind, "difference_kind")
        ensure_metadata_dict(self.metadata)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class ConstraintThreshold:
    """Numeric or probabilistic threshold configuration."""

    operator: ConstraintOperator
    value: object = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    inclusive: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_enum(self.operator, ConstraintOperator, "operator")
        lower_bound = ensure_non_negative(self.lower_bound, "lower_bound")
        upper_bound = ensure_non_negative(self.upper_bound, "upper_bound")
        if lower_bound is not None:
            object.__setattr__(self, "lower_bound", lower_bound)
        if upper_bound is not None:
            object.__setattr__(self, "upper_bound", upper_bound)
        if lower_bound is not None and upper_bound is not None and upper_bound < lower_bound:
            raise ValueError("upper_bound must be greater than or equal to lower_bound")
        if not isinstance(self.inclusive, bool):
            raise TypeError("inclusive must be a bool")
        ensure_metadata_dict(self.metadata)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class ConstraintDefinition:
    """Stable definition of one CEBA constraint."""

    constraint_id: UUID
    name: str
    kind: ConstraintKind
    scope: ConstraintScope
    status: ConstraintStatus
    target: ConstraintTarget
    threshold: ConstraintThreshold | None = None
    priority: float = 0.0
    weight: float = 1.0
    description: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_uuid(self.constraint_id, "constraint_id")
        ensure_non_empty_string(self.name, "name")
        _ensure_enum(self.kind, ConstraintKind, "kind")
        _ensure_enum(self.scope, ConstraintScope, "scope")
        _ensure_enum(self.status, ConstraintStatus, "status")
        _ensure_instance(self.target, ConstraintTarget, "target")
        if self.threshold is not None:
            _ensure_instance(self.threshold, ConstraintThreshold, "threshold")
        object.__setattr__(self, "priority", ensure_non_negative(self.priority, "priority"))
        object.__setattr__(self, "weight", ensure_non_negative(self.weight, "weight"))
        ensure_optional_non_empty_string(self.description, "description")
        ensure_timezone_aware(self.created_at, "created_at")
        ensure_metadata_dict(self.metadata)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class ConstraintInputRef:
    """Lightweight reference to an evaluated artifact."""

    artifact_id: str
    artifact_type: str
    difference_id: str | None = None
    observation_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ensure_non_empty_string(self.artifact_id, "artifact_id")
        ensure_non_empty_string(self.artifact_type, "artifact_type")
        ensure_optional_non_empty_string(self.difference_id, "difference_id")
        ensure_optional_non_empty_string(self.observation_id, "observation_id")
        ensure_metadata_dict(self.metadata)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class ConstraintEvaluation:
    """Result of evaluating one constraint against one input artifact."""

    evaluation_id: UUID
    constraint_id: UUID
    input_ref: ConstraintInputRef
    status: ConstraintEvaluationStatus
    violation_severity: ConstraintViolationSeverity | None
    score: float
    confidence: float
    message_code: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_uuid(self.evaluation_id, "evaluation_id")
        _ensure_uuid(self.constraint_id, "constraint_id")
        _ensure_instance(self.input_ref, ConstraintInputRef, "input_ref")
        _ensure_enum(self.status, ConstraintEvaluationStatus, "status")
        if self.violation_severity is not None:
            _ensure_enum(
                self.violation_severity,
                ConstraintViolationSeverity,
                "violation_severity",
            )
        object.__setattr__(
            self,
            "score",
            ensure_constraint_probability(self.score, "score"),
        )
        object.__setattr__(
            self,
            "confidence",
            ensure_constraint_probability(self.confidence, "confidence"),
        )
        ensure_optional_non_empty_string(self.message_code, "message_code")
        ensure_timezone_aware(self.created_at, "created_at")
        ensure_metadata_dict(self.metadata)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class ConstraintEvaluationSet:
    """Batch of constraint evaluations for one observation or difference set."""

    set_id: UUID
    observation_id: str
    evaluations: tuple[ConstraintEvaluation, ...]
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_uuid(self.set_id, "set_id")
        ensure_non_empty_string(self.observation_id, "observation_id")
        if isinstance(self.evaluations, list):
            object.__setattr__(self, "evaluations", tuple(self.evaluations))
        if not isinstance(self.evaluations, tuple):
            raise TypeError("evaluations must be a tuple of ConstraintEvaluation")
        for evaluation in self.evaluations:
            _ensure_instance(evaluation, ConstraintEvaluation, "evaluations")
        ensure_timezone_aware(self.created_at, "created_at")
        ensure_metadata_dict(self.metadata)
        object.__setattr__(self, "metadata", dict(self.metadata))


def new_constraint_id() -> UUID:
    """Return a new UUID for a constraint definition."""

    return uuid4()


def new_constraint_evaluation_id() -> UUID:
    """Return a new UUID for a constraint evaluation."""

    return uuid4()


def new_constraint_evaluation_set_id() -> UUID:
    """Return a new UUID for a constraint evaluation set."""

    return uuid4()


def ensure_constraint_probability(value: float, field_name: str) -> float:
    """Validate that a constraint value is in the probability range."""

    return ensure_probability(value, field_name)


def ensure_non_empty_string(value: str, field_name: str) -> str:
    """Validate a non-empty string field."""

    _ensure_field_name(field_name)
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


def ensure_optional_non_empty_string(value: str | None, field_name: str) -> str | None:
    """Validate an optional non-empty string field."""

    _ensure_field_name(field_name)
    if value is None:
        return None
    return ensure_non_empty_string(value, field_name)


def ensure_metadata_dict(
    value: dict[str, object],
    field_name: str = "metadata",
) -> dict[str, object]:
    """Validate a metadata dictionary with string keys."""

    _ensure_field_name(field_name)
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a dict")
    for key in value:
        if not isinstance(key, str):
            raise TypeError(f"{field_name} keys must be strings")
    return value


def _ensure_field_name(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("field_name must be a non-empty string")
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
        raise TypeError(f"{field_name} must be {expected_type.__name__}")
    return value


__all__ = [
    "ConstraintKind",
    "ConstraintScope",
    "ConstraintStatus",
    "ConstraintEvaluationStatus",
    "ConstraintViolationSeverity",
    "ConstraintOperator",
    "ConstraintTarget",
    "ConstraintThreshold",
    "ConstraintDefinition",
    "ConstraintInputRef",
    "ConstraintEvaluation",
    "ConstraintEvaluationSet",
    "new_constraint_id",
    "new_constraint_evaluation_id",
    "new_constraint_evaluation_set_id",
    "ensure_constraint_probability",
    "ensure_non_empty_string",
    "ensure_optional_non_empty_string",
    "ensure_metadata_dict",
]
