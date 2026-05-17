"""Strict schema objects for CEBA Core Metric Processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math
from uuid import UUID, uuid4

from ceba.core.constraints.schema import (
    ConstraintEvaluation,
    ConstraintEvaluationStatus,
    ConstraintViolationSeverity,
    ensure_constraint_probability,
    ensure_metadata_dict,
    ensure_non_empty_string,
    utc_now,
)


class MetricSpaceKind(Enum):
    EUCLIDEAN = "EUCLIDEAN"
    RIEMANNIAN = "RIEMANNIAN"
    FINSLER = "FINSLER"
    RANDERS = "RANDERS"
    MAHALANOBIS = "MAHALANOBIS"
    UNKNOWN = "UNKNOWN"


class MetricTensorStatus(Enum):
    IDENTITY = "IDENTITY"
    ACTIVE = "ACTIVE"
    UPDATED = "UPDATED"
    DEGRADED = "DEGRADED"
    INVALID = "INVALID"


class MetricUpdateReason(Enum):
    CONSTRAINT_PASS = "CONSTRAINT_PASS"
    CONSTRAINT_WARN = "CONSTRAINT_WARN"
    CONSTRAINT_FAIL = "CONSTRAINT_FAIL"
    VIOLATION_PRESSURE = "VIOLATION_PRESSURE"
    TELOS_ALIGNMENT = "TELOS_ALIGNMENT"
    STABILITY_CORRECTION = "STABILITY_CORRECTION"
    UNKNOWN = "UNKNOWN"


class DistanceKind(Enum):
    EUCLIDEAN = "EUCLIDEAN"
    MAHALANOBIS = "MAHALANOBIS"
    GEODESIC = "GEODESIC"
    FINSLER_LENGTH = "FINSLER_LENGTH"
    COSINE_DISTANCE = "COSINE_DISTANCE"
    UNKNOWN = "UNKNOWN"


class CostKind(Enum):
    ENERGY = "ENERGY"
    ENTROPY_PROXY = "ENTROPY_PROXY"
    UTILITY_GAP = "UTILITY_GAP"
    CONSTRAINT_PENALTY = "CONSTRAINT_PENALTY"
    TELOS_GAP = "TELOS_GAP"
    STABILITY_COST = "STABILITY_COST"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class MetricVectorRef:
    vector_id: str
    space: MetricSpaceKind
    dimension: int
    normalized: bool
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "vector_id", ensure_non_empty_string(self.vector_id, "vector_id"))
        if not isinstance(self.space, MetricSpaceKind):
            raise TypeError("space must be MetricSpaceKind")
        object.__setattr__(self, "dimension", ensure_positive_int(self.dimension, "dimension"))
        if not isinstance(self.normalized, bool):
            raise TypeError("normalized must be bool")
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class MetricTensorRef:
    tensor_id: str
    space: MetricSpaceKind
    dimension: int
    status: MetricTensorStatus
    is_spd: bool
    condition_number: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tensor_id", ensure_non_empty_string(self.tensor_id, "tensor_id"))
        if not isinstance(self.space, MetricSpaceKind):
            raise TypeError("space must be MetricSpaceKind")
        object.__setattr__(self, "dimension", ensure_positive_int(self.dimension, "dimension"))
        if not isinstance(self.status, MetricTensorStatus):
            raise TypeError("status must be MetricTensorStatus")
        if not isinstance(self.is_spd, bool):
            raise TypeError("is_spd must be bool")
        if self.condition_number is not None:
            object.__setattr__(
                self,
                "condition_number",
                ensure_non_negative_number(self.condition_number, "condition_number"),
            )
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class MetricDistanceResult:
    distance_id: UUID
    kind: DistanceKind
    left: MetricVectorRef
    right: MetricVectorRef
    tensor: MetricTensorRef | None
    value: float
    confidence: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_uuid(self.distance_id, "distance_id")
        if not isinstance(self.kind, DistanceKind):
            raise TypeError("kind must be DistanceKind")
        if not isinstance(self.left, MetricVectorRef):
            raise TypeError("left must be MetricVectorRef")
        if not isinstance(self.right, MetricVectorRef):
            raise TypeError("right must be MetricVectorRef")
        if self.tensor is not None and not isinstance(self.tensor, MetricTensorRef):
            raise TypeError("tensor must be None or MetricTensorRef")
        object.__setattr__(self, "value", ensure_non_negative_number(self.value, "value"))
        object.__setattr__(self, "confidence", ensure_constraint_probability(self.confidence, "confidence"))
        object.__setattr__(self, "created_at", ensure_timezone_aware(self.created_at, "created_at"))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class MetricCostResult:
    cost_id: UUID
    kind: CostKind
    value: float
    normalized_value: float | None
    confidence: float
    source_distance_id: UUID | None
    source_evaluation_id: UUID | None
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_uuid(self.cost_id, "cost_id")
        if not isinstance(self.kind, CostKind):
            raise TypeError("kind must be CostKind")
        object.__setattr__(self, "value", ensure_finite_number(self.value, "value"))
        if self.normalized_value is not None:
            object.__setattr__(
                self,
                "normalized_value",
                ensure_constraint_probability(self.normalized_value, "normalized_value"),
            )
        object.__setattr__(self, "confidence", ensure_constraint_probability(self.confidence, "confidence"))
        _ensure_optional_uuid(self.source_distance_id, "source_distance_id")
        _ensure_optional_uuid(self.source_evaluation_id, "source_evaluation_id")
        object.__setattr__(self, "created_at", ensure_timezone_aware(self.created_at, "created_at"))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class MetricUpdateSignal:
    signal_id: UUID
    reason: MetricUpdateReason
    source_evaluation_id: UUID | None
    source_cost_id: UUID | None
    severity: ConstraintViolationSeverity | None
    weight: float
    pressure: float
    confidence: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_uuid(self.signal_id, "signal_id")
        if not isinstance(self.reason, MetricUpdateReason):
            raise TypeError("reason must be MetricUpdateReason")
        _ensure_optional_uuid(self.source_evaluation_id, "source_evaluation_id")
        _ensure_optional_uuid(self.source_cost_id, "source_cost_id")
        if self.severity is not None and not isinstance(self.severity, ConstraintViolationSeverity):
            raise TypeError("severity must be None or ConstraintViolationSeverity")
        object.__setattr__(self, "weight", ensure_non_negative_number(self.weight, "weight"))
        object.__setattr__(self, "pressure", ensure_non_negative_number(self.pressure, "pressure"))
        object.__setattr__(self, "confidence", ensure_constraint_probability(self.confidence, "confidence"))
        object.__setattr__(self, "created_at", ensure_timezone_aware(self.created_at, "created_at"))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class MetricState:
    state_id: UUID
    space: MetricSpaceKind
    dimension: int
    tensor: MetricTensorRef
    lambda_value: float
    update_count: int
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_uuid(self.state_id, "state_id")
        if not isinstance(self.space, MetricSpaceKind):
            raise TypeError("space must be MetricSpaceKind")
        object.__setattr__(self, "dimension", ensure_positive_int(self.dimension, "dimension"))
        if not isinstance(self.tensor, MetricTensorRef):
            raise TypeError("tensor must be MetricTensorRef")
        if self.tensor.dimension != self.dimension:
            raise ValueError("tensor.dimension must equal dimension")
        object.__setattr__(self, "lambda_value", ensure_non_negative_number(self.lambda_value, "lambda_value"))
        if isinstance(self.update_count, bool) or not isinstance(self.update_count, int):
            raise TypeError("update_count must be int")
        if self.update_count < 0:
            raise ValueError("update_count must be non-negative")
        object.__setattr__(self, "created_at", ensure_timezone_aware(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", ensure_timezone_aware(self.updated_at, "updated_at"))
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be greater than or equal to created_at")
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))


def new_metric_distance_id() -> UUID:
    return uuid4()


def new_metric_cost_id() -> UUID:
    return uuid4()


def new_metric_update_signal_id() -> UUID:
    return uuid4()


def new_metric_state_id() -> UUID:
    return uuid4()


def ensure_positive_int(value: int, field_name: str) -> int:
    ensure_non_empty_string(field_name, "field_name")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def ensure_non_negative_number(value: float, field_name: str) -> float:
    number = ensure_finite_number(value, field_name)
    if number < 0.0:
        raise ValueError(f"{field_name} must be non-negative")
    return number


def ensure_finite_number(value: float, field_name: str) -> float:
    ensure_non_empty_string(field_name, "field_name")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be int or float")
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def ensure_timezone_aware(value: datetime, field_name: str) -> datetime:
    ensure_non_empty_string(field_name, "field_name")
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def metric_signal_from_constraint_evaluation(
    evaluation: ConstraintEvaluation,
) -> MetricUpdateSignal:
    if not isinstance(evaluation, ConstraintEvaluation):
        raise TypeError("evaluation must be ConstraintEvaluation")

    reason_by_status: dict[ConstraintEvaluationStatus, MetricUpdateReason] = {
        ConstraintEvaluationStatus.PASS: MetricUpdateReason.CONSTRAINT_PASS,
        ConstraintEvaluationStatus.WARN: MetricUpdateReason.CONSTRAINT_WARN,
        ConstraintEvaluationStatus.FAIL: MetricUpdateReason.CONSTRAINT_FAIL,
        ConstraintEvaluationStatus.SKIP: MetricUpdateReason.UNKNOWN,
    }
    reason = reason_by_status[evaluation.status]
    severity = evaluation.violation_severity
    weight = _weight_for_severity(severity)

    if evaluation.status is ConstraintEvaluationStatus.FAIL:
        pressure = weight
    elif evaluation.status is ConstraintEvaluationStatus.WARN:
        pressure = weight * 0.5
    else:
        pressure = 0.0

    return MetricUpdateSignal(
        signal_id=new_metric_update_signal_id(),
        reason=reason,
        source_evaluation_id=evaluation.evaluation_id,
        source_cost_id=None,
        severity=severity,
        weight=weight,
        pressure=pressure,
        confidence=evaluation.confidence,
        created_at=utc_now(),
        metadata={
            "source": "constraint_evaluation",
            "evaluation_status": evaluation.status.value,
            "message_code": evaluation.message_code,
        },
    )


def _ensure_uuid(value: UUID, field_name: str) -> UUID:
    if not isinstance(value, UUID):
        raise TypeError(f"{field_name} must be UUID")
    return value


def _ensure_optional_uuid(value: UUID | None, field_name: str) -> UUID | None:
    if value is not None:
        _ensure_uuid(value, field_name)
    return value


def _copy_metadata(value: dict[str, object]) -> dict[str, object]:
    return dict(ensure_metadata_dict(value, "metadata"))


def _weight_for_severity(severity: ConstraintViolationSeverity | None) -> float:
    if severity is ConstraintViolationSeverity.CRITICAL:
        return 1.0
    if severity is ConstraintViolationSeverity.HIGH:
        return 0.8
    if severity is ConstraintViolationSeverity.MEDIUM:
        return 0.5
    if severity is ConstraintViolationSeverity.LOW:
        return 0.25
    if severity is ConstraintViolationSeverity.TRACE:
        return 0.1
    if severity is None:
        return 0.0
    raise TypeError("severity must be None or ConstraintViolationSeverity")


__all__ = [
    "MetricSpaceKind",
    "MetricTensorStatus",
    "MetricUpdateReason",
    "DistanceKind",
    "CostKind",
    "MetricVectorRef",
    "MetricTensorRef",
    "MetricDistanceResult",
    "MetricCostResult",
    "MetricUpdateSignal",
    "MetricState",
    "new_metric_distance_id",
    "new_metric_cost_id",
    "new_metric_update_signal_id",
    "new_metric_state_id",
    "ensure_positive_int",
    "ensure_non_negative_number",
    "ensure_finite_number",
    "ensure_timezone_aware",
    "metric_signal_from_constraint_evaluation",
]
