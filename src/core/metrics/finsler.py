"""Deterministic lightweight Finsler metric utilities for CEBA Core."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from ceba.core.metrics.mahalanobis import ensure_vector_values, quadratic_form, vector_difference
from ceba.core.metrics.schema import (
    DistanceKind,
    MetricDistanceResult,
    MetricTensorRef,
    MetricVectorRef,
    ensure_constraint_probability,
    ensure_finite_number,
    ensure_non_negative_number,
    new_metric_distance_id,
    utc_now,
)


@dataclass(frozen=True, slots=True)
class FinslerPolicy:
    base_weight: float = 1.0
    directional_weight: float = 0.0
    normalize_direction: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_weight", ensure_non_negative_number(self.base_weight, "base_weight"))
        object.__setattr__(
            self,
            "directional_weight",
            ensure_non_negative_number(self.directional_weight, "directional_weight"),
        )
        if not isinstance(self.normalize_direction, bool):
            raise TypeError("normalize_direction must be bool")


@dataclass(frozen=True, slots=True)
class FinslerInput:
    left: MetricVectorRef
    right: MetricVectorRef
    left_values: tuple[float, ...]
    right_values: tuple[float, ...]
    direction_values: tuple[float, ...] | None
    tensor: MetricTensorRef | None
    tensor_matrix: tuple[tuple[float, ...], ...] | None
    policy: FinslerPolicy
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.left, MetricVectorRef):
            raise TypeError("left must be MetricVectorRef")
        if not isinstance(self.right, MetricVectorRef):
            raise TypeError("right must be MetricVectorRef")
        if self.left.dimension != self.right.dimension:
            raise ValueError("left.dimension must equal right.dimension")

        dimension = self.left.dimension
        object.__setattr__(
            self,
            "left_values",
            ensure_vector_values(self.left_values, dimension=dimension, field_name="left_values"),
        )
        object.__setattr__(
            self,
            "right_values",
            ensure_vector_values(self.right_values, dimension=dimension, field_name="right_values"),
        )
        if self.direction_values is not None:
            object.__setattr__(
                self,
                "direction_values",
                ensure_vector_values(self.direction_values, dimension=dimension, field_name="direction_values"),
            )
        if self.tensor is not None and not isinstance(self.tensor, MetricTensorRef):
            raise TypeError("tensor must be None or MetricTensorRef")
        if self.tensor is not None and self.tensor.dimension != dimension:
            raise ValueError("tensor.dimension must equal vector dimension")
        if self.tensor_matrix is not None:
            if self.tensor is None:
                raise ValueError("tensor_matrix requires tensor")
            object.__setattr__(
                self,
                "tensor_matrix",
                _ensure_matrix_values(self.tensor_matrix, dimension=dimension, field_name="tensor_matrix"),
            )
        if not isinstance(self.policy, FinslerPolicy):
            raise TypeError("policy must be FinslerPolicy")
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    @property
    def dimension(self) -> int:
        return self.left.dimension


def euclidean_norm(values: tuple[float, ...]) -> float:
    checked_values = _ensure_finite_tuple(values, "values")
    return ensure_non_negative_number(
        math.sqrt(sum(value * value for value in checked_values)),
        "euclidean_norm",
    )


def normalize_vector(values: tuple[float, ...]) -> tuple[float, ...]:
    checked_values = _ensure_finite_tuple(values, "values")
    norm = euclidean_norm(checked_values)
    if norm == 0.0:
        return checked_values
    return tuple(
        ensure_finite_number(value / norm, f"normalized[{index}]")
        for index, value in enumerate(checked_values)
    )


def directional_dot(
    *,
    vector: tuple[float, ...],
    direction: tuple[float, ...],
) -> float:
    checked_vector = _ensure_finite_tuple(vector, "vector")
    checked_direction = _ensure_finite_tuple(direction, "direction")
    if len(checked_vector) != len(checked_direction):
        raise ValueError("vector and direction must have the same length")
    return ensure_finite_number(
        sum(
            vector_value * direction_value
            for vector_value, direction_value in zip(checked_vector, checked_direction, strict=True)
        ),
        "directional_dot",
    )


def finsler_length_value(
    *,
    displacement: tuple[float, ...],
    policy: FinslerPolicy,
    direction: tuple[float, ...] | None = None,
    tensor_matrix: tuple[tuple[float, ...], ...] | None = None,
) -> float:
    checked_displacement = _ensure_finite_tuple(displacement, "displacement")
    if not isinstance(policy, FinslerPolicy):
        raise TypeError("policy must be FinslerPolicy")

    if tensor_matrix is None:
        base_length = euclidean_norm(checked_displacement)
    else:
        matrix = _ensure_matrix_values(tensor_matrix, dimension=len(checked_displacement), field_name="tensor_matrix")
        base_length = ensure_non_negative_number(
            math.sqrt(max(ensure_finite_number(quadratic_form(checked_displacement, matrix), "quadratic_form"), 0.0)),
            "base_length",
        )

    directional_term = 0.0
    if direction is not None:
        checked_direction = _ensure_finite_tuple(direction, "direction")
        if len(checked_direction) != len(checked_displacement):
            raise ValueError("direction length must equal displacement length")
        direction_used = normalize_vector(checked_direction) if policy.normalize_direction else checked_direction
        directional_term = ensure_non_negative_number(
            policy.directional_weight * abs(directional_dot(vector=checked_displacement, direction=direction_used)),
            "directional_term",
        )

    return ensure_non_negative_number(
        policy.base_weight * base_length + directional_term,
        "finsler_length",
    )


def create_finsler_input(
    *,
    left: MetricVectorRef,
    right: MetricVectorRef,
    left_values: tuple[float, ...] | list[float],
    right_values: tuple[float, ...] | list[float],
    direction_values: tuple[float, ...] | list[float] | None = None,
    tensor: MetricTensorRef | None = None,
    tensor_matrix: tuple[tuple[float, ...], ...] | list[list[float]] | None = None,
    policy: FinslerPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> FinslerInput:
    return FinslerInput(
        left=left,
        right=right,
        left_values=tuple(left_values) if isinstance(left_values, list) else left_values,
        right_values=tuple(right_values) if isinstance(right_values, list) else right_values,
        direction_values=tuple(direction_values) if isinstance(direction_values, list) else direction_values,
        tensor=tensor,
        tensor_matrix=tensor_matrix,
        policy=FinslerPolicy() if policy is None else policy,
        metadata={} if metadata is None else metadata,
    )


def compute_finsler_distance_result(
    *,
    finsler_input: FinslerInput,
    confidence: float = 1.0,
) -> MetricDistanceResult:
    if not isinstance(finsler_input, FinslerInput):
        raise TypeError("finsler_input must be FinslerInput")
    checked_confidence = ensure_constraint_probability(confidence, "confidence")
    displacement = vector_difference(finsler_input.left_values, finsler_input.right_values)
    value = finsler_length_value(
        displacement=displacement,
        policy=finsler_input.policy,
        direction=finsler_input.direction_values,
        tensor_matrix=finsler_input.tensor_matrix,
    )

    metadata = dict(finsler_input.metadata)
    metadata.update(
        {
            "distance_kind": "finsler_length",
            "dimension": finsler_input.dimension,
            "base_weight": finsler_input.policy.base_weight,
            "directional_weight": finsler_input.policy.directional_weight,
            "has_direction": finsler_input.direction_values is not None,
            "has_tensor": finsler_input.tensor is not None,
        }
    )

    return MetricDistanceResult(
        distance_id=new_metric_distance_id(),
        kind=DistanceKind.FINSLER_LENGTH,
        left=finsler_input.left,
        right=finsler_input.right,
        tensor=finsler_input.tensor,
        value=value,
        confidence=checked_confidence,
        created_at=utc_now(),
        metadata=metadata,
    )


def _ensure_finite_tuple(values: tuple[float, ...], field_name: str) -> tuple[float, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{field_name} must be tuple[float, ...]")
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    return tuple(ensure_finite_number(value, f"{field_name}[{index}]") for index, value in enumerate(values))


def _ensure_matrix_values(
    matrix: tuple[tuple[float, ...], ...] | list[list[float]],
    *,
    dimension: int,
    field_name: str,
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(matrix, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple[tuple[float, ...], ...] or list[list[float]]")
    if len(matrix) != dimension:
        raise ValueError(f"{field_name} row count must equal dimension")

    rows: list[tuple[float, ...]] = []
    for row_index, row in enumerate(matrix):
        if not isinstance(row, (tuple, list)):
            raise TypeError(f"{field_name}[{row_index}] must be tuple[float, ...] or list[float]")
        if len(row) != dimension:
            raise ValueError(f"{field_name} must be square with dimension rows and columns")
        rows.append(
            tuple(
                ensure_finite_number(value, f"{field_name}[{row_index}][{column_index}]")
                for column_index, value in enumerate(row)
            )
        )
    return tuple(rows)


def _copy_metadata(value: dict[str, object]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("metadata must be dict[str, object]")
    if any(not isinstance(key, str) for key in value):
        raise TypeError("metadata keys must be strings")
    return dict(value)


__all__ = [
    "FinslerPolicy",
    "FinslerInput",
    "euclidean_norm",
    "normalize_vector",
    "directional_dot",
    "finsler_length_value",
    "create_finsler_input",
    "compute_finsler_distance_result",
]
