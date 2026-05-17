"""Deterministic Mahalanobis distance utilities for CEBA Core Metric Processing."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from ceba.core.metrics.schema import (
    DistanceKind,
    MetricDistanceResult,
    MetricTensorRef,
    MetricTensorStatus,
    MetricVectorRef,
    ensure_finite_number,
    ensure_non_negative_number,
    ensure_positive_int,
    new_metric_distance_id,
    utc_now,
)
from ceba.core.metrics.spd import check_spd_candidate, ensure_matrix_shape


_NEGATIVE_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class MahalanobisInput:
    left: MetricVectorRef
    right: MetricVectorRef
    left_values: tuple[float, ...]
    right_values: tuple[float, ...]
    tensor: MetricTensorRef
    tensor_matrix: tuple[tuple[float, ...], ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.left, MetricVectorRef):
            raise TypeError("left must be MetricVectorRef")
        if not isinstance(self.right, MetricVectorRef):
            raise TypeError("right must be MetricVectorRef")
        if not isinstance(self.tensor, MetricTensorRef):
            raise TypeError("tensor must be MetricTensorRef")
        if self.left.dimension != self.right.dimension:
            raise ValueError("left.dimension must equal right.dimension")
        if self.left.dimension != self.tensor.dimension:
            raise ValueError("vector dimensions must equal tensor.dimension")

        object.__setattr__(
            self,
            "left_values",
            ensure_vector_values(self.left_values, dimension=self.tensor.dimension, field_name="left_values"),
        )
        object.__setattr__(
            self,
            "right_values",
            ensure_vector_values(self.right_values, dimension=self.tensor.dimension, field_name="right_values"),
        )
        matrix = ensure_matrix_shape(self.tensor_matrix)
        if len(matrix) != self.tensor.dimension:
            raise ValueError("tensor_matrix dimension must equal tensor.dimension")
        if any(len(row) != self.tensor.dimension for row in matrix):
            raise ValueError("tensor_matrix must be square")
        object.__setattr__(self, "tensor_matrix", matrix)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    @property
    def dimension(self) -> int:
        return self.tensor.dimension


def ensure_vector_values(
    values: tuple[float, ...] | list[float],
    *,
    dimension: int,
    field_name: str,
) -> tuple[float, ...]:
    expected_dimension = ensure_positive_int(dimension, "dimension")
    if not isinstance(field_name, str) or not field_name:
        raise ValueError("field_name must be a non-empty string")
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple[float, ...] or list[float]")
    if len(values) != expected_dimension:
        raise ValueError(f"{field_name} length must equal dimension")
    return tuple(ensure_finite_number(value, f"{field_name}[{index}]") for index, value in enumerate(values))


def vector_difference(
    left_values: tuple[float, ...],
    right_values: tuple[float, ...],
) -> tuple[float, ...]:
    _ensure_vector_tuple(left_values, "left_values")
    _ensure_vector_tuple(right_values, "right_values")
    if len(left_values) != len(right_values):
        raise ValueError("left_values and right_values must have the same length")
    return tuple(
        ensure_finite_number(left_value - right_value, f"difference[{index}]")
        for index, (left_value, right_value) in enumerate(zip(left_values, right_values, strict=True))
    )


def matrix_vector_product(
    matrix: tuple[tuple[float, ...], ...],
    vector: tuple[float, ...],
) -> tuple[float, ...]:
    frozen_matrix = ensure_matrix_shape(matrix)
    _ensure_square_matrix(frozen_matrix)
    _ensure_vector_tuple(vector, "vector")
    if len(vector) != len(frozen_matrix):
        raise ValueError("vector length must equal matrix dimension")
    return tuple(
        ensure_finite_number(
            sum(row[column_index] * vector[column_index] for column_index in range(len(vector))),
            f"matrix_vector_product[{row_index}]",
        )
        for row_index, row in enumerate(frozen_matrix)
    )


def dot_product(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> float:
    _ensure_vector_tuple(left, "left")
    _ensure_vector_tuple(right, "right")
    if len(left) != len(right):
        raise ValueError("left and right must have the same length")
    return ensure_finite_number(
        sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True)),
        "dot_product",
    )


def quadratic_form(
    vector: tuple[float, ...],
    matrix: tuple[tuple[float, ...], ...],
) -> float:
    frozen_matrix = ensure_matrix_shape(matrix)
    _ensure_square_matrix(frozen_matrix)
    _ensure_vector_tuple(vector, "vector")
    if len(vector) != len(frozen_matrix):
        raise ValueError("vector length must equal matrix dimension")
    return ensure_finite_number(dot_product(vector, matrix_vector_product(frozen_matrix, vector)), "quadratic_form")


def mahalanobis_squared(
    *,
    left_values: tuple[float, ...] | list[float],
    right_values: tuple[float, ...] | list[float],
    tensor_matrix: tuple[tuple[float, ...], ...] | list[list[float]],
) -> float:
    matrix = ensure_matrix_shape(tensor_matrix)
    _ensure_square_matrix(matrix)
    dimension = len(matrix)
    left_vector = ensure_vector_values(left_values, dimension=dimension, field_name="left_values")
    right_vector = ensure_vector_values(right_values, dimension=dimension, field_name="right_values")
    delta = vector_difference(left_vector, right_vector)
    value = quadratic_form(delta, matrix)
    if -_NEGATIVE_TOLERANCE <= value < 0.0:
        return 0.0
    return ensure_non_negative_number(value, "mahalanobis_squared")


def mahalanobis_distance(
    *,
    left_values: tuple[float, ...] | list[float],
    right_values: tuple[float, ...] | list[float],
    tensor_matrix: tuple[tuple[float, ...], ...] | list[list[float]],
) -> float:
    squared = mahalanobis_squared(
        left_values=left_values,
        right_values=right_values,
        tensor_matrix=tensor_matrix,
    )
    return ensure_non_negative_number(math.sqrt(squared), "mahalanobis_distance")


def create_mahalanobis_input(
    *,
    left: MetricVectorRef,
    right: MetricVectorRef,
    left_values: tuple[float, ...] | list[float],
    right_values: tuple[float, ...] | list[float],
    tensor: MetricTensorRef,
    tensor_matrix: tuple[tuple[float, ...], ...] | list[list[float]],
    metadata: dict[str, object] | None = None,
) -> MahalanobisInput:
    return MahalanobisInput(
        left=left,
        right=right,
        left_values=tuple(left_values) if isinstance(left_values, list) else left_values,
        right_values=tuple(right_values) if isinstance(right_values, list) else right_values,
        tensor=tensor,
        tensor_matrix=ensure_matrix_shape(tensor_matrix),
        metadata={} if metadata is None else metadata,
    )


def compute_mahalanobis_distance_result(
    *,
    mahalanobis_input: MahalanobisInput,
    confidence: float = 1.0,
    validate_spd_candidate: bool = True,
) -> MetricDistanceResult:
    if not isinstance(mahalanobis_input, MahalanobisInput):
        raise TypeError("mahalanobis_input must be MahalanobisInput")
    if not isinstance(validate_spd_candidate, bool):
        raise TypeError("validate_spd_candidate must be bool")

    squared_distance = mahalanobis_squared(
        left_values=mahalanobis_input.left_values,
        right_values=mahalanobis_input.right_values,
        tensor_matrix=mahalanobis_input.tensor_matrix,
    )
    distance = ensure_non_negative_number(math.sqrt(squared_distance), "distance")

    metadata = dict(mahalanobis_input.metadata)
    metadata.update(
        {
            "distance_kind": "mahalanobis",
            "squared_distance": squared_distance,
            "dimension": mahalanobis_input.dimension,
            "tensor_id": mahalanobis_input.tensor.tensor_id,
        }
    )
    if validate_spd_candidate:
        metadata["spd_check"] = check_spd_candidate(mahalanobis_input.tensor_matrix).to_dict()

    return MetricDistanceResult(
        distance_id=new_metric_distance_id(),
        kind=DistanceKind.MAHALANOBIS,
        left=mahalanobis_input.left,
        right=mahalanobis_input.right,
        tensor=mahalanobis_input.tensor,
        value=distance,
        confidence=confidence,
        created_at=utc_now(),
        metadata=metadata,
    )


def tensor_ref_is_compatible(
    *,
    tensor: MetricTensorRef,
    left: MetricVectorRef,
    right: MetricVectorRef,
) -> bool:
    if not isinstance(tensor, MetricTensorRef):
        raise TypeError("tensor must be MetricTensorRef")
    if not isinstance(left, MetricVectorRef):
        raise TypeError("left must be MetricVectorRef")
    if not isinstance(right, MetricVectorRef):
        raise TypeError("right must be MetricVectorRef")
    return (
        left.dimension == right.dimension == tensor.dimension
        and left.space is right.space
        and tensor.space is left.space
        and tensor.is_spd is True
        and tensor.status is not MetricTensorStatus.INVALID
    )


def _ensure_vector_tuple(values: tuple[float, ...], field_name: str) -> tuple[float, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{field_name} must be tuple[float, ...]")
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    return tuple(ensure_finite_number(value, f"{field_name}[{index}]") for index, value in enumerate(values))


def _ensure_square_matrix(matrix: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be square")
    return matrix


def _copy_metadata(value: dict[str, object]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("metadata must be dict[str, object]")
    if any(not isinstance(key, str) for key in value):
        raise TypeError("metadata keys must be strings")
    return dict(value)


__all__ = [
    "MahalanobisInput",
    "ensure_vector_values",
    "vector_difference",
    "matrix_vector_product",
    "dot_product",
    "quadratic_form",
    "mahalanobis_squared",
    "mahalanobis_distance",
    "create_mahalanobis_input",
    "compute_mahalanobis_distance_result",
    "tensor_ref_is_compatible",
]
