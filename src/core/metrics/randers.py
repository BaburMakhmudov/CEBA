"""Deterministic lightweight Randers metric utilities for CEBA Core."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from ceba.core.metrics.finsler import directional_dot, euclidean_norm, normalize_vector
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


DEFAULT_RANDERS_WIND_LIMIT = 0.999999
_NEGATIVE_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class RandersPolicy:
    base_weight: float = 1.0
    wind_weight: float = 1.0
    normalize_wind: bool = False
    enforce_wind_limit: bool = True
    wind_limit: float = DEFAULT_RANDERS_WIND_LIMIT

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_weight", ensure_non_negative_number(self.base_weight, "base_weight"))
        object.__setattr__(self, "wind_weight", ensure_non_negative_number(self.wind_weight, "wind_weight"))
        if not isinstance(self.normalize_wind, bool):
            raise TypeError("normalize_wind must be bool")
        if not isinstance(self.enforce_wind_limit, bool):
            raise TypeError("enforce_wind_limit must be bool")
        object.__setattr__(self, "wind_limit", ensure_constraint_probability(self.wind_limit, "wind_limit"))


@dataclass(frozen=True, slots=True)
class RandersInput:
    left: MetricVectorRef
    right: MetricVectorRef
    left_values: tuple[float, ...]
    right_values: tuple[float, ...]
    wind_values: tuple[float, ...]
    tensor: MetricTensorRef | None
    tensor_matrix: tuple[tuple[float, ...], ...] | None
    policy: RandersPolicy
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
        object.__setattr__(
            self,
            "wind_values",
            ensure_vector_values(self.wind_values, dimension=dimension, field_name="wind_values"),
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
        if not isinstance(self.policy, RandersPolicy):
            raise TypeError("policy must be RandersPolicy")
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    @property
    def dimension(self) -> int:
        return self.left.dimension


def wind_norm(
    wind_values: tuple[float, ...],
    *,
    tensor_matrix: tuple[tuple[float, ...], ...] | None = None,
) -> float:
    checked_wind = _ensure_finite_tuple(wind_values, "wind_values")
    if tensor_matrix is None:
        return ensure_non_negative_number(euclidean_norm(checked_wind), "wind_norm")

    matrix = _ensure_matrix_values(tensor_matrix, dimension=len(checked_wind), field_name="tensor_matrix")
    return ensure_non_negative_number(
        math.sqrt(max(ensure_finite_number(quadratic_form(checked_wind, matrix), "quadratic_form"), 0.0)),
        "wind_norm",
    )


def validate_randers_wind(
    *,
    wind_values: tuple[float, ...],
    policy: RandersPolicy,
    tensor_matrix: tuple[tuple[float, ...], ...] | None = None,
) -> tuple[float, ...]:
    checked_wind = _ensure_finite_tuple(wind_values, "wind_values")
    if not isinstance(policy, RandersPolicy):
        raise TypeError("policy must be RandersPolicy")

    wind_used = normalize_vector(checked_wind) if policy.normalize_wind else checked_wind
    if policy.enforce_wind_limit and wind_norm(wind_used, tensor_matrix=tensor_matrix) >= policy.wind_limit:
        raise ValueError("wind norm must be less than wind_limit")
    return wind_used


def randers_base_length(
    *,
    displacement: tuple[float, ...],
    tensor_matrix: tuple[tuple[float, ...], ...] | None = None,
) -> float:
    checked_displacement = _ensure_finite_tuple(displacement, "displacement")
    if tensor_matrix is None:
        return ensure_non_negative_number(euclidean_norm(checked_displacement), "randers_base_length")

    matrix = _ensure_matrix_values(tensor_matrix, dimension=len(checked_displacement), field_name="tensor_matrix")
    return ensure_non_negative_number(
        math.sqrt(max(ensure_finite_number(quadratic_form(checked_displacement, matrix), "quadratic_form"), 0.0)),
        "randers_base_length",
    )


def randers_wind_term(
    *,
    displacement: tuple[float, ...],
    wind_values: tuple[float, ...],
    policy: RandersPolicy,
) -> float:
    checked_displacement = _ensure_finite_tuple(displacement, "displacement")
    checked_wind = _ensure_finite_tuple(wind_values, "wind_values")
    if len(checked_displacement) != len(checked_wind):
        raise ValueError("displacement and wind_values must have the same length")
    if not isinstance(policy, RandersPolicy):
        raise TypeError("policy must be RandersPolicy")
    return ensure_finite_number(
        policy.wind_weight * directional_dot(vector=checked_displacement, direction=checked_wind),
        "randers_wind_term",
    )


def randers_length_value(
    *,
    displacement: tuple[float, ...],
    wind_values: tuple[float, ...],
    policy: RandersPolicy,
    tensor_matrix: tuple[tuple[float, ...], ...] | None = None,
) -> float:
    checked_displacement = _ensure_finite_tuple(displacement, "displacement")
    checked_wind = _ensure_finite_tuple(wind_values, "wind_values")
    if len(checked_displacement) != len(checked_wind):
        raise ValueError("displacement and wind_values must have the same length")
    if not isinstance(policy, RandersPolicy):
        raise TypeError("policy must be RandersPolicy")

    matrix = (
        _ensure_matrix_values(tensor_matrix, dimension=len(checked_displacement), field_name="tensor_matrix")
        if tensor_matrix is not None
        else None
    )
    wind_used = validate_randers_wind(wind_values=checked_wind, policy=policy, tensor_matrix=matrix)
    alpha = ensure_non_negative_number(
        policy.base_weight * randers_base_length(displacement=checked_displacement, tensor_matrix=matrix),
        "randers_alpha",
    )
    beta = randers_wind_term(displacement=checked_displacement, wind_values=wind_used, policy=policy)
    raw = ensure_finite_number(alpha + beta, "randers_raw_length")

    if raw < 0.0:
        if abs(raw) <= _NEGATIVE_TOLERANCE:
            return 0.0
        raise ValueError("randers length must be non-negative")
    return ensure_non_negative_number(raw, "randers_length")


def create_randers_input(
    *,
    left: MetricVectorRef,
    right: MetricVectorRef,
    left_values: tuple[float, ...] | list[float],
    right_values: tuple[float, ...] | list[float],
    wind_values: tuple[float, ...] | list[float],
    tensor: MetricTensorRef | None = None,
    tensor_matrix: tuple[tuple[float, ...], ...] | list[list[float]] | None = None,
    policy: RandersPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> RandersInput:
    return RandersInput(
        left=left,
        right=right,
        left_values=tuple(left_values) if isinstance(left_values, list) else left_values,
        right_values=tuple(right_values) if isinstance(right_values, list) else right_values,
        wind_values=tuple(wind_values) if isinstance(wind_values, list) else wind_values,
        tensor=tensor,
        tensor_matrix=tensor_matrix,
        policy=RandersPolicy() if policy is None else policy,
        metadata={} if metadata is None else metadata,
    )


def compute_randers_distance_result(
    *,
    randers_input: RandersInput,
    confidence: float = 1.0,
) -> MetricDistanceResult:
    if not isinstance(randers_input, RandersInput):
        raise TypeError("randers_input must be RandersInput")
    checked_confidence = ensure_constraint_probability(confidence, "confidence")
    displacement = vector_difference(randers_input.left_values, randers_input.right_values)
    matrix = randers_input.tensor_matrix
    wind_used = validate_randers_wind(
        wind_values=randers_input.wind_values,
        policy=randers_input.policy,
        tensor_matrix=matrix,
    )
    value = randers_length_value(
        displacement=displacement,
        wind_values=randers_input.wind_values,
        policy=randers_input.policy,
        tensor_matrix=matrix,
    )

    metadata = dict(randers_input.metadata)
    metadata.update(
        {
            "distance_kind": "randers_length",
            "dimension": randers_input.dimension,
            "base_weight": randers_input.policy.base_weight,
            "wind_weight": randers_input.policy.wind_weight,
            "normalize_wind": randers_input.policy.normalize_wind,
            "enforce_wind_limit": randers_input.policy.enforce_wind_limit,
            "wind_norm": wind_norm(wind_used, tensor_matrix=matrix),
            "has_tensor": randers_input.tensor is not None,
        }
    )

    return MetricDistanceResult(
        distance_id=new_metric_distance_id(),
        kind=DistanceKind.FINSLER_LENGTH,
        left=randers_input.left,
        right=randers_input.right,
        tensor=randers_input.tensor,
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
    "DEFAULT_RANDERS_WIND_LIMIT",
    "RandersPolicy",
    "RandersInput",
    "wind_norm",
    "validate_randers_wind",
    "randers_base_length",
    "randers_wind_term",
    "randers_length_value",
    "create_randers_input",
    "compute_randers_distance_result",
]
