"""Deterministic lightweight manifold primitives for CEBA Core Dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math

from ceba.core.metrics.schema import (
    ensure_finite_number,
    ensure_non_negative_number,
    ensure_positive_int,
    utc_now,
)


DEFAULT_VECTOR_TOLERANCE = 1e-12


class ManifoldKind(Enum):
    EUCLIDEAN = "EUCLIDEAN"
    UNIT_SPHERE = "UNIT_SPHERE"
    RIEMANNIAN_APPROX = "RIEMANNIAN_APPROX"
    FINSLER_APPROX = "FINSLER_APPROX"
    UNKNOWN = "UNKNOWN"


class ManifoldPointStatus(Enum):
    VALID = "VALID"
    NORMALIZED = "NORMALIZED"
    DEGENERATE = "DEGENERATE"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class ManifoldPoint:
    point_id: str
    kind: ManifoldKind
    values: tuple[float, ...]
    dimension: int
    status: ManifoldPointStatus
    normalized: bool
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_id", _ensure_non_empty_string(self.point_id, "point_id"))
        if not isinstance(self.kind, ManifoldKind):
            raise TypeError("kind must be ManifoldKind")
        object.__setattr__(self, "values", ensure_vector(self.values))
        object.__setattr__(self, "dimension", ensure_positive_int(self.dimension, "dimension"))
        if len(self.values) != self.dimension:
            raise ValueError("len(values) must equal dimension")
        if not isinstance(self.status, ManifoldPointStatus):
            raise TypeError("status must be ManifoldPointStatus")
        if not isinstance(self.normalized, bool):
            raise TypeError("normalized must be bool")
        object.__setattr__(self, "created_at", _ensure_timezone_aware(self.created_at, "created_at"))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "point_id": self.point_id,
            "kind": self.kind.value,
            "values": self.values,
            "dimension": self.dimension,
            "status": self.status.value,
            "normalized": self.normalized,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class TangentVector:
    vector_id: str
    base_point_id: str | None
    values: tuple[float, ...]
    dimension: int
    norm: float
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "vector_id", _ensure_non_empty_string(self.vector_id, "vector_id"))
        if self.base_point_id is not None:
            object.__setattr__(
                self,
                "base_point_id",
                _ensure_non_empty_string(self.base_point_id, "base_point_id"),
            )
        object.__setattr__(self, "values", ensure_vector(self.values))
        object.__setattr__(self, "dimension", ensure_positive_int(self.dimension, "dimension"))
        if len(self.values) != self.dimension:
            raise ValueError("len(values) must equal dimension")
        object.__setattr__(self, "norm", ensure_non_negative_number(self.norm, "norm"))
        computed_norm = euclidean_norm(self.values)
        if abs(self.norm - computed_norm) > DEFAULT_VECTOR_TOLERANCE:
            raise ValueError("norm must match euclidean_norm(values)")
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "vector_id": self.vector_id,
            "base_point_id": self.base_point_id,
            "values": self.values,
            "dimension": self.dimension,
            "norm": self.norm,
            "metadata": dict(self.metadata),
        }


def ensure_vector(
    values: tuple[float, ...] | list[float],
    *,
    field_name: str = "values",
) -> tuple[float, ...]:
    _ensure_non_empty_string(field_name, "field_name")
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple[float, ...] or list[float]")
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    return tuple(ensure_finite_number(value, f"{field_name}[{index}]") for index, value in enumerate(values))


def vector_dimension(values: tuple[float, ...] | list[float]) -> int:
    return len(ensure_vector(values))


def euclidean_norm(values: tuple[float, ...] | list[float]) -> float:
    checked_values = ensure_vector(values)
    return ensure_non_negative_number(
        math.sqrt(sum(value * value for value in checked_values)),
        "euclidean_norm",
    )


def normalize_vector(
    values: tuple[float, ...] | list[float],
    *,
    tolerance: float = DEFAULT_VECTOR_TOLERANCE,
) -> tuple[float, ...]:
    checked_values = ensure_vector(values)
    checked_tolerance = ensure_non_negative_number(tolerance, "tolerance")
    norm = euclidean_norm(checked_values)
    if norm <= checked_tolerance:
        raise ValueError("cannot normalize vector with norm less than or equal to tolerance")
    return tuple(
        ensure_finite_number(value / norm, f"normalized[{index}]")
        for index, value in enumerate(checked_values)
    )


def dot_product(
    left: tuple[float, ...] | list[float],
    right: tuple[float, ...] | list[float],
) -> float:
    left_values = ensure_vector(left, field_name="left")
    right_values = ensure_vector(right, field_name="right")
    _ensure_same_dimension(left_values, right_values)
    return ensure_finite_number(
        sum(left_value * right_value for left_value, right_value in zip(left_values, right_values, strict=True)),
        "dot_product",
    )


def vector_add(
    left: tuple[float, ...] | list[float],
    right: tuple[float, ...] | list[float],
) -> tuple[float, ...]:
    left_values = ensure_vector(left, field_name="left")
    right_values = ensure_vector(right, field_name="right")
    _ensure_same_dimension(left_values, right_values)
    return tuple(
        ensure_finite_number(left_value + right_value, f"sum[{index}]")
        for index, (left_value, right_value) in enumerate(zip(left_values, right_values, strict=True))
    )


def vector_subtract(
    left: tuple[float, ...] | list[float],
    right: tuple[float, ...] | list[float],
) -> tuple[float, ...]:
    left_values = ensure_vector(left, field_name="left")
    right_values = ensure_vector(right, field_name="right")
    _ensure_same_dimension(left_values, right_values)
    return tuple(
        ensure_finite_number(left_value - right_value, f"difference[{index}]")
        for index, (left_value, right_value) in enumerate(zip(left_values, right_values, strict=True))
    )


def scalar_multiply(
    scalar: float,
    values: tuple[float, ...] | list[float],
) -> tuple[float, ...]:
    checked_scalar = ensure_finite_number(scalar, "scalar")
    checked_values = ensure_vector(values)
    return tuple(
        ensure_finite_number(checked_scalar * value, f"scaled[{index}]")
        for index, value in enumerate(checked_values)
    )


def cosine_similarity(
    left: tuple[float, ...] | list[float],
    right: tuple[float, ...] | list[float],
    *,
    tolerance: float = DEFAULT_VECTOR_TOLERANCE,
) -> float:
    left_values = ensure_vector(left, field_name="left")
    right_values = ensure_vector(right, field_name="right")
    checked_tolerance = ensure_non_negative_number(tolerance, "tolerance")
    _ensure_same_dimension(left_values, right_values)
    left_norm = euclidean_norm(left_values)
    right_norm = euclidean_norm(right_values)
    if left_norm <= checked_tolerance or right_norm <= checked_tolerance:
        return 0.0
    value = ensure_finite_number(dot_product(left_values, right_values) / (left_norm * right_norm), "cosine")
    if value < -1.0:
        return -1.0
    if value > 1.0:
        return 1.0
    return value


def project_to_unit_sphere(
    values: tuple[float, ...] | list[float],
    *,
    tolerance: float = DEFAULT_VECTOR_TOLERANCE,
) -> tuple[float, ...]:
    return normalize_vector(values, tolerance=tolerance)


def point_status_for_values(
    values: tuple[float, ...],
    *,
    kind: ManifoldKind,
    tolerance: float = DEFAULT_VECTOR_TOLERANCE,
) -> ManifoldPointStatus:
    checked_values = ensure_vector(values)
    if not isinstance(kind, ManifoldKind):
        raise TypeError("kind must be ManifoldKind")
    checked_tolerance = ensure_non_negative_number(tolerance, "tolerance")
    norm = euclidean_norm(checked_values)
    if norm <= checked_tolerance:
        return ManifoldPointStatus.DEGENERATE
    if kind is ManifoldKind.UNIT_SPHERE and abs(norm - 1.0) <= checked_tolerance:
        return ManifoldPointStatus.NORMALIZED
    return ManifoldPointStatus.VALID


def create_manifold_point(
    *,
    point_id: str,
    values: tuple[float, ...] | list[float],
    kind: ManifoldKind = ManifoldKind.EUCLIDEAN,
    normalize: bool = False,
    metadata: dict[str, object] | None = None,
) -> ManifoldPoint:
    _ensure_non_empty_string(point_id, "point_id")
    if not isinstance(kind, ManifoldKind):
        raise TypeError("kind must be ManifoldKind")
    if not isinstance(normalize, bool):
        raise TypeError("normalize must be bool")
    point_values = ensure_vector(values)
    normalized = False
    if normalize:
        point_values = project_to_unit_sphere(point_values)
        normalized = True

    point_metadata: dict[str, object] = {"source": "manifold"}
    if metadata is not None:
        point_metadata.update(_copy_metadata(metadata))
    point_metadata["source"] = "manifold"

    return ManifoldPoint(
        point_id=point_id,
        kind=kind,
        values=point_values,
        dimension=len(point_values),
        status=point_status_for_values(point_values, kind=kind),
        normalized=normalized,
        created_at=utc_now(),
        metadata=point_metadata,
    )


def create_tangent_vector(
    *,
    vector_id: str,
    values: tuple[float, ...] | list[float],
    base_point_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> TangentVector:
    _ensure_non_empty_string(vector_id, "vector_id")
    if base_point_id is not None:
        _ensure_non_empty_string(base_point_id, "base_point_id")
    vector_values = ensure_vector(values)
    vector_metadata: dict[str, object] = {"source": "tangent_vector"}
    if metadata is not None:
        vector_metadata.update(_copy_metadata(metadata))
    vector_metadata["source"] = "tangent_vector"

    return TangentVector(
        vector_id=vector_id,
        base_point_id=base_point_id,
        values=vector_values,
        dimension=len(vector_values),
        norm=euclidean_norm(vector_values),
        metadata=vector_metadata,
    )


def tangent_from_points(
    *,
    vector_id: str,
    left: ManifoldPoint,
    right: ManifoldPoint,
) -> TangentVector:
    if not isinstance(left, ManifoldPoint):
        raise TypeError("left must be ManifoldPoint")
    if not isinstance(right, ManifoldPoint):
        raise TypeError("right must be ManifoldPoint")
    if left.dimension != right.dimension:
        raise ValueError("left.dimension must equal right.dimension")
    return create_tangent_vector(
        vector_id=vector_id,
        values=vector_subtract(left.values, right.values),
        base_point_id=right.point_id,
    )


def manifold_points_compatible(
    left: ManifoldPoint,
    right: ManifoldPoint,
) -> bool:
    if not isinstance(left, ManifoldPoint):
        raise TypeError("left must be ManifoldPoint")
    if not isinstance(right, ManifoldPoint):
        raise TypeError("right must be ManifoldPoint")
    return left.dimension == right.dimension and left.kind is right.kind


def _ensure_same_dimension(left: tuple[float, ...], right: tuple[float, ...]) -> None:
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")


def _ensure_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _ensure_timezone_aware(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _copy_metadata(value: dict[str, object]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("metadata must be dict[str, object]")
    if any(not isinstance(key, str) for key in value):
        raise TypeError("metadata keys must be strings")
    return dict(value)


__all__ = [
    "DEFAULT_VECTOR_TOLERANCE",
    "ManifoldKind",
    "ManifoldPointStatus",
    "ManifoldPoint",
    "TangentVector",
    "ensure_vector",
    "vector_dimension",
    "euclidean_norm",
    "normalize_vector",
    "dot_product",
    "vector_add",
    "vector_subtract",
    "scalar_multiply",
    "cosine_similarity",
    "project_to_unit_sphere",
    "point_status_for_values",
    "create_manifold_point",
    "create_tangent_vector",
    "tangent_from_points",
    "manifold_points_compatible",
]
