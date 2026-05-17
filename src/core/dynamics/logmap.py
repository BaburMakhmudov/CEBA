"""Deterministic lightweight logarithmic-map utilities for CEBA Core Dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ceba.core.dynamics.manifold import (
    DEFAULT_VECTOR_TOLERANCE,
    ManifoldKind,
    ManifoldPoint,
    TangentVector,
    create_tangent_vector,
    dot_product,
    ensure_vector,
    euclidean_norm,
    scalar_multiply,
    vector_subtract,
)
from ceba.core.metrics.schema import (
    ensure_finite_number,
    ensure_non_negative_number,
    utc_now,
)


class LogMapMode(Enum):
    EUCLIDEAN_SUBTRACT = "EUCLIDEAN_SUBTRACT"
    UNIT_SPHERE_TANGENT_APPROX = "UNIT_SPHERE_TANGENT_APPROX"
    RIEMANNIAN_APPROX = "RIEMANNIAN_APPROX"
    FINSLER_APPROX = "FINSLER_APPROX"


class LogMapStatus(Enum):
    COMPUTED = "COMPUTED"
    ZERO = "ZERO"
    DEGRADED = "DEGRADED"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class LogMapPolicy:
    project_unit_sphere_tangent: bool = True
    max_tangent_norm: float = 10.0
    zero_tolerance: float = DEFAULT_VECTOR_TOLERANCE

    def __post_init__(self) -> None:
        if not isinstance(self.project_unit_sphere_tangent, bool):
            raise TypeError("project_unit_sphere_tangent must be bool")
        object.__setattr__(
            self,
            "max_tangent_norm",
            ensure_non_negative_number(self.max_tangent_norm, "max_tangent_norm"),
        )
        object.__setattr__(
            self,
            "zero_tolerance",
            ensure_non_negative_number(self.zero_tolerance, "zero_tolerance"),
        )


@dataclass(frozen=True, slots=True)
class LogMapResult:
    source_point: ManifoldPoint
    target_point: ManifoldPoint
    tangent: TangentVector
    mode: LogMapMode
    status: LogMapStatus
    distance: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source_point, ManifoldPoint):
            raise TypeError("source_point must be ManifoldPoint")
        if not isinstance(self.target_point, ManifoldPoint):
            raise TypeError("target_point must be ManifoldPoint")
        if not isinstance(self.tangent, TangentVector):
            raise TypeError("tangent must be TangentVector")
        if not isinstance(self.mode, LogMapMode):
            raise TypeError("mode must be LogMapMode")
        if not isinstance(self.status, LogMapStatus):
            raise TypeError("status must be LogMapStatus")
        object.__setattr__(self, "distance", ensure_non_negative_number(self.distance, "distance"))
        object.__setattr__(self, "created_at", _ensure_timezone_aware(self.created_at, "created_at"))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "source_point": self.source_point.to_dict(),
            "target_point": self.target_point.to_dict(),
            "tangent": self.tangent.to_dict(),
            "mode": self.mode.value,
            "status": self.status.value,
            "distance": self.distance,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


def logmap_mode_for_points(
    source: ManifoldPoint,
    target: ManifoldPoint,
) -> LogMapMode:
    if not isinstance(source, ManifoldPoint):
        raise TypeError("source must be ManifoldPoint")
    if not isinstance(target, ManifoldPoint):
        raise TypeError("target must be ManifoldPoint")
    if source.kind is not target.kind:
        return LogMapMode.EUCLIDEAN_SUBTRACT
    if source.kind is ManifoldKind.UNIT_SPHERE:
        return LogMapMode.UNIT_SPHERE_TANGENT_APPROX
    if source.kind is ManifoldKind.EUCLIDEAN:
        return LogMapMode.EUCLIDEAN_SUBTRACT
    if source.kind is ManifoldKind.RIEMANNIAN_APPROX:
        return LogMapMode.RIEMANNIAN_APPROX
    if source.kind is ManifoldKind.FINSLER_APPROX:
        return LogMapMode.FINSLER_APPROX
    return LogMapMode.EUCLIDEAN_SUBTRACT


def project_to_tangent_space(
    *,
    base_values: tuple[float, ...],
    vector_values: tuple[float, ...],
) -> tuple[float, ...]:
    base = ensure_vector(base_values, field_name="base_values")
    vector = ensure_vector(vector_values, field_name="vector_values")
    if len(base) != len(vector):
        raise ValueError("base_values and vector_values must have the same length")
    projection_scale = dot_product(vector, base)
    projected_component = scalar_multiply(projection_scale, base)
    return vector_subtract(vector, projected_component)


def bound_tangent_values(
    *,
    values: tuple[float, ...],
    max_norm: float,
) -> tuple[float, ...]:
    checked_values = ensure_vector(values)
    checked_max_norm = ensure_non_negative_number(max_norm, "max_norm")
    norm = euclidean_norm(checked_values)
    if norm == 0.0:
        return checked_values
    if checked_max_norm == 0.0:
        return tuple(0.0 for _ in checked_values)
    if norm > checked_max_norm:
        return scalar_multiply(checked_max_norm / norm, checked_values)
    return tuple(ensure_finite_number(value, f"bounded[{index}]") for index, value in enumerate(checked_values))


def compute_logmap_values(
    *,
    source: ManifoldPoint,
    target: ManifoldPoint,
    policy: LogMapPolicy,
) -> tuple[float, ...]:
    if not isinstance(source, ManifoldPoint):
        raise TypeError("source must be ManifoldPoint")
    if not isinstance(target, ManifoldPoint):
        raise TypeError("target must be ManifoldPoint")
    if source.dimension != target.dimension:
        raise ValueError("source.dimension must equal target.dimension")
    if not isinstance(policy, LogMapPolicy):
        raise TypeError("policy must be LogMapPolicy")

    raw = vector_subtract(target.values, source.values)
    mode = logmap_mode_for_points(source, target)
    if mode is LogMapMode.UNIT_SPHERE_TANGENT_APPROX and policy.project_unit_sphere_tangent:
        raw = project_to_tangent_space(base_values=source.values, vector_values=raw)
    return bound_tangent_values(values=raw, max_norm=policy.max_tangent_norm)


def next_tangent_id(
    *,
    source_point_id: str,
    target_point_id: str,
    suffix: str = "logmap",
) -> str:
    source_id = _ensure_non_empty_string(source_point_id, "source_point_id")
    target_id = _ensure_non_empty_string(target_point_id, "target_point_id")
    checked_suffix = _ensure_non_empty_string(suffix, "suffix")
    return f"{source_id}:{target_id}:{checked_suffix}"


def apply_logmap(
    *,
    source: ManifoldPoint,
    target: ManifoldPoint,
    policy: LogMapPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> LogMapResult:
    if not isinstance(source, ManifoldPoint):
        raise TypeError("source must be ManifoldPoint")
    if not isinstance(target, ManifoldPoint):
        raise TypeError("target must be ManifoldPoint")
    if source.dimension != target.dimension:
        raise ValueError("source.dimension must equal target.dimension")

    active_policy = LogMapPolicy() if policy is None else policy
    if not isinstance(active_policy, LogMapPolicy):
        raise TypeError("policy must be LogMapPolicy or None")

    mode = logmap_mode_for_points(source, target)
    values = compute_logmap_values(source=source, target=target, policy=active_policy)
    distance = euclidean_norm(values)
    same_kind = source.kind is target.kind
    if distance <= active_policy.zero_tolerance:
        status = LogMapStatus.ZERO
    elif not same_kind:
        status = LogMapStatus.DEGRADED
    else:
        status = LogMapStatus.COMPUTED

    result_metadata: dict[str, object] = {
        "source": "logmap",
        "mode": mode.value,
        "source_point_id": source.point_id,
        "target_point_id": target.point_id,
        "same_kind": same_kind,
        "project_unit_sphere_tangent": active_policy.project_unit_sphere_tangent,
    }
    if metadata is not None:
        result_metadata.update(_copy_metadata(metadata))
    result_metadata["source"] = "logmap"

    tangent = create_tangent_vector(
        vector_id=next_tangent_id(source_point_id=source.point_id, target_point_id=target.point_id),
        values=values,
        base_point_id=source.point_id,
        metadata=result_metadata,
    )

    return LogMapResult(
        source_point=source,
        target_point=target,
        tangent=tangent,
        mode=mode,
        status=status,
        distance=distance,
        created_at=utc_now(),
        metadata=result_metadata,
    )


def logmap_distance(
    *,
    source: ManifoldPoint,
    target: ManifoldPoint,
    policy: LogMapPolicy | None = None,
) -> float:
    return apply_logmap(source=source, target=target, policy=policy).distance


def logmap_is_zero(result: LogMapResult) -> bool:
    if not isinstance(result, LogMapResult):
        raise TypeError("result must be LogMapResult")
    return result.status is LogMapStatus.ZERO or result.distance <= 0.0


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
    "LogMapMode",
    "LogMapStatus",
    "LogMapPolicy",
    "LogMapResult",
    "logmap_mode_for_points",
    "project_to_tangent_space",
    "bound_tangent_values",
    "compute_logmap_values",
    "next_tangent_id",
    "apply_logmap",
    "logmap_distance",
    "logmap_is_zero",
]
