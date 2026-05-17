"""Deterministic lightweight exponential-map utilities for CEBA Core Dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ceba.core.dynamics.manifold import (
    DEFAULT_VECTOR_TOLERANCE,
    ManifoldKind,
    ManifoldPoint,
    TangentVector,
    create_manifold_point,
    ensure_vector,
    euclidean_norm,
    project_to_unit_sphere,
    scalar_multiply,
    vector_add,
)
from ceba.core.dynamics.semantic_temperature import (
    SemanticTemperatureState,
    transition_scale_from_temperature,
)
from ceba.core.metrics.schema import (
    ensure_constraint_probability,
    ensure_finite_number,
    ensure_non_negative_number,
    utc_now,
)


class ExpMapMode(Enum):
    EUCLIDEAN_ADD = "EUCLIDEAN_ADD"
    UNIT_SPHERE_RENORMALIZE = "UNIT_SPHERE_RENORMALIZE"
    RIEMANNIAN_APPROX = "RIEMANNIAN_APPROX"
    FINSLER_APPROX = "FINSLER_APPROX"


class ExpMapStatus(Enum):
    APPLIED = "APPLIED"
    SKIPPED = "SKIPPED"
    DEGRADED = "DEGRADED"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class ExpMapPolicy:
    max_step_norm: float = 1.0
    min_step_norm: float = 0.0
    renormalize_unit_sphere: bool = True
    allow_zero_tangent: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_step_norm", ensure_non_negative_number(self.max_step_norm, "max_step_norm"))
        object.__setattr__(self, "min_step_norm", ensure_non_negative_number(self.min_step_norm, "min_step_norm"))
        if self.max_step_norm < self.min_step_norm:
            raise ValueError("max_step_norm must be greater than or equal to min_step_norm")
        if not isinstance(self.renormalize_unit_sphere, bool):
            raise TypeError("renormalize_unit_sphere must be bool")
        if not isinstance(self.allow_zero_tangent, bool):
            raise TypeError("allow_zero_tangent must be bool")


@dataclass(frozen=True, slots=True)
class ExpMapResult:
    previous_point: ManifoldPoint
    tangent: TangentVector
    next_point: ManifoldPoint
    mode: ExpMapMode
    status: ExpMapStatus
    scale: float
    step_norm: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.previous_point, ManifoldPoint):
            raise TypeError("previous_point must be ManifoldPoint")
        if not isinstance(self.tangent, TangentVector):
            raise TypeError("tangent must be TangentVector")
        if not isinstance(self.next_point, ManifoldPoint):
            raise TypeError("next_point must be ManifoldPoint")
        if not isinstance(self.mode, ExpMapMode):
            raise TypeError("mode must be ExpMapMode")
        if not isinstance(self.status, ExpMapStatus):
            raise TypeError("status must be ExpMapStatus")
        object.__setattr__(self, "scale", ensure_constraint_probability(self.scale, "scale"))
        object.__setattr__(self, "step_norm", ensure_non_negative_number(self.step_norm, "step_norm"))
        object.__setattr__(self, "created_at", _ensure_timezone_aware(self.created_at, "created_at"))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "previous_point": self.previous_point.to_dict(),
            "tangent": self.tangent.to_dict(),
            "next_point": self.next_point.to_dict(),
            "mode": self.mode.value,
            "status": self.status.value,
            "scale": self.scale,
            "step_norm": self.step_norm,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


def expmap_mode_for_point(point: ManifoldPoint) -> ExpMapMode:
    if not isinstance(point, ManifoldPoint):
        raise TypeError("point must be ManifoldPoint")
    if point.kind is ManifoldKind.UNIT_SPHERE:
        return ExpMapMode.UNIT_SPHERE_RENORMALIZE
    if point.kind is ManifoldKind.EUCLIDEAN:
        return ExpMapMode.EUCLIDEAN_ADD
    if point.kind is ManifoldKind.RIEMANNIAN_APPROX:
        return ExpMapMode.RIEMANNIAN_APPROX
    if point.kind is ManifoldKind.FINSLER_APPROX:
        return ExpMapMode.FINSLER_APPROX
    return ExpMapMode.EUCLIDEAN_ADD


def bounded_step_values(
    *,
    tangent: TangentVector,
    scale: float,
    policy: ExpMapPolicy,
) -> tuple[float, ...]:
    if not isinstance(tangent, TangentVector):
        raise TypeError("tangent must be TangentVector")
    checked_scale = ensure_constraint_probability(scale, "scale")
    if not isinstance(policy, ExpMapPolicy):
        raise TypeError("policy must be ExpMapPolicy")

    raw_step = scalar_multiply(checked_scale, tangent.values)
    raw_norm = euclidean_norm(raw_step)
    if raw_norm == 0.0:
        if policy.allow_zero_tangent:
            return raw_step
        raise ValueError("zero tangent is not allowed")
    if raw_norm > policy.max_step_norm:
        return scalar_multiply(policy.max_step_norm / raw_norm, raw_step)
    if raw_norm < policy.min_step_norm:
        return scalar_multiply(policy.min_step_norm / raw_norm, raw_step)
    return tuple(ensure_finite_number(value, f"step[{index}]") for index, value in enumerate(raw_step))


def expmap_euclidean_values(
    *,
    point_values: tuple[float, ...],
    step_values: tuple[float, ...],
) -> tuple[float, ...]:
    return vector_add(
        ensure_vector(point_values, field_name="point_values"),
        ensure_vector(step_values, field_name="step_values"),
    )


def expmap_unit_sphere_values(
    *,
    point_values: tuple[float, ...],
    step_values: tuple[float, ...],
    tolerance: float = DEFAULT_VECTOR_TOLERANCE,
) -> tuple[float, ...]:
    combined = expmap_euclidean_values(point_values=point_values, step_values=step_values)
    return project_to_unit_sphere(combined, tolerance=tolerance)


def next_point_id(
    *,
    previous_point_id: str,
    suffix: str = "expmap",
) -> str:
    previous = _ensure_non_empty_string(previous_point_id, "previous_point_id")
    checked_suffix = _ensure_non_empty_string(suffix, "suffix")
    return f"{previous}:{checked_suffix}"


def compute_expmap_values(
    *,
    point: ManifoldPoint,
    tangent: TangentVector,
    scale: float,
    policy: ExpMapPolicy,
) -> tuple[float, ...]:
    if not isinstance(point, ManifoldPoint):
        raise TypeError("point must be ManifoldPoint")
    if not isinstance(tangent, TangentVector):
        raise TypeError("tangent must be TangentVector")
    if point.dimension != tangent.dimension:
        raise ValueError("tangent.dimension must equal point.dimension")
    checked_scale = ensure_constraint_probability(scale, "scale")
    if not isinstance(policy, ExpMapPolicy):
        raise TypeError("policy must be ExpMapPolicy")

    step_values = bounded_step_values(tangent=tangent, scale=checked_scale, policy=policy)
    mode = expmap_mode_for_point(point)
    if mode is ExpMapMode.UNIT_SPHERE_RENORMALIZE:
        return expmap_unit_sphere_values(point_values=point.values, step_values=step_values)
    return expmap_euclidean_values(point_values=point.values, step_values=step_values)


def apply_expmap(
    *,
    point: ManifoldPoint,
    tangent: TangentVector,
    scale: float,
    policy: ExpMapPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> ExpMapResult:
    if not isinstance(point, ManifoldPoint):
        raise TypeError("point must be ManifoldPoint")
    if not isinstance(tangent, TangentVector):
        raise TypeError("tangent must be TangentVector")
    if tangent.dimension != point.dimension:
        raise ValueError("tangent.dimension must equal point.dimension")

    active_policy = ExpMapPolicy() if policy is None else policy
    if not isinstance(active_policy, ExpMapPolicy):
        raise TypeError("policy must be ExpMapPolicy or None")
    checked_scale = ensure_constraint_probability(scale, "scale")
    next_values = compute_expmap_values(
        point=point,
        tangent=tangent,
        scale=checked_scale,
        policy=active_policy,
    )
    step_values = bounded_step_values(tangent=tangent, scale=checked_scale, policy=active_policy)
    mode = expmap_mode_for_point(point)
    base_point_match = tangent.base_point_id is None or tangent.base_point_id == point.point_id

    if tangent.norm == 0.0 and checked_scale == 0.0:
        status = ExpMapStatus.SKIPPED
    elif not base_point_match:
        status = ExpMapStatus.DEGRADED
    else:
        status = ExpMapStatus.APPLIED

    result_metadata: dict[str, object] = {
        "source": "expmap",
        "mode": mode.value,
        "previous_point_id": point.point_id,
        "tangent_id": tangent.vector_id,
        "base_point_match": base_point_match,
    }
    if metadata is not None:
        result_metadata.update(_copy_metadata(metadata))
    result_metadata["source"] = "expmap"

    next_point = create_manifold_point(
        point_id=next_point_id(previous_point_id=point.point_id),
        values=next_values,
        kind=point.kind,
        normalize=point.kind is ManifoldKind.UNIT_SPHERE and active_policy.renormalize_unit_sphere,
        metadata=result_metadata,
    )

    return ExpMapResult(
        previous_point=point,
        tangent=tangent,
        next_point=next_point,
        mode=mode,
        status=status,
        scale=checked_scale,
        step_norm=euclidean_norm(step_values),
        created_at=utc_now(),
        metadata=result_metadata,
    )


def apply_expmap_with_temperature(
    *,
    point: ManifoldPoint,
    tangent: TangentVector,
    temperature: SemanticTemperatureState,
    policy: ExpMapPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> ExpMapResult:
    if not isinstance(temperature, SemanticTemperatureState):
        raise TypeError("temperature must be SemanticTemperatureState")
    return apply_expmap(
        point=point,
        tangent=tangent,
        scale=transition_scale_from_temperature(temperature),
        policy=policy,
        metadata=metadata,
    )


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
    "ExpMapMode",
    "ExpMapStatus",
    "ExpMapPolicy",
    "ExpMapResult",
    "expmap_mode_for_point",
    "bounded_step_values",
    "expmap_euclidean_values",
    "expmap_unit_sphere_values",
    "next_point_id",
    "compute_expmap_values",
    "apply_expmap",
    "apply_expmap_with_temperature",
]
