"""Deterministic lightweight gradient-flow utilities for CEBA Core Dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ceba.core.dynamics.energy import EnergyResult, EnergyState, EnergyStatus, energy_is_stable
from ceba.core.dynamics.expmap import ExpMapPolicy, ExpMapResult, apply_expmap
from ceba.core.dynamics.manifold import (
    ManifoldPoint,
    TangentVector,
    create_tangent_vector,
    normalize_vector,
    scalar_multiply,
)
from ceba.core.dynamics.semantic_temperature import (
    SemanticTemperatureState,
    semantic_temperature_is_frozen,
    transition_scale_from_temperature,
)
from ceba.core.metrics.schema import (
    ensure_constraint_probability,
    ensure_non_negative_number,
    utc_now,
)


class GradientFlowMode(Enum):
    DESCENT = "DESCENT"
    ASCENT_BLOCKED = "ASCENT_BLOCKED"
    ZERO = "ZERO"
    TEMPERATURE_SCALED = "TEMPERATURE_SCALED"
    ENERGY_STABLE = "ENERGY_STABLE"


class GradientFlowStatus(Enum):
    READY = "READY"
    APPLIED = "APPLIED"
    SKIPPED = "SKIPPED"
    BLOCKED = "BLOCKED"
    DEGRADED = "DEGRADED"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class GradientFlowPolicy:
    base_step_size: float = 0.1
    max_step_size: float = 1.0
    min_energy_threshold: float = 0.0
    allow_zero_gradient: bool = True
    normalize_gradient: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_step_size",
            ensure_non_negative_number(self.base_step_size, "base_step_size"),
        )
        object.__setattr__(
            self,
            "max_step_size",
            ensure_non_negative_number(self.max_step_size, "max_step_size"),
        )
        if self.max_step_size < self.base_step_size:
            raise ValueError("max_step_size must be greater than or equal to base_step_size")
        object.__setattr__(
            self,
            "min_energy_threshold",
            ensure_non_negative_number(self.min_energy_threshold, "min_energy_threshold"),
        )
        if not isinstance(self.allow_zero_gradient, bool):
            raise TypeError("allow_zero_gradient must be bool")
        if not isinstance(self.normalize_gradient, bool):
            raise TypeError("normalize_gradient must be bool")


@dataclass(frozen=True, slots=True)
class GradientFlowStep:
    source_point: ManifoldPoint
    gradient: TangentVector
    descent_tangent: TangentVector
    energy_value: float
    temperature_scale: float
    step_size: float
    mode: GradientFlowMode
    status: GradientFlowStatus
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source_point, ManifoldPoint):
            raise TypeError("source_point must be ManifoldPoint")
        if not isinstance(self.gradient, TangentVector):
            raise TypeError("gradient must be TangentVector")
        if not isinstance(self.descent_tangent, TangentVector):
            raise TypeError("descent_tangent must be TangentVector")
        object.__setattr__(self, "energy_value", ensure_non_negative_number(self.energy_value, "energy_value"))
        object.__setattr__(
            self,
            "temperature_scale",
            ensure_constraint_probability(self.temperature_scale, "temperature_scale"),
        )
        object.__setattr__(self, "step_size", ensure_non_negative_number(self.step_size, "step_size"))
        if not isinstance(self.mode, GradientFlowMode):
            raise TypeError("mode must be GradientFlowMode")
        if not isinstance(self.status, GradientFlowStatus):
            raise TypeError("status must be GradientFlowStatus")
        object.__setattr__(self, "created_at", _ensure_timezone_aware(self.created_at, "created_at"))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "source_point": self.source_point.to_dict(),
            "gradient": self.gradient.to_dict(),
            "descent_tangent": self.descent_tangent.to_dict(),
            "energy_value": self.energy_value,
            "temperature_scale": self.temperature_scale,
            "step_size": self.step_size,
            "mode": self.mode.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class GradientFlowResult:
    step: GradientFlowStep
    expmap_result: ExpMapResult | None
    status: GradientFlowStatus
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.step, GradientFlowStep):
            raise TypeError("step must be GradientFlowStep")
        if self.expmap_result is not None and not isinstance(self.expmap_result, ExpMapResult):
            raise TypeError("expmap_result must be None or ExpMapResult")
        if not isinstance(self.status, GradientFlowStatus):
            raise TypeError("status must be GradientFlowStatus")
        object.__setattr__(self, "created_at", _ensure_timezone_aware(self.created_at, "created_at"))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step.to_dict(),
            "expmap_result": self.expmap_result.to_dict() if self.expmap_result is not None else None,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


def compute_temperature_scale(
    temperature: SemanticTemperatureState | None,
) -> float:
    if temperature is None:
        return 1.0
    if not isinstance(temperature, SemanticTemperatureState):
        raise TypeError("temperature must be SemanticTemperatureState or None")
    return ensure_constraint_probability(transition_scale_from_temperature(temperature), "temperature_scale")


def compute_step_size(
    *,
    energy_value: float,
    temperature_scale: float,
    policy: GradientFlowPolicy,
) -> float:
    checked_energy = ensure_non_negative_number(energy_value, "energy_value")
    checked_scale = ensure_constraint_probability(temperature_scale, "temperature_scale")
    if not isinstance(policy, GradientFlowPolicy):
        raise TypeError("policy must be GradientFlowPolicy")
    raw_step = ensure_non_negative_number(
        policy.base_step_size * (1.0 + checked_energy) * checked_scale,
        "raw_step",
    )
    return ensure_non_negative_number(min(policy.max_step_size, max(0.0, raw_step)), "step_size")


def descent_values_from_gradient(
    *,
    gradient: TangentVector,
    policy: GradientFlowPolicy,
) -> tuple[float, ...]:
    if not isinstance(gradient, TangentVector):
        raise TypeError("gradient must be TangentVector")
    if not isinstance(policy, GradientFlowPolicy):
        raise TypeError("policy must be GradientFlowPolicy")
    if gradient.norm == 0.0:
        if policy.allow_zero_gradient:
            return tuple(0.0 for _ in gradient.values)
        raise ValueError("zero gradient is not allowed")
    if policy.normalize_gradient:
        return scalar_multiply(-1.0, normalize_vector(gradient.values))
    return scalar_multiply(-1.0, gradient.values)


def flow_mode_from_inputs(
    *,
    energy: EnergyResult | EnergyState,
    temperature: SemanticTemperatureState | None,
    gradient: TangentVector,
) -> GradientFlowMode:
    _ensure_energy_source(energy)
    if not isinstance(gradient, TangentVector):
        raise TypeError("gradient must be TangentVector")
    if temperature is not None:
        if not isinstance(temperature, SemanticTemperatureState):
            raise TypeError("temperature must be SemanticTemperatureState or None")
        if semantic_temperature_is_frozen(temperature):
            return GradientFlowMode.ZERO
    if _energy_source_is_stable(energy):
        return GradientFlowMode.ENERGY_STABLE
    if gradient.norm == 0.0:
        return GradientFlowMode.ZERO
    return GradientFlowMode.TEMPERATURE_SCALED


def flow_status_from_mode(
    mode: GradientFlowMode,
) -> GradientFlowStatus:
    if not isinstance(mode, GradientFlowMode):
        raise TypeError("mode must be GradientFlowMode")
    if mode is GradientFlowMode.ZERO:
        return GradientFlowStatus.SKIPPED
    if mode is GradientFlowMode.ENERGY_STABLE:
        return GradientFlowStatus.SKIPPED
    if mode is GradientFlowMode.ASCENT_BLOCKED:
        return GradientFlowStatus.BLOCKED
    return GradientFlowStatus.READY


def energy_value_of(
    energy: EnergyResult | EnergyState,
) -> float:
    if isinstance(energy, EnergyResult):
        return ensure_non_negative_number(energy.value, "energy_value")
    if isinstance(energy, EnergyState):
        return ensure_non_negative_number(energy.energy, "energy_value")
    raise TypeError("energy must be EnergyResult or EnergyState")


def build_gradient_flow_step(
    *,
    source_point: ManifoldPoint,
    gradient: TangentVector,
    energy: EnergyResult | EnergyState,
    temperature: SemanticTemperatureState | None = None,
    policy: GradientFlowPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> GradientFlowStep:
    if not isinstance(source_point, ManifoldPoint):
        raise TypeError("source_point must be ManifoldPoint")
    if not isinstance(gradient, TangentVector):
        raise TypeError("gradient must be TangentVector")
    if gradient.dimension != source_point.dimension:
        raise ValueError("gradient.dimension must equal source_point.dimension")
    _ensure_energy_source(energy)
    if temperature is not None and not isinstance(temperature, SemanticTemperatureState):
        raise TypeError("temperature must be SemanticTemperatureState or None")
    active_policy = GradientFlowPolicy() if policy is None else policy
    if not isinstance(active_policy, GradientFlowPolicy):
        raise TypeError("policy must be GradientFlowPolicy or None")

    value = energy_value_of(energy)
    temperature_scale = compute_temperature_scale(temperature)
    mode = flow_mode_from_inputs(energy=energy, temperature=temperature, gradient=gradient)
    status = flow_status_from_mode(mode)
    base_point_match = gradient.base_point_id is None or gradient.base_point_id == source_point.point_id
    if status not in (GradientFlowStatus.SKIPPED, GradientFlowStatus.BLOCKED) and not base_point_match:
        status = GradientFlowStatus.DEGRADED
    if value <= active_policy.min_energy_threshold and status is GradientFlowStatus.READY:
        status = GradientFlowStatus.SKIPPED

    step_size = compute_step_size(
        energy_value=value,
        temperature_scale=temperature_scale,
        policy=active_policy,
    )
    descent_values = descent_values_from_gradient(gradient=gradient, policy=active_policy)
    descent_step_values = scalar_multiply(step_size, descent_values)

    step_metadata: dict[str, object] = {
        "source": "gradient_flow",
        "gradient_id": gradient.vector_id,
        "energy_value": value,
        "temperature_scale": temperature_scale,
        "step_size": step_size,
        "base_point_match": base_point_match,
    }
    if metadata is not None:
        step_metadata.update(_copy_metadata(metadata))
    step_metadata["source"] = "gradient_flow"

    descent_tangent = create_tangent_vector(
        vector_id=f"{gradient.vector_id}:descent",
        values=descent_step_values,
        base_point_id=source_point.point_id,
        metadata=step_metadata,
    )

    return GradientFlowStep(
        source_point=source_point,
        gradient=gradient,
        descent_tangent=descent_tangent,
        energy_value=value,
        temperature_scale=temperature_scale,
        step_size=step_size,
        mode=mode,
        status=status,
        created_at=utc_now(),
        metadata=step_metadata,
    )


def apply_gradient_flow_step(
    *,
    step: GradientFlowStep,
    expmap_policy: ExpMapPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> GradientFlowResult:
    if not isinstance(step, GradientFlowStep):
        raise TypeError("step must be GradientFlowStep")

    result_metadata: dict[str, object] = {
        "source": "gradient_flow_apply",
        "step_status": step.status.value,
    }
    if metadata is not None:
        result_metadata.update(_copy_metadata(metadata))
    result_metadata["source"] = "gradient_flow_apply"

    if step.status in (GradientFlowStatus.SKIPPED, GradientFlowStatus.BLOCKED):
        return GradientFlowResult(
            step=step,
            expmap_result=None,
            status=step.status,
            created_at=utc_now(),
            metadata=result_metadata,
        )

    expmap_result = apply_expmap(
        point=step.source_point,
        tangent=step.descent_tangent,
        scale=1.0,
        policy=expmap_policy,
        metadata=result_metadata,
    )
    final_status = (
        GradientFlowStatus.DEGRADED
        if expmap_result.status.value == "DEGRADED"
        else GradientFlowStatus.APPLIED
    )
    return GradientFlowResult(
        step=step,
        expmap_result=expmap_result,
        status=final_status,
        created_at=utc_now(),
        metadata=result_metadata,
    )


def run_gradient_flow_step(
    *,
    source_point: ManifoldPoint,
    gradient: TangentVector,
    energy: EnergyResult | EnergyState,
    temperature: SemanticTemperatureState | None = None,
    policy: GradientFlowPolicy | None = None,
    expmap_policy: ExpMapPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> GradientFlowResult:
    step = build_gradient_flow_step(
        source_point=source_point,
        gradient=gradient,
        energy=energy,
        temperature=temperature,
        policy=policy,
        metadata=metadata,
    )
    return apply_gradient_flow_step(step=step, expmap_policy=expmap_policy, metadata=metadata)


def gradient_flow_result_point(
    result: GradientFlowResult,
) -> ManifoldPoint:
    if not isinstance(result, GradientFlowResult):
        raise TypeError("result must be GradientFlowResult")
    if result.expmap_result is not None:
        return result.expmap_result.next_point
    return result.step.source_point


def _ensure_energy_source(energy: EnergyResult | EnergyState) -> EnergyResult | EnergyState:
    if not isinstance(energy, (EnergyResult, EnergyState)):
        raise TypeError("energy must be EnergyResult or EnergyState")
    return energy


def _energy_source_is_stable(energy: EnergyResult | EnergyState) -> bool:
    if isinstance(energy, EnergyState):
        return energy_is_stable(energy)
    if isinstance(energy, EnergyResult):
        return energy.status in (EnergyStatus.STABLE, EnergyStatus.FROZEN)
    raise TypeError("energy must be EnergyResult or EnergyState")


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
    "GradientFlowMode",
    "GradientFlowStatus",
    "GradientFlowPolicy",
    "GradientFlowStep",
    "GradientFlowResult",
    "compute_temperature_scale",
    "compute_step_size",
    "descent_values_from_gradient",
    "flow_mode_from_inputs",
    "flow_status_from_mode",
    "energy_value_of",
    "build_gradient_flow_step",
    "apply_gradient_flow_step",
    "run_gradient_flow_step",
    "gradient_flow_result_point",
]
