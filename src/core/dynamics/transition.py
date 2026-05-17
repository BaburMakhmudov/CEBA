"""Deterministic semantic dynamics transition utilities for CEBA Core Dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib

from ceba.core.dynamics.energy import EnergyState
from ceba.core.dynamics.gradient_flow import (
    GradientFlowResult,
    GradientFlowStatus,
    gradient_flow_result_point,
)
from ceba.core.dynamics.manifold import ManifoldPoint, euclidean_norm, vector_subtract
from ceba.core.dynamics.semantic_temperature import (
    SemanticTemperatureState,
    transition_scale_from_temperature,
)
from ceba.core.dynamics.stability import (
    StabilityReport,
    assess_transition_readiness,
    stability_report_is_blocked,
    stability_report_is_passed,
)
from ceba.core.metrics.schema import (
    ensure_constraint_probability,
    ensure_non_negative_number,
    utc_now,
)


class DynamicsTransitionKind(Enum):
    GRADIENT_FLOW = "GRADIENT_FLOW"
    EXP_MAP = "EXP_MAP"
    TEMPERATURE_ONLY = "TEMPERATURE_ONLY"
    STABILITY_CHECK = "STABILITY_CHECK"
    NO_OP = "NO_OP"
    DEGRADED = "DEGRADED"


class DynamicsTransitionStatus(Enum):
    READY = "READY"
    APPLIED = "APPLIED"
    SKIPPED = "SKIPPED"
    BLOCKED = "BLOCKED"
    DEGRADED = "DEGRADED"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class DynamicsTransitionPolicy:
    allow_degraded_transition: bool = False
    allow_skipped_transition: bool = True
    max_displacement_norm: float = 1.0
    require_stability_pass: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.allow_degraded_transition, bool):
            raise TypeError("allow_degraded_transition must be bool")
        if not isinstance(self.allow_skipped_transition, bool):
            raise TypeError("allow_skipped_transition must be bool")
        object.__setattr__(
            self,
            "max_displacement_norm",
            ensure_non_negative_number(self.max_displacement_norm, "max_displacement_norm"),
        )
        if not isinstance(self.require_stability_pass, bool):
            raise TypeError("require_stability_pass must be bool")


@dataclass(frozen=True, slots=True)
class DynamicsTransition:
    transition_id: str
    kind: DynamicsTransitionKind
    status: DynamicsTransitionStatus
    previous_point: ManifoldPoint
    next_point: ManifoldPoint
    displacement_norm: float
    temperature_scale: float
    energy_value: float | None
    stability_score: float | None
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "transition_id", _ensure_non_empty_string(self.transition_id, "transition_id"))
        if not isinstance(self.kind, DynamicsTransitionKind):
            raise TypeError("kind must be DynamicsTransitionKind")
        if not isinstance(self.status, DynamicsTransitionStatus):
            raise TypeError("status must be DynamicsTransitionStatus")
        if not isinstance(self.previous_point, ManifoldPoint):
            raise TypeError("previous_point must be ManifoldPoint")
        if not isinstance(self.next_point, ManifoldPoint):
            raise TypeError("next_point must be ManifoldPoint")
        if self.previous_point.dimension != self.next_point.dimension:
            raise ValueError("previous_point.dimension must equal next_point.dimension")
        object.__setattr__(
            self,
            "displacement_norm",
            ensure_non_negative_number(self.displacement_norm, "displacement_norm"),
        )
        object.__setattr__(
            self,
            "temperature_scale",
            ensure_constraint_probability(self.temperature_scale, "temperature_scale"),
        )
        if self.energy_value is not None:
            object.__setattr__(self, "energy_value", ensure_non_negative_number(self.energy_value, "energy_value"))
        if self.stability_score is not None:
            object.__setattr__(
                self,
                "stability_score",
                ensure_constraint_probability(self.stability_score, "stability_score"),
            )
        object.__setattr__(self, "created_at", _ensure_timezone_aware(self.created_at, "created_at"))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "transition_id": self.transition_id,
            "kind": self.kind.value,
            "status": self.status.value,
            "previous_point": self.previous_point.to_dict(),
            "next_point": self.next_point.to_dict(),
            "displacement_norm": self.displacement_norm,
            "temperature_scale": self.temperature_scale,
            "energy_value": self.energy_value,
            "stability_score": self.stability_score,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class DynamicsTransitionResult:
    transition: DynamicsTransition
    gradient_result: GradientFlowResult | None
    energy_state: EnergyState | None
    temperature_state: SemanticTemperatureState | None
    stability_report: StabilityReport
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.transition, DynamicsTransition):
            raise TypeError("transition must be DynamicsTransition")
        if self.gradient_result is not None and not isinstance(self.gradient_result, GradientFlowResult):
            raise TypeError("gradient_result must be None or GradientFlowResult")
        if self.energy_state is not None and not isinstance(self.energy_state, EnergyState):
            raise TypeError("energy_state must be None or EnergyState")
        if self.temperature_state is not None and not isinstance(self.temperature_state, SemanticTemperatureState):
            raise TypeError("temperature_state must be None or SemanticTemperatureState")
        if not isinstance(self.stability_report, StabilityReport):
            raise TypeError("stability_report must be StabilityReport")
        object.__setattr__(self, "created_at", _ensure_timezone_aware(self.created_at, "created_at"))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    @property
    def applied(self) -> bool:
        return self.transition.status is DynamicsTransitionStatus.APPLIED

    @property
    def blocked(self) -> bool:
        return self.transition.status in (DynamicsTransitionStatus.BLOCKED, DynamicsTransitionStatus.INVALID)

    def to_dict(self) -> dict[str, object]:
        return {
            "transition": self.transition.to_dict(),
            "gradient_result": self.gradient_result.to_dict() if self.gradient_result is not None else None,
            "energy_state": self.energy_state.to_dict() if self.energy_state is not None else None,
            "temperature_state": self.temperature_state.to_dict() if self.temperature_state is not None else None,
            "stability_report": self.stability_report.to_dict(),
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
            "applied": self.applied,
            "blocked": self.blocked,
        }


def build_transition_id(
    *,
    previous_point: ManifoldPoint,
    next_point: ManifoldPoint,
    kind: DynamicsTransitionKind,
) -> str:
    if not isinstance(previous_point, ManifoldPoint):
        raise TypeError("previous_point must be ManifoldPoint")
    if not isinstance(next_point, ManifoldPoint):
        raise TypeError("next_point must be ManifoldPoint")
    if not isinstance(kind, DynamicsTransitionKind):
        raise TypeError("kind must be DynamicsTransitionKind")
    digest = hashlib.sha256()
    digest.update(previous_point.point_id.encode("utf-8"))
    digest.update(next_point.point_id.encode("utf-8"))
    digest.update(kind.value.encode("utf-8"))
    digest.update(repr(previous_point.values).encode("utf-8"))
    digest.update(repr(next_point.values).encode("utf-8"))
    return f"dyn_{digest.hexdigest()[:32]}"


def displacement_norm_between(
    *,
    previous_point: ManifoldPoint,
    next_point: ManifoldPoint,
) -> float:
    if not isinstance(previous_point, ManifoldPoint):
        raise TypeError("previous_point must be ManifoldPoint")
    if not isinstance(next_point, ManifoldPoint):
        raise TypeError("next_point must be ManifoldPoint")
    if previous_point.dimension != next_point.dimension:
        raise ValueError("previous_point.dimension must equal next_point.dimension")
    displacement = vector_subtract(next_point.values, previous_point.values)
    return ensure_non_negative_number(euclidean_norm(displacement), "displacement_norm")


def transition_kind_from_inputs(
    *,
    gradient_result: GradientFlowResult | None,
    temperature_state: SemanticTemperatureState | None,
) -> DynamicsTransitionKind:
    if gradient_result is not None:
        if not isinstance(gradient_result, GradientFlowResult):
            raise TypeError("gradient_result must be None or GradientFlowResult")
        if gradient_result.status is GradientFlowStatus.APPLIED:
            return DynamicsTransitionKind.GRADIENT_FLOW
        if gradient_result.status is GradientFlowStatus.DEGRADED:
            return DynamicsTransitionKind.DEGRADED
        if gradient_result.status is GradientFlowStatus.SKIPPED:
            return DynamicsTransitionKind.NO_OP
        if gradient_result.status in (GradientFlowStatus.BLOCKED, GradientFlowStatus.INVALID):
            return DynamicsTransitionKind.DEGRADED
    if temperature_state is not None:
        if not isinstance(temperature_state, SemanticTemperatureState):
            raise TypeError("temperature_state must be None or SemanticTemperatureState")
        return DynamicsTransitionKind.TEMPERATURE_ONLY
    return DynamicsTransitionKind.NO_OP


def transition_status_from_inputs(
    *,
    kind: DynamicsTransitionKind,
    gradient_result: GradientFlowResult | None,
    stability_report: StabilityReport,
    displacement_norm: float,
    policy: DynamicsTransitionPolicy,
) -> DynamicsTransitionStatus:
    if not isinstance(kind, DynamicsTransitionKind):
        raise TypeError("kind must be DynamicsTransitionKind")
    if gradient_result is not None and not isinstance(gradient_result, GradientFlowResult):
        raise TypeError("gradient_result must be None or GradientFlowResult")
    if not isinstance(stability_report, StabilityReport):
        raise TypeError("stability_report must be StabilityReport")
    checked_displacement = ensure_non_negative_number(displacement_norm, "displacement_norm")
    if not isinstance(policy, DynamicsTransitionPolicy):
        raise TypeError("policy must be DynamicsTransitionPolicy")

    if stability_report_is_blocked(stability_report):
        return DynamicsTransitionStatus.BLOCKED
    if policy.require_stability_pass and not stability_report_is_passed(stability_report):
        return DynamicsTransitionStatus.BLOCKED
    if checked_displacement > policy.max_displacement_norm:
        return DynamicsTransitionStatus.BLOCKED
    if kind is DynamicsTransitionKind.NO_OP:
        return DynamicsTransitionStatus.SKIPPED if policy.allow_skipped_transition else DynamicsTransitionStatus.BLOCKED
    if kind is DynamicsTransitionKind.DEGRADED:
        return (
            DynamicsTransitionStatus.DEGRADED
            if policy.allow_degraded_transition
            else DynamicsTransitionStatus.BLOCKED
        )
    if gradient_result is not None and gradient_result.status is GradientFlowStatus.DEGRADED:
        return (
            DynamicsTransitionStatus.DEGRADED
            if policy.allow_degraded_transition
            else DynamicsTransitionStatus.BLOCKED
        )
    return DynamicsTransitionStatus.APPLIED


def resolve_next_point(
    *,
    previous_point: ManifoldPoint,
    gradient_result: GradientFlowResult | None,
) -> ManifoldPoint:
    if not isinstance(previous_point, ManifoldPoint):
        raise TypeError("previous_point must be ManifoldPoint")
    if gradient_result is not None:
        if not isinstance(gradient_result, GradientFlowResult):
            raise TypeError("gradient_result must be None or GradientFlowResult")
        return gradient_flow_result_point(gradient_result)
    return previous_point


def create_dynamics_transition(
    *,
    previous_point: ManifoldPoint,
    next_point: ManifoldPoint,
    kind: DynamicsTransitionKind,
    status: DynamicsTransitionStatus,
    temperature_state: SemanticTemperatureState | None = None,
    energy_state: EnergyState | None = None,
    stability_report: StabilityReport | None = None,
    metadata: dict[str, object] | None = None,
) -> DynamicsTransition:
    if not isinstance(previous_point, ManifoldPoint):
        raise TypeError("previous_point must be ManifoldPoint")
    if not isinstance(next_point, ManifoldPoint):
        raise TypeError("next_point must be ManifoldPoint")
    if not isinstance(kind, DynamicsTransitionKind):
        raise TypeError("kind must be DynamicsTransitionKind")
    if not isinstance(status, DynamicsTransitionStatus):
        raise TypeError("status must be DynamicsTransitionStatus")
    if temperature_state is not None and not isinstance(temperature_state, SemanticTemperatureState):
        raise TypeError("temperature_state must be None or SemanticTemperatureState")
    if energy_state is not None and not isinstance(energy_state, EnergyState):
        raise TypeError("energy_state must be None or EnergyState")
    if stability_report is not None and not isinstance(stability_report, StabilityReport):
        raise TypeError("stability_report must be None or StabilityReport")

    transition_metadata: dict[str, object] = {
        "source": "dynamics_transition",
        "previous_point_id": previous_point.point_id,
        "next_point_id": next_point.point_id,
        "kind": kind.value,
        "status": status.value,
    }
    if metadata is not None:
        transition_metadata.update(_copy_metadata(metadata))
    transition_metadata["source"] = "dynamics_transition"

    return DynamicsTransition(
        transition_id=build_transition_id(previous_point=previous_point, next_point=next_point, kind=kind),
        kind=kind,
        status=status,
        previous_point=previous_point,
        next_point=next_point,
        displacement_norm=displacement_norm_between(previous_point=previous_point, next_point=next_point),
        temperature_scale=(
            transition_scale_from_temperature(temperature_state) if temperature_state is not None else 1.0
        ),
        energy_value=energy_state.energy if energy_state is not None else None,
        stability_score=stability_report.score if stability_report is not None else None,
        created_at=utc_now(),
        metadata=transition_metadata,
    )


def build_transition_result(
    *,
    previous_point: ManifoldPoint,
    gradient_result: GradientFlowResult | None = None,
    energy_state: EnergyState | None = None,
    temperature_state: SemanticTemperatureState | None = None,
    stability_report: StabilityReport | None = None,
    policy: DynamicsTransitionPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> DynamicsTransitionResult:
    if not isinstance(previous_point, ManifoldPoint):
        raise TypeError("previous_point must be ManifoldPoint")
    if gradient_result is not None and not isinstance(gradient_result, GradientFlowResult):
        raise TypeError("gradient_result must be None or GradientFlowResult")
    if energy_state is not None and not isinstance(energy_state, EnergyState):
        raise TypeError("energy_state must be None or EnergyState")
    if temperature_state is not None and not isinstance(temperature_state, SemanticTemperatureState):
        raise TypeError("temperature_state must be None or SemanticTemperatureState")
    if stability_report is not None and not isinstance(stability_report, StabilityReport):
        raise TypeError("stability_report must be None or StabilityReport")
    active_policy = DynamicsTransitionPolicy() if policy is None else policy
    if not isinstance(active_policy, DynamicsTransitionPolicy):
        raise TypeError("policy must be DynamicsTransitionPolicy or None")

    next_point = resolve_next_point(previous_point=previous_point, gradient_result=gradient_result)
    active_report = stability_report
    if active_report is None:
        active_report = assess_transition_readiness(
            point=next_point,
            energy=energy_state,
            temperature=temperature_state,
            gradient_result=gradient_result,
        )
    kind = transition_kind_from_inputs(gradient_result=gradient_result, temperature_state=temperature_state)
    displacement_norm = displacement_norm_between(previous_point=previous_point, next_point=next_point)
    status = transition_status_from_inputs(
        kind=kind,
        gradient_result=gradient_result,
        stability_report=active_report,
        displacement_norm=displacement_norm,
        policy=active_policy,
    )

    result_metadata: dict[str, object] = {
        "source": "dynamics_transition_result",
        "has_gradient_result": gradient_result is not None,
        "has_energy_state": energy_state is not None,
        "has_temperature_state": temperature_state is not None,
    }
    if metadata is not None:
        result_metadata.update(_copy_metadata(metadata))
    result_metadata["source"] = "dynamics_transition_result"

    transition = create_dynamics_transition(
        previous_point=previous_point,
        next_point=next_point,
        kind=kind,
        status=status,
        temperature_state=temperature_state,
        energy_state=energy_state,
        stability_report=active_report,
        metadata=result_metadata,
    )

    return DynamicsTransitionResult(
        transition=transition,
        gradient_result=gradient_result,
        energy_state=energy_state,
        temperature_state=temperature_state,
        stability_report=active_report,
        created_at=utc_now(),
        metadata=result_metadata,
    )


def transition_result_point(
    result: DynamicsTransitionResult,
) -> ManifoldPoint:
    if not isinstance(result, DynamicsTransitionResult):
        raise TypeError("result must be DynamicsTransitionResult")
    return result.transition.next_point


def transition_is_applied(result: DynamicsTransitionResult) -> bool:
    if not isinstance(result, DynamicsTransitionResult):
        raise TypeError("result must be DynamicsTransitionResult")
    return result.applied


def transition_is_blocked(result: DynamicsTransitionResult) -> bool:
    if not isinstance(result, DynamicsTransitionResult):
        raise TypeError("result must be DynamicsTransitionResult")
    return result.blocked


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
    "DynamicsTransitionKind",
    "DynamicsTransitionStatus",
    "DynamicsTransitionPolicy",
    "DynamicsTransition",
    "DynamicsTransitionResult",
    "build_transition_id",
    "displacement_norm_between",
    "transition_kind_from_inputs",
    "transition_status_from_inputs",
    "resolve_next_point",
    "create_dynamics_transition",
    "build_transition_result",
    "transition_result_point",
    "transition_is_applied",
    "transition_is_blocked",
]
