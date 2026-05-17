"""Deterministic semantic temperature utilities for CEBA Core Dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ceba.core.metrics.schema import (
    ensure_constraint_probability,
    ensure_finite_number,
    ensure_non_negative_number,
    utc_now,
)
from ceba.core.metrics.update import MetricUpdatePlan, MetricUpdatePlanStatus


class SemanticTemperatureMode(Enum):
    ACTIVE = "ACTIVE"
    COOLING = "COOLING"
    FROZEN = "FROZEN"
    RESET = "RESET"
    DEGRADED = "DEGRADED"


class SemanticTemperatureStatus(Enum):
    STABLE = "STABLE"
    UNSTABLE = "UNSTABLE"
    COOLING = "COOLING"
    FROZEN = "FROZEN"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class SemanticTemperaturePolicy:
    initial_temperature: float = 1.0
    min_temperature: float = 0.0
    max_temperature: float = 10.0
    cooling_rate: float = 0.05
    heating_rate: float = 0.10
    freeze_threshold: float = 0.01
    instability_threshold: float = 0.75

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "initial_temperature",
            ensure_non_negative_number(self.initial_temperature, "initial_temperature"),
        )
        object.__setattr__(
            self,
            "min_temperature",
            ensure_non_negative_number(self.min_temperature, "min_temperature"),
        )
        object.__setattr__(
            self,
            "max_temperature",
            ensure_non_negative_number(self.max_temperature, "max_temperature"),
        )
        if self.max_temperature < self.min_temperature:
            raise ValueError("max_temperature must be greater than or equal to min_temperature")
        if not self.min_temperature <= self.initial_temperature <= self.max_temperature:
            raise ValueError("initial_temperature must be between min_temperature and max_temperature")
        object.__setattr__(self, "cooling_rate", ensure_non_negative_number(self.cooling_rate, "cooling_rate"))
        object.__setattr__(self, "heating_rate", ensure_non_negative_number(self.heating_rate, "heating_rate"))
        object.__setattr__(
            self,
            "freeze_threshold",
            ensure_non_negative_number(self.freeze_threshold, "freeze_threshold"),
        )
        object.__setattr__(
            self,
            "instability_threshold",
            ensure_constraint_probability(self.instability_threshold, "instability_threshold"),
        )


@dataclass(frozen=True, slots=True)
class SemanticTemperatureState:
    temperature: float
    mode: SemanticTemperatureMode
    status: SemanticTemperatureStatus
    update_count: int
    last_pressure: float
    last_cost: float
    stability_score: float
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "temperature", ensure_non_negative_number(self.temperature, "temperature"))
        if not isinstance(self.mode, SemanticTemperatureMode):
            raise TypeError("mode must be SemanticTemperatureMode")
        if not isinstance(self.status, SemanticTemperatureStatus):
            raise TypeError("status must be SemanticTemperatureStatus")
        object.__setattr__(self, "update_count", _ensure_non_negative_int(self.update_count, "update_count"))
        object.__setattr__(self, "last_pressure", ensure_non_negative_number(self.last_pressure, "last_pressure"))
        object.__setattr__(self, "last_cost", ensure_non_negative_number(self.last_cost, "last_cost"))
        object.__setattr__(
            self,
            "stability_score",
            ensure_constraint_probability(self.stability_score, "stability_score"),
        )
        object.__setattr__(self, "created_at", _ensure_timezone_aware(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", _ensure_timezone_aware(self.updated_at, "updated_at"))
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be greater than or equal to created_at")
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "temperature": self.temperature,
            "mode": self.mode.value,
            "status": self.status.value,
            "update_count": self.update_count,
            "last_pressure": self.last_pressure,
            "last_cost": self.last_cost,
            "stability_score": self.stability_score,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class SemanticCoolingResult:
    previous_temperature: float
    next_temperature: float
    delta: float
    mode: SemanticTemperatureMode
    status: SemanticTemperatureStatus
    applied_pressure: float
    applied_cost: float
    stability_score: float
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "previous_temperature",
            ensure_non_negative_number(self.previous_temperature, "previous_temperature"),
        )
        object.__setattr__(
            self,
            "next_temperature",
            ensure_non_negative_number(self.next_temperature, "next_temperature"),
        )
        object.__setattr__(self, "delta", ensure_finite_number(self.delta, "delta"))
        if not isinstance(self.mode, SemanticTemperatureMode):
            raise TypeError("mode must be SemanticTemperatureMode")
        if not isinstance(self.status, SemanticTemperatureStatus):
            raise TypeError("status must be SemanticTemperatureStatus")
        object.__setattr__(
            self,
            "applied_pressure",
            ensure_non_negative_number(self.applied_pressure, "applied_pressure"),
        )
        object.__setattr__(self, "applied_cost", ensure_non_negative_number(self.applied_cost, "applied_cost"))
        object.__setattr__(
            self,
            "stability_score",
            ensure_constraint_probability(self.stability_score, "stability_score"),
        )
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "previous_temperature": self.previous_temperature,
            "next_temperature": self.next_temperature,
            "delta": self.delta,
            "mode": self.mode.value,
            "status": self.status.value,
            "applied_pressure": self.applied_pressure,
            "applied_cost": self.applied_cost,
            "stability_score": self.stability_score,
            "metadata": dict(self.metadata),
        }


def clamp_temperature(
    *,
    value: float,
    policy: SemanticTemperaturePolicy,
) -> float:
    number = ensure_finite_number(value, "value")
    if not isinstance(policy, SemanticTemperaturePolicy):
        raise TypeError("policy must be SemanticTemperaturePolicy")
    return ensure_non_negative_number(
        min(policy.max_temperature, max(policy.min_temperature, number)),
        "temperature",
    )


def status_from_temperature(
    *,
    temperature: float,
    stability_score: float,
    policy: SemanticTemperaturePolicy,
) -> SemanticTemperatureStatus:
    checked_temperature = ensure_non_negative_number(temperature, "temperature")
    checked_stability = ensure_constraint_probability(stability_score, "stability_score")
    if not isinstance(policy, SemanticTemperaturePolicy):
        raise TypeError("policy must be SemanticTemperaturePolicy")
    if checked_temperature <= policy.freeze_threshold:
        return SemanticTemperatureStatus.FROZEN
    if checked_stability < policy.instability_threshold:
        return SemanticTemperatureStatus.UNSTABLE
    if checked_temperature > policy.freeze_threshold:
        return SemanticTemperatureStatus.STABLE
    return SemanticTemperatureStatus.INVALID


def mode_from_update(
    *,
    previous_temperature: float,
    next_temperature: float,
    status: SemanticTemperatureStatus,
) -> SemanticTemperatureMode:
    previous = ensure_non_negative_number(previous_temperature, "previous_temperature")
    next_value = ensure_non_negative_number(next_temperature, "next_temperature")
    if not isinstance(status, SemanticTemperatureStatus):
        raise TypeError("status must be SemanticTemperatureStatus")
    if status is SemanticTemperatureStatus.FROZEN:
        return SemanticTemperatureMode.FROZEN
    if next_value < previous:
        return SemanticTemperatureMode.COOLING
    if next_value > previous:
        return SemanticTemperatureMode.ACTIVE
    return SemanticTemperatureMode.ACTIVE


def create_temperature_state(
    *,
    policy: SemanticTemperaturePolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> SemanticTemperatureState:
    active_policy = _ensure_policy(policy)
    timestamp = utc_now()
    state_metadata: dict[str, object] = {"state_kind": "semantic_temperature"}
    if metadata is not None:
        state_metadata.update(_copy_metadata(metadata))
    state_metadata["state_kind"] = "semantic_temperature"

    return SemanticTemperatureState(
        temperature=active_policy.initial_temperature,
        mode=SemanticTemperatureMode.ACTIVE,
        status=status_from_temperature(
            temperature=active_policy.initial_temperature,
            stability_score=1.0,
            policy=active_policy,
        ),
        update_count=0,
        last_pressure=0.0,
        last_cost=0.0,
        stability_score=1.0,
        created_at=timestamp,
        updated_at=timestamp,
        metadata=state_metadata,
    )


def pressure_from_update_plan(plan: MetricUpdatePlan) -> float:
    if not isinstance(plan, MetricUpdatePlan):
        raise TypeError("plan must be MetricUpdatePlan")
    return ensure_non_negative_number(plan.total_pressure, "total_pressure")


def cost_from_update_plan(plan: MetricUpdatePlan) -> float:
    if not isinstance(plan, MetricUpdatePlan):
        raise TypeError("plan must be MetricUpdatePlan")
    return ensure_non_negative_number(plan.total_cost_value, "total_cost_value")


def stability_from_pressure_cost(
    *,
    pressure: float,
    cost: float,
) -> float:
    checked_pressure = ensure_non_negative_number(pressure, "pressure")
    checked_cost = ensure_non_negative_number(cost, "cost")
    instability = ensure_non_negative_number(checked_pressure + checked_cost, "instability")
    return ensure_constraint_probability(1.0 / (1.0 + instability), "stability")


def compute_temperature_update(
    *,
    state: SemanticTemperatureState,
    plan: MetricUpdatePlan,
    policy: SemanticTemperaturePolicy | None = None,
) -> SemanticCoolingResult:
    if not isinstance(state, SemanticTemperatureState):
        raise TypeError("state must be SemanticTemperatureState")
    if not isinstance(plan, MetricUpdatePlan):
        raise TypeError("plan must be MetricUpdatePlan")
    active_policy = _ensure_policy(policy)
    pressure = pressure_from_update_plan(plan)
    cost = cost_from_update_plan(plan)
    stability_score = stability_from_pressure_cost(pressure=pressure, cost=cost)

    if plan.status is MetricUpdatePlanStatus.READY:
        heating = active_policy.heating_rate * pressure * (1.0 - stability_score)
        cooling = active_policy.cooling_rate * stability_score
        raw_next = state.temperature + heating - cooling
    elif plan.status is MetricUpdatePlanStatus.SKIPPED:
        raw_next = state.temperature - active_policy.cooling_rate
    elif plan.status is MetricUpdatePlanStatus.BLOCKED:
        raw_next = state.temperature + active_policy.heating_rate
    else:
        raise ValueError("unsupported metric update plan status")

    next_temperature = clamp_temperature(value=raw_next, policy=active_policy)
    status = status_from_temperature(
        temperature=next_temperature,
        stability_score=stability_score,
        policy=active_policy,
    )
    mode = mode_from_update(
        previous_temperature=state.temperature,
        next_temperature=next_temperature,
        status=status,
    )
    delta = ensure_finite_number(next_temperature - state.temperature, "delta")

    return SemanticCoolingResult(
        previous_temperature=state.temperature,
        next_temperature=next_temperature,
        delta=delta,
        mode=mode,
        status=status,
        applied_pressure=pressure,
        applied_cost=cost,
        stability_score=stability_score,
        metadata={
            "source": "metric_update_plan",
            "plan_status": plan.status.value,
            "plan_mode": plan.mode.value,
            "plan_reason": plan.reason.value,
        },
    )


def apply_temperature_update(
    *,
    state: SemanticTemperatureState,
    result: SemanticCoolingResult,
    metadata: dict[str, object] | None = None,
) -> SemanticTemperatureState:
    if not isinstance(state, SemanticTemperatureState):
        raise TypeError("state must be SemanticTemperatureState")
    if not isinstance(result, SemanticCoolingResult):
        raise TypeError("result must be SemanticCoolingResult")

    state_metadata: dict[str, object] = {
        "previous_temperature": result.previous_temperature,
        "delta": result.delta,
    }
    if metadata is not None:
        state_metadata.update(_copy_metadata(metadata))

    return SemanticTemperatureState(
        temperature=result.next_temperature,
        mode=result.mode,
        status=result.status,
        update_count=state.update_count + 1,
        last_pressure=result.applied_pressure,
        last_cost=result.applied_cost,
        stability_score=result.stability_score,
        created_at=state.created_at,
        updated_at=utc_now(),
        metadata=state_metadata,
    )


def semantic_temperature_is_frozen(state: SemanticTemperatureState) -> bool:
    if not isinstance(state, SemanticTemperatureState):
        raise TypeError("state must be SemanticTemperatureState")
    return state.status is SemanticTemperatureStatus.FROZEN or state.mode is SemanticTemperatureMode.FROZEN


def semantic_temperature_is_stable(state: SemanticTemperatureState) -> bool:
    if not isinstance(state, SemanticTemperatureState):
        raise TypeError("state must be SemanticTemperatureState")
    return state.status is SemanticTemperatureStatus.STABLE


def transition_scale_from_temperature(
    state: SemanticTemperatureState,
    *,
    policy: SemanticTemperaturePolicy | None = None,
) -> float:
    if not isinstance(state, SemanticTemperatureState):
        raise TypeError("state must be SemanticTemperatureState")
    active_policy = _ensure_policy(policy)
    if active_policy.max_temperature == 0.0:
        return 0.0
    scale = ensure_finite_number(state.temperature / active_policy.max_temperature, "transition_scale")
    if scale < 0.0:
        return 0.0
    if scale > 1.0:
        return 1.0
    return ensure_constraint_probability(scale, "transition_scale")


def _ensure_policy(policy: SemanticTemperaturePolicy | None) -> SemanticTemperaturePolicy:
    if policy is None:
        return SemanticTemperaturePolicy()
    if not isinstance(policy, SemanticTemperaturePolicy):
        raise TypeError("policy must be SemanticTemperaturePolicy or None")
    return policy


def _ensure_non_negative_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
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
    "SemanticTemperatureMode",
    "SemanticTemperatureStatus",
    "SemanticTemperaturePolicy",
    "SemanticTemperatureState",
    "SemanticCoolingResult",
    "clamp_temperature",
    "status_from_temperature",
    "mode_from_update",
    "create_temperature_state",
    "pressure_from_update_plan",
    "cost_from_update_plan",
    "stability_from_pressure_cost",
    "compute_temperature_update",
    "apply_temperature_update",
    "semantic_temperature_is_frozen",
    "semantic_temperature_is_stable",
    "transition_scale_from_temperature",
]
