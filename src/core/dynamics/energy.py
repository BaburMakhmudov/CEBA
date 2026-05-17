"""Deterministic scalar energy utilities for CEBA Core Dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ceba.core.dynamics.semantic_temperature import (
    SemanticTemperatureState,
    semantic_temperature_is_frozen,
    transition_scale_from_temperature,
)
from ceba.core.metrics.cost import cost_summary, total_cost
from ceba.core.metrics.schema import (
    MetricCostResult,
    ensure_constraint_probability,
    ensure_finite_number,
    ensure_non_negative_number,
    utc_now,
)
from ceba.core.metrics.update import MetricUpdatePlan, MetricUpdatePlanStatus


class EnergyKind(Enum):
    TOTAL_COST = "TOTAL_COST"
    NORMALIZED_COST = "NORMALIZED_COST"
    TRANSITION_PRESSURE = "TRANSITION_PRESSURE"
    TEMPERATURE_MODULATED = "TEMPERATURE_MODULATED"
    STABILITY_ENERGY = "STABILITY_ENERGY"
    UNKNOWN = "UNKNOWN"


class EnergyStatus(Enum):
    ACTIVE = "ACTIVE"
    STABLE = "STABLE"
    FROZEN = "FROZEN"
    DEGRADED = "DEGRADED"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class EnergyPolicy:
    cost_weight: float = 1.0
    pressure_weight: float = 1.0
    temperature_weight: float = 1.0
    stability_weight: float = 1.0
    normalize_energy: bool = True
    max_energy: float = 10.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "cost_weight", ensure_non_negative_number(self.cost_weight, "cost_weight"))
        object.__setattr__(
            self,
            "pressure_weight",
            ensure_non_negative_number(self.pressure_weight, "pressure_weight"),
        )
        object.__setattr__(
            self,
            "temperature_weight",
            ensure_non_negative_number(self.temperature_weight, "temperature_weight"),
        )
        object.__setattr__(
            self,
            "stability_weight",
            ensure_non_negative_number(self.stability_weight, "stability_weight"),
        )
        if not isinstance(self.normalize_energy, bool):
            raise TypeError("normalize_energy must be bool")
        object.__setattr__(self, "max_energy", _ensure_positive_finite_number(self.max_energy, "max_energy"))


@dataclass(frozen=True, slots=True)
class EnergyResult:
    kind: EnergyKind
    status: EnergyStatus
    value: float
    normalized_value: float | None
    confidence: float
    source_cost_ids: tuple[str, ...]
    source_plan_status: str | None
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EnergyKind):
            raise TypeError("kind must be EnergyKind")
        if not isinstance(self.status, EnergyStatus):
            raise TypeError("status must be EnergyStatus")
        object.__setattr__(self, "value", ensure_non_negative_number(self.value, "value"))
        if self.normalized_value is not None:
            object.__setattr__(
                self,
                "normalized_value",
                ensure_constraint_probability(self.normalized_value, "normalized_value"),
            )
        object.__setattr__(self, "confidence", ensure_constraint_probability(self.confidence, "confidence"))
        object.__setattr__(self, "source_cost_ids", _ensure_id_tuple(self.source_cost_ids, "source_cost_ids"))
        if self.source_plan_status is not None:
            object.__setattr__(
                self,
                "source_plan_status",
                _ensure_non_empty_string(self.source_plan_status, "source_plan_status"),
            )
        object.__setattr__(self, "created_at", _ensure_timezone_aware(self.created_at, "created_at"))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "status": self.status.value,
            "value": self.value,
            "normalized_value": self.normalized_value,
            "confidence": self.confidence,
            "source_cost_ids": self.source_cost_ids,
            "source_plan_status": self.source_plan_status,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class EnergyState:
    energy: float
    normalized_energy: float | None
    status: EnergyStatus
    update_count: int
    last_pressure: float
    last_temperature_scale: float
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "energy", ensure_non_negative_number(self.energy, "energy"))
        if self.normalized_energy is not None:
            object.__setattr__(
                self,
                "normalized_energy",
                ensure_constraint_probability(self.normalized_energy, "normalized_energy"),
            )
        if not isinstance(self.status, EnergyStatus):
            raise TypeError("status must be EnergyStatus")
        object.__setattr__(self, "update_count", _ensure_non_negative_int(self.update_count, "update_count"))
        object.__setattr__(self, "last_pressure", ensure_non_negative_number(self.last_pressure, "last_pressure"))
        object.__setattr__(
            self,
            "last_temperature_scale",
            ensure_constraint_probability(self.last_temperature_scale, "last_temperature_scale"),
        )
        object.__setattr__(self, "created_at", _ensure_timezone_aware(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", _ensure_timezone_aware(self.updated_at, "updated_at"))
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be greater than or equal to created_at")
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "energy": self.energy,
            "normalized_energy": self.normalized_energy,
            "status": self.status.value,
            "update_count": self.update_count,
            "last_pressure": self.last_pressure,
            "last_temperature_scale": self.last_temperature_scale,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }


def clamp_energy_probability(value: float) -> float:
    number = ensure_finite_number(value, "value")
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def normalize_energy_value(
    *,
    value: float,
    max_energy: float,
) -> float:
    checked_value = ensure_non_negative_number(value, "value")
    checked_max = _ensure_positive_finite_number(max_energy, "max_energy")
    return clamp_energy_probability(checked_value / checked_max)


def energy_status_from_inputs(
    *,
    temperature: SemanticTemperatureState | None,
    plan: MetricUpdatePlan | None,
    normalized_energy: float | None,
) -> EnergyStatus:
    if temperature is not None:
        if not isinstance(temperature, SemanticTemperatureState):
            raise TypeError("temperature must be SemanticTemperatureState or None")
        if semantic_temperature_is_frozen(temperature):
            return EnergyStatus.FROZEN
    if plan is not None:
        if not isinstance(plan, MetricUpdatePlan):
            raise TypeError("plan must be MetricUpdatePlan or None")
        if plan.status is MetricUpdatePlanStatus.BLOCKED:
            return EnergyStatus.DEGRADED
    if normalized_energy is not None and ensure_constraint_probability(normalized_energy, "normalized_energy") <= 0.05:
        return EnergyStatus.STABLE
    return EnergyStatus.ACTIVE


def confidence_from_costs(
    costs: tuple[MetricCostResult, ...] | list[MetricCostResult],
) -> float:
    checked_costs = _ensure_cost_sequence(costs)
    if not checked_costs:
        return 1.0
    return ensure_constraint_probability(
        sum(cost.confidence for cost in checked_costs) / len(checked_costs),
        "confidence",
    )


def compute_total_cost_energy(
    *,
    costs: tuple[MetricCostResult, ...] | list[MetricCostResult],
    policy: EnergyPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> EnergyResult:
    checked_costs = _ensure_cost_sequence(costs)
    active_policy = _ensure_policy(policy)
    value = ensure_non_negative_number(total_cost(checked_costs) * active_policy.cost_weight, "value")
    normalized_value = (
        normalize_energy_value(value=value, max_energy=active_policy.max_energy)
        if active_policy.normalize_energy
        else None
    )
    result_metadata: dict[str, object] = {
        "source": "costs",
        "cost_summary": cost_summary(checked_costs),
    }
    if metadata is not None:
        result_metadata.update(_copy_metadata(metadata))
    result_metadata["source"] = "costs"

    return EnergyResult(
        kind=EnergyKind.TOTAL_COST,
        status=energy_status_from_inputs(temperature=None, plan=None, normalized_energy=normalized_value),
        value=value,
        normalized_value=normalized_value,
        confidence=confidence_from_costs(checked_costs),
        source_cost_ids=tuple(str(cost.cost_id) for cost in checked_costs),
        source_plan_status=None,
        created_at=utc_now(),
        metadata=result_metadata,
    )


def compute_update_plan_energy(
    *,
    plan: MetricUpdatePlan,
    policy: EnergyPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> EnergyResult:
    if not isinstance(plan, MetricUpdatePlan):
        raise TypeError("plan must be MetricUpdatePlan")
    active_policy = _ensure_policy(policy)
    value = ensure_non_negative_number(
        plan.total_cost_value * active_policy.cost_weight + plan.total_pressure * active_policy.pressure_weight,
        "value",
    )
    normalized_value = (
        normalize_energy_value(value=value, max_energy=active_policy.max_energy)
        if active_policy.normalize_energy
        else None
    )
    confidence = _metadata_probability(plan.metadata.get("aggregate_signal_confidence"), default=1.0)
    result_metadata: dict[str, object] = {
        "source": "metric_update_plan",
        "plan_mode": plan.mode.value,
        "plan_reason": plan.reason.value,
        "pressure": plan.total_pressure,
    }
    if metadata is not None:
        result_metadata.update(_copy_metadata(metadata))
    result_metadata["source"] = "metric_update_plan"

    return EnergyResult(
        kind=EnergyKind.TRANSITION_PRESSURE,
        status=energy_status_from_inputs(temperature=None, plan=plan, normalized_energy=normalized_value),
        value=value,
        normalized_value=normalized_value,
        confidence=confidence,
        source_cost_ids=plan.cost_ids,
        source_plan_status=plan.status.value,
        created_at=utc_now(),
        metadata=result_metadata,
    )


def compute_temperature_modulated_energy(
    *,
    base_energy: EnergyResult,
    temperature: SemanticTemperatureState,
    policy: EnergyPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> EnergyResult:
    if not isinstance(base_energy, EnergyResult):
        raise TypeError("base_energy must be EnergyResult")
    if not isinstance(temperature, SemanticTemperatureState):
        raise TypeError("temperature must be SemanticTemperatureState")
    active_policy = _ensure_policy(policy)
    scale = transition_scale_from_temperature(temperature)
    value = ensure_non_negative_number(
        base_energy.value * (1.0 + active_policy.temperature_weight * scale),
        "value",
    )
    normalized_value = (
        normalize_energy_value(value=value, max_energy=active_policy.max_energy)
        if active_policy.normalize_energy
        else None
    )
    result_metadata: dict[str, object] = {
        "source": "temperature_modulation",
        "base_energy_kind": base_energy.kind.value,
        "temperature_scale": scale,
        "temperature_status": temperature.status.value,
    }
    if metadata is not None:
        result_metadata.update(_copy_metadata(metadata))
    result_metadata["source"] = "temperature_modulation"

    return EnergyResult(
        kind=EnergyKind.TEMPERATURE_MODULATED,
        status=energy_status_from_inputs(temperature=temperature, plan=None, normalized_energy=normalized_value),
        value=value,
        normalized_value=normalized_value,
        confidence=base_energy.confidence,
        source_cost_ids=base_energy.source_cost_ids,
        source_plan_status=base_energy.source_plan_status,
        created_at=utc_now(),
        metadata=result_metadata,
    )


def create_energy_state(
    *,
    energy: EnergyResult,
    temperature: SemanticTemperatureState | None = None,
    metadata: dict[str, object] | None = None,
) -> EnergyState:
    if not isinstance(energy, EnergyResult):
        raise TypeError("energy must be EnergyResult")
    if temperature is not None and not isinstance(temperature, SemanticTemperatureState):
        raise TypeError("temperature must be SemanticTemperatureState or None")
    timestamp = utc_now()
    temperature_scale = transition_scale_from_temperature(temperature) if temperature is not None else 0.0
    state_metadata: dict[str, object] = {
        "state_kind": "energy_state",
        "energy_kind": energy.kind.value,
    }
    if metadata is not None:
        state_metadata.update(_copy_metadata(metadata))
    state_metadata["state_kind"] = "energy_state"

    return EnergyState(
        energy=energy.value,
        normalized_energy=energy.normalized_value,
        status=energy_status_from_inputs(
            temperature=temperature,
            plan=None,
            normalized_energy=energy.normalized_value,
        ),
        update_count=0,
        last_pressure=_metadata_non_negative_number(energy.metadata.get("pressure"), default=0.0),
        last_temperature_scale=temperature_scale,
        created_at=timestamp,
        updated_at=timestamp,
        metadata=state_metadata,
    )


def transition_energy_state(
    *,
    state: EnergyState,
    energy: EnergyResult,
    temperature: SemanticTemperatureState | None = None,
    metadata: dict[str, object] | None = None,
) -> EnergyState:
    if not isinstance(state, EnergyState):
        raise TypeError("state must be EnergyState")
    if not isinstance(energy, EnergyResult):
        raise TypeError("energy must be EnergyResult")
    if temperature is not None and not isinstance(temperature, SemanticTemperatureState):
        raise TypeError("temperature must be SemanticTemperatureState or None")

    state_metadata: dict[str, object] = {
        "previous_energy": state.energy,
        "energy_kind": energy.kind.value,
    }
    if metadata is not None:
        state_metadata.update(_copy_metadata(metadata))

    return EnergyState(
        energy=energy.value,
        normalized_energy=energy.normalized_value,
        status=energy_status_from_inputs(
            temperature=temperature,
            plan=None,
            normalized_energy=energy.normalized_value,
        ),
        update_count=state.update_count + 1,
        last_pressure=_metadata_non_negative_number(energy.metadata.get("pressure"), default=state.last_pressure),
        last_temperature_scale=(
            transition_scale_from_temperature(temperature)
            if temperature is not None
            else state.last_temperature_scale
        ),
        created_at=state.created_at,
        updated_at=utc_now(),
        metadata=state_metadata,
    )


def energy_is_stable(state: EnergyState) -> bool:
    if not isinstance(state, EnergyState):
        raise TypeError("state must be EnergyState")
    return state.status in (EnergyStatus.STABLE, EnergyStatus.FROZEN)


def summarize_energy_results(
    results: tuple[EnergyResult, ...] | list[EnergyResult],
) -> dict[str, object]:
    checked_results = _ensure_energy_result_sequence(results)
    normalized_values = [
        result.normalized_value for result in checked_results if result.normalized_value is not None
    ]
    kinds: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for result in checked_results:
        kinds[result.kind.value] = kinds.get(result.kind.value, 0) + 1
        statuses[result.status.value] = statuses.get(result.status.value, 0) + 1

    return {
        "count": len(checked_results),
        "total_value": ensure_non_negative_number(
            sum(result.value for result in checked_results),
            "total_value",
        ),
        "mean_normalized_value": (
            ensure_constraint_probability(sum(normalized_values) / len(normalized_values), "mean_normalized_value")
            if normalized_values
            else None
        ),
        "max_value": max((result.value for result in checked_results), default=0.0),
        "kinds": kinds,
        "statuses": statuses,
    }


def _ensure_policy(policy: EnergyPolicy | None) -> EnergyPolicy:
    if policy is None:
        return EnergyPolicy()
    if not isinstance(policy, EnergyPolicy):
        raise TypeError("policy must be EnergyPolicy or None")
    return policy


def _ensure_cost_sequence(
    costs: tuple[MetricCostResult, ...] | list[MetricCostResult],
) -> tuple[MetricCostResult, ...]:
    if not isinstance(costs, (tuple, list)):
        raise TypeError("costs must be tuple[MetricCostResult, ...] or list[MetricCostResult]")
    checked_costs = tuple(costs)
    for index, cost in enumerate(checked_costs):
        if not isinstance(cost, MetricCostResult):
            raise TypeError(f"costs[{index}] must be MetricCostResult")
    return checked_costs


def _ensure_energy_result_sequence(
    results: tuple[EnergyResult, ...] | list[EnergyResult],
) -> tuple[EnergyResult, ...]:
    if not isinstance(results, (tuple, list)):
        raise TypeError("results must be tuple[EnergyResult, ...] or list[EnergyResult]")
    checked_results = tuple(results)
    for index, result in enumerate(checked_results):
        if not isinstance(result, EnergyResult):
            raise TypeError(f"results[{index}] must be EnergyResult")
    return checked_results


def _ensure_id_tuple(value: tuple[str, ...] | list[str], field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple[str, ...] or list[str]")
    checked_ids = tuple(value)
    for index, item in enumerate(checked_ids):
        _ensure_non_empty_string(item, f"{field_name}[{index}]")
    return checked_ids


def _metadata_probability(value: object, *, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return ensure_constraint_probability(default, "default")
    try:
        return ensure_constraint_probability(value, "metadata_probability")
    except ValueError:
        return ensure_constraint_probability(default, "default")


def _metadata_non_negative_number(value: object, *, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return ensure_non_negative_number(default, "default")
    try:
        return ensure_non_negative_number(value, "metadata_non_negative_number")
    except ValueError:
        return ensure_non_negative_number(default, "default")


def _ensure_positive_finite_number(value: float, field_name: str) -> float:
    number = ensure_finite_number(value, field_name)
    if number <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return number


def _ensure_non_negative_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


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
    "EnergyKind",
    "EnergyStatus",
    "EnergyPolicy",
    "EnergyResult",
    "EnergyState",
    "clamp_energy_probability",
    "normalize_energy_value",
    "energy_status_from_inputs",
    "confidence_from_costs",
    "compute_total_cost_energy",
    "compute_update_plan_energy",
    "compute_temperature_modulated_energy",
    "create_energy_state",
    "transition_energy_state",
    "energy_is_stable",
    "summarize_energy_results",
]
