"""Deterministic cost computation utilities for CEBA Core Metric Processing."""

from __future__ import annotations

from dataclasses import dataclass

from ceba.core.metrics.schema import (
    CostKind,
    MetricCostResult,
    MetricDistanceResult,
    MetricUpdateSignal,
    ensure_constraint_probability,
    ensure_finite_number,
    ensure_non_negative_number,
    new_metric_cost_id,
    utc_now,
)


DEFAULT_DISTANCE_NORMALIZER = 1.0
DEFAULT_SIGNAL_NORMALIZER = 1.0


@dataclass(frozen=True, slots=True)
class CostComputationPolicy:
    distance_normalizer: float = DEFAULT_DISTANCE_NORMALIZER
    signal_normalizer: float = DEFAULT_SIGNAL_NORMALIZER
    clamp_normalized: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "distance_normalizer",
            _ensure_positive_finite_number(self.distance_normalizer, "distance_normalizer"),
        )
        object.__setattr__(
            self,
            "signal_normalizer",
            _ensure_positive_finite_number(self.signal_normalizer, "signal_normalizer"),
        )
        if not isinstance(self.clamp_normalized, bool):
            raise TypeError("clamp_normalized must be bool")


def clamp_probability(value: float) -> float:
    number = ensure_finite_number(value, "value")
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def normalize_cost_value(
    *,
    value: float,
    normalizer: float,
    clamp: bool = True,
) -> float:
    non_negative_value = ensure_non_negative_number(value, "value")
    positive_normalizer = _ensure_positive_finite_number(normalizer, "normalizer")
    if not isinstance(clamp, bool):
        raise TypeError("clamp must be bool")

    normalized = ensure_non_negative_number(non_negative_value / positive_normalizer, "normalized")
    if clamp:
        return clamp_probability(normalized)
    return normalized


def cost_from_distance(
    *,
    distance: MetricDistanceResult,
    kind: CostKind = CostKind.ENERGY,
    policy: CostComputationPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> MetricCostResult:
    if not isinstance(distance, MetricDistanceResult):
        raise TypeError("distance must be MetricDistanceResult")
    if not isinstance(kind, CostKind):
        raise TypeError("kind must be CostKind")

    active_policy = _ensure_policy(policy)
    cost_metadata: dict[str, object] = {
        "source": "metric_distance",
        "distance_kind": distance.kind.value,
        "distance_value": distance.value,
    }
    if metadata is not None:
        cost_metadata.update(_copy_metadata(metadata))
    cost_metadata["source"] = "metric_distance"

    return MetricCostResult(
        cost_id=new_metric_cost_id(),
        kind=kind,
        value=distance.value,
        normalized_value=normalize_cost_value(
            value=distance.value,
            normalizer=active_policy.distance_normalizer,
            clamp=active_policy.clamp_normalized,
        ),
        confidence=distance.confidence,
        source_distance_id=distance.distance_id,
        source_evaluation_id=None,
        created_at=utc_now(),
        metadata=cost_metadata,
    )


def constraint_penalty_from_signal(
    *,
    signal: MetricUpdateSignal,
    policy: CostComputationPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> MetricCostResult:
    if not isinstance(signal, MetricUpdateSignal):
        raise TypeError("signal must be MetricUpdateSignal")

    active_policy = _ensure_policy(policy)
    value = ensure_non_negative_number(signal.pressure * signal.weight, "value")
    cost_metadata: dict[str, object] = {
        "source": "metric_update_signal",
        "signal_id": str(signal.signal_id),
        "reason": signal.reason.value,
        "pressure": signal.pressure,
        "weight": signal.weight,
    }
    if metadata is not None:
        cost_metadata.update(_copy_metadata(metadata))
    cost_metadata["source"] = "metric_update_signal"

    return MetricCostResult(
        cost_id=new_metric_cost_id(),
        kind=CostKind.CONSTRAINT_PENALTY,
        value=value,
        normalized_value=normalize_cost_value(
            value=value,
            normalizer=active_policy.signal_normalizer,
            clamp=active_policy.clamp_normalized,
        ),
        confidence=signal.confidence,
        source_distance_id=None,
        source_evaluation_id=signal.source_evaluation_id,
        created_at=utc_now(),
        metadata=cost_metadata,
    )


def total_cost(
    costs: tuple[MetricCostResult, ...] | list[MetricCostResult],
) -> float:
    checked_costs = _ensure_cost_sequence(costs)
    return ensure_finite_number(sum(cost.value for cost in checked_costs), "total_cost")


def mean_normalized_cost(
    costs: tuple[MetricCostResult, ...] | list[MetricCostResult],
) -> float | None:
    checked_costs = _ensure_cost_sequence(costs)
    normalized_values = [cost.normalized_value for cost in checked_costs if cost.normalized_value is not None]
    if not normalized_values:
        return None
    return ensure_constraint_probability(
        sum(normalized_values) / len(normalized_values),
        "mean_normalized_cost",
    )


def max_cost(
    costs: tuple[MetricCostResult, ...] | list[MetricCostResult],
) -> MetricCostResult | None:
    checked_costs = _ensure_cost_sequence(costs)
    if not checked_costs:
        return None

    maximum = checked_costs[0]
    for cost in checked_costs[1:]:
        if cost.value > maximum.value:
            maximum = cost
    return maximum


def cost_summary(
    costs: tuple[MetricCostResult, ...] | list[MetricCostResult],
) -> dict[str, object]:
    checked_costs = _ensure_cost_sequence(costs)
    maximum = max_cost(checked_costs)
    kinds: dict[str, int] = {}
    for cost in checked_costs:
        kind = cost.kind.value
        kinds[kind] = kinds.get(kind, 0) + 1

    return {
        "count": len(checked_costs),
        "total_cost": total_cost(checked_costs),
        "mean_normalized_cost": mean_normalized_cost(checked_costs),
        "max_cost_id": str(maximum.cost_id) if maximum is not None else None,
        "max_cost_value": maximum.value if maximum is not None else None,
        "kinds": kinds,
    }


def _ensure_policy(policy: CostComputationPolicy | None) -> CostComputationPolicy:
    if policy is None:
        return CostComputationPolicy()
    if not isinstance(policy, CostComputationPolicy):
        raise TypeError("policy must be CostComputationPolicy or None")
    return policy


def _ensure_positive_finite_number(value: float, field_name: str) -> float:
    number = ensure_finite_number(value, field_name)
    if number <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return number


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


def _copy_metadata(value: dict[str, object]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("metadata must be dict[str, object]")
    if any(not isinstance(key, str) for key in value):
        raise TypeError("metadata keys must be strings")
    return dict(value)


__all__ = [
    "DEFAULT_DISTANCE_NORMALIZER",
    "DEFAULT_SIGNAL_NORMALIZER",
    "CostComputationPolicy",
    "clamp_probability",
    "normalize_cost_value",
    "cost_from_distance",
    "constraint_penalty_from_signal",
    "total_cost",
    "mean_normalized_cost",
    "max_cost",
    "cost_summary",
]
