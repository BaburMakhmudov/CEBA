"""Deterministic metric update planning utilities for CEBA Core."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ceba.core.metrics.cost import mean_normalized_cost, total_cost
from ceba.core.metrics.metric_state import summarize_metric_state, update_lambda_from_signal
from ceba.core.metrics.schema import (
    MetricCostResult,
    MetricState,
    MetricUpdateReason,
    MetricUpdateSignal,
    ensure_constraint_probability,
    ensure_non_negative_number,
    utc_now,
)


class MetricUpdateMode(Enum):
    NONE = "NONE"
    LAMBDA_ONLY = "LAMBDA_ONLY"
    TENSOR_REF_ONLY = "TENSOR_REF_ONLY"
    LAMBDA_AND_TENSOR_REF = "LAMBDA_AND_TENSOR_REF"
    DEFERRED = "DEFERRED"


class MetricUpdatePlanStatus(Enum):
    READY = "READY"
    SKIPPED = "SKIPPED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class MetricUpdatePlan:
    mode: MetricUpdateMode
    status: MetricUpdatePlanStatus
    state_id: str
    signal_ids: tuple[str, ...]
    cost_ids: tuple[str, ...]
    lambda_before: float
    lambda_after: float
    total_pressure: float
    total_cost_value: float
    mean_normalized_cost_value: float | None
    reason: MetricUpdateReason
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.mode, MetricUpdateMode):
            raise TypeError("mode must be MetricUpdateMode")
        if not isinstance(self.status, MetricUpdatePlanStatus):
            raise TypeError("status must be MetricUpdatePlanStatus")
        object.__setattr__(self, "state_id", _ensure_non_empty_string(self.state_id, "state_id"))
        object.__setattr__(self, "signal_ids", _ensure_id_tuple(self.signal_ids, "signal_ids"))
        object.__setattr__(self, "cost_ids", _ensure_id_tuple(self.cost_ids, "cost_ids"))
        object.__setattr__(self, "lambda_before", ensure_non_negative_number(self.lambda_before, "lambda_before"))
        object.__setattr__(self, "lambda_after", ensure_non_negative_number(self.lambda_after, "lambda_after"))
        object.__setattr__(self, "total_pressure", ensure_non_negative_number(self.total_pressure, "total_pressure"))
        object.__setattr__(
            self,
            "total_cost_value",
            ensure_non_negative_number(self.total_cost_value, "total_cost_value"),
        )
        if self.mean_normalized_cost_value is not None:
            object.__setattr__(
                self,
                "mean_normalized_cost_value",
                ensure_constraint_probability(self.mean_normalized_cost_value, "mean_normalized_cost_value"),
            )
        if not isinstance(self.reason, MetricUpdateReason):
            raise TypeError("reason must be MetricUpdateReason")
        object.__setattr__(self, "created_at", _ensure_timezone_aware(self.created_at, "created_at"))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "status": self.status.value,
            "state_id": self.state_id,
            "signal_ids": self.signal_ids,
            "cost_ids": self.cost_ids,
            "lambda_before": self.lambda_before,
            "lambda_after": self.lambda_after,
            "total_pressure": self.total_pressure,
            "total_cost_value": self.total_cost_value,
            "mean_normalized_cost_value": self.mean_normalized_cost_value,
            "reason": self.reason.value,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


def dominant_update_reason(
    signals: tuple[MetricUpdateSignal, ...] | list[MetricUpdateSignal],
) -> MetricUpdateReason:
    checked_signals = _ensure_signal_sequence(signals)
    if not checked_signals:
        return MetricUpdateReason.UNKNOWN

    reasons = {signal.reason for signal in checked_signals}
    for reason in _REASON_PRIORITY:
        if reason in reasons:
            return reason
    return MetricUpdateReason.UNKNOWN


def total_signal_pressure(
    signals: tuple[MetricUpdateSignal, ...] | list[MetricUpdateSignal],
) -> float:
    checked_signals = _ensure_signal_sequence(signals)
    return ensure_non_negative_number(
        sum(signal.pressure for signal in checked_signals),
        "total_signal_pressure",
    )


def aggregate_signal_confidence(
    signals: tuple[MetricUpdateSignal, ...] | list[MetricUpdateSignal],
) -> float:
    checked_signals = _ensure_signal_sequence(signals)
    if not checked_signals:
        return 1.0
    return ensure_constraint_probability(
        sum(signal.confidence for signal in checked_signals) / len(checked_signals),
        "aggregate_signal_confidence",
    )


def select_update_mode(
    *,
    signals: tuple[MetricUpdateSignal, ...],
    costs: tuple[MetricCostResult, ...],
    pressure_threshold: float = 0.0,
) -> MetricUpdateMode:
    checked_signals = _ensure_signal_sequence(signals)
    checked_costs = _ensure_cost_sequence(costs)
    threshold = ensure_non_negative_number(pressure_threshold, "pressure_threshold")

    if not checked_signals and not checked_costs:
        return MetricUpdateMode.NONE
    if total_signal_pressure(checked_signals) <= threshold:
        return MetricUpdateMode.DEFERRED
    return MetricUpdateMode.LAMBDA_ONLY


def build_metric_update_plan(
    *,
    state: MetricState,
    signals: tuple[MetricUpdateSignal, ...] | list[MetricUpdateSignal],
    costs: tuple[MetricCostResult, ...] | list[MetricCostResult] = (),
    learning_rate: float = 0.1,
    max_lambda: float = 10.0,
    pressure_threshold: float = 0.0,
    metadata: dict[str, object] | None = None,
) -> MetricUpdatePlan:
    if not isinstance(state, MetricState):
        raise TypeError("state must be MetricState")
    checked_signals = _ensure_signal_sequence(signals)
    checked_costs = _ensure_cost_sequence(costs)
    ensure_non_negative_number(learning_rate, "learning_rate")
    ensure_non_negative_number(max_lambda, "max_lambda")

    mode = select_update_mode(
        signals=checked_signals,
        costs=checked_costs,
        pressure_threshold=pressure_threshold,
    )
    status = (
        MetricUpdatePlanStatus.SKIPPED
        if mode in (MetricUpdateMode.NONE, MetricUpdateMode.DEFERRED)
        else MetricUpdatePlanStatus.READY
    )

    lambda_before = ensure_non_negative_number(state.lambda_value, "lambda_before")
    lambda_after = lambda_before
    if checked_signals and mode not in (MetricUpdateMode.NONE, MetricUpdateMode.DEFERRED):
        for signal in checked_signals:
            lambda_after = update_lambda_from_signal(
                current_lambda=lambda_after,
                signal=signal,
                learning_rate=learning_rate,
                max_lambda=max_lambda,
            )

    plan_metadata: dict[str, object] = {
        "planner": "metric_update_v1",
        "state_summary": summarize_metric_state(state),
        "signal_count": len(checked_signals),
        "cost_count": len(checked_costs),
        "aggregate_signal_confidence": aggregate_signal_confidence(checked_signals),
    }
    if metadata is not None:
        plan_metadata.update(_copy_metadata(metadata))
    plan_metadata["planner"] = "metric_update_v1"

    return MetricUpdatePlan(
        mode=mode,
        status=status,
        state_id=str(state.state_id),
        signal_ids=tuple(str(signal.signal_id) for signal in checked_signals),
        cost_ids=tuple(str(cost.cost_id) for cost in checked_costs),
        lambda_before=lambda_before,
        lambda_after=lambda_after,
        total_pressure=total_signal_pressure(checked_signals),
        total_cost_value=ensure_non_negative_number(total_cost(checked_costs), "total_cost_value"),
        mean_normalized_cost_value=mean_normalized_cost(checked_costs),
        reason=dominant_update_reason(checked_signals),
        created_at=utc_now(),
        metadata=plan_metadata,
    )


def plan_is_ready(plan: MetricUpdatePlan) -> bool:
    if not isinstance(plan, MetricUpdatePlan):
        raise TypeError("plan must be MetricUpdatePlan")
    return plan.status is MetricUpdatePlanStatus.READY


def plan_is_skipped(plan: MetricUpdatePlan) -> bool:
    if not isinstance(plan, MetricUpdatePlan):
        raise TypeError("plan must be MetricUpdatePlan")
    return plan.status is MetricUpdatePlanStatus.SKIPPED


def summarize_update_plan(plan: MetricUpdatePlan) -> dict[str, object]:
    if not isinstance(plan, MetricUpdatePlan):
        raise TypeError("plan must be MetricUpdatePlan")
    return plan.to_dict()


def _ensure_signal_sequence(
    signals: tuple[MetricUpdateSignal, ...] | list[MetricUpdateSignal],
) -> tuple[MetricUpdateSignal, ...]:
    if not isinstance(signals, (tuple, list)):
        raise TypeError("signals must be tuple[MetricUpdateSignal, ...] or list[MetricUpdateSignal]")
    checked_signals = tuple(signals)
    for index, signal in enumerate(checked_signals):
        if not isinstance(signal, MetricUpdateSignal):
            raise TypeError(f"signals[{index}] must be MetricUpdateSignal")
    return checked_signals


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


def _ensure_id_tuple(value: tuple[str, ...] | list[str], field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple[str, ...] or list[str]")
    checked_ids = tuple(value)
    for index, item in enumerate(checked_ids):
        _ensure_non_empty_string(item, f"{field_name}[{index}]")
    return checked_ids


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


_REASON_PRIORITY: tuple[MetricUpdateReason, ...] = (
    MetricUpdateReason.CONSTRAINT_FAIL,
    MetricUpdateReason.VIOLATION_PRESSURE,
    MetricUpdateReason.CONSTRAINT_WARN,
    MetricUpdateReason.STABILITY_CORRECTION,
    MetricUpdateReason.TELOS_ALIGNMENT,
    MetricUpdateReason.CONSTRAINT_PASS,
    MetricUpdateReason.UNKNOWN,
)


__all__ = [
    "MetricUpdateMode",
    "MetricUpdatePlanStatus",
    "MetricUpdatePlan",
    "dominant_update_reason",
    "total_signal_pressure",
    "aggregate_signal_confidence",
    "select_update_mode",
    "build_metric_update_plan",
    "plan_is_ready",
    "plan_is_skipped",
    "summarize_update_plan",
]
