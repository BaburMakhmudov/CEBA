"""Deterministic MetricState helpers for CEBA Core Metric Processing."""

from __future__ import annotations

from dataclasses import dataclass, field

from ceba.core.metrics.schema import (
    MetricSpaceKind,
    MetricState,
    MetricTensorRef,
    MetricTensorStatus,
    MetricUpdateReason,
    MetricUpdateSignal,
    ensure_non_negative_number,
    ensure_positive_int,
    ensure_timezone_aware,
    new_metric_state_id,
    utc_now,
)


@dataclass(frozen=True, slots=True)
class MetricStateTransition:
    previous_state_id: str | None
    next_state_id: str
    reason: MetricUpdateReason
    update_count_before: int
    update_count_after: int
    lambda_before: float
    lambda_after: float
    signal_id: str | None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.previous_state_id is not None:
            object.__setattr__(
                self,
                "previous_state_id",
                _ensure_optional_non_empty_string(self.previous_state_id, "previous_state_id"),
            )
        object.__setattr__(self, "next_state_id", _ensure_non_empty_string(self.next_state_id, "next_state_id"))
        if not isinstance(self.reason, MetricUpdateReason):
            raise TypeError("reason must be MetricUpdateReason")
        object.__setattr__(
            self,
            "update_count_before",
            _ensure_non_negative_int(self.update_count_before, "update_count_before"),
        )
        object.__setattr__(
            self,
            "update_count_after",
            _ensure_non_negative_int(self.update_count_after, "update_count_after"),
        )
        if self.update_count_after < self.update_count_before:
            raise ValueError("update_count_after must be greater than or equal to update_count_before")
        object.__setattr__(self, "lambda_before", ensure_non_negative_number(self.lambda_before, "lambda_before"))
        object.__setattr__(self, "lambda_after", ensure_non_negative_number(self.lambda_after, "lambda_after"))
        if self.signal_id is not None:
            object.__setattr__(self, "signal_id", _ensure_optional_non_empty_string(self.signal_id, "signal_id"))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "previous_state_id": self.previous_state_id,
            "next_state_id": self.next_state_id,
            "reason": self.reason.value,
            "update_count_before": self.update_count_before,
            "update_count_after": self.update_count_after,
            "lambda_before": self.lambda_before,
            "lambda_after": self.lambda_after,
            "signal_id": self.signal_id,
            "metadata": dict(self.metadata),
        }


def create_identity_tensor_ref(
    *,
    tensor_id: str,
    dimension: int,
    space: MetricSpaceKind = MetricSpaceKind.RIEMANNIAN,
    metadata: dict[str, object] | None = None,
) -> MetricTensorRef:
    _ensure_non_empty_string(tensor_id, "tensor_id")
    ensure_positive_int(dimension, "dimension")
    if not isinstance(space, MetricSpaceKind):
        raise TypeError("space must be MetricSpaceKind")

    merged_metadata: dict[str, object] = {"tensor_kind": "identity"}
    if metadata is not None:
        merged_metadata.update(_copy_metadata(metadata))
    merged_metadata["tensor_kind"] = "identity"

    return MetricTensorRef(
        tensor_id=tensor_id,
        space=space,
        dimension=dimension,
        status=MetricTensorStatus.IDENTITY,
        is_spd=True,
        condition_number=1.0,
        metadata=merged_metadata,
    )


def create_metric_state(
    *,
    tensor: MetricTensorRef,
    lambda_value: float = 1.0,
    update_count: int = 0,
    metadata: dict[str, object] | None = None,
) -> MetricState:
    if not isinstance(tensor, MetricTensorRef):
        raise TypeError("tensor must be MetricTensorRef")
    lambda_number = ensure_non_negative_number(lambda_value, "lambda_value")
    update_count_value = _ensure_non_negative_int(update_count, "update_count")
    timestamp = utc_now()

    merged_metadata: dict[str, object] = {"state_kind": "metric_state"}
    if metadata is not None:
        merged_metadata.update(_copy_metadata(metadata))
    merged_metadata["state_kind"] = "metric_state"

    return MetricState(
        state_id=new_metric_state_id(),
        space=tensor.space,
        dimension=tensor.dimension,
        tensor=tensor,
        lambda_value=lambda_number,
        update_count=update_count_value,
        created_at=timestamp,
        updated_at=timestamp,
        metadata=merged_metadata,
    )


def create_identity_metric_state(
    *,
    tensor_id: str,
    dimension: int,
    space: MetricSpaceKind = MetricSpaceKind.RIEMANNIAN,
    lambda_value: float = 1.0,
    metadata: dict[str, object] | None = None,
) -> MetricState:
    tensor = create_identity_tensor_ref(
        tensor_id=tensor_id,
        dimension=dimension,
        space=space,
        metadata=metadata,
    )
    return create_metric_state(
        tensor=tensor,
        lambda_value=lambda_value,
        update_count=0,
        metadata=metadata,
    )


def update_lambda_from_signal(
    *,
    current_lambda: float,
    signal: MetricUpdateSignal,
    learning_rate: float = 0.1,
    max_lambda: float = 10.0,
) -> float:
    current = ensure_non_negative_number(current_lambda, "current_lambda")
    if not isinstance(signal, MetricUpdateSignal):
        raise TypeError("signal must be MetricUpdateSignal")
    rate = ensure_non_negative_number(learning_rate, "learning_rate")
    maximum = ensure_non_negative_number(max_lambda, "max_lambda")
    if maximum < current:
        return maximum

    delta = rate * signal.pressure * signal.confidence
    return ensure_non_negative_number(min(maximum, current + delta), "next_lambda")


def transition_metric_state(
    *,
    state: MetricState,
    signal: MetricUpdateSignal,
    tensor: MetricTensorRef | None = None,
    learning_rate: float = 0.1,
    max_lambda: float = 10.0,
    metadata: dict[str, object] | None = None,
) -> tuple[MetricState, MetricStateTransition]:
    if not isinstance(state, MetricState):
        raise TypeError("state must be MetricState")
    if not isinstance(signal, MetricUpdateSignal):
        raise TypeError("signal must be MetricUpdateSignal")
    next_tensor = state.tensor if tensor is None else tensor
    if not isinstance(next_tensor, MetricTensorRef):
        raise TypeError("tensor must be None or MetricTensorRef")
    if next_tensor.dimension != state.dimension:
        raise ValueError("tensor.dimension must equal state.dimension")
    if next_tensor.space is not state.space:
        raise ValueError("tensor.space must equal state.space")

    lambda_after = update_lambda_from_signal(
        current_lambda=state.lambda_value,
        signal=signal,
        learning_rate=learning_rate,
        max_lambda=max_lambda,
    )
    update_count_after = state.update_count + 1
    timestamp = utc_now()

    merged_metadata: dict[str, object] = {
        "transition_reason": signal.reason.value,
        "source_signal_id": str(signal.signal_id),
    }
    if metadata is not None:
        merged_metadata.update(_copy_metadata(metadata))
    merged_metadata["source_signal_id"] = str(signal.signal_id)

    next_state = MetricState(
        state_id=new_metric_state_id(),
        space=state.space,
        dimension=state.dimension,
        tensor=next_tensor,
        lambda_value=lambda_after,
        update_count=update_count_after,
        created_at=state.created_at,
        updated_at=timestamp,
        metadata=merged_metadata,
    )
    transition = MetricStateTransition(
        previous_state_id=str(state.state_id),
        next_state_id=str(next_state.state_id),
        reason=signal.reason,
        update_count_before=state.update_count,
        update_count_after=update_count_after,
        lambda_before=state.lambda_value,
        lambda_after=lambda_after,
        signal_id=str(signal.signal_id),
        metadata=merged_metadata,
    )
    return next_state, transition


def metric_state_is_valid(state: MetricState) -> bool:
    if not isinstance(state, MetricState):
        raise TypeError("state must be MetricState")
    return (
        state.tensor.dimension == state.dimension
        and state.tensor.space is state.space
        and ensure_timezone_aware(state.updated_at, "updated_at")
        >= ensure_timezone_aware(state.created_at, "created_at")
        and state.lambda_value >= 0.0
        and state.update_count >= 0
    )


def summarize_metric_state(state: MetricState) -> dict[str, object]:
    if not isinstance(state, MetricState):
        raise TypeError("state must be MetricState")
    return {
        "state_id": str(state.state_id),
        "space": state.space.value,
        "dimension": state.dimension,
        "tensor_id": state.tensor.tensor_id,
        "tensor_status": state.tensor.status.value,
        "tensor_is_spd": state.tensor.is_spd,
        "lambda_value": state.lambda_value,
        "update_count": state.update_count,
        "created_at": state.created_at.isoformat(),
        "updated_at": state.updated_at.isoformat(),
        "metadata": dict(state.metadata),
    }


def _ensure_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _ensure_optional_non_empty_string(value: str | None, field_name: str) -> str | None:
    if value is not None:
        _ensure_non_empty_string(value, field_name)
    return value


def _ensure_non_negative_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _copy_metadata(value: dict[str, object]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("metadata must be dict[str, object]")
    if any(not isinstance(key, str) for key in value):
        raise TypeError("metadata keys must be strings")
    return dict(value)


__all__ = [
    "MetricStateTransition",
    "create_identity_tensor_ref",
    "create_metric_state",
    "create_identity_metric_state",
    "update_lambda_from_signal",
    "transition_metric_state",
    "metric_state_is_valid",
    "summarize_metric_state",
]
