"""Deterministic update-policy utilities for CEBA Core Regulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ceba.core.dynamics.energy import EnergyState, EnergyStatus
from ceba.core.dynamics.semantic_temperature import (
    SemanticTemperatureState,
    semantic_temperature_is_frozen,
)
from ceba.core.dynamics.stability import (
    StabilityReport,
    risk_rank,
    stability_report_is_blocked,
    stability_report_is_passed,
)
from ceba.core.dynamics.transition import (
    DynamicsTransitionResult,
    DynamicsTransitionStatus,
    transition_is_blocked,
)
from ceba.core.metrics.schema import (
    ensure_constraint_probability,
    ensure_finite_number,
    ensure_non_negative_number,
    utc_now,
)
from ceba.core.regulation.precision_controller import (
    PrecisionState,
    PrecisionStatus,
    precision_allows_update,
    precision_scale,
)
from ceba.core.regulation.step_size import (
    StepSizeState,
    StepSizeStatus,
    step_size_allows_update,
    step_size_scale,
)


class UpdatePolicyMode(Enum):
    """Update-policy operating mode."""

    NORMAL = "NORMAL"
    CONSERVATIVE = "CONSERVATIVE"
    AGGRESSIVE = "AGGRESSIVE"
    DEFERRED = "DEFERRED"
    FROZEN = "FROZEN"
    BLOCKED = "BLOCKED"


class UpdatePolicyAction(Enum):
    """Update-policy action."""

    APPLY = "APPLY"
    APPLY_REDUCED = "APPLY_REDUCED"
    SKIP = "SKIP"
    DEFER = "DEFER"
    BLOCK = "BLOCK"


class UpdatePolicyStatus(Enum):
    """Update-policy status."""

    READY = "READY"
    REDUCED = "REDUCED"
    SKIPPED = "SKIPPED"
    DEFERRED = "DEFERRED"
    BLOCKED = "BLOCKED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class UpdatePolicyConfig:
    """Stable configuration for update-policy decisions."""

    min_precision_to_apply: float = 0.35
    min_step_size_to_apply: float = 1e-12
    max_risk_rank_to_apply: int = 2
    allow_reduced_updates: bool = True
    allow_deferred_updates: bool = True
    require_stability_pass: bool = True
    require_transition_not_blocked: bool = True

    def __post_init__(self) -> None:
        min_precision_to_apply = ensure_constraint_probability(
            self.min_precision_to_apply,
            field_name="min_precision_to_apply",
        )
        min_step_size_to_apply = ensure_non_negative_number(
            self.min_step_size_to_apply,
            field_name="min_step_size_to_apply",
        )
        max_risk_rank_to_apply = _ensure_risk_rank_int(
            self.max_risk_rank_to_apply,
            "max_risk_rank_to_apply",
        )
        if not isinstance(self.allow_reduced_updates, bool):
            raise TypeError("allow_reduced_updates must be bool")
        if not isinstance(self.allow_deferred_updates, bool):
            raise TypeError("allow_deferred_updates must be bool")
        if not isinstance(self.require_stability_pass, bool):
            raise TypeError("require_stability_pass must be bool")
        if not isinstance(self.require_transition_not_blocked, bool):
            raise TypeError("require_transition_not_blocked must be bool")

        object.__setattr__(
            self,
            "min_precision_to_apply",
            min_precision_to_apply,
        )
        object.__setattr__(
            self,
            "min_step_size_to_apply",
            min_step_size_to_apply,
        )
        object.__setattr__(
            self,
            "max_risk_rank_to_apply",
            max_risk_rank_to_apply,
        )


@dataclass(frozen=True)
class UpdateDecision:
    """Serializable-friendly result of one update-policy decision."""

    action: UpdatePolicyAction
    mode: UpdatePolicyMode
    status: UpdatePolicyStatus
    allowed: bool
    precision_value: float
    step_size_value: float
    risk_score: float
    confidence: float
    reason_code: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.action, UpdatePolicyAction):
            raise TypeError("action must be UpdatePolicyAction")
        if not isinstance(self.mode, UpdatePolicyMode):
            raise TypeError("mode must be UpdatePolicyMode")
        if not isinstance(self.status, UpdatePolicyStatus):
            raise TypeError("status must be UpdatePolicyStatus")
        if not isinstance(self.allowed, bool):
            raise TypeError("allowed must be bool")

        precision_value = ensure_constraint_probability(
            self.precision_value,
            field_name="precision_value",
        )
        step_size_value = ensure_non_negative_number(
            self.step_size_value,
            field_name="step_size_value",
        )
        risk_score = ensure_constraint_probability(
            self.risk_score,
            field_name="risk_score",
        )
        confidence = ensure_constraint_probability(
            self.confidence,
            field_name="confidence",
        )
        if not isinstance(self.reason_code, str) or not self.reason_code:
            raise ValueError("reason_code must be a non-empty string")
        _ensure_timezone_aware(self.created_at, "created_at")

        object.__setattr__(self, "precision_value", precision_value)
        object.__setattr__(self, "step_size_value", step_size_value)
        object.__setattr__(self, "risk_score", risk_score)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "action": self.action.value,
            "mode": self.mode.value,
            "status": self.status.value,
            "allowed": self.allowed,
            "precision_value": self.precision_value,
            "step_size_value": self.step_size_value,
            "risk_score": self.risk_score,
            "confidence": self.confidence,
            "reason_code": self.reason_code,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class UpdatePolicyState:
    """Runtime descriptor of update-policy state."""

    last_action: UpdatePolicyAction
    last_status: UpdatePolicyStatus
    allowed: bool
    update_count: int
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.last_action, UpdatePolicyAction):
            raise TypeError("last_action must be UpdatePolicyAction")
        if not isinstance(self.last_status, UpdatePolicyStatus):
            raise TypeError("last_status must be UpdatePolicyStatus")
        if not isinstance(self.allowed, bool):
            raise TypeError("allowed must be bool")
        update_count = _ensure_non_negative_int(self.update_count, "update_count")
        _ensure_timezone_aware(self.created_at, "created_at")
        _ensure_timezone_aware(self.updated_at, "updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be greater than or equal to created_at")

        object.__setattr__(self, "update_count", update_count)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "last_action": self.last_action.value,
            "last_status": self.last_status.value,
            "allowed": self.allowed,
            "update_count": self.update_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }


def risk_score_from_report(
    report: StabilityReport | None,
) -> float:
    """Return a normalized risk score from a stability report."""

    if report is None:
        return 0.0
    if not isinstance(report, StabilityReport):
        raise TypeError("report must be StabilityReport or None")
    return ensure_constraint_probability(
        risk_rank(report.risk) / 4.0,
        field_name="risk_score",
    )


def update_mode_from_inputs(
    *,
    precision: PrecisionState | None = None,
    step_size: StepSizeState | None = None,
    temperature: SemanticTemperatureState | None = None,
    stability: StabilityReport | None = None,
    energy: EnergyState | None = None,
    transition: DynamicsTransitionResult | None = None,
    config: UpdatePolicyConfig | None = None,
) -> UpdatePolicyMode:
    """Determine update-policy mode from runtime inputs."""

    if config is None:
        config = UpdatePolicyConfig()
    if not isinstance(config, UpdatePolicyConfig):
        raise TypeError("config must be UpdatePolicyConfig")
    _validate_optional_inputs(
        precision=precision,
        step_size=step_size,
        temperature=temperature,
        stability=stability,
        energy=energy,
        transition=transition,
    )

    if temperature is not None and semantic_temperature_is_frozen(temperature):
        return UpdatePolicyMode.FROZEN
    if precision is not None and not precision_allows_update(precision):
        return UpdatePolicyMode.BLOCKED
    if step_size is not None and not step_size_allows_update(step_size):
        if step_size.status is StepSizeStatus.SKIPPED:
            return UpdatePolicyMode.DEFERRED
        return UpdatePolicyMode.BLOCKED
    if stability is not None and stability_report_is_blocked(stability):
        return UpdatePolicyMode.BLOCKED
    if transition is not None and transition_is_blocked(transition):
        return UpdatePolicyMode.BLOCKED
    if energy is not None and energy.status is EnergyStatus.INVALID:
        return UpdatePolicyMode.BLOCKED
    if stability is not None and risk_rank(stability.risk) > config.max_risk_rank_to_apply:
        return UpdatePolicyMode.CONSERVATIVE
    if energy is not None and energy.status is EnergyStatus.DEGRADED:
        return UpdatePolicyMode.CONSERVATIVE
    if (
        transition is not None
        and transition.transition.status is DynamicsTransitionStatus.DEGRADED
    ):
        return UpdatePolicyMode.CONSERVATIVE
    if precision is not None and precision.status is PrecisionStatus.HIGH:
        return UpdatePolicyMode.AGGRESSIVE
    return UpdatePolicyMode.NORMAL


def update_action_from_mode(
    *,
    mode: UpdatePolicyMode,
    precision_value: float,
    step_size_value: float,
    stability: StabilityReport | None,
    transition: DynamicsTransitionResult | None,
    config: UpdatePolicyConfig,
) -> UpdatePolicyAction:
    """Map mode and thresholds to an update action."""

    if not isinstance(mode, UpdatePolicyMode):
        raise TypeError("mode must be UpdatePolicyMode")
    precision_value = ensure_constraint_probability(
        precision_value,
        field_name="precision_value",
    )
    step_size_value = ensure_non_negative_number(
        step_size_value,
        field_name="step_size_value",
    )
    if stability is not None and not isinstance(stability, StabilityReport):
        raise TypeError("stability must be StabilityReport or None")
    if transition is not None and not isinstance(transition, DynamicsTransitionResult):
        raise TypeError("transition must be DynamicsTransitionResult or None")
    if not isinstance(config, UpdatePolicyConfig):
        raise TypeError("config must be UpdatePolicyConfig")

    if mode is UpdatePolicyMode.BLOCKED:
        return UpdatePolicyAction.BLOCK
    if mode is UpdatePolicyMode.FROZEN:
        return (
            UpdatePolicyAction.DEFER
            if config.allow_deferred_updates
            else UpdatePolicyAction.BLOCK
        )
    if mode is UpdatePolicyMode.DEFERRED:
        return (
            UpdatePolicyAction.DEFER
            if config.allow_deferred_updates
            else UpdatePolicyAction.SKIP
        )
    if precision_value < config.min_precision_to_apply:
        return (
            UpdatePolicyAction.DEFER
            if config.allow_deferred_updates
            else UpdatePolicyAction.BLOCK
        )
    if step_size_value <= config.min_step_size_to_apply:
        return UpdatePolicyAction.SKIP
    if (
        config.require_stability_pass
        and stability is not None
        and not stability_report_is_passed(stability)
    ):
        return (
            UpdatePolicyAction.APPLY_REDUCED
            if config.allow_reduced_updates
            else UpdatePolicyAction.BLOCK
        )
    if (
        config.require_transition_not_blocked
        and transition is not None
        and transition_is_blocked(transition)
    ):
        return UpdatePolicyAction.BLOCK
    if mode is UpdatePolicyMode.CONSERVATIVE:
        return (
            UpdatePolicyAction.APPLY_REDUCED
            if config.allow_reduced_updates
            else UpdatePolicyAction.APPLY
        )
    return UpdatePolicyAction.APPLY


def update_status_from_action(
    action: UpdatePolicyAction,
) -> UpdatePolicyStatus:
    """Map an update action to status."""

    if not isinstance(action, UpdatePolicyAction):
        raise TypeError("action must be UpdatePolicyAction")
    if action is UpdatePolicyAction.APPLY:
        return UpdatePolicyStatus.READY
    if action is UpdatePolicyAction.APPLY_REDUCED:
        return UpdatePolicyStatus.REDUCED
    if action is UpdatePolicyAction.SKIP:
        return UpdatePolicyStatus.SKIPPED
    if action is UpdatePolicyAction.DEFER:
        return UpdatePolicyStatus.DEFERRED
    return UpdatePolicyStatus.BLOCKED


def update_allowed_from_action(
    action: UpdatePolicyAction,
) -> bool:
    """Return True only when an action applies an update."""

    if not isinstance(action, UpdatePolicyAction):
        raise TypeError("action must be UpdatePolicyAction")
    return action in (UpdatePolicyAction.APPLY, UpdatePolicyAction.APPLY_REDUCED)


def reason_code_from_decision(
    *,
    action: UpdatePolicyAction,
    mode: UpdatePolicyMode,
) -> str:
    """Build a deterministic reason code."""

    if not isinstance(action, UpdatePolicyAction):
        raise TypeError("action must be UpdatePolicyAction")
    if not isinstance(mode, UpdatePolicyMode):
        raise TypeError("mode must be UpdatePolicyMode")
    return f"UPDATE_{action.value}_{mode.value}"


def compute_update_decision(
    *,
    precision: PrecisionState | None = None,
    step_size: StepSizeState | None = None,
    temperature: SemanticTemperatureState | None = None,
    stability: StabilityReport | None = None,
    energy: EnergyState | None = None,
    transition: DynamicsTransitionResult | None = None,
    config: UpdatePolicyConfig | None = None,
    metadata: dict[str, object] | None = None,
) -> UpdateDecision:
    """Compute deterministic update decision from regulation/runtime inputs."""

    if config is None:
        config = UpdatePolicyConfig()
    if not isinstance(config, UpdatePolicyConfig):
        raise TypeError("config must be UpdatePolicyConfig")
    _validate_optional_inputs(
        precision=precision,
        step_size=step_size,
        temperature=temperature,
        stability=stability,
        energy=energy,
        transition=transition,
    )

    precision_value = (
        ensure_constraint_probability(
            precision_scale(precision),
            field_name="precision_value",
        )
        if precision is not None
        else 1.0
    )
    step_size_value = (
        ensure_non_negative_number(
            step_size_scale(step_size),
            field_name="step_size_value",
        )
        if step_size is not None
        else 0.0
    )
    risk_score = risk_score_from_report(stability)
    mode = update_mode_from_inputs(
        precision=precision,
        step_size=step_size,
        temperature=temperature,
        stability=stability,
        energy=energy,
        transition=transition,
        config=config,
    )
    action = update_action_from_mode(
        mode=mode,
        precision_value=precision_value,
        step_size_value=step_size_value,
        stability=stability,
        transition=transition,
        config=config,
    )
    status = update_status_from_action(action)
    allowed = update_allowed_from_action(action)
    confidence = _clamp_probability(precision_value * (1.0 - risk_score))
    reason_code = reason_code_from_decision(action=action, mode=mode)

    default_metadata: dict[str, object] = {
        "source": "update_policy",
        "has_precision": precision is not None,
        "has_step_size": step_size is not None,
        "has_temperature": temperature is not None,
        "has_stability": stability is not None,
        "has_energy": energy is not None,
        "has_transition": transition is not None,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "update_policy"

    return UpdateDecision(
        action=action,
        mode=mode,
        status=status,
        allowed=allowed,
        precision_value=precision_value,
        step_size_value=step_size_value,
        risk_score=risk_score,
        confidence=confidence,
        reason_code=reason_code,
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def create_update_policy_state(
    *,
    decision: UpdateDecision | None = None,
    config: UpdatePolicyConfig | None = None,
    metadata: dict[str, object] | None = None,
) -> UpdatePolicyState:
    """Create initial update-policy state."""

    if decision is None:
        decision = compute_update_decision(config=config)
    if not isinstance(decision, UpdateDecision):
        raise TypeError("decision must be UpdateDecision")

    timestamp = utc_now()
    default_metadata: dict[str, object] = {"state_kind": "update_policy_state"}
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["state_kind"] = "update_policy_state"

    return UpdatePolicyState(
        last_action=decision.action,
        last_status=decision.status,
        allowed=decision.allowed,
        update_count=0,
        created_at=timestamp,
        updated_at=timestamp,
        metadata=merged_metadata,
    )


def transition_update_policy_state(
    *,
    state: UpdatePolicyState,
    decision: UpdateDecision,
    metadata: dict[str, object] | None = None,
) -> UpdatePolicyState:
    """Create next update-policy state without mutating prior state."""

    if not isinstance(state, UpdatePolicyState):
        raise TypeError("state must be UpdatePolicyState")
    if not isinstance(decision, UpdateDecision):
        raise TypeError("decision must be UpdateDecision")

    default_metadata: dict[str, object] = {
        "previous_action": state.last_action.value,
        "previous_status": state.last_status.value,
        "decision_reason_code": decision.reason_code,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)

    return UpdatePolicyState(
        last_action=decision.action,
        last_status=decision.status,
        allowed=decision.allowed,
        update_count=state.update_count + 1,
        created_at=state.created_at,
        updated_at=utc_now(),
        metadata=merged_metadata,
    )


def update_policy_allows_apply(state: UpdatePolicyState) -> bool:
    """Return whether the update-policy state allows apply."""

    if not isinstance(state, UpdatePolicyState):
        raise TypeError("state must be UpdatePolicyState")
    return state.allowed


def update_decision_allows_apply(decision: UpdateDecision) -> bool:
    """Return whether the update decision allows apply."""

    if not isinstance(decision, UpdateDecision):
        raise TypeError("decision must be UpdateDecision")
    return decision.allowed


def _clamp_probability(value: float) -> float:
    value = ensure_finite_number(value, field_name="value")
    return min(max(value, 0.0), 1.0)


def _copy_metadata(metadata: dict[str, object]) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(metadata)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
    return copied


def _merge_metadata(
    default_metadata: dict[str, object],
    caller_metadata: dict[str, object] | None,
) -> dict[str, object]:
    merged = _copy_metadata(default_metadata)
    if caller_metadata is not None:
        merged.update(_copy_metadata(caller_metadata))
    return merged


def _ensure_non_negative_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _ensure_risk_rank_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value < 0 or value > 4:
        raise ValueError(f"{field_name} must be in range 0..4")
    return value


def _ensure_timezone_aware(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _validate_optional_inputs(
    *,
    precision: PrecisionState | None,
    step_size: StepSizeState | None,
    temperature: SemanticTemperatureState | None,
    stability: StabilityReport | None,
    energy: EnergyState | None,
    transition: DynamicsTransitionResult | None,
) -> None:
    if precision is not None and not isinstance(precision, PrecisionState):
        raise TypeError("precision must be PrecisionState or None")
    if step_size is not None and not isinstance(step_size, StepSizeState):
        raise TypeError("step_size must be StepSizeState or None")
    if temperature is not None and not isinstance(temperature, SemanticTemperatureState):
        raise TypeError("temperature must be SemanticTemperatureState or None")
    if stability is not None and not isinstance(stability, StabilityReport):
        raise TypeError("stability must be StabilityReport or None")
    if energy is not None and not isinstance(energy, EnergyState):
        raise TypeError("energy must be EnergyState or None")
    if transition is not None and not isinstance(transition, DynamicsTransitionResult):
        raise TypeError("transition must be DynamicsTransitionResult or None")


__all__ = [
    "UpdatePolicyMode",
    "UpdatePolicyAction",
    "UpdatePolicyStatus",
    "UpdatePolicyConfig",
    "UpdateDecision",
    "UpdatePolicyState",
    "risk_score_from_report",
    "update_mode_from_inputs",
    "update_action_from_mode",
    "update_status_from_action",
    "update_allowed_from_action",
    "reason_code_from_decision",
    "compute_update_decision",
    "create_update_policy_state",
    "transition_update_policy_state",
    "update_policy_allows_apply",
    "update_decision_allows_apply",
]
