"""Deterministic consolidation-policy utilities for CEBA Core Regulation."""

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
)
from ceba.core.dynamics.transition import (
    DynamicsTransitionResult,
    transition_is_applied,
    transition_is_blocked,
)
from ceba.core.metrics.schema import (
    ensure_constraint_probability,
    ensure_finite_number,
    utc_now,
)
from ceba.core.regulation.update_policy import (
    UpdateDecision,
    UpdatePolicyAction,
    update_decision_allows_apply,
)


class ConsolidationMode(Enum):
    """Consolidation-policy operating mode."""

    ELIGIBLE = "ELIGIBLE"
    CONSERVATIVE = "CONSERVATIVE"
    DEFERRED = "DEFERRED"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"


class ConsolidationAction(Enum):
    """Consolidation-policy action."""

    MARK_ELIGIBLE = "MARK_ELIGIBLE"
    MARK_REVIEW = "MARK_REVIEW"
    DEFER = "DEFER"
    REJECT = "REJECT"
    BLOCK = "BLOCK"


class ConsolidationStatus(Enum):
    """Consolidation-policy status."""

    ELIGIBLE = "ELIGIBLE"
    REVIEW = "REVIEW"
    DEFERRED = "DEFERRED"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class ConsolidationPolicyConfig:
    """Stable configuration for consolidation eligibility decisions."""

    min_stability_score: float = 0.75
    max_risk_rank: int = 2
    max_normalized_energy: float = 0.70
    allow_review_for_reduced_update: bool = True
    allow_deferred_consolidation: bool = True
    require_update_allowed: bool = True
    require_transition_applied: bool = True
    reject_frozen_temperature: bool = False

    def __post_init__(self) -> None:
        min_stability_score = ensure_constraint_probability(
            self.min_stability_score,
            field_name="min_stability_score",
        )
        max_risk_rank = _ensure_risk_rank_int(self.max_risk_rank, "max_risk_rank")
        max_normalized_energy = ensure_constraint_probability(
            self.max_normalized_energy,
            field_name="max_normalized_energy",
        )
        for field_name in (
            "allow_review_for_reduced_update",
            "allow_deferred_consolidation",
            "require_update_allowed",
            "require_transition_applied",
            "reject_frozen_temperature",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")

        object.__setattr__(self, "min_stability_score", min_stability_score)
        object.__setattr__(self, "max_risk_rank", max_risk_rank)
        object.__setattr__(self, "max_normalized_energy", max_normalized_energy)


@dataclass(frozen=True)
class ConsolidationDecision:
    """Serializable-friendly result of one consolidation-policy decision."""

    action: ConsolidationAction
    mode: ConsolidationMode
    status: ConsolidationStatus
    eligible: bool
    stability_score: float
    risk_score: float
    normalized_energy: float | None
    confidence: float
    reason_code: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.action, ConsolidationAction):
            raise TypeError("action must be ConsolidationAction")
        if not isinstance(self.mode, ConsolidationMode):
            raise TypeError("mode must be ConsolidationMode")
        if not isinstance(self.status, ConsolidationStatus):
            raise TypeError("status must be ConsolidationStatus")
        if not isinstance(self.eligible, bool):
            raise TypeError("eligible must be bool")

        stability_score = ensure_constraint_probability(
            self.stability_score,
            field_name="stability_score",
        )
        risk_score = ensure_constraint_probability(
            self.risk_score,
            field_name="risk_score",
        )
        normalized_energy = (
            None
            if self.normalized_energy is None
            else ensure_constraint_probability(
                self.normalized_energy,
                field_name="normalized_energy",
            )
        )
        confidence = ensure_constraint_probability(
            self.confidence,
            field_name="confidence",
        )
        if not isinstance(self.reason_code, str) or not self.reason_code:
            raise ValueError("reason_code must be a non-empty string")
        _ensure_timezone_aware(self.created_at, "created_at")

        object.__setattr__(self, "stability_score", stability_score)
        object.__setattr__(self, "risk_score", risk_score)
        object.__setattr__(self, "normalized_energy", normalized_energy)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "action": self.action.value,
            "mode": self.mode.value,
            "status": self.status.value,
            "eligible": self.eligible,
            "stability_score": self.stability_score,
            "risk_score": self.risk_score,
            "normalized_energy": self.normalized_energy,
            "confidence": self.confidence,
            "reason_code": self.reason_code,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ConsolidationPolicyState:
    """Runtime descriptor of consolidation-policy state."""

    last_action: ConsolidationAction
    last_status: ConsolidationStatus
    eligible: bool
    update_count: int
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.last_action, ConsolidationAction):
            raise TypeError("last_action must be ConsolidationAction")
        if not isinstance(self.last_status, ConsolidationStatus):
            raise TypeError("last_status must be ConsolidationStatus")
        if not isinstance(self.eligible, bool):
            raise TypeError("eligible must be bool")
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
            "eligible": self.eligible,
            "update_count": self.update_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }


def risk_score_from_report(report: StabilityReport | None) -> float:
    """Return a normalized risk score from a stability report."""

    if report is None:
        return 0.0
    if not isinstance(report, StabilityReport):
        raise TypeError("report must be StabilityReport or None")
    return ensure_constraint_probability(
        risk_rank(report.risk) / 4.0,
        field_name="risk_score",
    )


def stability_score_from_report(report: StabilityReport | None) -> float:
    """Return a stability score from a stability report."""

    if report is None:
        return 1.0
    if not isinstance(report, StabilityReport):
        raise TypeError("report must be StabilityReport or None")
    return ensure_constraint_probability(report.score, field_name="stability_score")


def normalized_energy_value(energy: EnergyState | None) -> float | None:
    """Return normalized energy when available."""

    if energy is None:
        return None
    if not isinstance(energy, EnergyState):
        raise TypeError("energy must be EnergyState or None")
    if energy.normalized_energy is None:
        return None
    return ensure_constraint_probability(
        energy.normalized_energy,
        field_name="normalized_energy",
    )


def consolidation_mode_from_inputs(
    *,
    update_decision: UpdateDecision | None = None,
    transition: DynamicsTransitionResult | None = None,
    stability: StabilityReport | None = None,
    energy: EnergyState | None = None,
    temperature: SemanticTemperatureState | None = None,
    config: ConsolidationPolicyConfig | None = None,
) -> ConsolidationMode:
    """Determine consolidation mode from runtime inputs."""

    if config is None:
        config = ConsolidationPolicyConfig()
    if not isinstance(config, ConsolidationPolicyConfig):
        raise TypeError("config must be ConsolidationPolicyConfig")
    _validate_optional_inputs(
        update_decision=update_decision,
        transition=transition,
        stability=stability,
        energy=energy,
        temperature=temperature,
    )

    if (
        update_decision is not None
        and config.require_update_allowed
        and not update_decision_allows_apply(update_decision)
    ):
        return ConsolidationMode.BLOCKED
    if transition is not None and transition_is_blocked(transition):
        return ConsolidationMode.BLOCKED
    if (
        transition is not None
        and config.require_transition_applied
        and not transition_is_applied(transition)
    ):
        return ConsolidationMode.DEFERRED
    if stability is not None and stability_report_is_blocked(stability):
        return ConsolidationMode.BLOCKED
    if stability is not None and risk_rank(stability.risk) > config.max_risk_rank:
        return ConsolidationMode.CONSERVATIVE
    if stability is not None and stability.score < config.min_stability_score:
        return ConsolidationMode.CONSERVATIVE
    if energy is not None and energy.status is EnergyStatus.INVALID:
        return ConsolidationMode.BLOCKED
    if (
        energy is not None
        and energy.normalized_energy is not None
        and energy.normalized_energy > config.max_normalized_energy
    ):
        return ConsolidationMode.CONSERVATIVE
    if temperature is not None and semantic_temperature_is_frozen(temperature):
        if config.reject_frozen_temperature:
            return ConsolidationMode.REJECTED
        return ConsolidationMode.DEFERRED
    return ConsolidationMode.ELIGIBLE


def consolidation_action_from_mode(
    *,
    mode: ConsolidationMode,
    update_decision: UpdateDecision | None,
    config: ConsolidationPolicyConfig,
) -> ConsolidationAction:
    """Map consolidation mode to action."""

    if not isinstance(mode, ConsolidationMode):
        raise TypeError("mode must be ConsolidationMode")
    if update_decision is not None and not isinstance(update_decision, UpdateDecision):
        raise TypeError("update_decision must be UpdateDecision or None")
    if not isinstance(config, ConsolidationPolicyConfig):
        raise TypeError("config must be ConsolidationPolicyConfig")

    if mode is ConsolidationMode.BLOCKED:
        return ConsolidationAction.BLOCK
    if mode is ConsolidationMode.REJECTED:
        return ConsolidationAction.REJECT
    if mode is ConsolidationMode.DEFERRED:
        return (
            ConsolidationAction.DEFER
            if config.allow_deferred_consolidation
            else ConsolidationAction.REJECT
        )
    if mode is ConsolidationMode.CONSERVATIVE:
        if (
            update_decision is not None
            and update_decision.action is UpdatePolicyAction.APPLY_REDUCED
            and config.allow_review_for_reduced_update
        ):
            return ConsolidationAction.MARK_REVIEW
        return ConsolidationAction.MARK_REVIEW
    return ConsolidationAction.MARK_ELIGIBLE


def consolidation_status_from_action(
    action: ConsolidationAction,
) -> ConsolidationStatus:
    """Map a consolidation action to status."""

    if not isinstance(action, ConsolidationAction):
        raise TypeError("action must be ConsolidationAction")
    if action is ConsolidationAction.MARK_ELIGIBLE:
        return ConsolidationStatus.ELIGIBLE
    if action is ConsolidationAction.MARK_REVIEW:
        return ConsolidationStatus.REVIEW
    if action is ConsolidationAction.DEFER:
        return ConsolidationStatus.DEFERRED
    if action is ConsolidationAction.REJECT:
        return ConsolidationStatus.REJECTED
    return ConsolidationStatus.BLOCKED


def consolidation_eligible_from_action(action: ConsolidationAction) -> bool:
    """Return True only for mark-eligible actions."""

    if not isinstance(action, ConsolidationAction):
        raise TypeError("action must be ConsolidationAction")
    return action is ConsolidationAction.MARK_ELIGIBLE


def reason_code_from_decision(
    *,
    action: ConsolidationAction,
    mode: ConsolidationMode,
) -> str:
    """Build a deterministic reason code."""

    if not isinstance(action, ConsolidationAction):
        raise TypeError("action must be ConsolidationAction")
    if not isinstance(mode, ConsolidationMode):
        raise TypeError("mode must be ConsolidationMode")
    return f"CONSOLIDATION_{action.value}_{mode.value}"


def compute_consolidation_decision(
    *,
    update_decision: UpdateDecision | None = None,
    transition: DynamicsTransitionResult | None = None,
    stability: StabilityReport | None = None,
    energy: EnergyState | None = None,
    temperature: SemanticTemperatureState | None = None,
    config: ConsolidationPolicyConfig | None = None,
    metadata: dict[str, object] | None = None,
) -> ConsolidationDecision:
    """Compute deterministic consolidation eligibility decision."""

    if config is None:
        config = ConsolidationPolicyConfig()
    if not isinstance(config, ConsolidationPolicyConfig):
        raise TypeError("config must be ConsolidationPolicyConfig")
    _validate_optional_inputs(
        update_decision=update_decision,
        transition=transition,
        stability=stability,
        energy=energy,
        temperature=temperature,
    )

    stability_score = stability_score_from_report(stability)
    risk_score = risk_score_from_report(stability)
    normalized_energy = normalized_energy_value(energy)
    mode = consolidation_mode_from_inputs(
        update_decision=update_decision,
        transition=transition,
        stability=stability,
        energy=energy,
        temperature=temperature,
        config=config,
    )
    action = consolidation_action_from_mode(
        mode=mode,
        update_decision=update_decision,
        config=config,
    )
    status = consolidation_status_from_action(action)
    eligible = consolidation_eligible_from_action(action)

    confidence = stability_score * (1.0 - risk_score)
    if normalized_energy is not None:
        confidence *= 1.0 - normalized_energy
    confidence = _clamp_probability(confidence)
    reason_code = reason_code_from_decision(action=action, mode=mode)

    default_metadata: dict[str, object] = {
        "source": "consolidation_policy",
        "has_update_decision": update_decision is not None,
        "has_transition": transition is not None,
        "has_stability": stability is not None,
        "has_energy": energy is not None,
        "has_temperature": temperature is not None,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "consolidation_policy"

    return ConsolidationDecision(
        action=action,
        mode=mode,
        status=status,
        eligible=eligible,
        stability_score=stability_score,
        risk_score=risk_score,
        normalized_energy=normalized_energy,
        confidence=confidence,
        reason_code=reason_code,
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def create_consolidation_policy_state(
    *,
    decision: ConsolidationDecision | None = None,
    config: ConsolidationPolicyConfig | None = None,
    metadata: dict[str, object] | None = None,
) -> ConsolidationPolicyState:
    """Create initial consolidation-policy state."""

    if decision is None:
        decision = compute_consolidation_decision(config=config)
    if not isinstance(decision, ConsolidationDecision):
        raise TypeError("decision must be ConsolidationDecision")

    timestamp = utc_now()
    default_metadata: dict[str, object] = {
        "state_kind": "consolidation_policy_state",
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["state_kind"] = "consolidation_policy_state"

    return ConsolidationPolicyState(
        last_action=decision.action,
        last_status=decision.status,
        eligible=decision.eligible,
        update_count=0,
        created_at=timestamp,
        updated_at=timestamp,
        metadata=merged_metadata,
    )


def transition_consolidation_policy_state(
    *,
    state: ConsolidationPolicyState,
    decision: ConsolidationDecision,
    metadata: dict[str, object] | None = None,
) -> ConsolidationPolicyState:
    """Create next consolidation-policy state without mutating prior state."""

    if not isinstance(state, ConsolidationPolicyState):
        raise TypeError("state must be ConsolidationPolicyState")
    if not isinstance(decision, ConsolidationDecision):
        raise TypeError("decision must be ConsolidationDecision")

    default_metadata: dict[str, object] = {
        "previous_action": state.last_action.value,
        "previous_status": state.last_status.value,
        "decision_reason_code": decision.reason_code,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)

    return ConsolidationPolicyState(
        last_action=decision.action,
        last_status=decision.status,
        eligible=decision.eligible,
        update_count=state.update_count + 1,
        created_at=state.created_at,
        updated_at=utc_now(),
        metadata=merged_metadata,
    )


def consolidation_policy_allows_consolidation(
    state: ConsolidationPolicyState,
) -> bool:
    """Return whether the consolidation-policy state allows consolidation."""

    if not isinstance(state, ConsolidationPolicyState):
        raise TypeError("state must be ConsolidationPolicyState")
    return state.eligible


def consolidation_decision_allows_consolidation(
    decision: ConsolidationDecision,
) -> bool:
    """Return whether the consolidation decision allows consolidation."""

    if not isinstance(decision, ConsolidationDecision):
        raise TypeError("decision must be ConsolidationDecision")
    return decision.eligible


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
    update_decision: UpdateDecision | None,
    transition: DynamicsTransitionResult | None,
    stability: StabilityReport | None,
    energy: EnergyState | None,
    temperature: SemanticTemperatureState | None,
) -> None:
    if update_decision is not None and not isinstance(update_decision, UpdateDecision):
        raise TypeError("update_decision must be UpdateDecision or None")
    if transition is not None and not isinstance(transition, DynamicsTransitionResult):
        raise TypeError("transition must be DynamicsTransitionResult or None")
    if stability is not None and not isinstance(stability, StabilityReport):
        raise TypeError("stability must be StabilityReport or None")
    if energy is not None and not isinstance(energy, EnergyState):
        raise TypeError("energy must be EnergyState or None")
    if temperature is not None and not isinstance(
        temperature,
        SemanticTemperatureState,
    ):
        raise TypeError("temperature must be SemanticTemperatureState or None")


__all__ = [
    "ConsolidationMode",
    "ConsolidationAction",
    "ConsolidationStatus",
    "ConsolidationPolicyConfig",
    "ConsolidationDecision",
    "ConsolidationPolicyState",
    "risk_score_from_report",
    "stability_score_from_report",
    "normalized_energy_value",
    "consolidation_mode_from_inputs",
    "consolidation_action_from_mode",
    "consolidation_status_from_action",
    "consolidation_eligible_from_action",
    "reason_code_from_decision",
    "compute_consolidation_decision",
    "create_consolidation_policy_state",
    "transition_consolidation_policy_state",
    "consolidation_policy_allows_consolidation",
    "consolidation_decision_allows_consolidation",
]
