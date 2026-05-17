"""Deterministic replay-policy utilities for CEBA Core Regulation."""

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
from ceba.core.regulation.consolidation_policy import (
    ConsolidationDecision,
    consolidation_decision_allows_consolidation,
)
from ceba.core.regulation.update_policy import (
    UpdateDecision,
    update_decision_allows_apply,
)


class ReplayMode(Enum):
    """Replay-policy operating mode."""

    ELIGIBLE = "ELIGIBLE"
    PRIORITIZED = "PRIORITIZED"
    CONSERVATIVE = "CONSERVATIVE"
    DEFERRED = "DEFERRED"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"


class ReplayAction(Enum):
    """Replay-policy action."""

    MARK_REPLAY_ELIGIBLE = "MARK_REPLAY_ELIGIBLE"
    MARK_PRIORITY_REPLAY = "MARK_PRIORITY_REPLAY"
    MARK_REVIEW = "MARK_REVIEW"
    DEFER = "DEFER"
    REJECT = "REJECT"
    BLOCK = "BLOCK"


class ReplayStatus(Enum):
    """Replay-policy status."""

    ELIGIBLE = "ELIGIBLE"
    PRIORITY = "PRIORITY"
    REVIEW = "REVIEW"
    DEFERRED = "DEFERRED"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class ReplayPolicyConfig:
    """Stable configuration for replay eligibility decisions."""

    min_stability_score: float = 0.80
    min_confidence: float = 0.60
    max_risk_rank: int = 2
    max_normalized_energy: float = 0.65
    priority_stability_score: float = 0.90
    priority_confidence: float = 0.85
    allow_review_for_conservative: bool = True
    allow_deferred_replay: bool = True
    require_consolidation_eligible: bool = True
    require_update_allowed: bool = True
    require_transition_applied: bool = True
    reject_frozen_temperature: bool = False

    def __post_init__(self) -> None:
        min_stability_score = ensure_constraint_probability(
            self.min_stability_score,
            field_name="min_stability_score",
        )
        min_confidence = ensure_constraint_probability(
            self.min_confidence,
            field_name="min_confidence",
        )
        max_risk_rank = _ensure_risk_rank_int(self.max_risk_rank, "max_risk_rank")
        max_normalized_energy = ensure_constraint_probability(
            self.max_normalized_energy,
            field_name="max_normalized_energy",
        )
        priority_stability_score = ensure_constraint_probability(
            self.priority_stability_score,
            field_name="priority_stability_score",
        )
        priority_confidence = ensure_constraint_probability(
            self.priority_confidence,
            field_name="priority_confidence",
        )
        for field_name in (
            "allow_review_for_conservative",
            "allow_deferred_replay",
            "require_consolidation_eligible",
            "require_update_allowed",
            "require_transition_applied",
            "reject_frozen_temperature",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")

        object.__setattr__(self, "min_stability_score", min_stability_score)
        object.__setattr__(self, "min_confidence", min_confidence)
        object.__setattr__(self, "max_risk_rank", max_risk_rank)
        object.__setattr__(self, "max_normalized_energy", max_normalized_energy)
        object.__setattr__(
            self,
            "priority_stability_score",
            priority_stability_score,
        )
        object.__setattr__(self, "priority_confidence", priority_confidence)


@dataclass(frozen=True)
class ReplayDecision:
    """Serializable-friendly result of one replay-policy decision."""

    action: ReplayAction
    mode: ReplayMode
    status: ReplayStatus
    eligible: bool
    priority: bool
    stability_score: float
    risk_score: float
    normalized_energy: float | None
    confidence: float
    reason_code: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.action, ReplayAction):
            raise TypeError("action must be ReplayAction")
        if not isinstance(self.mode, ReplayMode):
            raise TypeError("mode must be ReplayMode")
        if not isinstance(self.status, ReplayStatus):
            raise TypeError("status must be ReplayStatus")
        if not isinstance(self.eligible, bool):
            raise TypeError("eligible must be bool")
        if not isinstance(self.priority, bool):
            raise TypeError("priority must be bool")

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
            "priority": self.priority,
            "stability_score": self.stability_score,
            "risk_score": self.risk_score,
            "normalized_energy": self.normalized_energy,
            "confidence": self.confidence,
            "reason_code": self.reason_code,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ReplayPolicyState:
    """Runtime descriptor of replay-policy state."""

    last_action: ReplayAction
    last_status: ReplayStatus
    eligible: bool
    priority: bool
    update_count: int
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.last_action, ReplayAction):
            raise TypeError("last_action must be ReplayAction")
        if not isinstance(self.last_status, ReplayStatus):
            raise TypeError("last_status must be ReplayStatus")
        if not isinstance(self.eligible, bool):
            raise TypeError("eligible must be bool")
        if not isinstance(self.priority, bool):
            raise TypeError("priority must be bool")
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
            "priority": self.priority,
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


def replay_confidence_from_inputs(
    *,
    consolidation: ConsolidationDecision | None = None,
    update_decision: UpdateDecision | None = None,
    stability: StabilityReport | None = None,
    energy: EnergyState | None = None,
) -> float:
    """Compute deterministic replay confidence from available inputs."""

    _validate_confidence_inputs(
        consolidation=consolidation,
        update_decision=update_decision,
        stability=stability,
        energy=energy,
    )

    confidence = 1.0
    if consolidation is not None:
        confidence *= consolidation.confidence
    if update_decision is not None:
        confidence *= update_decision.confidence
    if stability is not None:
        confidence *= stability.score
    if energy is not None and energy.normalized_energy is not None:
        confidence *= 1.0 - energy.normalized_energy
    return _clamp_probability(confidence)


def replay_mode_from_inputs(
    *,
    consolidation: ConsolidationDecision | None = None,
    update_decision: UpdateDecision | None = None,
    transition: DynamicsTransitionResult | None = None,
    stability: StabilityReport | None = None,
    energy: EnergyState | None = None,
    temperature: SemanticTemperatureState | None = None,
    config: ReplayPolicyConfig | None = None,
) -> ReplayMode:
    """Determine replay mode from runtime inputs."""

    if config is None:
        config = ReplayPolicyConfig()
    if not isinstance(config, ReplayPolicyConfig):
        raise TypeError("config must be ReplayPolicyConfig")
    _validate_optional_inputs(
        consolidation=consolidation,
        update_decision=update_decision,
        transition=transition,
        stability=stability,
        energy=energy,
        temperature=temperature,
    )

    if (
        consolidation is not None
        and config.require_consolidation_eligible
        and not consolidation_decision_allows_consolidation(consolidation)
    ):
        return ReplayMode.BLOCKED
    if (
        update_decision is not None
        and config.require_update_allowed
        and not update_decision_allows_apply(update_decision)
    ):
        return ReplayMode.BLOCKED
    if transition is not None and transition_is_blocked(transition):
        return ReplayMode.BLOCKED
    if (
        transition is not None
        and config.require_transition_applied
        and not transition_is_applied(transition)
    ):
        return ReplayMode.DEFERRED
    if stability is not None and stability_report_is_blocked(stability):
        return ReplayMode.BLOCKED
    if energy is not None and energy.status is EnergyStatus.INVALID:
        return ReplayMode.BLOCKED
    if temperature is not None and semantic_temperature_is_frozen(temperature):
        if config.reject_frozen_temperature:
            return ReplayMode.REJECTED
        return ReplayMode.DEFERRED
    if stability is not None and risk_rank(stability.risk) > config.max_risk_rank:
        return ReplayMode.CONSERVATIVE
    if stability is not None and stability.score < config.min_stability_score:
        return ReplayMode.CONSERVATIVE
    if (
        energy is not None
        and energy.normalized_energy is not None
        and energy.normalized_energy > config.max_normalized_energy
    ):
        return ReplayMode.CONSERVATIVE

    confidence = replay_confidence_from_inputs(
        consolidation=consolidation,
        update_decision=update_decision,
        stability=stability,
        energy=energy,
    )
    if confidence < config.min_confidence:
        return ReplayMode.CONSERVATIVE

    stability_score = stability_score_from_report(stability)
    if (
        stability_score >= config.priority_stability_score
        and confidence >= config.priority_confidence
    ):
        return ReplayMode.PRIORITIZED
    return ReplayMode.ELIGIBLE


def replay_action_from_mode(
    *,
    mode: ReplayMode,
    config: ReplayPolicyConfig,
) -> ReplayAction:
    """Map replay mode to action."""

    if not isinstance(mode, ReplayMode):
        raise TypeError("mode must be ReplayMode")
    if not isinstance(config, ReplayPolicyConfig):
        raise TypeError("config must be ReplayPolicyConfig")

    if mode is ReplayMode.BLOCKED:
        return ReplayAction.BLOCK
    if mode is ReplayMode.REJECTED:
        return ReplayAction.REJECT
    if mode is ReplayMode.DEFERRED:
        return (
            ReplayAction.DEFER
            if config.allow_deferred_replay
            else ReplayAction.REJECT
        )
    if mode is ReplayMode.CONSERVATIVE:
        return (
            ReplayAction.MARK_REVIEW
            if config.allow_review_for_conservative
            else ReplayAction.REJECT
        )
    if mode is ReplayMode.PRIORITIZED:
        return ReplayAction.MARK_PRIORITY_REPLAY
    return ReplayAction.MARK_REPLAY_ELIGIBLE


def replay_status_from_action(action: ReplayAction) -> ReplayStatus:
    """Map a replay action to status."""

    if not isinstance(action, ReplayAction):
        raise TypeError("action must be ReplayAction")
    if action is ReplayAction.MARK_REPLAY_ELIGIBLE:
        return ReplayStatus.ELIGIBLE
    if action is ReplayAction.MARK_PRIORITY_REPLAY:
        return ReplayStatus.PRIORITY
    if action is ReplayAction.MARK_REVIEW:
        return ReplayStatus.REVIEW
    if action is ReplayAction.DEFER:
        return ReplayStatus.DEFERRED
    if action is ReplayAction.REJECT:
        return ReplayStatus.REJECTED
    return ReplayStatus.BLOCKED


def replay_eligible_from_action(action: ReplayAction) -> bool:
    """Return True for replay-eligible actions."""

    if not isinstance(action, ReplayAction):
        raise TypeError("action must be ReplayAction")
    return action in (
        ReplayAction.MARK_REPLAY_ELIGIBLE,
        ReplayAction.MARK_PRIORITY_REPLAY,
    )


def replay_priority_from_action(action: ReplayAction) -> bool:
    """Return True only for priority replay actions."""

    if not isinstance(action, ReplayAction):
        raise TypeError("action must be ReplayAction")
    return action is ReplayAction.MARK_PRIORITY_REPLAY


def reason_code_from_decision(
    *,
    action: ReplayAction,
    mode: ReplayMode,
) -> str:
    """Build a deterministic reason code."""

    if not isinstance(action, ReplayAction):
        raise TypeError("action must be ReplayAction")
    if not isinstance(mode, ReplayMode):
        raise TypeError("mode must be ReplayMode")
    return f"REPLAY_{action.value}_{mode.value}"


def compute_replay_decision(
    *,
    consolidation: ConsolidationDecision | None = None,
    update_decision: UpdateDecision | None = None,
    transition: DynamicsTransitionResult | None = None,
    stability: StabilityReport | None = None,
    energy: EnergyState | None = None,
    temperature: SemanticTemperatureState | None = None,
    config: ReplayPolicyConfig | None = None,
    metadata: dict[str, object] | None = None,
) -> ReplayDecision:
    """Compute deterministic replay eligibility decision."""

    if config is None:
        config = ReplayPolicyConfig()
    if not isinstance(config, ReplayPolicyConfig):
        raise TypeError("config must be ReplayPolicyConfig")
    _validate_optional_inputs(
        consolidation=consolidation,
        update_decision=update_decision,
        transition=transition,
        stability=stability,
        energy=energy,
        temperature=temperature,
    )

    stability_score = stability_score_from_report(stability)
    risk_score = risk_score_from_report(stability)
    normalized_energy = normalized_energy_value(energy)
    confidence = replay_confidence_from_inputs(
        consolidation=consolidation,
        update_decision=update_decision,
        stability=stability,
        energy=energy,
    )
    mode = replay_mode_from_inputs(
        consolidation=consolidation,
        update_decision=update_decision,
        transition=transition,
        stability=stability,
        energy=energy,
        temperature=temperature,
        config=config,
    )
    action = replay_action_from_mode(mode=mode, config=config)
    status = replay_status_from_action(action)
    eligible = replay_eligible_from_action(action)
    priority = replay_priority_from_action(action)
    reason_code = reason_code_from_decision(action=action, mode=mode)

    default_metadata: dict[str, object] = {
        "source": "replay_policy",
        "has_consolidation": consolidation is not None,
        "has_update_decision": update_decision is not None,
        "has_transition": transition is not None,
        "has_stability": stability is not None,
        "has_energy": energy is not None,
        "has_temperature": temperature is not None,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "replay_policy"

    return ReplayDecision(
        action=action,
        mode=mode,
        status=status,
        eligible=eligible,
        priority=priority,
        stability_score=stability_score,
        risk_score=risk_score,
        normalized_energy=normalized_energy,
        confidence=confidence,
        reason_code=reason_code,
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def create_replay_policy_state(
    *,
    decision: ReplayDecision | None = None,
    config: ReplayPolicyConfig | None = None,
    metadata: dict[str, object] | None = None,
) -> ReplayPolicyState:
    """Create initial replay-policy state."""

    if decision is None:
        decision = compute_replay_decision(config=config)
    if not isinstance(decision, ReplayDecision):
        raise TypeError("decision must be ReplayDecision")

    timestamp = utc_now()
    default_metadata: dict[str, object] = {"state_kind": "replay_policy_state"}
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["state_kind"] = "replay_policy_state"

    return ReplayPolicyState(
        last_action=decision.action,
        last_status=decision.status,
        eligible=decision.eligible,
        priority=decision.priority,
        update_count=0,
        created_at=timestamp,
        updated_at=timestamp,
        metadata=merged_metadata,
    )


def transition_replay_policy_state(
    *,
    state: ReplayPolicyState,
    decision: ReplayDecision,
    metadata: dict[str, object] | None = None,
) -> ReplayPolicyState:
    """Create next replay-policy state without mutating prior state."""

    if not isinstance(state, ReplayPolicyState):
        raise TypeError("state must be ReplayPolicyState")
    if not isinstance(decision, ReplayDecision):
        raise TypeError("decision must be ReplayDecision")

    default_metadata: dict[str, object] = {
        "previous_action": state.last_action.value,
        "previous_status": state.last_status.value,
        "decision_reason_code": decision.reason_code,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)

    return ReplayPolicyState(
        last_action=decision.action,
        last_status=decision.status,
        eligible=decision.eligible,
        priority=decision.priority,
        update_count=state.update_count + 1,
        created_at=state.created_at,
        updated_at=utc_now(),
        metadata=merged_metadata,
    )


def replay_policy_allows_replay(state: ReplayPolicyState) -> bool:
    """Return whether the replay-policy state allows replay."""

    if not isinstance(state, ReplayPolicyState):
        raise TypeError("state must be ReplayPolicyState")
    return state.eligible


def replay_decision_allows_replay(decision: ReplayDecision) -> bool:
    """Return whether the replay decision allows replay."""

    if not isinstance(decision, ReplayDecision):
        raise TypeError("decision must be ReplayDecision")
    return decision.eligible


def replay_decision_is_priority(decision: ReplayDecision) -> bool:
    """Return whether the replay decision has priority."""

    if not isinstance(decision, ReplayDecision):
        raise TypeError("decision must be ReplayDecision")
    return decision.priority


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


def _validate_confidence_inputs(
    *,
    consolidation: ConsolidationDecision | None,
    update_decision: UpdateDecision | None,
    stability: StabilityReport | None,
    energy: EnergyState | None,
) -> None:
    if consolidation is not None and not isinstance(
        consolidation,
        ConsolidationDecision,
    ):
        raise TypeError("consolidation must be ConsolidationDecision or None")
    if update_decision is not None and not isinstance(update_decision, UpdateDecision):
        raise TypeError("update_decision must be UpdateDecision or None")
    if stability is not None and not isinstance(stability, StabilityReport):
        raise TypeError("stability must be StabilityReport or None")
    if energy is not None and not isinstance(energy, EnergyState):
        raise TypeError("energy must be EnergyState or None")


def _validate_optional_inputs(
    *,
    consolidation: ConsolidationDecision | None,
    update_decision: UpdateDecision | None,
    transition: DynamicsTransitionResult | None,
    stability: StabilityReport | None,
    energy: EnergyState | None,
    temperature: SemanticTemperatureState | None,
) -> None:
    _validate_confidence_inputs(
        consolidation=consolidation,
        update_decision=update_decision,
        stability=stability,
        energy=energy,
    )
    if transition is not None and not isinstance(transition, DynamicsTransitionResult):
        raise TypeError("transition must be DynamicsTransitionResult or None")
    if temperature is not None and not isinstance(
        temperature,
        SemanticTemperatureState,
    ):
        raise TypeError("temperature must be SemanticTemperatureState or None")


__all__ = [
    "ReplayMode",
    "ReplayAction",
    "ReplayStatus",
    "ReplayPolicyConfig",
    "ReplayDecision",
    "ReplayPolicyState",
    "risk_score_from_report",
    "stability_score_from_report",
    "normalized_energy_value",
    "replay_confidence_from_inputs",
    "replay_mode_from_inputs",
    "replay_action_from_mode",
    "replay_status_from_action",
    "replay_eligible_from_action",
    "replay_priority_from_action",
    "reason_code_from_decision",
    "compute_replay_decision",
    "create_replay_policy_state",
    "transition_replay_policy_state",
    "replay_policy_allows_replay",
    "replay_decision_allows_replay",
    "replay_decision_is_priority",
]
