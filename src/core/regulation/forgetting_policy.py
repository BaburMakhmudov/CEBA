"""Deterministic forgetting-policy utilities for CEBA Core Regulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ceba.core.dynamics.energy import EnergyState, EnergyStatus
from ceba.core.dynamics.semantic_temperature import SemanticTemperatureState
from ceba.core.dynamics.stability import (
    StabilityReport,
    risk_rank,
    stability_report_is_blocked,
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
from ceba.core.regulation.replay_policy import (
    ReplayDecision,
    replay_decision_allows_replay,
    replay_decision_is_priority,
)
from ceba.core.regulation.update_policy import (
    UpdateDecision,
    UpdatePolicyStatus,
)


class ForgettingMode(Enum):
    """Forgetting-policy operating mode."""

    RETAIN = "RETAIN"
    DECAY = "DECAY"
    REVIEW = "REVIEW"
    MERGE_CANDIDATE = "MERGE_CANDIDATE"
    PRUNE_CANDIDATE = "PRUNE_CANDIDATE"
    BLOCKED = "BLOCKED"


class ForgettingAction(Enum):
    """Forgetting-policy action."""

    MARK_RETAIN = "MARK_RETAIN"
    MARK_DECAY = "MARK_DECAY"
    MARK_REVIEW = "MARK_REVIEW"
    MARK_MERGE_CANDIDATE = "MARK_MERGE_CANDIDATE"
    MARK_PRUNE_CANDIDATE = "MARK_PRUNE_CANDIDATE"
    BLOCK = "BLOCK"


class ForgettingStatus(Enum):
    """Forgetting-policy status."""

    RETAINED = "RETAINED"
    DECAYED = "DECAYED"
    REVIEW = "REVIEW"
    MERGE_CANDIDATE = "MERGE_CANDIDATE"
    PRUNE_CANDIDATE = "PRUNE_CANDIDATE"
    BLOCKED = "BLOCKED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class ForgettingPolicyConfig:
    """Stable configuration for forgetting/retention decisions."""

    min_stability_to_retain: float = 0.70
    min_confidence_to_retain: float = 0.60
    max_risk_rank_to_retain: int = 2
    decay_confidence_threshold: float = 0.45
    prune_confidence_threshold: float = 0.20
    max_normalized_energy_to_retain: float = 0.80
    retain_priority_replay: bool = True
    retain_consolidation_eligible: bool = True
    allow_merge_candidates: bool = True
    allow_prune_candidates: bool = True
    block_invalid_update: bool = True

    def __post_init__(self) -> None:
        min_stability_to_retain = ensure_constraint_probability(
            self.min_stability_to_retain,
            field_name="min_stability_to_retain",
        )
        min_confidence_to_retain = ensure_constraint_probability(
            self.min_confidence_to_retain,
            field_name="min_confidence_to_retain",
        )
        max_risk_rank_to_retain = _ensure_risk_rank_int(
            self.max_risk_rank_to_retain,
            "max_risk_rank_to_retain",
        )
        decay_confidence_threshold = ensure_constraint_probability(
            self.decay_confidence_threshold,
            field_name="decay_confidence_threshold",
        )
        prune_confidence_threshold = ensure_constraint_probability(
            self.prune_confidence_threshold,
            field_name="prune_confidence_threshold",
        )
        max_normalized_energy_to_retain = ensure_constraint_probability(
            self.max_normalized_energy_to_retain,
            field_name="max_normalized_energy_to_retain",
        )
        for field_name in (
            "retain_priority_replay",
            "retain_consolidation_eligible",
            "allow_merge_candidates",
            "allow_prune_candidates",
            "block_invalid_update",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")

        object.__setattr__(
            self,
            "min_stability_to_retain",
            min_stability_to_retain,
        )
        object.__setattr__(
            self,
            "min_confidence_to_retain",
            min_confidence_to_retain,
        )
        object.__setattr__(
            self,
            "max_risk_rank_to_retain",
            max_risk_rank_to_retain,
        )
        object.__setattr__(
            self,
            "decay_confidence_threshold",
            decay_confidence_threshold,
        )
        object.__setattr__(
            self,
            "prune_confidence_threshold",
            prune_confidence_threshold,
        )
        object.__setattr__(
            self,
            "max_normalized_energy_to_retain",
            max_normalized_energy_to_retain,
        )


@dataclass(frozen=True)
class ForgettingDecision:
    """Serializable-friendly result of one forgetting-policy decision."""

    action: ForgettingAction
    mode: ForgettingMode
    status: ForgettingStatus
    retain: bool
    decay: bool
    review: bool
    merge_candidate: bool
    prune_candidate: bool
    confidence: float
    stability_score: float
    risk_score: float
    normalized_energy: float | None
    decay_factor: float
    reason_code: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.action, ForgettingAction):
            raise TypeError("action must be ForgettingAction")
        if not isinstance(self.mode, ForgettingMode):
            raise TypeError("mode must be ForgettingMode")
        if not isinstance(self.status, ForgettingStatus):
            raise TypeError("status must be ForgettingStatus")
        for field_name in (
            "retain",
            "decay",
            "review",
            "merge_candidate",
            "prune_candidate",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")

        confidence = ensure_constraint_probability(
            self.confidence,
            field_name="confidence",
        )
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
        decay_factor = ensure_constraint_probability(
            self.decay_factor,
            field_name="decay_factor",
        )
        if not isinstance(self.reason_code, str) or not self.reason_code:
            raise ValueError("reason_code must be a non-empty string")
        _ensure_timezone_aware(self.created_at, "created_at")

        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "stability_score", stability_score)
        object.__setattr__(self, "risk_score", risk_score)
        object.__setattr__(self, "normalized_energy", normalized_energy)
        object.__setattr__(self, "decay_factor", decay_factor)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "action": self.action.value,
            "mode": self.mode.value,
            "status": self.status.value,
            "retain": self.retain,
            "decay": self.decay,
            "review": self.review,
            "merge_candidate": self.merge_candidate,
            "prune_candidate": self.prune_candidate,
            "confidence": self.confidence,
            "stability_score": self.stability_score,
            "risk_score": self.risk_score,
            "normalized_energy": self.normalized_energy,
            "decay_factor": self.decay_factor,
            "reason_code": self.reason_code,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ForgettingPolicyState:
    """Runtime descriptor of forgetting-policy state."""

    last_action: ForgettingAction
    last_status: ForgettingStatus
    retain: bool
    decay_factor: float
    update_count: int
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.last_action, ForgettingAction):
            raise TypeError("last_action must be ForgettingAction")
        if not isinstance(self.last_status, ForgettingStatus):
            raise TypeError("last_status must be ForgettingStatus")
        if not isinstance(self.retain, bool):
            raise TypeError("retain must be bool")
        decay_factor = ensure_constraint_probability(
            self.decay_factor,
            field_name="decay_factor",
        )
        update_count = _ensure_non_negative_int(self.update_count, "update_count")
        _ensure_timezone_aware(self.created_at, "created_at")
        _ensure_timezone_aware(self.updated_at, "updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be greater than or equal to created_at")

        object.__setattr__(self, "decay_factor", decay_factor)
        object.__setattr__(self, "update_count", update_count)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "last_action": self.last_action.value,
            "last_status": self.last_status.value,
            "retain": self.retain,
            "decay_factor": self.decay_factor,
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


def retention_confidence_from_inputs(
    *,
    replay: ReplayDecision | None = None,
    consolidation: ConsolidationDecision | None = None,
    update_decision: UpdateDecision | None = None,
    stability: StabilityReport | None = None,
    energy: EnergyState | None = None,
) -> float:
    """Compute deterministic retention confidence from available inputs."""

    _validate_confidence_inputs(
        replay=replay,
        consolidation=consolidation,
        update_decision=update_decision,
        stability=stability,
        energy=energy,
    )

    confidence = 1.0
    if replay is not None:
        confidence *= replay.confidence
    if consolidation is not None:
        confidence *= consolidation.confidence
    if update_decision is not None:
        confidence *= update_decision.confidence
    if stability is not None:
        confidence *= stability.score
    if energy is not None and energy.normalized_energy is not None:
        confidence *= 1.0 - energy.normalized_energy
    return _clamp_probability(confidence)


def decay_factor_from_confidence(
    *,
    confidence: float,
    config: ForgettingPolicyConfig,
) -> float:
    """Compute deterministic decay factor."""

    confidence = ensure_constraint_probability(confidence, field_name="confidence")
    if not isinstance(config, ForgettingPolicyConfig):
        raise TypeError("config must be ForgettingPolicyConfig")
    if confidence >= config.min_confidence_to_retain:
        return 0.0
    return _clamp_probability(1.0 - confidence)


def forgetting_mode_from_inputs(
    *,
    replay: ReplayDecision | None = None,
    consolidation: ConsolidationDecision | None = None,
    update_decision: UpdateDecision | None = None,
    stability: StabilityReport | None = None,
    energy: EnergyState | None = None,
    temperature: SemanticTemperatureState | None = None,
    config: ForgettingPolicyConfig | None = None,
) -> ForgettingMode:
    """Determine forgetting mode from regulatory inputs."""

    if config is None:
        config = ForgettingPolicyConfig()
    if not isinstance(config, ForgettingPolicyConfig):
        raise TypeError("config must be ForgettingPolicyConfig")
    _validate_optional_inputs(
        replay=replay,
        consolidation=consolidation,
        update_decision=update_decision,
        stability=stability,
        energy=energy,
        temperature=temperature,
    )

    if (
        update_decision is not None
        and config.block_invalid_update
        and update_decision.status is UpdatePolicyStatus.INVALID
    ):
        return ForgettingMode.BLOCKED
    if stability is not None and stability_report_is_blocked(stability):
        return ForgettingMode.BLOCKED
    if energy is not None and energy.status is EnergyStatus.INVALID:
        return ForgettingMode.BLOCKED
    if (
        replay is not None
        and replay_decision_is_priority(replay)
        and config.retain_priority_replay
    ):
        return ForgettingMode.RETAIN
    if (
        consolidation is not None
        and consolidation_decision_allows_consolidation(consolidation)
        and config.retain_consolidation_eligible
    ):
        return ForgettingMode.RETAIN
    if replay is not None and replay_decision_allows_replay(replay):
        return ForgettingMode.RETAIN
    if stability is not None and risk_rank(stability.risk) > (
        config.max_risk_rank_to_retain
    ):
        return ForgettingMode.REVIEW
    if stability is not None and stability.score < config.min_stability_to_retain:
        return ForgettingMode.REVIEW
    if (
        energy is not None
        and energy.normalized_energy is not None
        and energy.normalized_energy > config.max_normalized_energy_to_retain
    ):
        return ForgettingMode.REVIEW

    confidence = retention_confidence_from_inputs(
        replay=replay,
        consolidation=consolidation,
        update_decision=update_decision,
        stability=stability,
        energy=energy,
    )
    if (
        confidence <= config.prune_confidence_threshold
        and config.allow_prune_candidates
    ):
        return ForgettingMode.PRUNE_CANDIDATE
    if confidence <= config.decay_confidence_threshold:
        return ForgettingMode.DECAY
    if (
        config.allow_merge_candidates
        and confidence >= config.min_confidence_to_retain
        and stability_score_from_report(stability) >= config.min_stability_to_retain
    ):
        return ForgettingMode.MERGE_CANDIDATE
    return ForgettingMode.RETAIN


def forgetting_action_from_mode(
    *,
    mode: ForgettingMode,
) -> ForgettingAction:
    """Map forgetting mode to action."""

    if not isinstance(mode, ForgettingMode):
        raise TypeError("mode must be ForgettingMode")
    if mode is ForgettingMode.RETAIN:
        return ForgettingAction.MARK_RETAIN
    if mode is ForgettingMode.DECAY:
        return ForgettingAction.MARK_DECAY
    if mode is ForgettingMode.REVIEW:
        return ForgettingAction.MARK_REVIEW
    if mode is ForgettingMode.MERGE_CANDIDATE:
        return ForgettingAction.MARK_MERGE_CANDIDATE
    if mode is ForgettingMode.PRUNE_CANDIDATE:
        return ForgettingAction.MARK_PRUNE_CANDIDATE
    return ForgettingAction.BLOCK


def forgetting_status_from_action(action: ForgettingAction) -> ForgettingStatus:
    """Map forgetting action to status."""

    if not isinstance(action, ForgettingAction):
        raise TypeError("action must be ForgettingAction")
    if action is ForgettingAction.MARK_RETAIN:
        return ForgettingStatus.RETAINED
    if action is ForgettingAction.MARK_DECAY:
        return ForgettingStatus.DECAYED
    if action is ForgettingAction.MARK_REVIEW:
        return ForgettingStatus.REVIEW
    if action is ForgettingAction.MARK_MERGE_CANDIDATE:
        return ForgettingStatus.MERGE_CANDIDATE
    if action is ForgettingAction.MARK_PRUNE_CANDIDATE:
        return ForgettingStatus.PRUNE_CANDIDATE
    return ForgettingStatus.BLOCKED


def forgetting_flags_from_action(
    action: ForgettingAction,
) -> tuple[bool, bool, bool, bool, bool]:
    """Return retain, decay, review, merge, and prune flags."""

    if not isinstance(action, ForgettingAction):
        raise TypeError("action must be ForgettingAction")
    if action is ForgettingAction.MARK_RETAIN:
        return True, False, False, False, False
    if action is ForgettingAction.MARK_DECAY:
        return False, True, False, False, False
    if action is ForgettingAction.MARK_REVIEW:
        return False, False, True, False, False
    if action is ForgettingAction.MARK_MERGE_CANDIDATE:
        return False, False, False, True, False
    if action is ForgettingAction.MARK_PRUNE_CANDIDATE:
        return False, False, False, False, True
    return False, False, False, False, False


def reason_code_from_decision(
    *,
    action: ForgettingAction,
    mode: ForgettingMode,
) -> str:
    """Build a deterministic reason code."""

    if not isinstance(action, ForgettingAction):
        raise TypeError("action must be ForgettingAction")
    if not isinstance(mode, ForgettingMode):
        raise TypeError("mode must be ForgettingMode")
    return f"FORGETTING_{action.value}_{mode.value}"


def compute_forgetting_decision(
    *,
    replay: ReplayDecision | None = None,
    consolidation: ConsolidationDecision | None = None,
    update_decision: UpdateDecision | None = None,
    stability: StabilityReport | None = None,
    energy: EnergyState | None = None,
    temperature: SemanticTemperatureState | None = None,
    config: ForgettingPolicyConfig | None = None,
    metadata: dict[str, object] | None = None,
) -> ForgettingDecision:
    """Compute deterministic forgetting/retention decision."""

    if config is None:
        config = ForgettingPolicyConfig()
    if not isinstance(config, ForgettingPolicyConfig):
        raise TypeError("config must be ForgettingPolicyConfig")
    _validate_optional_inputs(
        replay=replay,
        consolidation=consolidation,
        update_decision=update_decision,
        stability=stability,
        energy=energy,
        temperature=temperature,
    )

    stability_score = stability_score_from_report(stability)
    risk_score = risk_score_from_report(stability)
    normalized_energy = normalized_energy_value(energy)
    confidence = retention_confidence_from_inputs(
        replay=replay,
        consolidation=consolidation,
        update_decision=update_decision,
        stability=stability,
        energy=energy,
    )
    mode = forgetting_mode_from_inputs(
        replay=replay,
        consolidation=consolidation,
        update_decision=update_decision,
        stability=stability,
        energy=energy,
        temperature=temperature,
        config=config,
    )
    action = forgetting_action_from_mode(mode=mode)
    status = forgetting_status_from_action(action)
    retain, decay, review, merge_candidate, prune_candidate = (
        forgetting_flags_from_action(action)
    )
    decay_factor = decay_factor_from_confidence(
        confidence=confidence,
        config=config,
    )
    reason_code = reason_code_from_decision(action=action, mode=mode)

    default_metadata: dict[str, object] = {
        "source": "forgetting_policy",
        "has_replay": replay is not None,
        "has_consolidation": consolidation is not None,
        "has_update_decision": update_decision is not None,
        "has_stability": stability is not None,
        "has_energy": energy is not None,
        "has_temperature": temperature is not None,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "forgetting_policy"

    return ForgettingDecision(
        action=action,
        mode=mode,
        status=status,
        retain=retain,
        decay=decay,
        review=review,
        merge_candidate=merge_candidate,
        prune_candidate=prune_candidate,
        confidence=confidence,
        stability_score=stability_score,
        risk_score=risk_score,
        normalized_energy=normalized_energy,
        decay_factor=decay_factor,
        reason_code=reason_code,
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def create_forgetting_policy_state(
    *,
    decision: ForgettingDecision | None = None,
    config: ForgettingPolicyConfig | None = None,
    metadata: dict[str, object] | None = None,
) -> ForgettingPolicyState:
    """Create initial forgetting-policy state."""

    if decision is None:
        decision = compute_forgetting_decision(config=config)
    if not isinstance(decision, ForgettingDecision):
        raise TypeError("decision must be ForgettingDecision")

    timestamp = utc_now()
    default_metadata: dict[str, object] = {"state_kind": "forgetting_policy_state"}
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["state_kind"] = "forgetting_policy_state"

    return ForgettingPolicyState(
        last_action=decision.action,
        last_status=decision.status,
        retain=decision.retain,
        decay_factor=decision.decay_factor,
        update_count=0,
        created_at=timestamp,
        updated_at=timestamp,
        metadata=merged_metadata,
    )


def transition_forgetting_policy_state(
    *,
    state: ForgettingPolicyState,
    decision: ForgettingDecision,
    metadata: dict[str, object] | None = None,
) -> ForgettingPolicyState:
    """Create next forgetting-policy state without mutating prior state."""

    if not isinstance(state, ForgettingPolicyState):
        raise TypeError("state must be ForgettingPolicyState")
    if not isinstance(decision, ForgettingDecision):
        raise TypeError("decision must be ForgettingDecision")

    default_metadata: dict[str, object] = {
        "previous_action": state.last_action.value,
        "previous_status": state.last_status.value,
        "decision_reason_code": decision.reason_code,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)

    return ForgettingPolicyState(
        last_action=decision.action,
        last_status=decision.status,
        retain=decision.retain,
        decay_factor=decision.decay_factor,
        update_count=state.update_count + 1,
        created_at=state.created_at,
        updated_at=utc_now(),
        metadata=merged_metadata,
    )


def forgetting_policy_retain(state: ForgettingPolicyState) -> bool:
    """Return whether the forgetting-policy state retains."""

    if not isinstance(state, ForgettingPolicyState):
        raise TypeError("state must be ForgettingPolicyState")
    return state.retain


def forgetting_decision_retain(decision: ForgettingDecision) -> bool:
    """Return whether the forgetting decision retains."""

    if not isinstance(decision, ForgettingDecision):
        raise TypeError("decision must be ForgettingDecision")
    return decision.retain


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
    replay: ReplayDecision | None,
    consolidation: ConsolidationDecision | None,
    update_decision: UpdateDecision | None,
    stability: StabilityReport | None,
    energy: EnergyState | None,
) -> None:
    if replay is not None and not isinstance(replay, ReplayDecision):
        raise TypeError("replay must be ReplayDecision or None")
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
    replay: ReplayDecision | None,
    consolidation: ConsolidationDecision | None,
    update_decision: UpdateDecision | None,
    stability: StabilityReport | None,
    energy: EnergyState | None,
    temperature: SemanticTemperatureState | None,
) -> None:
    _validate_confidence_inputs(
        replay=replay,
        consolidation=consolidation,
        update_decision=update_decision,
        stability=stability,
        energy=energy,
    )
    if temperature is not None and not isinstance(
        temperature,
        SemanticTemperatureState,
    ):
        raise TypeError("temperature must be SemanticTemperatureState or None")


__all__ = [
    "ForgettingMode",
    "ForgettingAction",
    "ForgettingStatus",
    "ForgettingPolicyConfig",
    "ForgettingDecision",
    "ForgettingPolicyState",
    "risk_score_from_report",
    "stability_score_from_report",
    "normalized_energy_value",
    "retention_confidence_from_inputs",
    "decay_factor_from_confidence",
    "forgetting_mode_from_inputs",
    "forgetting_action_from_mode",
    "forgetting_status_from_action",
    "forgetting_flags_from_action",
    "reason_code_from_decision",
    "compute_forgetting_decision",
    "create_forgetting_policy_state",
    "transition_forgetting_policy_state",
    "forgetting_policy_retain",
    "forgetting_decision_retain",
]
