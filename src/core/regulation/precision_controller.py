"""Deterministic precision-control utilities for CEBA Core Regulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ceba.core.dynamics.energy import EnergyState, EnergyStatus
from ceba.core.dynamics.semantic_temperature import (
    SemanticTemperatureState,
    SemanticTemperatureStatus,
    semantic_temperature_is_frozen,
    transition_scale_from_temperature,
)
from ceba.core.dynamics.stability import StabilityReport, StabilityStatus, risk_rank
from ceba.core.dynamics.transition import DynamicsTransitionResult, DynamicsTransitionStatus
from ceba.core.metrics.schema import (
    ensure_constraint_probability,
    ensure_finite_number,
    ensure_non_negative_number,
    utc_now,
)


class PrecisionMode(Enum):
    NORMAL = "NORMAL"
    CONSERVATIVE = "CONSERVATIVE"
    AGGRESSIVE = "AGGRESSIVE"
    FROZEN = "FROZEN"
    DEGRADED = "DEGRADED"


class PrecisionStatus(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    BLOCKED = "BLOCKED"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class PrecisionPolicy:
    high_threshold: float = 0.85
    medium_threshold: float = 0.60
    low_threshold: float = 0.35
    min_allowed_precision: float = 0.05
    uncertainty_penalty: float = 0.20
    risk_penalty_weight: float = 0.15
    conservative_mode_multiplier: float = 0.75
    aggressive_mode_multiplier: float = 1.25

    def __post_init__(self) -> None:
        object.__setattr__(self, "high_threshold", ensure_constraint_probability(self.high_threshold, "high_threshold"))
        object.__setattr__(
            self,
            "medium_threshold",
            ensure_constraint_probability(self.medium_threshold, "medium_threshold"),
        )
        object.__setattr__(self, "low_threshold", ensure_constraint_probability(self.low_threshold, "low_threshold"))
        if self.high_threshold < self.medium_threshold:
            raise ValueError("high_threshold must be greater than or equal to medium_threshold")
        if self.medium_threshold < self.low_threshold:
            raise ValueError("medium_threshold must be greater than or equal to low_threshold")
        object.__setattr__(
            self,
            "min_allowed_precision",
            ensure_constraint_probability(self.min_allowed_precision, "min_allowed_precision"),
        )
        object.__setattr__(
            self,
            "uncertainty_penalty",
            ensure_non_negative_number(self.uncertainty_penalty, "uncertainty_penalty"),
        )
        object.__setattr__(
            self,
            "risk_penalty_weight",
            ensure_non_negative_number(self.risk_penalty_weight, "risk_penalty_weight"),
        )
        object.__setattr__(
            self,
            "conservative_mode_multiplier",
            ensure_non_negative_number(self.conservative_mode_multiplier, "conservative_mode_multiplier"),
        )
        object.__setattr__(
            self,
            "aggressive_mode_multiplier",
            ensure_non_negative_number(self.aggressive_mode_multiplier, "aggressive_mode_multiplier"),
        )


@dataclass(frozen=True, slots=True)
class PrecisionState:
    precision: float
    mode: PrecisionMode
    status: PrecisionStatus
    uncertainty: float
    risk_score: float
    update_count: int
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "precision", ensure_constraint_probability(self.precision, "precision"))
        if not isinstance(self.mode, PrecisionMode):
            raise TypeError("mode must be PrecisionMode")
        if not isinstance(self.status, PrecisionStatus):
            raise TypeError("status must be PrecisionStatus")
        object.__setattr__(self, "uncertainty", ensure_constraint_probability(self.uncertainty, "uncertainty"))
        object.__setattr__(self, "risk_score", ensure_constraint_probability(self.risk_score, "risk_score"))
        object.__setattr__(self, "update_count", _ensure_non_negative_int(self.update_count, "update_count"))
        object.__setattr__(self, "created_at", _ensure_timezone_aware(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", _ensure_timezone_aware(self.updated_at, "updated_at"))
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be greater than or equal to created_at")
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "precision": self.precision,
            "mode": self.mode.value,
            "status": self.status.value,
            "uncertainty": self.uncertainty,
            "risk_score": self.risk_score,
            "update_count": self.update_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PrecisionAssessment:
    precision: float
    mode: PrecisionMode
    status: PrecisionStatus
    uncertainty: float
    risk_score: float
    confidence: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "precision", ensure_constraint_probability(self.precision, "precision"))
        if not isinstance(self.mode, PrecisionMode):
            raise TypeError("mode must be PrecisionMode")
        if not isinstance(self.status, PrecisionStatus):
            raise TypeError("status must be PrecisionStatus")
        object.__setattr__(self, "uncertainty", ensure_constraint_probability(self.uncertainty, "uncertainty"))
        object.__setattr__(self, "risk_score", ensure_constraint_probability(self.risk_score, "risk_score"))
        object.__setattr__(self, "confidence", ensure_constraint_probability(self.confidence, "confidence"))
        object.__setattr__(self, "created_at", _ensure_timezone_aware(self.created_at, "created_at"))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "precision": self.precision,
            "mode": self.mode.value,
            "status": self.status.value,
            "uncertainty": self.uncertainty,
            "risk_score": self.risk_score,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


def clamp_probability(value: float) -> float:
    number = ensure_finite_number(value, "value")
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def risk_score_from_stability(report: StabilityReport | None) -> float:
    if report is None:
        return 0.0
    if not isinstance(report, StabilityReport):
        raise TypeError("report must be StabilityReport or None")
    return ensure_constraint_probability(risk_rank(report.risk) / 4.0, "risk_score")


def uncertainty_from_inputs(
    *,
    energy: EnergyState | None = None,
    temperature: SemanticTemperatureState | None = None,
    stability: StabilityReport | None = None,
    transition: DynamicsTransitionResult | None = None,
) -> float:
    _validate_inputs(energy=energy, temperature=temperature, stability=stability, transition=transition)
    uncertainty = 0.0
    if energy is not None and energy.status in (EnergyStatus.DEGRADED, EnergyStatus.INVALID):
        uncertainty += 0.25
    if temperature is not None and temperature.status in (
        SemanticTemperatureStatus.UNSTABLE,
        SemanticTemperatureStatus.INVALID,
    ):
        uncertainty += 0.25
    if stability is not None and stability.status in (
        StabilityStatus.DEGRADED,
        StabilityStatus.UNSTABLE,
        StabilityStatus.BLOCKED,
        StabilityStatus.INVALID,
    ):
        uncertainty += 0.25
    if transition is not None and transition.transition.status in (
        DynamicsTransitionStatus.DEGRADED,
        DynamicsTransitionStatus.BLOCKED,
        DynamicsTransitionStatus.INVALID,
    ):
        uncertainty += 0.25
    return clamp_probability(uncertainty)


def base_precision_from_inputs(
    *,
    energy: EnergyState | None = None,
    temperature: SemanticTemperatureState | None = None,
    stability: StabilityReport | None = None,
    transition: DynamicsTransitionResult | None = None,
) -> float:
    _validate_inputs(energy=energy, temperature=temperature, stability=stability, transition=transition)
    precision = 1.0
    if energy is not None and energy.normalized_energy is not None:
        precision *= 1.0 - energy.normalized_energy
    if temperature is not None:
        precision *= transition_scale_from_temperature(temperature)
    if stability is not None:
        precision *= stability.score
    if transition is not None:
        precision *= (
            transition.transition.stability_score
            if transition.transition.stability_score is not None
            else 1.0
        )
    return clamp_probability(precision)


def precision_mode_from_inputs(
    *,
    temperature: SemanticTemperatureState | None,
    stability: StabilityReport | None,
    transition: DynamicsTransitionResult | None,
) -> PrecisionMode:
    _validate_inputs(temperature=temperature, stability=stability, transition=transition)
    if temperature is not None and semantic_temperature_is_frozen(temperature):
        return PrecisionMode.FROZEN
    if transition is not None and transition.transition.status is DynamicsTransitionStatus.DEGRADED:
        return PrecisionMode.DEGRADED
    if stability is not None and stability.status in (StabilityStatus.DEGRADED, StabilityStatus.UNSTABLE):
        return PrecisionMode.CONSERVATIVE
    return PrecisionMode.NORMAL


def precision_status_from_value(
    *,
    precision: float,
    mode: PrecisionMode,
    policy: PrecisionPolicy,
) -> PrecisionStatus:
    checked_precision = ensure_constraint_probability(precision, "precision")
    if not isinstance(mode, PrecisionMode):
        raise TypeError("mode must be PrecisionMode")
    if not isinstance(policy, PrecisionPolicy):
        raise TypeError("policy must be PrecisionPolicy")
    if mode is PrecisionMode.FROZEN:
        return PrecisionStatus.BLOCKED
    if mode is PrecisionMode.DEGRADED and checked_precision < policy.medium_threshold:
        return PrecisionStatus.LOW
    if checked_precision >= policy.high_threshold:
        return PrecisionStatus.HIGH
    if checked_precision >= policy.medium_threshold:
        return PrecisionStatus.MEDIUM
    if checked_precision >= policy.low_threshold:
        return PrecisionStatus.LOW
    if checked_precision < policy.min_allowed_precision:
        return PrecisionStatus.BLOCKED
    return PrecisionStatus.LOW


def apply_precision_mode(
    *,
    precision: float,
    mode: PrecisionMode,
    policy: PrecisionPolicy,
) -> float:
    checked_precision = ensure_constraint_probability(precision, "precision")
    if not isinstance(mode, PrecisionMode):
        raise TypeError("mode must be PrecisionMode")
    if not isinstance(policy, PrecisionPolicy):
        raise TypeError("policy must be PrecisionPolicy")
    if mode is PrecisionMode.NORMAL:
        return checked_precision
    if mode is PrecisionMode.CONSERVATIVE:
        return clamp_probability(checked_precision * policy.conservative_mode_multiplier)
    if mode is PrecisionMode.AGGRESSIVE:
        return clamp_probability(checked_precision * policy.aggressive_mode_multiplier)
    if mode is PrecisionMode.FROZEN:
        return 0.0
    if mode is PrecisionMode.DEGRADED:
        return clamp_probability(checked_precision * policy.conservative_mode_multiplier)
    return checked_precision


def compute_precision_assessment(
    *,
    energy: EnergyState | None = None,
    temperature: SemanticTemperatureState | None = None,
    stability: StabilityReport | None = None,
    transition: DynamicsTransitionResult | None = None,
    policy: PrecisionPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> PrecisionAssessment:
    _validate_inputs(energy=energy, temperature=temperature, stability=stability, transition=transition)
    active_policy = _ensure_policy(policy)
    base_precision = base_precision_from_inputs(
        energy=energy,
        temperature=temperature,
        stability=stability,
        transition=transition,
    )
    uncertainty = uncertainty_from_inputs(
        energy=energy,
        temperature=temperature,
        stability=stability,
        transition=transition,
    )
    risk_score = risk_score_from_stability(stability)
    adjusted = (
        base_precision
        - active_policy.uncertainty_penalty * uncertainty
        - active_policy.risk_penalty_weight * risk_score
    )
    mode = precision_mode_from_inputs(
        temperature=temperature,
        stability=stability,
        transition=transition,
    )
    precision = apply_precision_mode(precision=clamp_probability(adjusted), mode=mode, policy=active_policy)
    status = precision_status_from_value(precision=precision, mode=mode, policy=active_policy)

    assessment_metadata: dict[str, object] = {
        "source": "precision_controller",
        "base_precision": base_precision,
        "uncertainty": uncertainty,
        "risk_score": risk_score,
        "has_energy": energy is not None,
        "has_temperature": temperature is not None,
        "has_stability": stability is not None,
        "has_transition": transition is not None,
    }
    if metadata is not None:
        assessment_metadata.update(_copy_metadata(metadata))
    assessment_metadata["source"] = "precision_controller"

    return PrecisionAssessment(
        precision=precision,
        mode=mode,
        status=status,
        uncertainty=uncertainty,
        risk_score=risk_score,
        confidence=precision,
        created_at=utc_now(),
        metadata=assessment_metadata,
    )


def create_precision_state(
    *,
    assessment: PrecisionAssessment | None = None,
    policy: PrecisionPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> PrecisionState:
    if assessment is not None and not isinstance(assessment, PrecisionAssessment):
        raise TypeError("assessment must be PrecisionAssessment or None")
    active_assessment = assessment if assessment is not None else compute_precision_assessment(policy=policy)
    timestamp = utc_now()
    state_metadata: dict[str, object] = {"state_kind": "precision_state"}
    if metadata is not None:
        state_metadata.update(_copy_metadata(metadata))
    state_metadata["state_kind"] = "precision_state"

    return PrecisionState(
        precision=active_assessment.precision,
        mode=active_assessment.mode,
        status=active_assessment.status,
        uncertainty=active_assessment.uncertainty,
        risk_score=active_assessment.risk_score,
        update_count=0,
        created_at=timestamp,
        updated_at=timestamp,
        metadata=state_metadata,
    )


def transition_precision_state(
    *,
    state: PrecisionState,
    assessment: PrecisionAssessment,
    metadata: dict[str, object] | None = None,
) -> PrecisionState:
    if not isinstance(state, PrecisionState):
        raise TypeError("state must be PrecisionState")
    if not isinstance(assessment, PrecisionAssessment):
        raise TypeError("assessment must be PrecisionAssessment")
    state_metadata: dict[str, object] = {
        "previous_precision": state.precision,
        "assessment_status": assessment.status.value,
    }
    if metadata is not None:
        state_metadata.update(_copy_metadata(metadata))

    return PrecisionState(
        precision=assessment.precision,
        mode=assessment.mode,
        status=assessment.status,
        uncertainty=assessment.uncertainty,
        risk_score=assessment.risk_score,
        update_count=state.update_count + 1,
        created_at=state.created_at,
        updated_at=utc_now(),
        metadata=state_metadata,
    )


def precision_allows_update(state: PrecisionState) -> bool:
    if not isinstance(state, PrecisionState):
        raise TypeError("state must be PrecisionState")
    if state.status in (PrecisionStatus.BLOCKED, PrecisionStatus.INVALID):
        return False
    return state.precision > 0.0


def precision_scale(state: PrecisionState) -> float:
    if not isinstance(state, PrecisionState):
        raise TypeError("state must be PrecisionState")
    return state.precision


def _ensure_policy(policy: PrecisionPolicy | None) -> PrecisionPolicy:
    if policy is None:
        return PrecisionPolicy()
    if not isinstance(policy, PrecisionPolicy):
        raise TypeError("policy must be PrecisionPolicy or None")
    return policy


def _validate_inputs(
    *,
    energy: EnergyState | None = None,
    temperature: SemanticTemperatureState | None = None,
    stability: StabilityReport | None = None,
    transition: DynamicsTransitionResult | None = None,
) -> None:
    if energy is not None and not isinstance(energy, EnergyState):
        raise TypeError("energy must be EnergyState or None")
    if temperature is not None and not isinstance(temperature, SemanticTemperatureState):
        raise TypeError("temperature must be SemanticTemperatureState or None")
    if stability is not None and not isinstance(stability, StabilityReport):
        raise TypeError("stability must be StabilityReport or None")
    if transition is not None and not isinstance(transition, DynamicsTransitionResult):
        raise TypeError("transition must be DynamicsTransitionResult or None")


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
    "PrecisionMode",
    "PrecisionStatus",
    "PrecisionPolicy",
    "PrecisionState",
    "PrecisionAssessment",
    "clamp_probability",
    "risk_score_from_stability",
    "uncertainty_from_inputs",
    "base_precision_from_inputs",
    "precision_mode_from_inputs",
    "precision_status_from_value",
    "apply_precision_mode",
    "compute_precision_assessment",
    "create_precision_state",
    "transition_precision_state",
    "precision_allows_update",
    "precision_scale",
]
