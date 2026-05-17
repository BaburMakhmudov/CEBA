"""Deterministic step-size regulation utilities for CEBA Core Regulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ceba.core.dynamics.energy import EnergyState, EnergyStatus
from ceba.core.dynamics.semantic_temperature import (
    SemanticTemperatureState,
    semantic_temperature_is_frozen,
    transition_scale_from_temperature,
)
from ceba.core.dynamics.stability import (
    StabilityReport,
    StabilityStatus,
    stability_report_is_blocked,
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


class StepSizeMode(Enum):
    """Step-size regulation mode."""

    NORMAL = "NORMAL"
    REDUCED = "REDUCED"
    EXPANDED = "EXPANDED"
    FROZEN = "FROZEN"
    BLOCKED = "BLOCKED"


class StepSizeStatus(Enum):
    """Step-size regulation status."""

    READY = "READY"
    REDUCED = "REDUCED"
    SKIPPED = "SKIPPED"
    BLOCKED = "BLOCKED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class StepSizePolicy:
    """Stable policy for step-size regulation."""

    base_step_size: float = 0.1
    min_step_size: float = 0.0
    max_step_size: float = 1.0
    conservative_multiplier: float = 0.5
    aggressive_multiplier: float = 1.25
    instability_penalty: float = 0.5

    def __post_init__(self) -> None:
        base_step_size = ensure_non_negative_number(
            self.base_step_size,
            field_name="base_step_size",
        )
        min_step_size = ensure_non_negative_number(
            self.min_step_size,
            field_name="min_step_size",
        )
        max_step_size = ensure_non_negative_number(
            self.max_step_size,
            field_name="max_step_size",
        )
        conservative_multiplier = ensure_non_negative_number(
            self.conservative_multiplier,
            field_name="conservative_multiplier",
        )
        aggressive_multiplier = ensure_non_negative_number(
            self.aggressive_multiplier,
            field_name="aggressive_multiplier",
        )
        instability_penalty = ensure_non_negative_number(
            self.instability_penalty,
            field_name="instability_penalty",
        )

        if max_step_size < min_step_size:
            raise ValueError("max_step_size must be greater than or equal to min_step_size")
        if base_step_size < min_step_size or base_step_size > max_step_size:
            raise ValueError("base_step_size must be between min_step_size and max_step_size")

        object.__setattr__(self, "base_step_size", base_step_size)
        object.__setattr__(self, "min_step_size", min_step_size)
        object.__setattr__(self, "max_step_size", max_step_size)
        object.__setattr__(self, "conservative_multiplier", conservative_multiplier)
        object.__setattr__(self, "aggressive_multiplier", aggressive_multiplier)
        object.__setattr__(self, "instability_penalty", instability_penalty)


@dataclass(frozen=True)
class StepSizeAssessment:
    """Serializable-friendly result of one step-size computation."""

    step_size: float
    mode: StepSizeMode
    status: StepSizeStatus
    precision_scale_value: float
    temperature_scale_value: float
    stability_scale_value: float
    energy_scale_value: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        step_size = ensure_non_negative_number(self.step_size, field_name="step_size")
        if not isinstance(self.mode, StepSizeMode):
            raise TypeError("mode must be StepSizeMode")
        if not isinstance(self.status, StepSizeStatus):
            raise TypeError("status must be StepSizeStatus")

        precision_scale_value = ensure_constraint_probability(
            self.precision_scale_value,
            field_name="precision_scale_value",
        )
        temperature_scale_value = ensure_constraint_probability(
            self.temperature_scale_value,
            field_name="temperature_scale_value",
        )
        stability_scale_value = ensure_constraint_probability(
            self.stability_scale_value,
            field_name="stability_scale_value",
        )
        energy_scale_value = ensure_constraint_probability(
            self.energy_scale_value,
            field_name="energy_scale_value",
        )
        _ensure_timezone_aware(self.created_at, "created_at")

        object.__setattr__(self, "step_size", step_size)
        object.__setattr__(self, "precision_scale_value", precision_scale_value)
        object.__setattr__(self, "temperature_scale_value", temperature_scale_value)
        object.__setattr__(self, "stability_scale_value", stability_scale_value)
        object.__setattr__(self, "energy_scale_value", energy_scale_value)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "step_size": self.step_size,
            "mode": self.mode.value,
            "status": self.status.value,
            "precision_scale_value": self.precision_scale_value,
            "temperature_scale_value": self.temperature_scale_value,
            "stability_scale_value": self.stability_scale_value,
            "energy_scale_value": self.energy_scale_value,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class StepSizeState:
    """Runtime descriptor of step-size regulation state."""

    step_size: float
    mode: StepSizeMode
    status: StepSizeStatus
    update_count: int
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        step_size = ensure_non_negative_number(self.step_size, field_name="step_size")
        if not isinstance(self.mode, StepSizeMode):
            raise TypeError("mode must be StepSizeMode")
        if not isinstance(self.status, StepSizeStatus):
            raise TypeError("status must be StepSizeStatus")
        update_count = _ensure_non_negative_int(self.update_count, "update_count")
        _ensure_timezone_aware(self.created_at, "created_at")
        _ensure_timezone_aware(self.updated_at, "updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be greater than or equal to created_at")

        object.__setattr__(self, "step_size", step_size)
        object.__setattr__(self, "update_count", update_count)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "step_size": self.step_size,
            "mode": self.mode.value,
            "status": self.status.value,
            "update_count": self.update_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }


def clamp_step_size(
    *,
    value: float,
    policy: StepSizePolicy,
) -> float:
    """Clamp a step-size value to the policy bounds."""

    value = ensure_non_negative_number(value, field_name="value")
    if not isinstance(policy, StepSizePolicy):
        raise TypeError("policy must be StepSizePolicy")
    return min(max(value, policy.min_step_size), policy.max_step_size)


def temperature_step_scale(temperature: SemanticTemperatureState | None) -> float:
    """Return the temperature contribution to step size."""

    if temperature is None:
        return 1.0
    if not isinstance(temperature, SemanticTemperatureState):
        raise TypeError("temperature must be SemanticTemperatureState or None")
    if semantic_temperature_is_frozen(temperature):
        return 0.0
    return ensure_constraint_probability(
        transition_scale_from_temperature(temperature),
        field_name="temperature_step_scale",
    )


def stability_step_scale(stability: StabilityReport | None) -> float:
    """Return the stability contribution to step size."""

    if stability is None:
        return 1.0
    if not isinstance(stability, StabilityReport):
        raise TypeError("stability must be StabilityReport or None")
    if stability_report_is_blocked(stability):
        return 0.0
    return ensure_constraint_probability(stability.score, field_name="stability_step_scale")


def energy_step_scale(energy: EnergyState | None) -> float:
    """Return the energy contribution to step size."""

    if energy is None:
        return 1.0
    if not isinstance(energy, EnergyState):
        raise TypeError("energy must be EnergyState or None")
    if energy.status is EnergyStatus.INVALID:
        return 0.0
    if energy.normalized_energy is not None:
        return _clamp_probability(1.0 - energy.normalized_energy)
    return _clamp_probability(1.0 / (1.0 + energy.energy))


def step_size_mode_from_inputs(
    *,
    precision: PrecisionState | None = None,
    temperature: SemanticTemperatureState | None = None,
    stability: StabilityReport | None = None,
    energy: EnergyState | None = None,
) -> StepSizeMode:
    """Determine the step-size mode from regulation inputs."""

    _validate_optional_inputs(
        precision=precision,
        temperature=temperature,
        stability=stability,
        energy=energy,
    )
    if precision is not None and not precision_allows_update(precision):
        return StepSizeMode.BLOCKED
    if temperature is not None and semantic_temperature_is_frozen(temperature):
        return StepSizeMode.FROZEN
    if stability is not None and stability_report_is_blocked(stability):
        return StepSizeMode.BLOCKED
    if energy is not None and energy.status is EnergyStatus.DEGRADED:
        return StepSizeMode.REDUCED
    if stability is not None and stability.status in (
        StabilityStatus.DEGRADED,
        StabilityStatus.UNSTABLE,
    ):
        return StepSizeMode.REDUCED
    if precision is not None and precision.status is PrecisionStatus.HIGH:
        return StepSizeMode.EXPANDED
    return StepSizeMode.NORMAL


def step_size_status_from_mode(
    mode: StepSizeMode,
) -> StepSizeStatus:
    """Map a step-size mode to a status."""

    if not isinstance(mode, StepSizeMode):
        raise TypeError("mode must be StepSizeMode")
    if mode is StepSizeMode.BLOCKED:
        return StepSizeStatus.BLOCKED
    if mode is StepSizeMode.FROZEN:
        return StepSizeStatus.SKIPPED
    if mode is StepSizeMode.REDUCED:
        return StepSizeStatus.REDUCED
    return StepSizeStatus.READY


def compute_step_size_assessment(
    *,
    precision: PrecisionState | None = None,
    temperature: SemanticTemperatureState | None = None,
    stability: StabilityReport | None = None,
    energy: EnergyState | None = None,
    policy: StepSizePolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> StepSizeAssessment:
    """Compute bounded step-size assessment from regulation inputs."""

    if policy is None:
        policy = StepSizePolicy()
    if not isinstance(policy, StepSizePolicy):
        raise TypeError("policy must be StepSizePolicy")
    _validate_optional_inputs(
        precision=precision,
        temperature=temperature,
        stability=stability,
        energy=energy,
    )

    precision_scale_value = (
        ensure_constraint_probability(
            precision_scale(precision),
            field_name="precision_scale_value",
        )
        if precision is not None
        else 1.0
    )
    temperature_scale_value = temperature_step_scale(temperature)
    stability_scale_value = stability_step_scale(stability)
    energy_scale_value = energy_step_scale(energy)
    mode = step_size_mode_from_inputs(
        precision=precision,
        temperature=temperature,
        stability=stability,
        energy=energy,
    )

    raw_step = (
        policy.base_step_size
        * precision_scale_value
        * temperature_scale_value
        * stability_scale_value
        * energy_scale_value
    )
    if mode is StepSizeMode.REDUCED:
        raw_step *= policy.conservative_multiplier
    elif mode is StepSizeMode.EXPANDED:
        raw_step *= policy.aggressive_multiplier
    elif mode in (StepSizeMode.FROZEN, StepSizeMode.BLOCKED):
        raw_step = 0.0

    default_metadata: dict[str, object] = {
        "source": "step_size_regulation",
        "has_precision": precision is not None,
        "has_temperature": temperature is not None,
        "has_stability": stability is not None,
        "has_energy": energy is not None,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "step_size_regulation"

    return StepSizeAssessment(
        step_size=clamp_step_size(value=raw_step, policy=policy),
        mode=mode,
        status=step_size_status_from_mode(mode),
        precision_scale_value=precision_scale_value,
        temperature_scale_value=temperature_scale_value,
        stability_scale_value=stability_scale_value,
        energy_scale_value=energy_scale_value,
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def create_step_size_state(
    *,
    assessment: StepSizeAssessment | None = None,
    policy: StepSizePolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> StepSizeState:
    """Create initial step-size state."""

    if assessment is None:
        assessment = compute_step_size_assessment(policy=policy)
    if not isinstance(assessment, StepSizeAssessment):
        raise TypeError("assessment must be StepSizeAssessment")

    timestamp = utc_now()
    default_metadata: dict[str, object] = {"state_kind": "step_size_state"}
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["state_kind"] = "step_size_state"

    return StepSizeState(
        step_size=assessment.step_size,
        mode=assessment.mode,
        status=assessment.status,
        update_count=0,
        created_at=timestamp,
        updated_at=timestamp,
        metadata=merged_metadata,
    )


def transition_step_size_state(
    *,
    state: StepSizeState,
    assessment: StepSizeAssessment,
    metadata: dict[str, object] | None = None,
) -> StepSizeState:
    """Create next step-size state without mutating prior state."""

    if not isinstance(state, StepSizeState):
        raise TypeError("state must be StepSizeState")
    if not isinstance(assessment, StepSizeAssessment):
        raise TypeError("assessment must be StepSizeAssessment")

    default_metadata: dict[str, object] = {
        "previous_step_size": state.step_size,
        "assessment_status": assessment.status.value,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)

    return StepSizeState(
        step_size=assessment.step_size,
        mode=assessment.mode,
        status=assessment.status,
        update_count=state.update_count + 1,
        created_at=state.created_at,
        updated_at=utc_now(),
        metadata=merged_metadata,
    )


def step_size_allows_update(state: StepSizeState) -> bool:
    """Return True when the step-size state permits an update."""

    if not isinstance(state, StepSizeState):
        raise TypeError("state must be StepSizeState")
    if state.status in (
        StepSizeStatus.BLOCKED,
        StepSizeStatus.INVALID,
        StepSizeStatus.SKIPPED,
    ):
        return False
    return state.step_size > 0.0


def step_size_scale(state: StepSizeState) -> float:
    """Return the current regulated step size."""

    if not isinstance(state, StepSizeState):
        raise TypeError("state must be StepSizeState")
    return state.step_size


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


def _ensure_timezone_aware(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _validate_optional_inputs(
    *,
    precision: PrecisionState | None,
    temperature: SemanticTemperatureState | None,
    stability: StabilityReport | None,
    energy: EnergyState | None,
) -> None:
    if precision is not None and not isinstance(precision, PrecisionState):
        raise TypeError("precision must be PrecisionState or None")
    if temperature is not None and not isinstance(temperature, SemanticTemperatureState):
        raise TypeError("temperature must be SemanticTemperatureState or None")
    if stability is not None and not isinstance(stability, StabilityReport):
        raise TypeError("stability must be StabilityReport or None")
    if energy is not None and not isinstance(energy, EnergyState):
        raise TypeError("energy must be EnergyState or None")


__all__ = [
    "StepSizeMode",
    "StepSizeStatus",
    "StepSizePolicy",
    "StepSizeAssessment",
    "StepSizeState",
    "clamp_step_size",
    "temperature_step_scale",
    "stability_step_scale",
    "energy_step_scale",
    "step_size_mode_from_inputs",
    "step_size_status_from_mode",
    "compute_step_size_assessment",
    "create_step_size_state",
    "transition_step_size_state",
    "step_size_allows_update",
    "step_size_scale",
]
