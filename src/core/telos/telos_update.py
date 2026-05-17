"""Deterministic telos update-planning utilities for CEBA Core Telos Processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ceba.core.dynamics.energy import EnergyState, EnergyStatus
from ceba.core.dynamics.stability import (
    StabilityReport,
    risk_rank,
    stability_report_is_blocked,
    stability_report_is_passed,
)
from ceba.core.metrics.schema import (
    ensure_constraint_probability,
    ensure_finite_number,
    utc_now,
)
from ceba.core.regulation.update_policy import (
    UpdateDecision,
    update_decision_allows_apply,
)
from ceba.core.telos.attractor import (
    AttractorSelectionResult,
    AttractorSelectionStatus,
)
from ceba.core.telos.schema import (
    AttractorSignature,
    TelosAlignmentResult,
    TelosState,
    ensure_non_empty_string,
    ensure_non_negative_int,
)
from ceba.core.telos.telos_state import (
    alignment_score_from_results,
    summarize_telos_state,
    telos_state_is_degraded,
)


class TelosUpdateMode(Enum):
    """Stable telos update planning mode."""

    NONE = "NONE"
    ALIGNMENT_REFRESH = "ALIGNMENT_REFRESH"
    ATTRACTOR_SWITCH = "ATTRACTOR_SWITCH"
    PRIORITY_ADJUSTMENT = "PRIORITY_ADJUSTMENT"
    CONSERVATIVE = "CONSERVATIVE"
    BLOCKED = "BLOCKED"


class TelosUpdateAction(Enum):
    """Stable telos update planning action."""

    KEEP = "KEEP"
    REFRESH_ALIGNMENT = "REFRESH_ALIGNMENT"
    SWITCH_ATTRACTOR = "SWITCH_ATTRACTOR"
    ADJUST_PRIORITY = "ADJUST_PRIORITY"
    DEFER = "DEFER"
    BLOCK = "BLOCK"


class TelosUpdateStatus(Enum):
    """Stable telos update planning status."""

    READY = "READY"
    SKIPPED = "SKIPPED"
    DEFERRED = "DEFERRED"
    BLOCKED = "BLOCKED"
    DEGRADED = "DEGRADED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class TelosUpdatePolicy:
    """Stable policy for telos update planning."""

    min_alignment_to_keep: float = 0.50
    min_alignment_to_refresh: float = 0.35
    max_risk_rank_to_update: int = 2
    allow_attractor_switch: bool = True
    allow_priority_adjustment: bool = True
    require_update_allowed: bool = True
    require_stability_pass: bool = True

    def __post_init__(self) -> None:
        min_alignment_to_keep = ensure_constraint_probability(
            self.min_alignment_to_keep,
            field_name="min_alignment_to_keep",
        )
        min_alignment_to_refresh = ensure_constraint_probability(
            self.min_alignment_to_refresh,
            field_name="min_alignment_to_refresh",
        )
        if min_alignment_to_keep < min_alignment_to_refresh:
            raise ValueError(
                "min_alignment_to_keep must be greater than or equal to "
                "min_alignment_to_refresh"
            )
        max_risk_rank_to_update = ensure_risk_rank(
            self.max_risk_rank_to_update,
            "max_risk_rank_to_update",
        )
        for field_name in (
            "allow_attractor_switch",
            "allow_priority_adjustment",
            "require_update_allowed",
            "require_stability_pass",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")

        object.__setattr__(
            self,
            "min_alignment_to_keep",
            min_alignment_to_keep,
        )
        object.__setattr__(
            self,
            "min_alignment_to_refresh",
            min_alignment_to_refresh,
        )
        object.__setattr__(
            self,
            "max_risk_rank_to_update",
            max_risk_rank_to_update,
        )


@dataclass(frozen=True)
class TelosUpdatePlan:
    """Serializable-friendly telos update plan."""

    mode: TelosUpdateMode
    action: TelosUpdateAction
    status: TelosUpdateStatus
    state_id: str
    selected_attractor_id: str | None
    selected_attractor_name: str | None
    alignment_score: float
    confidence: float
    reason_code: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.mode, TelosUpdateMode):
            raise TypeError("mode must be TelosUpdateMode")
        if not isinstance(self.action, TelosUpdateAction):
            raise TypeError("action must be TelosUpdateAction")
        if not isinstance(self.status, TelosUpdateStatus):
            raise TypeError("status must be TelosUpdateStatus")
        state_id = ensure_non_empty_string(self.state_id, "state_id")
        selected_attractor_id = (
            None
            if self.selected_attractor_id is None
            else ensure_non_empty_string(
                self.selected_attractor_id,
                "selected_attractor_id",
            )
        )
        selected_attractor_name = (
            None
            if self.selected_attractor_name is None
            else ensure_non_empty_string(
                self.selected_attractor_name,
                "selected_attractor_name",
            )
        )
        alignment_score = ensure_constraint_probability(
            self.alignment_score,
            field_name="alignment_score",
        )
        confidence = ensure_constraint_probability(
            self.confidence,
            field_name="confidence",
        )
        reason_code = ensure_non_empty_string(self.reason_code, "reason_code")
        created_at = _ensure_timezone_aware(self.created_at, "created_at")

        object.__setattr__(self, "state_id", state_id)
        object.__setattr__(self, "selected_attractor_id", selected_attractor_id)
        object.__setattr__(self, "selected_attractor_name", selected_attractor_name)
        object.__setattr__(self, "alignment_score", alignment_score)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "reason_code", reason_code)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "mode": self.mode.value,
            "action": self.action.value,
            "status": self.status.value,
            "state_id": self.state_id,
            "selected_attractor_id": self.selected_attractor_id,
            "selected_attractor_name": self.selected_attractor_name,
            "alignment_score": self.alignment_score,
            "confidence": self.confidence,
            "reason_code": self.reason_code,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TelosUpdatePlanSummary:
    """Serializable-friendly summary of multiple telos update plans."""

    count: int
    ready_count: int
    blocked_count: int
    deferred_count: int
    mean_confidence: float
    actions: dict[str, int]
    modes: dict[str, int]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        count = ensure_non_negative_int(self.count, "count")
        ready_count = ensure_non_negative_int(self.ready_count, "ready_count")
        blocked_count = ensure_non_negative_int(self.blocked_count, "blocked_count")
        deferred_count = ensure_non_negative_int(
            self.deferred_count,
            "deferred_count",
        )
        mean_confidence = ensure_constraint_probability(
            self.mean_confidence,
            field_name="mean_confidence",
        )
        actions = _copy_count_dict(self.actions, "actions")
        modes = _copy_count_dict(self.modes, "modes")

        object.__setattr__(self, "count", count)
        object.__setattr__(self, "ready_count", ready_count)
        object.__setattr__(self, "blocked_count", blocked_count)
        object.__setattr__(self, "deferred_count", deferred_count)
        object.__setattr__(self, "mean_confidence", mean_confidence)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "modes", modes)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "count": self.count,
            "ready_count": self.ready_count,
            "blocked_count": self.blocked_count,
            "deferred_count": self.deferred_count,
            "mean_confidence": self.mean_confidence,
            "actions": dict(self.actions),
            "modes": dict(self.modes),
            "metadata": dict(self.metadata),
        }


def clamp_probability(value: float) -> float:
    """Clamp a finite number to the probability range."""

    value = ensure_finite_number(value, field_name="value")
    return min(max(value, 0.0), 1.0)


def ensure_risk_rank(value: int, field_name: str) -> int:
    """Validate a risk rank in the inclusive range 0..4."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value < 0 or value > 4:
        raise ValueError(f"{field_name} must be in range 0..4")
    return value


def telos_update_mode_from_inputs(
    *,
    state: TelosState,
    alignment_results: tuple[TelosAlignmentResult, ...]
    | list[TelosAlignmentResult] = (),
    attractor_selection: AttractorSelectionResult | None = None,
    update_decision: UpdateDecision | None = None,
    stability: StabilityReport | None = None,
    energy: EnergyState | None = None,
    policy: TelosUpdatePolicy | None = None,
) -> TelosUpdateMode:
    """Infer telos update mode from runtime inputs."""

    policy = _ensure_policy(policy)
    _validate_update_inputs(
        state=state,
        alignment_results=alignment_results,
        attractor_selection=attractor_selection,
        update_decision=update_decision,
        stability=stability,
        energy=energy,
    )

    if (
        update_decision is not None
        and policy.require_update_allowed
        and not update_decision_allows_apply(update_decision)
    ):
        return TelosUpdateMode.BLOCKED
    if stability is not None and stability_report_is_blocked(stability):
        return TelosUpdateMode.BLOCKED
    if (
        stability is not None
        and policy.require_stability_pass
        and not stability_report_is_passed(stability)
    ):
        return TelosUpdateMode.CONSERVATIVE
    if (
        stability is not None
        and risk_rank(stability.risk) > policy.max_risk_rank_to_update
    ):
        return TelosUpdateMode.CONSERVATIVE
    if energy is not None and energy.status is EnergyStatus.INVALID:
        return TelosUpdateMode.BLOCKED

    selected = selected_attractor_from_selection(attractor_selection)
    if (
        attractor_selection is not None
        and attractor_selection.status is AttractorSelectionStatus.SELECTED
        and selected_attractor_changed(state=state, selected=selected)
    ):
        return TelosUpdateMode.ATTRACTOR_SWITCH

    alignment_values = tuple(alignment_results)
    alignment_score = (
        alignment_score_from_results(alignment_values)
        if alignment_values
        else state.alignment_score
    )
    if alignment_score < policy.min_alignment_to_refresh:
        return TelosUpdateMode.ALIGNMENT_REFRESH
    if telos_state_is_degraded(state):
        return TelosUpdateMode.ALIGNMENT_REFRESH
    return TelosUpdateMode.NONE


def telos_update_action_from_mode(
    *,
    mode: TelosUpdateMode,
    policy: TelosUpdatePolicy,
) -> TelosUpdateAction:
    """Convert update mode into update action."""

    if not isinstance(mode, TelosUpdateMode):
        raise TypeError("mode must be TelosUpdateMode")
    if not isinstance(policy, TelosUpdatePolicy):
        raise TypeError("policy must be TelosUpdatePolicy")
    if mode is TelosUpdateMode.BLOCKED:
        return TelosUpdateAction.BLOCK
    if mode is TelosUpdateMode.CONSERVATIVE:
        return TelosUpdateAction.DEFER
    if mode is TelosUpdateMode.ATTRACTOR_SWITCH:
        if policy.allow_attractor_switch:
            return TelosUpdateAction.SWITCH_ATTRACTOR
        return TelosUpdateAction.DEFER
    if mode is TelosUpdateMode.PRIORITY_ADJUSTMENT:
        if policy.allow_priority_adjustment:
            return TelosUpdateAction.ADJUST_PRIORITY
        return TelosUpdateAction.DEFER
    if mode is TelosUpdateMode.ALIGNMENT_REFRESH:
        return TelosUpdateAction.REFRESH_ALIGNMENT
    return TelosUpdateAction.KEEP


def telos_update_status_from_action(action: TelosUpdateAction) -> TelosUpdateStatus:
    """Convert update action into update status."""

    if not isinstance(action, TelosUpdateAction):
        raise TypeError("action must be TelosUpdateAction")
    if action is TelosUpdateAction.KEEP:
        return TelosUpdateStatus.SKIPPED
    if action is TelosUpdateAction.REFRESH_ALIGNMENT:
        return TelosUpdateStatus.READY
    if action is TelosUpdateAction.SWITCH_ATTRACTOR:
        return TelosUpdateStatus.READY
    if action is TelosUpdateAction.ADJUST_PRIORITY:
        return TelosUpdateStatus.READY
    if action is TelosUpdateAction.DEFER:
        return TelosUpdateStatus.DEFERRED
    return TelosUpdateStatus.BLOCKED


def selected_attractor_from_selection(
    selection: AttractorSelectionResult | None,
) -> AttractorSignature | None:
    """Return selected attractor from a selection result."""

    if selection is None:
        return None
    if not isinstance(selection, AttractorSelectionResult):
        raise TypeError("selection must be AttractorSelectionResult or None")
    return selection.selected


def selected_attractor_changed(
    *,
    state: TelosState,
    selected: AttractorSignature | None,
) -> bool:
    """Return True when selected attractor differs from state's primary attractor."""

    if not isinstance(state, TelosState):
        raise TypeError("state must be TelosState")
    if selected is not None and not isinstance(selected, AttractorSignature):
        raise TypeError("selected must be AttractorSignature or None")
    if selected is None:
        return False
    if state.primary_attractor is None:
        return True
    return selected.attractor_id != state.primary_attractor.attractor_id


def confidence_from_inputs(
    *,
    alignment_score: float,
    update_decision: UpdateDecision | None = None,
    stability: StabilityReport | None = None,
    energy: EnergyState | None = None,
) -> float:
    """Compute deterministic confidence from available update inputs."""

    confidence = ensure_constraint_probability(
        alignment_score,
        field_name="alignment_score",
    )
    if update_decision is not None:
        if not isinstance(update_decision, UpdateDecision):
            raise TypeError("update_decision must be UpdateDecision or None")
        confidence *= update_decision.confidence
    if stability is not None:
        if not isinstance(stability, StabilityReport):
            raise TypeError("stability must be StabilityReport or None")
        confidence *= stability.score
    if energy is not None:
        if not isinstance(energy, EnergyState):
            raise TypeError("energy must be EnergyState or None")
        if energy.normalized_energy is not None:
            confidence *= 1.0 - energy.normalized_energy
    return clamp_probability(confidence)


def reason_code_from_update(
    *,
    action: TelosUpdateAction,
    mode: TelosUpdateMode,
) -> str:
    """Build deterministic telos update reason code."""

    if not isinstance(action, TelosUpdateAction):
        raise TypeError("action must be TelosUpdateAction")
    if not isinstance(mode, TelosUpdateMode):
        raise TypeError("mode must be TelosUpdateMode")
    return f"TELOS_UPDATE_{action.value}_{mode.value}"


def build_telos_update_plan(
    *,
    state: TelosState,
    alignment_results: tuple[TelosAlignmentResult, ...]
    | list[TelosAlignmentResult] = (),
    attractor_selection: AttractorSelectionResult | None = None,
    update_decision: UpdateDecision | None = None,
    stability: StabilityReport | None = None,
    energy: EnergyState | None = None,
    policy: TelosUpdatePolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> TelosUpdatePlan:
    """Build deterministic telos update plan."""

    policy = _ensure_policy(policy)
    _validate_update_inputs(
        state=state,
        alignment_results=alignment_results,
        attractor_selection=attractor_selection,
        update_decision=update_decision,
        stability=stability,
        energy=energy,
    )
    mode = telos_update_mode_from_inputs(
        state=state,
        alignment_results=alignment_results,
        attractor_selection=attractor_selection,
        update_decision=update_decision,
        stability=stability,
        energy=energy,
        policy=policy,
    )
    action = telos_update_action_from_mode(mode=mode, policy=policy)
    status = telos_update_status_from_action(action)
    alignment_values = tuple(alignment_results)
    alignment_score = (
        alignment_score_from_results(alignment_values)
        if alignment_values
        else state.alignment_score
    )
    selected = selected_attractor_from_selection(attractor_selection)
    confidence = confidence_from_inputs(
        alignment_score=alignment_score,
        update_decision=update_decision,
        stability=stability,
        energy=energy,
    )
    reason_code = reason_code_from_update(action=action, mode=mode)

    default_metadata: dict[str, object] = {
        "source": "telos_update_plan",
        "state_summary": summarize_telos_state(state),
        "has_alignment_results": bool(alignment_values),
        "has_attractor_selection": attractor_selection is not None,
        "has_update_decision": update_decision is not None,
        "has_stability": stability is not None,
        "has_energy": energy is not None,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "telos_update_plan"

    return TelosUpdatePlan(
        mode=mode,
        action=action,
        status=status,
        state_id=str(state.state_id),
        selected_attractor_id=None if selected is None else str(selected.attractor_id),
        selected_attractor_name=None if selected is None else selected.name,
        alignment_score=alignment_score,
        confidence=confidence,
        reason_code=reason_code,
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def plan_allows_telos_transition(plan: TelosUpdatePlan) -> bool:
    """Return True when plan allows a telos state transition."""

    if not isinstance(plan, TelosUpdatePlan):
        raise TypeError("plan must be TelosUpdatePlan")
    return plan.status is TelosUpdateStatus.READY


def plan_is_blocked(plan: TelosUpdatePlan) -> bool:
    """Return True when plan is blocked or invalid."""

    if not isinstance(plan, TelosUpdatePlan):
        raise TypeError("plan must be TelosUpdatePlan")
    return plan.status in (TelosUpdateStatus.BLOCKED, TelosUpdateStatus.INVALID)


def summarize_telos_update_plans(
    plans: tuple[TelosUpdatePlan, ...] | list[TelosUpdatePlan],
    *,
    metadata: dict[str, object] | None = None,
) -> TelosUpdatePlanSummary:
    """Summarize multiple telos update plans."""

    values = tuple(plans)
    for plan in values:
        if not isinstance(plan, TelosUpdatePlan):
            raise TypeError("plans must contain TelosUpdatePlan")
    count = len(values)
    ready_count = sum(1 for plan in values if plan.status is TelosUpdateStatus.READY)
    blocked_count = sum(
        1
        for plan in values
        if plan.status in (TelosUpdateStatus.BLOCKED, TelosUpdateStatus.INVALID)
    )
    deferred_count = sum(
        1 for plan in values if plan.status is TelosUpdateStatus.DEFERRED
    )
    mean_confidence = (
        0.0 if not values else sum(plan.confidence for plan in values) / count
    )
    actions: dict[str, int] = {}
    modes: dict[str, int] = {}
    for plan in values:
        actions[plan.action.value] = actions.get(plan.action.value, 0) + 1
        modes[plan.mode.value] = modes.get(plan.mode.value, 0) + 1

    default_metadata: dict[str, object] = {
        "source": "telos_update_plan_summary",
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "telos_update_plan_summary"

    return TelosUpdatePlanSummary(
        count=count,
        ready_count=ready_count,
        blocked_count=blocked_count,
        deferred_count=deferred_count,
        mean_confidence=mean_confidence,
        actions=actions,
        modes=modes,
        metadata=merged_metadata,
    )


def telos_update_plan_to_dict(plan: TelosUpdatePlan) -> dict[str, object]:
    """Return TelosUpdatePlan as a dictionary."""

    if not isinstance(plan, TelosUpdatePlan):
        raise TypeError("plan must be TelosUpdatePlan")
    return plan.to_dict()


def _ensure_policy(policy: TelosUpdatePolicy | None) -> TelosUpdatePolicy:
    if policy is None:
        return TelosUpdatePolicy()
    if not isinstance(policy, TelosUpdatePolicy):
        raise TypeError("policy must be TelosUpdatePolicy")
    return policy


def _validate_update_inputs(
    *,
    state: TelosState,
    alignment_results: tuple[TelosAlignmentResult, ...]
    | list[TelosAlignmentResult],
    attractor_selection: AttractorSelectionResult | None,
    update_decision: UpdateDecision | None,
    stability: StabilityReport | None,
    energy: EnergyState | None,
) -> None:
    if not isinstance(state, TelosState):
        raise TypeError("state must be TelosState")
    for result in tuple(alignment_results):
        if not isinstance(result, TelosAlignmentResult):
            raise TypeError("alignment_results must contain TelosAlignmentResult")
    if attractor_selection is not None and not isinstance(
        attractor_selection,
        AttractorSelectionResult,
    ):
        raise TypeError(
            "attractor_selection must be AttractorSelectionResult or None"
        )
    if update_decision is not None and not isinstance(update_decision, UpdateDecision):
        raise TypeError("update_decision must be UpdateDecision or None")
    if stability is not None and not isinstance(stability, StabilityReport):
        raise TypeError("stability must be StabilityReport or None")
    if energy is not None and not isinstance(energy, EnergyState):
        raise TypeError("energy must be EnergyState or None")


def _ensure_timezone_aware(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


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


def _copy_count_dict(values: dict[str, int], field_name: str) -> dict[str, int]:
    if not isinstance(values, dict):
        raise TypeError(f"{field_name} must be dict[str, int]")
    copied = dict(values)
    for key, value in copied.items():
        if not isinstance(key, str):
            raise TypeError(f"{field_name} keys must be strings")
        ensure_non_negative_int(value, f"{field_name}[{key!r}]")
    return copied


__all__ = [
    "TelosUpdateMode",
    "TelosUpdateAction",
    "TelosUpdateStatus",
    "TelosUpdatePolicy",
    "TelosUpdatePlan",
    "TelosUpdatePlanSummary",
    "clamp_probability",
    "ensure_risk_rank",
    "telos_update_mode_from_inputs",
    "telos_update_action_from_mode",
    "telos_update_status_from_action",
    "selected_attractor_from_selection",
    "selected_attractor_changed",
    "confidence_from_inputs",
    "reason_code_from_update",
    "build_telos_update_plan",
    "plan_allows_telos_transition",
    "plan_is_blocked",
    "summarize_telos_update_plans",
    "telos_update_plan_to_dict",
]
