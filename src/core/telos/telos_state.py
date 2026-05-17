"""Deterministic TelosState helpers for CEBA Core Telos Processing."""

from __future__ import annotations

from dataclasses import dataclass, field

from ceba.core.metrics.schema import ensure_constraint_probability, utc_now
from ceba.core.telos.alignment import best_alignment
from ceba.core.telos.attractor import active_attractors, attractor_summary
from ceba.core.telos.priority import sort_attractors_by_priority
from ceba.core.telos.schema import (
    AttractorSignature,
    TelosAlignmentResult,
    TelosPriority,
    TelosState,
    TelosStatus,
    create_telos_priority,
    create_telos_state,
    ensure_non_empty_string,
    ensure_non_negative_int,
    new_telos_state_id,
)


@dataclass(frozen=True)
class TelosStateTransition:
    """Serializable-friendly description of one TelosState transition."""

    previous_state_id: str | None
    next_state_id: str
    previous_status: TelosStatus | None
    next_status: TelosStatus
    previous_alignment_score: float | None
    next_alignment_score: float
    selected_attractor_id: str | None
    active_attractor_count: int
    update_count_before: int
    update_count_after: int
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        previous_state_id = (
            None
            if self.previous_state_id is None
            else ensure_non_empty_string(self.previous_state_id, "previous_state_id")
        )
        next_state_id = ensure_non_empty_string(self.next_state_id, "next_state_id")
        if self.previous_status is not None and not isinstance(
            self.previous_status,
            TelosStatus,
        ):
            raise TypeError("previous_status must be TelosStatus or None")
        if not isinstance(self.next_status, TelosStatus):
            raise TypeError("next_status must be TelosStatus")
        previous_alignment_score = (
            None
            if self.previous_alignment_score is None
            else ensure_constraint_probability(
                self.previous_alignment_score,
                field_name="previous_alignment_score",
            )
        )
        next_alignment_score = ensure_constraint_probability(
            self.next_alignment_score,
            field_name="next_alignment_score",
        )
        selected_attractor_id = (
            None
            if self.selected_attractor_id is None
            else ensure_non_empty_string(
                self.selected_attractor_id,
                "selected_attractor_id",
            )
        )
        active_attractor_count = ensure_non_negative_int(
            self.active_attractor_count,
            "active_attractor_count",
        )
        update_count_before = ensure_non_negative_int(
            self.update_count_before,
            "update_count_before",
        )
        update_count_after = ensure_non_negative_int(
            self.update_count_after,
            "update_count_after",
        )
        if update_count_after < update_count_before:
            raise ValueError("update_count_after must be >= update_count_before")

        object.__setattr__(self, "previous_state_id", previous_state_id)
        object.__setattr__(self, "next_state_id", next_state_id)
        object.__setattr__(
            self,
            "previous_alignment_score",
            previous_alignment_score,
        )
        object.__setattr__(self, "next_alignment_score", next_alignment_score)
        object.__setattr__(self, "selected_attractor_id", selected_attractor_id)
        object.__setattr__(self, "active_attractor_count", active_attractor_count)
        object.__setattr__(self, "update_count_before", update_count_before)
        object.__setattr__(self, "update_count_after", update_count_after)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "previous_state_id": self.previous_state_id,
            "next_state_id": self.next_state_id,
            "previous_status": (
                None if self.previous_status is None else self.previous_status.value
            ),
            "next_status": self.next_status.value,
            "previous_alignment_score": self.previous_alignment_score,
            "next_alignment_score": self.next_alignment_score,
            "selected_attractor_id": self.selected_attractor_id,
            "active_attractor_count": self.active_attractor_count,
            "update_count_before": self.update_count_before,
            "update_count_after": self.update_count_after,
            "metadata": dict(self.metadata),
        }


def telos_status_from_alignment(
    *,
    alignment_score: float,
    active_attractor_count: int,
    degraded_threshold: float = 0.35,
    active_threshold: float = 0.50,
) -> TelosStatus:
    """Infer telos status from alignment score and active attractor count."""

    alignment_score = ensure_constraint_probability(
        alignment_score,
        field_name="alignment_score",
    )
    active_attractor_count = ensure_non_negative_int(
        active_attractor_count,
        "active_attractor_count",
    )
    degraded_threshold = ensure_constraint_probability(
        degraded_threshold,
        field_name="degraded_threshold",
    )
    active_threshold = ensure_constraint_probability(
        active_threshold,
        field_name="active_threshold",
    )
    if active_threshold < degraded_threshold:
        raise ValueError("active_threshold must be >= degraded_threshold")
    if active_attractor_count <= 0:
        return TelosStatus.INACTIVE
    if alignment_score >= active_threshold:
        return TelosStatus.ACTIVE
    return TelosStatus.DEGRADED


def select_primary_attractor(
    *,
    attractors: tuple[AttractorSignature, ...] | list[AttractorSignature],
    priority: TelosPriority,
    min_strength: float = 0.0,
) -> AttractorSignature | None:
    """Select primary attractor from active attractors."""

    if not isinstance(priority, TelosPriority):
        raise TypeError("priority must be TelosPriority")
    min_strength = ensure_constraint_probability(
        min_strength,
        field_name="min_strength",
    )
    active = active_attractors(attractors, min_strength=min_strength)
    if not active:
        return None
    return sort_attractors_by_priority(active, priority=priority)[0]


def alignment_score_from_results(
    results: tuple[TelosAlignmentResult, ...] | list[TelosAlignmentResult],
) -> float:
    """Compute state alignment score from alignment results."""

    values = tuple(results)
    for result in values:
        if not isinstance(result, TelosAlignmentResult):
            raise TypeError("results must contain TelosAlignmentResult")
    if not values:
        return 1.0
    best = best_alignment(values)
    if best is None:
        return 1.0
    return ensure_constraint_probability(best.score, field_name="alignment_score")


def create_initial_telos_state(
    *,
    attractors: tuple[AttractorSignature, ...] | list[AttractorSignature] = (),
    priority: TelosPriority | None = None,
    alignment_results: tuple[TelosAlignmentResult, ...]
    | list[TelosAlignmentResult] = (),
    status: TelosStatus | None = None,
    metadata: dict[str, object] | None = None,
) -> TelosState:
    """Create initial TelosState from attractors and alignment results."""

    if priority is None:
        priority = create_telos_priority()
    if not isinstance(priority, TelosPriority):
        raise TypeError("priority must be TelosPriority")
    active = active_attractors(attractors, min_strength=0.0)
    primary = select_primary_attractor(attractors=active, priority=priority)
    alignment_score = alignment_score_from_results(alignment_results)
    if status is None:
        status = telos_status_from_alignment(
            alignment_score=alignment_score,
            active_attractor_count=len(active),
        )
    if not isinstance(status, TelosStatus):
        raise TypeError("status must be TelosStatus or None")

    default_metadata: dict[str, object] = {
        "source": "telos_state_initial",
        "active_attractor_count": len(active),
        "has_alignment_results": bool(tuple(alignment_results)),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "telos_state_initial"

    return create_telos_state(
        primary_attractor=primary,
        active_attractors=active,
        priority=priority,
        status=status,
        alignment_score=alignment_score,
        metadata=merged_metadata,
    )


def transition_telos_state(
    *,
    state: TelosState,
    attractors: tuple[AttractorSignature, ...] | list[AttractorSignature] | None = None,
    priority: TelosPriority | None = None,
    alignment_results: tuple[TelosAlignmentResult, ...]
    | list[TelosAlignmentResult] = (),
    status: TelosStatus | None = None,
    metadata: dict[str, object] | None = None,
) -> tuple[TelosState, TelosStateTransition]:
    """Create next TelosState descriptor without mutating prior state."""

    if not isinstance(state, TelosState):
        raise TypeError("state must be TelosState")
    selected_attractors = state.active_attractors if attractors is None else attractors
    selected_priority = state.priority if priority is None else priority
    if not isinstance(selected_priority, TelosPriority):
        raise TypeError("priority must be TelosPriority or None")

    active = active_attractors(selected_attractors, min_strength=0.0)
    primary = select_primary_attractor(attractors=active, priority=selected_priority)
    alignment_values = tuple(alignment_results)
    alignment_score = (
        alignment_score_from_results(alignment_values)
        if alignment_values
        else state.alignment_score
    )
    if status is None:
        status = telos_status_from_alignment(
            alignment_score=alignment_score,
            active_attractor_count=len(active),
        )
    if not isinstance(status, TelosStatus):
        raise TypeError("status must be TelosStatus or None")

    default_metadata: dict[str, object] = {
        "source": "telos_state_transition",
        "previous_state_id": str(state.state_id),
        "active_attractor_count": len(active),
        "has_alignment_results": bool(alignment_values),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "telos_state_transition"

    next_state = TelosState(
        state_id=new_telos_state_id(),
        status=status,
        primary_attractor=primary,
        active_attractors=active,
        priority=selected_priority,
        alignment_score=alignment_score,
        update_count=state.update_count + 1,
        created_at=state.created_at,
        updated_at=utc_now(),
        metadata=merged_metadata,
    )
    transition = TelosStateTransition(
        previous_state_id=str(state.state_id),
        next_state_id=str(next_state.state_id),
        previous_status=state.status,
        next_status=next_state.status,
        previous_alignment_score=state.alignment_score,
        next_alignment_score=next_state.alignment_score,
        selected_attractor_id=(
            None if primary is None else str(primary.attractor_id)
        ),
        active_attractor_count=len(active),
        update_count_before=state.update_count,
        update_count_after=next_state.update_count,
        metadata=merged_metadata,
    )
    return next_state, transition


def telos_state_is_active(state: TelosState) -> bool:
    """Return True when TelosState is active."""

    if not isinstance(state, TelosState):
        raise TypeError("state must be TelosState")
    return state.status is TelosStatus.ACTIVE


def telos_state_is_degraded(state: TelosState) -> bool:
    """Return True when TelosState is degraded or invalid."""

    if not isinstance(state, TelosState):
        raise TypeError("state must be TelosState")
    return state.status in (TelosStatus.DEGRADED, TelosStatus.INVALID)


def telos_state_has_attractors(state: TelosState) -> bool:
    """Return True when TelosState has active attractors."""

    if not isinstance(state, TelosState):
        raise TypeError("state must be TelosState")
    return len(state.active_attractors) > 0


def summarize_telos_state(state: TelosState) -> dict[str, object]:
    """Return shallow serializable TelosState summary."""

    if not isinstance(state, TelosState):
        raise TypeError("state must be TelosState")
    primary = state.primary_attractor
    return {
        "state_id": str(state.state_id),
        "status": state.status.value,
        "primary_attractor_id": None if primary is None else str(primary.attractor_id),
        "primary_attractor_name": None if primary is None else primary.name,
        "active_attractor_count": len(state.active_attractors),
        "priority": state.priority.to_dict(),
        "alignment_score": state.alignment_score,
        "update_count": state.update_count,
        "created_at": state.created_at.isoformat(),
        "updated_at": state.updated_at.isoformat(),
        "attractor_summary": attractor_summary(state.active_attractors),
        "metadata": dict(state.metadata),
    }


def summarize_telos_transition(
    transition: TelosStateTransition,
) -> dict[str, object]:
    """Return TelosStateTransition as a dictionary."""

    if not isinstance(transition, TelosStateTransition):
        raise TypeError("transition must be TelosStateTransition")
    return transition.to_dict()


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


__all__ = [
    "TelosStateTransition",
    "telos_status_from_alignment",
    "select_primary_attractor",
    "alignment_score_from_results",
    "create_initial_telos_state",
    "transition_telos_state",
    "telos_state_is_active",
    "telos_state_is_degraded",
    "telos_state_has_attractors",
    "summarize_telos_state",
    "summarize_telos_transition",
]
