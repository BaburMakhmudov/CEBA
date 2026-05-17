"""Deterministic attractor utilities for CEBA Core Telos Processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ceba.core.metrics.schema import (
    ensure_constraint_probability,
    ensure_finite_number,
    ensure_non_negative_number,
    utc_now,
)
from ceba.core.telos.priority import PriorityScore, score_attractor_priority
from ceba.core.telos.schema import (
    AttractorSignature,
    TelosKind,
    TelosPriority,
    TelosVectorRef,
    ensure_non_empty_string,
    new_attractor_id,
)
from ceba.core.telos.telos_vector import (
    TelosVectorPayload,
    cosine_alignment_score,
    vector_payloads_compatible,
)


class AttractorSelectionStatus(Enum):
    """Status for deterministic attractor selection."""

    SELECTED = "SELECTED"
    EMPTY = "EMPTY"
    BELOW_THRESHOLD = "BELOW_THRESHOLD"
    DEGRADED = "DEGRADED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class AttractorScore:
    """Serializable-friendly attractor score descriptor."""

    attractor: AttractorSignature
    priority_score: PriorityScore
    alignment_score: float | None
    final_score: float
    status: AttractorSelectionStatus
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.attractor, AttractorSignature):
            raise TypeError("attractor must be AttractorSignature")
        if not isinstance(self.priority_score, PriorityScore):
            raise TypeError("priority_score must be PriorityScore")
        alignment_score = (
            None
            if self.alignment_score is None
            else ensure_constraint_probability(
                self.alignment_score,
                field_name="alignment_score",
            )
        )
        final_score = ensure_non_negative_number(
            self.final_score,
            field_name="final_score",
        )
        if not isinstance(self.status, AttractorSelectionStatus):
            raise TypeError("status must be AttractorSelectionStatus")

        object.__setattr__(self, "alignment_score", alignment_score)
        object.__setattr__(self, "final_score", final_score)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "attractor": self.attractor.to_dict(),
            "priority_score": self.priority_score.to_dict(),
            "alignment_score": self.alignment_score,
            "final_score": self.final_score,
            "status": self.status.value,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AttractorSelectionResult:
    """Serializable-friendly result of deterministic attractor selection."""

    selected: AttractorSignature | None
    scores: tuple[AttractorScore, ...]
    status: AttractorSelectionStatus
    threshold: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.selected is not None and not isinstance(
            self.selected,
            AttractorSignature,
        ):
            raise TypeError("selected must be AttractorSignature or None")
        scores = tuple(self.scores)
        for score in scores:
            if not isinstance(score, AttractorScore):
                raise TypeError("scores must contain AttractorScore")
        if not isinstance(self.status, AttractorSelectionStatus):
            raise TypeError("status must be AttractorSelectionStatus")
        threshold = ensure_constraint_probability(
            self.threshold,
            field_name="threshold",
        )
        _ensure_timezone_aware(self.created_at, "created_at")

        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    @property
    def selected_id(self) -> str | None:
        """Return the selected attractor id, if present."""

        if self.selected is None:
            return None
        return str(self.selected.attractor_id)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "selected": None if self.selected is None else self.selected.to_dict(),
            "scores": tuple(score.to_dict() for score in self.scores),
            "status": self.status.value,
            "threshold": self.threshold,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


def clamp_probability(value: float) -> float:
    """Clamp a finite number into the probability range."""

    value = ensure_finite_number(value, field_name="value")
    return min(max(value, 0.0), 1.0)


def create_attractor_signature(
    *,
    name: str,
    kind: TelosKind = TelosKind.CONTEXTUAL,
    vector: TelosVectorRef | None = None,
    strength: float = 1.0,
    basin_radius: float | None = None,
    stability_weight: float = 1.0,
    metadata: dict[str, object] | None = None,
) -> AttractorSignature:
    """Create validated attractor signature."""

    name = ensure_non_empty_string(name, "name")
    if not isinstance(kind, TelosKind):
        raise TypeError("kind must be TelosKind")
    if vector is not None and not isinstance(vector, TelosVectorRef):
        raise TypeError("vector must be TelosVectorRef or None")
    strength = ensure_constraint_probability(strength, field_name="strength")
    basin_radius = (
        None
        if basin_radius is None
        else ensure_non_negative_number(basin_radius, field_name="basin_radius")
    )
    stability_weight = ensure_non_negative_number(
        stability_weight,
        field_name="stability_weight",
    )

    default_metadata: dict[str, object] = {"source": "telos_attractor"}
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "telos_attractor"

    return AttractorSignature(
        attractor_id=new_attractor_id(),
        name=name,
        kind=kind,
        vector=vector,
        strength=strength,
        basin_radius=basin_radius,
        stability_weight=stability_weight,
        metadata=merged_metadata,
    )


def create_attractor_from_payload(
    *,
    payload: TelosVectorPayload,
    name: str,
    kind: TelosKind = TelosKind.CONTEXTUAL,
    strength: float = 1.0,
    basin_radius: float | None = None,
    stability_weight: float = 1.0,
    metadata: dict[str, object] | None = None,
) -> AttractorSignature:
    """Create attractor signature from TelosVectorPayload."""

    if not isinstance(payload, TelosVectorPayload):
        raise TypeError("payload must be TelosVectorPayload")

    name = ensure_non_empty_string(name, "name")
    if not isinstance(kind, TelosKind):
        raise TypeError("kind must be TelosKind")

    strength = ensure_constraint_probability(strength, field_name="strength")
    basin_radius = (
        None
        if basin_radius is None
        else ensure_non_negative_number(basin_radius, field_name="basin_radius")
    )
    stability_weight = ensure_non_negative_number(
        stability_weight,
        field_name="stability_weight",
    )

    default_metadata: dict[str, object] = {
        "source": "telos_attractor_from_payload",
        "payload_vector_id": payload.vector_ref.vector_id,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "telos_attractor_from_payload"

    return AttractorSignature(
        attractor_id=new_attractor_id(),
        name=name,
        kind=kind,
        vector=payload.vector_ref,
        strength=strength,
        basin_radius=basin_radius,
        stability_weight=stability_weight,
        metadata=merged_metadata,
    )


def attractor_is_active(
    attractor: AttractorSignature,
    *,
    min_strength: float = 0.0,
) -> bool:
    """Return True when an attractor meets the minimum strength."""

    if not isinstance(attractor, AttractorSignature):
        raise TypeError("attractor must be AttractorSignature")
    min_strength = ensure_constraint_probability(
        min_strength,
        field_name="min_strength",
    )
    return attractor.strength >= min_strength


def attractor_base_score(
    attractor: AttractorSignature,
) -> float:
    """Compute base attractor score."""

    if not isinstance(attractor, AttractorSignature):
        raise TypeError("attractor must be AttractorSignature")
    return ensure_non_negative_number(
        attractor.strength * (1.0 + attractor.stability_weight),
        field_name="attractor_base_score",
    )


def score_attractor(
    *,
    attractor: AttractorSignature,
    priority: TelosPriority,
    source_payload: TelosVectorPayload | None = None,
    target_payload: TelosVectorPayload | None = None,
    max_score: float = 1.0,
    metadata: dict[str, object] | None = None,
) -> AttractorScore:
    """Compute deterministic attractor score."""

    if not isinstance(attractor, AttractorSignature):
        raise TypeError("attractor must be AttractorSignature")
    if not isinstance(priority, TelosPriority):
        raise TypeError("priority must be TelosPriority")
    if source_payload is not None and not isinstance(
        source_payload,
        TelosVectorPayload,
    ):
        raise TypeError("source_payload must be TelosVectorPayload or None")
    if target_payload is not None and not isinstance(
        target_payload,
        TelosVectorPayload,
    ):
        raise TypeError("target_payload must be TelosVectorPayload or None")

    alignment_score: float | None = None
    if source_payload is not None and target_payload is not None:
        if not vector_payloads_compatible(source_payload, target_payload):
            raise ValueError("source_payload and target_payload must be compatible")
        alignment_score = cosine_alignment_score(
            source_payload.values,
            target_payload.values,
        )

    priority_score = score_attractor_priority(
        attractor,
        priority=priority,
        max_score=max_score,
    )
    final_score = priority_score.normalized_score * attractor.strength
    if alignment_score is not None:
        final_score *= alignment_score
    final_score = ensure_non_negative_number(final_score, field_name="final_score")
    status = (
        AttractorSelectionStatus.SELECTED
        if final_score > 0.0
        else AttractorSelectionStatus.BELOW_THRESHOLD
    )

    default_metadata: dict[str, object] = {
        "source": "attractor_score",
        "attractor_id": str(attractor.attractor_id),
        "attractor_name": attractor.name,
        "has_alignment": alignment_score is not None,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "attractor_score"

    return AttractorScore(
        attractor=attractor,
        priority_score=priority_score,
        alignment_score=alignment_score,
        final_score=final_score,
        status=status,
        metadata=merged_metadata,
    )


def score_attractors(
    *,
    attractors: tuple[AttractorSignature, ...] | list[AttractorSignature],
    priority: TelosPriority,
    source_payload: TelosVectorPayload | None = None,
    target_payload: TelosVectorPayload | None = None,
    max_score: float = 1.0,
) -> tuple[AttractorScore, ...]:
    """Score attractors preserving input order."""

    values = tuple(attractors)
    for attractor in values:
        if not isinstance(attractor, AttractorSignature):
            raise TypeError("attractors must contain AttractorSignature")
    if not isinstance(priority, TelosPriority):
        raise TypeError("priority must be TelosPriority")
    return tuple(
        score_attractor(
            attractor=attractor,
            priority=priority,
            source_payload=source_payload,
            target_payload=target_payload,
            max_score=max_score,
        )
        for attractor in values
    )


def sort_attractor_scores(
    scores: tuple[AttractorScore, ...] | list[AttractorScore],
) -> tuple[AttractorScore, ...]:
    """Sort attractor scores deterministically."""

    values = tuple(scores)
    for score in values:
        if not isinstance(score, AttractorScore):
            raise TypeError("scores must contain AttractorScore")
    return tuple(
        sorted(
            values,
            key=lambda item: (
                -item.final_score,
                -item.attractor.strength,
                -item.attractor.stability_weight,
                item.attractor.name,
            ),
        )
    )


def select_best_attractor(
    *,
    attractors: tuple[AttractorSignature, ...] | list[AttractorSignature],
    priority: TelosPriority,
    source_payload: TelosVectorPayload | None = None,
    target_payload: TelosVectorPayload | None = None,
    threshold: float = 0.0,
    metadata: dict[str, object] | None = None,
) -> AttractorSelectionResult:
    """Select best attractor deterministically."""

    threshold = ensure_constraint_probability(threshold, field_name="threshold")
    values = tuple(attractors)
    for attractor in values:
        if not isinstance(attractor, AttractorSignature):
            raise TypeError("attractors must contain AttractorSignature")

    default_metadata: dict[str, object] = {
        "source": "attractor_selection",
        "attractor_count": len(values),
        "threshold": threshold,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "attractor_selection"

    if not values:
        return AttractorSelectionResult(
            selected=None,
            scores=(),
            status=AttractorSelectionStatus.EMPTY,
            threshold=threshold,
            created_at=utc_now(),
            metadata=merged_metadata,
        )

    sorted_scores = sort_attractor_scores(
        score_attractors(
            attractors=values,
            priority=priority,
            source_payload=source_payload,
            target_payload=target_payload,
        )
    )
    selected: AttractorSignature | None = None
    for score in sorted_scores:
        if score.final_score >= threshold:
            selected = score.attractor
            break

    status = (
        AttractorSelectionStatus.SELECTED
        if selected is not None
        else AttractorSelectionStatus.BELOW_THRESHOLD
    )
    return AttractorSelectionResult(
        selected=selected,
        scores=sorted_scores,
        status=status,
        threshold=threshold,
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def active_attractors(
    attractors: tuple[AttractorSignature, ...] | list[AttractorSignature],
    *,
    min_strength: float = 0.0,
) -> tuple[AttractorSignature, ...]:
    """Return active attractors preserving original order."""

    values = tuple(attractors)
    for attractor in values:
        if not isinstance(attractor, AttractorSignature):
            raise TypeError("attractors must contain AttractorSignature")
    return tuple(
        attractor
        for attractor in values
        if attractor_is_active(attractor, min_strength=min_strength)
    )


def attractor_summary(
    attractors: tuple[AttractorSignature, ...] | list[AttractorSignature],
) -> dict[str, object]:
    """Summarize attractor signatures."""

    values = tuple(attractors)
    for attractor in values:
        if not isinstance(attractor, AttractorSignature):
            raise TypeError("attractors must contain AttractorSignature")

    kinds: dict[str, int] = {}
    with_vector_count = 0
    total_strength = 0.0
    total_stability_weight = 0.0
    strongest: AttractorSignature | None = None
    for attractor in values:
        kinds[attractor.kind.value] = kinds.get(attractor.kind.value, 0) + 1
        if attractor.vector is not None:
            with_vector_count += 1
        total_strength += attractor.strength
        total_stability_weight += attractor.stability_weight
        if strongest is None or attractor.strength > strongest.strength:
            strongest = attractor

    count = len(values)
    return {
        "count": count,
        "kinds": kinds,
        "with_vector_count": with_vector_count,
        "average_strength": 0.0 if count == 0 else total_strength / count,
        "average_stability_weight": (
            0.0 if count == 0 else total_stability_weight / count
        ),
        "strongest_attractor_id": (
            None if strongest is None else str(strongest.attractor_id)
        ),
        "strongest_attractor_name": None if strongest is None else strongest.name,
    }


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


def _ensure_timezone_aware(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


__all__ = [
    "AttractorSelectionStatus",
    "AttractorScore",
    "AttractorSelectionResult",
    "clamp_probability",
    "create_attractor_signature",
    "create_attractor_from_payload",
    "attractor_is_active",
    "attractor_base_score",
    "score_attractor",
    "score_attractors",
    "sort_attractor_scores",
    "select_best_attractor",
    "active_attractors",
    "attractor_summary",
]
