"""Deterministic priority utilities for CEBA Core Telos Processing."""

from __future__ import annotations

from dataclasses import dataclass, field

from ceba.core.metrics.schema import (
    ensure_constraint_probability,
    ensure_finite_number,
    ensure_non_negative_number,
)
from ceba.core.telos.schema import (
    AttractorSignature,
    PriorityLevel,
    TelosPriority,
    create_telos_priority,
)


@dataclass(frozen=True)
class PriorityScore:
    """Serializable-friendly priority score descriptor."""

    priority: TelosPriority
    score: float
    normalized_score: float
    rank: int
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.priority, TelosPriority):
            raise TypeError("priority must be TelosPriority")
        score = ensure_non_negative_number(self.score, field_name="score")
        normalized_score = ensure_constraint_probability(
            self.normalized_score,
            field_name="normalized_score",
        )
        rank = ensure_non_negative_int(self.rank, "rank")

        object.__setattr__(self, "score", score)
        object.__setattr__(self, "normalized_score", normalized_score)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "priority": self.priority.to_dict(),
            "score": self.score,
            "normalized_score": self.normalized_score,
            "rank": self.rank,
            "metadata": dict(self.metadata),
        }


def clamp_probability(value: float) -> float:
    """Clamp a finite number to the probability range."""

    value = ensure_finite_number(value, field_name="value")
    return min(max(value, 0.0), 1.0)


def ensure_non_negative_int(value: int, field_name: str) -> int:
    """Validate a non-negative integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def priority_score_value(priority: TelosPriority) -> float:
    """Compute raw priority score."""

    if not isinstance(priority, TelosPriority):
        raise TypeError("priority must be TelosPriority")
    return ensure_non_negative_number(
        priority.weight / (1.0 + priority.rank),
        field_name="priority_score",
    )


def normalize_priority_score(
    *,
    score: float,
    max_score: float = 1.0,
) -> float:
    """Normalize a raw priority score."""

    score = ensure_non_negative_number(score, field_name="score")
    max_score = ensure_finite_number(max_score, field_name="max_score")
    if max_score <= 0.0:
        raise ValueError("max_score must be positive")
    return clamp_probability(score / max_score)


def score_priority(
    priority: TelosPriority,
    *,
    max_score: float = 1.0,
    metadata: dict[str, object] | None = None,
) -> PriorityScore:
    """Create PriorityScore from TelosPriority."""

    if not isinstance(priority, TelosPriority):
        raise TypeError("priority must be TelosPriority")
    score = priority_score_value(priority)
    default_metadata: dict[str, object] = {
        "source": "telos_priority_score",
        "priority_level": priority.level.value,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "telos_priority_score"

    return PriorityScore(
        priority=priority,
        score=score,
        normalized_score=normalize_priority_score(
            score=score,
            max_score=max_score,
        ),
        rank=priority.rank,
        metadata=merged_metadata,
    )


def score_attractor_priority(
    attractor: AttractorSignature,
    *,
    priority: TelosPriority,
    max_score: float = 1.0,
    metadata: dict[str, object] | None = None,
) -> PriorityScore:
    """Score an attractor using telos priority and attractor strength."""

    if not isinstance(attractor, AttractorSignature):
        raise TypeError("attractor must be AttractorSignature")
    if not isinstance(priority, TelosPriority):
        raise TypeError("priority must be TelosPriority")

    base_score = priority_score_value(priority)
    score = ensure_non_negative_number(
        base_score * attractor.strength * (1.0 + attractor.stability_weight),
        field_name="attractor_priority_score",
    )
    default_metadata: dict[str, object] = {
        "source": "attractor_priority_score",
        "attractor_id": str(attractor.attractor_id),
        "attractor_name": attractor.name,
        "priority_level": priority.level.value,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "attractor_priority_score"

    return PriorityScore(
        priority=priority,
        score=score,
        normalized_score=normalize_priority_score(
            score=score,
            max_score=max_score,
        ),
        rank=priority.rank,
        metadata=merged_metadata,
    )


def compare_priority(
    left: TelosPriority,
    right: TelosPriority,
) -> int:
    """Compare two priorities."""

    if not isinstance(left, TelosPriority):
        raise TypeError("left must be TelosPriority")
    if not isinstance(right, TelosPriority):
        raise TypeError("right must be TelosPriority")
    if left.rank < right.rank:
        return -1
    if left.rank > right.rank:
        return 1
    if left.weight > right.weight:
        return -1
    if left.weight < right.weight:
        return 1
    return 0


def sort_priorities(
    priorities: tuple[TelosPriority, ...] | list[TelosPriority],
) -> tuple[TelosPriority, ...]:
    """Sort priorities by rank ascending, then weight descending."""

    values = tuple(priorities)
    for priority in values:
        if not isinstance(priority, TelosPriority):
            raise TypeError("priorities must contain TelosPriority")
    return tuple(sorted(values, key=lambda item: (item.rank, -item.weight)))


def sort_attractors_by_priority(
    attractors: tuple[AttractorSignature, ...] | list[AttractorSignature],
    *,
    priority: TelosPriority,
) -> tuple[AttractorSignature, ...]:
    """Sort attractors by deterministic priority-relevant fields."""

    values = tuple(attractors)
    for attractor in values:
        if not isinstance(attractor, AttractorSignature):
            raise TypeError("attractors must contain AttractorSignature")
    if not isinstance(priority, TelosPriority):
        raise TypeError("priority must be TelosPriority")
    return tuple(
        sorted(
            values,
            key=lambda item: (-item.strength, -item.stability_weight, item.name),
        )
    )


def highest_priority(
    priorities: tuple[TelosPriority, ...] | list[TelosPriority],
) -> TelosPriority | None:
    """Return the highest priority, if any."""

    sorted_priorities = sort_priorities(priorities)
    if not sorted_priorities:
        return None
    return sorted_priorities[0]


def priority_from_level(
    level: PriorityLevel,
    *,
    metadata: dict[str, object] | None = None,
) -> TelosPriority:
    """Create a TelosPriority from a PriorityLevel."""

    if not isinstance(level, PriorityLevel):
        raise TypeError("level must be PriorityLevel")
    return create_telos_priority(level=level, metadata=metadata)


def priority_summary(
    priorities: tuple[TelosPriority, ...] | list[TelosPriority],
) -> dict[str, object]:
    """Summarize a collection of priorities."""

    values = tuple(priorities)
    for priority in values:
        if not isinstance(priority, TelosPriority):
            raise TypeError("priorities must contain TelosPriority")

    highest = highest_priority(values)
    levels: dict[str, int] = {}
    total_weight = 0.0
    for priority in values:
        levels[priority.level.value] = levels.get(priority.level.value, 0) + 1
        total_weight += priority.weight

    return {
        "count": len(values),
        "highest_level": None if highest is None else highest.level.value,
        "highest_rank": None if highest is None else highest.rank,
        "total_weight": ensure_non_negative_number(
            total_weight,
            field_name="total_weight",
        ),
        "levels": levels,
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


__all__ = [
    "PriorityScore",
    "clamp_probability",
    "ensure_non_negative_int",
    "priority_score_value",
    "normalize_priority_score",
    "score_priority",
    "score_attractor_priority",
    "compare_priority",
    "sort_priorities",
    "sort_attractors_by_priority",
    "highest_priority",
    "priority_from_level",
    "priority_summary",
]
