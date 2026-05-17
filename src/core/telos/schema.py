"""Strict schema objects for CEBA Core Telos Processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from ceba.core.metrics.schema import (
    ensure_constraint_probability,
    ensure_non_negative_number,
    ensure_positive_int,
    utc_now,
)


class TelosKind(Enum):
    """Stable telos category."""

    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"
    CONTEXTUAL = "CONTEXTUAL"
    SAFETY = "SAFETY"
    STABILITY = "STABILITY"
    EXPLORATORY = "EXPLORATORY"
    UNKNOWN = "UNKNOWN"


class TelosStatus(Enum):
    """Stable telos state status."""

    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    DEGRADED = "DEGRADED"
    FROZEN = "FROZEN"
    DEPRECATED = "DEPRECATED"
    INVALID = "INVALID"


class TelosVectorKind(Enum):
    """Stable telos vector category."""

    TARGET = "TARGET"
    GRADIENT = "GRADIENT"
    ATTRACTOR = "ATTRACTOR"
    PRIORITY = "PRIORITY"
    CONTEXT = "CONTEXT"
    UNKNOWN = "UNKNOWN"


class AlignmentStatus(Enum):
    """Stable telos alignment status."""

    ALIGNED = "ALIGNED"
    PARTIAL = "PARTIAL"
    MISALIGNED = "MISALIGNED"
    UNKNOWN = "UNKNOWN"
    INVALID = "INVALID"


class PriorityLevel(Enum):
    """Stable telos priority level."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    TRACE = "TRACE"


@dataclass(frozen=True)
class TelosVectorRef:
    """Lightweight reference to a telos vector without storing heavy arrays."""

    vector_id: str
    kind: TelosVectorKind
    dimension: int
    normalized: bool
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        vector_id = ensure_non_empty_string(self.vector_id, "vector_id")
        if not isinstance(self.kind, TelosVectorKind):
            raise TypeError("kind must be TelosVectorKind")
        dimension = ensure_positive_int(self.dimension, field_name="dimension")
        if not isinstance(self.normalized, bool):
            raise TypeError("normalized must be bool")

        object.__setattr__(self, "vector_id", vector_id)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "vector_id": self.vector_id,
            "kind": self.kind.value,
            "dimension": self.dimension,
            "normalized": self.normalized,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TelosPriority:
    """Stable priority descriptor for telos elements."""

    level: PriorityLevel
    weight: float
    rank: int
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.level, PriorityLevel):
            raise TypeError("level must be PriorityLevel")
        weight = ensure_non_negative_number(self.weight, field_name="weight")
        rank = ensure_non_negative_int(self.rank, "rank")

        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "level": self.level.value,
            "weight": self.weight,
            "rank": self.rank,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AttractorSignature:
    """Lightweight telos attractor descriptor."""

    attractor_id: UUID
    name: str
    kind: TelosKind
    vector: TelosVectorRef | None
    strength: float
    basin_radius: float | None
    stability_weight: float
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.attractor_id, UUID):
            raise TypeError("attractor_id must be UUID")
        name = ensure_non_empty_string(self.name, "name")
        if not isinstance(self.kind, TelosKind):
            raise TypeError("kind must be TelosKind")
        if self.vector is not None and not isinstance(self.vector, TelosVectorRef):
            raise TypeError("vector must be TelosVectorRef or None")
        strength = ensure_constraint_probability(self.strength, field_name="strength")
        basin_radius = (
            None
            if self.basin_radius is None
            else ensure_non_negative_number(
                self.basin_radius,
                field_name="basin_radius",
            )
        )
        stability_weight = ensure_non_negative_number(
            self.stability_weight,
            field_name="stability_weight",
        )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "strength", strength)
        object.__setattr__(self, "basin_radius", basin_radius)
        object.__setattr__(self, "stability_weight", stability_weight)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "attractor_id": str(self.attractor_id),
            "name": self.name,
            "kind": self.kind.value,
            "vector": None if self.vector is None else self.vector.to_dict(),
            "strength": self.strength,
            "basin_radius": self.basin_radius,
            "stability_weight": self.stability_weight,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TelosAlignmentResult:
    """Result descriptor for telos alignment computation."""

    alignment_id: UUID
    status: AlignmentStatus
    score: float
    confidence: float
    source_vector_id: str | None
    target_vector_id: str | None
    attractor_id: UUID | None
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.alignment_id, UUID):
            raise TypeError("alignment_id must be UUID")
        if not isinstance(self.status, AlignmentStatus):
            raise TypeError("status must be AlignmentStatus")
        score = ensure_constraint_probability(self.score, field_name="score")
        confidence = ensure_constraint_probability(
            self.confidence,
            field_name="confidence",
        )
        source_vector_id = ensure_optional_non_empty_string(
            self.source_vector_id,
            "source_vector_id",
        )
        target_vector_id = ensure_optional_non_empty_string(
            self.target_vector_id,
            "target_vector_id",
        )
        if self.attractor_id is not None and not isinstance(self.attractor_id, UUID):
            raise TypeError("attractor_id must be UUID or None")
        created_at = ensure_timezone_aware(self.created_at, "created_at")

        object.__setattr__(self, "score", score)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "source_vector_id", source_vector_id)
        object.__setattr__(self, "target_vector_id", target_vector_id)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "alignment_id": str(self.alignment_id),
            "status": self.status.value,
            "score": self.score,
            "confidence": self.confidence,
            "source_vector_id": self.source_vector_id,
            "target_vector_id": self.target_vector_id,
            "attractor_id": (
                None if self.attractor_id is None else str(self.attractor_id)
            ),
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TelosState:
    """Runtime descriptor of current telos structure."""

    state_id: UUID
    status: TelosStatus
    primary_attractor: AttractorSignature | None
    active_attractors: tuple[AttractorSignature, ...]
    priority: TelosPriority
    alignment_score: float
    update_count: int
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.state_id, UUID):
            raise TypeError("state_id must be UUID")
        if not isinstance(self.status, TelosStatus):
            raise TypeError("status must be TelosStatus")
        if (
            self.primary_attractor is not None
            and not isinstance(self.primary_attractor, AttractorSignature)
        ):
            raise TypeError("primary_attractor must be AttractorSignature or None")
        active_attractors = tuple(self.active_attractors)
        for attractor in active_attractors:
            if not isinstance(attractor, AttractorSignature):
                raise TypeError("active_attractors must contain AttractorSignature")
        if not isinstance(self.priority, TelosPriority):
            raise TypeError("priority must be TelosPriority")
        alignment_score = ensure_constraint_probability(
            self.alignment_score,
            field_name="alignment_score",
        )
        update_count = ensure_non_negative_int(self.update_count, "update_count")
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        updated_at = ensure_timezone_aware(self.updated_at, "updated_at")
        if updated_at < created_at:
            raise ValueError("updated_at must be greater than or equal to created_at")

        object.__setattr__(self, "active_attractors", active_attractors)
        object.__setattr__(self, "alignment_score", alignment_score)
        object.__setattr__(self, "update_count", update_count)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "state_id": str(self.state_id),
            "status": self.status.value,
            "primary_attractor": (
                None
                if self.primary_attractor is None
                else self.primary_attractor.to_dict()
            ),
            "active_attractors": tuple(
                attractor.to_dict() for attractor in self.active_attractors
            ),
            "priority": self.priority.to_dict(),
            "alignment_score": self.alignment_score,
            "update_count": self.update_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }


def new_attractor_id() -> UUID:
    """Return a new attractor UUID."""

    return uuid4()


def new_alignment_id() -> UUID:
    """Return a new alignment UUID."""

    return uuid4()


def new_telos_state_id() -> UUID:
    """Return a new telos state UUID."""

    return uuid4()


def ensure_non_empty_string(value: str, field_name: str) -> str:
    """Validate a non-empty string."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def ensure_optional_non_empty_string(
    value: str | None,
    field_name: str,
) -> str | None:
    """Validate an optional non-empty string."""

    if value is None:
        return None
    return ensure_non_empty_string(value, field_name)


def ensure_non_negative_int(value: int, field_name: str) -> int:
    """Validate a non-negative integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def ensure_timezone_aware(value: datetime, field_name: str) -> datetime:
    """Validate a timezone-aware datetime."""

    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def priority_weight_for_level(level: PriorityLevel) -> float:
    """Return the default weight for a priority level."""

    if not isinstance(level, PriorityLevel):
        raise TypeError("level must be PriorityLevel")
    return {
        PriorityLevel.CRITICAL: 1.0,
        PriorityLevel.HIGH: 0.8,
        PriorityLevel.MEDIUM: 0.5,
        PriorityLevel.LOW: 0.25,
        PriorityLevel.TRACE: 0.1,
    }[level]


def priority_rank_for_level(level: PriorityLevel) -> int:
    """Return the default rank for a priority level."""

    if not isinstance(level, PriorityLevel):
        raise TypeError("level must be PriorityLevel")
    return {
        PriorityLevel.CRITICAL: 0,
        PriorityLevel.HIGH: 1,
        PriorityLevel.MEDIUM: 2,
        PriorityLevel.LOW: 3,
        PriorityLevel.TRACE: 4,
    }[level]


def create_telos_priority(
    *,
    level: PriorityLevel = PriorityLevel.MEDIUM,
    weight: float | None = None,
    rank: int | None = None,
    metadata: dict[str, object] | None = None,
) -> TelosPriority:
    """Create a validated telos priority descriptor."""

    if not isinstance(level, PriorityLevel):
        raise TypeError("level must be PriorityLevel")
    if weight is None:
        weight = priority_weight_for_level(level)
    if rank is None:
        rank = priority_rank_for_level(level)

    default_metadata: dict[str, object] = {"source": "telos_priority"}
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "telos_priority"

    return TelosPriority(
        level=level,
        weight=weight,
        rank=rank,
        metadata=merged_metadata,
    )


def create_telos_state(
    *,
    primary_attractor: AttractorSignature | None = None,
    active_attractors: tuple[AttractorSignature, ...]
    | list[AttractorSignature] = (),
    priority: TelosPriority | None = None,
    status: TelosStatus = TelosStatus.ACTIVE,
    alignment_score: float = 1.0,
    metadata: dict[str, object] | None = None,
) -> TelosState:
    """Create a validated telos state descriptor."""

    if priority is None:
        priority = create_telos_priority()
    timestamp = utc_now()
    default_metadata: dict[str, object] = {"state_kind": "telos_state"}
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["state_kind"] = "telos_state"

    return TelosState(
        state_id=new_telos_state_id(),
        status=status,
        primary_attractor=primary_attractor,
        active_attractors=tuple(active_attractors),
        priority=priority,
        alignment_score=alignment_score,
        update_count=0,
        created_at=timestamp,
        updated_at=timestamp,
        metadata=merged_metadata,
    )


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
    "TelosKind",
    "TelosStatus",
    "TelosVectorKind",
    "AlignmentStatus",
    "PriorityLevel",
    "TelosVectorRef",
    "TelosPriority",
    "AttractorSignature",
    "TelosAlignmentResult",
    "TelosState",
    "new_attractor_id",
    "new_alignment_id",
    "new_telos_state_id",
    "ensure_non_empty_string",
    "ensure_optional_non_empty_string",
    "ensure_non_negative_int",
    "ensure_timezone_aware",
    "priority_weight_for_level",
    "priority_rank_for_level",
    "create_telos_priority",
    "create_telos_state",
]
