"""Strict schema objects for CEBA Core Valuation Processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from ceba.core.metrics.schema import (
    ensure_constraint_probability,
    ensure_finite_number,
    ensure_non_negative_number,
    utc_now,
)


class ValuationKind(Enum):
    """Stable valuation category."""

    UTILITY = "UTILITY"
    ENTROPY = "ENTROPY"
    FREE_ENERGY = "FREE_ENERGY"
    ALIGNMENT = "ALIGNMENT"
    STABILITY = "STABILITY"
    COST = "COST"
    COMPOSITE = "COMPOSITE"
    UNKNOWN = "UNKNOWN"


class ValuationStatus(Enum):
    """Stable valuation status."""

    VALID = "VALID"
    DEGRADED = "DEGRADED"
    UNSTABLE = "UNSTABLE"
    BLOCKED = "BLOCKED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class UtilityKind(Enum):
    """Stable utility category."""

    TELOS_ALIGNMENT = "TELOS_ALIGNMENT"
    TASK_RELEVANCE = "TASK_RELEVANCE"
    CONSTRAINT_SATISFACTION = "CONSTRAINT_SATISFACTION"
    STABILITY_GAIN = "STABILITY_GAIN"
    INFORMATION_GAIN = "INFORMATION_GAIN"
    UNKNOWN = "UNKNOWN"


class EntropyKind(Enum):
    """Stable entropy category."""

    STATE_ENTROPY = "STATE_ENTROPY"
    UNCERTAINTY = "UNCERTAINTY"
    DISPERSION = "DISPERSION"
    STRUCTURAL_ENTROPY = "STRUCTURAL_ENTROPY"
    UNKNOWN = "UNKNOWN"


class FreeEnergyKind(Enum):
    """Stable free-energy category."""

    VARIATIONAL = "VARIATIONAL"
    SEMANTIC = "SEMANTIC"
    PREDICTION_ERROR = "PREDICTION_ERROR"
    COST_UTILITY = "COST_UTILITY"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ValuationSourceRef:
    """Lightweight reference to the source artifact used for valuation."""

    source_id: str
    source_type: str
    confidence: float
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source_id = ensure_non_empty_string(self.source_id, "source_id")
        source_type = ensure_non_empty_string(self.source_type, "source_type")
        confidence = ensure_constraint_probability(
            self.confidence,
            field_name="confidence",
        )

        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "source_type", source_type)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class UtilityResult:
    """Serializable-friendly utility computation result descriptor."""

    utility_id: UUID
    kind: UtilityKind
    value: float
    normalized_value: float
    confidence: float
    source_refs: tuple[ValuationSourceRef, ...]
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.utility_id, UUID):
            raise TypeError("utility_id must be UUID")
        if not isinstance(self.kind, UtilityKind):
            raise TypeError("kind must be UtilityKind")
        value = ensure_finite_number(self.value, field_name="value")
        normalized_value = ensure_constraint_probability(
            self.normalized_value,
            field_name="normalized_value",
        )
        confidence = ensure_constraint_probability(
            self.confidence,
            field_name="confidence",
        )
        source_refs = _ensure_source_refs(self.source_refs)
        created_at = ensure_timezone_aware(self.created_at, "created_at")

        object.__setattr__(self, "value", value)
        object.__setattr__(self, "normalized_value", normalized_value)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "source_refs", source_refs)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "utility_id": str(self.utility_id),
            "kind": self.kind.value,
            "value": self.value,
            "normalized_value": self.normalized_value,
            "confidence": self.confidence,
            "source_refs": tuple(
                source_ref.to_dict() for source_ref in self.source_refs
            ),
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EntropyResult:
    """Serializable-friendly entropy computation result descriptor."""

    entropy_id: UUID
    kind: EntropyKind
    value: float
    normalized_value: float
    confidence: float
    source_refs: tuple[ValuationSourceRef, ...]
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.entropy_id, UUID):
            raise TypeError("entropy_id must be UUID")
        if not isinstance(self.kind, EntropyKind):
            raise TypeError("kind must be EntropyKind")
        value = ensure_non_negative_number(self.value, field_name="value")
        normalized_value = ensure_constraint_probability(
            self.normalized_value,
            field_name="normalized_value",
        )
        confidence = ensure_constraint_probability(
            self.confidence,
            field_name="confidence",
        )
        source_refs = _ensure_source_refs(self.source_refs)
        created_at = ensure_timezone_aware(self.created_at, "created_at")

        object.__setattr__(self, "value", value)
        object.__setattr__(self, "normalized_value", normalized_value)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "source_refs", source_refs)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "entropy_id": str(self.entropy_id),
            "kind": self.kind.value,
            "value": self.value,
            "normalized_value": self.normalized_value,
            "confidence": self.confidence,
            "source_refs": tuple(
                source_ref.to_dict() for source_ref in self.source_refs
            ),
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FreeEnergyResult:
    """Serializable-friendly free-energy result descriptor."""

    free_energy_id: UUID
    kind: FreeEnergyKind
    value: float
    normalized_value: float
    utility_value: float | None
    entropy_value: float | None
    confidence: float
    source_refs: tuple[ValuationSourceRef, ...]
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.free_energy_id, UUID):
            raise TypeError("free_energy_id must be UUID")
        if not isinstance(self.kind, FreeEnergyKind):
            raise TypeError("kind must be FreeEnergyKind")
        value = ensure_finite_number(self.value, field_name="value")
        normalized_value = ensure_constraint_probability(
            self.normalized_value,
            field_name="normalized_value",
        )
        utility_value = ensure_optional_finite_number(
            self.utility_value,
            "utility_value",
        )
        entropy_value = ensure_optional_non_negative_number(
            self.entropy_value,
            "entropy_value",
        )
        confidence = ensure_constraint_probability(
            self.confidence,
            field_name="confidence",
        )
        source_refs = _ensure_source_refs(self.source_refs)
        created_at = ensure_timezone_aware(self.created_at, "created_at")

        object.__setattr__(self, "value", value)
        object.__setattr__(self, "normalized_value", normalized_value)
        object.__setattr__(self, "utility_value", utility_value)
        object.__setattr__(self, "entropy_value", entropy_value)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "source_refs", source_refs)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "free_energy_id": str(self.free_energy_id),
            "kind": self.kind.value,
            "value": self.value,
            "normalized_value": self.normalized_value,
            "utility_value": self.utility_value,
            "entropy_value": self.entropy_value,
            "confidence": self.confidence,
            "source_refs": tuple(
                source_ref.to_dict() for source_ref in self.source_refs
            ),
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ValuationScore:
    """Generic scalar valuation score descriptor."""

    score_id: UUID
    kind: ValuationKind
    status: ValuationStatus
    value: float
    normalized_value: float
    confidence: float
    source_refs: tuple[ValuationSourceRef, ...]
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.score_id, UUID):
            raise TypeError("score_id must be UUID")
        if not isinstance(self.kind, ValuationKind):
            raise TypeError("kind must be ValuationKind")
        if not isinstance(self.status, ValuationStatus):
            raise TypeError("status must be ValuationStatus")
        value = ensure_finite_number(self.value, field_name="value")
        normalized_value = ensure_constraint_probability(
            self.normalized_value,
            field_name="normalized_value",
        )
        confidence = ensure_constraint_probability(
            self.confidence,
            field_name="confidence",
        )
        source_refs = _ensure_source_refs(self.source_refs)
        created_at = ensure_timezone_aware(self.created_at, "created_at")

        object.__setattr__(self, "value", value)
        object.__setattr__(self, "normalized_value", normalized_value)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "source_refs", source_refs)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "score_id": str(self.score_id),
            "kind": self.kind.value,
            "status": self.status.value,
            "value": self.value,
            "normalized_value": self.normalized_value,
            "confidence": self.confidence,
            "source_refs": tuple(
                source_ref.to_dict() for source_ref in self.source_refs
            ),
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ValuationEvaluation:
    """Runtime descriptor of a complete valuation evaluation."""

    evaluation_id: UUID
    status: ValuationStatus
    utility: UtilityResult | None
    entropy: EntropyResult | None
    free_energy: FreeEnergyResult | None
    scores: tuple[ValuationScore, ...]
    aggregate_score: float
    confidence: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation_id, UUID):
            raise TypeError("evaluation_id must be UUID")
        if not isinstance(self.status, ValuationStatus):
            raise TypeError("status must be ValuationStatus")
        if self.utility is not None and not isinstance(self.utility, UtilityResult):
            raise TypeError("utility must be UtilityResult or None")
        if self.entropy is not None and not isinstance(self.entropy, EntropyResult):
            raise TypeError("entropy must be EntropyResult or None")
        if self.free_energy is not None and not isinstance(
            self.free_energy,
            FreeEnergyResult,
        ):
            raise TypeError("free_energy must be FreeEnergyResult or None")
        scores = tuple(self.scores)
        for score in scores:
            if not isinstance(score, ValuationScore):
                raise TypeError("scores must contain ValuationScore")
        aggregate_score = ensure_constraint_probability(
            self.aggregate_score,
            field_name="aggregate_score",
        )
        confidence = ensure_constraint_probability(
            self.confidence,
            field_name="confidence",
        )
        created_at = ensure_timezone_aware(self.created_at, "created_at")

        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "aggregate_score", aggregate_score)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "evaluation_id": str(self.evaluation_id),
            "status": self.status.value,
            "utility": None if self.utility is None else self.utility.to_dict(),
            "entropy": None if self.entropy is None else self.entropy.to_dict(),
            "free_energy": (
                None if self.free_energy is None else self.free_energy.to_dict()
            ),
            "scores": tuple(score.to_dict() for score in self.scores),
            "aggregate_score": self.aggregate_score,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


def new_utility_id() -> UUID:
    """Return a new utility identifier."""

    return uuid4()


def new_entropy_id() -> UUID:
    """Return a new entropy identifier."""

    return uuid4()


def new_free_energy_id() -> UUID:
    """Return a new free-energy identifier."""

    return uuid4()


def new_valuation_score_id() -> UUID:
    """Return a new valuation score identifier."""

    return uuid4()


def new_valuation_evaluation_id() -> UUID:
    """Return a new valuation evaluation identifier."""

    return uuid4()


def ensure_non_empty_string(value: str, field_name: str) -> str:
    """Validate a non-empty string."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    if value == "":
        raise ValueError(f"{field_name} must be non-empty")
    return value


def ensure_optional_finite_number(
    value: float | None,
    field_name: str,
) -> float | None:
    """Validate an optional finite number."""

    if value is None:
        return None
    return ensure_finite_number(value, field_name=field_name)


def ensure_optional_non_negative_number(
    value: float | None,
    field_name: str,
) -> float | None:
    """Validate an optional non-negative finite number."""

    if value is None:
        return None
    return ensure_non_negative_number(value, field_name=field_name)


def ensure_timezone_aware(value: datetime, field_name: str) -> datetime:
    """Validate a timezone-aware datetime."""

    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def valuation_status_from_score(
    *,
    normalized_score: float,
    confidence: float,
    degraded_threshold: float = 0.35,
    valid_threshold: float = 0.60,
) -> ValuationStatus:
    """Infer valuation status from normalized score and confidence."""

    normalized_score = ensure_constraint_probability(
        normalized_score,
        field_name="normalized_score",
    )
    confidence = ensure_constraint_probability(confidence, field_name="confidence")
    degraded_threshold = ensure_constraint_probability(
        degraded_threshold,
        field_name="degraded_threshold",
    )
    valid_threshold = ensure_constraint_probability(
        valid_threshold,
        field_name="valid_threshold",
    )
    if valid_threshold < degraded_threshold:
        raise ValueError("valid_threshold must be >= degraded_threshold")
    if confidence <= 0.0:
        return ValuationStatus.UNKNOWN
    if normalized_score >= valid_threshold:
        return ValuationStatus.VALID
    if normalized_score >= degraded_threshold:
        return ValuationStatus.DEGRADED
    return ValuationStatus.UNSTABLE


def create_source_ref(
    *,
    source_id: str,
    source_type: str,
    confidence: float = 1.0,
    metadata: dict[str, object] | None = None,
) -> ValuationSourceRef:
    """Create a validated valuation source reference."""

    default_metadata: dict[str, object] = {"source": "valuation_source_ref"}
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "valuation_source_ref"

    return ValuationSourceRef(
        source_id=source_id,
        source_type=source_type,
        confidence=confidence,
        metadata=merged_metadata,
    )


def create_valuation_score(
    *,
    kind: ValuationKind,
    value: float,
    normalized_value: float,
    confidence: float = 1.0,
    source_refs: tuple[ValuationSourceRef, ...] | list[ValuationSourceRef] = (),
    status: ValuationStatus | None = None,
    metadata: dict[str, object] | None = None,
) -> ValuationScore:
    """Create a generic valuation score."""

    if not isinstance(kind, ValuationKind):
        raise TypeError("kind must be ValuationKind")
    value = ensure_finite_number(value, field_name="value")
    normalized_value = ensure_constraint_probability(
        normalized_value,
        field_name="normalized_value",
    )
    confidence = ensure_constraint_probability(confidence, field_name="confidence")
    source_refs = _ensure_source_refs(source_refs)
    if status is None:
        status = valuation_status_from_score(
            normalized_score=normalized_value,
            confidence=confidence,
        )
    if not isinstance(status, ValuationStatus):
        raise TypeError("status must be ValuationStatus or None")

    default_metadata: dict[str, object] = {"source": "valuation_score"}
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "valuation_score"

    return ValuationScore(
        score_id=new_valuation_score_id(),
        kind=kind,
        status=status,
        value=value,
        normalized_value=normalized_value,
        confidence=confidence,
        source_refs=source_refs,
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def create_valuation_evaluation(
    *,
    utility: UtilityResult | None = None,
    entropy: EntropyResult | None = None,
    free_energy: FreeEnergyResult | None = None,
    scores: tuple[ValuationScore, ...] | list[ValuationScore] = (),
    status: ValuationStatus | None = None,
    metadata: dict[str, object] | None = None,
) -> ValuationEvaluation:
    """Create a complete valuation evaluation descriptor."""

    if utility is not None and not isinstance(utility, UtilityResult):
        raise TypeError("utility must be UtilityResult or None")
    if entropy is not None and not isinstance(entropy, EntropyResult):
        raise TypeError("entropy must be EntropyResult or None")
    if free_energy is not None and not isinstance(free_energy, FreeEnergyResult):
        raise TypeError("free_energy must be FreeEnergyResult or None")
    scores = tuple(scores)
    for score in scores:
        if not isinstance(score, ValuationScore):
            raise TypeError("scores must contain ValuationScore")

    aggregate_score = (
        0.0
        if not scores
        else sum(score.normalized_value for score in scores) / len(scores)
    )
    confidences: list[float] = []
    if utility is not None:
        confidences.append(utility.confidence)
    if entropy is not None:
        confidences.append(entropy.confidence)
    if free_energy is not None:
        confidences.append(free_energy.confidence)
    confidences.extend(score.confidence for score in scores)
    confidence = 1.0 if not confidences else sum(confidences) / len(confidences)
    confidence = ensure_constraint_probability(confidence, field_name="confidence")

    if status is None:
        status = valuation_status_from_score(
            normalized_score=aggregate_score,
            confidence=confidence,
        )
    if not isinstance(status, ValuationStatus):
        raise TypeError("status must be ValuationStatus or None")

    default_metadata: dict[str, object] = {
        "source": "valuation_evaluation",
        "score_count": len(scores),
        "has_utility": utility is not None,
        "has_entropy": entropy is not None,
        "has_free_energy": free_energy is not None,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "valuation_evaluation"

    return ValuationEvaluation(
        evaluation_id=new_valuation_evaluation_id(),
        status=status,
        utility=utility,
        entropy=entropy,
        free_energy=free_energy,
        scores=scores,
        aggregate_score=aggregate_score,
        confidence=confidence,
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def _ensure_source_refs(
    source_refs: tuple[ValuationSourceRef, ...] | list[ValuationSourceRef],
) -> tuple[ValuationSourceRef, ...]:
    values = tuple(source_refs)
    for source_ref in values:
        if not isinstance(source_ref, ValuationSourceRef):
            raise TypeError("source_refs must contain ValuationSourceRef")
    return values


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
    "ValuationKind",
    "ValuationStatus",
    "UtilityKind",
    "EntropyKind",
    "FreeEnergyKind",
    "ValuationSourceRef",
    "UtilityResult",
    "EntropyResult",
    "FreeEnergyResult",
    "ValuationScore",
    "ValuationEvaluation",
    "new_utility_id",
    "new_entropy_id",
    "new_free_energy_id",
    "new_valuation_score_id",
    "new_valuation_evaluation_id",
    "ensure_non_empty_string",
    "ensure_optional_finite_number",
    "ensure_optional_non_negative_number",
    "ensure_timezone_aware",
    "valuation_status_from_score",
    "create_source_ref",
    "create_valuation_score",
    "create_valuation_evaluation",
]
