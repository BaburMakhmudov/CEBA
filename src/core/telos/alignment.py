"""Deterministic telos alignment utilities for CEBA Core Telos Processing."""

from __future__ import annotations

from dataclasses import dataclass, field

from ceba.core.metrics.schema import (
    ensure_constraint_probability,
    ensure_finite_number,
    utc_now,
)
from ceba.core.telos.attractor import AttractorScore
from ceba.core.telos.schema import (
    AlignmentStatus,
    AttractorSignature,
    TelosAlignmentResult,
    new_alignment_id,
)
from ceba.core.telos.telos_vector import (
    TelosVectorPayload,
    cosine_alignment_score,
    cosine_similarity,
    vector_payloads_compatible,
)


@dataclass(frozen=True)
class AlignmentPolicy:
    """Stable policy for telos alignment classification."""

    aligned_threshold: float = 0.85
    partial_threshold: float = 0.50
    confidence_floor: float = 0.0
    penalize_missing_target: bool = True

    def __post_init__(self) -> None:
        aligned_threshold = ensure_constraint_probability(
            self.aligned_threshold,
            field_name="aligned_threshold",
        )
        partial_threshold = ensure_constraint_probability(
            self.partial_threshold,
            field_name="partial_threshold",
        )
        confidence_floor = ensure_constraint_probability(
            self.confidence_floor,
            field_name="confidence_floor",
        )
        if aligned_threshold < partial_threshold:
            raise ValueError(
                "aligned_threshold must be greater than or equal to partial_threshold"
            )
        if not isinstance(self.penalize_missing_target, bool):
            raise TypeError("penalize_missing_target must be bool")

        object.__setattr__(self, "aligned_threshold", aligned_threshold)
        object.__setattr__(self, "partial_threshold", partial_threshold)
        object.__setattr__(self, "confidence_floor", confidence_floor)


@dataclass(frozen=True)
class AlignmentSummary:
    """Serializable-friendly summary of multiple telos alignment results."""

    count: int
    mean_score: float
    mean_confidence: float
    best_alignment_id: str | None
    best_score: float | None
    statuses: dict[str, int]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        count = ensure_non_negative_int(self.count, "count")
        mean_score = ensure_constraint_probability(
            self.mean_score,
            field_name="mean_score",
        )
        mean_confidence = ensure_constraint_probability(
            self.mean_confidence,
            field_name="mean_confidence",
        )
        if self.best_alignment_id is not None:
            _ensure_non_empty_string(self.best_alignment_id, "best_alignment_id")
        best_score = (
            None
            if self.best_score is None
            else ensure_constraint_probability(self.best_score, field_name="best_score")
        )
        statuses = _copy_statuses(self.statuses)

        object.__setattr__(self, "count", count)
        object.__setattr__(self, "mean_score", mean_score)
        object.__setattr__(self, "mean_confidence", mean_confidence)
        object.__setattr__(self, "best_score", best_score)
        object.__setattr__(self, "statuses", statuses)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "count": self.count,
            "mean_score": self.mean_score,
            "mean_confidence": self.mean_confidence,
            "best_alignment_id": self.best_alignment_id,
            "best_score": self.best_score,
            "statuses": dict(self.statuses),
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


def alignment_status_from_score(
    *,
    score: float,
    policy: AlignmentPolicy | None = None,
) -> AlignmentStatus:
    """Classify alignment score."""

    if policy is None:
        policy = AlignmentPolicy()
    if not isinstance(policy, AlignmentPolicy):
        raise TypeError("policy must be AlignmentPolicy")
    score = ensure_constraint_probability(score, field_name="score")
    if score >= policy.aligned_threshold:
        return AlignmentStatus.ALIGNED
    if score >= policy.partial_threshold:
        return AlignmentStatus.PARTIAL
    return AlignmentStatus.MISALIGNED


def confidence_from_alignment_score(
    *,
    score: float,
    policy: AlignmentPolicy | None = None,
) -> float:
    """Compute deterministic confidence from score."""

    if policy is None:
        policy = AlignmentPolicy()
    if not isinstance(policy, AlignmentPolicy):
        raise TypeError("policy must be AlignmentPolicy")
    score = ensure_constraint_probability(score, field_name="score")
    return ensure_constraint_probability(
        max(policy.confidence_floor, score),
        field_name="confidence",
    )


def compute_payload_alignment_score(
    *,
    source: TelosVectorPayload,
    target: TelosVectorPayload,
) -> float:
    """Compute alignment score between two telos vector payloads."""

    if not isinstance(source, TelosVectorPayload):
        raise TypeError("source must be TelosVectorPayload")
    if not isinstance(target, TelosVectorPayload):
        raise TypeError("target must be TelosVectorPayload")
    if not vector_payloads_compatible(source, target):
        raise ValueError("source and target payloads must be compatible")
    return cosine_alignment_score(source.values, target.values)


def align_payloads(
    *,
    source: TelosVectorPayload,
    target: TelosVectorPayload,
    policy: AlignmentPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> TelosAlignmentResult:
    """Create TelosAlignmentResult from two vector payloads."""

    if policy is None:
        policy = AlignmentPolicy()
    if not isinstance(policy, AlignmentPolicy):
        raise TypeError("policy must be AlignmentPolicy")
    score = compute_payload_alignment_score(source=source, target=target)
    default_metadata: dict[str, object] = {
        "source": "payload_alignment",
        "source_vector_kind": source.vector_ref.kind.value,
        "target_vector_kind": target.vector_ref.kind.value,
        "cosine_similarity": cosine_similarity(source.values, target.values),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "payload_alignment"

    return TelosAlignmentResult(
        alignment_id=new_alignment_id(),
        status=alignment_status_from_score(score=score, policy=policy),
        score=score,
        confidence=confidence_from_alignment_score(score=score, policy=policy),
        source_vector_id=source.vector_ref.vector_id,
        target_vector_id=target.vector_ref.vector_id,
        attractor_id=None,
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def align_payload_to_attractor(
    *,
    source: TelosVectorPayload,
    attractor: AttractorSignature,
    target_payload: TelosVectorPayload | None = None,
    policy: AlignmentPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> TelosAlignmentResult:
    """Create TelosAlignmentResult for a source payload against an attractor."""

    if not isinstance(source, TelosVectorPayload):
        raise TypeError("source must be TelosVectorPayload")
    if not isinstance(attractor, AttractorSignature):
        raise TypeError("attractor must be AttractorSignature")
    if target_payload is not None and not isinstance(
        target_payload,
        TelosVectorPayload,
    ):
        raise TypeError("target_payload must be TelosVectorPayload or None")
    if policy is None:
        policy = AlignmentPolicy()
    if not isinstance(policy, AlignmentPolicy):
        raise TypeError("policy must be AlignmentPolicy")

    target_vector_id: str | None
    if target_payload is not None:
        if not vector_payloads_compatible(source, target_payload):
            raise ValueError("source and target_payload must be compatible")
        score = compute_payload_alignment_score(source=source, target=target_payload)
        target_vector_id = target_payload.vector_ref.vector_id
    elif (
        attractor.vector is not None
        and attractor.vector.vector_id == source.vector_ref.vector_id
    ):
        score = 1.0
        target_vector_id = attractor.vector.vector_id
    elif policy.penalize_missing_target:
        score = 0.0
        target_vector_id = (
            None if attractor.vector is None else attractor.vector.vector_id
        )
    else:
        score = attractor.strength
        target_vector_id = (
            None if attractor.vector is None else attractor.vector.vector_id
        )

    score = clamp_probability(score * attractor.strength)
    default_metadata: dict[str, object] = {
        "source": "payload_attractor_alignment",
        "attractor_id": str(attractor.attractor_id),
        "attractor_name": attractor.name,
        "has_target_payload": target_payload is not None,
        "attractor_strength": attractor.strength,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "payload_attractor_alignment"

    return TelosAlignmentResult(
        alignment_id=new_alignment_id(),
        status=alignment_status_from_score(score=score, policy=policy),
        score=score,
        confidence=confidence_from_alignment_score(score=score, policy=policy),
        source_vector_id=source.vector_ref.vector_id,
        target_vector_id=target_vector_id,
        attractor_id=attractor.attractor_id,
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def alignment_from_attractor_score(
    *,
    score: AttractorScore,
    policy: AlignmentPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> TelosAlignmentResult:
    """Convert AttractorScore into TelosAlignmentResult."""

    if not isinstance(score, AttractorScore):
        raise TypeError("score must be AttractorScore")
    if policy is None:
        policy = AlignmentPolicy()
    if not isinstance(policy, AlignmentPolicy):
        raise TypeError("policy must be AlignmentPolicy")

    alignment_score = (
        score.alignment_score
        if score.alignment_score is not None
        else clamp_probability(score.final_score)
    )
    default_metadata: dict[str, object] = {
        "source": "attractor_score_alignment",
        "attractor_id": str(score.attractor.attractor_id),
        "attractor_name": score.attractor.name,
        "final_score": score.final_score,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "attractor_score_alignment"

    return TelosAlignmentResult(
        alignment_id=new_alignment_id(),
        status=alignment_status_from_score(score=alignment_score, policy=policy),
        score=alignment_score,
        confidence=confidence_from_alignment_score(
            score=alignment_score,
            policy=policy,
        ),
        source_vector_id=None,
        target_vector_id=(
            None
            if score.attractor.vector is None
            else score.attractor.vector.vector_id
        ),
        attractor_id=score.attractor.attractor_id,
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def best_alignment(
    results: tuple[TelosAlignmentResult, ...] | list[TelosAlignmentResult],
) -> TelosAlignmentResult | None:
    """Return the best alignment result, if any."""

    values = _ensure_alignment_results(results)
    if not values:
        return None
    return sorted(
        values,
        key=lambda item: (
            -item.score,
            -item.confidence,
            item.created_at,
            str(item.alignment_id),
        ),
    )[0]


def alignment_is_acceptable(
    result: TelosAlignmentResult,
    *,
    policy: AlignmentPolicy | None = None,
) -> bool:
    """Return True when an alignment is acceptable."""

    if not isinstance(result, TelosAlignmentResult):
        raise TypeError("result must be TelosAlignmentResult")
    if policy is None:
        policy = AlignmentPolicy()
    if not isinstance(policy, AlignmentPolicy):
        raise TypeError("policy must be AlignmentPolicy")
    return (
        result.score >= policy.partial_threshold
        and result.status is not AlignmentStatus.INVALID
    )


def summarize_alignments(
    results: tuple[TelosAlignmentResult, ...] | list[TelosAlignmentResult],
    *,
    metadata: dict[str, object] | None = None,
) -> AlignmentSummary:
    """Summarize telos alignment results."""

    values = _ensure_alignment_results(results)
    count = len(values)
    mean_score = 0.0 if count == 0 else sum(item.score for item in values) / count
    mean_confidence = (
        0.0 if count == 0 else sum(item.confidence for item in values) / count
    )
    best = best_alignment(values)
    statuses: dict[str, int] = {}
    for result in values:
        statuses[result.status.value] = statuses.get(result.status.value, 0) + 1

    default_metadata: dict[str, object] = {"source": "alignment_summary"}
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "alignment_summary"

    return AlignmentSummary(
        count=count,
        mean_score=mean_score,
        mean_confidence=mean_confidence,
        best_alignment_id=None if best is None else str(best.alignment_id),
        best_score=None if best is None else best.score,
        statuses=statuses,
        metadata=merged_metadata,
    )


def alignment_summary_to_dict(summary: AlignmentSummary) -> dict[str, object]:
    """Return AlignmentSummary as a dictionary."""

    if not isinstance(summary, AlignmentSummary):
        raise TypeError("summary must be AlignmentSummary")
    return summary.to_dict()


def _ensure_alignment_results(
    results: tuple[TelosAlignmentResult, ...] | list[TelosAlignmentResult],
) -> tuple[TelosAlignmentResult, ...]:
    values = tuple(results)
    for result in values:
        if not isinstance(result, TelosAlignmentResult):
            raise TypeError("results must contain TelosAlignmentResult")
    return values


def _ensure_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _copy_statuses(statuses: dict[str, int]) -> dict[str, int]:
    if not isinstance(statuses, dict):
        raise TypeError("statuses must be dict[str, int]")
    copied = dict(statuses)
    for key, value in copied.items():
        if not isinstance(key, str):
            raise TypeError("statuses keys must be strings")
        ensure_non_negative_int(value, "statuses value")
    return copied


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
    "AlignmentPolicy",
    "AlignmentSummary",
    "clamp_probability",
    "ensure_non_negative_int",
    "alignment_status_from_score",
    "confidence_from_alignment_score",
    "compute_payload_alignment_score",
    "align_payloads",
    "align_payload_to_attractor",
    "alignment_from_attractor_score",
    "best_alignment",
    "alignment_is_acceptable",
    "summarize_alignments",
    "alignment_summary_to_dict",
]
