"""Deterministic aggregation utilities for CEBA difference processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from ois.difference.errors import (
    DifferenceAggregationError,
    EmptyDifferenceSetError,
    build_error_context,
)
from ois.difference.schema import (
    DifferenceCandidate,
    DifferenceClassification,
    DifferenceKind,
    DifferenceSet,
    DifferenceSeverity,
    ensure_non_negative,
    ensure_probability,
)


class AggregationMode(str, Enum):
    """Stable aggregation modes for difference summaries."""

    BY_KIND = "BY_KIND"
    BY_SEVERITY = "BY_SEVERITY"
    BY_KIND_AND_SEVERITY = "BY_KIND_AND_SEVERITY"
    SUMMARY_ONLY = "SUMMARY_ONLY"


@dataclass(frozen=True, slots=True)
class DifferenceAggregateBucket:
    """One deterministic aggregation bucket."""

    key: str
    candidate_ids: tuple[str, ...]
    kind: DifferenceKind | None
    severity: DifferenceSeverity | None
    count: int
    mean_confidence: float
    max_magnitude: float | None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_non_empty_string(self.key, "key")
        if not isinstance(self.candidate_ids, tuple):
            if isinstance(self.candidate_ids, list):
                object.__setattr__(self, "candidate_ids", tuple(self.candidate_ids))
            else:
                raise TypeError("candidate_ids must be a tuple of strings")
        for candidate_id in self.candidate_ids:
            _ensure_non_empty_string(candidate_id, "candidate_ids")
        if self.kind is not None and not isinstance(self.kind, DifferenceKind):
            raise TypeError("kind must be DifferenceKind or None")
        if self.severity is not None and not isinstance(self.severity, DifferenceSeverity):
            raise TypeError("severity must be DifferenceSeverity or None")
        _ensure_non_negative_int(self.count, "count")
        object.__setattr__(
            self,
            "mean_confidence",
            ensure_probability(self.mean_confidence, "mean_confidence"),
        )
        object.__setattr__(
            self,
            "max_magnitude",
            ensure_non_negative(self.max_magnitude, "max_magnitude"),
        )
        _ensure_metadata_dict(self.metadata)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class DifferenceAggregateSummary:
    """Batch-level deterministic summary for one DifferenceSet."""

    observation_id: str
    total_candidates: int
    total_classifications: int
    buckets: tuple[DifferenceAggregateBucket, ...]
    max_severity: DifferenceSeverity | None
    mean_confidence: float
    max_magnitude: float | None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_non_empty_string(self.observation_id, "observation_id")
        _ensure_non_negative_int(self.total_candidates, "total_candidates")
        _ensure_non_negative_int(self.total_classifications, "total_classifications")
        if not isinstance(self.buckets, tuple):
            if isinstance(self.buckets, list):
                object.__setattr__(self, "buckets", tuple(self.buckets))
            else:
                raise TypeError("buckets must be a tuple of DifferenceAggregateBucket")
        for bucket in self.buckets:
            if not isinstance(bucket, DifferenceAggregateBucket):
                raise TypeError("buckets must contain DifferenceAggregateBucket")
        if self.max_severity is not None and not isinstance(
            self.max_severity,
            DifferenceSeverity,
        ):
            raise TypeError("max_severity must be DifferenceSeverity or None")
        object.__setattr__(
            self,
            "mean_confidence",
            ensure_probability(self.mean_confidence, "mean_confidence"),
        )
        object.__setattr__(
            self,
            "max_magnitude",
            ensure_non_negative(self.max_magnitude, "max_magnitude"),
        )
        _ensure_metadata_dict(self.metadata)
        object.__setattr__(self, "metadata", dict(self.metadata))


def severity_rank(severity: DifferenceSeverity) -> int:
    """Return the stable rank for a DifferenceSeverity."""

    if not isinstance(severity, DifferenceSeverity):
        raise TypeError("severity must be DifferenceSeverity")
    return {
        DifferenceSeverity.TRACE: 0,
        DifferenceSeverity.LOW: 1,
        DifferenceSeverity.MEDIUM: 2,
        DifferenceSeverity.HIGH: 3,
        DifferenceSeverity.CRITICAL: 4,
    }[severity]


def max_severity_of(
    classifications: tuple[DifferenceClassification, ...] | list[DifferenceClassification],
) -> DifferenceSeverity | None:
    """Return the highest severity in classifications."""

    if not isinstance(classifications, (tuple, list)):
        raise TypeError("classifications must be a tuple or list of DifferenceClassification")
    if not classifications:
        return None
    severities: list[DifferenceSeverity] = []
    for classification in classifications:
        if not isinstance(classification, DifferenceClassification):
            raise TypeError("classifications must contain DifferenceClassification")
        severities.append(classification.severity)
    return max(severities, key=severity_rank)


def classification_map_by_id(
    classifications: tuple[DifferenceClassification, ...] | list[DifferenceClassification],
) -> dict[str, DifferenceClassification]:
    """Build a deterministic lookup by difference_id string."""

    try:
        if not isinstance(classifications, (tuple, list)):
            raise TypeError("classifications must be a tuple or list")
        by_id: dict[str, DifferenceClassification] = {}
        for classification in classifications:
            if not isinstance(classification, DifferenceClassification):
                raise TypeError("classifications must contain DifferenceClassification")
            difference_id = str(classification.difference_id)
            if difference_id in by_id:
                raise ValueError("duplicate classification difference_id")
            by_id[difference_id] = classification
        return by_id
    except DifferenceAggregationError:
        raise
    except Exception as exc:
        raise DifferenceAggregationError(
            message="failed to build classification map",
            context=build_error_context(operation="classification_map_by_id"),
            cause=exc,
        ) from exc


def bucket_key_for(
    candidate: DifferenceCandidate,
    classification: DifferenceClassification | None,
    mode: AggregationMode,
) -> str:
    """Build a stable aggregation bucket key."""

    if not isinstance(candidate, DifferenceCandidate):
        raise TypeError("candidate must be DifferenceCandidate")
    if classification is not None and not isinstance(classification, DifferenceClassification):
        raise TypeError("classification must be DifferenceClassification or None")
    if not isinstance(mode, AggregationMode):
        raise TypeError("mode must be AggregationMode")

    if mode is AggregationMode.BY_KIND:
        return f"kind:{candidate.kind.value}"
    if mode is AggregationMode.BY_SEVERITY:
        severity = classification.severity.value if classification is not None else "UNCLASSIFIED"
        return f"severity:{severity}"
    if mode is AggregationMode.BY_KIND_AND_SEVERITY:
        severity = classification.severity.value if classification is not None else "UNCLASSIFIED"
        return f"kind:{candidate.kind.value}|severity:{severity}"
    return "summary:all"


def build_bucket(
    *,
    key: str,
    candidates: tuple[DifferenceCandidate, ...],
    classifications_by_id: dict[str, DifferenceClassification],
) -> DifferenceAggregateBucket:
    """Build one aggregate bucket."""

    _ensure_non_empty_string(key, "key")
    if not isinstance(candidates, tuple):
        raise TypeError("candidates must be a tuple")
    if not isinstance(classifications_by_id, dict):
        raise TypeError("classifications_by_id must be a dict")

    for classification_key, classification in classifications_by_id.items():
        _ensure_non_empty_string(classification_key, "classifications_by_id key")
        if not isinstance(classification, DifferenceClassification):
            raise TypeError("classifications_by_id values must be DifferenceClassification")

    candidate_ids: list[str] = []
    kinds: set[DifferenceKind] = set()
    severities: set[DifferenceSeverity] = set()
    confidences: list[float] = []
    magnitudes: list[float] = []
    classified_count = 0

    for candidate in candidates:
        if not isinstance(candidate, DifferenceCandidate):
            raise TypeError("candidates must contain DifferenceCandidate")

        candidate_id = str(candidate.difference_id)
        candidate_ids.append(candidate_id)
        kinds.add(candidate.kind)
        confidences.append(ensure_probability(candidate.confidence, "confidence"))

        magnitude = ensure_non_negative(candidate.magnitude, "magnitude")
        if magnitude is not None:
            magnitudes.append(magnitude)

        classification = classifications_by_id.get(candidate_id)
        if classification is not None:
            classified_count += 1
            severities.add(classification.severity)

    count = len(candidates)
    mean_confidence = sum(confidences) / count if count else 0.0
    max_magnitude = max(magnitudes) if magnitudes else None

    return DifferenceAggregateBucket(
        key=key,
        candidate_ids=tuple(sorted(candidate_ids)),
        kind=next(iter(kinds)) if len(kinds) == 1 else None,
        severity=next(iter(severities)) if len(severities) == 1 else None,
        count=count,
        mean_confidence=mean_confidence,
        max_magnitude=max_magnitude,
        metadata={
            "classified_count": classified_count,
            "unclassified_count": count - classified_count,
        },
    )


def aggregate_difference_set(
    difference_set: DifferenceSet,
    classifications: tuple[DifferenceClassification, ...] | list[DifferenceClassification],
    *,
    mode: AggregationMode = AggregationMode.BY_KIND_AND_SEVERITY,
) -> DifferenceAggregateSummary:
    """Aggregate one DifferenceSet and its classifications."""

    observation_id: str | None = None
    try:
        if isinstance(difference_set, DifferenceSet):
            observation_id = difference_set.observation_id
        else:
            raise TypeError("difference_set must be DifferenceSet")
        if not isinstance(classifications, (tuple, list)):
            raise TypeError("classifications must be a tuple or list")
        if not isinstance(mode, AggregationMode):
            raise TypeError("mode must be AggregationMode")
        if not difference_set.candidates:
            raise EmptyDifferenceSetError(message="difference_set candidates must not be empty")

        classifications_tuple = tuple(classifications)
        classifications_by_id = classification_map_by_id(classifications_tuple)
        candidate_ids = {str(candidate.difference_id) for candidate in difference_set.candidates}
        unknown_ids = set(classifications_by_id) - candidate_ids
        if unknown_ids:
            raise ValueError("classifications contain difference_id not present in difference_set")

        grouped: dict[str, list[DifferenceCandidate]] = {}
        for candidate in difference_set.candidates:
            if not isinstance(candidate, DifferenceCandidate):
                raise TypeError("difference_set candidates must contain DifferenceCandidate")
            candidate_id = str(candidate.difference_id)
            key = bucket_key_for(
                candidate,
                classifications_by_id.get(candidate_id),
                mode,
            )
            grouped.setdefault(key, []).append(candidate)

        buckets = tuple(
            build_bucket(
                key=key,
                candidates=tuple(grouped[key]),
                classifications_by_id=classifications_by_id,
            )
            for key in sorted(grouped)
        )

        confidences = [
            ensure_probability(candidate.confidence, "confidence")
            for candidate in difference_set.candidates
        ]
        magnitudes = [
            magnitude
            for candidate in difference_set.candidates
            if (magnitude := ensure_non_negative(candidate.magnitude, "magnitude")) is not None
        ]

        return DifferenceAggregateSummary(
            observation_id=difference_set.observation_id,
            total_candidates=len(difference_set.candidates),
            total_classifications=len(classifications_tuple),
            buckets=buckets,
            max_severity=max_severity_of(classifications_tuple),
            mean_confidence=sum(confidences) / len(confidences),
            max_magnitude=max(magnitudes) if magnitudes else None,
            metadata={
                "aggregation_mode": mode.value,
                "bucket_count": len(buckets),
            },
        )
    except EmptyDifferenceSetError:
        raise
    except DifferenceAggregationError:
        raise
    except Exception as exc:
        raise DifferenceAggregationError(
            message="failed to aggregate difference set",
            context=build_error_context(
                operation="aggregate_difference_set",
                observation_id=observation_id,
            ),
            cause=exc,
        ) from exc


def aggregate_candidates(
    candidates: tuple[DifferenceCandidate, ...] | list[DifferenceCandidate],
    classifications: tuple[DifferenceClassification, ...] | list[DifferenceClassification],
    *,
    observation_id: str,
    mode: AggregationMode = AggregationMode.BY_KIND_AND_SEVERITY,
) -> DifferenceAggregateSummary:
    """Aggregate candidates without requiring a caller-built DifferenceSet."""

    try:
        if not isinstance(candidates, (tuple, list)):
            raise TypeError("candidates must be a tuple or list")
        _ensure_non_empty_string(observation_id, "observation_id")
        difference_set = DifferenceSet(
            set_id=uuid4(),
            observation_id=observation_id,
            candidates=tuple(candidates),
            created_at=datetime.now(timezone.utc),
            metadata={},
        )
        return aggregate_difference_set(
            difference_set,
            classifications,
            mode=mode,
        )
    except EmptyDifferenceSetError:
        raise
    except DifferenceAggregationError:
        raise
    except Exception as exc:
        raise DifferenceAggregationError(
            message="failed to aggregate candidates",
            context=build_error_context(
                operation="aggregate_candidates",
                observation_id=observation_id if isinstance(observation_id, str) else None,
            ),
            cause=exc,
        ) from exc


def _ensure_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(field_name, str) or not field_name:
        raise ValueError("field_name must be a non-empty string")
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _ensure_non_negative_int(value: int, field_name: str) -> int:
    if not isinstance(field_name, str) or not field_name:
        raise ValueError("field_name must be a non-empty string")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{field_name} must be greater than or equal to 0")
    return value


def _ensure_metadata_dict(value: dict[str, object]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("metadata must be a dict")
    if not all(isinstance(key, str) for key in value):
        raise TypeError("metadata keys must be strings")
    return value


__all__ = [
    "AggregationMode",
    "DifferenceAggregateBucket",
    "DifferenceAggregateSummary",
    "severity_rank",
    "max_severity_of",
    "classification_map_by_id",
    "bucket_key_for",
    "build_bucket",
    "aggregate_difference_set",
    "aggregate_candidates",
]
