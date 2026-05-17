"""Deterministic feature adapters for CEBA Core Constraint Processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib

from ceba.core.constraints.errors import (
    ConstraintEvaluationError,
    ConstraintMatchingError,
    build_constraint_error_context,
)
from ceba.core.constraints.schema import (
    ConstraintInputRef,
    ensure_metadata_dict,
    ensure_non_empty_string,
)
from ceba.difference.aggregator import DifferenceAggregateSummary
from ceba.difference.schema import (
    DifferenceCandidate,
    DifferenceClassification,
    ensure_non_negative,
    ensure_probability,
)


class ConstraintFeatureSource(str, Enum):
    """Stable source categories for constraint features."""

    DIFFERENCE_CANDIDATE = "DIFFERENCE_CANDIDATE"
    DIFFERENCE_CLASSIFICATION = "DIFFERENCE_CLASSIFICATION"
    DIFFERENCE_AGGREGATE = "DIFFERENCE_AGGREGATE"
    DIFFERENCE_BUCKET = "DIFFERENCE_BUCKET"
    UNKNOWN = "UNKNOWN"


class ConstraintFeatureValueType(str, Enum):
    """Stable feature value type categories."""

    STRING = "STRING"
    NUMBER = "NUMBER"
    BOOLEAN = "BOOLEAN"
    ENUM = "ENUM"
    TUPLE = "TUPLE"
    DICT = "DICT"
    NONE = "NONE"
    OBJECT = "OBJECT"


@dataclass(frozen=True, slots=True)
class ConstraintFeature:
    """One deterministic feature available to constraint evaluation."""

    feature_id: str
    source: ConstraintFeatureSource
    input_ref: ConstraintInputRef
    name: str
    value: object
    value_type: ConstraintFeatureValueType
    confidence: float
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ensure_non_empty_string(self.feature_id, "feature_id")
        if not isinstance(self.source, ConstraintFeatureSource):
            raise TypeError("source must be ConstraintFeatureSource")
        if not isinstance(self.input_ref, ConstraintInputRef):
            raise TypeError("input_ref must be ConstraintInputRef")
        ensure_non_empty_string(self.name, "name")
        if not isinstance(self.value_type, ConstraintFeatureValueType):
            raise TypeError("value_type must be ConstraintFeatureValueType")
        object.__setattr__(self, "confidence", ensure_probability(self.confidence, "confidence"))
        ensure_metadata_dict(self.metadata)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serialization-friendly shallow representation."""

        return {
            "feature_id": self.feature_id,
            "source": self.source.value,
            "input_ref": {
                "artifact_id": self.input_ref.artifact_id,
                "artifact_type": self.input_ref.artifact_type,
                "difference_id": self.input_ref.difference_id,
                "observation_id": self.input_ref.observation_id,
            },
            "name": self.name,
            "value": self.value,
            "value_type": self.value_type.value,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ConstraintFeatureSet:
    """Batch of constraint features derived from one artifact or package."""

    source: ConstraintFeatureSource
    input_ref: ConstraintInputRef
    features: tuple[ConstraintFeature, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source, ConstraintFeatureSource):
            raise TypeError("source must be ConstraintFeatureSource")
        if not isinstance(self.input_ref, ConstraintInputRef):
            raise TypeError("input_ref must be ConstraintInputRef")
        if isinstance(self.features, list):
            object.__setattr__(self, "features", tuple(self.features))
        if not isinstance(self.features, tuple):
            raise TypeError("features must be a tuple of ConstraintFeature")
        for feature in self.features:
            if not isinstance(feature, ConstraintFeature):
                raise TypeError("features must contain ConstraintFeature")
        ensure_metadata_dict(self.metadata)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def by_name(self, name: str) -> ConstraintFeature | None:
        """Return a feature by name, or None when missing."""

        ensure_non_empty_string(name, "name")
        for feature in self.features:
            if feature.name == name:
                return feature
        return None

    def require(self, name: str) -> ConstraintFeature:
        """Return a feature by name or raise when missing."""

        feature = self.by_name(name)
        if feature is None:
            raise ConstraintMatchingError(
                message="constraint feature not found",
                context=build_constraint_error_context(
                    operation="ConstraintFeatureSet.require",
                    field_path=(name,),
                ),
            )
        return feature

    def names(self) -> tuple[str, ...]:
        """Return feature names in feature order."""

        return tuple(feature.name for feature in self.features)

    def to_dict(self) -> dict[str, object]:
        """Return a serialization-friendly representation."""

        return {
            "source": self.source.value,
            "input_ref": {
                "artifact_id": self.input_ref.artifact_id,
                "artifact_type": self.input_ref.artifact_type,
                "difference_id": self.input_ref.difference_id,
                "observation_id": self.input_ref.observation_id,
            },
            "features": [feature.to_dict() for feature in self.features],
            "metadata": dict(self.metadata),
        }


def infer_feature_value_type(value: object) -> ConstraintFeatureValueType:
    """Infer the stable feature value type for a value."""

    if value is None:
        return ConstraintFeatureValueType.NONE
    if isinstance(value, bool):
        return ConstraintFeatureValueType.BOOLEAN
    if isinstance(value, (int, float)):
        return ConstraintFeatureValueType.NUMBER
    if isinstance(value, str):
        return ConstraintFeatureValueType.STRING
    if isinstance(value, Enum):
        return ConstraintFeatureValueType.ENUM
    if isinstance(value, (tuple, list)):
        return ConstraintFeatureValueType.TUPLE
    if isinstance(value, dict):
        return ConstraintFeatureValueType.DICT
    return ConstraintFeatureValueType.OBJECT


def build_feature_id(
    *,
    source: ConstraintFeatureSource,
    input_ref: ConstraintInputRef,
    name: str,
    value: object,
) -> str:
    """Build a deterministic constraint feature identifier."""

    if not isinstance(source, ConstraintFeatureSource):
        raise TypeError("source must be ConstraintFeatureSource")
    if not isinstance(input_ref, ConstraintInputRef):
        raise TypeError("input_ref must be ConstraintInputRef")
    ensure_non_empty_string(name, "name")

    components = (
        source.value,
        input_ref.artifact_id,
        input_ref.artifact_type,
        input_ref.difference_id or "",
        input_ref.observation_id or "",
        name,
        repr(value),
    )
    digest = hashlib.sha256()
    for component in components:
        digest.update(component.encode("utf-8"))
        digest.update(b"\x00")
    return "cf_" + digest.hexdigest()[:32]


def create_constraint_feature(
    *,
    source: ConstraintFeatureSource,
    input_ref: ConstraintInputRef,
    name: str,
    value: object,
    confidence: float = 1.0,
    metadata: dict[str, object] | None = None,
) -> ConstraintFeature:
    """Create a validated constraint feature."""

    try:
        value_type = infer_feature_value_type(value)
        return ConstraintFeature(
            feature_id=build_feature_id(
                source=source,
                input_ref=input_ref,
                name=name,
                value=value,
            ),
            source=source,
            input_ref=input_ref,
            name=name,
            value=value,
            value_type=value_type,
            confidence=confidence,
            metadata={} if metadata is None else metadata,
        )
    except ConstraintMatchingError:
        raise
    except Exception as exc:
        raise ConstraintMatchingError(
            message="failed to create constraint feature",
            context=build_constraint_error_context(
                operation="create_constraint_feature",
                field_path=(name,) if isinstance(name, str) and name else (),
            ),
            cause=exc,
        ) from exc


def input_ref_from_candidate(candidate: DifferenceCandidate) -> ConstraintInputRef:
    """Build an input reference from a DifferenceCandidate."""

    if not isinstance(candidate, DifferenceCandidate):
        raise TypeError("candidate must be DifferenceCandidate")
    observation_id = candidate.metadata.get("observation_id")
    return ConstraintInputRef(
        artifact_id=str(candidate.difference_id),
        artifact_type="DifferenceCandidate",
        difference_id=str(candidate.difference_id),
        observation_id=observation_id if _is_non_empty_string(observation_id) else None,
        metadata={
            "source_status": candidate.status.value,
            "source_kind": candidate.kind.value,
        },
    )


def input_ref_from_classification(classification: DifferenceClassification) -> ConstraintInputRef:
    """Build an input reference from a DifferenceClassification."""

    if not isinstance(classification, DifferenceClassification):
        raise TypeError("classification must be DifferenceClassification")
    observation_id = classification.metadata.get("observation_id")
    return ConstraintInputRef(
        artifact_id=str(classification.difference_id),
        artifact_type="DifferenceClassification",
        difference_id=str(classification.difference_id),
        observation_id=observation_id if _is_non_empty_string(observation_id) else None,
        metadata={
            "source_kind": classification.kind.value,
            "source_severity": classification.severity.value,
        },
    )


def input_ref_from_summary(summary: DifferenceAggregateSummary) -> ConstraintInputRef:
    """Build an input reference from a DifferenceAggregateSummary."""

    if not isinstance(summary, DifferenceAggregateSummary):
        raise TypeError("summary must be DifferenceAggregateSummary")
    return ConstraintInputRef(
        artifact_id=summary.observation_id + ":aggregate_summary",
        artifact_type="DifferenceAggregateSummary",
        difference_id=None,
        observation_id=summary.observation_id,
        metadata={
            "total_candidates": summary.total_candidates,
            "total_classifications": summary.total_classifications,
        },
    )


def features_from_candidate(candidate: DifferenceCandidate) -> ConstraintFeatureSet:
    """Adapt DifferenceCandidate into constraint-readable features."""

    try:
        if not isinstance(candidate, DifferenceCandidate):
            raise TypeError("candidate must be DifferenceCandidate")
        ensure_probability(candidate.confidence, "confidence")
        ensure_non_negative(candidate.magnitude, "magnitude")

        source = ConstraintFeatureSource.DIFFERENCE_CANDIDATE
        input_ref = input_ref_from_candidate(candidate)
        features = tuple(
            create_constraint_feature(
                source=source,
                input_ref=input_ref,
                name=name,
                value=value,
                confidence=candidate.confidence,
            )
            for name, value in (
                ("kind", candidate.kind),
                ("status", candidate.status),
                ("direction", candidate.direction),
                ("confidence", candidate.confidence),
                ("magnitude", candidate.magnitude),
                ("has_left_value", candidate.left_value is not None),
                ("has_right_value", candidate.right_value is not None),
                ("evidence_ref_count", len(candidate.evidence_refs)),
            )
        )
        return ConstraintFeatureSet(
            source=source,
            input_ref=input_ref,
            features=features,
            metadata={"feature_scope": "candidate"},
        )
    except ConstraintMatchingError:
        raise
    except Exception as exc:
        raise ConstraintMatchingError(
            message="failed to adapt candidate features",
            context=build_constraint_error_context(
                operation="features_from_candidate",
                difference_id=str(candidate.difference_id)
                if isinstance(candidate, DifferenceCandidate)
                else None,
            ),
            cause=exc,
        ) from exc


def features_from_classification(classification: DifferenceClassification) -> ConstraintFeatureSet:
    """Adapt DifferenceClassification into constraint-readable features."""

    try:
        if not isinstance(classification, DifferenceClassification):
            raise TypeError("classification must be DifferenceClassification")
        ensure_probability(classification.confidence, "confidence")

        source = ConstraintFeatureSource.DIFFERENCE_CLASSIFICATION
        input_ref = input_ref_from_classification(classification)
        features = tuple(
            create_constraint_feature(
                source=source,
                input_ref=input_ref,
                name=name,
                value=value,
                confidence=classification.confidence,
            )
            for name, value in (
                ("kind", classification.kind),
                ("severity", classification.severity),
                ("confidence", classification.confidence),
                ("has_rationale_code", classification.rationale_code is not None),
                ("rationale_code", classification.rationale_code),
            )
        )
        return ConstraintFeatureSet(
            source=source,
            input_ref=input_ref,
            features=features,
            metadata={"feature_scope": "classification"},
        )
    except ConstraintMatchingError:
        raise
    except Exception as exc:
        raise ConstraintMatchingError(
            message="failed to adapt classification features",
            context=build_constraint_error_context(
                operation="features_from_classification",
                difference_id=str(classification.difference_id)
                if isinstance(classification, DifferenceClassification)
                else None,
            ),
            cause=exc,
        ) from exc


def features_from_aggregate_summary(summary: DifferenceAggregateSummary) -> ConstraintFeatureSet:
    """Adapt DifferenceAggregateSummary into constraint-readable features."""

    try:
        if not isinstance(summary, DifferenceAggregateSummary):
            raise TypeError("summary must be DifferenceAggregateSummary")
        ensure_probability(summary.mean_confidence, "mean_confidence")
        ensure_non_negative(summary.max_magnitude, "max_magnitude")

        source = ConstraintFeatureSource.DIFFERENCE_AGGREGATE
        input_ref = input_ref_from_summary(summary)
        features = tuple(
            create_constraint_feature(
                source=source,
                input_ref=input_ref,
                name=name,
                value=value,
                confidence=summary.mean_confidence,
            )
            for name, value in (
                ("total_candidates", summary.total_candidates),
                ("total_classifications", summary.total_classifications),
                ("bucket_count", len(summary.buckets)),
                ("max_severity", summary.max_severity),
                ("mean_confidence", summary.mean_confidence),
                ("max_magnitude", summary.max_magnitude),
            )
        )
        return ConstraintFeatureSet(
            source=source,
            input_ref=input_ref,
            features=features,
            metadata={"feature_scope": "aggregate_summary"},
        )
    except ConstraintMatchingError:
        raise
    except Exception as exc:
        raise ConstraintEvaluationError(
            message="failed to adapt aggregate summary features",
            context=build_constraint_error_context(
                operation="features_from_aggregate_summary",
                observation_id=summary.observation_id
                if isinstance(summary, DifferenceAggregateSummary)
                else None,
            ),
            cause=exc,
        ) from exc


def merge_feature_sets(
    feature_sets: tuple[ConstraintFeatureSet, ...] | list[ConstraintFeatureSet],
    *,
    input_ref: ConstraintInputRef,
    metadata: dict[str, object] | None = None,
) -> ConstraintFeatureSet:
    """Merge multiple feature sets into one deterministic feature set."""

    try:
        if not isinstance(feature_sets, (tuple, list)):
            raise TypeError("feature_sets must be a tuple or list")
        if not isinstance(input_ref, ConstraintInputRef):
            raise TypeError("input_ref must be ConstraintInputRef")

        features: list[ConstraintFeature] = []
        seen_names: set[str] = set()
        for feature_set in feature_sets:
            if not isinstance(feature_set, ConstraintFeatureSet):
                raise TypeError("feature_sets must contain ConstraintFeatureSet")
            for feature in feature_set.features:
                if feature.name in seen_names:
                    raise ConstraintMatchingError(
                        message="duplicate constraint feature name",
                        context=build_constraint_error_context(
                            operation="merge_feature_sets",
                            field_path=(feature.name,),
                        ),
                    )
                seen_names.add(feature.name)
                features.append(feature)

        resolved_metadata = {
            "merged": True,
            "source_set_count": len(feature_sets),
        }
        if metadata is not None:
            ensure_metadata_dict(metadata)
            resolved_metadata.update(metadata)
        resolved_metadata["merged"] = True

        return ConstraintFeatureSet(
            source=ConstraintFeatureSource.UNKNOWN,
            input_ref=input_ref,
            features=tuple(features),
            metadata=resolved_metadata,
        )
    except ConstraintMatchingError:
        raise
    except Exception as exc:
        raise ConstraintMatchingError(
            message="failed to merge constraint feature sets",
            context=build_constraint_error_context(operation="merge_feature_sets"),
            cause=exc,
        ) from exc


def _is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value)


__all__ = [
    "ConstraintFeatureSource",
    "ConstraintFeatureValueType",
    "ConstraintFeature",
    "ConstraintFeatureSet",
    "infer_feature_value_type",
    "build_feature_id",
    "create_constraint_feature",
    "input_ref_from_candidate",
    "input_ref_from_classification",
    "input_ref_from_summary",
    "features_from_candidate",
    "features_from_classification",
    "features_from_aggregate_summary",
    "merge_feature_sets",
]
