"""Deterministic entropy computation utilities for CEBA Core Valuation."""

from __future__ import annotations

from dataclasses import dataclass
from math import log

from ceba.core.metrics.schema import (
    ensure_constraint_probability,
    ensure_finite_number,
    ensure_non_negative_number,
    utc_now,
)
from ceba.core.telos.alignment import AlignmentSummary
from ceba.core.valuation.schema import (
    EntropyKind,
    EntropyResult,
    ValuationSourceRef,
    create_source_ref,
    new_entropy_id,
)


@dataclass(frozen=True)
class EntropyPolicy:
    """Stable policy for entropy computation."""

    max_entropy: float = 1.0
    log_base: float = 2.0
    confidence_floor: float = 0.0
    normalize: bool = True

    def __post_init__(self) -> None:
        max_entropy = ensure_finite_number(
            self.max_entropy,
            field_name="max_entropy",
        )
        if max_entropy <= 0.0:
            raise ValueError("max_entropy must be positive")
        log_base = _ensure_log_base(self.log_base)
        confidence_floor = ensure_constraint_probability(
            self.confidence_floor,
            field_name="confidence_floor",
        )
        if not isinstance(self.normalize, bool):
            raise TypeError("normalize must be bool")

        object.__setattr__(self, "max_entropy", max_entropy)
        object.__setattr__(self, "log_base", log_base)
        object.__setattr__(self, "confidence_floor", confidence_floor)


def clamp_probability(value: float) -> float:
    """Clamp a finite number to the probability range."""

    value = ensure_finite_number(value, field_name="value")
    return min(max(value, 0.0), 1.0)


def normalize_entropy_value(
    *,
    value: float,
    max_entropy: float = 1.0,
) -> float:
    """Normalize an entropy value."""

    value = ensure_non_negative_number(value, field_name="value")
    max_entropy = ensure_finite_number(max_entropy, field_name="max_entropy")
    if max_entropy <= 0.0:
        raise ValueError("max_entropy must be positive")
    return clamp_probability(value / max_entropy)


def ensure_probability_values(
    values: tuple[float, ...] | list[float],
    *,
    field_name: str = "values",
) -> tuple[float, ...]:
    """Validate a non-empty tuple of probability values."""

    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple[float, ...] or list[float]")
    result = tuple(
        ensure_constraint_probability(value, field_name=f"{field_name}[{index}]")
        for index, value in enumerate(values)
    )
    if not result:
        raise ValueError(f"{field_name} must be non-empty")
    return result


def normalize_distribution(
    values: tuple[float, ...] | list[float],
) -> tuple[float, ...]:
    """Normalize non-negative weights into a probability distribution."""

    if not isinstance(values, (tuple, list)):
        raise TypeError("values must be tuple[float, ...] or list[float]")
    weights = tuple(
        ensure_non_negative_number(value, field_name=f"values[{index}]")
        for index, value in enumerate(values)
    )
    if not weights:
        raise ValueError("values must be non-empty")
    total = sum(weights)
    if total == 0.0:
        uniform_value = 1.0 / len(weights)
        return tuple(uniform_value for _ in weights)
    return tuple(weight / total for weight in weights)


def shannon_entropy(
    values: tuple[float, ...] | list[float],
    *,
    log_base: float = 2.0,
) -> float:
    """Compute Shannon entropy from non-negative weights."""

    log_base = _ensure_log_base(log_base)
    distribution = normalize_distribution(values)
    entropy = -sum(
        probability * (log(probability) / log(log_base))
        for probability in distribution
        if probability > 0.0
    )
    return ensure_non_negative_number(entropy, field_name="entropy")


def binary_entropy(
    probability: float,
    *,
    log_base: float = 2.0,
) -> float:
    """Compute entropy of a Bernoulli variable."""

    probability = ensure_constraint_probability(
        probability,
        field_name="probability",
    )
    return shannon_entropy((probability, 1.0 - probability), log_base=log_base)


def confidence_entropy_value(
    confidence: float,
    *,
    policy: EntropyPolicy | None = None,
) -> tuple[float, float, float]:
    """Compute entropy values from confidence."""

    policy = _ensure_policy(policy)
    confidence = ensure_constraint_probability(confidence, field_name="confidence")
    uncertainty = 1.0 - confidence
    value = binary_entropy(uncertainty, log_base=policy.log_base)
    normalized_value = (
        normalize_entropy_value(value=value, max_entropy=policy.max_entropy)
        if policy.normalize
        else clamp_probability(value)
    )
    output_confidence = max(policy.confidence_floor, confidence)
    output_confidence = ensure_constraint_probability(
        output_confidence,
        field_name="output_confidence",
    )
    return value, normalized_value, output_confidence


def distribution_entropy_value(
    values: tuple[float, ...] | list[float],
    *,
    policy: EntropyPolicy | None = None,
) -> tuple[float, float, float]:
    """Compute entropy values from a distribution."""

    policy = _ensure_policy(policy)
    value = shannon_entropy(values, log_base=policy.log_base)
    normalized_value = (
        normalize_entropy_value(value=value, max_entropy=policy.max_entropy)
        if policy.normalize
        else clamp_probability(value)
    )
    confidence = max(policy.confidence_floor, 1.0 - normalized_value)
    confidence = ensure_constraint_probability(confidence, field_name="confidence")
    return value, normalized_value, confidence


def source_refs_confidence(
    source_refs: tuple[ValuationSourceRef, ...] | list[ValuationSourceRef],
) -> float:
    """Return arithmetic mean confidence for source references."""

    values = tuple(source_refs)
    if not values:
        return 1.0
    for source_ref in values:
        if not isinstance(source_ref, ValuationSourceRef):
            raise TypeError("source_refs must contain ValuationSourceRef")
    return ensure_constraint_probability(
        sum(source_ref.confidence for source_ref in values) / len(values),
        field_name="source_refs_confidence",
    )


def source_ref_from_alignment_summary(
    summary: AlignmentSummary,
) -> ValuationSourceRef:
    """Create a valuation source reference from AlignmentSummary."""

    if not isinstance(summary, AlignmentSummary):
        raise TypeError("summary must be AlignmentSummary")
    source_id = (
        summary.best_alignment_id
        if summary.best_alignment_id is not None
        else "alignment_summary"
    )
    return create_source_ref(
        source_id=source_id,
        source_type="AlignmentSummary",
        confidence=summary.mean_confidence,
        metadata={
            "count": summary.count,
            "mean_score": summary.mean_score,
            "best_score": summary.best_score,
        },
    )


def compute_confidence_entropy(
    *,
    confidence: float,
    source_id: str = "confidence",
    source_type: str = "scalar_confidence",
    policy: EntropyPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> EntropyResult:
    """Create EntropyResult from a confidence value."""

    policy = _ensure_policy(policy)
    confidence = ensure_constraint_probability(confidence, field_name="confidence")
    source_id = _ensure_non_empty_string(source_id, "source_id")
    source_type = _ensure_non_empty_string(source_type, "source_type")
    value, normalized_value, output_confidence = confidence_entropy_value(
        confidence,
        policy=policy,
    )
    source_ref = create_source_ref(
        source_id=source_id,
        source_type=source_type,
        confidence=output_confidence,
        metadata={"input_confidence": confidence},
    )

    default_metadata: dict[str, object] = {
        "source": "confidence_entropy",
        "input_confidence": confidence,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "confidence_entropy"

    return EntropyResult(
        entropy_id=new_entropy_id(),
        kind=EntropyKind.UNCERTAINTY,
        value=value,
        normalized_value=normalized_value,
        confidence=output_confidence,
        source_refs=(source_ref,),
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def compute_distribution_entropy(
    *,
    values: tuple[float, ...] | list[float],
    source_id: str = "distribution",
    source_type: str = "distribution",
    policy: EntropyPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> EntropyResult:
    """Create EntropyResult from a distribution."""

    policy = _ensure_policy(policy)
    weights = tuple(
        ensure_non_negative_number(value, field_name=f"values[{index}]")
        for index, value in enumerate(values)
    )
    if not weights:
        raise ValueError("values must be non-empty")
    source_id = _ensure_non_empty_string(source_id, "source_id")
    source_type = _ensure_non_empty_string(source_type, "source_type")
    value, normalized_value, confidence = distribution_entropy_value(
        weights,
        policy=policy,
    )
    source_ref = create_source_ref(
        source_id=source_id,
        source_type=source_type,
        confidence=confidence,
        metadata={"value_count": len(weights)},
    )

    default_metadata: dict[str, object] = {
        "source": "distribution_entropy",
        "value_count": len(weights),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "distribution_entropy"

    return EntropyResult(
        entropy_id=new_entropy_id(),
        kind=EntropyKind.DISPERSION,
        value=value,
        normalized_value=normalized_value,
        confidence=confidence,
        source_refs=(source_ref,),
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def compute_alignment_summary_entropy(
    *,
    summary: AlignmentSummary,
    policy: EntropyPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> EntropyResult:
    """Create EntropyResult from AlignmentSummary."""

    if not isinstance(summary, AlignmentSummary):
        raise TypeError("summary must be AlignmentSummary")
    policy = _ensure_policy(policy)
    values = (
        summary.mean_score,
        1.0 - summary.mean_score,
        1.0 - summary.mean_confidence,
    )
    value, normalized_value, confidence = distribution_entropy_value(
        values,
        policy=policy,
    )
    source_ref = source_ref_from_alignment_summary(summary)

    default_metadata: dict[str, object] = {
        "source": "alignment_summary_entropy",
        "alignment_count": summary.count,
        "mean_score": summary.mean_score,
        "mean_confidence": summary.mean_confidence,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "alignment_summary_entropy"

    return EntropyResult(
        entropy_id=new_entropy_id(),
        kind=EntropyKind.STRUCTURAL_ENTROPY,
        value=value,
        normalized_value=normalized_value,
        confidence=confidence,
        source_refs=(source_ref,),
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def combine_entropy_results(
    entropies: tuple[EntropyResult, ...] | list[EntropyResult],
    *,
    kind: EntropyKind = EntropyKind.UNKNOWN,
    policy: EntropyPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> EntropyResult:
    """Combine multiple EntropyResult objects into one EntropyResult."""

    _ensure_policy(policy)
    if not isinstance(kind, EntropyKind):
        raise TypeError("kind must be EntropyKind")
    values = tuple(entropies)
    for entropy in values:
        if not isinstance(entropy, EntropyResult):
            raise TypeError("entropies must contain EntropyResult")

    if kind is EntropyKind.UNKNOWN and values:
        kind = EntropyKind.STRUCTURAL_ENTROPY

    if not values:
        value = 0.0
        normalized_value = 0.0
        confidence = 1.0
        source_refs: tuple[ValuationSourceRef, ...] = ()
    else:
        value = sum(entropy.value for entropy in values) / len(values)
        normalized_value = (
            sum(entropy.normalized_value for entropy in values) / len(values)
        )
        confidence = sum(entropy.confidence for entropy in values) / len(values)
        refs: list[ValuationSourceRef] = []
        for entropy in values:
            refs.extend(entropy.source_refs)
        source_refs = tuple(refs)

    default_metadata: dict[str, object] = {
        "source": "combined_entropy",
        "entropy_count": len(values),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "combined_entropy"

    return EntropyResult(
        entropy_id=new_entropy_id(),
        kind=kind,
        value=value,
        normalized_value=ensure_constraint_probability(
            normalized_value,
            field_name="normalized_value",
        ),
        confidence=ensure_constraint_probability(confidence, field_name="confidence"),
        source_refs=source_refs,
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def entropy_is_low(
    entropy: EntropyResult,
    *,
    threshold: float = 0.35,
) -> bool:
    """Return True when entropy is low."""

    if not isinstance(entropy, EntropyResult):
        raise TypeError("entropy must be EntropyResult")
    threshold = ensure_constraint_probability(threshold, field_name="threshold")
    return entropy.normalized_value <= threshold


def entropy_summary(
    entropies: tuple[EntropyResult, ...] | list[EntropyResult],
) -> dict[str, object]:
    """Summarize entropy results."""

    values = tuple(entropies)
    for entropy in values:
        if not isinstance(entropy, EntropyResult):
            raise TypeError("entropies must contain EntropyResult")
    count = len(values)
    kinds: dict[str, int] = {}
    for entropy in values:
        kinds[entropy.kind.value] = kinds.get(entropy.kind.value, 0) + 1

    return {
        "count": count,
        "mean_value": 0.0 if not values else sum(item.value for item in values) / count,
        "mean_normalized_value": (
            0.0 if not values else sum(item.normalized_value for item in values) / count
        ),
        "mean_confidence": (
            0.0 if not values else sum(item.confidence for item in values) / count
        ),
        "kinds": kinds,
        "low_entropy_count": sum(1 for entropy in values if entropy_is_low(entropy)),
    }


def _ensure_policy(policy: EntropyPolicy | None) -> EntropyPolicy:
    if policy is None:
        return EntropyPolicy()
    if not isinstance(policy, EntropyPolicy):
        raise TypeError("policy must be EntropyPolicy")
    return policy


def _ensure_log_base(value: float) -> float:
    value = ensure_finite_number(value, field_name="log_base")
    if value <= 0.0:
        raise ValueError("log_base must be positive")
    if value == 1.0:
        raise ValueError("log_base must not equal 1.0")
    return value


def _ensure_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    if value == "":
        raise ValueError(f"{field_name} must be non-empty")
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


__all__ = [
    "EntropyPolicy",
    "clamp_probability",
    "normalize_entropy_value",
    "ensure_probability_values",
    "normalize_distribution",
    "shannon_entropy",
    "binary_entropy",
    "confidence_entropy_value",
    "distribution_entropy_value",
    "source_refs_confidence",
    "source_ref_from_alignment_summary",
    "compute_confidence_entropy",
    "compute_distribution_entropy",
    "compute_alignment_summary_entropy",
    "combine_entropy_results",
    "entropy_is_low",
    "entropy_summary",
]
