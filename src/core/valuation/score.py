"""Deterministic valuation-score utilities for CEBA Core Valuation."""

from __future__ import annotations

from dataclasses import dataclass

from ceba.core.dynamics.stability import (
    StabilityReport,
    stability_report_is_blocked,
    stability_report_is_passed,
)
from ceba.core.metrics.schema import (
    MetricCostResult,
    ensure_constraint_probability,
    ensure_finite_number,
    ensure_non_negative_number,
)
from ceba.core.telos.schema import TelosAlignmentResult
from ceba.core.valuation.entropy import entropy_is_low
from ceba.core.valuation.free_energy import (
    free_energy_is_low,
    source_ref_from_entropy,
    source_ref_from_utility,
)
from ceba.core.valuation.schema import (
    EntropyResult,
    FreeEnergyResult,
    UtilityResult,
    ValuationKind,
    ValuationScore,
    ValuationSourceRef,
    ValuationStatus,
    create_source_ref,
    create_valuation_score,
    valuation_status_from_score,
)
from ceba.core.valuation.utility import (
    source_ref_from_alignment,
    source_ref_from_stability_report,
)


@dataclass(frozen=True)
class ValuationScorePolicy:
    """Stable policy for valuation score computation."""

    utility_weight: float = 1.0
    entropy_weight: float = 1.0
    free_energy_weight: float = 1.0
    alignment_weight: float = 1.0
    stability_weight: float = 1.0
    cost_weight: float = 1.0
    min_valid_score: float = 0.35

    def __post_init__(self) -> None:
        ensure_non_negative_number(
            self.utility_weight,
            field_name="utility_weight",
        )
        ensure_non_negative_number(
            self.entropy_weight,
            field_name="entropy_weight",
        )
        ensure_non_negative_number(
            self.free_energy_weight,
            field_name="free_energy_weight",
        )
        ensure_non_negative_number(
            self.alignment_weight,
            field_name="alignment_weight",
        )
        ensure_non_negative_number(
            self.stability_weight,
            field_name="stability_weight",
        )
        ensure_non_negative_number(
            self.cost_weight,
            field_name="cost_weight",
        )
        ensure_constraint_probability(
            self.min_valid_score,
            field_name="min_valid_score",
        )


def _merge_metadata(
    defaults: dict[str, object],
    metadata: dict[str, object] | None,
) -> dict[str, object]:
    merged = dict(defaults)
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be dict[str, object]")
        for key in metadata:
            if not isinstance(key, str):
                raise TypeError("metadata keys must be strings")
        merged.update(metadata)
    return merged


def _ensure_policy(
    policy: ValuationScorePolicy | None,
) -> ValuationScorePolicy:
    if policy is None:
        return ValuationScorePolicy()
    if not isinstance(policy, ValuationScorePolicy):
        raise TypeError("policy must be ValuationScorePolicy")
    return policy


def _create_score(
    *,
    kind: ValuationKind,
    value: float,
    normalized_value: float,
    confidence: float,
    source_refs: tuple[ValuationSourceRef, ...],
    metadata: dict[str, object],
) -> ValuationScore:
    status = valuation_status_from_score(
        normalized_score=normalized_value,
        confidence=confidence,
    )
    created = create_valuation_score(
        kind=kind,
        value=value,
        normalized_value=normalized_value,
        confidence=confidence,
        source_refs=source_refs,
        status=status,
        metadata=metadata,
    )
    return ValuationScore(
        score_id=created.score_id,
        kind=created.kind,
        status=created.status,
        value=created.value,
        normalized_value=created.normalized_value,
        confidence=created.confidence,
        source_refs=created.source_refs,
        created_at=created.created_at,
        metadata=metadata,
    )


def _ensure_source_refs(
    refs: tuple[ValuationSourceRef, ...] | list[ValuationSourceRef],
    *,
    field_name: str = "source_refs",
) -> tuple[ValuationSourceRef, ...]:
    if not isinstance(refs, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple or list")
    values = tuple(refs)
    for ref in values:
        if not isinstance(ref, ValuationSourceRef):
            raise TypeError(f"{field_name} must contain ValuationSourceRef")
    return values


def clamp_probability(value: float) -> float:
    """Clamp a finite number to the probability range."""

    value = ensure_finite_number(value, field_name="value")
    return max(0.0, min(1.0, value))


def invert_probability(value: float) -> float:
    """Invert a probability value."""

    value = ensure_constraint_probability(value, field_name="value")
    return 1.0 - value


def source_ref_from_metric_cost(cost: MetricCostResult) -> ValuationSourceRef:
    """Create a valuation source reference from a metric cost result."""

    if not isinstance(cost, MetricCostResult):
        raise TypeError("cost must be MetricCostResult")
    return create_source_ref(
        source_id=str(cost.cost_id),
        source_type="MetricCostResult",
        confidence=cost.confidence,
        metadata={
            "cost_kind": cost.kind.value,
            "cost_value": cost.value,
            "normalized_cost": cost.normalized_value,
        },
    )


def score_from_utility(
    *,
    utility: UtilityResult,
    policy: ValuationScorePolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> ValuationScore:
    """Create ValuationScore from UtilityResult."""

    if not isinstance(utility, UtilityResult):
        raise TypeError("utility must be UtilityResult")
    policy = _ensure_policy(policy)
    value = ensure_finite_number(
        utility.value * policy.utility_weight,
        field_name="value",
    )
    normalized_value = ensure_constraint_probability(
        utility.normalized_value,
        field_name="normalized_value",
    )
    confidence = ensure_constraint_probability(
        utility.confidence,
        field_name="confidence",
    )
    source_refs = (
        source_ref_from_utility(utility),
        *_ensure_source_refs(utility.source_refs),
    )
    default_metadata: dict[str, object] = {
        "source": "utility_valuation_score",
        "utility_kind": utility.kind.value,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "utility_valuation_score"
    return _create_score(
        kind=ValuationKind.UTILITY,
        value=value,
        normalized_value=normalized_value,
        confidence=confidence,
        source_refs=source_refs,
        metadata=merged_metadata,
    )


def score_from_entropy(
    *,
    entropy: EntropyResult,
    policy: ValuationScorePolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> ValuationScore:
    """Create ValuationScore from EntropyResult."""

    if not isinstance(entropy, EntropyResult):
        raise TypeError("entropy must be EntropyResult")
    policy = _ensure_policy(policy)
    value = ensure_finite_number(
        entropy.value * policy.entropy_weight,
        field_name="value",
    )
    normalized_value = invert_probability(entropy.normalized_value)
    confidence = ensure_constraint_probability(
        entropy.confidence,
        field_name="confidence",
    )
    source_refs = (
        source_ref_from_entropy(entropy),
        *_ensure_source_refs(entropy.source_refs),
    )
    default_metadata: dict[str, object] = {
        "source": "entropy_valuation_score",
        "entropy_kind": entropy.kind.value,
        "entropy_low": entropy_is_low(entropy),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "entropy_valuation_score"
    return _create_score(
        kind=ValuationKind.ENTROPY,
        value=value,
        normalized_value=normalized_value,
        confidence=confidence,
        source_refs=source_refs,
        metadata=merged_metadata,
    )


def score_from_free_energy(
    *,
    free_energy: FreeEnergyResult,
    policy: ValuationScorePolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> ValuationScore:
    """Create ValuationScore from FreeEnergyResult."""

    if not isinstance(free_energy, FreeEnergyResult):
        raise TypeError("free_energy must be FreeEnergyResult")
    policy = _ensure_policy(policy)
    value = ensure_finite_number(
        free_energy.value * policy.free_energy_weight,
        field_name="value",
    )
    normalized_value = invert_probability(free_energy.normalized_value)
    confidence = ensure_constraint_probability(
        free_energy.confidence,
        field_name="confidence",
    )
    source_refs = _ensure_source_refs(free_energy.source_refs)
    default_metadata: dict[str, object] = {
        "source": "free_energy_valuation_score",
        "free_energy_kind": free_energy.kind.value,
        "free_energy_low": free_energy_is_low(free_energy),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "free_energy_valuation_score"
    return _create_score(
        kind=ValuationKind.FREE_ENERGY,
        value=value,
        normalized_value=normalized_value,
        confidence=confidence,
        source_refs=source_refs,
        metadata=merged_metadata,
    )


def score_from_alignment(
    *,
    alignment: TelosAlignmentResult,
    policy: ValuationScorePolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> ValuationScore:
    """Create ValuationScore from TelosAlignmentResult."""

    if not isinstance(alignment, TelosAlignmentResult):
        raise TypeError("alignment must be TelosAlignmentResult")
    policy = _ensure_policy(policy)
    value = ensure_finite_number(
        alignment.score * policy.alignment_weight,
        field_name="value",
    )
    normalized_value = ensure_constraint_probability(
        alignment.score,
        field_name="normalized_value",
    )
    confidence = ensure_constraint_probability(
        alignment.confidence,
        field_name="confidence",
    )
    default_metadata: dict[str, object] = {
        "source": "alignment_valuation_score",
        "alignment_status": alignment.status.value,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "alignment_valuation_score"
    return _create_score(
        kind=ValuationKind.ALIGNMENT,
        value=value,
        normalized_value=normalized_value,
        confidence=confidence,
        source_refs=(source_ref_from_alignment(alignment),),
        metadata=merged_metadata,
    )


def score_from_stability(
    *,
    stability: StabilityReport,
    policy: ValuationScorePolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> ValuationScore:
    """Create ValuationScore from StabilityReport."""

    if not isinstance(stability, StabilityReport):
        raise TypeError("stability must be StabilityReport")
    policy = _ensure_policy(policy)
    blocked = stability_report_is_blocked(stability)
    normalized_value = 0.0 if blocked else stability.score
    normalized_value = ensure_constraint_probability(
        normalized_value,
        field_name="normalized_value",
    )
    value = ensure_finite_number(
        normalized_value * policy.stability_weight,
        field_name="value",
    )
    confidence = ensure_constraint_probability(
        stability.score,
        field_name="confidence",
    )
    default_metadata: dict[str, object] = {
        "source": "stability_valuation_score",
        "stability_status": stability.status.value,
        "stability_passed": stability_report_is_passed(stability),
        "stability_blocked": blocked,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "stability_valuation_score"
    return _create_score(
        kind=ValuationKind.STABILITY,
        value=value,
        normalized_value=normalized_value,
        confidence=confidence,
        source_refs=(source_ref_from_stability_report(stability),),
        metadata=merged_metadata,
    )


def score_from_metric_cost(
    *,
    cost: MetricCostResult,
    policy: ValuationScorePolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> ValuationScore:
    """Create ValuationScore from MetricCostResult."""

    if not isinstance(cost, MetricCostResult):
        raise TypeError("cost must be MetricCostResult")
    if cost.normalized_value is None:
        raise ValueError("cost.normalized_value must not be None")
    policy = _ensure_policy(policy)
    value = ensure_finite_number(
        cost.value * policy.cost_weight,
        field_name="value",
    )
    normalized_value = invert_probability(cost.normalized_value)
    confidence = ensure_constraint_probability(
        cost.confidence,
        field_name="confidence",
    )
    default_metadata: dict[str, object] = {
        "source": "metric_cost_valuation_score",
        "cost_kind": cost.kind.value,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "metric_cost_valuation_score"
    return _create_score(
        kind=ValuationKind.COST,
        value=value,
        normalized_value=normalized_value,
        confidence=confidence,
        source_refs=(source_ref_from_metric_cost(cost),),
        metadata=merged_metadata,
    )


def combine_valuation_scores(
    scores: tuple[ValuationScore, ...] | list[ValuationScore],
    *,
    kind: ValuationKind = ValuationKind.COMPOSITE,
    metadata: dict[str, object] | None = None,
) -> ValuationScore:
    """Combine multiple ValuationScore objects into one composite score."""

    if not isinstance(scores, (tuple, list)):
        raise TypeError("scores must be tuple or list")
    if not isinstance(kind, ValuationKind):
        raise TypeError("kind must be ValuationKind")
    values = tuple(scores)
    for score in values:
        if not isinstance(score, ValuationScore):
            raise TypeError("scores must contain ValuationScore")

    if not values:
        value = 0.0
        normalized_value = 0.0
        confidence = 1.0
        source_refs: tuple[ValuationSourceRef, ...] = ()
    else:
        value = sum(score.value for score in values) / len(values)
        normalized_value = (
            sum(score.normalized_value for score in values) / len(values)
        )
        confidence = sum(score.confidence for score in values) / len(values)
        refs: list[ValuationSourceRef] = []
        for score in values:
            refs.extend(score.source_refs)
        source_refs = tuple(refs)

    value = ensure_finite_number(value, field_name="value")
    normalized_value = ensure_constraint_probability(
        normalized_value,
        field_name="normalized_value",
    )
    confidence = ensure_constraint_probability(
        confidence,
        field_name="confidence",
    )
    default_metadata: dict[str, object] = {
        "source": "combined_valuation_score",
        "score_count": len(values),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "combined_valuation_score"
    return _create_score(
        kind=kind,
        value=value,
        normalized_value=normalized_value,
        confidence=confidence,
        source_refs=source_refs,
        metadata=merged_metadata,
    )


def valuation_score_is_valid(
    score: ValuationScore,
    *,
    policy: ValuationScorePolicy | None = None,
) -> bool:
    """Return whether a valuation score is valid enough for use."""

    if not isinstance(score, ValuationScore):
        raise TypeError("score must be ValuationScore")
    policy = _ensure_policy(policy)
    return (
        score.status is ValuationStatus.VALID
        or score.normalized_value >= policy.min_valid_score
    )


def valuation_score_summary(
    scores: tuple[ValuationScore, ...] | list[ValuationScore],
) -> dict[str, object]:
    """Return a shallow deterministic summary of valuation scores."""

    if not isinstance(scores, (tuple, list)):
        raise TypeError("scores must be tuple or list")
    values = tuple(scores)
    for score in values:
        if not isinstance(score, ValuationScore):
            raise TypeError("scores must contain ValuationScore")

    count = len(values)
    if count == 0:
        mean_value = 0.0
        mean_normalized_value = 0.0
        mean_confidence = 0.0
    else:
        mean_value = sum(score.value for score in values) / count
        mean_normalized_value = (
            sum(score.normalized_value for score in values) / count
        )
        mean_confidence = sum(score.confidence for score in values) / count

    kinds: dict[str, int] = {}
    statuses: dict[str, int] = {}
    valid_count = 0
    for score in values:
        kinds[score.kind.value] = kinds.get(score.kind.value, 0) + 1
        statuses[score.status.value] = statuses.get(score.status.value, 0) + 1
        if valuation_score_is_valid(score):
            valid_count += 1

    return {
        "count": count,
        "mean_value": mean_value,
        "mean_normalized_value": mean_normalized_value,
        "mean_confidence": mean_confidence,
        "kinds": kinds,
        "statuses": statuses,
        "valid_count": valid_count,
    }


__all__ = [
    "ValuationScorePolicy",
    "clamp_probability",
    "invert_probability",
    "source_ref_from_metric_cost",
    "score_from_utility",
    "score_from_entropy",
    "score_from_free_energy",
    "score_from_alignment",
    "score_from_stability",
    "score_from_metric_cost",
    "combine_valuation_scores",
    "valuation_score_is_valid",
    "valuation_score_summary",
]
