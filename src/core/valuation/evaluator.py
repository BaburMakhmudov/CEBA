"""Deterministic valuation evaluator utilities for CEBA Core Valuation."""

from __future__ import annotations

from dataclasses import dataclass, field

from ceba.core.metrics.schema import (
    ensure_constraint_probability,
    ensure_finite_number,
)
from ceba.core.valuation.entropy import (
    combine_entropy_results,
    entropy_is_low,
    entropy_summary,
)
from ceba.core.valuation.free_energy import (
    combine_free_energy_results,
    free_energy_is_low,
    free_energy_summary,
)
from ceba.core.valuation.schema import (
    EntropyResult,
    FreeEnergyResult,
    UtilityResult,
    ValuationEvaluation,
    ValuationScore,
    ValuationStatus,
    create_valuation_evaluation,
    valuation_status_from_score,
)
from ceba.core.valuation.score import (
    combine_valuation_scores,
    score_from_entropy,
    score_from_free_energy,
    score_from_utility,
    valuation_score_summary,
)
from ceba.core.valuation.utility import (
    combine_utility_results,
    utility_summary,
)


@dataclass(frozen=True)
class ValuationEvaluatorPolicy:
    """Stable policy for valuation evaluation aggregation."""

    min_aggregate_score: float = 0.35
    min_confidence: float = 0.35
    require_low_free_energy: bool = False
    require_low_entropy: bool = False
    include_component_scores: bool = True
    include_composite_score: bool = True

    def __post_init__(self) -> None:
        min_aggregate_score = ensure_constraint_probability(
            self.min_aggregate_score,
            field_name="min_aggregate_score",
        )
        min_confidence = ensure_constraint_probability(
            self.min_confidence,
            field_name="min_confidence",
        )
        if not isinstance(self.require_low_free_energy, bool):
            raise TypeError("require_low_free_energy must be bool")
        if not isinstance(self.require_low_entropy, bool):
            raise TypeError("require_low_entropy must be bool")
        if not isinstance(self.include_component_scores, bool):
            raise TypeError("include_component_scores must be bool")
        if not isinstance(self.include_composite_score, bool):
            raise TypeError("include_composite_score must be bool")

        object.__setattr__(self, "min_aggregate_score", min_aggregate_score)
        object.__setattr__(self, "min_confidence", min_confidence)


@dataclass(frozen=True)
class ValuationEvaluationInput:
    """Immutable input bundle for valuation evaluation."""

    utilities: tuple[UtilityResult, ...] = ()
    entropies: tuple[EntropyResult, ...] = ()
    free_energies: tuple[FreeEnergyResult, ...] = ()
    scores: tuple[ValuationScore, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        utilities = _ensure_utilities(self.utilities)
        entropies = _ensure_entropies(self.entropies)
        free_energies = _ensure_free_energies(self.free_energies)
        scores = _ensure_scores(self.scores)

        object.__setattr__(self, "utilities", utilities)
        object.__setattr__(self, "entropies", entropies)
        object.__setattr__(self, "free_energies", free_energies)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "utilities": tuple(utility.to_dict() for utility in self.utilities),
            "entropies": tuple(entropy.to_dict() for entropy in self.entropies),
            "free_energies": tuple(
                free_energy.to_dict() for free_energy in self.free_energies
            ),
            "scores": tuple(score.to_dict() for score in self.scores),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ValuationEvaluationSummary:
    """Serializable-friendly summary of multiple ValuationEvaluation results."""

    count: int
    mean_aggregate_score: float
    mean_confidence: float
    statuses: dict[str, int]
    valid_count: int
    blocked_count: int
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        count = ensure_non_negative_int(self.count, "count")
        mean_aggregate_score = ensure_constraint_probability(
            self.mean_aggregate_score,
            field_name="mean_aggregate_score",
        )
        mean_confidence = ensure_constraint_probability(
            self.mean_confidence,
            field_name="mean_confidence",
        )
        statuses = _copy_counts(self.statuses, "statuses")
        valid_count = ensure_non_negative_int(self.valid_count, "valid_count")
        blocked_count = ensure_non_negative_int(
            self.blocked_count,
            "blocked_count",
        )

        object.__setattr__(self, "count", count)
        object.__setattr__(self, "mean_aggregate_score", mean_aggregate_score)
        object.__setattr__(self, "mean_confidence", mean_confidence)
        object.__setattr__(self, "statuses", statuses)
        object.__setattr__(self, "valid_count", valid_count)
        object.__setattr__(self, "blocked_count", blocked_count)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "count": self.count,
            "mean_aggregate_score": self.mean_aggregate_score,
            "mean_confidence": self.mean_confidence,
            "statuses": dict(self.statuses),
            "valid_count": self.valid_count,
            "blocked_count": self.blocked_count,
            "metadata": dict(self.metadata),
        }


def clamp_probability(value: float) -> float:
    """Clamp a finite number to the probability range."""

    value = ensure_finite_number(value, field_name="value")
    return max(0.0, min(1.0, value))


def ensure_non_negative_int(value: int, field_name: str) -> int:
    """Validate a non-negative integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def create_evaluation_input(
    *,
    utilities: tuple[UtilityResult, ...] | list[UtilityResult] = (),
    entropies: tuple[EntropyResult, ...] | list[EntropyResult] = (),
    free_energies: tuple[FreeEnergyResult, ...] | list[FreeEnergyResult] = (),
    scores: tuple[ValuationScore, ...] | list[ValuationScore] = (),
    metadata: dict[str, object] | None = None,
) -> ValuationEvaluationInput:
    """Create a validated valuation evaluation input bundle."""

    utilities_value = _ensure_utilities(utilities)
    entropies_value = _ensure_entropies(entropies)
    free_energies_value = _ensure_free_energies(free_energies)
    scores_value = _ensure_scores(scores)
    default_metadata: dict[str, object] = {
        "source": "valuation_evaluation_input",
        "utility_count": len(utilities_value),
        "entropy_count": len(entropies_value),
        "free_energy_count": len(free_energies_value),
        "score_count": len(scores_value),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "valuation_evaluation_input"
    return ValuationEvaluationInput(
        utilities=utilities_value,
        entropies=entropies_value,
        free_energies=free_energies_value,
        scores=scores_value,
        metadata=merged_metadata,
    )


def evaluation_status_from_inputs(
    *,
    aggregate_score: float,
    confidence: float,
    free_energy: FreeEnergyResult | None = None,
    entropy: EntropyResult | None = None,
    policy: ValuationEvaluatorPolicy | None = None,
) -> ValuationStatus:
    """Determine evaluation status from aggregate score and evaluator policy."""

    policy = _ensure_policy(policy)
    aggregate_score = ensure_constraint_probability(
        aggregate_score,
        field_name="aggregate_score",
    )
    confidence = ensure_constraint_probability(confidence, field_name="confidence")
    if free_energy is not None and not isinstance(free_energy, FreeEnergyResult):
        raise TypeError("free_energy must be FreeEnergyResult or None")
    if entropy is not None and not isinstance(entropy, EntropyResult):
        raise TypeError("entropy must be EntropyResult or None")

    if confidence < policy.min_confidence:
        return ValuationStatus.DEGRADED
    if aggregate_score < policy.min_aggregate_score:
        return ValuationStatus.UNSTABLE
    if (
        policy.require_low_free_energy
        and free_energy is not None
        and not free_energy_is_low(free_energy)
    ):
        return ValuationStatus.DEGRADED
    if (
        policy.require_low_entropy
        and entropy is not None
        and not entropy_is_low(entropy)
    ):
        return ValuationStatus.DEGRADED
    return valuation_status_from_score(
        normalized_score=aggregate_score,
        confidence=confidence,
    )


def build_component_scores(
    *,
    utility: UtilityResult | None = None,
    entropy: EntropyResult | None = None,
    free_energy: FreeEnergyResult | None = None,
    extra_scores: tuple[ValuationScore, ...] | list[ValuationScore] = (),
    policy: ValuationEvaluatorPolicy | None = None,
) -> tuple[ValuationScore, ...]:
    """Build component valuation scores from available artifacts."""

    policy = _ensure_policy(policy)
    if utility is not None and not isinstance(utility, UtilityResult):
        raise TypeError("utility must be UtilityResult or None")
    if entropy is not None and not isinstance(entropy, EntropyResult):
        raise TypeError("entropy must be EntropyResult or None")
    if free_energy is not None and not isinstance(free_energy, FreeEnergyResult):
        raise TypeError("free_energy must be FreeEnergyResult or None")

    scores = list(_ensure_scores(extra_scores))
    if policy.include_component_scores:
        if utility is not None:
            scores.append(score_from_utility(utility=utility))
        if entropy is not None:
            scores.append(score_from_entropy(entropy=entropy))
        if free_energy is not None:
            scores.append(score_from_free_energy(free_energy=free_energy))
    if policy.include_composite_score and scores:
        scores.append(combine_valuation_scores(tuple(scores)))
    return tuple(scores)


def aggregate_score_from_scores(
    scores: tuple[ValuationScore, ...] | list[ValuationScore],
) -> float:
    """Return arithmetic mean normalized valuation score."""

    values = _ensure_scores(scores)
    if not values:
        return 0.0
    return ensure_constraint_probability(
        sum(score.normalized_value for score in values) / len(values),
        field_name="aggregate_score",
    )


def aggregate_confidence_from_components(
    *,
    utility: UtilityResult | None = None,
    entropy: EntropyResult | None = None,
    free_energy: FreeEnergyResult | None = None,
    scores: tuple[ValuationScore, ...] | list[ValuationScore] = (),
) -> float:
    """Return aggregate confidence from available valuation components."""

    if utility is not None and not isinstance(utility, UtilityResult):
        raise TypeError("utility must be UtilityResult or None")
    if entropy is not None and not isinstance(entropy, EntropyResult):
        raise TypeError("entropy must be EntropyResult or None")
    if free_energy is not None and not isinstance(free_energy, FreeEnergyResult):
        raise TypeError("free_energy must be FreeEnergyResult or None")
    values: list[float] = []
    if utility is not None:
        values.append(utility.confidence)
    if entropy is not None:
        values.append(entropy.confidence)
    if free_energy is not None:
        values.append(free_energy.confidence)
    values.extend(score.confidence for score in _ensure_scores(scores))
    if not values:
        return 1.0
    return ensure_constraint_probability(
        sum(values) / len(values),
        field_name="confidence",
    )


def evaluate_components(
    *,
    utility: UtilityResult | None = None,
    entropy: EntropyResult | None = None,
    free_energy: FreeEnergyResult | None = None,
    scores: tuple[ValuationScore, ...] | list[ValuationScore] = (),
    policy: ValuationEvaluatorPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> ValuationEvaluation:
    """Create ValuationEvaluation from already combined components."""

    policy = _ensure_policy(policy)
    if utility is not None and not isinstance(utility, UtilityResult):
        raise TypeError("utility must be UtilityResult or None")
    if entropy is not None and not isinstance(entropy, EntropyResult):
        raise TypeError("entropy must be EntropyResult or None")
    if free_energy is not None and not isinstance(free_energy, FreeEnergyResult):
        raise TypeError("free_energy must be FreeEnergyResult or None")

    built_scores = build_component_scores(
        utility=utility,
        entropy=entropy,
        free_energy=free_energy,
        extra_scores=scores,
        policy=policy,
    )
    aggregate_score = aggregate_score_from_scores(built_scores)
    confidence = aggregate_confidence_from_components(
        utility=utility,
        entropy=entropy,
        free_energy=free_energy,
        scores=built_scores,
    )
    status = evaluation_status_from_inputs(
        aggregate_score=aggregate_score,
        confidence=confidence,
        free_energy=free_energy,
        entropy=entropy,
        policy=policy,
    )
    default_metadata: dict[str, object] = {
        "source": "valuation_evaluator",
        "has_utility": utility is not None,
        "has_entropy": entropy is not None,
        "has_free_energy": free_energy is not None,
        "score_count": len(built_scores),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "valuation_evaluator"
    return _create_evaluation_with_metadata(
        utility=utility,
        entropy=entropy,
        free_energy=free_energy,
        scores=built_scores,
        status=status,
        metadata=merged_metadata,
    )


def evaluate_input_bundle(
    *,
    bundle: ValuationEvaluationInput,
    policy: ValuationEvaluatorPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> ValuationEvaluation:
    """Create ValuationEvaluation from an input bundle."""

    if not isinstance(bundle, ValuationEvaluationInput):
        raise TypeError("bundle must be ValuationEvaluationInput")
    policy = _ensure_policy(policy)
    combined_utility = (
        None
        if not bundle.utilities
        else combine_utility_results(
            bundle.utilities,
            kind=bundle.utilities[0].kind,
        )
    )
    combined_entropy = (
        None if not bundle.entropies else combine_entropy_results(bundle.entropies)
    )
    combined_free_energy = (
        None
        if not bundle.free_energies
        else combine_free_energy_results(bundle.free_energies)
    )
    default_metadata: dict[str, object] = {
        "source": "valuation_input_bundle_evaluation",
        "input_metadata": dict(bundle.metadata),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "valuation_input_bundle_evaluation"
    evaluation = evaluate_components(
        utility=combined_utility,
        entropy=combined_entropy,
        free_energy=combined_free_energy,
        scores=bundle.scores,
        policy=policy,
    )
    return ValuationEvaluation(
        evaluation_id=evaluation.evaluation_id,
        status=evaluation.status,
        utility=evaluation.utility,
        entropy=evaluation.entropy,
        free_energy=evaluation.free_energy,
        scores=evaluation.scores,
        aggregate_score=evaluation.aggregate_score,
        confidence=evaluation.confidence,
        created_at=evaluation.created_at,
        metadata=merged_metadata,
    )


def valuation_evaluation_is_valid(
    evaluation: ValuationEvaluation,
    *,
    policy: ValuationEvaluatorPolicy | None = None,
) -> bool:
    """Return whether a valuation evaluation is valid under policy."""

    if not isinstance(evaluation, ValuationEvaluation):
        raise TypeError("evaluation must be ValuationEvaluation")
    policy = _ensure_policy(policy)
    return (
        evaluation.status in (ValuationStatus.VALID, ValuationStatus.DEGRADED)
        and evaluation.aggregate_score >= policy.min_aggregate_score
        and evaluation.confidence >= policy.min_confidence
    )


def valuation_evaluation_is_blocked(evaluation: ValuationEvaluation) -> bool:
    """Return whether a valuation evaluation is blocked."""

    if not isinstance(evaluation, ValuationEvaluation):
        raise TypeError("evaluation must be ValuationEvaluation")
    return evaluation.status in (ValuationStatus.BLOCKED, ValuationStatus.INVALID)


def summarize_valuation_evaluation(
    evaluation: ValuationEvaluation,
) -> dict[str, object]:
    """Return a shallow serializable summary of a valuation evaluation."""

    if not isinstance(evaluation, ValuationEvaluation):
        raise TypeError("evaluation must be ValuationEvaluation")
    utilities = () if evaluation.utility is None else (evaluation.utility,)
    entropies = () if evaluation.entropy is None else (evaluation.entropy,)
    free_energies = (
        () if evaluation.free_energy is None else (evaluation.free_energy,)
    )
    return {
        "evaluation_id": str(evaluation.evaluation_id),
        "status": evaluation.status.value,
        "aggregate_score": evaluation.aggregate_score,
        "confidence": evaluation.confidence,
        "has_utility": evaluation.utility is not None,
        "has_entropy": evaluation.entropy is not None,
        "has_free_energy": evaluation.free_energy is not None,
        "score_count": len(evaluation.scores),
        "utility_summary": utility_summary(utilities),
        "entropy_summary": entropy_summary(entropies),
        "free_energy_summary": free_energy_summary(free_energies),
        "score_summary": valuation_score_summary(evaluation.scores),
        "created_at": evaluation.created_at.isoformat(),
        "metadata": dict(evaluation.metadata),
    }


def summarize_valuation_evaluations(
    evaluations: tuple[ValuationEvaluation, ...] | list[ValuationEvaluation],
    *,
    metadata: dict[str, object] | None = None,
) -> ValuationEvaluationSummary:
    """Summarize multiple valuation evaluations."""

    values = _ensure_evaluations(evaluations)
    count = len(values)
    mean_aggregate_score = (
        0.0
        if not values
        else sum(evaluation.aggregate_score for evaluation in values) / count
    )
    mean_confidence = (
        0.0
        if not values
        else sum(evaluation.confidence for evaluation in values) / count
    )
    statuses: dict[str, int] = {}
    valid_count = 0
    blocked_count = 0
    for evaluation in values:
        statuses[evaluation.status.value] = (
            statuses.get(evaluation.status.value, 0) + 1
        )
        if valuation_evaluation_is_valid(evaluation):
            valid_count += 1
        if valuation_evaluation_is_blocked(evaluation):
            blocked_count += 1

    default_metadata: dict[str, object] = {
        "source": "valuation_evaluation_summary",
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "valuation_evaluation_summary"
    return ValuationEvaluationSummary(
        count=count,
        mean_aggregate_score=mean_aggregate_score,
        mean_confidence=mean_confidence,
        statuses=statuses,
        valid_count=valid_count,
        blocked_count=blocked_count,
        metadata=merged_metadata,
    )


def valuation_evaluation_summary_to_dict(
    summary: ValuationEvaluationSummary,
) -> dict[str, object]:
    """Return a dictionary representation of a valuation evaluation summary."""

    if not isinstance(summary, ValuationEvaluationSummary):
        raise TypeError("summary must be ValuationEvaluationSummary")
    return summary.to_dict()


def _create_evaluation_with_metadata(
    *,
    utility: UtilityResult | None,
    entropy: EntropyResult | None,
    free_energy: FreeEnergyResult | None,
    scores: tuple[ValuationScore, ...],
    status: ValuationStatus,
    metadata: dict[str, object],
) -> ValuationEvaluation:
    created = create_valuation_evaluation(
        utility=utility,
        entropy=entropy,
        free_energy=free_energy,
        scores=scores,
        status=status,
        metadata=metadata,
    )
    return ValuationEvaluation(
        evaluation_id=created.evaluation_id,
        status=created.status,
        utility=created.utility,
        entropy=created.entropy,
        free_energy=created.free_energy,
        scores=created.scores,
        aggregate_score=created.aggregate_score,
        confidence=created.confidence,
        created_at=created.created_at,
        metadata=metadata,
    )


def _ensure_policy(
    policy: ValuationEvaluatorPolicy | None,
) -> ValuationEvaluatorPolicy:
    if policy is None:
        return ValuationEvaluatorPolicy()
    if not isinstance(policy, ValuationEvaluatorPolicy):
        raise TypeError("policy must be ValuationEvaluatorPolicy")
    return policy


def _ensure_utilities(
    utilities: tuple[UtilityResult, ...] | list[UtilityResult],
) -> tuple[UtilityResult, ...]:
    if not isinstance(utilities, (tuple, list)):
        raise TypeError("utilities must be tuple or list")
    values = tuple(utilities)
    for utility in values:
        if not isinstance(utility, UtilityResult):
            raise TypeError("utilities must contain UtilityResult")
    return values


def _ensure_entropies(
    entropies: tuple[EntropyResult, ...] | list[EntropyResult],
) -> tuple[EntropyResult, ...]:
    if not isinstance(entropies, (tuple, list)):
        raise TypeError("entropies must be tuple or list")
    values = tuple(entropies)
    for entropy in values:
        if not isinstance(entropy, EntropyResult):
            raise TypeError("entropies must contain EntropyResult")
    return values


def _ensure_free_energies(
    free_energies: tuple[FreeEnergyResult, ...] | list[FreeEnergyResult],
) -> tuple[FreeEnergyResult, ...]:
    if not isinstance(free_energies, (tuple, list)):
        raise TypeError("free_energies must be tuple or list")
    values = tuple(free_energies)
    for free_energy in values:
        if not isinstance(free_energy, FreeEnergyResult):
            raise TypeError("free_energies must contain FreeEnergyResult")
    return values


def _ensure_scores(
    scores: tuple[ValuationScore, ...] | list[ValuationScore],
) -> tuple[ValuationScore, ...]:
    if not isinstance(scores, (tuple, list)):
        raise TypeError("scores must be tuple or list")
    values = tuple(scores)
    for score in values:
        if not isinstance(score, ValuationScore):
            raise TypeError("scores must contain ValuationScore")
    return values


def _ensure_evaluations(
    evaluations: tuple[ValuationEvaluation, ...] | list[ValuationEvaluation],
) -> tuple[ValuationEvaluation, ...]:
    if not isinstance(evaluations, (tuple, list)):
        raise TypeError("evaluations must be tuple or list")
    values = tuple(evaluations)
    for evaluation in values:
        if not isinstance(evaluation, ValuationEvaluation):
            raise TypeError("evaluations must contain ValuationEvaluation")
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


def _copy_counts(
    values: dict[str, int],
    field_name: str,
) -> dict[str, int]:
    if not isinstance(values, dict):
        raise TypeError(f"{field_name} must be dict[str, int]")
    copied = dict(values)
    for key, value in copied.items():
        if not isinstance(key, str):
            raise TypeError(f"{field_name} keys must be strings")
        ensure_non_negative_int(value, f"{field_name}[{key!r}]")
    return copied


__all__ = [
    "ValuationEvaluatorPolicy",
    "ValuationEvaluationInput",
    "ValuationEvaluationSummary",
    "clamp_probability",
    "ensure_non_negative_int",
    "create_evaluation_input",
    "evaluation_status_from_inputs",
    "build_component_scores",
    "aggregate_score_from_scores",
    "aggregate_confidence_from_components",
    "evaluate_components",
    "evaluate_input_bundle",
    "valuation_evaluation_is_valid",
    "valuation_evaluation_is_blocked",
    "summarize_valuation_evaluation",
    "summarize_valuation_evaluations",
    "valuation_evaluation_summary_to_dict",
]
