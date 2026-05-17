"""Deterministic free-energy computation utilities for CEBA Core Valuation."""

from __future__ import annotations

from dataclasses import dataclass

from ceba.core.metrics.cost import mean_normalized_cost, total_cost
from ceba.core.metrics.schema import (
    MetricCostResult,
    ensure_constraint_probability,
    ensure_finite_number,
    ensure_non_negative_number,
    utc_now,
)
from ceba.core.valuation.schema import (
    EntropyResult,
    FreeEnergyKind,
    FreeEnergyResult,
    UtilityResult,
    ValuationSourceRef,
    create_source_ref,
    new_free_energy_id,
)


@dataclass(frozen=True)
class FreeEnergyPolicy:
    """Stable policy for free-energy computation."""

    entropy_weight: float = 1.0
    utility_weight: float = 1.0
    cost_weight: float = 1.0
    prediction_error_weight: float = 1.0
    max_free_energy: float = 10.0
    confidence_floor: float = 0.0

    def __post_init__(self) -> None:
        entropy_weight = ensure_non_negative_number(
            self.entropy_weight,
            field_name="entropy_weight",
        )
        utility_weight = ensure_non_negative_number(
            self.utility_weight,
            field_name="utility_weight",
        )
        cost_weight = ensure_non_negative_number(
            self.cost_weight,
            field_name="cost_weight",
        )
        prediction_error_weight = ensure_non_negative_number(
            self.prediction_error_weight,
            field_name="prediction_error_weight",
        )
        max_free_energy = ensure_finite_number(
            self.max_free_energy,
            field_name="max_free_energy",
        )
        if max_free_energy <= 0.0:
            raise ValueError("max_free_energy must be positive")
        confidence_floor = ensure_constraint_probability(
            self.confidence_floor,
            field_name="confidence_floor",
        )

        object.__setattr__(self, "entropy_weight", entropy_weight)
        object.__setattr__(self, "utility_weight", utility_weight)
        object.__setattr__(self, "cost_weight", cost_weight)
        object.__setattr__(
            self,
            "prediction_error_weight",
            prediction_error_weight,
        )
        object.__setattr__(self, "max_free_energy", max_free_energy)
        object.__setattr__(self, "confidence_floor", confidence_floor)


def clamp_probability(value: float) -> float:
    """Clamp a finite number to the probability range."""

    value = ensure_finite_number(value, field_name="value")
    return min(max(value, 0.0), 1.0)


def normalize_free_energy_value(
    *,
    value: float,
    max_free_energy: float = 10.0,
) -> float:
    """Normalize a free-energy value using absolute magnitude."""

    value = ensure_finite_number(value, field_name="value")
    max_free_energy = ensure_finite_number(
        max_free_energy,
        field_name="max_free_energy",
    )
    if max_free_energy <= 0.0:
        raise ValueError("max_free_energy must be positive")
    return clamp_probability(abs(value) / max_free_energy)


def source_ref_from_utility(utility: UtilityResult) -> ValuationSourceRef:
    """Create a valuation source reference from UtilityResult."""

    if not isinstance(utility, UtilityResult):
        raise TypeError("utility must be UtilityResult")
    return create_source_ref(
        source_id=str(utility.utility_id),
        source_type="UtilityResult",
        confidence=utility.confidence,
        metadata={
            "utility_kind": utility.kind.value,
            "utility_value": utility.value,
            "normalized_utility": utility.normalized_value,
        },
    )


def source_ref_from_entropy(entropy: EntropyResult) -> ValuationSourceRef:
    """Create a valuation source reference from EntropyResult."""

    if not isinstance(entropy, EntropyResult):
        raise TypeError("entropy must be EntropyResult")
    return create_source_ref(
        source_id=str(entropy.entropy_id),
        source_type="EntropyResult",
        confidence=entropy.confidence,
        metadata={
            "entropy_kind": entropy.kind.value,
            "entropy_value": entropy.value,
            "normalized_entropy": entropy.normalized_value,
        },
    )


def source_ref_from_metric_cost(cost: MetricCostResult) -> ValuationSourceRef:
    """Create a valuation source reference from MetricCostResult."""

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


def free_energy_value(
    *,
    utility_value: float | None = None,
    entropy_value: float | None = None,
    cost_value: float | None = None,
    prediction_error: float | None = None,
    policy: FreeEnergyPolicy | None = None,
) -> float:
    """Compute scalar free-energy value."""

    policy = _ensure_policy(policy)
    utility_value = _ensure_optional_finite_number(utility_value, "utility_value")
    entropy_value = _ensure_optional_non_negative_number(
        entropy_value,
        "entropy_value",
    )
    cost_value = _ensure_optional_non_negative_number(cost_value, "cost_value")
    prediction_error = _ensure_optional_non_negative_number(
        prediction_error,
        "prediction_error",
    )

    value = 0.0
    if entropy_value is not None:
        value += policy.entropy_weight * entropy_value
    if cost_value is not None:
        value += policy.cost_weight * cost_value
    if prediction_error is not None:
        value += policy.prediction_error_weight * prediction_error
    if utility_value is not None:
        value -= policy.utility_weight * utility_value
    return ensure_finite_number(value, field_name="free_energy_value")


def free_energy_confidence(
    *,
    utility: UtilityResult | None = None,
    entropy: EntropyResult | None = None,
    costs: tuple[MetricCostResult, ...] | list[MetricCostResult] = (),
    prediction_error_confidence: float | None = None,
    policy: FreeEnergyPolicy | None = None,
) -> float:
    """Compute confidence for free-energy result."""

    policy = _ensure_policy(policy)
    confidence_values: list[float] = []
    if utility is not None:
        if not isinstance(utility, UtilityResult):
            raise TypeError("utility must be UtilityResult or None")
        confidence_values.append(utility.confidence)
    if entropy is not None:
        if not isinstance(entropy, EntropyResult):
            raise TypeError("entropy must be EntropyResult or None")
        confidence_values.append(entropy.confidence)
    for cost in _ensure_costs(costs):
        confidence_values.append(cost.confidence)
    if prediction_error_confidence is not None:
        confidence_values.append(
            ensure_constraint_probability(
                prediction_error_confidence,
                field_name="prediction_error_confidence",
            )
        )
    if not confidence_values:
        return 1.0
    confidence = sum(confidence_values) / len(confidence_values)
    confidence = max(policy.confidence_floor, confidence)
    return ensure_constraint_probability(confidence, field_name="confidence")


def compute_cost_utility_free_energy(
    *,
    utility: UtilityResult | None = None,
    entropy: EntropyResult | None = None,
    costs: tuple[MetricCostResult, ...] | list[MetricCostResult] = (),
    policy: FreeEnergyPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> FreeEnergyResult:
    """Create FreeEnergyResult from utility, entropy, and metric costs."""

    policy = _ensure_policy(policy)
    if utility is not None and not isinstance(utility, UtilityResult):
        raise TypeError("utility must be UtilityResult or None")
    if entropy is not None and not isinstance(entropy, EntropyResult):
        raise TypeError("entropy must be EntropyResult or None")
    checked_costs = _ensure_costs(costs)
    total_cost(checked_costs)
    mean_cost = mean_normalized_cost(checked_costs) if checked_costs else 0.0
    normalized_cost = 0.0 if mean_cost is None else mean_cost

    value = free_energy_value(
        utility_value=None if utility is None else utility.normalized_value,
        entropy_value=None if entropy is None else entropy.normalized_value,
        cost_value=normalized_cost,
        prediction_error=None,
        policy=policy,
    )
    normalized_value = normalize_free_energy_value(
        value=value,
        max_free_energy=policy.max_free_energy,
    )
    confidence = free_energy_confidence(
        utility=utility,
        entropy=entropy,
        costs=checked_costs,
        policy=policy,
    )
    source_refs = _source_refs_from_inputs(
        utility=utility,
        entropy=entropy,
        costs=checked_costs,
    )

    default_metadata: dict[str, object] = {
        "source": "cost_utility_free_energy",
        "has_utility": utility is not None,
        "has_entropy": entropy is not None,
        "cost_count": len(checked_costs),
        "normalized_cost": normalized_cost,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "cost_utility_free_energy"

    return FreeEnergyResult(
        free_energy_id=new_free_energy_id(),
        kind=FreeEnergyKind.COST_UTILITY,
        value=value,
        normalized_value=normalized_value,
        utility_value=None if utility is None else utility.normalized_value,
        entropy_value=None if entropy is None else entropy.normalized_value,
        confidence=confidence,
        source_refs=source_refs,
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def compute_prediction_error_free_energy(
    *,
    prediction_error: float,
    confidence: float = 1.0,
    source_id: str = "prediction_error",
    source_type: str = "prediction_error",
    utility: UtilityResult | None = None,
    entropy: EntropyResult | None = None,
    policy: FreeEnergyPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> FreeEnergyResult:
    """Create FreeEnergyResult from prediction-error-like scalar."""

    policy = _ensure_policy(policy)
    prediction_error = ensure_non_negative_number(
        prediction_error,
        field_name="prediction_error",
    )
    confidence = ensure_constraint_probability(confidence, field_name="confidence")
    source_id = _ensure_non_empty_string(source_id, "source_id")
    source_type = _ensure_non_empty_string(source_type, "source_type")
    if utility is not None and not isinstance(utility, UtilityResult):
        raise TypeError("utility must be UtilityResult or None")
    if entropy is not None and not isinstance(entropy, EntropyResult):
        raise TypeError("entropy must be EntropyResult or None")

    value = free_energy_value(
        utility_value=None if utility is None else utility.normalized_value,
        entropy_value=None if entropy is None else entropy.normalized_value,
        cost_value=None,
        prediction_error=prediction_error,
        policy=policy,
    )
    normalized_value = normalize_free_energy_value(
        value=value,
        max_free_energy=policy.max_free_energy,
    )
    output_confidence = free_energy_confidence(
        utility=utility,
        entropy=entropy,
        prediction_error_confidence=confidence,
        policy=policy,
    )
    source_refs = (
        create_source_ref(
            source_id=source_id,
            source_type=source_type,
            confidence=confidence,
            metadata={"prediction_error": prediction_error},
        ),
    ) + _source_refs_from_inputs(utility=utility, entropy=entropy)

    default_metadata: dict[str, object] = {
        "source": "prediction_error_free_energy",
        "prediction_error": prediction_error,
        "has_utility": utility is not None,
        "has_entropy": entropy is not None,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "prediction_error_free_energy"

    return FreeEnergyResult(
        free_energy_id=new_free_energy_id(),
        kind=FreeEnergyKind.PREDICTION_ERROR,
        value=value,
        normalized_value=normalized_value,
        utility_value=None if utility is None else utility.normalized_value,
        entropy_value=None if entropy is None else entropy.normalized_value,
        confidence=output_confidence,
        source_refs=source_refs,
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def compute_semantic_free_energy(
    *,
    utility: UtilityResult | None = None,
    entropy: EntropyResult | None = None,
    prediction_error: float | None = None,
    confidence: float = 1.0,
    policy: FreeEnergyPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> FreeEnergyResult:
    """Create semantic free-energy descriptor."""

    policy = _ensure_policy(policy)
    if utility is not None and not isinstance(utility, UtilityResult):
        raise TypeError("utility must be UtilityResult or None")
    if entropy is not None and not isinstance(entropy, EntropyResult):
        raise TypeError("entropy must be EntropyResult or None")
    prediction_error = _ensure_optional_non_negative_number(
        prediction_error,
        "prediction_error",
    )
    confidence = ensure_constraint_probability(confidence, field_name="confidence")

    value = free_energy_value(
        utility_value=None if utility is None else utility.normalized_value,
        entropy_value=None if entropy is None else entropy.normalized_value,
        cost_value=None,
        prediction_error=prediction_error,
        policy=policy,
    )
    normalized_value = normalize_free_energy_value(
        value=value,
        max_free_energy=policy.max_free_energy,
    )
    output_confidence = free_energy_confidence(
        utility=utility,
        entropy=entropy,
        prediction_error_confidence=(
            confidence if prediction_error is not None else None
        ),
        policy=policy,
    )
    source_refs = _source_refs_from_inputs(utility=utility, entropy=entropy)
    if prediction_error is not None:
        source_refs = source_refs + (
            create_source_ref(
                source_id="semantic_prediction_error",
                source_type="prediction_error",
                confidence=confidence,
                metadata={"prediction_error": prediction_error},
            ),
        )

    default_metadata: dict[str, object] = {
        "source": "semantic_free_energy",
        "has_utility": utility is not None,
        "has_entropy": entropy is not None,
        "has_prediction_error": prediction_error is not None,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "semantic_free_energy"

    return FreeEnergyResult(
        free_energy_id=new_free_energy_id(),
        kind=FreeEnergyKind.SEMANTIC,
        value=value,
        normalized_value=normalized_value,
        utility_value=None if utility is None else utility.normalized_value,
        entropy_value=None if entropy is None else entropy.normalized_value,
        confidence=output_confidence,
        source_refs=source_refs,
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def combine_free_energy_results(
    free_energies: tuple[FreeEnergyResult, ...] | list[FreeEnergyResult],
    *,
    kind: FreeEnergyKind = FreeEnergyKind.UNKNOWN,
    policy: FreeEnergyPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> FreeEnergyResult:
    """Combine multiple FreeEnergyResult objects into one FreeEnergyResult."""

    _ensure_policy(policy)
    if not isinstance(kind, FreeEnergyKind):
        raise TypeError("kind must be FreeEnergyKind")
    values = tuple(free_energies)
    for result in values:
        if not isinstance(result, FreeEnergyResult):
            raise TypeError("free_energies must contain FreeEnergyResult")

    if kind is FreeEnergyKind.UNKNOWN and values:
        kind = FreeEnergyKind.SEMANTIC

    if not values:
        value = 0.0
        normalized_value = 0.0
        confidence = 1.0
        source_refs: tuple[ValuationSourceRef, ...] = ()
        utility_value = None
        entropy_value = None
    else:
        value = sum(result.value for result in values) / len(values)
        normalized_value = (
            sum(result.normalized_value for result in values) / len(values)
        )
        confidence = sum(result.confidence for result in values) / len(values)
        refs: list[ValuationSourceRef] = []
        utility_values: list[float] = []
        entropy_values: list[float] = []
        for result in values:
            refs.extend(result.source_refs)
            if result.utility_value is not None:
                utility_values.append(result.utility_value)
            if result.entropy_value is not None:
                entropy_values.append(result.entropy_value)
        source_refs = tuple(refs)
        utility_value = (
            None
            if not utility_values
            else sum(utility_values) / len(utility_values)
        )
        entropy_value = (
            None
            if not entropy_values
            else sum(entropy_values) / len(entropy_values)
        )

    default_metadata: dict[str, object] = {
        "source": "combined_free_energy",
        "free_energy_count": len(values),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "combined_free_energy"

    return FreeEnergyResult(
        free_energy_id=new_free_energy_id(),
        kind=kind,
        value=value,
        normalized_value=ensure_constraint_probability(
            normalized_value,
            field_name="normalized_value",
        ),
        utility_value=utility_value,
        entropy_value=entropy_value,
        confidence=ensure_constraint_probability(confidence, field_name="confidence"),
        source_refs=source_refs,
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def free_energy_is_low(
    free_energy: FreeEnergyResult,
    *,
    threshold: float = 0.35,
) -> bool:
    """Return True when free energy is low."""

    if not isinstance(free_energy, FreeEnergyResult):
        raise TypeError("free_energy must be FreeEnergyResult")
    threshold = ensure_constraint_probability(threshold, field_name="threshold")
    return free_energy.normalized_value <= threshold


def free_energy_summary(
    free_energies: tuple[FreeEnergyResult, ...] | list[FreeEnergyResult],
) -> dict[str, object]:
    """Summarize free-energy results."""

    values = tuple(free_energies)
    for result in values:
        if not isinstance(result, FreeEnergyResult):
            raise TypeError("free_energies must contain FreeEnergyResult")
    count = len(values)
    kinds: dict[str, int] = {}
    for result in values:
        kinds[result.kind.value] = kinds.get(result.kind.value, 0) + 1

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
        "low_free_energy_count": sum(
            1 for result in values if free_energy_is_low(result)
        ),
    }


def _ensure_policy(policy: FreeEnergyPolicy | None) -> FreeEnergyPolicy:
    if policy is None:
        return FreeEnergyPolicy()
    if not isinstance(policy, FreeEnergyPolicy):
        raise TypeError("policy must be FreeEnergyPolicy")
    return policy


def _ensure_costs(
    costs: tuple[MetricCostResult, ...] | list[MetricCostResult],
) -> tuple[MetricCostResult, ...]:
    values = tuple(costs)
    for cost in values:
        if not isinstance(cost, MetricCostResult):
            raise TypeError("costs must contain MetricCostResult")
    return values


def _source_refs_from_inputs(
    *,
    utility: UtilityResult | None = None,
    entropy: EntropyResult | None = None,
    costs: tuple[MetricCostResult, ...] | list[MetricCostResult] = (),
) -> tuple[ValuationSourceRef, ...]:
    refs: list[ValuationSourceRef] = []
    if utility is not None:
        refs.append(source_ref_from_utility(utility))
    if entropy is not None:
        refs.append(source_ref_from_entropy(entropy))
    refs.extend(source_ref_from_metric_cost(cost) for cost in _ensure_costs(costs))
    return tuple(refs)


def _ensure_optional_finite_number(
    value: float | None,
    field_name: str,
) -> float | None:
    if value is None:
        return None
    return ensure_finite_number(value, field_name=field_name)


def _ensure_optional_non_negative_number(
    value: float | None,
    field_name: str,
) -> float | None:
    if value is None:
        return None
    return ensure_non_negative_number(value, field_name=field_name)


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
    "FreeEnergyPolicy",
    "clamp_probability",
    "normalize_free_energy_value",
    "source_ref_from_utility",
    "source_ref_from_entropy",
    "source_ref_from_metric_cost",
    "free_energy_value",
    "free_energy_confidence",
    "compute_cost_utility_free_energy",
    "compute_prediction_error_free_energy",
    "compute_semantic_free_energy",
    "combine_free_energy_results",
    "free_energy_is_low",
    "free_energy_summary",
]
