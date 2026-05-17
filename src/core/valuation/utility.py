"""Deterministic utility computation utilities for CEBA Core Valuation."""

from __future__ import annotations

from dataclasses import dataclass

from ceba.core.dynamics.stability import (
    StabilityReport,
    stability_report_is_blocked,
    stability_report_is_passed,
)
from ceba.core.metrics.schema import (
    ensure_constraint_probability,
    ensure_finite_number,
    ensure_non_negative_number,
    utc_now,
)
from ceba.core.regulation.update_policy import (
    UpdateDecision,
    update_decision_allows_apply,
)
from ceba.core.telos.alignment import alignment_is_acceptable, best_alignment
from ceba.core.telos.schema import TelosAlignmentResult
from ceba.core.valuation.schema import (
    UtilityKind,
    UtilityResult,
    ValuationSourceRef,
    create_source_ref,
    new_utility_id,
)


@dataclass(frozen=True)
class UtilityPolicy:
    """Stable policy for utility computation."""

    alignment_weight: float = 1.0
    stability_weight: float = 1.0
    update_weight: float = 1.0
    source_confidence_weight: float = 1.0
    min_useful_score: float = 0.35
    max_utility: float = 1.0

    def __post_init__(self) -> None:
        alignment_weight = ensure_non_negative_number(
            self.alignment_weight,
            field_name="alignment_weight",
        )
        stability_weight = ensure_non_negative_number(
            self.stability_weight,
            field_name="stability_weight",
        )
        update_weight = ensure_non_negative_number(
            self.update_weight,
            field_name="update_weight",
        )
        source_confidence_weight = ensure_non_negative_number(
            self.source_confidence_weight,
            field_name="source_confidence_weight",
        )
        min_useful_score = ensure_constraint_probability(
            self.min_useful_score,
            field_name="min_useful_score",
        )
        max_utility = ensure_finite_number(
            self.max_utility,
            field_name="max_utility",
        )
        if max_utility <= 0.0:
            raise ValueError("max_utility must be positive")

        object.__setattr__(self, "alignment_weight", alignment_weight)
        object.__setattr__(self, "stability_weight", stability_weight)
        object.__setattr__(self, "update_weight", update_weight)
        object.__setattr__(
            self,
            "source_confidence_weight",
            source_confidence_weight,
        )
        object.__setattr__(self, "min_useful_score", min_useful_score)
        object.__setattr__(self, "max_utility", max_utility)


def clamp_probability(value: float) -> float:
    """Clamp a finite number into the probability range."""

    value = ensure_finite_number(value, field_name="value")
    return min(max(value, 0.0), 1.0)


def normalize_utility_value(
    *,
    value: float,
    max_utility: float = 1.0,
) -> float:
    """Normalize a raw utility value."""

    value = ensure_finite_number(value, field_name="value")
    max_utility = ensure_finite_number(max_utility, field_name="max_utility")
    if max_utility <= 0.0:
        raise ValueError("max_utility must be positive")
    return clamp_probability(value / max_utility)


def source_refs_confidence(
    source_refs: tuple[ValuationSourceRef, ...] | list[ValuationSourceRef],
) -> float:
    """Return mean confidence for valuation source references."""

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


def source_ref_from_alignment(result: TelosAlignmentResult) -> ValuationSourceRef:
    """Create a valuation source reference from a telos alignment result."""

    if not isinstance(result, TelosAlignmentResult):
        raise TypeError("result must be TelosAlignmentResult")
    return create_source_ref(
        source_id=str(result.alignment_id),
        source_type="TelosAlignmentResult",
        confidence=result.confidence,
        metadata={
            "alignment_status": result.status.value,
            "alignment_score": result.score,
        },
    )


def source_ref_from_update_decision(decision: UpdateDecision) -> ValuationSourceRef:
    """Create a valuation source reference from an update decision."""

    if not isinstance(decision, UpdateDecision):
        raise TypeError("decision must be UpdateDecision")
    return create_source_ref(
        source_id=decision.reason_code,
        source_type="UpdateDecision",
        confidence=decision.confidence,
        metadata={
            "action": decision.action.value,
            "status": decision.status.value,
            "allowed": decision.allowed,
        },
    )


def source_ref_from_stability_report(report: StabilityReport) -> ValuationSourceRef:
    """Create a valuation source reference from a stability report."""

    if not isinstance(report, StabilityReport):
        raise TypeError("report must be StabilityReport")
    source_id = report.metadata.get("checked_object_type", "stability_report")
    if not isinstance(source_id, str) or source_id == "":
        source_id = "stability_report"
    return create_source_ref(
        source_id=source_id,
        source_type="StabilityReport",
        confidence=report.score,
        metadata={
            "status": report.status.value,
            "risk": report.risk.value,
            "score": report.score,
        },
    )


def utility_from_alignment_score(
    *,
    alignment_score: float,
    confidence: float = 1.0,
    policy: UtilityPolicy | None = None,
) -> tuple[float, float, float]:
    """Compute raw utility value, normalized value, and confidence."""

    policy = _ensure_policy(policy)
    alignment_score = ensure_constraint_probability(
        alignment_score,
        field_name="alignment_score",
    )
    confidence = ensure_constraint_probability(confidence, field_name="confidence")
    value = alignment_score * policy.alignment_weight
    normalized_value = normalize_utility_value(
        value=value,
        max_utility=policy.max_utility,
    )
    return value, normalized_value, confidence


def compute_alignment_utility(
    *,
    alignment: TelosAlignmentResult,
    policy: UtilityPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> UtilityResult:
    """Create UtilityResult from TelosAlignmentResult."""

    if not isinstance(alignment, TelosAlignmentResult):
        raise TypeError("alignment must be TelosAlignmentResult")
    policy = _ensure_policy(policy)
    value, normalized_value, confidence = utility_from_alignment_score(
        alignment_score=alignment.score,
        confidence=alignment.confidence,
        policy=policy,
    )
    default_metadata: dict[str, object] = {
        "source": "alignment_utility",
        "alignment_status": alignment.status.value,
        "acceptable": alignment_is_acceptable(alignment),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "alignment_utility"

    return UtilityResult(
        utility_id=new_utility_id(),
        kind=UtilityKind.TELOS_ALIGNMENT,
        value=value,
        normalized_value=normalized_value,
        confidence=confidence,
        source_refs=(source_ref_from_alignment(alignment),),
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def compute_best_alignment_utility(
    *,
    alignments: tuple[TelosAlignmentResult, ...] | list[TelosAlignmentResult],
    policy: UtilityPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> UtilityResult:
    """Create UtilityResult from the best alignment result."""

    policy = _ensure_policy(policy)
    values = tuple(alignments)
    for alignment in values:
        if not isinstance(alignment, TelosAlignmentResult):
            raise TypeError("alignments must contain TelosAlignmentResult")

    default_metadata: dict[str, object] = {
        "source": "best_alignment_utility",
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "best_alignment_utility"

    if not values:
        return UtilityResult(
            utility_id=new_utility_id(),
            kind=UtilityKind.TELOS_ALIGNMENT,
            value=0.0,
            normalized_value=0.0,
            confidence=1.0,
            source_refs=(),
            created_at=utc_now(),
            metadata=merged_metadata,
        )

    alignment = best_alignment(values)
    if alignment is None:
        value = 0.0
        normalized_value = 0.0
        confidence = 1.0
        source_refs: tuple[ValuationSourceRef, ...] = ()
    else:
        value, normalized_value, confidence = utility_from_alignment_score(
            alignment_score=alignment.score,
            confidence=alignment.confidence,
            policy=policy,
        )
        source_refs = (source_ref_from_alignment(alignment),)

    return UtilityResult(
        utility_id=new_utility_id(),
        kind=UtilityKind.TELOS_ALIGNMENT,
        value=value,
        normalized_value=normalized_value,
        confidence=confidence,
        source_refs=source_refs,
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def compute_stability_utility(
    *,
    stability: StabilityReport,
    policy: UtilityPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> UtilityResult:
    """Create UtilityResult from StabilityReport."""

    if not isinstance(stability, StabilityReport):
        raise TypeError("stability must be StabilityReport")
    policy = _ensure_policy(policy)
    base = 0.0 if stability_report_is_blocked(stability) else stability.score
    value = base * policy.stability_weight
    normalized_value = normalize_utility_value(
        value=value,
        max_utility=policy.max_utility,
    )
    default_metadata: dict[str, object] = {
        "source": "stability_utility",
        "stability_status": stability.status.value,
        "stability_passed": stability_report_is_passed(stability),
        "stability_blocked": stability_report_is_blocked(stability),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "stability_utility"

    return UtilityResult(
        utility_id=new_utility_id(),
        kind=UtilityKind.STABILITY_GAIN,
        value=value,
        normalized_value=normalized_value,
        confidence=stability.score,
        source_refs=(source_ref_from_stability_report(stability),),
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def compute_update_decision_utility(
    *,
    decision: UpdateDecision,
    policy: UtilityPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> UtilityResult:
    """Create UtilityResult from UpdateDecision."""

    if not isinstance(decision, UpdateDecision):
        raise TypeError("decision must be UpdateDecision")
    policy = _ensure_policy(policy)
    allowed = update_decision_allows_apply(decision)
    base = decision.confidence if allowed else 0.0
    value = base * policy.update_weight
    normalized_value = normalize_utility_value(
        value=value,
        max_utility=policy.max_utility,
    )
    default_metadata: dict[str, object] = {
        "source": "update_decision_utility",
        "action": decision.action.value,
        "status": decision.status.value,
        "allowed": decision.allowed,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "update_decision_utility"

    return UtilityResult(
        utility_id=new_utility_id(),
        kind=UtilityKind.CONSTRAINT_SATISFACTION,
        value=value,
        normalized_value=normalized_value,
        confidence=decision.confidence,
        source_refs=(source_ref_from_update_decision(decision),),
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def combine_utility_results(
    utilities: tuple[UtilityResult, ...] | list[UtilityResult],
    *,
    kind: UtilityKind = UtilityKind.UNKNOWN,
    policy: UtilityPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> UtilityResult:
    """Combine multiple UtilityResult objects into one UtilityResult."""

    _ensure_policy(policy)
    if not isinstance(kind, UtilityKind):
        raise TypeError("kind must be UtilityKind")
    values = tuple(utilities)
    for utility in values:
        if not isinstance(utility, UtilityResult):
            raise TypeError("utilities must contain UtilityResult")

    if kind is UtilityKind.UNKNOWN and values:
        kind = UtilityKind.COMPOSITE

    if not values:
        value = 0.0
        normalized_value = 0.0
        confidence = 1.0
        source_refs: tuple[ValuationSourceRef, ...] = ()
    else:
        value = sum(utility.value for utility in values) / len(values)
        normalized_value = (
            sum(utility.normalized_value for utility in values) / len(values)
        )
        confidence = sum(utility.confidence for utility in values) / len(values)
        refs: list[ValuationSourceRef] = []
        for utility in values:
            refs.extend(utility.source_refs)
        source_refs = tuple(refs)
    default_metadata: dict[str, object] = {
        "source": "combined_utility",
        "utility_count": len(values),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "combined_utility"

    return UtilityResult(
        utility_id=new_utility_id(),
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


def utility_is_useful(
    utility: UtilityResult,
    *,
    policy: UtilityPolicy | None = None,
) -> bool:
    """Return True when utility is useful under policy."""

    if not isinstance(utility, UtilityResult):
        raise TypeError("utility must be UtilityResult")
    policy = _ensure_policy(policy)
    return (
        utility.normalized_value >= policy.min_useful_score
        and utility.confidence > 0.0
    )


def utility_summary(
    utilities: tuple[UtilityResult, ...] | list[UtilityResult],
) -> dict[str, object]:
    """Summarize utility results."""

    values = tuple(utilities)
    for utility in values:
        if not isinstance(utility, UtilityResult):
            raise TypeError("utilities must contain UtilityResult")
    count = len(values)
    kinds: dict[str, int] = {}
    for utility in values:
        kinds[utility.kind.value] = kinds.get(utility.kind.value, 0) + 1

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
        "useful_count": sum(1 for utility in values if utility_is_useful(utility)),
    }


def _ensure_policy(policy: UtilityPolicy | None) -> UtilityPolicy:
    if policy is None:
        return UtilityPolicy()
    if not isinstance(policy, UtilityPolicy):
        raise TypeError("policy must be UtilityPolicy")
    return policy


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
    "UtilityPolicy",
    "clamp_probability",
    "normalize_utility_value",
    "source_refs_confidence",
    "source_ref_from_alignment",
    "source_ref_from_update_decision",
    "source_ref_from_stability_report",
    "utility_from_alignment_score",
    "compute_alignment_utility",
    "compute_best_alignment_utility",
    "compute_stability_utility",
    "compute_update_decision_utility",
    "combine_utility_results",
    "utility_is_useful",
    "utility_summary",
]
