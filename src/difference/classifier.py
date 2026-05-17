"""Deterministic classification utilities for CEBA difference processing."""

from __future__ import annotations

from dataclasses import dataclass, field

from ois.difference.errors import (
    DifferenceClassificationError,
    UnsupportedDifferenceKindError,
    build_error_context,
)
from ois.difference.schema import (
    DifferenceCandidate,
    DifferenceClassification,
    DifferenceKind,
    DifferenceSeverity,
    DifferenceStatus,
    ensure_non_negative,
    ensure_probability,
)


DEFAULT_LOW_THRESHOLD = 0.25
DEFAULT_MEDIUM_THRESHOLD = 0.50
DEFAULT_HIGH_THRESHOLD = 0.75
DEFAULT_CRITICAL_THRESHOLD = 0.90


@dataclass(frozen=True, slots=True)
class ClassificationThresholds:
    """Stable threshold configuration for severity assignment."""

    low: float = DEFAULT_LOW_THRESHOLD
    medium: float = DEFAULT_MEDIUM_THRESHOLD
    high: float = DEFAULT_HIGH_THRESHOLD
    critical: float = DEFAULT_CRITICAL_THRESHOLD

    def __post_init__(self) -> None:
        object.__setattr__(self, "low", ensure_probability(self.low, "low"))
        object.__setattr__(self, "medium", ensure_probability(self.medium, "medium"))
        object.__setattr__(self, "high", ensure_probability(self.high, "high"))
        object.__setattr__(self, "critical", ensure_probability(self.critical, "critical"))

        if not self.low <= self.medium <= self.high <= self.critical:
            raise ValueError("threshold ordering must be low <= medium <= high <= critical")


@dataclass(frozen=True, slots=True)
class ClassificationPolicy:
    """Deterministic policy for technical severity assignment."""

    thresholds: ClassificationThresholds = field(default_factory=ClassificationThresholds)
    contradiction_boost: float = 0.15
    missing_extra_boost: float = 0.10
    structural_boost: float = 0.05
    low_confidence_penalty: float = 0.10

    def __post_init__(self) -> None:
        if not isinstance(self.thresholds, ClassificationThresholds):
            raise TypeError("thresholds must be ClassificationThresholds")
        object.__setattr__(
            self,
            "contradiction_boost",
            ensure_non_negative(self.contradiction_boost, "contradiction_boost"),
        )
        object.__setattr__(
            self,
            "missing_extra_boost",
            ensure_non_negative(self.missing_extra_boost, "missing_extra_boost"),
        )
        object.__setattr__(
            self,
            "structural_boost",
            ensure_non_negative(self.structural_boost, "structural_boost"),
        )
        object.__setattr__(
            self,
            "low_confidence_penalty",
            ensure_non_negative(self.low_confidence_penalty, "low_confidence_penalty"),
        )


def compute_base_score(candidate: DifferenceCandidate) -> float:
    """Compute a base classification score from candidate fields."""

    if not isinstance(candidate, DifferenceCandidate):
        raise TypeError("candidate must be a DifferenceCandidate")

    confidence = ensure_probability(candidate.confidence, "confidence")
    if candidate.magnitude is not None:
        magnitude = ensure_non_negative(candidate.magnitude, "magnitude")
        if magnitude is None:
            base = confidence
        else:
            base = min(float(magnitude), 1.0)
    else:
        base = float(confidence)

    return _clamp_probability(base)


def apply_kind_modifiers(
    *,
    score: float,
    kind: DifferenceKind,
    policy: ClassificationPolicy,
) -> float:
    """Apply deterministic technical modifiers based on DifferenceKind."""

    score = ensure_probability(score, "score")
    if not isinstance(policy, ClassificationPolicy):
        raise TypeError("policy must be ClassificationPolicy")
    if not isinstance(kind, DifferenceKind):
        raise UnsupportedDifferenceKindError(message="unsupported difference kind")

    modifier = 0.0
    if kind is DifferenceKind.CONTRADICTION:
        modifier = policy.contradiction_boost
    elif kind in (DifferenceKind.MISSING, DifferenceKind.EXTRA):
        modifier = policy.missing_extra_boost
    elif kind is DifferenceKind.STRUCTURAL:
        modifier = policy.structural_boost

    return _clamp_probability(score + modifier)


def apply_confidence_penalty(
    *,
    score: float,
    confidence: float,
    policy: ClassificationPolicy,
) -> float:
    """Reduce severity score when candidate confidence is weak."""

    score = ensure_probability(score, "score")
    confidence = ensure_probability(confidence, "confidence")
    if not isinstance(policy, ClassificationPolicy):
        raise TypeError("policy must be ClassificationPolicy")

    if confidence < policy.thresholds.low:
        score -= policy.low_confidence_penalty

    return _clamp_probability(score)


def severity_from_score(
    score: float,
    thresholds: ClassificationThresholds,
) -> DifferenceSeverity:
    """Map a technical score to DifferenceSeverity."""

    score = ensure_probability(score, "score")
    if not isinstance(thresholds, ClassificationThresholds):
        raise TypeError("thresholds must be ClassificationThresholds")

    if score >= thresholds.critical:
        return DifferenceSeverity.CRITICAL
    if score >= thresholds.high:
        return DifferenceSeverity.HIGH
    if score >= thresholds.medium:
        return DifferenceSeverity.MEDIUM
    if score >= thresholds.low:
        return DifferenceSeverity.LOW
    return DifferenceSeverity.TRACE


def rationale_code_for(
    *,
    kind: DifferenceKind,
    severity: DifferenceSeverity,
) -> str:
    """Return a stable machine-readable rationale code."""

    if not isinstance(kind, DifferenceKind):
        raise UnsupportedDifferenceKindError(message="unsupported difference kind")
    if not isinstance(severity, DifferenceSeverity):
        raise TypeError("severity must be DifferenceSeverity")
    return f"KIND_{kind.value}__SEVERITY_{severity.value}"


def classify_candidate(
    candidate: DifferenceCandidate,
    *,
    policy: ClassificationPolicy | None = None,
) -> DifferenceClassification:
    """Classify one normalized DifferenceCandidate."""

    difference_id: str | None = None
    try:
        if isinstance(candidate, DifferenceCandidate):
            difference_id = str(candidate.difference_id)
        else:
            raise TypeError("candidate must be a DifferenceCandidate")

        if candidate.status is not DifferenceStatus.NORMALIZED:
            raise ValueError("candidate status must be NORMALIZED")

        resolved_policy = ClassificationPolicy() if policy is None else policy
        if not isinstance(resolved_policy, ClassificationPolicy):
            raise TypeError("policy must be ClassificationPolicy or None")

        base_score = compute_base_score(candidate)
        modified_score = apply_kind_modifiers(
            score=base_score,
            kind=candidate.kind,
            policy=resolved_policy,
        )
        final_score = apply_confidence_penalty(
            score=modified_score,
            confidence=candidate.confidence,
            policy=resolved_policy,
        )
        severity = severity_from_score(final_score, resolved_policy.thresholds)

        return DifferenceClassification(
            difference_id=candidate.difference_id,
            kind=candidate.kind,
            severity=severity,
            confidence=candidate.confidence,
            rationale_code=rationale_code_for(kind=candidate.kind, severity=severity),
            metadata={
                "classification_score": final_score,
                "base_score": base_score,
                "policy": "deterministic_v1",
            },
        )
    except UnsupportedDifferenceKindError:
        raise
    except DifferenceClassificationError:
        raise
    except Exception as exc:
        raise DifferenceClassificationError(
            message="failed to classify difference candidate",
            context=build_error_context(
                operation="classify_candidate",
                difference_id=difference_id,
            ),
            cause=exc,
        ) from exc


def classify_candidates(
    candidates: tuple[DifferenceCandidate, ...] | list[DifferenceCandidate],
    *,
    policy: ClassificationPolicy | None = None,
) -> tuple[DifferenceClassification, ...]:
    """Classify multiple normalized candidates in order."""

    try:
        if not isinstance(candidates, (tuple, list)):
            raise TypeError("candidates must be a tuple or list of DifferenceCandidate")
        return tuple(classify_candidate(candidate, policy=policy) for candidate in candidates)
    except UnsupportedDifferenceKindError:
        raise
    except DifferenceClassificationError as exc:
        raise DifferenceClassificationError(
            message="failed to classify candidates",
            context=build_error_context(operation="classify_candidates"),
            cause=exc,
        ) from exc
    except Exception as exc:
        raise DifferenceClassificationError(
            message="failed to classify candidates",
            context=build_error_context(operation="classify_candidates"),
            cause=exc,
        ) from exc


def candidate_is_classifiable(candidate: DifferenceCandidate) -> bool:
    """Return whether a candidate can be classified."""

    if not isinstance(candidate, DifferenceCandidate):
        raise TypeError("candidate must be a DifferenceCandidate")
    return candidate.status is DifferenceStatus.NORMALIZED


def _clamp_probability(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


__all__ = [
    "DEFAULT_LOW_THRESHOLD",
    "DEFAULT_MEDIUM_THRESHOLD",
    "DEFAULT_HIGH_THRESHOLD",
    "DEFAULT_CRITICAL_THRESHOLD",
    "ClassificationThresholds",
    "ClassificationPolicy",
    "compute_base_score",
    "apply_kind_modifiers",
    "apply_confidence_penalty",
    "severity_from_score",
    "rationale_code_for",
    "classify_candidate",
    "classify_candidates",
    "candidate_is_classifiable",
]
