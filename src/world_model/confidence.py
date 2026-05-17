"""Deterministic confidence utilities for CEBA World Model."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from ceba.world_model.schema import (
    ExpectedObservation,
    ExpectedObservationStatus,
    PredictionInput,
    PredictionResult,
    PredictionStatus,
    WorldModelEvidence,
    ensure_non_empty_string,
    ensure_probability,
)


__all__ = [
    "ConfidencePolicy",
    "ConfidenceScore",
    "ConfidenceReport",
    "WorldModelConfidenceEvaluator",
    "evaluate_prediction_input_confidence",
    "evaluate_expected_observation_confidence",
    "evaluate_prediction_result_confidence",
]


@dataclass(frozen=True)
class ConfidencePolicy:
    """Configuration object controlling deterministic confidence evaluation."""

    evidence_weight: float = 0.35
    prediction_weight: float = 0.35
    expected_observation_weight: float = 0.30
    uncertainty_penalty_weight: float = 0.25
    min_usable_confidence: float = 0.50
    max_usable_uncertainty: float = 0.75

    def __post_init__(self) -> None:
        evidence_weight = _ensure_non_negative_number(
            self.evidence_weight,
            "evidence_weight",
        )
        prediction_weight = _ensure_non_negative_number(
            self.prediction_weight,
            "prediction_weight",
        )
        expected_weight = _ensure_non_negative_number(
            self.expected_observation_weight,
            "expected_observation_weight",
        )
        uncertainty_weight = _ensure_non_negative_number(
            self.uncertainty_penalty_weight,
            "uncertainty_penalty_weight",
        )
        if evidence_weight + prediction_weight + expected_weight <= 0.0:
            raise ValueError("confidence weights must sum to > 0.0")
        min_usable = ensure_probability(
            self.min_usable_confidence,
            "min_usable_confidence",
        )
        max_uncertainty = ensure_probability(
            self.max_usable_uncertainty,
            "max_usable_uncertainty",
        )
        object.__setattr__(self, "evidence_weight", evidence_weight)
        object.__setattr__(self, "prediction_weight", prediction_weight)
        object.__setattr__(
            self,
            "expected_observation_weight",
            expected_weight,
        )
        object.__setattr__(
            self,
            "uncertainty_penalty_weight",
            uncertainty_weight,
        )
        object.__setattr__(self, "min_usable_confidence", min_usable)
        object.__setattr__(self, "max_usable_uncertainty", max_uncertainty)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "evidence_weight": self.evidence_weight,
            "prediction_weight": self.prediction_weight,
            "expected_observation_weight": self.expected_observation_weight,
            "uncertainty_penalty_weight": self.uncertainty_penalty_weight,
            "min_usable_confidence": self.min_usable_confidence,
            "max_usable_uncertainty": self.max_usable_uncertainty,
        }


@dataclass(frozen=True)
class ConfidenceScore:
    """Single confidence score descriptor."""

    score_name: str
    score: float
    weight: float
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        score_name = ensure_non_empty_string(self.score_name, "score_name")
        score = ensure_probability(self.score, "score")
        weight = _ensure_non_negative_number(self.weight, "weight")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "score_name", score_name)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "score_name": self.score_name,
            "score": self.score,
            "weight": self.weight,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ConfidenceReport:
    """Immutable result object returned by confidence evaluation."""

    score: float
    usable: bool
    scores: tuple[ConfidenceScore, ...]
    reason: str
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        score = ensure_probability(self.score, "score")
        if not isinstance(self.usable, bool):
            raise TypeError("usable must be bool")
        scores = _ensure_confidence_scores(self.scores)
        reason = ensure_non_empty_string(self.reason, "reason")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "score": self.score,
            "usable": self.usable,
            "scores": tuple(score.to_dict() for score in self.scores),
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }

    def score_count(self) -> int:
        """Return score descriptor count."""

        return len(self.scores)


class WorldModelConfidenceEvaluator:
    """Deterministic confidence evaluator for world model runtime objects."""

    def __init__(self, policy: ConfidencePolicy | None = None) -> None:
        self._policy = ConfidencePolicy() if policy is None else policy
        if not isinstance(self._policy, ConfidencePolicy):
            raise TypeError("policy must be ConfidencePolicy")

    def evaluate_prediction_input(
        self,
        *,
        prediction_input: PredictionInput,
    ) -> ConfidenceReport:
        """Evaluate confidence for one prediction input."""

        if not isinstance(prediction_input, PredictionInput):
            raise TypeError("prediction_input must be PredictionInput")
        evidence_score = _evidence_confidence(prediction_input.evidence)
        score = ConfidenceScore(
            score_name="evidence_confidence",
            score=evidence_score,
            weight=self._policy.evidence_weight,
        )
        usable = evidence_score >= self._policy.min_usable_confidence
        reason = (
            "prediction input confidence usable"
            if usable
            else "prediction input confidence below minimum"
        )
        return ConfidenceReport(
            score=evidence_score,
            usable=usable,
            scores=(score,),
            reason=reason,
            metadata={
                "source": "world_model_confidence_evaluator",
                "object_type": "prediction_input",
                "prediction_input_id": str(prediction_input.input_id),
                "evidence_count": len(prediction_input.evidence),
                "policy": self._policy.to_dict(),
            },
        )

    def evaluate_expected_observation(
        self,
        *,
        expected_observation: ExpectedObservation,
    ) -> ConfidenceReport:
        """Evaluate confidence for one expected observation."""

        if not isinstance(expected_observation, ExpectedObservation):
            raise TypeError(
                "expected_observation must be ExpectedObservation"
            )
        confidence_score = expected_observation.confidence
        uncertainty_score = _clamp_probability(
            1.0 - expected_observation.uncertainty
        )
        status_score = _expected_observation_status_score(
            expected_observation.status
        )
        scores = (
            ConfidenceScore(
                score_name="expected_observation_confidence",
                score=confidence_score,
                weight=self._policy.expected_observation_weight,
            ),
            ConfidenceScore(
                score_name="uncertainty_confidence",
                score=uncertainty_score,
                weight=self._policy.uncertainty_penalty_weight,
            ),
            ConfidenceScore(
                score_name="expected_observation_status",
                score=status_score,
                weight=self._policy.prediction_weight,
            ),
        )
        final_score = _weighted_average(scores)
        usable, reason = self._expected_observation_usability(
            expected_observation=expected_observation,
            score=final_score,
        )
        return ConfidenceReport(
            score=final_score,
            usable=usable,
            scores=scores,
            reason=reason,
            metadata={
                "source": "world_model_confidence_evaluator",
                "object_type": "expected_observation",
                "expected_id": str(expected_observation.expected_id),
                "prediction_input_id": str(
                    expected_observation.prediction_input_id
                ),
                "status": expected_observation.status.value,
                "policy": self._policy.to_dict(),
            },
        )

    def evaluate_prediction_result(
        self,
        *,
        prediction_result: PredictionResult,
    ) -> ConfidenceReport:
        """Evaluate confidence for one prediction result."""

        if not isinstance(prediction_result, PredictionResult):
            raise TypeError("prediction_result must be PredictionResult")
        prediction_score = prediction_result.confidence
        evidence_score = _evidence_confidence(prediction_result.evidence)
        expected_score = 0.0
        if prediction_result.expected_observation is not None:
            expected_score = self.evaluate_expected_observation(
                expected_observation=prediction_result.expected_observation
            ).score
        status_score = _prediction_status_score(prediction_result.status)
        scores = (
            ConfidenceScore(
                score_name="evidence_confidence",
                score=evidence_score,
                weight=self._policy.evidence_weight,
            ),
            ConfidenceScore(
                score_name="prediction_confidence",
                score=prediction_score,
                weight=self._policy.prediction_weight,
            ),
            ConfidenceScore(
                score_name="expected_observation_confidence",
                score=expected_score,
                weight=self._policy.expected_observation_weight,
            ),
            ConfidenceScore(
                score_name="prediction_status",
                score=status_score,
                weight=self._policy.prediction_weight,
            ),
        )
        final_score = _weighted_average(scores)
        usable, reason = self._prediction_result_usability(
            prediction_result=prediction_result,
            score=final_score,
        )
        return ConfidenceReport(
            score=final_score,
            usable=usable,
            scores=scores,
            reason=reason,
            metadata={
                "source": "world_model_confidence_evaluator",
                "object_type": "prediction_result",
                "result_id": str(prediction_result.result_id),
                "prediction_input_id": str(
                    prediction_result.prediction_input.input_id
                ),
                "status": prediction_result.status.value,
                "has_expected_observation": (
                    prediction_result.expected_observation is not None
                ),
                "policy": self._policy.to_dict(),
            },
        )

    def _expected_observation_usability(
        self,
        *,
        expected_observation: ExpectedObservation,
        score: float,
    ) -> tuple[bool, str]:
        if expected_observation.status in (
            ExpectedObservationStatus.INVALID,
            ExpectedObservationStatus.FAILED,
        ):
            return False, "expected observation status unusable"
        if expected_observation.uncertainty > self._policy.max_usable_uncertainty:
            return False, "expected observation uncertainty above maximum"
        if score < self._policy.min_usable_confidence:
            return False, "expected observation confidence below minimum"
        return True, "expected observation confidence usable"

    def _prediction_result_usability(
        self,
        *,
        prediction_result: PredictionResult,
        score: float,
    ) -> tuple[bool, str]:
        if prediction_result.status in (
            PredictionStatus.INVALID,
            PredictionStatus.FAILED,
        ):
            return False, "prediction result status unusable"
        if prediction_result.expected_observation is None:
            return False, "prediction result missing expected observation"
        if score < self._policy.min_usable_confidence:
            return False, "prediction result confidence below minimum"
        return True, "prediction result confidence usable"


def evaluate_prediction_input_confidence(
    *,
    prediction_input: PredictionInput,
    policy: ConfidencePolicy | None = None,
) -> ConfidenceReport:
    """Evaluate confidence for one prediction input."""

    evaluator = WorldModelConfidenceEvaluator(policy)
    return evaluator.evaluate_prediction_input(prediction_input=prediction_input)


def evaluate_expected_observation_confidence(
    *,
    expected_observation: ExpectedObservation,
    policy: ConfidencePolicy | None = None,
) -> ConfidenceReport:
    """Evaluate confidence for one expected observation."""

    evaluator = WorldModelConfidenceEvaluator(policy)
    return evaluator.evaluate_expected_observation(
        expected_observation=expected_observation
    )


def evaluate_prediction_result_confidence(
    *,
    prediction_result: PredictionResult,
    policy: ConfidencePolicy | None = None,
) -> ConfidenceReport:
    """Evaluate confidence for one prediction result."""

    evaluator = WorldModelConfidenceEvaluator(policy)
    return evaluator.evaluate_prediction_result(prediction_result=prediction_result)


def _evidence_confidence(
    evidence: tuple[WorldModelEvidence, ...] | list[WorldModelEvidence],
) -> float:
    if not isinstance(evidence, (tuple, list)):
        raise TypeError("evidence must be tuple or list")
    evidence_values = tuple(evidence)
    for item in evidence_values:
        if not isinstance(item, WorldModelEvidence):
            raise TypeError("evidence must contain WorldModelEvidence")
    if not evidence_values:
        return 1.0
    total = sum(item.confidence * item.relevance for item in evidence_values)
    return _clamp_probability(total / len(evidence_values))


def _expected_observation_status_score(
    status: ExpectedObservationStatus,
) -> float:
    if status is ExpectedObservationStatus.EXPECTED:
        return 1.0
    if status is ExpectedObservationStatus.UNCERTAIN:
        return 0.70
    if status is ExpectedObservationStatus.DEGRADED:
        return 0.40
    return 0.0


def _prediction_status_score(status: PredictionStatus) -> float:
    if status is PredictionStatus.PREDICTED:
        return 1.0
    if status is PredictionStatus.LOW_CONFIDENCE:
        return 0.60
    if status is PredictionStatus.PENDING:
        return 0.25
    return 0.0


def _weighted_average(scores: tuple[ConfidenceScore, ...]) -> float:
    denominator = sum(score.weight for score in scores)
    if denominator <= 0.0:
        return 0.0
    numerator = sum(score.score * score.weight for score in scores)
    return _clamp_probability(numerator / denominator)


def _ensure_confidence_scores(
    values: tuple[ConfidenceScore, ...] | list[ConfidenceScore],
) -> tuple[ConfidenceScore, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("scores must be tuple or list")
    scores = tuple(values)
    for score in scores:
        if not isinstance(score, ConfidenceScore):
            raise TypeError("scores must contain ConfidenceScore")
    score_names = tuple(score.score_name for score in scores)
    if len(set(score_names)) != len(score_names):
        raise ValueError("score_name values must be unique")
    return scores


def _ensure_non_negative_number(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be int or float")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    if number < 0.0:
        raise ValueError(f"{field_name} must be non-negative")
    return number


def _clamp_probability(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return ensure_probability(value, "value")


def _copy_metadata(metadata: dict[str, object]) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(metadata)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
    return copied
