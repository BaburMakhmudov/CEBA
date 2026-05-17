"""Deterministic prediction utilities for CEBA World Model."""

from __future__ import annotations

from dataclasses import dataclass, field

from ceba.world_model.schema import (
    ExpectedObservationStatus,
    PredictionInput,
    PredictionKind,
    PredictionResult,
    PredictionStatus,
    create_expected_observation,
    create_prediction_result,
    ensure_non_empty_string,
    ensure_probability,
)


__all__ = [
    "PredictionPolicy",
    "PredictionRule",
    "PredictionExecutionResult",
    "WorldModelPredictor",
    "predict_expected_observation",
    "predict_expected_observations",
]


@dataclass(frozen=True)
class PredictionPolicy:
    """Configuration object controlling deterministic prediction."""

    min_input_evidence_confidence: float = 0.0
    min_prediction_confidence: float = 0.50
    default_uncertainty: float = 0.50
    require_evidence: bool = False
    allow_low_confidence_prediction: bool = True

    def __post_init__(self) -> None:
        min_input_evidence_confidence = ensure_probability(
            self.min_input_evidence_confidence,
            "min_input_evidence_confidence",
        )
        min_prediction_confidence = ensure_probability(
            self.min_prediction_confidence,
            "min_prediction_confidence",
        )
        default_uncertainty = ensure_probability(
            self.default_uncertainty,
            "default_uncertainty",
        )
        if not isinstance(self.require_evidence, bool):
            raise TypeError("require_evidence must be bool")
        if not isinstance(self.allow_low_confidence_prediction, bool):
            raise TypeError("allow_low_confidence_prediction must be bool")
        object.__setattr__(
            self,
            "min_input_evidence_confidence",
            min_input_evidence_confidence,
        )
        object.__setattr__(
            self,
            "min_prediction_confidence",
            min_prediction_confidence,
        )
        object.__setattr__(self, "default_uncertainty", default_uncertainty)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "min_input_evidence_confidence": (
                self.min_input_evidence_confidence
            ),
            "min_prediction_confidence": self.min_prediction_confidence,
            "default_uncertainty": self.default_uncertainty,
            "require_evidence": self.require_evidence,
            "allow_low_confidence_prediction": (
                self.allow_low_confidence_prediction
            ),
        }


@dataclass(frozen=True)
class PredictionRule:
    """Explicit deterministic rule for expected observation references."""

    rule_id: str
    prediction_kind: PredictionKind
    observation_ref: str
    confidence: float = 1.0
    uncertainty: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rule_id = ensure_non_empty_string(self.rule_id, "rule_id")
        if not isinstance(self.prediction_kind, PredictionKind):
            raise TypeError("prediction_kind must be PredictionKind")
        observation_ref = ensure_non_empty_string(
            self.observation_ref,
            "observation_ref",
        )
        confidence = ensure_probability(self.confidence, "confidence")
        uncertainty = ensure_probability(self.uncertainty, "uncertainty")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "rule_id", rule_id)
        object.__setattr__(self, "observation_ref", observation_ref)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "rule_id": self.rule_id,
            "prediction_kind": self.prediction_kind.value,
            "observation_ref": self.observation_ref,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PredictionExecutionResult:
    """Immutable result object returned by prediction execution."""

    results: tuple[PredictionResult, ...]
    failed_input_ids: tuple[str, ...]
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        results = _ensure_results(self.results)
        failed_ids = _ensure_string_tuple(
            self.failed_input_ids,
            "failed_input_ids",
        )
        failure_reasons = _ensure_string_tuple(
            self.failure_reasons,
            "failure_reasons",
        )
        if len(failed_ids) != len(failure_reasons):
            raise ValueError("failed_input_ids and failure_reasons must match")
        object.__setattr__(self, "results", results)
        object.__setattr__(self, "failed_input_ids", failed_ids)
        object.__setattr__(self, "failure_reasons", failure_reasons)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "results": tuple(result.to_dict() for result in self.results),
            "failed_input_ids": self.failed_input_ids,
            "failure_reasons": self.failure_reasons,
        }

    def result_count(self) -> int:
        """Return prediction result count."""

        return len(self.results)

    def failed_count(self) -> int:
        """Return failed input count."""

        return len(self.failed_input_ids)


class WorldModelPredictor:
    """Deterministic service that produces prediction results."""

    def __init__(
        self,
        *,
        rules: tuple[PredictionRule, ...] | list[PredictionRule] = (),
        policy: PredictionPolicy | None = None,
    ) -> None:
        self._policy = PredictionPolicy() if policy is None else policy
        if not isinstance(self._policy, PredictionPolicy):
            raise TypeError("policy must be PredictionPolicy")
        self._rules = _ensure_rules(rules)

    def predict_one(
        self,
        *,
        prediction_input: PredictionInput,
    ) -> PredictionExecutionResult:
        """Predict one expected observation."""

        if not isinstance(prediction_input, PredictionInput):
            raise TypeError("prediction_input must be PredictionInput")
        failure_reason = self._failure_reason(prediction_input)
        if failure_reason is not None:
            return _failure_result(prediction_input, failure_reason)

        rule = self._select_rule(prediction_input)
        if rule is None:
            return _failure_result(prediction_input, "prediction rule missing")

        evidence_confidence = _evidence_confidence(prediction_input)
        final_confidence = _clamp_probability(
            rule.confidence * evidence_confidence
        )
        low_confidence = (
            final_confidence < self._policy.min_prediction_confidence
        )
        if low_confidence and not self._policy.allow_low_confidence_prediction:
            return _failure_result(
                prediction_input,
                "prediction confidence below minimum",
            )

        expected_status = (
            ExpectedObservationStatus.UNCERTAIN
            if low_confidence
            else ExpectedObservationStatus.EXPECTED
        )
        result_status = (
            PredictionStatus.LOW_CONFIDENCE
            if low_confidence
            else PredictionStatus.PREDICTED
        )
        uncertainty = _clamp_probability(
            max(
                rule.uncertainty,
                self._policy.default_uncertainty
                if low_confidence
                else rule.uncertainty,
            )
        )
        metadata = _metadata_for_prediction(
            prediction_input=prediction_input,
            rule=rule,
            final_confidence=final_confidence,
            policy=self._policy,
        )
        expected = create_expected_observation(
            prediction_input_id=prediction_input.input_id,
            observation_ref=rule.observation_ref,
            status=expected_status,
            confidence=final_confidence,
            uncertainty=uncertainty,
            metadata=metadata,
        )
        result = create_prediction_result(
            prediction_input=prediction_input,
            expected_observation=expected,
            status=result_status,
            confidence=final_confidence,
            evidence=prediction_input.evidence,
            metadata=metadata,
        )
        return PredictionExecutionResult(
            results=(result,),
            failed_input_ids=(),
            failure_reasons=(),
        )

    def predict_many(
        self,
        *,
        inputs: tuple[PredictionInput, ...] | list[PredictionInput],
    ) -> PredictionExecutionResult:
        """Predict expected observations for many inputs."""

        prediction_inputs = _ensure_inputs(inputs)
        results: list[PredictionResult] = []
        failed_ids: list[str] = []
        failure_reasons: list[str] = []
        for prediction_input in prediction_inputs:
            result = self.predict_one(prediction_input=prediction_input)
            results.extend(result.results)
            failed_ids.extend(result.failed_input_ids)
            failure_reasons.extend(result.failure_reasons)
        return PredictionExecutionResult(
            results=tuple(results),
            failed_input_ids=tuple(failed_ids),
            failure_reasons=tuple(failure_reasons),
        )

    def _failure_reason(self, prediction_input: PredictionInput) -> str | None:
        if self._policy.require_evidence and not prediction_input.evidence:
            return "prediction input has no evidence"
        if not prediction_input.evidence:
            return None
        if all(
            evidence.confidence < self._policy.min_input_evidence_confidence
            for evidence in prediction_input.evidence
        ):
            return "prediction evidence confidence below minimum"
        return None

    def _select_rule(self, prediction_input: PredictionInput) -> PredictionRule | None:
        for rule in self._rules:
            if rule.prediction_kind is prediction_input.kind:
                return rule
        for rule in self._rules:
            if rule.prediction_kind is PredictionKind.UNKNOWN:
                return rule
        return None


def predict_expected_observation(
    *,
    prediction_input: PredictionInput,
    rules: tuple[PredictionRule, ...] | list[PredictionRule] = (),
    policy: PredictionPolicy | None = None,
) -> PredictionExecutionResult:
    """Predict one expected observation."""

    predictor = WorldModelPredictor(rules=rules, policy=policy)
    return predictor.predict_one(prediction_input=prediction_input)


def predict_expected_observations(
    *,
    inputs: tuple[PredictionInput, ...] | list[PredictionInput],
    rules: tuple[PredictionRule, ...] | list[PredictionRule] = (),
    policy: PredictionPolicy | None = None,
) -> PredictionExecutionResult:
    """Predict expected observations for many inputs."""

    predictor = WorldModelPredictor(rules=rules, policy=policy)
    return predictor.predict_many(inputs=inputs)


def _ensure_rules(
    values: tuple[PredictionRule, ...] | list[PredictionRule],
) -> tuple[PredictionRule, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("rules must be tuple or list")
    rules = tuple(values)
    for rule in rules:
        if not isinstance(rule, PredictionRule):
            raise TypeError("rules must contain PredictionRule")
    rule_ids = tuple(rule.rule_id for rule in rules)
    if len(set(rule_ids)) != len(rule_ids):
        raise ValueError("rule_id values must be unique")
    return rules


def _ensure_inputs(
    values: tuple[PredictionInput, ...] | list[PredictionInput],
) -> tuple[PredictionInput, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("inputs must be tuple or list")
    inputs = tuple(values)
    for item in inputs:
        if not isinstance(item, PredictionInput):
            raise TypeError("inputs must contain PredictionInput")
    return inputs


def _ensure_results(
    values: tuple[PredictionResult, ...] | list[PredictionResult],
) -> tuple[PredictionResult, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("results must be tuple or list")
    results = tuple(values)
    for result in results:
        if not isinstance(result, PredictionResult):
            raise TypeError("results must contain PredictionResult")
    result_ids = tuple(result.result_id for result in results)
    if len(set(result_ids)) != len(result_ids):
        raise ValueError("result_id values must be unique")
    return results


def _ensure_string_tuple(
    values: tuple[str, ...] | list[str],
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple or list")
    return tuple(ensure_non_empty_string(value, field_name) for value in values)


def _evidence_confidence(prediction_input: PredictionInput) -> float:
    if not prediction_input.evidence:
        return 1.0
    total = sum(evidence.confidence for evidence in prediction_input.evidence)
    return _clamp_probability(total / len(prediction_input.evidence))


def _metadata_for_prediction(
    *,
    prediction_input: PredictionInput,
    rule: PredictionRule,
    final_confidence: float,
    policy: PredictionPolicy,
) -> dict[str, object]:
    metadata = dict(rule.metadata)
    metadata.update(
        {
            "source": "world_model_predictor",
            "prediction_input_id": str(prediction_input.input_id),
            "prediction_kind": prediction_input.kind.value,
            "rule_id": rule.rule_id,
            "evidence_count": len(prediction_input.evidence),
            "final_confidence": final_confidence,
            "policy": policy.to_dict(),
        }
    )
    metadata["source"] = "world_model_predictor"
    return metadata


def _failure_result(
    prediction_input: PredictionInput,
    reason: str,
) -> PredictionExecutionResult:
    return PredictionExecutionResult(
        results=(),
        failed_input_ids=(str(prediction_input.input_id),),
        failure_reasons=(reason,),
    )


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
