"""Deterministic symbolic constraint primitives for CEBA Core Constraint Processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ceba.core.constraints.errors import (
    ConstraintEvaluationError,
    InvalidConstraintThresholdError,
    UnsupportedConstraintOperatorError,
    build_constraint_error_context,
)
from ceba.core.constraints.feature_adapter import ConstraintFeature
from ceba.core.constraints.schema import (
    ConstraintOperator,
    ConstraintThreshold,
    ensure_constraint_probability,
    ensure_non_empty_string,
)


@dataclass(frozen=True, slots=True)
class SymbolicConstraintResult:
    """Deterministic result of one symbolic primitive check."""

    passed: bool
    score: float
    reason_code: str
    observed_value: object
    expected_value: object
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be a bool")
        object.__setattr__(
            self,
            "score",
            ensure_constraint_probability(self.score, "score"),
        )
        ensure_non_empty_string(self.reason_code, "reason_code")
        _ensure_metadata_dict(self.metadata)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serialization-friendly representation."""

        return {
            "passed": self.passed,
            "score": self.score,
            "reason_code": self.reason_code,
            "observed_value": self.observed_value,
            "expected_value": self.expected_value,
            "metadata": dict(self.metadata),
        }


def symbolic_value_from_feature(feature: ConstraintFeature) -> object:
    """Extract a symbolic value from a ConstraintFeature."""

    try:
        if not isinstance(feature, ConstraintFeature):
            raise TypeError("feature must be ConstraintFeature")
        return feature.value
    except ConstraintEvaluationError:
        raise
    except Exception as exc:
        raise ConstraintEvaluationError(
            message="failed to extract symbolic feature value",
            context=build_constraint_error_context(
                operation="symbolic_value_from_feature",
                field_path=(feature.name,) if isinstance(feature, ConstraintFeature) else (),
            ),
            cause=exc,
        ) from exc


def normalize_symbolic_value(value: object) -> object:
    """Normalize symbolic values for deterministic comparison."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return tuple(normalize_symbolic_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(normalize_symbolic_value(item) for item in value)
    if isinstance(value, dict):
        return {
            key: normalize_symbolic_value(item)
            for key, item in sorted(
                ((str(key), item) for key, item in value.items()),
                key=lambda item: item[0],
            )
        }
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return repr(value)


def validate_symbolic_threshold(threshold: ConstraintThreshold) -> ConstraintThreshold:
    """Validate a threshold for symbolic comparison."""

    if not isinstance(threshold, ConstraintThreshold):
        raise InvalidConstraintThresholdError(message="threshold must be ConstraintThreshold")

    if threshold.operator not in _SUPPORTED_SYMBOLIC_OPERATORS:
        raise UnsupportedConstraintOperatorError(
            message="unsupported symbolic constraint operator",
            context=build_constraint_error_context(
                operation="validate_symbolic_threshold",
                metadata={"operator": getattr(threshold.operator, "value", None)},
            ),
        )

    try:
        if threshold.operator in (ConstraintOperator.EXISTS, ConstraintOperator.NOT_EXISTS):
            return threshold
        if threshold.operator in (ConstraintOperator.EQUALS, ConstraintOperator.NOT_EQUALS):
            if threshold.value is None:
                raise ValueError("EQUALS and NOT_EQUALS require threshold.value")
            return threshold
        if threshold.operator in (ConstraintOperator.IN_SET, ConstraintOperator.NOT_IN_SET):
            if not isinstance(threshold.value, (tuple, list, set, frozenset)):
                raise TypeError("IN_SET and NOT_IN_SET require a collection value")
            return threshold
        if threshold.operator is ConstraintOperator.CUSTOM:
            return threshold
        raise UnsupportedConstraintOperatorError(message="unsupported symbolic constraint operator")
    except UnsupportedConstraintOperatorError:
        raise
    except Exception as exc:
        raise InvalidConstraintThresholdError(
            message="invalid symbolic constraint threshold",
            context=build_constraint_error_context(
                operation="validate_symbolic_threshold",
                metadata={"operator": threshold.operator.value},
            ),
            cause=exc,
        ) from exc


def expected_symbolic_value(threshold: ConstraintThreshold) -> object:
    """Resolve the expected symbolic value from a threshold."""

    try:
        validate_symbolic_threshold(threshold)
        if threshold.operator in (ConstraintOperator.EXISTS, ConstraintOperator.NOT_EXISTS):
            return None
        if threshold.operator in (ConstraintOperator.EQUALS, ConstraintOperator.NOT_EQUALS):
            return normalize_symbolic_value(threshold.value)
        if threshold.operator in (ConstraintOperator.IN_SET, ConstraintOperator.NOT_IN_SET):
            if not isinstance(threshold.value, (tuple, list, set, frozenset)):
                raise TypeError("set membership threshold value must be a collection")
            return tuple(normalize_symbolic_value(item) for item in threshold.value)
        if threshold.operator is ConstraintOperator.CUSTOM:
            return threshold.value
        raise UnsupportedConstraintOperatorError(message="unsupported symbolic constraint operator")
    except UnsupportedConstraintOperatorError:
        raise
    except InvalidConstraintThresholdError:
        raise
    except Exception as exc:
        raise InvalidConstraintThresholdError(
            message="failed to resolve expected symbolic value",
            context=build_constraint_error_context(operation="expected_symbolic_value"),
            cause=exc,
        ) from exc


def evaluate_symbolic_operator(
    *,
    observed: object,
    threshold: ConstraintThreshold,
) -> bool:
    """Evaluate a symbolic operator deterministically."""

    validate_symbolic_threshold(threshold)
    if threshold.operator is ConstraintOperator.CUSTOM:
        raise UnsupportedConstraintOperatorError(message="CUSTOM symbolic operator is not evaluated")

    normalized_observed = normalize_symbolic_value(observed)
    expected = expected_symbolic_value(threshold)

    if threshold.operator is ConstraintOperator.EXISTS:
        return observed is not None
    if threshold.operator is ConstraintOperator.NOT_EXISTS:
        return observed is None
    if threshold.operator is ConstraintOperator.EQUALS:
        return normalized_observed == expected
    if threshold.operator is ConstraintOperator.NOT_EQUALS:
        return normalized_observed != expected
    if threshold.operator is ConstraintOperator.IN_SET:
        if not isinstance(expected, tuple):
            raise InvalidConstraintThresholdError(message="IN_SET expected value must be tuple")
        return normalized_observed in expected
    if threshold.operator is ConstraintOperator.NOT_IN_SET:
        if not isinstance(expected, tuple):
            raise InvalidConstraintThresholdError(message="NOT_IN_SET expected value must be tuple")
        return normalized_observed not in expected
    raise UnsupportedConstraintOperatorError(message="unsupported symbolic constraint operator")


def symbolic_score(
    *,
    passed: bool,
    observed: object,
    threshold: ConstraintThreshold,
) -> float:
    """Return a deterministic score for symbolic evaluation."""

    if not isinstance(passed, bool):
        raise TypeError("passed must be a bool")
    normalize_symbolic_value(observed)
    validate_symbolic_threshold(threshold)
    return ensure_constraint_probability(1.0 if passed else 0.0, "score")


def evaluate_symbolic_constraint(
    *,
    feature: ConstraintFeature,
    threshold: ConstraintThreshold,
) -> SymbolicConstraintResult:
    """Evaluate a symbolic feature against a symbolic threshold."""

    try:
        if not isinstance(feature, ConstraintFeature):
            raise TypeError("feature must be ConstraintFeature")
        if not isinstance(threshold, ConstraintThreshold):
            raise TypeError("threshold must be ConstraintThreshold")

        observed = symbolic_value_from_feature(feature)
        validate_symbolic_threshold(threshold)
        passed = evaluate_symbolic_operator(observed=observed, threshold=threshold)
        score = symbolic_score(passed=passed, observed=observed, threshold=threshold)
        reason_code = f"SYMBOLIC_{threshold.operator.value}_{'PASS' if passed else 'FAIL'}"

        return SymbolicConstraintResult(
            passed=passed,
            score=score,
            reason_code=reason_code,
            observed_value=normalize_symbolic_value(observed),
            expected_value=expected_symbolic_value(threshold),
            metadata={
                "feature_name": feature.name,
                "feature_id": feature.feature_id,
                "operator": threshold.operator.value,
                "value_type": feature.value_type.value,
            },
        )
    except UnsupportedConstraintOperatorError:
        raise
    except InvalidConstraintThresholdError:
        raise
    except ConstraintEvaluationError:
        raise
    except Exception as exc:
        raise ConstraintEvaluationError(
            message="failed to evaluate symbolic constraint",
            context=build_constraint_error_context(operation="evaluate_symbolic_constraint"),
            cause=exc,
        ) from exc


def _ensure_metadata_dict(value: dict[str, object]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("metadata must be a dict")
    if not all(isinstance(key, str) for key in value):
        raise TypeError("metadata keys must be strings")
    return value


_SUPPORTED_SYMBOLIC_OPERATORS = frozenset(
    {
        ConstraintOperator.EXISTS,
        ConstraintOperator.NOT_EXISTS,
        ConstraintOperator.EQUALS,
        ConstraintOperator.NOT_EQUALS,
        ConstraintOperator.IN_SET,
        ConstraintOperator.NOT_IN_SET,
        ConstraintOperator.CUSTOM,
    }
)


__all__ = [
    "SymbolicConstraintResult",
    "symbolic_value_from_feature",
    "normalize_symbolic_value",
    "validate_symbolic_threshold",
    "expected_symbolic_value",
    "evaluate_symbolic_operator",
    "symbolic_score",
    "evaluate_symbolic_constraint",
]
