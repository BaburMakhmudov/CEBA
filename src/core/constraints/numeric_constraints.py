"""Deterministic numeric constraint primitives for CEBA Core Constraint Processing."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from ceba.core.constraints.errors import (
    ConstraintEvaluationError,
    InvalidConstraintThresholdError,
    UnsupportedConstraintOperatorError,
    build_constraint_error_context,
)
from ceba.core.constraints.feature_adapter import (
    ConstraintFeature,
    ConstraintFeatureValueType,
)
from ceba.core.constraints.schema import (
    ConstraintOperator,
    ConstraintThreshold,
    ensure_constraint_probability,
    ensure_non_empty_string,
)


@dataclass(frozen=True, slots=True)
class NumericConstraintResult:
    """Deterministic result of one numeric primitive check."""

    passed: bool
    score: float
    reason_code: str
    observed_value: float | None
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
        if self.observed_value is not None:
            object.__setattr__(
                self,
                "observed_value",
                coerce_numeric_value(self.observed_value, field_name="observed_value"),
            )
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


def coerce_numeric_value(value: object, *, field_name: str = "value") -> float:
    """Convert safe numeric values to finite float."""

    try:
        ensure_non_empty_string(field_name, "field_name")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field_name} must be numeric")
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            raise ValueError(f"{field_name} must be finite")
        return numeric
    except ConstraintEvaluationError:
        raise
    except Exception as exc:
        raise ConstraintEvaluationError(
            message="failed to coerce numeric value",
            context=build_constraint_error_context(
                operation="coerce_numeric_value",
                field_path=(field_name,) if isinstance(field_name, str) and field_name else (),
            ),
            cause=exc,
        ) from exc


def numeric_value_from_feature(feature: ConstraintFeature) -> float:
    """Extract a numeric value from a ConstraintFeature."""

    try:
        if not isinstance(feature, ConstraintFeature):
            raise TypeError("feature must be ConstraintFeature")
        if feature.value_type is not ConstraintFeatureValueType.NUMBER:
            raise TypeError("feature value_type must be NUMBER")
        return coerce_numeric_value(feature.value, field_name=feature.name)
    except ConstraintEvaluationError:
        raise
    except Exception as exc:
        raise ConstraintEvaluationError(
            message="failed to extract numeric feature value",
            context=build_constraint_error_context(
                operation="numeric_value_from_feature",
                field_path=(feature.name,) if isinstance(feature, ConstraintFeature) else (),
            ),
            cause=exc,
        ) from exc


def validate_numeric_threshold(threshold: ConstraintThreshold) -> ConstraintThreshold:
    """Validate a threshold for numeric comparison."""

    if not isinstance(threshold, ConstraintThreshold):
        raise InvalidConstraintThresholdError(message="threshold must be ConstraintThreshold")

    if threshold.operator not in _SUPPORTED_NUMERIC_OPERATORS:
        raise UnsupportedConstraintOperatorError(
            message="unsupported numeric constraint operator",
            context=build_constraint_error_context(
                operation="validate_numeric_threshold",
                metadata={"operator": getattr(threshold.operator, "value", None)},
            ),
        )

    try:
        if threshold.operator is ConstraintOperator.BETWEEN:
            if threshold.lower_bound is None or threshold.upper_bound is None:
                raise ValueError("BETWEEN requires lower_bound and upper_bound")
            coerce_numeric_value(threshold.lower_bound, field_name="lower_bound")
            coerce_numeric_value(threshold.upper_bound, field_name="upper_bound")
        elif threshold.operator in (
            ConstraintOperator.GREATER_THAN,
            ConstraintOperator.GREATER_THAN_OR_EQUAL,
        ):
            if threshold.lower_bound is None and threshold.value is None:
                raise ValueError("greater-than operators require lower_bound or value")
            coerce_numeric_value(
                threshold.lower_bound if threshold.lower_bound is not None else threshold.value,
                field_name="threshold",
            )
        elif threshold.operator in (
            ConstraintOperator.LESS_THAN,
            ConstraintOperator.LESS_THAN_OR_EQUAL,
        ):
            if threshold.upper_bound is None and threshold.value is None:
                raise ValueError("less-than operators require upper_bound or value")
            coerce_numeric_value(
                threshold.upper_bound if threshold.upper_bound is not None else threshold.value,
                field_name="threshold",
            )
        elif threshold.operator in (ConstraintOperator.EQUALS, ConstraintOperator.NOT_EQUALS):
            coerce_numeric_value(threshold.value, field_name="value")
        return threshold
    except ConstraintEvaluationError as exc:
        raise InvalidConstraintThresholdError(
            message="invalid numeric constraint threshold",
            context=build_constraint_error_context(
                operation="validate_numeric_threshold",
                metadata={"operator": threshold.operator.value},
            ),
            cause=exc,
        ) from exc
    except Exception as exc:
        raise InvalidConstraintThresholdError(
            message="invalid numeric constraint threshold",
            context=build_constraint_error_context(
                operation="validate_numeric_threshold",
                metadata={"operator": threshold.operator.value},
            ),
            cause=exc,
        ) from exc


def expected_numeric_value(threshold: ConstraintThreshold) -> float | tuple[float, float]:
    """Resolve the expected numeric value from a threshold."""

    try:
        validate_numeric_threshold(threshold)
        if threshold.operator is ConstraintOperator.BETWEEN:
            if threshold.lower_bound is None or threshold.upper_bound is None:
                raise ValueError("BETWEEN requires lower_bound and upper_bound")
            return (
                coerce_numeric_value(threshold.lower_bound, field_name="lower_bound"),
                coerce_numeric_value(threshold.upper_bound, field_name="upper_bound"),
            )
        if threshold.operator in (ConstraintOperator.EQUALS, ConstraintOperator.NOT_EQUALS):
            return coerce_numeric_value(threshold.value, field_name="value")
        if threshold.operator in (
            ConstraintOperator.GREATER_THAN,
            ConstraintOperator.GREATER_THAN_OR_EQUAL,
        ):
            return coerce_numeric_value(
                threshold.lower_bound if threshold.lower_bound is not None else threshold.value,
                field_name="threshold",
            )
        if threshold.operator in (
            ConstraintOperator.LESS_THAN,
            ConstraintOperator.LESS_THAN_OR_EQUAL,
        ):
            return coerce_numeric_value(
                threshold.upper_bound if threshold.upper_bound is not None else threshold.value,
                field_name="threshold",
            )
        raise UnsupportedConstraintOperatorError(message="unsupported numeric constraint operator")
    except UnsupportedConstraintOperatorError:
        raise
    except InvalidConstraintThresholdError:
        raise
    except Exception as exc:
        raise InvalidConstraintThresholdError(
            message="failed to resolve expected numeric value",
            context=build_constraint_error_context(operation="expected_numeric_value"),
            cause=exc,
        ) from exc


def evaluate_numeric_operator(
    *,
    observed: float,
    threshold: ConstraintThreshold,
) -> bool:
    """Evaluate a numeric operator deterministically."""

    observed_value = coerce_numeric_value(observed, field_name="observed")
    validate_numeric_threshold(threshold)
    expected = expected_numeric_value(threshold)

    if threshold.operator is ConstraintOperator.EQUALS:
        return observed_value == expected
    if threshold.operator is ConstraintOperator.NOT_EQUALS:
        return observed_value != expected
    if threshold.operator is ConstraintOperator.GREATER_THAN:
        return observed_value > expected
    if threshold.operator is ConstraintOperator.GREATER_THAN_OR_EQUAL:
        return observed_value >= expected
    if threshold.operator is ConstraintOperator.LESS_THAN:
        return observed_value < expected
    if threshold.operator is ConstraintOperator.LESS_THAN_OR_EQUAL:
        return observed_value <= expected
    if threshold.operator is ConstraintOperator.BETWEEN:
        if not isinstance(expected, tuple):
            raise InvalidConstraintThresholdError(message="BETWEEN expected value must be tuple")
        lower, upper = expected
        if threshold.inclusive:
            return lower <= observed_value <= upper
        return lower < observed_value < upper
    raise UnsupportedConstraintOperatorError(message="unsupported numeric constraint operator")


def numeric_score(
    *,
    passed: bool,
    observed: float,
    threshold: ConstraintThreshold,
) -> float:
    """Return a deterministic score for numeric evaluation."""

    if not isinstance(passed, bool):
        raise TypeError("passed must be a bool")
    coerce_numeric_value(observed, field_name="observed")
    validate_numeric_threshold(threshold)
    return ensure_constraint_probability(1.0 if passed else 0.0, "score")


def evaluate_numeric_constraint(
    *,
    feature: ConstraintFeature,
    threshold: ConstraintThreshold,
) -> NumericConstraintResult:
    """Evaluate a numeric feature against a numeric threshold."""

    try:
        if not isinstance(feature, ConstraintFeature):
            raise TypeError("feature must be ConstraintFeature")
        if not isinstance(threshold, ConstraintThreshold):
            raise TypeError("threshold must be ConstraintThreshold")

        observed = numeric_value_from_feature(feature)
        validate_numeric_threshold(threshold)
        passed = evaluate_numeric_operator(observed=observed, threshold=threshold)
        score = numeric_score(passed=passed, observed=observed, threshold=threshold)
        reason_code = f"NUMERIC_{threshold.operator.value}_{'PASS' if passed else 'FAIL'}"

        return NumericConstraintResult(
            passed=passed,
            score=score,
            reason_code=reason_code,
            observed_value=observed,
            expected_value=expected_numeric_value(threshold),
            metadata={
                "feature_name": feature.name,
                "feature_id": feature.feature_id,
                "operator": threshold.operator.value,
                "inclusive": threshold.inclusive,
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
            message="failed to evaluate numeric constraint",
            context=build_constraint_error_context(operation="evaluate_numeric_constraint"),
            cause=exc,
        ) from exc


def _ensure_metadata_dict(value: dict[str, object]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("metadata must be a dict")
    if not all(isinstance(key, str) for key in value):
        raise TypeError("metadata keys must be strings")
    return value


_SUPPORTED_NUMERIC_OPERATORS = frozenset(
    {
        ConstraintOperator.EQUALS,
        ConstraintOperator.NOT_EQUALS,
        ConstraintOperator.GREATER_THAN,
        ConstraintOperator.GREATER_THAN_OR_EQUAL,
        ConstraintOperator.LESS_THAN,
        ConstraintOperator.LESS_THAN_OR_EQUAL,
        ConstraintOperator.BETWEEN,
    }
)


__all__ = [
    "NumericConstraintResult",
    "coerce_numeric_value",
    "numeric_value_from_feature",
    "validate_numeric_threshold",
    "expected_numeric_value",
    "evaluate_numeric_operator",
    "numeric_score",
    "evaluate_numeric_constraint",
]
