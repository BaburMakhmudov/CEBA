"""Deterministic constraint evaluator utilities for CEBA Core Constraint Processing."""

from __future__ import annotations

from dataclasses import dataclass

from ceba.core.constraints.errors import (
    ConstraintEvaluationError,
    ConstraintMatchingError,
    InvalidConstraintThresholdError,
    UnsupportedConstraintOperatorError,
    build_constraint_error_context,
)
from ceba.core.constraints.feature_adapter import (
    ConstraintFeature,
    ConstraintFeatureSet,
    ConstraintFeatureValueType,
)
from ceba.core.constraints.numeric_constraints import (
    NumericConstraintResult,
    evaluate_numeric_constraint,
)
from ceba.core.constraints.schema import (
    ConstraintDefinition,
    ConstraintEvaluation,
    ConstraintEvaluationStatus,
    ConstraintOperator,
    ConstraintThreshold,
    ConstraintViolationSeverity,
    ensure_constraint_probability,
    new_constraint_evaluation_id,
    utc_now,
)
from ceba.core.constraints.symbolic_constraints import (
    SymbolicConstraintResult,
    evaluate_symbolic_constraint,
)


@dataclass(frozen=True, slots=True)
class ConstraintEvaluatorPolicy:
    """Stable evaluator policy."""

    missing_feature_status: ConstraintEvaluationStatus = ConstraintEvaluationStatus.SKIP
    missing_feature_score: float = 0.0
    missing_feature_confidence: float = 1.0
    fail_severity: ConstraintViolationSeverity = ConstraintViolationSeverity.HIGH
    warn_severity: ConstraintViolationSeverity = ConstraintViolationSeverity.MEDIUM
    pass_severity: ConstraintViolationSeverity | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.missing_feature_status, ConstraintEvaluationStatus):
            raise TypeError("missing_feature_status must be ConstraintEvaluationStatus")
        object.__setattr__(
            self,
            "missing_feature_score",
            ensure_constraint_probability(self.missing_feature_score, "missing_feature_score"),
        )
        object.__setattr__(
            self,
            "missing_feature_confidence",
            ensure_constraint_probability(
                self.missing_feature_confidence,
                "missing_feature_confidence",
            ),
        )
        if not isinstance(self.fail_severity, ConstraintViolationSeverity):
            raise TypeError("fail_severity must be ConstraintViolationSeverity")
        if not isinstance(self.warn_severity, ConstraintViolationSeverity):
            raise TypeError("warn_severity must be ConstraintViolationSeverity")
        if self.pass_severity is not None and not isinstance(
            self.pass_severity,
            ConstraintViolationSeverity,
        ):
            raise TypeError("pass_severity must be ConstraintViolationSeverity or None")


def threshold_is_numeric(threshold: ConstraintThreshold) -> bool:
    """Return whether a threshold operator is numeric-compatible."""

    if not isinstance(threshold, ConstraintThreshold):
        raise TypeError("threshold must be ConstraintThreshold")
    return threshold.operator in _NUMERIC_OPERATORS


def threshold_is_symbolic(threshold: ConstraintThreshold) -> bool:
    """Return whether a threshold operator is symbolic-compatible."""

    if not isinstance(threshold, ConstraintThreshold):
        raise TypeError("threshold must be ConstraintThreshold")
    return threshold.operator in _SYMBOLIC_OPERATORS


def select_feature_for_constraint(
    constraint: ConstraintDefinition,
    feature_set: ConstraintFeatureSet,
) -> ConstraintFeature | None:
    """Select the target feature from a feature set."""

    if not isinstance(constraint, ConstraintDefinition):
        raise TypeError("constraint must be ConstraintDefinition")
    if not isinstance(feature_set, ConstraintFeatureSet):
        raise TypeError("feature_set must be ConstraintFeatureSet")
    feature_name = (
        constraint.target.field_path[-1]
        if constraint.target.field_path
        else constraint.target.target_id
    )
    return feature_set.by_name(feature_name)


def result_to_status(passed: bool) -> ConstraintEvaluationStatus:
    """Map a primitive boolean result to evaluation status."""

    if not isinstance(passed, bool):
        raise TypeError("passed must be a bool")
    return ConstraintEvaluationStatus.PASS if passed else ConstraintEvaluationStatus.FAIL


def result_to_violation_severity(
    status: ConstraintEvaluationStatus,
    *,
    policy: ConstraintEvaluatorPolicy,
) -> ConstraintViolationSeverity | None:
    """Map evaluation status to violation severity."""

    if not isinstance(status, ConstraintEvaluationStatus):
        raise TypeError("status must be ConstraintEvaluationStatus")
    if not isinstance(policy, ConstraintEvaluatorPolicy):
        raise TypeError("policy must be ConstraintEvaluatorPolicy")
    if status is ConstraintEvaluationStatus.PASS:
        return policy.pass_severity
    if status is ConstraintEvaluationStatus.WARN:
        return policy.warn_severity
    if status is ConstraintEvaluationStatus.FAIL:
        return policy.fail_severity
    return None


def evaluate_feature_against_threshold(
    *,
    feature: ConstraintFeature,
    threshold: ConstraintThreshold,
) -> NumericConstraintResult | SymbolicConstraintResult:
    """Route feature and threshold to a primitive evaluator."""

    try:
        if not isinstance(feature, ConstraintFeature):
            raise TypeError("feature must be ConstraintFeature")
        if not isinstance(threshold, ConstraintThreshold):
            raise TypeError("threshold must be ConstraintThreshold")
        if (
            feature.value_type is ConstraintFeatureValueType.NUMBER
            and threshold_is_numeric(threshold)
        ):
            return evaluate_numeric_constraint(feature=feature, threshold=threshold)
        return evaluate_symbolic_constraint(feature=feature, threshold=threshold)
    except UnsupportedConstraintOperatorError:
        raise
    except InvalidConstraintThresholdError:
        raise
    except ConstraintEvaluationError:
        raise
    except Exception as exc:
        raise ConstraintEvaluationError(
            message="failed to evaluate feature against threshold",
            context=build_constraint_error_context(
                operation="evaluate_feature_against_threshold",
                field_path=(feature.name,) if isinstance(feature, ConstraintFeature) else (),
            ),
            cause=exc,
        ) from exc


def build_missing_feature_evaluation(
    *,
    constraint: ConstraintDefinition,
    feature_set: ConstraintFeatureSet,
    policy: ConstraintEvaluatorPolicy,
) -> ConstraintEvaluation:
    """Create an evaluation for a missing target feature."""

    if not isinstance(constraint, ConstraintDefinition):
        raise TypeError("constraint must be ConstraintDefinition")
    if not isinstance(feature_set, ConstraintFeatureSet):
        raise TypeError("feature_set must be ConstraintFeatureSet")
    if not isinstance(policy, ConstraintEvaluatorPolicy):
        raise TypeError("policy must be ConstraintEvaluatorPolicy")

    return ConstraintEvaluation(
        evaluation_id=new_constraint_evaluation_id(),
        constraint_id=constraint.constraint_id,
        input_ref=feature_set.input_ref,
        status=policy.missing_feature_status,
        violation_severity=result_to_violation_severity(
            policy.missing_feature_status,
            policy=policy,
        ),
        score=policy.missing_feature_score,
        confidence=policy.missing_feature_confidence,
        message_code="CONSTRAINT_FEATURE_MISSING",
        created_at=utc_now(),
        metadata={
            "constraint_name": constraint.name,
            "target_id": constraint.target.target_id,
            "field_path": tuple(constraint.target.field_path),
            "feature_set_source": feature_set.source.value,
        },
    )


def evaluate_constraint(
    *,
    constraint: ConstraintDefinition,
    feature_set: ConstraintFeatureSet,
    policy: ConstraintEvaluatorPolicy | None = None,
) -> ConstraintEvaluation:
    """Evaluate one selected constraint against one feature set."""

    try:
        if not isinstance(constraint, ConstraintDefinition):
            raise TypeError("constraint must be ConstraintDefinition")
        if not isinstance(feature_set, ConstraintFeatureSet):
            raise TypeError("feature_set must be ConstraintFeatureSet")

        resolved_policy = ConstraintEvaluatorPolicy() if policy is None else policy
        if not isinstance(resolved_policy, ConstraintEvaluatorPolicy):
            raise TypeError("policy must be ConstraintEvaluatorPolicy or None")

        if constraint.threshold is None:
            raise InvalidConstraintThresholdError(
                message="constraint threshold is required",
                context=build_constraint_error_context(
                    operation="evaluate_constraint",
                    constraint_id=str(constraint.constraint_id),
                ),
            )

        feature = select_feature_for_constraint(constraint, feature_set)
        if feature is None:
            return build_missing_feature_evaluation(
                constraint=constraint,
                feature_set=feature_set,
                policy=resolved_policy,
            )

        primitive_result = evaluate_feature_against_threshold(
            feature=feature,
            threshold=constraint.threshold,
        )
        status = result_to_status(primitive_result.passed)
        primitive_score = ensure_constraint_probability(primitive_result.score, "score")
        confidence = _evaluation_confidence(
            feature_confidence=feature.confidence,
            primitive_score=primitive_score,
        )

        return ConstraintEvaluation(
            evaluation_id=new_constraint_evaluation_id(),
            constraint_id=constraint.constraint_id,
            input_ref=feature.input_ref,
            status=status,
            violation_severity=result_to_violation_severity(status, policy=resolved_policy),
            score=primitive_score,
            confidence=confidence,
            message_code=primitive_result.reason_code,
            created_at=utc_now(),
            metadata={
                "constraint_name": constraint.name,
                "constraint_kind": constraint.kind.value,
                "constraint_scope": constraint.scope.value,
                "feature_name": feature.name,
                "feature_id": feature.feature_id,
                "primitive_result": primitive_result.to_dict(),
            },
        )
    except UnsupportedConstraintOperatorError:
        raise
    except InvalidConstraintThresholdError:
        raise
    except ConstraintMatchingError:
        raise
    except ConstraintEvaluationError:
        raise
    except Exception as exc:
        raise ConstraintEvaluationError(
            message="failed to evaluate constraint",
            context=build_constraint_error_context(
                operation="evaluate_constraint",
                constraint_id=str(constraint.constraint_id)
                if isinstance(constraint, ConstraintDefinition)
                else None,
            ),
            cause=exc,
        ) from exc


def evaluate_constraints(
    *,
    constraints: tuple[ConstraintDefinition, ...] | list[ConstraintDefinition],
    feature_set: ConstraintFeatureSet,
    policy: ConstraintEvaluatorPolicy | None = None,
) -> tuple[ConstraintEvaluation, ...]:
    """Evaluate multiple constraints against one feature set."""

    try:
        if not isinstance(constraints, (tuple, list)):
            raise TypeError("constraints must be a tuple or list")
        if not isinstance(feature_set, ConstraintFeatureSet):
            raise TypeError("feature_set must be ConstraintFeatureSet")
        for constraint in constraints:
            if not isinstance(constraint, ConstraintDefinition):
                raise TypeError("constraints must contain ConstraintDefinition")
        return tuple(
            evaluate_constraint(
                constraint=constraint,
                feature_set=feature_set,
                policy=policy,
            )
            for constraint in constraints
        )
    except UnsupportedConstraintOperatorError:
        raise
    except InvalidConstraintThresholdError:
        raise
    except ConstraintMatchingError:
        raise
    except ConstraintEvaluationError:
        raise
    except Exception as exc:
        raise ConstraintEvaluationError(
            message="failed to evaluate constraints",
            context=build_constraint_error_context(operation="evaluate_constraints"),
            cause=exc,
        ) from exc


def _evaluation_confidence(
    *,
    feature_confidence: float,
    primitive_score: float,
) -> float:
    feature_confidence = ensure_constraint_probability(feature_confidence, "feature_confidence")
    primitive_score = ensure_constraint_probability(primitive_score, "primitive_score")
    if primitive_score < 1.0:
        return ensure_constraint_probability(
            min(feature_confidence, primitive_score),
            "confidence",
        )
    return feature_confidence


_NUMERIC_OPERATORS = frozenset(
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

_SYMBOLIC_OPERATORS = frozenset(
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
    "ConstraintEvaluatorPolicy",
    "threshold_is_numeric",
    "threshold_is_symbolic",
    "select_feature_for_constraint",
    "result_to_status",
    "result_to_violation_severity",
    "evaluate_feature_against_threshold",
    "build_missing_feature_evaluation",
    "evaluate_constraint",
    "evaluate_constraints",
]
