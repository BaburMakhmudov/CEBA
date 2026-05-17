"""Deterministic validation utilities for CEBA Core Constraint Processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ceba.core.constraints.engine import ConstraintEngineResult
from ceba.core.constraints.errors import (
    ConstraintValidationError,
    EmptyConstraintSetError,
    build_constraint_error_context,
)
from ceba.core.constraints.feature_adapter import ConstraintFeature, ConstraintFeatureSet
from ceba.core.constraints.registry import ConstraintRegistry
from ceba.core.constraints.schema import (
    ConstraintDefinition,
    ConstraintEvaluation,
    ConstraintEvaluationSet,
    ConstraintEvaluationStatus,
    ConstraintInputRef,
    ConstraintOperator,
    ConstraintStatus,
    ConstraintTarget,
    ConstraintThreshold,
    ConstraintViolationSeverity,
    ensure_constraint_probability,
    ensure_metadata_dict,
    ensure_non_empty_string,
    ensure_timezone_aware,
)


class ConstraintValidationStatus(str, Enum):
    """Stable validation status for constraint artifacts."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class ConstraintValidationIssue:
    """One deterministic validation issue."""

    code: str
    message: str
    status: ConstraintValidationStatus
    constraint_id: str | None = None
    evaluation_id: str | None = None
    difference_id: str | None = None
    observation_id: str | None = None
    field_path: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ensure_non_empty_string(self.code, "code")
        ensure_non_empty_string(self.message, "message")
        if not isinstance(self.status, ConstraintValidationStatus):
            raise TypeError("status must be ConstraintValidationStatus")
        _ensure_optional_non_empty_string(self.constraint_id, "constraint_id")
        _ensure_optional_non_empty_string(self.evaluation_id, "evaluation_id")
        _ensure_optional_non_empty_string(self.difference_id, "difference_id")
        _ensure_optional_non_empty_string(self.observation_id, "observation_id")
        if isinstance(self.field_path, list):
            object.__setattr__(self, "field_path", tuple(self.field_path))
        if not isinstance(self.field_path, tuple):
            raise TypeError("field_path must be a tuple of strings")
        for part in self.field_path:
            ensure_non_empty_string(part, "field_path")
        ensure_metadata_dict(self.metadata)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic, serialization-friendly representation."""

        return {
            "code": self.code,
            "message": self.message,
            "status": self.status.value,
            "constraint_id": self.constraint_id,
            "evaluation_id": self.evaluation_id,
            "difference_id": self.difference_id,
            "observation_id": self.observation_id,
            "field_path": list(self.field_path),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ConstraintValidationReport:
    """Validation result for one constraint artifact or batch."""

    status: ConstraintValidationStatus
    issues: tuple[ConstraintValidationIssue, ...]
    checked_count: int
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, ConstraintValidationStatus):
            raise TypeError("status must be ConstraintValidationStatus")
        if isinstance(self.issues, list):
            object.__setattr__(self, "issues", tuple(self.issues))
        if not isinstance(self.issues, tuple):
            raise TypeError("issues must be a tuple of ConstraintValidationIssue")
        for validation_issue in self.issues:
            if not isinstance(validation_issue, ConstraintValidationIssue):
                raise TypeError("issues must contain ConstraintValidationIssue")
        _ensure_non_negative_int(self.checked_count, "checked_count")
        ensure_metadata_dict(self.metadata)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def passed(self) -> bool:
        """Return True when validation passed."""

        return self.status is ConstraintValidationStatus.PASS

    @property
    def failed(self) -> bool:
        """Return True when validation failed."""

        return self.status is ConstraintValidationStatus.FAIL

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic, serialization-friendly representation."""

        return {
            "status": self.status.value,
            "issues": [validation_issue.to_dict() for validation_issue in self.issues],
            "checked_count": self.checked_count,
            "metadata": dict(self.metadata),
        }


def issue(
    *,
    code: str,
    message: str,
    status: ConstraintValidationStatus,
    constraint_id: str | None = None,
    evaluation_id: str | None = None,
    difference_id: str | None = None,
    observation_id: str | None = None,
    field_path: tuple[str, ...] | list[str] = (),
    metadata: dict[str, object] | None = None,
) -> ConstraintValidationIssue:
    """Build one validated issue."""

    return ConstraintValidationIssue(
        code=code,
        message=message,
        status=status,
        constraint_id=constraint_id,
        evaluation_id=evaluation_id,
        difference_id=difference_id,
        observation_id=observation_id,
        field_path=tuple(field_path),
        metadata={} if metadata is None else metadata,
    )


def report_from_issues(
    issues: tuple[ConstraintValidationIssue, ...] | list[ConstraintValidationIssue],
    *,
    checked_count: int,
    metadata: dict[str, object] | None = None,
) -> ConstraintValidationReport:
    """Build a validation report with dominant status."""

    if not isinstance(issues, (tuple, list)):
        raise TypeError("issues must be a tuple or list")
    resolved_issues = tuple(issues)
    for validation_issue in resolved_issues:
        if not isinstance(validation_issue, ConstraintValidationIssue):
            raise TypeError("issues must contain ConstraintValidationIssue")
    _ensure_non_negative_int(checked_count, "checked_count")

    status = ConstraintValidationStatus.PASS
    if any(validation_issue.status is ConstraintValidationStatus.FAIL for validation_issue in resolved_issues):
        status = ConstraintValidationStatus.FAIL
    elif any(validation_issue.status is ConstraintValidationStatus.WARN for validation_issue in resolved_issues):
        status = ConstraintValidationStatus.WARN

    return ConstraintValidationReport(
        status=status,
        issues=resolved_issues,
        checked_count=checked_count,
        metadata={} if metadata is None else metadata,
    )


def validate_constraint_target(target: ConstraintTarget) -> ConstraintValidationReport:
    """Validate one ConstraintTarget."""

    try:
        if not isinstance(target, ConstraintTarget):
            raise TypeError("target must be ConstraintTarget")
        issues: list[ConstraintValidationIssue] = []
        _append_validation_issue(
            issues,
            code="INVALID_TARGET_ID",
            message="target_id must be non-empty",
            status=ConstraintValidationStatus.FAIL,
            field_path=("target_id",),
            validator=lambda: ensure_non_empty_string(target.target_id, "target_id"),
        )
        for index, part in enumerate(target.field_path):
            _append_validation_issue(
                issues,
                code="INVALID_FIELD_PATH",
                message="field_path parts must be non-empty",
                status=ConstraintValidationStatus.FAIL,
                field_path=("field_path", str(index)),
                validator=lambda part=part: ensure_non_empty_string(part, "field_path"),
            )
        return report_from_issues(issues, checked_count=1)
    except ConstraintValidationError:
        raise
    except Exception as exc:
        raise ConstraintValidationError(
            message="failed to validate constraint target",
            context=build_constraint_error_context(operation="validate_constraint_target"),
            cause=exc,
        ) from exc


def validate_constraint_threshold(
    threshold: ConstraintThreshold | None,
) -> ConstraintValidationReport:
    """Validate one optional ConstraintThreshold."""

    try:
        issues: list[ConstraintValidationIssue] = []
        if threshold is None:
            issues.append(
                issue(
                    code="MISSING_THRESHOLD",
                    message="constraint threshold is missing",
                    status=ConstraintValidationStatus.WARN,
                    field_path=("threshold",),
                )
            )
            return report_from_issues(issues, checked_count=1)
        if not isinstance(threshold, ConstraintThreshold):
            raise TypeError("threshold must be ConstraintThreshold or None")
        if not isinstance(threshold.operator, ConstraintOperator):
            issues.append(
                issue(
                    code="INVALID_OPERATOR",
                    message="threshold operator must be ConstraintOperator",
                    status=ConstraintValidationStatus.FAIL,
                    field_path=("operator",),
                )
            )
        if threshold.operator is ConstraintOperator.BETWEEN:
            if threshold.lower_bound is None or threshold.upper_bound is None:
                issues.append(
                    issue(
                        code="INVALID_BETWEEN_BOUNDS",
                        message="BETWEEN requires lower_bound and upper_bound",
                        status=ConstraintValidationStatus.FAIL,
                        field_path=("lower_bound", "upper_bound"),
                    )
                )
        if (
            threshold.lower_bound is not None
            and threshold.upper_bound is not None
            and threshold.upper_bound < threshold.lower_bound
        ):
            issues.append(
                issue(
                    code="INVALID_THRESHOLD_BOUNDS",
                    message="upper_bound must be greater than or equal to lower_bound",
                    status=ConstraintValidationStatus.FAIL,
                    field_path=("upper_bound",),
                )
            )
        return report_from_issues(issues, checked_count=1)
    except ConstraintValidationError:
        raise
    except Exception as exc:
        raise ConstraintValidationError(
            message="failed to validate constraint threshold",
            context=build_constraint_error_context(operation="validate_constraint_threshold"),
            cause=exc,
        ) from exc


def validate_constraint_definition(
    definition: ConstraintDefinition,
) -> ConstraintValidationReport:
    """Validate one ConstraintDefinition."""

    constraint_id: str | None = None
    try:
        if isinstance(definition, ConstraintDefinition):
            constraint_id = str(definition.constraint_id)
        else:
            raise TypeError("definition must be ConstraintDefinition")

        issues: list[ConstraintValidationIssue] = []
        _append_validation_issue(
            issues,
            code="INVALID_CONSTRAINT_NAME",
            message="constraint name must be non-empty",
            status=ConstraintValidationStatus.FAIL,
            constraint_id=constraint_id,
            field_path=("name",),
            validator=lambda: ensure_non_empty_string(definition.name, "name"),
        )
        if definition.status is ConstraintStatus.DEPRECATED:
            issues.append(
                issue(
                    code="DEPRECATED_CONSTRAINT",
                    message="constraint is deprecated",
                    status=ConstraintValidationStatus.WARN,
                    constraint_id=constraint_id,
                    field_path=("status",),
                )
            )
        if definition.status is ConstraintStatus.INACTIVE:
            issues.append(
                issue(
                    code="INACTIVE_CONSTRAINT",
                    message="constraint is inactive",
                    status=ConstraintValidationStatus.WARN,
                    constraint_id=constraint_id,
                    field_path=("status",),
                )
            )
        issues.extend(validate_constraint_target(definition.target).issues)
        issues.extend(validate_constraint_threshold(definition.threshold).issues)
        _append_validation_issue(
            issues,
            code="INVALID_PRIORITY",
            message="constraint priority must be non-negative",
            status=ConstraintValidationStatus.FAIL,
            constraint_id=constraint_id,
            field_path=("priority",),
            validator=lambda: _ensure_non_negative_number(definition.priority, "priority"),
        )
        _append_validation_issue(
            issues,
            code="INVALID_WEIGHT",
            message="constraint weight must be non-negative",
            status=ConstraintValidationStatus.FAIL,
            constraint_id=constraint_id,
            field_path=("weight",),
            validator=lambda: _ensure_non_negative_number(definition.weight, "weight"),
        )
        _append_validation_issue(
            issues,
            code="INVALID_CREATED_AT",
            message="constraint created_at must be timezone-aware",
            status=ConstraintValidationStatus.FAIL,
            constraint_id=constraint_id,
            field_path=("created_at",),
            validator=lambda: ensure_timezone_aware(definition.created_at, "created_at"),
        )
        return report_from_issues(issues, checked_count=1)
    except ConstraintValidationError:
        raise
    except Exception as exc:
        raise ConstraintValidationError(
            message="failed to validate constraint definition",
            context=build_constraint_error_context(
                operation="validate_constraint_definition",
                constraint_id=constraint_id,
            ),
            cause=exc,
        ) from exc


def validate_constraint_registry(registry: ConstraintRegistry) -> ConstraintValidationReport:
    """Validate one ConstraintRegistry."""

    try:
        if not isinstance(registry, ConstraintRegistry):
            raise TypeError("registry must be ConstraintRegistry")
        issues: list[ConstraintValidationIssue] = []
        if not registry.constraints:
            issues.append(
                issue(
                    code="EMPTY_REGISTRY",
                    message="constraint registry is empty",
                    status=ConstraintValidationStatus.WARN,
                    field_path=("constraints",),
                )
            )
        seen_ids: set[str] = set()
        for definition in registry.constraints:
            constraint_id = str(definition.constraint_id)
            if constraint_id in seen_ids:
                issues.append(
                    issue(
                        code="DUPLICATE_CONSTRAINT_ID",
                        message="duplicate constraint_id in registry",
                        status=ConstraintValidationStatus.FAIL,
                        constraint_id=constraint_id,
                        field_path=("constraints",),
                    )
                )
            seen_ids.add(constraint_id)
            issues.extend(validate_constraint_definition(definition).issues)
        return report_from_issues(issues, checked_count=len(registry.constraints))
    except ConstraintValidationError:
        raise
    except Exception as exc:
        raise ConstraintValidationError(
            message="failed to validate constraint registry",
            context=build_constraint_error_context(operation="validate_constraint_registry"),
            cause=exc,
        ) from exc


def validate_constraint_feature(feature: ConstraintFeature) -> ConstraintValidationReport:
    """Validate one ConstraintFeature."""

    try:
        if not isinstance(feature, ConstraintFeature):
            raise TypeError("feature must be ConstraintFeature")
        issues: list[ConstraintValidationIssue] = []
        _append_validation_issue(
            issues,
            code="INVALID_FEATURE_ID",
            message="feature_id must be non-empty",
            status=ConstraintValidationStatus.FAIL,
            field_path=("feature_id",),
            validator=lambda: ensure_non_empty_string(feature.feature_id, "feature_id"),
        )
        _append_validation_issue(
            issues,
            code="INVALID_FEATURE_NAME",
            message="feature name must be non-empty",
            status=ConstraintValidationStatus.FAIL,
            field_path=("name",),
            validator=lambda: ensure_non_empty_string(feature.name, "name"),
        )
        if not isinstance(feature.input_ref, ConstraintInputRef):
            issues.append(
                issue(
                    code="INVALID_INPUT_REF",
                    message="feature input_ref must be ConstraintInputRef",
                    status=ConstraintValidationStatus.FAIL,
                    field_path=("input_ref",),
                )
            )
        _append_validation_issue(
            issues,
            code="INVALID_CONFIDENCE",
            message="feature confidence must be a probability",
            status=ConstraintValidationStatus.FAIL,
            field_path=("confidence",),
            validator=lambda: ensure_constraint_probability(feature.confidence, "confidence"),
        )
        return report_from_issues(issues, checked_count=1)
    except ConstraintValidationError:
        raise
    except Exception as exc:
        raise ConstraintValidationError(
            message="failed to validate constraint feature",
            context=build_constraint_error_context(operation="validate_constraint_feature"),
            cause=exc,
        ) from exc


def validate_constraint_feature_set(feature_set: ConstraintFeatureSet) -> ConstraintValidationReport:
    """Validate one ConstraintFeatureSet."""

    try:
        if not isinstance(feature_set, ConstraintFeatureSet):
            raise TypeError("feature_set must be ConstraintFeatureSet")
        issues: list[ConstraintValidationIssue] = []
        if not feature_set.features:
            issues.append(
                issue(
                    code="EMPTY_FEATURE_SET",
                    message="constraint feature set is empty",
                    status=ConstraintValidationStatus.WARN,
                    field_path=("features",),
                )
            )
        seen_names: set[str] = set()
        for feature in feature_set.features:
            if feature.name in seen_names:
                issues.append(
                    issue(
                        code="DUPLICATE_FEATURE_NAME",
                        message="duplicate feature name",
                        status=ConstraintValidationStatus.FAIL,
                        field_path=("features", feature.name),
                    )
                )
            seen_names.add(feature.name)
            issues.extend(validate_constraint_feature(feature).issues)
        return report_from_issues(issues, checked_count=len(feature_set.features))
    except ConstraintValidationError:
        raise
    except Exception as exc:
        raise ConstraintValidationError(
            message="failed to validate constraint feature set",
            context=build_constraint_error_context(operation="validate_constraint_feature_set"),
            cause=exc,
        ) from exc


def validate_constraint_evaluation(
    evaluation: ConstraintEvaluation,
) -> ConstraintValidationReport:
    """Validate one ConstraintEvaluation."""

    evaluation_id: str | None = None
    try:
        if isinstance(evaluation, ConstraintEvaluation):
            evaluation_id = str(evaluation.evaluation_id)
        else:
            raise TypeError("evaluation must be ConstraintEvaluation")
        issues: list[ConstraintValidationIssue] = []
        if not isinstance(evaluation.input_ref, ConstraintInputRef):
            issues.append(
                issue(
                    code="INVALID_INPUT_REF",
                    message="evaluation input_ref must be ConstraintInputRef",
                    status=ConstraintValidationStatus.FAIL,
                    evaluation_id=evaluation_id,
                    field_path=("input_ref",),
                )
            )
        if not isinstance(evaluation.status, ConstraintEvaluationStatus):
            issues.append(
                issue(
                    code="INVALID_EVALUATION_STATUS",
                    message="evaluation status must be ConstraintEvaluationStatus",
                    status=ConstraintValidationStatus.FAIL,
                    evaluation_id=evaluation_id,
                    field_path=("status",),
                )
            )
        _append_validation_issue(
            issues,
            code="INVALID_SCORE",
            message="evaluation score must be a probability",
            status=ConstraintValidationStatus.FAIL,
            evaluation_id=evaluation_id,
            field_path=("score",),
            validator=lambda: ensure_constraint_probability(evaluation.score, "score"),
        )
        _append_validation_issue(
            issues,
            code="INVALID_CONFIDENCE",
            message="evaluation confidence must be a probability",
            status=ConstraintValidationStatus.FAIL,
            evaluation_id=evaluation_id,
            field_path=("confidence",),
            validator=lambda: ensure_constraint_probability(evaluation.confidence, "confidence"),
        )
        if evaluation.violation_severity is not None and not isinstance(
            evaluation.violation_severity,
            ConstraintViolationSeverity,
        ):
            issues.append(
                issue(
                    code="INVALID_VIOLATION_SEVERITY",
                    message="violation_severity must be ConstraintViolationSeverity or None",
                    status=ConstraintValidationStatus.FAIL,
                    evaluation_id=evaluation_id,
                    field_path=("violation_severity",),
                )
            )
        if (
            evaluation.status is ConstraintEvaluationStatus.FAIL
            and evaluation.violation_severity is None
        ):
            issues.append(
                issue(
                    code="FAIL_WITHOUT_SEVERITY",
                    message="FAIL evaluation has no violation severity",
                    status=ConstraintValidationStatus.WARN,
                    evaluation_id=evaluation_id,
                    field_path=("violation_severity",),
                )
            )
        if (
            evaluation.status is ConstraintEvaluationStatus.PASS
            and evaluation.violation_severity is not None
        ):
            issues.append(
                issue(
                    code="PASS_WITH_SEVERITY",
                    message="PASS evaluation has a violation severity",
                    status=ConstraintValidationStatus.WARN,
                    evaluation_id=evaluation_id,
                    field_path=("violation_severity",),
                )
            )
        if evaluation.message_code is None:
            issues.append(
                issue(
                    code="MISSING_MESSAGE_CODE",
                    message="evaluation message_code is missing",
                    status=ConstraintValidationStatus.WARN,
                    evaluation_id=evaluation_id,
                    field_path=("message_code",),
                )
            )
        _append_validation_issue(
            issues,
            code="INVALID_CREATED_AT",
            message="evaluation created_at must be timezone-aware",
            status=ConstraintValidationStatus.FAIL,
            evaluation_id=evaluation_id,
            field_path=("created_at",),
            validator=lambda: ensure_timezone_aware(evaluation.created_at, "created_at"),
        )
        return report_from_issues(issues, checked_count=1)
    except ConstraintValidationError:
        raise
    except Exception as exc:
        raise ConstraintValidationError(
            message="failed to validate constraint evaluation",
            context=build_constraint_error_context(
                operation="validate_constraint_evaluation",
                evaluation_id=evaluation_id,
            ),
            cause=exc,
        ) from exc


def validate_constraint_evaluation_set(
    evaluation_set: ConstraintEvaluationSet,
) -> ConstraintValidationReport:
    """Validate one ConstraintEvaluationSet."""

    observation_id: str | None = None
    try:
        if isinstance(evaluation_set, ConstraintEvaluationSet):
            observation_id = evaluation_set.observation_id
        else:
            raise TypeError("evaluation_set must be ConstraintEvaluationSet")
        if not evaluation_set.evaluations:
            raise EmptyConstraintSetError(message="evaluation_set evaluations must not be empty")

        issues: list[ConstraintValidationIssue] = []
        _append_validation_issue(
            issues,
            code="INVALID_OBSERVATION_ID",
            message="observation_id must be non-empty",
            status=ConstraintValidationStatus.FAIL,
            observation_id=observation_id,
            field_path=("observation_id",),
            validator=lambda: ensure_non_empty_string(evaluation_set.observation_id, "observation_id"),
        )
        seen_ids: set[str] = set()
        for evaluation in evaluation_set.evaluations:
            evaluation_id = str(evaluation.evaluation_id)
            if evaluation_id in seen_ids:
                issues.append(
                    issue(
                        code="DUPLICATE_EVALUATION_ID",
                        message="duplicate evaluation_id in evaluation set",
                        status=ConstraintValidationStatus.FAIL,
                        evaluation_id=evaluation_id,
                        observation_id=observation_id,
                        field_path=("evaluations",),
                    )
                )
            seen_ids.add(evaluation_id)
            issues.extend(validate_constraint_evaluation(evaluation).issues)
        _append_validation_issue(
            issues,
            code="INVALID_CREATED_AT",
            message="evaluation_set created_at must be timezone-aware",
            status=ConstraintValidationStatus.FAIL,
            observation_id=observation_id,
            field_path=("created_at",),
            validator=lambda: ensure_timezone_aware(evaluation_set.created_at, "created_at"),
        )
        return report_from_issues(issues, checked_count=len(evaluation_set.evaluations))
    except EmptyConstraintSetError:
        raise
    except ConstraintValidationError:
        raise
    except Exception as exc:
        raise ConstraintValidationError(
            message="failed to validate constraint evaluation set",
            context=build_constraint_error_context(
                operation="validate_constraint_evaluation_set",
                observation_id=observation_id,
            ),
            cause=exc,
        ) from exc


def validate_constraint_engine_result(
    result: ConstraintEngineResult,
) -> ConstraintValidationReport:
    """Validate one ConstraintEngineResult."""

    try:
        if not isinstance(result, ConstraintEngineResult):
            raise TypeError("result must be ConstraintEngineResult")
        issues: list[ConstraintValidationIssue] = []
        _append_validation_issue(
            issues,
            code="INVALID_SELECTED_CONSTRAINT_COUNT",
            message="selected_constraint_count must be non-negative",
            status=ConstraintValidationStatus.FAIL,
            field_path=("selected_constraint_count",),
            validator=lambda: _ensure_non_negative_int(
                result.selected_constraint_count,
                "selected_constraint_count",
            ),
        )
        _append_validation_issue(
            issues,
            code="INVALID_FEATURE_SET_COUNT",
            message="feature_set_count must be non-negative",
            status=ConstraintValidationStatus.FAIL,
            field_path=("feature_set_count",),
            validator=lambda: _ensure_non_negative_int(result.feature_set_count, "feature_set_count"),
        )
        issues.extend(validate_constraint_evaluation_set(result.evaluation_set).issues)
        if "engine" not in result.metadata:
            issues.append(
                issue(
                    code="MISSING_ENGINE_METADATA",
                    message='engine result metadata must include "engine"',
                    status=ConstraintValidationStatus.FAIL,
                    field_path=("metadata", "engine"),
                )
            )
        return report_from_issues(
            issues,
            checked_count=len(result.evaluation_set.evaluations),
        )
    except EmptyConstraintSetError:
        raise
    except ConstraintValidationError:
        raise
    except Exception as exc:
        raise ConstraintValidationError(
            message="failed to validate constraint engine result",
            context=build_constraint_error_context(operation="validate_constraint_engine_result"),
            cause=exc,
        ) from exc


def combine_reports(
    reports: tuple[ConstraintValidationReport, ...] | list[ConstraintValidationReport],
    *,
    metadata: dict[str, object] | None = None,
) -> ConstraintValidationReport:
    """Combine validation reports preserving issue order."""

    if not isinstance(reports, (tuple, list)):
        raise TypeError("reports must be a tuple or list")
    issues: list[ConstraintValidationIssue] = []
    checked_count = 0
    for report in reports:
        if not isinstance(report, ConstraintValidationReport):
            raise TypeError("reports must contain ConstraintValidationReport")
        issues.extend(report.issues)
        checked_count += report.checked_count
    return report_from_issues(
        tuple(issues),
        checked_count=checked_count,
        metadata={} if metadata is None else metadata,
    )


def _append_validation_issue(
    issues: list[ConstraintValidationIssue],
    *,
    code: str,
    message: str,
    status: ConstraintValidationStatus,
    field_path: tuple[str, ...],
    validator: object,
    constraint_id: str | None = None,
    evaluation_id: str | None = None,
    difference_id: str | None = None,
    observation_id: str | None = None,
) -> None:
    if not callable(validator):
        raise TypeError("validator must be callable")
    try:
        validator()
    except Exception as exc:
        issues.append(
            issue(
                code=code,
                message=message,
                status=status,
                constraint_id=constraint_id,
                evaluation_id=evaluation_id,
                difference_id=difference_id,
                observation_id=observation_id,
                field_path=field_path,
                metadata={
                    "cause_type": type(exc).__name__,
                    "cause_message": str(exc),
                },
            )
        )


def _ensure_optional_non_empty_string(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return ensure_non_empty_string(value, field_name)


def _ensure_non_negative_int(value: int, field_name: str) -> int:
    ensure_non_empty_string(field_name, "field_name")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{field_name} must be greater than or equal to 0")
    return value


def _ensure_non_negative_number(value: float, field_name: str) -> float:
    ensure_non_empty_string(field_name, "field_name")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a non-negative number")
    numeric = float(value)
    if numeric < 0.0:
        raise ValueError(f"{field_name} must be greater than or equal to 0.0")
    return numeric


__all__ = [
    "ConstraintValidationStatus",
    "ConstraintValidationIssue",
    "ConstraintValidationReport",
    "issue",
    "report_from_issues",
    "validate_constraint_target",
    "validate_constraint_threshold",
    "validate_constraint_definition",
    "validate_constraint_registry",
    "validate_constraint_feature",
    "validate_constraint_feature_set",
    "validate_constraint_evaluation",
    "validate_constraint_evaluation_set",
    "validate_constraint_engine_result",
    "combine_reports",
]
