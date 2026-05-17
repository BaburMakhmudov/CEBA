"""Deterministic validation utilities for CEBA World Model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar

from ceba.world_model.schema import (
    ExpectedObservation,
    ExpectedObservationStatus,
    PredictionInput,
    PredictionResult,
    PredictionStatus,
    WorldModelEvidence,
    WorldModelSnapshot,
    WorldStateRef,
    WorldStateStatus,
    ensure_non_empty_string,
)


__all__ = [
    "WorldModelValidationIssue",
    "WorldModelValidationReport",
    "WorldModelValidator",
    "validate_world_model_snapshot",
    "validate_world_model_components",
]


_VALID_SEVERITIES: frozenset[str] = frozenset({"INFO", "WARNING", "ERROR"})
_T = TypeVar("_T")


@dataclass(frozen=True)
class WorldModelValidationIssue:
    """Immutable validation issue descriptor."""

    code: str
    message: str
    severity: str
    object_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        code = ensure_non_empty_string(self.code, "code")
        message = ensure_non_empty_string(self.message, "message")
        severity = ensure_non_empty_string(self.severity, "severity")
        if severity not in _VALID_SEVERITIES:
            raise ValueError("severity must be INFO, WARNING, or ERROR")
        object_id = (
            None
            if self.object_id is None
            else ensure_non_empty_string(self.object_id, "object_id")
        )
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "object_id": self.object_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class WorldModelValidationReport:
    """Immutable validation report."""

    valid: bool
    issues: tuple[WorldModelValidationIssue, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be bool")
        issues = _ensure_issues(self.issues)
        if self.valid and any(issue.severity == "ERROR" for issue in issues):
            raise ValueError("valid must be False when ERROR issues exist")
        object.__setattr__(self, "issues", issues)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "valid": self.valid,
            "issues": tuple(issue.to_dict() for issue in self.issues),
        }

    def error_count(self) -> int:
        """Return the number of error issues."""

        return _count_severity(self.issues, "ERROR")

    def warning_count(self) -> int:
        """Return the number of warning issues."""

        return _count_severity(self.issues, "WARNING")

    def info_count(self) -> int:
        """Return the number of info issues."""

        return _count_severity(self.issues, "INFO")


class WorldModelValidator:
    """Deterministic validator for world model runtime structures."""

    def __init__(self) -> None:
        """Create a stateless world model validator."""

    def validate_components(
        self,
        *,
        states: tuple[WorldStateRef, ...] | list[WorldStateRef] = (),
        evidence: tuple[WorldModelEvidence, ...] | list[WorldModelEvidence] = (),
        prediction_inputs: (
            tuple[PredictionInput, ...] | list[PredictionInput]
        ) = (),
        prediction_results: (
            tuple[PredictionResult, ...] | list[PredictionResult]
        ) = (),
    ) -> WorldModelValidationReport:
        """Validate world model component consistency."""

        state_values = _ensure_collection(states, WorldStateRef, "states")
        evidence_values = _ensure_collection(
            evidence,
            WorldModelEvidence,
            "evidence",
        )
        input_values = _ensure_collection(
            prediction_inputs,
            PredictionInput,
            "prediction_inputs",
        )
        result_values = _ensure_collection(
            prediction_results,
            PredictionResult,
            "prediction_results",
        )

        issues: list[WorldModelValidationIssue] = []
        self._validate_empty_collections(
            issues=issues,
            states=state_values,
            evidence=evidence_values,
            prediction_inputs=input_values,
            prediction_results=result_values,
        )
        self._validate_uniqueness(
            issues=issues,
            states=state_values,
            evidence=evidence_values,
            prediction_inputs=input_values,
            prediction_results=result_values,
        )
        self._validate_references(
            issues=issues,
            states=state_values,
            evidence=evidence_values,
            prediction_inputs=input_values,
            prediction_results=result_values,
        )
        self._validate_statuses(
            issues=issues,
            states=state_values,
            evidence=evidence_values,
            prediction_results=result_values,
        )
        return _create_report(tuple(issues))

    def validate_snapshot(
        self,
        *,
        snapshot: WorldModelSnapshot,
    ) -> WorldModelValidationReport:
        """Validate a world model snapshot."""

        if not isinstance(snapshot, WorldModelSnapshot):
            raise TypeError("snapshot must be WorldModelSnapshot")
        return self.validate_components(
            states=snapshot.states,
            evidence=snapshot.evidence,
            prediction_inputs=snapshot.prediction_inputs,
            prediction_results=snapshot.prediction_results,
        )

    def _validate_empty_collections(
        self,
        *,
        issues: list[WorldModelValidationIssue],
        states: tuple[WorldStateRef, ...],
        evidence: tuple[WorldModelEvidence, ...],
        prediction_inputs: tuple[PredictionInput, ...],
        prediction_results: tuple[PredictionResult, ...],
    ) -> None:
        if not states:
            issues.append(
                _issue(
                    code="EMPTY_STATES",
                    message="empty states collection",
                    severity="INFO",
                )
            )
        if not evidence:
            issues.append(
                _issue(
                    code="EMPTY_EVIDENCE",
                    message="empty evidence collection",
                    severity="INFO",
                )
            )
        if not prediction_inputs:
            issues.append(
                _issue(
                    code="EMPTY_PREDICTION_INPUTS",
                    message="empty prediction_inputs collection",
                    severity="INFO",
                )
            )
        if not prediction_results:
            issues.append(
                _issue(
                    code="EMPTY_PREDICTION_RESULTS",
                    message="empty prediction_results collection",
                    severity="INFO",
                )
            )

    def _validate_uniqueness(
        self,
        *,
        issues: list[WorldModelValidationIssue],
        states: tuple[WorldStateRef, ...],
        evidence: tuple[WorldModelEvidence, ...],
        prediction_inputs: tuple[PredictionInput, ...],
        prediction_results: tuple[PredictionResult, ...],
    ) -> None:
        _add_duplicate_issues(
            issues=issues,
            ids=tuple(str(state.state_id) for state in states),
            code="DUPLICATE_STATE_ID",
            message="duplicate state_id",
        )
        _add_duplicate_issues(
            issues=issues,
            ids=tuple(str(item.evidence_id) for item in evidence),
            code="DUPLICATE_EVIDENCE_ID",
            message="duplicate evidence_id",
        )
        _add_duplicate_issues(
            issues=issues,
            ids=tuple(str(item.input_id) for item in prediction_inputs),
            code="DUPLICATE_PREDICTION_INPUT_ID",
            message="duplicate prediction input_id",
        )
        _add_duplicate_issues(
            issues=issues,
            ids=tuple(str(item.result_id) for item in prediction_results),
            code="DUPLICATE_PREDICTION_RESULT_ID",
            message="duplicate prediction result_id",
        )

    def _validate_references(
        self,
        *,
        issues: list[WorldModelValidationIssue],
        states: tuple[WorldStateRef, ...],
        evidence: tuple[WorldModelEvidence, ...],
        prediction_inputs: tuple[PredictionInput, ...],
        prediction_results: tuple[PredictionResult, ...],
    ) -> None:
        state_ids = {str(state.state_id) for state in states}
        evidence_ids = {str(item.evidence_id) for item in evidence}
        input_ids = {str(item.input_id) for item in prediction_inputs}
        check_evidence_refs = bool(evidence)

        for prediction_input in prediction_inputs:
            if prediction_input.current_state is not None:
                state_id = str(prediction_input.current_state.state_id)
                if state_id not in state_ids:
                    issues.append(
                        _issue(
                            code="PREDICTION_INPUT_STATE_MISSING",
                            message=(
                                "PredictionInput.current_state missing from states"
                            ),
                            severity="ERROR",
                            object_id=str(prediction_input.input_id),
                            metadata={"state_id": state_id},
                        )
                    )
            if check_evidence_refs:
                for item in prediction_input.evidence:
                    evidence_id = str(item.evidence_id)
                    if evidence_id not in evidence_ids:
                        issues.append(
                            _issue(
                                code="PREDICTION_INPUT_EVIDENCE_MISSING",
                                message=(
                                    "PredictionInput.evidence item missing "
                                    "from evidence"
                                ),
                                severity="ERROR",
                                object_id=str(prediction_input.input_id),
                                metadata={"evidence_id": evidence_id},
                            )
                        )

        for prediction_result in prediction_results:
            input_id = str(prediction_result.prediction_input.input_id)
            if input_id not in input_ids:
                issues.append(
                    _issue(
                        code="PREDICTION_RESULT_INPUT_MISSING",
                        message=(
                            "PredictionResult.prediction_input missing "
                            "from prediction_inputs"
                        ),
                        severity="ERROR",
                        object_id=str(prediction_result.result_id),
                        metadata={"prediction_input_id": input_id},
                    )
                )
            if check_evidence_refs:
                for item in prediction_result.evidence:
                    evidence_id = str(item.evidence_id)
                    if evidence_id not in evidence_ids:
                        issues.append(
                            _issue(
                                code="PREDICTION_RESULT_EVIDENCE_MISSING",
                                message=(
                                    "PredictionResult.evidence item missing "
                                    "from evidence"
                                ),
                                severity="ERROR",
                                object_id=str(prediction_result.result_id),
                                metadata={"evidence_id": evidence_id},
                            )
                        )
            self._validate_expected_observation_reference(
                issues=issues,
                prediction_result=prediction_result,
            )
            self._validate_world_model_before_delta(
                issues=issues,
                prediction_result=prediction_result,
            )

    def _validate_expected_observation_reference(
        self,
        *,
        issues: list[WorldModelValidationIssue],
        prediction_result: PredictionResult,
    ) -> None:
        expected = prediction_result.expected_observation
        if expected is None:
            return
        if expected.prediction_input_id != prediction_result.prediction_input.input_id:
            issues.append(
                _issue(
                    code="EXPECTED_OBSERVATION_INPUT_MISMATCH",
                    message="ExpectedObservation.prediction_input_id mismatch",
                    severity="ERROR",
                    object_id=str(expected.expected_id),
                    metadata={
                        "prediction_result_id": str(prediction_result.result_id),
                        "expected_prediction_input_id": str(
                            expected.prediction_input_id
                        ),
                        "result_prediction_input_id": str(
                            prediction_result.prediction_input.input_id
                        ),
                    },
                )
            )

    def _validate_world_model_before_delta(
        self,
        *,
        issues: list[WorldModelValidationIssue],
        prediction_result: PredictionResult,
    ) -> None:
        if (
            prediction_result.status == PredictionStatus.PREDICTED
            and prediction_result.expected_observation is None
        ):
            issues.append(
                _issue(
                    code="PREDICTED_RESULT_MISSING_EXPECTED_OBSERVATION",
                    message=(
                        "PredictionResult.status PREDICTED without "
                        "expected_observation"
                    ),
                    severity="ERROR",
                    object_id=str(prediction_result.result_id),
                    metadata={"invariant": "WORLD_MODEL_BEFORE_DELTA"},
                )
            )
        if (
            prediction_result.status == PredictionStatus.LOW_CONFIDENCE
            and prediction_result.expected_observation is None
        ):
            issues.append(
                _issue(
                    code="LOW_CONFIDENCE_RESULT_MISSING_EXPECTED_OBSERVATION",
                    message=(
                        "PredictionResult.status LOW_CONFIDENCE without "
                        "expected_observation"
                    ),
                    severity="WARNING",
                    object_id=str(prediction_result.result_id),
                    metadata={"invariant": "WORLD_MODEL_BEFORE_DELTA"},
                )
            )

    def _validate_statuses(
        self,
        *,
        issues: list[WorldModelValidationIssue],
        states: tuple[WorldStateRef, ...],
        evidence: tuple[WorldModelEvidence, ...],
        prediction_results: tuple[PredictionResult, ...],
    ) -> None:
        for state in states:
            if state.status == WorldStateStatus.INVALID:
                issues.append(
                    _issue(
                        code="WORLD_STATE_INVALID",
                        message="WorldStateRef.status INVALID",
                        severity="WARNING",
                        object_id=str(state.state_id),
                    )
                )
            if state.status == WorldStateStatus.STALE:
                issues.append(
                    _issue(
                        code="WORLD_STATE_STALE",
                        message="WorldStateRef.status STALE",
                        severity="WARNING",
                        object_id=str(state.state_id),
                    )
                )
            if state.status == WorldStateStatus.DEGRADED:
                issues.append(
                    _issue(
                        code="WORLD_STATE_DEGRADED",
                        message="WorldStateRef.status DEGRADED",
                        severity="WARNING",
                        object_id=str(state.state_id),
                    )
                )

        for item in evidence:
            if item.confidence == 0.0:
                issues.append(
                    _issue(
                        code="WORLD_MODEL_EVIDENCE_ZERO_CONFIDENCE",
                        message="WorldModelEvidence.confidence == 0.0",
                        severity="WARNING",
                        object_id=str(item.evidence_id),
                    )
                )
            if item.relevance == 0.0:
                issues.append(
                    _issue(
                        code="WORLD_MODEL_EVIDENCE_ZERO_RELEVANCE",
                        message="WorldModelEvidence.relevance == 0.0",
                        severity="WARNING",
                        object_id=str(item.evidence_id),
                    )
                )

        for prediction_result in prediction_results:
            self._validate_prediction_result_status(
                issues=issues,
                prediction_result=prediction_result,
            )
            if prediction_result.expected_observation is not None:
                self._validate_expected_observation_status(
                    issues=issues,
                    expected_observation=prediction_result.expected_observation,
                )

    def _validate_prediction_result_status(
        self,
        *,
        issues: list[WorldModelValidationIssue],
        prediction_result: PredictionResult,
    ) -> None:
        if prediction_result.status == PredictionStatus.INVALID:
            issues.append(
                _issue(
                    code="PREDICTION_RESULT_INVALID",
                    message="PredictionResult.status INVALID",
                    severity="WARNING",
                    object_id=str(prediction_result.result_id),
                )
            )
        if prediction_result.status == PredictionStatus.FAILED:
            issues.append(
                _issue(
                    code="PREDICTION_RESULT_FAILED",
                    message="PredictionResult.status FAILED",
                    severity="WARNING",
                    object_id=str(prediction_result.result_id),
                )
            )

    def _validate_expected_observation_status(
        self,
        *,
        issues: list[WorldModelValidationIssue],
        expected_observation: ExpectedObservation,
    ) -> None:
        if expected_observation.status == ExpectedObservationStatus.INVALID:
            issues.append(
                _issue(
                    code="EXPECTED_OBSERVATION_INVALID",
                    message="ExpectedObservation.status INVALID",
                    severity="WARNING",
                    object_id=str(expected_observation.expected_id),
                )
            )
        if expected_observation.status == ExpectedObservationStatus.FAILED:
            issues.append(
                _issue(
                    code="EXPECTED_OBSERVATION_FAILED",
                    message="ExpectedObservation.status FAILED",
                    severity="WARNING",
                    object_id=str(expected_observation.expected_id),
                )
            )
        if expected_observation.status == ExpectedObservationStatus.DEGRADED:
            issues.append(
                _issue(
                    code="EXPECTED_OBSERVATION_DEGRADED",
                    message="ExpectedObservation.status DEGRADED",
                    severity="WARNING",
                    object_id=str(expected_observation.expected_id),
                )
            )
        if expected_observation.status == ExpectedObservationStatus.UNCERTAIN:
            issues.append(
                _issue(
                    code="EXPECTED_OBSERVATION_UNCERTAIN",
                    message="ExpectedObservation.status UNCERTAIN",
                    severity="WARNING",
                    object_id=str(expected_observation.expected_id),
                )
            )


def validate_world_model_snapshot(
    *,
    snapshot: WorldModelSnapshot,
) -> WorldModelValidationReport:
    """Validate a world model snapshot."""

    validator = WorldModelValidator()
    return validator.validate_snapshot(snapshot=snapshot)


def validate_world_model_components(
    *,
    states: tuple[WorldStateRef, ...] | list[WorldStateRef] = (),
    evidence: tuple[WorldModelEvidence, ...] | list[WorldModelEvidence] = (),
    prediction_inputs: tuple[PredictionInput, ...] | list[PredictionInput] = (),
    prediction_results: tuple[PredictionResult, ...] | list[PredictionResult] = (),
) -> WorldModelValidationReport:
    """Validate world model components."""

    validator = WorldModelValidator()
    return validator.validate_components(
        states=states,
        evidence=evidence,
        prediction_inputs=prediction_inputs,
        prediction_results=prediction_results,
    )


def _ensure_issues(
    values: tuple[WorldModelValidationIssue, ...] | list[WorldModelValidationIssue],
) -> tuple[WorldModelValidationIssue, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("issues must be tuple or list")
    issues = tuple(values)
    for issue in issues:
        if not isinstance(issue, WorldModelValidationIssue):
            raise TypeError("issues must contain WorldModelValidationIssue")
    return issues


def _ensure_collection(
    values: tuple[_T, ...] | list[_T],
    expected_type: type[_T],
    field_name: str,
) -> tuple[_T, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple or list")
    items = tuple(values)
    for item in items:
        if not isinstance(item, expected_type):
            raise TypeError(f"{field_name} must contain {expected_type.__name__}")
    return items


def _add_duplicate_issues(
    *,
    issues: list[WorldModelValidationIssue],
    ids: tuple[str, ...],
    code: str,
    message: str,
) -> None:
    seen: set[str] = set()
    reported: set[str] = set()
    for object_id in ids:
        if object_id in seen and object_id not in reported:
            issues.append(
                _issue(
                    code=code,
                    message=message,
                    severity="ERROR",
                    object_id=object_id,
                )
            )
            reported.add(object_id)
        seen.add(object_id)


def _issue(
    *,
    code: str,
    message: str,
    severity: str,
    object_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> WorldModelValidationIssue:
    return WorldModelValidationIssue(
        code=code,
        message=message,
        severity=severity,
        object_id=object_id,
        metadata={} if metadata is None else metadata,
    )


def _create_report(
    issues: tuple[WorldModelValidationIssue, ...],
) -> WorldModelValidationReport:
    valid = not any(issue.severity == "ERROR" for issue in issues)
    return WorldModelValidationReport(valid=valid, issues=issues)


def _count_severity(
    issues: tuple[WorldModelValidationIssue, ...],
    severity: str,
) -> int:
    return sum(1 for issue in issues if issue.severity == severity)


def _copy_metadata(metadata: dict[str, object]) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(metadata)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
    return copied
