"""Deterministic expected-observation utilities for CEBA World Model."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from ceba.world_model.errors import (
    ExpectedObservationError,
    WorldModelInvariantError,
)
from ceba.world_model.schema import (
    ExpectedObservation,
    ExpectedObservationStatus,
    PredictionResult,
    ensure_non_empty_string,
    ensure_probability,
    utc_now,
)


__all__ = [
    "ExpectedObservationPolicy",
    "ExpectedObservationUpdate",
    "ExpectedObservationSelection",
    "ExpectedObservationManager",
    "require_expected_observation",
    "select_best_expected_observation",
]


@dataclass(frozen=True)
class ExpectedObservationPolicy:
    """Configuration object controlling expected-observation handling."""

    min_confidence: float = 0.0
    max_uncertainty: float = 1.0
    allow_uncertain: bool = True
    allow_degraded: bool = False
    allow_failed: bool = False

    def __post_init__(self) -> None:
        min_confidence = ensure_probability(
            self.min_confidence,
            "min_confidence",
        )
        max_uncertainty = ensure_probability(
            self.max_uncertainty,
            "max_uncertainty",
        )
        if not isinstance(self.allow_uncertain, bool):
            raise TypeError("allow_uncertain must be bool")
        if not isinstance(self.allow_degraded, bool):
            raise TypeError("allow_degraded must be bool")
        if not isinstance(self.allow_failed, bool):
            raise TypeError("allow_failed must be bool")
        object.__setattr__(self, "min_confidence", min_confidence)
        object.__setattr__(self, "max_uncertainty", max_uncertainty)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "min_confidence": self.min_confidence,
            "max_uncertainty": self.max_uncertainty,
            "allow_uncertain": self.allow_uncertain,
            "allow_degraded": self.allow_degraded,
            "allow_failed": self.allow_failed,
        }


@dataclass(frozen=True)
class ExpectedObservationUpdate:
    """Explicit update command for one ExpectedObservation."""

    expected_id: str
    status: ExpectedObservationStatus | None = None
    observation_ref: str | None = None
    confidence: float | None = None
    uncertainty: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected_id = ensure_non_empty_string(self.expected_id, "expected_id")
        _parse_uuid(expected_id, "expected_id")
        if self.status is not None and not isinstance(
            self.status,
            ExpectedObservationStatus,
        ):
            raise TypeError(
                "status must be ExpectedObservationStatus or None"
            )
        observation_ref = (
            None
            if self.observation_ref is None
            else ensure_non_empty_string(
                self.observation_ref,
                "observation_ref",
            )
        )
        confidence = (
            None
            if self.confidence is None
            else ensure_probability(self.confidence, "confidence")
        )
        uncertainty = (
            None
            if self.uncertainty is None
            else ensure_probability(self.uncertainty, "uncertainty")
        )
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "expected_id", expected_id)
        object.__setattr__(self, "observation_ref", observation_ref)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "expected_id": self.expected_id,
            "status": None if self.status is None else self.status.value,
            "observation_ref": self.observation_ref,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ExpectedObservationSelection:
    """Immutable result object for expected-observation selection."""

    selected: ExpectedObservation | None
    candidates: tuple[ExpectedObservation, ...]
    rejected_expected_ids: tuple[str, ...]
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.selected is not None and not isinstance(
            self.selected,
            ExpectedObservation,
        ):
            raise TypeError("selected must be ExpectedObservation or None")
        candidates = _ensure_expected_observations(self.candidates)
        if self.selected is not None:
            candidate_ids = {
                candidate.expected_id for candidate in candidates
            }
            if self.selected.expected_id not in candidate_ids:
                raise ValueError("selected expected_id must exist in candidates")
        rejected_ids = _ensure_string_tuple(
            self.rejected_expected_ids,
            "rejected_expected_ids",
        )
        rejection_reasons = _ensure_string_tuple(
            self.rejection_reasons,
            "rejection_reasons",
        )
        if len(rejected_ids) != len(rejection_reasons):
            raise ValueError(
                "rejected_expected_ids and rejection_reasons must match"
            )
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "rejected_expected_ids", rejected_ids)
        object.__setattr__(self, "rejection_reasons", rejection_reasons)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "selected": None if self.selected is None else self.selected.to_dict(),
            "candidates": tuple(
                candidate.to_dict() for candidate in self.candidates
            ),
            "rejected_expected_ids": self.rejected_expected_ids,
            "rejection_reasons": self.rejection_reasons,
        }

    def candidate_count(self) -> int:
        """Return candidate count."""

        return len(self.candidates)

    def rejected_count(self) -> int:
        """Return rejected expected observation count."""

        return len(self.rejected_expected_ids)

    def has_selected(self) -> bool:
        """Return whether a selected expected observation exists."""

        return self.selected is not None


class ExpectedObservationManager:
    """Service for validating, updating, and selecting expected observations."""

    def __init__(
        self,
        policy: ExpectedObservationPolicy | None = None,
    ) -> None:
        self._policy = (
            ExpectedObservationPolicy() if policy is None else policy
        )
        if not isinstance(self._policy, ExpectedObservationPolicy):
            raise TypeError("policy must be ExpectedObservationPolicy")

    def require(
        self,
        *,
        prediction_result: PredictionResult,
    ) -> ExpectedObservation:
        """Return a usable expected observation or raise."""

        if not isinstance(prediction_result, PredictionResult):
            raise TypeError("prediction_result must be PredictionResult")
        expected = prediction_result.expected_observation
        if expected is None:
            raise WorldModelInvariantError(
                "expected observation required before delta",
                context={"prediction_result_id": str(prediction_result.result_id)},
            )
        reason = self.rejection_reason(expected_observation=expected)
        if reason is not None:
            raise ExpectedObservationError(
                reason,
                context={"expected_id": str(expected.expected_id)},
            )
        return expected

    def is_usable(
        self,
        *,
        expected_observation: ExpectedObservation,
    ) -> bool:
        """Return whether an expected observation is usable."""

        return (
            self.rejection_reason(
                expected_observation=expected_observation,
            )
            is None
        )

    def rejection_reason(
        self,
        *,
        expected_observation: ExpectedObservation,
    ) -> str | None:
        """Return a stable rejection reason or None."""

        if not isinstance(expected_observation, ExpectedObservation):
            raise TypeError(
                "expected_observation must be ExpectedObservation"
            )
        status = expected_observation.status
        if status is ExpectedObservationStatus.INVALID:
            return "expected observation invalid"
        if (
            status is ExpectedObservationStatus.FAILED
            and not self._policy.allow_failed
        ):
            return "expected observation failed"
        if (
            status is ExpectedObservationStatus.DEGRADED
            and not self._policy.allow_degraded
        ):
            return "expected observation degraded"
        if (
            status is ExpectedObservationStatus.UNCERTAIN
            and not self._policy.allow_uncertain
        ):
            return "expected observation uncertain"
        if status not in (
            ExpectedObservationStatus.EXPECTED,
            ExpectedObservationStatus.UNCERTAIN,
            ExpectedObservationStatus.DEGRADED,
            ExpectedObservationStatus.FAILED,
        ):
            return "expected observation invalid"
        if expected_observation.confidence < self._policy.min_confidence:
            return "confidence below expected observation minimum"
        if expected_observation.uncertainty > self._policy.max_uncertainty:
            return "uncertainty above expected observation maximum"
        return None

    def update(
        self,
        *,
        expected_observation: ExpectedObservation,
        update: ExpectedObservationUpdate,
    ) -> ExpectedObservation:
        """Return an updated expected observation."""

        if not isinstance(expected_observation, ExpectedObservation):
            raise TypeError(
                "expected_observation must be ExpectedObservation"
            )
        if not isinstance(update, ExpectedObservationUpdate):
            raise TypeError("update must be ExpectedObservationUpdate")
        if update.expected_id != str(expected_observation.expected_id):
            raise ExpectedObservationError(
                "expected_id mismatch",
                context={
                    "expected_id": str(expected_observation.expected_id),
                    "update_expected_id": update.expected_id,
                },
            )
        metadata = dict(expected_observation.metadata)
        metadata.update(update.metadata)
        metadata.update(
            {
                "source": "expected_observation_manager",
                "operation": "update",
            }
        )
        metadata["source"] = "expected_observation_manager"
        return ExpectedObservation(
            expected_id=expected_observation.expected_id,
            prediction_input_id=expected_observation.prediction_input_id,
            status=(
                expected_observation.status
                if update.status is None
                else update.status
            ),
            observation_ref=(
                expected_observation.observation_ref
                if update.observation_ref is None
                else update.observation_ref
            ),
            confidence=(
                expected_observation.confidence
                if update.confidence is None
                else update.confidence
            ),
            uncertainty=(
                expected_observation.uncertainty
                if update.uncertainty is None
                else update.uncertainty
            ),
            predicted_at=utc_now(),
            metadata=metadata,
        )

    def select_best(
        self,
        *,
        candidates: tuple[ExpectedObservation, ...] | list[ExpectedObservation],
    ) -> ExpectedObservationSelection:
        """Select the best usable expected observation."""

        candidate_values = _ensure_expected_observations(candidates)
        accepted: list[ExpectedObservation] = []
        rejected_ids: list[str] = []
        rejection_reasons: list[str] = []
        for candidate in candidate_values:
            reason = self.rejection_reason(expected_observation=candidate)
            if reason is None:
                accepted.append(candidate)
            else:
                rejected_ids.append(str(candidate.expected_id))
                rejection_reasons.append(reason)

        accepted.sort(
            key=lambda item: (
                -item.confidence,
                item.uncertainty,
                -item.predicted_at.timestamp(),
                str(item.expected_id),
            )
        )
        selected = accepted[0] if accepted else None
        return ExpectedObservationSelection(
            selected=selected,
            candidates=tuple(accepted),
            rejected_expected_ids=tuple(rejected_ids),
            rejection_reasons=tuple(rejection_reasons),
        )


def require_expected_observation(
    *,
    prediction_result: PredictionResult,
    policy: ExpectedObservationPolicy | None = None,
) -> ExpectedObservation:
    """Require a usable expected observation from a prediction result."""

    manager = ExpectedObservationManager(policy)
    return manager.require(prediction_result=prediction_result)


def select_best_expected_observation(
    *,
    candidates: tuple[ExpectedObservation, ...] | list[ExpectedObservation],
    policy: ExpectedObservationPolicy | None = None,
) -> ExpectedObservationSelection:
    """Select the best usable expected observation."""

    manager = ExpectedObservationManager(policy)
    return manager.select_best(candidates=candidates)


def _ensure_expected_observations(
    values: tuple[ExpectedObservation, ...] | list[ExpectedObservation],
) -> tuple[ExpectedObservation, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("candidates must be tuple or list")
    candidates = tuple(values)
    for candidate in candidates:
        if not isinstance(candidate, ExpectedObservation):
            raise TypeError("candidates must contain ExpectedObservation")
    expected_ids = tuple(candidate.expected_id for candidate in candidates)
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("expected_id values must be unique")
    return candidates


def _parse_uuid(value: str, field_name: str) -> UUID:
    try:
        return UUID(ensure_non_empty_string(value, field_name))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be UUID string") from exc


def _ensure_string_tuple(
    values: tuple[str, ...] | list[str],
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple or list")
    return tuple(ensure_non_empty_string(value, field_name) for value in values)


def _copy_metadata(metadata: dict[str, object]) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(metadata)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
    return copied
