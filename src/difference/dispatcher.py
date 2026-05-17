"""Deterministic dispatch utilities for CEBA difference processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib

from ois.difference.aggregator import DifferenceAggregateSummary, severity_rank
from ois.difference.errors import DifferenceDispatchError, build_error_context
from ois.difference.schema import (
    DifferenceClassification,
    DifferenceSet,
    DifferenceSeverity,
    ensure_timezone_aware,
    utc_now,
)
from ois.difference.validation import (
    ValidationReport,
    ValidationStatus,
    validate_dispatch_ready,
)


class DifferenceDispatchTarget(str, Enum):
    """Stable downstream targets for difference dispatch."""

    CONSTRAINT_ENGINE = "CONSTRAINT_ENGINE"
    METRIC_ENGINE = "METRIC_ENGINE"
    WORLD_MODEL = "WORLD_MODEL"
    MEMORY_ORCHESTRATOR = "MEMORY_ORCHESTRATOR"
    AUDIT = "AUDIT"
    DEAD_LETTER = "DEAD_LETTER"


class DifferenceDispatchStatus(str, Enum):
    """Stable dispatch lifecycle status."""

    READY = "READY"
    BLOCKED = "BLOCKED"
    DEAD_LETTER = "DEAD_LETTER"


@dataclass(frozen=True, slots=True)
class DifferenceDispatchEnvelope:
    """Immutable dispatch package for validated difference artifacts."""

    envelope_id: str
    observation_id: str
    target: DifferenceDispatchTarget
    status: DifferenceDispatchStatus
    difference_set: DifferenceSet
    classifications: tuple[DifferenceClassification, ...]
    summary: DifferenceAggregateSummary
    validation_report: ValidationReport
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_non_empty_string(self.envelope_id, "envelope_id")
        _ensure_non_empty_string(self.observation_id, "observation_id")
        if not isinstance(self.target, DifferenceDispatchTarget):
            raise TypeError("target must be DifferenceDispatchTarget")
        if not isinstance(self.status, DifferenceDispatchStatus):
            raise TypeError("status must be DifferenceDispatchStatus")
        if not isinstance(self.difference_set, DifferenceSet):
            raise TypeError("difference_set must be DifferenceSet")
        if not isinstance(self.classifications, tuple):
            if isinstance(self.classifications, list):
                object.__setattr__(self, "classifications", tuple(self.classifications))
            else:
                raise TypeError("classifications must be a tuple of DifferenceClassification")
        for classification in self.classifications:
            if not isinstance(classification, DifferenceClassification):
                raise TypeError("classifications must contain DifferenceClassification")
        if not isinstance(self.summary, DifferenceAggregateSummary):
            raise TypeError("summary must be DifferenceAggregateSummary")
        if not isinstance(self.validation_report, ValidationReport):
            raise TypeError("validation_report must be ValidationReport")
        ensure_timezone_aware(self.created_at, "created_at")
        _ensure_metadata_dict(self.metadata)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serialization-friendly shallow representation."""

        return {
            "envelope_id": self.envelope_id,
            "observation_id": self.observation_id,
            "target": self.target.value,
            "status": self.status.value,
            "difference_set_id": str(self.difference_set.set_id),
            "classification_count": len(self.classifications),
            "total_candidates": self.summary.total_candidates,
            "max_severity": (
                self.summary.max_severity.value
                if self.summary.max_severity is not None
                else None
            ),
            "validation_status": self.validation_report.status.value,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


def build_envelope_id(
    *,
    observation_id: str,
    target: DifferenceDispatchTarget,
    summary: DifferenceAggregateSummary,
) -> str:
    """Build a deterministic dispatch envelope identifier."""

    _ensure_non_empty_string(observation_id, "observation_id")
    if not isinstance(target, DifferenceDispatchTarget):
        raise TypeError("target must be DifferenceDispatchTarget")
    if not isinstance(summary, DifferenceAggregateSummary):
        raise TypeError("summary must be DifferenceAggregateSummary")

    aggregation_mode = summary.metadata.get("aggregation_mode", "")
    if not isinstance(aggregation_mode, str):
        aggregation_mode = ""

    components = (
        observation_id,
        target.value,
        str(summary.total_candidates),
        str(summary.total_classifications),
        summary.max_severity.value if summary.max_severity is not None else "",
        aggregation_mode,
    )

    digest = hashlib.sha256()
    for component in components:
        digest.update(component.encode("utf-8"))
        digest.update(b"\x00")
    return "dispatch_" + digest.hexdigest()[:32]


def select_dispatch_target(
    summary: DifferenceAggregateSummary,
    validation_report: ValidationReport,
) -> DifferenceDispatchTarget:
    """Select a downstream target deterministically."""

    if not isinstance(summary, DifferenceAggregateSummary):
        raise TypeError("summary must be DifferenceAggregateSummary")
    if not isinstance(validation_report, ValidationReport):
        raise TypeError("validation_report must be ValidationReport")

    if validation_report.status is ValidationStatus.FAIL:
        return DifferenceDispatchTarget.DEAD_LETTER
    if summary.max_severity is None:
        return DifferenceDispatchTarget.AUDIT
    if not isinstance(summary.max_severity, DifferenceSeverity):
        raise TypeError("summary.max_severity must be DifferenceSeverity or None")

    rank = severity_rank(summary.max_severity)
    if rank >= severity_rank(DifferenceSeverity.HIGH):
        return DifferenceDispatchTarget.CONSTRAINT_ENGINE
    if summary.max_severity is DifferenceSeverity.MEDIUM:
        return DifferenceDispatchTarget.METRIC_ENGINE
    return DifferenceDispatchTarget.MEMORY_ORCHESTRATOR


def dispatch_status_for(
    validation_report: ValidationReport,
    target: DifferenceDispatchTarget,
) -> DifferenceDispatchStatus:
    """Determine dispatch status."""

    if not isinstance(validation_report, ValidationReport):
        raise TypeError("validation_report must be ValidationReport")
    if not isinstance(target, DifferenceDispatchTarget):
        raise TypeError("target must be DifferenceDispatchTarget")

    if validation_report.status is ValidationStatus.FAIL:
        return DifferenceDispatchStatus.DEAD_LETTER
    if target is DifferenceDispatchTarget.DEAD_LETTER:
        return DifferenceDispatchStatus.DEAD_LETTER
    if validation_report.status in (ValidationStatus.WARN, ValidationStatus.PASS):
        return DifferenceDispatchStatus.READY
    return DifferenceDispatchStatus.BLOCKED


def create_dispatch_envelope(
    *,
    difference_set: DifferenceSet,
    classifications: tuple[DifferenceClassification, ...] | list[DifferenceClassification],
    summary: DifferenceAggregateSummary,
    validation_report: ValidationReport | None = None,
    target: DifferenceDispatchTarget | None = None,
    metadata: dict[str, object] | None = None,
) -> DifferenceDispatchEnvelope:
    """Create a validated dispatch envelope."""

    observation_id: str | None = None
    try:
        if isinstance(difference_set, DifferenceSet):
            observation_id = difference_set.observation_id
        else:
            raise TypeError("difference_set must be DifferenceSet")
        if not isinstance(classifications, (tuple, list)):
            raise TypeError("classifications must be a tuple or list")
        classifications_tuple = tuple(classifications)
        for classification in classifications_tuple:
            if not isinstance(classification, DifferenceClassification):
                raise TypeError("classifications must contain DifferenceClassification")
        if not isinstance(summary, DifferenceAggregateSummary):
            raise TypeError("summary must be DifferenceAggregateSummary")
        if metadata is not None:
            _ensure_metadata_dict(metadata)

        resolved_validation_report = (
            validate_dispatch_ready(
                difference_set=difference_set,
                classifications=classifications_tuple,
                summary=summary,
            )
            if validation_report is None
            else validation_report
        )
        if not isinstance(resolved_validation_report, ValidationReport):
            raise TypeError("validation_report must be ValidationReport or None")

        resolved_target = (
            select_dispatch_target(summary, resolved_validation_report)
            if target is None
            else target
        )
        if not isinstance(resolved_target, DifferenceDispatchTarget):
            raise TypeError("target must be DifferenceDispatchTarget or None")

        return DifferenceDispatchEnvelope(
            envelope_id=build_envelope_id(
                observation_id=difference_set.observation_id,
                target=resolved_target,
                summary=summary,
            ),
            observation_id=difference_set.observation_id,
            target=resolved_target,
            status=dispatch_status_for(resolved_validation_report, resolved_target),
            difference_set=difference_set,
            classifications=classifications_tuple,
            summary=summary,
            validation_report=resolved_validation_report,
            created_at=utc_now(),
            metadata={} if metadata is None else metadata,
        )
    except DifferenceDispatchError:
        raise
    except Exception as exc:
        raise DifferenceDispatchError(
            message="failed to create dispatch envelope",
            context=build_error_context(
                operation="create_dispatch_envelope",
                observation_id=observation_id,
            ),
            cause=exc,
        ) from exc


def create_dead_letter_envelope(
    *,
    difference_set: DifferenceSet,
    classifications: tuple[DifferenceClassification, ...] | list[DifferenceClassification],
    summary: DifferenceAggregateSummary,
    validation_report: ValidationReport,
    metadata: dict[str, object] | None = None,
) -> DifferenceDispatchEnvelope:
    """Create an explicit dead-letter envelope."""

    resolved_metadata = {} if metadata is None else dict(metadata)
    resolved_metadata["dead_letter"] = True
    return create_dispatch_envelope(
        difference_set=difference_set,
        classifications=classifications,
        summary=summary,
        validation_report=validation_report,
        target=DifferenceDispatchTarget.DEAD_LETTER,
        metadata=resolved_metadata,
    )


def envelope_is_ready(envelope: DifferenceDispatchEnvelope) -> bool:
    """Return whether an envelope may be passed downstream."""

    if not isinstance(envelope, DifferenceDispatchEnvelope):
        raise TypeError("envelope must be DifferenceDispatchEnvelope")
    return envelope.status is DifferenceDispatchStatus.READY


def envelope_is_dead_letter(envelope: DifferenceDispatchEnvelope) -> bool:
    """Return whether an envelope is a dead-letter artifact."""

    if not isinstance(envelope, DifferenceDispatchEnvelope):
        raise TypeError("envelope must be DifferenceDispatchEnvelope")
    return (
        envelope.status is DifferenceDispatchStatus.DEAD_LETTER
        or envelope.target is DifferenceDispatchTarget.DEAD_LETTER
    )


def split_envelopes_by_status(
    envelopes: tuple[DifferenceDispatchEnvelope, ...] | list[DifferenceDispatchEnvelope],
) -> dict[str, tuple[DifferenceDispatchEnvelope, ...]]:
    """Split envelopes into deterministic status groups."""

    if not isinstance(envelopes, (tuple, list)):
        raise TypeError("envelopes must be a tuple or list")

    ready: list[DifferenceDispatchEnvelope] = []
    blocked: list[DifferenceDispatchEnvelope] = []
    dead_letter: list[DifferenceDispatchEnvelope] = []

    for envelope in envelopes:
        if not isinstance(envelope, DifferenceDispatchEnvelope):
            raise TypeError("envelopes must contain DifferenceDispatchEnvelope")
        if envelope.status is DifferenceDispatchStatus.READY:
            ready.append(envelope)
        elif envelope.status is DifferenceDispatchStatus.BLOCKED:
            blocked.append(envelope)
        elif envelope.status is DifferenceDispatchStatus.DEAD_LETTER:
            dead_letter.append(envelope)
        else:
            raise TypeError("envelope status must be DifferenceDispatchStatus")

    return {
        "ready": tuple(ready),
        "blocked": tuple(blocked),
        "dead_letter": tuple(dead_letter),
    }


def _ensure_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(field_name, str) or not field_name:
        raise ValueError("field_name must be a non-empty string")
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _ensure_metadata_dict(value: dict[str, object]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("metadata must be a dict")
    if not all(isinstance(key, str) for key in value):
        raise TypeError("metadata keys must be strings")
    return value


__all__ = [
    "DifferenceDispatchTarget",
    "DifferenceDispatchStatus",
    "DifferenceDispatchEnvelope",
    "build_envelope_id",
    "select_dispatch_target",
    "dispatch_status_for",
    "create_dispatch_envelope",
    "create_dead_letter_envelope",
    "envelope_is_ready",
    "envelope_is_dead_letter",
    "split_envelopes_by_status",
]
