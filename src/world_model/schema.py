"""Strict schema objects for CEBA World Model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from uuid import UUID, uuid4


__all__ = [
    "WorldStateKind",
    "WorldStateStatus",
    "PredictionKind",
    "PredictionStatus",
    "ExpectedObservationStatus",
    "WorldModelEvidenceKind",
    "WorldStateRef",
    "WorldModelEvidence",
    "PredictionInput",
    "ExpectedObservation",
    "PredictionResult",
    "WorldModelSnapshot",
    "utc_now",
    "new_world_state_id",
    "new_world_model_evidence_id",
    "new_prediction_input_id",
    "new_expected_observation_id",
    "new_prediction_result_id",
    "new_world_model_snapshot_id",
    "ensure_non_empty_string",
    "ensure_optional_non_empty_string",
    "ensure_probability",
    "ensure_timezone_aware",
    "create_world_state_ref",
    "create_world_model_evidence",
    "create_prediction_input",
    "create_expected_observation",
    "create_prediction_result",
    "create_world_model_snapshot",
]


class WorldStateKind(Enum):
    """Stable categories for world state references."""

    OBSERVATIONAL = "OBSERVATIONAL"
    STRUCTURAL = "STRUCTURAL"
    TEMPORAL = "TEMPORAL"
    RELATIONAL = "RELATIONAL"
    CAUSAL = "CAUSAL"
    CONTEXTUAL = "CONTEXTUAL"
    UNKNOWN = "UNKNOWN"


class WorldStateStatus(Enum):
    """Stable status values for world state references."""

    ACTIVE = "ACTIVE"
    STALE = "STALE"
    DEGRADED = "DEGRADED"
    SUPERSEDED = "SUPERSEDED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class PredictionKind(Enum):
    """Stable categories for prediction inputs."""

    EXPECTED_OBSERVATION = "EXPECTED_OBSERVATION"
    EXPECTED_STATE = "EXPECTED_STATE"
    EXPECTED_RELATION = "EXPECTED_RELATION"
    EXPECTED_CONSTRAINT = "EXPECTED_CONSTRAINT"
    EXPECTED_TRANSITION = "EXPECTED_TRANSITION"
    UNKNOWN = "UNKNOWN"


class PredictionStatus(Enum):
    """Stable status values for prediction results."""

    PENDING = "PENDING"
    PREDICTED = "PREDICTED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    FAILED = "FAILED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class ExpectedObservationStatus(Enum):
    """Stable status values for expected observations."""

    EXPECTED = "EXPECTED"
    UNCERTAIN = "UNCERTAIN"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class WorldModelEvidenceKind(Enum):
    """Stable categories for world model evidence."""

    MEMORY = "MEMORY"
    RETRIEVAL = "RETRIEVAL"
    CONTEXT = "CONTEXT"
    PRIOR_STATE = "PRIOR_STATE"
    RULE = "RULE"
    OBSERVATION = "OBSERVATION"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class WorldStateRef:
    """Lightweight reference to a runtime world state."""

    state_id: UUID
    kind: WorldStateKind
    status: WorldStateStatus
    state_ref: str
    confidence: float
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.state_id, UUID):
            raise TypeError("state_id must be UUID")
        if not isinstance(self.kind, WorldStateKind):
            raise TypeError("kind must be WorldStateKind")
        if not isinstance(self.status, WorldStateStatus):
            raise TypeError("status must be WorldStateStatus")
        state_ref = ensure_non_empty_string(self.state_ref, "state_ref")
        confidence = ensure_probability(self.confidence, "confidence")
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        updated_at = ensure_timezone_aware(self.updated_at, "updated_at")
        if updated_at < created_at:
            raise ValueError("updated_at must be >= created_at")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "state_ref", state_ref)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "state_id": str(self.state_id),
            "kind": self.kind.value,
            "status": self.status.value,
            "state_ref": self.state_ref,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class WorldModelEvidence:
    """Explicit evidence used by world model prediction."""

    evidence_id: UUID
    kind: WorldModelEvidenceKind
    source_id: str
    evidence_ref: str
    relevance: float
    confidence: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_id, UUID):
            raise TypeError("evidence_id must be UUID")
        if not isinstance(self.kind, WorldModelEvidenceKind):
            raise TypeError("kind must be WorldModelEvidenceKind")
        source_id = ensure_non_empty_string(self.source_id, "source_id")
        evidence_ref = ensure_non_empty_string(self.evidence_ref, "evidence_ref")
        relevance = ensure_probability(self.relevance, "relevance")
        confidence = ensure_probability(self.confidence, "confidence")
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "evidence_ref", evidence_ref)
        object.__setattr__(self, "relevance", relevance)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "evidence_id": str(self.evidence_id),
            "kind": self.kind.value,
            "source_id": self.source_id,
            "evidence_ref": self.evidence_ref,
            "relevance": self.relevance,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PredictionInput:
    """Explicit input descriptor for world model prediction."""

    input_id: UUID
    kind: PredictionKind
    context_ref: str
    current_state: WorldStateRef | None
    evidence: tuple[WorldModelEvidence, ...]
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.input_id, UUID):
            raise TypeError("input_id must be UUID")
        if not isinstance(self.kind, PredictionKind):
            raise TypeError("kind must be PredictionKind")
        context_ref = ensure_non_empty_string(self.context_ref, "context_ref")
        if self.current_state is not None and not isinstance(
            self.current_state,
            WorldStateRef,
        ):
            raise TypeError("current_state must be WorldStateRef or None")
        evidence = _ensure_evidence(self.evidence)
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "context_ref", context_ref)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "input_id": str(self.input_id),
            "kind": self.kind.value,
            "context_ref": self.context_ref,
            "current_state": (
                None if self.current_state is None else self.current_state.to_dict()
            ),
            "evidence": tuple(item.to_dict() for item in self.evidence),
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ExpectedObservation:
    """Predicted observation produced before delta comparison."""

    expected_id: UUID
    prediction_input_id: UUID
    status: ExpectedObservationStatus
    observation_ref: str
    confidence: float
    uncertainty: float
    predicted_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.expected_id, UUID):
            raise TypeError("expected_id must be UUID")
        if not isinstance(self.prediction_input_id, UUID):
            raise TypeError("prediction_input_id must be UUID")
        if not isinstance(self.status, ExpectedObservationStatus):
            raise TypeError("status must be ExpectedObservationStatus")
        observation_ref = ensure_non_empty_string(
            self.observation_ref,
            "observation_ref",
        )
        confidence = ensure_probability(self.confidence, "confidence")
        uncertainty = ensure_probability(self.uncertainty, "uncertainty")
        predicted_at = ensure_timezone_aware(self.predicted_at, "predicted_at")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "observation_ref", observation_ref)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "predicted_at", predicted_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "expected_id": str(self.expected_id),
            "prediction_input_id": str(self.prediction_input_id),
            "status": self.status.value,
            "observation_ref": self.observation_ref,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "predicted_at": self.predicted_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PredictionResult:
    """World model prediction result."""

    result_id: UUID
    prediction_input: PredictionInput
    expected_observation: ExpectedObservation | None
    status: PredictionStatus
    confidence: float
    evidence: tuple[WorldModelEvidence, ...]
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.result_id, UUID):
            raise TypeError("result_id must be UUID")
        if not isinstance(self.prediction_input, PredictionInput):
            raise TypeError("prediction_input must be PredictionInput")
        if self.expected_observation is not None:
            if not isinstance(self.expected_observation, ExpectedObservation):
                raise TypeError(
                    "expected_observation must be ExpectedObservation or None"
                )
            if (
                self.expected_observation.prediction_input_id
                != self.prediction_input.input_id
            ):
                raise ValueError("expected_observation input id must match")
        if not isinstance(self.status, PredictionStatus):
            raise TypeError("status must be PredictionStatus")
        confidence = ensure_probability(self.confidence, "confidence")
        evidence = _ensure_evidence(self.evidence)
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "result_id": str(self.result_id),
            "prediction_input": self.prediction_input.to_dict(),
            "expected_observation": (
                None
                if self.expected_observation is None
                else self.expected_observation.to_dict()
            ),
            "status": self.status.value,
            "confidence": self.confidence,
            "evidence": tuple(item.to_dict() for item in self.evidence),
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    def has_expected_observation(self) -> bool:
        """Return whether this result has an expected observation."""

        return self.expected_observation is not None


@dataclass(frozen=True)
class WorldModelSnapshot:
    """Immutable runtime snapshot of world model state."""

    snapshot_id: UUID
    states: tuple[WorldStateRef, ...]
    prediction_inputs: tuple[PredictionInput, ...]
    prediction_results: tuple[PredictionResult, ...]
    evidence: tuple[WorldModelEvidence, ...]
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot_id, UUID):
            raise TypeError("snapshot_id must be UUID")
        states = _ensure_states(self.states)
        prediction_inputs = _ensure_prediction_inputs(self.prediction_inputs)
        prediction_results = _ensure_prediction_results(self.prediction_results)
        evidence = _ensure_evidence(self.evidence)
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        _validate_snapshot_references(
            states=states,
            prediction_inputs=prediction_inputs,
            prediction_results=prediction_results,
        )
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "prediction_inputs", prediction_inputs)
        object.__setattr__(self, "prediction_results", prediction_results)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "snapshot_id": str(self.snapshot_id),
            "states": tuple(state.to_dict() for state in self.states),
            "prediction_inputs": tuple(
                item.to_dict() for item in self.prediction_inputs
            ),
            "prediction_results": tuple(
                result.to_dict() for result in self.prediction_results
            ),
            "evidence": tuple(item.to_dict() for item in self.evidence),
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    def state_count(self) -> int:
        """Return state count."""

        return len(self.states)

    def prediction_input_count(self) -> int:
        """Return prediction input count."""

        return len(self.prediction_inputs)

    def prediction_result_count(self) -> int:
        """Return prediction result count."""

        return len(self.prediction_results)

    def evidence_count(self) -> int:
        """Return evidence count."""

        return len(self.evidence)


def utc_now() -> datetime:
    """Return a timezone-aware current UTC datetime."""

    return datetime.now(timezone.utc)


def new_world_state_id() -> UUID:
    """Return a new world state identifier."""

    return uuid4()


def new_world_model_evidence_id() -> UUID:
    """Return a new world model evidence identifier."""

    return uuid4()


def new_prediction_input_id() -> UUID:
    """Return a new prediction input identifier."""

    return uuid4()


def new_expected_observation_id() -> UUID:
    """Return a new expected observation identifier."""

    return uuid4()


def new_prediction_result_id() -> UUID:
    """Return a new prediction result identifier."""

    return uuid4()


def new_world_model_snapshot_id() -> UUID:
    """Return a new world model snapshot identifier."""

    return uuid4()


def ensure_non_empty_string(value: str, field_name: str) -> str:
    """Return a stripped non-empty string."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must be non-empty")
    return stripped


def ensure_optional_non_empty_string(
    value: str | None,
    field_name: str,
) -> str | None:
    """Return None or a stripped non-empty string."""

    if value is None:
        return None
    return ensure_non_empty_string(value, field_name)


def ensure_probability(value: float, field_name: str) -> float:
    """Return a finite probability value."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be int or float")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{field_name} must be in range 0.0..1.0")
    return number


def ensure_timezone_aware(value: datetime, field_name: str) -> datetime:
    """Return a timezone-aware datetime."""

    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def create_world_state_ref(
    *,
    state_ref: str,
    kind: WorldStateKind = WorldStateKind.UNKNOWN,
    status: WorldStateStatus = WorldStateStatus.ACTIVE,
    confidence: float = 1.0,
    metadata: dict[str, object] | None = None,
) -> WorldStateRef:
    """Create a world state reference."""

    now = utc_now()
    default_metadata: dict[str, object] = {"source": "world_state_ref"}
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "world_state_ref"
    return WorldStateRef(
        state_id=new_world_state_id(),
        kind=kind,
        status=status,
        state_ref=state_ref,
        confidence=confidence,
        created_at=now,
        updated_at=now,
        metadata=merged_metadata,
    )


def create_world_model_evidence(
    *,
    source_id: str,
    evidence_ref: str,
    kind: WorldModelEvidenceKind = WorldModelEvidenceKind.UNKNOWN,
    relevance: float = 1.0,
    confidence: float = 1.0,
    metadata: dict[str, object] | None = None,
) -> WorldModelEvidence:
    """Create world model evidence."""

    default_metadata: dict[str, object] = {
        "source": "world_model_evidence",
        "source_id": source_id,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "world_model_evidence"
    return WorldModelEvidence(
        evidence_id=new_world_model_evidence_id(),
        kind=kind,
        source_id=source_id,
        evidence_ref=evidence_ref,
        relevance=relevance,
        confidence=confidence,
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def create_prediction_input(
    *,
    context_ref: str,
    kind: PredictionKind = PredictionKind.UNKNOWN,
    current_state: WorldStateRef | None = None,
    evidence: tuple[WorldModelEvidence, ...] | list[WorldModelEvidence] = (),
    metadata: dict[str, object] | None = None,
) -> PredictionInput:
    """Create a prediction input."""

    evidence_values = tuple(evidence)
    default_metadata: dict[str, object] = {
        "source": "prediction_input",
        "evidence_count": len(evidence_values),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "prediction_input"
    return PredictionInput(
        input_id=new_prediction_input_id(),
        kind=kind,
        context_ref=context_ref,
        current_state=current_state,
        evidence=evidence_values,
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def create_expected_observation(
    *,
    prediction_input_id: UUID,
    observation_ref: str,
    status: ExpectedObservationStatus = ExpectedObservationStatus.EXPECTED,
    confidence: float = 1.0,
    uncertainty: float = 0.0,
    metadata: dict[str, object] | None = None,
) -> ExpectedObservation:
    """Create an expected observation."""

    default_metadata: dict[str, object] = {
        "source": "expected_observation",
        "prediction_input_id": str(prediction_input_id),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "expected_observation"
    return ExpectedObservation(
        expected_id=new_expected_observation_id(),
        prediction_input_id=prediction_input_id,
        status=status,
        observation_ref=observation_ref,
        confidence=confidence,
        uncertainty=uncertainty,
        predicted_at=utc_now(),
        metadata=merged_metadata,
    )


def create_prediction_result(
    *,
    prediction_input: PredictionInput,
    expected_observation: ExpectedObservation | None = None,
    status: PredictionStatus = PredictionStatus.PREDICTED,
    confidence: float = 1.0,
    evidence: tuple[WorldModelEvidence, ...] | list[WorldModelEvidence] = (),
    metadata: dict[str, object] | None = None,
) -> PredictionResult:
    """Create a prediction result."""

    evidence_values = tuple(evidence)
    default_metadata: dict[str, object] = {
        "source": "prediction_result",
        "prediction_input_id": str(prediction_input.input_id),
        "has_expected_observation": expected_observation is not None,
        "evidence_count": len(evidence_values),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "prediction_result"
    return PredictionResult(
        result_id=new_prediction_result_id(),
        prediction_input=prediction_input,
        expected_observation=expected_observation,
        status=status,
        confidence=confidence,
        evidence=evidence_values,
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def create_world_model_snapshot(
    *,
    states: tuple[WorldStateRef, ...] | list[WorldStateRef] = (),
    prediction_inputs: tuple[PredictionInput, ...] | list[PredictionInput] = (),
    prediction_results: (
        tuple[PredictionResult, ...] | list[PredictionResult]
    ) = (),
    evidence: tuple[WorldModelEvidence, ...] | list[WorldModelEvidence] = (),
    metadata: dict[str, object] | None = None,
) -> WorldModelSnapshot:
    """Create a world model snapshot."""

    state_values = tuple(states)
    input_values = tuple(prediction_inputs)
    result_values = tuple(prediction_results)
    evidence_values = tuple(evidence)
    default_metadata: dict[str, object] = {
        "source": "world_model_snapshot",
        "state_count": len(state_values),
        "prediction_input_count": len(input_values),
        "prediction_result_count": len(result_values),
        "evidence_count": len(evidence_values),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "world_model_snapshot"
    return WorldModelSnapshot(
        snapshot_id=new_world_model_snapshot_id(),
        states=state_values,
        prediction_inputs=input_values,
        prediction_results=result_values,
        evidence=evidence_values,
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def _ensure_states(
    values: tuple[WorldStateRef, ...] | list[WorldStateRef],
) -> tuple[WorldStateRef, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("states must be tuple or list")
    states = tuple(values)
    for state in states:
        if not isinstance(state, WorldStateRef):
            raise TypeError("states must contain WorldStateRef")
    state_ids = tuple(state.state_id for state in states)
    if len(set(state_ids)) != len(state_ids):
        raise ValueError("state_id values must be unique")
    return states


def _ensure_evidence(
    values: tuple[WorldModelEvidence, ...] | list[WorldModelEvidence],
) -> tuple[WorldModelEvidence, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("evidence must be tuple or list")
    evidence = tuple(values)
    for item in evidence:
        if not isinstance(item, WorldModelEvidence):
            raise TypeError("evidence must contain WorldModelEvidence")
    evidence_ids = tuple(item.evidence_id for item in evidence)
    if len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError("evidence_id values must be unique")
    return evidence


def _ensure_prediction_inputs(
    values: tuple[PredictionInput, ...] | list[PredictionInput],
) -> tuple[PredictionInput, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("prediction_inputs must be tuple or list")
    prediction_inputs = tuple(values)
    for item in prediction_inputs:
        if not isinstance(item, PredictionInput):
            raise TypeError("prediction_inputs must contain PredictionInput")
    input_ids = tuple(item.input_id for item in prediction_inputs)
    if len(set(input_ids)) != len(input_ids):
        raise ValueError("input_id values must be unique")
    return prediction_inputs


def _ensure_prediction_results(
    values: tuple[PredictionResult, ...] | list[PredictionResult],
) -> tuple[PredictionResult, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("prediction_results must be tuple or list")
    prediction_results = tuple(values)
    for result in prediction_results:
        if not isinstance(result, PredictionResult):
            raise TypeError("prediction_results must contain PredictionResult")
    result_ids = tuple(result.result_id for result in prediction_results)
    if len(set(result_ids)) != len(result_ids):
        raise ValueError("result_id values must be unique")
    return prediction_results


def _validate_snapshot_references(
    *,
    states: tuple[WorldStateRef, ...],
    prediction_inputs: tuple[PredictionInput, ...],
    prediction_results: tuple[PredictionResult, ...],
) -> None:
    state_ids = {state.state_id for state in states}
    input_ids = {item.input_id for item in prediction_inputs}
    for prediction_input in prediction_inputs:
        if prediction_input.current_state is None:
            continue
        if prediction_input.current_state.state_id not in state_ids:
            raise ValueError("prediction input current_state must exist")
    for result in prediction_results:
        if result.prediction_input.input_id not in input_ids:
            raise ValueError("prediction result input must exist")


def _merge_metadata(
    default_metadata: dict[str, object],
    metadata: dict[str, object] | None,
) -> dict[str, object]:
    merged = _copy_metadata(default_metadata)
    if metadata is not None:
        merged.update(_copy_metadata(metadata))
    if "source" in default_metadata:
        merged["source"] = default_metadata["source"]
    return merged


def _copy_metadata(metadata: dict[str, object]) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(metadata)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
    return copied
