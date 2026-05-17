"""Deterministic evidence utilities for CEBA World Model."""

from __future__ import annotations

from dataclasses import dataclass, field

from ceba.world_model.errors import PredictionInputError
from ceba.world_model.schema import (
    PredictionInput,
    PredictionKind,
    WorldModelEvidence,
    WorldModelEvidenceKind,
    create_prediction_input,
    create_world_model_evidence,
    ensure_non_empty_string,
    ensure_probability,
)


__all__ = [
    "WorldEvidencePolicy",
    "WorldEvidenceInput",
    "WorldEvidenceSelection",
    "WorldEvidenceManager",
    "create_world_model_evidence_items",
    "select_world_model_evidence",
    "build_prediction_input_from_evidence",
]


@dataclass(frozen=True)
class WorldEvidencePolicy:
    """Configuration object controlling world model evidence handling."""

    min_relevance: float = 0.0
    min_confidence: float = 0.0
    max_evidence_items: int = 256
    require_evidence_for_prediction_input: bool = False
    default_prediction_kind: PredictionKind = PredictionKind.EXPECTED_OBSERVATION

    def __post_init__(self) -> None:
        min_relevance = ensure_probability(self.min_relevance, "min_relevance")
        min_confidence = ensure_probability(
            self.min_confidence,
            "min_confidence",
        )
        max_items = _ensure_positive_int(
            self.max_evidence_items,
            "max_evidence_items",
        )
        if not isinstance(self.require_evidence_for_prediction_input, bool):
            raise TypeError("require_evidence_for_prediction_input must be bool")
        if not isinstance(self.default_prediction_kind, PredictionKind):
            raise TypeError("default_prediction_kind must be PredictionKind")
        object.__setattr__(self, "min_relevance", min_relevance)
        object.__setattr__(self, "min_confidence", min_confidence)
        object.__setattr__(self, "max_evidence_items", max_items)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "min_relevance": self.min_relevance,
            "min_confidence": self.min_confidence,
            "max_evidence_items": self.max_evidence_items,
            "require_evidence_for_prediction_input": (
                self.require_evidence_for_prediction_input
            ),
            "default_prediction_kind": self.default_prediction_kind.value,
        }


@dataclass(frozen=True)
class WorldEvidenceInput:
    """Explicit input object for creating one WorldModelEvidence."""

    source_id: str
    evidence_ref: str
    kind: WorldModelEvidenceKind = WorldModelEvidenceKind.UNKNOWN
    relevance: float = 1.0
    confidence: float = 1.0
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, WorldModelEvidenceKind):
            raise TypeError("kind must be WorldModelEvidenceKind")
        source_id = ensure_non_empty_string(self.source_id, "source_id")
        evidence_ref = ensure_non_empty_string(self.evidence_ref, "evidence_ref")
        relevance = ensure_probability(self.relevance, "relevance")
        confidence = ensure_probability(self.confidence, "confidence")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "evidence_ref", evidence_ref)
        object.__setattr__(self, "relevance", relevance)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "kind": self.kind.value,
            "source_id": self.source_id,
            "evidence_ref": self.evidence_ref,
            "relevance": self.relevance,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class WorldEvidenceSelection:
    """Immutable result object returned by evidence selection."""

    evidence: tuple[WorldModelEvidence, ...]
    rejected_source_ids: tuple[str, ...]
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        evidence = _ensure_evidence(self.evidence)
        rejected_ids = _ensure_string_tuple(
            self.rejected_source_ids,
            "rejected_source_ids",
        )
        rejection_reasons = _ensure_string_tuple(
            self.rejection_reasons,
            "rejection_reasons",
        )
        if len(rejected_ids) != len(rejection_reasons):
            raise ValueError("rejected_source_ids and rejection_reasons must match")
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "rejected_source_ids", rejected_ids)
        object.__setattr__(self, "rejection_reasons", rejection_reasons)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "evidence": tuple(item.to_dict() for item in self.evidence),
            "rejected_source_ids": self.rejected_source_ids,
            "rejection_reasons": self.rejection_reasons,
        }

    def evidence_count(self) -> int:
        """Return evidence count."""

        return len(self.evidence)

    def rejected_count(self) -> int:
        """Return rejected source count."""

        return len(self.rejected_source_ids)


class WorldEvidenceManager:
    """Service for creating, filtering, ranking, and selecting evidence."""

    def __init__(self, policy: WorldEvidencePolicy | None = None) -> None:
        self._policy = WorldEvidencePolicy() if policy is None else policy
        if not isinstance(self._policy, WorldEvidencePolicy):
            raise TypeError("policy must be WorldEvidencePolicy")

    def create_many(
        self,
        *,
        inputs: tuple[WorldEvidenceInput, ...] | list[WorldEvidenceInput],
    ) -> WorldEvidenceSelection:
        """Create evidence from explicit local inputs."""

        input_values = _ensure_inputs(inputs)
        accepted: list[WorldModelEvidence] = []
        rejected_ids: list[str] = []
        rejection_reasons: list[str] = []
        for item in input_values:
            reason = self._input_rejection_reason(item)
            if reason is not None:
                rejected_ids.append(item.source_id)
                rejection_reasons.append(reason)
                continue
            if len(accepted) >= self._policy.max_evidence_items:
                rejected_ids.append(item.source_id)
                rejection_reasons.append("world model evidence limit exceeded")
                continue
            accepted.append(self._create_evidence(item))
        return WorldEvidenceSelection(
            evidence=tuple(accepted),
            rejected_source_ids=tuple(rejected_ids),
            rejection_reasons=tuple(rejection_reasons),
        )

    def select(
        self,
        *,
        evidence: tuple[WorldModelEvidence, ...] | list[WorldModelEvidence],
    ) -> WorldEvidenceSelection:
        """Select usable evidence deterministically."""

        evidence_values = _ensure_evidence(evidence)
        accepted: list[WorldModelEvidence] = []
        rejected_ids: list[str] = []
        rejection_reasons: list[str] = []
        for item in evidence_values:
            reason = self._evidence_rejection_reason(item)
            if reason is None:
                accepted.append(item)
            else:
                rejected_ids.append(item.source_id)
                rejection_reasons.append(reason)

        accepted.sort(
            key=lambda item: (
                -item.relevance,
                -item.confidence,
                -item.created_at.timestamp(),
                str(item.evidence_id),
            )
        )
        selected = accepted[: self._policy.max_evidence_items]
        overflow = accepted[self._policy.max_evidence_items :]
        for item in overflow:
            rejected_ids.append(item.source_id)
            rejection_reasons.append("world model evidence limit exceeded")
        return WorldEvidenceSelection(
            evidence=tuple(selected),
            rejected_source_ids=tuple(rejected_ids),
            rejection_reasons=tuple(rejection_reasons),
        )

    def build_prediction_input(
        self,
        *,
        context_ref: str,
        evidence: tuple[WorldModelEvidence, ...] | list[WorldModelEvidence],
        kind: PredictionKind | None = None,
        metadata: dict[str, object] | None = None,
    ) -> PredictionInput:
        """Build a prediction input from selected evidence."""

        context_ref_value = ensure_non_empty_string(context_ref, "context_ref")
        selection = self.select(evidence=evidence)
        if (
            not selection.evidence
            and self._policy.require_evidence_for_prediction_input
        ):
            raise PredictionInputError(
                "prediction input requires evidence",
                context={"context_ref": context_ref_value},
            )
        prediction_kind = self._policy.default_prediction_kind if kind is None else kind
        if not isinstance(prediction_kind, PredictionKind):
            raise TypeError("kind must be PredictionKind or None")
        default_metadata: dict[str, object] = {
            "source": "world_evidence_manager",
            "operation": "build_prediction_input",
            "selected_evidence_count": selection.evidence_count(),
            "rejected_evidence_count": selection.rejected_count(),
            "policy": self._policy.to_dict(),
        }
        prediction_input = create_prediction_input(
            kind=prediction_kind,
            context_ref=context_ref_value,
            current_state=None,
            evidence=selection.evidence,
            metadata=_merge_metadata(default_metadata, metadata),
        )
        input_metadata = dict(prediction_input.metadata)
        input_metadata.update(_merge_metadata(default_metadata, metadata))
        input_metadata["source"] = "world_evidence_manager"
        return PredictionInput(
            input_id=prediction_input.input_id,
            kind=prediction_input.kind,
            context_ref=prediction_input.context_ref,
            current_state=prediction_input.current_state,
            evidence=prediction_input.evidence,
            created_at=prediction_input.created_at,
            metadata=input_metadata,
        )

    def _create_evidence(self, item: WorldEvidenceInput) -> WorldModelEvidence:
        default_metadata: dict[str, object] = {
            "source": "world_evidence_manager",
            "operation": "create_many",
            "policy": self._policy.to_dict(),
        }
        evidence = create_world_model_evidence(
            kind=item.kind,
            source_id=item.source_id,
            evidence_ref=item.evidence_ref,
            relevance=item.relevance,
            confidence=item.confidence,
            metadata=_merge_metadata(default_metadata, item.metadata),
        )
        evidence_metadata = dict(evidence.metadata)
        evidence_metadata.update(_merge_metadata(default_metadata, item.metadata))
        evidence_metadata["source"] = "world_evidence_manager"
        return WorldModelEvidence(
            evidence_id=evidence.evidence_id,
            kind=evidence.kind,
            source_id=evidence.source_id,
            evidence_ref=evidence.evidence_ref,
            relevance=evidence.relevance,
            confidence=evidence.confidence,
            created_at=evidence.created_at,
            metadata=evidence_metadata,
        )

    def _input_rejection_reason(self, item: WorldEvidenceInput) -> str | None:
        if item.relevance < self._policy.min_relevance:
            return "evidence relevance below minimum"
        if item.confidence < self._policy.min_confidence:
            return "evidence confidence below minimum"
        return None

    def _evidence_rejection_reason(
        self,
        item: WorldModelEvidence,
    ) -> str | None:
        if item.relevance < self._policy.min_relevance:
            return "evidence relevance below minimum"
        if item.confidence < self._policy.min_confidence:
            return "evidence confidence below minimum"
        return None


def create_world_model_evidence_items(
    *,
    inputs: tuple[WorldEvidenceInput, ...] | list[WorldEvidenceInput],
    policy: WorldEvidencePolicy | None = None,
) -> WorldEvidenceSelection:
    """Create world model evidence items from explicit inputs."""

    manager = WorldEvidenceManager(policy)
    return manager.create_many(inputs=inputs)


def select_world_model_evidence(
    *,
    evidence: tuple[WorldModelEvidence, ...] | list[WorldModelEvidence],
    policy: WorldEvidencePolicy | None = None,
) -> WorldEvidenceSelection:
    """Select usable world model evidence."""

    manager = WorldEvidenceManager(policy)
    return manager.select(evidence=evidence)


def build_prediction_input_from_evidence(
    *,
    context_ref: str,
    evidence: tuple[WorldModelEvidence, ...] | list[WorldModelEvidence],
    kind: PredictionKind | None = None,
    policy: WorldEvidencePolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> PredictionInput:
    """Build a prediction input from world model evidence."""

    manager = WorldEvidenceManager(policy)
    return manager.build_prediction_input(
        context_ref=context_ref,
        evidence=evidence,
        kind=kind,
        metadata=metadata,
    )


def _ensure_inputs(
    values: tuple[WorldEvidenceInput, ...] | list[WorldEvidenceInput],
) -> tuple[WorldEvidenceInput, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("inputs must be tuple or list")
    inputs = tuple(values)
    for item in inputs:
        if not isinstance(item, WorldEvidenceInput):
            raise TypeError("inputs must contain WorldEvidenceInput")
    return inputs


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


def _ensure_string_tuple(
    values: tuple[str, ...] | list[str],
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple or list")
    return tuple(ensure_non_empty_string(value, field_name) for value in values)


def _ensure_positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


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
