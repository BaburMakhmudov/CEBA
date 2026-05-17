"""Deterministic relation weighting utilities for CEBA Core Relation."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from ceba.core.relation.schema import (
    RelationEdge,
    create_relation_edge,
    ensure_non_empty_string,
    relation_status_from_confidence,
)


@dataclass(frozen=True)
class RelationWeightingPolicy:
    """Configuration controlling deterministic relation weighting behavior."""

    min_weight: float = 0.0
    max_weight: float = 1.0
    base_confidence_weight: float = 0.40
    evidence_weight: float = 0.30
    prior_weight: float = 0.20
    activation_weight: float = 0.10
    normalize_output: bool = True

    def __post_init__(self) -> None:
        min_weight = _ensure_non_negative_number(self.min_weight, "min_weight")
        max_weight = _ensure_non_negative_number(self.max_weight, "max_weight")
        if max_weight < min_weight:
            raise ValueError("max_weight must be >= min_weight")
        base_confidence_weight = _ensure_probability(
            self.base_confidence_weight,
            "base_confidence_weight",
        )
        evidence_weight = _ensure_probability(
            self.evidence_weight,
            "evidence_weight",
        )
        prior_weight = _ensure_probability(self.prior_weight, "prior_weight")
        activation_weight = _ensure_probability(
            self.activation_weight,
            "activation_weight",
        )
        if not isinstance(self.normalize_output, bool):
            raise TypeError("normalize_output must be bool")

        object.__setattr__(self, "min_weight", min_weight)
        object.__setattr__(self, "max_weight", max_weight)
        object.__setattr__(
            self,
            "base_confidence_weight",
            base_confidence_weight,
        )
        object.__setattr__(self, "evidence_weight", evidence_weight)
        object.__setattr__(self, "prior_weight", prior_weight)
        object.__setattr__(self, "activation_weight", activation_weight)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "base_confidence_weight": self.base_confidence_weight,
            "evidence_weight": self.evidence_weight,
            "prior_weight": self.prior_weight,
            "activation_weight": self.activation_weight,
            "normalize_output": self.normalize_output,
        }


@dataclass(frozen=True)
class RelationWeightingSignal:
    """Explicit local signal used to adjust one relation edge weight."""

    edge_id: str
    evidence_strength: float = 0.0
    prior_strength: float = 0.0
    activation_strength: float = 0.0
    confidence_override: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        edge_id = ensure_non_empty_string(self.edge_id, "edge_id")
        evidence_strength = _ensure_probability(
            self.evidence_strength,
            "evidence_strength",
        )
        prior_strength = _ensure_probability(
            self.prior_strength,
            "prior_strength",
        )
        activation_strength = _ensure_probability(
            self.activation_strength,
            "activation_strength",
        )
        confidence_override = _ensure_optional_probability(
            self.confidence_override,
            "confidence_override",
        )
        metadata = _copy_metadata(self.metadata)

        object.__setattr__(self, "edge_id", edge_id)
        object.__setattr__(self, "evidence_strength", evidence_strength)
        object.__setattr__(self, "prior_strength", prior_strength)
        object.__setattr__(self, "activation_strength", activation_strength)
        object.__setattr__(self, "confidence_override", confidence_override)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "edge_id": self.edge_id,
            "evidence_strength": self.evidence_strength,
            "prior_strength": self.prior_strength,
            "activation_strength": self.activation_strength,
            "confidence_override": self.confidence_override,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RelationWeightingResult:
    """Immutable result object returned after weighting."""

    edges: tuple[RelationEdge, ...]
    skipped_edge_ids: tuple[str, ...]
    skip_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        edges = tuple(self.edges)
        for edge in edges:
            if not isinstance(edge, RelationEdge):
                raise TypeError("edges must contain RelationEdge")
        skipped_edge_ids = _ensure_string_tuple(
            self.skipped_edge_ids,
            "skipped_edge_ids",
        )
        skip_reasons = _ensure_string_tuple(self.skip_reasons, "skip_reasons")

        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "skipped_edge_ids", skipped_edge_ids)
        object.__setattr__(self, "skip_reasons", skip_reasons)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "edges": tuple(edge.to_dict() for edge in self.edges),
            "skipped_edge_ids": self.skipped_edge_ids,
            "skip_reasons": self.skip_reasons,
        }


class RelationWeightingEngine:
    """Deterministic service for re-weighting relation edges."""

    def __init__(self, policy: RelationWeightingPolicy | None = None) -> None:
        if policy is None:
            policy = RelationWeightingPolicy()
        if not isinstance(policy, RelationWeightingPolicy):
            raise TypeError("policy must be RelationWeightingPolicy or None")
        self.policy = policy

    def weight_edge(
        self,
        *,
        edge: RelationEdge,
        signal: RelationWeightingSignal,
        metadata: dict[str, object] | None = None,
    ) -> RelationEdge:
        """Create a re-weighted RelationEdge from an explicit signal."""

        if not isinstance(edge, RelationEdge):
            raise TypeError("edge must be RelationEdge")
        if not isinstance(signal, RelationWeightingSignal):
            raise TypeError("signal must be RelationWeightingSignal")
        if str(edge.edge_id) != signal.edge_id:
            raise ValueError("edge.edge_id must match signal.edge_id")

        weighted_score = self._weighted_score(edge=edge, signal=signal)
        weight = _clamp_weight(weighted_score, self.policy)
        confidence = (
            signal.confidence_override
            if signal.confidence_override is not None
            else edge.confidence
        )
        confidence = _ensure_probability(confidence, "confidence")
        status = relation_status_from_confidence(confidence=confidence)

        merged_metadata = _copy_metadata(edge.metadata)
        merged_metadata.update(_copy_metadata(signal.metadata))
        if metadata is not None:
            merged_metadata.update(_copy_metadata(metadata))
        merged_metadata.update(
            {
                "source": "relation_weighting_engine",
                "previous_weight": edge.weight,
                "weighted_score": weighted_score,
                "evidence_strength": signal.evidence_strength,
                "prior_strength": signal.prior_strength,
                "activation_strength": signal.activation_strength,
            }
        )

        return create_relation_edge(
            source=edge.source,
            target=edge.target,
            kind=edge.kind,
            weight=weight,
            confidence=confidence,
            evidence=edge.evidence,
            status=status,
            metadata=merged_metadata,
        )

    def weight_many(
        self,
        *,
        edges: tuple[RelationEdge, ...] | list[RelationEdge],
        signals: tuple[RelationWeightingSignal, ...] | list[
            RelationWeightingSignal
        ],
        metadata: dict[str, object] | None = None,
    ) -> RelationWeightingResult:
        """Weight many edges using matching explicit signals."""

        edge_values = _ensure_edges(edges)
        signal_values = _ensure_signals(signals)
        signal_lookup = {signal.edge_id: signal for signal in signal_values}
        known_edge_ids = {str(edge.edge_id) for edge in edge_values}

        weighted_edges: list[RelationEdge] = []
        skipped_edge_ids: list[str] = []
        skip_reasons: list[str] = []
        for edge in edge_values:
            edge_id = str(edge.edge_id)
            signal = signal_lookup.get(edge_id)
            if signal is None:
                skipped_edge_ids.append(edge_id)
                skip_reasons.append("missing weighting signal")
            else:
                weighted_edges.append(
                    self.weight_edge(
                        edge=edge,
                        signal=signal,
                        metadata=metadata,
                    )
                )

        for signal in signal_values:
            if signal.edge_id not in known_edge_ids:
                skipped_edge_ids.append(signal.edge_id)
                skip_reasons.append(f"unknown edge_id: {signal.edge_id}")

        return RelationWeightingResult(
            edges=tuple(weighted_edges),
            skipped_edge_ids=tuple(skipped_edge_ids),
            skip_reasons=tuple(skip_reasons),
        )

    def _weighted_score(
        self,
        *,
        edge: RelationEdge,
        signal: RelationWeightingSignal,
    ) -> float:
        weighted_score = (
            edge.confidence * self.policy.base_confidence_weight
            + signal.evidence_strength * self.policy.evidence_weight
            + signal.prior_strength * self.policy.prior_weight
            + signal.activation_strength * self.policy.activation_weight
        )
        if not self.policy.normalize_output:
            return weighted_score
        weight_sum = (
            self.policy.base_confidence_weight
            + self.policy.evidence_weight
            + self.policy.prior_weight
            + self.policy.activation_weight
        )
        if weight_sum <= 0.0:
            return 0.0
        return weighted_score / weight_sum


def weight_relation_edge(
    *,
    edge: RelationEdge,
    signal: RelationWeightingSignal,
    policy: RelationWeightingPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> RelationEdge:
    """Create a re-weighted relation edge."""

    engine = RelationWeightingEngine(policy)
    return engine.weight_edge(edge=edge, signal=signal, metadata=metadata)


def weight_relation_edges(
    *,
    edges: tuple[RelationEdge, ...] | list[RelationEdge],
    signals: tuple[RelationWeightingSignal, ...] | list[RelationWeightingSignal],
    policy: RelationWeightingPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> RelationWeightingResult:
    """Create re-weighted relation edges from matching signals."""

    engine = RelationWeightingEngine(policy)
    return engine.weight_many(edges=edges, signals=signals, metadata=metadata)


def _ensure_probability(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be finite number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{field_name} must be in range 0.0..1.0")
    return number


def _ensure_optional_probability(
    value: float | None,
    field_name: str,
) -> float | None:
    if value is None:
        return None
    return _ensure_probability(value, field_name)


def _ensure_non_negative_number(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be finite number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    if number < 0.0:
        raise ValueError(f"{field_name} must be non-negative")
    return number


def _clamp_weight(value: float, policy: RelationWeightingPolicy) -> float:
    if not isfinite(value):
        raise ValueError("weight must be finite")
    return max(policy.min_weight, min(policy.max_weight, value))


def _ensure_edges(
    edges: tuple[RelationEdge, ...] | list[RelationEdge],
) -> tuple[RelationEdge, ...]:
    if not isinstance(edges, (tuple, list)):
        raise TypeError("edges must be tuple or list")
    values = tuple(edges)
    for edge in values:
        if not isinstance(edge, RelationEdge):
            raise TypeError("edges must contain RelationEdge")
    return values


def _ensure_signals(
    signals: tuple[RelationWeightingSignal, ...] | list[
        RelationWeightingSignal
    ],
) -> tuple[RelationWeightingSignal, ...]:
    if not isinstance(signals, (tuple, list)):
        raise TypeError("signals must be tuple or list")
    values = tuple(signals)
    for signal in values:
        if not isinstance(signal, RelationWeightingSignal):
            raise TypeError("signals must contain RelationWeightingSignal")
    return values


def _ensure_string_tuple(
    values: tuple[str, ...] | list[str],
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple or list")
    result = tuple(values)
    for value in result:
        ensure_non_empty_string(value, field_name)
    return result


def _copy_metadata(metadata: dict[str, object]) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(metadata)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
    return copied


__all__ = [
    "RelationWeightingPolicy",
    "RelationWeightingSignal",
    "RelationWeightingResult",
    "RelationWeightingEngine",
    "weight_relation_edge",
    "weight_relation_edges",
]
