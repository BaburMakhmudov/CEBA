"""Deterministic runtime activation utilities for CEBA Core Relation."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from ceba.core.relation.schema import (
    RelationActivation,
    RelationActivationStatus,
    RelationEdge,
    activation_status_from_value,
    create_relation_activation,
    ensure_non_empty_string,
    utc_now,
)


@dataclass(frozen=True)
class RelationActivationPolicy:
    """Configuration controlling deterministic activation behavior."""

    base_activation: float = 1.0
    min_activation: float = 0.0
    max_activation: float = 1.0
    decay_rate: float = 0.05
    reinforcement_rate: float = 0.10
    suppress_rate: float = 0.20
    active_threshold: float = 0.60
    weak_threshold: float = 0.20

    def __post_init__(self) -> None:
        base_activation = _ensure_probability(
            self.base_activation,
            "base_activation",
        )
        min_activation = _ensure_probability(
            self.min_activation,
            "min_activation",
        )
        max_activation = _ensure_probability(
            self.max_activation,
            "max_activation",
        )
        decay_rate = _ensure_probability(self.decay_rate, "decay_rate")
        reinforcement_rate = _ensure_probability(
            self.reinforcement_rate,
            "reinforcement_rate",
        )
        suppress_rate = _ensure_probability(self.suppress_rate, "suppress_rate")
        active_threshold = _ensure_probability(
            self.active_threshold,
            "active_threshold",
        )
        weak_threshold = _ensure_probability(
            self.weak_threshold,
            "weak_threshold",
        )
        if max_activation < min_activation:
            raise ValueError("max_activation must be >= min_activation")
        if base_activation < min_activation or base_activation > max_activation:
            raise ValueError(
                "base_activation must be between min_activation and max_activation"
            )
        if active_threshold < weak_threshold:
            raise ValueError("active_threshold must be >= weak_threshold")

        object.__setattr__(self, "base_activation", base_activation)
        object.__setattr__(self, "min_activation", min_activation)
        object.__setattr__(self, "max_activation", max_activation)
        object.__setattr__(self, "decay_rate", decay_rate)
        object.__setattr__(self, "reinforcement_rate", reinforcement_rate)
        object.__setattr__(self, "suppress_rate", suppress_rate)
        object.__setattr__(self, "active_threshold", active_threshold)
        object.__setattr__(self, "weak_threshold", weak_threshold)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "base_activation": self.base_activation,
            "min_activation": self.min_activation,
            "max_activation": self.max_activation,
            "decay_rate": self.decay_rate,
            "reinforcement_rate": self.reinforcement_rate,
            "suppress_rate": self.suppress_rate,
            "active_threshold": self.active_threshold,
            "weak_threshold": self.weak_threshold,
        }


@dataclass(frozen=True)
class RelationActivationUpdate:
    """Immutable explicit update command for relation activation."""

    edge_id: str
    reinforcement: float = 0.0
    suppression: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        edge_id = ensure_non_empty_string(self.edge_id, "edge_id")
        reinforcement = _ensure_probability(
            self.reinforcement,
            "reinforcement",
        )
        suppression = _ensure_probability(self.suppression, "suppression")
        metadata = _copy_metadata(self.metadata)

        object.__setattr__(self, "edge_id", edge_id)
        object.__setattr__(self, "reinforcement", reinforcement)
        object.__setattr__(self, "suppression", suppression)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "edge_id": self.edge_id,
            "reinforcement": self.reinforcement,
            "suppression": self.suppression,
            "metadata": dict(self.metadata),
        }


class RelationActivationEngine:
    """Deterministic service for relation edge activation descriptors."""

    def __init__(self, policy: RelationActivationPolicy | None = None) -> None:
        if policy is None:
            policy = RelationActivationPolicy()
        if not isinstance(policy, RelationActivationPolicy):
            raise TypeError("policy must be RelationActivationPolicy or None")
        self.policy = policy

    def create_activation(
        self,
        *,
        edge: RelationEdge,
        activation: float | None = None,
        decay: float | None = None,
        update_count: int = 0,
        metadata: dict[str, object] | None = None,
    ) -> RelationActivation:
        """Create an activation descriptor for a relation edge."""

        if not isinstance(edge, RelationEdge):
            raise TypeError("edge must be RelationEdge")
        activation_value = (
            self.policy.base_activation if activation is None else activation
        )
        activation_value = _clamp_activation(
            _ensure_probability(activation_value, "activation"),
            self.policy,
        )
        decay_value = self.policy.decay_rate if decay is None else decay
        decay_value = _ensure_probability(decay_value, "decay")
        update_count = _ensure_non_negative_int(update_count, "update_count")
        status = self._status_from_activation(activation_value)
        default_metadata: dict[str, object] = {
            "source": "relation_activation_engine",
            "edge_id": str(edge.edge_id),
            "relation_kind": edge.kind.value,
            "source_node_id": edge.source.node_id,
            "target_node_id": edge.target.node_id,
        }
        merged_metadata = _merge_metadata(default_metadata, metadata)
        merged_metadata["source"] = "relation_activation_engine"
        return create_relation_activation(
            edge_id=edge.edge_id,
            activation=activation_value,
            decay=decay_value,
            update_count=update_count,
            status=status,
            metadata=merged_metadata,
        )

    def update_activation(
        self,
        *,
        activation: RelationActivation,
        update: RelationActivationUpdate,
        metadata: dict[str, object] | None = None,
    ) -> RelationActivation:
        """Create the next activation descriptor without mutating inputs."""

        if not isinstance(activation, RelationActivation):
            raise TypeError("activation must be RelationActivation")
        if not isinstance(update, RelationActivationUpdate):
            raise TypeError("update must be RelationActivationUpdate")
        if str(activation.edge_id) != update.edge_id:
            raise ValueError("activation.edge_id must match update.edge_id")

        activation_value = (
            activation.activation
            - self.policy.decay_rate
            + update.reinforcement * self.policy.reinforcement_rate
            - update.suppression * self.policy.suppress_rate
        )
        activation_value = _clamp_activation(activation_value, self.policy)
        status = self._status_from_activation(activation_value)
        update_count = activation.update_count + 1
        default_metadata = _copy_metadata(activation.metadata)
        default_metadata.update(_copy_metadata(update.metadata))
        caller_metadata = _copy_metadata(metadata) if metadata is not None else {}
        default_metadata.update(caller_metadata)
        default_metadata.update(
            {
                "source": "relation_activation_engine",
                "edge_id": update.edge_id,
                "update_count": update_count,
            }
        )

        return RelationActivation(
            activation_id=activation.activation_id,
            edge_id=activation.edge_id,
            status=status,
            activation=activation_value,
            decay=self.policy.decay_rate,
            update_count=update_count,
            created_at=activation.created_at,
            updated_at=utc_now(),
            metadata=default_metadata,
        )

    def activate_many(
        self,
        *,
        edges: tuple[RelationEdge, ...] | list[RelationEdge],
        metadata: dict[str, object] | None = None,
    ) -> tuple[RelationActivation, ...]:
        """Create one activation per edge preserving input order."""

        values = _ensure_edges(edges)
        return tuple(
            self.create_activation(edge=edge, metadata=metadata)
            for edge in values
        )

    def _status_from_activation(
        self,
        activation: float,
    ) -> RelationActivationStatus:
        return activation_status_from_value(
            activation=activation,
            active_threshold=self.policy.active_threshold,
            weak_threshold=self.policy.weak_threshold,
        )


def create_edge_activation(
    *,
    edge: RelationEdge,
    policy: RelationActivationPolicy | None = None,
    activation: float | None = None,
    decay: float | None = None,
    update_count: int = 0,
    metadata: dict[str, object] | None = None,
) -> RelationActivation:
    """Create a relation activation for an edge."""

    engine = RelationActivationEngine(policy)
    return engine.create_activation(
        edge=edge,
        activation=activation,
        decay=decay,
        update_count=update_count,
        metadata=metadata,
    )


def update_edge_activation(
    *,
    activation: RelationActivation,
    update: RelationActivationUpdate,
    policy: RelationActivationPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> RelationActivation:
    """Create the next activation descriptor from an explicit update."""

    engine = RelationActivationEngine(policy)
    return engine.update_activation(
        activation=activation,
        update=update,
        metadata=metadata,
    )


def activate_edges(
    *,
    edges: tuple[RelationEdge, ...] | list[RelationEdge],
    policy: RelationActivationPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> tuple[RelationActivation, ...]:
    """Create activation descriptors for relation edges."""

    engine = RelationActivationEngine(policy)
    return engine.activate_many(edges=edges, metadata=metadata)


def _ensure_probability(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be finite number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{field_name} must be in range 0.0..1.0")
    return number


def _ensure_non_negative_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _clamp_activation(
    value: float,
    policy: RelationActivationPolicy,
) -> float:
    if not isfinite(value):
        raise ValueError("activation must be finite")
    return max(policy.min_activation, min(policy.max_activation, value))


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


def _copy_metadata(metadata: dict[str, object]) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(metadata)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
    return copied


def _merge_metadata(
    default_metadata: dict[str, object],
    caller_metadata: dict[str, object] | None,
) -> dict[str, object]:
    merged = _copy_metadata(default_metadata)
    if caller_metadata is not None:
        merged.update(_copy_metadata(caller_metadata))
    return merged


__all__ = [
    "RelationActivationPolicy",
    "RelationActivationUpdate",
    "RelationActivationEngine",
    "create_edge_activation",
    "update_edge_activation",
    "activate_edges",
]
