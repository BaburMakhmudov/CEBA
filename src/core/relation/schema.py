"""Strict schema objects for CEBA Core Relation Processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from ceba.core.metrics.schema import (
    ensure_constraint_probability,
    ensure_non_negative_number,
    utc_now,
)


class RelationNodeKind(Enum):
    """Stable relation node category."""

    OBSERVATION = "OBSERVATION"
    SEGMENT = "SEGMENT"
    DIFFERENCE = "DIFFERENCE"
    CONSTRAINT = "CONSTRAINT"
    METRIC = "METRIC"
    DYNAMICS_STATE = "DYNAMICS_STATE"
    TELOS = "TELOS"
    VALUATION = "VALUATION"
    WORLD_STATE = "WORLD_STATE"
    CONCEPT = "CONCEPT"
    UNKNOWN = "UNKNOWN"


class RelationEdgeKind(Enum):
    """Stable relation edge category."""

    RELATES_TO = "RELATES_TO"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    CAUSES = "CAUSES"
    CONSTRAINS = "CONSTRAINS"
    ALIGNS_WITH = "ALIGNS_WITH"
    DERIVED_FROM = "DERIVED_FROM"
    UPDATES = "UPDATES"
    ACTIVATES = "ACTIVATES"
    SUPPRESSES = "SUPPRESSES"
    UNKNOWN = "UNKNOWN"


class RelationStatus(Enum):
    """Stable relation status."""

    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    DEGRADED = "DEGRADED"
    BLOCKED = "BLOCKED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class RelationEvidenceKind(Enum):
    """Stable relation evidence category."""

    DIRECT = "DIRECT"
    INFERRED = "INFERRED"
    RUNTIME = "RUNTIME"
    VALIDATION = "VALIDATION"
    ALIGNMENT = "ALIGNMENT"
    VALUATION = "VALUATION"
    UNKNOWN = "UNKNOWN"


class RelationActivationStatus(Enum):
    """Stable relation activation status."""

    ACTIVE = "ACTIVE"
    WEAK = "WEAK"
    DORMANT = "DORMANT"
    SUPPRESSED = "SUPPRESSED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class RelationNodeRef:
    """Lightweight reference to a runtime semantic node."""

    node_id: str
    kind: RelationNodeKind
    label: str | None
    confidence: float
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        node_id = ensure_non_empty_string(self.node_id, "node_id")
        if not isinstance(self.kind, RelationNodeKind):
            raise TypeError("kind must be RelationNodeKind")
        label = ensure_optional_non_empty_string(self.label, "label")
        confidence = ensure_constraint_probability(
            self.confidence,
            field_name="confidence",
        )

        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "node_id": self.node_id,
            "kind": self.kind.value,
            "label": self.label,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RelationEvidence:
    """Evidence supporting a relation edge."""

    evidence_id: UUID
    kind: RelationEvidenceKind
    source_id: str
    confidence: float
    weight: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_id, UUID):
            raise TypeError("evidence_id must be UUID")
        if not isinstance(self.kind, RelationEvidenceKind):
            raise TypeError("kind must be RelationEvidenceKind")
        source_id = ensure_non_empty_string(self.source_id, "source_id")
        confidence = ensure_constraint_probability(
            self.confidence,
            field_name="confidence",
        )
        weight = ensure_non_negative_number(self.weight, field_name="weight")
        created_at = ensure_timezone_aware(self.created_at, "created_at")

        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "evidence_id": str(self.evidence_id),
            "kind": self.kind.value,
            "source_id": self.source_id,
            "confidence": self.confidence,
            "weight": self.weight,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RelationEdge:
    """Typed directed relation between two semantic nodes."""

    edge_id: UUID
    source: RelationNodeRef
    target: RelationNodeRef
    kind: RelationEdgeKind
    status: RelationStatus
    weight: float
    confidence: float
    evidence: tuple[RelationEvidence, ...]
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.edge_id, UUID):
            raise TypeError("edge_id must be UUID")
        if not isinstance(self.source, RelationNodeRef):
            raise TypeError("source must be RelationNodeRef")
        if not isinstance(self.target, RelationNodeRef):
            raise TypeError("target must be RelationNodeRef")
        if self.source.node_id == self.target.node_id:
            raise ValueError("source.node_id and target.node_id must differ")
        if not isinstance(self.kind, RelationEdgeKind):
            raise TypeError("kind must be RelationEdgeKind")
        if not isinstance(self.status, RelationStatus):
            raise TypeError("status must be RelationStatus")
        weight = ensure_non_negative_number(self.weight, field_name="weight")
        confidence = ensure_constraint_probability(
            self.confidence,
            field_name="confidence",
        )
        evidence = _ensure_evidence(self.evidence)
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        updated_at = ensure_timezone_aware(self.updated_at, "updated_at")
        if updated_at < created_at:
            raise ValueError("updated_at must be >= created_at")

        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "edge_id": str(self.edge_id),
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "kind": self.kind.value,
            "status": self.status.value,
            "weight": self.weight,
            "confidence": self.confidence,
            "evidence": tuple(item.to_dict() for item in self.evidence),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RelationActivation:
    """Runtime activation descriptor for a relation edge."""

    activation_id: UUID
    edge_id: UUID
    status: RelationActivationStatus
    activation: float
    decay: float
    update_count: int
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.activation_id, UUID):
            raise TypeError("activation_id must be UUID")
        if not isinstance(self.edge_id, UUID):
            raise TypeError("edge_id must be UUID")
        if not isinstance(self.status, RelationActivationStatus):
            raise TypeError("status must be RelationActivationStatus")
        activation = ensure_constraint_probability(
            self.activation,
            field_name="activation",
        )
        decay = ensure_constraint_probability(self.decay, field_name="decay")
        update_count = ensure_non_negative_int(
            self.update_count,
            "update_count",
        )
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        updated_at = ensure_timezone_aware(self.updated_at, "updated_at")
        if updated_at < created_at:
            raise ValueError("updated_at must be >= created_at")

        object.__setattr__(self, "activation", activation)
        object.__setattr__(self, "decay", decay)
        object.__setattr__(self, "update_count", update_count)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "activation_id": str(self.activation_id),
            "edge_id": str(self.edge_id),
            "status": self.status.value,
            "activation": self.activation,
            "decay": self.decay,
            "update_count": self.update_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RelationGraphSnapshot:
    """Lightweight runtime snapshot of relation graph structure."""

    snapshot_id: UUID
    nodes: tuple[RelationNodeRef, ...]
    edges: tuple[RelationEdge, ...]
    activations: tuple[RelationActivation, ...]
    status: RelationStatus
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot_id, UUID):
            raise TypeError("snapshot_id must be UUID")
        nodes = _ensure_nodes(self.nodes)
        edges = _ensure_edges(self.edges)
        activations = _ensure_activations(self.activations)
        if not isinstance(self.status, RelationStatus):
            raise TypeError("status must be RelationStatus")
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        node_ids = {node.node_id for node in nodes}
        edge_ids = {edge.edge_id for edge in edges}
        for edge in edges:
            if edge.source.node_id not in node_ids:
                raise ValueError("edge source node_id must exist in nodes")
            if edge.target.node_id not in node_ids:
                raise ValueError("edge target node_id must exist in nodes")
        for activation in activations:
            if activation.edge_id not in edge_ids:
                raise ValueError("activation edge_id must exist in edges")

        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "activations", activations)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "snapshot_id": str(self.snapshot_id),
            "nodes": tuple(node.to_dict() for node in self.nodes),
            "edges": tuple(edge.to_dict() for edge in self.edges),
            "activations": tuple(
                activation.to_dict() for activation in self.activations
            ),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


def new_relation_evidence_id() -> UUID:
    """Return a new relation evidence identifier."""

    return uuid4()


def new_relation_edge_id() -> UUID:
    """Return a new relation edge identifier."""

    return uuid4()


def new_relation_activation_id() -> UUID:
    """Return a new relation activation identifier."""

    return uuid4()


def new_relation_snapshot_id() -> UUID:
    """Return a new relation snapshot identifier."""

    return uuid4()


def ensure_non_empty_string(value: str, field_name: str) -> str:
    """Validate a non-empty string."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    if value == "":
        raise ValueError(f"{field_name} must be non-empty")
    return value


def ensure_optional_non_empty_string(
    value: str | None,
    field_name: str,
) -> str | None:
    """Validate an optional non-empty string."""

    if value is None:
        return None
    return ensure_non_empty_string(value, field_name)


def ensure_non_negative_int(value: int, field_name: str) -> int:
    """Validate a non-negative integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def ensure_timezone_aware(value: datetime, field_name: str) -> datetime:
    """Validate a timezone-aware datetime."""

    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def relation_status_from_confidence(
    *,
    confidence: float,
    active_threshold: float = 0.60,
    degraded_threshold: float = 0.30,
) -> RelationStatus:
    """Infer relation status from confidence."""

    confidence = ensure_constraint_probability(
        confidence,
        field_name="confidence",
    )
    active_threshold = ensure_constraint_probability(
        active_threshold,
        field_name="active_threshold",
    )
    degraded_threshold = ensure_constraint_probability(
        degraded_threshold,
        field_name="degraded_threshold",
    )
    if active_threshold < degraded_threshold:
        raise ValueError("active_threshold must be >= degraded_threshold")
    if confidence >= active_threshold:
        return RelationStatus.ACTIVE
    if confidence >= degraded_threshold:
        return RelationStatus.DEGRADED
    return RelationStatus.INACTIVE


def activation_status_from_value(
    *,
    activation: float,
    active_threshold: float = 0.60,
    weak_threshold: float = 0.20,
) -> RelationActivationStatus:
    """Infer relation activation status from activation value."""

    activation = ensure_constraint_probability(
        activation,
        field_name="activation",
    )
    active_threshold = ensure_constraint_probability(
        active_threshold,
        field_name="active_threshold",
    )
    weak_threshold = ensure_constraint_probability(
        weak_threshold,
        field_name="weak_threshold",
    )
    if active_threshold < weak_threshold:
        raise ValueError("active_threshold must be >= weak_threshold")
    if activation >= active_threshold:
        return RelationActivationStatus.ACTIVE
    if activation >= weak_threshold:
        return RelationActivationStatus.WEAK
    return RelationActivationStatus.DORMANT


def create_relation_node_ref(
    *,
    node_id: str,
    kind: RelationNodeKind = RelationNodeKind.UNKNOWN,
    label: str | None = None,
    confidence: float = 1.0,
    metadata: dict[str, object] | None = None,
) -> RelationNodeRef:
    """Create a validated relation node reference."""

    default_metadata: dict[str, object] = {"source": "relation_node_ref"}
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "relation_node_ref"
    return RelationNodeRef(
        node_id=node_id,
        kind=kind,
        label=label,
        confidence=confidence,
        metadata=merged_metadata,
    )


def create_relation_evidence(
    *,
    kind: RelationEvidenceKind = RelationEvidenceKind.UNKNOWN,
    source_id: str,
    confidence: float = 1.0,
    weight: float = 1.0,
    metadata: dict[str, object] | None = None,
) -> RelationEvidence:
    """Create relation evidence."""

    default_metadata: dict[str, object] = {"source": "relation_evidence"}
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "relation_evidence"
    return RelationEvidence(
        evidence_id=new_relation_evidence_id(),
        kind=kind,
        source_id=source_id,
        confidence=confidence,
        weight=weight,
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def create_relation_edge(
    *,
    source: RelationNodeRef,
    target: RelationNodeRef,
    kind: RelationEdgeKind = RelationEdgeKind.RELATES_TO,
    weight: float = 1.0,
    confidence: float = 1.0,
    evidence: tuple[RelationEvidence, ...] | list[RelationEvidence] = (),
    status: RelationStatus | None = None,
    metadata: dict[str, object] | None = None,
) -> RelationEdge:
    """Create a typed directed relation edge."""

    if status is None:
        status = relation_status_from_confidence(confidence=confidence)
    if not isinstance(status, RelationStatus):
        raise TypeError("status must be RelationStatus or None")
    now = utc_now()
    source_node_id = (
        source.node_id if isinstance(source, RelationNodeRef) else None
    )
    target_node_id = (
        target.node_id if isinstance(target, RelationNodeRef) else None
    )
    default_metadata: dict[str, object] = {
        "source": "relation_edge",
        "source_node_id": source_node_id,
        "target_node_id": target_node_id,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "relation_edge"
    return RelationEdge(
        edge_id=new_relation_edge_id(),
        source=source,
        target=target,
        kind=kind,
        status=status,
        weight=weight,
        confidence=confidence,
        evidence=evidence,
        created_at=now,
        updated_at=now,
        metadata=merged_metadata,
    )


def create_relation_activation(
    *,
    edge_id: UUID,
    activation: float = 1.0,
    decay: float = 0.0,
    update_count: int = 0,
    status: RelationActivationStatus | None = None,
    metadata: dict[str, object] | None = None,
) -> RelationActivation:
    """Create a relation activation descriptor."""

    if status is None:
        status = activation_status_from_value(activation=activation)
    if not isinstance(status, RelationActivationStatus):
        raise TypeError("status must be RelationActivationStatus or None")
    now = utc_now()
    default_metadata: dict[str, object] = {
        "source": "relation_activation",
        "edge_id": str(edge_id),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "relation_activation"
    return RelationActivation(
        activation_id=new_relation_activation_id(),
        edge_id=edge_id,
        status=status,
        activation=activation,
        decay=decay,
        update_count=update_count,
        created_at=now,
        updated_at=now,
        metadata=merged_metadata,
    )


def create_relation_graph_snapshot(
    *,
    nodes: tuple[RelationNodeRef, ...] | list[RelationNodeRef] = (),
    edges: tuple[RelationEdge, ...] | list[RelationEdge] = (),
    activations: tuple[RelationActivation, ...] | list[RelationActivation] = (),
    status: RelationStatus = RelationStatus.ACTIVE,
    metadata: dict[str, object] | None = None,
) -> RelationGraphSnapshot:
    """Create a relation graph snapshot."""

    nodes_value = _ensure_nodes(nodes)
    edges_value = _ensure_edges(edges)
    activations_value = _ensure_activations(activations)
    default_metadata: dict[str, object] = {
        "source": "relation_graph_snapshot",
        "node_count": len(nodes_value),
        "edge_count": len(edges_value),
        "activation_count": len(activations_value),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "relation_graph_snapshot"
    return RelationGraphSnapshot(
        snapshot_id=new_relation_snapshot_id(),
        nodes=nodes_value,
        edges=edges_value,
        activations=activations_value,
        status=status,
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def _ensure_nodes(
    nodes: tuple[RelationNodeRef, ...] | list[RelationNodeRef],
) -> tuple[RelationNodeRef, ...]:
    values = tuple(nodes)
    for node in values:
        if not isinstance(node, RelationNodeRef):
            raise TypeError("nodes must contain RelationNodeRef")
    return values


def _ensure_edges(
    edges: tuple[RelationEdge, ...] | list[RelationEdge],
) -> tuple[RelationEdge, ...]:
    values = tuple(edges)
    for edge in values:
        if not isinstance(edge, RelationEdge):
            raise TypeError("edges must contain RelationEdge")
    return values


def _ensure_activations(
    activations: tuple[RelationActivation, ...] | list[RelationActivation],
) -> tuple[RelationActivation, ...]:
    values = tuple(activations)
    for activation in values:
        if not isinstance(activation, RelationActivation):
            raise TypeError("activations must contain RelationActivation")
    return values


def _ensure_evidence(
    evidence: tuple[RelationEvidence, ...] | list[RelationEvidence],
) -> tuple[RelationEvidence, ...]:
    values = tuple(evidence)
    for item in values:
        if not isinstance(item, RelationEvidence):
            raise TypeError("evidence must contain RelationEvidence")
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
    "RelationNodeKind",
    "RelationEdgeKind",
    "RelationStatus",
    "RelationEvidenceKind",
    "RelationActivationStatus",
    "RelationNodeRef",
    "RelationEvidence",
    "RelationEdge",
    "RelationActivation",
    "RelationGraphSnapshot",
    "new_relation_evidence_id",
    "new_relation_edge_id",
    "new_relation_activation_id",
    "new_relation_snapshot_id",
    "ensure_non_empty_string",
    "ensure_optional_non_empty_string",
    "ensure_non_negative_int",
    "ensure_timezone_aware",
    "relation_status_from_confidence",
    "activation_status_from_value",
    "create_relation_node_ref",
    "create_relation_evidence",
    "create_relation_edge",
    "create_relation_activation",
    "create_relation_graph_snapshot",
]
