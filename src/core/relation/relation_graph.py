"""Deterministic in-memory relation graph utilities for CEBA Core Relation."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from ceba.core.relation.schema import (
    RelationActivation,
    RelationEdge,
    RelationEdgeKind,
    RelationGraphSnapshot,
    RelationNodeRef,
    RelationStatus,
    create_relation_graph_snapshot,
    ensure_non_empty_string,
)


@dataclass(frozen=True)
class RelationGraphIndex:
    """Immutable index structure for fast deterministic graph inspection."""

    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    activation_ids: tuple[str, ...]
    outgoing_edges_by_node: dict[str, tuple[RelationEdge, ...]]
    incoming_edges_by_node: dict[str, tuple[RelationEdge, ...]]
    edges_by_kind: dict[str, tuple[RelationEdge, ...]]
    activations_by_edge_id: dict[str, RelationActivation]

    def __post_init__(self) -> None:
        node_ids = _ensure_string_tuple(self.node_ids, "node_ids")
        edge_ids = _ensure_string_tuple(self.edge_ids, "edge_ids")
        activation_ids = _ensure_string_tuple(
            self.activation_ids,
            "activation_ids",
        )
        outgoing_edges_by_node = _copy_edge_mapping(
            self.outgoing_edges_by_node,
            "outgoing_edges_by_node",
        )
        incoming_edges_by_node = _copy_edge_mapping(
            self.incoming_edges_by_node,
            "incoming_edges_by_node",
        )
        edges_by_kind = _copy_edge_mapping(self.edges_by_kind, "edges_by_kind")
        activations_by_edge_id = _copy_activation_mapping(
            self.activations_by_edge_id,
            "activations_by_edge_id",
        )

        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "edge_ids", edge_ids)
        object.__setattr__(self, "activation_ids", activation_ids)
        object.__setattr__(
            self,
            "outgoing_edges_by_node",
            outgoing_edges_by_node,
        )
        object.__setattr__(
            self,
            "incoming_edges_by_node",
            incoming_edges_by_node,
        )
        object.__setattr__(self, "edges_by_kind", edges_by_kind)
        object.__setattr__(
            self,
            "activations_by_edge_id",
            activations_by_edge_id,
        )

    def to_dict(self) -> dict[str, object]:
        """Return serializable-friendly structural identifiers."""

        return {
            "node_ids": self.node_ids,
            "edge_ids": self.edge_ids,
            "activation_ids": self.activation_ids,
            "outgoing_edge_ids_by_node": {
                node_id: tuple(str(edge.edge_id) for edge in edges)
                for node_id, edges in self.outgoing_edges_by_node.items()
            },
            "incoming_edge_ids_by_node": {
                node_id: tuple(str(edge.edge_id) for edge in edges)
                for node_id, edges in self.incoming_edges_by_node.items()
            },
            "edge_ids_by_kind": {
                kind: tuple(str(edge.edge_id) for edge in edges)
                for kind, edges in self.edges_by_kind.items()
            },
            "activation_ids_by_edge_id": {
                edge_id: str(activation.activation_id)
                for edge_id, activation in self.activations_by_edge_id.items()
            },
        }


@dataclass(frozen=True)
class RelationGraphView:
    """Immutable runtime graph view over relation nodes, edges, activations."""

    nodes: tuple[RelationNodeRef, ...]
    edges: tuple[RelationEdge, ...]
    activations: tuple[RelationActivation, ...]
    status: RelationStatus
    index: RelationGraphIndex
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        nodes = _ensure_nodes(self.nodes)
        edges = _ensure_edges(self.edges)
        activations = _ensure_activations(self.activations)
        if not isinstance(self.status, RelationStatus):
            raise TypeError("status must be RelationStatus")
        if not isinstance(self.index, RelationGraphIndex):
            raise TypeError("index must be RelationGraphIndex")
        metadata = _copy_metadata(self.metadata)
        _validate_graph_integrity(nodes, edges, activations)

        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "activations", activations)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "nodes": tuple(node.to_dict() for node in self.nodes),
            "edges": tuple(edge.to_dict() for edge in self.edges),
            "activations": tuple(
                activation.to_dict() for activation in self.activations
            ),
            "status": self.status.value,
            "index": self.index.to_dict(),
            "metadata": dict(self.metadata),
        }

    def node_count(self) -> int:
        """Return the number of nodes in the view."""

        return len(self.nodes)

    def edge_count(self) -> int:
        """Return the number of edges in the view."""

        return len(self.edges)

    def activation_count(self) -> int:
        """Return the number of activations in the view."""

        return len(self.activations)

    def has_node(self, node_id: str) -> bool:
        """Return whether a node id exists in the view."""

        node_id = ensure_non_empty_string(node_id, "node_id")
        return node_id in self.index.node_ids

    def get_node(self, node_id: str) -> RelationNodeRef | None:
        """Return a node reference by id when present."""

        node_id = ensure_non_empty_string(node_id, "node_id")
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def get_outgoing_edges(
        self,
        node_id: str,
        kind: RelationEdgeKind | None = None,
    ) -> tuple[RelationEdge, ...]:
        """Return outgoing edges for a node, optionally filtered by kind."""

        node_id = ensure_non_empty_string(node_id, "node_id")
        if kind is not None and not isinstance(kind, RelationEdgeKind):
            raise TypeError("kind must be RelationEdgeKind or None")
        edges = self.index.outgoing_edges_by_node.get(node_id, ())
        if kind is None:
            return edges
        return tuple(edge for edge in edges if edge.kind is kind)

    def get_incoming_edges(
        self,
        node_id: str,
        kind: RelationEdgeKind | None = None,
    ) -> tuple[RelationEdge, ...]:
        """Return incoming edges for a node, optionally filtered by kind."""

        node_id = ensure_non_empty_string(node_id, "node_id")
        if kind is not None and not isinstance(kind, RelationEdgeKind):
            raise TypeError("kind must be RelationEdgeKind or None")
        edges = self.index.incoming_edges_by_node.get(node_id, ())
        if kind is None:
            return edges
        return tuple(edge for edge in edges if edge.kind is kind)

    def get_edges_by_kind(
        self,
        kind: RelationEdgeKind,
    ) -> tuple[RelationEdge, ...]:
        """Return edges grouped under a relation edge kind."""

        if not isinstance(kind, RelationEdgeKind):
            raise TypeError("kind must be RelationEdgeKind")
        return self.index.edges_by_kind.get(kind.value, ())

    def get_activation_for_edge(
        self,
        edge_id: str,
    ) -> RelationActivation | None:
        """Return activation for an edge id when present."""

        edge_id = ensure_non_empty_string(edge_id, "edge_id")
        return self.index.activations_by_edge_id.get(edge_id)

    def to_snapshot(self) -> RelationGraphSnapshot:
        """Return a graph snapshot for this view."""

        return create_relation_graph_snapshot(
            nodes=self.nodes,
            edges=self.edges,
            activations=self.activations,
            status=self.status,
            metadata=self.metadata,
        )


def build_relation_graph_view(
    *,
    nodes: tuple[RelationNodeRef, ...] | list[RelationNodeRef],
    edges: tuple[RelationEdge, ...] | list[RelationEdge],
    activations: tuple[RelationActivation, ...] | list[RelationActivation] = (),
    status: RelationStatus = RelationStatus.ACTIVE,
    metadata: dict[str, object] | None = None,
) -> RelationGraphView:
    """Build an immutable deterministic relation graph view."""

    nodes_value = _ensure_nodes(nodes)
    edges_value = _ensure_edges(edges)
    activations_value = _ensure_activations(activations)
    if not isinstance(status, RelationStatus):
        raise TypeError("status must be RelationStatus")
    _validate_graph_integrity(nodes_value, edges_value, activations_value)
    index = _build_index(nodes_value, edges_value, activations_value)
    default_metadata: dict[str, object] = {
        "source": "relation_graph_view",
        "node_count": len(nodes_value),
        "edge_count": len(edges_value),
        "activation_count": len(activations_value),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "relation_graph_view"
    return RelationGraphView(
        nodes=nodes_value,
        edges=edges_value,
        activations=activations_value,
        status=status,
        index=index,
        metadata=merged_metadata,
    )


def build_relation_graph_snapshot(
    *,
    nodes: tuple[RelationNodeRef, ...] | list[RelationNodeRef],
    edges: tuple[RelationEdge, ...] | list[RelationEdge],
    activations: tuple[RelationActivation, ...] | list[RelationActivation] = (),
    status: RelationStatus = RelationStatus.ACTIVE,
    metadata: dict[str, object] | None = None,
) -> RelationGraphSnapshot:
    """Build a relation graph snapshot from graph view inputs."""

    view = build_relation_graph_view(
        nodes=nodes,
        edges=edges,
        activations=activations,
        status=status,
        metadata=metadata,
    )
    return view.to_snapshot()


def _build_index(
    nodes: tuple[RelationNodeRef, ...],
    edges: tuple[RelationEdge, ...],
    activations: tuple[RelationActivation, ...],
) -> RelationGraphIndex:
    node_ids = tuple(sorted(node.node_id for node in nodes))
    edge_ids = tuple(sorted(str(edge.edge_id) for edge in edges))
    activation_ids = tuple(
        sorted(str(activation.activation_id) for activation in activations)
    )

    outgoing: dict[str, list[RelationEdge]] = {}
    incoming: dict[str, list[RelationEdge]] = {}
    by_kind: dict[str, list[RelationEdge]] = {}
    for edge in edges:
        outgoing.setdefault(edge.source.node_id, []).append(edge)
        incoming.setdefault(edge.target.node_id, []).append(edge)
        by_kind.setdefault(edge.kind.value, []).append(edge)

    outgoing_edges_by_node = {
        node_id: _sort_edges(tuple(edge_values))
        for node_id, edge_values in outgoing.items()
    }
    incoming_edges_by_node = {
        node_id: _sort_edges(tuple(edge_values))
        for node_id, edge_values in incoming.items()
    }
    edges_by_kind = {
        kind: _sort_edges(tuple(edge_values))
        for kind, edge_values in by_kind.items()
    }
    activations_by_edge_id = {
        str(activation.edge_id): activation for activation in activations
    }

    return RelationGraphIndex(
        node_ids=node_ids,
        edge_ids=edge_ids,
        activation_ids=activation_ids,
        outgoing_edges_by_node=outgoing_edges_by_node,
        incoming_edges_by_node=incoming_edges_by_node,
        edges_by_kind=edges_by_kind,
        activations_by_edge_id=activations_by_edge_id,
    )


def _sort_edges(edges: tuple[RelationEdge, ...]) -> tuple[RelationEdge, ...]:
    return tuple(
        sorted(
            edges,
            key=lambda edge: (
                edge.source.node_id,
                edge.target.node_id,
                edge.kind.value,
                -edge.confidence,
                -edge.weight,
                str(edge.edge_id),
            ),
        )
    )


def _validate_graph_integrity(
    nodes: tuple[RelationNodeRef, ...],
    edges: tuple[RelationEdge, ...],
    activations: tuple[RelationActivation, ...],
) -> None:
    node_ids: set[str] = set()
    for node in nodes:
        if node.node_id in node_ids:
            raise ValueError("duplicate node_id")
        node_ids.add(node.node_id)

    edge_ids: set[UUID] = set()
    for edge in edges:
        if edge.edge_id in edge_ids:
            raise ValueError("duplicate edge_id")
        edge_ids.add(edge.edge_id)
        if edge.source.node_id not in node_ids:
            raise ValueError("edge source node_id must exist in nodes")
        if edge.target.node_id not in node_ids:
            raise ValueError("edge target node_id must exist in nodes")

    activation_ids: set[UUID] = set()
    for activation in activations:
        if activation.activation_id in activation_ids:
            raise ValueError("duplicate activation_id")
        activation_ids.add(activation.activation_id)
        if activation.edge_id not in edge_ids:
            raise ValueError("activation edge_id must exist in edges")


def _ensure_nodes(
    nodes: tuple[RelationNodeRef, ...] | list[RelationNodeRef],
) -> tuple[RelationNodeRef, ...]:
    if not isinstance(nodes, (tuple, list)):
        raise TypeError("nodes must be tuple or list")
    values = tuple(nodes)
    for node in values:
        if not isinstance(node, RelationNodeRef):
            raise TypeError("nodes must contain RelationNodeRef")
    return values


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


def _ensure_activations(
    activations: tuple[RelationActivation, ...] | list[RelationActivation],
) -> tuple[RelationActivation, ...]:
    if not isinstance(activations, (tuple, list)):
        raise TypeError("activations must be tuple or list")
    values = tuple(activations)
    for activation in values:
        if not isinstance(activation, RelationActivation):
            raise TypeError("activations must contain RelationActivation")
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


def _copy_edge_mapping(
    mapping: dict[str, tuple[RelationEdge, ...] | list[RelationEdge]],
    field_name: str,
) -> dict[str, tuple[RelationEdge, ...]]:
    if not isinstance(mapping, dict):
        raise TypeError(f"{field_name} must be dict")
    copied: dict[str, tuple[RelationEdge, ...]] = {}
    for key, values in mapping.items():
        key = ensure_non_empty_string(key, field_name)
        copied[key] = _ensure_edges(values)
    return copied


def _copy_activation_mapping(
    mapping: dict[str, RelationActivation],
    field_name: str,
) -> dict[str, RelationActivation]:
    if not isinstance(mapping, dict):
        raise TypeError(f"{field_name} must be dict")
    copied: dict[str, RelationActivation] = {}
    for key, activation in mapping.items():
        key = ensure_non_empty_string(key, field_name)
        if not isinstance(activation, RelationActivation):
            raise TypeError(f"{field_name} values must be RelationActivation")
        copied[key] = activation
    return copied


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
    "RelationGraphIndex",
    "RelationGraphView",
    "build_relation_graph_view",
    "build_relation_graph_snapshot",
]
