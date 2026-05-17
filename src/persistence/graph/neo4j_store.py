"""Deterministic local Neo4j adapter descriptors for CEBA graph persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from ceba.persistence.graph.schema import (
    GraphBackend,
    GraphEdgeRef,
    GraphElementStatus,
    GraphIndexConfig,
    GraphIndexSnapshot,
    GraphIndexStatus,
    GraphNodeRef,
    GraphPathMatch,
    GraphTraversalQuery,
    create_graph_index_config,
    create_graph_index_snapshot,
    create_graph_path_match,
    ensure_non_empty_string,
    ensure_timezone_aware,
    utc_now,
)


__all__ = [
    "Neo4jOperationKind",
    "Neo4jOperationStatus",
    "Neo4jGraphConfig",
    "Neo4jOperation",
    "Neo4jOperationResult",
    "Neo4jStoreAdapter",
    "create_neo4j_graph_config",
    "build_neo4j_store_adapter",
]


_FORBIDDEN = {"M_t", "Mt", "semantic_state", "runtime_semantic_state"}


class Neo4jOperationKind(Enum):
    CREATE_GRAPH = "CREATE_GRAPH"
    UPSERT_NODES = "UPSERT_NODES"
    UPSERT_EDGES = "UPSERT_EDGES"
    TRAVERSE = "TRAVERSE"
    DELETE_NODES = "DELETE_NODES"
    DELETE_EDGES = "DELETE_EDGES"
    SNAPSHOT = "SNAPSHOT"
    HEALTH_CHECK = "HEALTH_CHECK"
    UNKNOWN = "UNKNOWN"


class Neo4jOperationStatus(Enum):
    CREATED = "CREATED"
    READY = "READY"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Neo4jGraphConfig:
    graph_name: str
    index_config: GraphIndexConfig
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        graph_name = ensure_non_empty_string(self.graph_name, "graph_name")
        _reject_name(graph_name, "graph_name")
        if not isinstance(self.index_config, GraphIndexConfig):
            raise TypeError("index_config must be GraphIndexConfig")
        if self.index_config.backend is not GraphBackend.NEO4J:
            raise ValueError("index_config.backend must be GraphBackend.NEO4J")
        if self.index_config.graph_name != graph_name:
            raise ValueError("index_config.graph_name must equal graph_name")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "graph_name", graph_name)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        return {
            "graph_name": self.graph_name,
            "index_config": self.index_config.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Neo4jOperation:
    operation_id: UUID
    kind: Neo4jOperationKind
    status: Neo4jOperationStatus
    graph_name: str
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.operation_id, UUID):
            raise TypeError("operation_id must be UUID")
        if not isinstance(self.kind, Neo4jOperationKind):
            raise TypeError("kind must be Neo4jOperationKind")
        if not isinstance(self.status, Neo4jOperationStatus):
            raise TypeError("status must be Neo4jOperationStatus")
        graph_name = ensure_non_empty_string(self.graph_name, "graph_name")
        _reject_name(graph_name, "graph_name")
        node_ids = _id_tuple(self.node_ids, "node_ids")
        edge_ids = _id_tuple(self.edge_ids, "edge_ids")
        _unique(node_ids, "node_ids")
        _unique(edge_ids, "edge_ids")
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "graph_name", graph_name)
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "edge_ids", edge_ids)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        return {
            "operation_id": str(self.operation_id),
            "kind": self.kind.value,
            "status": self.status.value,
            "graph_name": self.graph_name,
            "node_ids": list(self.node_ids),
            "edge_ids": list(self.edge_ids),
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    def node_count(self) -> int:
        return len(self.node_ids)

    def edge_count(self) -> int:
        return len(self.edge_ids)


@dataclass(frozen=True)
class Neo4jOperationResult:
    operation: Neo4jOperation
    accepted_nodes: tuple[GraphNodeRef, ...]
    accepted_edges: tuple[GraphEdgeRef, ...]
    matches: tuple[GraphPathMatch, ...]
    snapshot: GraphIndexSnapshot | None
    changed: bool
    reason: str
    rejected_ids: tuple[str, ...]
    rejection_reasons: tuple[str, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.operation, Neo4jOperation):
            raise TypeError("operation must be Neo4jOperation")
        accepted_nodes = _node_tuple(self.accepted_nodes)
        accepted_edges = _edge_tuple(self.accepted_edges)
        matches = _match_tuple(self.matches)
        if self.snapshot is not None and not isinstance(self.snapshot, GraphIndexSnapshot):
            raise TypeError("snapshot must be None or GraphIndexSnapshot")
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        reason = ensure_non_empty_string(self.reason, "reason")
        rejected_ids = _id_tuple(self.rejected_ids, "rejected_ids")
        rejection_reasons = _id_tuple(self.rejection_reasons, "rejection_reasons")
        if len(rejected_ids) != len(rejection_reasons):
            raise ValueError("rejected_ids and rejection_reasons must have same length")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "accepted_nodes", accepted_nodes)
        object.__setattr__(self, "accepted_edges", accepted_edges)
        object.__setattr__(self, "matches", matches)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "rejected_ids", rejected_ids)
        object.__setattr__(self, "rejection_reasons", rejection_reasons)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        return {
            "operation": self.operation.to_dict(),
            "accepted_nodes": [node.to_dict() for node in self.accepted_nodes],
            "accepted_edges": [edge.to_dict() for edge in self.accepted_edges],
            "matches": [match.to_dict() for match in self.matches],
            "snapshot": None if self.snapshot is None else self.snapshot.to_dict(),
            "changed": self.changed,
            "reason": self.reason,
            "rejected_ids": list(self.rejected_ids),
            "rejection_reasons": list(self.rejection_reasons),
            "metadata": dict(self.metadata),
        }

    def accepted_node_count(self) -> int:
        return len(self.accepted_nodes)

    def accepted_edge_count(self) -> int:
        return len(self.accepted_edges)

    def match_count(self) -> int:
        return len(self.matches)

    def rejected_count(self) -> int:
        return len(self.rejected_ids)

    def has_snapshot(self) -> bool:
        return self.snapshot is not None

    def is_success(self) -> bool:
        return (
            self.changed
            and self.operation.status in {Neo4jOperationStatus.ACCEPTED, Neo4jOperationStatus.READY}
            and self.rejected_count() == 0
        )


class Neo4jStoreAdapter:
    def __init__(self, *, graph: Neo4jGraphConfig) -> None:
        if not isinstance(graph, Neo4jGraphConfig):
            raise TypeError("graph must be Neo4jGraphConfig")
        self._graph = graph

    def graph(self) -> Neo4jGraphConfig:
        return self._graph

    def create_graph_operation(self) -> Neo4jOperationResult:
        metadata = _op_metadata(self._graph.graph_name, "create_graph_operation")
        operation = _operation(
            kind=Neo4jOperationKind.CREATE_GRAPH,
            status=Neo4jOperationStatus.READY,
            graph_name=self._graph.graph_name,
            metadata=metadata,
        )
        return _result(operation=operation, changed=True, reason="neo4j graph operation prepared", metadata=metadata)

    def prepare_node_upsert(
        self,
        *,
        nodes: tuple[GraphNodeRef, ...] | list[GraphNodeRef],
        metadata: dict[str, object] | None = None,
    ) -> Neo4jOperationResult:
        accepted: list[GraphNodeRef] = []
        rejected: list[str] = []
        reasons: list[str] = []
        seen: set[str] = set()
        for node in _node_tuple(nodes):
            node_id = str(node.node_id)
            reason = None
            if node.status not in {GraphElementStatus.CREATED, GraphElementStatus.INDEXED}:
                reason = f"graph node {node.status.value.lower()}"
            elif node_id in seen:
                reason = "graph node_id duplicated"
            if reason is None:
                seen.add(node_id)
                accepted.append(node)
            else:
                rejected.append(node_id)
                reasons.append(reason)
        meta = _merge(_op_metadata(self._graph.graph_name, "prepare_node_upsert"), metadata)
        operation = _operation(
            kind=Neo4jOperationKind.UPSERT_NODES,
            status=Neo4jOperationStatus.ACCEPTED if accepted else Neo4jOperationStatus.REJECTED,
            graph_name=self._graph.graph_name,
            node_ids=tuple(str(node.node_id) for node in accepted),
            metadata=meta,
        )
        return _result(
            operation=operation,
            accepted_nodes=tuple(accepted),
            changed=bool(accepted),
            reason="neo4j node upsert operation prepared" if accepted else "neo4j node upsert operation rejected",
            rejected_ids=tuple(rejected),
            rejection_reasons=tuple(reasons),
            metadata=meta,
        )

    def prepare_edge_upsert(
        self,
        *,
        nodes: tuple[GraphNodeRef, ...] | list[GraphNodeRef],
        edges: tuple[GraphEdgeRef, ...] | list[GraphEdgeRef],
        metadata: dict[str, object] | None = None,
    ) -> Neo4jOperationResult:
        node_ids = {node.node_id for node in _node_tuple(nodes)}
        accepted: list[GraphEdgeRef] = []
        rejected: list[str] = []
        reasons: list[str] = []
        seen: set[str] = set()
        for edge in _edge_tuple(edges):
            edge_id = str(edge.edge_id)
            reason = None
            if edge.source_node_id not in node_ids or edge.target_node_id not in node_ids:
                reason = "graph edge endpoint missing"
            elif edge.status not in {GraphElementStatus.CREATED, GraphElementStatus.INDEXED}:
                reason = f"graph edge {edge.status.value.lower()}"
            elif edge_id in seen:
                reason = "graph edge_id duplicated"
            if reason is None:
                seen.add(edge_id)
                accepted.append(edge)
            else:
                rejected.append(edge_id)
                reasons.append(reason)
        meta = _merge(_op_metadata(self._graph.graph_name, "prepare_edge_upsert"), metadata)
        operation = _operation(
            kind=Neo4jOperationKind.UPSERT_EDGES,
            status=Neo4jOperationStatus.ACCEPTED if accepted else Neo4jOperationStatus.REJECTED,
            graph_name=self._graph.graph_name,
            edge_ids=tuple(str(edge.edge_id) for edge in accepted),
            metadata=meta,
        )
        return _result(
            operation=operation,
            accepted_edges=tuple(accepted),
            changed=bool(accepted),
            reason="neo4j edge upsert operation prepared" if accepted else "neo4j edge upsert operation rejected",
            rejected_ids=tuple(rejected),
            rejection_reasons=tuple(reasons),
            metadata=meta,
        )

    def prepare_traversal(
        self,
        *,
        query: GraphTraversalQuery,
        candidate_matches: tuple[GraphPathMatch, ...] | list[GraphPathMatch] = (),
        metadata: dict[str, object] | None = None,
    ) -> Neo4jOperationResult:
        if not isinstance(query, GraphTraversalQuery):
            raise TypeError("query must be GraphTraversalQuery")
        matches = [
            match
            for match in _match_tuple(candidate_matches)
            if match.score >= query.min_confidence
        ]
        matches = sorted(matches, key=lambda match: (-match.score, match.rank))
        reranked = tuple(
            create_graph_path_match(
                nodes=match.nodes,
                edges=match.edges,
                score=match.score,
                rank=index,
                metadata=match.metadata,
            )
            for index, match in enumerate(matches[: query.limit], start=1)
        )
        meta = _merge(_op_metadata(self._graph.graph_name, "prepare_traversal"), metadata)
        operation = _operation(
            kind=Neo4jOperationKind.TRAVERSE,
            status=Neo4jOperationStatus.ACCEPTED,
            graph_name=self._graph.graph_name,
            metadata=meta,
        )
        return _result(operation=operation, matches=reranked, changed=False, reason="neo4j traversal operation prepared", metadata=meta)

    def prepare_delete_nodes(
        self,
        *,
        node_ids: tuple[str, ...] | list[str],
        metadata: dict[str, object] | None = None,
    ) -> Neo4jOperationResult:
        ids = _id_tuple(node_ids, "node_ids")
        _unique(ids, "node_ids")
        meta = _merge(_op_metadata(self._graph.graph_name, "prepare_delete_nodes"), metadata)
        operation = _operation(
            kind=Neo4jOperationKind.DELETE_NODES,
            status=Neo4jOperationStatus.ACCEPTED if ids else Neo4jOperationStatus.REJECTED,
            graph_name=self._graph.graph_name,
            node_ids=ids,
            metadata=meta,
        )
        return _result(operation=operation, changed=bool(ids), reason="neo4j node delete operation prepared" if ids else "neo4j node delete operation rejected", metadata=meta)

    def prepare_delete_edges(
        self,
        *,
        edge_ids: tuple[str, ...] | list[str],
        metadata: dict[str, object] | None = None,
    ) -> Neo4jOperationResult:
        ids = _id_tuple(edge_ids, "edge_ids")
        _unique(ids, "edge_ids")
        meta = _merge(_op_metadata(self._graph.graph_name, "prepare_delete_edges"), metadata)
        operation = _operation(
            kind=Neo4jOperationKind.DELETE_EDGES,
            status=Neo4jOperationStatus.ACCEPTED if ids else Neo4jOperationStatus.REJECTED,
            graph_name=self._graph.graph_name,
            edge_ids=ids,
            metadata=meta,
        )
        return _result(operation=operation, changed=bool(ids), reason="neo4j edge delete operation prepared" if ids else "neo4j edge delete operation rejected", metadata=meta)

    def create_snapshot(
        self,
        *,
        nodes: tuple[GraphNodeRef, ...] | list[GraphNodeRef],
        edges: tuple[GraphEdgeRef, ...] | list[GraphEdgeRef],
        metadata: dict[str, object] | None = None,
    ) -> Neo4jOperationResult:
        node_tuple = _node_tuple(nodes)
        edge_tuple = _edge_tuple(edges)
        meta = _merge(_op_metadata(self._graph.graph_name, "create_snapshot"), metadata)
        snapshot = create_graph_index_snapshot(
            config=self._graph.index_config,
            nodes=node_tuple,
            edges=edge_tuple,
            status=self._graph.index_config.status,
            metadata=meta,
        )
        operation = _operation(
            kind=Neo4jOperationKind.SNAPSHOT,
            status=Neo4jOperationStatus.READY,
            graph_name=self._graph.graph_name,
            node_ids=tuple(str(node.node_id) for node in node_tuple),
            edge_ids=tuple(str(edge.edge_id) for edge in edge_tuple),
            metadata=meta,
        )
        return _result(operation=operation, accepted_nodes=node_tuple, accepted_edges=edge_tuple, snapshot=snapshot, changed=True, reason="neo4j snapshot created", metadata=meta)


def create_neo4j_graph_config(
    *,
    graph_name: str,
    status: GraphIndexStatus = GraphIndexStatus.CREATED,
    metadata: dict[str, object] | None = None,
) -> Neo4jGraphConfig:
    meta = _merge({"source": "neo4j_graph_config", "backend": "NEO4J", "graph_name": graph_name}, metadata)
    index_config = create_graph_index_config(
        backend=GraphBackend.NEO4J,
        graph_name=graph_name,
        status=status,
        metadata=meta,
    )
    return Neo4jGraphConfig(graph_name=graph_name, index_config=index_config, metadata=meta)


def build_neo4j_store_adapter(*, graph: Neo4jGraphConfig) -> Neo4jStoreAdapter:
    return Neo4jStoreAdapter(graph=graph)


def _operation(
    *,
    kind: Neo4jOperationKind,
    status: Neo4jOperationStatus,
    graph_name: str,
    node_ids: tuple[str, ...] | list[str] = (),
    edge_ids: tuple[str, ...] | list[str] = (),
    metadata: dict[str, object],
) -> Neo4jOperation:
    return Neo4jOperation(uuid4(), kind, status, graph_name, tuple(node_ids), tuple(edge_ids), utc_now(), metadata)


def _result(
    *,
    operation: Neo4jOperation,
    accepted_nodes: tuple[GraphNodeRef, ...] | list[GraphNodeRef] = (),
    accepted_edges: tuple[GraphEdgeRef, ...] | list[GraphEdgeRef] = (),
    matches: tuple[GraphPathMatch, ...] | list[GraphPathMatch] = (),
    snapshot: GraphIndexSnapshot | None = None,
    changed: bool,
    reason: str,
    rejected_ids: tuple[str, ...] | list[str] = (),
    rejection_reasons: tuple[str, ...] | list[str] = (),
    metadata: dict[str, object],
) -> Neo4jOperationResult:
    return Neo4jOperationResult(operation, tuple(accepted_nodes), tuple(accepted_edges), tuple(matches), snapshot, changed, reason, tuple(rejected_ids), tuple(rejection_reasons), metadata)


def _op_metadata(graph_name: str, operation: str) -> dict[str, object]:
    return {"source": "neo4j_store_adapter", "operation": operation, "backend": "NEO4J", "graph_name": graph_name}


def _copy_metadata(value: dict[str, object]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(value)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
        if key in _FORBIDDEN:
            raise ValueError("metadata must not persist runtime semantic state")
    return copied


def _merge(defaults: dict[str, object], metadata: dict[str, object] | None) -> dict[str, object]:
    merged = _copy_metadata(defaults)
    if metadata is not None:
        merged.update(_copy_metadata(metadata))
    if "source" in defaults:
        merged["source"] = defaults["source"]
    return merged


def _reject_name(value: str, field_name: str) -> None:
    if value in _FORBIDDEN:
        raise ValueError(f"{field_name} must not persist runtime semantic state")


def _id_tuple(value: tuple[str, ...] | list[str], field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple[str, ...]")
    return tuple(ensure_non_empty_string(item, field_name) for item in value)


def _unique(values: tuple[str, ...], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} values must be unique")


def _node_tuple(value: tuple[GraphNodeRef, ...] | list[GraphNodeRef]) -> tuple[GraphNodeRef, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("nodes must be tuple[GraphNodeRef, ...]")
    nodes = tuple(value)
    for node in nodes:
        if not isinstance(node, GraphNodeRef):
            raise TypeError("every node must be GraphNodeRef")
    return nodes


def _edge_tuple(value: tuple[GraphEdgeRef, ...] | list[GraphEdgeRef]) -> tuple[GraphEdgeRef, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("edges must be tuple[GraphEdgeRef, ...]")
    edges = tuple(value)
    for edge in edges:
        if not isinstance(edge, GraphEdgeRef):
            raise TypeError("every edge must be GraphEdgeRef")
    return edges


def _match_tuple(value: tuple[GraphPathMatch, ...] | list[GraphPathMatch]) -> tuple[GraphPathMatch, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("matches must be tuple[GraphPathMatch, ...]")
    matches = tuple(value)
    for match in matches:
        if not isinstance(match, GraphPathMatch):
            raise TypeError("every match must be GraphPathMatch")
    return matches
