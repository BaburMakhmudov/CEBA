"""Deterministic local relation repository for CEBA graph persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ceba.persistence.graph.schema import (
    GraphEdgeKind,
    GraphEdgeRef,
    GraphElementStatus,
    GraphNodeRef,
    create_graph_edge_ref,
    create_graph_payload,
    ensure_finite_number,
    ensure_non_empty_string,
    ensure_optional_non_empty_string,
    ensure_positive_int,
    ensure_probability,
)


__all__ = [
    "RelationStatusFilter",
    "RelationRepositoryPolicy",
    "RelationInput",
    "RelationRepositoryQuery",
    "RelationRepositoryResult",
    "RelationRepository",
    "build_relation_repository",
]


_FORBIDDEN = {"M_t", "Mt", "semantic_state", "runtime_semantic_state"}


class RelationStatusFilter(Enum):
    ANY = "ANY"
    CREATED = "CREATED"
    INDEXED = "INDEXED"
    STALE = "STALE"
    DELETED = "DELETED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class RelationRepositoryPolicy:
    max_relations: int = 100000
    allow_duplicate_relation_keys: bool = False
    reject_empty_payload: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_relations", ensure_positive_int(self.max_relations, "max_relations"))
        if not isinstance(self.allow_duplicate_relation_keys, bool):
            raise TypeError("allow_duplicate_relation_keys must be bool")
        if not isinstance(self.reject_empty_payload, bool):
            raise TypeError("reject_empty_payload must be bool")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_relations": self.max_relations,
            "allow_duplicate_relation_keys": self.allow_duplicate_relation_keys,
            "reject_empty_payload": self.reject_empty_payload,
        }


@dataclass(frozen=True)
class RelationInput:
    relation_key: str
    source_node: GraphNodeRef
    target_node: GraphNodeRef
    kind: GraphEdgeKind = GraphEdgeKind.RELATES_TO
    weight: float = 1.0
    confidence: float = 1.0
    source_record_id: str | None = None
    source_record_key: str | None = None
    payload_data: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        relation_key = ensure_non_empty_string(self.relation_key, "relation_key")
        _reject_name(relation_key, "relation_key")
        if not isinstance(self.source_node, GraphNodeRef):
            raise TypeError("source_node must be GraphNodeRef")
        if not isinstance(self.target_node, GraphNodeRef):
            raise TypeError("target_node must be GraphNodeRef")
        if self.source_node.node_id == self.target_node.node_id:
            raise ValueError("source_node and target_node must differ")
        if not isinstance(self.kind, GraphEdgeKind):
            raise TypeError("kind must be GraphEdgeKind")
        weight = ensure_finite_number(self.weight, "weight")
        if weight < 0.0:
            raise ValueError("weight must be non-negative")
        confidence = ensure_probability(self.confidence, "confidence")
        payload_data = _copy_dict(self.payload_data, "payload_data")
        metadata = _copy_dict(self.metadata, "metadata")
        _reject_keys(payload_data, "payload_data")
        _reject_keys(metadata, "metadata")
        object.__setattr__(self, "relation_key", relation_key)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "source_record_id", ensure_optional_non_empty_string(self.source_record_id, "source_record_id"))
        object.__setattr__(self, "source_record_key", ensure_optional_non_empty_string(self.source_record_key, "source_record_key"))
        object.__setattr__(self, "payload_data", payload_data)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        return {
            "relation_key": self.relation_key,
            "source_node_id": str(self.source_node.node_id),
            "target_node_id": str(self.target_node.node_id),
            "kind": self.kind.value,
            "weight": self.weight,
            "confidence": self.confidence,
            "payload_data": dict(self.payload_data),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RelationRepositoryQuery:
    relation_key: str | None = None
    source_node_id: str | None = None
    target_node_id: str | None = None
    edge_kind: GraphEdgeKind | None = None
    status_filter: RelationStatusFilter = RelationStatusFilter.ANY
    limit: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation_key", ensure_optional_non_empty_string(self.relation_key, "relation_key"))
        object.__setattr__(self, "source_node_id", ensure_optional_non_empty_string(self.source_node_id, "source_node_id"))
        object.__setattr__(self, "target_node_id", ensure_optional_non_empty_string(self.target_node_id, "target_node_id"))
        if self.edge_kind is not None and not isinstance(self.edge_kind, GraphEdgeKind):
            raise TypeError("edge_kind must be None or GraphEdgeKind")
        if not isinstance(self.status_filter, RelationStatusFilter):
            raise TypeError("status_filter must be RelationStatusFilter")
        object.__setattr__(self, "limit", ensure_positive_int(self.limit, "limit"))

    def to_dict(self) -> dict[str, object]:
        return {
            "relation_key": self.relation_key,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "edge_kind": None if self.edge_kind is None else self.edge_kind.value,
            "status_filter": self.status_filter.value,
            "limit": self.limit,
        }


@dataclass(frozen=True)
class RelationRepositoryResult:
    edges: tuple[GraphEdgeRef, ...]
    changed: bool
    reason: str
    rejected_relation_keys: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        edges = _edge_tuple(self.edges)
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        reason = ensure_non_empty_string(self.reason, "reason")
        rejected = _str_tuple(self.rejected_relation_keys, "rejected_relation_keys")
        reasons = _str_tuple(self.rejection_reasons, "rejection_reasons")
        if len(rejected) != len(reasons):
            raise ValueError("rejected_relation_keys and rejection_reasons must have same length")
        metadata = _copy_dict(self.metadata, "metadata")
        _reject_keys(metadata, "metadata")
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "rejected_relation_keys", rejected)
        object.__setattr__(self, "rejection_reasons", reasons)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        return {
            "edges": [edge.to_dict() for edge in self.edges],
            "changed": self.changed,
            "reason": self.reason,
            "rejected_relation_keys": list(self.rejected_relation_keys),
            "rejection_reasons": list(self.rejection_reasons),
            "metadata": dict(self.metadata),
        }


class RelationRepository:
    def __init__(
        self,
        *,
        edges: tuple[GraphEdgeRef, ...] | list[GraphEdgeRef] = (),
        policy: RelationRepositoryPolicy | None = None,
    ) -> None:
        self._policy = RelationRepositoryPolicy() if policy is None else policy
        if not isinstance(self._policy, RelationRepositoryPolicy):
            raise TypeError("policy must be RelationRepositoryPolicy")
        self._edges = _edge_tuple(edges)
        self._validate_edges(self._edges)

    def edges(self) -> tuple[GraphEdgeRef, ...]:
        return self._edges

    def edge_count(self) -> int:
        return len(self._edges)

    def create_relation(self, *, relation_input: RelationInput) -> RelationRepositoryResult:
        if not isinstance(relation_input, RelationInput):
            raise TypeError("relation_input must be RelationInput")
        if self._policy.reject_empty_payload and not relation_input.payload_data:
            return _result((), False, "relation payload is empty", (relation_input.relation_key,), ("relation payload is empty",))
        if len(self._edges) >= self._policy.max_relations:
            return _result((), False, "relation limit exceeded", (relation_input.relation_key,), ("relation limit exceeded",))
        if not self._policy.allow_duplicate_relation_keys and any(edge.edge_key == relation_input.relation_key for edge in self._edges):
            return _result((), False, "relation key already exists", (relation_input.relation_key,), ("relation key already exists",))
        payload = create_graph_payload(
            source_record_id=relation_input.source_record_id,
            source_record_key=relation_input.source_record_key,
            data=relation_input.payload_data,
            metadata={"source": "relation_repository", "operation": "create_relation_payload"},
        )
        edge = create_graph_edge_ref(
            edge_key=relation_input.relation_key,
            source_node_id=relation_input.source_node.node_id,
            target_node_id=relation_input.target_node.node_id,
            kind=relation_input.kind,
            weight=relation_input.weight,
            confidence=relation_input.confidence,
            payload=payload,
            status=GraphElementStatus.CREATED,
            metadata=_merge({"source": "relation_repository", "operation": "create_relation"}, relation_input.metadata),
        )
        self._edges = self._edges + (edge,)
        return _result((edge,), True, "relation created")

    def get_relation_by_key(self, *, relation_key: str) -> RelationRepositoryResult:
        key = ensure_non_empty_string(relation_key, "relation_key")
        for edge in self._edges:
            if edge.edge_key == key:
                return _result((edge,), False, "relation found")
        return _result((), False, "relation missing")

    def query_relations(self, *, query: RelationRepositoryQuery) -> RelationRepositoryResult:
        if not isinstance(query, RelationRepositoryQuery):
            raise TypeError("query must be RelationRepositoryQuery")
        edges = list(self._edges)
        if query.relation_key is not None:
            edges = [edge for edge in edges if edge.edge_key == query.relation_key]
        if query.source_node_id is not None:
            edges = [edge for edge in edges if str(edge.source_node_id) == query.source_node_id]
        if query.target_node_id is not None:
            edges = [edge for edge in edges if str(edge.target_node_id) == query.target_node_id]
        if query.edge_kind is not None:
            edges = [edge for edge in edges if edge.kind is query.edge_kind]
        if query.status_filter is not RelationStatusFilter.ANY:
            edges = [edge for edge in edges if edge.status.value == query.status_filter.value]
        edges = sorted(edges, key=lambda edge: (-edge.updated_at.timestamp(), edge.edge_key))
        return _result(tuple(edges[: query.limit]), False, "relations queried")

    def mark_indexed(self, *, edge_id: str) -> RelationRepositoryResult:
        return self._mark(edge_id=edge_id, status=GraphElementStatus.INDEXED, reason="relation indexed")

    def mark_stale(self, *, edge_id: str) -> RelationRepositoryResult:
        return self._mark(edge_id=edge_id, status=GraphElementStatus.STALE, reason="relation stale")

    def mark_deleted(self, *, edge_id: str) -> RelationRepositoryResult:
        return self._mark(edge_id=edge_id, status=GraphElementStatus.DELETED, reason="relation deleted")

    def replace_edges(self, *, edges: tuple[GraphEdgeRef, ...] | list[GraphEdgeRef]) -> RelationRepositoryResult:
        new_edges = _edge_tuple(edges)
        self._validate_edges(new_edges)
        self._edges = new_edges
        return _result(new_edges, True, "relation edges replaced")

    def _mark(self, *, edge_id: str, status: GraphElementStatus, reason: str) -> RelationRepositoryResult:
        normalized = ensure_non_empty_string(edge_id, "edge_id")
        updated: list[GraphEdgeRef] = []
        changed_edge: GraphEdgeRef | None = None
        for edge in self._edges:
            if str(edge.edge_id) == normalized:
                changed_edge = GraphEdgeRef(edge.edge_id, edge.edge_key, edge.source_node_id, edge.target_node_id, edge.kind, edge.weight, edge.confidence, edge.payload, status, edge.created_at, edge.updated_at, edge.metadata)
                updated.append(changed_edge)
            else:
                updated.append(edge)
        if changed_edge is None:
            return _result((), False, "relation missing", (normalized,), ("relation missing",))
        self._edges = tuple(updated)
        return _result((changed_edge,), True, reason)

    def _validate_edges(self, edges: tuple[GraphEdgeRef, ...]) -> None:
        if len(edges) > self._policy.max_relations:
            raise ValueError("relation limit exceeded")
        if not self._policy.allow_duplicate_relation_keys:
            keys = [edge.edge_key for edge in edges]
            if len(keys) != len(set(keys)):
                raise ValueError("relation keys must be unique")


def build_relation_repository(
    *,
    edges: tuple[GraphEdgeRef, ...] | list[GraphEdgeRef] = (),
    policy: RelationRepositoryPolicy | None = None,
) -> RelationRepository:
    return RelationRepository(edges=edges, policy=policy)


def _result(edges: tuple[GraphEdgeRef, ...], changed: bool, reason: str, rejected: tuple[str, ...] = (), reasons: tuple[str, ...] = ()) -> RelationRepositoryResult:
    return RelationRepositoryResult(edges, changed, reason, rejected, reasons, {"source": "relation_repository"})


def _edge_tuple(value: tuple[GraphEdgeRef, ...] | list[GraphEdgeRef]) -> tuple[GraphEdgeRef, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("edges must be tuple[GraphEdgeRef, ...]")
    edges = tuple(value)
    for edge in edges:
        if not isinstance(edge, GraphEdgeRef):
            raise TypeError("every edge must be GraphEdgeRef")
    return edges


def _copy_dict(value: dict[str, object], field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be dict[str, object]")
    copied = dict(value)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError(f"{field_name} keys must be strings")
    return copied


def _merge(defaults: dict[str, object], metadata: dict[str, object]) -> dict[str, object]:
    merged = _copy_dict(defaults, "metadata")
    merged.update(_copy_dict(metadata, "metadata"))
    if "source" in defaults:
        merged["source"] = defaults["source"]
    return merged


def _reject_name(value: str, field_name: str) -> None:
    if value in _FORBIDDEN:
        raise ValueError(f"{field_name} must not persist runtime semantic state")


def _reject_keys(value: dict[str, object], field_name: str) -> None:
    for key in value:
        if key in _FORBIDDEN:
            raise ValueError(f"{field_name} must not persist runtime semantic state")


def _str_tuple(value: tuple[str, ...] | list[str], field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple[str, ...]")
    return tuple(ensure_non_empty_string(item, field_name) for item in value)
