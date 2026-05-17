"""Strict schema objects for CEBA graph persistence surfaces."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from uuid import UUID, uuid4


__all__ = [
    "GraphBackend",
    "GraphIndexStatus",
    "GraphNodeKind",
    "GraphEdgeKind",
    "GraphElementStatus",
    "GraphTraversalDirection",
    "GraphIndexConfig",
    "GraphPayload",
    "GraphNodeRef",
    "GraphEdgeRef",
    "GraphTraversalQuery",
    "GraphPathMatch",
    "GraphIndexSnapshot",
    "utc_now",
    "new_graph_node_id",
    "new_graph_edge_id",
    "new_graph_snapshot_id",
    "ensure_non_empty_string",
    "ensure_optional_non_empty_string",
    "ensure_positive_int",
    "ensure_non_negative_int",
    "ensure_probability",
    "ensure_finite_number",
    "ensure_timezone_aware",
    "create_graph_index_config",
    "create_graph_payload",
    "create_graph_node_ref",
    "create_graph_edge_ref",
    "create_graph_traversal_query",
    "create_graph_path_match",
    "create_graph_index_snapshot",
]


_FORBIDDEN = {"M_t", "Mt", "semantic_state", "runtime_semantic_state"}


class GraphBackend(Enum):
    NEO4J = "NEO4J"
    IN_MEMORY = "IN_MEMORY"
    UNKNOWN = "UNKNOWN"


class GraphIndexStatus(Enum):
    CREATED = "CREATED"
    READY = "READY"
    DEGRADED = "DEGRADED"
    REBUILDING = "REBUILDING"
    DISABLED = "DISABLED"
    FAILED = "FAILED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class GraphNodeKind(Enum):
    OBSERVATION = "OBSERVATION"
    TRACE = "TRACE"
    EVENT = "EVENT"
    DECISION = "DECISION"
    CONSTRAINT = "CONSTRAINT"
    TELOS = "TELOS"
    AUDIT = "AUDIT"
    CONCEPT = "CONCEPT"
    ENTITY = "ENTITY"
    RELATION_ANCHOR = "RELATION_ANCHOR"
    UNKNOWN = "UNKNOWN"


class GraphEdgeKind(Enum):
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
    REFERENCES = "REFERENCES"
    UNKNOWN = "UNKNOWN"


class GraphElementStatus(Enum):
    CREATED = "CREATED"
    INDEXED = "INDEXED"
    STALE = "STALE"
    DELETED = "DELETED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class GraphTraversalDirection(Enum):
    OUTGOING = "OUTGOING"
    INCOMING = "INCOMING"
    BOTH = "BOTH"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class GraphIndexConfig:
    backend: GraphBackend
    graph_name: str
    status: GraphIndexStatus
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.backend, GraphBackend):
            raise TypeError("backend must be GraphBackend")
        graph_name = ensure_non_empty_string(self.graph_name, "graph_name")
        _reject_name(graph_name, "graph_name")
        if not isinstance(self.status, GraphIndexStatus):
            raise TypeError("status must be GraphIndexStatus")
        metadata = _copy_dict(self.metadata, "metadata")
        _reject_keys(metadata, "metadata")
        object.__setattr__(self, "graph_name", graph_name)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend.value,
            "graph_name": self.graph_name,
            "status": self.status.value,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GraphPayload:
    source_record_id: str | None
    source_record_key: str | None
    data: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source_record_id = ensure_optional_non_empty_string(
            self.source_record_id, "source_record_id"
        )
        source_record_key = ensure_optional_non_empty_string(
            self.source_record_key, "source_record_key"
        )
        data = _copy_dict(self.data, "data")
        metadata = _copy_dict(self.metadata, "metadata")
        _reject_keys(data, "data")
        _reject_keys(metadata, "metadata")
        object.__setattr__(self, "source_record_id", source_record_id)
        object.__setattr__(self, "source_record_key", source_record_key)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_record_id": self.source_record_id,
            "source_record_key": self.source_record_key,
            "data": dict(self.data),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GraphNodeRef:
    node_id: UUID
    node_key: str
    kind: GraphNodeKind
    label: str | None
    embedding_ref: str | None
    payload: GraphPayload
    status: GraphElementStatus
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, UUID):
            raise TypeError("node_id must be UUID")
        node_key = ensure_non_empty_string(self.node_key, "node_key")
        _reject_name(node_key, "node_key")
        if not isinstance(self.kind, GraphNodeKind):
            raise TypeError("kind must be GraphNodeKind")
        label = ensure_optional_non_empty_string(self.label, "label")
        if label is not None:
            _reject_name(label, "label")
        embedding_ref = ensure_optional_non_empty_string(
            self.embedding_ref, "embedding_ref"
        )
        if not isinstance(self.payload, GraphPayload):
            raise TypeError("payload must be GraphPayload")
        if not isinstance(self.status, GraphElementStatus):
            raise TypeError("status must be GraphElementStatus")
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        updated_at = ensure_timezone_aware(self.updated_at, "updated_at")
        if updated_at < created_at:
            raise ValueError("updated_at must be >= created_at")
        metadata = _copy_dict(self.metadata, "metadata")
        _reject_keys(metadata, "metadata")
        object.__setattr__(self, "node_key", node_key)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "embedding_ref", embedding_ref)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": str(self.node_id),
            "node_key": self.node_key,
            "kind": self.kind.value,
            "label": self.label,
            "embedding_ref": self.embedding_ref,
            "payload": self.payload.to_dict(),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GraphEdgeRef:
    edge_id: UUID
    edge_key: str
    source_node_id: UUID
    target_node_id: UUID
    kind: GraphEdgeKind
    weight: float
    confidence: float
    payload: GraphPayload
    status: GraphElementStatus
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.edge_id, UUID):
            raise TypeError("edge_id must be UUID")
        edge_key = ensure_non_empty_string(self.edge_key, "edge_key")
        _reject_name(edge_key, "edge_key")
        if not isinstance(self.source_node_id, UUID):
            raise TypeError("source_node_id must be UUID")
        if not isinstance(self.target_node_id, UUID):
            raise TypeError("target_node_id must be UUID")
        if self.source_node_id == self.target_node_id:
            raise ValueError("source_node_id and target_node_id must not be equal")
        if not isinstance(self.kind, GraphEdgeKind):
            raise TypeError("kind must be GraphEdgeKind")
        weight = ensure_finite_number(self.weight, "weight")
        if weight < 0.0:
            raise ValueError("weight must be non-negative")
        confidence = ensure_probability(self.confidence, "confidence")
        if not isinstance(self.payload, GraphPayload):
            raise TypeError("payload must be GraphPayload")
        if not isinstance(self.status, GraphElementStatus):
            raise TypeError("status must be GraphElementStatus")
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        updated_at = ensure_timezone_aware(self.updated_at, "updated_at")
        if updated_at < created_at:
            raise ValueError("updated_at must be >= created_at")
        metadata = _copy_dict(self.metadata, "metadata")
        _reject_keys(metadata, "metadata")
        object.__setattr__(self, "edge_key", edge_key)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        return {
            "edge_id": str(self.edge_id),
            "edge_key": self.edge_key,
            "source_node_id": str(self.source_node_id),
            "target_node_id": str(self.target_node_id),
            "kind": self.kind.value,
            "weight": self.weight,
            "confidence": self.confidence,
            "payload": self.payload.to_dict(),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GraphTraversalQuery:
    start_node_key: str | None = None
    node_kind: GraphNodeKind | None = None
    edge_kind: GraphEdgeKind | None = None
    direction: GraphTraversalDirection = GraphTraversalDirection.BOTH
    max_depth: int = 1
    limit: int = 100
    min_confidence: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        start_node_key = ensure_optional_non_empty_string(
            self.start_node_key, "start_node_key"
        )
        if start_node_key is not None:
            _reject_name(start_node_key, "start_node_key")
        if self.node_kind is not None and not isinstance(self.node_kind, GraphNodeKind):
            raise TypeError("node_kind must be None or GraphNodeKind")
        if self.edge_kind is not None and not isinstance(self.edge_kind, GraphEdgeKind):
            raise TypeError("edge_kind must be None or GraphEdgeKind")
        if not isinstance(self.direction, GraphTraversalDirection):
            raise TypeError("direction must be GraphTraversalDirection")
        max_depth = ensure_positive_int(self.max_depth, "max_depth")
        limit = ensure_positive_int(self.limit, "limit")
        min_confidence = ensure_probability(self.min_confidence, "min_confidence")
        metadata = _copy_dict(self.metadata, "metadata")
        _reject_keys(metadata, "metadata")
        object.__setattr__(self, "start_node_key", start_node_key)
        object.__setattr__(self, "max_depth", max_depth)
        object.__setattr__(self, "limit", limit)
        object.__setattr__(self, "min_confidence", min_confidence)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        return {
            "start_node_key": self.start_node_key,
            "node_kind": None if self.node_kind is None else self.node_kind.value,
            "edge_kind": None if self.edge_kind is None else self.edge_kind.value,
            "direction": self.direction.value,
            "max_depth": self.max_depth,
            "limit": self.limit,
            "min_confidence": self.min_confidence,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GraphPathMatch:
    nodes: tuple[GraphNodeRef, ...]
    edges: tuple[GraphEdgeRef, ...]
    score: float
    rank: int
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        nodes = _node_tuple(self.nodes)
        edges = _edge_tuple(self.edges)
        _unique([node.node_id for node in nodes], "node_id")
        _unique([edge.edge_id for edge in edges], "edge_id")
        node_ids = {node.node_id for node in nodes}
        for edge in edges:
            if edge.source_node_id not in node_ids or edge.target_node_id not in node_ids:
                raise ValueError("every edge endpoint must exist in nodes")
        score = ensure_probability(self.score, "score")
        rank = ensure_positive_int(self.rank, "rank")
        metadata = _copy_dict(self.metadata, "metadata")
        _reject_keys(metadata, "metadata")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "score": self.score,
            "rank": self.rank,
            "metadata": dict(self.metadata),
        }

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return len(self.edges)


@dataclass(frozen=True)
class GraphIndexSnapshot:
    snapshot_id: UUID
    config: GraphIndexConfig
    nodes: tuple[GraphNodeRef, ...]
    edges: tuple[GraphEdgeRef, ...]
    status: GraphIndexStatus
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot_id, UUID):
            raise TypeError("snapshot_id must be UUID")
        if not isinstance(self.config, GraphIndexConfig):
            raise TypeError("config must be GraphIndexConfig")
        nodes = _node_tuple(self.nodes)
        edges = _edge_tuple(self.edges)
        _unique([node.node_id for node in nodes], "node_id")
        _unique([node.node_key for node in nodes], "node_key")
        _unique([edge.edge_id for edge in edges], "edge_id")
        _unique([edge.edge_key for edge in edges], "edge_key")
        node_ids = {node.node_id for node in nodes}
        for edge in edges:
            if edge.source_node_id not in node_ids or edge.target_node_id not in node_ids:
                raise ValueError("every edge endpoint must exist in nodes")
        if not isinstance(self.status, GraphIndexStatus):
            raise TypeError("status must be GraphIndexStatus")
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        metadata = _copy_dict(self.metadata, "metadata")
        _reject_keys(metadata, "metadata")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        return {
            "snapshot_id": str(self.snapshot_id),
            "config": self.config.to_dict(),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return len(self.edges)


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_graph_node_id() -> UUID:
    return uuid4()


def new_graph_edge_id() -> UUID:
    return uuid4()


def new_graph_snapshot_id() -> UUID:
    return uuid4()


def ensure_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def ensure_optional_non_empty_string(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return ensure_non_empty_string(value, field_name)


def ensure_positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return value


def ensure_non_negative_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return value


def ensure_probability(value: float, field_name: str) -> float:
    number = ensure_finite_number(value, field_name)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return number


def ensure_finite_number(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be int or float")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def ensure_timezone_aware(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def create_graph_index_config(
    *,
    graph_name: str,
    backend: GraphBackend = GraphBackend.UNKNOWN,
    status: GraphIndexStatus = GraphIndexStatus.CREATED,
    metadata: dict[str, object] | None = None,
) -> GraphIndexConfig:
    return GraphIndexConfig(
        backend=backend,
        graph_name=graph_name,
        status=status,
        metadata=_merge({"source": "graph_index_config", "graph_name": graph_name}, metadata),
    )


def create_graph_payload(
    *,
    source_record_id: str | None = None,
    source_record_key: str | None = None,
    data: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
) -> GraphPayload:
    return GraphPayload(
        source_record_id=source_record_id,
        source_record_key=source_record_key,
        data={} if data is None else data,
        metadata=_merge({"source": "graph_payload"}, metadata),
    )


def create_graph_node_ref(
    *,
    node_key: str,
    payload: GraphPayload,
    kind: GraphNodeKind = GraphNodeKind.UNKNOWN,
    label: str | None = None,
    embedding_ref: str | None = None,
    status: GraphElementStatus = GraphElementStatus.CREATED,
    metadata: dict[str, object] | None = None,
) -> GraphNodeRef:
    now = utc_now()
    return GraphNodeRef(
        node_id=new_graph_node_id(),
        node_key=node_key,
        kind=kind,
        label=label,
        embedding_ref=embedding_ref,
        payload=payload,
        status=status,
        created_at=now,
        updated_at=now,
        metadata=_merge({"source": "graph_node_ref", "node_key": node_key}, metadata),
    )


def create_graph_edge_ref(
    *,
    edge_key: str,
    source_node_id: UUID,
    target_node_id: UUID,
    payload: GraphPayload,
    kind: GraphEdgeKind = GraphEdgeKind.UNKNOWN,
    weight: float = 1.0,
    confidence: float = 1.0,
    status: GraphElementStatus = GraphElementStatus.CREATED,
    metadata: dict[str, object] | None = None,
) -> GraphEdgeRef:
    now = utc_now()
    return GraphEdgeRef(
        edge_id=new_graph_edge_id(),
        edge_key=edge_key,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        kind=kind,
        weight=weight,
        confidence=confidence,
        payload=payload,
        status=status,
        created_at=now,
        updated_at=now,
        metadata=_merge({"source": "graph_edge_ref", "edge_key": edge_key}, metadata),
    )


def create_graph_traversal_query(
    *,
    start_node_key: str | None = None,
    node_kind: GraphNodeKind | None = None,
    edge_kind: GraphEdgeKind | None = None,
    direction: GraphTraversalDirection = GraphTraversalDirection.BOTH,
    max_depth: int = 1,
    limit: int = 100,
    min_confidence: float = 0.0,
    metadata: dict[str, object] | None = None,
) -> GraphTraversalQuery:
    return GraphTraversalQuery(
        start_node_key=start_node_key,
        node_kind=node_kind,
        edge_kind=edge_kind,
        direction=direction,
        max_depth=max_depth,
        limit=limit,
        min_confidence=min_confidence,
        metadata=_merge({"source": "graph_traversal_query"}, metadata),
    )


def create_graph_path_match(
    *,
    nodes: tuple[GraphNodeRef, ...] | list[GraphNodeRef],
    edges: tuple[GraphEdgeRef, ...] | list[GraphEdgeRef] = (),
    score: float = 1.0,
    rank: int = 1,
    metadata: dict[str, object] | None = None,
) -> GraphPathMatch:
    return GraphPathMatch(
        nodes=tuple(nodes),
        edges=tuple(edges),
        score=score,
        rank=rank,
        metadata=_merge({"source": "graph_path_match"}, metadata),
    )


def create_graph_index_snapshot(
    *,
    config: GraphIndexConfig,
    nodes: tuple[GraphNodeRef, ...] | list[GraphNodeRef] = (),
    edges: tuple[GraphEdgeRef, ...] | list[GraphEdgeRef] = (),
    status: GraphIndexStatus = GraphIndexStatus.CREATED,
    metadata: dict[str, object] | None = None,
) -> GraphIndexSnapshot:
    return GraphIndexSnapshot(
        snapshot_id=new_graph_snapshot_id(),
        config=config,
        nodes=tuple(nodes),
        edges=tuple(edges),
        status=status,
        created_at=utc_now(),
        metadata=_merge({"source": "graph_index_snapshot"}, metadata),
    )


def _copy_dict(value: dict[str, object], field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be dict[str, object]")
    copied = dict(value)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError(f"{field_name} keys must be strings")
    return copied


def _merge(defaults: dict[str, object], metadata: dict[str, object] | None) -> dict[str, object]:
    merged = _copy_dict(defaults, "metadata")
    if metadata is not None:
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


def _unique(values: list[object], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} values must be unique")
