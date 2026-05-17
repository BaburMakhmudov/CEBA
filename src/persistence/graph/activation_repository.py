"""Deterministic local activation repository for CEBA graph persistence."""

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
    ensure_non_empty_string,
    ensure_optional_non_empty_string,
    ensure_positive_int,
    ensure_probability,
)


__all__ = [
    "ActivationStatusFilter",
    "ActivationRepositoryPolicy",
    "ActivationInput",
    "ActivationRepositoryQuery",
    "ActivationRepositoryResult",
    "ActivationRepository",
    "build_activation_repository",
]


_FORBIDDEN = {"M_t", "Mt", "semantic_state", "runtime_semantic_state"}


class ActivationStatusFilter(Enum):
    ANY = "ANY"
    CREATED = "CREATED"
    INDEXED = "INDEXED"
    STALE = "STALE"
    DELETED = "DELETED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class ActivationRepositoryPolicy:
    max_activations: int = 100000
    allow_duplicate_activation_keys: bool = False
    reject_zero_activation: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_activations", ensure_positive_int(self.max_activations, "max_activations"))
        if not isinstance(self.allow_duplicate_activation_keys, bool):
            raise TypeError("allow_duplicate_activation_keys must be bool")
        if not isinstance(self.reject_zero_activation, bool):
            raise TypeError("reject_zero_activation must be bool")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_activations": self.max_activations,
            "allow_duplicate_activation_keys": self.allow_duplicate_activation_keys,
            "reject_zero_activation": self.reject_zero_activation,
        }


@dataclass(frozen=True)
class ActivationInput:
    activation_key: str
    source_node: GraphNodeRef
    target_node: GraphNodeRef
    activation: float
    decay: float
    confidence: float
    source_record_id: str | None = None
    source_record_key: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        activation_key = ensure_non_empty_string(self.activation_key, "activation_key")
        _reject_name(activation_key, "activation_key")
        if not isinstance(self.source_node, GraphNodeRef):
            raise TypeError("source_node must be GraphNodeRef")
        if not isinstance(self.target_node, GraphNodeRef):
            raise TypeError("target_node must be GraphNodeRef")
        if self.source_node.node_id == self.target_node.node_id:
            raise ValueError("source_node and target_node must differ")
        metadata = _copy_dict(self.metadata, "metadata")
        _reject_keys(metadata, "metadata")
        object.__setattr__(self, "activation_key", activation_key)
        object.__setattr__(self, "activation", ensure_probability(self.activation, "activation"))
        object.__setattr__(self, "decay", ensure_probability(self.decay, "decay"))
        object.__setattr__(self, "confidence", ensure_probability(self.confidence, "confidence"))
        object.__setattr__(self, "source_record_id", ensure_optional_non_empty_string(self.source_record_id, "source_record_id"))
        object.__setattr__(self, "source_record_key", ensure_optional_non_empty_string(self.source_record_key, "source_record_key"))
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        return {
            "activation_key": self.activation_key,
            "source_node_id": str(self.source_node.node_id),
            "target_node_id": str(self.target_node.node_id),
            "activation": self.activation,
            "decay": self.decay,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ActivationRepositoryQuery:
    activation_key: str | None = None
    source_node_id: str | None = None
    target_node_id: str | None = None
    min_activation: float = 0.0
    status_filter: ActivationStatusFilter = ActivationStatusFilter.ANY
    limit: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(self, "activation_key", ensure_optional_non_empty_string(self.activation_key, "activation_key"))
        object.__setattr__(self, "source_node_id", ensure_optional_non_empty_string(self.source_node_id, "source_node_id"))
        object.__setattr__(self, "target_node_id", ensure_optional_non_empty_string(self.target_node_id, "target_node_id"))
        object.__setattr__(self, "min_activation", ensure_probability(self.min_activation, "min_activation"))
        if not isinstance(self.status_filter, ActivationStatusFilter):
            raise TypeError("status_filter must be ActivationStatusFilter")
        object.__setattr__(self, "limit", ensure_positive_int(self.limit, "limit"))

    def to_dict(self) -> dict[str, object]:
        return {
            "activation_key": self.activation_key,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "min_activation": self.min_activation,
            "status_filter": self.status_filter.value,
            "limit": self.limit,
        }


@dataclass(frozen=True)
class ActivationRepositoryResult:
    edges: tuple[GraphEdgeRef, ...]
    changed: bool
    reason: str
    rejected_activation_keys: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        edges = _edge_tuple(self.edges)
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        reason = ensure_non_empty_string(self.reason, "reason")
        rejected = _str_tuple(self.rejected_activation_keys, "rejected_activation_keys")
        reasons = _str_tuple(self.rejection_reasons, "rejection_reasons")
        if len(rejected) != len(reasons):
            raise ValueError("rejected_activation_keys and rejection_reasons must have same length")
        metadata = _copy_dict(self.metadata, "metadata")
        _reject_keys(metadata, "metadata")
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "rejected_activation_keys", rejected)
        object.__setattr__(self, "rejection_reasons", reasons)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        return {
            "edges": [edge.to_dict() for edge in self.edges],
            "changed": self.changed,
            "reason": self.reason,
            "rejected_activation_keys": list(self.rejected_activation_keys),
            "rejection_reasons": list(self.rejection_reasons),
            "metadata": dict(self.metadata),
        }


class ActivationRepository:
    def __init__(
        self,
        *,
        edges: tuple[GraphEdgeRef, ...] | list[GraphEdgeRef] = (),
        policy: ActivationRepositoryPolicy | None = None,
    ) -> None:
        self._policy = ActivationRepositoryPolicy() if policy is None else policy
        if not isinstance(self._policy, ActivationRepositoryPolicy):
            raise TypeError("policy must be ActivationRepositoryPolicy")
        self._edges = _edge_tuple(edges)
        self._validate_edges(self._edges)

    def edges(self) -> tuple[GraphEdgeRef, ...]:
        return self._edges

    def edge_count(self) -> int:
        return len(self._edges)

    def create_activation(self, *, activation_input: ActivationInput) -> ActivationRepositoryResult:
        if not isinstance(activation_input, ActivationInput):
            raise TypeError("activation_input must be ActivationInput")
        if self._policy.reject_zero_activation and activation_input.activation == 0.0:
            return _result((), False, "activation is zero", (activation_input.activation_key,), ("activation is zero",))
        if len(self._edges) >= self._policy.max_activations:
            return _result((), False, "activation limit exceeded", (activation_input.activation_key,), ("activation limit exceeded",))
        if not self._policy.allow_duplicate_activation_keys and any(edge.edge_key == activation_input.activation_key for edge in self._edges):
            return _result((), False, "activation key already exists", (activation_input.activation_key,), ("activation key already exists",))
        payload = create_graph_payload(
            source_record_id=activation_input.source_record_id,
            source_record_key=activation_input.source_record_key,
            data={
                "activation": activation_input.activation,
                "decay": activation_input.decay,
                "confidence": activation_input.confidence,
            },
            metadata={"source": "activation_repository", "operation": "create_activation_payload"},
        )
        edge = create_graph_edge_ref(
            edge_key=activation_input.activation_key,
            source_node_id=activation_input.source_node.node_id,
            target_node_id=activation_input.target_node.node_id,
            kind=GraphEdgeKind.ACTIVATES,
            weight=activation_input.activation,
            confidence=activation_input.confidence,
            payload=payload,
            status=GraphElementStatus.CREATED,
            metadata=_merge({"source": "activation_repository", "operation": "create_activation"}, activation_input.metadata),
        )
        self._edges = self._edges + (edge,)
        return _result((edge,), True, "activation created")

    def get_activation_by_key(self, *, activation_key: str) -> ActivationRepositoryResult:
        key = ensure_non_empty_string(activation_key, "activation_key")
        for edge in self._edges:
            if edge.edge_key == key:
                return _result((edge,), False, "activation found")
        return _result((), False, "activation missing")

    def query_activations(self, *, query: ActivationRepositoryQuery) -> ActivationRepositoryResult:
        if not isinstance(query, ActivationRepositoryQuery):
            raise TypeError("query must be ActivationRepositoryQuery")
        edges = [edge for edge in self._edges if edge.kind is GraphEdgeKind.ACTIVATES]
        if query.activation_key is not None:
            edges = [edge for edge in edges if edge.edge_key == query.activation_key]
        if query.source_node_id is not None:
            edges = [edge for edge in edges if str(edge.source_node_id) == query.source_node_id]
        if query.target_node_id is not None:
            edges = [edge for edge in edges if str(edge.target_node_id) == query.target_node_id]
        edges = [edge for edge in edges if edge.weight >= query.min_activation]
        if query.status_filter is not ActivationStatusFilter.ANY:
            edges = [edge for edge in edges if edge.status.value == query.status_filter.value]
        edges = sorted(edges, key=lambda edge: (-edge.weight, -edge.updated_at.timestamp(), edge.edge_key))
        return _result(tuple(edges[: query.limit]), False, "activations queried")

    def mark_indexed(self, *, edge_id: str) -> ActivationRepositoryResult:
        return self._mark(edge_id=edge_id, status=GraphElementStatus.INDEXED, reason="activation indexed")

    def mark_stale(self, *, edge_id: str) -> ActivationRepositoryResult:
        return self._mark(edge_id=edge_id, status=GraphElementStatus.STALE, reason="activation stale")

    def mark_deleted(self, *, edge_id: str) -> ActivationRepositoryResult:
        return self._mark(edge_id=edge_id, status=GraphElementStatus.DELETED, reason="activation deleted")

    def replace_edges(self, *, edges: tuple[GraphEdgeRef, ...] | list[GraphEdgeRef]) -> ActivationRepositoryResult:
        new_edges = _edge_tuple(edges)
        self._validate_edges(new_edges)
        self._edges = new_edges
        return _result(new_edges, True, "activation edges replaced")

    def _mark(self, *, edge_id: str, status: GraphElementStatus, reason: str) -> ActivationRepositoryResult:
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
            return _result((), False, "activation missing", (normalized,), ("activation missing",))
        self._edges = tuple(updated)
        return _result((changed_edge,), True, reason)

    def _validate_edges(self, edges: tuple[GraphEdgeRef, ...]) -> None:
        if len(edges) > self._policy.max_activations:
            raise ValueError("activation limit exceeded")
        for edge in edges:
            if edge.kind is not GraphEdgeKind.ACTIVATES:
                raise TypeError("every edge must be GraphEdgeKind.ACTIVATES")
        if not self._policy.allow_duplicate_activation_keys:
            keys = [edge.edge_key for edge in edges]
            if len(keys) != len(set(keys)):
                raise ValueError("activation keys must be unique")


def build_activation_repository(
    *,
    edges: tuple[GraphEdgeRef, ...] | list[GraphEdgeRef] = (),
    policy: ActivationRepositoryPolicy | None = None,
) -> ActivationRepository:
    return ActivationRepository(edges=edges, policy=policy)


def _result(edges: tuple[GraphEdgeRef, ...], changed: bool, reason: str, rejected: tuple[str, ...] = (), reasons: tuple[str, ...] = ()) -> ActivationRepositoryResult:
    return ActivationRepositoryResult(edges, changed, reason, rejected, reasons, {"source": "activation_repository"})


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
