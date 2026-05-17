"""Deterministic local concept repository for CEBA graph persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ceba.persistence.graph.schema import (
    GraphElementStatus,
    GraphNodeKind,
    GraphNodeRef,
    create_graph_node_ref,
    create_graph_payload,
    ensure_non_empty_string,
    ensure_optional_non_empty_string,
    ensure_positive_int,
)


__all__ = [
    "ConceptStatusFilter",
    "ConceptRepositoryPolicy",
    "ConceptInput",
    "ConceptRepositoryQuery",
    "ConceptRepositoryResult",
    "ConceptRepository",
    "build_concept_repository",
]


_FORBIDDEN = {"M_t", "Mt", "semantic_state", "runtime_semantic_state"}


class ConceptStatusFilter(Enum):
    ANY = "ANY"
    CREATED = "CREATED"
    INDEXED = "INDEXED"
    STALE = "STALE"
    DELETED = "DELETED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class ConceptRepositoryPolicy:
    max_concepts: int = 100000
    allow_duplicate_concept_keys: bool = False
    reject_empty_payload: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_concepts", ensure_positive_int(self.max_concepts, "max_concepts"))
        if not isinstance(self.allow_duplicate_concept_keys, bool):
            raise TypeError("allow_duplicate_concept_keys must be bool")
        if not isinstance(self.reject_empty_payload, bool):
            raise TypeError("reject_empty_payload must be bool")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_concepts": self.max_concepts,
            "allow_duplicate_concept_keys": self.allow_duplicate_concept_keys,
            "reject_empty_payload": self.reject_empty_payload,
        }


@dataclass(frozen=True)
class ConceptInput:
    concept_key: str
    label: str | None = None
    embedding_ref: str | None = None
    source_record_id: str | None = None
    source_record_key: str | None = None
    payload_data: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        concept_key = ensure_non_empty_string(self.concept_key, "concept_key")
        _reject_name(concept_key, "concept_key")
        payload_data = _copy_dict(self.payload_data, "payload_data")
        metadata = _copy_dict(self.metadata, "metadata")
        _reject_keys(payload_data, "payload_data")
        _reject_keys(metadata, "metadata")
        object.__setattr__(self, "concept_key", concept_key)
        object.__setattr__(self, "label", ensure_optional_non_empty_string(self.label, "label"))
        object.__setattr__(self, "embedding_ref", ensure_optional_non_empty_string(self.embedding_ref, "embedding_ref"))
        object.__setattr__(self, "source_record_id", ensure_optional_non_empty_string(self.source_record_id, "source_record_id"))
        object.__setattr__(self, "source_record_key", ensure_optional_non_empty_string(self.source_record_key, "source_record_key"))
        object.__setattr__(self, "payload_data", payload_data)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        return {
            "concept_key": self.concept_key,
            "label": self.label,
            "embedding_ref": self.embedding_ref,
            "source_record_id": self.source_record_id,
            "source_record_key": self.source_record_key,
            "payload_data": dict(self.payload_data),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ConceptRepositoryQuery:
    concept_key: str | None = None
    label: str | None = None
    status_filter: ConceptStatusFilter = ConceptStatusFilter.ANY
    limit: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(self, "concept_key", ensure_optional_non_empty_string(self.concept_key, "concept_key"))
        object.__setattr__(self, "label", ensure_optional_non_empty_string(self.label, "label"))
        if not isinstance(self.status_filter, ConceptStatusFilter):
            raise TypeError("status_filter must be ConceptStatusFilter")
        object.__setattr__(self, "limit", ensure_positive_int(self.limit, "limit"))

    def to_dict(self) -> dict[str, object]:
        return {
            "concept_key": self.concept_key,
            "label": self.label,
            "status_filter": self.status_filter.value,
            "limit": self.limit,
        }


@dataclass(frozen=True)
class ConceptRepositoryResult:
    nodes: tuple[GraphNodeRef, ...]
    changed: bool
    reason: str
    rejected_concept_keys: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        nodes = _node_tuple(self.nodes)
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        reason = ensure_non_empty_string(self.reason, "reason")
        rejected = _str_tuple(self.rejected_concept_keys, "rejected_concept_keys")
        reasons = _str_tuple(self.rejection_reasons, "rejection_reasons")
        if len(rejected) != len(reasons):
            raise ValueError("rejected_concept_keys and rejection_reasons must have same length")
        metadata = _copy_dict(self.metadata, "metadata")
        _reject_keys(metadata, "metadata")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "rejected_concept_keys", rejected)
        object.__setattr__(self, "rejection_reasons", reasons)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "changed": self.changed,
            "reason": self.reason,
            "rejected_concept_keys": list(self.rejected_concept_keys),
            "rejection_reasons": list(self.rejection_reasons),
            "metadata": dict(self.metadata),
        }


class ConceptRepository:
    def __init__(
        self,
        *,
        nodes: tuple[GraphNodeRef, ...] | list[GraphNodeRef] = (),
        policy: ConceptRepositoryPolicy | None = None,
    ) -> None:
        self._policy = ConceptRepositoryPolicy() if policy is None else policy
        if not isinstance(self._policy, ConceptRepositoryPolicy):
            raise TypeError("policy must be ConceptRepositoryPolicy")
        self._nodes = _node_tuple(nodes)
        self._validate_nodes(self._nodes)

    def nodes(self) -> tuple[GraphNodeRef, ...]:
        return self._nodes

    def node_count(self) -> int:
        return len(self._nodes)

    def create_concept(self, *, concept_input: ConceptInput) -> ConceptRepositoryResult:
        if not isinstance(concept_input, ConceptInput):
            raise TypeError("concept_input must be ConceptInput")
        if self._policy.reject_empty_payload and not concept_input.payload_data:
            return _result((), False, "concept payload is empty", (concept_input.concept_key,), ("concept payload is empty",))
        if len(self._nodes) >= self._policy.max_concepts:
            return _result((), False, "concept limit exceeded", (concept_input.concept_key,), ("concept limit exceeded",))
        if not self._policy.allow_duplicate_concept_keys and any(node.node_key == concept_input.concept_key for node in self._nodes):
            return _result((), False, "concept key already exists", (concept_input.concept_key,), ("concept key already exists",))
        payload = create_graph_payload(
            source_record_id=concept_input.source_record_id,
            source_record_key=concept_input.source_record_key,
            data=concept_input.payload_data,
            metadata={"source": "concept_repository", "operation": "create_concept_payload"},
        )
        node = create_graph_node_ref(
            node_key=concept_input.concept_key,
            kind=GraphNodeKind.CONCEPT,
            label=concept_input.label,
            embedding_ref=concept_input.embedding_ref,
            payload=payload,
            status=GraphElementStatus.CREATED,
            metadata=_merge({"source": "concept_repository", "operation": "create_concept"}, concept_input.metadata),
        )
        self._nodes = self._nodes + (node,)
        return _result((node,), True, "concept created")

    def get_concept_by_key(self, *, concept_key: str) -> ConceptRepositoryResult:
        key = ensure_non_empty_string(concept_key, "concept_key")
        for node in self._nodes:
            if node.node_key == key:
                return _result((node,), False, "concept found")
        return _result((), False, "concept missing")

    def query_concepts(self, *, query: ConceptRepositoryQuery) -> ConceptRepositoryResult:
        if not isinstance(query, ConceptRepositoryQuery):
            raise TypeError("query must be ConceptRepositoryQuery")
        nodes = [node for node in self._nodes if node.kind is GraphNodeKind.CONCEPT]
        if query.concept_key is not None:
            nodes = [node for node in nodes if node.node_key == query.concept_key]
        if query.label is not None:
            nodes = [node for node in nodes if node.label == query.label]
        if query.status_filter is not ConceptStatusFilter.ANY:
            nodes = [node for node in nodes if node.status.value == query.status_filter.value]
        nodes = sorted(nodes, key=lambda node: (-node.updated_at.timestamp(), node.node_key))
        return _result(tuple(nodes[: query.limit]), False, "concepts queried")

    def mark_indexed(self, *, node_id: str) -> ConceptRepositoryResult:
        return self._mark(node_id=node_id, status=GraphElementStatus.INDEXED, reason="concept indexed")

    def mark_stale(self, *, node_id: str) -> ConceptRepositoryResult:
        return self._mark(node_id=node_id, status=GraphElementStatus.STALE, reason="concept stale")

    def mark_deleted(self, *, node_id: str) -> ConceptRepositoryResult:
        return self._mark(node_id=node_id, status=GraphElementStatus.DELETED, reason="concept deleted")

    def replace_nodes(self, *, nodes: tuple[GraphNodeRef, ...] | list[GraphNodeRef]) -> ConceptRepositoryResult:
        new_nodes = _node_tuple(nodes)
        self._validate_nodes(new_nodes)
        self._nodes = new_nodes
        return _result(new_nodes, True, "concept nodes replaced")

    def _mark(self, *, node_id: str, status: GraphElementStatus, reason: str) -> ConceptRepositoryResult:
        normalized = ensure_non_empty_string(node_id, "node_id")
        updated: list[GraphNodeRef] = []
        changed_node: GraphNodeRef | None = None
        for node in self._nodes:
            if str(node.node_id) == normalized:
                changed_node = GraphNodeRef(node.node_id, node.node_key, node.kind, node.label, node.embedding_ref, node.payload, status, node.created_at, node.updated_at, node.metadata)
                updated.append(changed_node)
            else:
                updated.append(node)
        if changed_node is None:
            return _result((), False, "concept missing", (normalized,), ("concept missing",))
        self._nodes = tuple(updated)
        return _result((changed_node,), True, reason)

    def _validate_nodes(self, nodes: tuple[GraphNodeRef, ...]) -> None:
        if len(nodes) > self._policy.max_concepts:
            raise ValueError("concept limit exceeded")
        for node in nodes:
            if node.kind is not GraphNodeKind.CONCEPT:
                raise TypeError("every node must be GraphNodeKind.CONCEPT")
        if not self._policy.allow_duplicate_concept_keys:
            keys = [node.node_key for node in nodes]
            if len(keys) != len(set(keys)):
                raise ValueError("concept keys must be unique")


def build_concept_repository(
    *,
    nodes: tuple[GraphNodeRef, ...] | list[GraphNodeRef] = (),
    policy: ConceptRepositoryPolicy | None = None,
) -> ConceptRepository:
    return ConceptRepository(nodes=nodes, policy=policy)


def _result(nodes: tuple[GraphNodeRef, ...], changed: bool, reason: str, rejected: tuple[str, ...] = (), reasons: tuple[str, ...] = ()) -> ConceptRepositoryResult:
    return ConceptRepositoryResult(nodes, changed, reason, rejected, reasons, {"source": "concept_repository"})


def _node_tuple(value: tuple[GraphNodeRef, ...] | list[GraphNodeRef]) -> tuple[GraphNodeRef, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("nodes must be tuple[GraphNodeRef, ...]")
    nodes = tuple(value)
    for node in nodes:
        if not isinstance(node, GraphNodeRef):
            raise TypeError("every node must be GraphNodeRef")
    return nodes


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
