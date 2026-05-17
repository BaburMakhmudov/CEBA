"""Deterministic relation extraction for CEBA Core Relation Processing."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ceba.core.relation.schema import (
    RelationEdge,
    RelationEdgeKind,
    RelationEvidence,
    RelationGraphSnapshot,
    RelationNodeRef,
    RelationStatus,
    create_relation_edge,
    create_relation_graph_snapshot,
    ensure_non_empty_string,
)


@dataclass(frozen=True)
class RelationExtractionPolicy:
    """Configuration controlling deterministic extraction behavior."""

    min_confidence: float = 0.0
    min_weight: float = 0.0
    allow_unknown_edge_kind: bool = False
    allow_self_edges: bool = False
    deduplicate: bool = True
    sort_output: bool = True
    max_edges: int | None = None

    def __post_init__(self) -> None:
        min_confidence = _ensure_probability(
            self.min_confidence,
            "min_confidence",
        )
        min_weight = _ensure_non_negative_number(self.min_weight, "min_weight")
        if not isinstance(self.allow_unknown_edge_kind, bool):
            raise TypeError("allow_unknown_edge_kind must be bool")
        if not isinstance(self.allow_self_edges, bool):
            raise TypeError("allow_self_edges must be bool")
        if not isinstance(self.deduplicate, bool):
            raise TypeError("deduplicate must be bool")
        if not isinstance(self.sort_output, bool):
            raise TypeError("sort_output must be bool")
        max_edges = _ensure_optional_positive_int(self.max_edges, "max_edges")

        object.__setattr__(self, "min_confidence", min_confidence)
        object.__setattr__(self, "min_weight", min_weight)
        object.__setattr__(self, "max_edges", max_edges)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "min_confidence": self.min_confidence,
            "min_weight": self.min_weight,
            "allow_unknown_edge_kind": self.allow_unknown_edge_kind,
            "allow_self_edges": self.allow_self_edges,
            "deduplicate": self.deduplicate,
            "sort_output": self.sort_output,
            "max_edges": self.max_edges,
        }


@dataclass(frozen=True)
class RelationExtractionResult:
    """Immutable result object returned by the extractor."""

    edges: tuple[RelationEdge, ...]
    rejected_evidence_ids: tuple[str, ...]
    rejection_reasons: tuple[str, ...]
    policy: RelationExtractionPolicy

    def __post_init__(self) -> None:
        edges = tuple(self.edges)
        for edge in edges:
            if not isinstance(edge, RelationEdge):
                raise TypeError("edges must contain RelationEdge")
        rejected_evidence_ids = _ensure_non_empty_string_tuple(
            self.rejected_evidence_ids,
            "rejected_evidence_ids",
        )
        rejection_reasons = _ensure_non_empty_string_tuple(
            self.rejection_reasons,
            "rejection_reasons",
        )
        if not isinstance(self.policy, RelationExtractionPolicy):
            raise TypeError("policy must be RelationExtractionPolicy")

        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "rejected_evidence_ids", rejected_evidence_ids)
        object.__setattr__(self, "rejection_reasons", rejection_reasons)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "edges": tuple(edge.to_dict() for edge in self.edges),
            "rejected_evidence_ids": self.rejected_evidence_ids,
            "rejection_reasons": self.rejection_reasons,
            "policy": self.policy.to_dict(),
        }


class RelationExtractor:
    """Deterministic service for extracting edges from explicit evidence."""

    def __init__(self, policy: RelationExtractionPolicy | None = None) -> None:
        if policy is None:
            policy = RelationExtractionPolicy()
        if not isinstance(policy, RelationExtractionPolicy):
            raise TypeError("policy must be RelationExtractionPolicy or None")
        self.policy = policy

    def extract(
        self,
        *,
        nodes: tuple[RelationNodeRef, ...] | list[RelationNodeRef],
        evidence: tuple[RelationEvidence, ...] | list[RelationEvidence],
    ) -> RelationExtractionResult:
        """Extract relation edges from explicit relation evidence."""

        node_values = self._validate_nodes(nodes)
        evidence_values = self._validate_evidence(evidence)
        node_lookup = self._build_node_lookup(node_values)
        edges: list[RelationEdge] = []
        rejected_ids: list[str] = []
        rejection_reasons: list[str] = []

        for item in evidence_values:
            edge, reason = self._edge_from_evidence(
                evidence=item,
                node_lookup=node_lookup,
            )
            if edge is None:
                rejected_ids.append(str(item.evidence_id))
                rejection_reasons.append(reason)
            else:
                edges.append(edge)

        edge_values = tuple(edges)
        if self.policy.deduplicate:
            edge_values = self._deduplicate_edges(edge_values)
        if self.policy.sort_output:
            edge_values = self._sort_edges(edge_values)
        if self.policy.max_edges is not None:
            edge_values = edge_values[: self.policy.max_edges]

        return RelationExtractionResult(
            edges=edge_values,
            rejected_evidence_ids=tuple(rejected_ids),
            rejection_reasons=tuple(rejection_reasons),
            policy=self.policy,
        )

    def extract_graph_snapshot(
        self,
        *,
        nodes: tuple[RelationNodeRef, ...] | list[RelationNodeRef],
        evidence: tuple[RelationEvidence, ...] | list[RelationEvidence],
        status: RelationStatus = RelationStatus.ACTIVE,
        metadata: dict[str, object] | None = None,
    ) -> RelationGraphSnapshot:
        """Extract edges and return a lightweight relation graph snapshot."""

        if not isinstance(status, RelationStatus):
            raise TypeError("status must be RelationStatus")
        result = self.extract(nodes=nodes, evidence=evidence)
        node_values = self._validate_nodes(nodes)
        default_metadata: dict[str, object] = {
            "source": "relation_extractor",
            "accepted_edge_count": len(result.edges),
            "rejected_evidence_count": len(result.rejected_evidence_ids),
        }
        merged_metadata = _merge_metadata(default_metadata, metadata)
        merged_metadata["source"] = "relation_extractor"
        return create_relation_graph_snapshot(
            nodes=node_values,
            edges=result.edges,
            activations=(),
            status=status,
            metadata=merged_metadata,
        )

    def _validate_nodes(
        self,
        nodes: tuple[RelationNodeRef, ...] | list[RelationNodeRef],
    ) -> tuple[RelationNodeRef, ...]:
        if not isinstance(nodes, (tuple, list)):
            raise TypeError("nodes must be tuple or list")
        values = tuple(nodes)
        for node in values:
            if not isinstance(node, RelationNodeRef):
                raise TypeError("nodes must contain RelationNodeRef")
        return values

    def _validate_evidence(
        self,
        evidence: tuple[RelationEvidence, ...] | list[RelationEvidence],
    ) -> tuple[RelationEvidence, ...]:
        if not isinstance(evidence, (tuple, list)):
            raise TypeError("evidence must be tuple or list")
        values = tuple(evidence)
        for item in values:
            if not isinstance(item, RelationEvidence):
                raise TypeError("evidence must contain RelationEvidence")
        return values

    def _build_node_lookup(
        self,
        nodes: tuple[RelationNodeRef, ...],
    ) -> dict[str, RelationNodeRef]:
        lookup: dict[str, RelationNodeRef] = {}
        for node in nodes:
            lookup[node.node_id] = node
        return lookup

    def _edge_from_evidence(
        self,
        *,
        evidence: RelationEvidence,
        node_lookup: dict[str, RelationNodeRef],
    ) -> tuple[RelationEdge | None, str]:
        metadata = evidence.metadata
        source_id = self._metadata_string(metadata, "source_node_id")
        if source_id is None:
            return None, "missing source_node_id"
        target_id = self._metadata_string(metadata, "target_node_id")
        if target_id is None:
            return None, "missing target_node_id"
        edge_kind_value = self._metadata_string(metadata, "edge_kind")
        if edge_kind_value is None:
            return None, "missing edge_kind"

        source = node_lookup.get(source_id)
        if source is None:
            return None, f"unknown source_node_id: {source_id}"
        target = node_lookup.get(target_id)
        if target is None:
            return None, f"unknown target_node_id: {target_id}"
        if source.node_id == target.node_id and not self.policy.allow_self_edges:
            return None, f"self edge rejected: {source.node_id}"

        edge_kind = self._map_edge_kind(edge_kind_value)
        if edge_kind is None:
            return None, f"unknown edge_kind: {edge_kind_value}"

        status_value = self._metadata_optional_string(metadata, "status")
        status = None
        if status_value is not None:
            status = self._map_status(status_value)
            if status is None:
                return None, f"invalid status: {status_value}"

        confidence = self._metadata_probability(
            metadata,
            "confidence",
            fallback=evidence.confidence,
        )
        if confidence is None:
            return None, "invalid confidence"
        if confidence < self.policy.min_confidence:
            return None, "confidence below policy minimum"

        weight = self._metadata_non_negative_number(
            metadata,
            "weight",
            fallback=evidence.weight,
        )
        if weight is None:
            return None, "invalid weight"
        if weight < self.policy.min_weight:
            return None, "weight below policy minimum"

        edge_metadata = self._metadata_dict(metadata, "edge_metadata")
        if edge_metadata is None and "edge_metadata" in metadata:
            return None, "invalid edge_metadata"
        edge_status = status
        edge = create_relation_edge(
            source=source,
            target=target,
            kind=edge_kind,
            weight=weight,
            confidence=confidence,
            evidence=(evidence,),
            status=edge_status,
            metadata=edge_metadata,
        )
        return edge, ""

    def _metadata_string(
        self,
        metadata: dict[str, object],
        key: str,
    ) -> str | None:
        value = metadata.get(key)
        if not isinstance(value, str):
            return None
        stripped = value.strip()
        if stripped == "":
            return None
        return stripped

    def _metadata_optional_string(
        self,
        metadata: dict[str, object],
        key: str,
    ) -> str | None:
        if key not in metadata:
            return None
        value = metadata[key]
        if not isinstance(value, str):
            return ""
        return value.strip()

    def _metadata_probability(
        self,
        metadata: dict[str, object],
        key: str,
        *,
        fallback: float,
    ) -> float | None:
        value = metadata.get(key, fallback)
        return _probability_or_none(value)

    def _metadata_non_negative_number(
        self,
        metadata: dict[str, object],
        key: str,
        *,
        fallback: float,
    ) -> float | None:
        value = metadata.get(key, fallback)
        return _non_negative_number_or_none(value)

    def _metadata_dict(
        self,
        metadata: dict[str, object],
        key: str,
    ) -> dict[str, object] | None:
        if key not in metadata:
            return {}
        value = metadata[key]
        if not isinstance(value, dict):
            return None
        copied = dict(value)
        for item_key in copied:
            if not isinstance(item_key, str):
                return None
        return copied

    def _map_edge_kind(self, value: str) -> RelationEdgeKind | None:
        if value == "":
            return None
        for kind in RelationEdgeKind:
            if value == kind.name or value == kind.value:
                return kind
        if self.policy.allow_unknown_edge_kind:
            return RelationEdgeKind.UNKNOWN
        return None

    def _map_status(self, value: str) -> RelationStatus | None:
        if value == "":
            return None
        for status in RelationStatus:
            if value == status.name or value == status.value:
                return status
        return None

    def _deduplication_key(
        self,
        edge: RelationEdge,
    ) -> tuple[str, str, str]:
        return (edge.source.node_id, edge.target.node_id, edge.kind.value)

    def _deduplicate_edges(
        self,
        edges: tuple[RelationEdge, ...],
    ) -> tuple[RelationEdge, ...]:
        selected: dict[tuple[str, str, str], RelationEdge] = {}
        order: list[tuple[str, str, str]] = []
        for edge in edges:
            key = self._deduplication_key(edge)
            current = selected.get(key)
            if current is None:
                selected[key] = edge
                order.append(key)
                continue
            if edge.confidence > current.confidence:
                selected[key] = edge
            elif (
                edge.confidence == current.confidence
                and edge.weight > current.weight
            ):
                selected[key] = edge
        return tuple(selected[key] for key in order)

    def _sort_edges(
        self,
        edges: tuple[RelationEdge, ...],
    ) -> tuple[RelationEdge, ...]:
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


def extract_relations(
    *,
    nodes: tuple[RelationNodeRef, ...] | list[RelationNodeRef],
    evidence: tuple[RelationEvidence, ...] | list[RelationEvidence],
    policy: RelationExtractionPolicy | None = None,
) -> RelationExtractionResult:
    """Extract relation edges from explicit evidence."""

    extractor = RelationExtractor(policy)
    return extractor.extract(nodes=nodes, evidence=evidence)


def extract_relation_graph_snapshot(
    *,
    nodes: tuple[RelationNodeRef, ...] | list[RelationNodeRef],
    evidence: tuple[RelationEvidence, ...] | list[RelationEvidence],
    policy: RelationExtractionPolicy | None = None,
    status: RelationStatus = RelationStatus.ACTIVE,
    metadata: dict[str, object] | None = None,
) -> RelationGraphSnapshot:
    """Extract relation edges and return a graph snapshot."""

    extractor = RelationExtractor(policy)
    return extractor.extract_graph_snapshot(
        nodes=nodes,
        evidence=evidence,
        status=status,
        metadata=metadata,
    )


def _ensure_probability(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be finite number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{field_name} must be in range 0.0..1.0")
    return number


def _ensure_non_negative_number(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be finite number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    if number < 0.0:
        raise ValueError(f"{field_name} must be non-negative")
    return number


def _ensure_optional_positive_int(
    value: int | None,
    field_name: str,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int or None")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _probability_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not isfinite(number) or number < 0.0 or number > 1.0:
        return None
    return number


def _non_negative_number_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not isfinite(number) or number < 0.0:
        return None
    return number


def _ensure_non_empty_string_tuple(
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


def _merge_metadata(
    default_metadata: dict[str, object],
    caller_metadata: dict[str, object] | None,
) -> dict[str, object]:
    merged = _copy_metadata(default_metadata)
    if caller_metadata is not None:
        merged.update(_copy_metadata(caller_metadata))
    return merged


__all__ = [
    "RelationExtractionPolicy",
    "RelationExtractionResult",
    "RelationExtractor",
    "extract_relations",
    "extract_relation_graph_snapshot",
]
