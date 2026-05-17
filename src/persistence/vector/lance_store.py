"""Local LanceDB store adapter descriptors for CEBA vector persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from ceba.persistence.vector.index_policy import (
    VectorIndexPolicy,
    VectorIndexPolicyManager,
)
from ceba.persistence.vector.schema import (
    VectorBackend,
    VectorDistanceMetric,
    VectorIndexConfig,
    VectorIndexSnapshot,
    VectorIndexStatus,
    VectorPointRef,
    VectorSearchMatch,
    VectorSearchQuery,
    create_vector_index_config,
    ensure_non_empty_string,
    ensure_positive_int,
    ensure_timezone_aware,
    utc_now,
)


__all__ = [
    "LanceOperationKind",
    "LanceOperationStatus",
    "LanceTableConfig",
    "LanceOperation",
    "LanceOperationResult",
    "LanceStoreAdapter",
    "create_lance_table_config",
    "build_lance_store_adapter",
]


_FORBIDDEN_RUNTIME_STATE_KEYS = {
    "M_t",
    "Mt",
    "semantic_state",
    "runtime_semantic_state",
}


class LanceOperationKind(Enum):
    """Stable LanceDB-style adapter operation categories."""

    CREATE_TABLE = "CREATE_TABLE"
    INSERT_POINTS = "INSERT_POINTS"
    UPSERT_POINTS = "UPSERT_POINTS"
    SEARCH = "SEARCH"
    DELETE_POINTS = "DELETE_POINTS"
    SNAPSHOT = "SNAPSHOT"
    HEALTH_CHECK = "HEALTH_CHECK"
    UNKNOWN = "UNKNOWN"


class LanceOperationStatus(Enum):
    """Stable local LanceDB-style operation statuses."""

    CREATED = "CREATED"
    READY = "READY"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class LanceTableConfig:
    """Immutable local LanceDB table descriptor."""

    dataset_uri_ref: str
    table_name: str
    vector_column: str
    dimension: int
    distance_metric: VectorDistanceMetric
    index_config: VectorIndexConfig
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dataset_uri_ref = ensure_non_empty_string(
            self.dataset_uri_ref,
            "dataset_uri_ref",
        )
        table_name = ensure_non_empty_string(self.table_name, "table_name")
        _reject_forbidden_runtime_state_name(table_name, "table_name")
        vector_column = ensure_non_empty_string(
            self.vector_column,
            "vector_column",
        )
        _reject_forbidden_runtime_state_name(vector_column, "vector_column")
        dimension = ensure_positive_int(self.dimension, "dimension")
        if not isinstance(self.distance_metric, VectorDistanceMetric):
            raise TypeError("distance_metric must be VectorDistanceMetric")
        if not isinstance(self.index_config, VectorIndexConfig):
            raise TypeError("index_config must be VectorIndexConfig")
        if self.index_config.backend is not VectorBackend.LANCE:
            raise ValueError("index_config.backend must be VectorBackend.LANCE")
        if self.index_config.collection_name != table_name:
            raise ValueError("index_config.collection_name must equal table_name")
        if self.index_config.dimension != dimension:
            raise ValueError("index_config.dimension must equal dimension")
        if self.index_config.distance_metric is not self.distance_metric:
            raise ValueError("index_config.distance_metric must equal distance_metric")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "dataset_uri_ref", dataset_uri_ref)
        object.__setattr__(self, "table_name", table_name)
        object.__setattr__(self, "vector_column", vector_column)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "dataset_uri_ref": self.dataset_uri_ref,
            "table_name": self.table_name,
            "vector_column": self.vector_column,
            "dimension": self.dimension,
            "distance_metric": self.distance_metric.value,
            "index_config": self.index_config.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LanceOperation:
    """Immutable local descriptor for one LanceDB adapter operation."""

    operation_id: UUID
    kind: LanceOperationKind
    status: LanceOperationStatus
    table_name: str
    point_ids: tuple[str, ...]
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.operation_id, UUID):
            raise TypeError("operation_id must be UUID")
        if not isinstance(self.kind, LanceOperationKind):
            raise TypeError("kind must be LanceOperationKind")
        if not isinstance(self.status, LanceOperationStatus):
            raise TypeError("status must be LanceOperationStatus")
        table_name = ensure_non_empty_string(self.table_name, "table_name")
        _reject_forbidden_runtime_state_name(table_name, "table_name")
        point_ids = _ensure_point_id_tuple(self.point_ids)
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("point ids must be unique")
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "table_name", table_name)
        object.__setattr__(self, "point_ids", point_ids)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "operation_id": str(self.operation_id),
            "kind": self.kind.value,
            "status": self.status.value,
            "table_name": self.table_name,
            "point_ids": list(self.point_ids),
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    def point_count(self) -> int:
        """Return number of point ids in the operation."""

        return len(self.point_ids)


@dataclass(frozen=True)
class LanceOperationResult:
    """Immutable result returned by local LanceDB adapter operations."""

    operation: LanceOperation
    accepted_points: tuple[VectorPointRef, ...]
    matches: tuple[VectorSearchMatch, ...]
    snapshot: VectorIndexSnapshot | None
    changed: bool
    reason: str
    rejected_point_ids: tuple[str, ...]
    rejection_reasons: tuple[str, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.operation, LanceOperation):
            raise TypeError("operation must be LanceOperation")
        accepted_points = _ensure_point_tuple(self.accepted_points)
        point_ids = [point.point_id for point in accepted_points]
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("point ids must be unique")
        matches = _ensure_match_tuple(self.matches)
        if self.snapshot is not None and not isinstance(
            self.snapshot,
            VectorIndexSnapshot,
        ):
            raise TypeError("snapshot must be None or VectorIndexSnapshot")
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        reason = ensure_non_empty_string(self.reason, "reason")
        rejected_point_ids = _ensure_point_id_tuple(self.rejected_point_ids)
        rejection_reasons = _ensure_string_tuple(
            self.rejection_reasons,
            "rejection_reasons",
        )
        if len(rejected_point_ids) != len(rejection_reasons):
            raise ValueError(
                "rejected_point_ids and rejection_reasons must have same length"
            )
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "accepted_points", accepted_points)
        object.__setattr__(self, "matches", matches)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "rejected_point_ids", rejected_point_ids)
        object.__setattr__(self, "rejection_reasons", rejection_reasons)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "operation": self.operation.to_dict(),
            "accepted_points": [point.to_dict() for point in self.accepted_points],
            "matches": [match.to_dict() for match in self.matches],
            "snapshot": None if self.snapshot is None else self.snapshot.to_dict(),
            "changed": self.changed,
            "reason": self.reason,
            "rejected_point_ids": list(self.rejected_point_ids),
            "rejection_reasons": list(self.rejection_reasons),
            "metadata": dict(self.metadata),
        }

    def accepted_count(self) -> int:
        """Return accepted point count."""

        return len(self.accepted_points)

    def match_count(self) -> int:
        """Return match count."""

        return len(self.matches)

    def rejected_count(self) -> int:
        """Return rejected point count."""

        return len(self.rejected_point_ids)

    def has_snapshot(self) -> bool:
        """Return whether a snapshot is attached."""

        return self.snapshot is not None

    def is_success(self) -> bool:
        """Return whether the operation result is successful."""

        return (
            self.changed is True
            and self.operation.status
            in {LanceOperationStatus.ACCEPTED, LanceOperationStatus.READY}
            and self.rejected_count() == 0
        )


class LanceStoreAdapter:
    """Deterministic local adapter boundary for LanceDB-style operations."""

    def __init__(
        self,
        *,
        table: LanceTableConfig,
        policy: VectorIndexPolicy | None = None,
    ) -> None:
        if not isinstance(table, LanceTableConfig):
            raise TypeError("table must be LanceTableConfig")
        if policy is None:
            policy = VectorIndexPolicy()
        if not isinstance(policy, VectorIndexPolicy):
            raise TypeError("policy must be VectorIndexPolicy")
        self._table = table
        self._policy = policy
        self._policy_manager = VectorIndexPolicyManager(policy=policy)

    def table(self) -> LanceTableConfig:
        """Return table descriptor."""

        return self._table

    def create_table_operation(self) -> LanceOperationResult:
        """Create a local table creation operation descriptor."""

        metadata = {
            "source": "lance_store_adapter",
            "operation": "create_table_operation",
            "backend": "LANCE",
            "dataset_uri_ref": self._table.dataset_uri_ref,
            "table_name": self._table.table_name,
            "vector_column": self._table.vector_column,
            "dimension": self._table.dimension,
        }
        operation = _create_operation(
            kind=LanceOperationKind.CREATE_TABLE,
            status=LanceOperationStatus.READY,
            table_name=self._table.table_name,
            point_ids=(),
            metadata=metadata,
        )
        return _operation_result(
            operation=operation,
            changed=True,
            reason="lance table operation prepared",
            metadata=metadata,
        )

    def prepare_insert(
        self,
        *,
        points: tuple[VectorPointRef, ...] | list[VectorPointRef],
        metadata: dict[str, object] | None = None,
    ) -> LanceOperationResult:
        """Prepare a local LanceDB-style insert operation descriptor."""

        return self._prepare_write(
            kind=LanceOperationKind.INSERT_POINTS,
            operation_name="prepare_insert",
            prepared_reason="lance insert operation prepared",
            rejected_reason="lance insert operation rejected",
            points=points,
            metadata=metadata,
        )

    def prepare_upsert(
        self,
        *,
        points: tuple[VectorPointRef, ...] | list[VectorPointRef],
        metadata: dict[str, object] | None = None,
    ) -> LanceOperationResult:
        """Prepare a local LanceDB-style upsert operation descriptor."""

        return self._prepare_write(
            kind=LanceOperationKind.UPSERT_POINTS,
            operation_name="prepare_upsert",
            prepared_reason="lance upsert operation prepared",
            rejected_reason="lance upsert operation rejected",
            points=points,
            metadata=metadata,
        )

    def prepare_search(
        self,
        *,
        query: VectorSearchQuery,
        candidate_matches: tuple[VectorSearchMatch, ...] | list[VectorSearchMatch] = (),
        metadata: dict[str, object] | None = None,
    ) -> LanceOperationResult:
        """Prepare a local LanceDB-style search operation descriptor."""

        if not isinstance(query, VectorSearchQuery):
            raise TypeError("query must be VectorSearchQuery")
        if query.dimension() != self._table.dimension:
            raise ValueError("query dimension must equal table.dimension")
        selection = self._policy_manager.select_matches(
            query=query,
            matches=candidate_matches,
        )
        point_ids = tuple(str(match.point.point_id) for match in selection.matches)
        required_metadata = {
            "source": "lance_store_adapter",
            "operation": "prepare_search",
            "backend": "LANCE",
            "dataset_uri_ref": self._table.dataset_uri_ref,
            "table_name": self._table.table_name,
            "query_dimension": query.dimension(),
            "match_count": selection.match_count(),
            "policy": self._policy.to_dict(),
        }
        merged_metadata = _merge_metadata(required_metadata, metadata)
        operation = _create_operation(
            kind=LanceOperationKind.SEARCH,
            status=LanceOperationStatus.ACCEPTED,
            table_name=self._table.table_name,
            point_ids=point_ids,
            metadata=merged_metadata,
        )
        return _operation_result(
            operation=operation,
            matches=selection.matches,
            changed=False,
            reason="lance search operation prepared",
            rejected_point_ids=selection.rejected_point_ids,
            rejection_reasons=selection.rejection_reasons,
            metadata=merged_metadata,
        )

    def prepare_delete(
        self,
        *,
        point_ids: tuple[str, ...] | list[str],
        metadata: dict[str, object] | None = None,
    ) -> LanceOperationResult:
        """Prepare a local LanceDB-style delete operation descriptor."""

        normalized_point_ids = _ensure_point_id_tuple(point_ids)
        if len(normalized_point_ids) != len(set(normalized_point_ids)):
            raise ValueError("point ids must be unique")
        accepted = bool(normalized_point_ids)
        required_metadata = {
            "source": "lance_store_adapter",
            "operation": "prepare_delete",
            "backend": "LANCE",
            "dataset_uri_ref": self._table.dataset_uri_ref,
            "table_name": self._table.table_name,
            "point_count": len(normalized_point_ids),
        }
        merged_metadata = _merge_metadata(required_metadata, metadata)
        operation = _create_operation(
            kind=LanceOperationKind.DELETE_POINTS,
            status=(
                LanceOperationStatus.ACCEPTED
                if accepted
                else LanceOperationStatus.REJECTED
            ),
            table_name=self._table.table_name,
            point_ids=normalized_point_ids,
            metadata=merged_metadata,
        )
        return _operation_result(
            operation=operation,
            changed=accepted,
            reason=(
                "lance delete operation prepared"
                if accepted
                else "lance delete operation rejected"
            ),
            metadata=merged_metadata,
        )

    def create_snapshot(
        self,
        *,
        points: tuple[VectorPointRef, ...] | list[VectorPointRef],
        metadata: dict[str, object] | None = None,
    ) -> LanceOperationResult:
        """Create a local LanceDB-style vector index snapshot descriptor."""

        selection = self._policy_manager.select_points(
            config=self._table.index_config,
            points=points,
        )
        required_metadata = {
            "source": "lance_store_adapter",
            "operation": "create_snapshot",
            "backend": "LANCE",
            "dataset_uri_ref": self._table.dataset_uri_ref,
            "table_name": self._table.table_name,
            "point_count": selection.point_count(),
            "policy": self._policy.to_dict(),
        }
        merged_metadata = _merge_metadata(required_metadata, metadata)
        snapshot = self._policy_manager.create_snapshot(
            config=self._table.index_config,
            points=selection.points,
            status=self._table.index_config.status,
            metadata=merged_metadata,
        )
        operation = _create_operation(
            kind=LanceOperationKind.SNAPSHOT,
            status=LanceOperationStatus.READY,
            table_name=self._table.table_name,
            point_ids=tuple(str(point.point_id) for point in selection.points),
            metadata=merged_metadata,
        )
        return _operation_result(
            operation=operation,
            accepted_points=selection.points,
            snapshot=snapshot,
            changed=True,
            reason="lance snapshot created",
            rejected_point_ids=selection.rejected_point_ids,
            rejection_reasons=selection.rejection_reasons,
            metadata=merged_metadata,
        )

    def _prepare_write(
        self,
        *,
        kind: LanceOperationKind,
        operation_name: str,
        prepared_reason: str,
        rejected_reason: str,
        points: tuple[VectorPointRef, ...] | list[VectorPointRef],
        metadata: dict[str, object] | None,
    ) -> LanceOperationResult:
        selection = self._policy_manager.select_points(
            config=self._table.index_config,
            points=points,
        )
        accepted_point_ids = tuple(str(point.point_id) for point in selection.points)
        accepted = selection.has_points()
        required_metadata = {
            "source": "lance_store_adapter",
            "operation": operation_name,
            "backend": "LANCE",
            "dataset_uri_ref": self._table.dataset_uri_ref,
            "table_name": self._table.table_name,
            "accepted_point_count": selection.point_count(),
            "rejected_point_count": selection.rejected_count(),
            "policy": self._policy.to_dict(),
        }
        merged_metadata = _merge_metadata(required_metadata, metadata)
        operation = _create_operation(
            kind=kind,
            status=(
                LanceOperationStatus.ACCEPTED
                if accepted
                else LanceOperationStatus.REJECTED
            ),
            table_name=self._table.table_name,
            point_ids=accepted_point_ids,
            metadata=merged_metadata,
        )
        return _operation_result(
            operation=operation,
            accepted_points=selection.points,
            changed=accepted,
            reason=prepared_reason if accepted else rejected_reason,
            rejected_point_ids=selection.rejected_point_ids,
            rejection_reasons=selection.rejection_reasons,
            metadata=merged_metadata,
        )


def create_lance_table_config(
    *,
    dataset_uri_ref: str,
    table_name: str,
    dimension: int,
    vector_column: str = "embedding",
    distance_metric: VectorDistanceMetric = VectorDistanceMetric.COSINE,
    status: VectorIndexStatus = VectorIndexStatus.CREATED,
    metadata: dict[str, object] | None = None,
) -> LanceTableConfig:
    """Create an immutable local LanceDB table descriptor."""

    required_metadata = {
        "source": "lance_table_config",
        "backend": "LANCE",
        "dataset_uri_ref": dataset_uri_ref,
        "table_name": table_name,
        "vector_column": vector_column,
        "dimension": dimension,
    }
    merged_metadata = _merge_metadata(required_metadata, metadata)
    index_config = create_vector_index_config(
        backend=VectorBackend.LANCE,
        collection_name=table_name,
        dimension=dimension,
        distance_metric=distance_metric,
        status=status,
        metadata=merged_metadata,
    )
    return LanceTableConfig(
        dataset_uri_ref=dataset_uri_ref,
        table_name=table_name,
        vector_column=vector_column,
        dimension=dimension,
        distance_metric=distance_metric,
        index_config=index_config,
        metadata=merged_metadata,
    )


def build_lance_store_adapter(
    *,
    table: LanceTableConfig,
    policy: VectorIndexPolicy | None = None,
) -> LanceStoreAdapter:
    """Build a deterministic local LanceDB-style store adapter."""

    return LanceStoreAdapter(table=table, policy=policy)


def _new_lance_operation_id() -> UUID:
    return uuid4()


def _create_operation(
    *,
    kind: LanceOperationKind,
    status: LanceOperationStatus,
    table_name: str,
    point_ids: tuple[str, ...] | list[str],
    metadata: dict[str, object],
) -> LanceOperation:
    return LanceOperation(
        operation_id=_new_lance_operation_id(),
        kind=kind,
        status=status,
        table_name=table_name,
        point_ids=point_ids,
        created_at=utc_now(),
        metadata=metadata,
    )


def _operation_result(
    *,
    operation: LanceOperation,
    accepted_points: tuple[VectorPointRef, ...] | list[VectorPointRef] = (),
    matches: tuple[VectorSearchMatch, ...] | list[VectorSearchMatch] = (),
    snapshot: VectorIndexSnapshot | None = None,
    changed: bool,
    reason: str,
    rejected_point_ids: tuple[str, ...] | list[str] = (),
    rejection_reasons: tuple[str, ...] | list[str] = (),
    metadata: dict[str, object],
) -> LanceOperationResult:
    return LanceOperationResult(
        operation=operation,
        accepted_points=tuple(accepted_points),
        matches=tuple(matches),
        snapshot=snapshot,
        changed=changed,
        reason=reason,
        rejected_point_ids=tuple(rejected_point_ids),
        rejection_reasons=tuple(rejection_reasons),
        metadata=metadata,
    )


def _ensure_point_tuple(
    value: tuple[VectorPointRef, ...] | list[VectorPointRef],
) -> tuple[VectorPointRef, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("accepted_points must be tuple[VectorPointRef, ...]")
    points = tuple(value)
    for point in points:
        if not isinstance(point, VectorPointRef):
            raise TypeError("every accepted point must be VectorPointRef")
    return points


def _ensure_match_tuple(
    value: tuple[VectorSearchMatch, ...] | list[VectorSearchMatch],
) -> tuple[VectorSearchMatch, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("matches must be tuple[VectorSearchMatch, ...]")
    matches = tuple(value)
    for match in matches:
        if not isinstance(match, VectorSearchMatch):
            raise TypeError("every match must be VectorSearchMatch")
    return matches


def _ensure_point_id_tuple(value: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("point_ids must be tuple[str, ...]")
    return tuple(ensure_non_empty_string(point_id, "point_id") for point_id in value)


def _ensure_string_tuple(
    value: tuple[str, ...] | list[str],
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple[str, ...]")
    return tuple(ensure_non_empty_string(item, field_name) for item in value)


def _copy_metadata(metadata: dict[str, object]) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(metadata)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
        if key in _FORBIDDEN_RUNTIME_STATE_KEYS:
            raise ValueError("metadata must not persist runtime semantic state")
    return copied


def _merge_metadata(
    default_metadata: dict[str, object],
    metadata: dict[str, object] | None,
) -> dict[str, object]:
    merged = _copy_metadata(default_metadata)
    if metadata is not None:
        merged.update(_copy_metadata(metadata))
    if "source" in default_metadata:
        merged["source"] = default_metadata["source"]
    return merged


def _reject_forbidden_runtime_state_name(value: str, field_name: str) -> None:
    if value in _FORBIDDEN_RUNTIME_STATE_KEYS:
        raise ValueError(f"{field_name} must not persist runtime semantic state")
