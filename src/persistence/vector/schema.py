"""Strict schema objects for CEBA vector persistence surfaces."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from uuid import UUID, uuid4


__all__ = [
    "VectorBackend",
    "VectorDistanceMetric",
    "VectorIndexStatus",
    "VectorPayloadKind",
    "VectorPointStatus",
    "VectorIndexConfig",
    "VectorPayload",
    "VectorPointRef",
    "VectorSearchQuery",
    "VectorSearchMatch",
    "VectorIndexSnapshot",
    "utc_now",
    "new_vector_point_id",
    "new_vector_snapshot_id",
    "ensure_non_empty_string",
    "ensure_optional_non_empty_string",
    "ensure_positive_int",
    "ensure_non_negative_int",
    "ensure_probability",
    "ensure_finite_number",
    "ensure_timezone_aware",
    "create_vector_index_config",
    "create_vector_payload",
    "create_vector_point_ref",
    "create_vector_search_query",
    "create_vector_search_match",
    "create_vector_index_snapshot",
]


_FORBIDDEN_RUNTIME_STATE_NAMES = {
    "M_t",
    "Mt",
    "semantic_state",
    "runtime_semantic_state",
}


class VectorBackend(Enum):
    """Stable vector backend categories."""

    QDRANT = "QDRANT"
    PGVECTOR = "PGVECTOR"
    LANCE = "LANCE"
    IN_MEMORY = "IN_MEMORY"
    UNKNOWN = "UNKNOWN"


class VectorDistanceMetric(Enum):
    """Stable vector distance metrics."""

    COSINE = "COSINE"
    DOT = "DOT"
    EUCLIDEAN = "EUCLIDEAN"
    MANHATTAN = "MANHATTAN"
    UNKNOWN = "UNKNOWN"


class VectorIndexStatus(Enum):
    """Stable vector index lifecycle statuses."""

    CREATED = "CREATED"
    READY = "READY"
    DEGRADED = "DEGRADED"
    REBUILDING = "REBUILDING"
    DISABLED = "DISABLED"
    FAILED = "FAILED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class VectorPayloadKind(Enum):
    """Stable vector payload categories."""

    OBSERVATION = "OBSERVATION"
    TRACE = "TRACE"
    EVENT = "EVENT"
    DECISION = "DECISION"
    CONSTRAINT = "CONSTRAINT"
    TELOS = "TELOS"
    AUDIT = "AUDIT"
    CONCEPT = "CONCEPT"
    RELATION = "RELATION"
    UNKNOWN = "UNKNOWN"


class VectorPointStatus(Enum):
    """Stable vector point lifecycle statuses."""

    CREATED = "CREATED"
    INDEXED = "INDEXED"
    STALE = "STALE"
    DELETED = "DELETED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class VectorIndexConfig:
    """Immutable vector index configuration descriptor."""

    backend: VectorBackend
    collection_name: str
    dimension: int
    distance_metric: VectorDistanceMetric
    status: VectorIndexStatus
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.backend, VectorBackend):
            raise TypeError("backend must be VectorBackend")
        collection_name = ensure_non_empty_string(
            self.collection_name,
            "collection_name",
        )
        _reject_forbidden_runtime_state_name(
            collection_name,
            "collection_name",
        )
        dimension = ensure_positive_int(self.dimension, "dimension")
        if not isinstance(self.distance_metric, VectorDistanceMetric):
            raise TypeError("distance_metric must be VectorDistanceMetric")
        if not isinstance(self.status, VectorIndexStatus):
            raise TypeError("status must be VectorIndexStatus")
        metadata = _copy_string_keyed_dict(self.metadata, "metadata")
        _reject_forbidden_runtime_state_keys(metadata, "metadata")
        object.__setattr__(self, "collection_name", collection_name)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "backend": self.backend.value,
            "collection_name": self.collection_name,
            "dimension": self.dimension,
            "distance_metric": self.distance_metric.value,
            "status": self.status.value,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class VectorPayload:
    """Immutable metadata payload attached to a vector point."""

    payload_kind: VectorPayloadKind
    source_record_id: str | None
    source_record_key: str | None
    data: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.payload_kind, VectorPayloadKind):
            raise TypeError("payload_kind must be VectorPayloadKind")
        source_record_id = ensure_optional_non_empty_string(
            self.source_record_id,
            "source_record_id",
        )
        source_record_key = ensure_optional_non_empty_string(
            self.source_record_key,
            "source_record_key",
        )
        data = _copy_string_keyed_dict(self.data, "data")
        metadata = _copy_string_keyed_dict(self.metadata, "metadata")
        _reject_forbidden_runtime_state_keys(data, "data")
        _reject_forbidden_runtime_state_keys(metadata, "metadata")
        object.__setattr__(self, "source_record_id", source_record_id)
        object.__setattr__(self, "source_record_key", source_record_key)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "payload_kind": self.payload_kind.value,
            "source_record_id": self.source_record_id,
            "source_record_key": self.source_record_key,
            "data": dict(self.data),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class VectorPointRef:
    """Immutable descriptor for one vector point in a retrieval surface."""

    point_id: UUID
    vector: tuple[float, ...]
    payload: VectorPayload
    status: VectorPointStatus
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.point_id, UUID):
            raise TypeError("point_id must be UUID")
        vector = _ensure_vector_tuple(self.vector, "vector")
        if not isinstance(self.payload, VectorPayload):
            raise TypeError("payload must be VectorPayload")
        if not isinstance(self.status, VectorPointStatus):
            raise TypeError("status must be VectorPointStatus")
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        updated_at = ensure_timezone_aware(self.updated_at, "updated_at")
        if updated_at < created_at:
            raise ValueError("updated_at must be >= created_at")
        metadata = _copy_string_keyed_dict(self.metadata, "metadata")
        _reject_forbidden_runtime_state_keys(metadata, "metadata")
        object.__setattr__(self, "vector", vector)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "point_id": str(self.point_id),
            "vector": list(self.vector),
            "payload": self.payload.to_dict(),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    def dimension(self) -> int:
        """Return vector dimensionality."""

        return len(self.vector)


@dataclass(frozen=True)
class VectorSearchQuery:
    """Immutable descriptor for a vector retrieval query."""

    query_vector: tuple[float, ...]
    top_k: int
    payload_kind: VectorPayloadKind | None
    min_score: float
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        query_vector = _ensure_vector_tuple(self.query_vector, "query_vector")
        top_k = ensure_positive_int(self.top_k, "top_k")
        if self.payload_kind is not None and not isinstance(
            self.payload_kind,
            VectorPayloadKind,
        ):
            raise TypeError("payload_kind must be None or VectorPayloadKind")
        min_score = ensure_probability(self.min_score, "min_score")
        metadata = _copy_string_keyed_dict(self.metadata, "metadata")
        _reject_forbidden_runtime_state_keys(metadata, "metadata")
        object.__setattr__(self, "query_vector", query_vector)
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(self, "min_score", min_score)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "query_vector": list(self.query_vector),
            "top_k": self.top_k,
            "payload_kind": None if self.payload_kind is None else self.payload_kind.value,
            "min_score": self.min_score,
            "metadata": dict(self.metadata),
        }

    def dimension(self) -> int:
        """Return query vector dimensionality."""

        return len(self.query_vector)


@dataclass(frozen=True)
class VectorSearchMatch:
    """Immutable descriptor for one vector search match."""

    point: VectorPointRef
    score: float
    rank: int
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.point, VectorPointRef):
            raise TypeError("point must be VectorPointRef")
        score = ensure_probability(self.score, "score")
        rank = ensure_positive_int(self.rank, "rank")
        metadata = _copy_string_keyed_dict(self.metadata, "metadata")
        _reject_forbidden_runtime_state_keys(metadata, "metadata")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "point": self.point.to_dict(),
            "score": self.score,
            "rank": self.rank,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class VectorIndexSnapshot:
    """Immutable snapshot of vector index state."""

    snapshot_id: UUID
    config: VectorIndexConfig
    points: tuple[VectorPointRef, ...]
    status: VectorIndexStatus
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot_id, UUID):
            raise TypeError("snapshot_id must be UUID")
        if not isinstance(self.config, VectorIndexConfig):
            raise TypeError("config must be VectorIndexConfig")
        points = _ensure_point_tuple(self.points)
        point_ids = [point.point_id for point in points]
        _ensure_unique_values(point_ids, "point_id")
        for point in points:
            if point.dimension() != self.config.dimension:
                raise ValueError("every point dimension must equal config.dimension")
        if not isinstance(self.status, VectorIndexStatus):
            raise TypeError("status must be VectorIndexStatus")
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        metadata = _copy_string_keyed_dict(self.metadata, "metadata")
        _reject_forbidden_runtime_state_keys(metadata, "metadata")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "snapshot_id": str(self.snapshot_id),
            "config": self.config.to_dict(),
            "points": [point.to_dict() for point in self.points],
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    def point_count(self) -> int:
        """Return number of vector points in the snapshot."""

        return len(self.points)


def utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""

    return datetime.now(UTC)


def new_vector_point_id() -> UUID:
    """Return a new vector point UUID."""

    return uuid4()


def new_vector_snapshot_id() -> UUID:
    """Return a new vector snapshot UUID."""

    return uuid4()


def ensure_non_empty_string(value: str, field_name: str) -> str:
    """Validate and normalize a non-empty string."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def ensure_optional_non_empty_string(
    value: str | None,
    field_name: str,
) -> str | None:
    """Validate and normalize an optional non-empty string."""

    if value is None:
        return None
    return ensure_non_empty_string(value, field_name)


def ensure_positive_int(value: int, field_name: str) -> int:
    """Validate a positive integer, rejecting bool."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return value


def ensure_non_negative_int(value: int, field_name: str) -> int:
    """Validate a non-negative integer, rejecting bool."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return value


def ensure_probability(value: float, field_name: str) -> float:
    """Validate a finite probability value in the inclusive range 0.0..1.0."""

    number = ensure_finite_number(value, field_name)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return number


def ensure_finite_number(value: float, field_name: str) -> float:
    """Validate a finite numeric value, rejecting bool."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be int or float")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def ensure_timezone_aware(value: datetime, field_name: str) -> datetime:
    """Validate a timezone-aware datetime."""

    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def create_vector_index_config(
    *,
    collection_name: str,
    dimension: int,
    backend: VectorBackend = VectorBackend.UNKNOWN,
    distance_metric: VectorDistanceMetric = VectorDistanceMetric.COSINE,
    status: VectorIndexStatus = VectorIndexStatus.CREATED,
    metadata: dict[str, object] | None = None,
) -> VectorIndexConfig:
    """Create an immutable vector index configuration descriptor."""

    required_metadata = {
        "source": "vector_index_config",
        "backend": backend.value if isinstance(backend, VectorBackend) else backend,
        "collection_name": collection_name,
        "dimension": dimension,
    }
    return VectorIndexConfig(
        backend=backend,
        collection_name=collection_name,
        dimension=dimension,
        distance_metric=distance_metric,
        status=status,
        metadata=_merge_metadata(required_metadata, metadata),
    )


def create_vector_payload(
    *,
    payload_kind: VectorPayloadKind = VectorPayloadKind.UNKNOWN,
    source_record_id: str | None = None,
    source_record_key: str | None = None,
    data: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
) -> VectorPayload:
    """Create an immutable vector payload descriptor."""

    required_metadata = {
        "source": "vector_payload",
        "payload_kind": (
            payload_kind.value
            if isinstance(payload_kind, VectorPayloadKind)
            else payload_kind
        ),
    }
    return VectorPayload(
        payload_kind=payload_kind,
        source_record_id=source_record_id,
        source_record_key=source_record_key,
        data={} if data is None else data,
        metadata=_merge_metadata(required_metadata, metadata),
    )


def create_vector_point_ref(
    *,
    vector: tuple[float, ...] | list[float],
    payload: VectorPayload,
    status: VectorPointStatus = VectorPointStatus.CREATED,
    metadata: dict[str, object] | None = None,
) -> VectorPointRef:
    """Create an immutable vector point reference."""

    timestamp = utc_now()
    required_metadata = {
        "source": "vector_point_ref",
        "dimension": len(vector),
        "point_status": status.value if isinstance(status, VectorPointStatus) else status,
    }
    return VectorPointRef(
        point_id=new_vector_point_id(),
        vector=tuple(vector),
        payload=payload,
        status=status,
        created_at=timestamp,
        updated_at=timestamp,
        metadata=_merge_metadata(required_metadata, metadata),
    )


def create_vector_search_query(
    *,
    query_vector: tuple[float, ...] | list[float],
    top_k: int = 10,
    payload_kind: VectorPayloadKind | None = None,
    min_score: float = 0.0,
    metadata: dict[str, object] | None = None,
) -> VectorSearchQuery:
    """Create an immutable vector search query descriptor."""

    required_metadata = {
        "source": "vector_search_query",
        "dimension": len(query_vector),
        "top_k": top_k,
    }
    return VectorSearchQuery(
        query_vector=tuple(query_vector),
        top_k=top_k,
        payload_kind=payload_kind,
        min_score=min_score,
        metadata=_merge_metadata(required_metadata, metadata),
    )


def create_vector_search_match(
    *,
    point: VectorPointRef,
    score: float,
    rank: int,
    metadata: dict[str, object] | None = None,
) -> VectorSearchMatch:
    """Create an immutable vector search match descriptor."""

    required_metadata = {
        "source": "vector_search_match",
        "point_id": str(point.point_id) if isinstance(point, VectorPointRef) else None,
        "score": score,
        "rank": rank,
    }
    return VectorSearchMatch(
        point=point,
        score=score,
        rank=rank,
        metadata=_merge_metadata(required_metadata, metadata),
    )


def create_vector_index_snapshot(
    *,
    config: VectorIndexConfig,
    points: tuple[VectorPointRef, ...] | list[VectorPointRef] = (),
    status: VectorIndexStatus = VectorIndexStatus.CREATED,
    metadata: dict[str, object] | None = None,
) -> VectorIndexSnapshot:
    """Create an immutable vector index snapshot."""

    points_tuple = tuple(points)
    required_metadata = {
        "source": "vector_index_snapshot",
        "backend": config.backend.value if isinstance(config, VectorIndexConfig) else None,
        "collection_name": (
            config.collection_name if isinstance(config, VectorIndexConfig) else None
        ),
        "dimension": config.dimension if isinstance(config, VectorIndexConfig) else None,
        "point_count": len(points_tuple),
    }
    return VectorIndexSnapshot(
        snapshot_id=new_vector_snapshot_id(),
        config=config,
        points=points_tuple,
        status=status,
        created_at=utc_now(),
        metadata=_merge_metadata(required_metadata, metadata),
    )


def _copy_string_keyed_dict(
    value: dict[str, object],
    field_name: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be dict[str, object]")
    copied = dict(value)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError(f"{field_name} keys must be strings")
    return copied


def _merge_metadata(
    default_metadata: dict[str, object],
    metadata: dict[str, object] | None,
) -> dict[str, object]:
    merged = _copy_string_keyed_dict(default_metadata, "metadata")
    if metadata is not None:
        merged.update(_copy_string_keyed_dict(metadata, "metadata"))
    if "source" in default_metadata:
        merged["source"] = default_metadata["source"]
    return merged


def _reject_forbidden_runtime_state_name(value: str, field_name: str) -> None:
    if value in _FORBIDDEN_RUNTIME_STATE_NAMES:
        raise ValueError(f"{field_name} must not persist runtime semantic state")


def _reject_forbidden_runtime_state_keys(
    value: dict[str, object],
    field_name: str,
) -> None:
    for key in value:
        if key in _FORBIDDEN_RUNTIME_STATE_NAMES:
            raise ValueError(f"{field_name} must not persist runtime semantic state")


def _ensure_vector_tuple(
    value: tuple[float, ...] | list[float],
    field_name: str,
) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple[float, ...]")
    if len(value) == 0:
        raise ValueError(f"{field_name} must not be empty")
    return tuple(
        ensure_finite_number(item, f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )


def _ensure_point_tuple(
    value: tuple[VectorPointRef, ...] | list[VectorPointRef],
) -> tuple[VectorPointRef, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("points must be tuple[VectorPointRef, ...]")
    points = tuple(value)
    for point in points:
        if not isinstance(point, VectorPointRef):
            raise TypeError("every point must be VectorPointRef")
    return points


def _ensure_unique_values(values: list[object], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} values must be unique")
