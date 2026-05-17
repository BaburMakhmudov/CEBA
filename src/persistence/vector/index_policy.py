"""Deterministic vector index policy utilities for CEBA vector persistence."""

from __future__ import annotations

from dataclasses import dataclass

from ceba.persistence.vector.schema import (
    VectorIndexConfig,
    VectorIndexSnapshot,
    VectorIndexStatus,
    VectorPointRef,
    VectorPointStatus,
    VectorSearchMatch,
    VectorSearchQuery,
    create_vector_index_snapshot,
    ensure_non_empty_string,
    ensure_positive_int,
    ensure_probability,
)


__all__ = [
    "VectorIndexPolicy",
    "VectorPointSelection",
    "VectorMatchSelection",
    "VectorIndexPolicyManager",
    "select_vector_points_for_index",
    "select_vector_matches",
    "snapshot_vector_index",
]


_FORBIDDEN_RUNTIME_STATE_KEYS = {
    "M_t",
    "Mt",
    "semantic_state",
    "runtime_semantic_state",
}


@dataclass(frozen=True)
class VectorIndexPolicy:
    """Configuration object controlling deterministic vector index behavior."""

    max_points: int = 1000000
    allow_stale_points: bool = False
    allow_deleted_points: bool = False
    allow_invalid_points: bool = False
    min_match_score: float = 0.0
    require_dimension_match: bool = True

    def __post_init__(self) -> None:
        max_points = ensure_positive_int(self.max_points, "max_points")
        if not isinstance(self.allow_stale_points, bool):
            raise TypeError("allow_stale_points must be bool")
        if not isinstance(self.allow_deleted_points, bool):
            raise TypeError("allow_deleted_points must be bool")
        if not isinstance(self.allow_invalid_points, bool):
            raise TypeError("allow_invalid_points must be bool")
        min_match_score = ensure_probability(self.min_match_score, "min_match_score")
        if not isinstance(self.require_dimension_match, bool):
            raise TypeError("require_dimension_match must be bool")
        object.__setattr__(self, "max_points", max_points)
        object.__setattr__(self, "min_match_score", min_match_score)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "max_points": self.max_points,
            "allow_stale_points": self.allow_stale_points,
            "allow_deleted_points": self.allow_deleted_points,
            "allow_invalid_points": self.allow_invalid_points,
            "min_match_score": self.min_match_score,
            "require_dimension_match": self.require_dimension_match,
        }


@dataclass(frozen=True)
class VectorPointSelection:
    """Immutable result object for vector point selection."""

    points: tuple[VectorPointRef, ...]
    rejected_point_ids: tuple[str, ...]
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        points = _ensure_point_tuple(self.points)
        point_ids = [point.point_id for point in points]
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("point_id values must be unique")
        rejected_point_ids = _ensure_string_tuple(
            self.rejected_point_ids,
            "rejected_point_ids",
        )
        rejection_reasons = _ensure_string_tuple(
            self.rejection_reasons,
            "rejection_reasons",
        )
        if len(rejected_point_ids) != len(rejection_reasons):
            raise ValueError(
                "rejected_point_ids and rejection_reasons must have same length"
            )
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "rejected_point_ids", rejected_point_ids)
        object.__setattr__(self, "rejection_reasons", rejection_reasons)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "points": [point.to_dict() for point in self.points],
            "rejected_point_ids": list(self.rejected_point_ids),
            "rejection_reasons": list(self.rejection_reasons),
        }

    def point_count(self) -> int:
        """Return accepted point count."""

        return len(self.points)

    def rejected_count(self) -> int:
        """Return rejected point count."""

        return len(self.rejected_point_ids)

    def has_points(self) -> bool:
        """Return whether any points were accepted."""

        return bool(self.points)


@dataclass(frozen=True)
class VectorMatchSelection:
    """Immutable result object for vector search match selection."""

    matches: tuple[VectorSearchMatch, ...]
    rejected_point_ids: tuple[str, ...]
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        matches = _ensure_match_tuple(self.matches)
        point_ids = [match.point.point_id for match in matches]
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("point_id values among matches must be unique")
        for match in matches:
            ensure_positive_int(match.rank, "rank")
        rejected_point_ids = _ensure_string_tuple(
            self.rejected_point_ids,
            "rejected_point_ids",
        )
        rejection_reasons = _ensure_string_tuple(
            self.rejection_reasons,
            "rejection_reasons",
        )
        if len(rejected_point_ids) != len(rejection_reasons):
            raise ValueError(
                "rejected_point_ids and rejection_reasons must have same length"
            )
        object.__setattr__(self, "matches", matches)
        object.__setattr__(self, "rejected_point_ids", rejected_point_ids)
        object.__setattr__(self, "rejection_reasons", rejection_reasons)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "matches": [match.to_dict() for match in self.matches],
            "rejected_point_ids": list(self.rejected_point_ids),
            "rejection_reasons": list(self.rejection_reasons),
        }

    def match_count(self) -> int:
        """Return accepted match count."""

        return len(self.matches)

    def rejected_count(self) -> int:
        """Return rejected match count."""

        return len(self.rejected_point_ids)

    def has_matches(self) -> bool:
        """Return whether any matches were accepted."""

        return bool(self.matches)


class VectorIndexPolicyManager:
    """Deterministic local policy manager for vector retrieval/index surfaces."""

    def __init__(
        self,
        *,
        policy: VectorIndexPolicy | None = None,
    ) -> None:
        if policy is None:
            policy = VectorIndexPolicy()
        if not isinstance(policy, VectorIndexPolicy):
            raise TypeError("policy must be VectorIndexPolicy")
        self._policy = policy

    def select_points(
        self,
        *,
        config: VectorIndexConfig,
        points: tuple[VectorPointRef, ...] | list[VectorPointRef],
    ) -> VectorPointSelection:
        """Select vector points accepted for an index."""

        if not isinstance(config, VectorIndexConfig):
            raise TypeError("config must be VectorIndexConfig")
        point_tuple = _ensure_point_tuple(points)
        accepted: list[VectorPointRef] = []
        rejected_ids: list[str] = []
        rejection_reasons: list[str] = []
        seen_ids: set[str] = set()

        for point in point_tuple:
            point_id = str(point.point_id)
            reason = self._point_rejection_reason(config=config, point=point)
            if reason is None and point_id in seen_ids:
                reason = "vector point_id duplicated"
            if reason is not None:
                rejected_ids.append(point_id)
                rejection_reasons.append(reason)
                continue
            seen_ids.add(point_id)
            accepted.append(point)

        sorted_points = _sort_points(tuple(accepted))
        kept_points = sorted_points[: self._policy.max_points]
        overflow_points = sorted_points[self._policy.max_points :]
        for point in overflow_points:
            rejected_ids.append(str(point.point_id))
            rejection_reasons.append("vector point limit exceeded")

        return VectorPointSelection(
            points=kept_points,
            rejected_point_ids=tuple(rejected_ids),
            rejection_reasons=tuple(rejection_reasons),
        )

    def select_matches(
        self,
        *,
        query: VectorSearchQuery,
        matches: tuple[VectorSearchMatch, ...] | list[VectorSearchMatch],
    ) -> VectorMatchSelection:
        """Select vector search matches accepted for a query."""

        if not isinstance(query, VectorSearchQuery):
            raise TypeError("query must be VectorSearchQuery")
        match_tuple = _ensure_match_tuple(matches)
        accepted: list[VectorSearchMatch] = []
        rejected_ids: list[str] = []
        rejection_reasons: list[str] = []
        seen_ids: set[str] = set()
        min_score = max(query.min_score, self._policy.min_match_score)

        for match in match_tuple:
            point_id = str(match.point.point_id)
            reason = self._match_rejection_reason(
                query=query,
                match=match,
                min_score=min_score,
            )
            if reason is None and point_id in seen_ids:
                reason = "vector match point_id duplicated"
            if reason is not None:
                rejected_ids.append(point_id)
                rejection_reasons.append(reason)
                continue
            seen_ids.add(point_id)
            accepted.append(match)

        sorted_matches = _sort_matches(tuple(accepted))
        reranked_matches = tuple(
            VectorSearchMatch(
                point=match.point,
                score=match.score,
                rank=index,
                metadata=match.metadata,
            )
            for index, match in enumerate(sorted_matches, start=1)
        )

        return VectorMatchSelection(
            matches=reranked_matches,
            rejected_point_ids=tuple(rejected_ids),
            rejection_reasons=tuple(rejection_reasons),
        )

    def create_snapshot(
        self,
        *,
        config: VectorIndexConfig,
        points: tuple[VectorPointRef, ...] | list[VectorPointRef],
        status: VectorIndexStatus | None = None,
        metadata: dict[str, object] | None = None,
    ) -> VectorIndexSnapshot:
        """Create a deterministic vector index snapshot from accepted points."""

        if not isinstance(config, VectorIndexConfig):
            raise TypeError("config must be VectorIndexConfig")
        selection = self.select_points(config=config, points=points)
        snapshot_status = config.status if status is None else status
        if not isinstance(snapshot_status, VectorIndexStatus):
            raise TypeError("status must be None or VectorIndexStatus")
        required_metadata = {
            "source": "vector_index_policy_manager",
            "operation": "create_snapshot",
            "backend": config.backend.value,
            "collection_name": config.collection_name,
            "dimension": config.dimension,
            "accepted_point_count": selection.point_count(),
            "rejected_point_count": selection.rejected_count(),
            "policy": self._policy.to_dict(),
        }
        return create_vector_index_snapshot(
            config=config,
            points=selection.points,
            status=snapshot_status,
            metadata=_merge_metadata(required_metadata, metadata),
        )

    def _point_rejection_reason(
        self,
        *,
        config: VectorIndexConfig,
        point: VectorPointRef,
    ) -> str | None:
        if self._policy.require_dimension_match and point.dimension() != config.dimension:
            return "vector point dimension mismatch"
        if point.status is VectorPointStatus.STALE and not self._policy.allow_stale_points:
            return "vector point stale"
        if (
            point.status is VectorPointStatus.DELETED
            and not self._policy.allow_deleted_points
        ):
            return "vector point deleted"
        if (
            point.status is VectorPointStatus.INVALID
            and not self._policy.allow_invalid_points
        ):
            return "vector point invalid"
        return None

    def _match_rejection_reason(
        self,
        *,
        query: VectorSearchQuery,
        match: VectorSearchMatch,
        min_score: float,
    ) -> str | None:
        if match.score < min_score:
            return "vector match score below minimum"
        if query.payload_kind is not None and (
            match.point.payload.payload_kind is not query.payload_kind
        ):
            return "vector match payload kind mismatch"
        if query.dimension() != match.point.dimension():
            return "vector match dimension mismatch"
        return None


def select_vector_points_for_index(
    *,
    config: VectorIndexConfig,
    points: tuple[VectorPointRef, ...] | list[VectorPointRef],
    policy: VectorIndexPolicy | None = None,
) -> VectorPointSelection:
    """Select vector points accepted for an index."""

    manager = VectorIndexPolicyManager(policy=policy)
    return manager.select_points(config=config, points=points)


def select_vector_matches(
    *,
    query: VectorSearchQuery,
    matches: tuple[VectorSearchMatch, ...] | list[VectorSearchMatch],
    policy: VectorIndexPolicy | None = None,
) -> VectorMatchSelection:
    """Select vector matches accepted for a query."""

    manager = VectorIndexPolicyManager(policy=policy)
    return manager.select_matches(query=query, matches=matches)


def snapshot_vector_index(
    *,
    config: VectorIndexConfig,
    points: tuple[VectorPointRef, ...] | list[VectorPointRef],
    status: VectorIndexStatus | None = None,
    policy: VectorIndexPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> VectorIndexSnapshot:
    """Create a deterministic vector index snapshot."""

    manager = VectorIndexPolicyManager(policy=policy)
    return manager.create_snapshot(
        config=config,
        points=points,
        status=status,
        metadata=metadata,
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


def _ensure_string_tuple(
    value: tuple[str, ...] | list[str],
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple[str, ...]")
    return tuple(ensure_non_empty_string(item, field_name) for item in value)


def _sort_points(points: tuple[VectorPointRef, ...]) -> tuple[VectorPointRef, ...]:
    return tuple(
        sorted(
            points,
            key=lambda point: (
                -point.updated_at.timestamp(),
                -point.created_at.timestamp(),
                str(point.point_id),
            ),
        )
    )


def _sort_matches(
    matches: tuple[VectorSearchMatch, ...],
) -> tuple[VectorSearchMatch, ...]:
    return tuple(
        sorted(
            matches,
            key=lambda match: (
                -match.score,
                match.rank,
                -match.point.updated_at.timestamp(),
                str(match.point.point_id),
            ),
        )
    )


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
