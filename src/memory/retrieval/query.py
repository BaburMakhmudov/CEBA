"""Deterministic retrieval query construction utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

from ceba.memory.retrieval.schema import (
    RetrievalQuery,
    RetrievalQueryKind,
    RetrievalSourceKind,
    create_retrieval_query,
    ensure_non_empty_string,
    ensure_positive_int,
    ensure_probability,
)


__all__ = [
    "RetrievalQueryPolicy",
    "RetrievalQueryInput",
    "RetrievalQueryResult",
    "RetrievalQueryBuilder",
    "build_retrieval_query",
    "build_retrieval_queries",
]


@dataclass(frozen=True)
class RetrievalQueryPolicy:
    """Configuration object controlling retrieval query construction."""

    default_kind: RetrievalQueryKind = RetrievalQueryKind.UNKNOWN
    default_source_kind: RetrievalSourceKind = RetrievalSourceKind.UNKNOWN
    default_limit: int = 10
    max_limit: int = 100
    default_min_score: float = 0.0
    require_query_ref: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.default_kind, RetrievalQueryKind):
            raise TypeError("default_kind must be RetrievalQueryKind")
        if not isinstance(self.default_source_kind, RetrievalSourceKind):
            raise TypeError("default_source_kind must be RetrievalSourceKind")
        default_limit = ensure_positive_int(self.default_limit, "default_limit")
        max_limit = ensure_positive_int(self.max_limit, "max_limit")
        if max_limit < default_limit:
            raise ValueError("max_limit must be >= default_limit")
        default_min_score = ensure_probability(
            self.default_min_score,
            "default_min_score",
        )
        if not isinstance(self.require_query_ref, bool):
            raise TypeError("require_query_ref must be bool")
        object.__setattr__(self, "default_limit", default_limit)
        object.__setattr__(self, "max_limit", max_limit)
        object.__setattr__(self, "default_min_score", default_min_score)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "default_kind": self.default_kind.value,
            "default_source_kind": self.default_source_kind.value,
            "default_limit": self.default_limit,
            "max_limit": self.max_limit,
            "default_min_score": self.default_min_score,
            "require_query_ref": self.require_query_ref,
        }


@dataclass(frozen=True)
class RetrievalQueryInput:
    """Explicit input object for creating one RetrievalQuery."""

    query_ref: str
    kind: RetrievalQueryKind | None = None
    source_kind: RetrievalSourceKind | None = None
    limit: int | None = None
    min_score: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        query_ref = ensure_non_empty_string(self.query_ref, "query_ref")
        if self.kind is not None and not isinstance(
            self.kind,
            RetrievalQueryKind,
        ):
            raise TypeError("kind must be RetrievalQueryKind or None")
        if self.source_kind is not None and not isinstance(
            self.source_kind,
            RetrievalSourceKind,
        ):
            raise TypeError("source_kind must be RetrievalSourceKind or None")
        limit = (
            None
            if self.limit is None
            else ensure_positive_int(self.limit, "limit")
        )
        min_score = (
            None
            if self.min_score is None
            else ensure_probability(self.min_score, "min_score")
        )
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "query_ref", query_ref)
        object.__setattr__(self, "limit", limit)
        object.__setattr__(self, "min_score", min_score)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "query_ref": self.query_ref,
            "kind": None if self.kind is None else self.kind.value,
            "source_kind": (
                None if self.source_kind is None else self.source_kind.value
            ),
            "limit": self.limit,
            "min_score": self.min_score,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RetrievalQueryResult:
    """Immutable result object returned by retrieval query construction."""

    queries: tuple[RetrievalQuery, ...]
    rejected_inputs: tuple[str, ...]
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        queries = _ensure_queries(self.queries)
        rejected_inputs = _ensure_string_tuple(
            self.rejected_inputs,
            "rejected_inputs",
        )
        rejection_reasons = _ensure_string_tuple(
            self.rejection_reasons,
            "rejection_reasons",
        )
        if len(rejected_inputs) != len(rejection_reasons):
            raise ValueError("rejected_inputs and rejection_reasons must match")
        object.__setattr__(self, "queries", queries)
        object.__setattr__(self, "rejected_inputs", rejected_inputs)
        object.__setattr__(self, "rejection_reasons", rejection_reasons)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "queries": tuple(query.to_dict() for query in self.queries),
            "rejected_inputs": self.rejected_inputs,
            "rejection_reasons": self.rejection_reasons,
        }

    def query_count(self) -> int:
        """Return query count."""

        return len(self.queries)

    def rejected_count(self) -> int:
        """Return rejected input count."""

        return len(self.rejected_inputs)


class RetrievalQueryBuilder:
    """Deterministic service that builds RetrievalQuery objects."""

    def __init__(self, policy: RetrievalQueryPolicy | None = None) -> None:
        self._policy = RetrievalQueryPolicy() if policy is None else policy
        if not isinstance(self._policy, RetrievalQueryPolicy):
            raise TypeError("policy must be RetrievalQueryPolicy")

    def build_one(
        self,
        *,
        query_input: RetrievalQueryInput,
    ) -> RetrievalQueryResult:
        """Build one RetrievalQuery from explicit input."""

        if not isinstance(query_input, RetrievalQueryInput):
            raise TypeError("query_input must be RetrievalQueryInput")
        rejection_reason = self._rejection_reason(query_input)
        if rejection_reason is not None:
            return RetrievalQueryResult(
                queries=(),
                rejected_inputs=(query_input.query_ref,),
                rejection_reasons=(rejection_reason,),
            )

        kind = query_input.kind or self._policy.default_kind
        source_kind = (
            query_input.source_kind or self._policy.default_source_kind
        )
        requested_limit = (
            query_input.limit
            if query_input.limit is not None
            else self._policy.default_limit
        )
        effective_limit = min(requested_limit, self._policy.max_limit)
        min_score = (
            query_input.min_score
            if query_input.min_score is not None
            else self._policy.default_min_score
        )
        metadata = self._metadata_for_input(
            query_input=query_input,
            requested_limit=requested_limit,
            effective_limit=effective_limit,
        )
        query = create_retrieval_query(
            kind=kind,
            source_kind=source_kind,
            query_ref=query_input.query_ref,
            limit=effective_limit,
            min_score=min_score,
            metadata=metadata,
        )
        return RetrievalQueryResult(
            queries=(query,),
            rejected_inputs=(),
            rejection_reasons=(),
        )

    def build_many(
        self,
        *,
        inputs: tuple[RetrievalQueryInput, ...] | list[RetrievalQueryInput],
    ) -> RetrievalQueryResult:
        """Build many RetrievalQuery objects in input order."""

        input_values = _ensure_inputs(inputs)
        queries: list[RetrievalQuery] = []
        rejected_inputs: list[str] = []
        rejection_reasons: list[str] = []
        for query_input in input_values:
            result = self.build_one(query_input=query_input)
            queries.extend(result.queries)
            rejected_inputs.extend(result.rejected_inputs)
            rejection_reasons.extend(result.rejection_reasons)
        return RetrievalQueryResult(
            queries=tuple(queries),
            rejected_inputs=tuple(rejected_inputs),
            rejection_reasons=tuple(rejection_reasons),
        )

    def _rejection_reason(self, query_input: RetrievalQueryInput) -> str | None:
        if self._policy.require_query_ref and query_input.query_ref == "":
            return "query_ref required"
        if query_input.limit is not None and query_input.limit <= 0:
            return "limit must be positive"
        if query_input.min_score is not None:
            if query_input.min_score < 0.0 or query_input.min_score > 1.0:
                return "min_score must be probability"
        return None

    def _metadata_for_input(
        self,
        *,
        query_input: RetrievalQueryInput,
        requested_limit: int,
        effective_limit: int,
    ) -> dict[str, object]:
        metadata = _copy_metadata(query_input.metadata)
        metadata.update(
            {
                "source": "retrieval_query_builder",
                "policy": self._policy.to_dict(),
                "requested_limit": requested_limit,
                "effective_limit": effective_limit,
            }
        )
        metadata["source"] = "retrieval_query_builder"
        return metadata


def build_retrieval_query(
    *,
    query_input: RetrievalQueryInput,
    policy: RetrievalQueryPolicy | None = None,
) -> RetrievalQueryResult:
    """Build one RetrievalQuery from explicit input."""

    builder = RetrievalQueryBuilder(policy)
    return builder.build_one(query_input=query_input)


def build_retrieval_queries(
    *,
    inputs: tuple[RetrievalQueryInput, ...] | list[RetrievalQueryInput],
    policy: RetrievalQueryPolicy | None = None,
) -> RetrievalQueryResult:
    """Build many RetrievalQuery objects."""

    builder = RetrievalQueryBuilder(policy)
    return builder.build_many(inputs=inputs)


def _ensure_queries(
    values: tuple[RetrievalQuery, ...] | list[RetrievalQuery],
) -> tuple[RetrievalQuery, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("queries must be tuple or list")
    queries = tuple(values)
    for query in queries:
        if not isinstance(query, RetrievalQuery):
            raise TypeError("queries must contain RetrievalQuery")
    query_ids = tuple(query.query_id for query in queries)
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("query_id values must be unique")
    return queries


def _ensure_inputs(
    values: tuple[RetrievalQueryInput, ...] | list[RetrievalQueryInput],
) -> tuple[RetrievalQueryInput, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("inputs must be tuple or list")
    inputs = tuple(values)
    for value in inputs:
        if not isinstance(value, RetrievalQueryInput):
            raise TypeError("inputs must contain RetrievalQueryInput")
    return inputs


def _ensure_string_tuple(
    values: tuple[str, ...] | list[str],
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple or list")
    return tuple(ensure_non_empty_string(value, field_name) for value in values)


def _copy_metadata(metadata: dict[str, object]) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(metadata)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
    return copied
