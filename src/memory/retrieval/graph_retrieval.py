"""Deterministic graph retrieval result adaptation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

from ceba.memory.retrieval.schema import (
    RankingSignal,
    RankingSignalKind,
    RetrievalCandidate,
    RetrievalCandidateKind,
    RetrievalQuery,
    RetrievalSourceKind,
    RetrievalStatus,
    create_ranking_signal,
    create_retrieval_candidate,
    ensure_non_empty_string,
    ensure_non_negative_number,
    ensure_probability,
)


__all__ = [
    "GraphRetrievalPolicy",
    "GraphRetrievalHit",
    "GraphRetrievalResult",
    "GraphRetrievalAdapter",
    "adapt_graph_hits",
]


@dataclass(frozen=True)
class GraphRetrievalPolicy:
    """Configuration object controlling deterministic graph hit adaptation."""

    min_score: float = 0.0
    max_candidates: int = 100
    default_candidate_kind: RetrievalCandidateKind = RetrievalCandidateKind.CONCEPT
    default_source_kind: RetrievalSourceKind = RetrievalSourceKind.GRAPH
    include_structural_distance_signal: bool = True
    structural_distance_weight: float = 1.0

    def __post_init__(self) -> None:
        min_score = ensure_probability(self.min_score, "min_score")
        if isinstance(self.max_candidates, bool) or not isinstance(
            self.max_candidates,
            int,
        ):
            raise TypeError("max_candidates must be int")
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if not isinstance(self.default_candidate_kind, RetrievalCandidateKind):
            raise TypeError(
                "default_candidate_kind must be RetrievalCandidateKind"
            )
        if not isinstance(self.default_source_kind, RetrievalSourceKind):
            raise TypeError("default_source_kind must be RetrievalSourceKind")
        if not isinstance(self.include_structural_distance_signal, bool):
            raise TypeError("include_structural_distance_signal must be bool")
        distance_weight = ensure_non_negative_number(
            self.structural_distance_weight,
            "structural_distance_weight",
        )
        object.__setattr__(self, "min_score", min_score)
        object.__setattr__(
            self,
            "structural_distance_weight",
            distance_weight,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "min_score": self.min_score,
            "max_candidates": self.max_candidates,
            "default_candidate_kind": self.default_candidate_kind.value,
            "default_source_kind": self.default_source_kind.value,
            "include_structural_distance_signal": (
                self.include_structural_distance_signal
            ),
            "structural_distance_weight": self.structural_distance_weight,
        }


@dataclass(frozen=True)
class GraphRetrievalHit:
    """Explicit graph-search hit returned by an external graph surface."""

    source_id: str
    score: float
    content_ref: str | None = None
    candidate_kind: RetrievalCandidateKind | None = None
    structural_distance: float | None = None
    confidence: float = 1.0
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source_id = ensure_non_empty_string(self.source_id, "source_id")
        score = ensure_probability(self.score, "score")
        content_ref = (
            None
            if self.content_ref is None
            else ensure_non_empty_string(self.content_ref, "content_ref")
        )
        if self.candidate_kind is not None and not isinstance(
            self.candidate_kind,
            RetrievalCandidateKind,
        ):
            raise TypeError(
                "candidate_kind must be RetrievalCandidateKind or None"
            )
        structural_distance = (
            None
            if self.structural_distance is None
            else ensure_non_negative_number(
                self.structural_distance,
                "structural_distance",
            )
        )
        confidence = ensure_probability(self.confidence, "confidence")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "content_ref", content_ref)
        object.__setattr__(self, "structural_distance", structural_distance)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "source_id": self.source_id,
            "score": self.score,
            "content_ref": self.content_ref,
            "candidate_kind": (
                None
                if self.candidate_kind is None
                else self.candidate_kind.value
            ),
            "structural_distance": self.structural_distance,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GraphRetrievalResult:
    """Immutable result object returned by graph hit adaptation."""

    candidates: tuple[RetrievalCandidate, ...]
    skipped_source_ids: tuple[str, ...]
    skip_reasons: tuple[str, ...]
    status: RetrievalStatus

    def __post_init__(self) -> None:
        candidates = _ensure_candidates(self.candidates)
        skipped_source_ids = _ensure_string_tuple(
            self.skipped_source_ids,
            "skipped_source_ids",
        )
        skip_reasons = _ensure_string_tuple(
            self.skip_reasons,
            "skip_reasons",
        )
        if len(skipped_source_ids) != len(skip_reasons):
            raise ValueError("skipped_source_ids and skip_reasons must match")
        if not isinstance(self.status, RetrievalStatus):
            raise TypeError("status must be RetrievalStatus")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "skipped_source_ids", skipped_source_ids)
        object.__setattr__(self, "skip_reasons", skip_reasons)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "candidates": tuple(
                candidate.to_dict() for candidate in self.candidates
            ),
            "skipped_source_ids": self.skipped_source_ids,
            "skip_reasons": self.skip_reasons,
            "status": self.status.value,
        }

    def candidate_count(self) -> int:
        """Return candidate count."""

        return len(self.candidates)

    def skipped_count(self) -> int:
        """Return skipped source count."""

        return len(self.skipped_source_ids)


class GraphRetrievalAdapter:
    """Deterministic service converting graph hits to retrieval candidates."""

    def __init__(self, policy: GraphRetrievalPolicy | None = None) -> None:
        self._policy = GraphRetrievalPolicy() if policy is None else policy
        if not isinstance(self._policy, GraphRetrievalPolicy):
            raise TypeError("policy must be GraphRetrievalPolicy")

    def adapt(
        self,
        *,
        query: RetrievalQuery,
        hits: tuple[GraphRetrievalHit, ...] | list[GraphRetrievalHit],
    ) -> GraphRetrievalResult:
        """Adapt explicit graph hits into RetrievalCandidate objects."""

        if not isinstance(query, RetrievalQuery):
            raise TypeError("query must be RetrievalQuery")
        hit_values = _ensure_hits(hits)
        candidates: list[RetrievalCandidate] = []
        skipped_source_ids: list[str] = []
        skip_reasons: list[str] = []
        for hit in hit_values:
            if hit.score < self._policy.min_score:
                skipped_source_ids.append(hit.source_id)
                skip_reasons.append("score below graph retrieval minimum")
                continue
            if len(candidates) >= self._policy.max_candidates:
                skipped_source_ids.append(hit.source_id)
                skip_reasons.append("graph retrieval limit exceeded")
                continue
            candidates.append(self._candidate_from_hit(query=query, hit=hit))

        status = RetrievalStatus.RETRIEVED if candidates else RetrievalStatus.EMPTY
        return GraphRetrievalResult(
            candidates=tuple(candidates),
            skipped_source_ids=tuple(skipped_source_ids),
            skip_reasons=tuple(skip_reasons),
            status=status,
        )

    def _candidate_from_hit(
        self,
        *,
        query: RetrievalQuery,
        hit: GraphRetrievalHit,
    ) -> RetrievalCandidate:
        signals: tuple[RankingSignal, ...] = ()
        if (
            self._policy.include_structural_distance_signal
            and hit.structural_distance is not None
        ):
            signals = (
                create_ranking_signal(
                    kind=RankingSignalKind.STRUCTURAL_DISTANCE,
                    score=1.0 / (1.0 + hit.structural_distance),
                    weight=self._policy.structural_distance_weight,
                    source_id=hit.source_id,
                    metadata={
                        "source": "graph_retrieval_adapter",
                        "query_id": str(query.query_id),
                    },
                ),
            )
        metadata = _copy_metadata(hit.metadata)
        metadata.update(
            {
                "source": "graph_retrieval_adapter",
                "query_id": str(query.query_id),
                "query_kind": query.kind.value,
                "query_source_kind": query.source_kind.value,
                "hit_source_id": hit.source_id,
                "structural_distance": hit.structural_distance,
                "policy": self._policy.to_dict(),
            }
        )
        metadata["source"] = "graph_retrieval_adapter"
        return create_retrieval_candidate(
            kind=hit.candidate_kind or self._policy.default_candidate_kind,
            source_kind=self._policy.default_source_kind,
            source_id=hit.source_id,
            content_ref=hit.content_ref,
            score=hit.score,
            confidence=hit.confidence,
            ranking_signals=signals,
            metadata=metadata,
        )


def adapt_graph_hits(
    *,
    query: RetrievalQuery,
    hits: tuple[GraphRetrievalHit, ...] | list[GraphRetrievalHit],
    policy: GraphRetrievalPolicy | None = None,
) -> GraphRetrievalResult:
    """Adapt explicit graph hits into RetrievalCandidate objects."""

    adapter = GraphRetrievalAdapter(policy)
    return adapter.adapt(query=query, hits=hits)


def _ensure_hits(
    values: tuple[GraphRetrievalHit, ...] | list[GraphRetrievalHit],
) -> tuple[GraphRetrievalHit, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("hits must be tuple or list")
    hits = tuple(values)
    for hit in hits:
        if not isinstance(hit, GraphRetrievalHit):
            raise TypeError("hits must contain GraphRetrievalHit")
    return hits


def _ensure_candidates(
    values: tuple[RetrievalCandidate, ...] | list[RetrievalCandidate],
) -> tuple[RetrievalCandidate, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("candidates must be tuple or list")
    candidates = tuple(values)
    for candidate in candidates:
        if not isinstance(candidate, RetrievalCandidate):
            raise TypeError("candidates must contain RetrievalCandidate")
    candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate_id values must be unique")
    return candidates


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
