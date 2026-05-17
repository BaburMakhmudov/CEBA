"""Deterministic hybrid retrieval fusion utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

from ceba.memory.retrieval.schema import (
    RankingSignal,
    RankingSignalKind,
    RetrievalCandidate,
    RetrievalQuery,
    RetrievalSourceKind,
    RetrievalStatus,
    create_ranking_signal,
    ensure_non_empty_string,
    ensure_non_negative_number,
    ensure_probability,
)


__all__ = [
    "HybridRetrievalPolicy",
    "HybridRetrievalInput",
    "HybridRetrievalResult",
    "HybridRetrievalFusion",
    "fuse_retrieval_candidates",
]


@dataclass(frozen=True)
class HybridRetrievalPolicy:
    """Configuration object controlling hybrid candidate fusion."""

    min_score: float = 0.0
    max_candidates: int = 100
    vector_weight: float = 0.50
    graph_weight: float = 0.30
    confidence_weight: float = 0.20
    deduplicate_by_source_id: bool = True
    include_hybrid_signal: bool = True

    def __post_init__(self) -> None:
        min_score = ensure_probability(self.min_score, "min_score")
        if isinstance(self.max_candidates, bool) or not isinstance(
            self.max_candidates,
            int,
        ):
            raise TypeError("max_candidates must be int")
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        vector_weight = ensure_non_negative_number(
            self.vector_weight,
            "vector_weight",
        )
        graph_weight = ensure_non_negative_number(
            self.graph_weight,
            "graph_weight",
        )
        confidence_weight = ensure_non_negative_number(
            self.confidence_weight,
            "confidence_weight",
        )
        if vector_weight + graph_weight + confidence_weight <= 0.0:
            raise ValueError("hybrid weights must sum to > 0.0")
        if not isinstance(self.deduplicate_by_source_id, bool):
            raise TypeError("deduplicate_by_source_id must be bool")
        if not isinstance(self.include_hybrid_signal, bool):
            raise TypeError("include_hybrid_signal must be bool")
        object.__setattr__(self, "min_score", min_score)
        object.__setattr__(self, "vector_weight", vector_weight)
        object.__setattr__(self, "graph_weight", graph_weight)
        object.__setattr__(self, "confidence_weight", confidence_weight)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "min_score": self.min_score,
            "max_candidates": self.max_candidates,
            "vector_weight": self.vector_weight,
            "graph_weight": self.graph_weight,
            "confidence_weight": self.confidence_weight,
            "deduplicate_by_source_id": self.deduplicate_by_source_id,
            "include_hybrid_signal": self.include_hybrid_signal,
        }


@dataclass(frozen=True)
class HybridRetrievalInput:
    """Explicit input object for hybrid fusion."""

    query: RetrievalQuery
    vector_candidates: tuple[RetrievalCandidate, ...]
    graph_candidates: tuple[RetrievalCandidate, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.query, RetrievalQuery):
            raise TypeError("query must be RetrievalQuery")
        vector_candidates = _ensure_candidates(
            self.vector_candidates,
            "vector_candidates",
        )
        graph_candidates = _ensure_candidates(
            self.graph_candidates,
            "graph_candidates",
        )
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "vector_candidates", vector_candidates)
        object.__setattr__(self, "graph_candidates", graph_candidates)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "query": self.query.to_dict(),
            "vector_candidates": tuple(
                candidate.to_dict() for candidate in self.vector_candidates
            ),
            "graph_candidates": tuple(
                candidate.to_dict() for candidate in self.graph_candidates
            ),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class HybridRetrievalResult:
    """Immutable result object returned by hybrid fusion."""

    candidates: tuple[RetrievalCandidate, ...]
    skipped_candidate_ids: tuple[str, ...]
    skip_reasons: tuple[str, ...]
    status: RetrievalStatus

    def __post_init__(self) -> None:
        candidates = _ensure_candidates(self.candidates, "candidates")
        skipped_ids = _ensure_string_tuple(
            self.skipped_candidate_ids,
            "skipped_candidate_ids",
        )
        skip_reasons = _ensure_string_tuple(
            self.skip_reasons,
            "skip_reasons",
        )
        if len(skipped_ids) != len(skip_reasons):
            raise ValueError("skipped_candidate_ids and skip_reasons must match")
        if not isinstance(self.status, RetrievalStatus):
            raise TypeError("status must be RetrievalStatus")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "skipped_candidate_ids", skipped_ids)
        object.__setattr__(self, "skip_reasons", skip_reasons)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "candidates": tuple(
                candidate.to_dict() for candidate in self.candidates
            ),
            "skipped_candidate_ids": self.skipped_candidate_ids,
            "skip_reasons": self.skip_reasons,
            "status": self.status.value,
        }

    def candidate_count(self) -> int:
        """Return fused candidate count."""

        return len(self.candidates)

    def skipped_count(self) -> int:
        """Return skipped candidate count."""

        return len(self.skipped_candidate_ids)


class HybridRetrievalFusion:
    """Deterministic service class that fuses retrieval candidates."""

    def __init__(self, policy: HybridRetrievalPolicy | None = None) -> None:
        self._policy = HybridRetrievalPolicy() if policy is None else policy
        if not isinstance(self._policy, HybridRetrievalPolicy):
            raise TypeError("policy must be HybridRetrievalPolicy")

    def fuse(
        self,
        *,
        hybrid_input: HybridRetrievalInput,
    ) -> HybridRetrievalResult:
        """Fuse vector and graph retrieval candidates."""

        if not isinstance(hybrid_input, HybridRetrievalInput):
            raise TypeError("hybrid_input must be HybridRetrievalInput")
        groups = self._build_groups(hybrid_input)
        accepted: list[RetrievalCandidate] = []
        skipped_ids: list[str] = []
        skip_reasons: list[str] = []
        for group in groups:
            candidate = self._fuse_group(
                group=group,
                hybrid_input=hybrid_input,
            )
            if candidate.score < self._policy.min_score:
                skipped_ids.append(str(candidate.candidate_id))
                skip_reasons.append("score below hybrid retrieval minimum")
            else:
                accepted.append(candidate)

        accepted.sort(
            key=lambda candidate: (
                -candidate.score,
                -candidate.confidence,
                -candidate.retrieved_at.timestamp(),
                str(candidate.candidate_id),
            )
        )
        candidates = accepted[: self._policy.max_candidates]
        overflow = accepted[self._policy.max_candidates :]
        for candidate in overflow:
            skipped_ids.append(str(candidate.candidate_id))
            skip_reasons.append("hybrid retrieval limit exceeded")
        status = RetrievalStatus.RETRIEVED if candidates else RetrievalStatus.EMPTY
        return HybridRetrievalResult(
            candidates=tuple(candidates),
            skipped_candidate_ids=tuple(skipped_ids),
            skip_reasons=tuple(skip_reasons),
            status=status,
        )

    def _build_groups(
        self,
        hybrid_input: HybridRetrievalInput,
    ) -> tuple[tuple[tuple[RetrievalCandidate, str], ...], ...]:
        ordered: list[tuple[RetrievalCandidate, str]] = []
        ordered.extend((candidate, "vector") for candidate in (
            hybrid_input.vector_candidates
        ))
        ordered.extend((candidate, "graph") for candidate in (
            hybrid_input.graph_candidates
        ))
        if not self._policy.deduplicate_by_source_id:
            return tuple((entry,) for entry in ordered)

        groups: dict[str, list[tuple[RetrievalCandidate, str]]] = {}
        order: list[str] = []
        for candidate, origin in ordered:
            key = candidate.source_id
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append((candidate, origin))
        return tuple(tuple(groups[key]) for key in order)

    def _fuse_group(
        self,
        *,
        group: tuple[tuple[RetrievalCandidate, str], ...],
        hybrid_input: HybridRetrievalInput,
    ) -> RetrievalCandidate:
        vector_score = max(
            (
                candidate.score
                for candidate, origin in group
                if (
                    origin == "vector"
                    or candidate.source_kind is RetrievalSourceKind.VECTOR
                )
            ),
            default=0.0,
        )
        graph_score = max(
            (
                candidate.score
                for candidate, origin in group
                if (
                    origin == "graph"
                    or candidate.source_kind is RetrievalSourceKind.GRAPH
                )
            ),
            default=0.0,
        )
        confidence = max(candidate.confidence for candidate, _ in group)
        representative = _representative_candidate(group)
        retrieved_at = max(candidate.retrieved_at for candidate, _ in group)
        hybrid_score = self._hybrid_score(
            vector_score=vector_score,
            graph_score=graph_score,
            confidence=confidence,
        )
        signals = self._ranking_signals(
            group=group,
            representative=representative,
            hybrid_score=hybrid_score,
        )
        metadata = _copy_metadata(representative.metadata)
        metadata.update(_copy_metadata(hybrid_input.metadata))
        metadata.update(
            {
                "source": "hybrid_retrieval_fusion",
                "query_id": str(hybrid_input.query.query_id),
                "query_kind": hybrid_input.query.kind.value,
                "vector_score": vector_score,
                "graph_score": graph_score,
                "hybrid_score": hybrid_score,
                "group_size": len(group),
                "policy": self._policy.to_dict(),
            }
        )
        metadata["source"] = "hybrid_retrieval_fusion"
        return RetrievalCandidate(
            candidate_id=representative.candidate_id,
            kind=representative.kind,
            source_kind=RetrievalSourceKind.HYBRID,
            source_id=representative.source_id,
            content_ref=representative.content_ref,
            score=hybrid_score,
            confidence=confidence,
            ranking_signals=signals,
            retrieved_at=retrieved_at,
            metadata=metadata,
        )

    def _hybrid_score(
        self,
        *,
        vector_score: float,
        graph_score: float,
        confidence: float,
    ) -> float:
        numerator = (
            vector_score * self._policy.vector_weight
            + graph_score * self._policy.graph_weight
            + confidence * self._policy.confidence_weight
        )
        denominator = (
            self._policy.vector_weight
            + self._policy.graph_weight
            + self._policy.confidence_weight
        )
        return _clamp_probability(numerator / denominator)

    def _ranking_signals(
        self,
        *,
        group: tuple[tuple[RetrievalCandidate, str], ...],
        representative: RetrievalCandidate,
        hybrid_score: float,
    ) -> tuple[RankingSignal, ...]:
        ordered_candidates = [representative]
        ordered_candidates.extend(
            candidate
            for candidate, _ in group
            if candidate.candidate_id != representative.candidate_id
        )
        signals: list[RankingSignal] = []
        seen_ids: set[object] = set()
        for candidate in ordered_candidates:
            for signal in candidate.ranking_signals:
                if signal.signal_id in seen_ids:
                    continue
                signals.append(signal)
                seen_ids.add(signal.signal_id)
        if self._policy.include_hybrid_signal:
            signal = create_ranking_signal(
                kind=RankingSignalKind.HYBRID_SCORE,
                score=hybrid_score,
                weight=1.0,
                source_id=representative.source_id,
                metadata={"source": "hybrid_retrieval_fusion"},
            )
            if signal.signal_id not in seen_ids:
                signals.append(signal)
        return tuple(signals)


def fuse_retrieval_candidates(
    *,
    hybrid_input: HybridRetrievalInput,
    policy: HybridRetrievalPolicy | None = None,
) -> HybridRetrievalResult:
    """Fuse vector and graph retrieval candidates."""

    fusion = HybridRetrievalFusion(policy)
    return fusion.fuse(hybrid_input=hybrid_input)


def _representative_candidate(
    group: tuple[tuple[RetrievalCandidate, str], ...],
) -> RetrievalCandidate:
    return sorted(
        (candidate for candidate, _ in group),
        key=lambda candidate: (
            -candidate.score,
            -candidate.confidence,
            -candidate.retrieved_at.timestamp(),
            str(candidate.candidate_id),
        ),
    )[0]


def _ensure_candidates(
    values: tuple[RetrievalCandidate, ...] | list[RetrievalCandidate],
    field_name: str,
) -> tuple[RetrievalCandidate, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple or list")
    candidates = tuple(values)
    for candidate in candidates:
        if not isinstance(candidate, RetrievalCandidate):
            raise TypeError(f"{field_name} must contain RetrievalCandidate")
    candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError(f"{field_name} candidate_id values must be unique")
    return candidates


def _ensure_string_tuple(
    values: tuple[str, ...] | list[str],
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple or list")
    return tuple(ensure_non_empty_string(value, field_name) for value in values)


def _clamp_probability(value: float) -> float:
    number = ensure_non_negative_number(value, "value")
    if number > 1.0:
        return 1.0
    return number


def _copy_metadata(metadata: dict[str, object]) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(metadata)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
    return copied
