"""Deterministic ranking utilities for CEBA Memory Retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field

from ceba.memory.retrieval.schema import (
    RankingSignal,
    RetrievalCandidate,
    RetrievalStatus,
    ensure_non_empty_string,
    ensure_non_negative_number,
    ensure_probability,
)


__all__ = [
    "RankingPolicy",
    "RankingInput",
    "RankingResult",
    "RetrievalRanker",
    "rank_retrieval_candidates",
]


@dataclass(frozen=True)
class RankingPolicy:
    """Configuration object controlling deterministic candidate ranking."""

    min_score: float = 0.0
    max_candidates: int = 100
    preserve_existing_score_weight: float = 0.30
    signal_score_weight: float = 0.50
    confidence_weight: float = 0.20
    normalize_signal_weights: bool = True

    def __post_init__(self) -> None:
        min_score = ensure_probability(self.min_score, "min_score")
        if isinstance(self.max_candidates, bool) or not isinstance(
            self.max_candidates,
            int,
        ):
            raise TypeError("max_candidates must be int")
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        preserve_weight = ensure_non_negative_number(
            self.preserve_existing_score_weight,
            "preserve_existing_score_weight",
        )
        signal_weight = ensure_non_negative_number(
            self.signal_score_weight,
            "signal_score_weight",
        )
        confidence_weight = ensure_non_negative_number(
            self.confidence_weight,
            "confidence_weight",
        )
        if preserve_weight + signal_weight + confidence_weight <= 0.0:
            raise ValueError("at least one ranking weight must be > 0.0")
        if not isinstance(self.normalize_signal_weights, bool):
            raise TypeError("normalize_signal_weights must be bool")
        object.__setattr__(self, "min_score", min_score)
        object.__setattr__(
            self,
            "preserve_existing_score_weight",
            preserve_weight,
        )
        object.__setattr__(self, "signal_score_weight", signal_weight)
        object.__setattr__(self, "confidence_weight", confidence_weight)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "min_score": self.min_score,
            "max_candidates": self.max_candidates,
            "preserve_existing_score_weight": (
                self.preserve_existing_score_weight
            ),
            "signal_score_weight": self.signal_score_weight,
            "confidence_weight": self.confidence_weight,
            "normalize_signal_weights": self.normalize_signal_weights,
        }


@dataclass(frozen=True)
class RankingInput:
    """Explicit input object for ranking retrieval candidates."""

    candidates: tuple[RetrievalCandidate, ...]
    additional_signals_by_candidate_id: (
        dict[str, tuple[RankingSignal, ...] | list[RankingSignal]]
    ) = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        candidates = _ensure_candidates(self.candidates)
        additional_signals = _ensure_signal_mapping(
            self.additional_signals_by_candidate_id,
        )
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(
            self,
            "additional_signals_by_candidate_id",
            additional_signals,
        )
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "candidates": tuple(
                candidate.to_dict() for candidate in self.candidates
            ),
            "additional_signals_by_candidate_id": {
                candidate_id: tuple(signal.to_dict() for signal in signals)
                for candidate_id, signals in (
                    self.additional_signals_by_candidate_id.items()
                )
            },
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RankingResult:
    """Immutable result object returned by ranking."""

    candidates: tuple[RetrievalCandidate, ...]
    filtered_candidate_ids: tuple[str, ...]
    filter_reasons: tuple[str, ...]
    status: RetrievalStatus

    def __post_init__(self) -> None:
        candidates = _ensure_candidates(self.candidates)
        filtered_ids = _ensure_string_tuple(
            self.filtered_candidate_ids,
            "filtered_candidate_ids",
        )
        reasons = _ensure_string_tuple(self.filter_reasons, "filter_reasons")
        if len(filtered_ids) != len(reasons):
            raise ValueError("filtered_candidate_ids and filter_reasons match")
        if not isinstance(self.status, RetrievalStatus):
            raise TypeError("status must be RetrievalStatus")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "filtered_candidate_ids", filtered_ids)
        object.__setattr__(self, "filter_reasons", reasons)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "candidates": tuple(
                candidate.to_dict() for candidate in self.candidates
            ),
            "filtered_candidate_ids": self.filtered_candidate_ids,
            "filter_reasons": self.filter_reasons,
            "status": self.status.value,
        }

    def candidate_count(self) -> int:
        """Return ranked candidate count."""

        return len(self.candidates)

    def filtered_count(self) -> int:
        """Return filtered candidate count."""

        return len(self.filtered_candidate_ids)


class RetrievalRanker:
    """Deterministic service class for ranking retrieval candidates."""

    def __init__(self, policy: RankingPolicy | None = None) -> None:
        self._policy = RankingPolicy() if policy is None else policy
        if not isinstance(self._policy, RankingPolicy):
            raise TypeError("policy must be RankingPolicy")

    def rank(
        self,
        *,
        ranking_input: RankingInput,
    ) -> RankingResult:
        """Rank retrieval candidates deterministically."""

        if not isinstance(ranking_input, RankingInput):
            raise TypeError("ranking_input must be RankingInput")
        accepted: list[RetrievalCandidate] = []
        filtered_ids: list[str] = []
        filter_reasons: list[str] = []
        for candidate in ranking_input.candidates:
            ranked = self._rank_candidate(
                candidate=candidate,
                additional_signals=(
                    ranking_input.additional_signals_by_candidate_id.get(
                        str(candidate.candidate_id),
                        (),
                    )
                ),
                caller_metadata=ranking_input.metadata,
            )
            if ranked.score < self._policy.min_score:
                filtered_ids.append(str(candidate.candidate_id))
                filter_reasons.append("score below ranking minimum")
            else:
                accepted.append(ranked)

        accepted.sort(
            key=lambda candidate: (
                -candidate.score,
                -candidate.confidence,
                -candidate.retrieved_at.timestamp(),
                str(candidate.candidate_id),
            )
        )
        ranked_candidates = accepted[: self._policy.max_candidates]
        overflow = accepted[self._policy.max_candidates :]
        for candidate in overflow:
            filtered_ids.append(str(candidate.candidate_id))
            filter_reasons.append("ranking limit exceeded")

        status = (
            RetrievalStatus.RANKED
            if ranked_candidates
            else RetrievalStatus.EMPTY
        )
        return RankingResult(
            candidates=tuple(ranked_candidates),
            filtered_candidate_ids=tuple(filtered_ids),
            filter_reasons=tuple(filter_reasons),
            status=status,
        )

    def _rank_candidate(
        self,
        *,
        candidate: RetrievalCandidate,
        additional_signals: tuple[RankingSignal, ...],
        caller_metadata: dict[str, object],
    ) -> RetrievalCandidate:
        signals = self._merge_signals(
            candidate.ranking_signals,
            additional_signals,
        )
        signal_score = self._signal_score(signals)
        final_score = self._final_score(
            candidate_score=candidate.score,
            signal_score=signal_score,
            confidence=candidate.confidence,
        )
        metadata = _copy_metadata(candidate.metadata)
        metadata.update(_copy_metadata(caller_metadata))
        metadata.update(
            {
                "source": "retrieval_ranker",
                "previous_score": candidate.score,
                "final_score": final_score,
                "signal_score": signal_score,
                "policy": self._policy.to_dict(),
            }
        )
        metadata["source"] = "retrieval_ranker"
        return RetrievalCandidate(
            candidate_id=candidate.candidate_id,
            kind=candidate.kind,
            source_kind=candidate.source_kind,
            source_id=candidate.source_id,
            content_ref=candidate.content_ref,
            score=final_score,
            confidence=candidate.confidence,
            ranking_signals=signals,
            retrieved_at=candidate.retrieved_at,
            metadata=metadata,
        )

    def _signal_score(self, signals: tuple[RankingSignal, ...]) -> float:
        if not signals:
            return 0.0
        weighted_sum = sum(signal.score * signal.weight for signal in signals)
        if self._policy.normalize_signal_weights:
            total_weight = sum(signal.weight for signal in signals)
            if total_weight == 0.0:
                return 0.0
            return ensure_probability(weighted_sum / total_weight, "signal_score")
        return _clamp_probability(weighted_sum)

    def _final_score(
        self,
        *,
        candidate_score: float,
        signal_score: float,
        confidence: float,
    ) -> float:
        numerator = (
            candidate_score * self._policy.preserve_existing_score_weight
            + signal_score * self._policy.signal_score_weight
            + confidence * self._policy.confidence_weight
        )
        denominator = (
            self._policy.preserve_existing_score_weight
            + self._policy.signal_score_weight
            + self._policy.confidence_weight
        )
        return _clamp_probability(numerator / denominator)

    def _merge_signals(
        self,
        existing: tuple[RankingSignal, ...],
        additional: tuple[RankingSignal, ...],
    ) -> tuple[RankingSignal, ...]:
        merged = existing + additional
        signal_ids = tuple(signal.signal_id for signal in merged)
        if len(set(signal_ids)) != len(signal_ids):
            raise ValueError("signal ids must be unique per candidate")
        return merged


def rank_retrieval_candidates(
    *,
    ranking_input: RankingInput,
    policy: RankingPolicy | None = None,
) -> RankingResult:
    """Rank retrieval candidates deterministically."""

    ranker = RetrievalRanker(policy)
    return ranker.rank(ranking_input=ranking_input)


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


def _ensure_signal_mapping(
    values: dict[str, tuple[RankingSignal, ...] | list[RankingSignal]],
) -> dict[str, tuple[RankingSignal, ...]]:
    if not isinstance(values, dict):
        raise TypeError("additional_signals_by_candidate_id must be dict")
    result: dict[str, tuple[RankingSignal, ...]] = {}
    for key, signals in values.items():
        candidate_id = ensure_non_empty_string(key, "candidate_id")
        result[candidate_id] = _ensure_signals(signals)
    return result


def _ensure_signals(
    values: tuple[RankingSignal, ...] | list[RankingSignal],
) -> tuple[RankingSignal, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("signals must be tuple or list")
    signals = tuple(values)
    for signal in signals:
        if not isinstance(signal, RankingSignal):
            raise TypeError("signals must contain RankingSignal")
    signal_ids = tuple(signal.signal_id for signal in signals)
    if len(set(signal_ids)) != len(signal_ids):
        raise ValueError("signal ids must be unique per candidate")
    return signals


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
