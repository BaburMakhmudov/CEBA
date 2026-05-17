"""Strict schema objects for CEBA Memory Retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from uuid import UUID, uuid4


class RetrievalSourceKind(Enum):
    """Stable retrieval source category."""

    VECTOR = "VECTOR"
    GRAPH = "GRAPH"
    LEDGER = "LEDGER"
    LONG_TERM_MEMORY = "LONG_TERM_MEMORY"
    SHORT_TERM_MEMORY = "SHORT_TERM_MEMORY"
    HYBRID = "HYBRID"
    CACHE = "CACHE"
    UNKNOWN = "UNKNOWN"


class RetrievalQueryKind(Enum):
    """Stable retrieval query category."""

    SEMANTIC = "SEMANTIC"
    STRUCTURAL = "STRUCTURAL"
    TRACE = "TRACE"
    EVIDENCE = "EVIDENCE"
    CONTEXT = "CONTEXT"
    HYBRID = "HYBRID"
    UNKNOWN = "UNKNOWN"


class RetrievalCandidateKind(Enum):
    """Stable retrieval candidate category."""

    TRACE = "TRACE"
    EXPERIENCE = "EXPERIENCE"
    OBSERVATION = "OBSERVATION"
    SEGMENT = "SEGMENT"
    CONCEPT = "CONCEPT"
    RELATION = "RELATION"
    EVIDENCE = "EVIDENCE"
    CONTEXT = "CONTEXT"
    UNKNOWN = "UNKNOWN"


class RetrievalStatus(Enum):
    """Stable retrieval status."""

    PENDING = "PENDING"
    RETRIEVED = "RETRIEVED"
    RANKED = "RANKED"
    FILTERED = "FILTERED"
    EMPTY = "EMPTY"
    FAILED = "FAILED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class EvidencePackStatus(Enum):
    """Stable evidence pack status."""

    READY = "READY"
    PARTIAL = "PARTIAL"
    EMPTY = "EMPTY"
    DEGRADED = "DEGRADED"
    INVALID = "INVALID"


class RankingSignalKind(Enum):
    """Stable ranking signal category."""

    SIMILARITY = "SIMILARITY"
    RECENCY = "RECENCY"
    CONFIDENCE = "CONFIDENCE"
    PRIORITY = "PRIORITY"
    PROVENANCE = "PROVENANCE"
    STRUCTURAL_DISTANCE = "STRUCTURAL_DISTANCE"
    HYBRID_SCORE = "HYBRID_SCORE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class RetrievalQuery:
    """Typed retrieval query descriptor."""

    query_id: UUID
    kind: RetrievalQueryKind
    source_kind: RetrievalSourceKind
    query_ref: str
    limit: int
    min_score: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, UUID):
            raise TypeError("query_id must be UUID")
        if not isinstance(self.kind, RetrievalQueryKind):
            raise TypeError("kind must be RetrievalQueryKind")
        if not isinstance(self.source_kind, RetrievalSourceKind):
            raise TypeError("source_kind must be RetrievalSourceKind")
        query_ref = ensure_non_empty_string(self.query_ref, "query_ref")
        limit = ensure_positive_int(self.limit, "limit")
        min_score = ensure_probability(self.min_score, "min_score")
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        object.__setattr__(self, "query_ref", query_ref)
        object.__setattr__(self, "limit", limit)
        object.__setattr__(self, "min_score", min_score)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "query_id": str(self.query_id),
            "kind": self.kind.value,
            "source_kind": self.source_kind.value,
            "query_ref": self.query_ref,
            "limit": self.limit,
            "min_score": self.min_score,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RankingSignal:
    """Single scoring signal attached to a retrieval candidate."""

    signal_id: UUID
    kind: RankingSignalKind
    score: float
    weight: float
    source_id: str
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.signal_id, UUID):
            raise TypeError("signal_id must be UUID")
        if not isinstance(self.kind, RankingSignalKind):
            raise TypeError("kind must be RankingSignalKind")
        score = ensure_probability(self.score, "score")
        weight = ensure_non_negative_number(self.weight, "weight")
        source_id = ensure_non_empty_string(self.source_id, "source_id")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "signal_id": str(self.signal_id),
            "kind": self.kind.value,
            "score": self.score,
            "weight": self.weight,
            "source_id": self.source_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RetrievalCandidate:
    """Candidate item returned by a retrieval surface."""

    candidate_id: UUID
    kind: RetrievalCandidateKind
    source_kind: RetrievalSourceKind
    source_id: str
    content_ref: str | None
    score: float
    confidence: float
    ranking_signals: tuple[RankingSignal, ...]
    retrieved_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, UUID):
            raise TypeError("candidate_id must be UUID")
        if not isinstance(self.kind, RetrievalCandidateKind):
            raise TypeError("kind must be RetrievalCandidateKind")
        if not isinstance(self.source_kind, RetrievalSourceKind):
            raise TypeError("source_kind must be RetrievalSourceKind")
        source_id = ensure_non_empty_string(self.source_id, "source_id")
        content_ref = ensure_optional_non_empty_string(
            self.content_ref,
            "content_ref",
        )
        score = ensure_probability(self.score, "score")
        confidence = ensure_probability(self.confidence, "confidence")
        signals = _ensure_ranking_signals(self.ranking_signals)
        retrieved_at = ensure_timezone_aware(self.retrieved_at, "retrieved_at")
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "content_ref", content_ref)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "ranking_signals", signals)
        object.__setattr__(self, "retrieved_at", retrieved_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "candidate_id": str(self.candidate_id),
            "kind": self.kind.value,
            "source_kind": self.source_kind.value,
            "source_id": self.source_id,
            "content_ref": self.content_ref,
            "score": self.score,
            "confidence": self.confidence,
            "ranking_signals": tuple(
                signal.to_dict() for signal in self.ranking_signals
            ),
            "retrieved_at": self.retrieved_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RetrievalEvidenceItem:
    """Evidence item extracted from a retrieval candidate."""

    evidence_id: UUID
    candidate_id: UUID
    source_id: str
    evidence_ref: str
    relevance: float
    confidence: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_id, UUID):
            raise TypeError("evidence_id must be UUID")
        if not isinstance(self.candidate_id, UUID):
            raise TypeError("candidate_id must be UUID")
        source_id = ensure_non_empty_string(self.source_id, "source_id")
        evidence_ref = ensure_non_empty_string(
            self.evidence_ref,
            "evidence_ref",
        )
        relevance = ensure_probability(self.relevance, "relevance")
        confidence = ensure_probability(self.confidence, "confidence")
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "evidence_ref", evidence_ref)
        object.__setattr__(self, "relevance", relevance)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "evidence_id": str(self.evidence_id),
            "candidate_id": str(self.candidate_id),
            "source_id": self.source_id,
            "evidence_ref": self.evidence_ref,
            "relevance": self.relevance,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RetrievalEvidencePack:
    """Runtime evidence pack assembled from retrieval evidence items."""

    pack_id: UUID
    query_id: UUID
    status: EvidencePackStatus
    evidence_items: tuple[RetrievalEvidenceItem, ...]
    candidates: tuple[RetrievalCandidate, ...]
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.pack_id, UUID):
            raise TypeError("pack_id must be UUID")
        if not isinstance(self.query_id, UUID):
            raise TypeError("query_id must be UUID")
        if not isinstance(self.status, EvidencePackStatus):
            raise TypeError("status must be EvidencePackStatus")
        evidence_items = _ensure_evidence_items(self.evidence_items)
        candidates = _ensure_candidates(self.candidates)
        candidate_ids = {candidate.candidate_id for candidate in candidates}
        for item in evidence_items:
            if item.candidate_id not in candidate_ids:
                raise ValueError("evidence_item candidate_id must exist")
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        object.__setattr__(self, "evidence_items", evidence_items)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "pack_id": str(self.pack_id),
            "query_id": str(self.query_id),
            "status": self.status.value,
            "evidence_items": tuple(
                item.to_dict() for item in self.evidence_items
            ),
            "candidates": tuple(
                candidate.to_dict() for candidate in self.candidates
            ),
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    def evidence_count(self) -> int:
        """Return evidence item count."""

        return len(self.evidence_items)

    def candidate_count(self) -> int:
        """Return candidate count."""

        return len(self.candidates)


@dataclass(frozen=True)
class RetrievalSnapshot:
    """Immutable runtime snapshot of retrieval state."""

    snapshot_id: UUID
    query: RetrievalQuery
    status: RetrievalStatus
    candidates: tuple[RetrievalCandidate, ...]
    evidence_pack: RetrievalEvidencePack | None
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot_id, UUID):
            raise TypeError("snapshot_id must be UUID")
        if not isinstance(self.query, RetrievalQuery):
            raise TypeError("query must be RetrievalQuery")
        if not isinstance(self.status, RetrievalStatus):
            raise TypeError("status must be RetrievalStatus")
        candidates = _ensure_candidates(self.candidates)
        evidence_pack = self.evidence_pack
        if evidence_pack is not None:
            if not isinstance(evidence_pack, RetrievalEvidencePack):
                raise TypeError(
                    "evidence_pack must be RetrievalEvidencePack or None"
                )
            if evidence_pack.query_id != self.query.query_id:
                raise ValueError("evidence_pack query_id must equal query_id")
            candidate_ids = {candidate.candidate_id for candidate in candidates}
            for candidate in evidence_pack.candidates:
                if candidate.candidate_id not in candidate_ids:
                    raise ValueError("evidence_pack candidate must exist")
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "snapshot_id": str(self.snapshot_id),
            "query": self.query.to_dict(),
            "status": self.status.value,
            "candidates": tuple(
                candidate.to_dict() for candidate in self.candidates
            ),
            "evidence_pack": (
                None
                if self.evidence_pack is None
                else self.evidence_pack.to_dict()
            ),
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    def candidate_count(self) -> int:
        """Return candidate count."""

        return len(self.candidates)

    def has_evidence_pack(self) -> bool:
        """Return whether this snapshot has an evidence pack."""

        return self.evidence_pack is not None


def utc_now() -> datetime:
    """Return a timezone-aware current UTC datetime."""

    return datetime.now(timezone.utc)


def new_retrieval_query_id() -> UUID:
    """Return a new retrieval query identifier."""

    return uuid4()


def new_ranking_signal_id() -> UUID:
    """Return a new ranking signal identifier."""

    return uuid4()


def new_retrieval_candidate_id() -> UUID:
    """Return a new retrieval candidate identifier."""

    return uuid4()


def new_retrieval_evidence_id() -> UUID:
    """Return a new retrieval evidence identifier."""

    return uuid4()


def new_retrieval_pack_id() -> UUID:
    """Return a new retrieval evidence pack identifier."""

    return uuid4()


def new_retrieval_snapshot_id() -> UUID:
    """Return a new retrieval snapshot identifier."""

    return uuid4()


def ensure_non_empty_string(value: str, field_name: str) -> str:
    """Validate a non-empty string and return its stripped value."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    stripped = value.strip()
    if stripped == "":
        raise ValueError(f"{field_name} must be non-empty")
    return stripped


def ensure_optional_non_empty_string(
    value: str | None,
    field_name: str,
) -> str | None:
    """Validate an optional non-empty string."""

    if value is None:
        return None
    return ensure_non_empty_string(value, field_name)


def ensure_probability(value: float, field_name: str) -> float:
    """Validate a finite probability value."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be int or float")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{field_name} must be in range 0.0..1.0")
    return number


def ensure_non_negative_number(value: float, field_name: str) -> float:
    """Validate a non-negative finite number."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be int or float")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    if number < 0.0:
        raise ValueError(f"{field_name} must be non-negative")
    return number


def ensure_positive_int(value: int, field_name: str) -> int:
    """Validate a positive integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def ensure_timezone_aware(value: datetime, field_name: str) -> datetime:
    """Validate a timezone-aware datetime."""

    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def create_retrieval_query(
    *,
    query_ref: str,
    kind: RetrievalQueryKind = RetrievalQueryKind.UNKNOWN,
    source_kind: RetrievalSourceKind = RetrievalSourceKind.UNKNOWN,
    limit: int = 10,
    min_score: float = 0.0,
    metadata: dict[str, object] | None = None,
) -> RetrievalQuery:
    """Create a retrieval query descriptor."""

    default_metadata: dict[str, object] = {"source": "retrieval_query"}
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "retrieval_query"
    return RetrievalQuery(
        query_id=new_retrieval_query_id(),
        kind=kind,
        source_kind=source_kind,
        query_ref=query_ref,
        limit=limit,
        min_score=min_score,
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def create_ranking_signal(
    *,
    score: float,
    source_id: str,
    kind: RankingSignalKind = RankingSignalKind.UNKNOWN,
    weight: float = 1.0,
    metadata: dict[str, object] | None = None,
) -> RankingSignal:
    """Create a ranking signal."""

    default_metadata: dict[str, object] = {"source": "ranking_signal"}
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "ranking_signal"
    return RankingSignal(
        signal_id=new_ranking_signal_id(),
        kind=kind,
        score=score,
        weight=weight,
        source_id=source_id,
        metadata=merged_metadata,
    )


def create_retrieval_candidate(
    *,
    source_id: str,
    kind: RetrievalCandidateKind = RetrievalCandidateKind.UNKNOWN,
    source_kind: RetrievalSourceKind = RetrievalSourceKind.UNKNOWN,
    content_ref: str | None = None,
    score: float = 0.0,
    confidence: float = 1.0,
    ranking_signals: tuple[RankingSignal, ...] | list[RankingSignal] = (),
    metadata: dict[str, object] | None = None,
) -> RetrievalCandidate:
    """Create a retrieval candidate."""

    default_metadata: dict[str, object] = {
        "source": "retrieval_candidate",
        "source_id": source_id,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "retrieval_candidate"
    return RetrievalCandidate(
        candidate_id=new_retrieval_candidate_id(),
        kind=kind,
        source_kind=source_kind,
        source_id=source_id,
        content_ref=content_ref,
        score=score,
        confidence=confidence,
        ranking_signals=tuple(ranking_signals),
        retrieved_at=utc_now(),
        metadata=merged_metadata,
    )


def create_retrieval_evidence_item(
    *,
    candidate_id: UUID,
    source_id: str,
    evidence_ref: str,
    relevance: float = 1.0,
    confidence: float = 1.0,
    metadata: dict[str, object] | None = None,
) -> RetrievalEvidenceItem:
    """Create a retrieval evidence item."""

    default_metadata: dict[str, object] = {
        "source": "retrieval_evidence_item",
        "candidate_id": str(candidate_id),
        "source_id": source_id,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "retrieval_evidence_item"
    return RetrievalEvidenceItem(
        evidence_id=new_retrieval_evidence_id(),
        candidate_id=candidate_id,
        source_id=source_id,
        evidence_ref=evidence_ref,
        relevance=relevance,
        confidence=confidence,
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def create_retrieval_evidence_pack(
    *,
    query_id: UUID,
    evidence_items: (
        tuple[RetrievalEvidenceItem, ...] | list[RetrievalEvidenceItem]
    ) = (),
    candidates: tuple[RetrievalCandidate, ...] | list[RetrievalCandidate] = (),
    status: EvidencePackStatus = EvidencePackStatus.READY,
    metadata: dict[str, object] | None = None,
) -> RetrievalEvidencePack:
    """Create a retrieval evidence pack."""

    evidence_values = tuple(evidence_items)
    candidate_values = tuple(candidates)
    default_metadata: dict[str, object] = {
        "source": "retrieval_evidence_pack",
        "evidence_count": len(evidence_values),
        "candidate_count": len(candidate_values),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "retrieval_evidence_pack"
    return RetrievalEvidencePack(
        pack_id=new_retrieval_pack_id(),
        query_id=query_id,
        status=status,
        evidence_items=evidence_values,
        candidates=candidate_values,
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def create_retrieval_snapshot(
    *,
    query: RetrievalQuery,
    status: RetrievalStatus = RetrievalStatus.PENDING,
    candidates: tuple[RetrievalCandidate, ...] | list[RetrievalCandidate] = (),
    evidence_pack: RetrievalEvidencePack | None = None,
    metadata: dict[str, object] | None = None,
) -> RetrievalSnapshot:
    """Create a retrieval snapshot."""

    candidate_values = tuple(candidates)
    default_metadata: dict[str, object] = {
        "source": "retrieval_snapshot",
        "candidate_count": len(candidate_values),
        "has_evidence_pack": evidence_pack is not None,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "retrieval_snapshot"
    return RetrievalSnapshot(
        snapshot_id=new_retrieval_snapshot_id(),
        query=query,
        status=status,
        candidates=candidate_values,
        evidence_pack=evidence_pack,
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def _ensure_ranking_signals(
    values: tuple[RankingSignal, ...] | list[RankingSignal],
) -> tuple[RankingSignal, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("ranking_signals must be tuple or list")
    result = tuple(values)
    for value in result:
        if not isinstance(value, RankingSignal):
            raise TypeError("ranking_signals must contain RankingSignal")
    signal_ids = tuple(signal.signal_id for signal in result)
    if len(set(signal_ids)) != len(signal_ids):
        raise ValueError("ranking signal ids must be unique")
    return result


def _ensure_candidates(
    values: tuple[RetrievalCandidate, ...] | list[RetrievalCandidate],
) -> tuple[RetrievalCandidate, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("candidates must be tuple or list")
    result = tuple(values)
    for value in result:
        if not isinstance(value, RetrievalCandidate):
            raise TypeError("candidates must contain RetrievalCandidate")
    candidate_ids = tuple(candidate.candidate_id for candidate in result)
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate_id values must be unique")
    return result


def _ensure_evidence_items(
    values: tuple[RetrievalEvidenceItem, ...] | list[RetrievalEvidenceItem],
) -> tuple[RetrievalEvidenceItem, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("evidence_items must be tuple or list")
    result = tuple(values)
    for value in result:
        if not isinstance(value, RetrievalEvidenceItem):
            raise TypeError("evidence_items must contain RetrievalEvidenceItem")
    evidence_ids = tuple(item.evidence_id for item in result)
    if len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError("evidence_id values must be unique")
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
    "RetrievalSourceKind",
    "RetrievalQueryKind",
    "RetrievalCandidateKind",
    "RetrievalStatus",
    "EvidencePackStatus",
    "RankingSignalKind",
    "RetrievalQuery",
    "RankingSignal",
    "RetrievalCandidate",
    "RetrievalEvidenceItem",
    "RetrievalEvidencePack",
    "RetrievalSnapshot",
    "utc_now",
    "new_retrieval_query_id",
    "new_ranking_signal_id",
    "new_retrieval_candidate_id",
    "new_retrieval_evidence_id",
    "new_retrieval_pack_id",
    "new_retrieval_snapshot_id",
    "ensure_non_empty_string",
    "ensure_optional_non_empty_string",
    "ensure_probability",
    "ensure_non_negative_number",
    "ensure_positive_int",
    "ensure_timezone_aware",
    "create_retrieval_query",
    "create_ranking_signal",
    "create_retrieval_candidate",
    "create_retrieval_evidence_item",
    "create_retrieval_evidence_pack",
    "create_retrieval_snapshot",
]
