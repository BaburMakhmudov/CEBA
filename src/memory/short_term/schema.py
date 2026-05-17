"""Strict schema objects for CEBA Short-Term Memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from uuid import UUID, uuid4


class ShortTermItemKind(Enum):
    """Stable short-term memory item category."""

    OBSERVATION = "OBSERVATION"
    SEGMENT = "SEGMENT"
    EMBEDDING_REF = "EMBEDDING_REF"
    DIFFERENCE = "DIFFERENCE"
    CONSTRAINT_SIGNAL = "CONSTRAINT_SIGNAL"
    METRIC_SIGNAL = "METRIC_SIGNAL"
    RELATION_SIGNAL = "RELATION_SIGNAL"
    TELOS_SIGNAL = "TELOS_SIGNAL"
    VALUATION_SIGNAL = "VALUATION_SIGNAL"
    WORLD_MODEL_SIGNAL = "WORLD_MODEL_SIGNAL"
    HYPOTHESIS = "HYPOTHESIS"
    CANDIDATE_TRACE = "CANDIDATE_TRACE"
    EVIDENCE_REF = "EVIDENCE_REF"
    UNKNOWN = "UNKNOWN"


class ShortTermItemStatus(Enum):
    """Stable short-term memory item status."""

    ACTIVE = "ACTIVE"
    RESOLVED = "RESOLVED"
    EXPIRED = "EXPIRED"
    EVICTED = "EVICTED"
    BLOCKED = "BLOCKED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class CandidateTraceStatus(Enum):
    """Stable candidate trace status."""

    CANDIDATE = "CANDIDATE"
    ELIGIBLE = "ELIGIBLE"
    REJECTED = "REJECTED"
    CONSOLIDATED = "CONSOLIDATED"
    EXPIRED = "EXPIRED"
    INVALID = "INVALID"


class HypothesisStatus(Enum):
    """Stable working hypothesis status."""

    OPEN = "OPEN"
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    RESOLVED = "RESOLVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    INVALID = "INVALID"


class ActiveContextStatus(Enum):
    """Stable active context status."""

    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    CLOSED = "CLOSED"
    EXPIRED = "EXPIRED"
    INVALID = "INVALID"


class EvictionReason(Enum):
    """Stable short-term memory eviction reason."""

    CAPACITY_LIMIT = "CAPACITY_LIMIT"
    TTL_EXPIRED = "TTL_EXPIRED"
    LOW_PRIORITY = "LOW_PRIORITY"
    RESOLVED = "RESOLVED"
    CONSOLIDATED = "CONSOLIDATED"
    INVALID = "INVALID"
    MANUAL = "MANUAL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ShortTermItem:
    """Atomic runtime item held in short-term memory."""

    item_id: UUID
    kind: ShortTermItemKind
    status: ShortTermItemStatus
    source_id: str
    content_ref: str | None
    priority: float
    confidence: float
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.item_id, UUID):
            raise TypeError("item_id must be UUID")
        if not isinstance(self.kind, ShortTermItemKind):
            raise TypeError("kind must be ShortTermItemKind")
        if not isinstance(self.status, ShortTermItemStatus):
            raise TypeError("status must be ShortTermItemStatus")
        source_id = ensure_non_empty_string(self.source_id, "source_id")
        content_ref = ensure_optional_non_empty_string(
            self.content_ref,
            "content_ref",
        )
        priority = ensure_probability(self.priority, "priority")
        confidence = ensure_probability(self.confidence, "confidence")
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        updated_at = ensure_timezone_aware(self.updated_at, "updated_at")
        if updated_at < created_at:
            raise ValueError("updated_at must be >= created_at")
        expires_at = (
            None
            if self.expires_at is None
            else ensure_timezone_aware(self.expires_at, "expires_at")
        )
        if expires_at is not None and expires_at <= created_at:
            raise ValueError("expires_at must be > created_at")

        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "content_ref", content_ref)
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "expires_at", expires_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "item_id": str(self.item_id),
            "kind": self.kind.value,
            "status": self.status.value,
            "source_id": self.source_id,
            "content_ref": self.content_ref,
            "priority": self.priority,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": (
                None if self.expires_at is None else self.expires_at.isoformat()
            ),
            "metadata": dict(self.metadata),
        }

    def is_expired(self, now: datetime | None = None) -> bool:
        """Return whether the item is expired at the given time."""

        if self.expires_at is None:
            return False
        now_value = utc_now() if now is None else ensure_timezone_aware(now, "now")
        return now_value >= self.expires_at


@dataclass(frozen=True)
class ActiveContextRef:
    """Reference to an active runtime context window."""

    context_id: UUID
    status: ActiveContextStatus
    root_item_id: UUID | None
    item_ids: tuple[UUID, ...]
    opened_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.context_id, UUID):
            raise TypeError("context_id must be UUID")
        if not isinstance(self.status, ActiveContextStatus):
            raise TypeError("status must be ActiveContextStatus")
        if self.root_item_id is not None and not isinstance(
            self.root_item_id,
            UUID,
        ):
            raise TypeError("root_item_id must be UUID or None")
        item_ids = _ensure_uuid_tuple(self.item_ids, "item_ids")
        _ensure_unique_uuids(item_ids, "item_ids")
        if self.root_item_id is not None and self.root_item_id not in item_ids:
            raise ValueError("root_item_id must exist in item_ids")
        opened_at = ensure_timezone_aware(self.opened_at, "opened_at")
        updated_at = ensure_timezone_aware(self.updated_at, "updated_at")
        if updated_at < opened_at:
            raise ValueError("updated_at must be >= opened_at")

        object.__setattr__(self, "item_ids", item_ids)
        object.__setattr__(self, "opened_at", opened_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "context_id": str(self.context_id),
            "status": self.status.value,
            "root_item_id": (
                None if self.root_item_id is None else str(self.root_item_id)
            ),
            "item_ids": tuple(str(item_id) for item_id in self.item_ids),
            "opened_at": self.opened_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CandidateTrace:
    """Runtime trace candidate before long-term consolidation."""

    trace_id: UUID
    status: CandidateTraceStatus
    item_ids: tuple[UUID, ...]
    evidence_ids: tuple[str, ...]
    confidence: float
    priority: float
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.trace_id, UUID):
            raise TypeError("trace_id must be UUID")
        if not isinstance(self.status, CandidateTraceStatus):
            raise TypeError("status must be CandidateTraceStatus")
        item_ids = _ensure_uuid_tuple(self.item_ids, "item_ids")
        _ensure_unique_uuids(item_ids, "item_ids")
        evidence_ids = _ensure_string_tuple(self.evidence_ids, "evidence_ids")
        _ensure_unique_strings(evidence_ids, "evidence_ids")
        confidence = ensure_probability(self.confidence, "confidence")
        priority = ensure_probability(self.priority, "priority")
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        updated_at = ensure_timezone_aware(self.updated_at, "updated_at")
        if updated_at < created_at:
            raise ValueError("updated_at must be >= created_at")

        object.__setattr__(self, "item_ids", item_ids)
        object.__setattr__(self, "evidence_ids", evidence_ids)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "trace_id": str(self.trace_id),
            "status": self.status.value,
            "item_ids": tuple(str(item_id) for item_id in self.item_ids),
            "evidence_ids": self.evidence_ids,
            "confidence": self.confidence,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class WorkingHypothesis:
    """Runtime hypothesis held before validation, rejection, or resolution."""

    hypothesis_id: UUID
    status: HypothesisStatus
    statement_ref: str
    supporting_item_ids: tuple[UUID, ...]
    contradicting_item_ids: tuple[UUID, ...]
    confidence: float
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.hypothesis_id, UUID):
            raise TypeError("hypothesis_id must be UUID")
        if not isinstance(self.status, HypothesisStatus):
            raise TypeError("status must be HypothesisStatus")
        statement_ref = ensure_non_empty_string(
            self.statement_ref,
            "statement_ref",
        )
        supporting_item_ids = _ensure_uuid_tuple(
            self.supporting_item_ids,
            "supporting_item_ids",
        )
        contradicting_item_ids = _ensure_uuid_tuple(
            self.contradicting_item_ids,
            "contradicting_item_ids",
        )
        _ensure_unique_uuids(supporting_item_ids, "supporting_item_ids")
        _ensure_unique_uuids(contradicting_item_ids, "contradicting_item_ids")
        confidence = ensure_probability(self.confidence, "confidence")
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        updated_at = ensure_timezone_aware(self.updated_at, "updated_at")
        if updated_at < created_at:
            raise ValueError("updated_at must be >= created_at")

        object.__setattr__(self, "statement_ref", statement_ref)
        object.__setattr__(self, "supporting_item_ids", supporting_item_ids)
        object.__setattr__(
            self,
            "contradicting_item_ids",
            contradicting_item_ids,
        )
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "hypothesis_id": str(self.hypothesis_id),
            "status": self.status.value,
            "statement_ref": self.statement_ref,
            "supporting_item_ids": tuple(
                str(item_id) for item_id in self.supporting_item_ids
            ),
            "contradicting_item_ids": tuple(
                str(item_id) for item_id in self.contradicting_item_ids
            ),
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EvictionRecord:
    """Runtime descriptor explaining why an item was removed from holding."""

    eviction_id: UUID
    item_id: UUID
    reason: EvictionReason
    evicted_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.eviction_id, UUID):
            raise TypeError("eviction_id must be UUID")
        if not isinstance(self.item_id, UUID):
            raise TypeError("item_id must be UUID")
        if not isinstance(self.reason, EvictionReason):
            raise TypeError("reason must be EvictionReason")
        evicted_at = ensure_timezone_aware(self.evicted_at, "evicted_at")

        object.__setattr__(self, "evicted_at", evicted_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "eviction_id": str(self.eviction_id),
            "item_id": str(self.item_id),
            "reason": self.reason.value,
            "evicted_at": self.evicted_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ShortTermMemorySnapshot:
    """Immutable runtime snapshot of short-term memory state."""

    snapshot_id: UUID
    items: tuple[ShortTermItem, ...]
    contexts: tuple[ActiveContextRef, ...]
    candidate_traces: tuple[CandidateTrace, ...]
    hypotheses: tuple[WorkingHypothesis, ...]
    evictions: tuple[EvictionRecord, ...]
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot_id, UUID):
            raise TypeError("snapshot_id must be UUID")
        items = _ensure_items(self.items)
        contexts = _ensure_contexts(self.contexts)
        candidate_traces = _ensure_candidate_traces(self.candidate_traces)
        hypotheses = _ensure_hypotheses(self.hypotheses)
        evictions = _ensure_evictions(self.evictions)
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        metadata = _copy_metadata(self.metadata)
        _validate_snapshot_references(
            items=items,
            contexts=contexts,
            candidate_traces=candidate_traces,
            hypotheses=hypotheses,
            evictions=evictions,
            metadata=metadata,
        )

        object.__setattr__(self, "items", items)
        object.__setattr__(self, "contexts", contexts)
        object.__setattr__(self, "candidate_traces", candidate_traces)
        object.__setattr__(self, "hypotheses", hypotheses)
        object.__setattr__(self, "evictions", evictions)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "snapshot_id": str(self.snapshot_id),
            "items": tuple(item.to_dict() for item in self.items),
            "contexts": tuple(context.to_dict() for context in self.contexts),
            "candidate_traces": tuple(
                trace.to_dict() for trace in self.candidate_traces
            ),
            "hypotheses": tuple(
                hypothesis.to_dict() for hypothesis in self.hypotheses
            ),
            "evictions": tuple(eviction.to_dict() for eviction in self.evictions),
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    def item_count(self) -> int:
        """Return item count."""

        return len(self.items)

    def context_count(self) -> int:
        """Return context count."""

        return len(self.contexts)

    def candidate_trace_count(self) -> int:
        """Return candidate trace count."""

        return len(self.candidate_traces)

    def hypothesis_count(self) -> int:
        """Return hypothesis count."""

        return len(self.hypotheses)

    def eviction_count(self) -> int:
        """Return eviction count."""

        return len(self.evictions)


def utc_now() -> datetime:
    """Return a timezone-aware current UTC datetime."""

    return datetime.now(timezone.utc)


def new_short_term_item_id() -> UUID:
    """Return a new short-term item identifier."""

    return uuid4()


def new_active_context_id() -> UUID:
    """Return a new active context identifier."""

    return uuid4()


def new_candidate_trace_id() -> UUID:
    """Return a new candidate trace identifier."""

    return uuid4()


def new_working_hypothesis_id() -> UUID:
    """Return a new working hypothesis identifier."""

    return uuid4()


def new_eviction_id() -> UUID:
    """Return a new eviction identifier."""

    return uuid4()


def new_short_term_snapshot_id() -> UUID:
    """Return a new short-term memory snapshot identifier."""

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


def ensure_timezone_aware(value: datetime, field_name: str) -> datetime:
    """Validate a timezone-aware datetime."""

    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def create_short_term_item(
    *,
    kind: ShortTermItemKind = ShortTermItemKind.UNKNOWN,
    status: ShortTermItemStatus = ShortTermItemStatus.ACTIVE,
    source_id: str,
    content_ref: str | None = None,
    priority: float = 0.5,
    confidence: float = 1.0,
    expires_at: datetime | None = None,
    metadata: dict[str, object] | None = None,
) -> ShortTermItem:
    """Create a short-term memory item."""

    now = utc_now()
    default_metadata: dict[str, object] = {"source": "short_term_item"}
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "short_term_item"
    return ShortTermItem(
        item_id=new_short_term_item_id(),
        kind=kind,
        status=status,
        source_id=source_id,
        content_ref=content_ref,
        priority=priority,
        confidence=confidence,
        created_at=now,
        updated_at=now,
        expires_at=expires_at,
        metadata=merged_metadata,
    )


def create_active_context_ref(
    *,
    root_item_id: UUID | None = None,
    item_ids: tuple[UUID, ...] | list[UUID] = (),
    status: ActiveContextStatus = ActiveContextStatus.ACTIVE,
    metadata: dict[str, object] | None = None,
) -> ActiveContextRef:
    """Create an active context reference."""

    item_ids_value = tuple(item_ids)
    if root_item_id is not None and not item_ids_value:
        item_ids_value = (root_item_id,)
    now = utc_now()
    default_metadata: dict[str, object] = {
        "source": "active_context_ref",
        "item_count": len(item_ids_value),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "active_context_ref"
    return ActiveContextRef(
        context_id=new_active_context_id(),
        status=status,
        root_item_id=root_item_id,
        item_ids=item_ids_value,
        opened_at=now,
        updated_at=now,
        metadata=merged_metadata,
    )


def create_candidate_trace(
    *,
    item_ids: tuple[UUID, ...] | list[UUID],
    evidence_ids: tuple[str, ...] | list[str] = (),
    status: CandidateTraceStatus = CandidateTraceStatus.CANDIDATE,
    confidence: float = 1.0,
    priority: float = 0.5,
    metadata: dict[str, object] | None = None,
) -> CandidateTrace:
    """Create a candidate trace."""

    item_ids_value = tuple(item_ids)
    evidence_ids_value = tuple(evidence_ids)
    now = utc_now()
    default_metadata: dict[str, object] = {
        "source": "candidate_trace",
        "item_count": len(item_ids_value),
        "evidence_count": len(evidence_ids_value),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "candidate_trace"
    return CandidateTrace(
        trace_id=new_candidate_trace_id(),
        status=status,
        item_ids=item_ids_value,
        evidence_ids=evidence_ids_value,
        confidence=confidence,
        priority=priority,
        created_at=now,
        updated_at=now,
        metadata=merged_metadata,
    )


def create_working_hypothesis(
    *,
    statement_ref: str,
    supporting_item_ids: tuple[UUID, ...] | list[UUID] = (),
    contradicting_item_ids: tuple[UUID, ...] | list[UUID] = (),
    status: HypothesisStatus = HypothesisStatus.OPEN,
    confidence: float = 0.5,
    metadata: dict[str, object] | None = None,
) -> WorkingHypothesis:
    """Create a working hypothesis."""

    supporting_ids = tuple(supporting_item_ids)
    contradicting_ids = tuple(contradicting_item_ids)
    now = utc_now()
    default_metadata: dict[str, object] = {
        "source": "working_hypothesis",
        "support_count": len(supporting_ids),
        "contradiction_count": len(contradicting_ids),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "working_hypothesis"
    return WorkingHypothesis(
        hypothesis_id=new_working_hypothesis_id(),
        status=status,
        statement_ref=statement_ref,
        supporting_item_ids=supporting_ids,
        contradicting_item_ids=contradicting_ids,
        confidence=confidence,
        created_at=now,
        updated_at=now,
        metadata=merged_metadata,
    )


def create_eviction_record(
    *,
    item_id: UUID,
    reason: EvictionReason = EvictionReason.UNKNOWN,
    metadata: dict[str, object] | None = None,
) -> EvictionRecord:
    """Create an eviction record."""

    default_metadata: dict[str, object] = {
        "source": "eviction_record",
        "item_id": str(item_id),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "eviction_record"
    return EvictionRecord(
        eviction_id=new_eviction_id(),
        item_id=item_id,
        reason=reason,
        evicted_at=utc_now(),
        metadata=merged_metadata,
    )


def create_short_term_memory_snapshot(
    *,
    items: tuple[ShortTermItem, ...] | list[ShortTermItem] = (),
    contexts: tuple[ActiveContextRef, ...] | list[ActiveContextRef] = (),
    candidate_traces: tuple[CandidateTrace, ...] | list[CandidateTrace] = (),
    hypotheses: tuple[WorkingHypothesis, ...] | list[WorkingHypothesis] = (),
    evictions: tuple[EvictionRecord, ...] | list[EvictionRecord] = (),
    metadata: dict[str, object] | None = None,
) -> ShortTermMemorySnapshot:
    """Create a short-term memory snapshot."""

    items_value = tuple(items)
    contexts_value = tuple(contexts)
    traces_value = tuple(candidate_traces)
    hypotheses_value = tuple(hypotheses)
    evictions_value = tuple(evictions)
    default_metadata: dict[str, object] = {
        "source": "short_term_memory_snapshot",
        "item_count": len(items_value),
        "context_count": len(contexts_value),
        "candidate_trace_count": len(traces_value),
        "hypothesis_count": len(hypotheses_value),
        "eviction_count": len(evictions_value),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "short_term_memory_snapshot"
    return ShortTermMemorySnapshot(
        snapshot_id=new_short_term_snapshot_id(),
        items=items_value,
        contexts=contexts_value,
        candidate_traces=traces_value,
        hypotheses=hypotheses_value,
        evictions=evictions_value,
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def _ensure_uuid_tuple(
    values: tuple[UUID, ...] | list[UUID],
    field_name: str,
) -> tuple[UUID, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple or list")
    result = tuple(values)
    for value in result:
        if not isinstance(value, UUID):
            raise TypeError(f"{field_name} must contain UUID")
    return result


def _ensure_string_tuple(
    values: tuple[str, ...] | list[str],
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple or list")
    return tuple(ensure_non_empty_string(value, field_name) for value in values)


def _ensure_unique_uuids(values: tuple[UUID, ...], field_name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must contain unique UUIDs")


def _ensure_unique_strings(values: tuple[str, ...], field_name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must contain unique strings")


def _ensure_items(
    values: tuple[ShortTermItem, ...] | list[ShortTermItem],
) -> tuple[ShortTermItem, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("items must be tuple or list")
    result = tuple(values)
    for item in result:
        if not isinstance(item, ShortTermItem):
            raise TypeError("items must contain ShortTermItem")
    _ensure_unique_ids(tuple(item.item_id for item in result), "item_id")
    return result


def _ensure_contexts(
    values: tuple[ActiveContextRef, ...] | list[ActiveContextRef],
) -> tuple[ActiveContextRef, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("contexts must be tuple or list")
    result = tuple(values)
    for context in result:
        if not isinstance(context, ActiveContextRef):
            raise TypeError("contexts must contain ActiveContextRef")
    _ensure_unique_ids(tuple(context.context_id for context in result), "context_id")
    return result


def _ensure_candidate_traces(
    values: tuple[CandidateTrace, ...] | list[CandidateTrace],
) -> tuple[CandidateTrace, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("candidate_traces must be tuple or list")
    result = tuple(values)
    for trace in result:
        if not isinstance(trace, CandidateTrace):
            raise TypeError("candidate_traces must contain CandidateTrace")
    _ensure_unique_ids(tuple(trace.trace_id for trace in result), "trace_id")
    return result


def _ensure_hypotheses(
    values: tuple[WorkingHypothesis, ...] | list[WorkingHypothesis],
) -> tuple[WorkingHypothesis, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("hypotheses must be tuple or list")
    result = tuple(values)
    for hypothesis in result:
        if not isinstance(hypothesis, WorkingHypothesis):
            raise TypeError("hypotheses must contain WorkingHypothesis")
    _ensure_unique_ids(
        tuple(hypothesis.hypothesis_id for hypothesis in result),
        "hypothesis_id",
    )
    return result


def _ensure_evictions(
    values: tuple[EvictionRecord, ...] | list[EvictionRecord],
) -> tuple[EvictionRecord, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("evictions must be tuple or list")
    result = tuple(values)
    for eviction in result:
        if not isinstance(eviction, EvictionRecord):
            raise TypeError("evictions must contain EvictionRecord")
    _ensure_unique_ids(
        tuple(eviction.eviction_id for eviction in result),
        "eviction_id",
    )
    return result


def _ensure_unique_ids(values: tuple[UUID, ...], field_name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} values must be unique")


def _validate_snapshot_references(
    *,
    items: tuple[ShortTermItem, ...],
    contexts: tuple[ActiveContextRef, ...],
    candidate_traces: tuple[CandidateTrace, ...],
    hypotheses: tuple[WorkingHypothesis, ...],
    evictions: tuple[EvictionRecord, ...],
    metadata: dict[str, object],
) -> None:
    item_ids = {item.item_id for item in items}
    for context in contexts:
        for item_id in context.item_ids:
            if item_id not in item_ids:
                raise ValueError("context item_id must exist in items")
    for trace in candidate_traces:
        for item_id in trace.item_ids:
            if item_id not in item_ids:
                raise ValueError("candidate_trace item_id must exist in items")
    for hypothesis in hypotheses:
        for item_id in hypothesis.supporting_item_ids:
            if item_id not in item_ids:
                raise ValueError("hypothesis supporting item_id must exist in items")
        for item_id in hypothesis.contradicting_item_ids:
            if item_id not in item_ids:
                raise ValueError(
                    "hypothesis contradicting item_id must exist in items"
                )
    known_evicted = _known_evicted_item_ids(metadata)
    for eviction in evictions:
        if eviction.item_id not in item_ids and eviction.item_id not in known_evicted:
            raise ValueError(
                "eviction.item_id must exist in items or known_evicted_item_ids"
            )


def _known_evicted_item_ids(metadata: dict[str, object]) -> set[UUID]:
    raw_values = metadata.get("known_evicted_item_ids", ())
    if raw_values is None:
        return set()
    if not isinstance(raw_values, (tuple, list, set, frozenset)):
        raise TypeError("known_evicted_item_ids must be a collection")
    result: set[UUID] = set()
    for value in raw_values:
        if isinstance(value, UUID):
            result.add(value)
        elif isinstance(value, str):
            result.add(UUID(value))
        else:
            raise TypeError("known_evicted_item_ids must contain UUID or str")
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
    "ShortTermItemKind",
    "ShortTermItemStatus",
    "CandidateTraceStatus",
    "HypothesisStatus",
    "ActiveContextStatus",
    "EvictionReason",
    "ShortTermItem",
    "ActiveContextRef",
    "CandidateTrace",
    "WorkingHypothesis",
    "EvictionRecord",
    "ShortTermMemorySnapshot",
    "utc_now",
    "new_short_term_item_id",
    "new_active_context_id",
    "new_candidate_trace_id",
    "new_working_hypothesis_id",
    "new_eviction_id",
    "new_short_term_snapshot_id",
    "ensure_non_empty_string",
    "ensure_optional_non_empty_string",
    "ensure_probability",
    "ensure_timezone_aware",
    "create_short_term_item",
    "create_active_context_ref",
    "create_candidate_trace",
    "create_working_hypothesis",
    "create_eviction_record",
    "create_short_term_memory_snapshot",
]
