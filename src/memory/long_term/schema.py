"""Strict schema objects for CEBA Long-Term Memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from uuid import UUID, uuid4


class LongTermTraceKind(Enum):
    """Stable long-term trace category."""

    OBSERVATION_TRACE = "OBSERVATION_TRACE"
    EPISODE_TRACE = "EPISODE_TRACE"
    DECISION_TRACE = "DECISION_TRACE"
    DELTA_TRACE = "DELTA_TRACE"
    CONSTRAINT_TRACE = "CONSTRAINT_TRACE"
    TELOS_TRACE = "TELOS_TRACE"
    VALUATION_TRACE = "VALUATION_TRACE"
    RELATION_TRACE = "RELATION_TRACE"
    WORLD_MODEL_TRACE = "WORLD_MODEL_TRACE"
    CONSOLIDATED_EXPERIENCE = "CONSOLIDATED_EXPERIENCE"
    UNKNOWN = "UNKNOWN"


class LongTermTraceStatus(Enum):
    """Stable long-term trace status."""

    ACTIVE = "ACTIVE"
    VERIFIED = "VERIFIED"
    CONSOLIDATED = "CONSOLIDATED"
    REPLAYABLE = "REPLAYABLE"
    SUPERSEDED = "SUPERSEDED"
    BLOCKED = "BLOCKED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class ConsolidationStatus(Enum):
    """Stable consolidation status."""

    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    CONSOLIDATED = "CONSOLIDATED"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"
    INVALID = "INVALID"


class ProvenanceStatus(Enum):
    """Stable provenance status."""

    VALID = "VALID"
    WEAK = "WEAK"
    MISSING = "MISSING"
    CONFLICTED = "CONFLICTED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class ReplayStatus(Enum):
    """Stable replay status."""

    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"
    USED = "USED"
    EXPIRED = "EXPIRED"
    BLOCKED = "BLOCKED"
    INVALID = "INVALID"


class EligibilityStatus(Enum):
    """Stable eligibility status."""

    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"
    PENDING = "PENDING"
    BLOCKED = "BLOCKED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class LongTermTrace:
    """Verified long-term trace record after consolidation."""

    trace_id: UUID
    kind: LongTermTraceKind
    status: LongTermTraceStatus
    source_candidate_id: UUID | None
    source_ids: tuple[str, ...]
    semantic_hash: str
    confidence: float
    priority: float
    replay_count: int
    created_at: datetime
    updated_at: datetime
    superseded_at: datetime | None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.trace_id, UUID):
            raise TypeError("trace_id must be UUID")
        if not isinstance(self.kind, LongTermTraceKind):
            raise TypeError("kind must be LongTermTraceKind")
        if not isinstance(self.status, LongTermTraceStatus):
            raise TypeError("status must be LongTermTraceStatus")
        if self.source_candidate_id is not None and not isinstance(
            self.source_candidate_id,
            UUID,
        ):
            raise TypeError("source_candidate_id must be UUID or None")
        source_ids = _ensure_string_tuple(self.source_ids, "source_ids")
        _ensure_unique_strings(source_ids, "source_ids")
        semantic_hash = ensure_non_empty_string(
            self.semantic_hash,
            "semantic_hash",
        )
        confidence = ensure_probability(self.confidence, "confidence")
        priority = ensure_probability(self.priority, "priority")
        replay_count = ensure_non_negative_int(
            self.replay_count,
            "replay_count",
        )
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        updated_at = ensure_timezone_aware(self.updated_at, "updated_at")
        if updated_at < created_at:
            raise ValueError("updated_at must be >= created_at")
        superseded_at = (
            None
            if self.superseded_at is None
            else ensure_timezone_aware(self.superseded_at, "superseded_at")
        )
        if superseded_at is not None and superseded_at < created_at:
            raise ValueError("superseded_at must be >= created_at")

        object.__setattr__(self, "source_ids", source_ids)
        object.__setattr__(self, "semantic_hash", semantic_hash)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "replay_count", replay_count)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "superseded_at", superseded_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "trace_id": str(self.trace_id),
            "kind": self.kind.value,
            "status": self.status.value,
            "source_candidate_id": (
                None
                if self.source_candidate_id is None
                else str(self.source_candidate_id)
            ),
            "source_ids": self.source_ids,
            "semantic_hash": self.semantic_hash,
            "confidence": self.confidence,
            "priority": self.priority,
            "replay_count": self.replay_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "superseded_at": (
                None
                if self.superseded_at is None
                else self.superseded_at.isoformat()
            ),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ConsolidatedExperience:
    """Verified experience unit built from one or more traces."""

    experience_id: UUID
    status: ConsolidationStatus
    trace_ids: tuple[UUID, ...]
    summary_ref: str | None
    confidence: float
    utility_score: float
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.experience_id, UUID):
            raise TypeError("experience_id must be UUID")
        if not isinstance(self.status, ConsolidationStatus):
            raise TypeError("status must be ConsolidationStatus")
        trace_ids = _ensure_uuid_tuple(self.trace_ids, "trace_ids")
        _ensure_unique_uuids(trace_ids, "trace_ids")
        summary_ref = ensure_optional_non_empty_string(
            self.summary_ref,
            "summary_ref",
        )
        confidence = ensure_probability(self.confidence, "confidence")
        utility_score = ensure_probability(self.utility_score, "utility_score")
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        updated_at = ensure_timezone_aware(self.updated_at, "updated_at")
        if updated_at < created_at:
            raise ValueError("updated_at must be >= created_at")

        object.__setattr__(self, "trace_ids", trace_ids)
        object.__setattr__(self, "summary_ref", summary_ref)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "utility_score", utility_score)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "experience_id": str(self.experience_id),
            "status": self.status.value,
            "trace_ids": tuple(str(trace_id) for trace_id in self.trace_ids),
            "summary_ref": self.summary_ref,
            "confidence": self.confidence,
            "utility_score": self.utility_score,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ProvenanceCheck:
    """Validation result for trace provenance before reuse."""

    check_id: UUID
    trace_id: UUID
    status: ProvenanceStatus
    source_ids: tuple[str, ...]
    confidence: float
    checked_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.check_id, UUID):
            raise TypeError("check_id must be UUID")
        if not isinstance(self.trace_id, UUID):
            raise TypeError("trace_id must be UUID")
        if not isinstance(self.status, ProvenanceStatus):
            raise TypeError("status must be ProvenanceStatus")
        source_ids = _ensure_string_tuple(self.source_ids, "source_ids")
        _ensure_unique_strings(source_ids, "source_ids")
        confidence = ensure_probability(self.confidence, "confidence")
        checked_at = ensure_timezone_aware(self.checked_at, "checked_at")

        object.__setattr__(self, "source_ids", source_ids)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "checked_at", checked_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "check_id": str(self.check_id),
            "trace_id": str(self.trace_id),
            "status": self.status.value,
            "source_ids": self.source_ids,
            "confidence": self.confidence,
            "checked_at": self.checked_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ReplayRecord:
    """Runtime replay descriptor for a trace or experience."""

    replay_id: UUID
    trace_id: UUID | None
    experience_id: UUID | None
    status: ReplayStatus
    replay_score: float
    used_at: datetime | None
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.replay_id, UUID):
            raise TypeError("replay_id must be UUID")
        if self.trace_id is not None and not isinstance(self.trace_id, UUID):
            raise TypeError("trace_id must be UUID or None")
        if self.experience_id is not None and not isinstance(
            self.experience_id,
            UUID,
        ):
            raise TypeError("experience_id must be UUID or None")
        if self.trace_id is None and self.experience_id is None:
            raise ValueError("trace_id or experience_id must be present")
        if not isinstance(self.status, ReplayStatus):
            raise TypeError("status must be ReplayStatus")
        replay_score = ensure_probability(self.replay_score, "replay_score")
        used_at = (
            None
            if self.used_at is None
            else ensure_timezone_aware(self.used_at, "used_at")
        )
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        if used_at is not None and used_at < created_at:
            raise ValueError("used_at must be >= created_at")

        object.__setattr__(self, "replay_score", replay_score)
        object.__setattr__(self, "used_at", used_at)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "replay_id": str(self.replay_id),
            "trace_id": None if self.trace_id is None else str(self.trace_id),
            "experience_id": (
                None if self.experience_id is None else str(self.experience_id)
            ),
            "status": self.status.value,
            "replay_score": self.replay_score,
            "used_at": None if self.used_at is None else self.used_at.isoformat(),
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EligibilityDecision:
    """Decision describing trace eligibility for consolidation or replay."""

    decision_id: UUID
    trace_id: UUID
    status: EligibilityStatus
    reason: str
    confidence: float
    decided_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.decision_id, UUID):
            raise TypeError("decision_id must be UUID")
        if not isinstance(self.trace_id, UUID):
            raise TypeError("trace_id must be UUID")
        if not isinstance(self.status, EligibilityStatus):
            raise TypeError("status must be EligibilityStatus")
        reason = ensure_non_empty_string(self.reason, "reason")
        confidence = ensure_probability(self.confidence, "confidence")
        decided_at = ensure_timezone_aware(self.decided_at, "decided_at")

        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "decided_at", decided_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "decision_id": str(self.decision_id),
            "trace_id": str(self.trace_id),
            "status": self.status.value,
            "reason": self.reason,
            "confidence": self.confidence,
            "decided_at": self.decided_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LongTermMemorySnapshot:
    """Immutable runtime snapshot of long-term memory state."""

    snapshot_id: UUID
    traces: tuple[LongTermTrace, ...]
    experiences: tuple[ConsolidatedExperience, ...]
    provenance_checks: tuple[ProvenanceCheck, ...]
    replay_records: tuple[ReplayRecord, ...]
    eligibility_decisions: tuple[EligibilityDecision, ...]
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot_id, UUID):
            raise TypeError("snapshot_id must be UUID")
        traces = _ensure_traces(self.traces)
        experiences = _ensure_experiences(self.experiences)
        provenance_checks = _ensure_provenance_checks(self.provenance_checks)
        replay_records = _ensure_replay_records(self.replay_records)
        eligibility_decisions = _ensure_eligibility_decisions(
            self.eligibility_decisions
        )
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        _validate_snapshot_references(
            traces=traces,
            experiences=experiences,
            provenance_checks=provenance_checks,
            replay_records=replay_records,
            eligibility_decisions=eligibility_decisions,
        )

        object.__setattr__(self, "traces", traces)
        object.__setattr__(self, "experiences", experiences)
        object.__setattr__(self, "provenance_checks", provenance_checks)
        object.__setattr__(self, "replay_records", replay_records)
        object.__setattr__(
            self,
            "eligibility_decisions",
            eligibility_decisions,
        )
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "snapshot_id": str(self.snapshot_id),
            "traces": tuple(trace.to_dict() for trace in self.traces),
            "experiences": tuple(
                experience.to_dict() for experience in self.experiences
            ),
            "provenance_checks": tuple(
                check.to_dict() for check in self.provenance_checks
            ),
            "replay_records": tuple(
                replay.to_dict() for replay in self.replay_records
            ),
            "eligibility_decisions": tuple(
                decision.to_dict() for decision in self.eligibility_decisions
            ),
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    def trace_count(self) -> int:
        """Return trace count."""

        return len(self.traces)

    def experience_count(self) -> int:
        """Return experience count."""

        return len(self.experiences)

    def provenance_check_count(self) -> int:
        """Return provenance check count."""

        return len(self.provenance_checks)

    def replay_record_count(self) -> int:
        """Return replay record count."""

        return len(self.replay_records)

    def eligibility_decision_count(self) -> int:
        """Return eligibility decision count."""

        return len(self.eligibility_decisions)


def utc_now() -> datetime:
    """Return a timezone-aware current UTC datetime."""

    return datetime.now(timezone.utc)


def new_long_term_trace_id() -> UUID:
    """Return a new long-term trace identifier."""

    return uuid4()


def new_consolidated_experience_id() -> UUID:
    """Return a new consolidated experience identifier."""

    return uuid4()


def new_provenance_check_id() -> UUID:
    """Return a new provenance check identifier."""

    return uuid4()


def new_replay_record_id() -> UUID:
    """Return a new replay record identifier."""

    return uuid4()


def new_eligibility_decision_id() -> UUID:
    """Return a new eligibility decision identifier."""

    return uuid4()


def new_long_term_snapshot_id() -> UUID:
    """Return a new long-term memory snapshot identifier."""

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


def ensure_non_negative_int(value: int, field_name: str) -> int:
    """Validate a non-negative integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def ensure_timezone_aware(value: datetime, field_name: str) -> datetime:
    """Validate a timezone-aware datetime."""

    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def create_long_term_trace(
    *,
    semantic_hash: str,
    kind: LongTermTraceKind = LongTermTraceKind.UNKNOWN,
    status: LongTermTraceStatus = LongTermTraceStatus.VERIFIED,
    source_candidate_id: UUID | None = None,
    source_ids: tuple[str, ...] | list[str] = (),
    confidence: float = 1.0,
    priority: float = 0.5,
    replay_count: int = 0,
    superseded_at: datetime | None = None,
    metadata: dict[str, object] | None = None,
) -> LongTermTrace:
    """Create a long-term trace."""

    source_id_values = tuple(source_ids)
    now = utc_now()
    default_metadata: dict[str, object] = {
        "source": "long_term_trace",
        "source_count": len(source_id_values),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "long_term_trace"
    return LongTermTrace(
        trace_id=new_long_term_trace_id(),
        kind=kind,
        status=status,
        source_candidate_id=source_candidate_id,
        source_ids=source_id_values,
        semantic_hash=semantic_hash,
        confidence=confidence,
        priority=priority,
        replay_count=replay_count,
        created_at=now,
        updated_at=now,
        superseded_at=superseded_at,
        metadata=merged_metadata,
    )


def create_consolidated_experience(
    *,
    trace_ids: tuple[UUID, ...] | list[UUID],
    status: ConsolidationStatus = ConsolidationStatus.CONSOLIDATED,
    summary_ref: str | None = None,
    confidence: float = 1.0,
    utility_score: float = 0.5,
    metadata: dict[str, object] | None = None,
) -> ConsolidatedExperience:
    """Create a consolidated experience."""

    trace_id_values = tuple(trace_ids)
    now = utc_now()
    default_metadata: dict[str, object] = {
        "source": "consolidated_experience",
        "trace_count": len(trace_id_values),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "consolidated_experience"
    return ConsolidatedExperience(
        experience_id=new_consolidated_experience_id(),
        status=status,
        trace_ids=trace_id_values,
        summary_ref=summary_ref,
        confidence=confidence,
        utility_score=utility_score,
        created_at=now,
        updated_at=now,
        metadata=merged_metadata,
    )


def create_provenance_check(
    *,
    trace_id: UUID,
    status: ProvenanceStatus = ProvenanceStatus.VALID,
    source_ids: tuple[str, ...] | list[str] = (),
    confidence: float = 1.0,
    metadata: dict[str, object] | None = None,
) -> ProvenanceCheck:
    """Create a provenance check."""

    source_id_values = tuple(source_ids)
    default_metadata: dict[str, object] = {
        "source": "provenance_check",
        "trace_id": str(trace_id),
        "source_count": len(source_id_values),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "provenance_check"
    return ProvenanceCheck(
        check_id=new_provenance_check_id(),
        trace_id=trace_id,
        status=status,
        source_ids=source_id_values,
        confidence=confidence,
        checked_at=utc_now(),
        metadata=merged_metadata,
    )


def create_replay_record(
    *,
    trace_id: UUID | None = None,
    experience_id: UUID | None = None,
    status: ReplayStatus = ReplayStatus.ELIGIBLE,
    replay_score: float = 1.0,
    used_at: datetime | None = None,
    metadata: dict[str, object] | None = None,
) -> ReplayRecord:
    """Create a replay record."""

    default_metadata: dict[str, object] = {
        "source": "replay_record",
        "trace_id": None if trace_id is None else str(trace_id),
        "experience_id": None if experience_id is None else str(experience_id),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "replay_record"
    return ReplayRecord(
        replay_id=new_replay_record_id(),
        trace_id=trace_id,
        experience_id=experience_id,
        status=status,
        replay_score=replay_score,
        used_at=used_at,
        created_at=utc_now(),
        metadata=merged_metadata,
    )


def create_eligibility_decision(
    *,
    trace_id: UUID,
    reason: str,
    status: EligibilityStatus = EligibilityStatus.PENDING,
    confidence: float = 1.0,
    metadata: dict[str, object] | None = None,
) -> EligibilityDecision:
    """Create an eligibility decision."""

    default_metadata: dict[str, object] = {
        "source": "eligibility_decision",
        "trace_id": str(trace_id),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "eligibility_decision"
    return EligibilityDecision(
        decision_id=new_eligibility_decision_id(),
        trace_id=trace_id,
        status=status,
        reason=reason,
        confidence=confidence,
        decided_at=utc_now(),
        metadata=merged_metadata,
    )


def create_long_term_memory_snapshot(
    *,
    traces: tuple[LongTermTrace, ...] | list[LongTermTrace] = (),
    experiences: (
        tuple[ConsolidatedExperience, ...] | list[ConsolidatedExperience]
    ) = (),
    provenance_checks: (
        tuple[ProvenanceCheck, ...] | list[ProvenanceCheck]
    ) = (),
    replay_records: tuple[ReplayRecord, ...] | list[ReplayRecord] = (),
    eligibility_decisions: (
        tuple[EligibilityDecision, ...] | list[EligibilityDecision]
    ) = (),
    metadata: dict[str, object] | None = None,
) -> LongTermMemorySnapshot:
    """Create a long-term memory snapshot."""

    trace_values = tuple(traces)
    experience_values = tuple(experiences)
    provenance_values = tuple(provenance_checks)
    replay_values = tuple(replay_records)
    decision_values = tuple(eligibility_decisions)
    default_metadata: dict[str, object] = {
        "source": "long_term_memory_snapshot",
        "trace_count": len(trace_values),
        "experience_count": len(experience_values),
        "provenance_check_count": len(provenance_values),
        "replay_record_count": len(replay_values),
        "eligibility_decision_count": len(decision_values),
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "long_term_memory_snapshot"
    return LongTermMemorySnapshot(
        snapshot_id=new_long_term_snapshot_id(),
        traces=trace_values,
        experiences=experience_values,
        provenance_checks=provenance_values,
        replay_records=replay_values,
        eligibility_decisions=decision_values,
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


def _ensure_unique_ids(values: tuple[UUID, ...], field_name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} values must be unique")


def _ensure_traces(
    values: tuple[LongTermTrace, ...] | list[LongTermTrace],
) -> tuple[LongTermTrace, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("traces must be tuple or list")
    result = tuple(values)
    for trace in result:
        if not isinstance(trace, LongTermTrace):
            raise TypeError("traces must contain LongTermTrace")
    _ensure_unique_ids(tuple(trace.trace_id for trace in result), "trace_id")
    return result


def _ensure_experiences(
    values: tuple[ConsolidatedExperience, ...] | list[ConsolidatedExperience],
) -> tuple[ConsolidatedExperience, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("experiences must be tuple or list")
    result = tuple(values)
    for experience in result:
        if not isinstance(experience, ConsolidatedExperience):
            raise TypeError("experiences must contain ConsolidatedExperience")
    _ensure_unique_ids(
        tuple(experience.experience_id for experience in result),
        "experience_id",
    )
    return result


def _ensure_provenance_checks(
    values: tuple[ProvenanceCheck, ...] | list[ProvenanceCheck],
) -> tuple[ProvenanceCheck, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("provenance_checks must be tuple or list")
    result = tuple(values)
    for check in result:
        if not isinstance(check, ProvenanceCheck):
            raise TypeError("provenance_checks must contain ProvenanceCheck")
    _ensure_unique_ids(tuple(check.check_id for check in result), "check_id")
    return result


def _ensure_replay_records(
    values: tuple[ReplayRecord, ...] | list[ReplayRecord],
) -> tuple[ReplayRecord, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("replay_records must be tuple or list")
    result = tuple(values)
    for replay in result:
        if not isinstance(replay, ReplayRecord):
            raise TypeError("replay_records must contain ReplayRecord")
    _ensure_unique_ids(tuple(replay.replay_id for replay in result), "replay_id")
    return result


def _ensure_eligibility_decisions(
    values: tuple[EligibilityDecision, ...] | list[EligibilityDecision],
) -> tuple[EligibilityDecision, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("eligibility_decisions must be tuple or list")
    result = tuple(values)
    for decision in result:
        if not isinstance(decision, EligibilityDecision):
            raise TypeError(
                "eligibility_decisions must contain EligibilityDecision"
            )
    _ensure_unique_ids(
        tuple(decision.decision_id for decision in result),
        "decision_id",
    )
    return result


def _validate_snapshot_references(
    *,
    traces: tuple[LongTermTrace, ...],
    experiences: tuple[ConsolidatedExperience, ...],
    provenance_checks: tuple[ProvenanceCheck, ...],
    replay_records: tuple[ReplayRecord, ...],
    eligibility_decisions: tuple[EligibilityDecision, ...],
) -> None:
    trace_ids = {trace.trace_id for trace in traces}
    experience_ids = {experience.experience_id for experience in experiences}
    for experience in experiences:
        for trace_id in experience.trace_ids:
            if trace_id not in trace_ids:
                raise ValueError("experience trace_id must exist in traces")
    for check in provenance_checks:
        if check.trace_id not in trace_ids:
            raise ValueError("provenance_check trace_id must exist in traces")
    for replay in replay_records:
        if replay.trace_id is not None and replay.trace_id not in trace_ids:
            raise ValueError("replay_record trace_id must exist in traces")
        if (
            replay.experience_id is not None
            and replay.experience_id not in experience_ids
        ):
            raise ValueError(
                "replay_record experience_id must exist in experiences"
            )
    for decision in eligibility_decisions:
        if decision.trace_id not in trace_ids:
            raise ValueError("eligibility_decision trace_id must exist in traces")


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
    "LongTermTraceKind",
    "LongTermTraceStatus",
    "ConsolidationStatus",
    "ProvenanceStatus",
    "ReplayStatus",
    "EligibilityStatus",
    "LongTermTrace",
    "ConsolidatedExperience",
    "ProvenanceCheck",
    "ReplayRecord",
    "EligibilityDecision",
    "LongTermMemorySnapshot",
    "utc_now",
    "new_long_term_trace_id",
    "new_consolidated_experience_id",
    "new_provenance_check_id",
    "new_replay_record_id",
    "new_eligibility_decision_id",
    "new_long_term_snapshot_id",
    "ensure_non_empty_string",
    "ensure_optional_non_empty_string",
    "ensure_probability",
    "ensure_non_negative_int",
    "ensure_timezone_aware",
    "create_long_term_trace",
    "create_consolidated_experience",
    "create_provenance_check",
    "create_replay_record",
    "create_eligibility_decision",
    "create_long_term_memory_snapshot",
]
