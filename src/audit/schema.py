"""Strict immutable schema objects for CEBA audit records."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from uuid import UUID, uuid4


__all__ = [
    "AuditSeverity",
    "AuditStatus",
    "AuditActorKind",
    "AuditEventKind",
    "AuditRecordKind",
    "AuditActor",
    "AuditRecordRef",
    "AuditEvent",
    "AuditBundle",
    "utc_now",
    "new_audit_event_id",
    "new_audit_bundle_id",
    "ensure_non_empty_string",
    "ensure_optional_non_empty_string",
    "ensure_positive_int",
    "ensure_non_negative_int",
    "ensure_probability",
    "ensure_finite_number",
    "ensure_timezone_aware",
    "create_audit_actor",
    "create_audit_record_ref",
    "create_audit_event",
    "create_audit_bundle",
]


_FORBIDDEN = {"M_t", "Mt", "semantic_state", "runtime_semantic_state"}


class AuditSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class AuditStatus(Enum):
    CREATED = "CREATED"
    RECORDED = "RECORDED"
    VERIFIED = "VERIFIED"
    EXPORTED = "EXPORTED"
    REJECTED = "REJECTED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class AuditActorKind(Enum):
    USER = "USER"
    SYSTEM = "SYSTEM"
    RUNTIME = "RUNTIME"
    WORLD_MODEL = "WORLD_MODEL"
    DELTA = "DELTA"
    MEMORY = "MEMORY"
    PERSISTENCE = "PERSISTENCE"
    SECURITY = "SECURITY"
    OBSERVABILITY = "OBSERVABILITY"
    UNKNOWN = "UNKNOWN"


class AuditEventKind(Enum):
    EVENT = "EVENT"
    PROVENANCE = "PROVENANCE"
    DECISION = "DECISION"
    DELTA = "DELTA"
    INVARIANT = "INVARIANT"
    EXPORT = "EXPORT"
    SECURITY = "SECURITY"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"


class AuditRecordKind(Enum):
    INPUT = "INPUT"
    PARSER = "PARSER"
    EMBEDDING = "EMBEDDING"
    DIFFERENCE = "DIFFERENCE"
    DELTA = "DELTA"
    CONSTRAINT = "CONSTRAINT"
    METRIC = "METRIC"
    DYNAMICS = "DYNAMICS"
    REGULATION = "REGULATION"
    TELOS = "TELOS"
    VALUATION = "VALUATION"
    RELATION = "RELATION"
    MEMORY = "MEMORY"
    WORLD_MODEL = "WORLD_MODEL"
    RUNTIME = "RUNTIME"
    PERSISTENCE = "PERSISTENCE"
    AUDIT = "AUDIT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class AuditActor:
    actor_id: str
    kind: AuditActorKind
    label: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        actor_id = ensure_non_empty_string(self.actor_id, "actor_id")
        _reject_name(actor_id, "actor_id")
        if not isinstance(self.kind, AuditActorKind):
            raise TypeError("kind must be AuditActorKind")
        label = ensure_optional_non_empty_string(self.label, "label")
        if label is not None:
            _reject_name(label, "label")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "actor_id", actor_id)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        return {
            "actor_id": self.actor_id,
            "kind": self.kind.value,
            "label": self.label,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AuditRecordRef:
    record_id: str
    kind: AuditRecordKind
    record_key: str | None
    source: str
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        record_id = ensure_non_empty_string(self.record_id, "record_id")
        _reject_name(record_id, "record_id")
        if not isinstance(self.kind, AuditRecordKind):
            raise TypeError("kind must be AuditRecordKind")
        record_key = ensure_optional_non_empty_string(self.record_key, "record_key")
        if record_key is not None:
            _reject_name(record_key, "record_key")
        source = ensure_non_empty_string(self.source, "source")
        _reject_name(source, "source")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "record_id", record_id)
        object.__setattr__(self, "record_key", record_key)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "kind": self.kind.value,
            "record_key": self.record_key,
            "source": self.source,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AuditEvent:
    event_id: UUID
    kind: AuditEventKind
    severity: AuditSeverity
    status: AuditStatus
    message: str
    actor: AuditActor
    record_ref: AuditRecordRef | None
    occurred_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, UUID):
            raise TypeError("event_id must be UUID")
        if not isinstance(self.kind, AuditEventKind):
            raise TypeError("kind must be AuditEventKind")
        if not isinstance(self.severity, AuditSeverity):
            raise TypeError("severity must be AuditSeverity")
        if not isinstance(self.status, AuditStatus):
            raise TypeError("status must be AuditStatus")
        message = ensure_non_empty_string(self.message, "message")
        if not isinstance(self.actor, AuditActor):
            raise TypeError("actor must be AuditActor")
        if self.record_ref is not None and not isinstance(self.record_ref, AuditRecordRef):
            raise TypeError("record_ref must be None or AuditRecordRef")
        occurred_at = ensure_timezone_aware(self.occurred_at, "occurred_at")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "occurred_at", occurred_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": str(self.event_id),
            "kind": self.kind.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "message": self.message,
            "actor": self.actor.to_dict(),
            "record_ref": None if self.record_ref is None else self.record_ref.to_dict(),
            "occurred_at": self.occurred_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    def is_error(self) -> bool:
        return self.severity in {AuditSeverity.ERROR, AuditSeverity.CRITICAL}


@dataclass(frozen=True)
class AuditBundle:
    bundle_id: UUID
    events: tuple[AuditEvent, ...]
    status: AuditStatus
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.bundle_id, UUID):
            raise TypeError("bundle_id must be UUID")
        events = _event_tuple(self.events)
        event_ids = [event.event_id for event in events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event_id values must be unique")
        if not isinstance(self.status, AuditStatus):
            raise TypeError("status must be AuditStatus")
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        return {
            "bundle_id": str(self.bundle_id),
            "events": [event.to_dict() for event in self.events],
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    def event_count(self) -> int:
        return len(self.events)

    def error_count(self) -> int:
        return sum(1 for event in self.events if event.is_error())

    def has_errors(self) -> bool:
        return self.error_count() > 0


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_audit_event_id() -> UUID:
    return uuid4()


def new_audit_bundle_id() -> UUID:
    return uuid4()


def ensure_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def ensure_optional_non_empty_string(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return ensure_non_empty_string(value, field_name)


def ensure_positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return value


def ensure_non_negative_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return value


def ensure_probability(value: float, field_name: str) -> float:
    number = ensure_finite_number(value, field_name)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return number


def ensure_finite_number(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be int or float")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def ensure_timezone_aware(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def create_audit_actor(
    *,
    actor_id: str,
    kind: AuditActorKind = AuditActorKind.UNKNOWN,
    label: str | None = None,
    metadata: dict[str, object] | None = None,
) -> AuditActor:
    return AuditActor(
        actor_id=actor_id,
        kind=kind,
        label=label,
        metadata=_merge_metadata({"source": "audit_actor"}, metadata),
    )


def create_audit_record_ref(
    *,
    record_id: str,
    kind: AuditRecordKind = AuditRecordKind.UNKNOWN,
    record_key: str | None = None,
    source: str = "unknown",
    metadata: dict[str, object] | None = None,
) -> AuditRecordRef:
    return AuditRecordRef(
        record_id=record_id,
        kind=kind,
        record_key=record_key,
        source=source,
        metadata=_merge_metadata({"source": "audit_record_ref"}, metadata),
    )


def create_audit_event(
    *,
    message: str,
    actor: AuditActor,
    kind: AuditEventKind = AuditEventKind.UNKNOWN,
    severity: AuditSeverity = AuditSeverity.INFO,
    status: AuditStatus = AuditStatus.CREATED,
    record_ref: AuditRecordRef | None = None,
    metadata: dict[str, object] | None = None,
) -> AuditEvent:
    return AuditEvent(
        event_id=new_audit_event_id(),
        kind=kind,
        severity=severity,
        status=status,
        message=message,
        actor=actor,
        record_ref=record_ref,
        occurred_at=utc_now(),
        metadata=_merge_metadata({"source": "audit_event"}, metadata),
    )


def create_audit_bundle(
    *,
    events: tuple[AuditEvent, ...] | list[AuditEvent] = (),
    status: AuditStatus = AuditStatus.CREATED,
    metadata: dict[str, object] | None = None,
) -> AuditBundle:
    return AuditBundle(
        bundle_id=new_audit_bundle_id(),
        events=tuple(events),
        status=status,
        created_at=utc_now(),
        metadata=_merge_metadata({"source": "audit_bundle"}, metadata),
    )


def _copy_metadata(value: dict[str, object]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(value)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
        if key in _FORBIDDEN:
            raise ValueError("metadata must not persist runtime semantic state")
    return copied


def _merge_metadata(
    defaults: dict[str, object],
    metadata: dict[str, object] | None,
) -> dict[str, object]:
    merged = _copy_metadata(defaults)
    if metadata is not None:
        merged.update(_copy_metadata(metadata))
    if "source" in defaults:
        merged["source"] = defaults["source"]
    return merged


def _reject_name(value: str, field_name: str) -> None:
    if value in _FORBIDDEN:
        raise ValueError(f"{field_name} must not persist runtime semantic state")


def _event_tuple(value: tuple[AuditEvent, ...] | list[AuditEvent]) -> tuple[AuditEvent, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("events must be tuple[AuditEvent, ...]")
    events = tuple(value)
    for event in events:
        if not isinstance(event, AuditEvent):
            raise TypeError("every event must be AuditEvent")
    return events
