"""Deterministic local event audit log utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ceba.audit.schema import (
    AuditActorKind,
    AuditBundle,
    AuditEvent,
    AuditEventKind,
    AuditRecordKind,
    AuditSeverity,
    AuditStatus,
    create_audit_actor,
    create_audit_bundle,
    create_audit_event,
    create_audit_record_ref,
    ensure_non_empty_string,
    ensure_optional_non_empty_string,
    ensure_positive_int,
)


__all__ = [
    "EventAuditStatusFilter",
    "EventAuditLogPolicy",
    "EventAuditInput",
    "EventAuditQuery",
    "EventAuditLogResult",
    "EventAuditLog",
    "build_event_audit_log",
]


_FORBIDDEN = {"M_t", "Mt", "semantic_state", "runtime_semantic_state"}


class EventAuditStatusFilter(Enum):
    ANY = "ANY"
    CREATED = "CREATED"
    RECORDED = "RECORDED"
    VERIFIED = "VERIFIED"
    EXPORTED = "EXPORTED"
    REJECTED = "REJECTED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class EventAuditLogPolicy:
    max_events: int = 100000
    allow_duplicate_event_messages: bool = True
    reject_empty_metadata: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_events", ensure_positive_int(self.max_events, "max_events"))
        _ensure_bool(self.allow_duplicate_event_messages, "allow_duplicate_event_messages")
        _ensure_bool(self.reject_empty_metadata, "reject_empty_metadata")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_events": self.max_events,
            "allow_duplicate_event_messages": self.allow_duplicate_event_messages,
            "reject_empty_metadata": self.reject_empty_metadata,
        }


@dataclass(frozen=True)
class EventAuditInput:
    message: str
    severity: AuditSeverity
    actor_id: str
    actor_kind: AuditActorKind
    actor_label: str | None = None
    record_id: str | None = None
    record_kind: AuditRecordKind | None = None
    record_key: str | None = None
    record_source: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "message", ensure_non_empty_string(self.message, "message"))
        if not isinstance(self.severity, AuditSeverity):
            raise TypeError("severity must be AuditSeverity")
        object.__setattr__(self, "actor_id", ensure_non_empty_string(self.actor_id, "actor_id"))
        if not isinstance(self.actor_kind, AuditActorKind):
            raise TypeError("actor_kind must be AuditActorKind")
        object.__setattr__(self, "actor_label", ensure_optional_non_empty_string(self.actor_label, "actor_label"))
        record_id = ensure_optional_non_empty_string(self.record_id, "record_id")
        record_source = ensure_optional_non_empty_string(self.record_source, "record_source")
        record_key = ensure_optional_non_empty_string(self.record_key, "record_key")
        if record_id is not None:
            if not isinstance(self.record_kind, AuditRecordKind):
                raise TypeError("record_kind must be AuditRecordKind when record_id is provided")
            if record_source is None:
                raise ValueError("record_source is required when record_id is provided")
        elif self.record_kind is not None and not isinstance(self.record_kind, AuditRecordKind):
            raise TypeError("record_kind must be None or AuditRecordKind")
        object.__setattr__(self, "record_id", record_id)
        object.__setattr__(self, "record_key", record_key)
        object.__setattr__(self, "record_source", record_source)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "message": self.message,
            "severity": self.severity.value,
            "actor_id": self.actor_id,
            "actor_kind": self.actor_kind.value,
            "actor_label": self.actor_label,
            "record_id": self.record_id,
            "record_kind": None if self.record_kind is None else self.record_kind.value,
            "record_key": self.record_key,
            "record_source": self.record_source,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EventAuditQuery:
    severity: AuditSeverity | None = None
    status_filter: EventAuditStatusFilter = EventAuditStatusFilter.ANY
    actor_id: str | None = None
    limit: int = 100

    def __post_init__(self) -> None:
        if self.severity is not None and not isinstance(self.severity, AuditSeverity):
            raise TypeError("severity must be None or AuditSeverity")
        if not isinstance(self.status_filter, EventAuditStatusFilter):
            raise TypeError("status_filter must be EventAuditStatusFilter")
        object.__setattr__(self, "actor_id", ensure_optional_non_empty_string(self.actor_id, "actor_id"))
        object.__setattr__(self, "limit", ensure_positive_int(self.limit, "limit"))

    def to_dict(self) -> dict[str, object]:
        return {
            "severity": None if self.severity is None else self.severity.value,
            "status_filter": self.status_filter.value,
            "actor_id": self.actor_id,
            "limit": self.limit,
        }


@dataclass(frozen=True)
class EventAuditLogResult:
    events: tuple[AuditEvent, ...] = ()
    bundle: AuditBundle | None = None
    changed: bool = False
    reason: str = "event audit result"
    rejected_messages: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        events = _event_tuple(self.events)
        if self.bundle is not None and not isinstance(self.bundle, AuditBundle):
            raise TypeError("bundle must be None or AuditBundle")
        _ensure_bool(self.changed, "changed")
        reason = ensure_non_empty_string(self.reason, "reason")
        rejected = _string_tuple(self.rejected_messages, "rejected_messages")
        reasons = _string_tuple(self.rejection_reasons, "rejection_reasons")
        if len(rejected) != len(reasons):
            raise ValueError("rejected_messages and rejection_reasons must have same length")
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "rejected_messages", rejected)
        object.__setattr__(self, "rejection_reasons", reasons)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "events": [event.to_dict() for event in self.events],
            "bundle": None if self.bundle is None else self.bundle.to_dict(),
            "changed": self.changed,
            "reason": self.reason,
            "rejected_messages": list(self.rejected_messages),
            "rejection_reasons": list(self.rejection_reasons),
            "metadata": dict(self.metadata),
        }


class EventAuditLog:
    def __init__(
        self,
        *,
        events: tuple[AuditEvent, ...] | list[AuditEvent] = (),
        policy: EventAuditLogPolicy | None = None,
    ) -> None:
        self._policy = EventAuditLogPolicy() if policy is None else policy
        if not isinstance(self._policy, EventAuditLogPolicy):
            raise TypeError("policy must be EventAuditLogPolicy")
        self._events = _event_tuple(events)

    def events(self) -> tuple[AuditEvent, ...]:
        return self._events

    def event_count(self) -> int:
        return len(self._events)

    def append_event(self, *, event_input: EventAuditInput) -> EventAuditLogResult:
        if not isinstance(event_input, EventAuditInput):
            raise TypeError("event_input must be EventAuditInput")
        if len(self._events) >= self._policy.max_events:
            return _result(reason="event audit log limit exceeded", rejected=(event_input.message,), rejection_reasons=("event audit log limit exceeded",))
        if self._policy.reject_empty_metadata and not event_input.metadata:
            return _result(reason="event audit metadata is empty", rejected=(event_input.message,), rejection_reasons=("event audit metadata is empty",))
        if not self._policy.allow_duplicate_event_messages and any(event.message == event_input.message for event in self._events):
            return _result(reason="event audit message duplicated", rejected=(event_input.message,), rejection_reasons=("event audit message duplicated",))
        actor = create_audit_actor(actor_id=event_input.actor_id, kind=event_input.actor_kind, label=event_input.actor_label)
        record_ref = None
        if event_input.record_id is not None:
            record_ref = create_audit_record_ref(
                record_id=event_input.record_id,
                kind=event_input.record_kind if event_input.record_kind is not None else AuditRecordKind.UNKNOWN,
                record_key=event_input.record_key,
                source=event_input.record_source if event_input.record_source is not None else "unknown",
            )
        event = create_audit_event(
            message=event_input.message,
            actor=actor,
            kind=AuditEventKind.EVENT,
            severity=event_input.severity,
            status=AuditStatus.RECORDED,
            record_ref=record_ref,
            metadata=_merge_metadata({"source": "event_audit_log", "event_kind": AuditEventKind.EVENT.value}, event_input.metadata),
        )
        self._events = self._events + (event,)
        return _result(events=(event,), changed=True, reason="event audit recorded")

    def query_events(self, *, query: EventAuditQuery) -> EventAuditLogResult:
        if not isinstance(query, EventAuditQuery):
            raise TypeError("query must be EventAuditQuery")
        events = []
        for event in self._events:
            if query.severity is not None and event.severity is not query.severity:
                continue
            if query.actor_id is not None and event.actor.actor_id != query.actor_id:
                continue
            if query.status_filter is not EventAuditStatusFilter.ANY and event.status.value != query.status_filter.value:
                continue
            events.append(event)
        return _result(events=tuple(_sort_events(events)[: query.limit]), reason="event audits queried")

    def mark_verified(self, *, event_id: str) -> EventAuditLogResult:
        return self._mark_status(event_id=event_id, status=AuditStatus.VERIFIED, reason="event audit verified")

    def mark_exported(self, *, event_id: str) -> EventAuditLogResult:
        return self._mark_status(event_id=event_id, status=AuditStatus.EXPORTED, reason="event audit exported")

    def create_bundle(self) -> EventAuditLogResult:
        bundle = create_audit_bundle(events=self._events, status=AuditStatus.CREATED, metadata={"source": "event_audit_log_bundle"})
        return _result(events=self._events, bundle=bundle, changed=False, reason="event audit bundle created")

    def replace_events(self, *, events: tuple[AuditEvent, ...] | list[AuditEvent]) -> EventAuditLogResult:
        self._events = _event_tuple(events)
        return _result(events=self._events, changed=True, reason="event audits replaced")

    def _mark_status(self, *, event_id: str, status: AuditStatus, reason: str) -> EventAuditLogResult:
        event_id = ensure_non_empty_string(event_id, "event_id")
        for index, event in enumerate(self._events):
            if str(event.event_id) == event_id:
                updated = AuditEvent(
                    event_id=event.event_id,
                    kind=event.kind,
                    severity=event.severity,
                    status=status,
                    message=event.message,
                    actor=event.actor,
                    record_ref=event.record_ref,
                    occurred_at=event.occurred_at,
                    metadata=event.metadata,
                )
                self._events = self._events[:index] + (updated,) + self._events[index + 1 :]
                return _result(events=(updated,), changed=True, reason=reason)
        return _result(reason="event audit missing")


def build_event_audit_log(
    *,
    events: tuple[AuditEvent, ...] | list[AuditEvent] = (),
    policy: EventAuditLogPolicy | None = None,
) -> EventAuditLog:
    return EventAuditLog(events=events, policy=policy)


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


def _merge_metadata(defaults: dict[str, object], metadata: dict[str, object] | None) -> dict[str, object]:
    merged = _copy_metadata(defaults)
    if metadata is not None:
        merged.update(_copy_metadata(metadata))
    if "source" in defaults:
        merged["source"] = defaults["source"]
    return merged


def _ensure_bool(value: bool, field_name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be bool")


def _event_tuple(value: tuple[AuditEvent, ...] | list[AuditEvent]) -> tuple[AuditEvent, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("events must be tuple[AuditEvent, ...]")
    events = tuple(value)
    for event in events:
        if not isinstance(event, AuditEvent):
            raise TypeError("every event must be AuditEvent")
    return events


def _string_tuple(value: tuple[str, ...] | list[str], field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple[str, ...]")
    return tuple(ensure_non_empty_string(item, field_name) for item in value)


def _sort_events(events: list[AuditEvent] | tuple[AuditEvent, ...]) -> tuple[AuditEvent, ...]:
    return tuple(sorted(events, key=lambda event: (event.occurred_at, str(event.event_id)), reverse=True))


def _result(
    *,
    events: tuple[AuditEvent, ...] = (),
    bundle: AuditBundle | None = None,
    changed: bool = False,
    reason: str,
    rejected: tuple[str, ...] = (),
    rejection_reasons: tuple[str, ...] = (),
) -> EventAuditLogResult:
    return EventAuditLogResult(
        events=events,
        bundle=bundle,
        changed=changed,
        reason=reason,
        rejected_messages=rejected,
        rejection_reasons=rejection_reasons,
        metadata={"source": "event_audit_log_result"},
    )
