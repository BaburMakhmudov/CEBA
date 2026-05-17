"""Deterministic local decision audit log utilities."""

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
    "DecisionStatusFilter",
    "DecisionLogPolicy",
    "DecisionAuditInput",
    "DecisionAuditQuery",
    "DecisionLogResult",
    "DecisionLog",
    "build_decision_log",
]


_FORBIDDEN = {"M_t", "Mt", "semantic_state", "runtime_semantic_state"}


class DecisionStatusFilter(Enum):
    ANY = "ANY"
    CREATED = "CREATED"
    RECORDED = "RECORDED"
    VERIFIED = "VERIFIED"
    EXPORTED = "EXPORTED"
    REJECTED = "REJECTED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class DecisionLogPolicy:
    max_events: int = 100000
    require_decision_key: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_events", ensure_positive_int(self.max_events, "max_events"))
        if not isinstance(self.require_decision_key, bool):
            raise TypeError("require_decision_key must be bool")

    def to_dict(self) -> dict[str, object]:
        return {"max_events": self.max_events, "require_decision_key": self.require_decision_key}


@dataclass(frozen=True)
class DecisionAuditInput:
    decision_key: str
    decision_type: str
    message: str
    actor_id: str
    actor_kind: AuditActorKind
    record_id: str | None = None
    record_kind: AuditRecordKind | None = None
    record_key: str | None = None
    record_source: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_key", ensure_non_empty_string(self.decision_key, "decision_key"))
        object.__setattr__(self, "decision_type", ensure_non_empty_string(self.decision_type, "decision_type"))
        object.__setattr__(self, "message", ensure_non_empty_string(self.message, "message"))
        object.__setattr__(self, "actor_id", ensure_non_empty_string(self.actor_id, "actor_id"))
        if not isinstance(self.actor_kind, AuditActorKind):
            raise TypeError("actor_kind must be AuditActorKind")
        record_id = ensure_optional_non_empty_string(self.record_id, "record_id")
        record_source = ensure_optional_non_empty_string(self.record_source, "record_source")
        if record_id is not None:
            if not isinstance(self.record_kind, AuditRecordKind):
                raise TypeError("record_kind must be AuditRecordKind when record_id is provided")
            if record_source is None:
                raise ValueError("record_source is required when record_id is provided")
        elif self.record_kind is not None and not isinstance(self.record_kind, AuditRecordKind):
            raise TypeError("record_kind must be None or AuditRecordKind")
        object.__setattr__(self, "record_id", record_id)
        object.__setattr__(self, "record_key", ensure_optional_non_empty_string(self.record_key, "record_key"))
        object.__setattr__(self, "record_source", record_source)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_key": self.decision_key,
            "decision_type": self.decision_type,
            "message": self.message,
            "actor_id": self.actor_id,
            "actor_kind": self.actor_kind.value,
            "record_id": self.record_id,
            "record_kind": None if self.record_kind is None else self.record_kind.value,
            "record_key": self.record_key,
            "record_source": self.record_source,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DecisionAuditQuery:
    decision_key: str | None = None
    decision_type: str | None = None
    status_filter: DecisionStatusFilter = DecisionStatusFilter.ANY
    limit: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_key", ensure_optional_non_empty_string(self.decision_key, "decision_key"))
        object.__setattr__(self, "decision_type", ensure_optional_non_empty_string(self.decision_type, "decision_type"))
        if not isinstance(self.status_filter, DecisionStatusFilter):
            raise TypeError("status_filter must be DecisionStatusFilter")
        object.__setattr__(self, "limit", ensure_positive_int(self.limit, "limit"))

    def to_dict(self) -> dict[str, object]:
        return {"decision_key": self.decision_key, "decision_type": self.decision_type, "status_filter": self.status_filter.value, "limit": self.limit}


@dataclass(frozen=True)
class DecisionLogResult:
    events: tuple[AuditEvent, ...] = ()
    bundle: AuditBundle | None = None
    changed: bool = False
    reason: str = "decision log result"
    rejected_decision_keys: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", _event_tuple(self.events))
        if self.bundle is not None and not isinstance(self.bundle, AuditBundle):
            raise TypeError("bundle must be None or AuditBundle")
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        object.__setattr__(self, "reason", ensure_non_empty_string(self.reason, "reason"))
        rejected = _string_tuple(self.rejected_decision_keys, "rejected_decision_keys")
        reasons = _string_tuple(self.rejection_reasons, "rejection_reasons")
        if len(rejected) != len(reasons):
            raise ValueError("rejected_decision_keys and rejection_reasons must have same length")
        object.__setattr__(self, "rejected_decision_keys", rejected)
        object.__setattr__(self, "rejection_reasons", reasons)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "events": [event.to_dict() for event in self.events],
            "bundle": None if self.bundle is None else self.bundle.to_dict(),
            "changed": self.changed,
            "reason": self.reason,
            "rejected_decision_keys": list(self.rejected_decision_keys),
            "rejection_reasons": list(self.rejection_reasons),
            "metadata": dict(self.metadata),
        }


class DecisionLog:
    def __init__(self, *, events: tuple[AuditEvent, ...] | list[AuditEvent] = (), policy: DecisionLogPolicy | None = None) -> None:
        self._policy = DecisionLogPolicy() if policy is None else policy
        if not isinstance(self._policy, DecisionLogPolicy):
            raise TypeError("policy must be DecisionLogPolicy")
        self._events = _event_tuple(events)

    def events(self) -> tuple[AuditEvent, ...]:
        return self._events

    def event_count(self) -> int:
        return len(self._events)

    def append_decision(self, *, decision_input: DecisionAuditInput) -> DecisionLogResult:
        if not isinstance(decision_input, DecisionAuditInput):
            raise TypeError("decision_input must be DecisionAuditInput")
        if len(self._events) >= self._policy.max_events:
            return _result(reason="decision log limit exceeded", rejected=(decision_input.decision_key,), rejection_reasons=("decision log limit exceeded",))
        if self._policy.require_decision_key and not decision_input.decision_key:
            return _result(reason="decision key required", rejected=(decision_input.decision_key,), rejection_reasons=("decision key required",))
        actor = create_audit_actor(actor_id=decision_input.actor_id, kind=decision_input.actor_kind)
        record_ref = _record_ref(decision_input.record_id, decision_input.record_kind, decision_input.record_key, decision_input.record_source)
        event = create_audit_event(
            message=decision_input.message,
            actor=actor,
            kind=AuditEventKind.DECISION,
            severity=AuditSeverity.INFO,
            status=AuditStatus.RECORDED,
            record_ref=record_ref,
            metadata=_merge_metadata(
                {"source": "decision_log", "decision_key": decision_input.decision_key, "decision_type": decision_input.decision_type},
                decision_input.metadata,
            ),
        )
        self._events = self._events + (event,)
        return _result(events=(event,), changed=True, reason="decision recorded")

    def query_decisions(self, *, query: DecisionAuditQuery) -> DecisionLogResult:
        if not isinstance(query, DecisionAuditQuery):
            raise TypeError("query must be DecisionAuditQuery")
        events = []
        for event in self._events:
            if query.decision_key is not None and event.metadata.get("decision_key") != query.decision_key:
                continue
            if query.decision_type is not None and event.metadata.get("decision_type") != query.decision_type:
                continue
            if query.status_filter is not DecisionStatusFilter.ANY and event.status.value != query.status_filter.value:
                continue
            events.append(event)
        return _result(events=_sort_events(events)[: query.limit], reason="decisions queried")

    def mark_verified(self, *, event_id: str) -> DecisionLogResult:
        return self._mark_status(event_id=event_id, status=AuditStatus.VERIFIED, reason="decision verified")

    def mark_exported(self, *, event_id: str) -> DecisionLogResult:
        return self._mark_status(event_id=event_id, status=AuditStatus.EXPORTED, reason="decision exported")

    def create_bundle(self) -> DecisionLogResult:
        bundle = create_audit_bundle(events=self._events, status=AuditStatus.CREATED, metadata={"source": "decision_log_bundle"})
        return _result(events=self._events, bundle=bundle, reason="decision bundle created")

    def replace_events(self, *, events: tuple[AuditEvent, ...] | list[AuditEvent]) -> DecisionLogResult:
        self._events = _event_tuple(events)
        return _result(events=self._events, changed=True, reason="decision events replaced")

    def _mark_status(self, *, event_id: str, status: AuditStatus, reason: str) -> DecisionLogResult:
        event_id = ensure_non_empty_string(event_id, "event_id")
        for index, event in enumerate(self._events):
            if str(event.event_id) == event_id:
                updated = AuditEvent(event.event_id, event.kind, event.severity, status, event.message, event.actor, event.record_ref, event.occurred_at, event.metadata)
                self._events = self._events[:index] + (updated,) + self._events[index + 1 :]
                return _result(events=(updated,), changed=True, reason=reason)
        return _result(reason="decision missing")


def build_decision_log(*, events: tuple[AuditEvent, ...] | list[AuditEvent] = (), policy: DecisionLogPolicy | None = None) -> DecisionLog:
    return DecisionLog(events=events, policy=policy)


def _record_ref(record_id: str | None, record_kind: AuditRecordKind | None, record_key: str | None, record_source: str | None):
    if record_id is None:
        return None
    return create_audit_record_ref(record_id=record_id, kind=record_kind if record_kind is not None else AuditRecordKind.UNKNOWN, record_key=record_key, source=record_source if record_source is not None else "unknown")


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
    events: tuple[AuditEvent, ...] | list[AuditEvent] = (),
    bundle: AuditBundle | None = None,
    changed: bool = False,
    reason: str,
    rejected: tuple[str, ...] = (),
    rejection_reasons: tuple[str, ...] = (),
) -> DecisionLogResult:
    return DecisionLogResult(events=tuple(events), bundle=bundle, changed=changed, reason=reason, rejected_decision_keys=rejected, rejection_reasons=rejection_reasons, metadata={"source": "decision_log_result"})
