"""Deterministic local provenance audit log utilities."""

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
    "ProvenanceStatusFilter",
    "ProvenanceLogPolicy",
    "ProvenanceInput",
    "ProvenanceQuery",
    "ProvenanceLogResult",
    "ProvenanceLog",
    "build_provenance_log",
]


_FORBIDDEN = {"M_t", "Mt", "semantic_state", "runtime_semantic_state"}


class ProvenanceStatusFilter(Enum):
    ANY = "ANY"
    CREATED = "CREATED"
    RECORDED = "RECORDED"
    VERIFIED = "VERIFIED"
    EXPORTED = "EXPORTED"
    REJECTED = "REJECTED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class ProvenanceLogPolicy:
    max_events: int = 100000
    require_record_ref: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_events", ensure_positive_int(self.max_events, "max_events"))
        if not isinstance(self.require_record_ref, bool):
            raise TypeError("require_record_ref must be bool")

    def to_dict(self) -> dict[str, object]:
        return {"max_events": self.max_events, "require_record_ref": self.require_record_ref}


@dataclass(frozen=True)
class ProvenanceInput:
    source_id: str
    source_ref: str
    message: str
    actor_id: str
    actor_kind: AuditActorKind
    record_id: str
    record_kind: AuditRecordKind
    record_source: str
    record_key: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", ensure_non_empty_string(self.source_id, "source_id"))
        object.__setattr__(self, "source_ref", ensure_non_empty_string(self.source_ref, "source_ref"))
        object.__setattr__(self, "message", ensure_non_empty_string(self.message, "message"))
        object.__setattr__(self, "actor_id", ensure_non_empty_string(self.actor_id, "actor_id"))
        if not isinstance(self.actor_kind, AuditActorKind):
            raise TypeError("actor_kind must be AuditActorKind")
        object.__setattr__(self, "record_id", ensure_non_empty_string(self.record_id, "record_id"))
        if not isinstance(self.record_kind, AuditRecordKind):
            raise TypeError("record_kind must be AuditRecordKind")
        object.__setattr__(self, "record_source", ensure_non_empty_string(self.record_source, "record_source"))
        object.__setattr__(self, "record_key", ensure_optional_non_empty_string(self.record_key, "record_key"))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "source_ref": self.source_ref,
            "message": self.message,
            "actor_id": self.actor_id,
            "actor_kind": self.actor_kind.value,
            "record_id": self.record_id,
            "record_kind": self.record_kind.value,
            "record_source": self.record_source,
            "record_key": self.record_key,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ProvenanceQuery:
    source_id: str | None = None
    record_id: str | None = None
    status_filter: ProvenanceStatusFilter = ProvenanceStatusFilter.ANY
    limit: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", ensure_optional_non_empty_string(self.source_id, "source_id"))
        object.__setattr__(self, "record_id", ensure_optional_non_empty_string(self.record_id, "record_id"))
        if not isinstance(self.status_filter, ProvenanceStatusFilter):
            raise TypeError("status_filter must be ProvenanceStatusFilter")
        object.__setattr__(self, "limit", ensure_positive_int(self.limit, "limit"))

    def to_dict(self) -> dict[str, object]:
        return {"source_id": self.source_id, "record_id": self.record_id, "status_filter": self.status_filter.value, "limit": self.limit}


@dataclass(frozen=True)
class ProvenanceLogResult:
    events: tuple[AuditEvent, ...] = ()
    bundle: AuditBundle | None = None
    changed: bool = False
    reason: str = "provenance log result"
    rejected_sources: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", _event_tuple(self.events))
        if self.bundle is not None and not isinstance(self.bundle, AuditBundle):
            raise TypeError("bundle must be None or AuditBundle")
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        object.__setattr__(self, "reason", ensure_non_empty_string(self.reason, "reason"))
        rejected = _string_tuple(self.rejected_sources, "rejected_sources")
        reasons = _string_tuple(self.rejection_reasons, "rejection_reasons")
        if len(rejected) != len(reasons):
            raise ValueError("rejected_sources and rejection_reasons must have same length")
        object.__setattr__(self, "rejected_sources", rejected)
        object.__setattr__(self, "rejection_reasons", reasons)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "events": [event.to_dict() for event in self.events],
            "bundle": None if self.bundle is None else self.bundle.to_dict(),
            "changed": self.changed,
            "reason": self.reason,
            "rejected_sources": list(self.rejected_sources),
            "rejection_reasons": list(self.rejection_reasons),
            "metadata": dict(self.metadata),
        }


class ProvenanceLog:
    def __init__(self, *, events: tuple[AuditEvent, ...] | list[AuditEvent] = (), policy: ProvenanceLogPolicy | None = None) -> None:
        self._policy = ProvenanceLogPolicy() if policy is None else policy
        if not isinstance(self._policy, ProvenanceLogPolicy):
            raise TypeError("policy must be ProvenanceLogPolicy")
        self._events = _event_tuple(events)

    def events(self) -> tuple[AuditEvent, ...]:
        return self._events

    def event_count(self) -> int:
        return len(self._events)

    def append_provenance(self, *, provenance_input: ProvenanceInput) -> ProvenanceLogResult:
        if not isinstance(provenance_input, ProvenanceInput):
            raise TypeError("provenance_input must be ProvenanceInput")
        if len(self._events) >= self._policy.max_events:
            return _result(reason="provenance log limit exceeded", rejected=(provenance_input.source_id,), rejection_reasons=("provenance log limit exceeded",))
        actor = create_audit_actor(actor_id=provenance_input.actor_id, kind=provenance_input.actor_kind)
        record_ref = create_audit_record_ref(
            record_id=provenance_input.record_id,
            kind=provenance_input.record_kind,
            record_key=provenance_input.record_key,
            source=provenance_input.record_source,
        )
        event = create_audit_event(
            message=provenance_input.message,
            actor=actor,
            kind=AuditEventKind.PROVENANCE,
            severity=AuditSeverity.INFO,
            status=AuditStatus.RECORDED,
            record_ref=record_ref,
            metadata=_merge_metadata(
                {"source": "provenance_log", "source_id": provenance_input.source_id, "source_ref": provenance_input.source_ref},
                provenance_input.metadata,
            ),
        )
        self._events = self._events + (event,)
        return _result(events=(event,), changed=True, reason="provenance recorded")

    def query_provenance(self, *, query: ProvenanceQuery) -> ProvenanceLogResult:
        if not isinstance(query, ProvenanceQuery):
            raise TypeError("query must be ProvenanceQuery")
        events = []
        for event in self._events:
            if query.source_id is not None and event.metadata.get("source_id") != query.source_id:
                continue
            if query.record_id is not None and (event.record_ref is None or event.record_ref.record_id != query.record_id):
                continue
            if query.status_filter is not ProvenanceStatusFilter.ANY and event.status.value != query.status_filter.value:
                continue
            events.append(event)
        return _result(events=_sort_events(events)[: query.limit], reason="provenance queried")

    def mark_verified(self, *, event_id: str) -> ProvenanceLogResult:
        return self._mark_status(event_id=event_id, status=AuditStatus.VERIFIED, reason="provenance verified")

    def mark_exported(self, *, event_id: str) -> ProvenanceLogResult:
        return self._mark_status(event_id=event_id, status=AuditStatus.EXPORTED, reason="provenance exported")

    def create_bundle(self) -> ProvenanceLogResult:
        bundle = create_audit_bundle(events=self._events, status=AuditStatus.CREATED, metadata={"source": "provenance_log_bundle"})
        return _result(events=self._events, bundle=bundle, reason="provenance bundle created")

    def replace_events(self, *, events: tuple[AuditEvent, ...] | list[AuditEvent]) -> ProvenanceLogResult:
        self._events = _event_tuple(events)
        return _result(events=self._events, changed=True, reason="provenance events replaced")

    def _mark_status(self, *, event_id: str, status: AuditStatus, reason: str) -> ProvenanceLogResult:
        event_id = ensure_non_empty_string(event_id, "event_id")
        for index, event in enumerate(self._events):
            if str(event.event_id) == event_id:
                updated = AuditEvent(event.event_id, event.kind, event.severity, status, event.message, event.actor, event.record_ref, event.occurred_at, event.metadata)
                self._events = self._events[:index] + (updated,) + self._events[index + 1 :]
                return _result(events=(updated,), changed=True, reason=reason)
        return _result(reason="provenance missing")


def build_provenance_log(*, events: tuple[AuditEvent, ...] | list[AuditEvent] = (), policy: ProvenanceLogPolicy | None = None) -> ProvenanceLog:
    return ProvenanceLog(events=events, policy=policy)


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
) -> ProvenanceLogResult:
    return ProvenanceLogResult(events=tuple(events), bundle=bundle, changed=changed, reason=reason, rejected_sources=rejected, rejection_reasons=rejection_reasons, metadata={"source": "provenance_log_result"})
