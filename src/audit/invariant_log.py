"""Deterministic local invariant audit log utilities."""

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
    "InvariantName",
    "InvariantStatusFilter",
    "InvariantLogPolicy",
    "InvariantAuditInput",
    "InvariantAuditQuery",
    "InvariantLogResult",
    "InvariantLog",
    "build_invariant_log",
]


_FORBIDDEN = {"M_t", "Mt", "semantic_state", "runtime_semantic_state"}


class InvariantName(Enum):
    WORLD_MODEL_BEFORE_DELTA = "WORLD_MODEL_BEFORE_DELTA"
    DELTA_ONLY_WRITER = "DELTA_ONLY_WRITER"
    M_T_NOT_PERSISTED = "M_T_NOT_PERSISTED"
    VECTOR_DB_NOT_MEMORY = "VECTOR_DB_NOT_MEMORY"
    STM_NOT_DATABASE = "STM_NOT_DATABASE"
    TEMPERATURE_NOT_MEMORY = "TEMPERATURE_NOT_MEMORY"
    UNKNOWN = "UNKNOWN"


class InvariantStatusFilter(Enum):
    ANY = "ANY"
    CREATED = "CREATED"
    RECORDED = "RECORDED"
    VERIFIED = "VERIFIED"
    EXPORTED = "EXPORTED"
    REJECTED = "REJECTED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class InvariantLogPolicy:
    max_events: int = 100000
    require_known_invariant: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_events", ensure_positive_int(self.max_events, "max_events"))
        if not isinstance(self.require_known_invariant, bool):
            raise TypeError("require_known_invariant must be bool")

    def to_dict(self) -> dict[str, object]:
        return {"max_events": self.max_events, "require_known_invariant": self.require_known_invariant}


@dataclass(frozen=True)
class InvariantAuditInput:
    invariant_name: InvariantName
    passed: bool
    message: str
    actor_id: str
    actor_kind: AuditActorKind
    record_id: str | None = None
    record_kind: AuditRecordKind | None = None
    record_key: str | None = None
    record_source: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.invariant_name, InvariantName):
            raise TypeError("invariant_name must be InvariantName")
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be bool")
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
            "invariant_name": self.invariant_name.value,
            "passed": self.passed,
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
class InvariantAuditQuery:
    invariant_name: InvariantName | None = None
    passed: bool | None = None
    status_filter: InvariantStatusFilter = InvariantStatusFilter.ANY
    limit: int = 100

    def __post_init__(self) -> None:
        if self.invariant_name is not None and not isinstance(self.invariant_name, InvariantName):
            raise TypeError("invariant_name must be None or InvariantName")
        if self.passed is not None and not isinstance(self.passed, bool):
            raise TypeError("passed must be None or bool")
        if not isinstance(self.status_filter, InvariantStatusFilter):
            raise TypeError("status_filter must be InvariantStatusFilter")
        object.__setattr__(self, "limit", ensure_positive_int(self.limit, "limit"))

    def to_dict(self) -> dict[str, object]:
        return {"invariant_name": None if self.invariant_name is None else self.invariant_name.value, "passed": self.passed, "status_filter": self.status_filter.value, "limit": self.limit}


@dataclass(frozen=True)
class InvariantLogResult:
    events: tuple[AuditEvent, ...] = ()
    bundle: AuditBundle | None = None
    changed: bool = False
    reason: str = "invariant log result"
    rejected_invariants: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", _event_tuple(self.events))
        if self.bundle is not None and not isinstance(self.bundle, AuditBundle):
            raise TypeError("bundle must be None or AuditBundle")
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        object.__setattr__(self, "reason", ensure_non_empty_string(self.reason, "reason"))
        rejected = _string_tuple(self.rejected_invariants, "rejected_invariants")
        reasons = _string_tuple(self.rejection_reasons, "rejection_reasons")
        if len(rejected) != len(reasons):
            raise ValueError("rejected_invariants and rejection_reasons must have same length")
        object.__setattr__(self, "rejected_invariants", rejected)
        object.__setattr__(self, "rejection_reasons", reasons)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "events": [event.to_dict() for event in self.events],
            "bundle": None if self.bundle is None else self.bundle.to_dict(),
            "changed": self.changed,
            "reason": self.reason,
            "rejected_invariants": list(self.rejected_invariants),
            "rejection_reasons": list(self.rejection_reasons),
            "metadata": dict(self.metadata),
        }


class InvariantLog:
    def __init__(self, *, events: tuple[AuditEvent, ...] | list[AuditEvent] = (), policy: InvariantLogPolicy | None = None) -> None:
        self._policy = InvariantLogPolicy() if policy is None else policy
        if not isinstance(self._policy, InvariantLogPolicy):
            raise TypeError("policy must be InvariantLogPolicy")
        self._events = _event_tuple(events)

    def events(self) -> tuple[AuditEvent, ...]:
        return self._events

    def event_count(self) -> int:
        return len(self._events)

    def append_invariant(self, *, invariant_input: InvariantAuditInput) -> InvariantLogResult:
        if not isinstance(invariant_input, InvariantAuditInput):
            raise TypeError("invariant_input must be InvariantAuditInput")
        if len(self._events) >= self._policy.max_events:
            return _result(reason="invariant log limit exceeded", rejected=(invariant_input.invariant_name.value,), rejection_reasons=("invariant log limit exceeded",))
        if self._policy.require_known_invariant and invariant_input.invariant_name is InvariantName.UNKNOWN:
            return _result(reason="known invariant required", rejected=(invariant_input.invariant_name.value,), rejection_reasons=("known invariant required",))
        actor = create_audit_actor(actor_id=invariant_input.actor_id, kind=invariant_input.actor_kind)
        record_ref = _record_ref(invariant_input.record_id, invariant_input.record_kind, invariant_input.record_key, invariant_input.record_source)
        event = create_audit_event(
            message=invariant_input.message,
            actor=actor,
            kind=AuditEventKind.INVARIANT,
            severity=AuditSeverity.INFO if invariant_input.passed else AuditSeverity.ERROR,
            status=AuditStatus.RECORDED,
            record_ref=record_ref,
            metadata=_merge_metadata(
                {"source": "invariant_log", "invariant_name": invariant_input.invariant_name.value, "passed": invariant_input.passed},
                invariant_input.metadata,
            ),
        )
        self._events = self._events + (event,)
        return _result(events=(event,), changed=True, reason="invariant recorded")

    def query_invariants(self, *, query: InvariantAuditQuery) -> InvariantLogResult:
        if not isinstance(query, InvariantAuditQuery):
            raise TypeError("query must be InvariantAuditQuery")
        events = []
        for event in self._events:
            if query.invariant_name is not None and event.metadata.get("invariant_name") != query.invariant_name.value:
                continue
            if query.passed is not None and event.metadata.get("passed") is not query.passed:
                continue
            if query.status_filter is not InvariantStatusFilter.ANY and event.status.value != query.status_filter.value:
                continue
            events.append(event)
        return _result(events=_sort_events(events)[: query.limit], reason="invariants queried")

    def mark_verified(self, *, event_id: str) -> InvariantLogResult:
        return self._mark_status(event_id=event_id, status=AuditStatus.VERIFIED, reason="invariant verified")

    def mark_exported(self, *, event_id: str) -> InvariantLogResult:
        return self._mark_status(event_id=event_id, status=AuditStatus.EXPORTED, reason="invariant exported")

    def create_bundle(self) -> InvariantLogResult:
        bundle = create_audit_bundle(events=self._events, status=AuditStatus.CREATED, metadata={"source": "invariant_log_bundle"})
        return _result(events=self._events, bundle=bundle, reason="invariant bundle created")

    def replace_events(self, *, events: tuple[AuditEvent, ...] | list[AuditEvent]) -> InvariantLogResult:
        self._events = _event_tuple(events)
        return _result(events=self._events, changed=True, reason="invariant events replaced")

    def _mark_status(self, *, event_id: str, status: AuditStatus, reason: str) -> InvariantLogResult:
        event_id = ensure_non_empty_string(event_id, "event_id")
        for index, event in enumerate(self._events):
            if str(event.event_id) == event_id:
                updated = AuditEvent(event.event_id, event.kind, event.severity, status, event.message, event.actor, event.record_ref, event.occurred_at, event.metadata)
                self._events = self._events[:index] + (updated,) + self._events[index + 1 :]
                return _result(events=(updated,), changed=True, reason=reason)
        return _result(reason="invariant missing")


def build_invariant_log(*, events: tuple[AuditEvent, ...] | list[AuditEvent] = (), policy: InvariantLogPolicy | None = None) -> InvariantLog:
    return InvariantLog(events=events, policy=policy)


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
) -> InvariantLogResult:
    return InvariantLogResult(events=tuple(events), bundle=bundle, changed=changed, reason=reason, rejected_invariants=rejected, rejection_reasons=rejection_reasons, metadata={"source": "invariant_log_result"})
