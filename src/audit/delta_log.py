"""Deterministic local delta audit log utilities."""

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
    ensure_finite_number,
    ensure_non_empty_string,
    ensure_optional_non_empty_string,
    ensure_positive_int,
)


__all__ = [
    "DeltaStatusFilter",
    "DeltaLogPolicy",
    "DeltaAuditInput",
    "DeltaAuditQuery",
    "DeltaLogResult",
    "DeltaLog",
    "build_delta_log",
]


_FORBIDDEN = {"M_t", "Mt", "semantic_state", "runtime_semantic_state"}


class DeltaStatusFilter(Enum):
    ANY = "ANY"
    CREATED = "CREATED"
    RECORDED = "RECORDED"
    VERIFIED = "VERIFIED"
    EXPORTED = "EXPORTED"
    REJECTED = "REJECTED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class DeltaLogPolicy:
    max_events: int = 100000
    require_delta_key: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_events", ensure_positive_int(self.max_events, "max_events"))
        if not isinstance(self.require_delta_key, bool):
            raise TypeError("require_delta_key must be bool")

    def to_dict(self) -> dict[str, object]:
        return {"max_events": self.max_events, "require_delta_key": self.require_delta_key}


@dataclass(frozen=True)
class DeltaAuditInput:
    delta_key: str
    delta_type: str
    delta_score: float
    message: str
    actor_id: str
    actor_kind: AuditActorKind
    record_id: str | None = None
    record_kind: AuditRecordKind | None = None
    record_key: str | None = None
    record_source: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "delta_key", ensure_non_empty_string(self.delta_key, "delta_key"))
        object.__setattr__(self, "delta_type", ensure_non_empty_string(self.delta_type, "delta_type"))
        score = ensure_finite_number(self.delta_score, "delta_score")
        if score < 0.0:
            raise ValueError("delta_score must be non-negative")
        object.__setattr__(self, "delta_score", score)
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
            "delta_key": self.delta_key,
            "delta_type": self.delta_type,
            "delta_score": self.delta_score,
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
class DeltaAuditQuery:
    delta_key: str | None = None
    delta_type: str | None = None
    min_delta_score: float = 0.0
    status_filter: DeltaStatusFilter = DeltaStatusFilter.ANY
    limit: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(self, "delta_key", ensure_optional_non_empty_string(self.delta_key, "delta_key"))
        object.__setattr__(self, "delta_type", ensure_optional_non_empty_string(self.delta_type, "delta_type"))
        score = ensure_finite_number(self.min_delta_score, "min_delta_score")
        if score < 0.0:
            raise ValueError("min_delta_score must be non-negative")
        object.__setattr__(self, "min_delta_score", score)
        if not isinstance(self.status_filter, DeltaStatusFilter):
            raise TypeError("status_filter must be DeltaStatusFilter")
        object.__setattr__(self, "limit", ensure_positive_int(self.limit, "limit"))

    def to_dict(self) -> dict[str, object]:
        return {"delta_key": self.delta_key, "delta_type": self.delta_type, "min_delta_score": self.min_delta_score, "status_filter": self.status_filter.value, "limit": self.limit}


@dataclass(frozen=True)
class DeltaLogResult:
    events: tuple[AuditEvent, ...] = ()
    bundle: AuditBundle | None = None
    changed: bool = False
    reason: str = "delta log result"
    rejected_delta_keys: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", _event_tuple(self.events))
        if self.bundle is not None and not isinstance(self.bundle, AuditBundle):
            raise TypeError("bundle must be None or AuditBundle")
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        object.__setattr__(self, "reason", ensure_non_empty_string(self.reason, "reason"))
        rejected = _string_tuple(self.rejected_delta_keys, "rejected_delta_keys")
        reasons = _string_tuple(self.rejection_reasons, "rejection_reasons")
        if len(rejected) != len(reasons):
            raise ValueError("rejected_delta_keys and rejection_reasons must have same length")
        object.__setattr__(self, "rejected_delta_keys", rejected)
        object.__setattr__(self, "rejection_reasons", reasons)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "events": [event.to_dict() for event in self.events],
            "bundle": None if self.bundle is None else self.bundle.to_dict(),
            "changed": self.changed,
            "reason": self.reason,
            "rejected_delta_keys": list(self.rejected_delta_keys),
            "rejection_reasons": list(self.rejection_reasons),
            "metadata": dict(self.metadata),
        }


class DeltaLog:
    def __init__(self, *, events: tuple[AuditEvent, ...] | list[AuditEvent] = (), policy: DeltaLogPolicy | None = None) -> None:
        self._policy = DeltaLogPolicy() if policy is None else policy
        if not isinstance(self._policy, DeltaLogPolicy):
            raise TypeError("policy must be DeltaLogPolicy")
        self._events = _event_tuple(events)

    def events(self) -> tuple[AuditEvent, ...]:
        return self._events

    def event_count(self) -> int:
        return len(self._events)

    def append_delta(self, *, delta_input: DeltaAuditInput) -> DeltaLogResult:
        if not isinstance(delta_input, DeltaAuditInput):
            raise TypeError("delta_input must be DeltaAuditInput")
        if len(self._events) >= self._policy.max_events:
            return _result(reason="delta log limit exceeded", rejected=(delta_input.delta_key,), rejection_reasons=("delta log limit exceeded",))
        if self._policy.require_delta_key and not delta_input.delta_key:
            return _result(reason="delta key required", rejected=(delta_input.delta_key,), rejection_reasons=("delta key required",))
        actor = create_audit_actor(actor_id=delta_input.actor_id, kind=delta_input.actor_kind)
        record_ref = _record_ref(delta_input.record_id, delta_input.record_kind, delta_input.record_key, delta_input.record_source)
        event = create_audit_event(
            message=delta_input.message,
            actor=actor,
            kind=AuditEventKind.DELTA,
            severity=AuditSeverity.INFO,
            status=AuditStatus.RECORDED,
            record_ref=record_ref,
            metadata=_merge_metadata(
                {"source": "delta_log", "delta_key": delta_input.delta_key, "delta_type": delta_input.delta_type, "delta_score": delta_input.delta_score},
                delta_input.metadata,
            ),
        )
        self._events = self._events + (event,)
        return _result(events=(event,), changed=True, reason="delta recorded")

    def query_deltas(self, *, query: DeltaAuditQuery) -> DeltaLogResult:
        if not isinstance(query, DeltaAuditQuery):
            raise TypeError("query must be DeltaAuditQuery")
        events = []
        for event in self._events:
            if query.delta_key is not None and event.metadata.get("delta_key") != query.delta_key:
                continue
            if query.delta_type is not None and event.metadata.get("delta_type") != query.delta_type:
                continue
            if float(event.metadata.get("delta_score", 0.0)) < query.min_delta_score:
                continue
            if query.status_filter is not DeltaStatusFilter.ANY and event.status.value != query.status_filter.value:
                continue
            events.append(event)
        return _result(events=_sort_events(events)[: query.limit], reason="deltas queried")

    def mark_verified(self, *, event_id: str) -> DeltaLogResult:
        return self._mark_status(event_id=event_id, status=AuditStatus.VERIFIED, reason="delta verified")

    def mark_exported(self, *, event_id: str) -> DeltaLogResult:
        return self._mark_status(event_id=event_id, status=AuditStatus.EXPORTED, reason="delta exported")

    def create_bundle(self) -> DeltaLogResult:
        bundle = create_audit_bundle(events=self._events, status=AuditStatus.CREATED, metadata={"source": "delta_log_bundle"})
        return _result(events=self._events, bundle=bundle, reason="delta bundle created")

    def replace_events(self, *, events: tuple[AuditEvent, ...] | list[AuditEvent]) -> DeltaLogResult:
        self._events = _event_tuple(events)
        return _result(events=self._events, changed=True, reason="delta events replaced")

    def _mark_status(self, *, event_id: str, status: AuditStatus, reason: str) -> DeltaLogResult:
        event_id = ensure_non_empty_string(event_id, "event_id")
        for index, event in enumerate(self._events):
            if str(event.event_id) == event_id:
                updated = AuditEvent(event.event_id, event.kind, event.severity, status, event.message, event.actor, event.record_ref, event.occurred_at, event.metadata)
                self._events = self._events[:index] + (updated,) + self._events[index + 1 :]
                return _result(events=(updated,), changed=True, reason=reason)
        return _result(reason="delta missing")


def build_delta_log(*, events: tuple[AuditEvent, ...] | list[AuditEvent] = (), policy: DeltaLogPolicy | None = None) -> DeltaLog:
    return DeltaLog(events=events, policy=policy)


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
) -> DeltaLogResult:
    return DeltaLogResult(events=tuple(events), bundle=bundle, changed=changed, reason=reason, rejected_delta_keys=rejected, rejection_reasons=rejection_reasons, metadata={"source": "delta_log_result"})
