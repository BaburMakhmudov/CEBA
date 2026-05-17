"""Deterministic local tracing descriptors for CEBA observability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from ceba.observability.logging import ensure_non_empty_string, ensure_optional_non_empty_string, ensure_positive_int, ensure_timezone_aware, utc_now


__all__ = [
    "TraceSpanKind",
    "TraceSpanStatus",
    "TraceStatus",
    "TraceSpan",
    "TraceContext",
    "TraceRecorderPolicy",
    "TraceRecorderResult",
    "TraceRecorder",
    "new_trace_id",
    "new_span_id",
    "create_trace_span",
    "create_trace_context",
    "build_trace_recorder",
]

_FORBIDDEN_STATE = {"M_t", "Mt", "semantic_state", "runtime_semantic_state"}
_FORBIDDEN_SECRET = {"password", "raw_password", "secret", "raw_secret", "api_key", "token", "access_token", "refresh_token", "private_key", "credential"}


class TraceSpanKind(Enum):
    INTERNAL = "INTERNAL"
    SERVER = "SERVER"
    CLIENT = "CLIENT"
    PRODUCER = "PRODUCER"
    CONSUMER = "CONSUMER"
    UNKNOWN = "UNKNOWN"


class TraceSpanStatus(Enum):
    CREATED = "CREATED"
    STARTED = "STARTED"
    ENDED = "ENDED"
    FAILED = "FAILED"
    EXPORTED = "EXPORTED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class TraceStatus(Enum):
    CREATED = "CREATED"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    EXPORTED = "EXPORTED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class TraceSpan:
    trace_id: UUID
    span_id: UUID
    parent_span_id: UUID | None
    name: str
    kind: TraceSpanKind
    status: TraceSpanStatus
    started_at: datetime
    ended_at: datetime | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.trace_id, UUID) or not isinstance(self.span_id, UUID):
            raise TypeError("trace_id and span_id must be UUID")
        if self.parent_span_id is not None and not isinstance(self.parent_span_id, UUID):
            raise TypeError("parent_span_id must be None or UUID")
        name = ensure_non_empty_string(self.name, "name")
        _reject_name(name, "name")
        if not isinstance(self.kind, TraceSpanKind):
            raise TypeError("kind must be TraceSpanKind")
        if not isinstance(self.status, TraceSpanStatus):
            raise TypeError("status must be TraceSpanStatus")
        started_at = ensure_timezone_aware(self.started_at, "started_at")
        ended_at = None if self.ended_at is None else ensure_timezone_aware(self.ended_at, "ended_at")
        if ended_at is not None and ended_at < started_at:
            raise ValueError("ended_at must be >= started_at")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(self, "ended_at", ended_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {"trace_id": str(self.trace_id), "span_id": str(self.span_id), "parent_span_id": None if self.parent_span_id is None else str(self.parent_span_id), "name": self.name, "kind": self.kind.value, "status": self.status.value, "started_at": self.started_at.isoformat(), "ended_at": None if self.ended_at is None else self.ended_at.isoformat(), "metadata": dict(self.metadata)}

    def duration_seconds(self) -> float | None:
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()

    def is_terminal(self) -> bool:
        return self.status in {TraceSpanStatus.ENDED, TraceSpanStatus.FAILED, TraceSpanStatus.EXPORTED, TraceSpanStatus.INVALID}


@dataclass(frozen=True)
class TraceContext:
    trace_id: UUID
    spans: tuple[TraceSpan, ...]
    status: TraceStatus
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.trace_id, UUID):
            raise TypeError("trace_id must be UUID")
        spans = _span_tuple(self.spans)
        ids = {span.span_id for span in spans}
        for span in spans:
            if span.trace_id != self.trace_id:
                raise ValueError("every span.trace_id must equal trace_id")
            if span.parent_span_id is not None and span.parent_span_id not in ids:
                raise ValueError("parent_span_id must exist in spans")
        if not isinstance(self.status, TraceStatus):
            raise TypeError("status must be TraceStatus")
        object.__setattr__(self, "spans", spans)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {"trace_id": str(self.trace_id), "spans": [span.to_dict() for span in self.spans], "status": self.status.value, "metadata": dict(self.metadata)}

    def span_count(self) -> int:
        return len(self.spans)

    def root_spans(self) -> tuple[TraceSpan, ...]:
        return tuple(span for span in self.spans if span.parent_span_id is None)


@dataclass(frozen=True)
class TraceRecorderPolicy:
    max_spans: int = 100000
    allow_orphan_spans: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_spans", ensure_positive_int(self.max_spans, "max_spans"))
        if not isinstance(self.allow_orphan_spans, bool):
            raise TypeError("allow_orphan_spans must be bool")


@dataclass(frozen=True)
class TraceRecorderResult:
    context: TraceContext
    changed: bool
    reason: str
    rejected_span_ids: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.context, TraceContext):
            raise TypeError("context must be TraceContext")
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        rejected = _string_tuple(self.rejected_span_ids, "rejected_span_ids")
        reasons = _string_tuple(self.rejection_reasons, "rejection_reasons")
        if len(rejected) != len(reasons):
            raise ValueError("rejected_span_ids and rejection_reasons must have same length")
        object.__setattr__(self, "reason", ensure_non_empty_string(self.reason, "reason"))
        object.__setattr__(self, "rejected_span_ids", rejected)
        object.__setattr__(self, "rejection_reasons", reasons)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))


class TraceRecorder:
    def __init__(self, *, context: TraceContext | None = None, policy: TraceRecorderPolicy | None = None) -> None:
        self._policy = TraceRecorderPolicy() if policy is None else policy
        if not isinstance(self._policy, TraceRecorderPolicy):
            raise TypeError("policy must be TraceRecorderPolicy")
        self._context = create_trace_context() if context is None else context
        if not isinstance(self._context, TraceContext):
            raise TypeError("context must be TraceContext")

    def context(self) -> TraceContext:
        return self._context

    def start_span(self, *, name: str, kind: TraceSpanKind = TraceSpanKind.INTERNAL, parent_span_id: UUID | None = None, metadata: dict[str, object] | None = None) -> TraceRecorderResult:
        if len(self._context.spans) >= self._policy.max_spans:
            return _result(self._context, False, "trace span limit exceeded", rejected=("limit",), rejection_reasons=("trace span limit exceeded",))
        if parent_span_id is not None and not self._policy.allow_orphan_spans and parent_span_id not in {span.span_id for span in self._context.spans}:
            return _result(self._context, False, "trace parent span missing", rejected=(str(parent_span_id),), rejection_reasons=("trace parent span missing",))
        span = create_trace_span(trace_id=self._context.trace_id, name=name, kind=kind, status=TraceSpanStatus.STARTED, parent_span_id=parent_span_id, metadata=metadata)
        self._context = TraceContext(self._context.trace_id, self._context.spans + (span,), TraceStatus.ACTIVE, self._context.metadata)
        return _result(self._context, True, "trace span started")

    def end_span(self, *, span_id: str, metadata: dict[str, object] | None = None) -> TraceRecorderResult:
        return self._replace_span(span_id=span_id, status=TraceSpanStatus.ENDED, reason="trace span ended", metadata=metadata)

    def fail_span(self, *, span_id: str, reason: str, metadata: dict[str, object] | None = None) -> TraceRecorderResult:
        reason = ensure_non_empty_string(reason, "reason")
        merged = _merge_metadata({"failure_reason": reason}, metadata)
        return self._replace_span(span_id=span_id, status=TraceSpanStatus.FAILED, reason="trace span failed", metadata=merged)

    def mark_exported(self, *, span_id: str) -> TraceRecorderResult:
        return self._replace_span(span_id=span_id, status=TraceSpanStatus.EXPORTED, reason="trace span exported", metadata=None)

    def replace_context(self, *, context: TraceContext) -> TraceRecorderResult:
        if not isinstance(context, TraceContext):
            raise TypeError("context must be TraceContext")
        self._context = context
        return _result(self._context, True, "trace context replaced")

    def _replace_span(self, *, span_id: str, status: TraceSpanStatus, reason: str, metadata: dict[str, object] | None) -> TraceRecorderResult:
        span_id = ensure_non_empty_string(span_id, "span_id")
        for index, span in enumerate(self._context.spans):
            if str(span.span_id) == span_id:
                merged = _merge_metadata(span.metadata, metadata)
                ended = utc_now() if status in {TraceSpanStatus.ENDED, TraceSpanStatus.FAILED} else span.ended_at
                updated = TraceSpan(span.trace_id, span.span_id, span.parent_span_id, span.name, span.kind, status, span.started_at, ended, merged)
                spans = self._context.spans[:index] + (updated,) + self._context.spans[index + 1 :]
                context_status = TraceStatus.FAILED if status is TraceSpanStatus.FAILED else self._context.status
                self._context = TraceContext(self._context.trace_id, spans, context_status, self._context.metadata)
                return _result(self._context, True, reason)
        return _result(self._context, False, "trace span missing")


def new_trace_id() -> UUID:
    return uuid4()


def new_span_id() -> UUID:
    return uuid4()


def create_trace_span(*, trace_id: UUID, name: str, kind: TraceSpanKind = TraceSpanKind.INTERNAL, status: TraceSpanStatus = TraceSpanStatus.CREATED, parent_span_id: UUID | None = None, metadata: dict[str, object] | None = None) -> TraceSpan:
    return TraceSpan(trace_id, new_span_id(), parent_span_id, name, kind, status, utc_now(), None, _merge_metadata({"source": "trace_span"}, metadata))


def create_trace_context(*, trace_id: UUID | None = None, spans: tuple[TraceSpan, ...] | list[TraceSpan] = (), status: TraceStatus = TraceStatus.CREATED, metadata: dict[str, object] | None = None) -> TraceContext:
    return TraceContext(new_trace_id() if trace_id is None else trace_id, tuple(spans), status, _merge_metadata({"source": "trace_context"}, metadata))


def build_trace_recorder(*, context: TraceContext | None = None, policy: TraceRecorderPolicy | None = None) -> TraceRecorder:
    return TraceRecorder(context=context, policy=policy)


def _copy_metadata(value: dict[str, object]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(value)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
        if key in _FORBIDDEN_STATE or key in _FORBIDDEN_SECRET:
            raise ValueError("metadata contains forbidden key")
    return copied


def _merge_metadata(defaults: dict[str, object], metadata: dict[str, object] | None) -> dict[str, object]:
    merged = _copy_metadata(defaults)
    if metadata is not None:
        merged.update(_copy_metadata(metadata))
    if "source" in defaults:
        merged["source"] = defaults["source"]
    return merged


def _reject_name(value: str, field_name: str) -> None:
    if value in _FORBIDDEN_STATE:
        raise ValueError(f"{field_name} must not persist runtime semantic state")


def _span_tuple(value: tuple[TraceSpan, ...] | list[TraceSpan]) -> tuple[TraceSpan, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("spans must be tuple[TraceSpan, ...]")
    spans = tuple(value)
    ids = [span.span_id for span in spans if isinstance(span, TraceSpan)]
    if len(ids) != len(set(ids)):
        raise ValueError("span_id values must be unique")
    for span in spans:
        if not isinstance(span, TraceSpan):
            raise TypeError("every span must be TraceSpan")
    return spans


def _string_tuple(value: tuple[str, ...] | list[str], field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple[str, ...]")
    return tuple(ensure_non_empty_string(item, field_name) for item in value)


def _result(context: TraceContext, changed: bool, reason: str, rejected: tuple[str, ...] = (), rejection_reasons: tuple[str, ...] = ()) -> TraceRecorderResult:
    return TraceRecorderResult(context, changed, reason, rejected, rejection_reasons, {"source": "trace_recorder_result"})
