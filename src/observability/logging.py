"""Deterministic local logging descriptors for CEBA observability."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from uuid import UUID, uuid4


__all__ = [
    "LogSeverity",
    "LogEventKind",
    "LogRecordStatus",
    "ObservabilityLogRecord",
    "LogBufferPolicy",
    "LogBufferResult",
    "LogBuffer",
    "utc_now",
    "new_log_record_id",
    "ensure_non_empty_string",
    "ensure_optional_non_empty_string",
    "ensure_positive_int",
    "ensure_non_negative_int",
    "ensure_probability",
    "ensure_finite_number",
    "ensure_timezone_aware",
    "create_log_record",
    "build_log_buffer",
]


_FORBIDDEN_STATE = {"M_t", "Mt", "semantic_state", "runtime_semantic_state"}
_FORBIDDEN_SECRET = {"password", "raw_password", "secret", "raw_secret", "api_key", "token", "access_token", "refresh_token", "private_key", "credential"}


class LogSeverity(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class LogEventKind(Enum):
    SYSTEM = "SYSTEM"
    RUNTIME = "RUNTIME"
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
    PERSISTENCE = "PERSISTENCE"
    AUDIT = "AUDIT"
    SECURITY = "SECURITY"
    OBSERVABILITY = "OBSERVABILITY"
    UNKNOWN = "UNKNOWN"


class LogRecordStatus(Enum):
    CREATED = "CREATED"
    BUFFERED = "BUFFERED"
    EXPORTED = "EXPORTED"
    DROPPED = "DROPPED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ObservabilityLogRecord:
    record_id: UUID
    severity: LogSeverity
    kind: LogEventKind
    status: LogRecordStatus
    message: str
    component: str
    occurred_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, UUID):
            raise TypeError("record_id must be UUID")
        if not isinstance(self.severity, LogSeverity):
            raise TypeError("severity must be LogSeverity")
        if not isinstance(self.kind, LogEventKind):
            raise TypeError("kind must be LogEventKind")
        if not isinstance(self.status, LogRecordStatus):
            raise TypeError("status must be LogRecordStatus")
        message = ensure_non_empty_string(self.message, "message")
        component = ensure_non_empty_string(self.component, "component")
        _reject_name(component, "component")
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "occurred_at", ensure_timezone_aware(self.occurred_at, "occurred_at"))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": str(self.record_id),
            "severity": self.severity.value,
            "kind": self.kind.value,
            "status": self.status.value,
            "message": self.message,
            "component": self.component,
            "occurred_at": self.occurred_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    def is_error(self) -> bool:
        return self.severity in {LogSeverity.ERROR, LogSeverity.CRITICAL}


@dataclass(frozen=True)
class LogBufferPolicy:
    max_records: int = 100000
    drop_debug: bool = False
    drop_unknown: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_records", ensure_positive_int(self.max_records, "max_records"))
        if not isinstance(self.drop_debug, bool):
            raise TypeError("drop_debug must be bool")
        if not isinstance(self.drop_unknown, bool):
            raise TypeError("drop_unknown must be bool")

    def to_dict(self) -> dict[str, object]:
        return {"max_records": self.max_records, "drop_debug": self.drop_debug, "drop_unknown": self.drop_unknown}


@dataclass(frozen=True)
class LogBufferResult:
    records: tuple[ObservabilityLogRecord, ...] = ()
    changed: bool = False
    reason: str = "log buffer result"
    rejected_record_ids: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        records = _record_tuple(self.records)
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        rejected = _string_tuple(self.rejected_record_ids, "rejected_record_ids")
        reasons = _string_tuple(self.rejection_reasons, "rejection_reasons")
        if len(rejected) != len(reasons):
            raise ValueError("rejected_record_ids and rejection_reasons must have same length")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "reason", ensure_non_empty_string(self.reason, "reason"))
        object.__setattr__(self, "rejected_record_ids", rejected)
        object.__setattr__(self, "rejection_reasons", reasons)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {"records": [record.to_dict() for record in self.records], "changed": self.changed, "reason": self.reason, "rejected_record_ids": list(self.rejected_record_ids), "rejection_reasons": list(self.rejection_reasons), "metadata": dict(self.metadata)}

    def record_count(self) -> int:
        return len(self.records)

    def rejected_count(self) -> int:
        return len(self.rejected_record_ids)

    def has_records(self) -> bool:
        return bool(self.records)


class LogBuffer:
    def __init__(self, *, records: tuple[ObservabilityLogRecord, ...] | list[ObservabilityLogRecord] = (), policy: LogBufferPolicy | None = None) -> None:
        self._policy = LogBufferPolicy() if policy is None else policy
        if not isinstance(self._policy, LogBufferPolicy):
            raise TypeError("policy must be LogBufferPolicy")
        self._records = _record_tuple(records)

    def records(self) -> tuple[ObservabilityLogRecord, ...]:
        return self._records

    def record_count(self) -> int:
        return len(self._records)

    def append(self, *, record: ObservabilityLogRecord) -> LogBufferResult:
        if not isinstance(record, ObservabilityLogRecord):
            raise TypeError("record must be ObservabilityLogRecord")
        reason = _reject_record(record, self._records, self._policy)
        if reason is not None:
            return _result(reason=reason, rejected=(str(record.record_id),), rejection_reasons=(reason,))
        buffered = ObservabilityLogRecord(record.record_id, record.severity, record.kind, LogRecordStatus.BUFFERED, record.message, record.component, record.occurred_at, record.metadata)
        self._records = self._records + (buffered,)
        return _result(records=(buffered,), changed=True, reason="log record buffered")

    def append_message(self, *, message: str, severity: LogSeverity = LogSeverity.INFO, kind: LogEventKind = LogEventKind.SYSTEM, component: str, metadata: dict[str, object] | None = None) -> LogBufferResult:
        return self.append(record=create_log_record(message=message, severity=severity, kind=kind, component=component, metadata=metadata))

    def query(self, *, severity: LogSeverity | None = None, kind: LogEventKind | None = None, component: str | None = None, limit: int = 100) -> LogBufferResult:
        if severity is not None and not isinstance(severity, LogSeverity):
            raise TypeError("severity must be None or LogSeverity")
        if kind is not None and not isinstance(kind, LogEventKind):
            raise TypeError("kind must be None or LogEventKind")
        component = ensure_optional_non_empty_string(component, "component")
        limit = ensure_positive_int(limit, "limit")
        records = [record for record in self._records if (severity is None or record.severity is severity) and (kind is None or record.kind is kind) and (component is None or record.component == component)]
        records.sort(key=lambda record: (record.occurred_at, str(record.record_id)), reverse=True)
        return _result(records=tuple(records[:limit]), reason="log records queried")

    def mark_exported(self, *, record_id: str) -> LogBufferResult:
        record_id = ensure_non_empty_string(record_id, "record_id")
        for index, record in enumerate(self._records):
            if str(record.record_id) == record_id:
                updated = ObservabilityLogRecord(record.record_id, record.severity, record.kind, LogRecordStatus.EXPORTED, record.message, record.component, record.occurred_at, record.metadata)
                self._records = self._records[:index] + (updated,) + self._records[index + 1 :]
                return _result(records=(updated,), changed=True, reason="log record exported")
        return _result(reason="log record missing")

    def replace_records(self, *, records: tuple[ObservabilityLogRecord, ...] | list[ObservabilityLogRecord]) -> LogBufferResult:
        self._records = _record_tuple(records)
        return _result(records=self._records, changed=True, reason="log records replaced")


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_log_record_id() -> UUID:
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


def create_log_record(*, message: str, severity: LogSeverity = LogSeverity.INFO, kind: LogEventKind = LogEventKind.SYSTEM, component: str, metadata: dict[str, object] | None = None) -> ObservabilityLogRecord:
    return ObservabilityLogRecord(new_log_record_id(), severity, kind, LogRecordStatus.CREATED, message, component, utc_now(), _merge_metadata({"source": "observability_log_record"}, metadata))


def build_log_buffer(*, records: tuple[ObservabilityLogRecord, ...] | list[ObservabilityLogRecord] = (), policy: LogBufferPolicy | None = None) -> LogBuffer:
    return LogBuffer(records=records, policy=policy)


def _copy_metadata(value: dict[str, object]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(value)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
        if key in _FORBIDDEN_STATE:
            raise ValueError("metadata must not persist runtime semantic state")
        if key in _FORBIDDEN_SECRET:
            raise ValueError("metadata must not include secret material")
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


def _string_tuple(value: tuple[str, ...] | list[str], field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple[str, ...]")
    return tuple(ensure_non_empty_string(item, field_name) for item in value)


def _record_tuple(value: tuple[ObservabilityLogRecord, ...] | list[ObservabilityLogRecord]) -> tuple[ObservabilityLogRecord, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("records must be tuple[ObservabilityLogRecord, ...]")
    records = tuple(value)
    ids = [record.record_id for record in records if isinstance(record, ObservabilityLogRecord)]
    if len(ids) != len(set(ids)):
        raise ValueError("record_id values must be unique")
    for record in records:
        if not isinstance(record, ObservabilityLogRecord):
            raise TypeError("every record must be ObservabilityLogRecord")
    return records


def _reject_record(record: ObservabilityLogRecord, records: tuple[ObservabilityLogRecord, ...], policy: LogBufferPolicy) -> str | None:
    if any(existing.record_id == record.record_id for existing in records):
        return "log record_id duplicated"
    if len(records) >= policy.max_records:
        return "log buffer limit exceeded"
    if policy.drop_debug and record.severity is LogSeverity.DEBUG:
        return "debug log record dropped"
    if policy.drop_unknown and record.kind is LogEventKind.UNKNOWN:
        return "unknown log record dropped"
    return None


def _result(*, records: tuple[ObservabilityLogRecord, ...] = (), changed: bool = False, reason: str, rejected: tuple[str, ...] = (), rejection_reasons: tuple[str, ...] = ()) -> LogBufferResult:
    return LogBufferResult(records, changed, reason, rejected, rejection_reasons, {"source": "log_buffer_result"})
