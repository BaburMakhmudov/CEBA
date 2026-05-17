"""CEBA Parser Layer audit utilities.

This module builds immutable technical audit records from ParseResult values. It
does not persist logs, infer meaning, compute final delta values, telos, world
model state, or memory updates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType

from ois.parser.errors import ParserAuditError
from ois.parser.extractor import build_parser_trace
from ois.parser.schema import ParseResult, ParseStatus, ParserTraceKind
from ois.parser.validation import ValidationMode, ValidationReport, validate_parse_result


_AUDIT_ID_PREFIX = "paudit"
_DEFAULT_AUDITOR_NAME = "parser_auditor"
_DEFAULT_AUDITOR_VERSION = "1.0.0"

_FORBIDDEN_METADATA_KEYS = frozenset({
    "meaning",
    "intent",
    "telos",
    "valuation",
    "world_model",
    "delta",
    "memory",
    "emotion",
    "goal",
    "belief",
    "semantic_label",
})


class ParserAuditStatus(str, Enum):
    ACCEPTED = "accepted"
    EMPTY = "empty"
    REJECTED = "rejected"
    INVALID = "invalid"


class ParserAuditLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ParserAuditEventKind(str, Enum):
    PARSE_RESULT_AUDITED = "parse_result_audited"
    PARSE_RESULT_INVALID = "parse_result_invalid"
    PARSE_RESULT_EMPTY = "parse_result_empty"


@dataclass(frozen=True, slots=True)
class ParserAuditConfig:
    auditor_name: str = _DEFAULT_AUDITOR_NAME
    auditor_version: str = _DEFAULT_AUDITOR_VERSION
    validate_result: bool = True
    include_validation_summary: bool = True
    append_audit_trace: bool = True
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_non_empty_string(self.auditor_name, field_name="auditor_name")
        _ensure_non_empty_string(self.auditor_version, field_name="auditor_version")
        _ensure_bool(self.validate_result, field_name="validate_result")
        _ensure_bool(
            self.include_validation_summary,
            field_name="include_validation_summary",
        )
        _ensure_bool(self.append_audit_trace, field_name="append_audit_trace")
        object.__setattr__(self, "metadata", _ensure_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ParserAuditRecord:
    audit_id: str
    request_id: str
    observation_id: str
    status: ParserAuditStatus
    level: ParserAuditLevel
    event_kind: ParserAuditEventKind
    created_at: datetime
    parse_status: ParseStatus
    segment_count: int
    token_count: int
    difference_candidate_count: int
    trace_count: int
    validation_valid: bool | None = None
    validation_issue_count: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_non_empty_string(self.audit_id, field_name="audit_id")
        _ensure_non_empty_string(self.request_id, field_name="request_id")
        _ensure_non_empty_string(self.observation_id, field_name="observation_id")
        object.__setattr__(self, "status", _ensure_audit_status(self.status))
        object.__setattr__(self, "level", _ensure_audit_level(self.level))
        object.__setattr__(
            self,
            "event_kind",
            _ensure_event_kind(self.event_kind),
        )
        object.__setattr__(
            self,
            "created_at",
            _ensure_timezone_aware(self.created_at, field_name="created_at"),
        )
        object.__setattr__(
            self,
            "parse_status",
            _ensure_parse_status(self.parse_status),
        )
        _ensure_non_negative_int(self.segment_count, field_name="segment_count")
        _ensure_non_negative_int(self.token_count, field_name="token_count")
        _ensure_non_negative_int(
            self.difference_candidate_count,
            field_name="difference_candidate_count",
        )
        _ensure_non_negative_int(self.trace_count, field_name="trace_count")
        _ensure_optional_bool(self.validation_valid, field_name="validation_valid")
        _ensure_optional_non_negative_int(
            self.validation_issue_count,
            field_name="validation_issue_count",
        )
        object.__setattr__(self, "metadata", _ensure_metadata(self.metadata))


def audit_parse_result(
    *,
    result: ParseResult,
    config: ParserAuditConfig | None = None,
    metadata: Mapping[str, object] | None = None,
) -> ParserAuditRecord:
    try:
        normalized_result = _ensure_parse_result(result)
        normalized_config = (
            ParserAuditConfig() if config is None else _ensure_audit_config(config)
        )
        request_id = normalized_result.request_id
        observation_id = normalized_result.observation_id
        validation_report = (
            validate_parse_result(normalized_result, mode=ValidationMode.COLLECT)
            if normalized_config.validate_result
            else None
        )
        _ensure_validation_report(validation_report)

        validation_valid = (
            validation_report.valid if validation_report is not None else None
        )
        validation_issue_count = (
            len(validation_report.issues) if validation_report is not None else None
        )
        segment_count = len(normalized_result.segments)
        token_count = sum(len(segment.tokens) for segment in normalized_result.segments)
        difference_candidate_count = len(normalized_result.difference_candidates)
        trace_count = len(normalized_result.traces)
        audit_status = _determine_status(
            parse_status=normalized_result.status,
            validation_valid=validation_valid,
        )
        audit_level = _determine_level(audit_status)
        event_kind = _determine_event_kind(audit_status)
        created_at = _utc_now()
        audit_id = make_audit_id(
            request_id=request_id,
            observation_id=observation_id,
            parse_status=normalized_result.status,
        )

        validation_metadata: Mapping[str, object] | None = None
        if (
            normalized_config.include_validation_summary
            and validation_report is not None
        ):
            validation_metadata = {
                "validation_valid": validation_report.valid,
                "validation_issue_count": len(validation_report.issues),
                "validation_id": validation_report.validation_id,
            }
        audit_metadata = _merge_metadata(
            normalized_config.metadata,
            metadata,
            validation_metadata,
            {
                "auditor_name": normalized_config.auditor_name,
                "auditor_version": normalized_config.auditor_version,
                "audit_id": audit_id,
                "audit_status": audit_status.value,
                "audit_level": audit_level.value,
                "event_kind": event_kind.value,
                "parse_status": normalized_result.status.value,
                "segment_count": segment_count,
                "token_count": token_count,
                "difference_candidate_count": difference_candidate_count,
                "trace_count": trace_count,
                "source": "parse_result",
            },
        )
        return ParserAuditRecord(
            audit_id=audit_id,
            request_id=request_id,
            observation_id=observation_id,
            status=audit_status,
            level=audit_level,
            event_kind=event_kind,
            created_at=created_at,
            parse_status=normalized_result.status,
            segment_count=segment_count,
            token_count=token_count,
            difference_candidate_count=difference_candidate_count,
            trace_count=trace_count,
            validation_valid=validation_valid,
            validation_issue_count=validation_issue_count,
            metadata=audit_metadata,
        )
    except ParserAuditError:
        raise
    except Exception as error:
        request_id = None
        observation_id = None
        if isinstance(result, ParseResult):
            request_id = result.request_id
            observation_id = result.observation_id
        raise ParserAuditError(
            "unexpected parser audit construction failure",
            request_id=request_id,
            observation_id=observation_id,
            context={"error_type": type(error).__name__},
        ) from error


def audit_and_append_trace(
    *,
    result: ParseResult,
    config: ParserAuditConfig | None = None,
    metadata: Mapping[str, object] | None = None,
) -> tuple[ParseResult, ParserAuditRecord]:
    normalized_config = ParserAuditConfig() if config is None else _ensure_audit_config(
        config
    )
    audit_record = audit_parse_result(
        result=result,
        config=normalized_config,
        metadata=metadata,
    )
    if not normalized_config.append_audit_trace:
        return result, audit_record
    return (
        append_audit_trace(
            result=result,
            audit_record=audit_record,
            metadata=metadata,
        ),
        audit_record,
    )


def append_audit_trace(
    *,
    result: ParseResult,
    audit_record: ParserAuditRecord,
    metadata: Mapping[str, object] | None = None,
) -> ParseResult:
    try:
        normalized_result = _ensure_parse_result(result)
        normalized_record = _ensure_audit_record(audit_record)
        auditor_name = _ensure_non_empty_string(
            str(normalized_record.metadata.get("auditor_name", _DEFAULT_AUDITOR_NAME)),
            field_name="auditor_name",
        )
        auditor_version = _ensure_non_empty_string(
            str(
                normalized_record.metadata.get(
                    "auditor_version",
                    _DEFAULT_AUDITOR_VERSION,
                )
            ),
            field_name="auditor_version",
        )
        trace_metadata = _merge_metadata(
            {
                "stage": "audit",
                "status": normalized_record.status.value,
                "level": normalized_record.level.value,
                "event_kind": normalized_record.event_kind.value,
                "audit_id": normalized_record.audit_id,
                "auditor_name": auditor_name,
                "auditor_version": auditor_version,
            },
            metadata,
        )
        audit_trace = build_parser_trace(
            request_id=normalized_result.request_id,
            observation_id=normalized_result.observation_id,
            kind=ParserTraceKind.AUDIT,
            message="parser audit completed",
            ordinal=len(normalized_result.traces),
            metadata=trace_metadata,
            created_at=_utc_now(),
        )
        result_metadata = _merge_metadata(
            normalized_result.metadata,
            normalized_record.metadata,
            metadata,
            {
                "stage": "audit",
                "status": normalized_record.status.value,
                "level": normalized_record.level.value,
                "audit_id": normalized_record.audit_id,
                "auditor_name": auditor_name,
                "auditor_version": auditor_version,
            },
        )
        return ParseResult(
            request_id=normalized_result.request_id,
            observation_id=normalized_result.observation_id,
            status=normalized_result.status,
            segments=normalized_result.segments,
            difference_candidates=normalized_result.difference_candidates,
            traces=normalized_result.traces + (audit_trace,),
            created_at=normalized_result.created_at,
            metadata=result_metadata,
        )
    except ParserAuditError:
        raise
    except Exception as error:
        request_id = None
        observation_id = None
        if isinstance(result, ParseResult):
            request_id = result.request_id
            observation_id = result.observation_id
        raise ParserAuditError(
            "unexpected parser audit trace construction failure",
            request_id=request_id,
            observation_id=observation_id,
            context={"error_type": type(error).__name__},
        ) from error


def make_audit_id(
    *,
    request_id: str,
    observation_id: str,
    parse_status: ParseStatus,
) -> str:
    normalized_request_id = _ensure_non_empty_string(
        request_id,
        field_name="request_id",
    )
    normalized_observation_id = _ensure_non_empty_string(
        observation_id,
        field_name="observation_id",
    )
    normalized_parse_status = _ensure_parse_status(parse_status)
    return (
        f"{_AUDIT_ID_PREFIX}:"
        f"{_sanitize_id_part(normalized_request_id)}:"
        f"{_sanitize_id_part(normalized_observation_id)}:"
        f"{normalized_parse_status.value}"
    )


def audit_record_to_dict(record: ParserAuditRecord) -> dict[str, object]:
    normalized_record = _ensure_audit_record(record)
    return {
        "audit_id": normalized_record.audit_id,
        "request_id": normalized_record.request_id,
        "observation_id": normalized_record.observation_id,
        "status": normalized_record.status.value,
        "level": normalized_record.level.value,
        "event_kind": normalized_record.event_kind.value,
        "created_at": normalized_record.created_at.isoformat(),
        "parse_status": normalized_record.parse_status.value,
        "segment_count": normalized_record.segment_count,
        "token_count": normalized_record.token_count,
        "difference_candidate_count": normalized_record.difference_candidate_count,
        "trace_count": normalized_record.trace_count,
        "validation_valid": normalized_record.validation_valid,
        "validation_issue_count": normalized_record.validation_issue_count,
        "metadata": dict(normalized_record.metadata),
    }


def _ensure_audit_config(value: ParserAuditConfig) -> ParserAuditConfig:
    if not isinstance(value, ParserAuditConfig):
        raise ParserAuditError(
            "config must be ParserAuditConfig",
            context={"type": type(value).__name__},
        )
    _ensure_non_empty_string(value.auditor_name, field_name="auditor_name")
    _ensure_non_empty_string(value.auditor_version, field_name="auditor_version")
    _ensure_bool(value.validate_result, field_name="validate_result")
    _ensure_bool(
        value.include_validation_summary,
        field_name="include_validation_summary",
    )
    _ensure_bool(value.append_audit_trace, field_name="append_audit_trace")
    _ensure_metadata(value.metadata)
    return value


def _ensure_audit_record(value: ParserAuditRecord) -> ParserAuditRecord:
    if not isinstance(value, ParserAuditRecord):
        raise ParserAuditError(
            "audit_record must be ParserAuditRecord",
            context={"type": type(value).__name__},
        )
    _ensure_non_empty_string(value.audit_id, field_name="audit_id")
    _ensure_non_empty_string(value.request_id, field_name="request_id")
    _ensure_non_empty_string(value.observation_id, field_name="observation_id")
    _ensure_audit_status(value.status)
    _ensure_audit_level(value.level)
    _ensure_event_kind(value.event_kind)
    _ensure_timezone_aware(value.created_at, field_name="created_at")
    _ensure_parse_status(value.parse_status)
    _ensure_non_negative_int(value.segment_count, field_name="segment_count")
    _ensure_non_negative_int(value.token_count, field_name="token_count")
    _ensure_non_negative_int(
        value.difference_candidate_count,
        field_name="difference_candidate_count",
    )
    _ensure_non_negative_int(value.trace_count, field_name="trace_count")
    _ensure_optional_bool(value.validation_valid, field_name="validation_valid")
    _ensure_optional_non_negative_int(
        value.validation_issue_count,
        field_name="validation_issue_count",
    )
    _ensure_metadata(value.metadata)
    return value


def _ensure_parse_result(value: ParseResult) -> ParseResult:
    if not isinstance(value, ParseResult):
        raise ParserAuditError(
            "result must be ParseResult",
            context={"type": type(value).__name__},
        )
    _ensure_non_empty_string(value.request_id, field_name="request_id")
    _ensure_non_empty_string(value.observation_id, field_name="observation_id")
    _ensure_parse_status(value.status)
    if not isinstance(value.segments, tuple):
        raise ParserAuditError(
            "result segments must be a tuple",
            request_id=value.request_id,
            observation_id=value.observation_id,
            context={"type": type(value.segments).__name__},
        )
    if not isinstance(value.difference_candidates, tuple):
        raise ParserAuditError(
            "result difference_candidates must be a tuple",
            request_id=value.request_id,
            observation_id=value.observation_id,
            context={"type": type(value.difference_candidates).__name__},
        )
    if not isinstance(value.traces, tuple):
        raise ParserAuditError(
            "result traces must be a tuple",
            request_id=value.request_id,
            observation_id=value.observation_id,
            context={"type": type(value.traces).__name__},
        )
    _ensure_metadata(value.metadata)
    return value


def _ensure_parse_status(value: ParseStatus) -> ParseStatus:
    if not isinstance(value, ParseStatus):
        raise ParserAuditError(
            "parse_status must be ParseStatus",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_audit_status(value: ParserAuditStatus) -> ParserAuditStatus:
    if not isinstance(value, ParserAuditStatus):
        raise ParserAuditError(
            "status must be ParserAuditStatus",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_audit_level(value: ParserAuditLevel) -> ParserAuditLevel:
    if not isinstance(value, ParserAuditLevel):
        raise ParserAuditError(
            "level must be ParserAuditLevel",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_event_kind(value: ParserAuditEventKind) -> ParserAuditEventKind:
    if not isinstance(value, ParserAuditEventKind):
        raise ParserAuditError(
            "event_kind must be ParserAuditEventKind",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_validation_report(
    value: ValidationReport | None,
) -> ValidationReport | None:
    if value is None:
        return None
    if not isinstance(value, ValidationReport):
        raise ParserAuditError(
            "validation_report must be ValidationReport",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_bool(value: bool, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ParserAuditError(
            f"{field_name} must be a bool",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_non_empty_string(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ParserAuditError(
            f"{field_name} must be a string",
            context={"type": type(value).__name__},
        )
    if not value:
        raise ParserAuditError(f"{field_name} must not be empty")
    if any(ord(character) < 32 for character in value):
        raise ParserAuditError(f"{field_name} must not contain control characters")
    return value


def _ensure_non_negative_int(value: int, *, field_name: str) -> int:
    if type(value) is not int:
        raise ParserAuditError(
            f"{field_name} must be an int",
            context={"type": type(value).__name__},
        )
    if value < 0:
        raise ParserAuditError(f"{field_name} must be greater than or equal to 0")
    return value


def _ensure_optional_non_negative_int(
    value: int | None,
    *,
    field_name: str,
) -> int | None:
    if value is None:
        return None
    return _ensure_non_negative_int(value, field_name=field_name)


def _ensure_optional_bool(value: bool | None, *, field_name: str) -> bool | None:
    if value is None:
        return None
    return _ensure_bool(value, field_name=field_name)


def _ensure_timezone_aware(value: datetime, *, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise ParserAuditError(
            f"{field_name} must be a datetime",
            context={"type": type(value).__name__},
        )
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ParserAuditError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _ensure_metadata(value: Mapping[str, object] | None) -> Mapping[str, object]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ParserAuditError(
            "metadata must be a mapping",
            context={"type": type(value).__name__},
        )
    copied: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ParserAuditError(
                "metadata keys must be strings",
                context={"type": type(key).__name__},
            )
        normalized_key = key.strip().lower().replace("-", "_")
        for forbidden_key in _FORBIDDEN_METADATA_KEYS:
            if normalized_key == forbidden_key or forbidden_key in normalized_key:
                raise ParserAuditError(
                    "metadata key crosses parser audit boundary",
                    context={"metadata_key": key},
                )
        copied[key] = item
    return MappingProxyType(copied)


def _merge_metadata(
    *metadata_items: Mapping[str, object] | None,
) -> Mapping[str, object]:
    merged: dict[str, object] = {}
    for metadata in metadata_items:
        merged.update(dict(_ensure_metadata(metadata)))
    return MappingProxyType(merged)


def _determine_status(
    *,
    parse_status: ParseStatus,
    validation_valid: bool | None,
) -> ParserAuditStatus:
    normalized_parse_status = _ensure_parse_status(parse_status)
    _ensure_optional_bool(validation_valid, field_name="validation_valid")
    if validation_valid is False:
        return ParserAuditStatus.INVALID
    if normalized_parse_status == ParseStatus.EMPTY:
        return ParserAuditStatus.EMPTY
    if normalized_parse_status == ParseStatus.REJECTED:
        return ParserAuditStatus.REJECTED
    return ParserAuditStatus.ACCEPTED


def _determine_level(status: ParserAuditStatus) -> ParserAuditLevel:
    normalized_status = _ensure_audit_status(status)
    if normalized_status in {ParserAuditStatus.INVALID, ParserAuditStatus.REJECTED}:
        return ParserAuditLevel.ERROR
    if normalized_status == ParserAuditStatus.EMPTY:
        return ParserAuditLevel.WARNING
    return ParserAuditLevel.INFO


def _determine_event_kind(status: ParserAuditStatus) -> ParserAuditEventKind:
    normalized_status = _ensure_audit_status(status)
    if normalized_status == ParserAuditStatus.INVALID:
        return ParserAuditEventKind.PARSE_RESULT_INVALID
    if normalized_status == ParserAuditStatus.EMPTY:
        return ParserAuditEventKind.PARSE_RESULT_EMPTY
    return ParserAuditEventKind.PARSE_RESULT_AUDITED


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _contains_disallowed_control_chars(value: str) -> bool:
    return any(character.isspace() or ord(character) < 32 for character in value)


def _sanitize_id_part(value: str) -> str:
    normalized_value = _ensure_non_empty_string(value, field_name="id_part")
    if _contains_disallowed_control_chars(normalized_value):
        raise ParserAuditError("id_part must not contain whitespace or controls")
    return normalized_value


__all__ = [
    "ParserAuditStatus",
    "ParserAuditLevel",
    "ParserAuditEventKind",
    "ParserAuditConfig",
    "ParserAuditRecord",
    "audit_parse_result",
    "audit_and_append_trace",
    "append_audit_trace",
    "make_audit_id",
    "audit_record_to_dict",
]
