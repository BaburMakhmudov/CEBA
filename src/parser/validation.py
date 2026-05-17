"""CEBA Parser Layer validation utilities.

This module validates parser artifacts and parser boundary invariants. It does
not parse content, infer meaning, compute final delta values, telos, world model
state, or memory updates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

from ois.parser.errors import ParserValidationError
from ois.parser.schema import (
    LocalDifferenceCandidate,
    ParseRequest,
    ParseResult,
    ParseStatus,
    ParsedSegment,
    ParserConfig,
    ParserTrace,
    TextSpan,
    TokenSpan,
)


_VALIDATION_ID_PREFIX = "pval"
_MAX_VALIDATION_ISSUES = 10_000

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


class ValidationSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


class ValidationTargetKind(str, Enum):
    PARSE_REQUEST = "parse_request"
    PARSE_RESULT = "parse_result"
    SEGMENT = "segment"
    TOKEN = "token"
    CANDIDATE = "candidate"
    TRACE = "trace"
    METADATA = "metadata"


class ValidationMode(str, Enum):
    COLLECT = "collect"
    RAISE = "raise"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    issue_id: str
    severity: ValidationSeverity
    target_kind: ValidationTargetKind
    code: str
    message: str
    request_id: str | None = None
    observation_id: str | None = None
    segment_id: str | None = None
    token_id: str | None = None
    candidate_id: str | None = None
    trace_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_non_empty_string(self.issue_id, field_name="issue_id")
        object.__setattr__(self, "severity", _ensure_severity(self.severity))
        object.__setattr__(
            self,
            "target_kind",
            _ensure_target_kind(self.target_kind),
        )
        _ensure_non_empty_string(self.code, field_name="code")
        _ensure_non_empty_string(self.message, field_name="message")
        _ensure_optional_id(self.request_id, field_name="request_id")
        _ensure_optional_id(self.observation_id, field_name="observation_id")
        _ensure_optional_id(self.segment_id, field_name="segment_id")
        _ensure_optional_id(self.token_id, field_name="token_id")
        _ensure_optional_id(self.candidate_id, field_name="candidate_id")
        _ensure_optional_id(self.trace_id, field_name="trace_id")
        object.__setattr__(self, "metadata", _ensure_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ValidationReport:
    validation_id: str
    target_kind: ValidationTargetKind
    valid: bool
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_non_empty_string(self.validation_id, field_name="validation_id")
        object.__setattr__(
            self,
            "target_kind",
            _ensure_target_kind(self.target_kind),
        )
        _ensure_bool(self.valid, field_name="valid")
        issues = _ensure_issues(self.issues)
        if self.valid and any(_is_error_issue(issue) for issue in issues):
            raise ParserValidationError(
                "valid report cannot contain error issues",
                context={"validation_id": self.validation_id},
            )
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "metadata", _ensure_metadata(self.metadata))


def validate_parse_request(
    request: ParseRequest,
    *,
    mode: ValidationMode = ValidationMode.RAISE,
) -> ValidationReport:
    normalized_mode = _ensure_validation_mode(mode)
    issues: list[ValidationIssue] = []
    request_id = _safe_id(getattr(request, "request_id", None))
    observation_id = _safe_id(
        str(getattr(getattr(request, "observation", None), "observation_id", ""))
        if getattr(request, "observation", None) is not None
        else None
    )

    if not isinstance(request, ParseRequest):
        _append_issue(
            issues,
            target_kind=ValidationTargetKind.PARSE_REQUEST,
            code="INVALID_TYPE",
            message="request must be ParseRequest",
            request_id=request_id,
            observation_id=observation_id,
            metadata={"actual": type(request).__name__},
        )
        return _apply_mode(
            _report(
                target_kind=ValidationTargetKind.PARSE_REQUEST,
                subject_id=request_id or "unknown",
                issues=issues,
            ),
            normalized_mode,
        )

    if not request.request_id:
        _append_issue(
            issues,
            target_kind=ValidationTargetKind.PARSE_REQUEST,
            code="EMPTY_ID",
            message="request_id must not be empty",
            request_id=request_id,
            observation_id=observation_id,
            metadata={"field": "request_id"},
        )
    if request.observation is None:
        _append_issue(
            issues,
            target_kind=ValidationTargetKind.PARSE_REQUEST,
            code="INVALID_TYPE",
            message="observation is required",
            request_id=request_id,
            metadata={"field": "observation"},
        )
    if not isinstance(request.config, ParserConfig):
        _append_issue(
            issues,
            target_kind=ValidationTargetKind.PARSE_REQUEST,
            code="INVALID_TYPE",
            message="config must be ParserConfig",
            request_id=request_id,
            observation_id=observation_id,
            metadata={"field": "config", "actual": type(request.config).__name__},
        )
    if not _is_timezone_aware(request.created_at):
        _append_issue(
            issues,
            target_kind=ValidationTargetKind.PARSE_REQUEST,
            code="TIMEZONE_REQUIRED",
            message="created_at must be timezone-aware",
            request_id=request_id,
            observation_id=observation_id,
            metadata={"field": "created_at"},
        )

    return _apply_mode(
        _report(
            target_kind=ValidationTargetKind.PARSE_REQUEST,
            subject_id=request_id or "unknown",
            issues=issues,
            metadata={"validation_target": ValidationTargetKind.PARSE_REQUEST.value},
        ),
        normalized_mode,
    )


def validate_parse_result(
    result: ParseResult,
    *,
    mode: ValidationMode = ValidationMode.RAISE,
) -> ValidationReport:
    normalized_mode = _ensure_validation_mode(mode)
    issues: list[ValidationIssue] = []
    request_id = _safe_id(getattr(result, "request_id", None))
    observation_id = _safe_id(getattr(result, "observation_id", None))

    if not isinstance(result, ParseResult):
        _append_issue(
            issues,
            target_kind=ValidationTargetKind.PARSE_RESULT,
            code="INVALID_TYPE",
            message="result must be ParseResult",
            request_id=request_id,
            observation_id=observation_id,
            metadata={"actual": type(result).__name__},
        )
        return _apply_mode(
            _report(
                target_kind=ValidationTargetKind.PARSE_RESULT,
                subject_id=request_id or observation_id or "unknown",
                issues=issues,
            ),
            normalized_mode,
        )

    if not result.request_id:
        _append_issue(
            issues,
            target_kind=ValidationTargetKind.PARSE_RESULT,
            code="EMPTY_ID",
            message="request_id must not be empty",
            request_id=request_id,
            observation_id=observation_id,
            metadata={"field": "request_id"},
        )
    if not result.observation_id:
        _append_issue(
            issues,
            target_kind=ValidationTargetKind.PARSE_RESULT,
            code="EMPTY_ID",
            message="observation_id must not be empty",
            request_id=request_id,
            observation_id=observation_id,
            metadata={"field": "observation_id"},
        )
    if not isinstance(result.status, ParseStatus):
        _append_issue(
            issues,
            target_kind=ValidationTargetKind.PARSE_RESULT,
            code="INVALID_TYPE",
            message="status must be ParseStatus",
            request_id=request_id,
            observation_id=observation_id,
            metadata={"field": "status", "actual": type(result.status).__name__},
        )
    if not _is_timezone_aware(result.created_at):
        _append_issue(
            issues,
            target_kind=ValidationTargetKind.PARSE_RESULT,
            code="TIMEZONE_REQUIRED",
            message="created_at must be timezone-aware",
            request_id=request_id,
            observation_id=observation_id,
            metadata={"field": "created_at"},
        )
    _validate_metadata_boundary(
        result.metadata,
        issues,
        target_kind=ValidationTargetKind.PARSE_RESULT,
        request_id=request_id,
        observation_id=observation_id,
    )

    segments = result.segments
    candidates = result.difference_candidates
    traces = result.traces
    if not isinstance(segments, tuple):
        _append_issue(
            issues,
            target_kind=ValidationTargetKind.PARSE_RESULT,
            code="INVALID_TYPE",
            message="segments must be a tuple",
            request_id=request_id,
            observation_id=observation_id,
            metadata={"field": "segments", "actual": type(segments).__name__},
        )
        segments = ()
    if not isinstance(candidates, tuple):
        _append_issue(
            issues,
            target_kind=ValidationTargetKind.PARSE_RESULT,
            code="INVALID_TYPE",
            message="difference_candidates must be a tuple",
            request_id=request_id,
            observation_id=observation_id,
            metadata={
                "field": "difference_candidates",
                "actual": type(candidates).__name__,
            },
        )
        candidates = ()
    if not isinstance(traces, tuple):
        _append_issue(
            issues,
            target_kind=ValidationTargetKind.PARSE_RESULT,
            code="INVALID_TYPE",
            message="traces must be a tuple",
            request_id=request_id,
            observation_id=observation_id,
            metadata={"field": "traces", "actual": type(traces).__name__},
        )
        traces = ()

    if (
        isinstance(result.status, ParseStatus)
        and result.status == ParseStatus.PARSED
        and not segments
    ):
        _append_issue(
            issues,
            target_kind=ValidationTargetKind.PARSE_RESULT,
            code="INVALID_STATUS_CONTENT",
            message="parsed result must contain at least one segment",
            request_id=request_id,
            observation_id=observation_id,
            metadata={"status": result.status.value},
        )
    if isinstance(result.status, ParseStatus) and result.status in {
        ParseStatus.EMPTY,
        ParseStatus.REJECTED,
    }:
        if segments:
            _append_issue(
                issues,
                target_kind=ValidationTargetKind.PARSE_RESULT,
                code="INVALID_STATUS_CONTENT",
                message="empty or rejected result must not contain segments",
                request_id=request_id,
                observation_id=observation_id,
                metadata={"status": result.status.value},
            )
        if candidates:
            _append_issue(
                issues,
                target_kind=ValidationTargetKind.PARSE_RESULT,
                code="INVALID_STATUS_CONTENT",
                message="empty or rejected result must not contain candidates",
                request_id=request_id,
                observation_id=observation_id,
                metadata={"status": result.status.value},
            )

    segments_by_id = _collect_segments(
        observation_id=observation_id,
        segments=segments,
        issues=issues,
        request_id=request_id,
    )
    _collect_tokens(
        observation_id=observation_id,
        segments=segments,
        issues=issues,
        request_id=request_id,
    )
    for issue_item in validate_candidates(
        observation_id=observation_id or "",
        candidates=candidates,
        segments_by_id=segments_by_id,
        mode=ValidationMode.COLLECT,
    ).issues:
        _append_existing_issue(issues, issue_item)
    for issue_item in validate_traces(
        observation_id=observation_id or "",
        traces=traces,
        mode=ValidationMode.COLLECT,
    ).issues:
        _append_existing_issue(issues, issue_item)

    return _apply_mode(
        _report(
            target_kind=ValidationTargetKind.PARSE_RESULT,
            subject_id=request_id or observation_id or "unknown",
            issues=issues,
            metadata={"validation_target": ValidationTargetKind.PARSE_RESULT.value},
        ),
        normalized_mode,
    )


def validate_segments(
    *,
    observation_id: str,
    segments: tuple[ParsedSegment, ...],
    mode: ValidationMode = ValidationMode.RAISE,
) -> ValidationReport:
    normalized_mode = _ensure_validation_mode(mode)
    issues: list[ValidationIssue] = []
    normalized_observation_id = _safe_id(observation_id)

    if normalized_observation_id is None:
        _append_issue(
            issues,
            target_kind=ValidationTargetKind.SEGMENT,
            code="EMPTY_ID",
            message="observation_id must not be empty",
            metadata={"field": "observation_id"},
        )
    if not isinstance(segments, tuple):
        _append_issue(
            issues,
            target_kind=ValidationTargetKind.SEGMENT,
            code="INVALID_TYPE",
            message="segments must be a tuple",
            observation_id=normalized_observation_id,
            metadata={"actual": type(segments).__name__},
        )
        return _apply_mode(
            _report(
                target_kind=ValidationTargetKind.SEGMENT,
                subject_id=normalized_observation_id or "unknown",
                issues=issues,
            ),
            normalized_mode,
        )

    _collect_segments(
        observation_id=normalized_observation_id,
        segments=segments,
        issues=issues,
        request_id=None,
    )
    _collect_tokens(
        observation_id=normalized_observation_id,
        segments=segments,
        issues=issues,
        request_id=None,
    )
    return _apply_mode(
        _report(
            target_kind=ValidationTargetKind.SEGMENT,
            subject_id=normalized_observation_id or "unknown",
            issues=issues,
            metadata={
                "validation_target": ValidationTargetKind.SEGMENT.value,
                "issue_count": len(issues),
            },
        ),
        normalized_mode,
    )


def validate_candidates(
    *,
    observation_id: str,
    candidates: tuple[LocalDifferenceCandidate, ...],
    segments_by_id: Mapping[str, ParsedSegment] | None = None,
    mode: ValidationMode = ValidationMode.RAISE,
) -> ValidationReport:
    normalized_mode = _ensure_validation_mode(mode)
    issues: list[ValidationIssue] = []
    normalized_observation_id = _safe_id(observation_id)

    if normalized_observation_id is None:
        _append_issue(
            issues,
            target_kind=ValidationTargetKind.CANDIDATE,
            code="EMPTY_ID",
            message="observation_id must not be empty",
            metadata={"field": "observation_id"},
        )
    if not isinstance(candidates, tuple):
        _append_issue(
            issues,
            target_kind=ValidationTargetKind.CANDIDATE,
            code="INVALID_TYPE",
            message="candidates must be a tuple",
            observation_id=normalized_observation_id,
            metadata={"actual": type(candidates).__name__},
        )
        return _apply_mode(
            _report(
                target_kind=ValidationTargetKind.CANDIDATE,
                subject_id=normalized_observation_id or "unknown",
                issues=issues,
            ),
            normalized_mode,
        )

    segment_lookup: Mapping[str, ParsedSegment] | None = None
    if segments_by_id is not None:
        if not isinstance(segments_by_id, Mapping):
            _append_issue(
                issues,
                target_kind=ValidationTargetKind.CANDIDATE,
                code="INVALID_TYPE",
                message="segments_by_id must be a mapping",
                observation_id=normalized_observation_id,
                metadata={"field": "segments_by_id"},
            )
        else:
            segment_lookup = segments_by_id

    seen_candidate_ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, LocalDifferenceCandidate):
            _append_issue(
                issues,
                target_kind=ValidationTargetKind.CANDIDATE,
                code="INVALID_TYPE",
                message="candidate must be LocalDifferenceCandidate",
                observation_id=normalized_observation_id,
                metadata={"actual": type(candidate).__name__},
            )
            continue
        candidate_id = _safe_id(candidate.candidate_id)
        if not candidate_id:
            _append_issue(
                issues,
                target_kind=ValidationTargetKind.CANDIDATE,
                code="EMPTY_ID",
                message="candidate_id must not be empty",
                observation_id=normalized_observation_id,
                candidate_id=candidate_id,
                metadata={"field": "candidate_id"},
            )
        elif candidate_id in seen_candidate_ids:
            _append_issue(
                issues,
                target_kind=ValidationTargetKind.CANDIDATE,
                code="DUPLICATE_CANDIDATE_ID",
                message="candidate_id must be unique",
                observation_id=normalized_observation_id,
                candidate_id=candidate_id,
            )
        else:
            seen_candidate_ids.add(candidate_id)
        if candidate.observation_id != normalized_observation_id:
            _append_issue(
                issues,
                target_kind=ValidationTargetKind.CANDIDATE,
                code="OBSERVATION_ID_MISMATCH",
                message="candidate observation_id must match validation observation_id",
                observation_id=normalized_observation_id,
                candidate_id=candidate_id,
                metadata={
                    "expected": normalized_observation_id or "",
                    "actual": candidate.observation_id,
                },
            )
        if not 0.0 <= candidate.confidence <= 1.0:
            _append_issue(
                issues,
                target_kind=ValidationTargetKind.CANDIDATE,
                code="CONFIDENCE_OUT_OF_RANGE",
                message="candidate confidence must be between 0.0 and 1.0",
                observation_id=normalized_observation_id,
                candidate_id=candidate_id,
                metadata={"actual": candidate.confidence},
            )
        _validate_metadata_boundary(
            candidate.metadata,
            issues,
            target_kind=ValidationTargetKind.CANDIDATE,
            observation_id=normalized_observation_id,
            candidate_id=candidate_id,
        )
        if (
            candidate.left_segment_id is None
            and candidate.right_segment_id is None
            and candidate.left_span is None
            and candidate.right_span is None
        ):
            _append_issue(
                issues,
                target_kind=ValidationTargetKind.CANDIDATE,
                code="CANDIDATE_REFERENCE_MISSING",
                message="candidate must include at least one segment or span reference",
                observation_id=normalized_observation_id,
                candidate_id=candidate_id,
            )
        _validate_candidate_side(
            candidate=candidate,
            segment_lookup=segment_lookup,
            side="left",
            observation_id=normalized_observation_id,
            issues=issues,
        )
        _validate_candidate_side(
            candidate=candidate,
            segment_lookup=segment_lookup,
            side="right",
            observation_id=normalized_observation_id,
            issues=issues,
        )

    return _apply_mode(
        _report(
            target_kind=ValidationTargetKind.CANDIDATE,
            subject_id=normalized_observation_id or "unknown",
            issues=issues,
            metadata={
                "validation_target": ValidationTargetKind.CANDIDATE.value,
                "issue_count": len(issues),
            },
        ),
        normalized_mode,
    )


def validate_traces(
    *,
    observation_id: str,
    traces: tuple[ParserTrace, ...],
    mode: ValidationMode = ValidationMode.RAISE,
) -> ValidationReport:
    normalized_mode = _ensure_validation_mode(mode)
    issues: list[ValidationIssue] = []
    normalized_observation_id = _safe_id(observation_id)

    if normalized_observation_id is None:
        _append_issue(
            issues,
            target_kind=ValidationTargetKind.TRACE,
            code="EMPTY_ID",
            message="observation_id must not be empty",
            metadata={"field": "observation_id"},
        )
    if not isinstance(traces, tuple):
        _append_issue(
            issues,
            target_kind=ValidationTargetKind.TRACE,
            code="INVALID_TYPE",
            message="traces must be a tuple",
            observation_id=normalized_observation_id,
            metadata={"actual": type(traces).__name__},
        )
        return _apply_mode(
            _report(
                target_kind=ValidationTargetKind.TRACE,
                subject_id=normalized_observation_id or "unknown",
                issues=issues,
            ),
            normalized_mode,
        )

    seen_trace_ids: set[str] = set()
    for trace in traces:
        if not isinstance(trace, ParserTrace):
            _append_issue(
                issues,
                target_kind=ValidationTargetKind.TRACE,
                code="INVALID_TYPE",
                message="trace must be ParserTrace",
                observation_id=normalized_observation_id,
                metadata={"actual": type(trace).__name__},
            )
            continue
        trace_id = _safe_id(trace.trace_id)
        if not trace_id:
            _append_issue(
                issues,
                target_kind=ValidationTargetKind.TRACE,
                code="EMPTY_ID",
                message="trace_id must not be empty",
                observation_id=normalized_observation_id,
                metadata={"field": "trace_id"},
            )
        elif trace_id in seen_trace_ids:
            _append_issue(
                issues,
                target_kind=ValidationTargetKind.TRACE,
                code="DUPLICATE_TRACE_ID",
                message="trace_id must be unique",
                observation_id=normalized_observation_id,
                trace_id=trace_id,
            )
        else:
            seen_trace_ids.add(trace_id)
        if trace.observation_id != normalized_observation_id:
            _append_issue(
                issues,
                target_kind=ValidationTargetKind.TRACE,
                code="OBSERVATION_ID_MISMATCH",
                message="trace observation_id must match validation observation_id",
                observation_id=normalized_observation_id,
                trace_id=trace_id,
                metadata={
                    "expected": normalized_observation_id or "",
                    "actual": trace.observation_id,
                },
            )
        if not trace.message:
            _append_issue(
                issues,
                target_kind=ValidationTargetKind.TRACE,
                code="EMPTY_ID",
                message="trace message must not be empty",
                observation_id=normalized_observation_id,
                trace_id=trace_id,
                metadata={"field": "message"},
            )
        if not _is_timezone_aware(trace.created_at):
            _append_issue(
                issues,
                target_kind=ValidationTargetKind.TRACE,
                code="TIMEZONE_REQUIRED",
                message="trace created_at must be timezone-aware",
                observation_id=normalized_observation_id,
                trace_id=trace_id,
                metadata={"field": "created_at"},
            )
        _validate_metadata_boundary(
            trace.metadata,
            issues,
            target_kind=ValidationTargetKind.TRACE,
            observation_id=normalized_observation_id,
            trace_id=trace_id,
        )

    return _apply_mode(
        _report(
            target_kind=ValidationTargetKind.TRACE,
            subject_id=normalized_observation_id or "unknown",
            issues=issues,
            metadata={
                "validation_target": ValidationTargetKind.TRACE.value,
                "issue_count": len(issues),
            },
        ),
        normalized_mode,
    )


def raise_if_invalid(report: ValidationReport) -> None:
    if not isinstance(report, ValidationReport):
        raise ParserValidationError(
            "report must be ValidationReport",
            context={"type": type(report).__name__},
        )
    if report.valid:
        return
    first_error = _first_error(report.issues)
    raise ParserValidationError(
        "parser validation failed",
        context={
            "validation_id": report.validation_id,
            "target_kind": report.target_kind.value,
            "issue_count": len(report.issues),
            "first_error_code": first_error.code if first_error is not None else "",
            "first_error_message": (
                first_error.message if first_error is not None else ""
            ),
        },
        request_id=first_error.request_id if first_error is not None else None,
        observation_id=first_error.observation_id if first_error is not None else None,
        segment_id=first_error.segment_id if first_error is not None else None,
        token_id=first_error.token_id if first_error is not None else None,
        candidate_id=first_error.candidate_id if first_error is not None else None,
    )


def make_validation_id(
    *,
    target_kind: ValidationTargetKind,
    subject_id: str,
) -> str:
    normalized_target_kind = _ensure_target_kind(target_kind)
    normalized_subject_id = _ensure_non_empty_string(
        subject_id,
        field_name="subject_id",
    )
    return (
        f"{_VALIDATION_ID_PREFIX}:"
        f"{normalized_target_kind.value}:"
        f"{_sanitize_id_part(normalized_subject_id)}"
    )


def _issue(
    *,
    target_kind: ValidationTargetKind,
    code: str,
    message: str,
    severity: ValidationSeverity = ValidationSeverity.ERROR,
    request_id: str | None = None,
    observation_id: str | None = None,
    segment_id: str | None = None,
    token_id: str | None = None,
    candidate_id: str | None = None,
    trace_id: str | None = None,
    metadata: Mapping[str, object] | None = None,
    ordinal: int,
) -> ValidationIssue:
    if type(ordinal) is not int:
        raise ParserValidationError(
            "ordinal must be an int",
            context={"type": type(ordinal).__name__},
        )
    if ordinal < 0:
        raise ParserValidationError(
            "ordinal must be greater than or equal to 0",
            context={"ordinal": ordinal},
        )
    normalized_target_kind = _ensure_target_kind(target_kind)
    subject_id = (
        request_id
        or observation_id
        or segment_id
        or token_id
        or candidate_id
        or trace_id
        or "unknown"
    )
    return ValidationIssue(
        issue_id=(
            f"{make_validation_id(target_kind=normalized_target_kind, subject_id=subject_id)}:"
            f"{ordinal}"
        ),
        severity=_ensure_severity(severity),
        target_kind=normalized_target_kind,
        code=_ensure_non_empty_string(code, field_name="code"),
        message=_ensure_non_empty_string(message, field_name="message"),
        request_id=_ensure_optional_id(request_id, field_name="request_id"),
        observation_id=_ensure_optional_id(observation_id, field_name="observation_id"),
        segment_id=_ensure_optional_id(segment_id, field_name="segment_id"),
        token_id=_ensure_optional_id(token_id, field_name="token_id"),
        candidate_id=_ensure_optional_id(candidate_id, field_name="candidate_id"),
        trace_id=_ensure_optional_id(trace_id, field_name="trace_id"),
        metadata=_ensure_metadata(metadata),
    )


def _report(
    *,
    target_kind: ValidationTargetKind,
    subject_id: str,
    issues: list[ValidationIssue],
    metadata: Mapping[str, object] | None = None,
) -> ValidationReport:
    normalized_target_kind = _ensure_target_kind(target_kind)
    normalized_subject_id = _ensure_non_empty_string(
        subject_id,
        field_name="subject_id",
    )
    limited_issues = tuple(issues[:_MAX_VALIDATION_ISSUES])
    error_count = sum(1 for issue_item in limited_issues if _is_error_issue(issue_item))
    warning_count = len(limited_issues) - error_count
    report_metadata = {
        **dict(_ensure_metadata(metadata)),
        "issue_count": len(limited_issues),
        "error_count": error_count,
        "warning_count": warning_count,
    }
    return ValidationReport(
        validation_id=make_validation_id(
            target_kind=normalized_target_kind,
            subject_id=normalized_subject_id,
        ),
        target_kind=normalized_target_kind,
        valid=error_count == 0,
        issues=limited_issues,
        metadata=report_metadata,
    )


def _apply_mode(report: ValidationReport, mode: ValidationMode) -> ValidationReport:
    normalized_mode = _ensure_validation_mode(mode)
    if normalized_mode == ValidationMode.RAISE:
        raise_if_invalid(report)
    return report


def _ensure_validation_mode(value: ValidationMode) -> ValidationMode:
    if not isinstance(value, ValidationMode):
        raise ParserValidationError(
            "mode must be ValidationMode",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_severity(value: ValidationSeverity) -> ValidationSeverity:
    if not isinstance(value, ValidationSeverity):
        raise ParserValidationError(
            "severity must be ValidationSeverity",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_target_kind(value: ValidationTargetKind) -> ValidationTargetKind:
    if not isinstance(value, ValidationTargetKind):
        raise ParserValidationError(
            "target_kind must be ValidationTargetKind",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_non_empty_string(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ParserValidationError(
            f"{field_name} must be a string",
            context={"type": type(value).__name__},
        )
    if not value:
        raise ParserValidationError(f"{field_name} must not be empty")
    if any(ord(character) < 32 for character in value):
        raise ParserValidationError(
            f"{field_name} must not contain control characters"
        )
    return value


def _ensure_optional_id(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    normalized_value = _ensure_non_empty_string(value, field_name=field_name)
    if _contains_disallowed_control_chars(normalized_value):
        raise ParserValidationError(
            f"{field_name} must not contain whitespace or control characters"
        )
    return normalized_value


def _ensure_bool(value: bool, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ParserValidationError(
            f"{field_name} must be a bool",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_timezone_aware(value: object, *, field_name: str) -> object:
    if not _is_timezone_aware(value):
        raise ParserValidationError(f"{field_name} must be timezone-aware")
    return value


def _ensure_metadata(value: Mapping[str, object] | None) -> Mapping[str, object]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ParserValidationError(
            "metadata must be a mapping",
            context={"type": type(value).__name__},
        )
    copied: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ParserValidationError(
                "metadata keys must be strings",
                context={"type": type(key).__name__},
            )
        normalized_key = key.strip().lower().replace("-", "_")
        for forbidden_key in _FORBIDDEN_METADATA_KEYS:
            if normalized_key == forbidden_key or forbidden_key in normalized_key:
                raise ParserValidationError(
                    "metadata key crosses parser validation boundary",
                    context={"metadata_key": key},
                )
        copied[key] = item
    return MappingProxyType(copied)


def _ensure_issues(value: tuple[ValidationIssue, ...]) -> tuple[ValidationIssue, ...]:
    if not isinstance(value, tuple):
        raise ParserValidationError(
            "issues must be a tuple",
            context={"type": type(value).__name__},
        )
    if len(value) > _MAX_VALIDATION_ISSUES:
        raise ParserValidationError(
            "too many validation issues",
            context={
                "issue_count": len(value),
                "max_issue_count": _MAX_VALIDATION_ISSUES,
            },
        )
    for issue_item in value:
        if not isinstance(issue_item, ValidationIssue):
            raise ParserValidationError(
                "issues must contain ValidationIssue values",
                context={"type": type(issue_item).__name__},
            )
    return value


def _immutable_metadata(value: Mapping[str, object]) -> Mapping[str, object]:
    return _ensure_metadata(value)


def _contains_disallowed_control_chars(value: str) -> bool:
    return any(character.isspace() or ord(character) < 32 for character in value)


def _sanitize_id_part(value: str) -> str:
    normalized_value = _ensure_non_empty_string(value, field_name="id_part")
    if _contains_disallowed_control_chars(normalized_value):
        raise ParserValidationError("id_part must not contain whitespace or controls")
    return normalized_value


def _span_fits_text(span: TextSpan, text: str) -> bool:
    if not isinstance(span, TextSpan) or not isinstance(text, str):
        return False
    return 0 <= span.start <= span.end <= len(text)


def _is_error_issue(issue: ValidationIssue) -> bool:
    return issue.severity == ValidationSeverity.ERROR


def _first_error(
    issues: tuple[ValidationIssue, ...],
) -> ValidationIssue | None:
    for issue_item in issues:
        if _is_error_issue(issue_item):
            return issue_item
    return None


def _append_issue(
    issues: list[ValidationIssue],
    *,
    target_kind: ValidationTargetKind,
    code: str,
    message: str,
    severity: ValidationSeverity = ValidationSeverity.ERROR,
    request_id: str | None = None,
    observation_id: str | None = None,
    segment_id: str | None = None,
    token_id: str | None = None,
    candidate_id: str | None = None,
    trace_id: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> None:
    if len(issues) >= _MAX_VALIDATION_ISSUES:
        return
    issues.append(
        _issue(
            target_kind=target_kind,
            code=code,
            message=message,
            severity=severity,
            request_id=request_id,
            observation_id=observation_id,
            segment_id=segment_id,
            token_id=token_id,
            candidate_id=candidate_id,
            trace_id=trace_id,
            metadata=metadata,
            ordinal=len(issues),
        )
    )


def _append_existing_issue(
    issues: list[ValidationIssue],
    issue_item: ValidationIssue,
) -> None:
    if len(issues) >= _MAX_VALIDATION_ISSUES:
        return
    issues.append(issue_item)


def _safe_id(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if _contains_disallowed_control_chars(value):
        return None
    return value


def _is_timezone_aware(value: object) -> bool:
    tzinfo = getattr(value, "tzinfo", None)
    if tzinfo is None:
        return False
    try:
        return tzinfo.utcoffset(value) is not None
    except Exception:
        return False


def _validate_metadata_boundary(
    metadata: object,
    issues: list[ValidationIssue],
    *,
    target_kind: ValidationTargetKind,
    request_id: str | None = None,
    observation_id: str | None = None,
    segment_id: str | None = None,
    token_id: str | None = None,
    candidate_id: str | None = None,
    trace_id: str | None = None,
) -> None:
    if not isinstance(metadata, Mapping):
        _append_issue(
            issues,
            target_kind=target_kind,
            code="INVALID_TYPE",
            message="metadata must be a mapping",
            request_id=request_id,
            observation_id=observation_id,
            segment_id=segment_id,
            token_id=token_id,
            candidate_id=candidate_id,
            trace_id=trace_id,
            metadata={"field": "metadata", "actual": type(metadata).__name__},
        )
        return
    for key in metadata:
        if not isinstance(key, str):
            _append_issue(
                issues,
                target_kind=target_kind,
                code="INVALID_TYPE",
                message="metadata keys must be strings",
                request_id=request_id,
                observation_id=observation_id,
                segment_id=segment_id,
                token_id=token_id,
                candidate_id=candidate_id,
                trace_id=trace_id,
                metadata={"field": "metadata", "actual": type(key).__name__},
            )
            continue
        normalized_key = key.strip().lower().replace("-", "_")
        for forbidden_key in _FORBIDDEN_METADATA_KEYS:
            if normalized_key == forbidden_key or forbidden_key in normalized_key:
                _append_issue(
                    issues,
                    target_kind=ValidationTargetKind.METADATA,
                    code="METADATA_BOUNDARY_VIOLATION",
                    message="metadata key crosses parser boundary",
                    request_id=request_id,
                    observation_id=observation_id,
                    segment_id=segment_id,
                    token_id=token_id,
                    candidate_id=candidate_id,
                    trace_id=trace_id,
                    metadata={"field": key},
                )
                break


def _collect_segments(
    *,
    observation_id: str | None,
    segments: tuple[ParsedSegment, ...],
    issues: list[ValidationIssue],
    request_id: str | None,
) -> dict[str, ParsedSegment]:
    segments_by_id: dict[str, ParsedSegment] = {}
    for index, segment in enumerate(segments):
        if not isinstance(segment, ParsedSegment):
            _append_issue(
                issues,
                target_kind=ValidationTargetKind.SEGMENT,
                code="INVALID_TYPE",
                message="segment must be ParsedSegment",
                request_id=request_id,
                observation_id=observation_id,
                metadata={"actual": type(segment).__name__},
            )
            continue
        segment_id = _safe_id(segment.segment_id)
        if not segment_id:
            _append_issue(
                issues,
                target_kind=ValidationTargetKind.SEGMENT,
                code="EMPTY_ID",
                message="segment_id must not be empty",
                request_id=request_id,
                observation_id=observation_id,
                metadata={"field": "segment_id"},
            )
        elif segment_id in segments_by_id:
            _append_issue(
                issues,
                target_kind=ValidationTargetKind.SEGMENT,
                code="DUPLICATE_SEGMENT_ID",
                message="segment_id must be unique",
                request_id=request_id,
                observation_id=observation_id,
                segment_id=segment_id,
            )
        else:
            segments_by_id[segment_id] = segment
        if observation_id is not None and segment.observation_id != observation_id:
            _append_issue(
                issues,
                target_kind=ValidationTargetKind.SEGMENT,
                code="OBSERVATION_ID_MISMATCH",
                message="segment observation_id must match validation observation_id",
                request_id=request_id,
                observation_id=observation_id,
                segment_id=segment_id,
                metadata={"expected": observation_id, "actual": segment.observation_id},
            )
        if type(segment.ordinal) is not int or segment.ordinal < 0:
            _append_issue(
                issues,
                target_kind=ValidationTargetKind.SEGMENT,
                code="INVALID_TYPE",
                message="segment ordinal must be a non-negative int",
                request_id=request_id,
                observation_id=observation_id,
                segment_id=segment_id,
                metadata={"field": "ordinal", "actual": segment.ordinal},
            )
        elif segment.ordinal != index:
            _append_issue(
                issues,
                target_kind=ValidationTargetKind.SEGMENT,
                code="SEGMENT_ORDINAL_MISMATCH",
                message="segment ordinal should match tuple index",
                severity=ValidationSeverity.WARNING,
                request_id=request_id,
                observation_id=observation_id,
                segment_id=segment_id,
                metadata={"expected": index, "actual": segment.ordinal},
            )
        if not isinstance(segment.path, tuple):
            _append_issue(
                issues,
                target_kind=ValidationTargetKind.SEGMENT,
                code="INVALID_TYPE",
                message="segment path must be a tuple",
                request_id=request_id,
                observation_id=observation_id,
                segment_id=segment_id,
                metadata={"field": "path", "actual": type(segment.path).__name__},
            )
        elif any(not isinstance(item, str) or not item for item in segment.path):
            _append_issue(
                issues,
                target_kind=ValidationTargetKind.SEGMENT,
                code="INVALID_TYPE",
                message="segment path must contain non-empty strings",
                request_id=request_id,
                observation_id=observation_id,
                segment_id=segment_id,
                metadata={"field": "path"},
            )
        _validate_metadata_boundary(
            segment.metadata,
            issues,
            target_kind=ValidationTargetKind.SEGMENT,
            request_id=request_id,
            observation_id=observation_id,
            segment_id=segment_id,
        )
        _validate_segment_span(
            segment=segment,
            issues=issues,
            request_id=request_id,
            observation_id=observation_id,
            segment_id=segment_id,
        )
    return segments_by_id


def _collect_tokens(
    *,
    observation_id: str | None,
    segments: tuple[ParsedSegment, ...],
    issues: list[ValidationIssue],
    request_id: str | None,
) -> dict[str, TokenSpan]:
    tokens_by_id: dict[str, TokenSpan] = {}
    for segment in segments:
        if not isinstance(segment, ParsedSegment):
            continue
        segment_id = _safe_id(segment.segment_id)
        if not isinstance(segment.tokens, tuple):
            _append_issue(
                issues,
                target_kind=ValidationTargetKind.TOKEN,
                code="INVALID_TYPE",
                message="segment tokens must be a tuple",
                request_id=request_id,
                observation_id=observation_id,
                segment_id=segment_id,
                metadata={"field": "tokens", "actual": type(segment.tokens).__name__},
            )
            continue
        seen_segment_token_ids: set[str] = set()
        for index, token in enumerate(segment.tokens):
            if not isinstance(token, TokenSpan):
                _append_issue(
                    issues,
                    target_kind=ValidationTargetKind.TOKEN,
                    code="INVALID_TYPE",
                    message="token must be TokenSpan",
                    request_id=request_id,
                    observation_id=observation_id,
                    segment_id=segment_id,
                    metadata={"actual": type(token).__name__},
                )
                continue
            token_id = _safe_id(token.token_id)
            if not token_id:
                _append_issue(
                    issues,
                    target_kind=ValidationTargetKind.TOKEN,
                    code="EMPTY_ID",
                    message="token_id must not be empty",
                    request_id=request_id,
                    observation_id=observation_id,
                    segment_id=segment_id,
                    metadata={"field": "token_id"},
                )
            else:
                if token_id in seen_segment_token_ids or token_id in tokens_by_id:
                    _append_issue(
                        issues,
                        target_kind=ValidationTargetKind.TOKEN,
                        code="DUPLICATE_TOKEN_ID",
                        message="token_id must be unique",
                        request_id=request_id,
                        observation_id=observation_id,
                        segment_id=segment_id,
                        token_id=token_id,
                    )
                seen_segment_token_ids.add(token_id)
                tokens_by_id[token_id] = token
            if token.segment_id != segment.segment_id:
                _append_issue(
                    issues,
                    target_kind=ValidationTargetKind.TOKEN,
                    code="TOKEN_SEGMENT_MISMATCH",
                    message="token segment_id must match containing segment_id",
                    request_id=request_id,
                    observation_id=observation_id,
                    segment_id=segment_id,
                    token_id=token_id,
                    metadata={"expected": segment.segment_id, "actual": token.segment_id},
                )
            if type(token.ordinal) is not int or token.ordinal < 0:
                _append_issue(
                    issues,
                    target_kind=ValidationTargetKind.TOKEN,
                    code="INVALID_TYPE",
                    message="token ordinal must be a non-negative int",
                    request_id=request_id,
                    observation_id=observation_id,
                    segment_id=segment_id,
                    token_id=token_id,
                    metadata={"field": "ordinal", "actual": token.ordinal},
                )
            elif token.ordinal != index:
                _append_issue(
                    issues,
                    target_kind=ValidationTargetKind.TOKEN,
                    code="TOKEN_ORDINAL_MISMATCH",
                    message="token ordinal should match tuple index",
                    severity=ValidationSeverity.WARNING,
                    request_id=request_id,
                    observation_id=observation_id,
                    segment_id=segment_id,
                    token_id=token_id,
                    metadata={"expected": index, "actual": token.ordinal},
                )
            if not isinstance(token.span, TextSpan):
                _append_issue(
                    issues,
                    target_kind=ValidationTargetKind.TOKEN,
                    code="INVALID_SPAN",
                    message="token span must be TextSpan",
                    request_id=request_id,
                    observation_id=observation_id,
                    segment_id=segment_id,
                    token_id=token_id,
                )
            elif segment.text is not None and not _span_fits_text(
                token.span,
                segment.text,
            ):
                _append_issue(
                    issues,
                    target_kind=ValidationTargetKind.TOKEN,
                    code="INVALID_SPAN",
                    message="token span must fit inside segment text",
                    request_id=request_id,
                    observation_id=observation_id,
                    segment_id=segment_id,
                    token_id=token_id,
                    metadata={
                        "expected": f"0..{len(segment.text)}",
                        "actual": f"{token.span.start}..{token.span.end}",
                    },
                )
    return tokens_by_id


def _validate_segment_span(
    *,
    segment: ParsedSegment,
    issues: list[ValidationIssue],
    request_id: str | None,
    observation_id: str | None,
    segment_id: str | None,
) -> None:
    if segment.text is None:
        if segment.span is not None:
            _append_issue(
                issues,
                target_kind=ValidationTargetKind.SEGMENT,
                code="INVALID_SPAN",
                message="segment span must be None when text is None",
                request_id=request_id,
                observation_id=observation_id,
                segment_id=segment_id,
            )
        return
    if segment.span is not None and not _span_fits_text(segment.span, segment.text):
        _append_issue(
            issues,
            target_kind=ValidationTargetKind.SEGMENT,
            code="INVALID_SPAN",
            message="segment span must fit inside segment text",
            request_id=request_id,
            observation_id=observation_id,
            segment_id=segment_id,
            metadata={
                "expected": f"0..{len(segment.text)}",
                "actual": f"{segment.span.start}..{segment.span.end}",
            },
        )


def _validate_candidate_side(
    *,
    candidate: LocalDifferenceCandidate,
    segment_lookup: Mapping[str, ParsedSegment] | None,
    side: str,
    observation_id: str | None,
    issues: list[ValidationIssue],
) -> None:
    candidate_id = _safe_id(candidate.candidate_id)
    segment_id = (
        candidate.left_segment_id if side == "left" else candidate.right_segment_id
    )
    span = candidate.left_span if side == "left" else candidate.right_span

    if segment_id is not None and not _safe_id(segment_id):
        _append_issue(
            issues,
            target_kind=ValidationTargetKind.CANDIDATE,
            code="EMPTY_ID",
            message=f"{side}_segment_id must not be empty",
            observation_id=observation_id,
            candidate_id=candidate_id,
            metadata={"field": f"{side}_segment_id"},
        )
        return
    referenced_segment = None
    if segment_id is not None and segment_lookup is not None:
        referenced_segment = segment_lookup.get(segment_id)
        if referenced_segment is None:
            _append_issue(
                issues,
                target_kind=ValidationTargetKind.CANDIDATE,
                code="UNKNOWN_SEGMENT_REFERENCE",
                message=f"{side}_segment_id must exist in segments_by_id",
                observation_id=observation_id,
                segment_id=segment_id,
                candidate_id=candidate_id,
            )
    if span is not None and not isinstance(span, TextSpan):
        _append_issue(
            issues,
            target_kind=ValidationTargetKind.CANDIDATE,
            code="INVALID_SPAN",
            message=f"{side}_span must be TextSpan",
            observation_id=observation_id,
            segment_id=segment_id,
            candidate_id=candidate_id,
            metadata={"field": f"{side}_span"},
        )
        return
    if (
        span is not None
        and referenced_segment is not None
        and referenced_segment.text is not None
        and not _span_fits_text(span, referenced_segment.text)
    ):
        _append_issue(
            issues,
            target_kind=ValidationTargetKind.CANDIDATE,
            code="INVALID_SPAN",
            message=f"{side}_span must fit inside referenced segment text",
            observation_id=observation_id,
            segment_id=segment_id,
            candidate_id=candidate_id,
            metadata={
                "expected": f"0..{len(referenced_segment.text)}",
                "actual": f"{span.start}..{span.end}",
            },
        )


__all__ = [
    "ValidationSeverity",
    "ValidationTargetKind",
    "ValidationMode",
    "ValidationIssue",
    "ValidationReport",
    "validate_parse_request",
    "validate_parse_result",
    "validate_segments",
    "validate_candidates",
    "validate_traces",
    "raise_if_invalid",
    "make_validation_id",
]
