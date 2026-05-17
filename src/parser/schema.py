"""Typed schema contract for the CEBA Parser Layer.

Parser structures describe technical decomposition artifacts derived from a
CanonicalObservation. They preserve comparable surface structure for later
parser modules without deciding meaning, intent, or downstream state changes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from types import MappingProxyType

from ois.input.schema import CanonicalObservation


class ParserMode(str, Enum):
    STRICT = "strict"
    PERMISSIVE = "permissive"


class ParseStatus(str, Enum):
    PARSED = "parsed"
    EMPTY = "empty"
    REJECTED = "rejected"


class SegmentKind(str, Enum):
    TEXT = "text"
    PARAGRAPH = "paragraph"
    LINE = "line"
    SENTENCE = "sentence"
    FIELD = "field"
    KEY_VALUE = "key_value"
    LIST_ITEM = "list_item"
    TABLE_ROW = "table_row"
    TABLE_CELL = "table_cell"
    CODE_BLOCK = "code_block"
    LOG_LINE = "log_line"
    ERROR_RECORD = "error_record"
    STRUCTURED_OBJECT = "structured_object"
    BINARY_METADATA = "binary_metadata"


class TokenKind(str, Enum):
    WORD = "word"
    NUMBER = "number"
    PUNCTUATION = "punctuation"
    SYMBOL = "symbol"
    WHITESPACE = "whitespace"
    CONTROL = "control"
    UNKNOWN = "unknown"


class LocalDifferenceKind(str, Enum):
    TOKEN_MISMATCH = "token_mismatch"
    TOKEN_INSERTION = "token_insertion"
    TOKEN_DELETION = "token_deletion"
    TOKEN_REORDER = "token_reorder"
    NUMERIC_CHANGE = "numeric_change"
    CASE_CHANGE = "case_change"
    PUNCTUATION_CHANGE = "punctuation_change"
    STRUCTURAL_CHANGE = "structural_change"
    FIELD_CHANGE = "field_change"
    REPETITION = "repetition"
    OMISSION = "omission"
    UNKNOWN = "unknown"


class ParserTraceKind(str, Enum):
    REQUEST_CREATED = "request_created"
    SEGMENTATION = "segmentation"
    TOKENIZATION = "tokenization"
    LOCAL_DIFFERENCE_EXTRACTION = "local_difference_extraction"
    VALIDATION = "validation"
    DISPATCH = "dispatch"
    AUDIT = "audit"


@dataclass(frozen=True, slots=True)
class TextSpan:
    start: int
    end: int

    def __post_init__(self) -> None:
        _ensure_non_negative_integer(self.start, field_name="start")
        _ensure_non_negative_integer(self.end, field_name="end")
        if self.end < self.start:
            raise ValueError("end must be greater than or equal to start")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class TokenSpan:
    token_id: str
    segment_id: str
    kind: TokenKind
    text: str
    normalized_text: str | None
    span: TextSpan
    ordinal: int

    def __post_init__(self) -> None:
        _ensure_non_empty_string(self.token_id, field_name="token_id")
        _ensure_non_empty_string(self.segment_id, field_name="segment_id")
        _ensure_enum(self.kind, TokenKind, field_name="kind")
        _ensure_string(self.text, field_name="text")
        if self.normalized_text is not None:
            _ensure_non_empty_string(
                self.normalized_text,
                field_name="normalized_text",
            )
        _ensure_instance(self.span, TextSpan, field_name="span")
        _ensure_non_negative_integer(self.ordinal, field_name="ordinal")


@dataclass(frozen=True, slots=True)
class ParsedSegment:
    segment_id: str
    observation_id: str
    kind: SegmentKind
    ordinal: int
    text: str | None
    span: TextSpan | None
    parent_segment_id: str | None
    path: tuple[str, ...] = field(default_factory=tuple)
    tokens: tuple[TokenSpan, ...] = field(default_factory=tuple)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_non_empty_string(self.segment_id, field_name="segment_id")
        _ensure_non_empty_string(self.observation_id, field_name="observation_id")
        _ensure_enum(self.kind, SegmentKind, field_name="kind")
        _ensure_non_negative_integer(self.ordinal, field_name="ordinal")
        if self.text is not None:
            _ensure_string(self.text, field_name="text")
        if self.span is not None:
            _ensure_instance(self.span, TextSpan, field_name="span")
        if self.parent_segment_id is not None:
            _ensure_non_empty_string(
                self.parent_segment_id,
                field_name="parent_segment_id",
            )
        object.__setattr__(
            self,
            "path",
            _tuple_of_strings(self.path, field_name="path"),
        )
        object.__setattr__(
            self,
            "tokens",
            _tuple_of_instances(self.tokens, TokenSpan, field_name="tokens"),
        )
        object.__setattr__(self, "metadata", _immutable_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class LocalDifferenceCandidate:
    candidate_id: str
    observation_id: str
    kind: LocalDifferenceKind
    left_segment_id: str | None
    right_segment_id: str | None
    left_span: TextSpan | None
    right_span: TextSpan | None
    surface_form: str | None
    normalized_form: str | None
    confidence: float
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_non_empty_string(self.candidate_id, field_name="candidate_id")
        _ensure_non_empty_string(self.observation_id, field_name="observation_id")
        _ensure_enum(self.kind, LocalDifferenceKind, field_name="kind")
        if self.left_segment_id is not None:
            _ensure_non_empty_string(
                self.left_segment_id,
                field_name="left_segment_id",
            )
        if self.right_segment_id is not None:
            _ensure_non_empty_string(
                self.right_segment_id,
                field_name="right_segment_id",
            )
        if self.left_span is not None:
            _ensure_instance(self.left_span, TextSpan, field_name="left_span")
        if self.right_span is not None:
            _ensure_instance(self.right_span, TextSpan, field_name="right_span")
        if self.surface_form is not None:
            _ensure_string(self.surface_form, field_name="surface_form")
        if self.normalized_form is not None:
            _ensure_non_empty_string(
                self.normalized_form,
                field_name="normalized_form",
            )
        object.__setattr__(self, "confidence", _ensure_confidence(self.confidence))
        if (
            self.left_segment_id is None
            and self.right_segment_id is None
            and self.left_span is None
            and self.right_span is None
        ):
            raise ValueError("at least one segment or span reference is required")
        object.__setattr__(self, "metadata", _immutable_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ParserTrace:
    trace_id: str
    observation_id: str
    kind: ParserTraceKind
    message: str
    created_at: datetime
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_non_empty_string(self.trace_id, field_name="trace_id")
        _ensure_non_empty_string(self.observation_id, field_name="observation_id")
        _ensure_enum(self.kind, ParserTraceKind, field_name="kind")
        _ensure_non_empty_string(self.message, field_name="message")
        object.__setattr__(
            self,
            "created_at",
            _ensure_timezone_aware(self.created_at, field_name="created_at"),
        )
        object.__setattr__(self, "metadata", _immutable_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ParserConfig:
    mode: ParserMode
    emit_whitespace_tokens: bool
    emit_control_tokens: bool
    max_segments: int
    max_tokens_per_segment: int
    max_difference_candidates: int

    def __post_init__(self) -> None:
        _ensure_enum(self.mode, ParserMode, field_name="mode")
        _ensure_bool(
            self.emit_whitespace_tokens,
            field_name="emit_whitespace_tokens",
        )
        _ensure_bool(self.emit_control_tokens, field_name="emit_control_tokens")
        _ensure_positive_integer(self.max_segments, field_name="max_segments")
        _ensure_positive_integer(
            self.max_tokens_per_segment,
            field_name="max_tokens_per_segment",
        )
        _ensure_positive_integer(
            self.max_difference_candidates,
            field_name="max_difference_candidates",
        )


@dataclass(frozen=True, slots=True)
class ParseRequest:
    request_id: str
    observation: CanonicalObservation
    config: ParserConfig
    created_at: datetime

    def __post_init__(self) -> None:
        _ensure_non_empty_string(self.request_id, field_name="request_id")
        _ensure_instance(
            self.observation,
            CanonicalObservation,
            field_name="observation",
        )
        _ensure_instance(self.config, ParserConfig, field_name="config")
        object.__setattr__(
            self,
            "created_at",
            _ensure_timezone_aware(self.created_at, field_name="created_at"),
        )


@dataclass(frozen=True, slots=True)
class ParseResult:
    request_id: str
    observation_id: str
    status: ParseStatus
    segments: tuple[ParsedSegment, ...]
    difference_candidates: tuple[LocalDifferenceCandidate, ...]
    traces: tuple[ParserTrace, ...]
    created_at: datetime
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_non_empty_string(self.request_id, field_name="request_id")
        _ensure_non_empty_string(self.observation_id, field_name="observation_id")
        _ensure_enum(self.status, ParseStatus, field_name="status")
        object.__setattr__(
            self,
            "segments",
            _tuple_of_instances(
                self.segments,
                ParsedSegment,
                field_name="segments",
            ),
        )
        object.__setattr__(
            self,
            "difference_candidates",
            _tuple_of_instances(
                self.difference_candidates,
                LocalDifferenceCandidate,
                field_name="difference_candidates",
            ),
        )
        object.__setattr__(
            self,
            "traces",
            _tuple_of_instances(self.traces, ParserTrace, field_name="traces"),
        )
        object.__setattr__(
            self,
            "created_at",
            _ensure_timezone_aware(self.created_at, field_name="created_at"),
        )
        object.__setattr__(self, "metadata", _immutable_metadata(self.metadata))


def _ensure_string(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value


def _ensure_non_empty_string(value: str, *, field_name: str) -> str:
    _ensure_string(value, field_name=field_name)
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


def _ensure_bool(value: bool, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool")
    return value


def _ensure_non_negative_integer(value: int, *, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an int")
    if value < 0:
        raise ValueError(f"{field_name} must be greater than or equal to 0")
    return value


def _ensure_positive_integer(value: int, *, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an int")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than 0")
    return value


def _ensure_confidence(value: float) -> float:
    if type(value) not in {float, int}:
        raise TypeError("confidence must be a number")
    numeric_value = float(value)
    if not isfinite(numeric_value):
        raise ValueError("confidence must be finite")
    if not 0.0 <= numeric_value <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")
    return numeric_value


def _ensure_timezone_aware(value: datetime, *, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _ensure_enum(
    value: object,
    enum_type: type[Enum],
    *,
    field_name: str,
) -> object:
    if not isinstance(value, enum_type):
        raise TypeError(f"{field_name} must be {enum_type.__name__}")
    return value


def _ensure_instance(
    value: object,
    expected_type: type[object],
    *,
    field_name: str,
) -> object:
    if not isinstance(value, expected_type):
        raise TypeError(f"{field_name} must be {expected_type.__name__}")
    return value


def _tuple_of_strings(value: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        value = tuple(value)
    for item in value:
        _ensure_non_empty_string(item, field_name=f"{field_name} item")
    return value


def _tuple_of_instances(
    value: tuple[object, ...],
    expected_type: type[object],
    *,
    field_name: str,
) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        value = tuple(value)
    for item in value:
        _ensure_instance(item, expected_type, field_name=f"{field_name} item")
    return value


def _immutable_metadata(value: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be a mapping")
    for key in value:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
    return MappingProxyType(dict(value))


__all__ = [
    "LocalDifferenceCandidate",
    "LocalDifferenceKind",
    "ParseRequest",
    "ParseResult",
    "ParseStatus",
    "ParsedSegment",
    "ParserConfig",
    "ParserMode",
    "ParserTrace",
    "ParserTraceKind",
    "SegmentKind",
    "TextSpan",
    "TokenKind",
    "TokenSpan",
]
