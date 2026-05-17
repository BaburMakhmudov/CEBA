"""CEBA Parser Layer technical context utilities.

This module builds immutable lookup and context snapshots from parser artifacts.
It does not compute meaning, final delta values, telos, world model state, or
memory updates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

from ois.parser.errors import ParserSchemaError
from ois.parser.schema import (
    LocalDifferenceCandidate,
    ParsedSegment,
    ParseResult,
    TextSpan,
    TokenSpan,
)


_CONTEXT_ID_PREFIX = "pctx"
_DEFAULT_WINDOW_RADIUS = 3
_MAX_WINDOW_RADIUS = 64
_MAX_CONTEXT_ITEMS = 100_000

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


class ParserContextKind(str, Enum):
    RESULT = "result"
    SEGMENT = "segment"
    TOKEN_WINDOW = "token_window"
    CANDIDATE = "candidate"


class ContextDirection(str, Enum):
    BEFORE = "before"
    AFTER = "after"
    AROUND = "around"


@dataclass(frozen=True, slots=True)
class SegmentContext:
    context_id: str
    observation_id: str
    segment_id: str
    segment: ParsedSegment
    previous_segment_id: str | None = None
    next_segment_id: str | None = None
    parent_segment_id: str | None = None
    child_segment_ids: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_non_empty_string(self.context_id, field_name="context_id")
        observation_id = _ensure_non_empty_string(
            self.observation_id,
            field_name="observation_id",
        )
        segment_id = _ensure_non_empty_string(
            self.segment_id,
            field_name="segment_id",
        )
        segment = _ensure_segment(self.segment)
        if segment.observation_id != observation_id:
            raise ParserSchemaError(
                "segment observation_id must match context observation_id",
                observation_id=observation_id,
                segment_id=segment.segment_id,
            )
        if segment.segment_id != segment_id:
            raise ParserSchemaError(
                "segment_id must match segment.segment_id",
                observation_id=observation_id,
                segment_id=segment.segment_id,
            )
        _ensure_optional_id(
            self.previous_segment_id,
            field_name="previous_segment_id",
        )
        _ensure_optional_id(self.next_segment_id, field_name="next_segment_id")
        parent_segment_id = _ensure_optional_id(
            self.parent_segment_id,
            field_name="parent_segment_id",
        )
        if parent_segment_id != segment.parent_segment_id:
            raise ParserSchemaError(
                "parent_segment_id must match segment.parent_segment_id",
                observation_id=observation_id,
                segment_id=segment_id,
            )
        object.__setattr__(
            self,
            "child_segment_ids",
            _tuple_of_ids(self.child_segment_ids, field_name="child_segment_ids"),
        )
        object.__setattr__(self, "metadata", _ensure_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class TokenWindow:
    context_id: str
    observation_id: str
    segment_id: str
    center_token_id: str | None
    direction: ContextDirection
    tokens: tuple[TokenSpan, ...]
    span: TextSpan | None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_non_empty_string(self.context_id, field_name="context_id")
        observation_id = _ensure_non_empty_string(
            self.observation_id,
            field_name="observation_id",
        )
        segment_id = _ensure_non_empty_string(
            self.segment_id,
            field_name="segment_id",
        )
        _ensure_optional_id(self.center_token_id, field_name="center_token_id")
        direction = _ensure_direction(self.direction)
        tokens = _ensure_tokens(self.tokens)
        for token in tokens:
            if token.segment_id != segment_id:
                raise ParserSchemaError(
                    "window token segment_id must match window segment_id",
                    observation_id=observation_id,
                    segment_id=segment_id,
                    token_id=token.token_id,
                )
        if tokens:
            if self.center_token_id is None:
                raise ParserSchemaError(
                    "center_token_id is required when window contains tokens",
                    observation_id=observation_id,
                    segment_id=segment_id,
                )
            expected_span = TextSpan(tokens[0].span.start, tokens[-1].span.end)
            if self.span != expected_span:
                raise ParserSchemaError(
                    "span must cover selected window tokens",
                    observation_id=observation_id,
                    segment_id=segment_id,
                    context={
                        "expected_start": expected_span.start,
                        "expected_end": expected_span.end,
                    },
                )
        elif self.span is not None:
            raise ParserSchemaError(
                "span must be None when window contains no tokens",
                observation_id=observation_id,
                segment_id=segment_id,
            )
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "metadata", _ensure_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class CandidateContext:
    context_id: str
    observation_id: str
    candidate_id: str
    candidate: LocalDifferenceCandidate
    left_segment: ParsedSegment | None
    right_segment: ParsedSegment | None
    left_tokens: tuple[TokenSpan, ...] = field(default_factory=tuple)
    right_tokens: tuple[TokenSpan, ...] = field(default_factory=tuple)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_non_empty_string(self.context_id, field_name="context_id")
        observation_id = _ensure_non_empty_string(
            self.observation_id,
            field_name="observation_id",
        )
        candidate_id = _ensure_non_empty_string(
            self.candidate_id,
            field_name="candidate_id",
        )
        candidate = _ensure_candidate(self.candidate)
        if candidate.observation_id != observation_id:
            raise ParserSchemaError(
                "candidate observation_id must match context observation_id",
                observation_id=observation_id,
                candidate_id=candidate.candidate_id,
            )
        if candidate.candidate_id != candidate_id:
            raise ParserSchemaError(
                "candidate_id must match candidate.candidate_id",
                observation_id=observation_id,
                candidate_id=candidate.candidate_id,
            )
        if self.left_segment is not None:
            left_segment = _ensure_segment(self.left_segment)
            if left_segment.segment_id != candidate.left_segment_id:
                raise ParserSchemaError(
                    "left_segment must match candidate.left_segment_id",
                    observation_id=observation_id,
                    segment_id=left_segment.segment_id,
                    candidate_id=candidate_id,
                )
        if self.right_segment is not None:
            right_segment = _ensure_segment(self.right_segment)
            if right_segment.segment_id != candidate.right_segment_id:
                raise ParserSchemaError(
                    "right_segment must match candidate.right_segment_id",
                    observation_id=observation_id,
                    segment_id=right_segment.segment_id,
                    candidate_id=candidate_id,
                )
        if self.left_segment is None and candidate.left_segment_id is not None:
            raise ParserSchemaError(
                "left_segment is required for candidate.left_segment_id",
                observation_id=observation_id,
                segment_id=candidate.left_segment_id,
                candidate_id=candidate_id,
            )
        if self.right_segment is None and candidate.right_segment_id is not None:
            raise ParserSchemaError(
                "right_segment is required for candidate.right_segment_id",
                observation_id=observation_id,
                segment_id=candidate.right_segment_id,
                candidate_id=candidate_id,
            )
        object.__setattr__(self, "left_tokens", _ensure_tokens(self.left_tokens))
        object.__setattr__(self, "right_tokens", _ensure_tokens(self.right_tokens))
        object.__setattr__(self, "metadata", _ensure_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ParserContextSnapshot:
    context_id: str
    request_id: str
    observation_id: str
    segments_by_id: Mapping[str, ParsedSegment]
    tokens_by_id: Mapping[str, TokenSpan]
    candidates_by_id: Mapping[str, LocalDifferenceCandidate]
    segment_contexts_by_id: Mapping[str, SegmentContext]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        observation_id = _ensure_non_empty_string(
            self.observation_id,
            field_name="observation_id",
        )
        _ensure_non_empty_string(self.context_id, field_name="context_id")
        _ensure_non_empty_string(self.request_id, field_name="request_id")
        segments = _copy_mapping(
            self.segments_by_id,
            expected_type=ParsedSegment,
            field_name="segments_by_id",
        )
        tokens = _copy_mapping(
            self.tokens_by_id,
            expected_type=TokenSpan,
            field_name="tokens_by_id",
        )
        candidates = _copy_mapping(
            self.candidates_by_id,
            expected_type=LocalDifferenceCandidate,
            field_name="candidates_by_id",
        )
        segment_contexts = _copy_mapping(
            self.segment_contexts_by_id,
            expected_type=SegmentContext,
            field_name="segment_contexts_by_id",
        )
        for key, segment in segments.items():
            if key != segment.segment_id:
                raise ParserSchemaError(
                    "segments_by_id key must match segment_id",
                    observation_id=observation_id,
                    segment_id=segment.segment_id,
                )
        for key, token in tokens.items():
            if key != token.token_id:
                raise ParserSchemaError(
                    "tokens_by_id key must match token_id",
                    observation_id=observation_id,
                    token_id=token.token_id,
                )
        for key, candidate in candidates.items():
            if key != candidate.candidate_id:
                raise ParserSchemaError(
                    "candidates_by_id key must match candidate_id",
                    observation_id=observation_id,
                    candidate_id=candidate.candidate_id,
                )
        object.__setattr__(
            self,
            "segments_by_id",
            _immutable_mapping(segments, field_name="segments_by_id"),
        )
        object.__setattr__(
            self,
            "tokens_by_id",
            _immutable_mapping(tokens, field_name="tokens_by_id"),
        )
        object.__setattr__(
            self,
            "candidates_by_id",
            _immutable_mapping(candidates, field_name="candidates_by_id"),
        )
        object.__setattr__(
            self,
            "segment_contexts_by_id",
            _immutable_mapping(segment_contexts, field_name="segment_contexts_by_id"),
        )
        object.__setattr__(self, "metadata", _ensure_metadata(self.metadata))


def build_context_snapshot(result: ParseResult) -> ParserContextSnapshot:
    normalized_result = _ensure_parse_result(result)
    request_id = _ensure_non_empty_string(
        normalized_result.request_id,
        field_name="request_id",
    )
    observation_id = _ensure_non_empty_string(
        normalized_result.observation_id,
        field_name="observation_id",
    )
    segments = _ensure_segments(normalized_result.segments)
    candidates = _ensure_candidates(normalized_result.difference_candidates)

    segments_by_id: dict[str, ParsedSegment] = {}
    tokens_by_id: dict[str, TokenSpan] = {}
    candidates_by_id: dict[str, LocalDifferenceCandidate] = {}

    for segment in segments:
        if segment.observation_id != observation_id:
            raise ParserSchemaError(
                "segment observation_id must match parse result observation_id",
                observation_id=observation_id,
                segment_id=segment.segment_id,
            )
        if segment.segment_id in segments_by_id:
            raise ParserSchemaError(
                "duplicate segment_id in parse result",
                observation_id=observation_id,
                segment_id=segment.segment_id,
            )
        segments_by_id[segment.segment_id] = segment
        for token in _ensure_tokens(segment.tokens):
            if token.token_id in tokens_by_id:
                raise ParserSchemaError(
                    "duplicate token_id in parse result",
                    observation_id=observation_id,
                    segment_id=segment.segment_id,
                    token_id=token.token_id,
                )
            tokens_by_id[token.token_id] = token

    for candidate in candidates:
        if candidate.observation_id != observation_id:
            raise ParserSchemaError(
                "candidate observation_id must match parse result observation_id",
                observation_id=observation_id,
                candidate_id=candidate.candidate_id,
            )
        if candidate.candidate_id in candidates_by_id:
            raise ParserSchemaError(
                "duplicate candidate_id in parse result",
                observation_id=observation_id,
                candidate_id=candidate.candidate_id,
            )
        candidates_by_id[candidate.candidate_id] = candidate

    total_items = len(segments_by_id) + len(tokens_by_id) + len(candidates_by_id)
    if total_items > _MAX_CONTEXT_ITEMS:
        raise ParserSchemaError(
            "parser context item limit exceeded",
            observation_id=observation_id,
            context={"item_count": total_items, "max_items": _MAX_CONTEXT_ITEMS},
        )

    segment_contexts = build_segment_contexts(
        observation_id=observation_id,
        segments=segments,
    )
    return ParserContextSnapshot(
        context_id=make_context_id(
            observation_id=observation_id,
            kind=ParserContextKind.RESULT,
            subject_id=request_id,
        ),
        request_id=request_id,
        observation_id=observation_id,
        segments_by_id=segments_by_id,
        tokens_by_id=tokens_by_id,
        candidates_by_id=candidates_by_id,
        segment_contexts_by_id=segment_contexts,
        metadata={
            "context_kind": ParserContextKind.RESULT.value,
            "segment_count": len(segments_by_id),
            "token_count": len(tokens_by_id),
            "candidate_count": len(candidates_by_id),
            "status": "built",
        },
    )


def build_segment_contexts(
    *,
    observation_id: str,
    segments: tuple[ParsedSegment, ...],
) -> Mapping[str, SegmentContext]:
    normalized_observation_id = _ensure_non_empty_string(
        observation_id,
        field_name="observation_id",
    )
    normalized_segments = _ensure_segments(segments)
    segments_by_id: dict[str, ParsedSegment] = {}
    children_by_parent_id: dict[str, list[str]] = {}

    for segment in normalized_segments:
        if segment.observation_id != normalized_observation_id:
            raise ParserSchemaError(
                "segment observation_id must match context observation_id",
                observation_id=normalized_observation_id,
                segment_id=segment.segment_id,
            )
        if segment.segment_id in segments_by_id:
            raise ParserSchemaError(
                "duplicate segment_id in segment contexts",
                observation_id=normalized_observation_id,
                segment_id=segment.segment_id,
            )
        segments_by_id[segment.segment_id] = segment

    for segment in normalized_segments:
        if segment.parent_segment_id is not None:
            if segment.parent_segment_id in segments_by_id:
                children_by_parent_id.setdefault(segment.parent_segment_id, []).append(
                    segment.segment_id
                )
            continue
        for parent in normalized_segments:
            if parent.segment_id == segment.segment_id:
                continue
            if _path_is_immediate_parent(parent.path, segment.path):
                children_by_parent_id.setdefault(parent.segment_id, []).append(
                    segment.segment_id
                )
                break

    contexts: dict[str, SegmentContext] = {}
    segment_count = len(normalized_segments)
    for index, segment in enumerate(normalized_segments):
        previous_segment_id = (
            normalized_segments[index - 1].segment_id if index > 0 else None
        )
        next_segment_id = (
            normalized_segments[index + 1].segment_id
            if index + 1 < segment_count
            else None
        )
        child_segment_ids = tuple(children_by_parent_id.get(segment.segment_id, ()))
        contexts[segment.segment_id] = SegmentContext(
            context_id=make_context_id(
                observation_id=normalized_observation_id,
                kind=ParserContextKind.SEGMENT,
                subject_id=segment.segment_id,
            ),
            observation_id=normalized_observation_id,
            segment_id=segment.segment_id,
            segment=segment,
            previous_segment_id=previous_segment_id,
            next_segment_id=next_segment_id,
            parent_segment_id=segment.parent_segment_id,
            child_segment_ids=child_segment_ids,
            metadata={
                "context_kind": ParserContextKind.SEGMENT.value,
                "source": "parse_result_segments",
            },
        )
    return _immutable_mapping(contexts, field_name="segment_contexts")


def get_token_window(
    *,
    segment: ParsedSegment,
    center_token_id: str | None = None,
    radius: int = _DEFAULT_WINDOW_RADIUS,
    direction: ContextDirection = ContextDirection.AROUND,
) -> TokenWindow:
    normalized_segment = _ensure_segment(segment)
    normalized_radius = _ensure_radius(radius)
    normalized_direction = _ensure_direction(direction)
    tokens = _ensure_tokens(normalized_segment.tokens)

    if not tokens:
        if center_token_id is not None:
            raise ParserSchemaError(
                "center_token_id must be None when segment has no tokens",
                observation_id=normalized_segment.observation_id,
                segment_id=normalized_segment.segment_id,
                token_id=center_token_id,
            )
        selected_tokens: tuple[TokenSpan, ...] = ()
        normalized_center_token_id = None
    else:
        normalized_center_token_id = (
            tokens[0].token_id
            if center_token_id is None
            else _ensure_non_empty_string(
                center_token_id,
                field_name="center_token_id",
            )
        )
        center_index = -1
        for index, token in enumerate(tokens):
            if token.token_id == normalized_center_token_id:
                center_index = index
                break
        if center_index < 0:
            raise ParserSchemaError(
                "unknown center_token_id for segment",
                observation_id=normalized_segment.observation_id,
                segment_id=normalized_segment.segment_id,
                token_id=normalized_center_token_id,
            )

        if normalized_direction == ContextDirection.BEFORE:
            start = max(0, center_index - normalized_radius)
            end = center_index
        elif normalized_direction == ContextDirection.AFTER:
            start = center_index + 1
            end = min(len(tokens), center_index + normalized_radius + 1)
        else:
            start = max(0, center_index - normalized_radius)
            end = min(len(tokens), center_index + normalized_radius + 1)
        selected_tokens = tokens[start:end]

    span = (
        TextSpan(selected_tokens[0].span.start, selected_tokens[-1].span.end)
        if selected_tokens
        else None
    )
    return TokenWindow(
        context_id=make_context_id(
            observation_id=normalized_segment.observation_id,
            kind=ParserContextKind.TOKEN_WINDOW,
            subject_id=(
                f"{normalized_segment.segment_id}:"
                f"{normalized_center_token_id or 'empty'}:"
                f"{normalized_direction.value}:"
                f"{normalized_radius}"
            ),
        ),
        observation_id=normalized_segment.observation_id,
        segment_id=normalized_segment.segment_id,
        center_token_id=normalized_center_token_id,
        direction=normalized_direction,
        tokens=selected_tokens,
        span=span,
        metadata={
            "context_kind": ParserContextKind.TOKEN_WINDOW.value,
            "window_radius": normalized_radius,
            "direction": normalized_direction.value,
        },
    )


def get_candidate_context(
    *,
    candidate: LocalDifferenceCandidate,
    segments_by_id: Mapping[str, ParsedSegment],
    radius: int = _DEFAULT_WINDOW_RADIUS,
) -> CandidateContext:
    normalized_candidate = _ensure_candidate(candidate)
    normalized_segments = _ensure_mapping_of_segments(segments_by_id)
    normalized_radius = _ensure_radius(radius)

    left_segment = None
    right_segment = None
    if normalized_candidate.left_segment_id is not None:
        left_segment = normalized_segments.get(normalized_candidate.left_segment_id)
        if left_segment is None:
            raise ParserSchemaError(
                "candidate left_segment_id is missing from segments_by_id",
                observation_id=normalized_candidate.observation_id,
                segment_id=normalized_candidate.left_segment_id,
                candidate_id=normalized_candidate.candidate_id,
            )
    if normalized_candidate.right_segment_id is not None:
        right_segment = normalized_segments.get(normalized_candidate.right_segment_id)
        if right_segment is None:
            raise ParserSchemaError(
                "candidate right_segment_id is missing from segments_by_id",
                observation_id=normalized_candidate.observation_id,
                segment_id=normalized_candidate.right_segment_id,
                candidate_id=normalized_candidate.candidate_id,
            )

    left_tokens = (
        _expand_token_window(
            tokens=left_segment.tokens,
            selected_tokens=_tokens_overlapping_span(
                left_segment.tokens,
                normalized_candidate.left_span,
            ),
            radius=normalized_radius,
        )
        if left_segment is not None and normalized_candidate.left_span is not None
        else ()
    )
    right_tokens = (
        _expand_token_window(
            tokens=right_segment.tokens,
            selected_tokens=_tokens_overlapping_span(
                right_segment.tokens,
                normalized_candidate.right_span,
            ),
            radius=normalized_radius,
        )
        if right_segment is not None and normalized_candidate.right_span is not None
        else ()
    )
    return CandidateContext(
        context_id=make_context_id(
            observation_id=normalized_candidate.observation_id,
            kind=ParserContextKind.CANDIDATE,
            subject_id=normalized_candidate.candidate_id,
        ),
        observation_id=normalized_candidate.observation_id,
        candidate_id=normalized_candidate.candidate_id,
        candidate=normalized_candidate,
        left_segment=left_segment,
        right_segment=right_segment,
        left_tokens=left_tokens,
        right_tokens=right_tokens,
        metadata={
            "context_kind": ParserContextKind.CANDIDATE.value,
            "window_radius": normalized_radius,
            "source": "difference_candidate",
        },
    )


def make_context_id(
    *,
    observation_id: str,
    kind: ParserContextKind,
    subject_id: str,
) -> str:
    normalized_observation_id = _ensure_non_empty_string(
        observation_id,
        field_name="observation_id",
    )
    normalized_kind = _ensure_context_kind(kind)
    normalized_subject_id = _ensure_non_empty_string(
        subject_id,
        field_name="subject_id",
    )
    return (
        f"{_CONTEXT_ID_PREFIX}:"
        f"{_sanitize_id_part(normalized_observation_id)}:"
        f"{normalized_kind.value}:"
        f"{_sanitize_id_part(normalized_subject_id)}"
    )


def _ensure_parse_result(value: ParseResult) -> ParseResult:
    if not isinstance(value, ParseResult):
        raise ParserSchemaError(
            "result must be ParseResult",
            context={"type": type(value).__name__},
        )
    _ensure_non_empty_string(value.request_id, field_name="request_id")
    _ensure_non_empty_string(value.observation_id, field_name="observation_id")
    _ensure_segments(value.segments)
    _ensure_candidates(value.difference_candidates)
    return value


def _ensure_segment(value: ParsedSegment) -> ParsedSegment:
    if not isinstance(value, ParsedSegment):
        raise ParserSchemaError(
            "segment must be ParsedSegment",
            context={"type": type(value).__name__},
        )
    _ensure_non_empty_string(value.observation_id, field_name="observation_id")
    _ensure_non_empty_string(value.segment_id, field_name="segment_id")
    _ensure_optional_id(value.parent_segment_id, field_name="parent_segment_id")
    _ensure_tokens(value.tokens)
    return value


def _ensure_token(value: TokenSpan) -> TokenSpan:
    if not isinstance(value, TokenSpan):
        raise ParserSchemaError(
            "token must be TokenSpan",
            context={"type": type(value).__name__},
        )
    _ensure_non_empty_string(value.token_id, field_name="token_id")
    _ensure_non_empty_string(value.segment_id, field_name="segment_id")
    return value


def _ensure_candidate(
    value: LocalDifferenceCandidate,
) -> LocalDifferenceCandidate:
    if not isinstance(value, LocalDifferenceCandidate):
        raise ParserSchemaError(
            "candidate must be LocalDifferenceCandidate",
            context={"type": type(value).__name__},
        )
    _ensure_non_empty_string(value.candidate_id, field_name="candidate_id")
    _ensure_non_empty_string(value.observation_id, field_name="observation_id")
    _ensure_optional_id(value.left_segment_id, field_name="left_segment_id")
    _ensure_optional_id(value.right_segment_id, field_name="right_segment_id")
    return value


def _ensure_segments(value: tuple[ParsedSegment, ...]) -> tuple[ParsedSegment, ...]:
    if not isinstance(value, tuple):
        raise ParserSchemaError(
            "segments must be a tuple",
            context={"type": type(value).__name__},
        )
    for segment in value:
        _ensure_segment(segment)
    return value


def _ensure_tokens(value: tuple[TokenSpan, ...]) -> tuple[TokenSpan, ...]:
    if not isinstance(value, tuple):
        raise ParserSchemaError(
            "tokens must be a tuple",
            context={"type": type(value).__name__},
        )
    for token in value:
        _ensure_token(token)
    return value


def _ensure_candidates(
    value: tuple[LocalDifferenceCandidate, ...],
) -> tuple[LocalDifferenceCandidate, ...]:
    if not isinstance(value, tuple):
        raise ParserSchemaError(
            "candidates must be a tuple",
            context={"type": type(value).__name__},
        )
    for candidate in value:
        _ensure_candidate(candidate)
    return value


def _ensure_mapping_of_segments(
    value: Mapping[str, ParsedSegment],
) -> Mapping[str, ParsedSegment]:
    if not isinstance(value, Mapping):
        raise ParserSchemaError(
            "segments_by_id must be a mapping",
            context={"type": type(value).__name__},
        )
    copied: dict[str, ParsedSegment] = {}
    for key, segment in value.items():
        normalized_key = _ensure_non_empty_string(key, field_name="segment_id")
        normalized_segment = _ensure_segment(segment)
        if normalized_key != normalized_segment.segment_id:
            raise ParserSchemaError(
                "segments_by_id key must match segment.segment_id",
                observation_id=normalized_segment.observation_id,
                segment_id=normalized_segment.segment_id,
            )
        copied[normalized_key] = normalized_segment
    return MappingProxyType(copied)


def _ensure_context_kind(value: ParserContextKind) -> ParserContextKind:
    if not isinstance(value, ParserContextKind):
        raise ParserSchemaError(
            "kind must be ParserContextKind",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_direction(value: ContextDirection) -> ContextDirection:
    if not isinstance(value, ContextDirection):
        raise ParserSchemaError(
            "direction must be ContextDirection",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_non_empty_string(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ParserSchemaError(
            f"{field_name} must be a string",
            context={"type": type(value).__name__},
        )
    if not value:
        raise ParserSchemaError(f"{field_name} must not be empty")
    if any(ord(character) < 32 for character in value):
        raise ParserSchemaError(f"{field_name} must not contain control characters")
    return value


def _ensure_optional_id(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _ensure_non_empty_string(value, field_name=field_name)


def _ensure_radius(value: int) -> int:
    if type(value) is not int:
        raise ParserSchemaError(
            "radius must be an int",
            context={"type": type(value).__name__},
        )
    if value < 0 or value > _MAX_WINDOW_RADIUS:
        raise ParserSchemaError(
            "radius must be between 0 and maximum window radius",
            context={"radius": value, "max_radius": _MAX_WINDOW_RADIUS},
        )
    return value


def _ensure_metadata(value: Mapping[str, object] | None) -> Mapping[str, object]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ParserSchemaError(
            "metadata must be a mapping",
            context={"type": type(value).__name__},
        )
    copied: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ParserSchemaError(
                "metadata keys must be strings",
                context={"type": type(key).__name__},
            )
        normalized_key = key.strip().lower().replace("-", "_")
        for forbidden_key in _FORBIDDEN_METADATA_KEYS:
            if normalized_key == forbidden_key or forbidden_key in normalized_key:
                raise ParserSchemaError(
                    "metadata key crosses parser context boundary",
                    context={"metadata_key": key},
                )
        copied[key] = item
    return MappingProxyType(copied)


def _immutable_mapping(
    value: Mapping[str, object],
    *,
    field_name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ParserSchemaError(
            f"{field_name} must be a mapping",
            context={"type": type(value).__name__},
        )
    copied: dict[str, object] = {}
    for key, item in value.items():
        copied[_ensure_non_empty_string(key, field_name=f"{field_name} key")] = item
    return MappingProxyType(copied)


def _contains_disallowed_control_chars(value: str) -> bool:
    return any(character.isspace() or ord(character) < 32 for character in value)


def _sanitize_id_part(value: str) -> str:
    normalized_value = _ensure_non_empty_string(value, field_name="id_part")
    if _contains_disallowed_control_chars(normalized_value):
        raise ParserSchemaError("id_part must not contain whitespace or controls")
    return normalized_value


def _spans_overlap(left: TextSpan, right: TextSpan) -> bool:
    if not isinstance(left, TextSpan):
        raise ParserSchemaError(
            "left span must be TextSpan",
            context={"type": type(left).__name__},
        )
    if not isinstance(right, TextSpan):
        raise ParserSchemaError(
            "right span must be TextSpan",
            context={"type": type(right).__name__},
        )
    return left.start < right.end and right.start < left.end


def _tokens_overlapping_span(
    tokens: tuple[TokenSpan, ...],
    span: TextSpan | None,
) -> tuple[TokenSpan, ...]:
    normalized_tokens = _ensure_tokens(tokens)
    if span is None:
        return ()
    if not isinstance(span, TextSpan):
        raise ParserSchemaError(
            "span must be TextSpan",
            context={"type": type(span).__name__},
        )
    return tuple(
        token for token in normalized_tokens if _spans_overlap(token.span, span)
    )


def _expand_token_window(
    *,
    tokens: tuple[TokenSpan, ...],
    selected_tokens: tuple[TokenSpan, ...],
    radius: int,
) -> tuple[TokenSpan, ...]:
    normalized_tokens = _ensure_tokens(tokens)
    normalized_selected_tokens = _ensure_tokens(selected_tokens)
    normalized_radius = _ensure_radius(radius)
    if not normalized_tokens or not normalized_selected_tokens:
        return ()

    selected_ids = {token.token_id for token in normalized_selected_tokens}
    selected_indexes = [
        index
        for index, token in enumerate(normalized_tokens)
        if token.token_id in selected_ids
    ]
    if not selected_indexes:
        return ()
    start = max(0, min(selected_indexes) - normalized_radius)
    end = min(len(normalized_tokens), max(selected_indexes) + normalized_radius + 1)
    return normalized_tokens[start:end]


def _path_is_immediate_parent(
    parent: tuple[str, ...],
    child: tuple[str, ...],
) -> bool:
    if not isinstance(parent, tuple):
        raise ParserSchemaError(
            "parent path must be a tuple",
            context={"type": type(parent).__name__},
        )
    if not isinstance(child, tuple):
        raise ParserSchemaError(
            "child path must be a tuple",
            context={"type": type(child).__name__},
        )
    for item in parent:
        _ensure_non_empty_string(item, field_name="parent path item")
    for item in child:
        _ensure_non_empty_string(item, field_name="child path item")
    return len(child) == len(parent) + 1 and child[: len(parent)] == parent


def _tuple_of_ids(value: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ParserSchemaError(
            f"{field_name} must be a tuple",
            context={"type": type(value).__name__},
        )
    return tuple(
        _ensure_non_empty_string(item, field_name=f"{field_name} item")
        for item in value
    )


def _copy_mapping(
    value: Mapping[str, object],
    *,
    expected_type: type[object],
    field_name: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ParserSchemaError(
            f"{field_name} must be a mapping",
            context={"type": type(value).__name__},
        )
    copied: dict[str, object] = {}
    for key, item in value.items():
        normalized_key = _ensure_non_empty_string(key, field_name=f"{field_name} key")
        if not isinstance(item, expected_type):
            raise ParserSchemaError(
                f"{field_name} values must be {expected_type.__name__}",
                context={"type": type(item).__name__},
            )
        copied[normalized_key] = item
    return copied


__all__ = [
    "ParserContextKind",
    "ContextDirection",
    "SegmentContext",
    "TokenWindow",
    "CandidateContext",
    "ParserContextSnapshot",
    "build_context_snapshot",
    "build_segment_contexts",
    "get_token_window",
    "get_candidate_context",
    "make_context_id",
]
