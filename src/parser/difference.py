"""CEBA Parser Layer local difference-candidate extraction.

This module produces LocalDifferenceCandidate values from tokenized
ParsedSegment objects. It does not compute final delta, meaning, telos,
valuation, world model state, or memory updates.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum

from ois.parser.errors import (
    LocalDifferenceExtractionError,
    ParserLimitExceededError,
)
from ois.parser.schema import (
    LocalDifferenceCandidate,
    LocalDifferenceKind,
    ParsedSegment,
    ParserConfig,
    TextSpan,
    TokenKind,
    TokenSpan,
)


_CANDIDATE_ID_PREFIX = "ldc"
_MAX_COMPARISON_TOKEN_WINDOW = 512
_MAX_REPETITION_RUN = 128
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


class DifferenceExtractionStrategy(str, Enum):
    LOCAL_SURFACE = "local_surface"


@dataclass(frozen=True, slots=True)
class DifferenceExtractionRequest:
    segments: tuple[ParsedSegment, ...]
    config: ParserConfig
    strategy: DifferenceExtractionStrategy = (
        DifferenceExtractionStrategy.LOCAL_SURFACE
    )

    def __post_init__(self) -> None:
        _ensure_segments(self.segments)
        _ensure_config(self.config)
        _ensure_strategy(self.strategy)


def extract_local_differences(
    request: DifferenceExtractionRequest,
) -> tuple[LocalDifferenceCandidate, ...]:
    normalized_request = _ensure_request(request)
    segments = _ensure_segments(normalized_request.segments)
    config = _ensure_config(normalized_request.config)
    _ensure_strategy(normalized_request.strategy)

    if not segments or not any(segment.tokens for segment in segments):
        return ()

    candidates: list[LocalDifferenceCandidate] = []
    for segment in segments:
        candidates.extend(
            extract_segment_repetition_candidates(segment=segment, config=config)
        )
        _enforce_candidate_limit(
            len(candidates),
            config,
            observation_id=segment.observation_id,
        )

    for left, right in _neighbor_pairs(segments):
        candidates.extend(
            compare_neighboring_segments(left=left, right=right, config=config)
        )
        _enforce_candidate_limit(
            len(candidates),
            config,
            observation_id=left.observation_id,
        )

    return tuple(candidates)


def extract_segment_repetition_candidates(
    *,
    segment: ParsedSegment,
    config: ParserConfig,
) -> tuple[LocalDifferenceCandidate, ...]:
    normalized_segment = _ensure_segment(segment)
    normalized_config = _ensure_config(config)
    tokens = _filtered_tokens(normalized_segment)
    if not tokens:
        return ()

    candidates: list[LocalDifferenceCandidate] = []
    cursor = 0
    while cursor < len(tokens):
        run_start = cursor
        compare_form = _token_compare_form(tokens[cursor])
        cursor += 1
        while (
            cursor < len(tokens)
            and cursor - run_start < _MAX_REPETITION_RUN
            and _token_compare_form(tokens[cursor]) == compare_form
        ):
            cursor += 1

        repeat_count = cursor - run_start
        if repeat_count > 1:
            run_tokens = tokens[run_start:cursor]
            first_token = run_tokens[0]
            last_token = run_tokens[-1]
            candidate = _build_candidate(
                observation_id=normalized_segment.observation_id,
                ordinal=len(candidates),
                kind=LocalDifferenceKind.REPETITION,
                left_segment_id=normalized_segment.segment_id,
                right_segment_id=normalized_segment.segment_id,
                left_span=first_token.span,
                right_span=TextSpan(first_token.span.start, last_token.span.end),
                surface_form=_join_token_surfaces(run_tokens),
                normalized_form=compare_form,
                confidence=1.0,
                metadata={
                    "source": "segment_repetition",
                    "token_kind": first_token.kind.value,
                    "repeat_count": repeat_count,
                    "start_token_ordinal": first_token.ordinal,
                    "end_token_ordinal": last_token.ordinal,
                },
            )
            candidates.append(candidate)
            _enforce_candidate_limit(
                len(candidates),
                normalized_config,
                observation_id=normalized_segment.observation_id,
            )

    return tuple(candidates)


def compare_neighboring_segments(
    *,
    left: ParsedSegment,
    right: ParsedSegment,
    config: ParserConfig,
) -> tuple[LocalDifferenceCandidate, ...]:
    left_segment = _ensure_segment(left)
    right_segment = _ensure_segment(right)
    normalized_config = _ensure_config(config)
    left_tokens = _filtered_tokens(left_segment)[:_MAX_COMPARISON_TOKEN_WINDOW]
    right_tokens = _filtered_tokens(right_segment)[:_MAX_COMPARISON_TOKEN_WINDOW]

    if not left_tokens and not right_tokens:
        return ()
    if left_segment.kind != right_segment.kind and not (left_tokens and right_tokens):
        return ()

    observation_id = left_segment.observation_id
    candidates: list[LocalDifferenceCandidate] = []
    aligned_count = min(len(left_tokens), len(right_tokens))

    for index in range(aligned_count):
        left_token = left_tokens[index]
        right_token = right_tokens[index]
        if _token_compare_form(left_token) == _token_compare_form(right_token):
            continue
        kind, confidence = _classify_aligned_difference(left_token, right_token)
        candidate = _build_candidate(
            observation_id=observation_id,
            ordinal=len(candidates),
            kind=kind,
            left_segment_id=left_segment.segment_id,
            right_segment_id=right_segment.segment_id,
            left_span=left_token.span,
            right_span=right_token.span,
            surface_form=f"{_token_surface_form(left_token)}|{_token_surface_form(right_token)}",
            normalized_form=(
                f"{_token_compare_form(left_token)}|"
                f"{_token_compare_form(right_token)}"
            ),
            confidence=confidence,
            metadata={
                "source": "neighbor_segment_comparison",
                "left_token_ordinal": left_token.ordinal,
                "right_token_ordinal": right_token.ordinal,
                "left_token_kind": left_token.kind.value,
                "right_token_kind": right_token.kind.value,
                "comparison_window": _MAX_COMPARISON_TOKEN_WINDOW,
            },
        )
        candidates.append(candidate)
        _enforce_candidate_limit(
            len(candidates),
            normalized_config,
            observation_id=observation_id,
        )

    for left_token in left_tokens[aligned_count:]:
        candidate = _build_candidate(
            observation_id=observation_id,
            ordinal=len(candidates),
            kind=LocalDifferenceKind.TOKEN_DELETION,
            left_segment_id=left_segment.segment_id,
            right_segment_id=right_segment.segment_id,
            left_span=left_token.span,
            right_span=None,
            surface_form=_token_surface_form(left_token),
            normalized_form=_token_compare_form(left_token),
            confidence=1.0,
            metadata={
                "source": "neighbor_segment_comparison",
                "left_token_ordinal": left_token.ordinal,
                "right_token_ordinal": None,
                "left_token_kind": left_token.kind.value,
                "right_token_kind": None,
                "comparison_window": _MAX_COMPARISON_TOKEN_WINDOW,
            },
        )
        candidates.append(candidate)
        _enforce_candidate_limit(
            len(candidates),
            normalized_config,
            observation_id=observation_id,
        )

    for right_token in right_tokens[aligned_count:]:
        candidate = _build_candidate(
            observation_id=observation_id,
            ordinal=len(candidates),
            kind=LocalDifferenceKind.TOKEN_INSERTION,
            left_segment_id=left_segment.segment_id,
            right_segment_id=right_segment.segment_id,
            left_span=None,
            right_span=right_token.span,
            surface_form=_token_surface_form(right_token),
            normalized_form=_token_compare_form(right_token),
            confidence=1.0,
            metadata={
                "source": "neighbor_segment_comparison",
                "left_token_ordinal": None,
                "right_token_ordinal": right_token.ordinal,
                "left_token_kind": None,
                "right_token_kind": right_token.kind.value,
                "comparison_window": _MAX_COMPARISON_TOKEN_WINDOW,
            },
        )
        candidates.append(candidate)
        _enforce_candidate_limit(
            len(candidates),
            normalized_config,
            observation_id=observation_id,
        )

    return tuple(candidates)


def make_candidate_id(
    *,
    observation_id: str,
    kind: LocalDifferenceKind,
    ordinal: int,
    left_segment_id: str | None = None,
    right_segment_id: str | None = None,
    left_span: TextSpan | None = None,
    right_span: TextSpan | None = None,
) -> str:
    normalized_observation_id = _sanitize_id_part(
        _ensure_non_empty_string(observation_id, field_name="observation_id")
    )
    normalized_kind = _ensure_candidate_kind(kind)
    if type(ordinal) is not int:
        raise LocalDifferenceExtractionError(
            "ordinal must be an int",
            observation_id=normalized_observation_id,
            context={"type": type(ordinal).__name__},
        )
    if ordinal < 0:
        raise LocalDifferenceExtractionError(
            "ordinal must be greater than or equal to 0",
            observation_id=normalized_observation_id,
            context={"ordinal": ordinal},
        )
    left_span = _ensure_span(left_span, field_name="left_span")
    right_span = _ensure_span(right_span, field_name="right_span")
    left_segment = _optional_id_part(left_segment_id, field_name="left_segment_id")
    right_segment = _optional_id_part(right_segment_id, field_name="right_segment_id")
    left_start = left_span.start if left_span is not None else "none"
    left_end = left_span.end if left_span is not None else "none"
    right_start = right_span.start if right_span is not None else "none"
    right_end = right_span.end if right_span is not None else "none"
    return (
        f"{_CANDIDATE_ID_PREFIX}:{normalized_observation_id}:{ordinal}:"
        f"{normalized_kind.value}:{left_segment}:{right_segment}:"
        f"{left_start}:{left_end}:{right_start}:{right_end}"
    )


def _build_candidate(
    *,
    observation_id: str,
    ordinal: int,
    kind: LocalDifferenceKind,
    left_segment_id: str | None,
    right_segment_id: str | None,
    left_span: TextSpan | None,
    right_span: TextSpan | None,
    surface_form: str | None,
    normalized_form: str | None,
    confidence: float,
    metadata: Mapping[str, object] | None = None,
) -> LocalDifferenceCandidate:
    normalized_observation_id = _ensure_non_empty_string(
        observation_id,
        field_name="observation_id",
    )
    normalized_kind = _ensure_candidate_kind(kind)
    normalized_left_span = _ensure_span(left_span, field_name="left_span")
    normalized_right_span = _ensure_span(right_span, field_name="right_span")
    normalized_confidence = _ensure_confidence(confidence)
    copied_metadata = _ensure_metadata(metadata)
    candidate_id = make_candidate_id(
        observation_id=normalized_observation_id,
        kind=normalized_kind,
        ordinal=ordinal,
        left_segment_id=left_segment_id,
        right_segment_id=right_segment_id,
        left_span=normalized_left_span,
        right_span=normalized_right_span,
    )
    if surface_form is not None and not isinstance(surface_form, str):
        raise LocalDifferenceExtractionError(
            "surface_form must be a string when provided",
            observation_id=normalized_observation_id,
            candidate_id=candidate_id,
            context={"type": type(surface_form).__name__},
        )
    if normalized_form is not None and not isinstance(normalized_form, str):
        raise LocalDifferenceExtractionError(
            "normalized_form must be a string when provided",
            observation_id=normalized_observation_id,
            candidate_id=candidate_id,
            context={"type": type(normalized_form).__name__},
        )
    return LocalDifferenceCandidate(
        candidate_id=candidate_id,
        observation_id=normalized_observation_id,
        kind=normalized_kind,
        left_segment_id=left_segment_id,
        right_segment_id=right_segment_id,
        left_span=normalized_left_span,
        right_span=normalized_right_span,
        surface_form=surface_form,
        normalized_form=normalized_form,
        confidence=normalized_confidence,
        metadata=copied_metadata,
    )


def _ensure_request(
    value: DifferenceExtractionRequest,
) -> DifferenceExtractionRequest:
    if not isinstance(value, DifferenceExtractionRequest):
        raise LocalDifferenceExtractionError(
            "request must be DifferenceExtractionRequest",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_segments(value: tuple[ParsedSegment, ...]) -> tuple[ParsedSegment, ...]:
    if not isinstance(value, tuple):
        raise LocalDifferenceExtractionError(
            "segments must be a tuple",
            context={"type": type(value).__name__},
        )
    for item in value:
        _ensure_segment(item)
    return value


def _ensure_segment(value: ParsedSegment) -> ParsedSegment:
    if not isinstance(value, ParsedSegment):
        raise LocalDifferenceExtractionError(
            "segment must be ParsedSegment",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_config(value: ParserConfig) -> ParserConfig:
    if not isinstance(value, ParserConfig):
        raise LocalDifferenceExtractionError(
            "config must be ParserConfig",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_strategy(
    value: DifferenceExtractionStrategy,
) -> DifferenceExtractionStrategy:
    if not isinstance(value, DifferenceExtractionStrategy):
        raise LocalDifferenceExtractionError(
            "strategy must be DifferenceExtractionStrategy",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_candidate_kind(value: LocalDifferenceKind) -> LocalDifferenceKind:
    if not isinstance(value, LocalDifferenceKind):
        raise LocalDifferenceExtractionError(
            "kind must be LocalDifferenceKind",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_span(value: TextSpan | None, *, field_name: str) -> TextSpan | None:
    if value is None:
        return None
    if not isinstance(value, TextSpan):
        raise LocalDifferenceExtractionError(
            f"{field_name} must be TextSpan",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_non_empty_string(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise LocalDifferenceExtractionError(
            f"{field_name} must be a string",
            context={"field_name": field_name, "type": type(value).__name__},
        )
    if not value:
        raise LocalDifferenceExtractionError(
            f"{field_name} must not be empty",
            context={"field_name": field_name},
        )
    if _contains_disallowed_control_chars(value):
        raise LocalDifferenceExtractionError(
            f"{field_name} must not contain control characters",
            context={"field_name": field_name},
        )
    return value


def _ensure_metadata(value: Mapping[str, object] | None) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise LocalDifferenceExtractionError(
            "metadata must be a mapping",
            context={"type": type(value).__name__},
        )
    copied: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise LocalDifferenceExtractionError(
                "metadata keys must be strings",
                context={"key_type": type(key).__name__},
            )
        if key in _FORBIDDEN_METADATA_KEYS:
            raise LocalDifferenceExtractionError(
                "metadata key crosses parser difference boundary",
                context={"key": key},
            )
        copied[key] = item
    return copied


def _ensure_confidence(value: float) -> float:
    if type(value) not in {float, int}:
        raise LocalDifferenceExtractionError(
            "confidence must be a number",
            context={"type": type(value).__name__},
        )
    numeric_value = float(value)
    if numeric_value != numeric_value or numeric_value in {
        float("inf"),
        float("-inf"),
    }:
        raise LocalDifferenceExtractionError("confidence must be finite")
    if not 0.0 <= numeric_value <= 1.0:
        raise LocalDifferenceExtractionError(
            "confidence must be between 0.0 and 1.0",
            context={"confidence": numeric_value},
        )
    return numeric_value


def _filtered_tokens(segment: ParsedSegment) -> tuple[TokenSpan, ...]:
    normalized_segment = _ensure_segment(segment)
    return tuple(
        token
        for token in normalized_segment.tokens
        if token.kind not in {TokenKind.WHITESPACE, TokenKind.CONTROL}
    )


def _token_compare_form(token: TokenSpan) -> str:
    if not isinstance(token, TokenSpan):
        raise LocalDifferenceExtractionError(
            "token must be TokenSpan",
            context={"type": type(token).__name__},
        )
    return token.normalized_text if token.normalized_text is not None else token.text


def _token_surface_form(token: TokenSpan) -> str:
    if not isinstance(token, TokenSpan):
        raise LocalDifferenceExtractionError(
            "token must be TokenSpan",
            context={"type": type(token).__name__},
        )
    return token.text


def _join_token_surfaces(tokens: tuple[TokenSpan, ...]) -> str:
    if not isinstance(tokens, tuple):
        raise LocalDifferenceExtractionError(
            "tokens must be a tuple",
            context={"type": type(tokens).__name__},
        )
    for token in tokens:
        if not isinstance(token, TokenSpan):
            raise LocalDifferenceExtractionError(
                "token item must be TokenSpan",
                context={"type": type(token).__name__},
            )
    return "".join(_token_surface_form(token) for token in tokens)


def _classify_aligned_difference(
    left: TokenSpan,
    right: TokenSpan,
) -> tuple[LocalDifferenceKind, float]:
    if not isinstance(left, TokenSpan) or not isinstance(right, TokenSpan):
        raise LocalDifferenceExtractionError("aligned values must be TokenSpan")
    if left.kind == TokenKind.NUMBER and right.kind == TokenKind.NUMBER:
        return LocalDifferenceKind.NUMERIC_CHANGE, 1.0
    if left.text.casefold() == right.text.casefold() and left.text != right.text:
        return LocalDifferenceKind.CASE_CHANGE, 1.0
    if left.kind == TokenKind.PUNCTUATION and right.kind == TokenKind.PUNCTUATION:
        return LocalDifferenceKind.PUNCTUATION_CHANGE, 1.0
    return LocalDifferenceKind.TOKEN_MISMATCH, 0.75


def _enforce_candidate_limit(
    count: int,
    config: ParserConfig,
    *,
    observation_id: str,
) -> None:
    if type(count) is not int:
        raise LocalDifferenceExtractionError(
            "candidate count must be an int",
            observation_id=observation_id,
            context={"type": type(count).__name__},
        )
    if count > config.max_difference_candidates:
        raise ParserLimitExceededError(
            "parser difference candidate limit exceeded",
            observation_id=observation_id,
            context={
                "count": count,
                "max_difference_candidates": config.max_difference_candidates,
            },
        )


def _contains_disallowed_control_chars(value: str) -> bool:
    return any(ord(character) < 32 for character in value)


def _sanitize_id_part(value: str) -> str:
    sanitized = []
    for character in value:
        if character.isspace() or ord(character) < 32:
            sanitized.append("_")
        else:
            sanitized.append(character)
    return "".join(sanitized)


def _neighbor_pairs(
    segments: tuple[ParsedSegment, ...],
) -> Iterable[tuple[ParsedSegment, ParsedSegment]]:
    for index in range(len(segments) - 1):
        yield segments[index], segments[index + 1]


def _optional_id_part(value: str | None, *, field_name: str) -> str:
    if value is None:
        return "none"
    return _sanitize_id_part(_ensure_non_empty_string(value, field_name=field_name))


__all__ = [
    "DifferenceExtractionStrategy",
    "DifferenceExtractionRequest",
    "extract_local_differences",
    "extract_segment_repetition_candidates",
    "compare_neighboring_segments",
    "make_candidate_id",
]
