"""CEBA Parser Layer tokenization utilities.

This module converts ParsedSegment text into TokenSpan objects. It does not
infer meaning, compute delta, or update memory.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
import unicodedata

from ois.parser.errors import ParserLimitExceededError, TokenizationError
from ois.parser.schema import (
    ParsedSegment,
    ParserConfig,
    TextSpan,
    TokenKind,
    TokenSpan,
)


_MAX_TOKEN_TEXT_LENGTH = 1_000_000
_TOKEN_ID_PREFIX = "tok"
_WORD_APOSTROPHES = frozenset({"'", "’", "ʼ"})
_NUMBER_SEPARATORS = frozenset({".", ",", "_", ":"})


class TokenizationStrategy(str, Enum):
    SURFACE = "surface"


@dataclass(frozen=True, slots=True)
class TokenizationRequest:
    segments: tuple[ParsedSegment, ...]
    config: ParserConfig
    strategy: TokenizationStrategy = TokenizationStrategy.SURFACE

    def __post_init__(self) -> None:
        _ensure_segments(self.segments)
        _ensure_config(self.config)
        _ensure_strategy(self.strategy)


def tokenize_segments(request: TokenizationRequest) -> tuple[ParsedSegment, ...]:
    if not isinstance(request, TokenizationRequest):
        raise TokenizationError(
            "request must be TokenizationRequest",
            context={"type": type(request).__name__},
        )
    segments = _ensure_segments(request.segments)
    config = _ensure_config(request.config)
    strategy = _ensure_strategy(request.strategy)
    return tuple(
        tokenize_segment(segment=segment, config=config, strategy=strategy)
        for segment in segments
    )


def tokenize_segment(
    *,
    segment: ParsedSegment,
    config: ParserConfig,
    strategy: TokenizationStrategy = TokenizationStrategy.SURFACE,
) -> ParsedSegment:
    normalized_segment = _ensure_segment(segment)
    normalized_config = _ensure_config(config)
    _ensure_strategy(strategy)

    if normalized_segment.text is None:
        return _copy_segment_with_tokens(normalized_segment, ())

    tokens = tokenize_text(
        observation_id=normalized_segment.observation_id,
        segment_id=normalized_segment.segment_id,
        text=normalized_segment.text,
        config=normalized_config,
    )
    return _copy_segment_with_tokens(normalized_segment, tokens)


def tokenize_text(
    *,
    observation_id: str,
    segment_id: str,
    text: str,
    config: ParserConfig,
) -> tuple[TokenSpan, ...]:
    normalized_observation_id = _ensure_non_empty_string(
        observation_id,
        field_name="observation_id",
    )
    normalized_segment_id = _ensure_non_empty_string(
        segment_id,
        field_name="segment_id",
    )
    normalized_config = _ensure_config(config)
    if not isinstance(text, str):
        raise TokenizationError(
            "text must be a string",
            observation_id=normalized_observation_id,
            segment_id=normalized_segment_id,
            context={"type": type(text).__name__},
        )
    if len(text) > _MAX_TOKEN_TEXT_LENGTH:
        raise TokenizationError(
            "text exceeds maximum tokenization length",
            observation_id=normalized_observation_id,
            segment_id=normalized_segment_id,
            context={"length": len(text), "max_length": _MAX_TOKEN_TEXT_LENGTH},
        )

    tokens: list[TokenSpan] = []
    for kind, start, end in _scan_token_specs(text):
        if not _should_emit_token(kind, normalized_config):
            continue
        token_text = text[start:end]
        span = TextSpan(start, end)
        ordinal = len(tokens)
        token_id = make_token_id(
            segment_id=normalized_segment_id,
            ordinal=ordinal,
            kind=kind,
            span=span,
        )
        tokens.append(
            TokenSpan(
                token_id=token_id,
                segment_id=normalized_segment_id,
                kind=kind,
                text=token_text,
                normalized_text=normalize_token_text(text=token_text, kind=kind),
                span=span,
                ordinal=ordinal,
            )
        )
        _enforce_token_limit(
            len(tokens),
            normalized_config,
            observation_id=normalized_observation_id,
            segment_id=normalized_segment_id,
        )
    return tuple(tokens)


def classify_character(character: str) -> TokenKind:
    if not isinstance(character, str):
        raise TokenizationError(
            "character must be a string",
            context={"type": type(character).__name__},
        )
    if len(character) != 1:
        raise TokenizationError(
            "character must contain exactly one code point",
            context={"length": len(character)},
        )

    if character.isspace():
        return TokenKind.WHITESPACE

    category = unicodedata.category(character)
    if category.startswith("C"):
        return TokenKind.CONTROL
    if category.startswith("N"):
        return TokenKind.NUMBER
    if category.startswith(("L", "M")) or character == "_":
        return TokenKind.WORD
    if category.startswith("P"):
        return TokenKind.PUNCTUATION
    if category.startswith("S"):
        return TokenKind.SYMBOL
    return TokenKind.UNKNOWN


def normalize_token_text(
    *,
    text: str,
    kind: TokenKind,
) -> str | None:
    if not isinstance(text, str):
        raise TokenizationError(
            "text must be a string",
            context={"type": type(text).__name__},
        )
    normalized_kind = _ensure_token_kind(kind)
    if normalized_kind in {TokenKind.WHITESPACE, TokenKind.CONTROL}:
        return None
    if normalized_kind == TokenKind.WORD:
        return text.casefold()
    return text


def make_token_id(
    *,
    segment_id: str,
    ordinal: int,
    kind: TokenKind,
    span: TextSpan,
) -> str:
    normalized_segment_id = _ensure_non_empty_string(
        segment_id,
        field_name="segment_id",
    )
    if type(ordinal) is not int:
        raise TokenizationError(
            "ordinal must be an int",
            segment_id=normalized_segment_id,
            context={"type": type(ordinal).__name__},
        )
    if ordinal < 0:
        raise TokenizationError(
            "ordinal must be greater than or equal to 0",
            segment_id=normalized_segment_id,
            context={"ordinal": ordinal},
        )
    normalized_kind = _ensure_token_kind(kind)
    normalized_span = _ensure_span(span)
    sanitized_segment_id = _sanitize_id_part(normalized_segment_id)
    return (
        f"{_TOKEN_ID_PREFIX}:{sanitized_segment_id}:{ordinal}:"
        f"{normalized_kind.value}:{normalized_span.start}:{normalized_span.end}"
    )


def _copy_segment_with_tokens(
    segment: ParsedSegment,
    tokens: tuple[TokenSpan, ...],
) -> ParsedSegment:
    normalized_segment = _ensure_segment(segment)
    normalized_tokens = _ensure_tokens(tokens)
    metadata: Mapping[str, object] = normalized_segment.metadata
    return ParsedSegment(
        segment_id=normalized_segment.segment_id,
        observation_id=normalized_segment.observation_id,
        kind=normalized_segment.kind,
        ordinal=normalized_segment.ordinal,
        text=normalized_segment.text,
        span=normalized_segment.span,
        parent_segment_id=normalized_segment.parent_segment_id,
        path=normalized_segment.path,
        tokens=normalized_tokens,
        metadata=dict(metadata),
    )


def _ensure_segments(value: tuple[ParsedSegment, ...]) -> tuple[ParsedSegment, ...]:
    if not isinstance(value, tuple):
        raise TokenizationError(
            "segments must be a tuple",
            context={"type": type(value).__name__},
        )
    for item in value:
        _ensure_segment(item)
    return value


def _ensure_segment(value: ParsedSegment) -> ParsedSegment:
    if not isinstance(value, ParsedSegment):
        raise TokenizationError(
            "segment must be ParsedSegment",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_config(value: ParserConfig) -> ParserConfig:
    if not isinstance(value, ParserConfig):
        raise TokenizationError(
            "config must be ParserConfig",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_strategy(value: TokenizationStrategy) -> TokenizationStrategy:
    if not isinstance(value, TokenizationStrategy):
        raise TokenizationError(
            "strategy must be TokenizationStrategy",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_non_empty_string(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TokenizationError(
            f"{field_name} must be a string",
            context={"field_name": field_name, "type": type(value).__name__},
        )
    if not value:
        raise TokenizationError(
            f"{field_name} must not be empty",
            context={"field_name": field_name},
        )
    if _contains_disallowed_control_chars(value):
        raise TokenizationError(
            f"{field_name} must not contain control characters",
            context={"field_name": field_name},
        )
    return value


def _ensure_token_kind(value: TokenKind) -> TokenKind:
    if not isinstance(value, TokenKind):
        raise TokenizationError(
            "kind must be TokenKind",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_span(value: TextSpan) -> TextSpan:
    if not isinstance(value, TextSpan):
        raise TokenizationError(
            "span must be TextSpan",
            context={"type": type(value).__name__},
        )
    return value


def _enforce_token_limit(
    count: int,
    config: ParserConfig,
    *,
    observation_id: str,
    segment_id: str,
) -> None:
    if type(count) is not int:
        raise TokenizationError(
            "token count must be an int",
            observation_id=observation_id,
            segment_id=segment_id,
            context={"type": type(count).__name__},
        )
    if count > config.max_tokens_per_segment:
        raise ParserLimitExceededError(
            "parser token limit exceeded",
            observation_id=observation_id,
            segment_id=segment_id,
            context={
                "count": count,
                "max_tokens_per_segment": config.max_tokens_per_segment,
            },
        )


def _should_emit_token(kind: TokenKind, config: ParserConfig) -> bool:
    normalized_kind = _ensure_token_kind(kind)
    normalized_config = _ensure_config(config)
    if normalized_kind == TokenKind.WHITESPACE:
        return normalized_config.emit_whitespace_tokens
    if normalized_kind == TokenKind.CONTROL:
        return normalized_config.emit_control_tokens
    return True


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


def _scan_token_specs(text: str) -> Iterable[tuple[TokenKind, int, int]]:
    cursor = 0
    length = len(text)
    while cursor < length:
        kind = classify_character(text[cursor])
        if kind == TokenKind.WORD:
            end = _scan_word(text, cursor)
        elif kind == TokenKind.NUMBER:
            end = _scan_number(text, cursor)
        else:
            end = _scan_same_kind(text, cursor, kind)
        yield kind, cursor, end
        cursor = end


def _scan_word(text: str, start: int) -> int:
    cursor = start + 1
    while cursor < len(text):
        character = text[cursor]
        kind = classify_character(character)
        if kind == TokenKind.WORD:
            cursor += 1
            continue
        if (
            character in _WORD_APOSTROPHES
            and cursor > start
            and cursor + 1 < len(text)
            and classify_character(text[cursor - 1]) == TokenKind.WORD
            and classify_character(text[cursor + 1]) == TokenKind.WORD
        ):
            cursor += 1
            continue
        break
    return cursor


def _scan_number(text: str, start: int) -> int:
    cursor = start + 1
    while cursor < len(text):
        character = text[cursor]
        kind = classify_character(character)
        if kind == TokenKind.NUMBER:
            cursor += 1
            continue
        if (
            character in _NUMBER_SEPARATORS
            and cursor > start
            and cursor + 1 < len(text)
            and classify_character(text[cursor - 1]) == TokenKind.NUMBER
            and classify_character(text[cursor + 1]) == TokenKind.NUMBER
        ):
            cursor += 1
            continue
        break
    return cursor


def _scan_same_kind(text: str, start: int, kind: TokenKind) -> int:
    cursor = start + 1
    while cursor < len(text) and classify_character(text[cursor]) == kind:
        cursor += 1
    return cursor


def _ensure_tokens(value: tuple[TokenSpan, ...]) -> tuple[TokenSpan, ...]:
    if not isinstance(value, tuple):
        raise TokenizationError(
            "tokens must be a tuple",
            context={"type": type(value).__name__},
        )
    for item in value:
        if not isinstance(item, TokenSpan):
            raise TokenizationError(
                "token item must be TokenSpan",
                context={"type": type(item).__name__},
            )
    return value


__all__ = [
    "TokenizationStrategy",
    "TokenizationRequest",
    "tokenize_segments",
    "tokenize_segment",
    "tokenize_text",
    "classify_character",
    "normalize_token_text",
    "make_token_id",
]
