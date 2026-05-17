"""CEBA Parser Layer segmentation utilities.

This module converts CanonicalObservation surface content into ParsedSegment
objects. It does not tokenize content, infer meaning, compute delta, or update
memory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from ois.input.schema import CanonicalObservation
from ois.parser.errors import (
    EmptyObservationError,
    ParserLimitExceededError,
    SegmentationError,
)
from ois.parser.schema import ParsedSegment, ParserConfig, SegmentKind, TextSpan


_DEFAULT_ROOT_PATH = ("root",)

_MAX_TEXT_LENGTH = 5_000_000
_MAX_STRUCTURED_DEPTH = 16
_MAX_STRUCTURED_ITEMS = 50_000
_MAX_SEGMENT_TEXT_LENGTH = 1_000_000

_SENTENCE_TERMINATORS = frozenset({".", "!", "?", "؟", "。", "！", "？"})
_CODE_FENCE_PREFIXES = ("```", "~~~")


class SegmentationStrategy(str, Enum):
    AUTO = "auto"
    TEXT_LINES = "text_lines"
    TEXT_PARAGRAPHS = "text_paragraphs"
    STRUCTURED_FIELDS = "structured_fields"
    SINGLE_SEGMENT = "single_segment"


@dataclass(frozen=True, slots=True)
class SegmentRequest:
    observation: CanonicalObservation
    config: ParserConfig
    strategy: SegmentationStrategy = SegmentationStrategy.AUTO

    def __post_init__(self) -> None:
        _ensure_observation(self.observation)
        _ensure_parser_config(self.config)
        _ensure_strategy(self.strategy)


def segment_observation(request: SegmentRequest) -> tuple[ParsedSegment, ...]:
    if not isinstance(request, SegmentRequest):
        raise SegmentationError(
            "request must be SegmentRequest",
            context={"type": type(request).__name__},
        )

    observation = _ensure_observation(request.observation)
    config = _ensure_parser_config(request.config)
    strategy = _ensure_strategy(request.strategy)
    observation_id = str(observation.observation_id)
    surface = extract_parseable_surface(observation)

    if strategy == SegmentationStrategy.AUTO:
        if isinstance(surface, str):
            segments = segment_text(
                observation_id=observation_id,
                text=surface,
                config=config,
                strategy=strategy,
                metadata={"source": "canonical_content"},
            )
        elif isinstance(surface, Mapping) or _is_sequence_but_not_text(surface):
            segments = segment_structured(
                observation_id=observation_id,
                value=surface,
                config=config,
                metadata={"source": "canonical_content"},
            )
        elif _is_json_safe_scalar(surface):
            segment = create_single_segment(
                observation_id=observation_id,
                value=surface,
                kind=SegmentKind.FIELD,
                metadata={
                    "source": "canonical_content",
                    "strategy": SegmentationStrategy.SINGLE_SEGMENT.value,
                },
            )
            segments = (segment,)
        else:
            raise SegmentationError(
                "canonical content contains unsupported parseable surface",
                observation_id=observation_id,
                context={"type": type(surface).__name__},
            )
    elif strategy in {
        SegmentationStrategy.TEXT_LINES,
        SegmentationStrategy.TEXT_PARAGRAPHS,
    }:
        if not isinstance(surface, str):
            raise SegmentationError(
                "text segmentation requires string surface content",
                observation_id=observation_id,
                context={"type": type(surface).__name__},
            )
        segments = segment_text(
            observation_id=observation_id,
            text=surface,
            config=config,
            strategy=strategy,
            metadata={"source": "canonical_content"},
        )
    elif strategy == SegmentationStrategy.STRUCTURED_FIELDS:
        segments = segment_structured(
            observation_id=observation_id,
            value=surface,
            config=config,
            metadata={"source": "canonical_content"},
        )
    elif strategy == SegmentationStrategy.SINGLE_SEGMENT:
        segment = create_single_segment(
            observation_id=observation_id,
            value=surface,
            kind=_single_segment_kind(surface),
            metadata={
                "source": "canonical_content",
                "strategy": strategy.value,
            },
        )
        segments = (segment,)
    else:
        raise SegmentationError(
            "unsupported segmentation strategy",
            observation_id=observation_id,
            context={"strategy": strategy.value},
        )

    if not segments:
        raise EmptyObservationError(
            "canonical observation contains no parseable segments",
            observation_id=observation_id,
        )
    _enforce_segment_limit(len(segments), config, observation_id=observation_id)
    return tuple(segments)


def extract_parseable_surface(observation: CanonicalObservation) -> object:
    normalized_observation = _ensure_observation(observation)
    observation_id = str(normalized_observation.observation_id)
    content = normalized_observation.content

    if content.text is not None:
        if content.text == "":
            raise EmptyObservationError(
                "canonical observation text content is empty",
                observation_id=observation_id,
            )
        return content.text

    if content.data is not None:
        if not content.data:
            raise EmptyObservationError(
                "canonical observation structured content is empty",
                observation_id=observation_id,
            )
        return content.data

    raise EmptyObservationError(
        "canonical observation has no inline parseable surface content",
        observation_id=observation_id,
        context={"content_kind": content.kind.value},
    )


def segment_text(
    *,
    observation_id: str,
    text: str,
    config: ParserConfig,
    strategy: SegmentationStrategy = SegmentationStrategy.AUTO,
    metadata: Mapping[str, object] | None = None,
) -> tuple[ParsedSegment, ...]:
    normalized_observation_id = _ensure_non_empty_string(
        observation_id,
        field_name="observation_id",
    )
    normalized_config = _ensure_parser_config(config)
    normalized_strategy = _ensure_strategy(strategy)
    base_metadata = _ensure_metadata(metadata)

    if not isinstance(text, str):
        raise SegmentationError(
            "text must be a string",
            observation_id=normalized_observation_id,
            context={"type": type(text).__name__},
        )
    if len(text) > _MAX_TEXT_LENGTH:
        raise SegmentationError(
            "text exceeds maximum segmentation length",
            observation_id=normalized_observation_id,
            context={"length": len(text), "max_length": _MAX_TEXT_LENGTH},
        )
    if text == "":
        raise EmptyObservationError(
            "text contains no parseable surface content",
            observation_id=normalized_observation_id,
        )

    if normalized_strategy == SegmentationStrategy.TEXT_LINES:
        segments = _line_segments(
            observation_id=normalized_observation_id,
            text=text,
            base_metadata=base_metadata,
            initial_ordinal=0,
        )
    elif normalized_strategy == SegmentationStrategy.TEXT_PARAGRAPHS:
        segments = _paragraph_segments(
            observation_id=normalized_observation_id,
            text=text,
            base_metadata=base_metadata,
            initial_ordinal=0,
        )
    elif normalized_strategy == SegmentationStrategy.SINGLE_SEGMENT:
        segments = (
            create_single_segment(
                observation_id=normalized_observation_id,
                value=text,
                kind=SegmentKind.TEXT,
                metadata={**base_metadata, "strategy": normalized_strategy.value},
            ),
        )
    elif normalized_strategy == SegmentationStrategy.AUTO:
        if _has_code_fence(text):
            segments = _code_aware_text_segments(
                observation_id=normalized_observation_id,
                text=text,
                base_metadata=base_metadata,
            )
        elif _has_multiple_non_empty_paragraphs(text):
            segments = _paragraph_segments(
                observation_id=normalized_observation_id,
                text=text,
                base_metadata=base_metadata,
                initial_ordinal=0,
            )
        elif "\n" in text or "\r" in text:
            segments = _line_segments(
                observation_id=normalized_observation_id,
                text=text,
                base_metadata=base_metadata,
                initial_ordinal=0,
            )
        else:
            segments = (
                create_single_segment(
                    observation_id=normalized_observation_id,
                    value=text,
                    kind=SegmentKind.TEXT,
                    metadata={**base_metadata, "strategy": normalized_strategy.value},
                ),
            )
    else:
        raise SegmentationError(
            "text segmentation received incompatible strategy",
            observation_id=normalized_observation_id,
            context={"strategy": normalized_strategy.value},
        )

    if not segments:
        raise EmptyObservationError(
            "text contains no parseable segments",
            observation_id=normalized_observation_id,
        )
    _enforce_segment_limit(
        len(segments),
        normalized_config,
        observation_id=normalized_observation_id,
    )
    return tuple(segments)


def segment_structured(
    *,
    observation_id: str,
    value: object,
    config: ParserConfig,
    path: tuple[str, ...] = _DEFAULT_ROOT_PATH,
    metadata: Mapping[str, object] | None = None,
) -> tuple[ParsedSegment, ...]:
    normalized_observation_id = _ensure_non_empty_string(
        observation_id,
        field_name="observation_id",
    )
    normalized_config = _ensure_parser_config(config)
    normalized_path = _ensure_path(path)
    base_metadata = _ensure_metadata(metadata)
    segments: list[ParsedSegment] = []

    try:
        _walk_structured_value(
            observation_id=normalized_observation_id,
            value=value,
            config=normalized_config,
            path=normalized_path,
            metadata=base_metadata,
            segments=segments,
            depth=0,
            item_counter={"count": 0},
        )
    except ParserLimitExceededError:
        raise
    except SegmentationError:
        raise
    except Exception as error:
        raise SegmentationError(
            "structured segmentation failed",
            observation_id=normalized_observation_id,
            context={"error_type": error.__class__.__name__},
        ) from error

    if not segments:
        raise EmptyObservationError(
            "structured content contains no parseable segments",
            observation_id=normalized_observation_id,
        )
    _enforce_segment_limit(
        len(segments),
        normalized_config,
        observation_id=normalized_observation_id,
    )
    return tuple(segments)


def create_single_segment(
    *,
    observation_id: str,
    value: object,
    kind: SegmentKind,
    ordinal: int = 0,
    path: tuple[str, ...] = _DEFAULT_ROOT_PATH,
    metadata: Mapping[str, object] | None = None,
) -> ParsedSegment:
    normalized_observation_id = _ensure_non_empty_string(
        observation_id,
        field_name="observation_id",
    )
    if not isinstance(kind, SegmentKind):
        raise SegmentationError(
            "kind must be SegmentKind",
            observation_id=normalized_observation_id,
            context={"type": type(kind).__name__},
        )
    if type(ordinal) is not int:
        raise SegmentationError(
            "ordinal must be an int",
            observation_id=normalized_observation_id,
            context={"type": type(ordinal).__name__},
        )
    if ordinal < 0:
        raise SegmentationError(
            "ordinal must be greater than or equal to 0",
            observation_id=normalized_observation_id,
            context={"ordinal": ordinal},
        )

    normalized_path = _ensure_path(path)
    segment_metadata = _ensure_metadata(metadata)
    text = _scalar_to_text(value)
    if text is not None and len(text) > _MAX_SEGMENT_TEXT_LENGTH:
        raise SegmentationError(
            "segment text exceeds maximum length",
            observation_id=normalized_observation_id,
            context={
                "length": len(text),
                "max_length": _MAX_SEGMENT_TEXT_LENGTH,
            },
        )

    if isinstance(value, Mapping):
        _validate_mapping_keys(value, observation_id=normalized_observation_id)
        segment_metadata = {
            **segment_metadata,
            "structural_type": "mapping",
            "item_count": len(value),
        }
    elif _is_sequence_but_not_text(value):
        segment_metadata = {
            **segment_metadata,
            "structural_type": "sequence",
            "item_count": len(value),
        }
    elif not _is_json_safe_scalar(value):
        raise SegmentationError(
            "unsupported value for single segment",
            observation_id=normalized_observation_id,
            context={"type": type(value).__name__},
        )

    span = TextSpan(0, len(text)) if text is not None else None
    segment_id = _make_segment_id(
        observation_id=normalized_observation_id,
        kind=kind,
        ordinal=ordinal,
        path=normalized_path,
        span=span,
    )
    return ParsedSegment(
        segment_id=segment_id,
        observation_id=normalized_observation_id,
        kind=kind,
        ordinal=ordinal,
        text=text,
        span=span,
        parent_segment_id=None,
        path=normalized_path,
        tokens=(),
        metadata=segment_metadata,
    )


def _make_segment_id(
    *,
    observation_id: str,
    kind: SegmentKind,
    ordinal: int,
    path: tuple[str, ...],
    span: TextSpan | None = None,
) -> str:
    normalized_observation_id = _sanitize_id_part(
        _ensure_non_empty_string(observation_id, field_name="observation_id")
    )
    normalized_path = _ensure_path(path)
    if not isinstance(kind, SegmentKind):
        raise SegmentationError(
            "kind must be SegmentKind",
            observation_id=normalized_observation_id,
            context={"type": type(kind).__name__},
        )
    if type(ordinal) is not int or ordinal < 0:
        raise SegmentationError(
            "ordinal must be a non-negative int",
            observation_id=normalized_observation_id,
            context={"ordinal": ordinal},
        )
    path_joined = ".".join(_sanitize_id_part(part) for part in normalized_path)
    start = span.start if span is not None else "none"
    end = span.end if span is not None else "none"
    return (
        f"seg:{normalized_observation_id}:{ordinal}:{kind.value}:"
        f"{path_joined}:{start}:{end}"
    )


def _ensure_non_empty_string(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise SegmentationError(
            f"{field_name} must be a string",
            context={"field_name": field_name, "type": type(value).__name__},
        )
    if not value:
        raise SegmentationError(
            f"{field_name} must not be empty",
            context={"field_name": field_name},
        )
    if _contains_disallowed_control_chars(value):
        raise SegmentationError(
            f"{field_name} must not contain control characters",
            context={"field_name": field_name},
        )
    return value


def _ensure_parser_config(value: ParserConfig) -> ParserConfig:
    if not isinstance(value, ParserConfig):
        raise SegmentationError(
            "config must be ParserConfig",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_observation(value: CanonicalObservation) -> CanonicalObservation:
    if not isinstance(value, CanonicalObservation):
        raise SegmentationError(
            "observation must be CanonicalObservation",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_strategy(value: SegmentationStrategy) -> SegmentationStrategy:
    if not isinstance(value, SegmentationStrategy):
        raise SegmentationError(
            "strategy must be SegmentationStrategy",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_metadata(value: Mapping[str, object] | None) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SegmentationError(
            "metadata must be a mapping",
            context={"type": type(value).__name__},
        )
    copied: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise SegmentationError(
                "metadata keys must be strings",
                context={"key_type": type(key).__name__},
            )
        copied[key] = item
    return copied


def _ensure_path(value: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise SegmentationError(
            "path must be a tuple",
            context={"type": type(value).__name__},
        )
    if not value:
        raise SegmentationError("path must not be empty")
    for part in value:
        _ensure_non_empty_string(part, field_name="path part")
    return value


def _enforce_segment_limit(
    count: int,
    config: ParserConfig,
    *,
    observation_id: str,
) -> None:
    if type(count) is not int:
        raise SegmentationError(
            "segment count must be an int",
            observation_id=observation_id,
            context={"type": type(count).__name__},
        )
    if count > config.max_segments:
        raise ParserLimitExceededError(
            "parser segment limit exceeded",
            observation_id=observation_id,
            context={"count": count, "max_segments": config.max_segments},
        )


def _is_sequence_but_not_text(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    )


def _is_json_safe_scalar(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return _is_finite_float(value)
    return False


def _scalar_to_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value)
    if type(value) is int:
        return str(value)
    if isinstance(value, float):
        if not _is_finite_float(value):
            raise SegmentationError("float scalar must be finite")
        return str(value)
    if isinstance(value, Mapping) or _is_sequence_but_not_text(value):
        return None
    raise SegmentationError(
        "unsupported scalar value",
        context={"type": type(value).__name__},
    )


def _contains_disallowed_control_chars(value: str) -> bool:
    return any(ord(character) < 32 for character in value)


def _line_segments(
    *,
    observation_id: str,
    text: str,
    base_metadata: Mapping[str, object],
    initial_ordinal: int,
) -> tuple[ParsedSegment, ...]:
    segments: list[ParsedSegment] = []
    cursor = 0
    line_number = 0
    for line in text.splitlines(keepends=True):
        start = cursor
        end = cursor + len(line)
        cursor = end
        if line.strip() == "":
            line_number += 1
            continue
        span = TextSpan(start, end)
        ordinal = initial_ordinal + len(segments)
        path = (*_DEFAULT_ROOT_PATH, f"line:{line_number}")
        metadata = {
            **base_metadata,
            "strategy": SegmentationStrategy.TEXT_LINES.value,
            "line_number": line_number,
            "path": ".".join(path),
        }
        segments.append(
            _build_segment(
                observation_id=observation_id,
                kind=SegmentKind.LINE,
                ordinal=ordinal,
                text=line,
                span=span,
                path=path,
                metadata=metadata,
            )
        )
        line_number += 1

    if cursor < len(text):
        line = text[cursor:]
        if line.strip() != "":
            span = TextSpan(cursor, len(text))
            ordinal = initial_ordinal + len(segments)
            path = (*_DEFAULT_ROOT_PATH, f"line:{line_number}")
            metadata = {
                **base_metadata,
                "strategy": SegmentationStrategy.TEXT_LINES.value,
                "line_number": line_number,
                "path": ".".join(path),
            }
            segments.append(
                _build_segment(
                    observation_id=observation_id,
                    kind=SegmentKind.LINE,
                    ordinal=ordinal,
                    text=line,
                    span=span,
                    path=path,
                    metadata=metadata,
                )
            )
    return tuple(segments)


def _paragraph_segments(
    *,
    observation_id: str,
    text: str,
    base_metadata: Mapping[str, object],
    initial_ordinal: int,
) -> tuple[ParsedSegment, ...]:
    segments: list[ParsedSegment] = []
    paragraph_start: int | None = None
    cursor = 0

    for line in text.splitlines(keepends=True):
        line_start = cursor
        line_end = cursor + len(line)
        cursor = line_end
        if line.strip() == "":
            if paragraph_start is not None:
                _append_paragraph_segment(
                    segments=segments,
                    observation_id=observation_id,
                    text=text,
                    start=paragraph_start,
                    end=line_start,
                    base_metadata=base_metadata,
                    initial_ordinal=initial_ordinal,
                )
                paragraph_start = None
        elif paragraph_start is None:
            paragraph_start = line_start

    if paragraph_start is not None:
        _append_paragraph_segment(
            segments=segments,
            observation_id=observation_id,
            text=text,
            start=paragraph_start,
            end=len(text),
            base_metadata=base_metadata,
            initial_ordinal=initial_ordinal,
        )
    return tuple(segments)


def _append_paragraph_segment(
    *,
    segments: list[ParsedSegment],
    observation_id: str,
    text: str,
    start: int,
    end: int,
    base_metadata: Mapping[str, object],
    initial_ordinal: int,
) -> None:
    paragraph_text = text[start:end]
    if paragraph_text.strip() == "":
        return
    span = TextSpan(start, end)
    paragraph_index = len(segments)
    ordinal = initial_ordinal + paragraph_index
    path = (*_DEFAULT_ROOT_PATH, f"paragraph:{paragraph_index}")
    metadata = {
        **base_metadata,
        "strategy": SegmentationStrategy.TEXT_PARAGRAPHS.value,
        "paragraph_index": paragraph_index,
        "path": ".".join(path),
    }
    segments.append(
        _build_segment(
            observation_id=observation_id,
            kind=SegmentKind.PARAGRAPH,
            ordinal=ordinal,
            text=paragraph_text,
            span=span,
            path=path,
            metadata=metadata,
        )
    )


def _code_aware_text_segments(
    *,
    observation_id: str,
    text: str,
    base_metadata: Mapping[str, object],
) -> tuple[ParsedSegment, ...]:
    segments: list[ParsedSegment] = []
    lines = text.splitlines(keepends=True)
    cursor = 0
    block_start: int | None = None
    block_prefix: str | None = None
    text_start: int | None = None
    code_block_index = 0

    for line in lines:
        line_start = cursor
        line_end = cursor + len(line)
        stripped = line.lstrip()
        fence_prefix = _line_fence_prefix(stripped)

        if block_start is None and fence_prefix is not None:
            if text_start is not None and text_start < line_start:
                _append_surrounding_text_segments(
                    segments=segments,
                    observation_id=observation_id,
                    text=text[text_start:line_start],
                    absolute_start=text_start,
                    base_metadata=base_metadata,
                )
                text_start = None
            block_start = line_start
            block_prefix = fence_prefix
        elif block_start is not None and fence_prefix == block_prefix:
            _append_code_segment(
                segments=segments,
                observation_id=observation_id,
                text=text,
                start=block_start,
                end=line_end,
                base_metadata=base_metadata,
                code_block_index=code_block_index,
            )
            code_block_index += 1
            block_start = None
            block_prefix = None
        elif block_start is None and text_start is None and line.strip() != "":
            text_start = line_start

        cursor = line_end

    if block_start is not None:
        _append_code_segment(
            segments=segments,
            observation_id=observation_id,
            text=text,
            start=block_start,
            end=len(text),
            base_metadata=base_metadata,
            code_block_index=code_block_index,
        )
    elif text_start is not None and text_start < len(text):
        _append_surrounding_text_segments(
            segments=segments,
            observation_id=observation_id,
            text=text[text_start:],
            absolute_start=text_start,
            base_metadata=base_metadata,
        )

    return tuple(
        _rebuild_segment_with_ordinal(segment, ordinal)
        for ordinal, segment in enumerate(segments)
    )


def _append_surrounding_text_segments(
    *,
    segments: list[ParsedSegment],
    observation_id: str,
    text: str,
    absolute_start: int,
    base_metadata: Mapping[str, object],
) -> None:
    if text.strip() == "":
        return
    if _has_multiple_non_empty_paragraphs(text):
        local_segments = _paragraph_segments(
            observation_id=observation_id,
            text=text,
            base_metadata=base_metadata,
            initial_ordinal=len(segments),
        )
    else:
        local_segments = _line_segments(
            observation_id=observation_id,
            text=text,
            base_metadata=base_metadata,
            initial_ordinal=len(segments),
        )
    for segment in local_segments:
        if segment.span is None:
            segments.append(segment)
            continue
        adjusted_span = TextSpan(
            absolute_start + segment.span.start,
            absolute_start + segment.span.end,
        )
        segments.append(
            _build_segment(
                observation_id=observation_id,
                kind=segment.kind,
                ordinal=segment.ordinal,
                text=segment.text,
                span=adjusted_span,
                path=segment.path,
                metadata=dict(segment.metadata),
            )
        )


def _append_code_segment(
    *,
    segments: list[ParsedSegment],
    observation_id: str,
    text: str,
    start: int,
    end: int,
    base_metadata: Mapping[str, object],
    code_block_index: int,
) -> None:
    code_text = text[start:end]
    span = TextSpan(start, end)
    ordinal = len(segments)
    path = (*_DEFAULT_ROOT_PATH, f"code_block:{code_block_index}")
    metadata = {
        **base_metadata,
        "strategy": SegmentationStrategy.AUTO.value,
        "code_block_index": code_block_index,
        "path": ".".join(path),
    }
    segments.append(
        _build_segment(
            observation_id=observation_id,
            kind=SegmentKind.CODE_BLOCK,
            ordinal=ordinal,
            text=code_text,
            span=span,
            path=path,
            metadata=metadata,
        )
    )


def _walk_structured_value(
    *,
    observation_id: str,
    value: object,
    config: ParserConfig,
    path: tuple[str, ...],
    metadata: Mapping[str, object],
    segments: list[ParsedSegment],
    depth: int,
    item_counter: dict[str, int],
) -> None:
    if depth > _MAX_STRUCTURED_DEPTH:
        raise SegmentationError(
            "structured content exceeds maximum depth",
            observation_id=observation_id,
            context={"depth": depth, "max_depth": _MAX_STRUCTURED_DEPTH},
        )
    _increment_item_counter(item_counter, observation_id=observation_id)

    if isinstance(value, Mapping):
        _validate_mapping_keys(value, observation_id=observation_id)
        if not value:
            _append_structured_segment(
                observation_id=observation_id,
                kind=SegmentKind.STRUCTURED_OBJECT,
                value=value,
                path=path,
                metadata=metadata,
                segments=segments,
                extra_metadata={"structural_type": "mapping", "item_count": 0},
            )
            _enforce_segment_limit(
                len(segments),
                config,
                observation_id=observation_id,
            )
            return
        for key, item in value.items():
            child_path = (*path, key)
            if isinstance(item, Mapping) or _is_sequence_but_not_text(item):
                _append_structured_segment(
                    observation_id=observation_id,
                    kind=SegmentKind.FIELD,
                    value=item,
                    path=child_path,
                    metadata=metadata,
                    segments=segments,
                    extra_metadata={
                        "key": key,
                        "structural_type": _structural_type(item),
                        "item_count": len(item),
                    },
                )
                _enforce_segment_limit(
                    len(segments),
                    config,
                    observation_id=observation_id,
                )
                _walk_structured_value(
                    observation_id=observation_id,
                    value=item,
                    config=config,
                    path=child_path,
                    metadata=metadata,
                    segments=segments,
                    depth=depth + 1,
                    item_counter=item_counter,
                )
            elif _is_json_safe_scalar(item):
                _append_structured_segment(
                    observation_id=observation_id,
                    kind=SegmentKind.KEY_VALUE,
                    value=item,
                    path=child_path,
                    metadata=metadata,
                    segments=segments,
                    extra_metadata={"key": key, "structural_type": "scalar"},
                )
                _enforce_segment_limit(
                    len(segments),
                    config,
                    observation_id=observation_id,
                )
            else:
                raise SegmentationError(
                    "unsupported structured mapping value",
                    observation_id=observation_id,
                    context={"path": ".".join(child_path), "type": type(item).__name__},
                )
    elif _is_sequence_but_not_text(value):
        if not value:
            _append_structured_segment(
                observation_id=observation_id,
                kind=SegmentKind.STRUCTURED_OBJECT,
                value=value,
                path=path,
                metadata=metadata,
                segments=segments,
                extra_metadata={"structural_type": "sequence", "item_count": 0},
            )
            _enforce_segment_limit(
                len(segments),
                config,
                observation_id=observation_id,
            )
            return
        for index, item in enumerate(value):
            child_path = (*path, str(index))
            if isinstance(item, Mapping) or _is_sequence_but_not_text(item):
                _append_structured_segment(
                    observation_id=observation_id,
                    kind=SegmentKind.LIST_ITEM,
                    value=item,
                    path=child_path,
                    metadata=metadata,
                    segments=segments,
                    extra_metadata={
                        "index": index,
                        "structural_type": _structural_type(item),
                        "item_count": len(item),
                    },
                )
                _enforce_segment_limit(
                    len(segments),
                    config,
                    observation_id=observation_id,
                )
                _walk_structured_value(
                    observation_id=observation_id,
                    value=item,
                    config=config,
                    path=child_path,
                    metadata=metadata,
                    segments=segments,
                    depth=depth + 1,
                    item_counter=item_counter,
                )
            elif _is_json_safe_scalar(item):
                _append_structured_segment(
                    observation_id=observation_id,
                    kind=SegmentKind.LIST_ITEM,
                    value=item,
                    path=child_path,
                    metadata=metadata,
                    segments=segments,
                    extra_metadata={"index": index, "structural_type": "scalar"},
                )
                _enforce_segment_limit(
                    len(segments),
                    config,
                    observation_id=observation_id,
                )
            else:
                raise SegmentationError(
                    "unsupported structured sequence value",
                    observation_id=observation_id,
                    context={"path": ".".join(child_path), "type": type(item).__name__},
                )
    elif _is_json_safe_scalar(value):
        _append_structured_segment(
            observation_id=observation_id,
            kind=SegmentKind.FIELD if path != _DEFAULT_ROOT_PATH else SegmentKind.STRUCTURED_OBJECT,
            value=value,
            path=path,
            metadata=metadata,
            segments=segments,
            extra_metadata={"structural_type": "scalar"},
        )
        _enforce_segment_limit(len(segments), config, observation_id=observation_id)
    else:
        raise SegmentationError(
            "unsupported structured value",
            observation_id=observation_id,
            context={"path": ".".join(path), "type": type(value).__name__},
        )


def _append_structured_segment(
    *,
    observation_id: str,
    kind: SegmentKind,
    value: object,
    path: tuple[str, ...],
    metadata: Mapping[str, object],
    segments: list[ParsedSegment],
    extra_metadata: Mapping[str, object],
) -> None:
    ordinal = len(segments)
    segment_metadata = {
        **metadata,
        **extra_metadata,
        "strategy": SegmentationStrategy.STRUCTURED_FIELDS.value,
        "path": ".".join(path),
    }
    text = _scalar_to_text(value)
    if text is not None and len(text) > _MAX_SEGMENT_TEXT_LENGTH:
        raise SegmentationError(
            "segment text exceeds maximum length",
            observation_id=observation_id,
            context={
                "length": len(text),
                "max_length": _MAX_SEGMENT_TEXT_LENGTH,
                "path": ".".join(path),
            },
        )
    span = TextSpan(0, len(text)) if text is not None else None
    segments.append(
        _build_segment(
            observation_id=observation_id,
            kind=kind,
            ordinal=ordinal,
            text=text,
            span=span,
            path=path,
            metadata=segment_metadata,
        )
    )


def _build_segment(
    *,
    observation_id: str,
    kind: SegmentKind,
    ordinal: int,
    text: str | None,
    span: TextSpan | None,
    path: tuple[str, ...],
    metadata: Mapping[str, object],
) -> ParsedSegment:
    if text is not None and len(text) > _MAX_SEGMENT_TEXT_LENGTH:
        raise SegmentationError(
            "segment text exceeds maximum length",
            observation_id=observation_id,
            context={
                "length": len(text),
                "max_length": _MAX_SEGMENT_TEXT_LENGTH,
            },
        )
    segment_id = _make_segment_id(
        observation_id=observation_id,
        kind=kind,
        ordinal=ordinal,
        path=path,
        span=span,
    )
    return ParsedSegment(
        segment_id=segment_id,
        observation_id=observation_id,
        kind=kind,
        ordinal=ordinal,
        text=text,
        span=span,
        parent_segment_id=None,
        path=path,
        tokens=(),
        metadata=dict(metadata),
    )


def _rebuild_segment_with_ordinal(
    segment: ParsedSegment,
    ordinal: int,
) -> ParsedSegment:
    segment_id = _make_segment_id(
        observation_id=segment.observation_id,
        kind=segment.kind,
        ordinal=ordinal,
        path=segment.path,
        span=segment.span,
    )
    return ParsedSegment(
        segment_id=segment_id,
        observation_id=segment.observation_id,
        kind=segment.kind,
        ordinal=ordinal,
        text=segment.text,
        span=segment.span,
        parent_segment_id=segment.parent_segment_id,
        path=segment.path,
        tokens=(),
        metadata=dict(segment.metadata),
    )


def _has_code_fence(text: str) -> bool:
    return any(
        line.lstrip().startswith(prefix)
        for line in text.splitlines()
        for prefix in _CODE_FENCE_PREFIXES
    )


def _line_fence_prefix(stripped_line: str) -> str | None:
    for prefix in _CODE_FENCE_PREFIXES:
        if stripped_line.startswith(prefix):
            return prefix
    return None


def _has_multiple_non_empty_paragraphs(text: str) -> bool:
    return len(_paragraph_ranges(text)) > 1


def _paragraph_ranges(text: str) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    paragraph_start: int | None = None
    cursor = 0
    for line in text.splitlines(keepends=True):
        line_start = cursor
        line_end = cursor + len(line)
        cursor = line_end
        if line.strip() == "":
            if paragraph_start is not None:
                ranges.append((paragraph_start, line_start))
                paragraph_start = None
        elif paragraph_start is None:
            paragraph_start = line_start
    if paragraph_start is not None:
        ranges.append((paragraph_start, len(text)))
    return tuple(ranges)


def _increment_item_counter(
    item_counter: dict[str, int],
    *,
    observation_id: str,
) -> None:
    item_counter["count"] = item_counter.get("count", 0) + 1
    if item_counter["count"] > _MAX_STRUCTURED_ITEMS:
        raise SegmentationError(
            "structured content exceeds maximum item count",
            observation_id=observation_id,
            context={
                "count": item_counter["count"],
                "max_items": _MAX_STRUCTURED_ITEMS,
            },
        )


def _validate_mapping_keys(
    value: Mapping[object, object],
    *,
    observation_id: str,
) -> None:
    for key in value:
        if not isinstance(key, str):
            raise SegmentationError(
                "structured mapping keys must be strings",
                observation_id=observation_id,
                context={"key_type": type(key).__name__},
            )
        _ensure_non_empty_string(key, field_name="mapping key")


def _structural_type(value: object) -> str:
    if isinstance(value, Mapping):
        return "mapping"
    if _is_sequence_but_not_text(value):
        return "sequence"
    if _is_json_safe_scalar(value):
        return "scalar"
    raise SegmentationError(
        "unsupported structural value type",
        context={"type": type(value).__name__},
    )


def _single_segment_kind(value: object) -> SegmentKind:
    if isinstance(value, str):
        return SegmentKind.TEXT
    if isinstance(value, Mapping) or _is_sequence_but_not_text(value):
        return SegmentKind.STRUCTURED_OBJECT
    return SegmentKind.FIELD


def _sanitize_id_part(value: str) -> str:
    sanitized = []
    for character in value:
        if character.isspace() or ord(character) < 32:
            sanitized.append("_")
        elif character in {":", "/"}:
            sanitized.append("_")
        else:
            sanitized.append(character)
    return "".join(sanitized)


def _is_finite_float(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


__all__ = [
    "SegmentationStrategy",
    "SegmentRequest",
    "segment_observation",
    "extract_parseable_surface",
    "segment_text",
    "segment_structured",
    "create_single_segment",
]
