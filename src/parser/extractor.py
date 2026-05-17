"""CEBA Parser Layer extraction pipeline.

This module composes segmentation, tokenization, and local difference-candidate
extraction into deterministic parser artifacts. It does not compute final
meaning, telos, world model state, final delta values, or memory updates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from ois.parser.difference import (
    DifferenceExtractionRequest,
    DifferenceExtractionStrategy,
    extract_local_differences,
)
from ois.parser.errors import EmptyObservationError, ParserError, ParserSchemaError
from ois.parser.schema import (
    LocalDifferenceCandidate,
    ParsedSegment,
    ParseRequest,
    ParseResult,
    ParseStatus,
    ParserConfig,
    ParserTrace,
    ParserTraceKind,
)
from ois.parser.segments import (
    SegmentRequest,
    SegmentationStrategy,
    segment_observation,
)
from ois.parser.tokenizer import (
    TokenizationRequest,
    TokenizationStrategy,
    tokenize_segments,
)


_TRACE_ID_PREFIX = "ptrace"
_DEFAULT_EXTRACTOR_VERSION = "1.0.0"
_FORBIDDEN_METADATA_KEYS = frozenset(
    {
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
    }
)


class ParserExtractionStrategy(str, Enum):
    DEFAULT = "default"


@dataclass(frozen=True, slots=True)
class ParserExtractionRequest:
    parse_request: ParseRequest
    segmentation_strategy: SegmentationStrategy = SegmentationStrategy.AUTO
    tokenization_strategy: TokenizationStrategy = TokenizationStrategy.SURFACE
    difference_strategy: DifferenceExtractionStrategy = (
        DifferenceExtractionStrategy.LOCAL_SURFACE
    )
    extraction_strategy: ParserExtractionStrategy = ParserExtractionStrategy.DEFAULT
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        _ensure_parse_request(self.parse_request)
        _ensure_segmentation_strategy(self.segmentation_strategy)
        _ensure_tokenization_strategy(self.tokenization_strategy)
        _ensure_difference_strategy(self.difference_strategy)
        _ensure_extraction_strategy(self.extraction_strategy)
        object.__setattr__(self, "metadata", _ensure_metadata(self.metadata))


def extract_parse_result(request: ParserExtractionRequest) -> ParseResult:
    try:
        extraction_request = _ensure_extraction_request(request)
        parse_request = _ensure_parse_request(extraction_request.parse_request)
        config = _ensure_config(parse_request.config)
        request_id = _ensure_non_empty_string(
            parse_request.request_id,
            field_name="request_id",
        )
        observation_id = _ensure_non_empty_string(
            str(parse_request.observation.observation_id),
            field_name="observation_id",
        )
        created_at = _utc_now()
        traces = [
            build_parser_trace(
                request_id=request_id,
                observation_id=observation_id,
                kind=ParserTraceKind.REQUEST_CREATED,
                message="parser extraction request accepted",
                ordinal=0,
                metadata={"stage": "request", "status": "accepted"},
                created_at=created_at,
            )
        ]

        try:
            segment_request = SegmentRequest(
                parse_request.observation,
                config,
                extraction_request.segmentation_strategy,
            )
            segments = segment_observation(segment_request)
        except EmptyObservationError as error:
            traces.append(
                build_parser_trace(
                    request_id=request_id,
                    observation_id=observation_id,
                    kind=ParserTraceKind.SEGMENTATION,
                    message="segmentation produced empty parseable content",
                    ordinal=1,
                    metadata={
                        "stage": "segmentation",
                        "status": "empty",
                        "empty_reason": error.message,
                    },
                    created_at=created_at,
                )
            )
            return ParseResult(
                request_id=request_id,
                observation_id=observation_id,
                status=ParseStatus.EMPTY,
                segments=(),
                difference_candidates=(),
                traces=tuple(traces),
                created_at=created_at,
                metadata={
                    "extractor_version": _DEFAULT_EXTRACTOR_VERSION,
                    "status": ParseStatus.EMPTY.value,
                    "empty_reason": error.message,
                },
            )

        segments = _ensure_segments(segments)
        traces.append(
            build_parser_trace(
                request_id=request_id,
                observation_id=observation_id,
                kind=ParserTraceKind.SEGMENTATION,
                message="segmentation completed",
                ordinal=1,
                metadata={
                    "stage": "segmentation",
                    "status": "completed",
                    "segment_count": len(segments),
                },
                created_at=created_at,
            )
        )

        tokenization_request = TokenizationRequest(
            segments,
            config,
            extraction_request.tokenization_strategy,
        )
        tokenized_segments = _ensure_segments(tokenize_segments(tokenization_request))
        traces.append(
            build_parser_trace(
                request_id=request_id,
                observation_id=observation_id,
                kind=ParserTraceKind.TOKENIZATION,
                message="tokenization completed",
                ordinal=2,
                metadata={
                    "stage": "tokenization",
                    "status": "completed",
                    "tokenized_segment_count": len(tokenized_segments),
                },
                created_at=created_at,
            )
        )

        difference_request = DifferenceExtractionRequest(
            tokenized_segments,
            config,
            extraction_request.difference_strategy,
        )
        difference_candidates = _ensure_candidates(
            extract_local_differences(difference_request)
        )
        traces.append(
            build_parser_trace(
                request_id=request_id,
                observation_id=observation_id,
                kind=ParserTraceKind.LOCAL_DIFFERENCE_EXTRACTION,
                message="local difference extraction completed",
                ordinal=3,
                metadata={
                    "stage": "local_difference_extraction",
                    "status": "completed",
                    "difference_candidate_count": len(difference_candidates),
                },
                created_at=created_at,
            )
        )

        metadata = {
            **_ensure_metadata(extraction_request.metadata),
            "extractor_version": _DEFAULT_EXTRACTOR_VERSION,
            "extraction_strategy": extraction_request.extraction_strategy.value,
            "segmentation_strategy": extraction_request.segmentation_strategy.value,
            "tokenization_strategy": extraction_request.tokenization_strategy.value,
            "difference_strategy": extraction_request.difference_strategy.value,
            "segment_count": len(tokenized_segments),
            "difference_candidate_count": len(difference_candidates),
        }
        return ParseResult(
            request_id=request_id,
            observation_id=observation_id,
            status=ParseStatus.PARSED,
            segments=tokenized_segments,
            difference_candidates=difference_candidates,
            traces=tuple(traces),
            created_at=created_at,
            metadata=metadata,
        )
    except ParserError:
        raise
    except Exception as error:
        request_id = None
        observation_id = None
        if isinstance(request, ParserExtractionRequest) and isinstance(
            request.parse_request,
            ParseRequest,
        ):
            request_id = request.parse_request.request_id
            observation_id = str(request.parse_request.observation.observation_id)
        raise ParserSchemaError(
            "unexpected parser extraction failure",
            request_id=request_id,
            observation_id=observation_id,
            context={"error_type": type(error).__name__},
        ) from error


def extract_parser_artifacts(
    *,
    parse_request: ParseRequest,
    segmentation_strategy: SegmentationStrategy = SegmentationStrategy.AUTO,
    tokenization_strategy: TokenizationStrategy = TokenizationStrategy.SURFACE,
    difference_strategy: DifferenceExtractionStrategy = (
        DifferenceExtractionStrategy.LOCAL_SURFACE
    ),
    metadata: Mapping[str, object] | None = None,
) -> ParseResult:
    return extract_parse_result(
        ParserExtractionRequest(
            parse_request=parse_request,
            segmentation_strategy=segmentation_strategy,
            tokenization_strategy=tokenization_strategy,
            difference_strategy=difference_strategy,
            metadata=metadata,
        )
    )


def build_parser_trace(
    *,
    request_id: str,
    observation_id: str,
    kind: ParserTraceKind,
    message: str,
    ordinal: int,
    metadata: Mapping[str, object] | None = None,
    created_at: datetime | None = None,
) -> ParserTrace:
    normalized_request_id = _ensure_non_empty_string(
        request_id,
        field_name="request_id",
    )
    normalized_observation_id = _ensure_non_empty_string(
        observation_id,
        field_name="observation_id",
    )
    normalized_kind = _ensure_trace_kind(kind)
    normalized_message = _ensure_non_empty_string(message, field_name="message")
    if type(ordinal) is not int:
        raise ParserSchemaError(
            "ordinal must be an int",
            request_id=normalized_request_id,
            observation_id=normalized_observation_id,
            context={"type": type(ordinal).__name__},
        )
    if ordinal < 0:
        raise ParserSchemaError(
            "ordinal must be greater than or equal to 0",
            request_id=normalized_request_id,
            observation_id=normalized_observation_id,
            context={"ordinal": ordinal},
        )
    trace_created_at = (
        _utc_now()
        if created_at is None
        else _ensure_timezone_aware(created_at, field_name="created_at")
    )
    return ParserTrace(
        trace_id=make_parser_trace_id(
            request_id=normalized_request_id,
            observation_id=normalized_observation_id,
            kind=normalized_kind,
            ordinal=ordinal,
        ),
        observation_id=normalized_observation_id,
        kind=normalized_kind,
        message=normalized_message,
        created_at=trace_created_at,
        metadata=_ensure_metadata(metadata),
    )


def make_parser_trace_id(
    *,
    request_id: str,
    observation_id: str,
    kind: ParserTraceKind,
    ordinal: int,
) -> str:
    normalized_request_id = _ensure_non_empty_string(
        request_id,
        field_name="request_id",
    )
    normalized_observation_id = _ensure_non_empty_string(
        observation_id,
        field_name="observation_id",
    )
    normalized_kind = _ensure_trace_kind(kind)
    if type(ordinal) is not int:
        raise ParserSchemaError(
            "ordinal must be an int",
            request_id=normalized_request_id,
            observation_id=normalized_observation_id,
            context={"type": type(ordinal).__name__},
        )
    if ordinal < 0:
        raise ParserSchemaError(
            "ordinal must be greater than or equal to 0",
            request_id=normalized_request_id,
            observation_id=normalized_observation_id,
            context={"ordinal": ordinal},
        )
    return (
        f"{_TRACE_ID_PREFIX}:"
        f"{_sanitize_id_part(normalized_request_id)}:"
        f"{_sanitize_id_part(normalized_observation_id)}:"
        f"{ordinal}:"
        f"{normalized_kind.value}"
    )


def _ensure_extraction_request(
    value: ParserExtractionRequest,
) -> ParserExtractionRequest:
    if not isinstance(value, ParserExtractionRequest):
        raise ParserSchemaError(
            "request must be ParserExtractionRequest",
            context={"type": type(value).__name__},
        )
    _ensure_parse_request(value.parse_request)
    _ensure_segmentation_strategy(value.segmentation_strategy)
    _ensure_tokenization_strategy(value.tokenization_strategy)
    _ensure_difference_strategy(value.difference_strategy)
    _ensure_extraction_strategy(value.extraction_strategy)
    _ensure_metadata(value.metadata)
    return value


def _ensure_parse_request(value: ParseRequest) -> ParseRequest:
    if not isinstance(value, ParseRequest):
        raise ParserSchemaError(
            "parse_request must be ParseRequest",
            context={"type": type(value).__name__},
        )
    _ensure_non_empty_string(value.request_id, field_name="request_id")
    _ensure_config(value.config)
    _ensure_timezone_aware(value.created_at, field_name="created_at")
    return value


def _ensure_config(value: ParserConfig) -> ParserConfig:
    if not isinstance(value, ParserConfig):
        raise ParserSchemaError(
            "config must be ParserConfig",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_segmentation_strategy(value: SegmentationStrategy) -> SegmentationStrategy:
    if not isinstance(value, SegmentationStrategy):
        raise ParserSchemaError(
            "segmentation_strategy must be SegmentationStrategy",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_tokenization_strategy(value: TokenizationStrategy) -> TokenizationStrategy:
    if not isinstance(value, TokenizationStrategy):
        raise ParserSchemaError(
            "tokenization_strategy must be TokenizationStrategy",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_difference_strategy(
    value: DifferenceExtractionStrategy,
) -> DifferenceExtractionStrategy:
    if not isinstance(value, DifferenceExtractionStrategy):
        raise ParserSchemaError(
            "difference_strategy must be DifferenceExtractionStrategy",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_extraction_strategy(
    value: ParserExtractionStrategy,
) -> ParserExtractionStrategy:
    if not isinstance(value, ParserExtractionStrategy):
        raise ParserSchemaError(
            "extraction_strategy must be ParserExtractionStrategy",
            context={"type": type(value).__name__},
        )
    if value != ParserExtractionStrategy.DEFAULT:
        raise ParserSchemaError(
            "unsupported parser extraction strategy",
            context={"extraction_strategy": value.value},
        )
    return value


def _ensure_trace_kind(value: ParserTraceKind) -> ParserTraceKind:
    if not isinstance(value, ParserTraceKind):
        raise ParserSchemaError(
            "kind must be ParserTraceKind",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_segments(value: tuple[ParsedSegment, ...]) -> tuple[ParsedSegment, ...]:
    if not isinstance(value, tuple):
        raise ParserSchemaError(
            "segments must be a tuple",
            context={"type": type(value).__name__},
        )
    for item in value:
        if not isinstance(item, ParsedSegment):
            raise ParserSchemaError(
                "segments must contain ParsedSegment values",
                context={"type": type(item).__name__},
            )
    return value


def _ensure_candidates(
    value: tuple[LocalDifferenceCandidate, ...],
) -> tuple[LocalDifferenceCandidate, ...]:
    if not isinstance(value, tuple):
        raise ParserSchemaError(
            "difference_candidates must be a tuple",
            context={"type": type(value).__name__},
        )
    for item in value:
        if not isinstance(item, LocalDifferenceCandidate):
            raise ParserSchemaError(
                "difference_candidates must contain LocalDifferenceCandidate values",
                context={"type": type(item).__name__},
            )
    return value


def _ensure_metadata(value: Mapping[str, object] | None) -> dict[str, object]:
    if value is None:
        return {}
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
                    "metadata key crosses parser extraction boundary",
                    context={"metadata_key": key},
                )
        copied[key] = item
    return copied


def _ensure_non_empty_string(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ParserSchemaError(
            f"{field_name} must be a string",
            context={"type": type(value).__name__},
        )
    if not value:
        raise ParserSchemaError(f"{field_name} must not be empty")
    return value


def _ensure_timezone_aware(value: datetime, *, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise ParserSchemaError(
            f"{field_name} must be a datetime",
            context={"type": type(value).__name__},
        )
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ParserSchemaError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _contains_disallowed_control_chars(value: str) -> bool:
    return any(character.isspace() or ord(character) < 32 for character in value)


def _sanitize_id_part(value: str) -> str:
    normalized_value = _ensure_non_empty_string(value, field_name="id_part")
    if _contains_disallowed_control_chars(normalized_value):
        raise ParserSchemaError("id_part must not contain whitespace or controls")
    return normalized_value


__all__ = [
    "ParserExtractionStrategy",
    "ParserExtractionRequest",
    "extract_parse_result",
    "extract_parser_artifacts",
    "build_parser_trace",
    "make_parser_trace_id",
]
