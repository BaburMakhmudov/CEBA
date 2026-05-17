"""CEBA Parser Layer dispatcher.

This module routes ParseRequest values through extraction and validation. It
does not infer meaning, compute final delta values, telos, world model state, or
memory updates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType

from ois.parser.difference import DifferenceExtractionStrategy
from ois.parser.errors import ParserDispatchError, ParserError, ParserValidationError
from ois.parser.extractor import (
    ParserExtractionRequest,
    ParserExtractionStrategy,
    build_parser_trace,
    extract_parse_result,
)
from ois.parser.schema import ParseRequest, ParseResult, ParseStatus, ParserTraceKind
from ois.parser.segments import SegmentationStrategy
from ois.parser.tokenizer import TokenizationStrategy
from ois.parser.validation import (
    ValidationMode,
    validate_parse_request,
    validate_parse_result,
)


_DEFAULT_DISPATCHER_NAME = "parser_dispatcher"
_DEFAULT_DISPATCHER_VERSION = "1.0.0"

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


class ParserDispatchMode(str, Enum):
    STRICT = "strict"
    PERMISSIVE = "permissive"


class ParserDispatchStatus(str, Enum):
    DISPATCHED = "dispatched"
    EMPTY = "empty"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ParserDispatchConfig:
    mode: ParserDispatchMode = ParserDispatchMode.STRICT
    validate_request: bool = True
    validate_result: bool = True
    segmentation_strategy: SegmentationStrategy = SegmentationStrategy.AUTO
    tokenization_strategy: TokenizationStrategy = TokenizationStrategy.SURFACE
    difference_strategy: DifferenceExtractionStrategy = (
        DifferenceExtractionStrategy.LOCAL_SURFACE
    )
    extraction_strategy: ParserExtractionStrategy = ParserExtractionStrategy.DEFAULT
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_dispatch_mode(self.mode)
        _ensure_bool(self.validate_request, field_name="validate_request")
        _ensure_bool(self.validate_result, field_name="validate_result")
        _ensure_segmentation_strategy(self.segmentation_strategy)
        _ensure_tokenization_strategy(self.tokenization_strategy)
        _ensure_difference_strategy(self.difference_strategy)
        _ensure_extraction_strategy(self.extraction_strategy)
        object.__setattr__(self, "metadata", _ensure_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ParserDispatchRequest:
    parse_request: ParseRequest
    config: ParserDispatchConfig = field(default_factory=ParserDispatchConfig)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_parse_request(self.parse_request)
        _ensure_dispatch_config(self.config)
        object.__setattr__(self, "metadata", _ensure_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ParserDispatchOutcome:
    status: ParserDispatchStatus
    result: ParseResult | None
    request_id: str
    observation_id: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_status = _ensure_dispatch_status(self.status)
        if self.result is None and normalized_status != ParserDispatchStatus.FAILED:
            raise ParserDispatchError(
                "result may be None only when dispatch status is failed",
                request_id=self.request_id if isinstance(self.request_id, str) else None,
                observation_id=(
                    self.observation_id
                    if isinstance(self.observation_id, str)
                    else None
                ),
            )
        if self.result is not None:
            _ensure_parse_result(self.result)
        _ensure_non_empty_string(self.request_id, field_name="request_id")
        _ensure_non_empty_string(self.observation_id, field_name="observation_id")
        object.__setattr__(self, "status", normalized_status)
        object.__setattr__(self, "metadata", _ensure_metadata(self.metadata))


class ParserDispatcher:
    name = _DEFAULT_DISPATCHER_NAME
    version = _DEFAULT_DISPATCHER_VERSION

    def dispatch(self, request: ParserDispatchRequest) -> ParserDispatchOutcome:
        try:
            dispatch_request = _ensure_dispatch_request(request)
            config = _ensure_dispatch_config(dispatch_request.config)
            parse_request = _ensure_parse_request(dispatch_request.parse_request)
            request_id = _ensure_non_empty_string(
                parse_request.request_id,
                field_name="request_id",
            )
            observation_id = _ensure_non_empty_string(
                str(parse_request.observation.observation_id),
                field_name="observation_id",
            )
            validation_mode = _validation_mode_for_dispatch(config.mode)
            dispatch_metadata: dict[str, object] = {
                "dispatcher_name": self.name,
                "dispatcher_version": self.version,
                "dispatch_mode": config.mode.value,
                "validate_request": config.validate_request,
                "validate_result": config.validate_result,
                "segmentation_strategy": config.segmentation_strategy.value,
                "tokenization_strategy": config.tokenization_strategy.value,
                "difference_strategy": config.difference_strategy.value,
                "extraction_strategy": config.extraction_strategy.value,
            }

            if config.validate_request:
                request_report = validate_parse_request(
                    parse_request,
                    mode=validation_mode,
                )
                dispatch_metadata["request_validation_valid"] = request_report.valid
                dispatch_metadata["request_validation_issue_count"] = len(
                    request_report.issues
                )
                if validation_mode == ValidationMode.COLLECT and not request_report.valid:
                    raise ParserDispatchError(
                        "parse request validation failed",
                        request_id=request_id,
                        observation_id=observation_id,
                        context={
                            "validation_id": request_report.validation_id,
                            "issue_count": len(request_report.issues),
                        },
                    )

            extraction_request = ParserExtractionRequest(
                parse_request=parse_request,
                segmentation_strategy=config.segmentation_strategy,
                tokenization_strategy=config.tokenization_strategy,
                difference_strategy=config.difference_strategy,
                extraction_strategy=config.extraction_strategy,
                metadata=_merge_metadata(
                    config.metadata,
                    dispatch_request.metadata,
                    dispatch_metadata,
                ),
            )
            result = _ensure_parse_result(extract_parse_result(extraction_request))

            if config.validate_result:
                result_report = validate_parse_result(result, mode=validation_mode)
                dispatch_metadata["result_validation_valid"] = result_report.valid
                dispatch_metadata["result_validation_issue_count"] = len(
                    result_report.issues
                )
                if validation_mode == ValidationMode.COLLECT and not result_report.valid:
                    raise ParserDispatchError(
                        "parse result validation failed",
                        request_id=request_id,
                        observation_id=observation_id,
                        context={
                            "validation_id": result_report.validation_id,
                            "issue_count": len(result_report.issues),
                        },
                    )

            dispatch_status = _status_from_parse_result(result)
            result_with_trace = append_dispatch_trace(
                result=result,
                dispatcher_name=self.name,
                dispatcher_version=self.version,
                status=dispatch_status,
                metadata=dispatch_metadata,
            )
            outcome_metadata = _merge_metadata(
                config.metadata,
                dispatch_request.metadata,
                dispatch_metadata,
                {
                    "dispatch_status": dispatch_status.value,
                    "status": dispatch_status.value,
                },
            )
            return ParserDispatchOutcome(
                status=dispatch_status,
                result=result_with_trace,
                request_id=request_id,
                observation_id=observation_id,
                metadata=outcome_metadata,
            )
        except ParserValidationError:
            raise
        except ParserError:
            raise
        except Exception as error:
            request_id = None
            observation_id = None
            if isinstance(request, ParserDispatchRequest) and isinstance(
                request.parse_request,
                ParseRequest,
            ):
                request_id = request.parse_request.request_id
                observation_id = str(request.parse_request.observation.observation_id)
            raise ParserDispatchError(
                "unexpected parser dispatch failure",
                request_id=request_id,
                observation_id=observation_id,
                context={"error_type": type(error).__name__},
            ) from error

    def supports(self, request: object) -> bool:
        return isinstance(request, (ParserDispatchRequest, ParseRequest))


def dispatch_parse_request(
    parse_request: ParseRequest,
    *,
    config: ParserDispatchConfig | None = None,
    metadata: Mapping[str, object] | None = None,
) -> ParseResult:
    dispatch_config = ParserDispatchConfig() if config is None else config
    outcome = ParserDispatcher().dispatch(
        ParserDispatchRequest(
            parse_request=parse_request,
            config=dispatch_config,
            metadata={} if metadata is None else metadata,
        )
    )
    if outcome.result is None:
        raise ParserDispatchError(
            "parser dispatch produced no ParseResult",
            request_id=outcome.request_id,
            observation_id=outcome.observation_id,
        )
    return outcome.result


def dispatch_parser_request(
    request: ParserDispatchRequest,
) -> ParserDispatchOutcome:
    return ParserDispatcher().dispatch(request)


def create_parser_dispatcher() -> ParserDispatcher:
    return ParserDispatcher()


def append_dispatch_trace(
    *,
    result: ParseResult,
    dispatcher_name: str,
    dispatcher_version: str,
    status: ParserDispatchStatus,
    metadata: Mapping[str, object] | None = None,
) -> ParseResult:
    normalized_result = _ensure_parse_result(result)
    normalized_dispatcher_name = _ensure_non_empty_string(
        dispatcher_name,
        field_name="dispatcher_name",
    )
    normalized_dispatcher_version = _ensure_non_empty_string(
        dispatcher_version,
        field_name="dispatcher_version",
    )
    normalized_status = _ensure_dispatch_status(status)
    trace_metadata = _merge_metadata(
        {
            "stage": "dispatch",
            "status": normalized_status.value,
            "dispatcher_name": normalized_dispatcher_name,
            "dispatcher_version": normalized_dispatcher_version,
        },
        metadata,
    )
    dispatch_trace = build_parser_trace(
        request_id=normalized_result.request_id,
        observation_id=normalized_result.observation_id,
        kind=ParserTraceKind.DISPATCH,
        message="parser dispatch completed",
        ordinal=len(normalized_result.traces),
        metadata=trace_metadata,
        created_at=_utc_now(),
    )
    result_metadata = _merge_metadata(
        normalized_result.metadata,
        metadata,
        {
            "dispatcher_name": normalized_dispatcher_name,
            "dispatcher_version": normalized_dispatcher_version,
            "dispatch_status": normalized_status.value,
            "stage": "dispatch",
            "status": normalized_status.value,
        },
    )
    return ParseResult(
        request_id=normalized_result.request_id,
        observation_id=normalized_result.observation_id,
        status=normalized_result.status,
        segments=normalized_result.segments,
        difference_candidates=normalized_result.difference_candidates,
        traces=normalized_result.traces + (dispatch_trace,),
        created_at=normalized_result.created_at,
        metadata=result_metadata,
    )


def _ensure_dispatch_request(value: ParserDispatchRequest) -> ParserDispatchRequest:
    if not isinstance(value, ParserDispatchRequest):
        raise ParserDispatchError(
            "request must be ParserDispatchRequest",
            context={"type": type(value).__name__},
        )
    _ensure_parse_request(value.parse_request)
    _ensure_dispatch_config(value.config)
    _ensure_metadata(value.metadata)
    return value


def _ensure_dispatch_config(value: ParserDispatchConfig) -> ParserDispatchConfig:
    if not isinstance(value, ParserDispatchConfig):
        raise ParserDispatchError(
            "config must be ParserDispatchConfig",
            context={"type": type(value).__name__},
        )
    _ensure_dispatch_mode(value.mode)
    _ensure_bool(value.validate_request, field_name="validate_request")
    _ensure_bool(value.validate_result, field_name="validate_result")
    _ensure_segmentation_strategy(value.segmentation_strategy)
    _ensure_tokenization_strategy(value.tokenization_strategy)
    _ensure_difference_strategy(value.difference_strategy)
    _ensure_extraction_strategy(value.extraction_strategy)
    _ensure_metadata(value.metadata)
    return value


def _ensure_parse_request(value: ParseRequest) -> ParseRequest:
    if not isinstance(value, ParseRequest):
        raise ParserDispatchError(
            "parse_request must be ParseRequest",
            context={"type": type(value).__name__},
        )
    _ensure_non_empty_string(value.request_id, field_name="request_id")
    return value


def _ensure_parse_result(value: ParseResult) -> ParseResult:
    if not isinstance(value, ParseResult):
        raise ParserDispatchError(
            "result must be ParseResult",
            context={"type": type(value).__name__},
        )
    _ensure_non_empty_string(value.request_id, field_name="request_id")
    _ensure_non_empty_string(value.observation_id, field_name="observation_id")
    return value


def _ensure_dispatch_mode(value: ParserDispatchMode) -> ParserDispatchMode:
    if not isinstance(value, ParserDispatchMode):
        raise ParserDispatchError(
            "mode must be ParserDispatchMode",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_dispatch_status(value: ParserDispatchStatus) -> ParserDispatchStatus:
    if not isinstance(value, ParserDispatchStatus):
        raise ParserDispatchError(
            "status must be ParserDispatchStatus",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_segmentation_strategy(value: SegmentationStrategy) -> SegmentationStrategy:
    if not isinstance(value, SegmentationStrategy):
        raise ParserDispatchError(
            "segmentation_strategy must be SegmentationStrategy",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_tokenization_strategy(value: TokenizationStrategy) -> TokenizationStrategy:
    if not isinstance(value, TokenizationStrategy):
        raise ParserDispatchError(
            "tokenization_strategy must be TokenizationStrategy",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_difference_strategy(
    value: DifferenceExtractionStrategy,
) -> DifferenceExtractionStrategy:
    if not isinstance(value, DifferenceExtractionStrategy):
        raise ParserDispatchError(
            "difference_strategy must be DifferenceExtractionStrategy",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_extraction_strategy(
    value: ParserExtractionStrategy,
) -> ParserExtractionStrategy:
    if not isinstance(value, ParserExtractionStrategy):
        raise ParserDispatchError(
            "extraction_strategy must be ParserExtractionStrategy",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_bool(value: bool, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ParserDispatchError(
            f"{field_name} must be a bool",
            context={"type": type(value).__name__},
        )
    return value


def _ensure_non_empty_string(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ParserDispatchError(
            f"{field_name} must be a string",
            context={"type": type(value).__name__},
        )
    if not value:
        raise ParserDispatchError(f"{field_name} must not be empty")
    if any(ord(character) < 32 for character in value):
        raise ParserDispatchError(f"{field_name} must not contain control characters")
    return value


def _ensure_metadata(value: Mapping[str, object] | None) -> Mapping[str, object]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ParserDispatchError(
            "metadata must be a mapping",
            context={"type": type(value).__name__},
        )
    copied: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ParserDispatchError(
                "metadata keys must be strings",
                context={"type": type(key).__name__},
            )
        normalized_key = key.strip().lower().replace("-", "_")
        for forbidden_key in _FORBIDDEN_METADATA_KEYS:
            if normalized_key == forbidden_key or forbidden_key in normalized_key:
                raise ParserDispatchError(
                    "metadata key crosses parser dispatch boundary",
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


def _validation_mode_for_dispatch(mode: ParserDispatchMode) -> ValidationMode:
    normalized_mode = _ensure_dispatch_mode(mode)
    if normalized_mode == ParserDispatchMode.STRICT:
        return ValidationMode.RAISE
    return ValidationMode.COLLECT


def _status_from_parse_result(result: ParseResult) -> ParserDispatchStatus:
    normalized_result = _ensure_parse_result(result)
    if normalized_result.status == ParseStatus.EMPTY:
        return ParserDispatchStatus.EMPTY
    return ParserDispatchStatus.DISPATCHED


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = [
    "ParserDispatchMode",
    "ParserDispatchStatus",
    "ParserDispatchConfig",
    "ParserDispatchRequest",
    "ParserDispatchOutcome",
    "ParserDispatcher",
    "dispatch_parse_request",
    "dispatch_parser_request",
    "create_parser_dispatcher",
    "append_dispatch_trace",
]
