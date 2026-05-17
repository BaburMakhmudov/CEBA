"""Typed CEBA Parser Layer errors.

Parser errors preserve parser-layer failure context while converting
CanonicalObservation values into parser artifacts. They do not perform parsing,
tokenization, segmentation, difference extraction, or infer meaning.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType


PARSER_ERROR = "PARSER_ERROR"
PARSER_SCHEMA_ERROR = "PARSER_SCHEMA_ERROR"
PARSER_REQUEST_ERROR = "PARSER_REQUEST_ERROR"
PARSER_UNSUPPORTED_OBSERVATION = "PARSER_UNSUPPORTED_OBSERVATION"
PARSER_EMPTY_OBSERVATION = "PARSER_EMPTY_OBSERVATION"
PARSER_SEGMENTATION_ERROR = "PARSER_SEGMENTATION_ERROR"
PARSER_TOKENIZATION_ERROR = "PARSER_TOKENIZATION_ERROR"
PARSER_LOCAL_DIFFERENCE_EXTRACTION_ERROR = (
    "PARSER_LOCAL_DIFFERENCE_EXTRACTION_ERROR"
)
PARSER_VALIDATION_ERROR = "PARSER_VALIDATION_ERROR"
PARSER_LIMIT_EXCEEDED = "PARSER_LIMIT_EXCEEDED"
PARSER_DISPATCH_ERROR = "PARSER_DISPATCH_ERROR"
PARSER_AUDIT_ERROR = "PARSER_AUDIT_ERROR"
PARSER_BOUNDARY_ERROR = "PARSER_BOUNDARY_ERROR"


class ParserError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: str = PARSER_ERROR,
        context: Mapping[str, object] | None = None,
        request_id: str | None = None,
        observation_id: str | None = None,
        segment_id: str | None = None,
        token_id: str | None = None,
        candidate_id: str | None = None,
    ) -> None:
        normalized_message = _ensure_non_empty_string(
            message,
            field_name="message",
        )
        normalized_code = _ensure_non_empty_string(code, field_name="code")
        self.message = normalized_message
        self.code = normalized_code
        self.context = _build_error_context(
            context,
            request_id=request_id,
            observation_id=observation_id,
            segment_id=segment_id,
            token_id=token_id,
            candidate_id=candidate_id,
        )
        super().__init__(normalized_message)

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "context": dict(self.context),
        }


class ParserSchemaError(ParserError):
    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, object] | None = None,
        request_id: str | None = None,
        observation_id: str | None = None,
        segment_id: str | None = None,
        token_id: str | None = None,
        candidate_id: str | None = None,
    ) -> None:
        super().__init__(
            message,
            code=PARSER_SCHEMA_ERROR,
            context=context,
            request_id=request_id,
            observation_id=observation_id,
            segment_id=segment_id,
            token_id=token_id,
            candidate_id=candidate_id,
        )


class ParserRequestError(ParserError):
    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, object] | None = None,
        request_id: str | None = None,
        observation_id: str | None = None,
        segment_id: str | None = None,
        token_id: str | None = None,
        candidate_id: str | None = None,
    ) -> None:
        super().__init__(
            message,
            code=PARSER_REQUEST_ERROR,
            context=context,
            request_id=request_id,
            observation_id=observation_id,
            segment_id=segment_id,
            token_id=token_id,
            candidate_id=candidate_id,
        )


class UnsupportedObservationError(ParserError):
    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, object] | None = None,
        request_id: str | None = None,
        observation_id: str | None = None,
        segment_id: str | None = None,
        token_id: str | None = None,
        candidate_id: str | None = None,
    ) -> None:
        super().__init__(
            message,
            code=PARSER_UNSUPPORTED_OBSERVATION,
            context=context,
            request_id=request_id,
            observation_id=observation_id,
            segment_id=segment_id,
            token_id=token_id,
            candidate_id=candidate_id,
        )


class EmptyObservationError(ParserError):
    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, object] | None = None,
        request_id: str | None = None,
        observation_id: str | None = None,
        segment_id: str | None = None,
        token_id: str | None = None,
        candidate_id: str | None = None,
    ) -> None:
        super().__init__(
            message,
            code=PARSER_EMPTY_OBSERVATION,
            context=context,
            request_id=request_id,
            observation_id=observation_id,
            segment_id=segment_id,
            token_id=token_id,
            candidate_id=candidate_id,
        )


class SegmentationError(ParserError):
    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, object] | None = None,
        request_id: str | None = None,
        observation_id: str | None = None,
        segment_id: str | None = None,
        token_id: str | None = None,
        candidate_id: str | None = None,
    ) -> None:
        super().__init__(
            message,
            code=PARSER_SEGMENTATION_ERROR,
            context=context,
            request_id=request_id,
            observation_id=observation_id,
            segment_id=segment_id,
            token_id=token_id,
            candidate_id=candidate_id,
        )


class TokenizationError(ParserError):
    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, object] | None = None,
        request_id: str | None = None,
        observation_id: str | None = None,
        segment_id: str | None = None,
        token_id: str | None = None,
        candidate_id: str | None = None,
    ) -> None:
        super().__init__(
            message,
            code=PARSER_TOKENIZATION_ERROR,
            context=context,
            request_id=request_id,
            observation_id=observation_id,
            segment_id=segment_id,
            token_id=token_id,
            candidate_id=candidate_id,
        )


class LocalDifferenceExtractionError(ParserError):
    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, object] | None = None,
        request_id: str | None = None,
        observation_id: str | None = None,
        segment_id: str | None = None,
        token_id: str | None = None,
        candidate_id: str | None = None,
    ) -> None:
        super().__init__(
            message,
            code=PARSER_LOCAL_DIFFERENCE_EXTRACTION_ERROR,
            context=context,
            request_id=request_id,
            observation_id=observation_id,
            segment_id=segment_id,
            token_id=token_id,
            candidate_id=candidate_id,
        )


class ParserValidationError(ParserError):
    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, object] | None = None,
        request_id: str | None = None,
        observation_id: str | None = None,
        segment_id: str | None = None,
        token_id: str | None = None,
        candidate_id: str | None = None,
    ) -> None:
        super().__init__(
            message,
            code=PARSER_VALIDATION_ERROR,
            context=context,
            request_id=request_id,
            observation_id=observation_id,
            segment_id=segment_id,
            token_id=token_id,
            candidate_id=candidate_id,
        )


class ParserLimitExceededError(ParserError):
    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, object] | None = None,
        request_id: str | None = None,
        observation_id: str | None = None,
        segment_id: str | None = None,
        token_id: str | None = None,
        candidate_id: str | None = None,
    ) -> None:
        super().__init__(
            message,
            code=PARSER_LIMIT_EXCEEDED,
            context=context,
            request_id=request_id,
            observation_id=observation_id,
            segment_id=segment_id,
            token_id=token_id,
            candidate_id=candidate_id,
        )


class ParserDispatchError(ParserError):
    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, object] | None = None,
        request_id: str | None = None,
        observation_id: str | None = None,
        segment_id: str | None = None,
        token_id: str | None = None,
        candidate_id: str | None = None,
    ) -> None:
        super().__init__(
            message,
            code=PARSER_DISPATCH_ERROR,
            context=context,
            request_id=request_id,
            observation_id=observation_id,
            segment_id=segment_id,
            token_id=token_id,
            candidate_id=candidate_id,
        )


class ParserAuditError(ParserError):
    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, object] | None = None,
        request_id: str | None = None,
        observation_id: str | None = None,
        segment_id: str | None = None,
        token_id: str | None = None,
        candidate_id: str | None = None,
    ) -> None:
        super().__init__(
            message,
            code=PARSER_AUDIT_ERROR,
            context=context,
            request_id=request_id,
            observation_id=observation_id,
            segment_id=segment_id,
            token_id=token_id,
            candidate_id=candidate_id,
        )


class ParserBoundaryError(ParserError):
    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, object] | None = None,
        request_id: str | None = None,
        observation_id: str | None = None,
        segment_id: str | None = None,
        token_id: str | None = None,
        candidate_id: str | None = None,
    ) -> None:
        super().__init__(
            message,
            code=PARSER_BOUNDARY_ERROR,
            context=context,
            request_id=request_id,
            observation_id=observation_id,
            segment_id=segment_id,
            token_id=token_id,
            candidate_id=candidate_id,
        )


def _ensure_non_empty_string(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _build_error_context(
    context: Mapping[str, object] | None,
    *,
    request_id: str | None = None,
    observation_id: str | None = None,
    segment_id: str | None = None,
    token_id: str | None = None,
    candidate_id: str | None = None,
) -> Mapping[str, object]:
    if context is None:
        copied_context: dict[str, object] = {}
    else:
        if not isinstance(context, Mapping):
            raise TypeError("context must be a mapping")
        copied_context = {}
        for key, value in context.items():
            if not isinstance(key, str):
                raise TypeError("context keys must be strings")
            copied_context[key] = value

    explicit_context = {
        "request_id": request_id,
        "observation_id": observation_id,
        "segment_id": segment_id,
        "token_id": token_id,
        "candidate_id": candidate_id,
    }
    for key, value in explicit_context.items():
        if value is not None:
            copied_context[key] = _ensure_non_empty_string(value, field_name=key)

    return MappingProxyType(copied_context)


__all__ = [
    "EmptyObservationError",
    "LocalDifferenceExtractionError",
    "PARSER_AUDIT_ERROR",
    "PARSER_BOUNDARY_ERROR",
    "PARSER_DISPATCH_ERROR",
    "PARSER_EMPTY_OBSERVATION",
    "PARSER_ERROR",
    "PARSER_LIMIT_EXCEEDED",
    "PARSER_LOCAL_DIFFERENCE_EXTRACTION_ERROR",
    "PARSER_REQUEST_ERROR",
    "PARSER_SCHEMA_ERROR",
    "PARSER_SEGMENTATION_ERROR",
    "PARSER_TOKENIZATION_ERROR",
    "PARSER_UNSUPPORTED_OBSERVATION",
    "PARSER_VALIDATION_ERROR",
    "ParserAuditError",
    "ParserBoundaryError",
    "ParserDispatchError",
    "ParserError",
    "ParserLimitExceededError",
    "ParserRequestError",
    "ParserSchemaError",
    "ParserValidationError",
    "SegmentationError",
    "TokenizationError",
    "UnsupportedObservationError",
]
