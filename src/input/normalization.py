"""CEBA Input normalization utilities.

Normalization makes observations technically comparable by stabilizing text,
timestamps, MIME values, encodings, language hints, and metadata. It does not
compute meaning.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any
import mimetypes
import unicodedata

from pydantic import ValidationError

from ois.input.errors import NormalizationError, SchemaViolationError
from ois.input.schema import ObservationMetadata, RawObservation


_DEFAULT_ENCODING = "utf-8"
_DEFAULT_TEXT_MIME = "text/plain"
_MAX_TEXT_LENGTH = 5_000_000
_MAX_METADATA_KEYS = 128
_MAX_METADATA_KEY_LENGTH = 128
_MAX_METADATA_STRING_VALUE_LENGTH = 4096

_ALLOWED_METADATA_VALUE_TYPES = (str, int, float, bool, type(None))


def normalize_text(value: str) -> str:
    if not isinstance(value, str):
        raise NormalizationError(
            "text value must be a string",
            field_path="text",
            details={"type": type(value).__name__},
        )
    normalized = unicodedata.normalize("NFC", value)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(normalized) > _MAX_TEXT_LENGTH:
        raise NormalizationError(
            "text exceeds maximum length",
            field_path="text",
            details={"max_length": _MAX_TEXT_LENGTH, "length": len(normalized)},
        )
    return normalized


def normalize_optional_text(value: str | None, *, field_path: str) -> str | None:
    if value is None:
        return None
    try:
        return normalize_text(value)
    except NormalizationError as error:
        raise NormalizationError(
            error.payload.message,
            field_path=field_path,
            details=error.details,
        ) from error


def normalize_datetime(value: datetime, *, field_path: str) -> datetime:
    if not isinstance(value, datetime):
        raise NormalizationError(
            "datetime value is required",
            field_path=field_path,
            details={"type": type(value).__name__},
        )
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise NormalizationError(
            "datetime must be timezone-aware",
            field_path=field_path,
        )
    return value.astimezone(timezone.utc)


def normalize_optional_datetime(
    value: datetime | None,
    *,
    field_path: str,
) -> datetime | None:
    if value is None:
        return None
    return normalize_datetime(value, field_path=field_path)


def normalize_mime_type(
    value: str | None,
    *,
    filename: str | None = None,
) -> str | None:
    if value is not None:
        if not isinstance(value, str):
            raise NormalizationError(
                "mime_type must be a string",
                field_path="mime_type",
                details={"type": type(value).__name__},
            )
        normalized = value.strip().lower()
        if not normalized:
            return None
        return normalized.split(";", 1)[0].strip() or None
    if filename is not None:
        guessed = mimetypes.guess_type(filename)[0]
        if guessed:
            return guessed.lower()
    return None


def normalize_encoding(value: str | None) -> str:
    if value is None:
        return _DEFAULT_ENCODING
    if not isinstance(value, str):
        raise NormalizationError(
            "encoding must be a string",
            field_path="encoding",
            details={"type": type(value).__name__},
        )
    normalized = value.strip().lower().replace("_", "-")
    if not normalized:
        return _DEFAULT_ENCODING
    aliases = {
        "utf8": "utf-8",
        "utf-16le": "utf-16-le",
        "utf-16be": "utf-16-be",
        "latin1": "latin-1",
    }
    return aliases.get(normalized, normalized)


def normalize_language_hint(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise NormalizationError(
            "language_hint must be a string",
            field_path="language_hint",
            details={"type": type(value).__name__},
        )
    normalized = value.strip().lower().replace("_", "-")
    if not normalized:
        return None
    if len(normalized) > 32:
        raise NormalizationError(
            "language_hint exceeds maximum length",
            field_path="language_hint",
            details={"max_length": 32, "length": len(normalized)},
        )
    if _contains_disallowed_control_chars(normalized):
        raise NormalizationError(
            "language_hint contains disallowed control characters",
            field_path="language_hint",
        )
    return normalized


def normalize_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    field_path: str = "metadata",
) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise NormalizationError(
            "metadata must be a mapping",
            field_path=field_path,
            details={"type": type(metadata).__name__},
        )
    if len(metadata) > _MAX_METADATA_KEYS:
        raise NormalizationError(
            "metadata exceeds maximum key count",
            field_path=field_path,
            details={"max_keys": _MAX_METADATA_KEYS, "key_count": len(metadata)},
        )

    normalized_metadata: dict[str, Any] = {}
    for raw_key, raw_value in metadata.items():
        if not isinstance(raw_key, str):
            raise NormalizationError(
                "metadata keys must be strings",
                field_path=field_path,
                details={"key_type": type(raw_key).__name__},
            )
        key = raw_key.strip()
        if not key:
            raise NormalizationError(
                "metadata key must not be empty",
                field_path=field_path,
            )
        if len(key) > _MAX_METADATA_KEY_LENGTH:
            raise NormalizationError(
                "metadata key exceeds maximum length",
                field_path=f"{field_path}.{key}",
                details={"max_length": _MAX_METADATA_KEY_LENGTH, "length": len(key)},
            )
        if _contains_disallowed_control_chars(key):
            raise NormalizationError(
                "metadata key contains disallowed control characters",
                field_path=f"{field_path}.{key}",
            )
        normalized_metadata[key] = _normalize_metadata_value(
            raw_value,
            field_path=f"{field_path}.{key}",
        )
    return normalized_metadata


def build_observation_metadata(
    *,
    mime_type: str | None = None,
    encoding: str | None = None,
    language_hint: str | None = None,
    length_chars: int | None = None,
    size_bytes: int | None = None,
    source_metadata: Mapping[str, Any] | None = None,
    transport_metadata: Mapping[str, Any] | None = None,
) -> ObservationMetadata:
    try:
        return ObservationMetadata(
            mime_type=normalize_mime_type(mime_type),
            encoding=normalize_encoding(encoding),
            language_hint=normalize_language_hint(language_hint),
            length_chars=_validate_non_negative_int(
                length_chars,
                field_path="length_chars",
            ),
            size_bytes=_validate_non_negative_int(
                size_bytes,
                field_path="size_bytes",
            ),
            source_metadata=normalize_metadata(
                source_metadata,
                field_path="source_metadata",
            ),
            transport_metadata=normalize_metadata(
                transport_metadata,
                field_path="transport_metadata",
            ),
        )
    except ValidationError as error:
        raise NormalizationError(
            "failed to build observation metadata",
            details={"errors": error.errors()},
        ) from error


def normalize_raw_observation(raw: RawObservation) -> RawObservation:
    if not isinstance(raw, RawObservation):
        raise NormalizationError(
            "raw must be RawObservation",
            field_path="raw",
            details={"type": type(raw).__name__},
        )
    try:
        return raw.model_copy(
            update={
                "received_at": normalize_datetime(
                    raw.received_at,
                    field_path="received_at",
                ),
                "observed_at": normalize_optional_datetime(
                    raw.observed_at,
                    field_path="observed_at",
                ),
                "raw_mime_type": normalize_mime_type(raw.raw_mime_type),
                "raw_encoding": normalize_encoding(raw.raw_encoding),
                "source_metadata": normalize_metadata(
                    raw.source_metadata,
                    field_path="source_metadata",
                ),
                "transport_metadata": normalize_metadata(
                    raw.transport_metadata,
                    field_path="transport_metadata",
                ),
            }
        )
    except ValidationError as error:
        raise NormalizationError(
            "failed to normalize raw observation",
            details={"errors": error.errors()},
        ) from error


def _contains_disallowed_control_chars(
    value: str,
    *,
    allow_multiline: bool = False,
) -> bool:
    allowed_multiline_chars = {"\t", "\n", "\r"} if allow_multiline else set()
    return any(
        (ord(char) < 32 or ord(char) == 127) and char not in allowed_multiline_chars
        for char in value
    )


def _normalize_metadata_value(value: Any, *, field_path: str) -> Any:
    if not isinstance(value, _ALLOWED_METADATA_VALUE_TYPES):
        raise NormalizationError(
            "metadata value has unsupported type",
            field_path=field_path,
            details={"value_type": type(value).__name__},
        )
    if isinstance(value, str):
        normalized = unicodedata.normalize("NFC", value).strip()
        if len(normalized) > _MAX_METADATA_STRING_VALUE_LENGTH:
            raise NormalizationError(
                "metadata string value exceeds maximum length",
                field_path=field_path,
                details={
                    "max_length": _MAX_METADATA_STRING_VALUE_LENGTH,
                    "length": len(normalized),
                },
            )
        if _contains_disallowed_control_chars(normalized, allow_multiline=True):
            raise NormalizationError(
                "metadata string value contains disallowed control characters",
                field_path=field_path,
            )
        return normalized
    return value


def _validate_non_negative_int(value: int | None, *, field_path: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaViolationError(
            "value must be an integer",
            field_path=field_path,
            details={"type": type(value).__name__},
        )
    if value < 0:
        raise NormalizationError(
            "value must be greater than or equal to 0",
            field_path=field_path,
        )
    return value


__all__ = [
    "build_observation_metadata",
    "normalize_datetime",
    "normalize_encoding",
    "normalize_language_hint",
    "normalize_metadata",
    "normalize_mime_type",
    "normalize_optional_datetime",
    "normalize_optional_text",
    "normalize_raw_observation",
    "normalize_text",
]
