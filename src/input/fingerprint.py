"""CEBA Input fingerprint utilities.

Fingerprints provide deterministic hashes that make input observations
reproducible and auditable. They serialize and hash technical input structures
without computing meaning.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from ois.input.errors import HashMismatchError, SchemaViolationError
from ois.input.schema import CanonicalObservation, InputEnvelope, RawObservation


_HASH_PREFIX = "sha256:"
_STRUCTURAL_STRING = "<str>"
_STRUCTURAL_INT = "<int>"
_STRUCTURAL_FLOAT = "<float>"
_STRUCTURAL_BOOL = "<bool>"
_STRUCTURAL_NULL = "<null>"
_STRUCTURAL_UNKNOWN = "<unknown>"


def canonical_json_dumps(value: Any) -> str:
    return json.dumps(
        _to_json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_digest(value: str | bytes) -> str:
    if isinstance(value, str):
        payload = value.encode("utf-8")
    elif isinstance(value, bytes):
        payload = value
    else:
        raise SchemaViolationError(
            "sha256_digest value must be str or bytes",
            field_path="value",
            details={"type": type(value).__name__},
        )
    return f"{_HASH_PREFIX}{hashlib.sha256(payload).hexdigest()}"


def compute_content_hash(value: Any) -> str:
    return sha256_digest(canonical_json_dumps(value))


def compute_structural_hash(value: Any) -> str:
    return sha256_digest(canonical_json_dumps(_to_structural_form(value)))


def compute_raw_payload_hash(raw: RawObservation) -> str:
    return compute_content_hash(raw.raw_payload)


def compute_raw_observation_hash(raw: RawObservation) -> str:
    return compute_content_hash(raw.model_dump(mode="json"))


def compute_envelope_hash(envelope: InputEnvelope) -> str:
    return compute_content_hash(envelope.model_dump(mode="json"))


def compute_canonical_content_hash(observation: CanonicalObservation) -> str:
    return compute_content_hash(observation.content.model_dump(mode="json"))


def compute_canonical_observation_structural_hash(
    observation: CanonicalObservation,
) -> str:
    payload = observation.model_dump(
        mode="json",
        exclude={"content_hash", "structural_hash"},
    )
    return compute_structural_hash(payload)


def is_sha256_hash(value: str) -> bool:
    if not isinstance(value, str):
        return False
    if not value.startswith(_HASH_PREFIX):
        return False
    digest = value[len(_HASH_PREFIX) :]
    return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)


def verify_hash(
    *,
    expected: str,
    actual: str,
    field_path: str,
) -> None:
    if not is_sha256_hash(expected):
        raise HashMismatchError(
            "expected hash is not a valid sha256 hash",
            field_path=field_path,
            details={"expected": expected},
        )
    if not is_sha256_hash(actual):
        raise HashMismatchError(
            "actual hash is not a valid sha256 hash",
            field_path=field_path,
            details={"actual": actual},
        )
    if expected != actual:
        raise HashMismatchError(
            "hash mismatch",
            field_path=field_path,
            details={"expected": expected, "actual": actual},
        )


def verify_content_hash(
    *,
    value: Any,
    expected_hash: str,
    field_path: str = "content_hash",
) -> None:
    verify_hash(
        expected=expected_hash,
        actual=compute_content_hash(value),
        field_path=field_path,
    )


def verify_structural_hash(
    *,
    value: Any,
    expected_hash: str,
    field_path: str = "structural_hash",
) -> None:
    verify_hash(
        expected=expected_hash,
        actual=compute_structural_hash(value),
        field_path=field_path,
    )


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return _ensure_timezone_aware(value).isoformat()
    if isinstance(value, Mapping):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise SchemaViolationError(
                "non-finite float values are not JSON-safe",
                field_path="value",
                details={"value": str(value)},
            )
        return value
    if isinstance(value, str) or value is None:
        return value
    if isinstance(value, (bytes, bytearray)):
        raise SchemaViolationError(
            "bytes are not valid JSON-safe input",
            field_path="value",
            details={"type": type(value).__name__},
        )
    raise SchemaViolationError(
        "unsupported value for JSON serialization",
        field_path="value",
        details={"type": type(value).__name__},
    )


def _to_structural_form(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _to_structural_form(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {
            str(key): _to_structural_form(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_structural_form(item) for item in value]
    if isinstance(value, Enum):
        return {"enum": value.__class__.__name__, "value": _STRUCTURAL_STRING}
    if isinstance(value, str):
        return _STRUCTURAL_STRING
    if isinstance(value, bool):
        return _STRUCTURAL_BOOL
    if isinstance(value, int):
        return _STRUCTURAL_INT
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise SchemaViolationError(
                "non-finite float values cannot be structurally hashed",
                field_path="value",
                details={"value": str(value)},
            )
        return _STRUCTURAL_FLOAT
    if value is None:
        return _STRUCTURAL_NULL
    if isinstance(value, UUID):
        return "<uuid>"
    if isinstance(value, datetime):
        _ensure_timezone_aware(value)
        return "<datetime>"
    if isinstance(value, (bytes, bytearray)):
        raise SchemaViolationError(
            "bytes cannot be structurally hashed",
            field_path="value",
            details={"type": type(value).__name__},
        )
    _to_json_safe(value)
    return _STRUCTURAL_UNKNOWN


def _ensure_timezone_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise SchemaViolationError(
            "datetime must be timezone-aware",
            field_path="value",
        )
    return value


__all__ = [
    "canonical_json_dumps",
    "compute_canonical_content_hash",
    "compute_canonical_observation_structural_hash",
    "compute_content_hash",
    "compute_envelope_hash",
    "compute_raw_observation_hash",
    "compute_raw_payload_hash",
    "compute_structural_hash",
    "is_sha256_hash",
    "sha256_digest",
    "verify_content_hash",
    "verify_hash",
    "verify_structural_hash",
]
