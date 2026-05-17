"""CEBA Input security utilities.

Security checks protect the technical input boundary with deterministic size,
MIME, source, path, redaction, and risk-label checks. They do not compute
meaning.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
import re

from ois.input.errors import SecurityPolicyViolation, SchemaViolationError
from ois.input.schema import (
    InputSource,
    PayloadRef,
    PayloadStorageKind,
    RawObservation,
    SecurityReport,
    SecurityStatus,
)
from ois.input.source import (
    source_allows_external_payload,
    source_allows_inline_payload,
    validate_source_boundary,
)


_DEFAULT_MAX_INLINE_PAYLOAD_BYTES = 5_000_000
_DEFAULT_MAX_METADATA_STRING_LENGTH = 4096
_MAX_SECURITY_LABELS = 64

_ALLOWED_TEXT_MIME_PREFIXES = frozenset({"text/"})
_ALLOWED_EXACT_MIME_TYPES = frozenset(
    {
        "application/json",
        "application/xml",
        "application/pdf",
        "application/octet-stream",
    }
)
_DENIED_MIME_TYPES = frozenset(
    {
        "application/x-msdownload",
        "application/x-dosexec",
        "application/x-sh",
        "application/x-bat",
        "application/x-msdos-program",
    }
)

_PATH_TRAVERSAL_PATTERN = re.compile(r"(^|[\\/])\.\.([\\/]|$)")
_SECRET_PATTERNS = {
    "api_key": re.compile(r"(?i)(api[_-]?key)\s*[:=]\s*([A-Za-z0-9_\-]{16,})"),
    "bearer_token": re.compile(r"(?i)(bearer)\s+([A-Za-z0-9._\-]{20,})"),
    "password": re.compile(r"(?i)(password|passwd|pwd)\s*[:=]\s*([^\s,;]{6,})"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}


@dataclass(frozen=True, slots=True)
class InputSecurityPolicy:
    max_inline_payload_bytes: int = _DEFAULT_MAX_INLINE_PAYLOAD_BYTES
    max_metadata_string_length: int = _DEFAULT_MAX_METADATA_STRING_LENGTH
    allowed_text_mime_prefixes: frozenset[str] = field(
        default_factory=lambda: _ALLOWED_TEXT_MIME_PREFIXES
    )
    allowed_exact_mime_types: frozenset[str] = field(
        default_factory=lambda: _ALLOWED_EXACT_MIME_TYPES
    )
    denied_mime_types: frozenset[str] = field(default_factory=lambda: _DENIED_MIME_TYPES)
    redact_secrets: bool = True
    block_denied_mime_types: bool = True
    block_path_traversal: bool = True


def passed_security_report(labels: list[str] | None = None) -> SecurityReport:
    return SecurityReport(
        status=SecurityStatus.PASSED,
        labels=deduplicate_labels(labels or []),
    )


def warning_security_report(labels: list[str]) -> SecurityReport:
    return SecurityReport(
        status=SecurityStatus.WARNING,
        labels=deduplicate_labels(labels),
    )


def blocked_security_report(
    *,
    reason: str,
    labels: list[str] | None = None,
) -> SecurityReport:
    if not isinstance(reason, str) or not reason.strip():
        raise SchemaViolationError(
            "blocked security report reason must be a non-empty string",
            field_path="blocked_reason",
            details={"type": type(reason).__name__},
        )
    return SecurityReport(
        status=SecurityStatus.BLOCKED,
        labels=deduplicate_labels(labels or []),
        blocked_reason=reason.strip(),
    )


def redacted_security_report(
    *,
    labels: list[str],
    redacted_fields: list[str],
) -> SecurityReport:
    return SecurityReport(
        status=SecurityStatus.REDACTED,
        labels=deduplicate_labels(labels),
        redacted_fields=deduplicate_labels(redacted_fields),
    )


def is_mime_allowed(
    mime_type: str | None,
    *,
    policy: InputSecurityPolicy | None = None,
) -> bool:
    active_policy = policy or InputSecurityPolicy()
    normalized = _normalize_mime_type(mime_type)
    if normalized is None:
        return True
    if normalized in active_policy.denied_mime_types:
        return False
    if normalized in active_policy.allowed_exact_mime_types:
        return True
    return any(
        normalized.startswith(prefix)
        for prefix in active_policy.allowed_text_mime_prefixes
    )


def validate_mime_type(
    mime_type: str | None,
    *,
    policy: InputSecurityPolicy | None = None,
    field_path: str = "mime_type",
) -> None:
    active_policy = policy or InputSecurityPolicy()
    normalized = _normalize_mime_type(mime_type)
    if normalized is None:
        return
    if normalized in active_policy.denied_mime_types:
        if active_policy.block_denied_mime_types:
            raise SecurityPolicyViolation(
                "denied MIME type blocked",
                field_path=field_path,
                details={"mime_type": normalized},
            )
        return
    if not is_mime_allowed(normalized, policy=active_policy):
        raise SecurityPolicyViolation(
            "MIME type is not allowed",
            field_path=field_path,
            details={"mime_type": normalized},
        )


def validate_payload_size(
    *,
    size_bytes: int | None,
    policy: InputSecurityPolicy | None = None,
    field_path: str = "size_bytes",
) -> None:
    if size_bytes is None:
        return
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int):
        raise SchemaViolationError(
            "size_bytes must be an integer",
            field_path=field_path,
            details={"type": type(size_bytes).__name__},
        )
    if size_bytes < 0:
        raise SchemaViolationError(
            "size_bytes must be greater than or equal to 0",
            field_path=field_path,
        )
    active_policy = policy or InputSecurityPolicy()
    if size_bytes > active_policy.max_inline_payload_bytes:
        raise SecurityPolicyViolation(
            "payload exceeds maximum allowed size",
            field_path=field_path,
            details={
                "size_bytes": size_bytes,
                "max_inline_payload_bytes": active_policy.max_inline_payload_bytes,
            },
        )


def contains_path_traversal(value: str) -> bool:
    return bool(_PATH_TRAVERSAL_PATTERN.search(value)) or "%2e%2e" in value.lower()


def validate_no_path_traversal(
    value: str | None,
    *,
    policy: InputSecurityPolicy | None = None,
    field_path: str,
) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise SchemaViolationError(
            "path value must be a string",
            field_path=field_path,
            details={"type": type(value).__name__},
        )
    active_policy = policy or InputSecurityPolicy()
    if active_policy.block_path_traversal and contains_path_traversal(value):
        raise SecurityPolicyViolation(
            "path traversal pattern blocked",
            field_path=field_path,
        )


def detect_secret_labels(value: str) -> list[str]:
    labels: list[str] = []
    if _SECRET_PATTERNS["api_key"].search(value):
        labels.append("contains_api_key")
    if _SECRET_PATTERNS["bearer_token"].search(value):
        labels.append("contains_bearer_token")
    if _SECRET_PATTERNS["password"].search(value):
        labels.append("contains_password")
    if _SECRET_PATTERNS["private_key"].search(value):
        labels.append("contains_private_key")
    return labels


def redact_secrets(value: str) -> tuple[str, list[str]]:
    labels = detect_secret_labels(value)
    redacted = value
    redacted = _SECRET_PATTERNS["api_key"].sub(r"\1=[REDACTED]", redacted)
    redacted = _SECRET_PATTERNS["bearer_token"].sub(r"\1 [REDACTED]", redacted)
    redacted = _SECRET_PATTERNS["password"].sub(r"\1=[REDACTED]", redacted)
    redacted = _SECRET_PATTERNS["private_key"].sub("[REDACTED]", redacted)
    return redacted, labels


def redact_metadata_secrets(
    metadata: Mapping[str, Any],
    *,
    field_path: str,
    policy: InputSecurityPolicy | None = None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    active_policy = policy or InputSecurityPolicy()
    copied = dict(metadata)
    if not active_policy.redact_secrets:
        return copied, [], []

    labels: list[str] = []
    redacted_fields: list[str] = []
    redacted_metadata: dict[str, Any] = {}
    for key, value in copied.items():
        if isinstance(value, str):
            redacted_value, value_labels = redact_secrets(value)
            redacted_metadata[key] = redacted_value
            if value_labels:
                labels.extend(value_labels)
                redacted_fields.append(f"{field_path}.{key}")
        else:
            redacted_metadata[key] = value
    return redacted_metadata, deduplicate_labels(labels), deduplicate_labels(redacted_fields)


def validate_source_security(
    source: InputSource,
    *,
    uses_external_payload: bool = False,
    uses_inline_payload: bool = False,
) -> None:
    validate_source_boundary(source)
    if uses_external_payload and not source_allows_external_payload(source.source_type):
        raise SecurityPolicyViolation(
            "source type does not allow external payloads",
            field_path="source.source_type",
            details={"source_type": source.source_type.value},
        )
    if uses_inline_payload and not source_allows_inline_payload(source.source_type):
        raise SecurityPolicyViolation(
            "source type does not allow inline payloads",
            field_path="source.source_type",
            details={"source_type": source.source_type.value},
        )


def validate_payload_ref_security(
    payload_ref: PayloadRef,
    *,
    policy: InputSecurityPolicy | None = None,
) -> SecurityReport:
    labels: list[str] = []
    try:
        validate_mime_type(
            payload_ref.mime_type,
            policy=policy,
            field_path="payload_ref.mime_type",
        )
        if payload_ref.storage_kind == PayloadStorageKind.EXTERNAL:
            validate_no_path_traversal(
                payload_ref.uri,
                policy=policy,
                field_path="payload_ref.uri",
            )
            labels.append("external_payload")
        if payload_ref.storage_kind == PayloadStorageKind.REFERENCE:
            validate_no_path_traversal(
                payload_ref.uri,
                policy=policy,
                field_path="payload_ref.uri",
            )
            labels.append("referenced_payload")
        if payload_ref.storage_kind == PayloadStorageKind.INLINE:
            validate_payload_size(
                size_bytes=payload_ref.size_bytes,
                policy=policy,
                field_path="payload_ref.size_bytes",
            )
            labels.append("inline_payload")
        return passed_security_report(labels)
    except SecurityPolicyViolation as error:
        return blocked_security_report(reason=str(error), labels=["security_block"])


def scan_raw_observation_security(
    raw: RawObservation,
    *,
    policy: InputSecurityPolicy | None = None,
) -> tuple[RawObservation, SecurityReport]:
    if not isinstance(raw, RawObservation):
        raise SchemaViolationError(
            "raw must be RawObservation",
            field_path="raw",
            details={"type": type(raw).__name__},
        )

    active_policy = policy or InputSecurityPolicy()
    labels: list[str] = []
    redacted_fields: list[str] = []
    try:
        validate_mime_type(
            raw.raw_mime_type,
            policy=active_policy,
            field_path="raw_mime_type",
        )
        validate_payload_size(
            size_bytes=raw.raw_size_bytes,
            policy=active_policy,
            field_path="raw_size_bytes",
        )
        validate_source_security(
            InputSource(
                source_type=raw.source_type,
                source_name=raw.source_name,
                source_id=None,
                trust_level=raw.trust_level,
                metadata=raw.source_metadata,
            )
        )
    except SecurityPolicyViolation as error:
        return raw, blocked_security_report(reason=str(error), labels=["security_block"])

    source_metadata, source_labels, source_redacted_fields = redact_metadata_secrets(
        raw.source_metadata,
        field_path="source_metadata",
        policy=active_policy,
    )
    transport_metadata, transport_labels, transport_redacted_fields = (
        redact_metadata_secrets(
            raw.transport_metadata,
            field_path="transport_metadata",
            policy=active_policy,
        )
    )
    labels.extend(source_labels)
    labels.extend(transport_labels)
    redacted_fields.extend(source_redacted_fields)
    redacted_fields.extend(transport_redacted_fields)

    raw_payload = raw.raw_payload
    if isinstance(raw_payload, str) and active_policy.redact_secrets:
        redacted_payload, payload_labels = redact_secrets(raw_payload)
        if payload_labels:
            raw_payload = redacted_payload
            labels.extend(payload_labels)
            redacted_fields.append("raw_payload")

    updated_raw = raw.model_copy(
        update={
            "source_metadata": source_metadata,
            "transport_metadata": transport_metadata,
            "raw_payload": raw_payload,
        }
    )
    labels = deduplicate_labels(labels)
    redacted_fields = deduplicate_labels(redacted_fields)

    if redacted_fields:
        return updated_raw, redacted_security_report(
            labels=labels,
            redacted_fields=redacted_fields,
        )
    if labels:
        return updated_raw, warning_security_report(labels)
    return updated_raw, passed_security_report()


def deduplicate_labels(labels: list[str]) -> list[str]:
    deduplicated: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if not label:
            continue
        if label in seen:
            continue
        seen.add(label)
        deduplicated.append(label)
        if len(deduplicated) >= _MAX_SECURITY_LABELS:
            break
    return deduplicated


def _normalize_mime_type(mime_type: str | None) -> str | None:
    if mime_type is None:
        return None
    if not isinstance(mime_type, str):
        raise SchemaViolationError(
            "mime_type must be a string",
            field_path="mime_type",
            details={"type": type(mime_type).__name__},
        )
    normalized = mime_type.strip().lower().split(";", 1)[0].strip()
    return normalized or None


__all__ = [
    "InputSecurityPolicy",
    "blocked_security_report",
    "contains_path_traversal",
    "deduplicate_labels",
    "detect_secret_labels",
    "is_mime_allowed",
    "passed_security_report",
    "redact_metadata_secrets",
    "redact_secrets",
    "redacted_security_report",
    "scan_raw_observation_security",
    "validate_mime_type",
    "validate_no_path_traversal",
    "validate_payload_ref_security",
    "validate_payload_size",
    "validate_source_security",
    "warning_security_report",
]
