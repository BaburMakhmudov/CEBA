"""Deterministic JSONB serialization utilities for CEBA PostgreSQL persistence."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from ceba.persistence.postgres.connection import (
    ensure_non_empty_string,
    ensure_non_negative_int,
)


__all__ = [
    "JsonbPayloadKind",
    "JsonbValidationMode",
    "JsonbPayload",
    "JsonbValidationIssue",
    "JsonbValidationReport",
    "JsonbNormalizer",
    "validate_jsonb_value",
    "normalize_jsonb_value",
    "create_jsonb_payload",
]


_FORBIDDEN_RUNTIME_STATE_KEYS = {
    "M_t",
    "Mt",
    "semantic_state",
    "runtime_semantic_state",
}

_VALID_SEVERITIES = {"INFO", "WARNING", "ERROR"}


class JsonbPayloadKind(Enum):
    """Stable JSONB payload categories."""

    LEDGER_PAYLOAD = "LEDGER_PAYLOAD"
    LEDGER_METADATA = "LEDGER_METADATA"
    LEDGER_PROVENANCE = "LEDGER_PROVENANCE"
    EVENT_PAYLOAD = "EVENT_PAYLOAD"
    AUDIT_PAYLOAD = "AUDIT_PAYLOAD"
    METRIC_PAYLOAD = "METRIC_PAYLOAD"
    OUTBOX_PAYLOAD = "OUTBOX_PAYLOAD"
    UNKNOWN = "UNKNOWN"


class JsonbValidationMode(Enum):
    """Stable JSONB validation modes."""

    STRICT = "STRICT"
    PERMISSIVE = "PERMISSIVE"


@dataclass(frozen=True)
class JsonbPayload:
    """Immutable descriptor for a JSONB-ready payload."""

    kind: JsonbPayloadKind
    schema_version: int
    value: dict[str, object]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, JsonbPayloadKind):
            raise TypeError("kind must be JsonbPayloadKind")
        schema_version = ensure_non_negative_int(
            self.schema_version,
            "schema_version",
        )
        value = _normalize_dict(self.value, "value")
        metadata = _normalize_dict(self.metadata, "metadata")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "kind": self.kind.value,
            "schema_version": self.schema_version,
            "value": dict(self.value),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class JsonbValidationIssue:
    """Immutable JSONB validation issue descriptor."""

    path: str
    code: str
    message: str
    severity: str

    def __post_init__(self) -> None:
        path = ensure_non_empty_string(self.path, "path")
        code = ensure_non_empty_string(self.code, "code")
        message = ensure_non_empty_string(self.message, "message")
        severity = ensure_non_empty_string(self.severity, "severity")
        if severity not in _VALID_SEVERITIES:
            raise ValueError("severity must be INFO, WARNING, or ERROR")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "severity", severity)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "path": self.path,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class JsonbValidationReport:
    """Immutable validation report for JSONB values."""

    valid: bool
    issues: tuple[JsonbValidationIssue, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be bool")
        issues = _ensure_issue_tuple(self.issues)
        has_error = any(issue.severity == "ERROR" for issue in issues)
        if has_error and self.valid:
            raise ValueError("valid must be False when errors are present")
        object.__setattr__(self, "issues", issues)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "valid": self.valid,
            "issues": tuple(issue.to_dict() for issue in self.issues),
        }

    def error_count(self) -> int:
        """Return error issue count."""

        return _count_severity(self.issues, "ERROR")

    def warning_count(self) -> int:
        """Return warning issue count."""

        return _count_severity(self.issues, "WARNING")

    def info_count(self) -> int:
        """Return info issue count."""

        return _count_severity(self.issues, "INFO")

    def has_errors(self) -> bool:
        """Return whether the report contains errors."""

        return self.error_count() > 0


class JsonbNormalizer:
    """Deterministic JSONB validator and normalizer."""

    def __init__(
        self,
        *,
        mode: JsonbValidationMode = JsonbValidationMode.STRICT,
    ) -> None:
        if not isinstance(mode, JsonbValidationMode):
            raise TypeError("mode must be JsonbValidationMode")
        self._mode = mode

    def validate(
        self,
        *,
        value: object,
    ) -> JsonbValidationReport:
        """Validate a value recursively for JSONB compatibility."""

        issues: list[JsonbValidationIssue] = []
        _validate_value(
            value=value,
            path="$",
            mode=self._mode,
            issues=issues,
        )
        issues_tuple = tuple(issues)
        return JsonbValidationReport(
            valid=not any(issue.severity == "ERROR" for issue in issues_tuple),
            issues=issues_tuple,
        )

    def normalize(
        self,
        *,
        value: object,
    ) -> object:
        """Return a JSONB-normalized copy of a valid value."""

        report = self.validate(value=value)
        if report.has_errors():
            raise ValueError("value is not JSONB-safe")
        return _normalize_value(value)

    def create_payload(
        self,
        *,
        kind: JsonbPayloadKind = JsonbPayloadKind.UNKNOWN,
        schema_version: int = 1,
        value: dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> JsonbPayload:
        """Create an immutable JSONB payload descriptor."""

        if not isinstance(kind, JsonbPayloadKind):
            raise TypeError("kind must be JsonbPayloadKind")
        normalized_value = {}
        if value is not None:
            normalized = self.normalize(value=value)
            if not isinstance(normalized, dict):
                raise TypeError("value must normalize to dict[str, object]")
            normalized_value = normalized
        required_metadata = {
            "source": "jsonb_normalizer",
            "payload_kind": kind.value,
            "schema_version": schema_version,
            "validation_mode": self._mode.value,
        }
        normalized_metadata = self.normalize(
            value=_merge_metadata(required_metadata, metadata),
        )
        if not isinstance(normalized_metadata, dict):
            raise TypeError("metadata must normalize to dict[str, object]")
        return JsonbPayload(
            kind=kind,
            schema_version=schema_version,
            value=normalized_value,
            metadata=normalized_metadata,
        )


def validate_jsonb_value(
    *,
    value: object,
    mode: JsonbValidationMode = JsonbValidationMode.STRICT,
) -> JsonbValidationReport:
    """Validate a value recursively for JSONB compatibility."""

    return JsonbNormalizer(mode=mode).validate(value=value)


def normalize_jsonb_value(
    *,
    value: object,
    mode: JsonbValidationMode = JsonbValidationMode.STRICT,
) -> object:
    """Return a JSONB-normalized copy of a valid value."""

    return JsonbNormalizer(mode=mode).normalize(value=value)


def create_jsonb_payload(
    *,
    kind: JsonbPayloadKind = JsonbPayloadKind.UNKNOWN,
    schema_version: int = 1,
    value: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
    mode: JsonbValidationMode = JsonbValidationMode.STRICT,
) -> JsonbPayload:
    """Create an immutable JSONB payload descriptor."""

    return JsonbNormalizer(mode=mode).create_payload(
        kind=kind,
        schema_version=schema_version,
        value=value,
        metadata=metadata,
    )


def _validate_value(
    *,
    value: object,
    path: str,
    mode: JsonbValidationMode,
    issues: list[JsonbValidationIssue],
) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            issues.append(
                _issue(
                    path=path,
                    code="NON_FINITE_FLOAT",
                    message="JSONB value contains non-finite float",
                    severity="ERROR",
                )
            )
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            if not isinstance(key, str):
                issues.append(
                    _issue(
                        path=path,
                        code="NON_STRING_KEY",
                        message="JSONB object contains non-string key",
                        severity="ERROR",
                    )
                )
                continue
            child_path = _child_path(path, key)
            if key in _FORBIDDEN_RUNTIME_STATE_KEYS:
                issues.append(
                    _issue(
                        path=child_path,
                        code="FORBIDDEN_RUNTIME_STATE_KEY",
                        message="JSONB value must not persist runtime semantic state",
                        severity="ERROR",
                    )
                )
            _validate_value(value=nested, path=child_path, mode=mode, issues=issues)
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _validate_value(
                value=nested,
                path=f"{path}[{index}]",
                mode=mode,
                issues=issues,
            )
        return
    if isinstance(value, tuple):
        if mode == JsonbValidationMode.PERMISSIVE:
            issues.append(
                _issue(
                    path=path,
                    code="TUPLE_NORMALIZES_TO_LIST",
                    message="tuple will normalize to list",
                    severity="WARNING",
                )
            )
        for index, nested in enumerate(value):
            _validate_value(
                value=nested,
                path=f"{path}[{index}]",
                mode=mode,
                issues=issues,
            )
        return
    issues.append(
        _issue(
            path=path,
            code="UNSUPPORTED_TYPE",
            message="JSONB value contains unsupported type",
            severity="ERROR",
        )
    )


def _normalize_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, dict):
        normalized: dict[str, object] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError("JSONB object keys must be strings")
            if key in _FORBIDDEN_RUNTIME_STATE_KEYS:
                raise ValueError("JSONB value must not persist runtime semantic state")
            normalized[key] = _normalize_value(nested)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_value(nested) for nested in value]
    raise TypeError("unsupported JSONB value type")


def _normalize_dict(value: dict[str, object], field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be dict[str, object]")
    report = validate_jsonb_value(value=value)
    if report.has_errors():
        raise ValueError(f"{field_name} is not JSONB-safe")
    normalized = _normalize_value(value)
    if not isinstance(normalized, dict):
        raise TypeError(f"{field_name} must normalize to dict[str, object]")
    return normalized


def _merge_metadata(
    default_metadata: dict[str, object],
    metadata: dict[str, object] | None,
) -> dict[str, object]:
    merged = _copy_dict(default_metadata)
    if metadata is not None:
        merged.update(_copy_dict(metadata))
    if "source" in default_metadata:
        merged["source"] = default_metadata["source"]
    return merged


def _copy_dict(value: dict[str, object]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("value must be dict[str, object]")
    copied = dict(value)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("dict keys must be strings")
    return copied


def _ensure_issue_tuple(
    issues: tuple[JsonbValidationIssue, ...] | list[JsonbValidationIssue],
) -> tuple[JsonbValidationIssue, ...]:
    if not isinstance(issues, (tuple, list)):
        raise TypeError("issues must be tuple[JsonbValidationIssue, ...]")
    issues_tuple = tuple(issues)
    for issue in issues_tuple:
        if not isinstance(issue, JsonbValidationIssue):
            raise TypeError("every issue must be JsonbValidationIssue")
    return issues_tuple


def _issue(
    *,
    path: str,
    code: str,
    message: str,
    severity: str,
) -> JsonbValidationIssue:
    return JsonbValidationIssue(
        path=path,
        code=code,
        message=message,
        severity=severity,
    )


def _child_path(path: str, key: str) -> str:
    return f"{path}.{key}"


def _count_severity(
    issues: tuple[JsonbValidationIssue, ...],
    severity: str,
) -> int:
    return sum(1 for issue in issues if issue.severity == severity)
