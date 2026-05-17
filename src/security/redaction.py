"""Deterministic redaction utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ceba.security.schema import (
    SecurityRiskLevel,
    SecurityValidationReport,
    SecurityValidationStatus,
    create_security_validation_issue,
    create_security_validation_report,
    ensure_non_empty_string,
    ensure_positive_int,
)


__all__ = ["RedactionMode", "RedactionPolicy", "RedactionResult", "Redactor", "redact_value", "build_redactor"]

_FORBIDDEN_STATE = {"M_t", "Mt", "semantic_state", "runtime_semantic_state"}
_FORBIDDEN_SECRET = {"password", "raw_password", "secret", "raw_secret", "api_key", "token", "access_token", "refresh_token", "private_key", "credential"}


class RedactionMode(Enum):
    STRICT = "STRICT"
    PERMISSIVE = "PERMISSIVE"


@dataclass(frozen=True)
class RedactionPolicy:
    mode: RedactionMode = RedactionMode.STRICT
    replacement: str = "[REDACTED]"
    max_depth: int = 16
    sensitive_keys: tuple[str, ...] = tuple(sorted(_FORBIDDEN_SECRET | _FORBIDDEN_STATE))

    def __post_init__(self) -> None:
        if not isinstance(self.mode, RedactionMode):
            raise TypeError("mode must be RedactionMode")
        object.__setattr__(self, "replacement", ensure_non_empty_string(self.replacement, "replacement"))
        object.__setattr__(self, "max_depth", ensure_positive_int(self.max_depth, "max_depth"))
        object.__setattr__(self, "sensitive_keys", _key_tuple(self.sensitive_keys))

    def to_dict(self) -> dict[str, object]:
        return {"mode": self.mode.value, "replacement": self.replacement, "max_depth": self.max_depth, "sensitive_keys": list(self.sensitive_keys)}


@dataclass(frozen=True)
class RedactionResult:
    value: object
    report: SecurityValidationReport
    changed: bool
    reason: str
    redacted_paths: tuple[str, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.report, SecurityValidationReport):
            raise TypeError("report must be SecurityValidationReport")
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        object.__setattr__(self, "reason", ensure_non_empty_string(self.reason, "reason"))
        object.__setattr__(self, "redacted_paths", _key_tuple(self.redacted_paths))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {"value": self.value, "report": self.report.to_dict(), "changed": self.changed, "reason": self.reason, "redacted_paths": list(self.redacted_paths), "metadata": dict(self.metadata)}

    def redacted_count(self) -> int:
        return len(self.redacted_paths)


class Redactor:
    def __init__(self, *, policy: RedactionPolicy | None = None) -> None:
        self._policy = RedactionPolicy() if policy is None else policy
        if not isinstance(self._policy, RedactionPolicy):
            raise TypeError("policy must be RedactionPolicy")

    def redact(self, *, value: object) -> RedactionResult:
        issues = []
        paths: list[str] = []
        try:
            redacted = self._redact(value, "$", 0, paths)
            status = SecurityValidationStatus.WARNING if paths else SecurityValidationStatus.VALID
        except TypeError as exc:
            issue = create_security_validation_issue(code="REDACTION_UNSUPPORTED", message=str(exc), risk_level=SecurityRiskLevel.HIGH)
            issues.append(issue)
            redacted = None
            status = SecurityValidationStatus.BLOCKED
        if paths:
            issues.append(create_security_validation_issue(code="REDACTION_APPLIED", message="sensitive values redacted", risk_level=SecurityRiskLevel.MEDIUM))
        report = create_security_validation_report(status=status, issues=tuple(issues))
        return RedactionResult(redacted, report, bool(paths), "value redacted" if paths else "value did not require redaction", tuple(paths), {"source": "redaction_result"})

    def _redact(self, value: object, path: str, depth: int, paths: list[str]) -> object:
        if depth > self._policy.max_depth:
            if self._policy.mode is RedactionMode.STRICT:
                raise TypeError("redaction max depth exceeded")
            return str(value)
        if isinstance(value, dict):
            copied: dict[str, object] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    if self._policy.mode is RedactionMode.STRICT:
                        raise TypeError("dict keys must be strings")
                    key = str(key)
                next_path = f"{path}.{key}"
                if key in self._policy.sensitive_keys:
                    copied[key] = self._policy.replacement
                    paths.append(next_path)
                else:
                    copied[key] = self._redact(item, next_path, depth + 1, paths)
            return copied
        if isinstance(value, list):
            return [self._redact(item, f"{path}[{index}]", depth + 1, paths) for index, item in enumerate(value)]
        if isinstance(value, tuple):
            return tuple(self._redact(item, f"{path}[{index}]", depth + 1, paths) for index, item in enumerate(value))
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if self._policy.mode is RedactionMode.PERMISSIVE:
            return str(value)
        raise TypeError("unsupported value for redaction")


def redact_value(*, value: object, policy: RedactionPolicy | None = None) -> RedactionResult:
    return Redactor(policy=policy).redact(value=value)


def build_redactor(*, policy: RedactionPolicy | None = None) -> Redactor:
    return Redactor(policy=policy)


def _key_tuple(value: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("value must be tuple[str, ...]")
    return tuple(ensure_non_empty_string(item, "key") for item in value)


def _copy_metadata(value: dict[str, object]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(value)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
        if key in _FORBIDDEN_SECRET or key in _FORBIDDEN_STATE:
            raise ValueError("metadata must not include sensitive keys")
    return copied
