"""Deterministic sandbox policy descriptors."""

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


__all__ = ["SandboxCapability", "SandboxDecision", "SandboxPolicy", "SandboxRequest", "SandboxResult", "SandboxValidator", "build_sandbox_validator"]

_FORBIDDEN_STATE = {"M_t", "Mt", "semantic_state", "runtime_semantic_state"}
_FORBIDDEN_SECRET = {"password", "raw_password", "secret", "raw_secret", "api_key", "token", "access_token", "refresh_token", "private_key", "credential"}


class SandboxCapability(Enum):
    FILE_READ = "FILE_READ"
    FILE_WRITE = "FILE_WRITE"
    NETWORK = "NETWORK"
    SUBPROCESS = "SUBPROCESS"
    DATABASE = "DATABASE"
    ENVIRONMENT = "ENVIRONMENT"
    IMPORT_EXTERNAL = "IMPORT_EXTERNAL"
    LLM_CALL = "LLM_CALL"
    UNKNOWN = "UNKNOWN"


class SandboxDecision(Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    BLOCK = "BLOCK"
    INVALID = "INVALID"


@dataclass(frozen=True)
class SandboxPolicy:
    allowed_capabilities: tuple[SandboxCapability, ...] = ()
    blocked_capabilities: tuple[SandboxCapability, ...] = (
        SandboxCapability.FILE_READ,
        SandboxCapability.FILE_WRITE,
        SandboxCapability.NETWORK,
        SandboxCapability.SUBPROCESS,
        SandboxCapability.DATABASE,
        SandboxCapability.ENVIRONMENT,
        SandboxCapability.IMPORT_EXTERNAL,
        SandboxCapability.LLM_CALL,
    )
    max_operations: int = 128
    default_decision: SandboxDecision = SandboxDecision.DENY

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_capabilities", _cap_tuple(self.allowed_capabilities, "allowed_capabilities"))
        object.__setattr__(self, "blocked_capabilities", _cap_tuple(self.blocked_capabilities, "blocked_capabilities"))
        object.__setattr__(self, "max_operations", ensure_positive_int(self.max_operations, "max_operations"))
        if not isinstance(self.default_decision, SandboxDecision):
            raise TypeError("default_decision must be SandboxDecision")


@dataclass(frozen=True)
class SandboxRequest:
    capability: SandboxCapability
    operation_name: str
    requested_by: str
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.capability, SandboxCapability):
            raise TypeError("capability must be SandboxCapability")
        operation_name = ensure_non_empty_string(self.operation_name, "operation_name")
        requested_by = ensure_non_empty_string(self.requested_by, "requested_by")
        _reject_name(operation_name, "operation_name")
        _reject_name(requested_by, "requested_by")
        object.__setattr__(self, "operation_name", operation_name)
        object.__setattr__(self, "requested_by", requested_by)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))


@dataclass(frozen=True)
class SandboxResult:
    decision: SandboxDecision
    report: SecurityValidationReport
    changed: bool
    reason: str
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.decision, SandboxDecision):
            raise TypeError("decision must be SandboxDecision")
        if not isinstance(self.report, SecurityValidationReport):
            raise TypeError("report must be SecurityValidationReport")
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        object.__setattr__(self, "reason", ensure_non_empty_string(self.reason, "reason"))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {"decision": self.decision.value, "report": self.report.to_dict(), "changed": self.changed, "reason": self.reason, "metadata": dict(self.metadata)}

    def is_allowed(self) -> bool:
        return self.decision is SandboxDecision.ALLOW and self.report.is_valid()


class SandboxValidator:
    def __init__(self, *, policy: SandboxPolicy | None = None) -> None:
        self._policy = SandboxPolicy() if policy is None else policy
        if not isinstance(self._policy, SandboxPolicy):
            raise TypeError("policy must be SandboxPolicy")

    def validate(self, *, request: SandboxRequest) -> SandboxResult:
        if not isinstance(request, SandboxRequest):
            raise TypeError("request must be SandboxRequest")
        if request.capability is SandboxCapability.UNKNOWN:
            return _result(SandboxDecision.INVALID, "sandbox capability invalid", "SANDBOX_UNKNOWN")
        if request.capability in self._policy.blocked_capabilities:
            return _result(SandboxDecision.BLOCK, "sandbox capability blocked", "SANDBOX_BLOCKED")
        if request.capability in self._policy.allowed_capabilities:
            return _result(SandboxDecision.ALLOW, "sandbox capability allowed")
        return _result(self._policy.default_decision, "sandbox default decision", None if self._policy.default_decision is SandboxDecision.ALLOW else "SANDBOX_DEFAULT_DENY")

    def block(self, *, request: SandboxRequest, reason: str) -> SandboxResult:
        if not isinstance(request, SandboxRequest):
            raise TypeError("request must be SandboxRequest")
        return _result(SandboxDecision.BLOCK, ensure_non_empty_string(reason, "reason"), "SANDBOX_MANUAL_BLOCK")


def build_sandbox_validator(*, policy: SandboxPolicy | None = None) -> SandboxValidator:
    return SandboxValidator(policy=policy)


def _result(decision: SandboxDecision, reason: str, code: str | None = None) -> SandboxResult:
    issues = () if code is None else (create_security_validation_issue(code=code, message=reason, risk_level=SecurityRiskLevel.HIGH),)
    status = SecurityValidationStatus.VALID if decision is SandboxDecision.ALLOW and code is None else SecurityValidationStatus.BLOCKED
    if decision is SandboxDecision.INVALID:
        status = SecurityValidationStatus.INVALID
    return SandboxResult(decision, create_security_validation_report(status=status, issues=issues), False, reason, {"source": "sandbox_result"})


def _cap_tuple(value: tuple[SandboxCapability, ...] | list[SandboxCapability], field_name: str) -> tuple[SandboxCapability, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple[SandboxCapability, ...]")
    result = tuple(value)
    for item in result:
        if not isinstance(item, SandboxCapability):
            raise TypeError(f"{field_name} values must be SandboxCapability")
    return result


def _copy_metadata(value: dict[str, object]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(value)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
        _reject_name(key, "metadata key")
        if key in _FORBIDDEN_SECRET:
            raise ValueError("metadata must not include secret material")
    return copied


def _reject_name(value: str, field_name: str) -> None:
    if value in _FORBIDDEN_STATE:
        raise ValueError(f"{field_name} must not persist runtime semantic state")
