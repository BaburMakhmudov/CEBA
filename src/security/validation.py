"""Deterministic security validation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

from ceba.security.auth import AuthResult
from ceba.security.permissions import PermissionResult
from ceba.security.redaction import RedactionResult, redact_value
from ceba.security.sandbox import SandboxResult
from ceba.security.schema import (
    SecurityEnvelope,
    SecurityRiskLevel,
    SecurityValidationReport,
    SecurityValidationStatus,
    create_security_validation_issue,
    create_security_validation_report,
)
from ceba.security.secrets import SecretValidationResult


__all__ = ["SecurityValidationPolicy", "SecurityValidationInput", "SecurityValidator", "validate_security_input", "build_security_validator"]

_FORBIDDEN_STATE = {"M_t", "Mt", "semantic_state", "runtime_semantic_state"}
_FORBIDDEN_SECRET = {"password", "raw_password", "secret", "raw_secret", "api_key", "token", "access_token", "refresh_token", "private_key", "credential"}


@dataclass(frozen=True)
class SecurityValidationPolicy:
    require_authenticated_actor: bool = True
    require_non_expired_envelope: bool = True
    require_permissions: bool = True
    block_critical_risk: bool = True
    redact_before_validation: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "require_authenticated_actor",
            "require_non_expired_envelope",
            "require_permissions",
            "block_critical_risk",
            "redact_before_validation",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")


@dataclass(frozen=True)
class SecurityValidationInput:
    envelope: SecurityEnvelope | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.envelope is not None and not isinstance(self.envelope, SecurityEnvelope):
            raise TypeError("envelope must be None or SecurityEnvelope")
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata, strict=True))


class SecurityValidator:
    def __init__(self, *, policy: SecurityValidationPolicy | None = None) -> None:
        self._policy = SecurityValidationPolicy() if policy is None else policy
        if not isinstance(self._policy, SecurityValidationPolicy):
            raise TypeError("policy must be SecurityValidationPolicy")

    def validate(self, *, validation_input: SecurityValidationInput) -> SecurityValidationReport:
        if not isinstance(validation_input, SecurityValidationInput):
            raise TypeError("validation_input must be SecurityValidationInput")
        issues = []
        if self._policy.redact_before_validation:
            redaction = redact_value(value=validation_input.metadata)
            if isinstance(redaction, RedactionResult) and redaction.redacted_count() > 0:
                issues.append(create_security_validation_issue(code="SECURITY_METADATA_REDACTED", message="metadata contained sensitive keys", risk_level=SecurityRiskLevel.CRITICAL))
        envelope = validation_input.envelope
        if envelope is None:
            if self._policy.require_authenticated_actor:
                issues.append(create_security_validation_issue(code="SECURITY_ENVELOPE_REQUIRED", message="security envelope required", risk_level=SecurityRiskLevel.CRITICAL))
        else:
            if self._policy.require_authenticated_actor and not envelope.actor.is_authenticated():
                issues.append(create_security_validation_issue(code="SECURITY_ACTOR_UNAUTHENTICATED", message="authenticated actor required", risk_level=SecurityRiskLevel.HIGH))
            if self._policy.require_non_expired_envelope and envelope.is_expired():
                issues.append(create_security_validation_issue(code="SECURITY_ENVELOPE_EXPIRED", message="security envelope expired", risk_level=SecurityRiskLevel.HIGH))
            if self._policy.require_permissions and envelope.permission_count() == 0:
                issues.append(create_security_validation_issue(code="SECURITY_PERMISSIONS_REQUIRED", message="permissions required", risk_level=SecurityRiskLevel.HIGH))
        critical = any(issue.risk_level is SecurityRiskLevel.CRITICAL for issue in issues)
        status = SecurityValidationStatus.VALID
        if issues:
            status = SecurityValidationStatus.BLOCKED if critical and self._policy.block_critical_risk else SecurityValidationStatus.WARNING
        return create_security_validation_report(status=status, issues=tuple(issues), metadata={"source": "security_validator"})


def validate_security_input(*, validation_input: SecurityValidationInput, policy: SecurityValidationPolicy | None = None) -> SecurityValidationReport:
    return SecurityValidator(policy=policy).validate(validation_input=validation_input)


def build_security_validator(*, policy: SecurityValidationPolicy | None = None) -> SecurityValidator:
    return SecurityValidator(policy=policy)


def _copy_metadata(value: dict[str, object], *, strict: bool) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(value)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
        if key in _FORBIDDEN_STATE:
            raise ValueError("metadata must not persist runtime semantic state")
        if strict and key in _FORBIDDEN_SECRET:
            raise ValueError("metadata must not include secret material")
    return copied
