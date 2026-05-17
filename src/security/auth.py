"""Deterministic local authentication descriptors."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ceba.security.schema import (
    SecurityActor,
    SecurityActorKind,
    SecurityAuthStatus,
    SecurityCredentialKind,
    SecurityCredentialRef,
    SecurityEnvelope,
    SecurityRiskLevel,
    SecurityValidationReport,
    SecurityValidationStatus,
    create_security_actor,
    create_security_credential_ref,
    create_security_envelope,
    create_security_validation_issue,
    create_security_validation_report,
    ensure_non_empty_string,
    ensure_optional_non_empty_string,
    ensure_positive_int,
    ensure_timezone_aware,
)


__all__ = ["AuthPolicy", "AuthInput", "AuthResult", "AuthManager", "build_auth_manager"]


_FORBIDDEN_STATE = {"M_t", "Mt", "semantic_state", "runtime_semantic_state"}
_FORBIDDEN_SECRET = {"password", "raw_password", "secret", "raw_secret", "api_key", "token", "access_token", "refresh_token", "private_key", "credential"}


@dataclass(frozen=True)
class AuthPolicy:
    allow_anonymous: bool = False
    require_credential_ref: bool = True
    allow_expired_credentials: bool = False
    max_metadata_keys: int = 64

    def __post_init__(self) -> None:
        for name in ("allow_anonymous", "require_credential_ref", "allow_expired_credentials"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        object.__setattr__(self, "max_metadata_keys", ensure_positive_int(self.max_metadata_keys, "max_metadata_keys"))

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class AuthInput:
    actor_key: str
    actor_kind: SecurityActorKind
    credential_kind: SecurityCredentialKind
    credential_reference: str | None = None
    label: str | None = None
    expires_at: datetime | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        actor_key = ensure_non_empty_string(self.actor_key, "actor_key")
        _reject_name(actor_key, "actor_key")
        if not isinstance(self.actor_kind, SecurityActorKind):
            raise TypeError("actor_kind must be SecurityActorKind")
        if not isinstance(self.credential_kind, SecurityCredentialKind):
            raise TypeError("credential_kind must be SecurityCredentialKind")
        object.__setattr__(self, "actor_key", actor_key)
        object.__setattr__(self, "credential_reference", ensure_optional_non_empty_string(self.credential_reference, "credential_reference"))
        object.__setattr__(self, "label", ensure_optional_non_empty_string(self.label, "label"))
        object.__setattr__(self, "expires_at", None if self.expires_at is None else ensure_timezone_aware(self.expires_at, "expires_at"))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "actor_key": self.actor_key,
            "actor_kind": self.actor_kind.value,
            "credential_kind": self.credential_kind.value,
            "credential_reference": self.credential_reference,
            "label": self.label,
            "expires_at": None if self.expires_at is None else self.expires_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AuthResult:
    envelope: SecurityEnvelope | None
    actor: SecurityActor | None
    credential_ref: SecurityCredentialRef | None
    report: SecurityValidationReport
    changed: bool
    reason: str
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.envelope is not None and not isinstance(self.envelope, SecurityEnvelope):
            raise TypeError("envelope must be None or SecurityEnvelope")
        if self.actor is not None and not isinstance(self.actor, SecurityActor):
            raise TypeError("actor must be None or SecurityActor")
        if self.credential_ref is not None and not isinstance(self.credential_ref, SecurityCredentialRef):
            raise TypeError("credential_ref must be None or SecurityCredentialRef")
        if not isinstance(self.report, SecurityValidationReport):
            raise TypeError("report must be SecurityValidationReport")
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        object.__setattr__(self, "reason", ensure_non_empty_string(self.reason, "reason"))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "envelope": None if self.envelope is None else self.envelope.to_dict(),
            "actor": None if self.actor is None else self.actor.to_dict(),
            "credential_ref": None if self.credential_ref is None else self.credential_ref.to_dict(),
            "report": self.report.to_dict(),
            "changed": self.changed,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }

    def is_authenticated(self) -> bool:
        return self.actor is not None and self.actor.is_authenticated() and self.envelope is not None and self.report.is_valid()


class AuthManager:
    def __init__(self, *, policy: AuthPolicy | None = None) -> None:
        self._policy = AuthPolicy() if policy is None else policy
        if not isinstance(self._policy, AuthPolicy):
            raise TypeError("policy must be AuthPolicy")

    def authenticate(self, *, auth_input: AuthInput) -> AuthResult:
        if not isinstance(auth_input, AuthInput):
            raise TypeError("auth_input must be AuthInput")
        issues = []
        if len(auth_input.metadata) > self._policy.max_metadata_keys:
            issues.append(create_security_validation_issue(code="AUTH_METADATA_LIMIT", message="auth metadata key limit exceeded", risk_level=SecurityRiskLevel.HIGH))
        if auth_input.actor_kind is SecurityActorKind.ANONYMOUS and not self._policy.allow_anonymous:
            issues.append(create_security_validation_issue(code="AUTH_ANONYMOUS_BLOCKED", message="anonymous actor blocked", risk_level=SecurityRiskLevel.HIGH))
        if auth_input.credential_reference is None and self._policy.require_credential_ref:
            issues.append(create_security_validation_issue(code="AUTH_CREDENTIAL_REQUIRED", message="credential reference required", risk_level=SecurityRiskLevel.HIGH))
        credential = None
        if auth_input.credential_reference is not None:
            credential = create_security_credential_ref(kind=auth_input.credential_kind, reference=auth_input.credential_reference, expires_at=auth_input.expires_at)
            if credential.is_expired() and not self._policy.allow_expired_credentials:
                issues.append(create_security_validation_issue(code="AUTH_CREDENTIAL_EXPIRED", message="credential reference expired", risk_level=SecurityRiskLevel.HIGH))
        status = SecurityValidationStatus.VALID if not issues else SecurityValidationStatus.BLOCKED
        auth_status = SecurityAuthStatus.AUTHENTICATED if not issues else SecurityAuthStatus.BLOCKED
        actor = create_security_actor(actor_key=auth_input.actor_key, kind=auth_input.actor_kind, auth_status=auth_status, label=auth_input.label, metadata=auth_input.metadata)
        report = create_security_validation_report(status=status, issues=tuple(issues))
        envelope = None if issues else create_security_envelope(actor=actor, credential_ref=credential, metadata={"source": "auth_manager"})
        return AuthResult(envelope, actor, credential, report, not issues, "security actor authenticated" if not issues else "security authentication blocked", {"source": "auth_result"})

    def reject(self, *, auth_input: AuthInput, reason: str) -> AuthResult:
        if not isinstance(auth_input, AuthInput):
            raise TypeError("auth_input must be AuthInput")
        reason = ensure_non_empty_string(reason, "reason")
        actor = create_security_actor(actor_key=auth_input.actor_key, kind=auth_input.actor_kind, auth_status=SecurityAuthStatus.BLOCKED, label=auth_input.label)
        issue = create_security_validation_issue(code="AUTH_REJECTED", message=reason, risk_level=SecurityRiskLevel.HIGH)
        report = create_security_validation_report(status=SecurityValidationStatus.BLOCKED, issues=(issue,))
        return AuthResult(None, actor, None, report, False, reason, {"source": "auth_result"})

    def validate_envelope(self, *, envelope: SecurityEnvelope) -> AuthResult:
        if not isinstance(envelope, SecurityEnvelope):
            raise TypeError("envelope must be SecurityEnvelope")
        issues = []
        if not envelope.actor.is_authenticated():
            issues.append(create_security_validation_issue(code="AUTH_ACTOR_NOT_AUTHENTICATED", message="actor is not authenticated", risk_level=SecurityRiskLevel.HIGH))
        if envelope.is_expired():
            issues.append(create_security_validation_issue(code="AUTH_ENVELOPE_EXPIRED", message="security envelope expired", risk_level=SecurityRiskLevel.HIGH))
        if envelope.credential_ref is not None and envelope.credential_ref.is_expired() and not self._policy.allow_expired_credentials:
            issues.append(create_security_validation_issue(code="AUTH_CREDENTIAL_EXPIRED", message="credential reference expired", risk_level=SecurityRiskLevel.HIGH))
        report = create_security_validation_report(status=SecurityValidationStatus.VALID if not issues else SecurityValidationStatus.BLOCKED, issues=tuple(issues))
        return AuthResult(envelope if not issues else None, envelope.actor, envelope.credential_ref, report, False, "security envelope valid" if not issues else "security envelope invalid", {"source": "auth_result"})


def build_auth_manager(*, policy: AuthPolicy | None = None) -> AuthManager:
    return AuthManager(policy=policy)


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
