"""Deterministic secret reference utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ceba.security.schema import (
    SecurityRiskLevel,
    SecurityValidationReport,
    SecurityValidationStatus,
    create_security_validation_issue,
    create_security_validation_report,
    ensure_non_empty_string,
    ensure_optional_non_empty_string,
    ensure_positive_int,
    ensure_timezone_aware,
    utc_now,
)


__all__ = ["SecretRefKind", "SecretRefStatus", "SecretRef", "SecretValidationResult", "SecretRegistryPolicy", "SecretRegistry", "build_secret_registry"]

_FORBIDDEN_STATE = {"M_t", "Mt", "semantic_state", "runtime_semantic_state"}
_FORBIDDEN_SECRET = {"password", "raw_password", "secret", "raw_secret", "api_key", "token", "access_token", "refresh_token", "private_key", "credential"}


class SecretRefKind(Enum):
    PASSWORD = "PASSWORD"
    API_KEY = "API_KEY"
    TOKEN = "TOKEN"
    SSH_KEY = "SSH_KEY"
    OAUTH_CLIENT_SECRET = "OAUTH_CLIENT_SECRET"
    DATABASE_PASSWORD = "DATABASE_PASSWORD"
    UNKNOWN = "UNKNOWN"


class SecretRefStatus(Enum):
    CREATED = "CREATED"
    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"
    BLOCKED = "BLOCKED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class SecretRef:
    secret_key: str
    kind: SecretRefKind
    reference: str
    fingerprint: str | None = None
    status: SecretRefStatus = SecretRefStatus.CREATED
    created_at: datetime = field(default_factory=utc_now)
    expires_at: datetime | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        secret_key = ensure_non_empty_string(self.secret_key, "secret_key")
        reference = ensure_non_empty_string(self.reference, "reference")
        _reject_name(secret_key, "secret_key")
        _reject_raw(reference, "reference")
        if not isinstance(self.kind, SecretRefKind):
            raise TypeError("kind must be SecretRefKind")
        if not isinstance(self.status, SecretRefStatus):
            raise TypeError("status must be SecretRefStatus")
        object.__setattr__(self, "secret_key", secret_key)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "fingerprint", ensure_optional_non_empty_string(self.fingerprint, "fingerprint"))
        object.__setattr__(self, "created_at", ensure_timezone_aware(self.created_at, "created_at"))
        object.__setattr__(self, "expires_at", None if self.expires_at is None else ensure_timezone_aware(self.expires_at, "expires_at"))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {"secret_key": self.secret_key, "kind": self.kind.value, "reference": self.reference, "fingerprint": self.fingerprint, "status": self.status.value, "created_at": self.created_at.isoformat(), "expires_at": None if self.expires_at is None else self.expires_at.isoformat(), "metadata": dict(self.metadata)}

    def is_active(self, *, at: datetime | None = None) -> bool:
        check_at = utc_now() if at is None else ensure_timezone_aware(at, "at")
        return self.status is SecretRefStatus.ACTIVE and (self.expires_at is None or self.expires_at > check_at)


@dataclass(frozen=True)
class SecretValidationResult:
    secret_ref: SecretRef | None
    report: SecurityValidationReport
    changed: bool
    reason: str
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.secret_ref is not None and not isinstance(self.secret_ref, SecretRef):
            raise TypeError("secret_ref must be None or SecretRef")
        if not isinstance(self.report, SecurityValidationReport):
            raise TypeError("report must be SecurityValidationReport")
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        object.__setattr__(self, "reason", ensure_non_empty_string(self.reason, "reason"))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {"secret_ref": None if self.secret_ref is None else self.secret_ref.to_dict(), "report": self.report.to_dict(), "changed": self.changed, "reason": self.reason, "metadata": dict(self.metadata)}


@dataclass(frozen=True)
class SecretRegistryPolicy:
    allow_duplicate_secret_keys: bool = False
    reject_expired: bool = True
    max_secrets: int = 10000

    def __post_init__(self) -> None:
        if not isinstance(self.allow_duplicate_secret_keys, bool):
            raise TypeError("allow_duplicate_secret_keys must be bool")
        if not isinstance(self.reject_expired, bool):
            raise TypeError("reject_expired must be bool")
        object.__setattr__(self, "max_secrets", ensure_positive_int(self.max_secrets, "max_secrets"))


class SecretRegistry:
    def __init__(self, *, secrets: tuple[SecretRef, ...] | list[SecretRef] = (), policy: SecretRegistryPolicy | None = None) -> None:
        self._policy = SecretRegistryPolicy() if policy is None else policy
        if not isinstance(self._policy, SecretRegistryPolicy):
            raise TypeError("policy must be SecretRegistryPolicy")
        self._secrets = _secret_tuple(secrets)

    def secrets(self) -> tuple[SecretRef, ...]:
        return self._secrets

    def register(self, *, secret_ref: SecretRef) -> SecretValidationResult:
        if not isinstance(secret_ref, SecretRef):
            raise TypeError("secret_ref must be SecretRef")
        if len(self._secrets) >= self._policy.max_secrets:
            return _result(None, "secret registry limit exceeded", False, SecurityValidationStatus.BLOCKED, "SECRET_LIMIT")
        if not self._policy.allow_duplicate_secret_keys and any(item.secret_key == secret_ref.secret_key for item in self._secrets):
            return _result(secret_ref, "secret key duplicated", False, SecurityValidationStatus.BLOCKED, "SECRET_DUPLICATE")
        if self._policy.reject_expired and not secret_ref.is_active() and secret_ref.status is SecretRefStatus.EXPIRED:
            return _result(secret_ref, "secret reference expired", False, SecurityValidationStatus.BLOCKED, "SECRET_EXPIRED")
        self._secrets = self._secrets + (secret_ref,)
        return _result(secret_ref, "secret reference registered", True, SecurityValidationStatus.VALID)

    def get(self, *, secret_key: str) -> SecretValidationResult:
        secret_key = ensure_non_empty_string(secret_key, "secret_key")
        for secret in self._secrets:
            if secret.secret_key == secret_key:
                return _result(secret, "secret reference found", False, SecurityValidationStatus.VALID)
        return _result(None, "secret reference missing", False, SecurityValidationStatus.BLOCKED, "SECRET_MISSING")

    def revoke(self, *, secret_key: str) -> SecretValidationResult:
        found = self.get(secret_key=secret_key).secret_ref
        if found is None:
            return _result(None, "secret reference missing", False, SecurityValidationStatus.BLOCKED, "SECRET_MISSING")
        updated = SecretRef(found.secret_key, found.kind, found.reference, found.fingerprint, SecretRefStatus.REVOKED, found.created_at, found.expires_at, found.metadata)
        self._secrets = tuple(updated if item.secret_key == secret_key else item for item in self._secrets)
        return _result(updated, "secret reference revoked", True, SecurityValidationStatus.BLOCKED, "SECRET_REVOKED")

    def validate(self, *, secret_key: str) -> SecretValidationResult:
        result = self.get(secret_key=secret_key)
        if result.secret_ref is None:
            return result
        secret = result.secret_ref
        if self._policy.reject_expired and not secret.is_active():
            return _result(secret, "secret reference inactive", False, SecurityValidationStatus.BLOCKED, "SECRET_INACTIVE")
        return _result(secret, "secret reference valid", False, SecurityValidationStatus.VALID)


def build_secret_registry(*, secrets: tuple[SecretRef, ...] | list[SecretRef] = (), policy: SecretRegistryPolicy | None = None) -> SecretRegistry:
    return SecretRegistry(secrets=secrets, policy=policy)


def _result(secret_ref: SecretRef | None, reason: str, changed: bool, status: SecurityValidationStatus, code: str | None = None) -> SecretValidationResult:
    issues = () if code is None else (create_security_validation_issue(code=code, message=reason, risk_level=SecurityRiskLevel.HIGH),)
    return SecretValidationResult(secret_ref, create_security_validation_report(status=status, issues=issues), changed, reason, {"source": "secret_validation_result"})


def _secret_tuple(value: tuple[SecretRef, ...] | list[SecretRef]) -> tuple[SecretRef, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("secrets must be tuple[SecretRef, ...]")
    secrets = tuple(value)
    for secret in secrets:
        if not isinstance(secret, SecretRef):
            raise TypeError("every secret must be SecretRef")
    return secrets


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


def _reject_raw(value: str, field_name: str) -> None:
    lowered = value.lower()
    for token in ("password=", "secret=", "token=", "api_key=", "private_key="):
        if token in lowered:
            raise ValueError(f"{field_name} must be a reference only")
