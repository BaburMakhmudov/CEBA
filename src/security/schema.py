"""Strict immutable schema objects for CEBA security."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from uuid import UUID, uuid4


__all__ = [
    "SecurityActorKind",
    "SecurityCredentialKind",
    "SecurityAuthStatus",
    "SecurityPermissionEffect",
    "SecurityResourceKind",
    "SecurityRiskLevel",
    "SecurityValidationStatus",
    "SecurityActor",
    "SecurityCredentialRef",
    "SecurityResourceRef",
    "SecurityPermission",
    "SecurityEnvelope",
    "SecurityValidationIssue",
    "SecurityValidationReport",
    "utc_now",
    "new_security_actor_id",
    "new_security_credential_id",
    "new_security_resource_id",
    "new_security_envelope_id",
    "ensure_non_empty_string",
    "ensure_optional_non_empty_string",
    "ensure_positive_int",
    "ensure_non_negative_int",
    "ensure_probability",
    "ensure_finite_number",
    "ensure_timezone_aware",
    "create_security_actor",
    "create_security_credential_ref",
    "create_security_resource_ref",
    "create_security_permission",
    "create_security_envelope",
    "create_security_validation_issue",
    "create_security_validation_report",
]


_FORBIDDEN_STATE = {"M_t", "Mt", "semantic_state", "runtime_semantic_state"}
_FORBIDDEN_SECRET = {
    "password",
    "raw_password",
    "secret",
    "raw_secret",
    "api_key",
    "token",
    "access_token",
    "refresh_token",
    "private_key",
    "credential",
}


class SecurityActorKind(Enum):
    USER = "USER"
    SERVICE = "SERVICE"
    SYSTEM = "SYSTEM"
    RUNTIME = "RUNTIME"
    API_CLIENT = "API_CLIENT"
    ADMIN = "ADMIN"
    ANONYMOUS = "ANONYMOUS"
    UNKNOWN = "UNKNOWN"


class SecurityCredentialKind(Enum):
    PASSWORD_REF = "PASSWORD_REF"
    API_KEY_REF = "API_KEY_REF"
    TOKEN_REF = "TOKEN_REF"
    SESSION_REF = "SESSION_REF"
    SSH_KEY_REF = "SSH_KEY_REF"
    OAUTH_REF = "OAUTH_REF"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"


class SecurityAuthStatus(Enum):
    UNAUTHENTICATED = "UNAUTHENTICATED"
    AUTHENTICATED = "AUTHENTICATED"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"
    BLOCKED = "BLOCKED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class SecurityPermissionEffect(Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    UNKNOWN = "UNKNOWN"


class SecurityResourceKind(Enum):
    API = "API"
    RUNTIME = "RUNTIME"
    MEMORY = "MEMORY"
    WORLD_MODEL = "WORLD_MODEL"
    PERSISTENCE = "PERSISTENCE"
    LEDGER = "LEDGER"
    VECTOR_SURFACE = "VECTOR_SURFACE"
    GRAPH_SURFACE = "GRAPH_SURFACE"
    AUDIT = "AUDIT"
    SECURITY = "SECURITY"
    CONFIG = "CONFIG"
    UNKNOWN = "UNKNOWN"


class SecurityRiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class SecurityValidationStatus(Enum):
    VALID = "VALID"
    WARNING = "WARNING"
    BLOCKED = "BLOCKED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class SecurityActor:
    actor_id: UUID
    actor_key: str
    kind: SecurityActorKind
    auth_status: SecurityAuthStatus
    label: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.actor_id, UUID):
            raise TypeError("actor_id must be UUID")
        actor_key = ensure_non_empty_string(self.actor_key, "actor_key")
        _reject_name(actor_key, "actor_key")
        if not isinstance(self.kind, SecurityActorKind):
            raise TypeError("kind must be SecurityActorKind")
        if not isinstance(self.auth_status, SecurityAuthStatus):
            raise TypeError("auth_status must be SecurityAuthStatus")
        label = ensure_optional_non_empty_string(self.label, "label")
        if label is not None:
            _reject_name(label, "label")
        object.__setattr__(self, "actor_key", actor_key)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "actor_id": str(self.actor_id),
            "actor_key": self.actor_key,
            "kind": self.kind.value,
            "auth_status": self.auth_status.value,
            "label": self.label,
            "metadata": dict(self.metadata),
        }

    def is_authenticated(self) -> bool:
        return self.auth_status is SecurityAuthStatus.AUTHENTICATED


@dataclass(frozen=True)
class SecurityCredentialRef:
    credential_id: UUID
    kind: SecurityCredentialKind
    reference: str
    expires_at: datetime | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.credential_id, UUID):
            raise TypeError("credential_id must be UUID")
        if not isinstance(self.kind, SecurityCredentialKind):
            raise TypeError("kind must be SecurityCredentialKind")
        reference = ensure_non_empty_string(self.reference, "reference")
        _reject_secret_reference(reference, "reference")
        expires_at = None if self.expires_at is None else ensure_timezone_aware(self.expires_at, "expires_at")
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "expires_at", expires_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "credential_id": str(self.credential_id),
            "kind": self.kind.value,
            "reference": self.reference,
            "expires_at": None if self.expires_at is None else self.expires_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    def is_expired(self, *, at: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        check_at = utc_now() if at is None else ensure_timezone_aware(at, "at")
        return self.expires_at <= check_at


@dataclass(frozen=True)
class SecurityResourceRef:
    resource_id: UUID
    resource_key: str
    kind: SecurityResourceKind
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.resource_id, UUID):
            raise TypeError("resource_id must be UUID")
        resource_key = ensure_non_empty_string(self.resource_key, "resource_key")
        _reject_name(resource_key, "resource_key")
        if not isinstance(self.kind, SecurityResourceKind):
            raise TypeError("kind must be SecurityResourceKind")
        object.__setattr__(self, "resource_key", resource_key)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "resource_id": str(self.resource_id),
            "resource_key": self.resource_key,
            "kind": self.kind.value,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SecurityPermission:
    actor: SecurityActor
    resource: SecurityResourceRef
    effect: SecurityPermissionEffect
    actions: tuple[str, ...]
    risk_level: SecurityRiskLevel = SecurityRiskLevel.LOW
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.actor, SecurityActor):
            raise TypeError("actor must be SecurityActor")
        if not isinstance(self.resource, SecurityResourceRef):
            raise TypeError("resource must be SecurityResourceRef")
        if not isinstance(self.effect, SecurityPermissionEffect):
            raise TypeError("effect must be SecurityPermissionEffect")
        actions = _string_tuple(self.actions, "actions")
        if len(actions) != len(set(actions)):
            raise ValueError("action names must be unique")
        for action in actions:
            _reject_name(action, "action")
        if not isinstance(self.risk_level, SecurityRiskLevel):
            raise TypeError("risk_level must be SecurityRiskLevel")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "actor": self.actor.to_dict(),
            "resource": self.resource.to_dict(),
            "effect": self.effect.value,
            "actions": list(self.actions),
            "risk_level": self.risk_level.value,
            "metadata": dict(self.metadata),
        }

    def allows(self, *, action: str) -> bool:
        return self.effect is SecurityPermissionEffect.ALLOW and ensure_non_empty_string(action, "action") in self.actions

    def denies(self, *, action: str) -> bool:
        return self.effect is SecurityPermissionEffect.DENY and ensure_non_empty_string(action, "action") in self.actions


@dataclass(frozen=True)
class SecurityEnvelope:
    envelope_id: UUID
    actor: SecurityActor
    credential_ref: SecurityCredentialRef | None
    permissions: tuple[SecurityPermission, ...]
    created_at: datetime
    expires_at: datetime | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.envelope_id, UUID):
            raise TypeError("envelope_id must be UUID")
        if not isinstance(self.actor, SecurityActor):
            raise TypeError("actor must be SecurityActor")
        if self.credential_ref is not None and not isinstance(self.credential_ref, SecurityCredentialRef):
            raise TypeError("credential_ref must be None or SecurityCredentialRef")
        permissions = _permission_tuple(self.permissions)
        for permission in permissions:
            if permission.actor.actor_id != self.actor.actor_id:
                raise ValueError("permission actor must match envelope actor")
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        expires_at = None if self.expires_at is None else ensure_timezone_aware(self.expires_at, "expires_at")
        if expires_at is not None and expires_at < created_at:
            raise ValueError("expires_at must be >= created_at")
        object.__setattr__(self, "permissions", permissions)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "expires_at", expires_at)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "envelope_id": str(self.envelope_id),
            "actor": self.actor.to_dict(),
            "credential_ref": None if self.credential_ref is None else self.credential_ref.to_dict(),
            "permissions": [permission.to_dict() for permission in self.permissions],
            "created_at": self.created_at.isoformat(),
            "expires_at": None if self.expires_at is None else self.expires_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    def permission_count(self) -> int:
        return len(self.permissions)

    def is_expired(self, *, at: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        check_at = utc_now() if at is None else ensure_timezone_aware(at, "at")
        return self.expires_at <= check_at

    def has_permission(self, *, resource_key: str, action: str) -> bool:
        resource_key = ensure_non_empty_string(resource_key, "resource_key")
        action = ensure_non_empty_string(action, "action")
        denied = any(permission.resource.resource_key == resource_key and permission.denies(action=action) for permission in self.permissions)
        if denied:
            return False
        return any(permission.resource.resource_key == resource_key and permission.allows(action=action) for permission in self.permissions)


@dataclass(frozen=True)
class SecurityValidationIssue:
    code: str
    message: str
    risk_level: SecurityRiskLevel
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", ensure_non_empty_string(self.code, "code"))
        object.__setattr__(self, "message", ensure_non_empty_string(self.message, "message"))
        if not isinstance(self.risk_level, SecurityRiskLevel):
            raise TypeError("risk_level must be SecurityRiskLevel")
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, "risk_level": self.risk_level.value, "metadata": dict(self.metadata)}


@dataclass(frozen=True)
class SecurityValidationReport:
    status: SecurityValidationStatus
    issues: tuple[SecurityValidationIssue, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, SecurityValidationStatus):
            raise TypeError("status must be SecurityValidationStatus")
        issues = _issue_tuple(self.issues)
        if any(issue.risk_level is SecurityRiskLevel.CRITICAL for issue in issues) and self.status not in {SecurityValidationStatus.BLOCKED, SecurityValidationStatus.INVALID}:
            raise ValueError("status must be BLOCKED or INVALID when critical issues exist")
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {"status": self.status.value, "issues": [issue.to_dict() for issue in self.issues], "metadata": dict(self.metadata)}

    def issue_count(self) -> int:
        return len(self.issues)

    def has_critical(self) -> bool:
        return any(issue.risk_level is SecurityRiskLevel.CRITICAL for issue in self.issues)

    def is_valid(self) -> bool:
        return self.status is SecurityValidationStatus.VALID and not self.has_critical()


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_security_actor_id() -> UUID:
    return uuid4()


def new_security_credential_id() -> UUID:
    return uuid4()


def new_security_resource_id() -> UUID:
    return uuid4()


def new_security_envelope_id() -> UUID:
    return uuid4()


def ensure_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def ensure_optional_non_empty_string(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return ensure_non_empty_string(value, field_name)


def ensure_positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return value


def ensure_non_negative_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return value


def ensure_probability(value: float, field_name: str) -> float:
    number = ensure_finite_number(value, field_name)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return number


def ensure_finite_number(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be int or float")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def ensure_timezone_aware(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def create_security_actor(
    *,
    actor_key: str,
    kind: SecurityActorKind = SecurityActorKind.UNKNOWN,
    auth_status: SecurityAuthStatus = SecurityAuthStatus.UNAUTHENTICATED,
    label: str | None = None,
    metadata: dict[str, object] | None = None,
) -> SecurityActor:
    return SecurityActor(new_security_actor_id(), actor_key, kind, auth_status, label, _merge_metadata({"source": "security_actor"}, metadata))


def create_security_credential_ref(
    *,
    kind: SecurityCredentialKind,
    reference: str,
    expires_at: datetime | None = None,
    metadata: dict[str, object] | None = None,
) -> SecurityCredentialRef:
    return SecurityCredentialRef(new_security_credential_id(), kind, reference, expires_at, _merge_metadata({"source": "security_credential_ref"}, metadata))


def create_security_resource_ref(
    *,
    resource_key: str,
    kind: SecurityResourceKind = SecurityResourceKind.UNKNOWN,
    metadata: dict[str, object] | None = None,
) -> SecurityResourceRef:
    return SecurityResourceRef(new_security_resource_id(), resource_key, kind, _merge_metadata({"source": "security_resource_ref"}, metadata))


def create_security_permission(
    *,
    actor: SecurityActor,
    resource: SecurityResourceRef,
    effect: SecurityPermissionEffect,
    actions: tuple[str, ...] | list[str],
    risk_level: SecurityRiskLevel = SecurityRiskLevel.LOW,
    metadata: dict[str, object] | None = None,
) -> SecurityPermission:
    return SecurityPermission(actor, resource, effect, tuple(actions), risk_level, _merge_metadata({"source": "security_permission"}, metadata))


def create_security_envelope(
    *,
    actor: SecurityActor,
    credential_ref: SecurityCredentialRef | None = None,
    permissions: tuple[SecurityPermission, ...] | list[SecurityPermission] = (),
    expires_at: datetime | None = None,
    metadata: dict[str, object] | None = None,
) -> SecurityEnvelope:
    return SecurityEnvelope(new_security_envelope_id(), actor, credential_ref, tuple(permissions), utc_now(), expires_at, _merge_metadata({"source": "security_envelope"}, metadata))


def create_security_validation_issue(
    *,
    code: str,
    message: str,
    risk_level: SecurityRiskLevel = SecurityRiskLevel.LOW,
    metadata: dict[str, object] | None = None,
) -> SecurityValidationIssue:
    return SecurityValidationIssue(code, message, risk_level, _merge_metadata({"source": "security_validation_issue"}, metadata))


def create_security_validation_report(
    *,
    status: SecurityValidationStatus = SecurityValidationStatus.VALID,
    issues: tuple[SecurityValidationIssue, ...] | list[SecurityValidationIssue] = (),
    metadata: dict[str, object] | None = None,
) -> SecurityValidationReport:
    return SecurityValidationReport(status, tuple(issues), _merge_metadata({"source": "security_validation_report"}, metadata))


def _copy_metadata(value: dict[str, object]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(value)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
        _reject_name(key, "metadata key")
        _reject_secret_key(key, "metadata key")
    return copied


def _merge_metadata(defaults: dict[str, object], metadata: dict[str, object] | None) -> dict[str, object]:
    merged = _copy_metadata(defaults)
    if metadata is not None:
        merged.update(_copy_metadata(metadata))
    if "source" in defaults:
        merged["source"] = defaults["source"]
    return merged


def _reject_name(value: str, field_name: str) -> None:
    if value in _FORBIDDEN_STATE:
        raise ValueError(f"{field_name} must not persist runtime semantic state")


def _reject_secret_key(value: str, field_name: str) -> None:
    if value in _FORBIDDEN_SECRET:
        raise ValueError(f"{field_name} must not include secret material")


def _reject_secret_reference(value: str, field_name: str) -> None:
    lowered = value.lower()
    for token in ("password=", "secret=", "token=", "api_key=", "private_key="):
        if token in lowered:
            raise ValueError(f"{field_name} must be a reference only")


def _string_tuple(value: tuple[str, ...] | list[str], field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple[str, ...]")
    return tuple(ensure_non_empty_string(item, field_name) for item in value)


def _permission_tuple(value: tuple[SecurityPermission, ...] | list[SecurityPermission]) -> tuple[SecurityPermission, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("permissions must be tuple[SecurityPermission, ...]")
    permissions = tuple(value)
    for permission in permissions:
        if not isinstance(permission, SecurityPermission):
            raise TypeError("every permission must be SecurityPermission")
    return permissions


def _issue_tuple(value: tuple[SecurityValidationIssue, ...] | list[SecurityValidationIssue]) -> tuple[SecurityValidationIssue, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("issues must be tuple[SecurityValidationIssue, ...]")
    issues = tuple(value)
    for issue in issues:
        if not isinstance(issue, SecurityValidationIssue):
            raise TypeError("every issue must be SecurityValidationIssue")
    return issues
