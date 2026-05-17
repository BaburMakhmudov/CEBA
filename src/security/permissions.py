"""Deterministic local permission policy utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ceba.security.schema import (
    SecurityActor,
    SecurityEnvelope,
    SecurityPermission,
    SecurityPermissionEffect,
    SecurityResourceKind,
    SecurityResourceRef,
    SecurityRiskLevel,
    SecurityValidationReport,
    SecurityValidationStatus,
    create_security_permission,
    create_security_resource_ref,
    create_security_validation_issue,
    create_security_validation_report,
    ensure_non_empty_string,
    ensure_positive_int,
)


__all__ = ["PermissionDecision", "PermissionPolicy", "PermissionRequest", "PermissionResult", "PermissionManager", "build_permission_manager"]

_FORBIDDEN_STATE = {"M_t", "Mt", "semantic_state", "runtime_semantic_state"}
_FORBIDDEN_SECRET = {"password", "raw_password", "secret", "raw_secret", "api_key", "token", "access_token", "refresh_token", "private_key", "credential"}


class PermissionDecision(Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    NO_MATCH = "NO_MATCH"
    INVALID = "INVALID"


@dataclass(frozen=True)
class PermissionPolicy:
    default_decision: PermissionDecision = PermissionDecision.DENY
    deny_overrides_allow: bool = True
    require_authenticated_actor: bool = True
    max_permissions: int = 1024

    def __post_init__(self) -> None:
        if not isinstance(self.default_decision, PermissionDecision):
            raise TypeError("default_decision must be PermissionDecision")
        if not isinstance(self.deny_overrides_allow, bool):
            raise TypeError("deny_overrides_allow must be bool")
        if not isinstance(self.require_authenticated_actor, bool):
            raise TypeError("require_authenticated_actor must be bool")
        object.__setattr__(self, "max_permissions", ensure_positive_int(self.max_permissions, "max_permissions"))

    def to_dict(self) -> dict[str, object]:
        return {"default_decision": self.default_decision.value, "deny_overrides_allow": self.deny_overrides_allow, "require_authenticated_actor": self.require_authenticated_actor, "max_permissions": self.max_permissions}


@dataclass(frozen=True)
class PermissionRequest:
    envelope: SecurityEnvelope
    resource_key: str
    resource_kind: SecurityResourceKind
    action: str
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.envelope, SecurityEnvelope):
            raise TypeError("envelope must be SecurityEnvelope")
        resource_key = ensure_non_empty_string(self.resource_key, "resource_key")
        action = ensure_non_empty_string(self.action, "action")
        _reject_name(resource_key, "resource_key")
        _reject_name(action, "action")
        if not isinstance(self.resource_kind, SecurityResourceKind):
            raise TypeError("resource_kind must be SecurityResourceKind")
        object.__setattr__(self, "resource_key", resource_key)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))


@dataclass(frozen=True)
class PermissionResult:
    decision: PermissionDecision
    permission: SecurityPermission | None
    report: SecurityValidationReport
    changed: bool
    reason: str
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.decision, PermissionDecision):
            raise TypeError("decision must be PermissionDecision")
        if self.permission is not None and not isinstance(self.permission, SecurityPermission):
            raise TypeError("permission must be None or SecurityPermission")
        if not isinstance(self.report, SecurityValidationReport):
            raise TypeError("report must be SecurityValidationReport")
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        object.__setattr__(self, "reason", ensure_non_empty_string(self.reason, "reason"))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {"decision": self.decision.value, "permission": None if self.permission is None else self.permission.to_dict(), "report": self.report.to_dict(), "changed": self.changed, "reason": self.reason, "metadata": dict(self.metadata)}

    def is_allowed(self) -> bool:
        return self.decision is PermissionDecision.ALLOW and self.report.is_valid()


class PermissionManager:
    def __init__(self, *, policy: PermissionPolicy | None = None) -> None:
        self._policy = PermissionPolicy() if policy is None else policy
        if not isinstance(self._policy, PermissionPolicy):
            raise TypeError("policy must be PermissionPolicy")

    def evaluate(self, *, request: PermissionRequest) -> PermissionResult:
        if not isinstance(request, PermissionRequest):
            raise TypeError("request must be PermissionRequest")
        issues = []
        if self._policy.require_authenticated_actor and not request.envelope.actor.is_authenticated():
            issues.append(create_security_validation_issue(code="PERMISSION_UNAUTHENTICATED", message="authenticated actor required", risk_level=SecurityRiskLevel.HIGH))
        if len(request.envelope.permissions) > self._policy.max_permissions:
            issues.append(create_security_validation_issue(code="PERMISSION_LIMIT", message="permission limit exceeded", risk_level=SecurityRiskLevel.HIGH))
        matches = [p for p in request.envelope.permissions if p.resource.resource_key == request.resource_key and request.action in p.actions]
        denied = [p for p in matches if p.effect is SecurityPermissionEffect.DENY]
        allowed = [p for p in matches if p.effect is SecurityPermissionEffect.ALLOW]
        permission = None
        decision = self._policy.default_decision
        if denied and self._policy.deny_overrides_allow:
            decision, permission = PermissionDecision.DENY, denied[0]
        elif allowed:
            decision, permission = PermissionDecision.ALLOW, allowed[0]
        elif denied:
            decision, permission = PermissionDecision.DENY, denied[0]
        elif not matches:
            decision = PermissionDecision.NO_MATCH if self._policy.default_decision is PermissionDecision.NO_MATCH else self._policy.default_decision
        status = SecurityValidationStatus.VALID if not issues and decision is PermissionDecision.ALLOW else SecurityValidationStatus.BLOCKED
        report = create_security_validation_report(status=status, issues=tuple(issues))
        return PermissionResult(decision, permission, report, False, "permission allowed" if decision is PermissionDecision.ALLOW and not issues else "permission denied", {"source": "permission_result"})

    def grant(self, *, actor: SecurityActor, resource_key: str, resource_kind: SecurityResourceKind, actions: tuple[str, ...] | list[str], risk_level: SecurityRiskLevel = SecurityRiskLevel.LOW, metadata: dict[str, object] | None = None) -> SecurityPermission:
        return self._make_permission(actor=actor, resource_key=resource_key, resource_kind=resource_kind, actions=actions, risk_level=risk_level, effect=SecurityPermissionEffect.ALLOW, metadata=metadata)

    def deny(self, *, actor: SecurityActor, resource_key: str, resource_kind: SecurityResourceKind, actions: tuple[str, ...] | list[str], risk_level: SecurityRiskLevel = SecurityRiskLevel.LOW, metadata: dict[str, object] | None = None) -> SecurityPermission:
        return self._make_permission(actor=actor, resource_key=resource_key, resource_kind=resource_kind, actions=actions, risk_level=risk_level, effect=SecurityPermissionEffect.DENY, metadata=metadata)

    def _make_permission(self, *, actor: SecurityActor, resource_key: str, resource_kind: SecurityResourceKind, actions: tuple[str, ...] | list[str], risk_level: SecurityRiskLevel, effect: SecurityPermissionEffect, metadata: dict[str, object] | None) -> SecurityPermission:
        if not isinstance(actor, SecurityActor):
            raise TypeError("actor must be SecurityActor")
        resource = create_security_resource_ref(resource_key=resource_key, kind=resource_kind)
        return create_security_permission(actor=actor, resource=resource, effect=effect, actions=tuple(actions), risk_level=risk_level, metadata=metadata)


def build_permission_manager(*, policy: PermissionPolicy | None = None) -> PermissionManager:
    return PermissionManager(policy=policy)


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
