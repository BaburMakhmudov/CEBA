"""Deterministic local health check descriptors for CEBA observability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from ceba.observability.logging import ensure_non_empty_string, ensure_positive_int, ensure_timezone_aware, utc_now


__all__ = [
    "HealthStatus",
    "HealthComponentKind",
    "HealthCheck",
    "HealthReport",
    "HealthPolicy",
    "HealthMonitorResult",
    "HealthMonitor",
    "new_health_check_id",
    "create_health_check",
    "create_health_report",
    "build_health_monitor",
]

_FORBIDDEN_STATE = {"M_t", "Mt", "semantic_state", "runtime_semantic_state"}
_FORBIDDEN_SECRET = {"password", "raw_password", "secret", "raw_secret", "api_key", "token", "access_token", "refresh_token", "private_key", "credential"}


class HealthStatus(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    UNKNOWN = "UNKNOWN"
    INVALID = "INVALID"


class HealthComponentKind(Enum):
    API = "API"
    RUNTIME = "RUNTIME"
    WORLD_MODEL = "WORLD_MODEL"
    MEMORY = "MEMORY"
    PERSISTENCE = "PERSISTENCE"
    LEDGER = "LEDGER"
    VECTOR_SURFACE = "VECTOR_SURFACE"
    GRAPH_SURFACE = "GRAPH_SURFACE"
    SECURITY = "SECURITY"
    AUDIT = "AUDIT"
    OBSERVABILITY = "OBSERVABILITY"
    CONFIG = "CONFIG"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class HealthCheck:
    check_id: UUID
    component: str
    kind: HealthComponentKind
    status: HealthStatus
    message: str
    checked_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.check_id, UUID):
            raise TypeError("check_id must be UUID")
        component = ensure_non_empty_string(self.component, "component")
        _reject_name(component, "component")
        if not isinstance(self.kind, HealthComponentKind):
            raise TypeError("kind must be HealthComponentKind")
        if not isinstance(self.status, HealthStatus):
            raise TypeError("status must be HealthStatus")
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "message", ensure_non_empty_string(self.message, "message"))
        object.__setattr__(self, "checked_at", ensure_timezone_aware(self.checked_at, "checked_at"))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {"check_id": str(self.check_id), "component": self.component, "kind": self.kind.value, "status": self.status.value, "message": self.message, "checked_at": self.checked_at.isoformat(), "metadata": dict(self.metadata)}

    def is_healthy(self) -> bool:
        return self.status is HealthStatus.HEALTHY


@dataclass(frozen=True)
class HealthReport:
    status: HealthStatus
    checks: tuple[HealthCheck, ...]
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        checks = _check_tuple(self.checks)
        if not isinstance(self.status, HealthStatus):
            raise TypeError("status must be HealthStatus")
        expected = _worst_status(checks)
        if self.status is not expected:
            raise ValueError("status must be worst status from checks")
        object.__setattr__(self, "checks", checks)
        object.__setattr__(self, "created_at", ensure_timezone_aware(self.created_at, "created_at"))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {"status": self.status.value, "checks": [check.to_dict() for check in self.checks], "created_at": self.created_at.isoformat(), "metadata": dict(self.metadata)}

    def check_count(self) -> int:
        return len(self.checks)

    def unhealthy_count(self) -> int:
        return sum(1 for check in self.checks if check.status is HealthStatus.UNHEALTHY)

    def degraded_count(self) -> int:
        return sum(1 for check in self.checks if check.status is HealthStatus.DEGRADED)

    def is_healthy(self) -> bool:
        return self.status is HealthStatus.HEALTHY


@dataclass(frozen=True)
class HealthPolicy:
    max_checks: int = 10000
    allow_unknown: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_checks", ensure_positive_int(self.max_checks, "max_checks"))
        if not isinstance(self.allow_unknown, bool):
            raise TypeError("allow_unknown must be bool")


@dataclass(frozen=True)
class HealthMonitorResult:
    report: HealthReport
    changed: bool
    reason: str
    rejected_check_ids: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.report, HealthReport):
            raise TypeError("report must be HealthReport")
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        rejected = _string_tuple(self.rejected_check_ids, "rejected_check_ids")
        reasons = _string_tuple(self.rejection_reasons, "rejection_reasons")
        if len(rejected) != len(reasons):
            raise ValueError("rejected_check_ids and rejection_reasons must have same length")
        object.__setattr__(self, "reason", ensure_non_empty_string(self.reason, "reason"))
        object.__setattr__(self, "rejected_check_ids", rejected)
        object.__setattr__(self, "rejection_reasons", reasons)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))


class HealthMonitor:
    def __init__(self, *, checks: tuple[HealthCheck, ...] | list[HealthCheck] = (), policy: HealthPolicy | None = None) -> None:
        self._policy = HealthPolicy() if policy is None else policy
        if not isinstance(self._policy, HealthPolicy):
            raise TypeError("policy must be HealthPolicy")
        self._checks = _check_tuple(checks)

    def checks(self) -> tuple[HealthCheck, ...]:
        return self._checks

    def add_check(self, *, check: HealthCheck) -> HealthMonitorResult:
        if not isinstance(check, HealthCheck):
            raise TypeError("check must be HealthCheck")
        if len(self._checks) >= self._policy.max_checks:
            return _result(self._checks, False, "health check limit exceeded", rejected=(str(check.check_id),), rejection_reasons=("health check limit exceeded",))
        if not self._policy.allow_unknown and check.status is HealthStatus.UNKNOWN:
            return _result(self._checks, False, "unknown health check rejected", rejected=(str(check.check_id),), rejection_reasons=("unknown health check rejected",))
        if any(existing.check_id == check.check_id for existing in self._checks):
            return _result(self._checks, False, "health check_id duplicated", rejected=(str(check.check_id),), rejection_reasons=("health check_id duplicated",))
        self._checks = self._checks + (check,)
        return _result((check,), True, "health check recorded")

    def report(self) -> HealthMonitorResult:
        return _result(self._checks, False, "health report created")

    def replace_checks(self, *, checks: tuple[HealthCheck, ...] | list[HealthCheck]) -> HealthMonitorResult:
        self._checks = _check_tuple(checks)
        return _result(self._checks, True, "health checks replaced")


def new_health_check_id() -> UUID:
    return uuid4()


def create_health_check(*, component: str, kind: HealthComponentKind = HealthComponentKind.UNKNOWN, status: HealthStatus = HealthStatus.UNKNOWN, message: str, metadata: dict[str, object] | None = None) -> HealthCheck:
    return HealthCheck(new_health_check_id(), component, kind, status, message, utc_now(), _merge_metadata({"source": "health_check"}, metadata))


def create_health_report(*, checks: tuple[HealthCheck, ...] | list[HealthCheck] = (), metadata: dict[str, object] | None = None) -> HealthReport:
    checks_tuple = _check_tuple(checks)
    return HealthReport(_worst_status(checks_tuple), checks_tuple, utc_now(), _merge_metadata({"source": "health_report"}, metadata))


def build_health_monitor(*, checks: tuple[HealthCheck, ...] | list[HealthCheck] = (), policy: HealthPolicy | None = None) -> HealthMonitor:
    return HealthMonitor(checks=checks, policy=policy)


def _copy_metadata(value: dict[str, object]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(value)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
        if key in _FORBIDDEN_STATE or key in _FORBIDDEN_SECRET:
            raise ValueError("metadata contains forbidden key")
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


def _check_tuple(value: tuple[HealthCheck, ...] | list[HealthCheck]) -> tuple[HealthCheck, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("checks must be tuple[HealthCheck, ...]")
    checks = tuple(value)
    ids = [check.check_id for check in checks if isinstance(check, HealthCheck)]
    if len(ids) != len(set(ids)):
        raise ValueError("check_id values must be unique")
    for check in checks:
        if not isinstance(check, HealthCheck):
            raise TypeError("every check must be HealthCheck")
    return checks


def _worst_status(checks: tuple[HealthCheck, ...]) -> HealthStatus:
    if any(check.status is HealthStatus.UNHEALTHY for check in checks):
        return HealthStatus.UNHEALTHY
    if any(check.status is HealthStatus.DEGRADED for check in checks):
        return HealthStatus.DEGRADED
    if any(check.status is HealthStatus.INVALID for check in checks):
        return HealthStatus.INVALID
    if any(check.status is HealthStatus.UNKNOWN for check in checks):
        return HealthStatus.UNKNOWN
    return HealthStatus.HEALTHY


def _string_tuple(value: tuple[str, ...] | list[str], field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple[str, ...]")
    return tuple(ensure_non_empty_string(item, field_name) for item in value)


def _result(checks: tuple[HealthCheck, ...], changed: bool, reason: str, rejected: tuple[str, ...] = (), rejection_reasons: tuple[str, ...] = ()) -> HealthMonitorResult:
    return HealthMonitorResult(create_health_report(checks=checks), changed, reason, rejected, rejection_reasons, {"source": "health_monitor_result"})
