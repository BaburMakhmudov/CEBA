"""Deterministic local diagnostic descriptors for CEBA observability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from ceba.observability.health import HealthCheck, HealthReport, HealthStatus
from ceba.observability.logging import ensure_non_empty_string, ensure_positive_int, ensure_timezone_aware, utc_now
from ceba.observability.metrics import MetricSample, MetricStatus
from ceba.observability.tracing import TraceContext, TraceSpanStatus


__all__ = [
    "DiagnosticSeverity",
    "DiagnosticStatus",
    "DiagnosticFindingKind",
    "DiagnosticFinding",
    "DiagnosticReport",
    "DiagnosticsPolicy",
    "DiagnosticsResult",
    "DiagnosticsRunner",
    "new_diagnostic_finding_id",
    "create_diagnostic_finding",
    "create_diagnostic_report",
    "build_diagnostics_runner",
]

_FORBIDDEN_STATE = {"M_t", "Mt", "semantic_state", "runtime_semantic_state"}
_FORBIDDEN_SECRET = {"password", "raw_password", "secret", "raw_secret", "api_key", "token", "access_token", "refresh_token", "private_key", "credential"}


class DiagnosticSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class DiagnosticStatus(Enum):
    CREATED = "CREATED"
    RECORDED = "RECORDED"
    RESOLVED = "RESOLVED"
    IGNORED = "IGNORED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class DiagnosticFindingKind(Enum):
    HEALTH = "HEALTH"
    METRIC = "METRIC"
    TRACE = "TRACE"
    LOG = "LOG"
    CONFIG = "CONFIG"
    SECURITY = "SECURITY"
    PERFORMANCE = "PERFORMANCE"
    INVARIANT = "INVARIANT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class DiagnosticFinding:
    finding_id: UUID
    kind: DiagnosticFindingKind
    severity: DiagnosticSeverity
    status: DiagnosticStatus
    component: str
    message: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.finding_id, UUID):
            raise TypeError("finding_id must be UUID")
        if not isinstance(self.kind, DiagnosticFindingKind):
            raise TypeError("kind must be DiagnosticFindingKind")
        if not isinstance(self.severity, DiagnosticSeverity):
            raise TypeError("severity must be DiagnosticSeverity")
        if not isinstance(self.status, DiagnosticStatus):
            raise TypeError("status must be DiagnosticStatus")
        component = ensure_non_empty_string(self.component, "component")
        _reject_name(component, "component")
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "message", ensure_non_empty_string(self.message, "message"))
        object.__setattr__(self, "created_at", ensure_timezone_aware(self.created_at, "created_at"))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {"finding_id": str(self.finding_id), "kind": self.kind.value, "severity": self.severity.value, "status": self.status.value, "component": self.component, "message": self.message, "created_at": self.created_at.isoformat(), "metadata": dict(self.metadata)}

    def is_error(self) -> bool:
        return self.severity in {DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL}


@dataclass(frozen=True)
class DiagnosticReport:
    findings: tuple[DiagnosticFinding, ...]
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "findings", _finding_tuple(self.findings))
        object.__setattr__(self, "created_at", ensure_timezone_aware(self.created_at, "created_at"))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {"findings": [finding.to_dict() for finding in self.findings], "created_at": self.created_at.isoformat(), "metadata": dict(self.metadata)}

    def finding_count(self) -> int:
        return len(self.findings)

    def error_count(self) -> int:
        return sum(1 for finding in self.findings if finding.is_error())

    def critical_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity is DiagnosticSeverity.CRITICAL)

    def has_errors(self) -> bool:
        return self.error_count() > 0


@dataclass(frozen=True)
class DiagnosticsPolicy:
    max_findings: int = 10000
    include_healthy_findings: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_findings", ensure_positive_int(self.max_findings, "max_findings"))
        if not isinstance(self.include_healthy_findings, bool):
            raise TypeError("include_healthy_findings must be bool")


@dataclass(frozen=True)
class DiagnosticsResult:
    report: DiagnosticReport
    changed: bool
    reason: str
    rejected_finding_ids: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.report, DiagnosticReport):
            raise TypeError("report must be DiagnosticReport")
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        rejected = _string_tuple(self.rejected_finding_ids, "rejected_finding_ids")
        reasons = _string_tuple(self.rejection_reasons, "rejection_reasons")
        if len(rejected) != len(reasons):
            raise ValueError("rejected_finding_ids and rejection_reasons must have same length")
        object.__setattr__(self, "reason", ensure_non_empty_string(self.reason, "reason"))
        object.__setattr__(self, "rejected_finding_ids", rejected)
        object.__setattr__(self, "rejection_reasons", reasons)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))


class DiagnosticsRunner:
    def __init__(self, *, findings: tuple[DiagnosticFinding, ...] | list[DiagnosticFinding] = (), policy: DiagnosticsPolicy | None = None) -> None:
        self._policy = DiagnosticsPolicy() if policy is None else policy
        if not isinstance(self._policy, DiagnosticsPolicy):
            raise TypeError("policy must be DiagnosticsPolicy")
        self._findings = _finding_tuple(findings)

    def findings(self) -> tuple[DiagnosticFinding, ...]:
        return self._findings

    def add_finding(self, *, finding: DiagnosticFinding) -> DiagnosticsResult:
        if not isinstance(finding, DiagnosticFinding):
            raise TypeError("finding must be DiagnosticFinding")
        if len(self._findings) >= self._policy.max_findings:
            return _result(self._findings, False, "diagnostic finding limit exceeded", rejected=(str(finding.finding_id),), rejection_reasons=("diagnostic finding limit exceeded",))
        if any(existing.finding_id == finding.finding_id for existing in self._findings):
            return _result(self._findings, False, "diagnostic finding_id duplicated", rejected=(str(finding.finding_id),), rejection_reasons=("diagnostic finding_id duplicated",))
        recorded = DiagnosticFinding(finding.finding_id, finding.kind, finding.severity, DiagnosticStatus.RECORDED, finding.component, finding.message, finding.created_at, finding.metadata)
        self._findings = self._findings + (recorded,)
        return _result((recorded,), True, "diagnostic finding recorded")

    def from_health_report(self, *, health_report: HealthReport) -> DiagnosticsResult:
        if not isinstance(health_report, HealthReport):
            raise TypeError("health_report must be HealthReport")
        findings = []
        for check in health_report.checks:
            if check.status is HealthStatus.HEALTHY and not self._policy.include_healthy_findings:
                continue
            if check.status in {HealthStatus.DEGRADED, HealthStatus.UNHEALTHY, HealthStatus.INVALID, HealthStatus.UNKNOWN} or self._policy.include_healthy_findings:
                findings.append(_finding_from_check(check))
        return self._add_many(tuple(findings), "health diagnostics recorded")

    def from_metric_samples(self, *, samples: tuple[MetricSample, ...] | list[MetricSample]) -> DiagnosticsResult:
        samples_tuple = _sample_tuple(samples)
        findings = tuple(create_diagnostic_finding(kind=DiagnosticFindingKind.METRIC, severity=DiagnosticSeverity.WARNING, component=sample.name, message=f"metric status {sample.status.value}", metadata={"sample_id": str(sample.sample_id)}) for sample in samples_tuple if sample.status in {MetricStatus.UNKNOWN, MetricStatus.INVALID})
        return self._add_many(findings, "metric diagnostics recorded")

    def from_trace_context(self, *, trace_context: TraceContext) -> DiagnosticsResult:
        if not isinstance(trace_context, TraceContext):
            raise TypeError("trace_context must be TraceContext")
        findings = tuple(create_diagnostic_finding(kind=DiagnosticFindingKind.TRACE, severity=DiagnosticSeverity.ERROR, component=span.name, message=f"trace span status {span.status.value}", metadata={"span_id": str(span.span_id)}) for span in trace_context.spans if span.status in {TraceSpanStatus.FAILED, TraceSpanStatus.INVALID})
        return self._add_many(findings, "trace diagnostics recorded")

    def report(self) -> DiagnosticsResult:
        return _result(self._findings, False, "diagnostic report created")

    def replace_findings(self, *, findings: tuple[DiagnosticFinding, ...] | list[DiagnosticFinding]) -> DiagnosticsResult:
        self._findings = _finding_tuple(findings)
        return _result(self._findings, True, "diagnostic findings replaced")

    def _add_many(self, findings: tuple[DiagnosticFinding, ...], reason: str) -> DiagnosticsResult:
        accepted = []
        rejected = []
        rejection_reasons = []
        for finding in findings:
            if len(self._findings) + len(accepted) >= self._policy.max_findings:
                rejected.append(str(finding.finding_id))
                rejection_reasons.append("diagnostic finding limit exceeded")
            elif any(existing.finding_id == finding.finding_id for existing in self._findings + tuple(accepted)):
                rejected.append(str(finding.finding_id))
                rejection_reasons.append("diagnostic finding_id duplicated")
            else:
                accepted.append(DiagnosticFinding(finding.finding_id, finding.kind, finding.severity, DiagnosticStatus.RECORDED, finding.component, finding.message, finding.created_at, finding.metadata))
        self._findings = self._findings + tuple(accepted)
        return DiagnosticsResult(create_diagnostic_report(findings=tuple(accepted)), bool(accepted), reason, tuple(rejected), tuple(rejection_reasons), {"source": "diagnostics_result"})


def new_diagnostic_finding_id() -> UUID:
    return uuid4()


def create_diagnostic_finding(*, kind: DiagnosticFindingKind = DiagnosticFindingKind.UNKNOWN, severity: DiagnosticSeverity = DiagnosticSeverity.INFO, status: DiagnosticStatus = DiagnosticStatus.CREATED, component: str, message: str, metadata: dict[str, object] | None = None) -> DiagnosticFinding:
    return DiagnosticFinding(new_diagnostic_finding_id(), kind, severity, status, component, message, utc_now(), _merge_metadata({"source": "diagnostic_finding"}, metadata))


def create_diagnostic_report(*, findings: tuple[DiagnosticFinding, ...] | list[DiagnosticFinding] = (), metadata: dict[str, object] | None = None) -> DiagnosticReport:
    return DiagnosticReport(tuple(findings), utc_now(), _merge_metadata({"source": "diagnostic_report"}, metadata))


def build_diagnostics_runner(*, findings: tuple[DiagnosticFinding, ...] | list[DiagnosticFinding] = (), policy: DiagnosticsPolicy | None = None) -> DiagnosticsRunner:
    return DiagnosticsRunner(findings=findings, policy=policy)


def _finding_from_check(check: HealthCheck) -> DiagnosticFinding:
    severity = DiagnosticSeverity.INFO
    if check.status is HealthStatus.DEGRADED:
        severity = DiagnosticSeverity.WARNING
    elif check.status is HealthStatus.UNHEALTHY:
        severity = DiagnosticSeverity.ERROR
    elif check.status is HealthStatus.INVALID:
        severity = DiagnosticSeverity.CRITICAL
    elif check.status is HealthStatus.UNKNOWN:
        severity = DiagnosticSeverity.WARNING
    return create_diagnostic_finding(kind=DiagnosticFindingKind.HEALTH, severity=severity, component=check.component, message=check.message, metadata={"check_id": str(check.check_id)})


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


def _finding_tuple(value: tuple[DiagnosticFinding, ...] | list[DiagnosticFinding]) -> tuple[DiagnosticFinding, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("findings must be tuple[DiagnosticFinding, ...]")
    findings = tuple(value)
    ids = [finding.finding_id for finding in findings if isinstance(finding, DiagnosticFinding)]
    if len(ids) != len(set(ids)):
        raise ValueError("finding_id values must be unique")
    for finding in findings:
        if not isinstance(finding, DiagnosticFinding):
            raise TypeError("every finding must be DiagnosticFinding")
    return findings


def _sample_tuple(value: tuple[MetricSample, ...] | list[MetricSample]) -> tuple[MetricSample, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("samples must be tuple[MetricSample, ...]")
    samples = tuple(value)
    for sample in samples:
        if not isinstance(sample, MetricSample):
            raise TypeError("every sample must be MetricSample")
    return samples


def _string_tuple(value: tuple[str, ...] | list[str], field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple[str, ...]")
    return tuple(ensure_non_empty_string(item, field_name) for item in value)


def _result(findings: tuple[DiagnosticFinding, ...], changed: bool, reason: str, rejected: tuple[str, ...] = (), rejection_reasons: tuple[str, ...] = ()) -> DiagnosticsResult:
    return DiagnosticsResult(create_diagnostic_report(findings=findings), changed, reason, rejected, rejection_reasons, {"source": "diagnostics_result"})
