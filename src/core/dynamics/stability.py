"""Deterministic stability assessment utilities for CEBA Core Dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ceba.core.dynamics.energy import EnergyState, EnergyStatus
from ceba.core.dynamics.gradient_flow import (
    GradientFlowResult,
    GradientFlowStatus,
    gradient_flow_result_point,
)
from ceba.core.dynamics.manifold import ManifoldPoint, ManifoldPointStatus
from ceba.core.dynamics.semantic_temperature import (
    SemanticTemperatureState,
    SemanticTemperatureStatus,
    transition_scale_from_temperature,
)
from ceba.core.metrics.schema import (
    ensure_constraint_probability,
    ensure_finite_number,
    ensure_non_negative_number,
    utc_now,
)


class StabilityStatus(Enum):
    STABLE = "STABLE"
    ACCEPTABLE = "ACCEPTABLE"
    DEGRADED = "DEGRADED"
    UNSTABLE = "UNSTABLE"
    BLOCKED = "BLOCKED"
    INVALID = "INVALID"


class StabilityRiskLevel(Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True, slots=True)
class StabilityPolicy:
    stable_score_threshold: float = 0.85
    acceptable_score_threshold: float = 0.60
    degraded_score_threshold: float = 0.35
    max_allowed_energy: float = 10.0
    max_allowed_step_norm: float = 1.0
    min_temperature_scale: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stable_score_threshold",
            ensure_constraint_probability(self.stable_score_threshold, "stable_score_threshold"),
        )
        object.__setattr__(
            self,
            "acceptable_score_threshold",
            ensure_constraint_probability(self.acceptable_score_threshold, "acceptable_score_threshold"),
        )
        object.__setattr__(
            self,
            "degraded_score_threshold",
            ensure_constraint_probability(self.degraded_score_threshold, "degraded_score_threshold"),
        )
        if self.stable_score_threshold < self.acceptable_score_threshold:
            raise ValueError("stable_score_threshold must be greater than or equal to acceptable_score_threshold")
        if self.acceptable_score_threshold < self.degraded_score_threshold:
            raise ValueError("acceptable_score_threshold must be greater than or equal to degraded_score_threshold")
        object.__setattr__(
            self,
            "max_allowed_energy",
            ensure_non_negative_number(self.max_allowed_energy, "max_allowed_energy"),
        )
        object.__setattr__(
            self,
            "max_allowed_step_norm",
            ensure_non_negative_number(self.max_allowed_step_norm, "max_allowed_step_norm"),
        )
        object.__setattr__(
            self,
            "min_temperature_scale",
            ensure_constraint_probability(self.min_temperature_scale, "min_temperature_scale"),
        )


@dataclass(frozen=True, slots=True)
class StabilityIssue:
    code: str
    message: str
    risk: StabilityRiskLevel
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _ensure_non_empty_string(self.code, "code"))
        object.__setattr__(self, "message", _ensure_non_empty_string(self.message, "message"))
        if not isinstance(self.risk, StabilityRiskLevel):
            raise TypeError("risk must be StabilityRiskLevel")
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "risk": self.risk.value,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class StabilityReport:
    status: StabilityStatus
    risk: StabilityRiskLevel
    score: float
    issues: tuple[StabilityIssue, ...]
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, StabilityStatus):
            raise TypeError("status must be StabilityStatus")
        if not isinstance(self.risk, StabilityRiskLevel):
            raise TypeError("risk must be StabilityRiskLevel")
        object.__setattr__(self, "score", ensure_constraint_probability(self.score, "score"))
        object.__setattr__(self, "issues", _ensure_issue_tuple(self.issues))
        object.__setattr__(self, "created_at", _ensure_timezone_aware(self.created_at, "created_at"))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    @property
    def passed(self) -> bool:
        return self.status in (StabilityStatus.STABLE, StabilityStatus.ACCEPTABLE)

    @property
    def blocked(self) -> bool:
        return self.status in (StabilityStatus.BLOCKED, StabilityStatus.INVALID)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "risk": self.risk.value,
            "score": self.score,
            "issues": tuple(item.to_dict() for item in self.issues),
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
            "passed": self.passed,
            "blocked": self.blocked,
        }


def issue(
    *,
    code: str,
    message: str,
    risk: StabilityRiskLevel,
    metadata: dict[str, object] | None = None,
) -> StabilityIssue:
    return StabilityIssue(
        code=code,
        message=message,
        risk=risk,
        metadata={} if metadata is None else metadata,
    )


def risk_rank(risk: StabilityRiskLevel) -> int:
    if not isinstance(risk, StabilityRiskLevel):
        raise TypeError("risk must be StabilityRiskLevel")
    ranks: dict[StabilityRiskLevel, int] = {
        StabilityRiskLevel.NONE: 0,
        StabilityRiskLevel.LOW: 1,
        StabilityRiskLevel.MEDIUM: 2,
        StabilityRiskLevel.HIGH: 3,
        StabilityRiskLevel.CRITICAL: 4,
    }
    return ranks[risk]


def max_risk(
    risks: tuple[StabilityRiskLevel, ...] | list[StabilityRiskLevel],
) -> StabilityRiskLevel:
    if not isinstance(risks, (tuple, list)):
        raise TypeError("risks must be tuple[StabilityRiskLevel, ...] or list[StabilityRiskLevel]")
    if not risks:
        return StabilityRiskLevel.NONE
    checked_risks = tuple(risks)
    for index, risk in enumerate(checked_risks):
        if not isinstance(risk, StabilityRiskLevel):
            raise TypeError(f"risks[{index}] must be StabilityRiskLevel")
    return max(checked_risks, key=risk_rank)


def status_from_score_and_risk(
    *,
    score: float,
    risk: StabilityRiskLevel,
    policy: StabilityPolicy,
) -> StabilityStatus:
    checked_score = ensure_constraint_probability(score, "score")
    if not isinstance(risk, StabilityRiskLevel):
        raise TypeError("risk must be StabilityRiskLevel")
    if not isinstance(policy, StabilityPolicy):
        raise TypeError("policy must be StabilityPolicy")
    if risk is StabilityRiskLevel.CRITICAL:
        return StabilityStatus.BLOCKED
    if risk is StabilityRiskLevel.HIGH:
        return StabilityStatus.UNSTABLE
    if checked_score >= policy.stable_score_threshold:
        return StabilityStatus.STABLE
    if checked_score >= policy.acceptable_score_threshold:
        return StabilityStatus.ACCEPTABLE
    if checked_score >= policy.degraded_score_threshold:
        return StabilityStatus.DEGRADED
    return StabilityStatus.UNSTABLE


def stability_score_from_energy_temperature(
    *,
    energy: EnergyState | None,
    temperature: SemanticTemperatureState | None,
    policy: StabilityPolicy,
) -> float:
    if energy is not None and not isinstance(energy, EnergyState):
        raise TypeError("energy must be EnergyState or None")
    if temperature is not None and not isinstance(temperature, SemanticTemperatureState):
        raise TypeError("temperature must be SemanticTemperatureState or None")
    if not isinstance(policy, StabilityPolicy):
        raise TypeError("policy must be StabilityPolicy")

    energy_factor = 1.0 if energy is None else 1.0 / (1.0 + energy.energy)
    temperature_factor = 1.0 if temperature is None else transition_scale_from_temperature(temperature)
    score = ensure_finite_number(0.5 * energy_factor + 0.5 * temperature_factor, "score")
    return _clamp_probability(score)


def assess_manifold_point_stability(
    point: ManifoldPoint,
    *,
    policy: StabilityPolicy | None = None,
) -> StabilityReport:
    if not isinstance(point, ManifoldPoint):
        raise TypeError("point must be ManifoldPoint")
    active_policy = _ensure_policy(policy)
    issues: list[StabilityIssue] = []
    if point.status is ManifoldPointStatus.INVALID:
        issues.append(
            issue(
                code="point_invalid",
                message="Manifold point status is invalid.",
                risk=StabilityRiskLevel.CRITICAL,
            )
        )
        score = 0.0
    elif point.status is ManifoldPointStatus.DEGENERATE:
        issues.append(
            issue(
                code="point_degenerate",
                message="Manifold point is degenerate.",
                risk=StabilityRiskLevel.HIGH,
            )
        )
        score = 0.25
    else:
        score = 1.0

    risk = max_risk([item.risk for item in issues])
    return StabilityReport(
        status=status_from_score_and_risk(score=score, risk=risk, policy=active_policy),
        risk=risk,
        score=score,
        issues=tuple(issues),
        created_at=utc_now(),
        metadata={
            "assessment": "manifold_point",
            "point_id": point.point_id,
            "point_status": point.status.value,
            "dimension": point.dimension,
        },
    )


def assess_temperature_stability(
    temperature: SemanticTemperatureState,
    *,
    policy: StabilityPolicy | None = None,
) -> StabilityReport:
    if not isinstance(temperature, SemanticTemperatureState):
        raise TypeError("temperature must be SemanticTemperatureState")
    active_policy = _ensure_policy(policy)
    issues: list[StabilityIssue] = []
    if temperature.status is SemanticTemperatureStatus.FROZEN:
        issues.append(
            issue(code="temperature_frozen", message="Semantic temperature is frozen.", risk=StabilityRiskLevel.MEDIUM)
        )
    elif temperature.status is SemanticTemperatureStatus.UNSTABLE:
        issues.append(
            issue(
                code="temperature_unstable",
                message="Semantic temperature is unstable.",
                risk=StabilityRiskLevel.HIGH,
            )
        )
    elif temperature.status is SemanticTemperatureStatus.INVALID:
        issues.append(
            issue(
                code="temperature_invalid",
                message="Semantic temperature is invalid.",
                risk=StabilityRiskLevel.CRITICAL,
            )
        )

    transition_scale = transition_scale_from_temperature(temperature)
    if transition_scale < active_policy.min_temperature_scale:
        issues.append(
            issue(
                code="temperature_scale_low",
                message="Semantic temperature transition scale is below the configured minimum.",
                risk=StabilityRiskLevel.HIGH,
                metadata={"transition_scale": transition_scale},
            )
        )

    score = ensure_constraint_probability(temperature.stability_score, "score")
    risk = max_risk([item.risk for item in issues])
    return StabilityReport(
        status=status_from_score_and_risk(score=score, risk=risk, policy=active_policy),
        risk=risk,
        score=score,
        issues=tuple(issues),
        created_at=utc_now(),
        metadata={
            "assessment": "semantic_temperature",
            "temperature": temperature.temperature,
            "temperature_status": temperature.status.value,
            "temperature_mode": temperature.mode.value,
            "transition_scale": transition_scale,
        },
    )


def assess_energy_stability(
    energy: EnergyState,
    *,
    policy: StabilityPolicy | None = None,
) -> StabilityReport:
    if not isinstance(energy, EnergyState):
        raise TypeError("energy must be EnergyState")
    active_policy = _ensure_policy(policy)
    issues: list[StabilityIssue] = []
    if energy.status is EnergyStatus.INVALID:
        issues.append(
            issue(code="energy_invalid", message="Energy state is invalid.", risk=StabilityRiskLevel.CRITICAL)
        )
    elif energy.status is EnergyStatus.DEGRADED:
        issues.append(issue(code="energy_degraded", message="Energy state is degraded.", risk=StabilityRiskLevel.HIGH))
    if energy.energy > active_policy.max_allowed_energy:
        issues.append(
            issue(
                code="energy_too_high",
                message="Energy exceeds the configured maximum.",
                risk=StabilityRiskLevel.HIGH,
                metadata={"max_allowed_energy": active_policy.max_allowed_energy},
            )
        )

    if energy.normalized_energy is not None:
        score = _clamp_probability(1.0 - energy.normalized_energy)
    else:
        score = _clamp_probability(1.0 / (1.0 + energy.energy))
    risk = max_risk([item.risk for item in issues])
    return StabilityReport(
        status=status_from_score_and_risk(score=score, risk=risk, policy=active_policy),
        risk=risk,
        score=score,
        issues=tuple(issues),
        created_at=utc_now(),
        metadata={
            "assessment": "energy_state",
            "energy": energy.energy,
            "normalized_energy": energy.normalized_energy,
            "energy_status": energy.status.value,
        },
    )


def assess_gradient_flow_stability(
    result: GradientFlowResult,
    *,
    policy: StabilityPolicy | None = None,
) -> StabilityReport:
    if not isinstance(result, GradientFlowResult):
        raise TypeError("result must be GradientFlowResult")
    active_policy = _ensure_policy(policy)
    issues: list[StabilityIssue] = []
    if result.status is GradientFlowStatus.BLOCKED:
        issues.append(
            issue(code="gradient_blocked", message="Gradient flow result is blocked.", risk=StabilityRiskLevel.CRITICAL)
        )
    elif result.status is GradientFlowStatus.DEGRADED:
        issues.append(
            issue(code="gradient_degraded", message="Gradient flow result is degraded.", risk=StabilityRiskLevel.HIGH)
        )
    elif result.status is GradientFlowStatus.INVALID:
        issues.append(
            issue(code="gradient_invalid", message="Gradient flow result is invalid.", risk=StabilityRiskLevel.CRITICAL)
        )
    if result.step.step_size > active_policy.max_allowed_step_norm:
        issues.append(
            issue(
                code="gradient_step_too_large",
                message="Gradient flow step size exceeds the configured maximum.",
                risk=StabilityRiskLevel.HIGH,
                metadata={"max_allowed_step_norm": active_policy.max_allowed_step_norm},
            )
        )

    score = _gradient_score(result.status)
    risk = max_risk([item.risk for item in issues])
    result_point = gradient_flow_result_point(result)
    return StabilityReport(
        status=status_from_score_and_risk(score=score, risk=risk, policy=active_policy),
        risk=risk,
        score=score,
        issues=tuple(issues),
        created_at=utc_now(),
        metadata={
            "assessment": "gradient_flow",
            "flow_status": result.status.value,
            "step_size": result.step.step_size,
            "result_point_id": result_point.point_id,
        },
    )


def combine_stability_reports(
    reports: tuple[StabilityReport, ...] | list[StabilityReport],
    *,
    policy: StabilityPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> StabilityReport:
    checked_reports = _ensure_report_sequence(reports)
    active_policy = _ensure_policy(policy)
    score = (
        ensure_constraint_probability(sum(report.score for report in checked_reports) / len(checked_reports), "score")
        if checked_reports
        else 1.0
    )
    risk = max_risk([report.risk for report in checked_reports])
    issues: list[StabilityIssue] = []
    for report in checked_reports:
        issues.extend(report.issues)

    report_metadata: dict[str, object] = {
        "assessment": "combined",
        "report_count": len(checked_reports),
    }
    if metadata is not None:
        report_metadata.update(_copy_metadata(metadata))
    report_metadata["assessment"] = "combined"

    return StabilityReport(
        status=status_from_score_and_risk(score=score, risk=risk, policy=active_policy),
        risk=risk,
        score=score,
        issues=tuple(issues),
        created_at=utc_now(),
        metadata=report_metadata,
    )


def assess_transition_readiness(
    *,
    point: ManifoldPoint,
    energy: EnergyState | None = None,
    temperature: SemanticTemperatureState | None = None,
    gradient_result: GradientFlowResult | None = None,
    policy: StabilityPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> StabilityReport:
    if not isinstance(point, ManifoldPoint):
        raise TypeError("point must be ManifoldPoint")
    if energy is not None and not isinstance(energy, EnergyState):
        raise TypeError("energy must be EnergyState or None")
    if temperature is not None and not isinstance(temperature, SemanticTemperatureState):
        raise TypeError("temperature must be SemanticTemperatureState or None")
    if gradient_result is not None and not isinstance(gradient_result, GradientFlowResult):
        raise TypeError("gradient_result must be GradientFlowResult or None")
    active_policy = _ensure_policy(policy)

    reports: list[StabilityReport] = [assess_manifold_point_stability(point, policy=active_policy)]
    if energy is not None:
        reports.append(assess_energy_stability(energy, policy=active_policy))
    if temperature is not None:
        reports.append(assess_temperature_stability(temperature, policy=active_policy))
    if gradient_result is not None:
        reports.append(assess_gradient_flow_stability(gradient_result, policy=active_policy))

    report_metadata: dict[str, object] = {
        "assessment": "transition_readiness",
        "has_energy": energy is not None,
        "has_temperature": temperature is not None,
        "has_gradient_result": gradient_result is not None,
    }
    if metadata is not None:
        report_metadata.update(_copy_metadata(metadata))
    report_metadata["assessment"] = "transition_readiness"

    combined = combine_stability_reports(reports, policy=active_policy, metadata=report_metadata)
    return StabilityReport(
        status=combined.status,
        risk=combined.risk,
        score=combined.score,
        issues=combined.issues,
        created_at=combined.created_at,
        metadata=report_metadata,
    )


def stability_report_is_passed(report: StabilityReport) -> bool:
    if not isinstance(report, StabilityReport):
        raise TypeError("report must be StabilityReport")
    return report.passed


def stability_report_is_blocked(report: StabilityReport) -> bool:
    if not isinstance(report, StabilityReport):
        raise TypeError("report must be StabilityReport")
    return report.blocked


def _gradient_score(status: GradientFlowStatus) -> float:
    if status is GradientFlowStatus.APPLIED:
        return 1.0
    if status is GradientFlowStatus.SKIPPED:
        return 0.8
    if status is GradientFlowStatus.DEGRADED:
        return 0.5
    if status in (GradientFlowStatus.BLOCKED, GradientFlowStatus.INVALID):
        return 0.0
    return 0.8


def _ensure_policy(policy: StabilityPolicy | None) -> StabilityPolicy:
    if policy is None:
        return StabilityPolicy()
    if not isinstance(policy, StabilityPolicy):
        raise TypeError("policy must be StabilityPolicy or None")
    return policy


def _ensure_issue_tuple(value: tuple[StabilityIssue, ...] | list[StabilityIssue]) -> tuple[StabilityIssue, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("issues must be tuple[StabilityIssue, ...] or list[StabilityIssue]")
    checked_issues = tuple(value)
    for index, item in enumerate(checked_issues):
        if not isinstance(item, StabilityIssue):
            raise TypeError(f"issues[{index}] must be StabilityIssue")
    return checked_issues


def _ensure_report_sequence(
    reports: tuple[StabilityReport, ...] | list[StabilityReport],
) -> tuple[StabilityReport, ...]:
    if not isinstance(reports, (tuple, list)):
        raise TypeError("reports must be tuple[StabilityReport, ...] or list[StabilityReport]")
    checked_reports = tuple(reports)
    for index, report in enumerate(checked_reports):
        if not isinstance(report, StabilityReport):
            raise TypeError(f"reports[{index}] must be StabilityReport")
    return checked_reports


def _clamp_probability(value: float) -> float:
    number = ensure_finite_number(value, "value")
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return ensure_constraint_probability(number, "value")


def _ensure_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _ensure_timezone_aware(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _copy_metadata(value: dict[str, object]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("metadata must be dict[str, object]")
    if any(not isinstance(key, str) for key in value):
        raise TypeError("metadata keys must be strings")
    return dict(value)


__all__ = [
    "StabilityStatus",
    "StabilityRiskLevel",
    "StabilityPolicy",
    "StabilityIssue",
    "StabilityReport",
    "issue",
    "risk_rank",
    "max_risk",
    "status_from_score_and_risk",
    "stability_score_from_energy_temperature",
    "assess_manifold_point_stability",
    "assess_temperature_stability",
    "assess_energy_stability",
    "assess_gradient_flow_stability",
    "combine_stability_reports",
    "assess_transition_readiness",
    "stability_report_is_passed",
    "stability_report_is_blocked",
]
