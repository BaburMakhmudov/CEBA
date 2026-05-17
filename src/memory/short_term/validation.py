"""Deterministic validation utilities for CEBA Short-Term Memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar
from uuid import UUID

from ceba.memory.short_term.schema import (
    ActiveContextRef,
    ActiveContextStatus,
    CandidateTrace,
    CandidateTraceStatus,
    EvictionRecord,
    HypothesisStatus,
    ShortTermItem,
    ShortTermItemStatus,
    ShortTermMemorySnapshot,
    WorkingHypothesis,
    ensure_non_empty_string,
    utc_now,
)


__all__ = [
    "ShortTermValidationIssue",
    "ShortTermValidationReport",
    "ShortTermValidator",
    "validate_short_term_snapshot",
    "validate_short_term_components",
]


_VALID_SEVERITIES = ("INFO", "WARNING", "ERROR")
_T = TypeVar("_T")


@dataclass(frozen=True)
class ShortTermValidationIssue:
    """Immutable validation issue descriptor."""

    code: str
    message: str
    severity: str
    object_id: str | None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        code = ensure_non_empty_string(self.code, "code")
        message = ensure_non_empty_string(self.message, "message")
        severity = ensure_non_empty_string(self.severity, "severity")
        if severity not in _VALID_SEVERITIES:
            raise ValueError("severity must be INFO, WARNING, or ERROR")
        object_id = (
            None
            if self.object_id is None
            else ensure_non_empty_string(self.object_id, "object_id")
        )
        metadata = _copy_metadata(self.metadata)

        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "object_id": self.object_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ShortTermValidationReport:
    """Immutable validation report."""

    valid: bool
    issues: tuple[ShortTermValidationIssue, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be bool")
        issues = _ensure_issues(self.issues)
        has_error = any(issue.severity == "ERROR" for issue in issues)
        if has_error and self.valid:
            raise ValueError("valid must be False when ERROR issues exist")
        object.__setattr__(self, "issues", issues)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "valid": self.valid,
            "issues": tuple(issue.to_dict() for issue in self.issues),
        }

    def error_count(self) -> int:
        """Return ERROR issue count."""

        return sum(1 for issue in self.issues if issue.severity == "ERROR")

    def warning_count(self) -> int:
        """Return WARNING issue count."""

        return sum(1 for issue in self.issues if issue.severity == "WARNING")

    def info_count(self) -> int:
        """Return INFO issue count."""

        return sum(1 for issue in self.issues if issue.severity == "INFO")


class ShortTermValidator:
    """Deterministic validator for STM runtime structures."""

    def __init__(self) -> None:
        pass

    def validate_components(
        self,
        *,
        items: tuple[ShortTermItem, ...] | list[ShortTermItem] = (),
        contexts: tuple[ActiveContextRef, ...] | list[ActiveContextRef] = (),
        candidate_traces: tuple[CandidateTrace, ...] | list[CandidateTrace] = (),
        hypotheses: tuple[WorkingHypothesis, ...] | list[WorkingHypothesis] = (),
        evictions: tuple[EvictionRecord, ...] | list[EvictionRecord] = (),
    ) -> ShortTermValidationReport:
        """Validate STM component consistency."""

        item_values = _ensure_collection(items, ShortTermItem, "items")
        context_values = _ensure_collection(
            contexts,
            ActiveContextRef,
            "contexts",
        )
        trace_values = _ensure_collection(
            candidate_traces,
            CandidateTrace,
            "candidate_traces",
        )
        hypothesis_values = _ensure_collection(
            hypotheses,
            WorkingHypothesis,
            "hypotheses",
        )
        eviction_values = _ensure_collection(
            evictions,
            EvictionRecord,
            "evictions",
        )

        issues: list[ShortTermValidationIssue] = []
        self._add_empty_collection_info(
            issues,
            items=item_values,
            contexts=context_values,
            candidate_traces=trace_values,
            hypotheses=hypothesis_values,
        )
        item_ids = self._check_duplicate_ids(
            issues,
            ids=tuple(item.item_id for item in item_values),
            code="DUPLICATE_ITEM_ID",
            message="duplicate item_id",
            object_type="ShortTermItem",
        )
        self._check_duplicate_ids(
            issues,
            ids=tuple(context.context_id for context in context_values),
            code="DUPLICATE_CONTEXT_ID",
            message="duplicate context_id",
            object_type="ActiveContextRef",
        )
        self._check_duplicate_ids(
            issues,
            ids=tuple(trace.trace_id for trace in trace_values),
            code="DUPLICATE_TRACE_ID",
            message="duplicate trace_id",
            object_type="CandidateTrace",
        )
        self._check_duplicate_ids(
            issues,
            ids=tuple(
                hypothesis.hypothesis_id for hypothesis in hypothesis_values
            ),
            code="DUPLICATE_HYPOTHESIS_ID",
            message="duplicate hypothesis_id",
            object_type="WorkingHypothesis",
        )
        self._check_duplicate_ids(
            issues,
            ids=tuple(eviction.eviction_id for eviction in eviction_values),
            code="DUPLICATE_EVICTION_ID",
            message="duplicate eviction_id",
            object_type="EvictionRecord",
        )

        self._validate_context_references(issues, context_values, item_ids)
        self._validate_trace_references(issues, trace_values, item_ids)
        self._validate_hypothesis_references(
            issues,
            hypothesis_values,
            item_ids,
        )
        self._validate_eviction_references(issues, eviction_values, item_ids)
        self._validate_status_warnings(
            issues,
            item_values,
            context_values,
            trace_values,
            hypothesis_values,
        )

        valid = not any(issue.severity == "ERROR" for issue in issues)
        return ShortTermValidationReport(valid=valid, issues=tuple(issues))

    def validate_snapshot(
        self,
        *,
        snapshot: ShortTermMemorySnapshot,
    ) -> ShortTermValidationReport:
        """Validate a short-term memory snapshot."""

        if not isinstance(snapshot, ShortTermMemorySnapshot):
            raise TypeError("snapshot must be ShortTermMemorySnapshot")
        return self.validate_components(
            items=snapshot.items,
            contexts=snapshot.contexts,
            candidate_traces=snapshot.candidate_traces,
            hypotheses=snapshot.hypotheses,
            evictions=snapshot.evictions,
        )

    def _add_empty_collection_info(
        self,
        issues: list[ShortTermValidationIssue],
        *,
        items: tuple[ShortTermItem, ...],
        contexts: tuple[ActiveContextRef, ...],
        candidate_traces: tuple[CandidateTrace, ...],
        hypotheses: tuple[WorkingHypothesis, ...],
    ) -> None:
        if not items:
            issues.append(_issue("EMPTY_ITEMS", "empty items collection", "INFO"))
        if not contexts:
            issues.append(
                _issue("EMPTY_CONTEXTS", "empty contexts collection", "INFO")
            )
        if not candidate_traces:
            issues.append(
                _issue(
                    "EMPTY_CANDIDATE_TRACES",
                    "empty candidate_traces collection",
                    "INFO",
                )
            )
        if not hypotheses:
            issues.append(
                _issue(
                    "EMPTY_HYPOTHESES",
                    "empty hypotheses collection",
                    "INFO",
                )
            )

    def _check_duplicate_ids(
        self,
        issues: list[ShortTermValidationIssue],
        *,
        ids: tuple[UUID, ...],
        code: str,
        message: str,
        object_type: str,
    ) -> set[UUID]:
        seen: set[UUID] = set()
        duplicates: set[UUID] = set()
        for value in ids:
            if value in seen:
                duplicates.add(value)
            seen.add(value)
        for duplicate in sorted(duplicates, key=str):
            issues.append(
                _issue(
                    code,
                    message,
                    "ERROR",
                    object_id=str(duplicate),
                    metadata={"object_type": object_type},
                )
            )
        return seen

    def _validate_context_references(
        self,
        issues: list[ShortTermValidationIssue],
        contexts: tuple[ActiveContextRef, ...],
        item_ids: set[UUID],
    ) -> None:
        for context in contexts:
            for item_id in context.item_ids:
                if item_id not in item_ids:
                    issues.append(
                        _issue(
                            "CONTEXT_ITEM_MISSING",
                            "context item_id missing from items",
                            "ERROR",
                            object_id=str(context.context_id),
                            metadata={"item_id": str(item_id)},
                        )
                    )

    def _validate_trace_references(
        self,
        issues: list[ShortTermValidationIssue],
        traces: tuple[CandidateTrace, ...],
        item_ids: set[UUID],
    ) -> None:
        for trace in traces:
            for item_id in trace.item_ids:
                if item_id not in item_ids:
                    issues.append(
                        _issue(
                            "TRACE_ITEM_MISSING",
                            "candidate_trace item_id missing from items",
                            "ERROR",
                            object_id=str(trace.trace_id),
                            metadata={"item_id": str(item_id)},
                        )
                    )

    def _validate_hypothesis_references(
        self,
        issues: list[ShortTermValidationIssue],
        hypotheses: tuple[WorkingHypothesis, ...],
        item_ids: set[UUID],
    ) -> None:
        for hypothesis in hypotheses:
            for item_id in hypothesis.supporting_item_ids:
                if item_id not in item_ids:
                    issues.append(
                        _issue(
                            "HYPOTHESIS_SUPPORTING_ITEM_MISSING",
                            "hypothesis supporting item_id missing from items",
                            "ERROR",
                            object_id=str(hypothesis.hypothesis_id),
                            metadata={"item_id": str(item_id)},
                        )
                    )
            for item_id in hypothesis.contradicting_item_ids:
                if item_id not in item_ids:
                    issues.append(
                        _issue(
                            "HYPOTHESIS_CONTRADICTING_ITEM_MISSING",
                            "hypothesis contradicting item_id missing from items",
                            "ERROR",
                            object_id=str(hypothesis.hypothesis_id),
                            metadata={"item_id": str(item_id)},
                        )
                    )

    def _validate_eviction_references(
        self,
        issues: list[ShortTermValidationIssue],
        evictions: tuple[EvictionRecord, ...],
        item_ids: set[UUID],
    ) -> None:
        evicted_item_ids = {eviction.item_id for eviction in evictions}
        for eviction in evictions:
            if (
                eviction.item_id not in item_ids
                and eviction.item_id not in evicted_item_ids
            ):
                issues.append(
                    _issue(
                        "EVICTION_ITEM_MISSING",
                        "eviction item_id missing from items and evictions",
                        "ERROR",
                        object_id=str(eviction.eviction_id),
                        metadata={"item_id": str(eviction.item_id)},
                    )
                )

    def _validate_status_warnings(
        self,
        issues: list[ShortTermValidationIssue],
        items: tuple[ShortTermItem, ...],
        contexts: tuple[ActiveContextRef, ...],
        traces: tuple[CandidateTrace, ...],
        hypotheses: tuple[WorkingHypothesis, ...],
    ) -> None:
        now = utc_now()
        for item in items:
            if item.status is ShortTermItemStatus.INVALID:
                issues.append(
                    _issue(
                        "INVALID_ITEM_STATUS",
                        "ShortTermItem.status is INVALID",
                        "WARNING",
                        object_id=str(item.item_id),
                    )
                )
            if item.status is ShortTermItemStatus.ACTIVE and item.is_expired(now):
                issues.append(
                    _issue(
                        "EXPIRED_ACTIVE_ITEM",
                        "expired item still ACTIVE",
                        "WARNING",
                        object_id=str(item.item_id),
                    )
                )

        for context in contexts:
            if context.status is ActiveContextStatus.INVALID:
                issues.append(
                    _issue(
                        "INVALID_CONTEXT_STATUS",
                        "ActiveContextRef.status is INVALID",
                        "WARNING",
                        object_id=str(context.context_id),
                    )
                )

        for trace in traces:
            if trace.status is CandidateTraceStatus.INVALID:
                issues.append(
                    _issue(
                        "INVALID_TRACE_STATUS",
                        "CandidateTrace.status is INVALID",
                        "WARNING",
                        object_id=str(trace.trace_id),
                    )
                )
            if (
                trace.status is CandidateTraceStatus.ELIGIBLE
                and not trace.evidence_ids
            ):
                issues.append(
                    _issue(
                        "ELIGIBLE_TRACE_WITHOUT_EVIDENCE",
                        "CandidateTrace.status is ELIGIBLE with no evidence_ids",
                        "WARNING",
                        object_id=str(trace.trace_id),
                    )
                )

        for hypothesis in hypotheses:
            if hypothesis.status is HypothesisStatus.INVALID:
                issues.append(
                    _issue(
                        "INVALID_HYPOTHESIS_STATUS",
                        "WorkingHypothesis.status is INVALID",
                        "WARNING",
                        object_id=str(hypothesis.hypothesis_id),
                    )
                )
            if (
                hypothesis.status is HypothesisStatus.SUPPORTED
                and not hypothesis.supporting_item_ids
            ):
                issues.append(
                    _issue(
                        "SUPPORTED_HYPOTHESIS_WITHOUT_SUPPORT",
                        "WorkingHypothesis.status is SUPPORTED with no support",
                        "WARNING",
                        object_id=str(hypothesis.hypothesis_id),
                    )
                )
            if (
                hypothesis.status is HypothesisStatus.CONTRADICTED
                and not hypothesis.contradicting_item_ids
            ):
                issues.append(
                    _issue(
                        "CONTRADICTED_HYPOTHESIS_WITHOUT_CONTRADICTION",
                        "WorkingHypothesis.status is CONTRADICTED with no "
                        "contradicting_item_ids",
                        "WARNING",
                        object_id=str(hypothesis.hypothesis_id),
                    )
                )


def validate_short_term_snapshot(
    *,
    snapshot: ShortTermMemorySnapshot,
) -> ShortTermValidationReport:
    """Validate a short-term memory snapshot."""

    validator = ShortTermValidator()
    return validator.validate_snapshot(snapshot=snapshot)


def validate_short_term_components(
    *,
    items: tuple[ShortTermItem, ...] | list[ShortTermItem] = (),
    contexts: tuple[ActiveContextRef, ...] | list[ActiveContextRef] = (),
    candidate_traces: tuple[CandidateTrace, ...] | list[CandidateTrace] = (),
    hypotheses: tuple[WorkingHypothesis, ...] | list[WorkingHypothesis] = (),
    evictions: tuple[EvictionRecord, ...] | list[EvictionRecord] = (),
) -> ShortTermValidationReport:
    """Validate short-term memory components."""

    validator = ShortTermValidator()
    return validator.validate_components(
        items=items,
        contexts=contexts,
        candidate_traces=candidate_traces,
        hypotheses=hypotheses,
        evictions=evictions,
    )


def _issue(
    code: str,
    message: str,
    severity: str,
    *,
    object_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> ShortTermValidationIssue:
    return ShortTermValidationIssue(
        code=code,
        message=message,
        severity=severity,
        object_id=object_id,
        metadata={} if metadata is None else metadata,
    )


def _ensure_collection(
    values: tuple[_T, ...] | list[_T],
    item_type: type[_T],
    field_name: str,
) -> tuple[_T, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple or list")
    result = tuple(values)
    for value in result:
        if not isinstance(value, item_type):
            raise TypeError(f"{field_name} contains invalid item type")
    return result


def _ensure_issues(
    values: tuple[ShortTermValidationIssue, ...] | list[ShortTermValidationIssue],
) -> tuple[ShortTermValidationIssue, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("issues must be tuple or list")
    result = tuple(values)
    for issue in result:
        if not isinstance(issue, ShortTermValidationIssue):
            raise TypeError("issues must contain ShortTermValidationIssue")
    return result


def _copy_metadata(metadata: dict[str, object]) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(metadata)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
    return copied
