"""Deterministic runtime candidate trace management for CEBA STM."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from ceba.memory.short_term.schema import (
    CandidateTrace,
    CandidateTraceStatus,
    ShortTermItem,
    create_candidate_trace,
    ensure_non_empty_string,
    ensure_probability,
    utc_now,
)


__all__ = [
    "CandidateTracePolicy",
    "CandidateTraceResult",
    "CandidateTraceManager",
    "build_candidate_trace_manager",
]


_UNCHANGED = object()


@dataclass(frozen=True)
class CandidateTracePolicy:
    """Configuration object controlling deterministic candidate trace behavior."""

    max_items_per_trace: int = 256
    max_evidence_refs: int = 512
    eligibility_confidence_threshold: float = 0.60
    eligibility_priority_threshold: float = 0.50
    allow_empty_trace: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.max_items_per_trace, bool) or not isinstance(
            self.max_items_per_trace,
            int,
        ):
            raise TypeError("max_items_per_trace must be int")
        if self.max_items_per_trace <= 0:
            raise ValueError("max_items_per_trace must be positive")
        if isinstance(self.max_evidence_refs, bool) or not isinstance(
            self.max_evidence_refs,
            int,
        ):
            raise TypeError("max_evidence_refs must be int")
        if self.max_evidence_refs < 0:
            raise ValueError("max_evidence_refs must be non-negative")
        confidence_threshold = ensure_probability(
            self.eligibility_confidence_threshold,
            "eligibility_confidence_threshold",
        )
        priority_threshold = ensure_probability(
            self.eligibility_priority_threshold,
            "eligibility_priority_threshold",
        )
        if not isinstance(self.allow_empty_trace, bool):
            raise TypeError("allow_empty_trace must be bool")
        object.__setattr__(
            self,
            "eligibility_confidence_threshold",
            confidence_threshold,
        )
        object.__setattr__(
            self,
            "eligibility_priority_threshold",
            priority_threshold,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "max_items_per_trace": self.max_items_per_trace,
            "max_evidence_refs": self.max_evidence_refs,
            "eligibility_confidence_threshold": (
                self.eligibility_confidence_threshold
            ),
            "eligibility_priority_threshold": self.eligibility_priority_threshold,
            "allow_empty_trace": self.allow_empty_trace,
        }


@dataclass(frozen=True)
class CandidateTraceResult:
    """Immutable result object for candidate trace operations."""

    trace: CandidateTrace
    changed: bool
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.trace, CandidateTrace):
            raise TypeError("trace must be CandidateTrace")
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        reason = ensure_non_empty_string(self.reason, "reason")
        object.__setattr__(self, "reason", reason)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "trace": self.trace.to_dict(),
            "changed": self.changed,
            "reason": self.reason,
        }


class CandidateTraceManager:
    """Deterministic runtime manager for one CandidateTrace."""

    def __init__(
        self,
        *,
        trace: CandidateTrace | None = None,
        items: tuple[ShortTermItem, ...] | list[ShortTermItem] = (),
        evidence_ids: tuple[str, ...] | list[str] = (),
        policy: CandidateTracePolicy | None = None,
    ) -> None:
        self._policy = CandidateTracePolicy() if policy is None else policy
        if not isinstance(self._policy, CandidateTracePolicy):
            raise TypeError("policy must be CandidateTracePolicy")

        item_values = _ensure_items(items)
        known_item_ids = tuple(item.item_id for item in item_values)
        evidence_id_values = _ensure_evidence_ids(evidence_ids)
        if trace is None:
            trace = create_candidate_trace(
                item_ids=known_item_ids,
                evidence_ids=evidence_id_values,
            )
        elif not isinstance(trace, CandidateTrace):
            raise TypeError("trace must be CandidateTrace")

        if known_item_ids:
            known = set(known_item_ids)
            for item_id in trace.item_ids:
                if item_id not in known:
                    raise ValueError("trace.item_ids must exist in items")

        item_ids = self._enforce_item_capacity(trace.item_ids)
        evidence_ids_value = self._enforce_evidence_capacity(trace.evidence_ids)
        self._trace = CandidateTrace(
            trace_id=trace.trace_id,
            status=trace.status,
            item_ids=item_ids,
            evidence_ids=evidence_ids_value,
            confidence=trace.confidence,
            priority=trace.priority,
            created_at=trace.created_at,
            updated_at=trace.updated_at,
            metadata=dict(trace.metadata),
        )
        self._known_item_ids = known_item_ids

    def trace(self) -> CandidateTrace:
        """Return current trace."""

        return self._trace

    def item_ids(self) -> tuple[str, ...]:
        """Return current trace item ids as strings."""

        return tuple(str(item_id) for item_id in self._trace.item_ids)

    def evidence_ids(self) -> tuple[str, ...]:
        """Return current evidence ids."""

        return self._trace.evidence_ids

    def item_count(self) -> int:
        """Return number of item ids in current trace."""

        return len(self._trace.item_ids)

    def evidence_count(self) -> int:
        """Return number of evidence ids in current trace."""

        return len(self._trace.evidence_ids)

    def status(self) -> CandidateTraceStatus:
        """Return current trace status."""

        return self._trace.status

    def add_item(self, item: ShortTermItem) -> CandidateTraceResult:
        """Add an item to the candidate trace."""

        if not isinstance(item, ShortTermItem):
            raise TypeError("item must be ShortTermItem")
        if item.item_id not in self._known_item_ids:
            self._known_item_ids = (*self._known_item_ids, item.item_id)
        return self._add_uuid(item.item_id, "add_item")

    def add_item_id(self, item_id: str) -> CandidateTraceResult:
        """Add an item id to the candidate trace."""

        return self._add_uuid(_parse_uuid(item_id, "item_id"), "add_item_id")

    def remove_item_id(self, item_id: str) -> CandidateTraceResult:
        """Remove an item id from the candidate trace."""

        parsed = _parse_uuid(item_id, "item_id")
        if parsed not in self._trace.item_ids:
            return self._result(False, "item_id missing")
        item_ids = tuple(
            current_id for current_id in self._trace.item_ids if current_id != parsed
        )
        if not item_ids and not self._policy.allow_empty_trace:
            raise ValueError("empty trace is not allowed")
        self._trace = self._updated_trace(
            item_ids=item_ids,
            operation="remove_item_id",
        )
        return self._result(True, "item_id removed")

    def add_evidence_id(self, evidence_id: str) -> CandidateTraceResult:
        """Add an evidence id to the candidate trace."""

        evidence_id_value = ensure_non_empty_string(evidence_id, "evidence_id")
        if evidence_id_value in self._trace.evidence_ids:
            return self._result(False, "evidence_id already present")
        evidence_ids = self._enforce_evidence_capacity(
            (*self._trace.evidence_ids, evidence_id_value)
        )
        self._trace = self._updated_trace(
            evidence_ids=evidence_ids,
            operation="add_evidence_id",
        )
        return self._result(True, "evidence_id added")

    def remove_evidence_id(self, evidence_id: str) -> CandidateTraceResult:
        """Remove an evidence id from the candidate trace."""

        evidence_id_value = ensure_non_empty_string(evidence_id, "evidence_id")
        if evidence_id_value not in self._trace.evidence_ids:
            return self._result(False, "evidence_id missing")
        evidence_ids = tuple(
            current_id
            for current_id in self._trace.evidence_ids
            if current_id != evidence_id_value
        )
        self._trace = self._updated_trace(
            evidence_ids=evidence_ids,
            operation="remove_evidence_id",
        )
        return self._result(True, "evidence_id removed")

    def set_confidence(self, confidence: float) -> CandidateTraceResult:
        """Set trace confidence."""

        confidence_value = ensure_probability(confidence, "confidence")
        if self._trace.confidence == confidence_value:
            return self._result(False, "confidence unchanged")
        self._trace = self._updated_trace(
            confidence=confidence_value,
            operation="set_confidence",
        )
        return self._result(True, "confidence set")

    def set_priority(self, priority: float) -> CandidateTraceResult:
        """Set trace priority."""

        priority_value = ensure_probability(priority, "priority")
        if self._trace.priority == priority_value:
            return self._result(False, "priority unchanged")
        self._trace = self._updated_trace(
            priority=priority_value,
            operation="set_priority",
        )
        return self._result(True, "priority set")

    def set_status(self, status: CandidateTraceStatus) -> CandidateTraceResult:
        """Set trace status."""

        if not isinstance(status, CandidateTraceStatus):
            raise TypeError("status must be CandidateTraceStatus")
        if self._trace.status is status:
            return self._result(False, "status unchanged")
        self._trace = self._updated_trace(
            status=status,
            operation="set_status",
        )
        return self._result(True, "status set")

    def mark_eligible(self) -> CandidateTraceResult:
        """Mark trace eligible for consolidation."""

        if (
            self._trace.confidence
            < self._policy.eligibility_confidence_threshold
        ):
            raise ValueError("confidence below eligibility threshold")
        if self._trace.priority < self._policy.eligibility_priority_threshold:
            raise ValueError("priority below eligibility threshold")
        if not self._trace.item_ids and not self._policy.allow_empty_trace:
            raise ValueError("eligible trace must contain at least one item")
        return self.set_status(CandidateTraceStatus.ELIGIBLE)

    def reject(self) -> CandidateTraceResult:
        """Reject the candidate trace."""

        return self.set_status(CandidateTraceStatus.REJECTED)

    def expire(self) -> CandidateTraceResult:
        """Expire the candidate trace."""

        return self.set_status(CandidateTraceStatus.EXPIRED)

    def mark_consolidated(self) -> CandidateTraceResult:
        """Mark the candidate trace as consolidated."""

        return self.set_status(CandidateTraceStatus.CONSOLIDATED)

    def reset(
        self,
        *,
        item_ids: tuple[str, ...] | list[str] = (),
        evidence_ids: tuple[str, ...] | list[str] = (),
        confidence: float = 1.0,
        priority: float = 0.5,
    ) -> CandidateTraceResult:
        """Reset the candidate trace contents."""

        parsed_item_ids = tuple(_parse_uuid(value, "item_ids") for value in item_ids)
        item_ids_value = self._enforce_item_capacity(
            _deduplicate_uuids(parsed_item_ids)
        )
        if not item_ids_value and not self._policy.allow_empty_trace:
            raise ValueError("empty trace is not allowed")
        evidence_ids_value = self._enforce_evidence_capacity(
            _deduplicate_strings(_ensure_evidence_ids(evidence_ids))
        )
        self._trace = self._updated_trace(
            status=CandidateTraceStatus.CANDIDATE,
            item_ids=item_ids_value,
            evidence_ids=evidence_ids_value,
            confidence=ensure_probability(confidence, "confidence"),
            priority=ensure_probability(priority, "priority"),
            operation="reset",
        )
        return self._result(True, "trace reset")

    def _add_uuid(self, item_id: UUID, operation: str) -> CandidateTraceResult:
        if item_id in self._trace.item_ids:
            return self._result(False, "item_id already present")
        item_ids = self._enforce_item_capacity((*self._trace.item_ids, item_id))
        self._trace = self._updated_trace(
            item_ids=item_ids,
            operation=operation,
        )
        return self._result(True, "item_id added")

    def _updated_trace(
        self,
        *,
        status: CandidateTraceStatus | None = None,
        item_ids: tuple[UUID, ...] | None = None,
        evidence_ids: tuple[str, ...] | None = None,
        confidence: float | object = _UNCHANGED,
        priority: float | object = _UNCHANGED,
        operation: str,
    ) -> CandidateTrace:
        metadata = dict(self._trace.metadata)
        metadata["last_operation"] = operation
        confidence_value = (
            self._trace.confidence if confidence is _UNCHANGED else confidence
        )
        priority_value = self._trace.priority if priority is _UNCHANGED else priority
        if not isinstance(confidence_value, (int, float)):
            raise TypeError("confidence must be int or float")
        if not isinstance(priority_value, (int, float)):
            raise TypeError("priority must be int or float")
        return CandidateTrace(
            trace_id=self._trace.trace_id,
            status=self._trace.status if status is None else status,
            item_ids=self._trace.item_ids if item_ids is None else item_ids,
            evidence_ids=(
                self._trace.evidence_ids
                if evidence_ids is None
                else evidence_ids
            ),
            confidence=confidence_value,
            priority=priority_value,
            created_at=self._trace.created_at,
            updated_at=utc_now(),
            metadata=metadata,
        )

    def _enforce_item_capacity(
        self,
        item_ids: tuple[UUID, ...],
    ) -> tuple[UUID, ...]:
        if len(item_ids) <= self._policy.max_items_per_trace:
            return item_ids
        return item_ids[-self._policy.max_items_per_trace :]

    def _enforce_evidence_capacity(
        self,
        evidence_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        if self._policy.max_evidence_refs == 0:
            return ()
        if len(evidence_ids) <= self._policy.max_evidence_refs:
            return evidence_ids
        return evidence_ids[-self._policy.max_evidence_refs :]

    def _result(self, changed: bool, reason: str) -> CandidateTraceResult:
        return CandidateTraceResult(
            trace=self._trace,
            changed=changed,
            reason=reason,
        )


def build_candidate_trace_manager(
    *,
    trace: CandidateTrace | None = None,
    items: tuple[ShortTermItem, ...] | list[ShortTermItem] = (),
    evidence_ids: tuple[str, ...] | list[str] = (),
    policy: CandidateTracePolicy | None = None,
) -> CandidateTraceManager:
    """Build a candidate trace manager."""

    return CandidateTraceManager(
        trace=trace,
        items=items,
        evidence_ids=evidence_ids,
        policy=policy,
    )


def _ensure_items(
    values: tuple[ShortTermItem, ...] | list[ShortTermItem],
) -> tuple[ShortTermItem, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("items must be tuple or list")
    result = tuple(values)
    for item in result:
        if not isinstance(item, ShortTermItem):
            raise TypeError("items must contain ShortTermItem")
    return result


def _ensure_evidence_ids(
    values: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("evidence_ids must be tuple or list")
    return tuple(ensure_non_empty_string(value, "evidence_ids") for value in values)


def _parse_uuid(value: str, field_name: str) -> UUID:
    text = ensure_non_empty_string(value, field_name)
    try:
        return UUID(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be UUID string") from exc


def _deduplicate_uuids(values: tuple[UUID, ...]) -> tuple[UUID, ...]:
    seen: set[UUID] = set()
    result: list[UUID] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return tuple(result)


def _deduplicate_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return tuple(result)
