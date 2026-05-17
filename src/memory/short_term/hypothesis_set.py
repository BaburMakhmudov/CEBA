"""Deterministic runtime working hypothesis management for CEBA STM."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from ceba.memory.short_term.schema import (
    HypothesisStatus,
    ShortTermItem,
    WorkingHypothesis,
    create_working_hypothesis,
    ensure_non_empty_string,
    ensure_probability,
    utc_now,
)


__all__ = [
    "HypothesisSetPolicy",
    "HypothesisSetResult",
    "HypothesisSet",
    "build_hypothesis_set",
]


@dataclass(frozen=True)
class HypothesisSetPolicy:
    """Configuration object controlling deterministic hypothesis behavior."""

    max_hypotheses: int = 128
    max_supporting_items: int = 256
    max_contradicting_items: int = 256
    support_confidence_boost: float = 0.05
    contradiction_confidence_penalty: float = 0.10
    supported_threshold: float = 0.70
    contradicted_threshold: float = 0.30

    def __post_init__(self) -> None:
        _ensure_positive_int(self.max_hypotheses, "max_hypotheses")
        _ensure_positive_int(self.max_supporting_items, "max_supporting_items")
        _ensure_positive_int(
            self.max_contradicting_items,
            "max_contradicting_items",
        )
        support_boost = ensure_probability(
            self.support_confidence_boost,
            "support_confidence_boost",
        )
        contradiction_penalty = ensure_probability(
            self.contradiction_confidence_penalty,
            "contradiction_confidence_penalty",
        )
        supported_threshold = ensure_probability(
            self.supported_threshold,
            "supported_threshold",
        )
        contradicted_threshold = ensure_probability(
            self.contradicted_threshold,
            "contradicted_threshold",
        )
        if supported_threshold < contradicted_threshold:
            raise ValueError("supported_threshold must be >= contradicted_threshold")

        object.__setattr__(self, "support_confidence_boost", support_boost)
        object.__setattr__(
            self,
            "contradiction_confidence_penalty",
            contradiction_penalty,
        )
        object.__setattr__(self, "supported_threshold", supported_threshold)
        object.__setattr__(self, "contradicted_threshold", contradicted_threshold)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "max_hypotheses": self.max_hypotheses,
            "max_supporting_items": self.max_supporting_items,
            "max_contradicting_items": self.max_contradicting_items,
            "support_confidence_boost": self.support_confidence_boost,
            "contradiction_confidence_penalty": (
                self.contradiction_confidence_penalty
            ),
            "supported_threshold": self.supported_threshold,
            "contradicted_threshold": self.contradicted_threshold,
        }


@dataclass(frozen=True)
class HypothesisSetResult:
    """Immutable result object for hypothesis set operations."""

    hypotheses: tuple[WorkingHypothesis, ...]
    changed: bool
    reason: str

    def __post_init__(self) -> None:
        hypotheses = _ensure_hypotheses(self.hypotheses)
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        reason = ensure_non_empty_string(self.reason, "reason")
        object.__setattr__(self, "hypotheses", hypotheses)
        object.__setattr__(self, "reason", reason)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "hypotheses": tuple(
                hypothesis.to_dict() for hypothesis in self.hypotheses
            ),
            "changed": self.changed,
            "reason": self.reason,
        }


class HypothesisSet:
    """Deterministic runtime manager for WorkingHypothesis objects."""

    def __init__(
        self,
        *,
        hypotheses: (
            tuple[WorkingHypothesis, ...] | list[WorkingHypothesis]
        ) = (),
        items: tuple[ShortTermItem, ...] | list[ShortTermItem] = (),
        policy: HypothesisSetPolicy | None = None,
    ) -> None:
        self._policy = HypothesisSetPolicy() if policy is None else policy
        if not isinstance(self._policy, HypothesisSetPolicy):
            raise TypeError("policy must be HypothesisSetPolicy")
        self._known_item_ids = tuple(item.item_id for item in _ensure_items(items))
        hypothesis_values = _ensure_hypotheses(hypotheses)
        if self._known_item_ids:
            self._validate_hypothesis_item_refs(hypothesis_values)
        self._hypotheses = self._enforce_hypothesis_capacity(hypothesis_values)

    def hypotheses(self) -> tuple[WorkingHypothesis, ...]:
        """Return current hypotheses."""

        return self._hypotheses

    def hypothesis_count(self) -> int:
        """Return current hypothesis count."""

        return len(self._hypotheses)

    def item_ids(self) -> tuple[str, ...]:
        """Return known item ids as strings."""

        return tuple(str(item_id) for item_id in self._known_item_ids)

    def has_hypothesis(self, hypothesis_id: str) -> bool:
        """Return whether a hypothesis exists."""

        parsed = _parse_uuid(hypothesis_id, "hypothesis_id")
        return any(
            hypothesis.hypothesis_id == parsed
            for hypothesis in self._hypotheses
        )

    def get_hypothesis(self, hypothesis_id: str) -> WorkingHypothesis | None:
        """Return a matching hypothesis or None."""

        parsed = _parse_uuid(hypothesis_id, "hypothesis_id")
        for hypothesis in self._hypotheses:
            if hypothesis.hypothesis_id == parsed:
                return hypothesis
        return None

    def add_hypothesis(
        self,
        hypothesis: WorkingHypothesis,
    ) -> HypothesisSetResult:
        """Add or replace one hypothesis."""

        if not isinstance(hypothesis, WorkingHypothesis):
            raise TypeError("hypothesis must be WorkingHypothesis")
        self._validate_hypothesis_item_refs((hypothesis,))
        self._hypotheses = self._enforce_hypothesis_capacity(
            _replace_hypothesis(self._hypotheses, hypothesis)
        )
        return self._result(True, "hypothesis added")

    def create_hypothesis(
        self,
        *,
        statement_ref: str,
        supporting_item_ids: tuple[str, ...] | list[str] = (),
        contradicting_item_ids: tuple[str, ...] | list[str] = (),
        confidence: float = 0.5,
    ) -> HypothesisSetResult:
        """Create and add a working hypothesis."""

        statement = ensure_non_empty_string(statement_ref, "statement_ref")
        support_ids = self._enforce_support_capacity(
            _deduplicate_uuids(
                tuple(
                    _parse_uuid(value, "supporting_item_ids")
                    for value in supporting_item_ids
                )
            )
        )
        contradiction_ids = self._enforce_contradiction_capacity(
            _deduplicate_uuids(
                tuple(
                    _parse_uuid(value, "contradicting_item_ids")
                    for value in contradicting_item_ids
                )
            )
        )
        hypothesis = create_working_hypothesis(
            statement_ref=statement,
            supporting_item_ids=support_ids,
            contradicting_item_ids=contradiction_ids,
            confidence=confidence,
        )
        return self.add_hypothesis(hypothesis)

    def remove_hypothesis(self, hypothesis_id: str) -> HypothesisSetResult:
        """Remove one hypothesis."""

        parsed = _parse_uuid(hypothesis_id, "hypothesis_id")
        kept = tuple(
            hypothesis
            for hypothesis in self._hypotheses
            if hypothesis.hypothesis_id != parsed
        )
        if len(kept) == len(self._hypotheses):
            return self._result(False, "hypothesis missing")
        self._hypotheses = kept
        return self._result(True, "hypothesis removed")

    def add_support(
        self,
        hypothesis_id: str,
        item: ShortTermItem,
    ) -> HypothesisSetResult:
        """Add supporting item evidence to a hypothesis."""

        if not isinstance(item, ShortTermItem):
            raise TypeError("item must be ShortTermItem")
        if item.item_id not in self._known_item_ids:
            self._known_item_ids = (*self._known_item_ids, item.item_id)
        hypothesis = self.get_hypothesis(hypothesis_id)
        if hypothesis is None:
            return self._result(False, "hypothesis missing")
        support_ids = hypothesis.supporting_item_ids
        if item.item_id not in support_ids:
            support_ids = (*support_ids, item.item_id)
        support_ids = self._enforce_support_capacity(support_ids)
        contradiction_ids = tuple(
            item_id
            for item_id in hypothesis.contradicting_item_ids
            if item_id != item.item_id
        )
        confidence = min(
            1.0,
            hypothesis.confidence + self._policy.support_confidence_boost,
        )
        updated = self._updated_hypothesis(
            hypothesis,
            supporting_item_ids=support_ids,
            contradicting_item_ids=contradiction_ids,
            confidence=confidence,
            operation="add_support",
        )
        self._hypotheses = _replace_hypothesis(self._hypotheses, updated)
        return self._result(True, "support added")

    def add_contradiction(
        self,
        hypothesis_id: str,
        item: ShortTermItem,
    ) -> HypothesisSetResult:
        """Add contradicting item evidence to a hypothesis."""

        if not isinstance(item, ShortTermItem):
            raise TypeError("item must be ShortTermItem")
        if item.item_id not in self._known_item_ids:
            self._known_item_ids = (*self._known_item_ids, item.item_id)
        hypothesis = self.get_hypothesis(hypothesis_id)
        if hypothesis is None:
            return self._result(False, "hypothesis missing")
        contradiction_ids = hypothesis.contradicting_item_ids
        if item.item_id not in contradiction_ids:
            contradiction_ids = (*contradiction_ids, item.item_id)
        contradiction_ids = self._enforce_contradiction_capacity(
            contradiction_ids
        )
        support_ids = tuple(
            item_id
            for item_id in hypothesis.supporting_item_ids
            if item_id != item.item_id
        )
        confidence = max(
            0.0,
            hypothesis.confidence
            - self._policy.contradiction_confidence_penalty,
        )
        updated = self._updated_hypothesis(
            hypothesis,
            supporting_item_ids=support_ids,
            contradicting_item_ids=contradiction_ids,
            confidence=confidence,
            operation="add_contradiction",
        )
        self._hypotheses = _replace_hypothesis(self._hypotheses, updated)
        return self._result(True, "contradiction added")

    def set_confidence(
        self,
        hypothesis_id: str,
        confidence: float,
    ) -> HypothesisSetResult:
        """Set hypothesis confidence."""

        confidence_value = ensure_probability(confidence, "confidence")
        hypothesis = self.get_hypothesis(hypothesis_id)
        if hypothesis is None:
            return self._result(False, "hypothesis missing")
        if hypothesis.confidence == confidence_value:
            return self._result(False, "confidence unchanged")
        updated = self._updated_hypothesis(
            hypothesis,
            confidence=confidence_value,
            operation="set_confidence",
        )
        self._hypotheses = _replace_hypothesis(self._hypotheses, updated)
        return self._result(True, "confidence set")

    def set_status(
        self,
        hypothesis_id: str,
        status: HypothesisStatus,
    ) -> HypothesisSetResult:
        """Set hypothesis status."""

        if not isinstance(status, HypothesisStatus):
            raise TypeError("status must be HypothesisStatus")
        hypothesis = self.get_hypothesis(hypothesis_id)
        if hypothesis is None:
            return self._result(False, "hypothesis missing")
        if hypothesis.status is status:
            return self._result(False, "status unchanged")
        updated = self._updated_hypothesis(
            hypothesis,
            status=status,
            operation="set_status",
        )
        self._hypotheses = _replace_hypothesis(self._hypotheses, updated)
        return self._result(True, "status set")

    def resolve(self, hypothesis_id: str) -> HypothesisSetResult:
        """Resolve a hypothesis."""

        return self.set_status(hypothesis_id, HypothesisStatus.RESOLVED)

    def reject(self, hypothesis_id: str) -> HypothesisSetResult:
        """Reject a hypothesis."""

        return self.set_status(hypothesis_id, HypothesisStatus.REJECTED)

    def expire(self, hypothesis_id: str) -> HypothesisSetResult:
        """Expire a hypothesis."""

        return self.set_status(hypothesis_id, HypothesisStatus.EXPIRED)

    def _updated_hypothesis(
        self,
        hypothesis: WorkingHypothesis,
        *,
        status: HypothesisStatus | None = None,
        supporting_item_ids: tuple[UUID, ...] | None = None,
        contradicting_item_ids: tuple[UUID, ...] | None = None,
        confidence: float | None = None,
        operation: str,
    ) -> WorkingHypothesis:
        confidence_value = (
            hypothesis.confidence if confidence is None else confidence
        )
        confidence_value = ensure_probability(confidence_value, "confidence")
        next_status = status
        if next_status is None:
            next_status = self._status_from_confidence(confidence_value)
        metadata = dict(hypothesis.metadata)
        metadata["last_operation"] = operation
        return WorkingHypothesis(
            hypothesis_id=hypothesis.hypothesis_id,
            status=next_status,
            statement_ref=hypothesis.statement_ref,
            supporting_item_ids=(
                hypothesis.supporting_item_ids
                if supporting_item_ids is None
                else supporting_item_ids
            ),
            contradicting_item_ids=(
                hypothesis.contradicting_item_ids
                if contradicting_item_ids is None
                else contradicting_item_ids
            ),
            confidence=confidence_value,
            created_at=hypothesis.created_at,
            updated_at=utc_now(),
            metadata=metadata,
        )

    def _status_from_confidence(self, confidence: float) -> HypothesisStatus:
        if confidence >= self._policy.supported_threshold:
            return HypothesisStatus.SUPPORTED
        if confidence <= self._policy.contradicted_threshold:
            return HypothesisStatus.CONTRADICTED
        return HypothesisStatus.OPEN

    def _enforce_hypothesis_capacity(
        self,
        hypotheses: tuple[WorkingHypothesis, ...],
    ) -> tuple[WorkingHypothesis, ...]:
        if len(hypotheses) <= self._policy.max_hypotheses:
            return hypotheses
        sorted_values = sorted(
            hypotheses,
            key=lambda hypothesis: (
                -hypothesis.updated_at.timestamp(),
                -hypothesis.created_at.timestamp(),
                str(hypothesis.hypothesis_id),
            ),
        )
        kept = sorted_values[: self._policy.max_hypotheses]
        kept_ids = {hypothesis.hypothesis_id for hypothesis in kept}
        return tuple(
            hypothesis
            for hypothesis in hypotheses
            if hypothesis.hypothesis_id in kept_ids
        )

    def _enforce_support_capacity(
        self,
        item_ids: tuple[UUID, ...],
    ) -> tuple[UUID, ...]:
        if len(item_ids) <= self._policy.max_supporting_items:
            return item_ids
        return item_ids[-self._policy.max_supporting_items :]

    def _enforce_contradiction_capacity(
        self,
        item_ids: tuple[UUID, ...],
    ) -> tuple[UUID, ...]:
        if len(item_ids) <= self._policy.max_contradicting_items:
            return item_ids
        return item_ids[-self._policy.max_contradicting_items :]

    def _validate_hypothesis_item_refs(
        self,
        hypotheses: tuple[WorkingHypothesis, ...],
    ) -> None:
        if not self._known_item_ids:
            return
        known = set(self._known_item_ids)
        for hypothesis in hypotheses:
            for item_id in hypothesis.supporting_item_ids:
                if item_id not in known:
                    raise ValueError("supporting_item_ids must exist in items")
            for item_id in hypothesis.contradicting_item_ids:
                if item_id not in known:
                    raise ValueError("contradicting_item_ids must exist in items")

    def _result(self, changed: bool, reason: str) -> HypothesisSetResult:
        return HypothesisSetResult(
            hypotheses=self._hypotheses,
            changed=changed,
            reason=reason,
        )


def build_hypothesis_set(
    *,
    hypotheses: tuple[WorkingHypothesis, ...] | list[WorkingHypothesis] = (),
    items: tuple[ShortTermItem, ...] | list[ShortTermItem] = (),
    policy: HypothesisSetPolicy | None = None,
) -> HypothesisSet:
    """Build a hypothesis set."""

    return HypothesisSet(hypotheses=hypotheses, items=items, policy=policy)


def _ensure_positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


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


def _ensure_hypotheses(
    values: tuple[WorkingHypothesis, ...] | list[WorkingHypothesis],
) -> tuple[WorkingHypothesis, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("hypotheses must be tuple or list")
    result = tuple(values)
    for hypothesis in result:
        if not isinstance(hypothesis, WorkingHypothesis):
            raise TypeError("hypotheses must contain WorkingHypothesis")
    hypothesis_ids = tuple(
        str(hypothesis.hypothesis_id) for hypothesis in result
    )
    if len(set(hypothesis_ids)) != len(hypothesis_ids):
        raise ValueError("hypothesis_id values must be unique")
    return result


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


def _replace_hypothesis(
    hypotheses: tuple[WorkingHypothesis, ...],
    replacement: WorkingHypothesis,
) -> tuple[WorkingHypothesis, ...]:
    replacement_id = replacement.hypothesis_id
    replaced = False
    result: list[WorkingHypothesis] = []
    for hypothesis in hypotheses:
        if hypothesis.hypothesis_id == replacement_id:
            result.append(replacement)
            replaced = True
        else:
            result.append(hypothesis)
    if not replaced:
        result.append(replacement)
    return tuple(result)
