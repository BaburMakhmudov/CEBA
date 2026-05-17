"""Deterministic eviction policy utilities for CEBA Short-Term Memory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ceba.memory.short_term.schema import (
    EvictionReason,
    EvictionRecord,
    ShortTermItem,
    ShortTermItemStatus,
    create_eviction_record,
    ensure_probability,
    utc_now,
)


__all__ = [
    "EvictionPolicy",
    "EvictionDecision",
    "EvictionPlanner",
    "plan_evictions",
    "create_eviction_records",
]


@dataclass(frozen=True)
class EvictionPolicy:
    """Configuration object controlling deterministic STM eviction decisions."""

    max_items: int = 512
    evict_expired_first: bool = True
    evict_resolved_first: bool = True
    evict_invalid_first: bool = True
    min_priority_to_keep: float = 0.0
    min_confidence_to_keep: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.max_items, bool) or not isinstance(self.max_items, int):
            raise TypeError("max_items must be int")
        if self.max_items <= 0:
            raise ValueError("max_items must be positive")
        if not isinstance(self.evict_expired_first, bool):
            raise TypeError("evict_expired_first must be bool")
        if not isinstance(self.evict_resolved_first, bool):
            raise TypeError("evict_resolved_first must be bool")
        if not isinstance(self.evict_invalid_first, bool):
            raise TypeError("evict_invalid_first must be bool")
        min_priority = ensure_probability(
            self.min_priority_to_keep,
            "min_priority_to_keep",
        )
        min_confidence = ensure_probability(
            self.min_confidence_to_keep,
            "min_confidence_to_keep",
        )
        object.__setattr__(self, "min_priority_to_keep", min_priority)
        object.__setattr__(self, "min_confidence_to_keep", min_confidence)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "max_items": self.max_items,
            "evict_expired_first": self.evict_expired_first,
            "evict_resolved_first": self.evict_resolved_first,
            "evict_invalid_first": self.evict_invalid_first,
            "min_priority_to_keep": self.min_priority_to_keep,
            "min_confidence_to_keep": self.min_confidence_to_keep,
        }


@dataclass(frozen=True)
class EvictionDecision:
    """Immutable deterministic eviction decision result."""

    kept_items: tuple[ShortTermItem, ...]
    evicted_items: tuple[ShortTermItem, ...]
    reasons: tuple[EvictionReason, ...]

    def __post_init__(self) -> None:
        kept_items = _ensure_items(self.kept_items, "kept_items")
        evicted_items = _ensure_items(self.evicted_items, "evicted_items")
        reasons = _ensure_reasons(self.reasons)
        if len(evicted_items) != len(reasons):
            raise ValueError("evicted_items and reasons must have same length")
        kept_ids = {item.item_id for item in kept_items}
        evicted_ids = {item.item_id for item in evicted_items}
        if kept_ids.intersection(evicted_ids):
            raise ValueError("item_id must not appear in both kept and evicted")

        object.__setattr__(self, "kept_items", kept_items)
        object.__setattr__(self, "evicted_items", evicted_items)
        object.__setattr__(self, "reasons", reasons)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "kept_items": tuple(item.to_dict() for item in self.kept_items),
            "evicted_items": tuple(
                item.to_dict() for item in self.evicted_items
            ),
            "reasons": tuple(reason.value for reason in self.reasons),
        }

    def kept_count(self) -> int:
        """Return kept item count."""

        return len(self.kept_items)

    def evicted_count(self) -> int:
        """Return evicted item count."""

        return len(self.evicted_items)


class EvictionPlanner:
    """Deterministic service class that plans STM evictions."""

    def __init__(self, policy: EvictionPolicy | None = None) -> None:
        self._policy = EvictionPolicy() if policy is None else policy
        if not isinstance(self._policy, EvictionPolicy):
            raise TypeError("policy must be EvictionPolicy")

    def plan(
        self,
        *,
        items: tuple[ShortTermItem, ...] | list[ShortTermItem],
    ) -> EvictionDecision:
        """Plan evictions without mutating items."""

        item_values = _ensure_items(items, "items")
        now = utc_now()
        kept = list(item_values)
        evicted: list[ShortTermItem] = []
        reasons: list[EvictionReason] = []

        for item in item_values:
            reason = self._mandatory_reason(item, now)
            if reason is None:
                continue
            kept.remove(item)
            evicted.append(item)
            reasons.append(reason)

        while len(kept) > self._policy.max_items:
            candidate = min(kept, key=lambda item: self._eviction_sort_key(item, now))
            kept.remove(candidate)
            evicted.append(candidate)
            reasons.append(self._reason_for_item(candidate, now))

        return EvictionDecision(
            kept_items=tuple(kept),
            evicted_items=tuple(evicted),
            reasons=tuple(reasons),
        )

    def records_for_decision(
        self,
        *,
        decision: EvictionDecision,
    ) -> tuple[EvictionRecord, ...]:
        """Create eviction records for a decision."""

        if not isinstance(decision, EvictionDecision):
            raise TypeError("decision must be EvictionDecision")
        return tuple(
            create_eviction_record(item_id=item.item_id, reason=reason)
            for item, reason in zip(decision.evicted_items, decision.reasons)
        )

    def _mandatory_reason(
        self,
        item: ShortTermItem,
        now: datetime,
    ) -> EvictionReason | None:
        if self._policy.evict_expired_first and item.is_expired(now):
            return EvictionReason.TTL_EXPIRED
        if (
            self._policy.evict_invalid_first
            and item.status is ShortTermItemStatus.INVALID
        ):
            return EvictionReason.INVALID
        return None

    def _eviction_sort_key(
        self,
        item: ShortTermItem,
        now: datetime,
    ) -> tuple[object, ...]:
        expired_rank = 0
        if self._policy.evict_expired_first and item.is_expired(now):
            expired_rank = -1
        invalid_rank = 0
        if (
            self._policy.evict_invalid_first
            and item.status is ShortTermItemStatus.INVALID
        ):
            invalid_rank = -1
        resolved_rank = 0
        if (
            self._policy.evict_resolved_first
            and item.status is ShortTermItemStatus.RESOLVED
        ):
            resolved_rank = -1
        low_priority_rank = (
            -1 if item.priority < self._policy.min_priority_to_keep else 0
        )
        low_confidence_rank = (
            -1 if item.confidence < self._policy.min_confidence_to_keep else 0
        )
        return (
            expired_rank,
            invalid_rank,
            resolved_rank,
            low_priority_rank,
            low_confidence_rank,
            item.priority,
            item.confidence,
            item.updated_at,
            item.created_at,
            str(item.item_id),
        )

    def _reason_for_item(
        self,
        item: ShortTermItem,
        now: datetime,
    ) -> EvictionReason:
        if item.is_expired(now):
            return EvictionReason.TTL_EXPIRED
        if item.status is ShortTermItemStatus.INVALID:
            return EvictionReason.INVALID
        if item.status is ShortTermItemStatus.RESOLVED:
            return EvictionReason.RESOLVED
        if item.priority < self._policy.min_priority_to_keep:
            return EvictionReason.LOW_PRIORITY
        if item.confidence < self._policy.min_confidence_to_keep:
            return EvictionReason.LOW_PRIORITY
        return EvictionReason.CAPACITY_LIMIT


def plan_evictions(
    *,
    items: tuple[ShortTermItem, ...] | list[ShortTermItem],
    policy: EvictionPolicy | None = None,
) -> EvictionDecision:
    """Plan STM evictions."""

    planner = EvictionPlanner(policy)
    return planner.plan(items=items)


def create_eviction_records(
    *,
    decision: EvictionDecision,
    policy: EvictionPolicy | None = None,
) -> tuple[EvictionRecord, ...]:
    """Create eviction records for a decision."""

    planner = EvictionPlanner(policy)
    return planner.records_for_decision(decision=decision)


def _ensure_items(
    values: tuple[ShortTermItem, ...] | list[ShortTermItem],
    field_name: str,
) -> tuple[ShortTermItem, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple or list")
    result = tuple(values)
    for item in result:
        if not isinstance(item, ShortTermItem):
            raise TypeError(f"{field_name} must contain ShortTermItem")
    item_ids = tuple(item.item_id for item in result)
    if len(set(item_ids)) != len(item_ids):
        raise ValueError(f"{field_name} item_id values must be unique")
    return result


def _ensure_reasons(
    values: tuple[EvictionReason, ...] | list[EvictionReason],
) -> tuple[EvictionReason, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("reasons must be tuple or list")
    result = tuple(values)
    for reason in result:
        if not isinstance(reason, EvictionReason):
            raise TypeError("reasons must contain EvictionReason")
    return result
