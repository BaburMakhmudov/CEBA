"""Deterministic runtime working buffer for CEBA Short-Term Memory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ceba.memory.short_term.schema import (
    EvictionReason,
    EvictionRecord,
    ShortTermItem,
    ShortTermItemStatus,
    ShortTermMemorySnapshot,
    create_eviction_record,
    create_short_term_memory_snapshot,
    ensure_non_empty_string,
    ensure_probability,
    utc_now,
)


__all__ = [
    "WorkingBufferPolicy",
    "WorkingBufferResult",
    "WorkingBuffer",
    "build_working_buffer",
]


@dataclass(frozen=True)
class WorkingBufferPolicy:
    """Configuration object controlling bounded STM buffer behavior."""

    max_items: int = 512
    evict_expired_first: bool = True
    evict_resolved_first: bool = True
    min_priority_to_keep: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.max_items, bool) or not isinstance(self.max_items, int):
            raise TypeError("max_items must be int")
        if self.max_items <= 0:
            raise ValueError("max_items must be positive")
        if not isinstance(self.evict_expired_first, bool):
            raise TypeError("evict_expired_first must be bool")
        if not isinstance(self.evict_resolved_first, bool):
            raise TypeError("evict_resolved_first must be bool")
        min_priority = ensure_probability(
            self.min_priority_to_keep,
            "min_priority_to_keep",
        )
        object.__setattr__(self, "min_priority_to_keep", min_priority)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "max_items": self.max_items,
            "evict_expired_first": self.evict_expired_first,
            "evict_resolved_first": self.evict_resolved_first,
            "min_priority_to_keep": self.min_priority_to_keep,
        }


@dataclass(frozen=True)
class WorkingBufferResult:
    """Immutable result object for buffer operations."""

    items: tuple[ShortTermItem, ...]
    evictions: tuple[EvictionRecord, ...]

    def __post_init__(self) -> None:
        items = _ensure_items(self.items)
        evictions = _ensure_evictions(self.evictions)
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "evictions", evictions)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "items": tuple(item.to_dict() for item in self.items),
            "evictions": tuple(eviction.to_dict() for eviction in self.evictions),
        }


class WorkingBuffer:
    """Deterministic in-memory runtime buffer for ShortTermItem objects."""

    def __init__(
        self,
        *,
        items: tuple[ShortTermItem, ...] | list[ShortTermItem] = (),
        evictions: tuple[EvictionRecord, ...] | list[EvictionRecord] = (),
        policy: WorkingBufferPolicy | None = None,
    ) -> None:
        self._policy = WorkingBufferPolicy() if policy is None else policy
        if not isinstance(self._policy, WorkingBufferPolicy):
            raise TypeError("policy must be WorkingBufferPolicy")
        self._items = _deduplicate_items(_ensure_items(items, require_unique=False))
        self._evictions = _ensure_evictions(evictions)
        result = self._apply_capacity_policy(self._items, self._evictions)
        self._items = result.items
        self._evictions = result.evictions

    def items(self) -> tuple[ShortTermItem, ...]:
        """Return current items."""

        return self._items

    def evictions(self) -> tuple[EvictionRecord, ...]:
        """Return current evictions."""

        return self._evictions

    def item_count(self) -> int:
        """Return number of current items."""

        return len(self._items)

    def has_item(self, item_id: str) -> bool:
        """Return whether an item exists."""

        target_id = ensure_non_empty_string(item_id, "item_id")
        return any(str(item.item_id) == target_id for item in self._items)

    def get_item(self, item_id: str) -> ShortTermItem | None:
        """Return a matching item or None."""

        target_id = ensure_non_empty_string(item_id, "item_id")
        for item in self._items:
            if str(item.item_id) == target_id:
                return item
        return None

    def add_item(self, item: ShortTermItem) -> WorkingBufferResult:
        """Add or replace one item."""

        if not isinstance(item, ShortTermItem):
            raise TypeError("item must be ShortTermItem")
        items = _replace_item(self._items, item)
        result = self._apply_capacity_policy(items, self._evictions)
        self._items = result.items
        self._evictions = result.evictions
        return result

    def add_many(
        self,
        items: tuple[ShortTermItem, ...] | list[ShortTermItem],
    ) -> WorkingBufferResult:
        """Add or replace many items in input order."""

        next_items = self._items
        for item in _ensure_items(items, require_unique=False):
            next_items = _replace_item(next_items, item)
        result = self._apply_capacity_policy(next_items, self._evictions)
        self._items = result.items
        self._evictions = result.evictions
        return result

    def remove_item(
        self,
        item_id: str,
        reason: EvictionReason = EvictionReason.MANUAL,
    ) -> WorkingBufferResult:
        """Remove one item and record an eviction."""

        target_id = ensure_non_empty_string(item_id, "item_id")
        if not isinstance(reason, EvictionReason):
            raise TypeError("reason must be EvictionReason")
        removed: ShortTermItem | None = None
        kept: list[ShortTermItem] = []
        for item in self._items:
            if str(item.item_id) == target_id:
                removed = item
            else:
                kept.append(item)
        if removed is None:
            return self._result()

        eviction = create_eviction_record(item_id=removed.item_id, reason=reason)
        self._items = tuple(kept)
        self._evictions = (*self._evictions, eviction)
        return self._result()

    def expire_items(self) -> WorkingBufferResult:
        """Remove all expired items."""

        now = utc_now()
        kept: list[ShortTermItem] = []
        evictions = list(self._evictions)
        for item in self._items:
            if item.is_expired(now):
                evictions.append(
                    create_eviction_record(
                        item_id=item.item_id,
                        reason=EvictionReason.TTL_EXPIRED,
                    )
                )
            else:
                kept.append(item)
        self._items = tuple(kept)
        self._evictions = tuple(evictions)
        return self._result()

    def resolve_item(self, item_id: str) -> WorkingBufferResult:
        """Mark one item as resolved."""

        target_id = ensure_non_empty_string(item_id, "item_id")
        now = utc_now()
        resolved_items: list[ShortTermItem] = []
        found = False
        for item in self._items:
            if str(item.item_id) != target_id:
                resolved_items.append(item)
                continue
            metadata = dict(item.metadata)
            metadata["resolved"] = True
            resolved_items.append(
                ShortTermItem(
                    item_id=item.item_id,
                    kind=item.kind,
                    status=ShortTermItemStatus.RESOLVED,
                    source_id=item.source_id,
                    content_ref=item.content_ref,
                    priority=item.priority,
                    confidence=item.confidence,
                    created_at=item.created_at,
                    updated_at=now,
                    expires_at=item.expires_at,
                    metadata=metadata,
                )
            )
            found = True
        if not found:
            return self._result()
        self._items = tuple(resolved_items)
        return self._result()

    def snapshot(self) -> ShortTermMemorySnapshot:
        """Return an immutable short-term memory snapshot."""

        return create_short_term_memory_snapshot(
            items=self._items,
            evictions=self._evictions,
            metadata={
                "runtime_source": "working_buffer_snapshot",
                "policy": self._policy.to_dict(),
                "known_evicted_item_ids": tuple(
                    eviction.item_id for eviction in self._evictions
                ),
            },
        )

    def _result(self) -> WorkingBufferResult:
        return WorkingBufferResult(items=self._items, evictions=self._evictions)

    def _apply_capacity_policy(
        self,
        items: tuple[ShortTermItem, ...],
        evictions: tuple[EvictionRecord, ...],
    ) -> WorkingBufferResult:
        kept = list(items)
        recorded = list(evictions)
        now = utc_now()
        while len(kept) > self._policy.max_items:
            candidate = min(
                kept,
                key=lambda item: self._eviction_sort_key(item, now),
            )
            kept.remove(candidate)
            recorded.append(
                create_eviction_record(
                    item_id=candidate.item_id,
                    reason=self._eviction_reason(candidate, now),
                )
            )
        return WorkingBufferResult(items=tuple(kept), evictions=tuple(recorded))

    def _eviction_sort_key(
        self,
        item: ShortTermItem,
        now: datetime,
    ) -> tuple[object, ...]:
        expired_rank = 0
        if self._policy.evict_expired_first and item.is_expired(now):
            expired_rank = -1
        resolved_rank = 0
        if (
            self._policy.evict_resolved_first
            and item.status is ShortTermItemStatus.RESOLVED
        ):
            resolved_rank = -1
        low_priority_rank = (
            -1 if item.priority < self._policy.min_priority_to_keep else 0
        )
        return (
            expired_rank,
            resolved_rank,
            low_priority_rank,
            item.priority,
            item.confidence,
            item.updated_at,
            item.created_at,
            str(item.item_id),
        )

    def _eviction_reason(self, item: ShortTermItem, now: datetime) -> EvictionReason:
        if item.is_expired(now):
            return EvictionReason.TTL_EXPIRED
        if item.status is ShortTermItemStatus.RESOLVED:
            return EvictionReason.RESOLVED
        if item.priority < self._policy.min_priority_to_keep:
            return EvictionReason.LOW_PRIORITY
        return EvictionReason.CAPACITY_LIMIT


def build_working_buffer(
    *,
    items: tuple[ShortTermItem, ...] | list[ShortTermItem] = (),
    evictions: tuple[EvictionRecord, ...] | list[EvictionRecord] = (),
    policy: WorkingBufferPolicy | None = None,
) -> WorkingBuffer:
    """Build a working buffer."""

    return WorkingBuffer(items=items, evictions=evictions, policy=policy)


def _ensure_items(
    values: tuple[ShortTermItem, ...] | list[ShortTermItem],
    *,
    require_unique: bool = True,
) -> tuple[ShortTermItem, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("items must be tuple or list")
    result = tuple(values)
    for item in result:
        if not isinstance(item, ShortTermItem):
            raise TypeError("items must contain ShortTermItem")
    if require_unique:
        item_ids = tuple(str(item.item_id) for item in result)
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("item_id values must be unique")
    return result


def _ensure_evictions(
    values: tuple[EvictionRecord, ...] | list[EvictionRecord],
) -> tuple[EvictionRecord, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("evictions must be tuple or list")
    result = tuple(values)
    for eviction in result:
        if not isinstance(eviction, EvictionRecord):
            raise TypeError("evictions must contain EvictionRecord")
    eviction_ids = tuple(str(eviction.eviction_id) for eviction in result)
    if len(set(eviction_ids)) != len(eviction_ids):
        raise ValueError("eviction_id values must be unique")
    return result


def _deduplicate_items(
    items: tuple[ShortTermItem, ...],
) -> tuple[ShortTermItem, ...]:
    by_id: dict[str, ShortTermItem] = {}
    order: list[str] = []
    for item in items:
        item_id = str(item.item_id)
        if item_id not in by_id:
            order.append(item_id)
        by_id[item_id] = item
    return tuple(by_id[item_id] for item_id in order)


def _replace_item(
    items: tuple[ShortTermItem, ...],
    replacement: ShortTermItem,
) -> tuple[ShortTermItem, ...]:
    replacement_id = str(replacement.item_id)
    replaced = False
    result: list[ShortTermItem] = []
    for item in items:
        if str(item.item_id) == replacement_id:
            result.append(replacement)
            replaced = True
        else:
            result.append(item)
    if not replaced:
        result.append(replacement)
    return tuple(result)
