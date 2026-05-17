"""Deterministic runtime active context management for CEBA STM."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from ceba.memory.short_term.schema import (
    ActiveContextRef,
    ActiveContextStatus,
    ShortTermItem,
    create_active_context_ref,
    ensure_non_empty_string,
    utc_now,
)


__all__ = [
    "ActiveContextPolicy",
    "ActiveContextResult",
    "ActiveContextManager",
    "build_active_context_manager",
]


_UNCHANGED = object()


@dataclass(frozen=True)
class ActiveContextPolicy:
    """Configuration object controlling deterministic active context behavior."""

    max_items_per_context: int = 128
    allow_empty_context: bool = False
    auto_degrade_empty_context: bool = True
    auto_close_empty_context: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.max_items_per_context, bool) or not isinstance(
            self.max_items_per_context,
            int,
        ):
            raise TypeError("max_items_per_context must be int")
        if self.max_items_per_context <= 0:
            raise ValueError("max_items_per_context must be positive")
        if not isinstance(self.allow_empty_context, bool):
            raise TypeError("allow_empty_context must be bool")
        if not isinstance(self.auto_degrade_empty_context, bool):
            raise TypeError("auto_degrade_empty_context must be bool")
        if not isinstance(self.auto_close_empty_context, bool):
            raise TypeError("auto_close_empty_context must be bool")
        if self.auto_degrade_empty_context and self.auto_close_empty_context:
            raise ValueError(
                "auto_degrade_empty_context and auto_close_empty_context "
                "must not both be True"
            )

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "max_items_per_context": self.max_items_per_context,
            "allow_empty_context": self.allow_empty_context,
            "auto_degrade_empty_context": self.auto_degrade_empty_context,
            "auto_close_empty_context": self.auto_close_empty_context,
        }


@dataclass(frozen=True)
class ActiveContextResult:
    """Immutable result object for active context operations."""

    context: ActiveContextRef
    changed: bool
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.context, ActiveContextRef):
            raise TypeError("context must be ActiveContextRef")
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        reason = ensure_non_empty_string(self.reason, "reason")
        object.__setattr__(self, "reason", reason)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "context": self.context.to_dict(),
            "changed": self.changed,
            "reason": self.reason,
        }


class ActiveContextManager:
    """Deterministic runtime manager for one ActiveContextRef."""

    def __init__(
        self,
        *,
        context: ActiveContextRef | None = None,
        items: tuple[ShortTermItem, ...] | list[ShortTermItem] = (),
        policy: ActiveContextPolicy | None = None,
    ) -> None:
        self._policy = ActiveContextPolicy() if policy is None else policy
        if not isinstance(self._policy, ActiveContextPolicy):
            raise TypeError("policy must be ActiveContextPolicy")

        item_values = _ensure_items(items)
        known_item_ids = tuple(item.item_id for item in item_values)
        if context is None:
            context = create_active_context_ref(item_ids=known_item_ids)
        elif not isinstance(context, ActiveContextRef):
            raise TypeError("context must be ActiveContextRef")

        if known_item_ids:
            known = set(known_item_ids)
            for item_id in context.item_ids:
                if item_id not in known:
                    raise ValueError("context.item_ids must exist in items")

        item_ids, root_item_id = self._enforce_capacity(
            context.item_ids,
            context.root_item_id,
        )
        status = self._apply_empty_context_policy(context.status, item_ids)
        self._context = ActiveContextRef(
            context_id=context.context_id,
            status=status,
            root_item_id=root_item_id,
            item_ids=item_ids,
            opened_at=context.opened_at,
            updated_at=context.updated_at,
            metadata=dict(context.metadata),
        )
        self._known_item_ids = known_item_ids

    def context(self) -> ActiveContextRef:
        """Return current context."""

        return self._context

    def item_ids(self) -> tuple[str, ...]:
        """Return current context item ids as strings."""

        return tuple(str(item_id) for item_id in self._context.item_ids)

    def item_count(self) -> int:
        """Return number of item ids in current context."""

        return len(self._context.item_ids)

    def status(self) -> ActiveContextStatus:
        """Return current context status."""

        return self._context.status

    def has_item(self, item_id: str) -> bool:
        """Return whether an item id exists in the context."""

        parsed = _parse_uuid(item_id, "item_id")
        return parsed in self._context.item_ids

    def add_item(self, item: ShortTermItem) -> ActiveContextResult:
        """Add an item to the active context."""

        if not isinstance(item, ShortTermItem):
            raise TypeError("item must be ShortTermItem")
        if item.item_id not in self._known_item_ids:
            self._known_item_ids = (*self._known_item_ids, item.item_id)
        return self._add_uuid(item.item_id, "add_item")

    def add_item_id(self, item_id: str) -> ActiveContextResult:
        """Add an item id to the active context."""

        return self._add_uuid(_parse_uuid(item_id, "item_id"), "add_item_id")

    def remove_item_id(self, item_id: str) -> ActiveContextResult:
        """Remove an item id from the active context."""

        parsed = _parse_uuid(item_id, "item_id")
        if parsed not in self._context.item_ids:
            return self._result(False, "item_id missing")
        item_ids = tuple(
            current_id for current_id in self._context.item_ids if current_id != parsed
        )
        root_item_id = self._context.root_item_id
        if root_item_id == parsed:
            root_item_id = item_ids[0] if item_ids else None
        status = self._apply_empty_context_policy(self._context.status, item_ids)
        self._context = self._updated_context(
            status=status,
            root_item_id=root_item_id,
            item_ids=item_ids,
            operation="remove_item_id",
        )
        return self._result(True, "item_id removed")

    def set_root_item_id(self, item_id: str) -> ActiveContextResult:
        """Set the context root item id."""

        parsed = _parse_uuid(item_id, "item_id")
        if parsed not in self._context.item_ids:
            raise ValueError("item_id must exist in context.item_ids")
        if self._context.root_item_id == parsed:
            return self._result(False, "root_item_id unchanged")
        self._context = self._updated_context(
            root_item_id=parsed,
            operation="set_root_item_id",
        )
        return self._result(True, "root_item_id set")

    def set_status(self, status: ActiveContextStatus) -> ActiveContextResult:
        """Set context status."""

        if not isinstance(status, ActiveContextStatus):
            raise TypeError("status must be ActiveContextStatus")
        if self._context.status is status:
            return self._result(False, "status unchanged")
        self._context = self._updated_context(
            status=status,
            operation="set_status",
        )
        return self._result(True, "status set")

    def degrade(self) -> ActiveContextResult:
        """Set context status to DEGRADED."""

        return self.set_status(ActiveContextStatus.DEGRADED)

    def close(self) -> ActiveContextResult:
        """Set context status to CLOSED."""

        return self.set_status(ActiveContextStatus.CLOSED)

    def expire(self) -> ActiveContextResult:
        """Set context status to EXPIRED."""

        return self.set_status(ActiveContextStatus.EXPIRED)

    def reset(
        self,
        *,
        root_item_id: str | None = None,
        item_ids: tuple[str, ...] | list[str] = (),
    ) -> ActiveContextResult:
        """Reset the active context contents."""

        parsed_item_ids = tuple(_parse_uuid(value, "item_ids") for value in item_ids)
        parsed_root = (
            None
            if root_item_id is None
            else _parse_uuid(root_item_id, "root_item_id")
        )
        if parsed_root is not None and not parsed_item_ids:
            parsed_item_ids = (parsed_root,)
        if parsed_root is not None and parsed_root not in parsed_item_ids:
            raise ValueError("root_item_id must exist in item_ids")
        item_ids_value = _deduplicate_uuids(parsed_item_ids)
        item_ids_value, parsed_root = self._enforce_capacity(
            item_ids_value,
            parsed_root,
        )
        status = self._apply_empty_context_policy(
            ActiveContextStatus.ACTIVE,
            item_ids_value,
        )
        self._context = self._updated_context(
            status=status,
            root_item_id=parsed_root,
            item_ids=item_ids_value,
            operation="reset",
        )
        return self._result(True, "context reset")

    def _add_uuid(self, item_id: UUID, operation: str) -> ActiveContextResult:
        if item_id in self._context.item_ids:
            return self._result(False, "item_id already present")
        item_ids = (*self._context.item_ids, item_id)
        root_item_id = self._context.root_item_id
        if root_item_id is None:
            root_item_id = item_ids[0]
        item_ids, root_item_id = self._enforce_capacity(item_ids, root_item_id)
        status = self._apply_empty_context_policy(self._context.status, item_ids)
        self._context = self._updated_context(
            status=status,
            root_item_id=root_item_id,
            item_ids=item_ids,
            operation=operation,
        )
        return self._result(True, "item_id added")

    def _updated_context(
        self,
        *,
        status: ActiveContextStatus | None = None,
        root_item_id: UUID | None | object = _UNCHANGED,
        item_ids: tuple[UUID, ...] | None = None,
        operation: str,
    ) -> ActiveContextRef:
        metadata = dict(self._context.metadata)
        metadata["last_operation"] = operation
        next_root = (
            self._context.root_item_id if root_item_id is _UNCHANGED else root_item_id
        )
        if next_root is not None and not isinstance(next_root, UUID):
            raise TypeError("root_item_id must be UUID or None")
        return ActiveContextRef(
            context_id=self._context.context_id,
            status=self._context.status if status is None else status,
            root_item_id=next_root,
            item_ids=self._context.item_ids if item_ids is None else item_ids,
            opened_at=self._context.opened_at,
            updated_at=utc_now(),
            metadata=metadata,
        )

    def _enforce_capacity(
        self,
        item_ids: tuple[UUID, ...],
        root_item_id: UUID | None,
    ) -> tuple[tuple[UUID, ...], UUID | None]:
        if len(item_ids) <= self._policy.max_items_per_context:
            return item_ids, root_item_id
        kept = item_ids[-self._policy.max_items_per_context :]
        if root_item_id not in kept:
            root_item_id = kept[0] if kept else None
        return kept, root_item_id

    def _apply_empty_context_policy(
        self,
        status: ActiveContextStatus,
        item_ids: tuple[UUID, ...],
    ) -> ActiveContextStatus:
        if item_ids or self._policy.allow_empty_context:
            return status
        if self._policy.auto_degrade_empty_context:
            return ActiveContextStatus.DEGRADED
        if self._policy.auto_close_empty_context:
            return ActiveContextStatus.CLOSED
        raise ValueError("empty context is not allowed")

    def _result(self, changed: bool, reason: str) -> ActiveContextResult:
        return ActiveContextResult(
            context=self._context,
            changed=changed,
            reason=reason,
        )


def build_active_context_manager(
    *,
    context: ActiveContextRef | None = None,
    items: tuple[ShortTermItem, ...] | list[ShortTermItem] = (),
    policy: ActiveContextPolicy | None = None,
) -> ActiveContextManager:
    """Build an active context manager."""

    return ActiveContextManager(context=context, items=items, policy=policy)


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
