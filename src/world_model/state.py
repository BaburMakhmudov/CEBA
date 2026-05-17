"""Deterministic runtime world state management for CEBA World Model."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from ceba.world_model.errors import WorldModelStateError
from ceba.world_model.schema import (
    WorldModelSnapshot,
    WorldStateKind,
    WorldStateRef,
    WorldStateStatus,
    create_world_model_snapshot,
    create_world_state_ref,
    ensure_non_empty_string,
    ensure_probability,
    utc_now,
)


__all__ = [
    "WorldStatePolicy",
    "WorldStateUpdate",
    "WorldStateResult",
    "WorldStateManager",
    "build_world_state_manager",
]


@dataclass(frozen=True)
class WorldStatePolicy:
    """Configuration object controlling deterministic world state management."""

    max_states: int = 512
    min_confidence_to_keep: float = 0.0
    allow_empty_state_set: bool = True
    evict_invalid_first: bool = True
    evict_stale_first: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.max_states, bool) or not isinstance(
            self.max_states,
            int,
        ):
            raise TypeError("max_states must be int")
        if self.max_states <= 0:
            raise ValueError("max_states must be positive")
        min_confidence = ensure_probability(
            self.min_confidence_to_keep,
            "min_confidence_to_keep",
        )
        if not isinstance(self.allow_empty_state_set, bool):
            raise TypeError("allow_empty_state_set must be bool")
        if not isinstance(self.evict_invalid_first, bool):
            raise TypeError("evict_invalid_first must be bool")
        if not isinstance(self.evict_stale_first, bool):
            raise TypeError("evict_stale_first must be bool")
        object.__setattr__(
            self,
            "min_confidence_to_keep",
            min_confidence,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "max_states": self.max_states,
            "min_confidence_to_keep": self.min_confidence_to_keep,
            "allow_empty_state_set": self.allow_empty_state_set,
            "evict_invalid_first": self.evict_invalid_first,
            "evict_stale_first": self.evict_stale_first,
        }


@dataclass(frozen=True)
class WorldStateUpdate:
    """Explicit update command for one WorldStateRef."""

    state_id: str
    status: WorldStateStatus | None = None
    confidence: float | None = None
    state_ref: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        state_id = ensure_non_empty_string(self.state_id, "state_id")
        _parse_uuid(state_id, "state_id")
        if self.status is not None and not isinstance(
            self.status,
            WorldStateStatus,
        ):
            raise TypeError("status must be WorldStateStatus or None")
        confidence = (
            None
            if self.confidence is None
            else ensure_probability(self.confidence, "confidence")
        )
        state_ref = (
            None
            if self.state_ref is None
            else ensure_non_empty_string(self.state_ref, "state_ref")
        )
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "state_id", state_id)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "state_ref", state_ref)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "state_id": self.state_id,
            "status": None if self.status is None else self.status.value,
            "confidence": self.confidence,
            "state_ref": self.state_ref,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class WorldStateResult:
    """Immutable result object returned by world state operations."""

    states: tuple[WorldStateRef, ...]
    changed: bool
    reason: str
    skipped_state_ids: tuple[str, ...] = ()
    skip_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        states = _ensure_states(self.states)
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        reason = ensure_non_empty_string(self.reason, "reason")
        skipped_ids = _ensure_string_tuple(
            self.skipped_state_ids,
            "skipped_state_ids",
        )
        skip_reasons = _ensure_string_tuple(
            self.skip_reasons,
            "skip_reasons",
        )
        if len(skipped_ids) != len(skip_reasons):
            raise ValueError("skipped_state_ids and skip_reasons must match")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "skipped_state_ids", skipped_ids)
        object.__setattr__(self, "skip_reasons", skip_reasons)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "states": tuple(state.to_dict() for state in self.states),
            "changed": self.changed,
            "reason": self.reason,
            "skipped_state_ids": self.skipped_state_ids,
            "skip_reasons": self.skip_reasons,
        }

    def state_count(self) -> int:
        """Return state count."""

        return len(self.states)

    def skipped_count(self) -> int:
        """Return skipped state count."""

        return len(self.skipped_state_ids)


class WorldStateManager:
    """Deterministic runtime manager for WorldStateRef objects."""

    def __init__(
        self,
        *,
        states: tuple[WorldStateRef, ...] | list[WorldStateRef] = (),
        policy: WorldStatePolicy | None = None,
    ) -> None:
        self._policy = WorldStatePolicy() if policy is None else policy
        if not isinstance(self._policy, WorldStatePolicy):
            raise TypeError("policy must be WorldStatePolicy")
        self._states = _ensure_states(states)
        result = self._enforce_capacity(self._states)
        self._states = result.states

    def states(self) -> tuple[WorldStateRef, ...]:
        """Return current world states."""

        return self._states

    def state_count(self) -> int:
        """Return number of current states."""

        return len(self._states)

    def has_state(self, state_id: str) -> bool:
        """Return whether a state exists."""

        target_id = ensure_non_empty_string(state_id, "state_id")
        return any(str(state.state_id) == target_id for state in self._states)

    def get_state(self, state_id: str) -> WorldStateRef | None:
        """Return a matching state or None."""

        target_id = ensure_non_empty_string(state_id, "state_id")
        for state in self._states:
            if str(state.state_id) == target_id:
                return state
        return None

    def add_state(self, state: WorldStateRef) -> WorldStateResult:
        """Add or replace one world state."""

        if not isinstance(state, WorldStateRef):
            raise TypeError("state must be WorldStateRef")
        states = _replace_state(self._states, state)
        result = self._enforce_capacity(states)
        self._states = result.states
        return WorldStateResult(
            states=self._states,
            changed=True,
            reason="world state added",
            skipped_state_ids=result.skipped_state_ids,
            skip_reasons=result.skip_reasons,
        )

    def create_state(
        self,
        *,
        state_ref: str,
        kind: WorldStateKind = WorldStateKind.UNKNOWN,
        status: WorldStateStatus = WorldStateStatus.ACTIVE,
        confidence: float = 1.0,
        metadata: dict[str, object] | None = None,
    ) -> WorldStateResult:
        """Create and add a world state."""

        default_metadata: dict[str, object] = {
            "source": "world_state_manager",
            "operation": "create_state",
        }
        state = create_world_state_ref(
            state_ref=state_ref,
            kind=kind,
            status=status,
            confidence=confidence,
            metadata=_merge_metadata(default_metadata, metadata),
        )
        state_metadata = dict(state.metadata)
        state_metadata.update(_merge_metadata(default_metadata, metadata))
        state_metadata["source"] = "world_state_manager"
        managed_state = WorldStateRef(
            state_id=state.state_id,
            kind=state.kind,
            status=state.status,
            state_ref=state.state_ref,
            confidence=state.confidence,
            created_at=state.created_at,
            updated_at=state.updated_at,
            metadata=state_metadata,
        )
        return self.add_state(managed_state)

    def update_state(self, update: WorldStateUpdate) -> WorldStateResult:
        """Apply one world state update."""

        if not isinstance(update, WorldStateUpdate):
            raise TypeError("update must be WorldStateUpdate")
        existing = self.get_state(update.state_id)
        if existing is None:
            return self._result(
                changed=False,
                reason="world state unchanged",
                skipped_state_ids=(update.state_id,),
                skip_reasons=("state_id missing",),
            )

        metadata = dict(existing.metadata)
        metadata.update(update.metadata)
        metadata.update(
            {
                "source": "world_state_manager",
                "operation": "update_state",
            }
        )
        metadata["source"] = "world_state_manager"
        updated = WorldStateRef(
            state_id=existing.state_id,
            kind=existing.kind,
            status=existing.status if update.status is None else update.status,
            state_ref=(
                existing.state_ref
                if update.state_ref is None
                else update.state_ref
            ),
            confidence=(
                existing.confidence
                if update.confidence is None
                else update.confidence
            ),
            created_at=existing.created_at,
            updated_at=utc_now(),
            metadata=metadata,
        )
        states = _replace_state(self._states, updated)
        capacity_result = self._enforce_capacity(states)
        self._states = capacity_result.states
        return WorldStateResult(
            states=self._states,
            changed=True,
            reason="world state updated",
            skipped_state_ids=capacity_result.skipped_state_ids,
            skip_reasons=capacity_result.skip_reasons,
        )

    def set_status(
        self,
        *,
        state_id: str,
        status: WorldStateStatus,
    ) -> WorldStateResult:
        """Set one world state status."""

        if not isinstance(status, WorldStateStatus):
            raise TypeError("status must be WorldStateStatus")
        return self.update_state(
            WorldStateUpdate(state_id=state_id, status=status)
        )

    def mark_stale(self, state_id: str) -> WorldStateResult:
        """Mark one world state as stale."""

        return self.set_status(
            state_id=state_id,
            status=WorldStateStatus.STALE,
        )

    def mark_degraded(self, state_id: str) -> WorldStateResult:
        """Mark one world state as degraded."""

        return self.set_status(
            state_id=state_id,
            status=WorldStateStatus.DEGRADED,
        )

    def mark_superseded(self, state_id: str) -> WorldStateResult:
        """Mark one world state as superseded."""

        return self.set_status(
            state_id=state_id,
            status=WorldStateStatus.SUPERSEDED,
        )

    def mark_invalid(self, state_id: str) -> WorldStateResult:
        """Mark one world state as invalid."""

        return self.set_status(
            state_id=state_id,
            status=WorldStateStatus.INVALID,
        )

    def remove_state(self, state_id: str) -> WorldStateResult:
        """Remove one world state."""

        target_id = ensure_non_empty_string(state_id, "state_id")
        if not self.has_state(target_id):
            return self._result(
                changed=False,
                reason="world state unchanged",
                skipped_state_ids=(target_id,),
                skip_reasons=("state_id missing",),
            )
        states = tuple(
            state for state in self._states if str(state.state_id) != target_id
        )
        if not states and not self._policy.allow_empty_state_set:
            raise WorldModelStateError(
                "world state set cannot be empty",
                context={"state_id": target_id},
            )
        self._states = states
        return self._result(changed=True, reason="world state removed")

    def snapshot(self) -> WorldModelSnapshot:
        """Return a world model snapshot for current states."""

        return create_world_model_snapshot(
            states=self._states,
            prediction_inputs=(),
            prediction_results=(),
            evidence=(),
            metadata={
                "runtime_source": "world_state_manager",
                "policy": self._policy.to_dict(),
            },
        )

    def _enforce_capacity(
        self,
        states: tuple[WorldStateRef, ...],
    ) -> WorldStateResult:
        kept = list(states)
        skipped_ids: list[str] = []
        skip_reasons: list[str] = []
        while len(kept) > self._policy.max_states:
            candidate = min(kept, key=self._capacity_sort_key)
            kept.remove(candidate)
            skipped_ids.append(str(candidate.state_id))
            skip_reasons.append("state capacity evicted")
        return WorldStateResult(
            states=tuple(kept),
            changed=bool(skipped_ids),
            reason="world state capacity enforced",
            skipped_state_ids=tuple(skipped_ids),
            skip_reasons=tuple(skip_reasons),
        )

    def _capacity_sort_key(self, state: WorldStateRef) -> tuple[object, ...]:
        invalid_rank = 0
        if (
            self._policy.evict_invalid_first
            and state.status is WorldStateStatus.INVALID
        ):
            invalid_rank = -1
        stale_rank = 0
        if (
            self._policy.evict_stale_first
            and state.status is WorldStateStatus.STALE
        ):
            stale_rank = -1
        low_confidence_rank = (
            -1
            if state.confidence < self._policy.min_confidence_to_keep
            else 0
        )
        return (
            invalid_rank,
            stale_rank,
            low_confidence_rank,
            state.confidence,
            state.updated_at,
            state.created_at,
            str(state.state_id),
        )

    def _result(
        self,
        *,
        changed: bool,
        reason: str,
        skipped_state_ids: tuple[str, ...] = (),
        skip_reasons: tuple[str, ...] = (),
    ) -> WorldStateResult:
        return WorldStateResult(
            states=self._states,
            changed=changed,
            reason=reason,
            skipped_state_ids=skipped_state_ids,
            skip_reasons=skip_reasons,
        )


def build_world_state_manager(
    *,
    states: tuple[WorldStateRef, ...] | list[WorldStateRef] = (),
    policy: WorldStatePolicy | None = None,
) -> WorldStateManager:
    """Build a world state manager."""

    return WorldStateManager(states=states, policy=policy)


def _ensure_states(
    values: tuple[WorldStateRef, ...] | list[WorldStateRef],
) -> tuple[WorldStateRef, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("states must be tuple or list")
    states = tuple(values)
    for state in states:
        if not isinstance(state, WorldStateRef):
            raise TypeError("states must contain WorldStateRef")
    state_ids = tuple(state.state_id for state in states)
    if len(set(state_ids)) != len(state_ids):
        raise ValueError("state_id values must be unique")
    return states


def _replace_state(
    states: tuple[WorldStateRef, ...],
    state: WorldStateRef,
) -> tuple[WorldStateRef, ...]:
    replaced = False
    next_states: list[WorldStateRef] = []
    for current in states:
        if current.state_id == state.state_id:
            next_states.append(state)
            replaced = True
        else:
            next_states.append(current)
    if not replaced:
        next_states.append(state)
    return tuple(next_states)


def _parse_uuid(value: str, field_name: str) -> UUID:
    try:
        return UUID(ensure_non_empty_string(value, field_name))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be UUID string") from exc


def _ensure_string_tuple(
    values: tuple[str, ...] | list[str],
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple or list")
    return tuple(ensure_non_empty_string(value, field_name) for value in values)


def _merge_metadata(
    default_metadata: dict[str, object],
    metadata: dict[str, object] | None,
) -> dict[str, object]:
    merged = _copy_metadata(default_metadata)
    if metadata is not None:
        merged.update(_copy_metadata(metadata))
    if "source" in default_metadata:
        merged["source"] = default_metadata["source"]
    return merged


def _copy_metadata(metadata: dict[str, object]) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(metadata)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
    return copied
