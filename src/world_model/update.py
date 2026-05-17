"""Deterministic update utilities for CEBA World Model snapshots."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from ceba.world_model.errors import WorldModelSnapshotError
from ceba.world_model.schema import (
    PredictionInput,
    PredictionResult,
    WorldModelEvidence,
    WorldModelSnapshot,
    WorldStateRef,
    create_world_model_snapshot,
    ensure_non_empty_string,
)


__all__ = [
    "WorldModelUpdatePolicy",
    "WorldModelUpdateResult",
    "WorldModelUpdater",
    "update_world_model_snapshot",
]


_T = TypeVar("_T")


@dataclass(frozen=True)
class WorldModelUpdatePolicy:
    """Configuration object controlling world model snapshot updates."""

    max_states: int = 512
    max_prediction_inputs: int = 512
    max_prediction_results: int = 512
    max_evidence_items: int = 1024
    allow_empty_snapshot: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_states",
            _ensure_positive_int(self.max_states, "max_states"),
        )
        object.__setattr__(
            self,
            "max_prediction_inputs",
            _ensure_positive_int(
                self.max_prediction_inputs,
                "max_prediction_inputs",
            ),
        )
        object.__setattr__(
            self,
            "max_prediction_results",
            _ensure_positive_int(
                self.max_prediction_results,
                "max_prediction_results",
            ),
        )
        object.__setattr__(
            self,
            "max_evidence_items",
            _ensure_positive_int(
                self.max_evidence_items,
                "max_evidence_items",
            ),
        )
        if not isinstance(self.allow_empty_snapshot, bool):
            raise TypeError("allow_empty_snapshot must be bool")

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "max_states": self.max_states,
            "max_prediction_inputs": self.max_prediction_inputs,
            "max_prediction_results": self.max_prediction_results,
            "max_evidence_items": self.max_evidence_items,
            "allow_empty_snapshot": self.allow_empty_snapshot,
        }


@dataclass(frozen=True)
class WorldModelUpdateResult:
    """Immutable result object returned by world model snapshot update."""

    snapshot: WorldModelSnapshot
    changed: bool
    reason: str
    skipped_ids: tuple[str, ...] = ()
    skip_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, WorldModelSnapshot):
            raise TypeError("snapshot must be WorldModelSnapshot")
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        reason = ensure_non_empty_string(self.reason, "reason")
        skipped_ids = _ensure_string_tuple(self.skipped_ids, "skipped_ids")
        skip_reasons = _ensure_string_tuple(self.skip_reasons, "skip_reasons")
        if len(skipped_ids) != len(skip_reasons):
            raise ValueError("skipped_ids and skip_reasons must match")
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "skipped_ids", skipped_ids)
        object.__setattr__(self, "skip_reasons", skip_reasons)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "snapshot": self.snapshot.to_dict(),
            "changed": self.changed,
            "reason": self.reason,
            "skipped_ids": self.skipped_ids,
            "skip_reasons": self.skip_reasons,
        }

    def skipped_count(self) -> int:
        """Return skipped object count."""

        return len(self.skipped_ids)


class WorldModelUpdater:
    """Deterministic service class that updates WorldModelSnapshot structure."""

    def __init__(self, policy: WorldModelUpdatePolicy | None = None) -> None:
        self._policy = WorldModelUpdatePolicy() if policy is None else policy
        if not isinstance(self._policy, WorldModelUpdatePolicy):
            raise TypeError("policy must be WorldModelUpdatePolicy")

    def update(
        self,
        *,
        snapshot: WorldModelSnapshot,
        states: tuple[WorldStateRef, ...] | list[WorldStateRef] = (),
        prediction_inputs: (
            tuple[PredictionInput, ...] | list[PredictionInput]
        ) = (),
        prediction_results: (
            tuple[PredictionResult, ...] | list[PredictionResult]
        ) = (),
        evidence: tuple[WorldModelEvidence, ...] | list[WorldModelEvidence] = (),
        metadata: dict[str, object] | None = None,
    ) -> WorldModelUpdateResult:
        """Update a world model snapshot structure."""

        if not isinstance(snapshot, WorldModelSnapshot):
            raise TypeError("snapshot must be WorldModelSnapshot")
        state_values = _ensure_states(states)
        input_values = _ensure_prediction_inputs(prediction_inputs)
        result_values = _ensure_prediction_results(prediction_results)
        evidence_values = _ensure_evidence(evidence)

        next_states = _replace_by_id(
            snapshot.states,
            state_values,
            lambda state: state.state_id,
        )
        next_inputs = _replace_by_id(
            snapshot.prediction_inputs,
            input_values,
            lambda item: item.input_id,
        )
        next_results = _replace_by_id(
            snapshot.prediction_results,
            result_values,
            lambda result: result.result_id,
        )
        next_evidence = _replace_by_id(
            snapshot.evidence,
            evidence_values,
            lambda item: item.evidence_id,
        )

        skipped_ids: list[str] = []
        skip_reasons: list[str] = []
        next_states = self._limit_states(next_states, skipped_ids, skip_reasons)
        next_inputs = self._limit_prediction_inputs(
            next_inputs,
            skipped_ids,
            skip_reasons,
        )
        next_results = self._limit_prediction_results(
            next_results,
            skipped_ids,
            skip_reasons,
        )
        next_evidence = self._limit_evidence(
            next_evidence,
            skipped_ids,
            skip_reasons,
        )

        self._ensure_non_empty(
            states=next_states,
            prediction_inputs=next_inputs,
            prediction_results=next_results,
            evidence=next_evidence,
        )
        changed = bool(
            state_values
            or input_values
            or result_values
            or evidence_values
            or skipped_ids
        )
        if not changed:
            return WorldModelUpdateResult(
                snapshot=snapshot,
                changed=False,
                reason="world model snapshot unchanged",
            )

        update_metadata: dict[str, object] = {
            "source": "world_model_updater",
            "operation": "update",
            "policy": self._policy.to_dict(),
            "previous_snapshot_id": str(snapshot.snapshot_id),
        }
        update_metadata.update(_copy_metadata({} if metadata is None else metadata))
        update_metadata["source"] = "world_model_updater"
        updated_snapshot = create_world_model_snapshot(
            states=_order_states(next_states),
            prediction_inputs=_order_prediction_inputs(next_inputs),
            prediction_results=_order_prediction_results(next_results),
            evidence=_order_evidence(next_evidence),
            metadata=update_metadata,
        )
        snapshot_metadata = dict(updated_snapshot.metadata)
        snapshot_metadata.update(update_metadata)
        snapshot_metadata["source"] = "world_model_updater"
        updated_snapshot = WorldModelSnapshot(
            snapshot_id=updated_snapshot.snapshot_id,
            states=updated_snapshot.states,
            prediction_inputs=updated_snapshot.prediction_inputs,
            prediction_results=updated_snapshot.prediction_results,
            evidence=updated_snapshot.evidence,
            created_at=updated_snapshot.created_at,
            metadata=snapshot_metadata,
        )
        return WorldModelUpdateResult(
            snapshot=updated_snapshot,
            changed=True,
            reason="world model snapshot updated",
            skipped_ids=tuple(skipped_ids),
            skip_reasons=tuple(skip_reasons),
        )

    def add_state(
        self,
        *,
        snapshot: WorldModelSnapshot,
        state: WorldStateRef,
    ) -> WorldModelUpdateResult:
        """Add or replace one world state."""

        if not isinstance(state, WorldStateRef):
            raise TypeError("state must be WorldStateRef")
        return self.update(snapshot=snapshot, states=(state,))

    def add_prediction_input(
        self,
        *,
        snapshot: WorldModelSnapshot,
        prediction_input: PredictionInput,
    ) -> WorldModelUpdateResult:
        """Add or replace one prediction input."""

        if not isinstance(prediction_input, PredictionInput):
            raise TypeError("prediction_input must be PredictionInput")
        return self.update(
            snapshot=snapshot,
            prediction_inputs=(prediction_input,),
        )

    def add_prediction_result(
        self,
        *,
        snapshot: WorldModelSnapshot,
        prediction_result: PredictionResult,
    ) -> WorldModelUpdateResult:
        """Add or replace one prediction result."""

        if not isinstance(prediction_result, PredictionResult):
            raise TypeError("prediction_result must be PredictionResult")
        return self.update(
            snapshot=snapshot,
            prediction_results=(prediction_result,),
        )

    def add_evidence(
        self,
        *,
        snapshot: WorldModelSnapshot,
        evidence: WorldModelEvidence,
    ) -> WorldModelUpdateResult:
        """Add or replace one world model evidence item."""

        if not isinstance(evidence, WorldModelEvidence):
            raise TypeError("evidence must be WorldModelEvidence")
        return self.update(snapshot=snapshot, evidence=(evidence,))

    def _limit_states(
        self,
        states: tuple[WorldStateRef, ...],
        skipped_ids: list[str],
        skip_reasons: list[str],
    ) -> tuple[WorldStateRef, ...]:
        return _limit_by_keep_order(
            values=states,
            max_count=self._policy.max_states,
            keep_key=lambda state: (
                state.updated_at,
                state.created_at,
                str(state.state_id),
            ),
            id_func=lambda state: str(state.state_id),
            skipped_ids=skipped_ids,
            skip_reasons=skip_reasons,
        )

    def _limit_prediction_inputs(
        self,
        inputs: tuple[PredictionInput, ...],
        skipped_ids: list[str],
        skip_reasons: list[str],
    ) -> tuple[PredictionInput, ...]:
        return _limit_by_keep_order(
            values=inputs,
            max_count=self._policy.max_prediction_inputs,
            keep_key=lambda item: (item.created_at, str(item.input_id)),
            id_func=lambda item: str(item.input_id),
            skipped_ids=skipped_ids,
            skip_reasons=skip_reasons,
        )

    def _limit_prediction_results(
        self,
        results: tuple[PredictionResult, ...],
        skipped_ids: list[str],
        skip_reasons: list[str],
    ) -> tuple[PredictionResult, ...]:
        return _limit_by_keep_order(
            values=results,
            max_count=self._policy.max_prediction_results,
            keep_key=lambda result: (result.created_at, str(result.result_id)),
            id_func=lambda result: str(result.result_id),
            skipped_ids=skipped_ids,
            skip_reasons=skip_reasons,
        )

    def _limit_evidence(
        self,
        evidence: tuple[WorldModelEvidence, ...],
        skipped_ids: list[str],
        skip_reasons: list[str],
    ) -> tuple[WorldModelEvidence, ...]:
        return _limit_by_keep_order(
            values=evidence,
            max_count=self._policy.max_evidence_items,
            keep_key=lambda item: (item.created_at, str(item.evidence_id)),
            id_func=lambda item: str(item.evidence_id),
            skipped_ids=skipped_ids,
            skip_reasons=skip_reasons,
        )

    def _ensure_non_empty(
        self,
        *,
        states: tuple[WorldStateRef, ...],
        prediction_inputs: tuple[PredictionInput, ...],
        prediction_results: tuple[PredictionResult, ...],
        evidence: tuple[WorldModelEvidence, ...],
    ) -> None:
        if self._policy.allow_empty_snapshot:
            return
        if states or prediction_inputs or prediction_results or evidence:
            return
        raise WorldModelSnapshotError("world model snapshot cannot be empty")


def update_world_model_snapshot(
    *,
    snapshot: WorldModelSnapshot,
    states: tuple[WorldStateRef, ...] | list[WorldStateRef] = (),
    prediction_inputs: tuple[PredictionInput, ...] | list[PredictionInput] = (),
    prediction_results: (
        tuple[PredictionResult, ...] | list[PredictionResult]
    ) = (),
    evidence: tuple[WorldModelEvidence, ...] | list[WorldModelEvidence] = (),
    policy: WorldModelUpdatePolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> WorldModelUpdateResult:
    """Update a world model snapshot."""

    updater = WorldModelUpdater(policy)
    return updater.update(
        snapshot=snapshot,
        states=states,
        prediction_inputs=prediction_inputs,
        prediction_results=prediction_results,
        evidence=evidence,
        metadata=metadata,
    )


def _ensure_states(
    values: tuple[WorldStateRef, ...] | list[WorldStateRef],
) -> tuple[WorldStateRef, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("states must be tuple or list")
    states = tuple(values)
    for state in states:
        if not isinstance(state, WorldStateRef):
            raise TypeError("states must contain WorldStateRef")
    return states


def _ensure_prediction_inputs(
    values: tuple[PredictionInput, ...] | list[PredictionInput],
) -> tuple[PredictionInput, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("prediction_inputs must be tuple or list")
    inputs = tuple(values)
    for item in inputs:
        if not isinstance(item, PredictionInput):
            raise TypeError("prediction_inputs must contain PredictionInput")
    return inputs


def _ensure_prediction_results(
    values: tuple[PredictionResult, ...] | list[PredictionResult],
) -> tuple[PredictionResult, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("prediction_results must be tuple or list")
    results = tuple(values)
    for result in results:
        if not isinstance(result, PredictionResult):
            raise TypeError("prediction_results must contain PredictionResult")
    return results


def _ensure_evidence(
    values: tuple[WorldModelEvidence, ...] | list[WorldModelEvidence],
) -> tuple[WorldModelEvidence, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("evidence must be tuple or list")
    evidence = tuple(values)
    for item in evidence:
        if not isinstance(item, WorldModelEvidence):
            raise TypeError("evidence must contain WorldModelEvidence")
    return evidence


def _replace_by_id(
    existing: tuple[_T, ...],
    updates: tuple[_T, ...],
    id_func: Callable[[_T], object],
) -> tuple[_T, ...]:
    result = list(existing)
    positions: dict[object, int] = {}
    for index, value in enumerate(result):
        positions[id_func(value)] = index
    for value in updates:
        key = id_func(value)
        if key in positions:
            result[positions[key]] = value
        else:
            positions[key] = len(result)
            result.append(value)
    return tuple(result)


def _limit_by_keep_order(
    *,
    values: tuple[_T, ...],
    max_count: int,
    keep_key: Callable[[_T], object],
    id_func: Callable[[_T], str],
    skipped_ids: list[str],
    skip_reasons: list[str],
) -> tuple[_T, ...]:
    if len(values) <= max_count:
        return values
    ordered = sorted(values, key=keep_key, reverse=True)
    kept = tuple(ordered[:max_count])
    evicted = ordered[max_count:]
    for value in evicted:
        skipped_ids.append(id_func(value))
        skip_reasons.append("world model update capacity exceeded")
    return kept


def _order_states(states: tuple[WorldStateRef, ...]) -> tuple[WorldStateRef, ...]:
    return tuple(
        sorted(states, key=lambda state: (state.created_at, str(state.state_id)))
    )


def _order_prediction_inputs(
    inputs: tuple[PredictionInput, ...],
) -> tuple[PredictionInput, ...]:
    return tuple(sorted(inputs, key=lambda item: (item.created_at, str(item.input_id))))


def _order_prediction_results(
    results: tuple[PredictionResult, ...],
) -> tuple[PredictionResult, ...]:
    return tuple(
        sorted(results, key=lambda item: (item.created_at, str(item.result_id)))
    )


def _order_evidence(
    evidence: tuple[WorldModelEvidence, ...],
) -> tuple[WorldModelEvidence, ...]:
    return tuple(
        sorted(evidence, key=lambda item: (item.created_at, str(item.evidence_id)))
    )


def _ensure_positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _ensure_string_tuple(
    values: tuple[str, ...] | list[str],
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple or list")
    return tuple(ensure_non_empty_string(value, field_name) for value in values)


def _copy_metadata(metadata: dict[str, object]) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(metadata)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
    return copied
