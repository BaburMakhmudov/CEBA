"""Deterministic local Canonical Ledger utilities for CEBA persistence."""

from __future__ import annotations

from dataclasses import dataclass, field

from ceba.persistence.ledger.schema import (
    CanonicalLedgerRecord,
    LedgerPayload,
    LedgerProvenance,
    LedgerRecordKind,
    LedgerRecordRef,
    LedgerRecordStatus,
    LedgerSnapshot,
    create_canonical_ledger_record,
    create_ledger_snapshot,
    ensure_non_empty_string,
    ensure_non_negative_int,
    utc_now,
)


__all__ = [
    "CanonicalLedgerPolicy",
    "CanonicalLedgerState",
    "CanonicalLedgerResult",
    "CanonicalLedger",
    "build_canonical_ledger",
]


_FORBIDDEN_RUNTIME_STATE_KEYS = {
    "M_t",
    "Mt",
    "semantic_state",
    "runtime_semantic_state",
}


@dataclass(frozen=True)
class CanonicalLedgerPolicy:
    """Configuration object controlling local canonical ledger behavior."""

    allow_empty_ledger: bool = True
    allow_duplicate_record_keys: bool = False
    max_records: int = 100000
    reject_runtime_semantic_state: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.allow_empty_ledger, bool):
            raise TypeError("allow_empty_ledger must be bool")
        if not isinstance(self.allow_duplicate_record_keys, bool):
            raise TypeError("allow_duplicate_record_keys must be bool")
        object.__setattr__(
            self,
            "max_records",
            _ensure_positive_int(self.max_records, "max_records"),
        )
        if not isinstance(self.reject_runtime_semantic_state, bool):
            raise TypeError("reject_runtime_semantic_state must be bool")

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "allow_empty_ledger": self.allow_empty_ledger,
            "allow_duplicate_record_keys": self.allow_duplicate_record_keys,
            "max_records": self.max_records,
            "reject_runtime_semantic_state": self.reject_runtime_semantic_state,
        }


@dataclass(frozen=True)
class CanonicalLedgerState:
    """Immutable local state bundle for canonical ledger records."""

    records: tuple[CanonicalLedgerRecord, ...] = ()
    snapshot: LedgerSnapshot | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        records = _ensure_record_tuple(self.records)
        _ensure_unique_values(
            tuple(str(record.record_id) for record in records),
            "record_id",
        )
        metadata = _copy_metadata(self.metadata)
        if metadata.get("allow_duplicate_record_keys") is not True:
            _ensure_unique_values(
                tuple(record.record_key for record in records),
                "record_key",
            )
        if self.snapshot is not None and not isinstance(self.snapshot, LedgerSnapshot):
            raise TypeError("snapshot must be LedgerSnapshot or None")
        if self.snapshot is not None:
            record_ids = {record.record_id for record in records}
            for snapshot_record in self.snapshot.records:
                if snapshot_record.record_id not in record_ids:
                    raise ValueError("every snapshot record_id must exist in records")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "records": tuple(record.to_dict() for record in self.records),
            "snapshot": None if self.snapshot is None else self.snapshot.to_dict(),
            "metadata": dict(self.metadata),
        }

    def record_count(self) -> int:
        """Return record count."""

        return len(self.records)

    def has_snapshot(self) -> bool:
        """Return whether state has a snapshot."""

        return self.snapshot is not None

    def refs(self) -> tuple[LedgerRecordRef, ...]:
        """Return ledger record references."""

        return tuple(record.to_ref() for record in self.records)


@dataclass(frozen=True)
class CanonicalLedgerResult:
    """Immutable result returned by canonical ledger operations."""

    state: CanonicalLedgerState
    changed: bool
    reason: str
    record: CanonicalLedgerRecord | None = None
    snapshot: LedgerSnapshot | None = None
    rejected_record_keys: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.state, CanonicalLedgerState):
            raise TypeError("state must be CanonicalLedgerState")
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        reason = ensure_non_empty_string(self.reason, "reason")
        if self.record is not None and not isinstance(
            self.record,
            CanonicalLedgerRecord,
        ):
            raise TypeError("record must be CanonicalLedgerRecord or None")
        if self.snapshot is not None and not isinstance(self.snapshot, LedgerSnapshot):
            raise TypeError("snapshot must be LedgerSnapshot or None")
        rejected_record_keys = _ensure_string_tuple(
            self.rejected_record_keys,
            "rejected_record_keys",
        )
        rejection_reasons = _ensure_string_tuple(
            self.rejection_reasons,
            "rejection_reasons",
        )
        if len(rejected_record_keys) != len(rejection_reasons):
            raise ValueError("rejected_record_keys and rejection_reasons must match")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "rejected_record_keys", rejected_record_keys)
        object.__setattr__(self, "rejection_reasons", rejection_reasons)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "state": self.state.to_dict(),
            "changed": self.changed,
            "reason": self.reason,
            "record": None if self.record is None else self.record.to_dict(),
            "snapshot": None if self.snapshot is None else self.snapshot.to_dict(),
            "rejected_record_keys": self.rejected_record_keys,
            "rejection_reasons": self.rejection_reasons,
            "metadata": dict(self.metadata),
        }

    def rejected_count(self) -> int:
        """Return rejected record count."""

        return len(self.rejected_record_keys)

    def has_record(self) -> bool:
        """Return whether result includes a record."""

        return self.record is not None

    def has_snapshot(self) -> bool:
        """Return whether result includes a snapshot."""

        return self.snapshot is not None


class CanonicalLedger:
    """Deterministic local manager for canonical ledger records."""

    def __init__(
        self,
        *,
        state: CanonicalLedgerState | None = None,
        policy: CanonicalLedgerPolicy | None = None,
    ) -> None:
        self._policy = CanonicalLedgerPolicy() if policy is None else policy
        if not isinstance(self._policy, CanonicalLedgerPolicy):
            raise TypeError("policy must be CanonicalLedgerPolicy")
        if state is None:
            state = CanonicalLedgerState(
                records=(),
                snapshot=None,
                metadata=self._state_metadata(record_count=0),
            )
        if not isinstance(state, CanonicalLedgerState):
            raise TypeError("state must be CanonicalLedgerState")
        self._validate_records_under_policy(state.records)
        self._state = self._state_for_policy(state)

    def state(self) -> CanonicalLedgerState:
        """Return current ledger state."""

        return self._state

    def records(self) -> tuple[CanonicalLedgerRecord, ...]:
        """Return current records."""

        return self._state.records

    def record_count(self) -> int:
        """Return current record count."""

        return len(self._state.records)

    def has_record_id(self, record_id: str) -> bool:
        """Return whether a record id exists."""

        normalized = ensure_non_empty_string(record_id, "record_id")
        return any(str(record.record_id) == normalized for record in self._state.records)

    def has_record_key(self, record_key: str) -> bool:
        """Return whether a record key exists."""

        normalized = ensure_non_empty_string(record_key, "record_key")
        return any(record.record_key == normalized for record in self._state.records)

    def get_record_by_id(self, record_id: str) -> CanonicalLedgerRecord | None:
        """Return a record by id, if present."""

        normalized = ensure_non_empty_string(record_id, "record_id")
        for record in self._state.records:
            if str(record.record_id) == normalized:
                return record
        return None

    def get_record_by_key(self, record_key: str) -> CanonicalLedgerRecord | None:
        """Return the first record with matching key, if present."""

        normalized = ensure_non_empty_string(record_key, "record_key")
        for record in self._state.records:
            if record.record_key == normalized:
                return record
        return None

    def append_record(
        self,
        record: CanonicalLedgerRecord,
        *,
        metadata: dict[str, object] | None = None,
    ) -> CanonicalLedgerResult:
        """Append a canonical ledger record."""

        if not isinstance(record, CanonicalLedgerRecord):
            raise TypeError("record must be CanonicalLedgerRecord")
        required_metadata = self._operation_metadata(
            operation="append_record",
            record=record,
        )
        result_metadata = _merge_metadata(required_metadata, metadata)
        rejection = self._record_rejection_reason(record)
        if rejection is not None:
            return _result(
                state=self._state,
                changed=False,
                reason=rejection,
                record=None,
                snapshot=None,
                rejected_record_keys=(record.record_key,),
                rejection_reasons=(rejection,),
                metadata=result_metadata,
            )
        records = self._state.records + (record,)
        updated_state = CanonicalLedgerState(
            records=records,
            snapshot=self._state.snapshot,
            metadata=self._state_metadata(record_count=len(records)),
        )
        self._state = updated_state
        return _result(
            state=updated_state,
            changed=True,
            reason="canonical ledger record appended",
            record=record,
            snapshot=updated_state.snapshot,
            metadata=result_metadata,
        )

    def create_record(
        self,
        *,
        record_key: str,
        payload: LedgerPayload,
        kind: LedgerRecordKind = LedgerRecordKind.UNKNOWN,
        status: LedgerRecordStatus = LedgerRecordStatus.CREATED,
        provenance: tuple[LedgerProvenance, ...] | list[LedgerProvenance] = (),
        semantic_hash: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> CanonicalLedgerResult:
        """Create and append a canonical ledger record."""

        required_metadata = {
            "source": "canonical_ledger",
            "operation": "create_record",
        }
        record = create_canonical_ledger_record(
            record_key=record_key,
            payload=payload,
            kind=kind,
            status=status,
            provenance=provenance,
            semantic_hash=semantic_hash,
            metadata=_merge_metadata(required_metadata, metadata),
        )
        return self.append_record(
            record,
            metadata=_merge_metadata(required_metadata, metadata),
        )

    def update_record_status(
        self,
        *,
        record_id: str,
        status: LedgerRecordStatus,
        metadata: dict[str, object] | None = None,
    ) -> CanonicalLedgerResult:
        """Update a record status immutably."""

        normalized_id = ensure_non_empty_string(record_id, "record_id")
        if not isinstance(status, LedgerRecordStatus):
            raise TypeError("status must be LedgerRecordStatus")
        existing = self.get_record_by_id(normalized_id)
        if existing is None:
            reason = "ledger record_id missing"
            return _result(
                state=self._state,
                changed=False,
                reason=reason,
                rejected_record_keys=(normalized_id,),
                rejection_reasons=(reason,),
                metadata=_merge_metadata(
                    {
                        "source": "canonical_ledger",
                        "operation": "update_record_status",
                        "record_id": normalized_id,
                    },
                    metadata,
                ),
            )
        required_metadata = {
            "source": "canonical_ledger",
            "operation": "update_record_status",
            "previous_status": existing.status.value,
            "new_status": status.value,
        }
        updated_record = CanonicalLedgerRecord(
            record_id=existing.record_id,
            kind=existing.kind,
            status=status,
            record_key=existing.record_key,
            payload=existing.payload,
            provenance=existing.provenance,
            semantic_hash=existing.semantic_hash,
            created_at=existing.created_at,
            updated_at=utc_now(),
            metadata=_merge_runtime_metadata(
                existing.metadata,
                required_metadata,
                metadata,
            ),
        )
        updated_records = _replace_record(
            records=self._state.records,
            record=updated_record,
        )
        updated_state = CanonicalLedgerState(
            records=updated_records,
            snapshot=self._state.snapshot,
            metadata=self._state_metadata(record_count=len(updated_records)),
        )
        self._state = updated_state
        return _result(
            state=updated_state,
            changed=True,
            reason="canonical ledger record status updated",
            record=updated_record,
            snapshot=updated_state.snapshot,
            metadata=_merge_metadata(required_metadata, metadata),
        )

    def mark_verified(
        self,
        *,
        record_id: str,
        metadata: dict[str, object] | None = None,
    ) -> CanonicalLedgerResult:
        """Mark a record verified."""

        return self.update_record_status(
            record_id=record_id,
            status=LedgerRecordStatus.VERIFIED,
            metadata=metadata,
        )

    def mark_superseded(
        self,
        *,
        record_id: str,
        metadata: dict[str, object] | None = None,
    ) -> CanonicalLedgerResult:
        """Mark a record superseded."""

        return self.update_record_status(
            record_id=record_id,
            status=LedgerRecordStatus.SUPERSEDED,
            metadata=metadata,
        )

    def mark_rejected(
        self,
        *,
        record_id: str,
        metadata: dict[str, object] | None = None,
    ) -> CanonicalLedgerResult:
        """Mark a record rejected."""

        return self.update_record_status(
            record_id=record_id,
            status=LedgerRecordStatus.REJECTED,
            metadata=metadata,
        )

    def mark_archived(
        self,
        *,
        record_id: str,
        metadata: dict[str, object] | None = None,
    ) -> CanonicalLedgerResult:
        """Mark a record archived."""

        return self.update_record_status(
            record_id=record_id,
            status=LedgerRecordStatus.ARCHIVED,
            metadata=metadata,
        )

    def create_snapshot(
        self,
        *,
        metadata: dict[str, object] | None = None,
    ) -> CanonicalLedgerResult:
        """Create and attach a ledger snapshot."""

        if not self._state.records and not self._policy.allow_empty_ledger:
            raise ValueError("empty canonical ledger is not allowed")
        required_metadata = {
            "source": "canonical_ledger",
            "operation": "create_snapshot",
            "record_count": len(self._state.records),
            "policy": self._policy.to_dict(),
        }
        snapshot = create_ledger_snapshot(
            records=self._state.records,
            metadata=_merge_metadata(required_metadata, metadata),
        )
        updated_state = CanonicalLedgerState(
            records=self._state.records,
            snapshot=snapshot,
            metadata=self._state_metadata(record_count=len(self._state.records)),
        )
        self._state = updated_state
        return _result(
            state=updated_state,
            changed=True,
            reason="canonical ledger snapshot created",
            snapshot=snapshot,
            metadata=_merge_metadata(required_metadata, metadata),
        )

    def replace_state(
        self,
        *,
        state: CanonicalLedgerState,
        metadata: dict[str, object] | None = None,
    ) -> CanonicalLedgerResult:
        """Replace canonical ledger state."""

        if not isinstance(state, CanonicalLedgerState):
            raise TypeError("state must be CanonicalLedgerState")
        self._validate_records_under_policy(state.records)
        updated_state = self._state_for_policy(
            CanonicalLedgerState(
                records=state.records,
                snapshot=state.snapshot,
                metadata=_merge_runtime_metadata(
                    state.metadata,
                    self._state_metadata(record_count=len(state.records)),
                    None,
                ),
            )
        )
        self._state = updated_state
        required_metadata = {
            "source": "canonical_ledger",
            "operation": "replace_state",
            "record_count": len(updated_state.records),
            "policy": self._policy.to_dict(),
        }
        return _result(
            state=updated_state,
            changed=True,
            reason="canonical ledger state replaced",
            snapshot=updated_state.snapshot,
            metadata=_merge_metadata(required_metadata, metadata),
        )

    def _record_rejection_reason(
        self,
        record: CanonicalLedgerRecord,
    ) -> str | None:
        if self.has_record_id(str(record.record_id)):
            return "ledger record_id already exists"
        if (
            not self._policy.allow_duplicate_record_keys
            and self.has_record_key(record.record_key)
        ):
            return "ledger record_key already exists"
        if self._policy.reject_runtime_semantic_state:
            _reject_runtime_semantic_state(record)
        if len(self._state.records) + 1 > self._policy.max_records:
            return "canonical ledger record limit exceeded"
        return None

    def _validate_records_under_policy(
        self,
        records: tuple[CanonicalLedgerRecord, ...],
    ) -> None:
        if len(records) > self._policy.max_records:
            raise ValueError("canonical ledger record limit exceeded")
        if not records and not self._policy.allow_empty_ledger:
            raise ValueError("empty canonical ledger is not allowed")
        if self._policy.reject_runtime_semantic_state:
            for record in records:
                _reject_runtime_semantic_state(record)
        if not self._policy.allow_duplicate_record_keys:
            _ensure_unique_values(tuple(record.record_key for record in records), "record_key")

    def _state_for_policy(self, state: CanonicalLedgerState) -> CanonicalLedgerState:
        return CanonicalLedgerState(
            records=state.records,
            snapshot=state.snapshot,
            metadata=_merge_runtime_metadata(
                state.metadata,
                self._state_metadata(record_count=len(state.records)),
                None,
            ),
        )

    def _state_metadata(self, *, record_count: int) -> dict[str, object]:
        return {
            "source": "canonical_ledger_state",
            "record_count": record_count,
            "allow_duplicate_record_keys": self._policy.allow_duplicate_record_keys,
            "policy": self._policy.to_dict(),
        }

    def _operation_metadata(
        self,
        *,
        operation: str,
        record: CanonicalLedgerRecord,
    ) -> dict[str, object]:
        return {
            "source": "canonical_ledger",
            "operation": operation,
            "record_id": str(record.record_id),
            "record_key": record.record_key,
            "policy": self._policy.to_dict(),
        }


def build_canonical_ledger(
    *,
    state: CanonicalLedgerState | None = None,
    policy: CanonicalLedgerPolicy | None = None,
) -> CanonicalLedger:
    """Build a canonical ledger manager."""

    return CanonicalLedger(state=state, policy=policy)


def _result(
    *,
    state: CanonicalLedgerState,
    changed: bool,
    reason: str,
    record: CanonicalLedgerRecord | None = None,
    snapshot: LedgerSnapshot | None = None,
    rejected_record_keys: tuple[str, ...] | list[str] = (),
    rejection_reasons: tuple[str, ...] | list[str] = (),
    metadata: dict[str, object] | None = None,
) -> CanonicalLedgerResult:
    return CanonicalLedgerResult(
        state=state,
        changed=changed,
        reason=reason,
        record=record,
        snapshot=snapshot,
        rejected_record_keys=tuple(rejected_record_keys),
        rejection_reasons=tuple(rejection_reasons),
        metadata={} if metadata is None else metadata,
    )


def _ensure_record_tuple(
    records: tuple[CanonicalLedgerRecord, ...] | list[CanonicalLedgerRecord],
) -> tuple[CanonicalLedgerRecord, ...]:
    if not isinstance(records, (tuple, list)):
        raise TypeError("records must be tuple[CanonicalLedgerRecord, ...]")
    records_tuple = tuple(records)
    for record in records_tuple:
        if not isinstance(record, CanonicalLedgerRecord):
            raise TypeError("every record must be CanonicalLedgerRecord")
    return records_tuple


def _replace_record(
    *,
    records: tuple[CanonicalLedgerRecord, ...],
    record: CanonicalLedgerRecord,
) -> tuple[CanonicalLedgerRecord, ...]:
    return tuple(
        record if existing.record_id == record.record_id else existing
        for existing in records
    )


def _ensure_positive_int(value: int, field_name: str) -> int:
    normalized = ensure_non_negative_int(value, field_name)
    if normalized <= 0:
        raise ValueError(f"{field_name} must be positive")
    return normalized


def _ensure_string_tuple(
    values: tuple[str, ...] | list[str],
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple[str, ...]")
    return tuple(ensure_non_empty_string(value, field_name) for value in values)


def _ensure_unique_values(values: tuple[str, ...], field_name: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"{field_name} values must be unique")
        seen.add(value)


def _reject_runtime_semantic_state(record: CanonicalLedgerRecord) -> None:
    if record.record_key in _FORBIDDEN_RUNTIME_STATE_KEYS:
        raise ValueError("record_key must not identify runtime semantic state")
    _reject_forbidden_keys(record.payload.data, "payload data")
    _reject_forbidden_keys(record.payload.metadata, "payload metadata")
    _reject_forbidden_keys(record.metadata, "record metadata")


def _reject_forbidden_keys(value: dict[str, object], field_name: str) -> None:
    for key in value:
        if key in _FORBIDDEN_RUNTIME_STATE_KEYS:
            raise ValueError(f"{field_name} must not persist runtime semantic state")


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


def _merge_runtime_metadata(
    existing_metadata: dict[str, object],
    required_metadata: dict[str, object],
    caller_metadata: dict[str, object] | None,
) -> dict[str, object]:
    merged = _copy_metadata(existing_metadata)
    if caller_metadata is not None:
        merged.update(_copy_metadata(caller_metadata))
    merged.update(_copy_metadata(required_metadata))
    return merged


def _copy_metadata(metadata: dict[str, object]) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(metadata)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
    return copied
