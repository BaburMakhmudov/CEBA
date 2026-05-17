"""Strict schema objects for CEBA Canonical Ledger persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from uuid import UUID, uuid4


__all__ = [
    "LedgerRecordKind",
    "LedgerRecordStatus",
    "LedgerProvenanceKind",
    "LedgerRecordRef",
    "LedgerProvenance",
    "LedgerPayload",
    "CanonicalLedgerRecord",
    "LedgerSnapshot",
    "utc_now",
    "new_ledger_record_id",
    "new_ledger_snapshot_id",
    "ensure_non_empty_string",
    "ensure_optional_non_empty_string",
    "ensure_timezone_aware",
    "ensure_non_negative_int",
    "create_ledger_record_ref",
    "create_ledger_provenance",
    "create_ledger_payload",
    "create_canonical_ledger_record",
    "create_ledger_snapshot",
]


_FORBIDDEN_RUNTIME_STATE_KEYS = {
    "M_t",
    "Mt",
    "semantic_state",
    "runtime_semantic_state",
}


class LedgerRecordKind(Enum):
    """Stable categories for canonical ledger records."""

    OBSERVATION = "OBSERVATION"
    EPISODE = "EPISODE"
    DECISION = "DECISION"
    CONSTRAINT = "CONSTRAINT"
    TELOS = "TELOS"
    DELTA = "DELTA"
    EVENT = "EVENT"
    AUDIT = "AUDIT"
    METRIC_SNAPSHOT = "METRIC_SNAPSHOT"
    WORLD_MODEL = "WORLD_MODEL"
    MEMORY_TRACE = "MEMORY_TRACE"
    CONSOLIDATED_EXPERIENCE = "CONSOLIDATED_EXPERIENCE"
    REPLAY_RECORD = "REPLAY_RECORD"
    UNKNOWN = "UNKNOWN"


class LedgerRecordStatus(Enum):
    """Stable lifecycle statuses for canonical ledger records."""

    CREATED = "CREATED"
    VERIFIED = "VERIFIED"
    SUPERSEDED = "SUPERSEDED"
    REJECTED = "REJECTED"
    ARCHIVED = "ARCHIVED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class LedgerProvenanceKind(Enum):
    """Stable provenance categories for canonical ledger records."""

    INPUT = "INPUT"
    RUNTIME = "RUNTIME"
    WORLD_MODEL = "WORLD_MODEL"
    DELTA = "DELTA"
    MEMORY = "MEMORY"
    CONSTRAINT = "CONSTRAINT"
    TELOS = "TELOS"
    AUDIT = "AUDIT"
    SYSTEM = "SYSTEM"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class LedgerRecordRef:
    """Lightweight reference to a canonical ledger record."""

    record_id: UUID
    kind: LedgerRecordKind
    record_key: str
    semantic_hash: str | None
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, UUID):
            raise TypeError("record_id must be UUID")
        if not isinstance(self.kind, LedgerRecordKind):
            raise TypeError("kind must be LedgerRecordKind")
        record_key = ensure_non_empty_string(self.record_key, "record_key")
        semantic_hash = ensure_optional_non_empty_string(
            self.semantic_hash,
            "semantic_hash",
        )
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        metadata = _copy_string_keyed_dict(self.metadata, "metadata")
        object.__setattr__(self, "record_key", record_key)
        object.__setattr__(self, "semantic_hash", semantic_hash)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "record_id": str(self.record_id),
            "kind": self.kind.value,
            "record_key": self.record_key,
            "semantic_hash": self.semantic_hash,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LedgerProvenance:
    """Immutable provenance descriptor for a canonical ledger record."""

    provenance_id: UUID
    kind: LedgerProvenanceKind
    source_id: str
    source_ref: str
    captured_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.provenance_id, UUID):
            raise TypeError("provenance_id must be UUID")
        if not isinstance(self.kind, LedgerProvenanceKind):
            raise TypeError("kind must be LedgerProvenanceKind")
        source_id = ensure_non_empty_string(self.source_id, "source_id")
        source_ref = ensure_non_empty_string(self.source_ref, "source_ref")
        captured_at = ensure_timezone_aware(self.captured_at, "captured_at")
        metadata = _copy_string_keyed_dict(self.metadata, "metadata")
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "source_ref", source_ref)
        object.__setattr__(self, "captured_at", captured_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "provenance_id": str(self.provenance_id),
            "kind": self.kind.value,
            "source_id": self.source_id,
            "source_ref": self.source_ref,
            "captured_at": self.captured_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LedgerPayload:
    """Typed immutable envelope around ledger payload data."""

    payload_kind: str
    schema_version: int
    data: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        payload_kind = ensure_non_empty_string(self.payload_kind, "payload_kind")
        schema_version = ensure_non_negative_int(
            self.schema_version,
            "schema_version",
        )
        data = _copy_string_keyed_dict(self.data, "data")
        metadata = _copy_string_keyed_dict(self.metadata, "metadata")
        _reject_forbidden_runtime_state_keys(data, "data")
        _reject_forbidden_runtime_state_keys(metadata, "metadata")
        object.__setattr__(self, "payload_kind", payload_kind)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "payload_kind": self.payload_kind,
            "schema_version": self.schema_version,
            "data": dict(self.data),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CanonicalLedgerRecord:
    """Immutable canonical ledger record."""

    record_id: UUID
    kind: LedgerRecordKind
    status: LedgerRecordStatus
    record_key: str
    payload: LedgerPayload
    provenance: tuple[LedgerProvenance, ...]
    semantic_hash: str | None
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, UUID):
            raise TypeError("record_id must be UUID")
        if not isinstance(self.kind, LedgerRecordKind):
            raise TypeError("kind must be LedgerRecordKind")
        if not isinstance(self.status, LedgerRecordStatus):
            raise TypeError("status must be LedgerRecordStatus")
        record_key = ensure_non_empty_string(self.record_key, "record_key")
        if record_key in _FORBIDDEN_RUNTIME_STATE_KEYS:
            raise ValueError("record_key must not identify runtime semantic state")
        if not isinstance(self.payload, LedgerPayload):
            raise TypeError("payload must be LedgerPayload")
        provenance = _ensure_provenance_tuple(self.provenance)
        _ensure_unique_uuid_values(
            tuple(item.provenance_id for item in provenance),
            "provenance_id",
        )
        semantic_hash = ensure_optional_non_empty_string(
            self.semantic_hash,
            "semantic_hash",
        )
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        updated_at = ensure_timezone_aware(self.updated_at, "updated_at")
        if updated_at < created_at:
            raise ValueError("updated_at must be >= created_at")
        metadata = _copy_string_keyed_dict(self.metadata, "metadata")
        _reject_forbidden_runtime_state_keys(metadata, "metadata")
        object.__setattr__(self, "record_key", record_key)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "semantic_hash", semantic_hash)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "record_id": str(self.record_id),
            "kind": self.kind.value,
            "status": self.status.value,
            "record_key": self.record_key,
            "payload": self.payload.to_dict(),
            "provenance": tuple(item.to_dict() for item in self.provenance),
            "semantic_hash": self.semantic_hash,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    def to_ref(self) -> LedgerRecordRef:
        """Return a lightweight ledger record reference."""

        return LedgerRecordRef(
            record_id=self.record_id,
            kind=self.kind,
            record_key=self.record_key,
            semantic_hash=self.semantic_hash,
            created_at=self.created_at,
            metadata={
                "source": "canonical_ledger_record",
                "record_status": self.status.value,
            },
        )

    def provenance_count(self) -> int:
        """Return provenance item count."""

        return len(self.provenance)


@dataclass(frozen=True)
class LedgerSnapshot:
    """Immutable snapshot of canonical ledger records."""

    snapshot_id: UUID
    records: tuple[CanonicalLedgerRecord, ...]
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot_id, UUID):
            raise TypeError("snapshot_id must be UUID")
        records = _ensure_record_tuple(self.records)
        _ensure_unique_uuid_values(
            tuple(record.record_id for record in records),
            "record_id",
        )
        _ensure_unique_string_values(
            tuple(record.record_key for record in records),
            "record_key",
        )
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        metadata = _copy_string_keyed_dict(self.metadata, "metadata")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "snapshot_id": str(self.snapshot_id),
            "records": tuple(record.to_dict() for record in self.records),
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    def record_count(self) -> int:
        """Return record count."""

        return len(self.records)

    def refs(self) -> tuple[LedgerRecordRef, ...]:
        """Return ledger record references."""

        return tuple(record.to_ref() for record in self.records)


def utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""

    return datetime.now(UTC)


def new_ledger_record_id() -> UUID:
    """Return a new ledger record identifier."""

    return uuid4()


def new_ledger_snapshot_id() -> UUID:
    """Return a new ledger snapshot identifier."""

    return uuid4()


def ensure_non_empty_string(value: str, field_name: str) -> str:
    """Validate and normalize a non-empty string."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must be non-empty")
    return stripped


def ensure_optional_non_empty_string(
    value: str | None,
    field_name: str,
) -> str | None:
    """Validate and normalize an optional non-empty string."""

    if value is None:
        return None
    return ensure_non_empty_string(value, field_name)


def ensure_timezone_aware(value: datetime, field_name: str) -> datetime:
    """Validate a timezone-aware datetime."""

    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def ensure_non_negative_int(value: int, field_name: str) -> int:
    """Validate a non-negative integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return value


def create_ledger_record_ref(
    *,
    record_id: UUID,
    kind: LedgerRecordKind,
    record_key: str,
    semantic_hash: str | None = None,
    metadata: dict[str, object] | None = None,
) -> LedgerRecordRef:
    """Create a ledger record reference."""

    required_metadata: dict[str, object] = {"source": "ledger_record_ref"}
    return LedgerRecordRef(
        record_id=record_id,
        kind=kind,
        record_key=record_key,
        semantic_hash=semantic_hash,
        created_at=utc_now(),
        metadata=_merge_metadata(required_metadata, metadata),
    )


def create_ledger_provenance(
    *,
    source_id: str,
    source_ref: str,
    kind: LedgerProvenanceKind = LedgerProvenanceKind.UNKNOWN,
    metadata: dict[str, object] | None = None,
) -> LedgerProvenance:
    """Create a ledger provenance descriptor."""

    required_metadata: dict[str, object] = {
        "source": "ledger_provenance",
        "source_id": source_id,
    }
    return LedgerProvenance(
        provenance_id=uuid4(),
        kind=kind,
        source_id=source_id,
        source_ref=source_ref,
        captured_at=utc_now(),
        metadata=_merge_metadata(required_metadata, metadata),
    )


def create_ledger_payload(
    *,
    payload_kind: str,
    schema_version: int = 1,
    data: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
) -> LedgerPayload:
    """Create a ledger payload envelope."""

    required_metadata: dict[str, object] = {
        "source": "ledger_payload",
        "payload_kind": payload_kind,
        "schema_version": schema_version,
    }
    return LedgerPayload(
        payload_kind=payload_kind,
        schema_version=schema_version,
        data={} if data is None else data,
        metadata=_merge_metadata(required_metadata, metadata),
    )


def create_canonical_ledger_record(
    *,
    record_key: str,
    payload: LedgerPayload,
    kind: LedgerRecordKind = LedgerRecordKind.UNKNOWN,
    status: LedgerRecordStatus = LedgerRecordStatus.CREATED,
    provenance: tuple[LedgerProvenance, ...] | list[LedgerProvenance] = (),
    semantic_hash: str | None = None,
    metadata: dict[str, object] | None = None,
) -> CanonicalLedgerRecord:
    """Create a canonical ledger record."""

    now = utc_now()
    provenance_tuple = tuple(provenance)
    required_metadata: dict[str, object] = {
        "source": "canonical_ledger_record",
        "record_kind": kind.value if isinstance(kind, LedgerRecordKind) else kind,
        "record_status": (
            status.value if isinstance(status, LedgerRecordStatus) else status
        ),
        "record_key": record_key,
        "provenance_count": len(provenance_tuple),
    }
    return CanonicalLedgerRecord(
        record_id=new_ledger_record_id(),
        kind=kind,
        status=status,
        record_key=record_key,
        payload=payload,
        provenance=provenance_tuple,
        semantic_hash=semantic_hash,
        created_at=now,
        updated_at=now,
        metadata=_merge_metadata(required_metadata, metadata),
    )


def create_ledger_snapshot(
    *,
    records: tuple[CanonicalLedgerRecord, ...] | list[CanonicalLedgerRecord] = (),
    metadata: dict[str, object] | None = None,
) -> LedgerSnapshot:
    """Create a ledger snapshot."""

    records_tuple = tuple(records)
    required_metadata: dict[str, object] = {
        "source": "ledger_snapshot",
        "record_count": len(records_tuple),
    }
    return LedgerSnapshot(
        snapshot_id=new_ledger_snapshot_id(),
        records=records_tuple,
        created_at=utc_now(),
        metadata=_merge_metadata(required_metadata, metadata),
    )


def _ensure_provenance_tuple(
    provenance: tuple[LedgerProvenance, ...] | list[LedgerProvenance],
) -> tuple[LedgerProvenance, ...]:
    if not isinstance(provenance, (tuple, list)):
        raise TypeError("provenance must be tuple[LedgerProvenance, ...]")
    provenance_tuple = tuple(provenance)
    for item in provenance_tuple:
        if not isinstance(item, LedgerProvenance):
            raise TypeError("every provenance item must be LedgerProvenance")
    return provenance_tuple


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


def _ensure_unique_uuid_values(values: tuple[UUID, ...], field_name: str) -> None:
    seen: set[UUID] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"{field_name} values must be unique")
        seen.add(value)


def _ensure_unique_string_values(values: tuple[str, ...], field_name: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"{field_name} values must be unique")
        seen.add(value)


def _copy_string_keyed_dict(
    value: dict[str, object],
    field_name: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be dict[str, object]")
    copied = dict(value)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError(f"{field_name} keys must be strings")
    return copied


def _merge_metadata(
    default_metadata: dict[str, object],
    metadata: dict[str, object] | None,
) -> dict[str, object]:
    merged = _copy_string_keyed_dict(default_metadata, "metadata")
    if metadata is not None:
        merged.update(_copy_string_keyed_dict(metadata, "metadata"))
    if "source" in default_metadata:
        merged["source"] = default_metadata["source"]
    return merged


def _reject_forbidden_runtime_state_keys(
    value: dict[str, object],
    field_name: str,
) -> None:
    for key in value:
        if key in _FORBIDDEN_RUNTIME_STATE_KEYS:
            raise ValueError(f"{field_name} must not persist runtime semantic state")
