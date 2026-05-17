"""Deterministic local audit repository utilities for CEBA ledger persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ceba.persistence.ledger.canonical_ledger import (
    CanonicalLedger,
    CanonicalLedgerPolicy,
    CanonicalLedgerResult,
    CanonicalLedgerState,
    build_canonical_ledger,
)
from ceba.persistence.ledger.schema import (
    CanonicalLedgerRecord,
    LedgerProvenanceKind,
    LedgerRecordKind,
    LedgerRecordRef,
    LedgerRecordStatus,
    create_ledger_payload,
    create_ledger_provenance,
    ensure_non_empty_string,
    ensure_non_negative_int,
    ensure_optional_non_empty_string,
)


__all__ = [
    "AuditRecordStatusFilter",
    "AuditRepositoryPolicy",
    "AuditPayloadInput",
    "AuditRepositoryQuery",
    "AuditRepositoryResult",
    "AuditRepository",
    "build_audit_repository",
]


_FORBIDDEN_RUNTIME_STATE_KEYS = {
    "M_t",
    "Mt",
    "semantic_state",
    "runtime_semantic_state",
}


class AuditRecordStatusFilter(Enum):
    """Stable status filters for audit repository queries."""

    ANY = "ANY"
    CREATED = "CREATED"
    VERIFIED = "VERIFIED"
    SUPERSEDED = "SUPERSEDED"
    REJECTED = "REJECTED"
    ARCHIVED = "ARCHIVED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class AuditRepositoryPolicy:
    """Configuration object controlling audit repository behavior."""

    ledger_policy: CanonicalLedgerPolicy = field(default_factory=CanonicalLedgerPolicy)
    default_schema_version: int = 1
    auto_verify_on_create: bool = False
    reject_empty_payload: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.ledger_policy, CanonicalLedgerPolicy):
            raise TypeError("ledger_policy must be CanonicalLedgerPolicy")
        object.__setattr__(
            self,
            "default_schema_version",
            ensure_non_negative_int(
                self.default_schema_version,
                "default_schema_version",
            ),
        )
        if not isinstance(self.auto_verify_on_create, bool):
            raise TypeError("auto_verify_on_create must be bool")
        if not isinstance(self.reject_empty_payload, bool):
            raise TypeError("reject_empty_payload must be bool")

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "ledger_policy": self.ledger_policy.to_dict(),
            "default_schema_version": self.default_schema_version,
            "auto_verify_on_create": self.auto_verify_on_create,
            "reject_empty_payload": self.reject_empty_payload,
        }


@dataclass(frozen=True)
class AuditPayloadInput:
    """Explicit input for creating one audit ledger record."""

    audit_key: str
    audit_type: str
    payload_data: dict[str, object]
    provenance_source_id: str
    provenance_source_ref: str
    provenance_kind: LedgerProvenanceKind = LedgerProvenanceKind.AUDIT
    semantic_hash: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        audit_key = ensure_non_empty_string(self.audit_key, "audit_key")
        audit_type = ensure_non_empty_string(self.audit_type, "audit_type")
        payload_data = _copy_string_keyed_dict(self.payload_data, "payload_data")
        _reject_forbidden_keys(payload_data, "payload_data")
        provenance_source_id = ensure_non_empty_string(
            self.provenance_source_id,
            "provenance_source_id",
        )
        provenance_source_ref = ensure_non_empty_string(
            self.provenance_source_ref,
            "provenance_source_ref",
        )
        if not isinstance(self.provenance_kind, LedgerProvenanceKind):
            raise TypeError("provenance_kind must be LedgerProvenanceKind")
        semantic_hash = ensure_optional_non_empty_string(
            self.semantic_hash,
            "semantic_hash",
        )
        metadata = _copy_string_keyed_dict(self.metadata, "metadata")
        _reject_forbidden_keys(metadata, "metadata")
        object.__setattr__(self, "audit_key", audit_key)
        object.__setattr__(self, "audit_type", audit_type)
        object.__setattr__(self, "payload_data", payload_data)
        object.__setattr__(self, "provenance_source_id", provenance_source_id)
        object.__setattr__(self, "provenance_source_ref", provenance_source_ref)
        object.__setattr__(self, "semantic_hash", semantic_hash)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "audit_key": self.audit_key,
            "audit_type": self.audit_type,
            "payload_data": dict(self.payload_data),
            "provenance_source_id": self.provenance_source_id,
            "provenance_source_ref": self.provenance_source_ref,
            "provenance_kind": self.provenance_kind.value,
            "semantic_hash": self.semantic_hash,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AuditRepositoryQuery:
    """Explicit local query descriptor for audit records."""

    audit_key: str | None = None
    audit_type: str | None = None
    status_filter: AuditRecordStatusFilter = AuditRecordStatusFilter.ANY
    limit: int = 100

    def __post_init__(self) -> None:
        audit_key = ensure_optional_non_empty_string(
            self.audit_key,
            "audit_key",
        )
        audit_type = ensure_optional_non_empty_string(
            self.audit_type,
            "audit_type",
        )
        if not isinstance(self.status_filter, AuditRecordStatusFilter):
            raise TypeError("status_filter must be AuditRecordStatusFilter")
        limit = _ensure_positive_int(self.limit, "limit")
        object.__setattr__(self, "audit_key", audit_key)
        object.__setattr__(self, "audit_type", audit_type)
        object.__setattr__(self, "limit", limit)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "audit_key": self.audit_key,
            "audit_type": self.audit_type,
            "status_filter": self.status_filter.value,
            "limit": self.limit,
        }


@dataclass(frozen=True)
class AuditRepositoryResult:
    """Immutable result returned by audit repository operations."""

    ledger_result: CanonicalLedgerResult | None
    records: tuple[CanonicalLedgerRecord, ...]
    refs: tuple[LedgerRecordRef, ...]
    changed: bool
    reason: str
    rejected_audit_keys: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ledger_result is not None and not isinstance(
            self.ledger_result,
            CanonicalLedgerResult,
        ):
            raise TypeError("ledger_result must be CanonicalLedgerResult or None")
        records = _ensure_audit_record_tuple(self.records)
        refs = _ensure_ref_tuple(self.refs)
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        reason = ensure_non_empty_string(self.reason, "reason")
        rejected_audit_keys = _ensure_string_tuple(
            self.rejected_audit_keys,
            "rejected_audit_keys",
        )
        rejection_reasons = _ensure_string_tuple(
            self.rejection_reasons,
            "rejection_reasons",
        )
        if len(rejected_audit_keys) != len(rejection_reasons):
            raise ValueError("rejected_audit_keys and rejection_reasons must match")
        metadata = _copy_string_keyed_dict(self.metadata, "metadata")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "refs", refs)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "rejected_audit_keys", rejected_audit_keys)
        object.__setattr__(self, "rejection_reasons", rejection_reasons)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "ledger_result": (
                None if self.ledger_result is None else self.ledger_result.to_dict()
            ),
            "records": tuple(record.to_dict() for record in self.records),
            "refs": tuple(ref.to_dict() for ref in self.refs),
            "changed": self.changed,
            "reason": self.reason,
            "rejected_audit_keys": self.rejected_audit_keys,
            "rejection_reasons": self.rejection_reasons,
            "metadata": dict(self.metadata),
        }

    def record_count(self) -> int:
        """Return record count."""

        return len(self.records)

    def ref_count(self) -> int:
        """Return reference count."""

        return len(self.refs)

    def rejected_count(self) -> int:
        """Return rejected audit count."""

        return len(self.rejected_audit_keys)

    def has_records(self) -> bool:
        """Return whether result contains records."""

        return bool(self.records)


class AuditRepository:
    """Deterministic local repository boundary for audit ledger records."""

    def __init__(
        self,
        *,
        ledger: CanonicalLedger | None = None,
        policy: AuditRepositoryPolicy | None = None,
    ) -> None:
        self._policy = AuditRepositoryPolicy() if policy is None else policy
        if not isinstance(self._policy, AuditRepositoryPolicy):
            raise TypeError("policy must be AuditRepositoryPolicy")
        self._ledger = (
            build_canonical_ledger(policy=self._policy.ledger_policy)
            if ledger is None
            else ledger
        )
        if not isinstance(self._ledger, CanonicalLedger):
            raise TypeError("ledger must be CanonicalLedger")

    def ledger(self) -> CanonicalLedger:
        """Return internal canonical ledger."""

        return self._ledger

    def state(self) -> CanonicalLedgerState:
        """Return current canonical ledger state."""

        return self._ledger.state()

    def create_audit(
        self,
        *,
        audit_input: AuditPayloadInput,
    ) -> AuditRepositoryResult:
        """Create an audit ledger record."""

        if not isinstance(audit_input, AuditPayloadInput):
            raise TypeError("audit_input must be AuditPayloadInput")
        if self._policy.reject_empty_payload and not audit_input.payload_data:
            reason = "audit payload is empty"
            return _result(
                ledger_result=None,
                records=(),
                refs=(),
                changed=False,
                reason=reason,
                rejected_audit_keys=(audit_input.audit_key,),
                rejection_reasons=(reason,),
                metadata=self._operation_metadata(
                    operation="create_audit",
                    audit_key=audit_input.audit_key,
                    audit_type=audit_input.audit_type,
                ),
            )
        payload = create_ledger_payload(
            payload_kind="audit",
            schema_version=self._policy.default_schema_version,
            data=audit_input.payload_data,
            metadata=_merge_metadata(
                {
                    "source": "audit_repository",
                    "operation": "create_audit_payload",
                    "audit_key": audit_input.audit_key,
                    "audit_type": audit_input.audit_type,
                },
                audit_input.metadata,
            ),
        )
        provenance = create_ledger_provenance(
            source_id=audit_input.provenance_source_id,
            source_ref=audit_input.provenance_source_ref,
            kind=audit_input.provenance_kind,
            metadata={
                "source": "audit_repository",
                "operation": "create_audit_provenance",
                "audit_key": audit_input.audit_key,
                "audit_type": audit_input.audit_type,
            },
        )
        ledger_result = self._ledger.create_record(
            kind=LedgerRecordKind.AUDIT,
            status=LedgerRecordStatus.CREATED,
            record_key=audit_input.audit_key,
            payload=payload,
            provenance=(provenance,),
            semantic_hash=audit_input.semantic_hash,
            metadata=_merge_metadata(
                {
                    "source": "audit_repository",
                    "operation": "create_audit",
                    "audit_key": audit_input.audit_key,
                    "audit_type": audit_input.audit_type,
                },
                audit_input.metadata,
            ),
        )
        reason = "audit record created"
        if self._policy.auto_verify_on_create and ledger_result.record is not None:
            ledger_result = self._ledger.mark_verified(
                record_id=str(ledger_result.record.record_id)
            )
            reason = "audit record created and verified"
        if ledger_result.record is None:
            reason = ledger_result.reason
        return _from_ledger_result(
            ledger_result=ledger_result,
            reason=reason,
            metadata=self._operation_metadata(
                operation="create_audit",
                audit_key=audit_input.audit_key,
                audit_type=audit_input.audit_type,
            ),
        )

    def get_audit_by_key(
        self,
        *,
        audit_key: str,
    ) -> AuditRepositoryResult:
        """Return the first audit record by audit key."""

        normalized = ensure_non_empty_string(audit_key, "audit_key")
        for record in self._ledger.state().records:
            if (
                record.kind == LedgerRecordKind.AUDIT
                and record.record_key == normalized
            ):
                return _result(
                    ledger_result=None,
                    records=(record,),
                    refs=(record.to_ref(),),
                    changed=False,
                    reason="audit record found",
                    metadata=self._operation_metadata(
                        operation="get_audit_by_key",
                        audit_key=normalized,
                        audit_type=str(record.metadata.get("audit_type", "")),
                    ),
                )
        return _result(
            ledger_result=None,
            records=(),
            refs=(),
            changed=False,
            reason="audit record missing",
            metadata={
                "source": "audit_repository",
                "operation": "get_audit_by_key",
                "audit_key": normalized,
            },
        )

    def query_audits(
        self,
        *,
        query: AuditRepositoryQuery,
    ) -> AuditRepositoryResult:
        """Query audit records from local ledger state."""

        if not isinstance(query, AuditRepositoryQuery):
            raise TypeError("query must be AuditRepositoryQuery")
        records = _sort_audit_records(
            tuple(
                record
                for record in self._ledger.state().records
                if _matches_query(record=record, query=query)
            )
        )[: query.limit]
        return _result(
            ledger_result=None,
            records=records,
            refs=tuple(record.to_ref() for record in records),
            changed=False,
            reason="audit records queried",
            metadata={
                "source": "audit_repository",
                "operation": "query_audits",
                "query": query.to_dict(),
            },
        )

    def mark_verified(
        self,
        *,
        record_id: str,
    ) -> AuditRepositoryResult:
        """Mark an audit record verified."""

        normalized = ensure_non_empty_string(record_id, "record_id")
        record = self._ledger.get_record_by_id(normalized)
        if record is None:
            return _result(
                ledger_result=None,
                records=(),
                refs=(),
                changed=False,
                reason="audit record missing",
                metadata={
                    "source": "audit_repository",
                    "operation": "mark_verified",
                    "record_id": normalized,
                },
            )
        if record.kind != LedgerRecordKind.AUDIT:
            reason = "audit record kind mismatch"
            return _result(
                ledger_result=None,
                records=(),
                refs=(),
                changed=False,
                reason=reason,
                rejected_audit_keys=(record.record_key,),
                rejection_reasons=(reason,),
                metadata={
                    "source": "audit_repository",
                    "operation": "mark_verified",
                    "record_id": normalized,
                },
            )
        ledger_result = self._ledger.mark_verified(record_id=normalized)
        return _from_ledger_result(
            ledger_result=ledger_result,
            reason="audit record verified",
            metadata={
                "source": "audit_repository",
                "operation": "mark_verified",
                "record_id": normalized,
            },
        )

    def mark_archived(
        self,
        *,
        record_id: str,
    ) -> AuditRepositoryResult:
        """Mark an audit record archived."""

        normalized = ensure_non_empty_string(record_id, "record_id")
        record = self._ledger.get_record_by_id(normalized)
        if record is None:
            return _result(
                ledger_result=None,
                records=(),
                refs=(),
                changed=False,
                reason="audit record missing",
                metadata={
                    "source": "audit_repository",
                    "operation": "mark_archived",
                    "record_id": normalized,
                },
            )
        if record.kind != LedgerRecordKind.AUDIT:
            reason = "audit record kind mismatch"
            return _result(
                ledger_result=None,
                records=(),
                refs=(),
                changed=False,
                reason=reason,
                rejected_audit_keys=(record.record_key,),
                rejection_reasons=(reason,),
                metadata={
                    "source": "audit_repository",
                    "operation": "mark_archived",
                    "record_id": normalized,
                },
            )
        ledger_result = self._ledger.mark_archived(record_id=normalized)
        return _from_ledger_result(
            ledger_result=ledger_result,
            reason="audit record archived",
            metadata={
                "source": "audit_repository",
                "operation": "mark_archived",
                "record_id": normalized,
            },
        )

    def create_snapshot(self) -> AuditRepositoryResult:
        """Create an audit repository snapshot."""

        ledger_result = self._ledger.create_snapshot(
            metadata={
                "source": "audit_repository",
                "operation": "create_snapshot",
            }
        )
        refs: tuple[LedgerRecordRef, ...] = ()
        if ledger_result.snapshot is not None:
            refs = tuple(
                ref
                for ref in ledger_result.snapshot.refs()
                if _record_by_ref_kind(
                    records=ledger_result.snapshot.records,
                    ref=ref,
                    kind=LedgerRecordKind.AUDIT,
                )
            )
        return _result(
            ledger_result=ledger_result,
            records=(),
            refs=refs,
            changed=ledger_result.changed,
            reason="audit repository snapshot created",
            metadata={
                "source": "audit_repository",
                "operation": "create_snapshot",
            },
        )

    def _operation_metadata(
        self,
        *,
        operation: str,
        audit_key: str,
        audit_type: str,
    ) -> dict[str, object]:
        return {
            "source": "audit_repository",
            "operation": operation,
            "audit_key": audit_key,
            "audit_type": audit_type,
            "policy": self._policy.to_dict(),
        }


def build_audit_repository(
    *,
    ledger: CanonicalLedger | None = None,
    policy: AuditRepositoryPolicy | None = None,
) -> AuditRepository:
    """Build an audit repository."""

    return AuditRepository(ledger=ledger, policy=policy)


def _from_ledger_result(
    *,
    ledger_result: CanonicalLedgerResult,
    reason: str,
    metadata: dict[str, object],
) -> AuditRepositoryResult:
    records: tuple[CanonicalLedgerRecord, ...] = ()
    refs: tuple[LedgerRecordRef, ...] = ()
    if (
        ledger_result.record is not None
        and ledger_result.record.kind == LedgerRecordKind.AUDIT
    ):
        records = (ledger_result.record,)
        refs = (ledger_result.record.to_ref(),)
    return _result(
        ledger_result=ledger_result,
        records=records,
        refs=refs,
        changed=ledger_result.changed,
        reason=reason,
        rejected_audit_keys=ledger_result.rejected_record_keys,
        rejection_reasons=ledger_result.rejection_reasons,
        metadata=metadata,
    )


def _result(
    *,
    ledger_result: CanonicalLedgerResult | None,
    records: tuple[CanonicalLedgerRecord, ...] | list[CanonicalLedgerRecord],
    refs: tuple[LedgerRecordRef, ...] | list[LedgerRecordRef],
    changed: bool,
    reason: str,
    rejected_audit_keys: tuple[str, ...] | list[str] = (),
    rejection_reasons: tuple[str, ...] | list[str] = (),
    metadata: dict[str, object] | None = None,
) -> AuditRepositoryResult:
    return AuditRepositoryResult(
        ledger_result=ledger_result,
        records=tuple(records),
        refs=tuple(refs),
        changed=changed,
        reason=reason,
        rejected_audit_keys=tuple(rejected_audit_keys),
        rejection_reasons=tuple(rejection_reasons),
        metadata={} if metadata is None else metadata,
    )


def _matches_query(
    *,
    record: CanonicalLedgerRecord,
    query: AuditRepositoryQuery,
) -> bool:
    if record.kind != LedgerRecordKind.AUDIT:
        return False
    if query.audit_key is not None and record.record_key != query.audit_key:
        return False
    if (
        query.audit_type is not None
        and record.metadata.get("audit_type") != query.audit_type
    ):
        return False
    if query.status_filter == AuditRecordStatusFilter.ANY:
        return True
    return record.status.value == query.status_filter.value


def _sort_audit_records(
    records: tuple[CanonicalLedgerRecord, ...],
) -> tuple[CanonicalLedgerRecord, ...]:
    return tuple(
        sorted(
            records,
            key=lambda record: (
                -record.updated_at.timestamp(),
                -record.created_at.timestamp(),
                str(record.record_id),
            ),
        )
    )


def _record_by_ref_kind(
    *,
    records: tuple[CanonicalLedgerRecord, ...],
    ref: LedgerRecordRef,
    kind: LedgerRecordKind,
) -> bool:
    for record in records:
        if record.record_id == ref.record_id and record.kind == kind:
            return True
    return False


def _ensure_audit_record_tuple(
    records: tuple[CanonicalLedgerRecord, ...] | list[CanonicalLedgerRecord],
) -> tuple[CanonicalLedgerRecord, ...]:
    if not isinstance(records, (tuple, list)):
        raise TypeError("records must be tuple[CanonicalLedgerRecord, ...]")
    records_tuple = tuple(records)
    seen: set[str] = set()
    for record in records_tuple:
        if not isinstance(record, CanonicalLedgerRecord):
            raise TypeError("every record must be CanonicalLedgerRecord")
        if record.kind != LedgerRecordKind.AUDIT:
            raise ValueError("every record.kind must be LedgerRecordKind.AUDIT")
        record_id = str(record.record_id)
        if record_id in seen:
            raise ValueError("record_id values must be unique")
        seen.add(record_id)
    return records_tuple


def _ensure_ref_tuple(
    refs: tuple[LedgerRecordRef, ...] | list[LedgerRecordRef],
) -> tuple[LedgerRecordRef, ...]:
    if not isinstance(refs, (tuple, list)):
        raise TypeError("refs must be tuple[LedgerRecordRef, ...]")
    refs_tuple = tuple(refs)
    for ref in refs_tuple:
        if not isinstance(ref, LedgerRecordRef):
            raise TypeError("every ref must be LedgerRecordRef")
    return refs_tuple


def _ensure_string_tuple(
    values: tuple[str, ...] | list[str],
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple[str, ...]")
    return tuple(ensure_non_empty_string(value, field_name) for value in values)


def _ensure_positive_int(value: int, field_name: str) -> int:
    normalized = ensure_non_negative_int(value, field_name)
    if normalized <= 0:
        raise ValueError(f"{field_name} must be positive")
    return normalized


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


def _reject_forbidden_keys(value: dict[str, object], field_name: str) -> None:
    for key in value:
        if key in _FORBIDDEN_RUNTIME_STATE_KEYS:
            raise ValueError(f"{field_name} must not persist runtime semantic state")
