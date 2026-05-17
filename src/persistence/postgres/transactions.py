"""Deterministic PostgreSQL transaction descriptors for CEBA persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from ceba.persistence.postgres.connection import (
    ensure_non_empty_string,
    ensure_optional_non_empty_string,
    ensure_positive_int,
    ensure_timezone_aware,
    utc_now,
)
from ceba.persistence.postgres.models import PostgresTableKind


__all__ = [
    "PostgresTransactionStatus",
    "PostgresTransactionOperationKind",
    "PostgresTransactionIsolationLevel",
    "PostgresTransactionPolicy",
    "PostgresTransactionOperation",
    "PostgresTransactionDescriptor",
    "PostgresTransactionReport",
    "PostgresTransactionManager",
    "new_postgres_transaction_id",
    "new_postgres_transaction_operation_id",
    "create_postgres_transaction_operation",
    "create_postgres_transaction_descriptor",
    "build_postgres_transaction_manager",
]


_FORBIDDEN_RUNTIME_STATE_NAMES = {
    "M_t",
    "Mt",
    "semantic_state",
    "runtime_semantic_state",
}


class PostgresTransactionStatus(Enum):
    """Stable PostgreSQL transaction descriptor statuses."""

    CREATED = "CREATED"
    OPEN = "OPEN"
    COMMITTED = "COMMITTED"
    ROLLED_BACK = "ROLLED_BACK"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class PostgresTransactionOperationKind(Enum):
    """Stable PostgreSQL transaction operation kinds."""

    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    UPSERT = "UPSERT"
    SELECT = "SELECT"
    LOCK = "LOCK"
    CHECK = "CHECK"
    UNKNOWN = "UNKNOWN"


class PostgresTransactionIsolationLevel(Enum):
    """Stable PostgreSQL transaction isolation levels."""

    READ_COMMITTED = "READ_COMMITTED"
    REPEATABLE_READ = "REPEATABLE_READ"
    SERIALIZABLE = "SERIALIZABLE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class PostgresTransactionPolicy:
    """Configuration object controlling local transaction descriptors."""

    isolation_level: PostgresTransactionIsolationLevel = (
        PostgresTransactionIsolationLevel.READ_COMMITTED
    )
    max_operations: int = 1024
    allow_empty_transaction: bool = False
    allow_select_operations: bool = True
    block_forbidden_runtime_state: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.isolation_level, PostgresTransactionIsolationLevel):
            raise TypeError("isolation_level must be PostgresTransactionIsolationLevel")
        object.__setattr__(
            self,
            "max_operations",
            ensure_positive_int(self.max_operations, "max_operations"),
        )
        object.__setattr__(
            self,
            "allow_empty_transaction",
            _ensure_bool(self.allow_empty_transaction, "allow_empty_transaction"),
        )
        object.__setattr__(
            self,
            "allow_select_operations",
            _ensure_bool(self.allow_select_operations, "allow_select_operations"),
        )
        object.__setattr__(
            self,
            "block_forbidden_runtime_state",
            _ensure_bool(
                self.block_forbidden_runtime_state,
                "block_forbidden_runtime_state",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "isolation_level": self.isolation_level.value,
            "max_operations": self.max_operations,
            "allow_empty_transaction": self.allow_empty_transaction,
            "allow_select_operations": self.allow_select_operations,
            "block_forbidden_runtime_state": self.block_forbidden_runtime_state,
        }


@dataclass(frozen=True)
class PostgresTransactionOperation:
    """Immutable descriptor for one local PostgreSQL transaction operation."""

    operation_id: UUID
    kind: PostgresTransactionOperationKind
    table_kind: PostgresTableKind
    table_name: str
    record_key: str | None
    payload_ref: str | None
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.operation_id, UUID):
            raise TypeError("operation_id must be UUID")
        if not isinstance(self.kind, PostgresTransactionOperationKind):
            raise TypeError("kind must be PostgresTransactionOperationKind")
        if not isinstance(self.table_kind, PostgresTableKind):
            raise TypeError("table_kind must be PostgresTableKind")
        table_name = _ensure_safe_name(self.table_name, "table_name")
        record_key = ensure_optional_non_empty_string(self.record_key, "record_key")
        if record_key is not None:
            _reject_forbidden_name(record_key, "record_key")
        payload_ref = ensure_optional_non_empty_string(self.payload_ref, "payload_ref")
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        metadata = _copy_metadata(self.metadata)
        _reject_forbidden_metadata_keys(metadata)
        object.__setattr__(self, "table_name", table_name)
        object.__setattr__(self, "record_key", record_key)
        object.__setattr__(self, "payload_ref", payload_ref)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "operation_id": str(self.operation_id),
            "kind": self.kind.value,
            "table_kind": self.table_kind.value,
            "table_name": self.table_name,
            "record_key": self.record_key,
            "payload_ref": self.payload_ref,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PostgresTransactionDescriptor:
    """Immutable local PostgreSQL transaction boundary descriptor."""

    transaction_id: UUID
    status: PostgresTransactionStatus
    isolation_level: PostgresTransactionIsolationLevel
    operations: tuple[PostgresTransactionOperation, ...]
    created_at: datetime
    updated_at: datetime
    failure_reason: str | None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.transaction_id, UUID):
            raise TypeError("transaction_id must be UUID")
        if not isinstance(self.status, PostgresTransactionStatus):
            raise TypeError("status must be PostgresTransactionStatus")
        if not isinstance(self.isolation_level, PostgresTransactionIsolationLevel):
            raise TypeError("isolation_level must be PostgresTransactionIsolationLevel")
        operations = _ensure_operation_tuple(self.operations)
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        updated_at = ensure_timezone_aware(self.updated_at, "updated_at")
        if updated_at < created_at:
            raise ValueError("updated_at must be >= created_at")
        failure_reason = ensure_optional_non_empty_string(
            self.failure_reason,
            "failure_reason",
        )
        metadata = _copy_metadata(self.metadata)
        _reject_forbidden_metadata_keys(metadata)
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "failure_reason", failure_reason)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "transaction_id": str(self.transaction_id),
            "status": self.status.value,
            "isolation_level": self.isolation_level.value,
            "operations": tuple(operation.to_dict() for operation in self.operations),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "failure_reason": self.failure_reason,
            "metadata": dict(self.metadata),
        }

    def operation_count(self) -> int:
        """Return operation count."""

        return len(self.operations)

    def is_terminal(self) -> bool:
        """Return whether transaction descriptor is terminal."""

        return self.status in {
            PostgresTransactionStatus.COMMITTED,
            PostgresTransactionStatus.ROLLED_BACK,
            PostgresTransactionStatus.FAILED,
            PostgresTransactionStatus.BLOCKED,
            PostgresTransactionStatus.INVALID,
        }


@dataclass(frozen=True)
class PostgresTransactionReport:
    """Immutable result returned by local transaction descriptor operations."""

    transaction: PostgresTransactionDescriptor
    changed: bool
    reason: str
    blocked: bool
    rejected_operation_ids: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.transaction, PostgresTransactionDescriptor):
            raise TypeError("transaction must be PostgresTransactionDescriptor")
        changed = _ensure_bool(self.changed, "changed")
        reason = ensure_non_empty_string(self.reason, "reason")
        blocked = _ensure_bool(self.blocked, "blocked")
        rejected_operation_ids = _ensure_string_tuple(
            self.rejected_operation_ids,
            "rejected_operation_ids",
        )
        rejection_reasons = _ensure_string_tuple(
            self.rejection_reasons,
            "rejection_reasons",
        )
        if len(rejected_operation_ids) != len(rejection_reasons):
            raise ValueError("rejected_operation_ids and rejection_reasons must match")
        metadata = _copy_metadata(self.metadata)
        _reject_forbidden_metadata_keys(metadata)
        object.__setattr__(self, "changed", changed)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "blocked", blocked)
        object.__setattr__(self, "rejected_operation_ids", rejected_operation_ids)
        object.__setattr__(self, "rejection_reasons", rejection_reasons)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "transaction": self.transaction.to_dict(),
            "changed": self.changed,
            "reason": self.reason,
            "blocked": self.blocked,
            "rejected_operation_ids": self.rejected_operation_ids,
            "rejection_reasons": self.rejection_reasons,
            "metadata": dict(self.metadata),
        }

    def rejected_count(self) -> int:
        """Return rejected operation count."""

        return len(self.rejected_operation_ids)

    def is_success(self) -> bool:
        """Return whether the report represents a successful commit."""

        return (
            not self.blocked
            and self.transaction.status == PostgresTransactionStatus.COMMITTED
        )


class PostgresTransactionManager:
    """Deterministic local manager for PostgreSQL transaction descriptors."""

    def __init__(self, policy: PostgresTransactionPolicy | None = None) -> None:
        self._policy = PostgresTransactionPolicy() if policy is None else policy
        if not isinstance(self._policy, PostgresTransactionPolicy):
            raise TypeError("policy must be PostgresTransactionPolicy")

    def begin(
        self,
        *,
        metadata: dict[str, object] | None = None,
    ) -> PostgresTransactionReport:
        """Open a local PostgreSQL transaction descriptor."""

        transaction = create_postgres_transaction_descriptor(
            status=PostgresTransactionStatus.OPEN,
            isolation_level=self._policy.isolation_level,
            operations=(),
            failure_reason=None,
            metadata=_merge_metadata(
                {
                    "source": "postgres_transaction_manager",
                    "operation": "begin",
                    "policy": self._policy.to_dict(),
                },
                metadata,
            ),
        )
        return _report(
            transaction=transaction,
            changed=True,
            reason="postgres transaction opened",
            blocked=False,
        )

    def add_operation(
        self,
        *,
        transaction: PostgresTransactionDescriptor,
        operation: PostgresTransactionOperation,
        metadata: dict[str, object] | None = None,
    ) -> PostgresTransactionReport:
        """Append an operation to an open local transaction descriptor."""

        self._require_transaction(transaction)
        if not isinstance(operation, PostgresTransactionOperation):
            raise TypeError("operation must be PostgresTransactionOperation")
        if operation.kind == PostgresTransactionOperationKind.SELECT:
            if not self._policy.allow_select_operations:
                return _rejected_report(
                    transaction=transaction,
                    operation=operation,
                    reason="postgres select operation not allowed",
                    metadata=metadata,
                )
        existing_ids = {item.operation_id for item in transaction.operations}
        if operation.operation_id in existing_ids:
            return _rejected_report(
                transaction=transaction,
                operation=operation,
                reason="postgres transaction operation_id already exists",
                metadata=metadata,
            )
        if len(transaction.operations) + 1 > self._policy.max_operations:
            return _rejected_report(
                transaction=transaction,
                operation=operation,
                reason="postgres transaction operation limit exceeded",
                metadata=metadata,
            )
        if self._policy.block_forbidden_runtime_state:
            _reject_forbidden_operation(operation)
        updated = _replace_transaction(
            transaction=transaction,
            status=transaction.status,
            operations=transaction.operations + (operation,),
            failure_reason=transaction.failure_reason,
            metadata=_merge_metadata(
                {
                    "source": "postgres_transaction_manager",
                    "operation": "add_operation",
                    "operation_id": str(operation.operation_id),
                    "operation_kind": operation.kind.value,
                    "policy": self._policy.to_dict(),
                },
                metadata,
            ),
        )
        return _report(
            transaction=updated,
            changed=True,
            reason="postgres transaction operation added",
            blocked=False,
        )

    def commit(
        self,
        *,
        transaction: PostgresTransactionDescriptor,
        metadata: dict[str, object] | None = None,
    ) -> PostgresTransactionReport:
        """Commit a local transaction descriptor."""

        self._require_transaction(transaction)
        if not self._policy.allow_empty_transaction and not transaction.operations:
            raise ValueError("postgres transaction cannot be empty")
        updated = _replace_transaction(
            transaction=transaction,
            status=PostgresTransactionStatus.COMMITTED,
            operations=transaction.operations,
            failure_reason=None,
            metadata=_merge_metadata(
                {
                    "source": "postgres_transaction_manager",
                    "operation": "commit",
                    "operation_count": transaction.operation_count(),
                    "policy": self._policy.to_dict(),
                },
                metadata,
            ),
        )
        return _report(
            transaction=updated,
            changed=True,
            reason="postgres transaction committed",
            blocked=False,
        )

    def rollback(
        self,
        *,
        transaction: PostgresTransactionDescriptor,
        reason: str,
        metadata: dict[str, object] | None = None,
    ) -> PostgresTransactionReport:
        """Roll back a local transaction descriptor."""

        if not isinstance(transaction, PostgresTransactionDescriptor):
            raise TypeError("transaction must be PostgresTransactionDescriptor")
        normalized_reason = ensure_non_empty_string(reason, "reason")
        updated = _replace_transaction(
            transaction=transaction,
            status=PostgresTransactionStatus.ROLLED_BACK,
            operations=transaction.operations,
            failure_reason=normalized_reason,
            metadata=_merge_metadata(
                {
                    "source": "postgres_transaction_manager",
                    "operation": "rollback",
                    "rollback_reason": normalized_reason,
                    "policy": self._policy.to_dict(),
                },
                metadata,
            ),
        )
        return _report(
            transaction=updated,
            changed=True,
            reason="postgres transaction rolled back",
            blocked=True,
        )

    def fail(
        self,
        *,
        transaction: PostgresTransactionDescriptor,
        reason: str,
        metadata: dict[str, object] | None = None,
    ) -> PostgresTransactionReport:
        """Fail a local transaction descriptor."""

        if not isinstance(transaction, PostgresTransactionDescriptor):
            raise TypeError("transaction must be PostgresTransactionDescriptor")
        normalized_reason = ensure_non_empty_string(reason, "reason")
        updated = _replace_transaction(
            transaction=transaction,
            status=PostgresTransactionStatus.FAILED,
            operations=transaction.operations,
            failure_reason=normalized_reason,
            metadata=_merge_metadata(
                {
                    "source": "postgres_transaction_manager",
                    "operation": "fail",
                    "failure_reason": normalized_reason,
                    "policy": self._policy.to_dict(),
                },
                metadata,
            ),
        )
        return _report(
            transaction=updated,
            changed=True,
            reason="postgres transaction failed",
            blocked=True,
        )

    def _require_transaction(self, transaction: PostgresTransactionDescriptor) -> None:
        if not isinstance(transaction, PostgresTransactionDescriptor):
            raise TypeError("transaction must be PostgresTransactionDescriptor")
        if transaction.status != PostgresTransactionStatus.OPEN:
            raise ValueError("postgres transaction must be OPEN")


def new_postgres_transaction_id() -> UUID:
    """Return a new PostgreSQL transaction descriptor UUID."""

    return uuid4()


def new_postgres_transaction_operation_id() -> UUID:
    """Return a new PostgreSQL transaction operation UUID."""

    return uuid4()


def create_postgres_transaction_operation(
    *,
    kind: PostgresTransactionOperationKind,
    table_kind: PostgresTableKind,
    table_name: str,
    record_key: str | None = None,
    payload_ref: str | None = None,
    metadata: dict[str, object] | None = None,
) -> PostgresTransactionOperation:
    """Create an immutable PostgreSQL transaction operation descriptor."""

    if not isinstance(kind, PostgresTransactionOperationKind):
        raise TypeError("kind must be PostgresTransactionOperationKind")
    if not isinstance(table_kind, PostgresTableKind):
        raise TypeError("table_kind must be PostgresTableKind")
    return PostgresTransactionOperation(
        operation_id=new_postgres_transaction_operation_id(),
        kind=kind,
        table_kind=table_kind,
        table_name=table_name,
        record_key=record_key,
        payload_ref=payload_ref,
        created_at=utc_now(),
        metadata=_merge_metadata(
            {
                "source": "postgres_transaction_operation",
                "operation_kind": kind.value,
                "table_kind": table_kind.value,
                "table_name": table_name,
            },
            metadata,
        ),
    )


def create_postgres_transaction_descriptor(
    *,
    status: PostgresTransactionStatus = PostgresTransactionStatus.CREATED,
    isolation_level: PostgresTransactionIsolationLevel = (
        PostgresTransactionIsolationLevel.READ_COMMITTED
    ),
    operations: tuple[PostgresTransactionOperation, ...]
    | list[PostgresTransactionOperation] = (),
    failure_reason: str | None = None,
    metadata: dict[str, object] | None = None,
) -> PostgresTransactionDescriptor:
    """Create an immutable local PostgreSQL transaction boundary descriptor."""

    if not isinstance(status, PostgresTransactionStatus):
        raise TypeError("status must be PostgresTransactionStatus")
    if not isinstance(isolation_level, PostgresTransactionIsolationLevel):
        raise TypeError("isolation_level must be PostgresTransactionIsolationLevel")
    now = utc_now()
    operations_tuple = tuple(operations)
    return PostgresTransactionDescriptor(
        transaction_id=new_postgres_transaction_id(),
        status=status,
        isolation_level=isolation_level,
        operations=operations_tuple,
        created_at=now,
        updated_at=now,
        failure_reason=failure_reason,
        metadata=_merge_metadata(
            {
                "source": "postgres_transaction_descriptor",
                "status": status.value,
                "isolation_level": isolation_level.value,
                "operation_count": len(operations_tuple),
            },
            metadata,
        ),
    )


def build_postgres_transaction_manager(
    *,
    policy: PostgresTransactionPolicy | None = None,
) -> PostgresTransactionManager:
    """Build a PostgreSQL transaction descriptor manager."""

    return PostgresTransactionManager(policy=policy)


def _replace_transaction(
    *,
    transaction: PostgresTransactionDescriptor,
    status: PostgresTransactionStatus,
    operations: tuple[PostgresTransactionOperation, ...],
    failure_reason: str | None,
    metadata: dict[str, object],
) -> PostgresTransactionDescriptor:
    return PostgresTransactionDescriptor(
        transaction_id=transaction.transaction_id,
        status=status,
        isolation_level=transaction.isolation_level,
        operations=operations,
        created_at=transaction.created_at,
        updated_at=utc_now(),
        failure_reason=failure_reason,
        metadata=_merge_metadata(transaction.metadata, metadata),
    )


def _report(
    *,
    transaction: PostgresTransactionDescriptor,
    changed: bool,
    reason: str,
    blocked: bool,
    rejected_operation_ids: tuple[str, ...] = (),
    rejection_reasons: tuple[str, ...] = (),
    metadata: dict[str, object] | None = None,
) -> PostgresTransactionReport:
    return PostgresTransactionReport(
        transaction=transaction,
        changed=changed,
        reason=reason,
        blocked=blocked,
        rejected_operation_ids=rejected_operation_ids,
        rejection_reasons=rejection_reasons,
        metadata={} if metadata is None else metadata,
    )


def _rejected_report(
    *,
    transaction: PostgresTransactionDescriptor,
    operation: PostgresTransactionOperation,
    reason: str,
    metadata: dict[str, object] | None,
) -> PostgresTransactionReport:
    return _report(
        transaction=transaction,
        changed=False,
        reason=reason,
        blocked=False,
        rejected_operation_ids=(str(operation.operation_id),),
        rejection_reasons=(reason,),
        metadata=_merge_metadata(
            {
                "source": "postgres_transaction_manager",
                "operation": "add_operation",
                "operation_id": str(operation.operation_id),
            },
            metadata,
        ),
    )


def _ensure_operation_tuple(
    operations: tuple[PostgresTransactionOperation, ...]
    | list[PostgresTransactionOperation],
) -> tuple[PostgresTransactionOperation, ...]:
    if not isinstance(operations, (tuple, list)):
        raise TypeError("operations must be tuple[PostgresTransactionOperation, ...]")
    operations_tuple = tuple(operations)
    seen: set[UUID] = set()
    for operation in operations_tuple:
        if not isinstance(operation, PostgresTransactionOperation):
            raise TypeError("every operation must be PostgresTransactionOperation")
        if operation.operation_id in seen:
            raise ValueError("operation_id values must be unique")
        seen.add(operation.operation_id)
    return operations_tuple


def _ensure_string_tuple(
    values: tuple[str, ...] | list[str],
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple[str, ...]")
    return tuple(ensure_non_empty_string(value, field_name) for value in values)


def _ensure_bool(value: bool, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be bool")
    return value


def _ensure_safe_name(value: str, field_name: str) -> str:
    normalized = ensure_non_empty_string(value, field_name)
    _reject_forbidden_name(normalized, field_name)
    return normalized


def _reject_forbidden_name(value: str, field_name: str) -> None:
    if value in _FORBIDDEN_RUNTIME_STATE_NAMES:
        raise ValueError(f"{field_name} must not persist runtime semantic state")


def _reject_forbidden_metadata_keys(metadata: dict[str, object]) -> None:
    for key in metadata:
        if key in _FORBIDDEN_RUNTIME_STATE_NAMES:
            raise ValueError("metadata keys must not persist runtime semantic state")


def _reject_forbidden_operation(operation: PostgresTransactionOperation) -> None:
    _reject_forbidden_name(operation.table_name, "table_name")
    if operation.record_key is not None:
        _reject_forbidden_name(operation.record_key, "record_key")
    _reject_forbidden_metadata_keys(operation.metadata)


def _copy_metadata(value: dict[str, object]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(value)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
    return copied


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
