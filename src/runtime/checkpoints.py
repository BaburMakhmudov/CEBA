"""Deterministic runtime checkpoint utilities for CEBA Runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from ceba.runtime.context import (
    RuntimeContext,
    RuntimeContextBuilder,
    RuntimeContextStatus,
    ensure_non_empty_string,
    ensure_timezone_aware,
    utc_now,
)
from ceba.runtime.errors import RuntimeCheckpointError
from ceba.runtime.invariants import (
    RuntimeInvariantPolicy,
    RuntimeInvariantReport,
    validate_runtime_invariants,
)
from ceba.runtime.transaction import RuntimeTransaction, RuntimeTransactionStatus


__all__ = [
    "RuntimeCheckpointKind",
    "RuntimeCheckpointStatus",
    "RuntimeCheckpointPolicy",
    "RuntimeCheckpoint",
    "RuntimeCheckpointResult",
    "RuntimeCheckpointSelection",
    "RuntimeCheckpointManager",
    "new_runtime_checkpoint_id",
    "create_runtime_checkpoint",
    "build_runtime_checkpoint_manager",
]


class RuntimeCheckpointKind(Enum):
    """Stable categories for runtime checkpoints."""

    CYCLE = "CYCLE"
    ENGINE = "ENGINE"
    TRANSACTION = "TRANSACTION"
    SCHEDULER = "SCHEDULER"
    MANUAL = "MANUAL"
    RECOVERY = "RECOVERY"
    VALIDATION = "VALIDATION"
    UNKNOWN = "UNKNOWN"


class RuntimeCheckpointStatus(Enum):
    """Stable status values for runtime checkpoints."""

    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    RESTORED = "RESTORED"
    SUPERSEDED = "SUPERSEDED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class RuntimeCheckpointPolicy:
    """Configuration object controlling checkpoint behavior."""

    require_invariants_on_create: bool = True
    allow_invalid_checkpoint: bool = False
    allow_blocked_transaction_checkpoint: bool = True
    max_checkpoints: int = 128
    invariant_policy: RuntimeInvariantPolicy = field(
        default_factory=RuntimeInvariantPolicy
    )

    def __post_init__(self) -> None:
        for field_name in (
            "require_invariants_on_create",
            "allow_invalid_checkpoint",
            "allow_blocked_transaction_checkpoint",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")
        max_checkpoints = _ensure_positive_int(
            self.max_checkpoints,
            "max_checkpoints",
        )
        if not isinstance(self.invariant_policy, RuntimeInvariantPolicy):
            raise TypeError("invariant_policy must be RuntimeInvariantPolicy")
        object.__setattr__(self, "max_checkpoints", max_checkpoints)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "require_invariants_on_create": self.require_invariants_on_create,
            "allow_invalid_checkpoint": self.allow_invalid_checkpoint,
            "allow_blocked_transaction_checkpoint": (
                self.allow_blocked_transaction_checkpoint
            ),
            "max_checkpoints": self.max_checkpoints,
            "invariant_policy": self.invariant_policy.to_dict(),
        }


@dataclass(frozen=True)
class RuntimeCheckpoint:
    """Immutable local runtime checkpoint descriptor."""

    checkpoint_id: UUID
    kind: RuntimeCheckpointKind
    status: RuntimeCheckpointStatus
    context: RuntimeContext
    transaction: RuntimeTransaction | None
    invariant_report: RuntimeInvariantReport | None
    label: str | None
    created_at: datetime
    updated_at: datetime
    restored_at: datetime | None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint_id, UUID):
            raise TypeError("checkpoint_id must be UUID")
        if not isinstance(self.kind, RuntimeCheckpointKind):
            raise TypeError("kind must be RuntimeCheckpointKind")
        if not isinstance(self.status, RuntimeCheckpointStatus):
            raise TypeError("status must be RuntimeCheckpointStatus")
        if not isinstance(self.context, RuntimeContext):
            raise TypeError("context must be RuntimeContext")
        if self.transaction is not None:
            if not isinstance(self.transaction, RuntimeTransaction):
                raise TypeError("transaction must be RuntimeTransaction or None")
            if self.transaction.context.context_id != self.context.context_id:
                raise ValueError("transaction context_id must match context")
        if self.invariant_report is not None and not isinstance(
            self.invariant_report,
            RuntimeInvariantReport,
        ):
            raise TypeError("invariant_report must be RuntimeInvariantReport or None")
        label = (
            None
            if self.label is None
            else ensure_non_empty_string(self.label, "label")
        )
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        updated_at = ensure_timezone_aware(self.updated_at, "updated_at")
        if updated_at < created_at:
            raise ValueError("updated_at must be >= created_at")
        restored_at = None
        if self.restored_at is not None:
            restored_at = ensure_timezone_aware(self.restored_at, "restored_at")
            if restored_at < created_at:
                raise ValueError("restored_at must be >= created_at")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "restored_at", restored_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "checkpoint_id": str(self.checkpoint_id),
            "kind": self.kind.value,
            "status": self.status.value,
            "context": self.context.to_dict(),
            "transaction": (
                None if self.transaction is None else self.transaction.to_dict()
            ),
            "invariant_report": (
                None
                if self.invariant_report is None
                else self.invariant_report.to_dict()
            ),
            "label": self.label,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "restored_at": (
                None if self.restored_at is None else self.restored_at.isoformat()
            ),
            "metadata": dict(self.metadata),
        }

    def has_transaction(self) -> bool:
        """Return whether this checkpoint has a transaction."""

        return self.transaction is not None

    def has_invariant_report(self) -> bool:
        """Return whether this checkpoint has an invariant report."""

        return self.invariant_report is not None

    def is_usable(self) -> bool:
        """Return whether this checkpoint is usable."""

        if self.status not in {
            RuntimeCheckpointStatus.CREATED,
            RuntimeCheckpointStatus.VALIDATED,
            RuntimeCheckpointStatus.RESTORED,
        }:
            return False
        if self.context.status == RuntimeContextStatus.INVALID:
            return False
        if self.transaction is not None:
            return self.transaction.status != RuntimeTransactionStatus.INVALID
        return True


@dataclass(frozen=True)
class RuntimeCheckpointResult:
    """Immutable result returned by checkpoint operations."""

    checkpoint: RuntimeCheckpoint
    changed: bool
    reason: str
    blocked: bool
    invariant_report: RuntimeInvariantReport | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, RuntimeCheckpoint):
            raise TypeError("checkpoint must be RuntimeCheckpoint")
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        reason = ensure_non_empty_string(self.reason, "reason")
        if not isinstance(self.blocked, bool):
            raise TypeError("blocked must be bool")
        if self.invariant_report is not None and not isinstance(
            self.invariant_report,
            RuntimeInvariantReport,
        ):
            raise TypeError("invariant_report must be RuntimeInvariantReport or None")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "checkpoint": self.checkpoint.to_dict(),
            "changed": self.changed,
            "reason": self.reason,
            "blocked": self.blocked,
            "invariant_report": (
                None
                if self.invariant_report is None
                else self.invariant_report.to_dict()
            ),
            "metadata": dict(self.metadata),
        }

    def is_success(self) -> bool:
        """Return whether the checkpoint operation succeeded."""

        return not self.blocked and self.checkpoint.is_usable()


@dataclass(frozen=True)
class RuntimeCheckpointSelection:
    """Immutable result object for checkpoint selection."""

    selected: RuntimeCheckpoint | None
    checkpoints: tuple[RuntimeCheckpoint, ...]
    rejected_checkpoint_ids: tuple[str, ...]
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.selected is not None and not isinstance(
            self.selected,
            RuntimeCheckpoint,
        ):
            raise TypeError("selected must be RuntimeCheckpoint or None")
        checkpoints = _ensure_checkpoints(self.checkpoints)
        checkpoint_ids = {checkpoint.checkpoint_id for checkpoint in checkpoints}
        if self.selected is not None and self.selected.checkpoint_id not in checkpoint_ids:
            raise ValueError("selected checkpoint_id must exist in checkpoints")
        rejected_ids = _ensure_string_tuple(
            self.rejected_checkpoint_ids,
            "rejected_checkpoint_ids",
        )
        rejection_reasons = _ensure_string_tuple(
            self.rejection_reasons,
            "rejection_reasons",
        )
        if len(rejected_ids) != len(rejection_reasons):
            raise ValueError("rejected_checkpoint_ids and rejection_reasons must match")
        object.__setattr__(self, "checkpoints", checkpoints)
        object.__setattr__(self, "rejected_checkpoint_ids", rejected_ids)
        object.__setattr__(self, "rejection_reasons", rejection_reasons)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "selected": None if self.selected is None else self.selected.to_dict(),
            "checkpoints": tuple(checkpoint.to_dict() for checkpoint in self.checkpoints),
            "rejected_checkpoint_ids": self.rejected_checkpoint_ids,
            "rejection_reasons": self.rejection_reasons,
        }

    def checkpoint_count(self) -> int:
        """Return checkpoint count."""

        return len(self.checkpoints)

    def rejected_count(self) -> int:
        """Return rejected checkpoint count."""

        return len(self.rejected_checkpoint_ids)

    def has_selected(self) -> bool:
        """Return whether a checkpoint is selected."""

        return self.selected is not None


class RuntimeCheckpointManager:
    """Deterministic manager for local runtime checkpoints."""

    def __init__(self, policy: RuntimeCheckpointPolicy | None = None) -> None:
        self._policy = RuntimeCheckpointPolicy() if policy is None else policy
        if not isinstance(self._policy, RuntimeCheckpointPolicy):
            raise TypeError("policy must be RuntimeCheckpointPolicy")

    def create(
        self,
        *,
        context: RuntimeContext,
        kind: RuntimeCheckpointKind = RuntimeCheckpointKind.CYCLE,
        transaction: RuntimeTransaction | None = None,
        label: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> RuntimeCheckpointResult:
        """Create a local runtime checkpoint."""

        if not isinstance(context, RuntimeContext):
            raise TypeError("context must be RuntimeContext")
        if not isinstance(kind, RuntimeCheckpointKind):
            raise TypeError("kind must be RuntimeCheckpointKind")
        _validate_transaction_context(context=context, transaction=transaction)
        label_value = (
            None if label is None else ensure_non_empty_string(label, "label")
        )
        if (
            transaction is not None
            and transaction.status == RuntimeTransactionStatus.BLOCKED
            and not self._policy.allow_blocked_transaction_checkpoint
        ):
            raise RuntimeCheckpointError("blocked transaction checkpoint not allowed")

        invariant_report: RuntimeInvariantReport | None = None
        status = RuntimeCheckpointStatus.CREATED
        blocked = False
        reason = "runtime checkpoint created"
        if self._policy.require_invariants_on_create:
            invariant_report = validate_runtime_invariants(
                context=context,
                policy=self._policy.invariant_policy,
            )
            if invariant_report.has_errors() and not self._policy.allow_invalid_checkpoint:
                status = RuntimeCheckpointStatus.BLOCKED
                blocked = True
                reason = "runtime checkpoint blocked by invariant errors"
            elif not invariant_report.has_errors():
                status = RuntimeCheckpointStatus.VALIDATED

        required_metadata = self._manager_metadata(
            operation="create",
            context=context,
            transaction=transaction,
            invariant_report=invariant_report,
        )
        checkpoint = create_runtime_checkpoint(
            context=context,
            kind=kind,
            status=status,
            transaction=transaction,
            invariant_report=invariant_report,
            label=label_value,
            metadata=_merge_metadata(required_metadata, metadata),
        )
        checkpoint = _replace_checkpoint(
            checkpoint=checkpoint,
            status=checkpoint.status,
            context=checkpoint.context,
            transaction=checkpoint.transaction,
            invariant_report=checkpoint.invariant_report,
            label=checkpoint.label,
            restored_at=checkpoint.restored_at,
            metadata=_force_required_metadata(checkpoint.metadata, required_metadata),
        )
        return _result(
            checkpoint=checkpoint,
            changed=True,
            reason=reason,
            blocked=blocked,
            invariant_report=invariant_report,
        )

    def validate(
        self,
        *,
        checkpoint: RuntimeCheckpoint,
    ) -> RuntimeCheckpointResult:
        """Validate a runtime checkpoint."""

        if not isinstance(checkpoint, RuntimeCheckpoint):
            raise TypeError("checkpoint must be RuntimeCheckpoint")
        report = validate_runtime_invariants(
            context=checkpoint.context,
            policy=self._policy.invariant_policy,
        )
        blocked = report.has_errors() and not self._policy.allow_invalid_checkpoint
        status = (
            RuntimeCheckpointStatus.BLOCKED
            if blocked
            else RuntimeCheckpointStatus.VALIDATED
        )
        updated = _replace_checkpoint(
            checkpoint=checkpoint,
            status=status,
            context=checkpoint.context,
            transaction=checkpoint.transaction,
            invariant_report=report,
            label=checkpoint.label,
            restored_at=checkpoint.restored_at,
            metadata=_manager_metadata(
                existing_metadata=checkpoint.metadata,
                required_metadata={
                    "source": "runtime_checkpoint_manager",
                    "operation": "validate",
                },
                caller_metadata=None,
            ),
        )
        return _result(
            checkpoint=updated,
            changed=True,
            reason=(
                "runtime checkpoint blocked by invariant errors"
                if blocked
                else "runtime checkpoint validated"
            ),
            blocked=blocked,
            invariant_report=report,
        )

    def mark_restored(
        self,
        *,
        checkpoint: RuntimeCheckpoint,
        metadata: dict[str, object] | None = None,
    ) -> RuntimeCheckpointResult:
        """Mark a checkpoint as restored."""

        if not isinstance(checkpoint, RuntimeCheckpoint):
            raise TypeError("checkpoint must be RuntimeCheckpoint")
        if checkpoint.status in {
            RuntimeCheckpointStatus.INVALID,
            RuntimeCheckpointStatus.FAILED,
        }:
            raise RuntimeCheckpointError("checkpoint cannot be restored")
        restored_at = utc_now()
        context = RuntimeContextBuilder().mark_ready(checkpoint.context)
        updated = _replace_checkpoint(
            checkpoint=checkpoint,
            status=RuntimeCheckpointStatus.RESTORED,
            context=context,
            transaction=checkpoint.transaction,
            invariant_report=checkpoint.invariant_report,
            label=checkpoint.label,
            restored_at=restored_at,
            updated_at=restored_at,
            metadata=_manager_metadata(
                existing_metadata=checkpoint.metadata,
                required_metadata={
                    "source": "runtime_checkpoint_manager",
                    "operation": "mark_restored",
                },
                caller_metadata=metadata,
            ),
        )
        return _result(
            checkpoint=updated,
            changed=True,
            reason="runtime checkpoint restored",
            blocked=False,
            invariant_report=checkpoint.invariant_report,
        )

    def mark_superseded(
        self,
        *,
        checkpoint: RuntimeCheckpoint,
        metadata: dict[str, object] | None = None,
    ) -> RuntimeCheckpointResult:
        """Mark a checkpoint as superseded."""

        if not isinstance(checkpoint, RuntimeCheckpoint):
            raise TypeError("checkpoint must be RuntimeCheckpoint")
        updated = _replace_checkpoint(
            checkpoint=checkpoint,
            status=RuntimeCheckpointStatus.SUPERSEDED,
            context=checkpoint.context,
            transaction=checkpoint.transaction,
            invariant_report=checkpoint.invariant_report,
            label=checkpoint.label,
            restored_at=checkpoint.restored_at,
            metadata=_manager_metadata(
                existing_metadata=checkpoint.metadata,
                required_metadata={
                    "source": "runtime_checkpoint_manager",
                    "operation": "mark_superseded",
                },
                caller_metadata=metadata,
            ),
        )
        return _result(
            checkpoint=updated,
            changed=True,
            reason="runtime checkpoint superseded",
            blocked=False,
            invariant_report=checkpoint.invariant_report,
        )

    def mark_failed(
        self,
        *,
        checkpoint: RuntimeCheckpoint,
        reason: str,
        metadata: dict[str, object] | None = None,
    ) -> RuntimeCheckpointResult:
        """Mark a checkpoint as failed."""

        if not isinstance(checkpoint, RuntimeCheckpoint):
            raise TypeError("checkpoint must be RuntimeCheckpoint")
        reason_value = ensure_non_empty_string(reason, "reason")
        context = RuntimeContextBuilder().mark_failed(
            checkpoint.context,
            reason=reason_value,
        )
        updated = _replace_checkpoint(
            checkpoint=checkpoint,
            status=RuntimeCheckpointStatus.FAILED,
            context=context,
            transaction=checkpoint.transaction,
            invariant_report=checkpoint.invariant_report,
            label=checkpoint.label,
            restored_at=checkpoint.restored_at,
            metadata=_manager_metadata(
                existing_metadata=checkpoint.metadata,
                required_metadata={
                    "source": "runtime_checkpoint_manager",
                    "operation": "mark_failed",
                    "failure_reason": reason_value,
                },
                caller_metadata=metadata,
            ),
        )
        return _result(
            checkpoint=updated,
            changed=True,
            reason="runtime checkpoint failed",
            blocked=True,
            invariant_report=checkpoint.invariant_report,
        )

    def select_latest(
        self,
        *,
        checkpoints: tuple[RuntimeCheckpoint, ...] | list[RuntimeCheckpoint],
    ) -> RuntimeCheckpointSelection:
        """Select the latest usable checkpoint."""

        checkpoint_values = _ensure_checkpoints(checkpoints)
        accepted: list[RuntimeCheckpoint] = []
        rejected_ids: list[str] = []
        rejection_reasons: list[str] = []
        for checkpoint in checkpoint_values:
            if checkpoint.is_usable():
                accepted.append(checkpoint)
            else:
                rejected_ids.append(str(checkpoint.checkpoint_id))
                rejection_reasons.append("checkpoint not usable")
        sorted_checkpoints = _sort_checkpoints(tuple(accepted))
        selected = sorted_checkpoints[0] if sorted_checkpoints else None
        return RuntimeCheckpointSelection(
            selected=selected,
            checkpoints=sorted_checkpoints,
            rejected_checkpoint_ids=tuple(rejected_ids),
            rejection_reasons=tuple(rejection_reasons),
        )

    def limit_checkpoints(
        self,
        *,
        checkpoints: tuple[RuntimeCheckpoint, ...] | list[RuntimeCheckpoint],
    ) -> RuntimeCheckpointSelection:
        """Limit checkpoints according to checkpoint policy."""

        checkpoint_values = _sort_checkpoints(_ensure_checkpoints(checkpoints))
        kept = checkpoint_values[: self._policy.max_checkpoints]
        rejected = checkpoint_values[self._policy.max_checkpoints :]
        selected = kept[0] if kept else None
        return RuntimeCheckpointSelection(
            selected=selected,
            checkpoints=kept,
            rejected_checkpoint_ids=tuple(
                str(checkpoint.checkpoint_id) for checkpoint in rejected
            ),
            rejection_reasons=tuple(
                "runtime checkpoint limit exceeded" for _ in rejected
            ),
        )

    def _manager_metadata(
        self,
        *,
        operation: str,
        context: RuntimeContext,
        transaction: RuntimeTransaction | None,
        invariant_report: RuntimeInvariantReport | None,
    ) -> dict[str, object]:
        return {
            "source": "runtime_checkpoint_manager",
            "operation": operation,
            "context_id": str(context.context_id),
            "has_transaction": transaction is not None,
            "has_invariant_report": invariant_report is not None,
            "policy": self._policy.to_dict(),
        }


def new_runtime_checkpoint_id() -> UUID:
    """Return a new runtime checkpoint identifier."""

    return uuid4()


def create_runtime_checkpoint(
    *,
    context: RuntimeContext,
    kind: RuntimeCheckpointKind = RuntimeCheckpointKind.CYCLE,
    status: RuntimeCheckpointStatus = RuntimeCheckpointStatus.CREATED,
    transaction: RuntimeTransaction | None = None,
    invariant_report: RuntimeInvariantReport | None = None,
    label: str | None = None,
    metadata: dict[str, object] | None = None,
) -> RuntimeCheckpoint:
    """Create a runtime checkpoint."""

    if not isinstance(context, RuntimeContext):
        raise TypeError("context must be RuntimeContext")
    _validate_transaction_context(context=context, transaction=transaction)
    now = utc_now()
    required_metadata: dict[str, object] = {
        "source": "runtime_checkpoint",
        "context_id": str(context.context_id),
        "checkpoint_kind": kind.value if isinstance(kind, RuntimeCheckpointKind) else kind,
        "checkpoint_status": (
            status.value if isinstance(status, RuntimeCheckpointStatus) else status
        ),
        "has_transaction": transaction is not None,
        "has_invariant_report": invariant_report is not None,
    }
    return RuntimeCheckpoint(
        checkpoint_id=new_runtime_checkpoint_id(),
        kind=kind,
        status=status,
        context=context,
        transaction=transaction,
        invariant_report=invariant_report,
        label=label,
        created_at=now,
        updated_at=now,
        restored_at=None,
        metadata=_merge_metadata(required_metadata, metadata),
    )


def build_runtime_checkpoint_manager(
    *,
    policy: RuntimeCheckpointPolicy | None = None,
) -> RuntimeCheckpointManager:
    """Build a runtime checkpoint manager."""

    return RuntimeCheckpointManager(policy=policy)


def _ensure_checkpoints(
    values: tuple[RuntimeCheckpoint, ...] | list[RuntimeCheckpoint],
) -> tuple[RuntimeCheckpoint, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("checkpoints must be tuple or list")
    checkpoints = tuple(values)
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, RuntimeCheckpoint):
            raise TypeError("checkpoints must contain RuntimeCheckpoint")
    checkpoint_ids = tuple(checkpoint.checkpoint_id for checkpoint in checkpoints)
    if len(set(checkpoint_ids)) != len(checkpoint_ids):
        raise ValueError("checkpoint_id values must be unique")
    return checkpoints


def _ensure_string_tuple(
    values: tuple[str, ...] | list[str],
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple or list")
    return tuple(ensure_non_empty_string(value, field_name) for value in values)


def _validate_transaction_context(
    *,
    context: RuntimeContext,
    transaction: RuntimeTransaction | None,
) -> None:
    if transaction is None:
        return
    if not isinstance(transaction, RuntimeTransaction):
        raise TypeError("transaction must be RuntimeTransaction or None")
    if transaction.context.context_id != context.context_id:
        raise ValueError("transaction context_id must match context")


def _replace_checkpoint(
    *,
    checkpoint: RuntimeCheckpoint,
    status: RuntimeCheckpointStatus,
    context: RuntimeContext,
    transaction: RuntimeTransaction | None,
    invariant_report: RuntimeInvariantReport | None,
    label: str | None,
    restored_at: datetime | None,
    metadata: dict[str, object],
    updated_at: datetime | None = None,
) -> RuntimeCheckpoint:
    return RuntimeCheckpoint(
        checkpoint_id=checkpoint.checkpoint_id,
        kind=checkpoint.kind,
        status=status,
        context=context,
        transaction=transaction,
        invariant_report=invariant_report,
        label=label,
        created_at=checkpoint.created_at,
        updated_at=utc_now() if updated_at is None else updated_at,
        restored_at=restored_at,
        metadata=metadata,
    )


def _sort_checkpoints(
    checkpoints: tuple[RuntimeCheckpoint, ...],
) -> tuple[RuntimeCheckpoint, ...]:
    return tuple(
        sorted(
            checkpoints,
            key=lambda checkpoint: (
                -checkpoint.updated_at.timestamp(),
                -checkpoint.created_at.timestamp(),
                str(checkpoint.checkpoint_id),
            ),
        )
    )


def _result(
    *,
    checkpoint: RuntimeCheckpoint,
    changed: bool,
    reason: str,
    blocked: bool,
    invariant_report: RuntimeInvariantReport | None,
    metadata: dict[str, object] | None = None,
) -> RuntimeCheckpointResult:
    return RuntimeCheckpointResult(
        checkpoint=checkpoint,
        changed=changed,
        reason=reason,
        blocked=blocked,
        invariant_report=invariant_report,
        metadata={} if metadata is None else metadata,
    )


def _ensure_positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


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


def _manager_metadata(
    *,
    existing_metadata: dict[str, object],
    required_metadata: dict[str, object],
    caller_metadata: dict[str, object] | None,
) -> dict[str, object]:
    merged = _copy_metadata(existing_metadata)
    if caller_metadata is not None:
        merged.update(_copy_metadata(caller_metadata))
    merged.update(_copy_metadata(required_metadata))
    return merged


def _force_required_metadata(
    metadata: dict[str, object],
    required_metadata: dict[str, object],
) -> dict[str, object]:
    merged = _copy_metadata(metadata)
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
