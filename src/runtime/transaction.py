"""Deterministic runtime transaction utilities for CEBA Runtime."""

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
from ceba.runtime.errors import RuntimeTransactionError
from ceba.runtime.invariants import (
    RuntimeInvariantPolicy,
    RuntimeInvariantReport,
    validate_runtime_invariants,
)


__all__ = [
    "RuntimeTransactionStatus",
    "RuntimeTransactionStepStatus",
    "RuntimeTransactionStepKind",
    "RuntimeTransactionPolicy",
    "RuntimeTransactionStep",
    "RuntimeTransaction",
    "RuntimeTransactionResult",
    "RuntimeTransactionManager",
    "new_runtime_transaction_id",
    "new_runtime_transaction_step_id",
    "create_runtime_transaction_step",
    "create_runtime_transaction",
    "build_runtime_transaction_manager",
]


class RuntimeTransactionStatus(Enum):
    """Stable status values for runtime transactions."""

    CREATED = "CREATED"
    OPEN = "OPEN"
    COMMITTED = "COMMITTED"
    ROLLED_BACK = "ROLLED_BACK"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class RuntimeTransactionStepStatus(Enum):
    """Stable status values for runtime transaction steps."""

    PENDING = "PENDING"
    APPLIED = "APPLIED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class RuntimeTransactionStepKind(Enum):
    """Stable categories for runtime transaction steps."""

    CONTEXT_UPDATE = "CONTEXT_UPDATE"
    INVARIANT_CHECK = "INVARIANT_CHECK"
    CYCLE_STEP = "CYCLE_STEP"
    ENGINE_STEP = "ENGINE_STEP"
    MEMORY_COORDINATION = "MEMORY_COORDINATION"
    WORLD_MODEL_STEP = "WORLD_MODEL_STEP"
    DELTA_STEP = "DELTA_STEP"
    CHECKPOINT_STEP = "CHECKPOINT_STEP"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class RuntimeTransactionPolicy:
    """Configuration object controlling runtime transaction behavior."""

    require_invariants_before_commit: bool = True
    allow_empty_transaction: bool = False
    allow_commit_with_warnings: bool = True
    fail_on_invalid_step: bool = True
    invariant_policy: RuntimeInvariantPolicy = field(
        default_factory=RuntimeInvariantPolicy
    )

    def __post_init__(self) -> None:
        for field_name in (
            "require_invariants_before_commit",
            "allow_empty_transaction",
            "allow_commit_with_warnings",
            "fail_on_invalid_step",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")
        if not isinstance(self.invariant_policy, RuntimeInvariantPolicy):
            raise TypeError("invariant_policy must be RuntimeInvariantPolicy")

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "require_invariants_before_commit": (
                self.require_invariants_before_commit
            ),
            "allow_empty_transaction": self.allow_empty_transaction,
            "allow_commit_with_warnings": self.allow_commit_with_warnings,
            "fail_on_invalid_step": self.fail_on_invalid_step,
            "invariant_policy": self.invariant_policy.to_dict(),
        }


@dataclass(frozen=True)
class RuntimeTransactionStep:
    """Immutable descriptor for one local runtime transaction step."""

    step_id: UUID
    kind: RuntimeTransactionStepKind
    status: RuntimeTransactionStepStatus
    name: str
    context_id: UUID | None
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.step_id, UUID):
            raise TypeError("step_id must be UUID")
        if not isinstance(self.kind, RuntimeTransactionStepKind):
            raise TypeError("kind must be RuntimeTransactionStepKind")
        if not isinstance(self.status, RuntimeTransactionStepStatus):
            raise TypeError("status must be RuntimeTransactionStepStatus")
        name = ensure_non_empty_string(self.name, "name")
        if self.context_id is not None and not isinstance(self.context_id, UUID):
            raise TypeError("context_id must be UUID or None")
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        updated_at = ensure_timezone_aware(self.updated_at, "updated_at")
        if updated_at < created_at:
            raise ValueError("updated_at must be >= created_at")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "step_id": str(self.step_id),
            "kind": self.kind.value,
            "status": self.status.value,
            "name": self.name,
            "context_id": None if self.context_id is None else str(self.context_id),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RuntimeTransaction:
    """Immutable runtime transaction boundary."""

    transaction_id: UUID
    status: RuntimeTransactionStatus
    context: RuntimeContext
    steps: tuple[RuntimeTransactionStep, ...]
    invariant_report: RuntimeInvariantReport | None
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.transaction_id, UUID):
            raise TypeError("transaction_id must be UUID")
        if not isinstance(self.status, RuntimeTransactionStatus):
            raise TypeError("status must be RuntimeTransactionStatus")
        if not isinstance(self.context, RuntimeContext):
            raise TypeError("context must be RuntimeContext")
        steps = _ensure_steps(self.steps)
        if self.invariant_report is not None and not isinstance(
            self.invariant_report,
            RuntimeInvariantReport,
        ):
            raise TypeError("invariant_report must be RuntimeInvariantReport or None")
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        updated_at = ensure_timezone_aware(self.updated_at, "updated_at")
        if updated_at < created_at:
            raise ValueError("updated_at must be >= created_at")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "transaction_id": str(self.transaction_id),
            "status": self.status.value,
            "context": self.context.to_dict(),
            "steps": tuple(step.to_dict() for step in self.steps),
            "invariant_report": (
                None
                if self.invariant_report is None
                else self.invariant_report.to_dict()
            ),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    def step_count(self) -> int:
        """Return transaction step count."""

        return len(self.steps)

    def failed_step_count(self) -> int:
        """Return failed or invalid step count."""

        return sum(
            1
            for step in self.steps
            if step.status
            in {
                RuntimeTransactionStepStatus.FAILED,
                RuntimeTransactionStepStatus.INVALID,
            }
        )

    def has_invariant_report(self) -> bool:
        """Return whether this transaction has an invariant report."""

        return self.invariant_report is not None


@dataclass(frozen=True)
class RuntimeTransactionResult:
    """Immutable result returned by transaction operations."""

    transaction: RuntimeTransaction
    changed: bool
    reason: str
    blocked: bool
    invariant_report: RuntimeInvariantReport | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.transaction, RuntimeTransaction):
            raise TypeError("transaction must be RuntimeTransaction")
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
            "transaction": self.transaction.to_dict(),
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
        """Return whether this transaction operation succeeded."""

        return (
            not self.blocked
            and self.transaction.status == RuntimeTransactionStatus.COMMITTED
        )


class RuntimeTransactionManager:
    """Deterministic manager for local runtime transaction boundaries."""

    def __init__(self, policy: RuntimeTransactionPolicy | None = None) -> None:
        self._policy = RuntimeTransactionPolicy() if policy is None else policy
        if not isinstance(self._policy, RuntimeTransactionPolicy):
            raise TypeError("policy must be RuntimeTransactionPolicy")

    def begin(
        self,
        *,
        context: RuntimeContext,
        metadata: dict[str, object] | None = None,
    ) -> RuntimeTransactionResult:
        """Open a runtime transaction."""

        if not isinstance(context, RuntimeContext):
            raise TypeError("context must be RuntimeContext")
        updated_context = RuntimeContextBuilder().mark_ready(context)
        required_metadata: dict[str, object] = {
            "source": "runtime_transaction_manager",
            "operation": "begin",
        }
        transaction = create_runtime_transaction(
            context=updated_context,
            status=RuntimeTransactionStatus.OPEN,
            steps=(),
            invariant_report=None,
            metadata=_merge_metadata(required_metadata, metadata),
        )
        transaction = _replace_transaction(
            transaction=transaction,
            status=transaction.status,
            context=transaction.context,
            invariant_report=transaction.invariant_report,
            metadata=_force_required_metadata(transaction.metadata, required_metadata),
        )
        return _result(
            transaction=transaction,
            changed=True,
            reason="runtime transaction opened",
            blocked=False,
        )

    def add_step(
        self,
        *,
        transaction: RuntimeTransaction,
        step: RuntimeTransactionStep,
    ) -> RuntimeTransactionResult:
        """Add or replace a transaction step."""

        _ensure_open_transaction(transaction)
        if not isinstance(step, RuntimeTransactionStep):
            raise TypeError("step must be RuntimeTransactionStep")
        updated = _replace_transaction_step(
            transaction=transaction,
            step=step,
            status=transaction.status,
            context=transaction.context,
            invariant_report=transaction.invariant_report,
            metadata=transaction.metadata,
        )
        return _result(
            transaction=updated,
            changed=True,
            reason="runtime transaction step added",
            blocked=False,
        )

    def apply_step(
        self,
        *,
        transaction: RuntimeTransaction,
        step_id: UUID,
        metadata: dict[str, object] | None = None,
    ) -> RuntimeTransactionResult:
        """Mark a transaction step as applied."""

        _ensure_open_transaction(transaction)
        step = _find_step(transaction.steps, step_id)
        updated_step = _update_step(
            step=step,
            status=RuntimeTransactionStepStatus.APPLIED,
            metadata=_manager_metadata(
                existing_metadata=step.metadata,
                required_metadata={
                    "source": "runtime_transaction_manager",
                    "operation": "apply_step",
                },
                caller_metadata=metadata,
            ),
        )
        updated = _replace_transaction_step(
            transaction=transaction,
            step=updated_step,
            status=transaction.status,
            context=transaction.context,
            invariant_report=transaction.invariant_report,
            metadata=transaction.metadata,
        )
        return _result(
            transaction=updated,
            changed=True,
            reason="runtime transaction step applied",
            blocked=False,
        )

    def fail_step(
        self,
        *,
        transaction: RuntimeTransaction,
        step_id: UUID,
        reason: str,
        metadata: dict[str, object] | None = None,
    ) -> RuntimeTransactionResult:
        """Mark a transaction step as failed."""

        _ensure_open_transaction(transaction)
        reason_value = ensure_non_empty_string(reason, "reason")
        step = _find_step(transaction.steps, step_id)
        updated_step = _update_step(
            step=step,
            status=RuntimeTransactionStepStatus.FAILED,
            metadata=_manager_metadata(
                existing_metadata=step.metadata,
                required_metadata={
                    "source": "runtime_transaction_manager",
                    "operation": "fail_step",
                    "failure_reason": reason_value,
                },
                caller_metadata=metadata,
            ),
        )
        blocked = self._policy.fail_on_invalid_step
        context = transaction.context
        status = transaction.status
        if blocked:
            context = RuntimeContextBuilder().mark_failed(
                context,
                reason=reason_value,
            )
            status = RuntimeTransactionStatus.FAILED
        updated = _replace_transaction_step(
            transaction=transaction,
            step=updated_step,
            status=status,
            context=context,
            invariant_report=transaction.invariant_report,
            metadata=transaction.metadata,
        )
        return _result(
            transaction=updated,
            changed=True,
            reason="runtime transaction step failed",
            blocked=blocked,
        )

    def validate_before_commit(
        self,
        *,
        transaction: RuntimeTransaction,
    ) -> RuntimeTransactionResult:
        """Validate runtime invariants before commit."""

        _ensure_open_transaction(transaction)
        report = validate_runtime_invariants(
            context=transaction.context,
            policy=self._policy.invariant_policy,
        )
        if report.has_errors():
            updated = _replace_transaction(
                transaction=transaction,
                status=RuntimeTransactionStatus.BLOCKED,
                context=_mark_context_blocked(transaction.context),
                invariant_report=report,
                metadata=transaction.metadata,
            )
            return _result(
                transaction=updated,
                changed=True,
                reason="runtime transaction blocked by invariant errors",
                blocked=True,
                invariant_report=report,
            )
        if report.warning_count() > 0 and not self._policy.allow_commit_with_warnings:
            updated = _replace_transaction(
                transaction=transaction,
                status=RuntimeTransactionStatus.BLOCKED,
                context=_mark_context_blocked(transaction.context),
                invariant_report=report,
                metadata=transaction.metadata,
            )
            return _result(
                transaction=updated,
                changed=True,
                reason="runtime transaction blocked by invariant warnings",
                blocked=True,
                invariant_report=report,
            )
        updated = _replace_transaction(
            transaction=transaction,
            status=RuntimeTransactionStatus.OPEN,
            context=transaction.context,
            invariant_report=report,
            metadata=transaction.metadata,
        )
        return _result(
            transaction=updated,
            changed=True,
            reason="runtime transaction validated before commit",
            blocked=False,
            invariant_report=report,
        )

    def commit(
        self,
        *,
        transaction: RuntimeTransaction,
    ) -> RuntimeTransactionResult:
        """Commit a local runtime transaction boundary."""

        _ensure_open_transaction(transaction)
        if not self._policy.allow_empty_transaction and not transaction.steps:
            raise RuntimeTransactionError("runtime transaction has no steps")
        if self._policy.fail_on_invalid_step and transaction.failed_step_count() > 0:
            raise RuntimeTransactionError("runtime transaction has failed steps")
        target = transaction
        invariant_report = transaction.invariant_report
        if self._policy.require_invariants_before_commit:
            validation = self.validate_before_commit(transaction=transaction)
            if validation.blocked:
                return validation
            target = validation.transaction
            invariant_report = validation.invariant_report
        context = RuntimeContextBuilder().mark_completed(target.context)
        updated = _replace_transaction(
            transaction=target,
            status=RuntimeTransactionStatus.COMMITTED,
            context=context,
            invariant_report=invariant_report,
            metadata=_manager_metadata(
                existing_metadata=target.metadata,
                required_metadata={
                    "source": "runtime_transaction_manager",
                    "operation": "commit",
                },
                caller_metadata=None,
            ),
        )
        return _result(
            transaction=updated,
            changed=True,
            reason="runtime transaction committed",
            blocked=False,
            invariant_report=invariant_report,
        )

    def rollback(
        self,
        *,
        transaction: RuntimeTransaction,
        reason: str,
    ) -> RuntimeTransactionResult:
        """Roll back a local runtime transaction boundary."""

        if not isinstance(transaction, RuntimeTransaction):
            raise TypeError("transaction must be RuntimeTransaction")
        reason_value = ensure_non_empty_string(reason, "reason")
        rolled_steps = tuple(_rollback_step(step) for step in transaction.steps)
        context = RuntimeContextBuilder().mark_failed(
            transaction.context,
            reason=reason_value,
        )
        updated = _replace_transaction(
            transaction=transaction,
            status=RuntimeTransactionStatus.ROLLED_BACK,
            context=context,
            steps=rolled_steps,
            invariant_report=transaction.invariant_report,
            metadata=_manager_metadata(
                existing_metadata=transaction.metadata,
                required_metadata={
                    "source": "runtime_transaction_manager",
                    "operation": "rollback",
                    "rollback_reason": reason_value,
                },
                caller_metadata=None,
            ),
        )
        return _result(
            transaction=updated,
            changed=True,
            reason="runtime transaction rolled back",
            blocked=True,
            invariant_report=transaction.invariant_report,
        )

    def mark_failed(
        self,
        *,
        transaction: RuntimeTransaction,
        reason: str,
    ) -> RuntimeTransactionResult:
        """Mark a transaction as failed."""

        if not isinstance(transaction, RuntimeTransaction):
            raise TypeError("transaction must be RuntimeTransaction")
        reason_value = ensure_non_empty_string(reason, "reason")
        context = RuntimeContextBuilder().mark_failed(
            transaction.context,
            reason=reason_value,
        )
        updated = _replace_transaction(
            transaction=transaction,
            status=RuntimeTransactionStatus.FAILED,
            context=context,
            invariant_report=transaction.invariant_report,
            metadata=_manager_metadata(
                existing_metadata=transaction.metadata,
                required_metadata={
                    "source": "runtime_transaction_manager",
                    "operation": "mark_failed",
                    "failure_reason": reason_value,
                },
                caller_metadata=None,
            ),
        )
        return _result(
            transaction=updated,
            changed=True,
            reason="runtime transaction failed",
            blocked=True,
            invariant_report=transaction.invariant_report,
        )


def new_runtime_transaction_id() -> UUID:
    """Return a new runtime transaction identifier."""

    return uuid4()


def new_runtime_transaction_step_id() -> UUID:
    """Return a new runtime transaction step identifier."""

    return uuid4()


def create_runtime_transaction_step(
    *,
    name: str,
    kind: RuntimeTransactionStepKind = RuntimeTransactionStepKind.UNKNOWN,
    status: RuntimeTransactionStepStatus = RuntimeTransactionStepStatus.PENDING,
    context_id: UUID | None = None,
    metadata: dict[str, object] | None = None,
) -> RuntimeTransactionStep:
    """Create a runtime transaction step."""

    now = utc_now()
    required_metadata: dict[str, object] = {
        "source": "runtime_transaction_step",
        "step_kind": kind.value if isinstance(kind, RuntimeTransactionStepKind) else kind,
        "step_status": (
            status.value if isinstance(status, RuntimeTransactionStepStatus) else status
        ),
    }
    return RuntimeTransactionStep(
        step_id=new_runtime_transaction_step_id(),
        kind=kind,
        status=status,
        name=name,
        context_id=context_id,
        created_at=now,
        updated_at=now,
        metadata=_merge_metadata(required_metadata, metadata),
    )


def create_runtime_transaction(
    *,
    context: RuntimeContext,
    status: RuntimeTransactionStatus = RuntimeTransactionStatus.CREATED,
    steps: tuple[RuntimeTransactionStep, ...] | list[RuntimeTransactionStep] = (),
    invariant_report: RuntimeInvariantReport | None = None,
    metadata: dict[str, object] | None = None,
) -> RuntimeTransaction:
    """Create a runtime transaction."""

    if not isinstance(context, RuntimeContext):
        raise TypeError("context must be RuntimeContext")
    steps_value = _ensure_steps(steps)
    now = utc_now()
    required_metadata: dict[str, object] = {
        "source": "runtime_transaction",
        "context_id": str(context.context_id),
        "step_count": len(steps_value),
        "has_invariant_report": invariant_report is not None,
    }
    return RuntimeTransaction(
        transaction_id=new_runtime_transaction_id(),
        status=status,
        context=context,
        steps=steps_value,
        invariant_report=invariant_report,
        created_at=now,
        updated_at=now,
        metadata=_merge_metadata(required_metadata, metadata),
    )


def build_runtime_transaction_manager(
    *,
    policy: RuntimeTransactionPolicy | None = None,
) -> RuntimeTransactionManager:
    """Build a runtime transaction manager."""

    return RuntimeTransactionManager(policy=policy)


def _ensure_steps(
    values: tuple[RuntimeTransactionStep, ...] | list[RuntimeTransactionStep],
) -> tuple[RuntimeTransactionStep, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("steps must be tuple or list")
    steps = tuple(values)
    for step in steps:
        if not isinstance(step, RuntimeTransactionStep):
            raise TypeError("steps must contain RuntimeTransactionStep")
    step_ids = tuple(step.step_id for step in steps)
    if len(set(step_ids)) != len(step_ids):
        raise ValueError("step_id values must be unique")
    return steps


def _ensure_open_transaction(transaction: RuntimeTransaction) -> None:
    if not isinstance(transaction, RuntimeTransaction):
        raise TypeError("transaction must be RuntimeTransaction")
    if transaction.status != RuntimeTransactionStatus.OPEN:
        raise RuntimeTransactionError("runtime transaction is not open")


def _find_step(
    steps: tuple[RuntimeTransactionStep, ...],
    step_id: UUID,
) -> RuntimeTransactionStep:
    if not isinstance(step_id, UUID):
        raise TypeError("step_id must be UUID")
    for step in steps:
        if step.step_id == step_id:
            return step
    raise RuntimeTransactionError(
        "runtime transaction step missing",
        context={"step_id": str(step_id)},
    )


def _update_step(
    *,
    step: RuntimeTransactionStep,
    status: RuntimeTransactionStepStatus,
    metadata: dict[str, object],
) -> RuntimeTransactionStep:
    return RuntimeTransactionStep(
        step_id=step.step_id,
        kind=step.kind,
        status=status,
        name=step.name,
        context_id=step.context_id,
        created_at=step.created_at,
        updated_at=utc_now(),
        metadata=_copy_metadata(metadata),
    )


def _rollback_step(step: RuntimeTransactionStep) -> RuntimeTransactionStep:
    if step.status == RuntimeTransactionStepStatus.FAILED:
        return step
    return RuntimeTransactionStep(
        step_id=step.step_id,
        kind=step.kind,
        status=RuntimeTransactionStepStatus.ROLLED_BACK,
        name=step.name,
        context_id=step.context_id,
        created_at=step.created_at,
        updated_at=utc_now(),
        metadata=_manager_metadata(
            existing_metadata=step.metadata,
            required_metadata={
                "source": "runtime_transaction_manager",
                "operation": "rollback_step",
            },
            caller_metadata=None,
        ),
    )


def _replace_transaction_step(
    *,
    transaction: RuntimeTransaction,
    step: RuntimeTransactionStep,
    status: RuntimeTransactionStatus,
    context: RuntimeContext,
    invariant_report: RuntimeInvariantReport | None,
    metadata: dict[str, object],
) -> RuntimeTransaction:
    steps = tuple(
        step if existing.step_id == step.step_id else existing
        for existing in transaction.steps
    )
    if all(existing.step_id != step.step_id for existing in transaction.steps):
        steps = transaction.steps + (step,)
    return _replace_transaction(
        transaction=transaction,
        status=status,
        context=context,
        steps=steps,
        invariant_report=invariant_report,
        metadata=metadata,
    )


def _replace_transaction(
    *,
    transaction: RuntimeTransaction,
    status: RuntimeTransactionStatus,
    context: RuntimeContext,
    invariant_report: RuntimeInvariantReport | None,
    metadata: dict[str, object],
    steps: tuple[RuntimeTransactionStep, ...] | None = None,
) -> RuntimeTransaction:
    return RuntimeTransaction(
        transaction_id=transaction.transaction_id,
        status=status,
        context=context,
        steps=transaction.steps if steps is None else steps,
        invariant_report=invariant_report,
        created_at=transaction.created_at,
        updated_at=utc_now(),
        metadata=metadata,
    )


def _mark_context_blocked(context: RuntimeContext) -> RuntimeContext:
    return RuntimeContext(
        context_id=context.context_id,
        kind=context.kind,
        status=RuntimeContextStatus.BLOCKED,
        input_refs=context.input_refs,
        memory_state=context.memory_state,
        world_model_snapshot=context.world_model_snapshot,
        prediction_input=context.prediction_input,
        prediction_result=context.prediction_result,
        expected_observation=context.expected_observation,
        invariant_flags=context.invariant_flags,
        created_at=context.created_at,
        updated_at=utc_now(),
        metadata=_manager_metadata(
            existing_metadata=context.metadata,
            required_metadata={
                "source": "runtime_transaction_manager",
                "operation": "mark_context_blocked",
            },
            caller_metadata=None,
        ),
    )


def _result(
    *,
    transaction: RuntimeTransaction,
    changed: bool,
    reason: str,
    blocked: bool,
    invariant_report: RuntimeInvariantReport | None = None,
    metadata: dict[str, object] | None = None,
) -> RuntimeTransactionResult:
    return RuntimeTransactionResult(
        transaction=transaction,
        changed=changed,
        reason=reason,
        blocked=blocked,
        invariant_report=invariant_report,
        metadata={} if metadata is None else metadata,
    )


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
