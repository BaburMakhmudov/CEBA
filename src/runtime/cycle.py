"""Deterministic runtime cycle coordination utilities for CEBA Runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from ceba.runtime.checkpoints import (
    RuntimeCheckpoint,
    RuntimeCheckpointKind,
    RuntimeCheckpointPolicy,
    RuntimeCheckpointResult,
    build_runtime_checkpoint_manager,
)
from ceba.runtime.context import (
    RuntimeContext,
    RuntimeContextBuilder,
    RuntimeContextStatus,
    ensure_non_empty_string,
    ensure_timezone_aware,
    utc_now,
)
from ceba.runtime.errors import RuntimeCycleError
from ceba.runtime.invariants import (
    RuntimeInvariantPolicy,
    RuntimeInvariantReport,
    validate_runtime_invariants,
)
from ceba.runtime.scheduler import (
    RuntimeSchedule,
    RuntimeScheduleResult,
    RuntimeScheduleStatus,
    RuntimeSchedulerPolicy,
    RuntimeTask,
    build_runtime_scheduler,
)
from ceba.runtime.transaction import (
    RuntimeTransaction,
    RuntimeTransactionPolicy,
    RuntimeTransactionResult,
    build_runtime_transaction_manager,
)


__all__ = [
    "RuntimeCyclePhase",
    "RuntimeCycleStatus",
    "RuntimeCyclePolicy",
    "RuntimeCycle",
    "RuntimeCycleResult",
    "RuntimeCycleCoordinator",
    "new_runtime_cycle_id",
    "create_runtime_cycle",
    "build_runtime_cycle_coordinator",
]


class RuntimeCyclePhase(Enum):
    """Stable phases for runtime cycles."""

    CREATED = "CREATED"
    CONTEXT_READY = "CONTEXT_READY"
    INVARIANTS_VALIDATED = "INVARIANTS_VALIDATED"
    SCHEDULE_BUILT = "SCHEDULE_BUILT"
    TRANSACTION_OPENED = "TRANSACTION_OPENED"
    TASKS_COORDINATED = "TASKS_COORDINATED"
    TRANSACTION_COMMITTED = "TRANSACTION_COMMITTED"
    CHECKPOINT_CREATED = "CHECKPOINT_CREATED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    UNKNOWN = "UNKNOWN"


class RuntimeCycleStatus(Enum):
    """Stable status values for runtime cycles."""

    CREATED = "CREATED"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class RuntimeCyclePolicy:
    """Configuration object controlling runtime cycle coordination."""

    require_invariants_before_cycle: bool = True
    require_schedule: bool = True
    require_transaction: bool = True
    create_checkpoint_on_complete: bool = True
    create_checkpoint_on_failure: bool = True
    allow_schedule_warnings: bool = True
    invariant_policy: RuntimeInvariantPolicy = field(
        default_factory=RuntimeInvariantPolicy
    )
    scheduler_policy: RuntimeSchedulerPolicy = field(
        default_factory=RuntimeSchedulerPolicy
    )
    transaction_policy: RuntimeTransactionPolicy = field(
        default_factory=RuntimeTransactionPolicy
    )
    checkpoint_policy: RuntimeCheckpointPolicy = field(
        default_factory=RuntimeCheckpointPolicy
    )

    def __post_init__(self) -> None:
        for field_name in (
            "require_invariants_before_cycle",
            "require_schedule",
            "require_transaction",
            "create_checkpoint_on_complete",
            "create_checkpoint_on_failure",
            "allow_schedule_warnings",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")
        if not isinstance(self.invariant_policy, RuntimeInvariantPolicy):
            raise TypeError("invariant_policy must be RuntimeInvariantPolicy")
        if not isinstance(self.scheduler_policy, RuntimeSchedulerPolicy):
            raise TypeError("scheduler_policy must be RuntimeSchedulerPolicy")
        if not isinstance(self.transaction_policy, RuntimeTransactionPolicy):
            raise TypeError("transaction_policy must be RuntimeTransactionPolicy")
        if not isinstance(self.checkpoint_policy, RuntimeCheckpointPolicy):
            raise TypeError("checkpoint_policy must be RuntimeCheckpointPolicy")

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "require_invariants_before_cycle": self.require_invariants_before_cycle,
            "require_schedule": self.require_schedule,
            "require_transaction": self.require_transaction,
            "create_checkpoint_on_complete": self.create_checkpoint_on_complete,
            "create_checkpoint_on_failure": self.create_checkpoint_on_failure,
            "allow_schedule_warnings": self.allow_schedule_warnings,
            "invariant_policy": self.invariant_policy.to_dict(),
            "scheduler_policy": self.scheduler_policy.to_dict(),
            "transaction_policy": self.transaction_policy.to_dict(),
            "checkpoint_policy": self.checkpoint_policy.to_dict(),
        }


@dataclass(frozen=True)
class RuntimeCycle:
    """Immutable runtime cycle descriptor."""

    cycle_id: UUID
    status: RuntimeCycleStatus
    phase: RuntimeCyclePhase
    context: RuntimeContext
    schedule: RuntimeSchedule | None
    transaction: RuntimeTransaction | None
    checkpoint: RuntimeCheckpoint | None
    invariant_report: RuntimeInvariantReport | None
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.cycle_id, UUID):
            raise TypeError("cycle_id must be UUID")
        if not isinstance(self.status, RuntimeCycleStatus):
            raise TypeError("status must be RuntimeCycleStatus")
        if not isinstance(self.phase, RuntimeCyclePhase):
            raise TypeError("phase must be RuntimeCyclePhase")
        if not isinstance(self.context, RuntimeContext):
            raise TypeError("context must be RuntimeContext")
        _validate_optional_context_link(
            owner=self.schedule,
            expected_type=RuntimeSchedule,
            context=self.context,
            field_name="schedule",
        )
        _validate_optional_context_link(
            owner=self.transaction,
            expected_type=RuntimeTransaction,
            context=self.context,
            field_name="transaction",
        )
        _validate_optional_context_link(
            owner=self.checkpoint,
            expected_type=RuntimeCheckpoint,
            context=self.context,
            field_name="checkpoint",
        )
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
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "cycle_id": str(self.cycle_id),
            "status": self.status.value,
            "phase": self.phase.value,
            "context": self.context.to_dict(),
            "schedule": None if self.schedule is None else self.schedule.to_dict(),
            "transaction": (
                None if self.transaction is None else self.transaction.to_dict()
            ),
            "checkpoint": (
                None if self.checkpoint is None else self.checkpoint.to_dict()
            ),
            "invariant_report": (
                None
                if self.invariant_report is None
                else self.invariant_report.to_dict()
            ),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    def has_schedule(self) -> bool:
        """Return whether this cycle has a schedule."""

        return self.schedule is not None

    def has_transaction(self) -> bool:
        """Return whether this cycle has a transaction."""

        return self.transaction is not None

    def has_checkpoint(self) -> bool:
        """Return whether this cycle has a checkpoint."""

        return self.checkpoint is not None

    def has_invariant_report(self) -> bool:
        """Return whether this cycle has an invariant report."""

        return self.invariant_report is not None

    def is_terminal(self) -> bool:
        """Return whether this cycle is terminal."""

        return self.status in {
            RuntimeCycleStatus.COMPLETED,
            RuntimeCycleStatus.FAILED,
            RuntimeCycleStatus.BLOCKED,
            RuntimeCycleStatus.INVALID,
        }


@dataclass(frozen=True)
class RuntimeCycleResult:
    """Immutable result returned by runtime cycle coordination operations."""

    cycle: RuntimeCycle
    changed: bool
    reason: str
    blocked: bool
    invariant_report: RuntimeInvariantReport | None = None
    schedule_result: RuntimeScheduleResult | None = None
    transaction_result: RuntimeTransactionResult | None = None
    checkpoint_result: RuntimeCheckpointResult | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.cycle, RuntimeCycle):
            raise TypeError("cycle must be RuntimeCycle")
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        reason = ensure_non_empty_string(self.reason, "reason")
        if not isinstance(self.blocked, bool):
            raise TypeError("blocked must be bool")
        _ensure_optional(self.invariant_report, RuntimeInvariantReport, "invariant_report")
        _ensure_optional(self.schedule_result, RuntimeScheduleResult, "schedule_result")
        _ensure_optional(
            self.transaction_result,
            RuntimeTransactionResult,
            "transaction_result",
        )
        _ensure_optional(
            self.checkpoint_result,
            RuntimeCheckpointResult,
            "checkpoint_result",
        )
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "cycle": self.cycle.to_dict(),
            "changed": self.changed,
            "reason": self.reason,
            "blocked": self.blocked,
            "invariant_report": _optional_to_dict(self.invariant_report),
            "schedule_result": _optional_to_dict(self.schedule_result),
            "transaction_result": _optional_to_dict(self.transaction_result),
            "checkpoint_result": _optional_to_dict(self.checkpoint_result),
            "metadata": dict(self.metadata),
        }

    def is_success(self) -> bool:
        """Return whether cycle coordination succeeded."""

        return not self.blocked and self.cycle.status == RuntimeCycleStatus.COMPLETED


class RuntimeCycleCoordinator:
    """Deterministic coordinator for one CEBA runtime cycle."""

    def __init__(self, policy: RuntimeCyclePolicy | None = None) -> None:
        self._policy = RuntimeCyclePolicy() if policy is None else policy
        if not isinstance(self._policy, RuntimeCyclePolicy):
            raise TypeError("policy must be RuntimeCyclePolicy")

    def create(
        self,
        *,
        context: RuntimeContext,
        metadata: dict[str, object] | None = None,
    ) -> RuntimeCycleResult:
        """Create a runtime cycle."""

        if not isinstance(context, RuntimeContext):
            raise TypeError("context must be RuntimeContext")
        cycle_context = RuntimeContextBuilder().mark_ready(context)
        required_metadata = self._manager_metadata(
            operation="create",
            context=cycle_context,
        )
        cycle = create_runtime_cycle(
            context=cycle_context,
            status=RuntimeCycleStatus.READY,
            phase=RuntimeCyclePhase.CONTEXT_READY,
            metadata=_merge_metadata(required_metadata, metadata),
        )
        cycle = _replace_cycle(
            cycle=cycle,
            status=cycle.status,
            phase=cycle.phase,
            context=cycle.context,
            schedule=cycle.schedule,
            transaction=cycle.transaction,
            checkpoint=cycle.checkpoint,
            invariant_report=cycle.invariant_report,
            metadata=_force_required_metadata(cycle.metadata, required_metadata),
        )
        return _result(
            cycle=cycle,
            changed=True,
            reason="runtime cycle created",
            blocked=False,
        )

    def validate_invariants(
        self,
        *,
        cycle: RuntimeCycle,
    ) -> RuntimeCycleResult:
        """Validate runtime invariants for a cycle."""

        _ensure_cycle(cycle)
        report = validate_runtime_invariants(
            context=cycle.context,
            policy=self._policy.invariant_policy,
        )
        if report.has_errors():
            blocked_cycle = _replace_cycle(
                cycle=cycle,
                status=RuntimeCycleStatus.BLOCKED,
                phase=RuntimeCyclePhase.BLOCKED,
                context=_mark_context_blocked(cycle.context),
                schedule=cycle.schedule,
                transaction=cycle.transaction,
                checkpoint=cycle.checkpoint,
                invariant_report=report,
                metadata=_manager_metadata(
                    existing_metadata=cycle.metadata,
                    required_metadata=self._manager_metadata(
                        operation="validate_invariants",
                        context=cycle.context,
                    ),
                    caller_metadata=None,
                ),
            )
            return _result(
                cycle=blocked_cycle,
                changed=True,
                reason="runtime cycle blocked by invariant errors",
                blocked=True,
                invariant_report=report,
            )
        updated = _replace_cycle(
            cycle=cycle,
            status=(
                RuntimeCycleStatus.RUNNING
                if cycle.status == RuntimeCycleStatus.RUNNING
                else RuntimeCycleStatus.READY
            ),
            phase=RuntimeCyclePhase.INVARIANTS_VALIDATED,
            context=cycle.context,
            schedule=cycle.schedule,
            transaction=cycle.transaction,
            checkpoint=cycle.checkpoint,
            invariant_report=report,
            metadata=cycle.metadata,
        )
        return _result(
            cycle=updated,
            changed=True,
            reason="runtime cycle invariants validated",
            blocked=False,
            invariant_report=report,
        )

    def build_schedule(
        self,
        *,
        cycle: RuntimeCycle,
        tasks: tuple[RuntimeTask, ...] | list[RuntimeTask],
    ) -> RuntimeCycleResult:
        """Build and attach a runtime schedule."""

        _ensure_cycle(cycle)
        target = cycle
        invariant_report = cycle.invariant_report
        if self._policy.require_invariants_before_cycle and invariant_report is None:
            validation = self.validate_invariants(cycle=cycle)
            if validation.blocked:
                return validation
            target = validation.cycle
            invariant_report = validation.invariant_report
        scheduler = build_runtime_scheduler(policy=self._policy.scheduler_policy)
        schedule_result = scheduler.build_schedule(context=target.context, tasks=tasks)
        report = schedule_result.invariant_report or invariant_report
        if schedule_result.blocked:
            blocked = _replace_cycle(
                cycle=target,
                status=RuntimeCycleStatus.BLOCKED,
                phase=RuntimeCyclePhase.BLOCKED,
                context=target.context,
                schedule=schedule_result.schedule,
                transaction=target.transaction,
                checkpoint=target.checkpoint,
                invariant_report=report,
                metadata=target.metadata,
            )
            return _result(
                cycle=blocked,
                changed=True,
                reason="runtime cycle blocked by schedule",
                blocked=True,
                invariant_report=report,
                schedule_result=schedule_result,
            )
        updated = _replace_cycle(
            cycle=target,
            status=RuntimeCycleStatus.RUNNING,
            phase=RuntimeCyclePhase.SCHEDULE_BUILT,
            context=target.context,
            schedule=schedule_result.schedule,
            transaction=target.transaction,
            checkpoint=target.checkpoint,
            invariant_report=report,
            metadata=target.metadata,
        )
        return _result(
            cycle=updated,
            changed=True,
            reason="runtime cycle schedule built",
            blocked=False,
            invariant_report=report,
            schedule_result=schedule_result,
        )

    def open_transaction(self, *, cycle: RuntimeCycle) -> RuntimeCycleResult:
        """Open a runtime transaction for the cycle."""

        _ensure_cycle(cycle)
        if not self._policy.require_transaction:
            return _result(
                cycle=cycle,
                changed=False,
                reason="runtime transaction not required",
                blocked=False,
            )
        manager = build_runtime_transaction_manager(
            policy=self._policy.transaction_policy
        )
        transaction_result = manager.begin(context=cycle.context)
        if transaction_result.blocked:
            blocked = _replace_cycle(
                cycle=cycle,
                status=RuntimeCycleStatus.BLOCKED,
                phase=RuntimeCyclePhase.BLOCKED,
                context=transaction_result.transaction.context,
                schedule=cycle.schedule,
                transaction=transaction_result.transaction,
                checkpoint=cycle.checkpoint,
                invariant_report=cycle.invariant_report,
                metadata=cycle.metadata,
            )
            return _result(
                cycle=blocked,
                changed=True,
                reason=transaction_result.reason,
                blocked=True,
                transaction_result=transaction_result,
            )
        updated = _replace_cycle(
            cycle=cycle,
            status=RuntimeCycleStatus.RUNNING,
            phase=RuntimeCyclePhase.TRANSACTION_OPENED,
            context=transaction_result.transaction.context,
            schedule=cycle.schedule,
            transaction=transaction_result.transaction,
            checkpoint=cycle.checkpoint,
            invariant_report=cycle.invariant_report,
            metadata=cycle.metadata,
        )
        return _result(
            cycle=updated,
            changed=True,
            reason="runtime cycle transaction opened",
            blocked=False,
            transaction_result=transaction_result,
        )

    def coordinate_tasks(self, *, cycle: RuntimeCycle) -> RuntimeCycleResult:
        """Coordinate schedule readiness without executing tasks."""

        _ensure_cycle(cycle)
        if self._policy.require_schedule and cycle.schedule is None:
            raise RuntimeCycleError("runtime schedule is required")
        if cycle.schedule is None:
            return _result(
                cycle=cycle,
                changed=False,
                reason="runtime schedule not required",
                blocked=False,
            )
        if cycle.schedule.status in {
            RuntimeScheduleStatus.BLOCKED,
            RuntimeScheduleStatus.INVALID,
        }:
            blocked = _replace_cycle(
                cycle=cycle,
                status=RuntimeCycleStatus.BLOCKED,
                phase=RuntimeCyclePhase.BLOCKED,
                context=cycle.context,
                schedule=cycle.schedule,
                transaction=cycle.transaction,
                checkpoint=cycle.checkpoint,
                invariant_report=cycle.invariant_report,
                metadata=cycle.metadata,
            )
            return _result(
                cycle=blocked,
                changed=True,
                reason="runtime cycle schedule not executable",
                blocked=True,
                invariant_report=cycle.invariant_report,
            )
        updated = _replace_cycle(
            cycle=cycle,
            status=RuntimeCycleStatus.RUNNING,
            phase=RuntimeCyclePhase.TASKS_COORDINATED,
            context=cycle.context,
            schedule=cycle.schedule,
            transaction=cycle.transaction,
            checkpoint=cycle.checkpoint,
            invariant_report=cycle.invariant_report,
            metadata=cycle.metadata,
        )
        return _result(
            cycle=updated,
            changed=True,
            reason="runtime cycle tasks coordinated",
            blocked=False,
            invariant_report=cycle.invariant_report,
        )

    def commit_transaction(self, *, cycle: RuntimeCycle) -> RuntimeCycleResult:
        """Commit the cycle transaction."""

        _ensure_cycle(cycle)
        if not self._policy.require_transaction:
            return _result(
                cycle=cycle,
                changed=False,
                reason="runtime transaction not required",
                blocked=False,
            )
        if cycle.transaction is None:
            raise RuntimeCycleError("runtime transaction is required")
        manager = build_runtime_transaction_manager(
            policy=self._policy.transaction_policy
        )
        transaction_result = manager.commit(transaction=cycle.transaction)
        report = transaction_result.invariant_report or cycle.invariant_report
        if transaction_result.blocked:
            blocked = _replace_cycle(
                cycle=cycle,
                status=RuntimeCycleStatus.BLOCKED,
                phase=RuntimeCyclePhase.BLOCKED,
                context=transaction_result.transaction.context,
                schedule=cycle.schedule,
                transaction=transaction_result.transaction,
                checkpoint=cycle.checkpoint,
                invariant_report=report,
                metadata=cycle.metadata,
            )
            return _result(
                cycle=blocked,
                changed=True,
                reason=transaction_result.reason,
                blocked=True,
                invariant_report=report,
                transaction_result=transaction_result,
            )
        updated = _replace_cycle(
            cycle=cycle,
            status=RuntimeCycleStatus.RUNNING,
            phase=RuntimeCyclePhase.TRANSACTION_COMMITTED,
            context=transaction_result.transaction.context,
            schedule=cycle.schedule,
            transaction=transaction_result.transaction,
            checkpoint=cycle.checkpoint,
            invariant_report=report,
            metadata=cycle.metadata,
        )
        return _result(
            cycle=updated,
            changed=True,
            reason="runtime cycle transaction committed",
            blocked=False,
            invariant_report=report,
            transaction_result=transaction_result,
        )

    def create_checkpoint(
        self,
        *,
        cycle: RuntimeCycle,
        kind: RuntimeCheckpointKind = RuntimeCheckpointKind.CYCLE,
    ) -> RuntimeCycleResult:
        """Create and attach a runtime checkpoint."""

        _ensure_cycle(cycle)
        manager = build_runtime_checkpoint_manager(policy=self._policy.checkpoint_policy)
        checkpoint_result = manager.create(
            context=cycle.context,
            kind=kind,
            transaction=cycle.transaction,
        )
        report = checkpoint_result.invariant_report or cycle.invariant_report
        if checkpoint_result.blocked:
            blocked = _replace_cycle(
                cycle=cycle,
                status=RuntimeCycleStatus.BLOCKED,
                phase=RuntimeCyclePhase.BLOCKED,
                context=checkpoint_result.checkpoint.context,
                schedule=cycle.schedule,
                transaction=cycle.transaction,
                checkpoint=checkpoint_result.checkpoint,
                invariant_report=report,
                metadata=cycle.metadata,
            )
            return _result(
                cycle=blocked,
                changed=True,
                reason=checkpoint_result.reason,
                blocked=True,
                invariant_report=report,
                checkpoint_result=checkpoint_result,
            )
        updated = _replace_cycle(
            cycle=cycle,
            status=cycle.status,
            phase=RuntimeCyclePhase.CHECKPOINT_CREATED,
            context=checkpoint_result.checkpoint.context,
            schedule=cycle.schedule,
            transaction=cycle.transaction,
            checkpoint=checkpoint_result.checkpoint,
            invariant_report=report,
            metadata=cycle.metadata,
        )
        return _result(
            cycle=updated,
            changed=True,
            reason="runtime cycle checkpoint created",
            blocked=False,
            invariant_report=report,
            checkpoint_result=checkpoint_result,
        )

    def complete(self, *, cycle: RuntimeCycle) -> RuntimeCycleResult:
        """Complete a runtime cycle."""

        _ensure_cycle(cycle)
        target = cycle
        checkpoint_result: RuntimeCheckpointResult | None = None
        if self._policy.create_checkpoint_on_complete and target.checkpoint is None:
            checkpoint = self.create_checkpoint(cycle=target)
            if checkpoint.blocked:
                return checkpoint
            target = checkpoint.cycle
            checkpoint_result = checkpoint.checkpoint_result
        context = RuntimeContextBuilder().mark_completed(target.context)
        required_metadata = self._manager_metadata(operation="complete", context=context)
        updated = _replace_cycle(
            cycle=target,
            status=RuntimeCycleStatus.COMPLETED,
            phase=RuntimeCyclePhase.COMPLETED,
            context=context,
            schedule=target.schedule,
            transaction=target.transaction,
            checkpoint=target.checkpoint,
            invariant_report=target.invariant_report,
            metadata=_manager_metadata(
                existing_metadata=target.metadata,
                required_metadata=required_metadata,
                caller_metadata=None,
            ),
        )
        return _result(
            cycle=updated,
            changed=True,
            reason="runtime cycle completed",
            blocked=False,
            invariant_report=updated.invariant_report,
            checkpoint_result=checkpoint_result,
        )

    def fail(self, *, cycle: RuntimeCycle, reason: str) -> RuntimeCycleResult:
        """Fail a runtime cycle."""

        _ensure_cycle(cycle)
        failure_reason = ensure_non_empty_string(reason, "reason")
        context = RuntimeContextBuilder().mark_failed(cycle.context, reason=failure_reason)
        target = _replace_cycle(
            cycle=cycle,
            status=cycle.status,
            phase=cycle.phase,
            context=context,
            schedule=cycle.schedule,
            transaction=cycle.transaction,
            checkpoint=cycle.checkpoint,
            invariant_report=cycle.invariant_report,
            metadata=cycle.metadata,
        )
        checkpoint_result: RuntimeCheckpointResult | None = None
        if self._policy.create_checkpoint_on_failure and target.checkpoint is None:
            checkpoint = self.create_checkpoint(
                cycle=target,
                kind=RuntimeCheckpointKind.RECOVERY,
            )
            if checkpoint.checkpoint_result is not None:
                target = checkpoint.cycle
                checkpoint_result = checkpoint.checkpoint_result
        required_metadata = self._manager_metadata(operation="fail", context=context)
        required_metadata["failure_reason"] = failure_reason
        failed = _replace_cycle(
            cycle=target,
            status=RuntimeCycleStatus.FAILED,
            phase=RuntimeCyclePhase.FAILED,
            context=target.context,
            schedule=target.schedule,
            transaction=target.transaction,
            checkpoint=target.checkpoint,
            invariant_report=target.invariant_report,
            metadata=_manager_metadata(
                existing_metadata=target.metadata,
                required_metadata=required_metadata,
                caller_metadata=None,
            ),
        )
        return _result(
            cycle=failed,
            changed=True,
            reason="runtime cycle failed",
            blocked=True,
            invariant_report=failed.invariant_report,
            checkpoint_result=checkpoint_result,
        )

    def run_prepared(
        self,
        *,
        context: RuntimeContext,
        tasks: tuple[RuntimeTask, ...] | list[RuntimeTask],
        metadata: dict[str, object] | None = None,
    ) -> RuntimeCycleResult:
        """Coordinate a prepared cycle without executing domain tasks."""

        result = self.create(context=context, metadata=metadata)
        if result.blocked:
            return result
        current = result.cycle
        if self._policy.require_invariants_before_cycle:
            result = self.validate_invariants(cycle=current)
            if result.blocked:
                return result
            current = result.cycle
        if self._policy.require_schedule:
            result = self.build_schedule(cycle=current, tasks=tasks)
            if result.blocked:
                return result
            current = result.cycle
        if self._policy.require_transaction:
            result = self.open_transaction(cycle=current)
            if result.blocked:
                return result
            current = result.cycle
        result = self.coordinate_tasks(cycle=current)
        if result.blocked:
            return result
        current = result.cycle
        if self._policy.require_transaction:
            result = self.commit_transaction(cycle=current)
            if result.blocked:
                return result
            current = result.cycle
        if self._policy.create_checkpoint_on_complete:
            result = self.create_checkpoint(cycle=current)
            if result.blocked:
                return result
            current = result.cycle
        return self.complete(cycle=current)

    def _manager_metadata(
        self,
        *,
        operation: str,
        context: RuntimeContext,
    ) -> dict[str, object]:
        return {
            "source": "runtime_cycle_coordinator",
            "operation": operation,
            "context_id": str(context.context_id),
            "policy": self._policy.to_dict(),
        }


def new_runtime_cycle_id() -> UUID:
    """Return a new runtime cycle identifier."""

    return uuid4()


def create_runtime_cycle(
    *,
    context: RuntimeContext,
    status: RuntimeCycleStatus = RuntimeCycleStatus.CREATED,
    phase: RuntimeCyclePhase = RuntimeCyclePhase.CREATED,
    schedule: RuntimeSchedule | None = None,
    transaction: RuntimeTransaction | None = None,
    checkpoint: RuntimeCheckpoint | None = None,
    invariant_report: RuntimeInvariantReport | None = None,
    metadata: dict[str, object] | None = None,
) -> RuntimeCycle:
    """Create a runtime cycle descriptor."""

    if not isinstance(context, RuntimeContext):
        raise TypeError("context must be RuntimeContext")
    now = utc_now()
    required_metadata: dict[str, object] = {
        "source": "runtime_cycle",
        "context_id": str(context.context_id),
        "cycle_status": status.value if isinstance(status, RuntimeCycleStatus) else status,
        "cycle_phase": phase.value if isinstance(phase, RuntimeCyclePhase) else phase,
        "has_schedule": schedule is not None,
        "has_transaction": transaction is not None,
        "has_checkpoint": checkpoint is not None,
        "has_invariant_report": invariant_report is not None,
    }
    return RuntimeCycle(
        cycle_id=new_runtime_cycle_id(),
        status=status,
        phase=phase,
        context=context,
        schedule=schedule,
        transaction=transaction,
        checkpoint=checkpoint,
        invariant_report=invariant_report,
        created_at=now,
        updated_at=now,
        metadata=_merge_metadata(required_metadata, metadata),
    )


def build_runtime_cycle_coordinator(
    *,
    policy: RuntimeCyclePolicy | None = None,
) -> RuntimeCycleCoordinator:
    """Build a runtime cycle coordinator."""

    return RuntimeCycleCoordinator(policy=policy)


def _ensure_cycle(cycle: RuntimeCycle) -> None:
    if not isinstance(cycle, RuntimeCycle):
        raise TypeError("cycle must be RuntimeCycle")


def _ensure_optional(value: object, expected_type: type[object], field_name: str) -> None:
    if value is not None and not isinstance(value, expected_type):
        raise TypeError(f"{field_name} must be {expected_type.__name__} or None")


def _validate_optional_context_link(
    *,
    owner: object,
    expected_type: type[object],
    context: RuntimeContext,
    field_name: str,
) -> None:
    if owner is None:
        return
    if not isinstance(owner, expected_type):
        raise TypeError(f"{field_name} must be {expected_type.__name__} or None")
    owner_context = getattr(owner, "context")
    if owner_context.context_id != context.context_id:
        raise ValueError(f"{field_name} context_id must match context")


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
                "source": "runtime_cycle_coordinator",
                "operation": "mark_context_blocked",
            },
            caller_metadata=None,
        ),
    )


def _replace_cycle(
    *,
    cycle: RuntimeCycle,
    status: RuntimeCycleStatus,
    phase: RuntimeCyclePhase,
    context: RuntimeContext,
    schedule: RuntimeSchedule | None,
    transaction: RuntimeTransaction | None,
    checkpoint: RuntimeCheckpoint | None,
    invariant_report: RuntimeInvariantReport | None,
    metadata: dict[str, object],
) -> RuntimeCycle:
    return RuntimeCycle(
        cycle_id=cycle.cycle_id,
        status=status,
        phase=phase,
        context=context,
        schedule=schedule,
        transaction=transaction,
        checkpoint=checkpoint,
        invariant_report=invariant_report,
        created_at=cycle.created_at,
        updated_at=utc_now(),
        metadata=metadata,
    )


def _result(
    *,
    cycle: RuntimeCycle,
    changed: bool,
    reason: str,
    blocked: bool,
    invariant_report: RuntimeInvariantReport | None = None,
    schedule_result: RuntimeScheduleResult | None = None,
    transaction_result: RuntimeTransactionResult | None = None,
    checkpoint_result: RuntimeCheckpointResult | None = None,
    metadata: dict[str, object] | None = None,
) -> RuntimeCycleResult:
    return RuntimeCycleResult(
        cycle=cycle,
        changed=changed,
        reason=reason,
        blocked=blocked,
        invariant_report=invariant_report,
        schedule_result=schedule_result,
        transaction_result=transaction_result,
        checkpoint_result=checkpoint_result,
        metadata={} if metadata is None else metadata,
    )


def _optional_to_dict(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    to_dict = getattr(value, "to_dict")
    return to_dict()


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
