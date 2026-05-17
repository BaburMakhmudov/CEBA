"""Deterministic runtime scheduling utilities for CEBA Runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from ceba.runtime.context import (
    RuntimeContext,
    ensure_non_empty_string,
    ensure_timezone_aware,
    utc_now,
)
from ceba.runtime.errors import RuntimeSchedulerError
from ceba.runtime.invariants import (
    RuntimeInvariantPolicy,
    RuntimeInvariantReport,
    validate_runtime_invariants,
)


__all__ = [
    "RuntimeTaskKind",
    "RuntimeTaskStatus",
    "RuntimeScheduleStatus",
    "RuntimeSchedulerPolicy",
    "RuntimeTask",
    "RuntimeSchedule",
    "RuntimeScheduleResult",
    "RuntimeScheduler",
    "new_runtime_task_id",
    "new_runtime_schedule_id",
    "create_runtime_task",
    "create_runtime_schedule",
    "build_runtime_scheduler",
]


class RuntimeTaskKind(Enum):
    """Stable categories for runtime tasks."""

    CONTEXT_PREPARE = "CONTEXT_PREPARE"
    INVARIANT_VALIDATE = "INVARIANT_VALIDATE"
    WORLD_MODEL_PREDICT = "WORLD_MODEL_PREDICT"
    EXPECTED_OBSERVATION_REQUIRE = "EXPECTED_OBSERVATION_REQUIRE"
    DELTA_COMPUTE = "DELTA_COMPUTE"
    MEMORY_COORDINATE = "MEMORY_COORDINATE"
    TRANSACTION_COMMIT = "TRANSACTION_COMMIT"
    CHECKPOINT_CREATE = "CHECKPOINT_CREATE"
    ENGINE_STEP = "ENGINE_STEP"
    UNKNOWN = "UNKNOWN"


class RuntimeTaskStatus(Enum):
    """Stable status values for runtime tasks."""

    PENDING = "PENDING"
    READY = "READY"
    BLOCKED = "BLOCKED"
    SCHEDULED = "SCHEDULED"
    SKIPPED = "SKIPPED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class RuntimeScheduleStatus(Enum):
    """Stable status values for runtime schedules."""

    CREATED = "CREATED"
    READY = "READY"
    BLOCKED = "BLOCKED"
    EMPTY = "EMPTY"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class RuntimeSchedulerPolicy:
    """Configuration object controlling deterministic runtime scheduling."""

    require_invariants_before_schedule: bool = True
    allow_empty_schedule: bool = False
    allow_blocked_tasks: bool = False
    max_tasks: int = 128
    invariant_policy: RuntimeInvariantPolicy = field(
        default_factory=RuntimeInvariantPolicy
    )

    def __post_init__(self) -> None:
        for field_name in (
            "require_invariants_before_schedule",
            "allow_empty_schedule",
            "allow_blocked_tasks",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")
        max_tasks = _ensure_positive_int(self.max_tasks, "max_tasks")
        if not isinstance(self.invariant_policy, RuntimeInvariantPolicy):
            raise TypeError("invariant_policy must be RuntimeInvariantPolicy")
        object.__setattr__(self, "max_tasks", max_tasks)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "require_invariants_before_schedule": (
                self.require_invariants_before_schedule
            ),
            "allow_empty_schedule": self.allow_empty_schedule,
            "allow_blocked_tasks": self.allow_blocked_tasks,
            "max_tasks": self.max_tasks,
            "invariant_policy": self.invariant_policy.to_dict(),
        }


@dataclass(frozen=True)
class RuntimeTask:
    """Immutable runtime task descriptor."""

    task_id: UUID
    kind: RuntimeTaskKind
    status: RuntimeTaskStatus
    name: str
    priority: int
    depends_on: tuple[UUID, ...]
    context_id: UUID | None
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, UUID):
            raise TypeError("task_id must be UUID")
        if not isinstance(self.kind, RuntimeTaskKind):
            raise TypeError("kind must be RuntimeTaskKind")
        if not isinstance(self.status, RuntimeTaskStatus):
            raise TypeError("status must be RuntimeTaskStatus")
        name = ensure_non_empty_string(self.name, "name")
        priority = _ensure_non_negative_int(self.priority, "priority")
        depends_on = _ensure_uuid_tuple(self.depends_on, "depends_on")
        if self.context_id is not None and not isinstance(self.context_id, UUID):
            raise TypeError("context_id must be UUID or None")
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        updated_at = ensure_timezone_aware(self.updated_at, "updated_at")
        if updated_at < created_at:
            raise ValueError("updated_at must be >= created_at")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "depends_on", depends_on)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "task_id": str(self.task_id),
            "kind": self.kind.value,
            "status": self.status.value,
            "name": self.name,
            "priority": self.priority,
            "depends_on": tuple(str(task_id) for task_id in self.depends_on),
            "context_id": None if self.context_id is None else str(self.context_id),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    def dependency_count(self) -> int:
        """Return dependency count."""

        return len(self.depends_on)


@dataclass(frozen=True)
class RuntimeSchedule:
    """Immutable deterministic runtime schedule."""

    schedule_id: UUID
    status: RuntimeScheduleStatus
    context: RuntimeContext
    tasks: tuple[RuntimeTask, ...]
    invariant_report: RuntimeInvariantReport | None
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.schedule_id, UUID):
            raise TypeError("schedule_id must be UUID")
        if not isinstance(self.status, RuntimeScheduleStatus):
            raise TypeError("status must be RuntimeScheduleStatus")
        if not isinstance(self.context, RuntimeContext):
            raise TypeError("context must be RuntimeContext")
        tasks = _ensure_tasks(self.tasks)
        _validate_task_dependencies(tasks)
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
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "schedule_id": str(self.schedule_id),
            "status": self.status.value,
            "context": self.context.to_dict(),
            "tasks": tuple(task.to_dict() for task in self.tasks),
            "invariant_report": (
                None
                if self.invariant_report is None
                else self.invariant_report.to_dict()
            ),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    def task_count(self) -> int:
        """Return task count."""

        return len(self.tasks)

    def blocked_task_count(self) -> int:
        """Return blocked task count."""

        return sum(1 for task in self.tasks if task.status == RuntimeTaskStatus.BLOCKED)

    def has_invariant_report(self) -> bool:
        """Return whether this schedule has an invariant report."""

        return self.invariant_report is not None


@dataclass(frozen=True)
class RuntimeScheduleResult:
    """Immutable result returned by scheduler operations."""

    schedule: RuntimeSchedule
    changed: bool
    reason: str
    blocked: bool
    invariant_report: RuntimeInvariantReport | None
    rejected_task_ids: tuple[str, ...]
    rejection_reasons: tuple[str, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, RuntimeSchedule):
            raise TypeError("schedule must be RuntimeSchedule")
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
        rejected_ids = _ensure_string_tuple(self.rejected_task_ids, "rejected_task_ids")
        rejection_reasons = _ensure_string_tuple(
            self.rejection_reasons,
            "rejection_reasons",
        )
        if len(rejected_ids) != len(rejection_reasons):
            raise ValueError("rejected_task_ids and rejection_reasons must match")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "rejected_task_ids", rejected_ids)
        object.__setattr__(self, "rejection_reasons", rejection_reasons)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "schedule": self.schedule.to_dict(),
            "changed": self.changed,
            "reason": self.reason,
            "blocked": self.blocked,
            "invariant_report": (
                None
                if self.invariant_report is None
                else self.invariant_report.to_dict()
            ),
            "rejected_task_ids": self.rejected_task_ids,
            "rejection_reasons": self.rejection_reasons,
            "metadata": dict(self.metadata),
        }

    def rejected_count(self) -> int:
        """Return rejected task count."""

        return len(self.rejected_task_ids)

    def is_ready(self) -> bool:
        """Return whether the schedule is ready."""

        return not self.blocked and self.schedule.status == RuntimeScheduleStatus.READY


class RuntimeScheduler:
    """Deterministic scheduler for CEBA runtime tasks."""

    def __init__(self, policy: RuntimeSchedulerPolicy | None = None) -> None:
        self._policy = RuntimeSchedulerPolicy() if policy is None else policy
        if not isinstance(self._policy, RuntimeSchedulerPolicy):
            raise TypeError("policy must be RuntimeSchedulerPolicy")

    def build_schedule(
        self,
        *,
        context: RuntimeContext,
        tasks: tuple[RuntimeTask, ...] | list[RuntimeTask],
        metadata: dict[str, object] | None = None,
    ) -> RuntimeScheduleResult:
        """Build a deterministic runtime schedule."""

        if not isinstance(context, RuntimeContext):
            raise TypeError("context must be RuntimeContext")
        task_values = _ensure_tasks(tasks)
        if not task_values and not self._policy.allow_empty_schedule:
            raise RuntimeSchedulerError("runtime schedule has no tasks")

        invariant_report: RuntimeInvariantReport | None = None
        blocked_by_invariants = False
        if self._policy.require_invariants_before_schedule:
            invariant_report = validate_runtime_invariants(
                context=context,
                policy=self._policy.invariant_policy,
            )
            blocked_by_invariants = invariant_report.has_errors()

        accepted: list[RuntimeTask] = []
        rejected_ids: list[str] = []
        rejection_reasons: list[str] = []
        for task in task_values:
            if len(accepted) >= self._policy.max_tasks:
                rejected_ids.append(str(task.task_id))
                rejection_reasons.append("runtime schedule task limit exceeded")
            elif task.status == RuntimeTaskStatus.INVALID:
                rejected_ids.append(str(task.task_id))
                rejection_reasons.append("runtime task invalid")
            elif (
                task.status == RuntimeTaskStatus.BLOCKED
                and not self._policy.allow_blocked_tasks
            ):
                rejected_ids.append(str(task.task_id))
                rejection_reasons.append("runtime task blocked")
            else:
                accepted.append(task)

        accepted, dependency_rejected_ids, dependency_reasons = (
            _reject_missing_dependencies(tuple(accepted))
        )
        rejected_ids.extend(dependency_rejected_ids)
        rejection_reasons.extend(dependency_reasons)
        ordered_tasks = _order_tasks(accepted)
        status = RuntimeScheduleStatus.READY
        reason = "runtime schedule ready"
        if not ordered_tasks:
            status = RuntimeScheduleStatus.EMPTY
            reason = "runtime schedule empty"
        if blocked_by_invariants:
            status = RuntimeScheduleStatus.BLOCKED
            reason = "runtime schedule blocked by invariant errors"

        required_metadata: dict[str, object] = {
            "source": "runtime_scheduler",
            "operation": "build_schedule",
            "context_id": str(context.context_id),
            "policy": self._policy.to_dict(),
        }
        schedule = create_runtime_schedule(
            context=context,
            status=status,
            tasks=ordered_tasks,
            invariant_report=invariant_report,
            metadata=_merge_metadata(required_metadata, metadata),
        )
        schedule = _replace_schedule(
            schedule=schedule,
            status=schedule.status,
            tasks=schedule.tasks,
            invariant_report=schedule.invariant_report,
            metadata=_force_required_metadata(schedule.metadata, required_metadata),
        )
        return _result(
            schedule=schedule,
            changed=True,
            reason=reason,
            blocked=blocked_by_invariants,
            invariant_report=invariant_report,
            rejected_task_ids=tuple(rejected_ids),
            rejection_reasons=tuple(rejection_reasons),
        )

    def mark_task_scheduled(
        self,
        *,
        schedule: RuntimeSchedule,
        task_id: UUID,
        metadata: dict[str, object] | None = None,
    ) -> RuntimeScheduleResult:
        """Mark a task as scheduled."""

        return self._mark_task(
            schedule=schedule,
            task_id=task_id,
            status=RuntimeTaskStatus.SCHEDULED,
            reason="runtime task scheduled",
            blocked=False,
            metadata=metadata,
        )

    def mark_task_completed(
        self,
        *,
        schedule: RuntimeSchedule,
        task_id: UUID,
        metadata: dict[str, object] | None = None,
    ) -> RuntimeScheduleResult:
        """Mark a task as completed."""

        return self._mark_task(
            schedule=schedule,
            task_id=task_id,
            status=RuntimeTaskStatus.COMPLETED,
            reason="runtime task completed",
            blocked=False,
            metadata=metadata,
        )

    def mark_task_failed(
        self,
        *,
        schedule: RuntimeSchedule,
        task_id: UUID,
        reason: str,
        metadata: dict[str, object] | None = None,
    ) -> RuntimeScheduleResult:
        """Mark a task as failed."""

        failure_reason = ensure_non_empty_string(reason, "reason")
        return self._mark_task(
            schedule=schedule,
            task_id=task_id,
            status=RuntimeTaskStatus.FAILED,
            reason="runtime task failed",
            blocked=True,
            schedule_status=RuntimeScheduleStatus.BLOCKED,
            metadata=_merge_metadata(
                {"failure_reason": failure_reason},
                metadata,
            ),
        )

    def _mark_task(
        self,
        *,
        schedule: RuntimeSchedule,
        task_id: UUID,
        status: RuntimeTaskStatus,
        reason: str,
        blocked: bool,
        metadata: dict[str, object] | None,
        schedule_status: RuntimeScheduleStatus | None = None,
    ) -> RuntimeScheduleResult:
        if not isinstance(schedule, RuntimeSchedule):
            raise TypeError("schedule must be RuntimeSchedule")
        if not isinstance(task_id, UUID):
            raise TypeError("task_id must be UUID")
        task = _find_task(schedule.tasks, task_id)
        required_metadata: dict[str, object] = {
            "source": "runtime_scheduler",
            "operation": (
                "mark_task_failed"
                if status == RuntimeTaskStatus.FAILED
                else (
                    "mark_task_completed"
                    if status == RuntimeTaskStatus.COMPLETED
                    else "mark_task_scheduled"
                )
            ),
        }
        updated_task = _update_task(
            task=task,
            status=status,
            metadata=_manager_metadata(
                existing_metadata=task.metadata,
                required_metadata=required_metadata,
                caller_metadata=metadata,
            ),
        )
        updated_tasks = tuple(
            updated_task if item.task_id == task_id else item
            for item in schedule.tasks
        )
        updated_schedule = _replace_schedule(
            schedule=schedule,
            status=schedule.status if schedule_status is None else schedule_status,
            tasks=updated_tasks,
            invariant_report=schedule.invariant_report,
            metadata=schedule.metadata,
        )
        return _result(
            schedule=updated_schedule,
            changed=True,
            reason=reason,
            blocked=blocked,
            invariant_report=schedule.invariant_report,
            rejected_task_ids=(),
            rejection_reasons=(),
        )


def new_runtime_task_id() -> UUID:
    """Return a new runtime task identifier."""

    return uuid4()


def new_runtime_schedule_id() -> UUID:
    """Return a new runtime schedule identifier."""

    return uuid4()


def create_runtime_task(
    *,
    name: str,
    kind: RuntimeTaskKind = RuntimeTaskKind.UNKNOWN,
    status: RuntimeTaskStatus = RuntimeTaskStatus.PENDING,
    priority: int = 0,
    depends_on: tuple[UUID, ...] | list[UUID] = (),
    context_id: UUID | None = None,
    metadata: dict[str, object] | None = None,
) -> RuntimeTask:
    """Create a runtime task."""

    now = utc_now()
    required_metadata: dict[str, object] = {
        "source": "runtime_task",
        "task_kind": kind.value if isinstance(kind, RuntimeTaskKind) else kind,
        "task_status": status.value if isinstance(status, RuntimeTaskStatus) else status,
    }
    return RuntimeTask(
        task_id=new_runtime_task_id(),
        kind=kind,
        status=status,
        name=name,
        priority=priority,
        depends_on=depends_on,
        context_id=context_id,
        created_at=now,
        updated_at=now,
        metadata=_merge_metadata(required_metadata, metadata),
    )


def create_runtime_schedule(
    *,
    context: RuntimeContext,
    status: RuntimeScheduleStatus = RuntimeScheduleStatus.CREATED,
    tasks: tuple[RuntimeTask, ...] | list[RuntimeTask] = (),
    invariant_report: RuntimeInvariantReport | None = None,
    metadata: dict[str, object] | None = None,
) -> RuntimeSchedule:
    """Create a runtime schedule."""

    if not isinstance(context, RuntimeContext):
        raise TypeError("context must be RuntimeContext")
    tasks_value = _ensure_tasks(tasks)
    now = utc_now()
    required_metadata: dict[str, object] = {
        "source": "runtime_schedule",
        "context_id": str(context.context_id),
        "task_count": len(tasks_value),
        "has_invariant_report": invariant_report is not None,
    }
    return RuntimeSchedule(
        schedule_id=new_runtime_schedule_id(),
        status=status,
        context=context,
        tasks=tasks_value,
        invariant_report=invariant_report,
        created_at=now,
        updated_at=now,
        metadata=_merge_metadata(required_metadata, metadata),
    )


def build_runtime_scheduler(
    *,
    policy: RuntimeSchedulerPolicy | None = None,
) -> RuntimeScheduler:
    """Build a runtime scheduler."""

    return RuntimeScheduler(policy=policy)


def _ensure_tasks(
    values: tuple[RuntimeTask, ...] | list[RuntimeTask],
) -> tuple[RuntimeTask, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("tasks must be tuple or list")
    tasks = tuple(values)
    for task in tasks:
        if not isinstance(task, RuntimeTask):
            raise TypeError("tasks must contain RuntimeTask")
    task_ids = tuple(task.task_id for task in tasks)
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task_id values must be unique")
    return tasks


def _validate_task_dependencies(tasks: tuple[RuntimeTask, ...]) -> None:
    task_ids = {task.task_id for task in tasks}
    for task in tasks:
        if task.task_id in task.depends_on:
            raise ValueError("task may not depend on itself")
        for dependency_id in task.depends_on:
            if dependency_id not in task_ids:
                raise ValueError("dependency id must exist in tasks")


def _reject_missing_dependencies(
    tasks: tuple[RuntimeTask, ...],
) -> tuple[tuple[RuntimeTask, ...], tuple[str, ...], tuple[str, ...]]:
    accepted = list(tasks)
    rejected_ids: list[str] = []
    rejection_reasons: list[str] = []
    changed = True
    while changed:
        changed = False
        accepted_ids = {task.task_id for task in accepted}
        next_accepted: list[RuntimeTask] = []
        for task in accepted:
            if all(dependency_id in accepted_ids for dependency_id in task.depends_on):
                next_accepted.append(task)
            else:
                rejected_ids.append(str(task.task_id))
                rejection_reasons.append("runtime task dependency missing")
                changed = True
        accepted = next_accepted
    return tuple(accepted), tuple(rejected_ids), tuple(rejection_reasons)


def _order_tasks(tasks: tuple[RuntimeTask, ...]) -> tuple[RuntimeTask, ...]:
    remaining = {task.task_id: task for task in tasks}
    completed: set[UUID] = set()
    ordered: list[RuntimeTask] = []
    while remaining:
        ready = tuple(
            task
            for task in remaining.values()
            if all(dependency_id in completed for dependency_id in task.depends_on)
        )
        if not ready:
            raise RuntimeSchedulerError("runtime task dependency cycle detected")
        next_task = sorted(
            ready,
            key=lambda task: (
                -task.priority,
                task.created_at.timestamp(),
                str(task.task_id),
            ),
        )[0]
        ordered.append(next_task)
        completed.add(next_task.task_id)
        del remaining[next_task.task_id]
    return tuple(ordered)


def _find_task(tasks: tuple[RuntimeTask, ...], task_id: UUID) -> RuntimeTask:
    for task in tasks:
        if task.task_id == task_id:
            return task
    raise RuntimeSchedulerError(
        "runtime task missing",
        context={"task_id": str(task_id)},
    )


def _update_task(
    *,
    task: RuntimeTask,
    status: RuntimeTaskStatus,
    metadata: dict[str, object],
) -> RuntimeTask:
    return RuntimeTask(
        task_id=task.task_id,
        kind=task.kind,
        status=status,
        name=task.name,
        priority=task.priority,
        depends_on=task.depends_on,
        context_id=task.context_id,
        created_at=task.created_at,
        updated_at=utc_now(),
        metadata=metadata,
    )


def _replace_schedule(
    *,
    schedule: RuntimeSchedule,
    status: RuntimeScheduleStatus,
    tasks: tuple[RuntimeTask, ...],
    invariant_report: RuntimeInvariantReport | None,
    metadata: dict[str, object],
) -> RuntimeSchedule:
    return RuntimeSchedule(
        schedule_id=schedule.schedule_id,
        status=status,
        context=schedule.context,
        tasks=tasks,
        invariant_report=invariant_report,
        created_at=schedule.created_at,
        updated_at=utc_now(),
        metadata=metadata,
    )


def _result(
    *,
    schedule: RuntimeSchedule,
    changed: bool,
    reason: str,
    blocked: bool,
    invariant_report: RuntimeInvariantReport | None,
    rejected_task_ids: tuple[str, ...],
    rejection_reasons: tuple[str, ...],
    metadata: dict[str, object] | None = None,
) -> RuntimeScheduleResult:
    return RuntimeScheduleResult(
        schedule=schedule,
        changed=changed,
        reason=reason,
        blocked=blocked,
        invariant_report=invariant_report,
        rejected_task_ids=rejected_task_ids,
        rejection_reasons=rejection_reasons,
        metadata={} if metadata is None else metadata,
    )


def _ensure_uuid_tuple(
    values: tuple[UUID, ...] | list[UUID],
    field_name: str,
) -> tuple[UUID, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple or list")
    ids = tuple(values)
    for value in ids:
        if not isinstance(value, UUID):
            raise TypeError(f"{field_name} must contain UUID")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{field_name} values must be unique")
    return ids


def _ensure_string_tuple(
    values: tuple[str, ...] | list[str],
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple or list")
    return tuple(ensure_non_empty_string(value, field_name) for value in values)


def _ensure_positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _ensure_non_negative_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
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
