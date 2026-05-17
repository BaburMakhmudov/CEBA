"""Deterministic runtime engine coordination utilities for CEBA Runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from ceba.runtime.context import (
    RuntimeContext,
    RuntimeContextBuilder,
    RuntimeContextStatus,
    RuntimeInputRef,
    RuntimeInvariantFlag,
    create_runtime_context,
    ensure_non_empty_string,
    ensure_timezone_aware,
    utc_now,
)
from ceba.runtime.cycle import (
    RuntimeCycle,
    RuntimeCyclePolicy,
    RuntimeCycleResult,
    RuntimeCycleStatus,
    build_runtime_cycle_coordinator,
)
from ceba.runtime.errors import RuntimeEngineError
from ceba.runtime.invariants import RuntimeInvariantReport
from ceba.runtime.scheduler import RuntimeTask


__all__ = [
    "RuntimeEngineStatus",
    "RuntimeEnginePolicy",
    "RuntimeEngineState",
    "RuntimeEngineResult",
    "RuntimeEngine",
    "new_runtime_engine_id",
    "create_runtime_engine_state",
    "build_runtime_engine",
]


class RuntimeEngineStatus(Enum):
    """Stable status values for runtime engines."""

    CREATED = "CREATED"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    STOPPED = "STOPPED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class RuntimeEnginePolicy:
    """Configuration object controlling runtime engine coordination."""

    cycle_policy: RuntimeCyclePolicy = field(default_factory=RuntimeCyclePolicy)
    require_context_before_run: bool = True
    allow_rerun_after_terminal_cycle: bool = False
    stop_on_blocked_cycle: bool = True
    stop_on_failed_cycle: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.cycle_policy, RuntimeCyclePolicy):
            raise TypeError("cycle_policy must be RuntimeCyclePolicy")
        for field_name in (
            "require_context_before_run",
            "allow_rerun_after_terminal_cycle",
            "stop_on_blocked_cycle",
            "stop_on_failed_cycle",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "cycle_policy": self.cycle_policy.to_dict(),
            "require_context_before_run": self.require_context_before_run,
            "allow_rerun_after_terminal_cycle": self.allow_rerun_after_terminal_cycle,
            "stop_on_blocked_cycle": self.stop_on_blocked_cycle,
            "stop_on_failed_cycle": self.stop_on_failed_cycle,
        }


@dataclass(frozen=True)
class RuntimeEngineState:
    """Immutable runtime engine state descriptor."""

    engine_id: UUID
    status: RuntimeEngineStatus
    context: RuntimeContext | None
    cycle: RuntimeCycle | None
    last_cycle_result: RuntimeCycleResult | None
    started_at: datetime
    updated_at: datetime
    stopped_at: datetime | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.engine_id, UUID):
            raise TypeError("engine_id must be UUID")
        if not isinstance(self.status, RuntimeEngineStatus):
            raise TypeError("status must be RuntimeEngineStatus")
        if self.context is not None and not isinstance(self.context, RuntimeContext):
            raise TypeError("context must be RuntimeContext or None")
        if self.cycle is not None and not isinstance(self.cycle, RuntimeCycle):
            raise TypeError("cycle must be RuntimeCycle or None")
        if self.context is not None and self.cycle is not None:
            if self.cycle.context.context_id != self.context.context_id:
                raise ValueError("cycle context_id must match context")
        if self.last_cycle_result is not None and not isinstance(
            self.last_cycle_result,
            RuntimeCycleResult,
        ):
            raise TypeError("last_cycle_result must be RuntimeCycleResult or None")
        started_at = ensure_timezone_aware(self.started_at, "started_at")
        updated_at = ensure_timezone_aware(self.updated_at, "updated_at")
        if updated_at < started_at:
            raise ValueError("updated_at must be >= started_at")
        stopped_at = self.stopped_at
        if stopped_at is not None:
            stopped_at = ensure_timezone_aware(stopped_at, "stopped_at")
            if stopped_at < started_at:
                raise ValueError("stopped_at must be >= started_at")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "stopped_at", stopped_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "engine_id": str(self.engine_id),
            "status": self.status.value,
            "context": None if self.context is None else self.context.to_dict(),
            "cycle": None if self.cycle is None else self.cycle.to_dict(),
            "last_cycle_result": (
                None
                if self.last_cycle_result is None
                else self.last_cycle_result.to_dict()
            ),
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "stopped_at": None if self.stopped_at is None else self.stopped_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    def has_context(self) -> bool:
        """Return whether this engine state has a context."""

        return self.context is not None

    def has_cycle(self) -> bool:
        """Return whether this engine state has a cycle."""

        return self.cycle is not None

    def has_last_cycle_result(self) -> bool:
        """Return whether this engine state has a last cycle result."""

        return self.last_cycle_result is not None

    def is_terminal(self) -> bool:
        """Return whether this engine state is terminal."""

        return self.status in {
            RuntimeEngineStatus.COMPLETED,
            RuntimeEngineStatus.FAILED,
            RuntimeEngineStatus.BLOCKED,
            RuntimeEngineStatus.STOPPED,
            RuntimeEngineStatus.INVALID,
        }


@dataclass(frozen=True)
class RuntimeEngineResult:
    """Immutable result returned by runtime engine operations."""

    state: RuntimeEngineState
    changed: bool
    reason: str
    blocked: bool
    cycle_result: RuntimeCycleResult | None = None
    invariant_report: RuntimeInvariantReport | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.state, RuntimeEngineState):
            raise TypeError("state must be RuntimeEngineState")
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        reason = ensure_non_empty_string(self.reason, "reason")
        if not isinstance(self.blocked, bool):
            raise TypeError("blocked must be bool")
        if self.cycle_result is not None and not isinstance(
            self.cycle_result,
            RuntimeCycleResult,
        ):
            raise TypeError("cycle_result must be RuntimeCycleResult or None")
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
            "state": self.state.to_dict(),
            "changed": self.changed,
            "reason": self.reason,
            "blocked": self.blocked,
            "cycle_result": (
                None if self.cycle_result is None else self.cycle_result.to_dict()
            ),
            "invariant_report": (
                None
                if self.invariant_report is None
                else self.invariant_report.to_dict()
            ),
            "metadata": dict(self.metadata),
        }

    def is_success(self) -> bool:
        """Return whether engine coordination succeeded."""

        return not self.blocked and self.state.status == RuntimeEngineStatus.COMPLETED


class RuntimeEngine:
    """Deterministic top-level coordinator for prepared runtime cycles."""

    def __init__(
        self,
        *,
        state: RuntimeEngineState | None = None,
        policy: RuntimeEnginePolicy | None = None,
    ) -> None:
        self._policy = RuntimeEnginePolicy() if policy is None else policy
        if not isinstance(self._policy, RuntimeEnginePolicy):
            raise TypeError("policy must be RuntimeEnginePolicy")
        self._state = (
            create_runtime_engine_state(status=RuntimeEngineStatus.CREATED)
            if state is None
            else state
        )
        if not isinstance(self._state, RuntimeEngineState):
            raise TypeError("state must be RuntimeEngineState")

    def state(self) -> RuntimeEngineState:
        """Return current engine state."""

        return self._state

    def status(self) -> RuntimeEngineStatus:
        """Return current engine status."""

        return self._state.status

    def context(self) -> RuntimeContext | None:
        """Return current runtime context."""

        return self._state.context

    def cycle(self) -> RuntimeCycle | None:
        """Return current runtime cycle."""

        return self._state.cycle

    def start(
        self,
        *,
        metadata: dict[str, object] | None = None,
    ) -> RuntimeEngineResult:
        """Start the runtime engine."""

        required_metadata = self._operation_metadata("start")
        updated = _replace_engine_state(
            state=self._state,
            status=RuntimeEngineStatus.READY,
            context=self._state.context,
            cycle=self._state.cycle,
            last_cycle_result=self._state.last_cycle_result,
            stopped_at=None,
            metadata=_merge_runtime_metadata(
                self._state.metadata,
                required_metadata,
                metadata,
            ),
        )
        self._state = updated
        return _engine_result(
            state=updated,
            changed=True,
            reason="runtime engine started",
            blocked=False,
        )

    def attach_context(
        self,
        *,
        context: RuntimeContext,
        metadata: dict[str, object] | None = None,
    ) -> RuntimeEngineResult:
        """Attach a runtime context to the engine."""

        if not isinstance(context, RuntimeContext):
            raise TypeError("context must be RuntimeContext")
        ready_context = RuntimeContextBuilder().mark_ready(context)
        preserved_cycle = self._state.cycle
        if preserved_cycle is not None:
            if preserved_cycle.context.context_id != ready_context.context_id:
                preserved_cycle = None
        required_metadata = self._operation_metadata("attach_context")
        updated = _replace_engine_state(
            state=self._state,
            status=RuntimeEngineStatus.READY,
            context=ready_context,
            cycle=preserved_cycle,
            last_cycle_result=self._state.last_cycle_result,
            stopped_at=self._state.stopped_at,
            metadata=_merge_runtime_metadata(
                self._state.metadata,
                required_metadata,
                metadata,
            ),
        )
        self._state = updated
        return _engine_result(
            state=updated,
            changed=True,
            reason="runtime engine context attached",
            blocked=False,
        )

    def build_context(
        self,
        *,
        input_refs: tuple[RuntimeInputRef, ...] | list[RuntimeInputRef] = (),
        invariant_flags: tuple[RuntimeInvariantFlag, ...] | list[RuntimeInvariantFlag] = (),
        metadata: dict[str, object] | None = None,
    ) -> RuntimeEngineResult:
        """Build and attach a runtime context."""

        required_metadata = self._operation_metadata("build_context")
        context = create_runtime_context(
            status=RuntimeContextStatus.READY,
            input_refs=input_refs,
            invariant_flags=invariant_flags,
            metadata=_merge_metadata(required_metadata, metadata),
        )
        ready_context = RuntimeContextBuilder().mark_ready(context)
        preserved_cycle = self._state.cycle
        if preserved_cycle is not None:
            if preserved_cycle.context.context_id != ready_context.context_id:
                preserved_cycle = None
        updated = _replace_engine_state(
            state=self._state,
            status=RuntimeEngineStatus.READY,
            context=ready_context,
            cycle=preserved_cycle,
            last_cycle_result=self._state.last_cycle_result,
            stopped_at=self._state.stopped_at,
            metadata=_merge_runtime_metadata(
                self._state.metadata,
                required_metadata,
                metadata,
            ),
        )
        self._state = updated
        return _engine_result(
            state=updated,
            changed=True,
            reason="runtime engine context built",
            blocked=False,
        )

    def run_prepared_cycle(
        self,
        *,
        tasks: tuple[RuntimeTask, ...] | list[RuntimeTask],
        metadata: dict[str, object] | None = None,
    ) -> RuntimeEngineResult:
        """Run a prepared runtime cycle through the cycle coordinator."""

        if self._policy.require_context_before_run and self._state.context is None:
            raise RuntimeEngineError("runtime context is required before run")
        if self._state.context is None:
            raise RuntimeEngineError("runtime context is missing")
        if (
            self._state.cycle is not None
            and self._state.cycle.is_terminal()
            and not self._policy.allow_rerun_after_terminal_cycle
        ):
            raise RuntimeEngineError("terminal runtime cycle cannot be rerun")
        coordinator = build_runtime_cycle_coordinator(policy=self._policy.cycle_policy)
        cycle_result = coordinator.run_prepared(
            context=self._state.context,
            tasks=tasks,
            metadata=metadata,
        )
        engine_status = _status_from_cycle_result(cycle_result)
        stopped_at: datetime | None = self._state.stopped_at
        if self._policy.stop_on_blocked_cycle and cycle_result.blocked:
            stopped_at = utc_now()
        if (
            self._policy.stop_on_failed_cycle
            and cycle_result.cycle.status == RuntimeCycleStatus.FAILED
        ):
            stopped_at = utc_now()
        required_metadata = self._operation_metadata("run_prepared_cycle")
        updated = _replace_engine_state(
            state=self._state,
            status=engine_status,
            context=cycle_result.cycle.context,
            cycle=cycle_result.cycle,
            last_cycle_result=cycle_result,
            stopped_at=stopped_at,
            metadata=_merge_runtime_metadata(
                self._state.metadata,
                required_metadata,
                metadata,
            ),
        )
        self._state = updated
        return _engine_result(
            state=updated,
            changed=True,
            reason=cycle_result.reason,
            blocked=cycle_result.blocked,
            cycle_result=cycle_result,
            invariant_report=cycle_result.invariant_report,
        )

    def fail(
        self,
        *,
        reason: str,
        metadata: dict[str, object] | None = None,
    ) -> RuntimeEngineResult:
        """Fail the runtime engine."""

        failure_reason = ensure_non_empty_string(reason, "reason")
        context = self._state.context
        if context is not None:
            context = RuntimeContextBuilder().mark_failed(
                context,
                reason=failure_reason,
            )
        stopped_at = utc_now()
        required_metadata = self._operation_metadata("fail")
        required_metadata["failure_reason"] = failure_reason
        updated = _replace_engine_state(
            state=self._state,
            status=RuntimeEngineStatus.FAILED,
            context=context,
            cycle=self._state.cycle,
            last_cycle_result=self._state.last_cycle_result,
            stopped_at=stopped_at,
            metadata=_merge_runtime_metadata(
                self._state.metadata,
                required_metadata,
                metadata,
            ),
        )
        self._state = updated
        return _engine_result(
            state=updated,
            changed=True,
            reason="runtime engine failed",
            blocked=True,
        )

    def stop(
        self,
        *,
        metadata: dict[str, object] | None = None,
    ) -> RuntimeEngineResult:
        """Stop the runtime engine."""

        stopped_at = utc_now()
        required_metadata = self._operation_metadata("stop")
        updated = _replace_engine_state(
            state=self._state,
            status=RuntimeEngineStatus.STOPPED,
            context=self._state.context,
            cycle=self._state.cycle,
            last_cycle_result=self._state.last_cycle_result,
            stopped_at=stopped_at,
            metadata=_merge_runtime_metadata(
                self._state.metadata,
                required_metadata,
                metadata,
            ),
        )
        self._state = updated
        return _engine_result(
            state=updated,
            changed=True,
            reason="runtime engine stopped",
            blocked=False,
        )

    def reset(
        self,
        *,
        metadata: dict[str, object] | None = None,
    ) -> RuntimeEngineResult:
        """Reset the runtime engine."""

        required_metadata = self._operation_metadata("reset")
        updated = RuntimeEngineState(
            engine_id=self._state.engine_id,
            status=RuntimeEngineStatus.CREATED,
            context=None,
            cycle=None,
            last_cycle_result=None,
            started_at=self._state.started_at,
            updated_at=utc_now(),
            stopped_at=None,
            metadata=_merge_runtime_metadata(
                self._state.metadata,
                required_metadata,
                metadata,
            ),
        )
        self._state = updated
        return _engine_result(
            state=updated,
            changed=True,
            reason="runtime engine reset",
            blocked=False,
        )

    def _operation_metadata(self, operation: str) -> dict[str, object]:
        return {
            "source": "runtime_engine",
            "operation": operation,
            "policy": self._policy.to_dict(),
        }


def new_runtime_engine_id() -> UUID:
    """Return a new runtime engine identifier."""

    return uuid4()


def create_runtime_engine_state(
    *,
    status: RuntimeEngineStatus = RuntimeEngineStatus.CREATED,
    context: RuntimeContext | None = None,
    cycle: RuntimeCycle | None = None,
    last_cycle_result: RuntimeCycleResult | None = None,
    stopped_at: datetime | None = None,
    metadata: dict[str, object] | None = None,
) -> RuntimeEngineState:
    """Create a runtime engine state descriptor."""

    now = utc_now()
    required_metadata: dict[str, object] = {
        "source": "runtime_engine_state",
        "engine_status": status.value if isinstance(status, RuntimeEngineStatus) else status,
        "has_context": context is not None,
        "has_cycle": cycle is not None,
        "has_last_cycle_result": last_cycle_result is not None,
    }
    return RuntimeEngineState(
        engine_id=new_runtime_engine_id(),
        status=status,
        context=context,
        cycle=cycle,
        last_cycle_result=last_cycle_result,
        started_at=now,
        updated_at=now,
        stopped_at=stopped_at,
        metadata=_merge_metadata(required_metadata, metadata),
    )


def build_runtime_engine(
    *,
    state: RuntimeEngineState | None = None,
    policy: RuntimeEnginePolicy | None = None,
) -> RuntimeEngine:
    """Build a runtime engine."""

    return RuntimeEngine(state=state, policy=policy)


def _replace_engine_state(
    *,
    state: RuntimeEngineState,
    status: RuntimeEngineStatus,
    context: RuntimeContext | None,
    cycle: RuntimeCycle | None,
    last_cycle_result: RuntimeCycleResult | None,
    stopped_at: datetime | None,
    metadata: dict[str, object],
) -> RuntimeEngineState:
    return RuntimeEngineState(
        engine_id=state.engine_id,
        status=status,
        context=context,
        cycle=cycle,
        last_cycle_result=last_cycle_result,
        started_at=state.started_at,
        updated_at=utc_now(),
        stopped_at=stopped_at,
        metadata=metadata,
    )


def _status_from_cycle_result(cycle_result: RuntimeCycleResult) -> RuntimeEngineStatus:
    if not isinstance(cycle_result, RuntimeCycleResult):
        raise TypeError("cycle_result must be RuntimeCycleResult")
    if cycle_result.blocked:
        return RuntimeEngineStatus.BLOCKED
    if cycle_result.cycle.status == RuntimeCycleStatus.COMPLETED:
        return RuntimeEngineStatus.COMPLETED
    if cycle_result.cycle.status == RuntimeCycleStatus.FAILED:
        return RuntimeEngineStatus.FAILED
    if cycle_result.cycle.status == RuntimeCycleStatus.BLOCKED:
        return RuntimeEngineStatus.BLOCKED
    return RuntimeEngineStatus.RUNNING


def _engine_result(
    *,
    state: RuntimeEngineState,
    changed: bool,
    reason: str,
    blocked: bool,
    cycle_result: RuntimeCycleResult | None = None,
    invariant_report: RuntimeInvariantReport | None = None,
    metadata: dict[str, object] | None = None,
) -> RuntimeEngineResult:
    return RuntimeEngineResult(
        state=state,
        changed=changed,
        reason=reason,
        blocked=blocked,
        cycle_result=cycle_result,
        invariant_report=invariant_report,
        metadata={} if metadata is None else metadata,
    )


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
