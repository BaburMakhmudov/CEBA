"""Deterministic runtime context structures for CEBA Runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from ceba.memory import MemoryOrchestratorState
from ceba.runtime.errors import RuntimeContextError
from ceba.world_model import (
    ExpectedObservation,
    PredictionInput,
    PredictionResult,
    WorldModelSnapshot,
)


__all__ = [
    "RuntimeContextKind",
    "RuntimeContextStatus",
    "RuntimeInvariantFlag",
    "RuntimeInputRef",
    "RuntimeContext",
    "RuntimeContextUpdate",
    "RuntimeContextBuilder",
    "utc_now",
    "new_runtime_context_id",
    "new_runtime_input_ref_id",
    "ensure_non_empty_string",
    "ensure_optional_non_empty_string",
    "ensure_timezone_aware",
    "create_runtime_input_ref",
    "create_runtime_context",
    "build_runtime_context",
]


class RuntimeContextKind(Enum):
    """Stable categories for runtime contexts."""

    CYCLE = "CYCLE"
    ENGINE = "ENGINE"
    SCHEDULER = "SCHEDULER"
    TRANSACTION = "TRANSACTION"
    CHECKPOINT = "CHECKPOINT"
    VALIDATION = "VALIDATION"
    UNKNOWN = "UNKNOWN"


class RuntimeContextStatus(Enum):
    """Stable status values for runtime contexts."""

    CREATED = "CREATED"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class RuntimeInvariantFlag(Enum):
    """Stable architectural invariant flags carried by runtime context."""

    WORLD_MODEL_BEFORE_DELTA = "WORLD_MODEL_BEFORE_DELTA"
    DELTA_ONLY_WRITER = "DELTA_ONLY_WRITER"
    M_T_NOT_PERSISTED = "M_T_NOT_PERSISTED"
    VECTOR_DB_NOT_MEMORY = "VECTOR_DB_NOT_MEMORY"
    STM_NOT_DATABASE = "STM_NOT_DATABASE"
    TEMPERATURE_NOT_MEMORY = "TEMPERATURE_NOT_MEMORY"


@dataclass(frozen=True)
class RuntimeInputRef:
    """Lightweight reference to an upstream runtime object."""

    input_ref_id: UUID
    source: str
    object_ref: str
    object_kind: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.input_ref_id, UUID):
            raise TypeError("input_ref_id must be UUID")
        source = ensure_non_empty_string(self.source, "source")
        object_ref = ensure_non_empty_string(self.object_ref, "object_ref")
        object_kind = ensure_non_empty_string(self.object_kind, "object_kind")
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "object_ref", object_ref)
        object.__setattr__(self, "object_kind", object_kind)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "input_ref_id": str(self.input_ref_id),
            "source": self.source,
            "object_ref": self.object_ref,
            "object_kind": self.object_kind,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RuntimeContext:
    """Immutable runtime execution context for one CEBA operation."""

    context_id: UUID
    kind: RuntimeContextKind
    status: RuntimeContextStatus
    input_refs: tuple[RuntimeInputRef, ...]
    memory_state: MemoryOrchestratorState | None
    world_model_snapshot: WorldModelSnapshot | None
    prediction_input: PredictionInput | None
    prediction_result: PredictionResult | None
    expected_observation: ExpectedObservation | None
    invariant_flags: tuple[RuntimeInvariantFlag, ...]
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.context_id, UUID):
            raise TypeError("context_id must be UUID")
        if not isinstance(self.kind, RuntimeContextKind):
            raise TypeError("kind must be RuntimeContextKind")
        if not isinstance(self.status, RuntimeContextStatus):
            raise TypeError("status must be RuntimeContextStatus")
        input_refs = _ensure_input_refs(self.input_refs)
        _ensure_optional_type(
            self.memory_state,
            MemoryOrchestratorState,
            "memory_state",
        )
        _ensure_optional_type(
            self.world_model_snapshot,
            WorldModelSnapshot,
            "world_model_snapshot",
        )
        _ensure_optional_type(
            self.prediction_input,
            PredictionInput,
            "prediction_input",
        )
        _ensure_optional_type(
            self.prediction_result,
            PredictionResult,
            "prediction_result",
        )
        _ensure_optional_type(
            self.expected_observation,
            ExpectedObservation,
            "expected_observation",
        )
        _validate_expected_observation_link(
            self.prediction_result,
            self.expected_observation,
        )
        invariant_flags = _ensure_invariant_flags(self.invariant_flags)
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        updated_at = ensure_timezone_aware(self.updated_at, "updated_at")
        if updated_at < created_at:
            raise ValueError("updated_at must be >= created_at")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "input_refs", input_refs)
        object.__setattr__(self, "invariant_flags", invariant_flags)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "context_id": str(self.context_id),
            "kind": self.kind.value,
            "status": self.status.value,
            "input_refs": tuple(item.to_dict() for item in self.input_refs),
            "memory_state": _to_optional_dict(self.memory_state),
            "world_model_snapshot": _to_optional_dict(self.world_model_snapshot),
            "prediction_input": _to_optional_dict(self.prediction_input),
            "prediction_result": _to_optional_dict(self.prediction_result),
            "expected_observation": _to_optional_dict(self.expected_observation),
            "invariant_flags": tuple(flag.value for flag in self.invariant_flags),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    def input_ref_count(self) -> int:
        """Return input reference count."""

        return len(self.input_refs)

    def has_world_model_prediction(self) -> bool:
        """Return whether this context has prediction input and result."""

        return self.prediction_input is not None and self.prediction_result is not None

    def has_expected_observation(self) -> bool:
        """Return whether this context has an expected observation."""

        return self.expected_observation is not None

    def has_invariant_flag(self, flag: RuntimeInvariantFlag) -> bool:
        """Return whether the context carries an invariant flag."""

        if not isinstance(flag, RuntimeInvariantFlag):
            raise TypeError("flag must be RuntimeInvariantFlag")
        return flag in self.invariant_flags


@dataclass(frozen=True)
class RuntimeContextUpdate:
    """Explicit update command for RuntimeContext."""

    status: RuntimeContextStatus | None = None
    input_refs: tuple[RuntimeInputRef, ...] | list[RuntimeInputRef] | None = None
    memory_state: MemoryOrchestratorState | None = None
    world_model_snapshot: WorldModelSnapshot | None = None
    prediction_input: PredictionInput | None = None
    prediction_result: PredictionResult | None = None
    expected_observation: ExpectedObservation | None = None
    invariant_flags: (
        tuple[RuntimeInvariantFlag, ...] | list[RuntimeInvariantFlag] | None
    ) = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status is not None and not isinstance(
            self.status,
            RuntimeContextStatus,
        ):
            raise TypeError("status must be RuntimeContextStatus or None")
        input_refs = (
            None if self.input_refs is None else _ensure_input_refs(self.input_refs)
        )
        _ensure_optional_type(
            self.memory_state,
            MemoryOrchestratorState,
            "memory_state",
        )
        _ensure_optional_type(
            self.world_model_snapshot,
            WorldModelSnapshot,
            "world_model_snapshot",
        )
        _ensure_optional_type(
            self.prediction_input,
            PredictionInput,
            "prediction_input",
        )
        _ensure_optional_type(
            self.prediction_result,
            PredictionResult,
            "prediction_result",
        )
        _ensure_optional_type(
            self.expected_observation,
            ExpectedObservation,
            "expected_observation",
        )
        invariant_flags = (
            None
            if self.invariant_flags is None
            else _ensure_invariant_flags(self.invariant_flags)
        )
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "input_refs", input_refs)
        object.__setattr__(self, "invariant_flags", invariant_flags)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "status": None if self.status is None else self.status.value,
            "input_refs": (
                None
                if self.input_refs is None
                else tuple(item.to_dict() for item in self.input_refs)
            ),
            "memory_state": _to_optional_dict(self.memory_state),
            "world_model_snapshot": _to_optional_dict(self.world_model_snapshot),
            "prediction_input": _to_optional_dict(self.prediction_input),
            "prediction_result": _to_optional_dict(self.prediction_result),
            "expected_observation": _to_optional_dict(self.expected_observation),
            "invariant_flags": (
                None
                if self.invariant_flags is None
                else tuple(flag.value for flag in self.invariant_flags)
            ),
            "metadata": dict(self.metadata),
        }


class RuntimeContextBuilder:
    """Deterministic builder and updater for RuntimeContext objects."""

    def __init__(self) -> None:
        """Create a stateless runtime context builder."""

    def create(
        self,
        *,
        kind: RuntimeContextKind = RuntimeContextKind.CYCLE,
        status: RuntimeContextStatus = RuntimeContextStatus.CREATED,
        input_refs: tuple[RuntimeInputRef, ...] | list[RuntimeInputRef] = (),
        memory_state: MemoryOrchestratorState | None = None,
        world_model_snapshot: WorldModelSnapshot | None = None,
        prediction_input: PredictionInput | None = None,
        prediction_result: PredictionResult | None = None,
        expected_observation: ExpectedObservation | None = None,
        invariant_flags: (
            tuple[RuntimeInvariantFlag, ...] | list[RuntimeInvariantFlag]
        ) = (),
        metadata: dict[str, object] | None = None,
    ) -> RuntimeContext:
        """Create a runtime context."""

        required_metadata: dict[str, object] = {
            "source": "runtime_context_builder",
            "operation": "create",
        }
        context = create_runtime_context(
            kind=kind,
            status=status,
            input_refs=input_refs,
            memory_state=memory_state,
            world_model_snapshot=world_model_snapshot,
            prediction_input=prediction_input,
            prediction_result=prediction_result,
            expected_observation=expected_observation,
            invariant_flags=invariant_flags,
            metadata=_merge_metadata(required_metadata, metadata),
        )
        context_metadata = dict(context.metadata)
        context_metadata.update(_merge_metadata(required_metadata, metadata))
        context_metadata["source"] = "runtime_context_builder"
        return RuntimeContext(
            context_id=context.context_id,
            kind=context.kind,
            status=context.status,
            input_refs=context.input_refs,
            memory_state=context.memory_state,
            world_model_snapshot=context.world_model_snapshot,
            prediction_input=context.prediction_input,
            prediction_result=context.prediction_result,
            expected_observation=context.expected_observation,
            invariant_flags=context.invariant_flags,
            created_at=context.created_at,
            updated_at=context.updated_at,
            metadata=context_metadata,
        )

    def update(
        self,
        *,
        context: RuntimeContext,
        update: RuntimeContextUpdate,
    ) -> RuntimeContext:
        """Update a runtime context immutably."""

        if not isinstance(context, RuntimeContext):
            raise TypeError("context must be RuntimeContext")
        if not isinstance(update, RuntimeContextUpdate):
            raise TypeError("update must be RuntimeContextUpdate")
        required_metadata: dict[str, object] = {
            "source": "runtime_context_builder",
            "operation": "update",
        }
        metadata = _merge_metadata(context.metadata, update.metadata)
        metadata.update(required_metadata)
        return RuntimeContext(
            context_id=context.context_id,
            kind=context.kind,
            status=context.status if update.status is None else update.status,
            input_refs=(
                context.input_refs if update.input_refs is None else update.input_refs
            ),
            memory_state=(
                context.memory_state
                if update.memory_state is None
                else update.memory_state
            ),
            world_model_snapshot=(
                context.world_model_snapshot
                if update.world_model_snapshot is None
                else update.world_model_snapshot
            ),
            prediction_input=(
                context.prediction_input
                if update.prediction_input is None
                else update.prediction_input
            ),
            prediction_result=(
                context.prediction_result
                if update.prediction_result is None
                else update.prediction_result
            ),
            expected_observation=(
                context.expected_observation
                if update.expected_observation is None
                else update.expected_observation
            ),
            invariant_flags=(
                context.invariant_flags
                if update.invariant_flags is None
                else update.invariant_flags
            ),
            created_at=context.created_at,
            updated_at=utc_now(),
            metadata=metadata,
        )

    def mark_ready(self, context: RuntimeContext) -> RuntimeContext:
        """Mark a runtime context as ready."""

        return self.update(
            context=context,
            update=RuntimeContextUpdate(status=RuntimeContextStatus.READY),
        )

    def mark_running(self, context: RuntimeContext) -> RuntimeContext:
        """Mark a runtime context as running."""

        return self.update(
            context=context,
            update=RuntimeContextUpdate(status=RuntimeContextStatus.RUNNING),
        )

    def mark_completed(self, context: RuntimeContext) -> RuntimeContext:
        """Mark a runtime context as completed."""

        return self.update(
            context=context,
            update=RuntimeContextUpdate(status=RuntimeContextStatus.COMPLETED),
        )

    def mark_failed(
        self,
        context: RuntimeContext,
        *,
        reason: str,
    ) -> RuntimeContext:
        """Mark a runtime context as failed."""

        reason_value = ensure_non_empty_string(reason, "reason")
        return self.update(
            context=context,
            update=RuntimeContextUpdate(
                status=RuntimeContextStatus.FAILED,
                metadata={"failure_reason": reason_value},
            ),
        )


def utc_now() -> datetime:
    """Return a timezone-aware current UTC datetime."""

    return datetime.now(timezone.utc)


def new_runtime_context_id() -> UUID:
    """Return a new runtime context identifier."""

    return uuid4()


def new_runtime_input_ref_id() -> UUID:
    """Return a new runtime input reference identifier."""

    return uuid4()


def ensure_non_empty_string(value: str, field_name: str) -> str:
    """Return a stripped non-empty string."""

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
    """Return None or a stripped non-empty string."""

    if value is None:
        return None
    return ensure_non_empty_string(value, field_name)


def ensure_timezone_aware(value: datetime, field_name: str) -> datetime:
    """Return a timezone-aware datetime."""

    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def create_runtime_input_ref(
    *,
    source: str,
    object_ref: str,
    object_kind: str,
    metadata: dict[str, object] | None = None,
) -> RuntimeInputRef:
    """Create a runtime input reference."""

    required_metadata: dict[str, object] = {"source": "runtime_input_ref"}
    return RuntimeInputRef(
        input_ref_id=new_runtime_input_ref_id(),
        source=source,
        object_ref=object_ref,
        object_kind=object_kind,
        created_at=utc_now(),
        metadata=_merge_metadata(required_metadata, metadata),
    )


def create_runtime_context(
    *,
    kind: RuntimeContextKind = RuntimeContextKind.CYCLE,
    status: RuntimeContextStatus = RuntimeContextStatus.CREATED,
    input_refs: tuple[RuntimeInputRef, ...] | list[RuntimeInputRef] = (),
    memory_state: MemoryOrchestratorState | None = None,
    world_model_snapshot: WorldModelSnapshot | None = None,
    prediction_input: PredictionInput | None = None,
    prediction_result: PredictionResult | None = None,
    expected_observation: ExpectedObservation | None = None,
    invariant_flags: tuple[RuntimeInvariantFlag, ...] | list[RuntimeInvariantFlag] = (),
    metadata: dict[str, object] | None = None,
) -> RuntimeContext:
    """Create a runtime context."""

    now = utc_now()
    input_refs_value = _ensure_input_refs(input_refs)
    required_metadata: dict[str, object] = {
        "source": "runtime_context",
        "input_ref_count": len(input_refs_value),
        "has_memory_state": memory_state is not None,
        "has_world_model_snapshot": world_model_snapshot is not None,
        "has_prediction_input": prediction_input is not None,
        "has_prediction_result": prediction_result is not None,
        "has_expected_observation": expected_observation is not None,
    }
    return RuntimeContext(
        context_id=new_runtime_context_id(),
        kind=kind,
        status=status,
        input_refs=input_refs_value,
        memory_state=memory_state,
        world_model_snapshot=world_model_snapshot,
        prediction_input=prediction_input,
        prediction_result=prediction_result,
        expected_observation=expected_observation,
        invariant_flags=invariant_flags,
        created_at=now,
        updated_at=now,
        metadata=_merge_metadata(required_metadata, metadata),
    )


def build_runtime_context(
    *,
    kind: RuntimeContextKind = RuntimeContextKind.CYCLE,
    status: RuntimeContextStatus = RuntimeContextStatus.CREATED,
    input_refs: tuple[RuntimeInputRef, ...] | list[RuntimeInputRef] = (),
    memory_state: MemoryOrchestratorState | None = None,
    world_model_snapshot: WorldModelSnapshot | None = None,
    prediction_input: PredictionInput | None = None,
    prediction_result: PredictionResult | None = None,
    expected_observation: ExpectedObservation | None = None,
    invariant_flags: tuple[RuntimeInvariantFlag, ...] | list[RuntimeInvariantFlag] = (),
    metadata: dict[str, object] | None = None,
) -> RuntimeContext:
    """Build a runtime context with RuntimeContextBuilder."""

    builder = RuntimeContextBuilder()
    return builder.create(
        kind=kind,
        status=status,
        input_refs=input_refs,
        memory_state=memory_state,
        world_model_snapshot=world_model_snapshot,
        prediction_input=prediction_input,
        prediction_result=prediction_result,
        expected_observation=expected_observation,
        invariant_flags=invariant_flags,
        metadata=metadata,
    )


def _ensure_input_refs(
    values: tuple[RuntimeInputRef, ...] | list[RuntimeInputRef],
) -> tuple[RuntimeInputRef, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("input_refs must be tuple or list")
    input_refs = tuple(values)
    for item in input_refs:
        if not isinstance(item, RuntimeInputRef):
            raise TypeError("input_refs must contain RuntimeInputRef")
    input_ref_ids = tuple(item.input_ref_id for item in input_refs)
    if len(set(input_ref_ids)) != len(input_ref_ids):
        raise ValueError("input_ref_id values must be unique")
    return input_refs


def _ensure_invariant_flags(
    values: tuple[RuntimeInvariantFlag, ...] | list[RuntimeInvariantFlag],
) -> tuple[RuntimeInvariantFlag, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("invariant_flags must be tuple or list")
    flags = tuple(values)
    for flag in flags:
        if not isinstance(flag, RuntimeInvariantFlag):
            raise TypeError("invariant_flags must contain RuntimeInvariantFlag")
    if len(set(flags)) != len(flags):
        raise ValueError("invariant_flags must be unique")
    return flags


def _ensure_optional_type(
    value: object,
    expected_type: type[object],
    field_name: str,
) -> None:
    if value is not None and not isinstance(value, expected_type):
        raise TypeError(f"{field_name} must be {expected_type.__name__} or None")


def _validate_expected_observation_link(
    prediction_result: PredictionResult | None,
    expected_observation: ExpectedObservation | None,
) -> None:
    if prediction_result is None or expected_observation is None:
        return
    result_expected = prediction_result.expected_observation
    if result_expected is None:
        return
    if result_expected.expected_id != expected_observation.expected_id:
        raise RuntimeContextError(
            "expected_observation must match prediction_result expected observation",
            context={
                "prediction_result_id": str(prediction_result.result_id),
                "expected_observation_id": str(expected_observation.expected_id),
                "prediction_result_expected_id": str(result_expected.expected_id),
            },
        )


def _to_optional_dict(value: object) -> dict[str, object] | None:
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


def _copy_metadata(metadata: dict[str, object]) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(metadata)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
    return copied
