"""Deterministic trace-building utilities for CEBA Long-Term Memory."""

from __future__ import annotations

from dataclasses import dataclass, field

from ceba.memory.long_term.schema import (
    LongTermTrace,
    LongTermTraceKind,
    LongTermTraceStatus,
    create_long_term_trace,
    ensure_non_empty_string,
    ensure_non_negative_int,
    ensure_probability,
)


__all__ = [
    "TraceBuildPolicy",
    "TraceBuildInput",
    "TraceBuildResult",
    "TraceBuilder",
    "build_long_term_trace",
    "build_long_term_traces",
]


@dataclass(frozen=True)
class TraceBuildPolicy:
    """Configuration object controlling deterministic trace building."""

    default_kind: LongTermTraceKind = LongTermTraceKind.UNKNOWN
    default_status: LongTermTraceStatus = LongTermTraceStatus.VERIFIED
    min_confidence: float = 0.0
    min_priority: float = 0.0
    require_source_ids: bool = True
    require_semantic_hash: bool = True
    default_replay_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.default_kind, LongTermTraceKind):
            raise TypeError("default_kind must be LongTermTraceKind")
        if not isinstance(self.default_status, LongTermTraceStatus):
            raise TypeError("default_status must be LongTermTraceStatus")
        min_confidence = ensure_probability(
            self.min_confidence,
            "min_confidence",
        )
        min_priority = ensure_probability(self.min_priority, "min_priority")
        if not isinstance(self.require_source_ids, bool):
            raise TypeError("require_source_ids must be bool")
        if not isinstance(self.require_semantic_hash, bool):
            raise TypeError("require_semantic_hash must be bool")
        replay_count = ensure_non_negative_int(
            self.default_replay_count,
            "default_replay_count",
        )
        object.__setattr__(self, "min_confidence", min_confidence)
        object.__setattr__(self, "min_priority", min_priority)
        object.__setattr__(self, "default_replay_count", replay_count)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "default_kind": self.default_kind.value,
            "default_status": self.default_status.value,
            "min_confidence": self.min_confidence,
            "min_priority": self.min_priority,
            "require_source_ids": self.require_source_ids,
            "require_semantic_hash": self.require_semantic_hash,
            "default_replay_count": self.default_replay_count,
        }


@dataclass(frozen=True)
class TraceBuildInput:
    """Explicit input object for building one LongTermTrace."""

    semantic_hash: str
    source_ids: tuple[str, ...]
    kind: LongTermTraceKind | None = None
    status: LongTermTraceStatus | None = None
    confidence: float = 1.0
    priority: float = 0.5
    replay_count: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        semantic_hash = ensure_non_empty_string(
            self.semantic_hash,
            "semantic_hash",
        )
        source_ids = _ensure_string_tuple(self.source_ids, "source_ids")
        _ensure_unique_strings(source_ids, "source_ids")
        if self.kind is not None and not isinstance(
            self.kind,
            LongTermTraceKind,
        ):
            raise TypeError("kind must be LongTermTraceKind or None")
        if self.status is not None and not isinstance(
            self.status,
            LongTermTraceStatus,
        ):
            raise TypeError("status must be LongTermTraceStatus or None")
        confidence = ensure_probability(self.confidence, "confidence")
        priority = ensure_probability(self.priority, "priority")
        replay_count = (
            None
            if self.replay_count is None
            else ensure_non_negative_int(self.replay_count, "replay_count")
        )
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "semantic_hash", semantic_hash)
        object.__setattr__(self, "source_ids", source_ids)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "replay_count", replay_count)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "semantic_hash": self.semantic_hash,
            "source_ids": self.source_ids,
            "kind": None if self.kind is None else self.kind.value,
            "status": None if self.status is None else self.status.value,
            "confidence": self.confidence,
            "priority": self.priority,
            "replay_count": self.replay_count,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TraceBuildResult:
    """Immutable result object returned by trace building."""

    traces: tuple[LongTermTrace, ...]
    rejected_inputs: tuple[str, ...]
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        traces = _ensure_traces(self.traces)
        rejected_inputs = _ensure_string_tuple(
            self.rejected_inputs,
            "rejected_inputs",
        )
        rejection_reasons = _ensure_string_tuple(
            self.rejection_reasons,
            "rejection_reasons",
        )
        if len(rejected_inputs) != len(rejection_reasons):
            raise ValueError("rejected_inputs and rejection_reasons must match")
        object.__setattr__(self, "traces", traces)
        object.__setattr__(self, "rejected_inputs", rejected_inputs)
        object.__setattr__(self, "rejection_reasons", rejection_reasons)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "traces": tuple(trace.to_dict() for trace in self.traces),
            "rejected_inputs": self.rejected_inputs,
            "rejection_reasons": self.rejection_reasons,
        }

    def trace_count(self) -> int:
        """Return trace count."""

        return len(self.traces)

    def rejected_count(self) -> int:
        """Return rejected input count."""

        return len(self.rejected_inputs)


class TraceBuilder:
    """Deterministic service that builds LongTermTrace objects."""

    def __init__(self, policy: TraceBuildPolicy | None = None) -> None:
        self._policy = TraceBuildPolicy() if policy is None else policy
        if not isinstance(self._policy, TraceBuildPolicy):
            raise TypeError("policy must be TraceBuildPolicy")

    def build_one(
        self,
        *,
        trace_input: TraceBuildInput,
    ) -> TraceBuildResult:
        """Build one long-term trace."""

        if not isinstance(trace_input, TraceBuildInput):
            raise TypeError("trace_input must be TraceBuildInput")
        rejection_reason = self._rejection_reason(trace_input)
        if rejection_reason is not None:
            return TraceBuildResult(
                traces=(),
                rejected_inputs=(trace_input.semantic_hash,),
                rejection_reasons=(rejection_reason,),
            )

        kind = trace_input.kind or self._policy.default_kind
        status = trace_input.status or self._policy.default_status
        replay_count = (
            trace_input.replay_count
            if trace_input.replay_count is not None
            else self._policy.default_replay_count
        )
        metadata = self._metadata_for_input(trace_input)
        trace = create_long_term_trace(
            semantic_hash=trace_input.semantic_hash,
            kind=kind,
            status=status,
            source_candidate_id=None,
            source_ids=trace_input.source_ids,
            confidence=trace_input.confidence,
            priority=trace_input.priority,
            replay_count=replay_count,
            superseded_at=None,
            metadata=metadata,
        )
        return TraceBuildResult(
            traces=(trace,),
            rejected_inputs=(),
            rejection_reasons=(),
        )

    def build_many(
        self,
        *,
        inputs: tuple[TraceBuildInput, ...] | list[TraceBuildInput],
    ) -> TraceBuildResult:
        """Build many long-term traces in input order."""

        input_values = _ensure_inputs(inputs)
        traces: list[LongTermTrace] = []
        rejected_inputs: list[str] = []
        rejection_reasons: list[str] = []
        for trace_input in input_values:
            result = self.build_one(trace_input=trace_input)
            traces.extend(result.traces)
            rejected_inputs.extend(result.rejected_inputs)
            rejection_reasons.extend(result.rejection_reasons)
        return TraceBuildResult(
            traces=tuple(traces),
            rejected_inputs=tuple(rejected_inputs),
            rejection_reasons=tuple(rejection_reasons),
        )

    def _rejection_reason(self, trace_input: TraceBuildInput) -> str | None:
        if self._policy.require_semantic_hash and trace_input.semantic_hash == "":
            return "semantic_hash required"
        if self._policy.require_source_ids and not trace_input.source_ids:
            return "source_ids required"
        if trace_input.confidence < self._policy.min_confidence:
            return "confidence below trace build minimum"
        if trace_input.priority < self._policy.min_priority:
            return "priority below trace build minimum"
        return None

    def _metadata_for_input(
        self,
        trace_input: TraceBuildInput,
    ) -> dict[str, object]:
        metadata = _copy_metadata(trace_input.metadata)
        metadata.update(
            {
                "source": "trace_builder",
                "source_count": len(trace_input.source_ids),
                "semantic_hash": trace_input.semantic_hash,
            }
        )
        metadata["source"] = "trace_builder"
        return metadata


def build_long_term_trace(
    *,
    trace_input: TraceBuildInput,
    policy: TraceBuildPolicy | None = None,
) -> TraceBuildResult:
    """Build one long-term trace."""

    builder = TraceBuilder(policy)
    return builder.build_one(trace_input=trace_input)


def build_long_term_traces(
    *,
    inputs: tuple[TraceBuildInput, ...] | list[TraceBuildInput],
    policy: TraceBuildPolicy | None = None,
) -> TraceBuildResult:
    """Build many long-term traces."""

    builder = TraceBuilder(policy)
    return builder.build_many(inputs=inputs)


def _ensure_string_tuple(
    values: tuple[str, ...] | list[str],
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple or list")
    return tuple(ensure_non_empty_string(value, field_name) for value in values)


def _ensure_unique_strings(values: tuple[str, ...], field_name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must contain unique strings")


def _ensure_traces(
    values: tuple[LongTermTrace, ...] | list[LongTermTrace],
) -> tuple[LongTermTrace, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("traces must be tuple or list")
    result = tuple(values)
    for trace in result:
        if not isinstance(trace, LongTermTrace):
            raise TypeError("traces must contain LongTermTrace")
    trace_ids = tuple(trace.trace_id for trace in result)
    if len(set(trace_ids)) != len(trace_ids):
        raise ValueError("trace_id values must be unique")
    return result


def _ensure_inputs(
    values: tuple[TraceBuildInput, ...] | list[TraceBuildInput],
) -> tuple[TraceBuildInput, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("inputs must be tuple or list")
    result = tuple(values)
    for value in result:
        if not isinstance(value, TraceBuildInput):
            raise TypeError("inputs must contain TraceBuildInput")
    return result


def _copy_metadata(metadata: dict[str, object]) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(metadata)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
    return copied
