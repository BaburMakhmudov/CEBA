"""Deterministic eligibility decision utilities for CEBA Long-Term Memory."""

from __future__ import annotations

from dataclasses import dataclass, field

from ceba.memory.long_term.schema import (
    EligibilityDecision,
    EligibilityStatus,
    LongTermTrace,
    LongTermTraceStatus,
    create_eligibility_decision,
    ensure_non_empty_string,
    ensure_probability,
)


__all__ = [
    "EligibilityPolicy",
    "EligibilityInput",
    "EligibilityResult",
    "EligibilityEvaluator",
    "evaluate_trace_eligibility",
    "evaluate_traces_eligibility",
]


@dataclass(frozen=True)
class EligibilityPolicy:
    """Configuration object controlling deterministic trace eligibility."""

    min_confidence: float = 0.60
    min_priority: float = 0.50
    require_verified_status: bool = True
    allow_replayable_status: bool = True
    block_superseded: bool = True
    block_invalid: bool = True
    block_blocked: bool = True

    def __post_init__(self) -> None:
        min_confidence = ensure_probability(
            self.min_confidence,
            "min_confidence",
        )
        min_priority = ensure_probability(self.min_priority, "min_priority")
        if not isinstance(self.require_verified_status, bool):
            raise TypeError("require_verified_status must be bool")
        if not isinstance(self.allow_replayable_status, bool):
            raise TypeError("allow_replayable_status must be bool")
        if not isinstance(self.block_superseded, bool):
            raise TypeError("block_superseded must be bool")
        if not isinstance(self.block_invalid, bool):
            raise TypeError("block_invalid must be bool")
        if not isinstance(self.block_blocked, bool):
            raise TypeError("block_blocked must be bool")
        object.__setattr__(self, "min_confidence", min_confidence)
        object.__setattr__(self, "min_priority", min_priority)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "min_confidence": self.min_confidence,
            "min_priority": self.min_priority,
            "require_verified_status": self.require_verified_status,
            "allow_replayable_status": self.allow_replayable_status,
            "block_superseded": self.block_superseded,
            "block_invalid": self.block_invalid,
            "block_blocked": self.block_blocked,
        }


@dataclass(frozen=True)
class EligibilityInput:
    """Explicit input object for evaluating one LongTermTrace."""

    trace: LongTermTrace
    reason_context: str | None = None
    confidence_override: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.trace, LongTermTrace):
            raise TypeError("trace must be LongTermTrace")
        reason_context = (
            None
            if self.reason_context is None
            else ensure_non_empty_string(self.reason_context, "reason_context")
        )
        confidence_override = (
            None
            if self.confidence_override is None
            else ensure_probability(
                self.confidence_override,
                "confidence_override",
            )
        )
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "reason_context", reason_context)
        object.__setattr__(self, "confidence_override", confidence_override)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "trace": self.trace.to_dict(),
            "reason_context": self.reason_context,
            "confidence_override": self.confidence_override,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EligibilityResult:
    """Immutable result object returned by eligibility evaluation."""

    decisions: tuple[EligibilityDecision, ...]
    blocked_trace_ids: tuple[str, ...]
    block_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        decisions = _ensure_decisions(self.decisions)
        blocked_trace_ids = _ensure_string_tuple(
            self.blocked_trace_ids,
            "blocked_trace_ids",
        )
        block_reasons = _ensure_string_tuple(
            self.block_reasons,
            "block_reasons",
        )
        if len(blocked_trace_ids) != len(block_reasons):
            raise ValueError("blocked_trace_ids and block_reasons must match")
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(self, "blocked_trace_ids", blocked_trace_ids)
        object.__setattr__(self, "block_reasons", block_reasons)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "decisions": tuple(decision.to_dict() for decision in self.decisions),
            "blocked_trace_ids": self.blocked_trace_ids,
            "block_reasons": self.block_reasons,
        }

    def decision_count(self) -> int:
        """Return eligibility decision count."""

        return len(self.decisions)

    def blocked_count(self) -> int:
        """Return blocked trace count."""

        return len(self.blocked_trace_ids)


class EligibilityEvaluator:
    """Deterministic service that creates EligibilityDecision objects."""

    def __init__(self, policy: EligibilityPolicy | None = None) -> None:
        self._policy = EligibilityPolicy() if policy is None else policy
        if not isinstance(self._policy, EligibilityPolicy):
            raise TypeError("policy must be EligibilityPolicy")

    def evaluate_one(
        self,
        *,
        eligibility_input: EligibilityInput,
    ) -> EligibilityResult:
        """Evaluate one trace eligibility input."""

        if not isinstance(eligibility_input, EligibilityInput):
            raise TypeError("eligibility_input must be EligibilityInput")
        trace = eligibility_input.trace
        status, reason = self._status_and_reason(trace)
        confidence = (
            eligibility_input.confidence_override
            if eligibility_input.confidence_override is not None
            else trace.confidence
        )
        metadata = self._metadata_for_input(eligibility_input)
        decision = create_eligibility_decision(
            trace_id=trace.trace_id,
            reason=reason,
            status=status,
            confidence=confidence,
            metadata=metadata,
        )
        blocked_ids: tuple[str, ...] = ()
        block_reasons: tuple[str, ...] = ()
        if status in (EligibilityStatus.BLOCKED, EligibilityStatus.INVALID):
            blocked_ids = (str(trace.trace_id),)
            block_reasons = (reason,)
        return EligibilityResult(
            decisions=(decision,),
            blocked_trace_ids=blocked_ids,
            block_reasons=block_reasons,
        )

    def evaluate_many(
        self,
        *,
        inputs: tuple[EligibilityInput, ...] | list[EligibilityInput],
    ) -> EligibilityResult:
        """Evaluate many trace eligibility inputs in input order."""

        input_values = _ensure_inputs(inputs)
        decisions: list[EligibilityDecision] = []
        blocked_trace_ids: list[str] = []
        block_reasons: list[str] = []
        for eligibility_input in input_values:
            result = self.evaluate_one(eligibility_input=eligibility_input)
            decisions.extend(result.decisions)
            blocked_trace_ids.extend(result.blocked_trace_ids)
            block_reasons.extend(result.block_reasons)
        return EligibilityResult(
            decisions=tuple(decisions),
            blocked_trace_ids=tuple(blocked_trace_ids),
            block_reasons=tuple(block_reasons),
        )

    def _status_and_reason(
        self,
        trace: LongTermTrace,
    ) -> tuple[EligibilityStatus, str]:
        if (
            self._policy.block_invalid
            and trace.status is LongTermTraceStatus.INVALID
        ):
            return EligibilityStatus.INVALID, "trace status invalid"
        if (
            self._policy.block_blocked
            and trace.status is LongTermTraceStatus.BLOCKED
        ):
            return EligibilityStatus.BLOCKED, "trace status blocked"
        if (
            self._policy.block_superseded
            and trace.status is LongTermTraceStatus.SUPERSEDED
        ):
            return EligibilityStatus.BLOCKED, "trace superseded"
        verified_statuses = (
            LongTermTraceStatus.VERIFIED,
            LongTermTraceStatus.CONSOLIDATED,
            LongTermTraceStatus.REPLAYABLE,
        )
        if self._policy.require_verified_status and (
            trace.status not in verified_statuses
        ):
            return EligibilityStatus.INELIGIBLE, "trace status not eligible"
        if (
            not self._policy.allow_replayable_status
            and trace.status is LongTermTraceStatus.REPLAYABLE
        ):
            return (
                EligibilityStatus.INELIGIBLE,
                "replayable status not allowed",
            )
        if trace.confidence < self._policy.min_confidence:
            return (
                EligibilityStatus.INELIGIBLE,
                "confidence below eligibility minimum",
            )
        if trace.priority < self._policy.min_priority:
            return (
                EligibilityStatus.INELIGIBLE,
                "priority below eligibility minimum",
            )
        return EligibilityStatus.ELIGIBLE, "trace eligible"

    def _metadata_for_input(
        self,
        eligibility_input: EligibilityInput,
    ) -> dict[str, object]:
        trace = eligibility_input.trace
        metadata = _copy_metadata(eligibility_input.metadata)
        metadata.update(
            {
                "source": "eligibility_evaluator",
                "trace_id": str(trace.trace_id),
                "trace_kind": trace.kind.value,
                "trace_status": trace.status.value,
                "trace_confidence": trace.confidence,
                "trace_priority": trace.priority,
                "policy": self._policy.to_dict(),
            }
        )
        if eligibility_input.reason_context is not None:
            metadata["reason_context"] = eligibility_input.reason_context
        metadata["source"] = "eligibility_evaluator"
        return metadata


def evaluate_trace_eligibility(
    *,
    eligibility_input: EligibilityInput,
    policy: EligibilityPolicy | None = None,
) -> EligibilityResult:
    """Evaluate one trace eligibility input."""

    evaluator = EligibilityEvaluator(policy)
    return evaluator.evaluate_one(eligibility_input=eligibility_input)


def evaluate_traces_eligibility(
    *,
    inputs: tuple[EligibilityInput, ...] | list[EligibilityInput],
    policy: EligibilityPolicy | None = None,
) -> EligibilityResult:
    """Evaluate many trace eligibility inputs."""

    evaluator = EligibilityEvaluator(policy)
    return evaluator.evaluate_many(inputs=inputs)


def _ensure_string_tuple(
    values: tuple[str, ...] | list[str],
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple or list")
    return tuple(ensure_non_empty_string(value, field_name) for value in values)


def _ensure_decisions(
    values: tuple[EligibilityDecision, ...] | list[EligibilityDecision],
) -> tuple[EligibilityDecision, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("decisions must be tuple or list")
    decisions = tuple(values)
    for decision in decisions:
        if not isinstance(decision, EligibilityDecision):
            raise TypeError("decisions must contain EligibilityDecision")
    decision_ids = tuple(decision.decision_id for decision in decisions)
    if len(set(decision_ids)) != len(decision_ids):
        raise ValueError("decision_id values must be unique")
    return decisions


def _ensure_inputs(
    values: tuple[EligibilityInput, ...] | list[EligibilityInput],
) -> tuple[EligibilityInput, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("inputs must be tuple or list")
    inputs = tuple(values)
    for value in inputs:
        if not isinstance(value, EligibilityInput):
            raise TypeError("inputs must contain EligibilityInput")
    return inputs


def _copy_metadata(metadata: dict[str, object]) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(metadata)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
    return copied
