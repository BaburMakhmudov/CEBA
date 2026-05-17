"""Deterministic consolidation utilities for CEBA Long-Term Memory."""

from __future__ import annotations

from dataclasses import dataclass, field

from ceba.memory.long_term.schema import (
    ConsolidatedExperience,
    ConsolidationStatus,
    EligibilityDecision,
    EligibilityStatus,
    LongTermTrace,
    LongTermTraceKind,
    LongTermTraceStatus,
    create_consolidated_experience,
    create_eligibility_decision,
    create_long_term_trace,
    ensure_non_empty_string,
    ensure_probability,
)
from ceba.memory.short_term.schema import CandidateTrace, CandidateTraceStatus


__all__ = [
    "ConsolidationPolicy",
    "ConsolidationInput",
    "ConsolidationResult",
    "Consolidator",
    "consolidate_candidate_trace",
    "consolidate_candidate_traces",
]


@dataclass(frozen=True)
class ConsolidationPolicy:
    """Configuration object controlling deterministic LTM consolidation."""

    min_confidence: float = 0.60
    min_priority: float = 0.50
    require_eligible_status: bool = True
    require_evidence: bool = True
    default_trace_kind: LongTermTraceKind = (
        LongTermTraceKind.CONSOLIDATED_EXPERIENCE
    )
    default_trace_status: LongTermTraceStatus = LongTermTraceStatus.CONSOLIDATED
    default_experience_status: ConsolidationStatus = (
        ConsolidationStatus.CONSOLIDATED
    )

    def __post_init__(self) -> None:
        min_confidence = ensure_probability(
            self.min_confidence,
            "min_confidence",
        )
        min_priority = ensure_probability(self.min_priority, "min_priority")
        if not isinstance(self.require_eligible_status, bool):
            raise TypeError("require_eligible_status must be bool")
        if not isinstance(self.require_evidence, bool):
            raise TypeError("require_evidence must be bool")
        if not isinstance(self.default_trace_kind, LongTermTraceKind):
            raise TypeError("default_trace_kind must be LongTermTraceKind")
        if not isinstance(self.default_trace_status, LongTermTraceStatus):
            raise TypeError("default_trace_status must be LongTermTraceStatus")
        if not isinstance(self.default_experience_status, ConsolidationStatus):
            raise TypeError(
                "default_experience_status must be ConsolidationStatus"
            )
        object.__setattr__(self, "min_confidence", min_confidence)
        object.__setattr__(self, "min_priority", min_priority)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "min_confidence": self.min_confidence,
            "min_priority": self.min_priority,
            "require_eligible_status": self.require_eligible_status,
            "require_evidence": self.require_evidence,
            "default_trace_kind": self.default_trace_kind.value,
            "default_trace_status": self.default_trace_status.value,
            "default_experience_status": self.default_experience_status.value,
        }


@dataclass(frozen=True)
class ConsolidationInput:
    """Explicit input object for consolidating one CandidateTrace."""

    candidate_trace: CandidateTrace
    semantic_hash: str
    source_ids: tuple[str, ...]
    summary_ref: str | None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_trace, CandidateTrace):
            raise TypeError("candidate_trace must be CandidateTrace")
        semantic_hash = ensure_non_empty_string(
            self.semantic_hash,
            "semantic_hash",
        )
        source_ids = _ensure_string_tuple(self.source_ids, "source_ids")
        _ensure_unique_strings(source_ids, "source_ids")
        summary_ref = (
            None
            if self.summary_ref is None
            else ensure_non_empty_string(self.summary_ref, "summary_ref")
        )
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "semantic_hash", semantic_hash)
        object.__setattr__(self, "source_ids", source_ids)
        object.__setattr__(self, "summary_ref", summary_ref)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "candidate_trace": self.candidate_trace.to_dict(),
            "semantic_hash": self.semantic_hash,
            "source_ids": self.source_ids,
            "summary_ref": self.summary_ref,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ConsolidationResult:
    """Immutable result object returned by consolidation."""

    traces: tuple[LongTermTrace, ...]
    experiences: tuple[ConsolidatedExperience, ...]
    eligibility_decisions: tuple[EligibilityDecision, ...]
    rejected_candidate_ids: tuple[str, ...]
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        traces = _ensure_traces(self.traces)
        experiences = _ensure_experiences(self.experiences)
        decisions = _ensure_eligibility_decisions(self.eligibility_decisions)
        rejected_ids = _ensure_string_tuple(
            self.rejected_candidate_ids,
            "rejected_candidate_ids",
        )
        reasons = _ensure_string_tuple(
            self.rejection_reasons,
            "rejection_reasons",
        )
        if len(rejected_ids) != len(reasons):
            raise ValueError(
                "rejected_candidate_ids and rejection_reasons must match"
            )
        object.__setattr__(self, "traces", traces)
        object.__setattr__(self, "experiences", experiences)
        object.__setattr__(self, "eligibility_decisions", decisions)
        object.__setattr__(self, "rejected_candidate_ids", rejected_ids)
        object.__setattr__(self, "rejection_reasons", reasons)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "traces": tuple(trace.to_dict() for trace in self.traces),
            "experiences": tuple(
                experience.to_dict() for experience in self.experiences
            ),
            "eligibility_decisions": tuple(
                decision.to_dict() for decision in self.eligibility_decisions
            ),
            "rejected_candidate_ids": self.rejected_candidate_ids,
            "rejection_reasons": self.rejection_reasons,
        }

    def trace_count(self) -> int:
        """Return trace count."""

        return len(self.traces)

    def experience_count(self) -> int:
        """Return experience count."""

        return len(self.experiences)

    def rejected_count(self) -> int:
        """Return rejected candidate count."""

        return len(self.rejected_candidate_ids)


class Consolidator:
    """Deterministic service for consolidating eligible CandidateTrace objects."""

    def __init__(self, policy: ConsolidationPolicy | None = None) -> None:
        self._policy = ConsolidationPolicy() if policy is None else policy
        if not isinstance(self._policy, ConsolidationPolicy):
            raise TypeError("policy must be ConsolidationPolicy")

    def consolidate_one(
        self,
        *,
        consolidation_input: ConsolidationInput,
    ) -> ConsolidationResult:
        """Consolidate one candidate trace."""

        if not isinstance(consolidation_input, ConsolidationInput):
            raise TypeError("consolidation_input must be ConsolidationInput")
        candidate = consolidation_input.candidate_trace
        rejection_reason = self._rejection_reason(candidate)
        if rejection_reason is not None:
            return ConsolidationResult(
                traces=(),
                experiences=(),
                eligibility_decisions=(),
                rejected_candidate_ids=(str(candidate.trace_id),),
                rejection_reasons=(rejection_reason,),
            )

        metadata = self._metadata_for_input(consolidation_input)
        source_ids = (
            consolidation_input.source_ids
            if consolidation_input.source_ids
            else candidate.evidence_ids
        )
        trace = create_long_term_trace(
            semantic_hash=consolidation_input.semantic_hash,
            kind=self._policy.default_trace_kind,
            status=self._policy.default_trace_status,
            source_candidate_id=candidate.trace_id,
            source_ids=source_ids,
            confidence=candidate.confidence,
            priority=candidate.priority,
            replay_count=0,
            superseded_at=None,
            metadata=metadata,
        )
        experience = create_consolidated_experience(
            trace_ids=(trace.trace_id,),
            status=self._policy.default_experience_status,
            summary_ref=consolidation_input.summary_ref,
            confidence=candidate.confidence,
            utility_score=candidate.priority,
            metadata=metadata,
        )
        decision = create_eligibility_decision(
            trace_id=trace.trace_id,
            reason="candidate trace accepted",
            status=EligibilityStatus.ELIGIBLE,
            confidence=candidate.confidence,
            metadata=metadata,
        )
        return ConsolidationResult(
            traces=(trace,),
            experiences=(experience,),
            eligibility_decisions=(decision,),
            rejected_candidate_ids=(),
            rejection_reasons=(),
        )

    def consolidate_many(
        self,
        *,
        inputs: tuple[ConsolidationInput, ...] | list[ConsolidationInput],
    ) -> ConsolidationResult:
        """Consolidate many candidate traces in input order."""

        input_values = _ensure_inputs(inputs)
        traces: list[LongTermTrace] = []
        experiences: list[ConsolidatedExperience] = []
        decisions: list[EligibilityDecision] = []
        rejected_ids: list[str] = []
        reasons: list[str] = []
        for consolidation_input in input_values:
            result = self.consolidate_one(
                consolidation_input=consolidation_input
            )
            traces.extend(result.traces)
            experiences.extend(result.experiences)
            decisions.extend(result.eligibility_decisions)
            rejected_ids.extend(result.rejected_candidate_ids)
            reasons.extend(result.rejection_reasons)
        return ConsolidationResult(
            traces=tuple(traces),
            experiences=tuple(experiences),
            eligibility_decisions=tuple(decisions),
            rejected_candidate_ids=tuple(rejected_ids),
            rejection_reasons=tuple(reasons),
        )

    def _rejection_reason(self, candidate: CandidateTrace) -> str | None:
        if (
            self._policy.require_eligible_status
            and candidate.status is not CandidateTraceStatus.ELIGIBLE
        ):
            return "candidate trace is not eligible"
        if candidate.confidence < self._policy.min_confidence:
            return "confidence below consolidation minimum"
        if candidate.priority < self._policy.min_priority:
            return "priority below consolidation minimum"
        if self._policy.require_evidence and not candidate.evidence_ids:
            return "candidate trace has no evidence"
        if not candidate.item_ids:
            return "candidate trace has no items"
        return None

    def _metadata_for_input(
        self,
        consolidation_input: ConsolidationInput,
    ) -> dict[str, object]:
        candidate = consolidation_input.candidate_trace
        metadata = _copy_metadata(consolidation_input.metadata)
        metadata.update(
            {
                "source": "consolidator",
                "candidate_trace_id": str(candidate.trace_id),
                "candidate_status": candidate.status.value,
                "candidate_item_count": len(candidate.item_ids),
                "candidate_evidence_count": len(candidate.evidence_ids),
                "candidate_evidence_ids": candidate.evidence_ids,
            }
        )
        metadata["source"] = "consolidator"
        return metadata


def consolidate_candidate_trace(
    *,
    consolidation_input: ConsolidationInput,
    policy: ConsolidationPolicy | None = None,
) -> ConsolidationResult:
    """Consolidate one candidate trace."""

    consolidator = Consolidator(policy)
    return consolidator.consolidate_one(
        consolidation_input=consolidation_input
    )


def consolidate_candidate_traces(
    *,
    inputs: tuple[ConsolidationInput, ...] | list[ConsolidationInput],
    policy: ConsolidationPolicy | None = None,
) -> ConsolidationResult:
    """Consolidate many candidate traces."""

    consolidator = Consolidator(policy)
    return consolidator.consolidate_many(inputs=inputs)


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


def _ensure_experiences(
    values: tuple[ConsolidatedExperience, ...] | list[ConsolidatedExperience],
) -> tuple[ConsolidatedExperience, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("experiences must be tuple or list")
    result = tuple(values)
    for experience in result:
        if not isinstance(experience, ConsolidatedExperience):
            raise TypeError("experiences must contain ConsolidatedExperience")
    experience_ids = tuple(experience.experience_id for experience in result)
    if len(set(experience_ids)) != len(experience_ids):
        raise ValueError("experience_id values must be unique")
    return result


def _ensure_eligibility_decisions(
    values: tuple[EligibilityDecision, ...] | list[EligibilityDecision],
) -> tuple[EligibilityDecision, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("eligibility_decisions must be tuple or list")
    result = tuple(values)
    for decision in result:
        if not isinstance(decision, EligibilityDecision):
            raise TypeError(
                "eligibility_decisions must contain EligibilityDecision"
            )
    decision_ids = tuple(decision.decision_id for decision in result)
    if len(set(decision_ids)) != len(decision_ids):
        raise ValueError("decision_id values must be unique")
    return result


def _ensure_inputs(
    values: tuple[ConsolidationInput, ...] | list[ConsolidationInput],
) -> tuple[ConsolidationInput, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("inputs must be tuple or list")
    result = tuple(values)
    for value in result:
        if not isinstance(value, ConsolidationInput):
            raise TypeError("inputs must contain ConsolidationInput")
    return result


def _copy_metadata(metadata: dict[str, object]) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(metadata)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
    return copied
