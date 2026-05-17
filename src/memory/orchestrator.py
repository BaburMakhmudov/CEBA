"""Deterministic runtime MemoryOrchestrator for CEBA Memory."""

from __future__ import annotations

from dataclasses import dataclass, field

from ceba.memory.short_term import (
    ShortTermMemorySnapshot,
    ShortTermValidationReport,
    WorkingBuffer,
    WorkingBufferPolicy,
    build_working_buffer,
    create_short_term_memory_snapshot,
    ensure_non_empty_string as ensure_short_term_non_empty_string,
    validate_short_term_snapshot,
)
from ceba.memory.long_term import (
    ConsolidationInput,
    ConsolidationPolicy,
    ConsolidationResult,
    Consolidator,
    LongTermMemorySnapshot,
    LongTermValidationReport,
    create_long_term_memory_snapshot,
    validate_long_term_snapshot,
)
from ceba.memory.retrieval import (
    EvidencePackBuilder,
    EvidencePackInput,
    EvidencePackPolicy,
    EvidencePackResult,
    GraphRetrievalAdapter,
    GraphRetrievalHit,
    GraphRetrievalPolicy,
    GraphRetrievalResult,
    HybridRetrievalFusion,
    HybridRetrievalInput,
    HybridRetrievalPolicy,
    HybridRetrievalResult,
    RankingInput,
    RankingPolicy,
    RankingResult,
    RetrievalCandidate,
    RetrievalEvidencePack,
    RetrievalQuery,
    RetrievalQueryBuilder,
    RetrievalQueryInput,
    RetrievalQueryPolicy,
    RetrievalQueryResult,
    RetrievalRanker,
    RetrievalSnapshot,
    RetrievalStatus,
    VectorRetrievalAdapter,
    VectorRetrievalHit,
    VectorRetrievalPolicy,
    VectorRetrievalResult,
    create_retrieval_snapshot,
)


__all__ = [
    "MemoryOrchestratorPolicy",
    "MemoryOrchestratorState",
    "MemoryOrchestratorReport",
    "MemoryOrchestrator",
    "build_memory_orchestrator",
]


@dataclass(frozen=True)
class MemoryOrchestratorPolicy:
    """Configuration object controlling deterministic memory orchestration."""

    working_buffer_policy: WorkingBufferPolicy = field(
        default_factory=WorkingBufferPolicy
    )
    consolidation_policy: ConsolidationPolicy = field(
        default_factory=ConsolidationPolicy
    )
    retrieval_query_policy: RetrievalQueryPolicy = field(
        default_factory=RetrievalQueryPolicy
    )
    ranking_policy: RankingPolicy = field(default_factory=RankingPolicy)
    evidence_pack_policy: EvidencePackPolicy = field(
        default_factory=EvidencePackPolicy
    )
    vector_retrieval_policy: VectorRetrievalPolicy = field(
        default_factory=VectorRetrievalPolicy
    )
    graph_retrieval_policy: GraphRetrievalPolicy = field(
        default_factory=GraphRetrievalPolicy
    )
    hybrid_retrieval_policy: HybridRetrievalPolicy = field(
        default_factory=HybridRetrievalPolicy
    )
    validate_after_update: bool = True
    allow_invalid_snapshots: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.working_buffer_policy, WorkingBufferPolicy):
            raise TypeError("working_buffer_policy must be WorkingBufferPolicy")
        if not isinstance(self.consolidation_policy, ConsolidationPolicy):
            raise TypeError("consolidation_policy must be ConsolidationPolicy")
        if not isinstance(self.retrieval_query_policy, RetrievalQueryPolicy):
            raise TypeError("retrieval_query_policy must be RetrievalQueryPolicy")
        if not isinstance(self.ranking_policy, RankingPolicy):
            raise TypeError("ranking_policy must be RankingPolicy")
        if not isinstance(self.evidence_pack_policy, EvidencePackPolicy):
            raise TypeError("evidence_pack_policy must be EvidencePackPolicy")
        if not isinstance(self.vector_retrieval_policy, VectorRetrievalPolicy):
            raise TypeError(
                "vector_retrieval_policy must be VectorRetrievalPolicy"
            )
        if not isinstance(self.graph_retrieval_policy, GraphRetrievalPolicy):
            raise TypeError("graph_retrieval_policy must be GraphRetrievalPolicy")
        if not isinstance(self.hybrid_retrieval_policy, HybridRetrievalPolicy):
            raise TypeError(
                "hybrid_retrieval_policy must be HybridRetrievalPolicy"
            )
        if not isinstance(self.validate_after_update, bool):
            raise TypeError("validate_after_update must be bool")
        if not isinstance(self.allow_invalid_snapshots, bool):
            raise TypeError("allow_invalid_snapshots must be bool")

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "working_buffer_policy": self.working_buffer_policy.to_dict(),
            "consolidation_policy": self.consolidation_policy.to_dict(),
            "retrieval_query_policy": self.retrieval_query_policy.to_dict(),
            "ranking_policy": self.ranking_policy.to_dict(),
            "evidence_pack_policy": self.evidence_pack_policy.to_dict(),
            "vector_retrieval_policy": self.vector_retrieval_policy.to_dict(),
            "graph_retrieval_policy": self.graph_retrieval_policy.to_dict(),
            "hybrid_retrieval_policy": self.hybrid_retrieval_policy.to_dict(),
            "validate_after_update": self.validate_after_update,
            "allow_invalid_snapshots": self.allow_invalid_snapshots,
        }


@dataclass(frozen=True)
class MemoryOrchestratorState:
    """Immutable runtime state bundle managed by MemoryOrchestrator."""

    short_term_snapshot: ShortTermMemorySnapshot
    long_term_snapshot: LongTermMemorySnapshot
    retrieval_snapshots: tuple[RetrievalSnapshot, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.short_term_snapshot, ShortTermMemorySnapshot):
            raise TypeError("short_term_snapshot must be ShortTermMemorySnapshot")
        if not isinstance(self.long_term_snapshot, LongTermMemorySnapshot):
            raise TypeError("long_term_snapshot must be LongTermMemorySnapshot")
        retrieval_snapshots = _ensure_retrieval_snapshots(
            self.retrieval_snapshots
        )
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "retrieval_snapshots", retrieval_snapshots)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "short_term_snapshot": self.short_term_snapshot.to_dict(),
            "long_term_snapshot": self.long_term_snapshot.to_dict(),
            "retrieval_snapshots": tuple(
                snapshot.to_dict() for snapshot in self.retrieval_snapshots
            ),
            "metadata": dict(self.metadata),
        }

    def retrieval_snapshot_count(self) -> int:
        """Return retrieval snapshot count."""

        return len(self.retrieval_snapshots)


@dataclass(frozen=True)
class MemoryOrchestratorReport:
    """Immutable report returned by orchestration operations."""

    state: MemoryOrchestratorState
    short_term_validation: ShortTermValidationReport | None
    long_term_validation: LongTermValidationReport | None
    changed: bool
    reason: str
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.state, MemoryOrchestratorState):
            raise TypeError("state must be MemoryOrchestratorState")
        if (
            self.short_term_validation is not None
            and not isinstance(
                self.short_term_validation,
                ShortTermValidationReport,
            )
        ):
            raise TypeError(
                "short_term_validation must be ShortTermValidationReport or None"
            )
        if (
            self.long_term_validation is not None
            and not isinstance(
                self.long_term_validation,
                LongTermValidationReport,
            )
        ):
            raise TypeError(
                "long_term_validation must be LongTermValidationReport or None"
            )
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        reason = ensure_short_term_non_empty_string(self.reason, "reason")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "state": self.state.to_dict(),
            "short_term_validation": (
                None
                if self.short_term_validation is None
                else self.short_term_validation.to_dict()
            ),
            "long_term_validation": (
                None
                if self.long_term_validation is None
                else self.long_term_validation.to_dict()
            ),
            "changed": self.changed,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }

    def is_valid(self) -> bool:
        """Return whether included validation reports are valid."""

        short_valid = (
            True
            if self.short_term_validation is None
            else self.short_term_validation.valid
        )
        long_valid = (
            True
            if self.long_term_validation is None
            else self.long_term_validation.valid
        )
        return short_valid and long_valid


class MemoryOrchestrator:
    """Deterministic runtime coordinator for STM, LTM, and retrieval."""

    def __init__(
        self,
        *,
        state: MemoryOrchestratorState | None = None,
        policy: MemoryOrchestratorPolicy | None = None,
    ) -> None:
        self._policy = MemoryOrchestratorPolicy() if policy is None else policy
        if not isinstance(self._policy, MemoryOrchestratorPolicy):
            raise TypeError("policy must be MemoryOrchestratorPolicy")
        self._state = _empty_state() if state is None else state
        if not isinstance(self._state, MemoryOrchestratorState):
            raise TypeError("state must be MemoryOrchestratorState")

    def state(self) -> MemoryOrchestratorState:
        """Return current state."""

        return self._state

    def short_term_snapshot(self) -> ShortTermMemorySnapshot:
        """Return current short-term snapshot."""

        return self._state.short_term_snapshot

    def long_term_snapshot(self) -> LongTermMemorySnapshot:
        """Return current long-term snapshot."""

        return self._state.long_term_snapshot

    def retrieval_snapshots(self) -> tuple[RetrievalSnapshot, ...]:
        """Return current retrieval snapshots."""

        return self._state.retrieval_snapshots

    def validate(self) -> MemoryOrchestratorReport:
        """Validate current STM and LTM snapshots."""

        return MemoryOrchestratorReport(
            state=self._state,
            short_term_validation=validate_short_term_snapshot(
                snapshot=self._state.short_term_snapshot
            ),
            long_term_validation=validate_long_term_snapshot(
                snapshot=self._state.long_term_snapshot
            ),
            changed=False,
            reason="validated memory orchestrator state",
        )

    def replace_short_term_snapshot(
        self,
        snapshot: ShortTermMemorySnapshot,
    ) -> MemoryOrchestratorReport:
        """Replace only the short-term snapshot."""

        if not isinstance(snapshot, ShortTermMemorySnapshot):
            raise TypeError("snapshot must be ShortTermMemorySnapshot")
        validation = self._validate_short_term_for_update(snapshot)
        self._state = self._state_with(short_term_snapshot=snapshot)
        return self._report(
            changed=True,
            reason="short-term snapshot replaced",
            short_term_validation=validation,
        )

    def replace_long_term_snapshot(
        self,
        snapshot: LongTermMemorySnapshot,
    ) -> MemoryOrchestratorReport:
        """Replace only the long-term snapshot."""

        if not isinstance(snapshot, LongTermMemorySnapshot):
            raise TypeError("snapshot must be LongTermMemorySnapshot")
        validation = self._validate_long_term_for_update(snapshot)
        self._state = self._state_with(long_term_snapshot=snapshot)
        return self._report(
            changed=True,
            reason="long-term snapshot replaced",
            long_term_validation=validation,
        )

    def add_retrieval_snapshot(
        self,
        snapshot: RetrievalSnapshot,
    ) -> MemoryOrchestratorReport:
        """Add or replace a retrieval snapshot."""

        if not isinstance(snapshot, RetrievalSnapshot):
            raise TypeError("snapshot must be RetrievalSnapshot")
        retrieval_snapshots = _replace_retrieval_snapshot(
            self._state.retrieval_snapshots,
            snapshot,
        )
        self._state = self._state_with(retrieval_snapshots=retrieval_snapshots)
        return self._report(
            changed=True,
            reason="retrieval snapshot added",
        )

    def build_working_buffer_from_state(self) -> WorkingBuffer:
        """Build a working buffer from the current short-term snapshot."""

        return build_working_buffer(
            items=self._state.short_term_snapshot.items,
            evictions=self._state.short_term_snapshot.evictions,
            policy=self._policy.working_buffer_policy,
        )

    def update_short_term_from_buffer(
        self,
        buffer: WorkingBuffer,
    ) -> MemoryOrchestratorReport:
        """Update short-term snapshot from a working buffer."""

        if not isinstance(buffer, WorkingBuffer):
            raise TypeError("buffer must be WorkingBuffer")
        current = self._state.short_term_snapshot
        snapshot = create_short_term_memory_snapshot(
            items=buffer.items(),
            contexts=current.contexts,
            candidate_traces=current.candidate_traces,
            hypotheses=current.hypotheses,
            evictions=buffer.evictions(),
            metadata={
                "runtime_source": "memory_orchestrator",
                "operation": "update_short_term_from_buffer",
            },
        )
        return self.replace_short_term_snapshot(snapshot)

    def consolidate_one(
        self,
        consolidation_input: ConsolidationInput,
    ) -> MemoryOrchestratorReport:
        """Consolidate one candidate trace and update long-term state."""

        if not isinstance(consolidation_input, ConsolidationInput):
            raise TypeError("consolidation_input must be ConsolidationInput")
        result = Consolidator(self._policy.consolidation_policy).consolidate_one(
            consolidation_input=consolidation_input
        )
        return self._apply_consolidation_result(result)

    def consolidate_many(
        self,
        inputs: tuple[ConsolidationInput, ...] | list[ConsolidationInput],
    ) -> MemoryOrchestratorReport:
        """Consolidate many candidate traces and update long-term state."""

        result = Consolidator(
            self._policy.consolidation_policy
        ).consolidate_many(inputs=_ensure_consolidation_inputs(inputs))
        return self._apply_consolidation_result(result)

    def build_retrieval_query(
        self,
        query_input: RetrievalQueryInput,
    ) -> RetrievalQueryResult:
        """Build a retrieval query without mutating state."""

        return RetrievalQueryBuilder(self._policy.retrieval_query_policy).build_one(
            query_input=query_input
        )

    def adapt_vector_hits(
        self,
        *,
        query: RetrievalQuery,
        hits: tuple[VectorRetrievalHit, ...] | list[VectorRetrievalHit],
    ) -> VectorRetrievalResult:
        """Adapt explicit vector hits without mutating state."""

        return VectorRetrievalAdapter(self._policy.vector_retrieval_policy).adapt(
            query=query,
            hits=hits,
        )

    def adapt_graph_hits(
        self,
        *,
        query: RetrievalQuery,
        hits: tuple[GraphRetrievalHit, ...] | list[GraphRetrievalHit],
    ) -> GraphRetrievalResult:
        """Adapt explicit graph hits without mutating state."""

        return GraphRetrievalAdapter(self._policy.graph_retrieval_policy).adapt(
            query=query,
            hits=hits,
        )

    def fuse_hybrid_candidates(
        self,
        hybrid_input: HybridRetrievalInput,
    ) -> HybridRetrievalResult:
        """Fuse vector and graph candidates without mutating state."""

        return HybridRetrievalFusion(self._policy.hybrid_retrieval_policy).fuse(
            hybrid_input=hybrid_input
        )

    def rank_candidates(
        self,
        ranking_input: RankingInput,
    ) -> RankingResult:
        """Rank candidates without mutating state."""

        return RetrievalRanker(self._policy.ranking_policy).rank(
            ranking_input=ranking_input
        )

    def build_evidence_pack(
        self,
        pack_input: EvidencePackInput,
    ) -> EvidencePackResult:
        """Build an evidence pack without mutating state."""

        return EvidencePackBuilder(self._policy.evidence_pack_policy).build(
            pack_input=pack_input
        )

    def create_retrieval_snapshot_from_candidates(
        self,
        *,
        query: RetrievalQuery,
        status: RetrievalStatus,
        candidates: tuple[RetrievalCandidate, ...] | list[RetrievalCandidate],
        evidence_pack: RetrievalEvidencePack | None = None,
        metadata: dict[str, object] | None = None,
    ) -> MemoryOrchestratorReport:
        """Create and add a retrieval snapshot from candidates."""

        default_metadata: dict[str, object] = {
            "runtime_source": "memory_orchestrator",
            "operation": "create_retrieval_snapshot_from_candidates",
        }
        snapshot = create_retrieval_snapshot(
            query=query,
            status=status,
            candidates=candidates,
            evidence_pack=evidence_pack,
            metadata=_merge_metadata(default_metadata, metadata),
        )
        return self.add_retrieval_snapshot(snapshot)

    def _apply_consolidation_result(
        self,
        result: ConsolidationResult,
    ) -> MemoryOrchestratorReport:
        if not isinstance(result, ConsolidationResult):
            raise TypeError("result must be ConsolidationResult")
        if result.trace_count() == 0:
            return self._report(
                changed=False,
                reason="consolidation produced no traces",
                metadata={"consolidation_result": result.to_dict()},
            )
        current = self._state.long_term_snapshot
        snapshot = create_long_term_memory_snapshot(
            traces=(*current.traces, *result.traces),
            experiences=(*current.experiences, *result.experiences),
            provenance_checks=current.provenance_checks,
            replay_records=current.replay_records,
            eligibility_decisions=(
                *current.eligibility_decisions,
                *result.eligibility_decisions,
            ),
            metadata={
                "runtime_source": "memory_orchestrator",
                "operation": "consolidation",
            },
        )
        report = self.replace_long_term_snapshot(snapshot)
        return MemoryOrchestratorReport(
            state=report.state,
            short_term_validation=report.short_term_validation,
            long_term_validation=report.long_term_validation,
            changed=True,
            reason="long-term snapshot updated from consolidation",
            metadata={"consolidation_result": result.to_dict()},
        )

    def _validate_short_term_for_update(
        self,
        snapshot: ShortTermMemorySnapshot,
    ) -> ShortTermValidationReport | None:
        if not self._policy.validate_after_update:
            return None
        validation = validate_short_term_snapshot(snapshot=snapshot)
        if not validation.valid and not self._policy.allow_invalid_snapshots:
            raise ValueError("short-term snapshot is invalid")
        return validation

    def _validate_long_term_for_update(
        self,
        snapshot: LongTermMemorySnapshot,
    ) -> LongTermValidationReport | None:
        if not self._policy.validate_after_update:
            return None
        validation = validate_long_term_snapshot(snapshot=snapshot)
        if not validation.valid and not self._policy.allow_invalid_snapshots:
            raise ValueError("long-term snapshot is invalid")
        return validation

    def _state_with(
        self,
        *,
        short_term_snapshot: ShortTermMemorySnapshot | None = None,
        long_term_snapshot: LongTermMemorySnapshot | None = None,
        retrieval_snapshots: tuple[RetrievalSnapshot, ...] | None = None,
    ) -> MemoryOrchestratorState:
        return MemoryOrchestratorState(
            short_term_snapshot=(
                self._state.short_term_snapshot
                if short_term_snapshot is None
                else short_term_snapshot
            ),
            long_term_snapshot=(
                self._state.long_term_snapshot
                if long_term_snapshot is None
                else long_term_snapshot
            ),
            retrieval_snapshots=(
                self._state.retrieval_snapshots
                if retrieval_snapshots is None
                else retrieval_snapshots
            ),
            metadata=self._state.metadata,
        )

    def _report(
        self,
        *,
        changed: bool,
        reason: str,
        short_term_validation: ShortTermValidationReport | None = None,
        long_term_validation: LongTermValidationReport | None = None,
        metadata: dict[str, object] | None = None,
    ) -> MemoryOrchestratorReport:
        if self._policy.validate_after_update:
            if short_term_validation is None:
                short_term_validation = validate_short_term_snapshot(
                    snapshot=self._state.short_term_snapshot
                )
            if long_term_validation is None:
                long_term_validation = validate_long_term_snapshot(
                    snapshot=self._state.long_term_snapshot
                )
        return MemoryOrchestratorReport(
            state=self._state,
            short_term_validation=short_term_validation,
            long_term_validation=long_term_validation,
            changed=changed,
            reason=reason,
            metadata={} if metadata is None else metadata,
        )


def build_memory_orchestrator(
    *,
    state: MemoryOrchestratorState | None = None,
    policy: MemoryOrchestratorPolicy | None = None,
) -> MemoryOrchestrator:
    """Build a memory orchestrator."""

    return MemoryOrchestrator(state=state, policy=policy)


def _empty_state() -> MemoryOrchestratorState:
    return MemoryOrchestratorState(
        short_term_snapshot=create_short_term_memory_snapshot(),
        long_term_snapshot=create_long_term_memory_snapshot(),
        retrieval_snapshots=(),
        metadata={"source": "memory_orchestrator"},
    )


def _ensure_retrieval_snapshots(
    values: tuple[RetrievalSnapshot, ...] | list[RetrievalSnapshot],
) -> tuple[RetrievalSnapshot, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("retrieval_snapshots must be tuple or list")
    snapshots = tuple(values)
    for snapshot in snapshots:
        if not isinstance(snapshot, RetrievalSnapshot):
            raise TypeError("retrieval_snapshots must contain RetrievalSnapshot")
    snapshot_ids = tuple(snapshot.snapshot_id for snapshot in snapshots)
    if len(set(snapshot_ids)) != len(snapshot_ids):
        raise ValueError("retrieval snapshot ids must be unique")
    return snapshots


def _replace_retrieval_snapshot(
    snapshots: tuple[RetrievalSnapshot, ...],
    snapshot: RetrievalSnapshot,
) -> tuple[RetrievalSnapshot, ...]:
    replaced = False
    next_snapshots: list[RetrievalSnapshot] = []
    for current in snapshots:
        if current.snapshot_id == snapshot.snapshot_id:
            next_snapshots.append(snapshot)
            replaced = True
        else:
            next_snapshots.append(current)
    if not replaced:
        next_snapshots.append(snapshot)
    return tuple(next_snapshots)


def _ensure_consolidation_inputs(
    values: tuple[ConsolidationInput, ...] | list[ConsolidationInput],
) -> tuple[ConsolidationInput, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("inputs must be tuple or list")
    inputs = tuple(values)
    for value in inputs:
        if not isinstance(value, ConsolidationInput):
            raise TypeError("inputs must contain ConsolidationInput")
    return inputs


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
