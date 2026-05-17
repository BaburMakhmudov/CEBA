"""Deterministic evidence pack assembly utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from ceba.memory.retrieval.schema import (
    EvidencePackStatus,
    RetrievalCandidate,
    RetrievalEvidenceItem,
    RetrievalEvidencePack,
    RetrievalQuery,
    create_retrieval_evidence_item,
    create_retrieval_evidence_pack,
    ensure_non_empty_string,
    ensure_probability,
)


__all__ = [
    "EvidencePackPolicy",
    "EvidenceItemInput",
    "EvidencePackInput",
    "EvidencePackResult",
    "EvidencePackBuilder",
    "build_evidence_pack",
]


@dataclass(frozen=True)
class EvidencePackPolicy:
    """Configuration object controlling evidence pack assembly."""

    min_relevance: float = 0.0
    min_confidence: float = 0.0
    max_evidence_items: int = 256
    allow_empty_pack: bool = True
    default_status: EvidencePackStatus = EvidencePackStatus.READY

    def __post_init__(self) -> None:
        min_relevance = ensure_probability(self.min_relevance, "min_relevance")
        min_confidence = ensure_probability(
            self.min_confidence,
            "min_confidence",
        )
        if isinstance(self.max_evidence_items, bool) or not isinstance(
            self.max_evidence_items,
            int,
        ):
            raise TypeError("max_evidence_items must be int")
        if self.max_evidence_items <= 0:
            raise ValueError("max_evidence_items must be positive")
        if not isinstance(self.allow_empty_pack, bool):
            raise TypeError("allow_empty_pack must be bool")
        if not isinstance(self.default_status, EvidencePackStatus):
            raise TypeError("default_status must be EvidencePackStatus")
        object.__setattr__(self, "min_relevance", min_relevance)
        object.__setattr__(self, "min_confidence", min_confidence)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "min_relevance": self.min_relevance,
            "min_confidence": self.min_confidence,
            "max_evidence_items": self.max_evidence_items,
            "allow_empty_pack": self.allow_empty_pack,
            "default_status": self.default_status.value,
        }


@dataclass(frozen=True)
class EvidenceItemInput:
    """Explicit input object for creating one RetrievalEvidenceItem."""

    candidate_id: str
    source_id: str
    evidence_ref: str
    relevance: float = 1.0
    confidence: float = 1.0
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        candidate_id = ensure_non_empty_string(
            self.candidate_id,
            "candidate_id",
        )
        _parse_uuid(candidate_id, "candidate_id")
        source_id = ensure_non_empty_string(self.source_id, "source_id")
        evidence_ref = ensure_non_empty_string(
            self.evidence_ref,
            "evidence_ref",
        )
        relevance = ensure_probability(self.relevance, "relevance")
        confidence = ensure_probability(self.confidence, "confidence")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "evidence_ref", evidence_ref)
        object.__setattr__(self, "relevance", relevance)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "candidate_id": self.candidate_id,
            "source_id": self.source_id,
            "evidence_ref": self.evidence_ref,
            "relevance": self.relevance,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EvidencePackInput:
    """Explicit input object for building a RetrievalEvidencePack."""

    query: RetrievalQuery
    candidates: tuple[RetrievalCandidate, ...]
    evidence_inputs: tuple[EvidenceItemInput, ...]
    status: EvidencePackStatus | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.query, RetrievalQuery):
            raise TypeError("query must be RetrievalQuery")
        candidates = _ensure_candidates(self.candidates)
        evidence_inputs = _ensure_evidence_inputs(self.evidence_inputs)
        if self.status is not None and not isinstance(
            self.status,
            EvidencePackStatus,
        ):
            raise TypeError("status must be EvidencePackStatus or None")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "evidence_inputs", evidence_inputs)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "query": self.query.to_dict(),
            "candidates": tuple(
                candidate.to_dict() for candidate in self.candidates
            ),
            "evidence_inputs": tuple(
                evidence_input.to_dict()
                for evidence_input in self.evidence_inputs
            ),
            "status": None if self.status is None else self.status.value,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EvidencePackResult:
    """Immutable result object returned by evidence pack assembly."""

    evidence_pack: RetrievalEvidencePack | None
    evidence_items: tuple[RetrievalEvidenceItem, ...]
    rejected_evidence_refs: tuple[str, ...]
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.evidence_pack is not None and not isinstance(
            self.evidence_pack,
            RetrievalEvidencePack,
        ):
            raise TypeError("evidence_pack must be RetrievalEvidencePack or None")
        evidence_items = _ensure_evidence_items(self.evidence_items)
        rejected_refs = _ensure_string_tuple(
            self.rejected_evidence_refs,
            "rejected_evidence_refs",
        )
        rejection_reasons = _ensure_string_tuple(
            self.rejection_reasons,
            "rejection_reasons",
        )
        if len(rejected_refs) != len(rejection_reasons):
            raise ValueError(
                "rejected_evidence_refs and rejection_reasons must match"
            )
        object.__setattr__(self, "evidence_items", evidence_items)
        object.__setattr__(self, "rejected_evidence_refs", rejected_refs)
        object.__setattr__(self, "rejection_reasons", rejection_reasons)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "evidence_pack": (
                None
                if self.evidence_pack is None
                else self.evidence_pack.to_dict()
            ),
            "evidence_items": tuple(
                item.to_dict() for item in self.evidence_items
            ),
            "rejected_evidence_refs": self.rejected_evidence_refs,
            "rejection_reasons": self.rejection_reasons,
        }

    def evidence_count(self) -> int:
        """Return evidence item count."""

        return len(self.evidence_items)

    def rejected_count(self) -> int:
        """Return rejected evidence count."""

        return len(self.rejected_evidence_refs)

    def has_pack(self) -> bool:
        """Return whether this result contains an evidence pack."""

        return self.evidence_pack is not None


class EvidencePackBuilder:
    """Deterministic service class for assembling evidence packs."""

    def __init__(self, policy: EvidencePackPolicy | None = None) -> None:
        self._policy = EvidencePackPolicy() if policy is None else policy
        if not isinstance(self._policy, EvidencePackPolicy):
            raise TypeError("policy must be EvidencePackPolicy")

    def build(
        self,
        *,
        pack_input: EvidencePackInput,
    ) -> EvidencePackResult:
        """Build a RetrievalEvidencePack from explicit evidence inputs."""

        if not isinstance(pack_input, EvidencePackInput):
            raise TypeError("pack_input must be EvidencePackInput")
        candidates_by_id = {
            str(candidate.candidate_id): candidate
            for candidate in pack_input.candidates
        }
        evidence_items: list[RetrievalEvidenceItem] = []
        rejected_refs: list[str] = []
        rejection_reasons: list[str] = []
        for evidence_input in pack_input.evidence_inputs:
            candidate = candidates_by_id.get(evidence_input.candidate_id)
            reason = self._rejection_reason(evidence_input, candidate)
            if reason is not None:
                rejected_refs.append(evidence_input.evidence_ref)
                rejection_reasons.append(reason)
                continue
            if len(evidence_items) >= self._policy.max_evidence_items:
                rejected_refs.append(evidence_input.evidence_ref)
                rejection_reasons.append("evidence pack limit exceeded")
                continue
            if candidate is None:
                raise RuntimeError("candidate lookup changed during build")
            evidence_items.append(self._create_evidence_item(
                pack_input=pack_input,
                evidence_input=evidence_input,
                candidate=candidate,
            ))

        evidence_values = tuple(evidence_items)
        status = self._status_for_pack(pack_input, evidence_values)
        if not evidence_values and not self._policy.allow_empty_pack:
            return EvidencePackResult(
                evidence_pack=None,
                evidence_items=evidence_values,
                rejected_evidence_refs=tuple(rejected_refs),
                rejection_reasons=tuple(rejection_reasons),
            )
        pack_candidates = _referenced_candidates(
            candidates=pack_input.candidates,
            evidence_items=evidence_values,
        )
        evidence_pack = create_retrieval_evidence_pack(
            query_id=pack_input.query.query_id,
            evidence_items=evidence_values,
            candidates=pack_candidates,
            status=status,
            metadata=self._pack_metadata(pack_input, evidence_values),
        )
        return EvidencePackResult(
            evidence_pack=evidence_pack,
            evidence_items=evidence_values,
            rejected_evidence_refs=tuple(rejected_refs),
            rejection_reasons=tuple(rejection_reasons),
        )

    def _rejection_reason(
        self,
        evidence_input: EvidenceItemInput,
        candidate: RetrievalCandidate | None,
    ) -> str | None:
        if candidate is None:
            return "candidate_id missing"
        if evidence_input.relevance < self._policy.min_relevance:
            return "relevance below evidence minimum"
        if evidence_input.confidence < self._policy.min_confidence:
            return "confidence below evidence minimum"
        return None

    def _create_evidence_item(
        self,
        *,
        pack_input: EvidencePackInput,
        evidence_input: EvidenceItemInput,
        candidate: RetrievalCandidate,
    ) -> RetrievalEvidenceItem:
        metadata = _copy_metadata(evidence_input.metadata)
        metadata.update(_copy_metadata(pack_input.metadata))
        metadata.update(
            {
                "source": "evidence_pack_builder",
                "query_id": str(pack_input.query.query_id),
                "candidate_id": str(candidate.candidate_id),
                "candidate_kind": candidate.kind.value,
                "candidate_source_kind": candidate.source_kind.value,
                "candidate_score": candidate.score,
                "candidate_confidence": candidate.confidence,
            }
        )
        metadata["source"] = "evidence_pack_builder"
        return create_retrieval_evidence_item(
            candidate_id=candidate.candidate_id,
            source_id=evidence_input.source_id,
            evidence_ref=evidence_input.evidence_ref,
            relevance=evidence_input.relevance,
            confidence=evidence_input.confidence,
            metadata=metadata,
        )

    def _status_for_pack(
        self,
        pack_input: EvidencePackInput,
        evidence_items: tuple[RetrievalEvidenceItem, ...],
    ) -> EvidencePackStatus:
        if pack_input.status is not None:
            return pack_input.status
        if not evidence_items:
            return EvidencePackStatus.EMPTY
        return self._policy.default_status

    def _pack_metadata(
        self,
        pack_input: EvidencePackInput,
        evidence_items: tuple[RetrievalEvidenceItem, ...],
    ) -> dict[str, object]:
        metadata = _copy_metadata(pack_input.metadata)
        metadata.update(
            {
                "source": "evidence_pack_builder",
                "query_id": str(pack_input.query.query_id),
                "evidence_count": len(evidence_items),
                "policy": self._policy.to_dict(),
            }
        )
        metadata["source"] = "evidence_pack_builder"
        return metadata


def build_evidence_pack(
    *,
    pack_input: EvidencePackInput,
    policy: EvidencePackPolicy | None = None,
) -> EvidencePackResult:
    """Build a RetrievalEvidencePack from explicit evidence inputs."""

    builder = EvidencePackBuilder(policy)
    return builder.build(pack_input=pack_input)


def _ensure_candidates(
    values: tuple[RetrievalCandidate, ...] | list[RetrievalCandidate],
) -> tuple[RetrievalCandidate, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("candidates must be tuple or list")
    candidates = tuple(values)
    for candidate in candidates:
        if not isinstance(candidate, RetrievalCandidate):
            raise TypeError("candidates must contain RetrievalCandidate")
    candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate_id values must be unique")
    return candidates


def _ensure_evidence_inputs(
    values: tuple[EvidenceItemInput, ...] | list[EvidenceItemInput],
) -> tuple[EvidenceItemInput, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("evidence_inputs must be tuple or list")
    evidence_inputs = tuple(values)
    for evidence_input in evidence_inputs:
        if not isinstance(evidence_input, EvidenceItemInput):
            raise TypeError("evidence_inputs must contain EvidenceItemInput")
    return evidence_inputs


def _ensure_evidence_items(
    values: tuple[RetrievalEvidenceItem, ...] | list[RetrievalEvidenceItem],
) -> tuple[RetrievalEvidenceItem, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("evidence_items must be tuple or list")
    evidence_items = tuple(values)
    for item in evidence_items:
        if not isinstance(item, RetrievalEvidenceItem):
            raise TypeError("evidence_items must contain RetrievalEvidenceItem")
    evidence_ids = tuple(item.evidence_id for item in evidence_items)
    if len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError("evidence_id values must be unique")
    return evidence_items


def _ensure_string_tuple(
    values: tuple[str, ...] | list[str],
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple or list")
    return tuple(ensure_non_empty_string(value, field_name) for value in values)


def _referenced_candidates(
    *,
    candidates: tuple[RetrievalCandidate, ...],
    evidence_items: tuple[RetrievalEvidenceItem, ...],
) -> tuple[RetrievalCandidate, ...]:
    referenced_ids = {item.candidate_id for item in evidence_items}
    return tuple(
        candidate
        for candidate in candidates
        if candidate.candidate_id in referenced_ids
    )


def _parse_uuid(value: str, field_name: str) -> UUID:
    try:
        return UUID(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be UUID string") from exc


def _copy_metadata(metadata: dict[str, object]) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(metadata)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
    return copied
