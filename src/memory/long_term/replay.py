"""Deterministic replay utilities for CEBA Long-Term Memory."""

from __future__ import annotations

from dataclasses import dataclass

from ceba.memory.long_term.schema import (
    ConsolidatedExperience,
    LongTermTrace,
    ReplayRecord,
    ReplayStatus,
    create_replay_record,
    ensure_probability,
    utc_now,
)


__all__ = [
    "ReplayPolicy",
    "ReplaySelection",
    "ReplayManager",
    "create_trace_replay_record",
    "create_experience_replay_record",
    "mark_replay_used",
    "select_replay_records",
]


@dataclass(frozen=True)
class ReplayPolicy:
    """Configuration object controlling deterministic replay behavior."""

    min_replay_score: float = 0.0
    max_replay_records: int = 128
    replay_score_confidence_weight: float = 0.60
    replay_score_priority_weight: float = 0.40

    def __post_init__(self) -> None:
        min_replay_score = ensure_probability(
            self.min_replay_score,
            "min_replay_score",
        )
        if isinstance(self.max_replay_records, bool) or not isinstance(
            self.max_replay_records,
            int,
        ):
            raise TypeError("max_replay_records must be int")
        if self.max_replay_records <= 0:
            raise ValueError("max_replay_records must be positive")
        confidence_weight = ensure_probability(
            self.replay_score_confidence_weight,
            "replay_score_confidence_weight",
        )
        priority_weight = ensure_probability(
            self.replay_score_priority_weight,
            "replay_score_priority_weight",
        )
        if confidence_weight + priority_weight <= 0.0:
            raise ValueError("replay score weights must sum to > 0.0")
        object.__setattr__(self, "min_replay_score", min_replay_score)
        object.__setattr__(
            self,
            "replay_score_confidence_weight",
            confidence_weight,
        )
        object.__setattr__(
            self,
            "replay_score_priority_weight",
            priority_weight,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "min_replay_score": self.min_replay_score,
            "max_replay_records": self.max_replay_records,
            "replay_score_confidence_weight": (
                self.replay_score_confidence_weight
            ),
            "replay_score_priority_weight": self.replay_score_priority_weight,
        }


@dataclass(frozen=True)
class ReplaySelection:
    """Immutable result object for replay selection."""

    records: tuple[ReplayRecord, ...]
    skipped_ids: tuple[str, ...]
    skip_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        records = _ensure_records(self.records)
        skipped_ids = _ensure_string_tuple(self.skipped_ids, "skipped_ids")
        skip_reasons = _ensure_string_tuple(
            self.skip_reasons,
            "skip_reasons",
        )
        if len(skipped_ids) != len(skip_reasons):
            raise ValueError("skipped_ids and skip_reasons must match")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "skipped_ids", skipped_ids)
        object.__setattr__(self, "skip_reasons", skip_reasons)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "records": tuple(record.to_dict() for record in self.records),
            "skipped_ids": self.skipped_ids,
            "skip_reasons": self.skip_reasons,
        }

    def record_count(self) -> int:
        """Return selected record count."""

        return len(self.records)

    def skipped_count(self) -> int:
        """Return skipped record count."""

        return len(self.skipped_ids)


class ReplayManager:
    """Deterministic service for creating and updating replay records."""

    def __init__(self, policy: ReplayPolicy | None = None) -> None:
        self._policy = ReplayPolicy() if policy is None else policy
        if not isinstance(self._policy, ReplayPolicy):
            raise TypeError("policy must be ReplayPolicy")

    def create_for_trace(
        self,
        *,
        trace: LongTermTrace,
        metadata: dict[str, object] | None = None,
    ) -> ReplayRecord:
        """Create a replay record for a long-term trace."""

        if not isinstance(trace, LongTermTrace):
            raise TypeError("trace must be LongTermTrace")
        replay_score = self._score(
            confidence=trace.confidence,
            priority=trace.priority,
        )
        status = self._status_for_score(replay_score)
        replay_metadata = _merge_metadata(
            {
                "source": "replay_manager",
                "trace_id": str(trace.trace_id),
                "trace_kind": trace.kind.value,
                "trace_status": trace.status.value,
            },
            metadata,
        )
        replay_metadata["source"] = "replay_manager"
        return create_replay_record(
            trace_id=trace.trace_id,
            experience_id=None,
            status=status,
            replay_score=replay_score,
            used_at=None,
            metadata=replay_metadata,
        )

    def create_for_experience(
        self,
        *,
        experience: ConsolidatedExperience,
        metadata: dict[str, object] | None = None,
    ) -> ReplayRecord:
        """Create a replay record for a consolidated experience."""

        if not isinstance(experience, ConsolidatedExperience):
            raise TypeError("experience must be ConsolidatedExperience")
        replay_score = self._score(
            confidence=experience.confidence,
            priority=experience.utility_score,
        )
        status = self._status_for_score(replay_score)
        replay_metadata = _merge_metadata(
            {
                "source": "replay_manager",
                "experience_id": str(experience.experience_id),
                "experience_status": experience.status.value,
            },
            metadata,
        )
        replay_metadata["source"] = "replay_manager"
        return create_replay_record(
            trace_id=None,
            experience_id=experience.experience_id,
            status=status,
            replay_score=replay_score,
            used_at=None,
            metadata=replay_metadata,
        )

    def mark_used(
        self,
        *,
        record: ReplayRecord,
        metadata: dict[str, object] | None = None,
    ) -> ReplayRecord:
        """Return a replay record marked as used."""

        if not isinstance(record, ReplayRecord):
            raise TypeError("record must be ReplayRecord")
        replay_metadata = _merge_metadata(record.metadata, metadata)
        replay_metadata.update(
            {
                "source": "replay_manager",
                "last_operation": "mark_used",
            }
        )
        replay_metadata["source"] = "replay_manager"
        return ReplayRecord(
            replay_id=record.replay_id,
            trace_id=record.trace_id,
            experience_id=record.experience_id,
            status=ReplayStatus.USED,
            replay_score=record.replay_score,
            used_at=utc_now(),
            created_at=record.created_at,
            metadata=replay_metadata,
        )

    def set_status(
        self,
        *,
        record: ReplayRecord,
        status: ReplayStatus,
        metadata: dict[str, object] | None = None,
    ) -> ReplayRecord:
        """Return a replay record with an updated status."""

        if not isinstance(record, ReplayRecord):
            raise TypeError("record must be ReplayRecord")
        if not isinstance(status, ReplayStatus):
            raise TypeError("status must be ReplayStatus")
        replay_metadata = _merge_metadata(record.metadata, metadata)
        replay_metadata.update(
            {
                "source": "replay_manager",
                "last_operation": "set_status",
            }
        )
        replay_metadata["source"] = "replay_manager"
        return ReplayRecord(
            replay_id=record.replay_id,
            trace_id=record.trace_id,
            experience_id=record.experience_id,
            status=status,
            replay_score=record.replay_score,
            used_at=record.used_at,
            created_at=record.created_at,
            metadata=replay_metadata,
        )

    def select_records(
        self,
        *,
        records: tuple[ReplayRecord, ...] | list[ReplayRecord],
    ) -> ReplaySelection:
        """Select replay records deterministically."""

        values = _ensure_records(records)
        selected: list[ReplayRecord] = []
        skipped_ids: list[str] = []
        skip_reasons: list[str] = []
        for record in values:
            if record.status is not ReplayStatus.ELIGIBLE:
                skipped_ids.append(str(record.replay_id))
                skip_reasons.append("record not eligible")
            elif record.replay_score < self._policy.min_replay_score:
                skipped_ids.append(str(record.replay_id))
                skip_reasons.append("replay_score below minimum")
            else:
                selected.append(record)

        selected.sort(
            key=lambda record: (
                -record.replay_score,
                -record.created_at.timestamp(),
                str(record.replay_id),
            )
        )
        limited = selected[: self._policy.max_replay_records]
        overflow = selected[self._policy.max_replay_records :]
        for record in overflow:
            skipped_ids.append(str(record.replay_id))
            skip_reasons.append("replay selection limit exceeded")
        return ReplaySelection(
            records=tuple(limited),
            skipped_ids=tuple(skipped_ids),
            skip_reasons=tuple(skip_reasons),
        )

    def _score(self, *, confidence: float, priority: float) -> float:
        numerator = (
            confidence * self._policy.replay_score_confidence_weight
            + priority * self._policy.replay_score_priority_weight
        )
        denominator = (
            self._policy.replay_score_confidence_weight
            + self._policy.replay_score_priority_weight
        )
        return ensure_probability(numerator / denominator, "replay_score")

    def _status_for_score(self, replay_score: float) -> ReplayStatus:
        if replay_score >= self._policy.min_replay_score:
            return ReplayStatus.ELIGIBLE
        return ReplayStatus.INELIGIBLE


def create_trace_replay_record(
    *,
    trace: LongTermTrace,
    policy: ReplayPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> ReplayRecord:
    """Create a replay record for a long-term trace."""

    manager = ReplayManager(policy)
    return manager.create_for_trace(trace=trace, metadata=metadata)


def create_experience_replay_record(
    *,
    experience: ConsolidatedExperience,
    policy: ReplayPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> ReplayRecord:
    """Create a replay record for a consolidated experience."""

    manager = ReplayManager(policy)
    return manager.create_for_experience(
        experience=experience,
        metadata=metadata,
    )


def mark_replay_used(
    *,
    record: ReplayRecord,
    policy: ReplayPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> ReplayRecord:
    """Return a replay record marked as used."""

    manager = ReplayManager(policy)
    return manager.mark_used(record=record, metadata=metadata)


def select_replay_records(
    *,
    records: tuple[ReplayRecord, ...] | list[ReplayRecord],
    policy: ReplayPolicy | None = None,
) -> ReplaySelection:
    """Select replay records deterministically."""

    manager = ReplayManager(policy)
    return manager.select_records(records=records)


def _ensure_records(
    values: tuple[ReplayRecord, ...] | list[ReplayRecord],
) -> tuple[ReplayRecord, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("records must be tuple or list")
    records = tuple(values)
    for record in records:
        if not isinstance(record, ReplayRecord):
            raise TypeError("records must contain ReplayRecord")
    replay_ids = tuple(record.replay_id for record in records)
    if len(set(replay_ids)) != len(replay_ids):
        raise ValueError("replay_id values must be unique")
    return records


def _ensure_string_tuple(
    values: tuple[str, ...] | list[str],
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple or list")
    result = tuple(values)
    for value in result:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must contain str")
        if value == "":
            raise ValueError(f"{field_name} must contain non-empty strings")
    return result


def _copy_metadata(metadata: dict[str, object]) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(metadata)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
    return copied


def _merge_metadata(
    base_metadata: dict[str, object],
    caller_metadata: dict[str, object] | None,
) -> dict[str, object]:
    merged = _copy_metadata(base_metadata)
    if caller_metadata is not None:
        merged.update(_copy_metadata(caller_metadata))
    return merged
