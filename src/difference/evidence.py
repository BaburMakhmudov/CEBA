"""Evidence construction utilities for CEBA difference processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib

from ois.difference.errors import DifferenceEvidenceError, build_error_context
from ois.difference.schema import (
    DifferenceAnchor,
    DifferenceEvidenceRef,
    DifferenceValue,
    ensure_probability,
    ensure_timezone_aware,
    utc_now,
)


class EvidenceSourceKind(str, Enum):
    """Stable categories for evidence sources."""

    OBSERVED = "OBSERVED"
    EXPECTED = "EXPECTED"
    COMPARISON = "COMPARISON"
    RULE = "RULE"
    TRACE = "TRACE"
    EXTERNAL = "EXTERNAL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class DifferenceEvidenceRecord:
    """Concrete evidence artifact before reduction to a schema reference."""

    evidence_id: str
    source_kind: EvidenceSourceKind
    source_id: str | None = None
    anchor: DifferenceAnchor | None = None
    value: DifferenceValue | None = None
    confidence: float = 1.0
    created_at: datetime = field(default_factory=utc_now)
    note: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_non_empty_string(self.evidence_id, "evidence_id")
        _ensure_source_kind(self.source_kind)
        _ensure_optional_string(self.source_id, "source_id")
        _ensure_optional_instance(self.anchor, DifferenceAnchor, "anchor")
        _ensure_optional_instance(self.value, DifferenceValue, "value")
        object.__setattr__(self, "confidence", ensure_probability(self.confidence, "confidence"))
        ensure_timezone_aware(self.created_at, "created_at")
        _ensure_optional_string(self.note, "note")
        _ensure_metadata_dict(self.metadata)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_ref(self) -> DifferenceEvidenceRef:
        """Return a lightweight schema evidence reference."""

        return DifferenceEvidenceRef(
            evidence_id=self.evidence_id,
            source_id=self.source_id,
            confidence=self.confidence,
            note=self.note,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic, serialization-friendly representation."""

        return {
            "evidence_id": self.evidence_id,
            "source_kind": self.source_kind.value,
            "source_id": self.source_id,
            "anchor": _anchor_to_dict(self.anchor),
            "value": _value_to_dict(self.value),
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "note": self.note,
            "metadata": dict(self.metadata),
        }


def build_evidence_id(
    *,
    source_kind: EvidenceSourceKind,
    source_id: str | None,
    anchor: DifferenceAnchor | None,
    value: DifferenceValue | None,
) -> str:
    """Build a deterministic evidence identifier from stable fields."""

    _ensure_source_kind(source_kind)
    _ensure_optional_string(source_id, "source_id")
    _ensure_optional_instance(anchor, DifferenceAnchor, "anchor")
    _ensure_optional_instance(value, DifferenceValue, "value")

    digest = hashlib.sha256()
    for component in _evidence_id_components(source_kind, source_id, anchor, value):
        digest.update(component.encode("utf-8"))
        digest.update(b"\x00")
    return "ev_" + digest.hexdigest()[:32]


def create_evidence_record(
    *,
    source_kind: EvidenceSourceKind,
    source_id: str | None = None,
    anchor: DifferenceAnchor | None = None,
    value: DifferenceValue | None = None,
    confidence: float = 1.0,
    note: str | None = None,
    metadata: dict[str, object] | None = None,
) -> DifferenceEvidenceRecord:
    """Create a validated evidence record with a deterministic identifier."""

    try:
        resolved_metadata: dict[str, object] = {} if metadata is None else metadata
        evidence_id = build_evidence_id(
            source_kind=source_kind,
            source_id=source_id,
            anchor=anchor,
            value=value,
        )
        return DifferenceEvidenceRecord(
            evidence_id=evidence_id,
            source_kind=source_kind,
            source_id=source_id,
            anchor=anchor,
            value=value,
            confidence=confidence,
            created_at=utc_now(),
            note=note,
            metadata=resolved_metadata,
        )
    except DifferenceEvidenceError:
        raise
    except Exception as exc:
        raise DifferenceEvidenceError(
            message="failed to create evidence record",
            context=build_error_context(operation="create_evidence_record"),
            cause=exc,
        ) from exc


def evidence_ref_from_record(
    record: DifferenceEvidenceRecord,
) -> DifferenceEvidenceRef:
    """Convert a full evidence record into a lightweight schema reference."""

    try:
        if not isinstance(record, DifferenceEvidenceRecord):
            raise TypeError("record must be a DifferenceEvidenceRecord")
        return record.to_ref()
    except DifferenceEvidenceError:
        raise
    except Exception as exc:
        raise DifferenceEvidenceError(
            message="failed to build evidence reference from record",
            context=build_error_context(operation="evidence_ref_from_record"),
            cause=exc,
        ) from exc


def evidence_refs_from_records(
    records: tuple[DifferenceEvidenceRecord, ...] | list[DifferenceEvidenceRecord],
) -> tuple[DifferenceEvidenceRef, ...]:
    """Convert multiple evidence records into schema references."""

    try:
        if not isinstance(records, (tuple, list)):
            raise TypeError("records must be a tuple or list of DifferenceEvidenceRecord")
        return tuple(evidence_ref_from_record(record) for record in records)
    except DifferenceEvidenceError:
        raise
    except Exception as exc:
        raise DifferenceEvidenceError(
            message="failed to build evidence references from records",
            context=build_error_context(operation="evidence_refs_from_records"),
            cause=exc,
        ) from exc


def merge_evidence_metadata(
    base: dict[str, object],
    extra: dict[str, object],
) -> dict[str, object]:
    """Return merged evidence metadata without mutating inputs."""

    try:
        _ensure_metadata_dict(base)
        _ensure_metadata_dict(extra)
        return {**base, **extra}
    except DifferenceEvidenceError:
        raise
    except Exception as exc:
        raise DifferenceEvidenceError(
            message="failed to merge evidence metadata",
            context=build_error_context(operation="merge_evidence_metadata"),
            cause=exc,
        ) from exc


def _evidence_id_components(
    source_kind: EvidenceSourceKind,
    source_id: str | None,
    anchor: DifferenceAnchor | None,
    value: DifferenceValue | None,
) -> tuple[str, ...]:
    components: list[str] = [
        source_kind.value,
        source_id or "",
    ]

    if anchor is None:
        components.extend(["", "", repr(()), repr(None), repr(None), repr(None)])
    else:
        components.extend(
            [
                anchor.anchor_id,
                anchor.source_id or "",
                repr(anchor.field_path),
                repr(anchor.span_start),
                repr(anchor.span_end),
                repr(anchor.label),
            ]
        )

    if value is None:
        components.extend([repr(None), repr(None), repr(None)])
    else:
        components.extend(
            [
                repr(value.raw_value),
                repr(value.normalized_value),
                repr(value.value_type),
            ]
        )

    return tuple(components)


def _anchor_to_dict(anchor: DifferenceAnchor | None) -> dict[str, object] | None:
    if anchor is None:
        return None
    return {
        "anchor_id": anchor.anchor_id,
        "source_id": anchor.source_id,
        "field_path": list(anchor.field_path),
        "span_start": anchor.span_start,
        "span_end": anchor.span_end,
        "label": anchor.label,
    }


def _value_to_dict(value: DifferenceValue | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "raw_value": value.raw_value,
        "normalized_value": value.normalized_value,
        "value_type": value.value_type,
    }


def _ensure_source_kind(value: EvidenceSourceKind) -> EvidenceSourceKind:
    if not isinstance(value, EvidenceSourceKind):
        raise TypeError("source_kind must be an EvidenceSourceKind")
    return value


def _ensure_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(field_name, str) or not field_name:
        raise ValueError("field_name must be a non-empty string")
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _ensure_optional_string(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return _ensure_non_empty_string(value, field_name)


def _ensure_optional_instance(
    value: object | None,
    expected_type: type[object],
    field_name: str,
) -> object | None:
    if value is None:
        return None
    if not isinstance(value, expected_type):
        raise TypeError(f"{field_name} must be {expected_type.__name__} or None")
    return value


def _ensure_metadata_dict(value: dict[str, object]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("metadata must be a dict")
    if not all(isinstance(key, str) for key in value):
        raise TypeError("metadata keys must be strings")
    return value


__all__ = [
    "EvidenceSourceKind",
    "DifferenceEvidenceRecord",
    "build_evidence_id",
    "create_evidence_record",
    "evidence_ref_from_record",
    "evidence_refs_from_records",
    "merge_evidence_metadata",
]
