"""Deterministic normalization utilities for CEBA difference processing."""

from __future__ import annotations

import math
import re
import unicodedata

from ois.difference.errors import DifferenceNormalizationError, build_error_context
from ois.difference.schema import (
    DifferenceAnchor,
    DifferenceCandidate,
    DifferenceEvidenceRef,
    DifferenceSet,
    DifferenceStatus,
    DifferenceValue,
    ensure_non_negative,
    ensure_probability,
    ensure_timezone_aware,
)


MAX_STRING_LENGTH = 10000
_SPACES_AND_TABS = re.compile(r"[ \t]+")


def normalize_text_value(value: str) -> str:
    """Normalize text without semantic interpretation."""

    if not isinstance(value, str):
        raise TypeError("value must be a string")

    normalized = unicodedata.normalize("NFC", value)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.strip()
    normalized = "\n".join(_SPACES_AND_TABS.sub(" ", line) for line in normalized.split("\n"))

    if len(normalized) > MAX_STRING_LENGTH:
        raise ValueError("normalized string exceeds MAX_STRING_LENGTH")

    return normalized


def normalize_scalar_value(value: object) -> object:
    """Normalize scalar values safely."""

    if isinstance(value, str):
        return normalize_text_value(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError("float value must not be NaN or infinity")
        return float(value)
    if value is None:
        return None
    return value


def normalize_container_value(value: object) -> object:
    """Normalize simple JSON-like containers recursively."""

    if isinstance(value, dict):
        normalized_items: list[tuple[str, object]] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("container dict keys must be strings")
            normalized_items.append(
                (
                    normalize_text_value(key),
                    normalize_container_value(item),
                )
            )
        return dict(sorted(normalized_items, key=lambda item: item[0]))

    if isinstance(value, (list, tuple)):
        return tuple(normalize_container_value(item) for item in value)

    return normalize_scalar_value(value)


def normalize_difference_value(value: DifferenceValue | None) -> DifferenceValue | None:
    """Normalize a DifferenceValue without changing raw_value."""

    if value is None:
        return None
    if not isinstance(value, DifferenceValue):
        raise TypeError("value must be a DifferenceValue or None")

    source_value = value.normalized_value if value.normalized_value is not None else value.raw_value
    normalized_value = normalize_container_value(source_value)
    value_type = normalize_text_value(value.value_type) if value.value_type is not None else None

    return DifferenceValue(
        raw_value=value.raw_value,
        normalized_value=normalized_value,
        value_type=value_type,
    )


def normalize_anchor(anchor: DifferenceAnchor | None) -> DifferenceAnchor | None:
    """Normalize anchor metadata."""

    if anchor is None:
        return None
    if not isinstance(anchor, DifferenceAnchor):
        raise TypeError("anchor must be a DifferenceAnchor or None")

    return DifferenceAnchor(
        anchor_id=normalize_text_value(anchor.anchor_id),
        source_id=normalize_text_value(anchor.source_id) if anchor.source_id is not None else None,
        field_path=tuple(normalize_text_value(part) for part in anchor.field_path),
        span_start=anchor.span_start,
        span_end=anchor.span_end,
        label=normalize_text_value(anchor.label) if anchor.label is not None else None,
    )


def normalize_evidence_ref(ref: DifferenceEvidenceRef) -> DifferenceEvidenceRef:
    """Normalize evidence reference metadata."""

    if not isinstance(ref, DifferenceEvidenceRef):
        raise TypeError("ref must be a DifferenceEvidenceRef")

    return DifferenceEvidenceRef(
        evidence_id=normalize_text_value(ref.evidence_id),
        source_id=normalize_text_value(ref.source_id) if ref.source_id is not None else None,
        confidence=ensure_probability(ref.confidence, "confidence"),
        note=normalize_text_value(ref.note) if ref.note is not None else None,
    )


def normalize_metadata(metadata: dict[str, object]) -> dict[str, object]:
    """Normalize metadata without mutating the original mapping."""

    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dict")

    normalized_items: list[tuple[str, object]] = []
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
        normalized_items.append(
            (
                normalize_text_value(key),
                normalize_container_value(value),
            )
        )

    return dict(sorted(normalized_items, key=lambda item: item[0]))


def normalize_candidate(candidate: DifferenceCandidate) -> DifferenceCandidate:
    """Return a normalized copy of a DifferenceCandidate."""

    difference_id: str | None = None
    try:
        if isinstance(candidate, DifferenceCandidate):
            difference_id = str(candidate.difference_id)
        else:
            raise TypeError("candidate must be a DifferenceCandidate")

        status = (
            DifferenceStatus.REJECTED
            if candidate.status is DifferenceStatus.REJECTED
            else DifferenceStatus.NORMALIZED
        )

        return DifferenceCandidate(
            difference_id=candidate.difference_id,
            kind=candidate.kind,
            status=status,
            direction=candidate.direction,
            left_anchor=normalize_anchor(candidate.left_anchor),
            right_anchor=normalize_anchor(candidate.right_anchor),
            left_value=normalize_difference_value(candidate.left_value),
            right_value=normalize_difference_value(candidate.right_value),
            evidence_refs=tuple(
                normalize_evidence_ref(ref) for ref in candidate.evidence_refs
            ),
            magnitude=ensure_non_negative(candidate.magnitude, "magnitude"),
            confidence=ensure_probability(candidate.confidence, "confidence"),
            created_at=ensure_timezone_aware(candidate.created_at, "created_at"),
            metadata=normalize_metadata(candidate.metadata),
        )
    except DifferenceNormalizationError:
        raise
    except Exception as exc:
        raise DifferenceNormalizationError(
            message="failed to normalize difference candidate",
            context=build_error_context(
                operation="normalize_candidate",
                difference_id=difference_id,
            ),
            cause=exc,
        ) from exc


def normalize_difference_set(difference_set: DifferenceSet) -> DifferenceSet:
    """Return a normalized copy of a DifferenceSet."""

    observation_id: str | None = None
    try:
        if isinstance(difference_set, DifferenceSet):
            observation_id = difference_set.observation_id
        else:
            raise TypeError("difference_set must be a DifferenceSet")

        return DifferenceSet(
            set_id=difference_set.set_id,
            observation_id=normalize_text_value(difference_set.observation_id),
            candidates=tuple(
                normalize_candidate(candidate) for candidate in difference_set.candidates
            ),
            created_at=ensure_timezone_aware(difference_set.created_at, "created_at"),
            metadata=normalize_metadata(difference_set.metadata),
        )
    except DifferenceNormalizationError:
        raise
    except Exception as exc:
        raise DifferenceNormalizationError(
            message="failed to normalize difference set",
            context=build_error_context(
                operation="normalize_difference_set",
                observation_id=observation_id,
            ),
            cause=exc,
        ) from exc


def is_normalized_candidate(candidate: DifferenceCandidate) -> bool:
    """Return whether a candidate has NORMALIZED status."""

    if not isinstance(candidate, DifferenceCandidate):
        raise TypeError("candidate must be a DifferenceCandidate")
    return candidate.status is DifferenceStatus.NORMALIZED


__all__ = [
    "MAX_STRING_LENGTH",
    "normalize_text_value",
    "normalize_scalar_value",
    "normalize_container_value",
    "normalize_difference_value",
    "normalize_anchor",
    "normalize_evidence_ref",
    "normalize_metadata",
    "normalize_candidate",
    "normalize_difference_set",
    "is_normalized_candidate",
]
