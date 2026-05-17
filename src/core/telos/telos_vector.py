"""Deterministic telos vector utilities for CEBA Core Telos Processing."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

from ceba.core.metrics.schema import (
    ensure_constraint_probability,
    ensure_finite_number,
    ensure_non_negative_number,
    ensure_positive_int,
)
from ceba.core.telos.schema import (
    TelosVectorKind,
    TelosVectorRef,
    ensure_non_empty_string,
)


DEFAULT_VECTOR_TOLERANCE = 1e-12


@dataclass(frozen=True)
class TelosVectorPayload:
    """Small immutable vector payload for telos-layer computations."""

    vector_ref: TelosVectorRef
    values: tuple[float, ...]
    norm: float
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.vector_ref, TelosVectorRef):
            raise TypeError("vector_ref must be TelosVectorRef")
        values = ensure_vector_values(
            self.values,
            dimension=self.vector_ref.dimension,
            field_name="values",
        )
        norm = ensure_non_negative_number(self.norm, field_name="norm")
        expected_norm = euclidean_norm(values)
        if abs(norm - expected_norm) > DEFAULT_VECTOR_TOLERANCE:
            raise ValueError("norm must match euclidean_norm(values)")

        object.__setattr__(self, "values", values)
        object.__setattr__(self, "norm", norm)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "vector_ref": self.vector_ref.to_dict(),
            "values": self.values,
            "norm": self.norm,
            "metadata": dict(self.metadata),
        }


def ensure_vector_values(
    values: tuple[float, ...] | list[float],
    *,
    dimension: int | None = None,
    field_name: str = "values",
) -> tuple[float, ...]:
    """Validate finite vector values."""

    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple[float, ...] or list[float]")
    vector = tuple(
        ensure_finite_number(value, field_name=field_name) for value in values
    )
    if not vector:
        raise ValueError(f"{field_name} must be non-empty")
    if dimension is not None:
        dimension = ensure_positive_int(dimension, field_name="dimension")
        if len(vector) != dimension:
            raise ValueError(f"{field_name} length must equal dimension")
    return vector


def euclidean_norm(values: tuple[float, ...] | list[float]) -> float:
    """Return Euclidean norm for finite vector values."""

    vector = ensure_vector_values(values)
    return ensure_non_negative_number(
        math.sqrt(sum(value * value for value in vector)),
        field_name="euclidean_norm",
    )


def normalize_vector(
    values: tuple[float, ...] | list[float],
    *,
    tolerance: float = DEFAULT_VECTOR_TOLERANCE,
) -> tuple[float, ...]:
    """Normalize finite vector values."""

    vector = ensure_vector_values(values)
    tolerance = ensure_non_negative_number(tolerance, field_name="tolerance")
    norm = euclidean_norm(vector)
    if norm <= tolerance:
        raise ValueError("cannot normalize vector with norm below tolerance")
    return tuple(value / norm for value in vector)


def dot_product(
    left: tuple[float, ...] | list[float],
    right: tuple[float, ...] | list[float],
) -> float:
    """Return dot product for two finite vectors."""

    left_values = ensure_vector_values(left, field_name="left")
    right_values = ensure_vector_values(right, field_name="right")
    if len(left_values) != len(right_values):
        raise ValueError("vectors must have the same dimension")
    value = sum(
        left_value * right_value
        for left_value, right_value in zip(left_values, right_values)
    )
    return ensure_finite_number(
        value,
        field_name="dot_product",
    )


def cosine_similarity(
    left: tuple[float, ...] | list[float],
    right: tuple[float, ...] | list[float],
    *,
    tolerance: float = DEFAULT_VECTOR_TOLERANCE,
) -> float:
    """Return cosine similarity clamped to -1.0..1.0."""

    left_values = ensure_vector_values(left, field_name="left")
    right_values = ensure_vector_values(right, field_name="right")
    if len(left_values) != len(right_values):
        raise ValueError("vectors must have the same dimension")
    tolerance = ensure_non_negative_number(tolerance, field_name="tolerance")
    left_norm = euclidean_norm(left_values)
    right_norm = euclidean_norm(right_values)
    if left_norm <= tolerance or right_norm <= tolerance:
        return 0.0
    cosine = dot_product(left_values, right_values) / (left_norm * right_norm)
    cosine = ensure_finite_number(cosine, field_name="cosine_similarity")
    return min(max(cosine, -1.0), 1.0)


def cosine_alignment_score(
    left: tuple[float, ...] | list[float],
    right: tuple[float, ...] | list[float],
    *,
    tolerance: float = DEFAULT_VECTOR_TOLERANCE,
) -> float:
    """Convert cosine similarity into a 0.0..1.0 alignment score."""

    cosine = cosine_similarity(left, right, tolerance=tolerance)
    return ensure_constraint_probability(
        (cosine + 1.0) / 2.0,
        field_name="cosine_alignment_score",
    )


def build_telos_vector_id(
    *,
    kind: TelosVectorKind,
    dimension: int,
    label: str | None = None,
) -> str:
    """Build deterministic lightweight vector id."""

    if not isinstance(kind, TelosVectorKind):
        raise TypeError("kind must be TelosVectorKind")
    dimension = ensure_positive_int(dimension, field_name="dimension")
    label = None if label is None else ensure_non_empty_string(label, "label")
    digest_input = f"{kind.value}:{dimension}"
    if label is not None:
        digest_input = f"{digest_input}:{label}"
    digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
    return f"telos_vec_{digest[:24]}"


def create_telos_vector_ref(
    *,
    kind: TelosVectorKind,
    dimension: int,
    normalized: bool = False,
    vector_id: str | None = None,
    label: str | None = None,
    metadata: dict[str, object] | None = None,
) -> TelosVectorRef:
    """Create validated TelosVectorRef."""

    if not isinstance(kind, TelosVectorKind):
        raise TypeError("kind must be TelosVectorKind")
    dimension = ensure_positive_int(dimension, field_name="dimension")
    if not isinstance(normalized, bool):
        raise TypeError("normalized must be bool")
    if vector_id is None:
        vector_id = build_telos_vector_id(
            kind=kind,
            dimension=dimension,
            label=label,
        )
    else:
        vector_id = ensure_non_empty_string(vector_id, "vector_id")
    label = None if label is None else ensure_non_empty_string(label, "label")

    default_metadata: dict[str, object] = {
        "source": "telos_vector_ref",
        "label": label,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "telos_vector_ref"

    return TelosVectorRef(
        vector_id=vector_id,
        kind=kind,
        dimension=dimension,
        normalized=normalized,
        metadata=merged_metadata,
    )


def create_telos_vector_payload(
    *,
    values: tuple[float, ...] | list[float],
    kind: TelosVectorKind = TelosVectorKind.TARGET,
    normalize: bool = False,
    vector_id: str | None = None,
    label: str | None = None,
    metadata: dict[str, object] | None = None,
) -> TelosVectorPayload:
    """Create validated telos vector payload."""

    if not isinstance(kind, TelosVectorKind):
        raise TypeError("kind must be TelosVectorKind")
    if not isinstance(normalize, bool):
        raise TypeError("normalize must be bool")
    vector_values = ensure_vector_values(values)
    if normalize:
        vector_values = normalize_vector(vector_values)

    vector_ref = create_telos_vector_ref(
        kind=kind,
        dimension=len(vector_values),
        normalized=normalize,
        vector_id=vector_id,
        label=label,
    )
    default_metadata: dict[str, object] = {
        "source": "telos_vector_payload",
        "normalized": normalize,
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "telos_vector_payload"

    return TelosVectorPayload(
        vector_ref=vector_ref,
        values=vector_values,
        norm=euclidean_norm(vector_values),
        metadata=merged_metadata,
    )


def payload_from_ref_and_values(
    *,
    vector_ref: TelosVectorRef,
    values: tuple[float, ...] | list[float],
    normalize: bool = False,
    metadata: dict[str, object] | None = None,
) -> TelosVectorPayload:
    """Create payload from an existing TelosVectorRef and values."""

    if not isinstance(vector_ref, TelosVectorRef):
        raise TypeError("vector_ref must be TelosVectorRef")
    if not isinstance(normalize, bool):
        raise TypeError("normalize must be bool")
    vector_values = ensure_vector_values(values, dimension=vector_ref.dimension)
    if normalize:
        vector_values = normalize_vector(vector_values)

    default_metadata: dict[str, object] = {
        "source": "telos_vector_payload_from_ref",
    }
    merged_metadata = _merge_metadata(default_metadata, metadata)
    merged_metadata["source"] = "telos_vector_payload_from_ref"

    return TelosVectorPayload(
        vector_ref=vector_ref,
        values=vector_values,
        norm=euclidean_norm(vector_values),
        metadata=merged_metadata,
    )


def vector_payloads_compatible(
    left: TelosVectorPayload,
    right: TelosVectorPayload,
) -> bool:
    """Return True when two vector payload dimensions match."""

    if not isinstance(left, TelosVectorPayload):
        raise TypeError("left must be TelosVectorPayload")
    if not isinstance(right, TelosVectorPayload):
        raise TypeError("right must be TelosVectorPayload")
    return left.vector_ref.dimension == right.vector_ref.dimension


def telos_vector_summary(
    payloads: tuple[TelosVectorPayload, ...] | list[TelosVectorPayload],
) -> dict[str, object]:
    """Summarize telos vector payloads."""

    values = tuple(payloads)
    for payload in values:
        if not isinstance(payload, TelosVectorPayload):
            raise TypeError("payloads must contain TelosVectorPayload")

    dimensions: dict[int, int] = {}
    kinds: dict[str, int] = {}
    normalized_count = 0
    total_norm = 0.0
    for payload in values:
        dimension = payload.vector_ref.dimension
        kind = payload.vector_ref.kind.value
        dimensions[dimension] = dimensions.get(dimension, 0) + 1
        kinds[kind] = kinds.get(kind, 0) + 1
        if payload.vector_ref.normalized:
            normalized_count += 1
        total_norm += payload.norm

    average_norm = 0.0 if not values else total_norm / len(values)
    return {
        "count": len(values),
        "dimensions": dimensions,
        "kinds": kinds,
        "normalized_count": normalized_count,
        "average_norm": ensure_non_negative_number(
            average_norm,
            field_name="average_norm",
        ),
    }


def _copy_metadata(metadata: dict[str, object]) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(metadata)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
    return copied


def _merge_metadata(
    default_metadata: dict[str, object],
    caller_metadata: dict[str, object] | None,
) -> dict[str, object]:
    merged = _copy_metadata(default_metadata)
    if caller_metadata is not None:
        merged.update(_copy_metadata(caller_metadata))
    return merged


__all__ = [
    "DEFAULT_VECTOR_TOLERANCE",
    "TelosVectorPayload",
    "ensure_vector_values",
    "euclidean_norm",
    "normalize_vector",
    "dot_product",
    "cosine_similarity",
    "cosine_alignment_score",
    "build_telos_vector_id",
    "create_telos_vector_ref",
    "create_telos_vector_payload",
    "payload_from_ref_and_values",
    "vector_payloads_compatible",
    "telos_vector_summary",
]
