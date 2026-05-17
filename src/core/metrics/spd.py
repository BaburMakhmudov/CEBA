"""Deterministic SPD matrix utilities for CEBA Core Metric Processing."""

from __future__ import annotations

from dataclasses import dataclass, field

from ceba.core.metrics.schema import (
    MetricSpaceKind,
    MetricTensorRef,
    MetricTensorStatus,
    ensure_finite_number,
    ensure_non_negative_number,
    ensure_positive_int,
)


DEFAULT_SYMMETRY_TOLERANCE = 1e-9
DEFAULT_POSITIVE_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class SPDCheckResult:
    is_square: bool
    is_symmetric: bool
    is_positive_diagonal: bool
    is_strictly_diagonally_dominant: bool
    is_spd_candidate: bool
    dimension: int | None
    min_diagonal: float | None
    max_abs_off_diagonal: float | None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "is_square",
            "is_symmetric",
            "is_positive_diagonal",
            "is_strictly_diagonally_dominant",
            "is_spd_candidate",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")
        if self.dimension is not None:
            object.__setattr__(self, "dimension", ensure_positive_int(self.dimension, "dimension"))
        if self.min_diagonal is not None:
            object.__setattr__(self, "min_diagonal", ensure_finite_number(self.min_diagonal, "min_diagonal"))
        if self.max_abs_off_diagonal is not None:
            object.__setattr__(
                self,
                "max_abs_off_diagonal",
                ensure_non_negative_number(self.max_abs_off_diagonal, "max_abs_off_diagonal"),
            )
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "is_square": self.is_square,
            "is_symmetric": self.is_symmetric,
            "is_positive_diagonal": self.is_positive_diagonal,
            "is_strictly_diagonally_dominant": self.is_strictly_diagonally_dominant,
            "is_spd_candidate": self.is_spd_candidate,
            "dimension": self.dimension,
            "min_diagonal": self.min_diagonal,
            "max_abs_off_diagonal": self.max_abs_off_diagonal,
            "metadata": dict(self.metadata),
        }


def ensure_matrix_shape(matrix: tuple[tuple[float, ...], ...] | list[list[float]]) -> tuple[tuple[float, ...], ...]:
    if not isinstance(matrix, (tuple, list)):
        raise TypeError("matrix must be tuple[tuple[float, ...], ...] or list[list[float]]")
    if not matrix:
        raise ValueError("matrix must not be empty")

    frozen_rows: list[tuple[float, ...]] = []
    expected_width: int | None = None
    for row_index, row in enumerate(matrix):
        if not isinstance(row, (tuple, list)):
            raise TypeError("matrix rows must be tuple or list")
        if expected_width is None:
            expected_width = len(row)
            if expected_width == 0:
                raise ValueError("matrix rows must not be empty")
        elif len(row) != expected_width:
            raise ValueError("all matrix rows must have the same length")

        frozen_row = tuple(
            ensure_finite_number(value, f"matrix[{row_index}][{column_index}]")
            for column_index, value in enumerate(row)
        )
        frozen_rows.append(frozen_row)

    return tuple(frozen_rows)


def matrix_dimension(matrix: tuple[tuple[float, ...], ...]) -> int:
    if not is_square_matrix(matrix):
        raise ValueError("matrix must be square")
    return len(matrix)


def is_square_matrix(matrix: tuple[tuple[float, ...], ...]) -> bool:
    _ensure_frozen_matrix(matrix)
    return all(len(row) == len(matrix) for row in matrix)


def is_symmetric_matrix(
    matrix: tuple[tuple[float, ...], ...],
    *,
    tolerance: float = DEFAULT_SYMMETRY_TOLERANCE,
) -> bool:
    _ensure_frozen_matrix(matrix)
    tolerance_value = ensure_non_negative_number(tolerance, "tolerance")
    if not is_square_matrix(matrix):
        return False
    dimension = len(matrix)
    for row_index in range(dimension):
        for column_index in range(row_index + 1, dimension):
            if abs(matrix[row_index][column_index] - matrix[column_index][row_index]) > tolerance_value:
                return False
    return True


def positive_diagonal(
    matrix: tuple[tuple[float, ...], ...],
    *,
    tolerance: float = DEFAULT_POSITIVE_TOLERANCE,
) -> bool:
    _ensure_frozen_matrix(matrix)
    tolerance_value = ensure_non_negative_number(tolerance, "tolerance")
    if not is_square_matrix(matrix):
        return False
    return all(matrix[index][index] > tolerance_value for index in range(len(matrix)))


def strictly_diagonally_dominant(
    matrix: tuple[tuple[float, ...], ...],
    *,
    tolerance: float = DEFAULT_POSITIVE_TOLERANCE,
) -> bool:
    _ensure_frozen_matrix(matrix)
    tolerance_value = ensure_non_negative_number(tolerance, "tolerance")
    if not is_square_matrix(matrix):
        return False
    for row_index, row in enumerate(matrix):
        diagonal = abs(row[row_index])
        off_diagonal_sum = sum(abs(value) for column_index, value in enumerate(row) if column_index != row_index)
        if diagonal <= off_diagonal_sum + tolerance_value:
            return False
    return True


def min_diagonal_value(matrix: tuple[tuple[float, ...], ...]) -> float | None:
    _ensure_frozen_matrix(matrix)
    if not is_square_matrix(matrix):
        return None
    return min(matrix[index][index] for index in range(len(matrix)))


def max_abs_off_diagonal_value(matrix: tuple[tuple[float, ...], ...]) -> float | None:
    _ensure_frozen_matrix(matrix)
    if not is_square_matrix(matrix):
        return None
    values = [
        abs(value)
        for row_index, row in enumerate(matrix)
        for column_index, value in enumerate(row)
        if column_index != row_index
    ]
    if not values:
        return 0.0
    return max(values)


def check_spd_candidate(
    matrix: tuple[tuple[float, ...], ...] | list[list[float]],
    *,
    symmetry_tolerance: float = DEFAULT_SYMMETRY_TOLERANCE,
    positive_tolerance: float = DEFAULT_POSITIVE_TOLERANCE,
) -> SPDCheckResult:
    frozen = ensure_matrix_shape(matrix)
    symmetry_tolerance_value = ensure_non_negative_number(symmetry_tolerance, "symmetry_tolerance")
    positive_tolerance_value = ensure_non_negative_number(positive_tolerance, "positive_tolerance")
    square = is_square_matrix(frozen)
    dimension = len(frozen) if square else None
    symmetric = is_symmetric_matrix(frozen, tolerance=symmetry_tolerance_value) if square else False
    positive = positive_diagonal(frozen, tolerance=positive_tolerance_value) if square else False
    dominant = strictly_diagonally_dominant(frozen, tolerance=positive_tolerance_value) if square else False
    spd_candidate = square and symmetric and positive and dominant

    return SPDCheckResult(
        is_square=square,
        is_symmetric=symmetric,
        is_positive_diagonal=positive,
        is_strictly_diagonally_dominant=dominant,
        is_spd_candidate=spd_candidate,
        dimension=dimension,
        min_diagonal=min_diagonal_value(frozen),
        max_abs_off_diagonal=max_abs_off_diagonal_value(frozen),
        metadata={
            "symmetry_tolerance": symmetry_tolerance_value,
            "positive_tolerance": positive_tolerance_value,
        },
    )


def build_metric_tensor_ref_from_check(
    *,
    tensor_id: str,
    check: SPDCheckResult,
    space: MetricSpaceKind = MetricSpaceKind.RIEMANNIAN,
    metadata: dict[str, object] | None = None,
) -> MetricTensorRef:
    if not isinstance(tensor_id, str) or not tensor_id:
        raise ValueError("tensor_id must be a non-empty string")
    if not isinstance(check, SPDCheckResult):
        raise TypeError("check must be SPDCheckResult")
    if not isinstance(space, MetricSpaceKind):
        raise TypeError("space must be MetricSpaceKind")
    if check.dimension is None:
        raise ValueError("check.dimension must not be None")

    merged_metadata: dict[str, object] = {
        "source": "spd_check",
        "is_square": check.is_square,
        "is_symmetric": check.is_symmetric,
        "is_positive_diagonal": check.is_positive_diagonal,
        "is_strictly_diagonally_dominant": check.is_strictly_diagonally_dominant,
    }
    if metadata is not None:
        merged_metadata.update(_copy_metadata(metadata))
    merged_metadata["source"] = "spd_check"

    return MetricTensorRef(
        tensor_id=tensor_id,
        space=space,
        dimension=check.dimension,
        status=MetricTensorStatus.ACTIVE if check.is_spd_candidate else MetricTensorStatus.INVALID,
        is_spd=check.is_spd_candidate,
        condition_number=None,
        metadata=merged_metadata,
    )


def _ensure_frozen_matrix(matrix: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    if not isinstance(matrix, tuple) or not matrix:
        raise TypeError("matrix must be a non-empty tuple of tuples")
    for row in matrix:
        if not isinstance(row, tuple) or not row:
            raise TypeError("matrix rows must be non-empty tuples")
        for value in row:
            ensure_finite_number(value, "matrix value")
    return matrix


def _copy_metadata(value: dict[str, object]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("metadata must be dict[str, object]")
    if any(not isinstance(key, str) for key in value):
        raise TypeError("metadata keys must be strings")
    return dict(value)


__all__ = [
    "DEFAULT_SYMMETRY_TOLERANCE",
    "DEFAULT_POSITIVE_TOLERANCE",
    "SPDCheckResult",
    "ensure_matrix_shape",
    "matrix_dimension",
    "is_square_matrix",
    "is_symmetric_matrix",
    "positive_diagonal",
    "strictly_diagonally_dominant",
    "min_diagonal_value",
    "max_abs_off_diagonal_value",
    "check_spd_candidate",
    "build_metric_tensor_ref_from_check",
]
