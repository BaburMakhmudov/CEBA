"""Deterministic in-memory registry utilities for CEBA Core Constraint Processing."""

from __future__ import annotations

from dataclasses import dataclass, field

from ceba.core.constraints.errors import (
    ConstraintNotFoundError,
    ConstraintRegistryError,
    DuplicateConstraintIdError,
    build_constraint_error_context,
)
from ceba.core.constraints.schema import (
    ConstraintDefinition,
    ConstraintKind,
    ConstraintScope,
    ConstraintStatus,
    ensure_metadata_dict,
    ensure_non_empty_string,
)


@dataclass(frozen=True, slots=True)
class ConstraintRegistrySnapshot:
    """Immutable serializable-friendly view of registry state."""

    constraint_ids: tuple[str, ...]
    active_count: int
    inactive_count: int
    deprecated_count: int
    experimental_count: int
    total_count: int
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.constraint_ids, list):
            object.__setattr__(self, "constraint_ids", tuple(self.constraint_ids))
        if not isinstance(self.constraint_ids, tuple):
            raise TypeError("constraint_ids must be a tuple of strings")
        for constraint_id in self.constraint_ids:
            ensure_non_empty_string(constraint_id, "constraint_ids")

        _ensure_non_negative_int(self.active_count, "active_count")
        _ensure_non_negative_int(self.inactive_count, "inactive_count")
        _ensure_non_negative_int(self.deprecated_count, "deprecated_count")
        _ensure_non_negative_int(self.experimental_count, "experimental_count")
        _ensure_non_negative_int(self.total_count, "total_count")

        if self.total_count != len(self.constraint_ids):
            raise ValueError("total_count must equal len(constraint_ids)")

        ensure_metadata_dict(self.metadata)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic, serialization-friendly representation."""

        return {
            "constraint_ids": list(self.constraint_ids),
            "active_count": self.active_count,
            "inactive_count": self.inactive_count,
            "deprecated_count": self.deprecated_count,
            "experimental_count": self.experimental_count,
            "total_count": self.total_count,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ConstraintRegistry:
    """Immutable registry of ConstraintDefinition objects."""

    constraints: tuple[ConstraintDefinition, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.constraints, list):
            object.__setattr__(self, "constraints", tuple(self.constraints))
        if not isinstance(self.constraints, tuple):
            raise TypeError("constraints must be a tuple of ConstraintDefinition")

        seen_ids: set[str] = set()
        for constraint in self.constraints:
            if not isinstance(constraint, ConstraintDefinition):
                raise TypeError("constraints must contain ConstraintDefinition")
            constraint_id = str(constraint.constraint_id)
            if constraint_id in seen_ids:
                raise DuplicateConstraintIdError(
                    message="duplicate constraint_id",
                    context=build_constraint_error_context(
                        operation="ConstraintRegistry.__post_init__",
                        constraint_id=constraint_id,
                    ),
                )
            seen_ids.add(constraint_id)

        ensure_metadata_dict(self.metadata)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def ids(self) -> tuple[str, ...]:
        """Return constraint IDs in registry order."""

        return tuple(str(constraint.constraint_id) for constraint in self.constraints)

    def active(self) -> tuple[ConstraintDefinition, ...]:
        """Return active constraints in registry order."""

        return tuple(
            constraint
            for constraint in self.constraints
            if constraint.status is ConstraintStatus.ACTIVE
        )

    def by_id(self, constraint_id: str) -> ConstraintDefinition:
        """Return one constraint by ID."""

        ensure_non_empty_string(constraint_id, "constraint_id")
        for constraint in self.constraints:
            if str(constraint.constraint_id) == constraint_id:
                return constraint
        raise ConstraintNotFoundError(
            message="constraint not found",
            context=build_constraint_error_context(
                operation="ConstraintRegistry.by_id",
                constraint_id=constraint_id,
            ),
        )

    def contains(self, constraint_id: str) -> bool:
        """Return whether the registry contains a constraint ID."""

        ensure_non_empty_string(constraint_id, "constraint_id")
        return any(str(constraint.constraint_id) == constraint_id for constraint in self.constraints)

    def snapshot(self) -> ConstraintRegistrySnapshot:
        """Return a deterministic snapshot of registry state."""

        active_count = 0
        inactive_count = 0
        deprecated_count = 0
        experimental_count = 0

        for constraint in self.constraints:
            if constraint.status is ConstraintStatus.ACTIVE:
                active_count += 1
            elif constraint.status is ConstraintStatus.INACTIVE:
                inactive_count += 1
            elif constraint.status is ConstraintStatus.DEPRECATED:
                deprecated_count += 1
            elif constraint.status is ConstraintStatus.EXPERIMENTAL:
                experimental_count += 1

        return ConstraintRegistrySnapshot(
            constraint_ids=self.ids(),
            active_count=active_count,
            inactive_count=inactive_count,
            deprecated_count=deprecated_count,
            experimental_count=experimental_count,
            total_count=len(self.constraints),
            metadata=dict(self.metadata),
        )

    def add(self, constraint: ConstraintDefinition) -> ConstraintRegistry:
        """Return a new registry with one constraint appended."""

        if not isinstance(constraint, ConstraintDefinition):
            raise TypeError("constraint must be ConstraintDefinition")
        constraint_id = str(constraint.constraint_id)
        if self.contains(constraint_id):
            raise DuplicateConstraintIdError(
                message="duplicate constraint_id",
                context=build_constraint_error_context(
                    operation="ConstraintRegistry.add",
                    constraint_id=constraint_id,
                ),
            )
        return ConstraintRegistry(
            constraints=self.constraints + (constraint,),
            metadata=dict(self.metadata),
        )

    def remove(self, constraint_id: str) -> ConstraintRegistry:
        """Return a new registry without one constraint."""

        ensure_non_empty_string(constraint_id, "constraint_id")
        if not self.contains(constraint_id):
            raise ConstraintNotFoundError(
                message="constraint not found",
                context=build_constraint_error_context(
                    operation="ConstraintRegistry.remove",
                    constraint_id=constraint_id,
                ),
            )
        return ConstraintRegistry(
            constraints=tuple(
                constraint
                for constraint in self.constraints
                if str(constraint.constraint_id) != constraint_id
            ),
            metadata=dict(self.metadata),
        )

    def replace(self, constraint: ConstraintDefinition) -> ConstraintRegistry:
        """Return a new registry with one constraint replaced."""

        if not isinstance(constraint, ConstraintDefinition):
            raise TypeError("constraint must be ConstraintDefinition")
        constraint_id = str(constraint.constraint_id)
        if not self.contains(constraint_id):
            raise ConstraintNotFoundError(
                message="constraint not found",
                context=build_constraint_error_context(
                    operation="ConstraintRegistry.replace",
                    constraint_id=constraint_id,
                ),
            )
        return ConstraintRegistry(
            constraints=tuple(
                constraint if str(existing.constraint_id) == constraint_id else existing
                for existing in self.constraints
            ),
            metadata=dict(self.metadata),
        )


def create_constraint_registry(
    constraints: tuple[ConstraintDefinition, ...] | list[ConstraintDefinition] = (),
    *,
    metadata: dict[str, object] | None = None,
) -> ConstraintRegistry:
    """Create a validated registry."""

    try:
        if not isinstance(constraints, (tuple, list)):
            raise TypeError("constraints must be a tuple or list")
        return ConstraintRegistry(
            constraints=tuple(constraints),
            metadata={} if metadata is None else metadata,
        )
    except ConstraintRegistryError:
        raise
    except Exception as exc:
        raise ConstraintRegistryError(
            message="failed to create constraint registry",
            context=build_constraint_error_context(operation="create_constraint_registry"),
            cause=exc,
        ) from exc


def registry_from_mapping(
    mapping: dict[str, ConstraintDefinition],
    *,
    metadata: dict[str, object] | None = None,
) -> ConstraintRegistry:
    """Build a registry from a mapping."""

    try:
        if not isinstance(mapping, dict):
            raise TypeError("mapping must be a dict")
        constraints: list[ConstraintDefinition] = []
        for key, definition in mapping.items():
            ensure_non_empty_string(key, "mapping key")
            if not isinstance(definition, ConstraintDefinition):
                raise TypeError("mapping values must be ConstraintDefinition")
            if key != str(definition.constraint_id):
                raise ValueError("mapping key must equal str(definition.constraint_id)")
            constraints.append(definition)
        return ConstraintRegistry(
            constraints=tuple(constraints),
            metadata={} if metadata is None else metadata,
        )
    except ConstraintRegistryError:
        raise
    except Exception as exc:
        raise ConstraintRegistryError(
            message="failed to build registry from mapping",
            context=build_constraint_error_context(operation="registry_from_mapping"),
            cause=exc,
        ) from exc


def find_constraints_by_kind(
    registry: ConstraintRegistry,
    kind: ConstraintKind,
) -> tuple[ConstraintDefinition, ...]:
    """Return matching constraints in registry order."""

    _ensure_registry(registry)
    if not isinstance(kind, ConstraintKind):
        raise TypeError("kind must be ConstraintKind")
    return tuple(constraint for constraint in registry.constraints if constraint.kind is kind)


def find_constraints_by_scope(
    registry: ConstraintRegistry,
    scope: ConstraintScope,
) -> tuple[ConstraintDefinition, ...]:
    """Return matching constraints in registry order."""

    _ensure_registry(registry)
    if not isinstance(scope, ConstraintScope):
        raise TypeError("scope must be ConstraintScope")
    return tuple(constraint for constraint in registry.constraints if constraint.scope is scope)


def find_constraints_by_status(
    registry: ConstraintRegistry,
    status: ConstraintStatus,
) -> tuple[ConstraintDefinition, ...]:
    """Return matching constraints in registry order."""

    _ensure_registry(registry)
    if not isinstance(status, ConstraintStatus):
        raise TypeError("status must be ConstraintStatus")
    return tuple(constraint for constraint in registry.constraints if constraint.status is status)


def find_active_constraints(
    registry: ConstraintRegistry,
) -> tuple[ConstraintDefinition, ...]:
    """Return active constraints in registry order."""

    _ensure_registry(registry)
    return registry.active()


def require_constraint(
    registry: ConstraintRegistry,
    constraint_id: str,
) -> ConstraintDefinition:
    """Return one required constraint by ID."""

    _ensure_registry(registry)
    ensure_non_empty_string(constraint_id, "constraint_id")
    return registry.by_id(constraint_id)


def merge_registries(
    left: ConstraintRegistry,
    right: ConstraintRegistry,
    *,
    metadata: dict[str, object] | None = None,
) -> ConstraintRegistry:
    """Merge two registries into a new registry."""

    _ensure_registry(left)
    _ensure_registry(right)

    left_ids = set(left.ids())
    for constraint_id in right.ids():
        if constraint_id in left_ids:
            raise DuplicateConstraintIdError(
                message="duplicate constraint_id across registries",
                context=build_constraint_error_context(
                    operation="merge_registries",
                    constraint_id=constraint_id,
                ),
            )

    merged_metadata = {**left.metadata, **right.metadata}
    if metadata is not None:
        ensure_metadata_dict(metadata)
        merged_metadata.update(metadata)

    return ConstraintRegistry(
        constraints=left.constraints + right.constraints,
        metadata=merged_metadata,
    )


def _ensure_registry(registry: ConstraintRegistry) -> ConstraintRegistry:
    if not isinstance(registry, ConstraintRegistry):
        raise TypeError("registry must be ConstraintRegistry")
    return registry


def _ensure_non_negative_int(value: int, field_name: str) -> int:
    ensure_non_empty_string(field_name, "field_name")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{field_name} must be greater than or equal to 0")
    return value


__all__ = [
    "ConstraintRegistrySnapshot",
    "ConstraintRegistry",
    "create_constraint_registry",
    "registry_from_mapping",
    "find_constraints_by_kind",
    "find_constraints_by_scope",
    "find_constraints_by_status",
    "find_active_constraints",
    "require_constraint",
    "merge_registries",
]
