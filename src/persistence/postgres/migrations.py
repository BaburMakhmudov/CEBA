"""Deterministic PostgreSQL migration descriptors for CEBA persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ceba.persistence.postgres.connection import (
    ensure_non_empty_string,
    ensure_non_negative_int,
    ensure_optional_non_empty_string,
    ensure_positive_int,
)


__all__ = [
    "PostgresMigrationStatus",
    "PostgresMigrationDirection",
    "PostgresMigrationDescriptor",
    "PostgresMigrationPlan",
    "PostgresMigrationIssue",
    "PostgresMigrationReport",
    "PostgresMigrationPlanner",
    "create_postgres_migration_descriptor",
    "create_postgres_migration_plan",
    "build_postgres_migration_planner",
]


_FORBIDDEN_RUNTIME_STATE_NAMES = {
    "M_t",
    "Mt",
    "semantic_state",
    "runtime_semantic_state",
}

_VALID_SEVERITIES = {"INFO", "WARNING", "ERROR"}


class PostgresMigrationStatus(Enum):
    """Stable local PostgreSQL migration statuses."""

    PENDING = "PENDING"
    PLANNED = "PLANNED"
    APPLIED = "APPLIED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class PostgresMigrationDirection(Enum):
    """Stable PostgreSQL migration directions."""

    UP = "UP"
    DOWN = "DOWN"
    VERIFY = "VERIFY"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class PostgresMigrationDescriptor:
    """Immutable descriptor for one PostgreSQL migration."""

    version: int
    name: str
    filename: str
    direction: PostgresMigrationDirection
    checksum: str | None
    status: PostgresMigrationStatus
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        version = ensure_positive_int(self.version, "version")
        name = ensure_non_empty_string(self.name, "name")
        _reject_forbidden_name(name, "name")
        filename = ensure_non_empty_string(self.filename, "filename")
        _reject_forbidden_name(filename, "filename")
        if not filename.endswith(".sql"):
            raise ValueError("filename must end with .sql")
        if not isinstance(self.direction, PostgresMigrationDirection):
            raise TypeError("direction must be PostgresMigrationDirection")
        checksum = ensure_optional_non_empty_string(self.checksum, "checksum")
        if not isinstance(self.status, PostgresMigrationStatus):
            raise TypeError("status must be PostgresMigrationStatus")
        metadata = _copy_metadata(self.metadata)
        _reject_forbidden_metadata_keys(metadata)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "filename", filename)
        object.__setattr__(self, "checksum", checksum)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "version": self.version,
            "name": self.name,
            "filename": self.filename,
            "direction": self.direction.value,
            "checksum": self.checksum,
            "status": self.status.value,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PostgresMigrationPlan:
    """Immutable deterministic migration plan."""

    migrations: tuple[PostgresMigrationDescriptor, ...]
    direction: PostgresMigrationDirection
    target_version: int | None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        migrations = _ensure_migration_tuple(self.migrations)
        if not isinstance(self.direction, PostgresMigrationDirection):
            raise TypeError("direction must be PostgresMigrationDirection")
        if self.direction == PostgresMigrationDirection.UNKNOWN:
            raise ValueError("direction must not be UNKNOWN")
        for migration in migrations:
            if migration.direction not in {
                self.direction,
                PostgresMigrationDirection.VERIFY,
            }:
                raise ValueError("every migration.direction must equal direction or VERIFY")
        target_version = None
        if self.target_version is not None:
            target_version = ensure_non_negative_int(
                self.target_version,
                "target_version",
            )
        _validate_ordering(migrations=migrations, direction=self.direction)
        metadata = _copy_metadata(self.metadata)
        _reject_forbidden_metadata_keys(metadata)
        object.__setattr__(self, "migrations", migrations)
        object.__setattr__(self, "target_version", target_version)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "migrations": tuple(migration.to_dict() for migration in self.migrations),
            "direction": self.direction.value,
            "target_version": self.target_version,
            "metadata": dict(self.metadata),
        }

    def migration_count(self) -> int:
        """Return migration count."""

        return len(self.migrations)

    def latest_version(self) -> int | None:
        """Return latest version in the plan."""

        if not self.migrations:
            return None
        return max(migration.version for migration in self.migrations)


@dataclass(frozen=True)
class PostgresMigrationIssue:
    """Immutable migration validation issue descriptor."""

    code: str
    message: str
    severity: str
    migration_version: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        code = ensure_non_empty_string(self.code, "code")
        message = ensure_non_empty_string(self.message, "message")
        severity = ensure_non_empty_string(self.severity, "severity")
        if severity not in _VALID_SEVERITIES:
            raise ValueError("severity must be INFO, WARNING, or ERROR")
        migration_version = None
        if self.migration_version is not None:
            migration_version = ensure_positive_int(
                self.migration_version,
                "migration_version",
            )
        metadata = _copy_metadata(self.metadata)
        _reject_forbidden_metadata_keys(metadata)
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "migration_version", migration_version)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "migration_version": self.migration_version,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PostgresMigrationReport:
    """Immutable migration validation/planning report."""

    valid: bool
    plan: PostgresMigrationPlan | None
    issues: tuple[PostgresMigrationIssue, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be bool")
        if self.plan is not None and not isinstance(self.plan, PostgresMigrationPlan):
            raise TypeError("plan must be PostgresMigrationPlan or None")
        issues = _ensure_issue_tuple(self.issues)
        if self.valid and any(issue.severity == "ERROR" for issue in issues):
            raise ValueError("valid must be False when errors are present")
        metadata = _copy_metadata(self.metadata)
        _reject_forbidden_metadata_keys(metadata)
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "valid": self.valid,
            "plan": None if self.plan is None else self.plan.to_dict(),
            "issues": tuple(issue.to_dict() for issue in self.issues),
            "metadata": dict(self.metadata),
        }

    def error_count(self) -> int:
        """Return error issue count."""

        return _count_issues(self.issues, "ERROR")

    def warning_count(self) -> int:
        """Return warning issue count."""

        return _count_issues(self.issues, "WARNING")

    def info_count(self) -> int:
        """Return info issue count."""

        return _count_issues(self.issues, "INFO")

    def has_errors(self) -> bool:
        """Return whether report contains errors."""

        return self.error_count() > 0


class PostgresMigrationPlanner:
    """Deterministic local planner for PostgreSQL migration descriptors."""

    def __init__(self) -> None:
        pass

    def validate_descriptors(
        self,
        *,
        migrations: tuple[PostgresMigrationDescriptor, ...]
        | list[PostgresMigrationDescriptor],
    ) -> PostgresMigrationReport:
        """Validate migration descriptors without mutating them."""

        issues: list[PostgresMigrationIssue] = []
        migrations_tuple = _coerce_migration_collection(migrations)
        _append_duplicate_issues(migrations_tuple, issues)
        for migration in migrations_tuple:
            if migration.direction == PostgresMigrationDirection.UNKNOWN:
                issues.append(
                    _issue(
                        code="UNKNOWN_MIGRATION_DIRECTION",
                        message="migration direction is UNKNOWN",
                        severity="ERROR",
                        migration_version=migration.version,
                    )
                )
            if migration.status == PostgresMigrationStatus.UNKNOWN:
                issues.append(
                    _issue(
                        code="UNKNOWN_MIGRATION_STATUS",
                        message="migration status is UNKNOWN",
                        severity="WARNING",
                        migration_version=migration.version,
                    )
                )
            if migration.checksum is None:
                issues.append(
                    _issue(
                        code="MISSING_MIGRATION_CHECKSUM",
                        message="migration checksum is missing",
                        severity="WARNING",
                        migration_version=migration.version,
                    )
                )
            for key in migration.metadata:
                if key in _FORBIDDEN_RUNTIME_STATE_NAMES:
                    issues.append(
                        _issue(
                            code="FORBIDDEN_RUNTIME_STATE_METADATA",
                            message="migration metadata contains forbidden runtime semantic-state key",
                            severity="ERROR",
                            migration_version=migration.version,
                        )
                    )
        issues_tuple = tuple(issues)
        return PostgresMigrationReport(
            valid=not any(issue.severity == "ERROR" for issue in issues_tuple),
            plan=None,
            issues=issues_tuple,
            metadata={
                "source": "postgres_migration_planner",
                "operation": "validate_descriptors",
            },
        )

    def plan(
        self,
        *,
        migrations: tuple[PostgresMigrationDescriptor, ...]
        | list[PostgresMigrationDescriptor],
        direction: PostgresMigrationDirection,
        current_version: int = 0,
        target_version: int | None = None,
        metadata: dict[str, object] | None = None,
    ) -> PostgresMigrationReport:
        """Create a deterministic migration plan without executing migrations."""

        if not isinstance(direction, PostgresMigrationDirection):
            raise TypeError("direction must be PostgresMigrationDirection")
        if direction == PostgresMigrationDirection.UNKNOWN:
            raise ValueError("direction must not be UNKNOWN")
        current_version = ensure_non_negative_int(current_version, "current_version")
        normalized_target_version = None
        if target_version is not None:
            normalized_target_version = ensure_non_negative_int(
                target_version,
                "target_version",
            )
        validation_report = self.validate_descriptors(migrations=migrations)
        if validation_report.has_errors():
            return PostgresMigrationReport(
                valid=False,
                plan=None,
                issues=validation_report.issues,
                metadata=_merge_metadata(
                    {
                        "source": "postgres_migration_planner",
                        "operation": "plan",
                        "direction": direction.value,
                        "current_version": current_version,
                        "target_version": normalized_target_version,
                    },
                    metadata,
                ),
            )
        migrations_tuple = tuple(migrations)
        planned_migrations = _select_migrations(
            migrations=migrations_tuple,
            direction=direction,
            current_version=current_version,
            target_version=normalized_target_version,
        )
        plan = create_postgres_migration_plan(
            migrations=planned_migrations,
            direction=direction,
            target_version=normalized_target_version,
            metadata=_merge_metadata(
                {
                    "source": "postgres_migration_planner",
                    "operation": "plan",
                    "direction": direction.value,
                    "current_version": current_version,
                    "target_version": normalized_target_version,
                },
                metadata,
            ),
        )
        return PostgresMigrationReport(
            valid=True,
            plan=plan,
            issues=validation_report.issues,
            metadata=_merge_metadata(
                {
                    "source": "postgres_migration_planner",
                    "operation": "plan",
                    "direction": direction.value,
                    "current_version": current_version,
                    "target_version": normalized_target_version,
                },
                metadata,
            ),
        )


def create_postgres_migration_descriptor(
    *,
    version: int,
    name: str,
    filename: str,
    direction: PostgresMigrationDirection = PostgresMigrationDirection.UP,
    checksum: str | None = None,
    status: PostgresMigrationStatus = PostgresMigrationStatus.PENDING,
    metadata: dict[str, object] | None = None,
) -> PostgresMigrationDescriptor:
    """Create an immutable PostgreSQL migration descriptor."""

    return PostgresMigrationDescriptor(
        version=version,
        name=name,
        filename=filename,
        direction=direction,
        checksum=checksum,
        status=status,
        metadata=_merge_metadata(
            {
                "source": "postgres_migration_descriptor",
                "version": version,
                "filename": filename,
            },
            metadata,
        ),
    )


def create_postgres_migration_plan(
    *,
    migrations: tuple[PostgresMigrationDescriptor, ...]
    | list[PostgresMigrationDescriptor],
    direction: PostgresMigrationDirection,
    target_version: int | None = None,
    metadata: dict[str, object] | None = None,
) -> PostgresMigrationPlan:
    """Create an immutable PostgreSQL migration plan."""

    sorted_migrations = _sort_migrations(tuple(migrations), direction)
    return PostgresMigrationPlan(
        migrations=sorted_migrations,
        direction=direction,
        target_version=target_version,
        metadata=_merge_metadata(
            {
                "source": "postgres_migration_plan",
                "direction": direction.value
                if isinstance(direction, PostgresMigrationDirection)
                else direction,
                "target_version": target_version,
            },
            metadata,
        ),
    )


def build_postgres_migration_planner() -> PostgresMigrationPlanner:
    """Build a PostgreSQL migration planner."""

    return PostgresMigrationPlanner()


def _select_migrations(
    *,
    migrations: tuple[PostgresMigrationDescriptor, ...],
    direction: PostgresMigrationDirection,
    current_version: int,
    target_version: int | None,
) -> tuple[PostgresMigrationDescriptor, ...]:
    if direction == PostgresMigrationDirection.UP:
        selected = tuple(
            migration
            for migration in migrations
            if migration.version > current_version
            and (target_version is None or migration.version <= target_version)
        )
        return _sort_migrations(selected, direction)
    if direction == PostgresMigrationDirection.DOWN:
        selected = tuple(
            migration
            for migration in migrations
            if migration.version <= current_version
            and (target_version is None or migration.version > target_version)
        )
        return _sort_migrations(selected, direction)
    selected = tuple(
        migration
        for migration in migrations
        if migration.version <= current_version
        and (target_version is None or migration.version <= target_version)
    )
    return _sort_migrations(selected, direction)


def _sort_migrations(
    migrations: tuple[PostgresMigrationDescriptor, ...],
    direction: PostgresMigrationDirection,
) -> tuple[PostgresMigrationDescriptor, ...]:
    if direction == PostgresMigrationDirection.DOWN:
        return tuple(sorted(migrations, key=lambda migration: migration.version, reverse=True))
    return tuple(sorted(migrations, key=lambda migration: migration.version))


def _validate_ordering(
    *,
    migrations: tuple[PostgresMigrationDescriptor, ...],
    direction: PostgresMigrationDirection,
) -> None:
    versions = tuple(migration.version for migration in migrations)
    if direction == PostgresMigrationDirection.DOWN:
        if versions != tuple(sorted(versions, reverse=True)):
            raise ValueError("DOWN plans must be ordered by version descending")
        return
    if versions != tuple(sorted(versions)):
        raise ValueError("UP and VERIFY plans must be ordered by version ascending")


def _ensure_migration_tuple(
    migrations: tuple[PostgresMigrationDescriptor, ...]
    | list[PostgresMigrationDescriptor],
) -> tuple[PostgresMigrationDescriptor, ...]:
    if not isinstance(migrations, (tuple, list)):
        raise TypeError("migrations must be tuple[PostgresMigrationDescriptor, ...]")
    migrations_tuple = tuple(migrations)
    versions: set[int] = set()
    filenames: set[str] = set()
    for migration in migrations_tuple:
        if not isinstance(migration, PostgresMigrationDescriptor):
            raise TypeError("every migration must be PostgresMigrationDescriptor")
        if migration.version in versions:
            raise ValueError("migration versions must be unique")
        if migration.filename in filenames:
            raise ValueError("migration filenames must be unique")
        versions.add(migration.version)
        filenames.add(migration.filename)
    return migrations_tuple


def _coerce_migration_collection(
    migrations: tuple[PostgresMigrationDescriptor, ...]
    | list[PostgresMigrationDescriptor],
) -> tuple[PostgresMigrationDescriptor, ...]:
    if not isinstance(migrations, (tuple, list)):
        raise TypeError("migrations must be tuple[PostgresMigrationDescriptor, ...]")
    migrations_tuple = tuple(migrations)
    for migration in migrations_tuple:
        if not isinstance(migration, PostgresMigrationDescriptor):
            raise TypeError("every migration must be PostgresMigrationDescriptor")
    return migrations_tuple


def _append_duplicate_issues(
    migrations: tuple[PostgresMigrationDescriptor, ...],
    issues: list[PostgresMigrationIssue],
) -> None:
    seen_versions: set[int] = set()
    duplicate_versions: set[int] = set()
    seen_filenames: set[str] = set()
    duplicate_filenames: set[str] = set()
    for migration in migrations:
        if migration.version in seen_versions:
            duplicate_versions.add(migration.version)
        seen_versions.add(migration.version)
        if migration.filename in seen_filenames:
            duplicate_filenames.add(migration.filename)
        seen_filenames.add(migration.filename)
    for version in sorted(duplicate_versions):
        issues.append(
            _issue(
                code="DUPLICATE_MIGRATION_VERSION",
                message="duplicate migration version detected",
                severity="ERROR",
                migration_version=version,
            )
        )
    for filename in sorted(duplicate_filenames):
        issues.append(
            _issue(
                code="DUPLICATE_MIGRATION_FILENAME",
                message="duplicate migration filename detected",
                severity="ERROR",
                metadata={"filename": filename},
            )
        )


def _ensure_issue_tuple(
    issues: tuple[PostgresMigrationIssue, ...] | list[PostgresMigrationIssue],
) -> tuple[PostgresMigrationIssue, ...]:
    if not isinstance(issues, (tuple, list)):
        raise TypeError("issues must be tuple[PostgresMigrationIssue, ...]")
    issues_tuple = tuple(issues)
    for issue in issues_tuple:
        if not isinstance(issue, PostgresMigrationIssue):
            raise TypeError("every issue must be PostgresMigrationIssue")
    return issues_tuple


def _issue(
    *,
    code: str,
    message: str,
    severity: str,
    migration_version: int | None = None,
    metadata: dict[str, object] | None = None,
) -> PostgresMigrationIssue:
    return PostgresMigrationIssue(
        code=code,
        message=message,
        severity=severity,
        migration_version=migration_version,
        metadata={} if metadata is None else metadata,
    )


def _count_issues(
    issues: tuple[PostgresMigrationIssue, ...],
    severity: str,
) -> int:
    return sum(1 for issue in issues if issue.severity == severity)


def _copy_metadata(value: dict[str, object]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(value)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
    return copied


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


def _reject_forbidden_name(value: str, field_name: str) -> None:
    if value in _FORBIDDEN_RUNTIME_STATE_NAMES:
        raise ValueError(f"{field_name} must not persist runtime semantic state")


def _reject_forbidden_metadata_keys(metadata: dict[str, object]) -> None:
    for key in metadata:
        if key in _FORBIDDEN_RUNTIME_STATE_NAMES:
            raise ValueError("metadata keys must not persist runtime semantic state")
