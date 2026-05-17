"""Deterministic PostgreSQL persistence model descriptors for CEBA."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ceba.persistence.postgres.connection import (
    ensure_non_empty_string,
    ensure_non_negative_int,
    ensure_optional_non_empty_string,
)


__all__ = [
    "PostgresColumnType",
    "PostgresIndexKind",
    "PostgresConstraintKind",
    "PostgresTableKind",
    "PostgresColumn",
    "PostgresIndex",
    "PostgresConstraint",
    "PostgresTableModel",
    "PostgresSchemaManifest",
    "create_postgres_column",
    "create_postgres_index",
    "create_postgres_constraint",
    "create_postgres_table_model",
    "create_default_schema_manifest",
]


_FORBIDDEN_RUNTIME_STATE_NAMES = {
    "M_t",
    "Mt",
    "semantic_state",
    "runtime_semantic_state",
}


class PostgresColumnType(Enum):
    """Stable PostgreSQL column type descriptors."""

    UUID = "UUID"
    TEXT = "TEXT"
    INTEGER = "INTEGER"
    BIGINT = "BIGINT"
    BOOLEAN = "BOOLEAN"
    TIMESTAMPTZ = "TIMESTAMPTZ"
    JSONB = "JSONB"
    NUMERIC = "NUMERIC"
    BYTEA = "BYTEA"
    UNKNOWN = "UNKNOWN"


class PostgresIndexKind(Enum):
    """Stable PostgreSQL index kind descriptors."""

    BTREE = "BTREE"
    HASH = "HASH"
    GIN = "GIN"
    GIST = "GIST"
    UNIQUE = "UNIQUE"
    UNKNOWN = "UNKNOWN"


class PostgresConstraintKind(Enum):
    """Stable PostgreSQL constraint kind descriptors."""

    PRIMARY_KEY = "PRIMARY_KEY"
    FOREIGN_KEY = "FOREIGN_KEY"
    UNIQUE = "UNIQUE"
    CHECK = "CHECK"
    NOT_NULL = "NOT_NULL"
    UNKNOWN = "UNKNOWN"


class PostgresTableKind(Enum):
    """Stable PostgreSQL table kind descriptors."""

    CANONICAL_LEDGER = "CANONICAL_LEDGER"
    LEDGER_TRACE = "LEDGER_TRACE"
    LEDGER_EVENT = "LEDGER_EVENT"
    LEDGER_DECISION = "LEDGER_DECISION"
    LEDGER_CONSTRAINT = "LEDGER_CONSTRAINT"
    LEDGER_TELOS = "LEDGER_TELOS"
    LEDGER_AUDIT = "LEDGER_AUDIT"
    METRIC_SNAPSHOT = "METRIC_SNAPSHOT"
    OUTBOX_EVENT = "OUTBOX_EVENT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class PostgresColumn:
    """Immutable descriptor for one PostgreSQL column."""

    name: str
    column_type: PostgresColumnType
    nullable: bool
    primary_key: bool
    default_sql: str | None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = _ensure_safe_name(self.name, "name")
        if not isinstance(self.column_type, PostgresColumnType):
            raise TypeError("column_type must be PostgresColumnType")
        _ensure_bool(self.nullable, "nullable")
        _ensure_bool(self.primary_key, "primary_key")
        default_sql = ensure_optional_non_empty_string(
            self.default_sql,
            "default_sql",
        )
        metadata = _copy_metadata(self.metadata)
        _reject_forbidden_metadata_keys(metadata)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "default_sql", default_sql)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "name": self.name,
            "column_type": self.column_type.value,
            "nullable": self.nullable,
            "primary_key": self.primary_key,
            "default_sql": self.default_sql,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PostgresIndex:
    """Immutable descriptor for one PostgreSQL index."""

    name: str
    kind: PostgresIndexKind
    columns: tuple[str, ...]
    unique: bool
    where_sql: str | None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = ensure_non_empty_string(self.name, "name")
        if not isinstance(self.kind, PostgresIndexKind):
            raise TypeError("kind must be PostgresIndexKind")
        columns = _ensure_name_tuple(self.columns, "columns", require_unique=True)
        unique = _ensure_bool(self.unique, "unique")
        where_sql = ensure_optional_non_empty_string(self.where_sql, "where_sql")
        metadata = _copy_metadata(self.metadata)
        _reject_forbidden_metadata_keys(metadata)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "unique", unique)
        object.__setattr__(self, "where_sql", where_sql)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "name": self.name,
            "kind": self.kind.value,
            "columns": self.columns,
            "unique": self.unique,
            "where_sql": self.where_sql,
            "metadata": dict(self.metadata),
        }

    def column_count(self) -> int:
        """Return indexed column count."""

        return len(self.columns)


@dataclass(frozen=True)
class PostgresConstraint:
    """Immutable descriptor for one PostgreSQL constraint."""

    name: str
    kind: PostgresConstraintKind
    columns: tuple[str, ...]
    expression: str | None
    referenced_table: str | None
    referenced_columns: tuple[str, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = ensure_non_empty_string(self.name, "name")
        if not isinstance(self.kind, PostgresConstraintKind):
            raise TypeError("kind must be PostgresConstraintKind")
        columns = _ensure_name_tuple(self.columns, "columns", require_unique=False)
        expression = ensure_optional_non_empty_string(self.expression, "expression")
        referenced_table = ensure_optional_non_empty_string(
            self.referenced_table,
            "referenced_table",
        )
        if referenced_table is not None:
            _reject_forbidden_name(referenced_table, "referenced_table")
        referenced_columns = _ensure_name_tuple(
            self.referenced_columns,
            "referenced_columns",
            require_unique=False,
        )
        metadata = _copy_metadata(self.metadata)
        _reject_forbidden_metadata_keys(metadata)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "expression", expression)
        object.__setattr__(self, "referenced_table", referenced_table)
        object.__setattr__(self, "referenced_columns", referenced_columns)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "name": self.name,
            "kind": self.kind.value,
            "columns": self.columns,
            "expression": self.expression,
            "referenced_table": self.referenced_table,
            "referenced_columns": self.referenced_columns,
            "metadata": dict(self.metadata),
        }

    def column_count(self) -> int:
        """Return constrained column count."""

        return len(self.columns)


@dataclass(frozen=True)
class PostgresTableModel:
    """Immutable descriptor for one PostgreSQL table model."""

    table_name: str
    kind: PostgresTableKind
    columns: tuple[PostgresColumn, ...]
    indexes: tuple[PostgresIndex, ...]
    constraints: tuple[PostgresConstraint, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        table_name = _ensure_safe_name(self.table_name, "table_name")
        if not isinstance(self.kind, PostgresTableKind):
            raise TypeError("kind must be PostgresTableKind")
        columns = _ensure_column_tuple(self.columns)
        indexes = _ensure_index_tuple(self.indexes)
        constraints = _ensure_constraint_tuple(self.constraints)
        column_names = {column.name for column in columns}
        for index in indexes:
            for column_name in index.columns:
                if column_name not in column_names:
                    raise ValueError("every index column must exist in columns")
        for constraint in constraints:
            for column_name in constraint.columns:
                if column_name not in column_names:
                    raise ValueError("every constraint column must exist in columns")
        metadata = _copy_metadata(self.metadata)
        _reject_forbidden_metadata_keys(metadata)
        object.__setattr__(self, "table_name", table_name)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "indexes", indexes)
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "table_name": self.table_name,
            "kind": self.kind.value,
            "columns": tuple(column.to_dict() for column in self.columns),
            "indexes": tuple(index.to_dict() for index in self.indexes),
            "constraints": tuple(
                constraint.to_dict() for constraint in self.constraints
            ),
            "metadata": dict(self.metadata),
        }

    def column_count(self) -> int:
        """Return column count."""

        return len(self.columns)

    def index_count(self) -> int:
        """Return index count."""

        return len(self.indexes)

    def constraint_count(self) -> int:
        """Return constraint count."""

        return len(self.constraints)

    def has_column(self, column_name: str) -> bool:
        """Return whether the table contains a column by name."""

        normalized = _ensure_safe_name(column_name, "column_name")
        return any(column.name == normalized for column in self.columns)


@dataclass(frozen=True)
class PostgresSchemaManifest:
    """Immutable descriptor for a PostgreSQL schema manifest."""

    schema_name: str
    tables: tuple[PostgresTableModel, ...]
    version: int
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        schema_name = _ensure_safe_name(self.schema_name, "schema_name")
        tables = _ensure_table_tuple(self.tables)
        version = ensure_non_negative_int(self.version, "version")
        metadata = _copy_metadata(self.metadata)
        _reject_forbidden_metadata_keys(metadata)
        object.__setattr__(self, "schema_name", schema_name)
        object.__setattr__(self, "tables", tables)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "schema_name": self.schema_name,
            "tables": tuple(table.to_dict() for table in self.tables),
            "version": self.version,
            "metadata": dict(self.metadata),
        }

    def table_count(self) -> int:
        """Return table count."""

        return len(self.tables)

    def has_table(self, table_name: str) -> bool:
        """Return whether the manifest contains a table by name."""

        normalized = _ensure_safe_name(table_name, "table_name")
        return any(table.table_name == normalized for table in self.tables)

    def get_table(self, table_name: str) -> PostgresTableModel | None:
        """Return a table by name when present."""

        normalized = _ensure_safe_name(table_name, "table_name")
        for table in self.tables:
            if table.table_name == normalized:
                return table
        return None


def create_postgres_column(
    *,
    name: str,
    column_type: PostgresColumnType,
    nullable: bool = True,
    primary_key: bool = False,
    default_sql: str | None = None,
    metadata: dict[str, object] | None = None,
) -> PostgresColumn:
    """Create an immutable PostgreSQL column descriptor."""

    return PostgresColumn(
        name=name,
        column_type=column_type,
        nullable=nullable,
        primary_key=primary_key,
        default_sql=default_sql,
        metadata=_merge_metadata({"source": "postgres_column"}, metadata),
    )


def create_postgres_index(
    *,
    name: str,
    columns: tuple[str, ...] | list[str],
    kind: PostgresIndexKind = PostgresIndexKind.BTREE,
    unique: bool = False,
    where_sql: str | None = None,
    metadata: dict[str, object] | None = None,
) -> PostgresIndex:
    """Create an immutable PostgreSQL index descriptor."""

    return PostgresIndex(
        name=name,
        kind=kind,
        columns=tuple(columns),
        unique=unique,
        where_sql=where_sql,
        metadata=_merge_metadata(
            {
                "source": "postgres_index",
                "index_kind": kind.value if isinstance(kind, PostgresIndexKind) else kind,
            },
            metadata,
        ),
    )


def create_postgres_constraint(
    *,
    name: str,
    kind: PostgresConstraintKind,
    columns: tuple[str, ...] | list[str],
    expression: str | None = None,
    referenced_table: str | None = None,
    referenced_columns: tuple[str, ...] | list[str] = (),
    metadata: dict[str, object] | None = None,
) -> PostgresConstraint:
    """Create an immutable PostgreSQL constraint descriptor."""

    return PostgresConstraint(
        name=name,
        kind=kind,
        columns=tuple(columns),
        expression=expression,
        referenced_table=referenced_table,
        referenced_columns=tuple(referenced_columns),
        metadata=_merge_metadata(
            {
                "source": "postgres_constraint",
                "constraint_kind": (
                    kind.value if isinstance(kind, PostgresConstraintKind) else kind
                ),
            },
            metadata,
        ),
    )


def create_postgres_table_model(
    *,
    table_name: str,
    columns: tuple[PostgresColumn, ...] | list[PostgresColumn],
    kind: PostgresTableKind = PostgresTableKind.UNKNOWN,
    indexes: tuple[PostgresIndex, ...] | list[PostgresIndex] = (),
    constraints: tuple[PostgresConstraint, ...] | list[PostgresConstraint] = (),
    metadata: dict[str, object] | None = None,
) -> PostgresTableModel:
    """Create an immutable PostgreSQL table model descriptor."""

    return PostgresTableModel(
        table_name=table_name,
        kind=kind,
        columns=tuple(columns),
        indexes=tuple(indexes),
        constraints=tuple(constraints),
        metadata=_merge_metadata(
            {
                "source": "postgres_table_model",
                "table_kind": kind.value if isinstance(kind, PostgresTableKind) else kind,
                "table_name": table_name,
            },
            metadata,
        ),
    )


def create_default_schema_manifest(
    *,
    schema_name: str = "ceba",
    version: int = 1,
    metadata: dict[str, object] | None = None,
) -> PostgresSchemaManifest:
    """Create a safe default CEBA PostgreSQL schema manifest."""

    required_metadata = {
        "source": "postgres_schema_manifest",
        "version": version,
    }
    tables = (
        _create_ledger_table(
            table_name="canonical_ledger_records",
            kind=PostgresTableKind.CANONICAL_LEDGER,
        ),
        _create_ledger_table(
            table_name="ledger_trace_records",
            kind=PostgresTableKind.LEDGER_TRACE,
        ),
        _create_ledger_table(
            table_name="ledger_event_records",
            kind=PostgresTableKind.LEDGER_EVENT,
        ),
        _create_ledger_table(
            table_name="ledger_decision_records",
            kind=PostgresTableKind.LEDGER_DECISION,
        ),
        _create_ledger_table(
            table_name="ledger_constraint_records",
            kind=PostgresTableKind.LEDGER_CONSTRAINT,
        ),
        _create_ledger_table(
            table_name="ledger_telos_records",
            kind=PostgresTableKind.LEDGER_TELOS,
        ),
        _create_ledger_table(
            table_name="ledger_audit_records",
            kind=PostgresTableKind.LEDGER_AUDIT,
        ),
        _create_metric_snapshots_table(),
        _create_outbox_events_table(),
    )
    return PostgresSchemaManifest(
        schema_name=schema_name,
        tables=tables,
        version=version,
        metadata=_merge_metadata(required_metadata, metadata),
    )


def _create_ledger_table(
    *,
    table_name: str,
    kind: PostgresTableKind,
) -> PostgresTableModel:
    columns = (
        create_postgres_column(
            name="record_id",
            column_type=PostgresColumnType.UUID,
            nullable=False,
            primary_key=True,
        ),
        create_postgres_column(
            name="record_kind",
            column_type=PostgresColumnType.TEXT,
            nullable=False,
        ),
        create_postgres_column(
            name="record_status",
            column_type=PostgresColumnType.TEXT,
            nullable=False,
        ),
        create_postgres_column(
            name="record_key",
            column_type=PostgresColumnType.TEXT,
            nullable=False,
        ),
        create_postgres_column(
            name="payload",
            column_type=PostgresColumnType.JSONB,
            nullable=False,
        ),
        create_postgres_column(
            name="provenance",
            column_type=PostgresColumnType.JSONB,
            nullable=False,
        ),
        create_postgres_column(
            name="semantic_hash",
            column_type=PostgresColumnType.TEXT,
            nullable=True,
        ),
        create_postgres_column(
            name="created_at",
            column_type=PostgresColumnType.TIMESTAMPTZ,
            nullable=False,
        ),
        create_postgres_column(
            name="updated_at",
            column_type=PostgresColumnType.TIMESTAMPTZ,
            nullable=False,
        ),
        create_postgres_column(
            name="metadata",
            column_type=PostgresColumnType.JSONB,
            nullable=False,
        ),
    )
    indexes = (
        create_postgres_index(
            name=f"{table_name}_record_key_uidx",
            kind=PostgresIndexKind.UNIQUE,
            columns=("record_key",),
            unique=True,
        ),
        create_postgres_index(
            name=f"{table_name}_record_status_idx",
            columns=("record_status",),
        ),
        create_postgres_index(
            name=f"{table_name}_created_at_idx",
            columns=("created_at",),
        ),
        create_postgres_index(
            name=f"{table_name}_updated_at_idx",
            columns=("updated_at",),
        ),
        create_postgres_index(
            name=f"{table_name}_metadata_gin_idx",
            kind=PostgresIndexKind.GIN,
            columns=("metadata",),
        ),
    )
    constraints = (
        create_postgres_constraint(
            name=f"{table_name}_pkey",
            kind=PostgresConstraintKind.PRIMARY_KEY,
            columns=("record_id",),
        ),
    )
    return create_postgres_table_model(
        table_name=table_name,
        kind=kind,
        columns=columns,
        indexes=indexes,
        constraints=constraints,
    )


def _create_metric_snapshots_table() -> PostgresTableModel:
    columns = (
        create_postgres_column(
            name="snapshot_id",
            column_type=PostgresColumnType.UUID,
            nullable=False,
            primary_key=True,
        ),
        create_postgres_column(
            name="metric_kind",
            column_type=PostgresColumnType.TEXT,
            nullable=False,
        ),
        create_postgres_column(
            name="payload",
            column_type=PostgresColumnType.JSONB,
            nullable=False,
        ),
        create_postgres_column(
            name="created_at",
            column_type=PostgresColumnType.TIMESTAMPTZ,
            nullable=False,
        ),
        create_postgres_column(
            name="metadata",
            column_type=PostgresColumnType.JSONB,
            nullable=False,
        ),
    )
    indexes = (
        create_postgres_index(
            name="metric_snapshots_metric_kind_idx",
            columns=("metric_kind",),
        ),
        create_postgres_index(
            name="metric_snapshots_created_at_idx",
            columns=("created_at",),
        ),
        create_postgres_index(
            name="metric_snapshots_metadata_gin_idx",
            kind=PostgresIndexKind.GIN,
            columns=("metadata",),
        ),
    )
    constraints = (
        create_postgres_constraint(
            name="metric_snapshots_pkey",
            kind=PostgresConstraintKind.PRIMARY_KEY,
            columns=("snapshot_id",),
        ),
    )
    return create_postgres_table_model(
        table_name="metric_snapshots",
        kind=PostgresTableKind.METRIC_SNAPSHOT,
        columns=columns,
        indexes=indexes,
        constraints=constraints,
    )


def _create_outbox_events_table() -> PostgresTableModel:
    columns = (
        create_postgres_column(
            name="event_id",
            column_type=PostgresColumnType.UUID,
            nullable=False,
            primary_key=True,
        ),
        create_postgres_column(
            name="event_kind",
            column_type=PostgresColumnType.TEXT,
            nullable=False,
        ),
        create_postgres_column(
            name="payload",
            column_type=PostgresColumnType.JSONB,
            nullable=False,
        ),
        create_postgres_column(
            name="status",
            column_type=PostgresColumnType.TEXT,
            nullable=False,
        ),
        create_postgres_column(
            name="created_at",
            column_type=PostgresColumnType.TIMESTAMPTZ,
            nullable=False,
        ),
        create_postgres_column(
            name="published_at",
            column_type=PostgresColumnType.TIMESTAMPTZ,
            nullable=True,
        ),
        create_postgres_column(
            name="metadata",
            column_type=PostgresColumnType.JSONB,
            nullable=False,
        ),
    )
    indexes = (
        create_postgres_index(
            name="outbox_events_event_kind_idx",
            columns=("event_kind",),
        ),
        create_postgres_index(
            name="outbox_events_status_idx",
            columns=("status",),
        ),
        create_postgres_index(
            name="outbox_events_created_at_idx",
            columns=("created_at",),
        ),
        create_postgres_index(
            name="outbox_events_metadata_gin_idx",
            kind=PostgresIndexKind.GIN,
            columns=("metadata",),
        ),
    )
    constraints = (
        create_postgres_constraint(
            name="outbox_events_pkey",
            kind=PostgresConstraintKind.PRIMARY_KEY,
            columns=("event_id",),
        ),
    )
    return create_postgres_table_model(
        table_name="outbox_events",
        kind=PostgresTableKind.OUTBOX_EVENT,
        columns=columns,
        indexes=indexes,
        constraints=constraints,
    )


def _ensure_column_tuple(
    columns: tuple[PostgresColumn, ...] | list[PostgresColumn],
) -> tuple[PostgresColumn, ...]:
    if not isinstance(columns, (tuple, list)):
        raise TypeError("columns must be tuple[PostgresColumn, ...]")
    columns_tuple = tuple(columns)
    if not columns_tuple:
        raise ValueError("at least one column required")
    seen: set[str] = set()
    for column in columns_tuple:
        if not isinstance(column, PostgresColumn):
            raise TypeError("every column must be PostgresColumn")
        if column.name in seen:
            raise ValueError("column names must be unique")
        seen.add(column.name)
    return columns_tuple


def _ensure_index_tuple(
    indexes: tuple[PostgresIndex, ...] | list[PostgresIndex],
) -> tuple[PostgresIndex, ...]:
    if not isinstance(indexes, (tuple, list)):
        raise TypeError("indexes must be tuple[PostgresIndex, ...]")
    indexes_tuple = tuple(indexes)
    seen: set[str] = set()
    for index in indexes_tuple:
        if not isinstance(index, PostgresIndex):
            raise TypeError("every index must be PostgresIndex")
        if index.name in seen:
            raise ValueError("index names must be unique")
        seen.add(index.name)
    return indexes_tuple


def _ensure_constraint_tuple(
    constraints: tuple[PostgresConstraint, ...] | list[PostgresConstraint],
) -> tuple[PostgresConstraint, ...]:
    if not isinstance(constraints, (tuple, list)):
        raise TypeError("constraints must be tuple[PostgresConstraint, ...]")
    constraints_tuple = tuple(constraints)
    seen: set[str] = set()
    for constraint in constraints_tuple:
        if not isinstance(constraint, PostgresConstraint):
            raise TypeError("every constraint must be PostgresConstraint")
        if constraint.name in seen:
            raise ValueError("constraint names must be unique")
        seen.add(constraint.name)
    return constraints_tuple


def _ensure_table_tuple(
    tables: tuple[PostgresTableModel, ...] | list[PostgresTableModel],
) -> tuple[PostgresTableModel, ...]:
    if not isinstance(tables, (tuple, list)):
        raise TypeError("tables must be tuple[PostgresTableModel, ...]")
    tables_tuple = tuple(tables)
    seen: set[str] = set()
    for table in tables_tuple:
        if not isinstance(table, PostgresTableModel):
            raise TypeError("every table must be PostgresTableModel")
        if table.table_name in seen:
            raise ValueError("table names must be unique")
        seen.add(table.table_name)
    return tables_tuple


def _ensure_name_tuple(
    values: tuple[str, ...] | list[str],
    field_name: str,
    *,
    require_unique: bool,
) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple[str, ...]")
    normalized = tuple(_ensure_safe_name(value, field_name) for value in values)
    if require_unique and len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} values must be unique")
    return normalized


def _ensure_bool(value: bool, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be bool")
    return value


def _ensure_safe_name(value: str, field_name: str) -> str:
    normalized = ensure_non_empty_string(value, field_name)
    _reject_forbidden_name(normalized, field_name)
    return normalized


def _reject_forbidden_name(value: str, field_name: str) -> None:
    if value in _FORBIDDEN_RUNTIME_STATE_NAMES:
        raise ValueError(f"{field_name} must not persist runtime semantic state")


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


def _reject_forbidden_metadata_keys(metadata: dict[str, object]) -> None:
    for key in metadata:
        if key in _FORBIDDEN_RUNTIME_STATE_NAMES:
            raise ValueError("metadata keys must not persist runtime semantic state")
