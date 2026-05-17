"""Strict PostgreSQL connection configuration and local state descriptors."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4


__all__ = [
    "PostgresSSLMode",
    "PostgresConnectionStatus",
    "PostgresConnectionConfig",
    "PostgresConnectionIdentity",
    "PostgresConnectionState",
    "utc_now",
    "new_postgres_connection_id",
    "ensure_non_empty_string",
    "ensure_optional_non_empty_string",
    "ensure_positive_int",
    "ensure_non_negative_int",
    "ensure_timezone_aware",
    "create_postgres_connection_config",
    "create_postgres_connection_identity",
    "create_postgres_connection_state",
]


_SECRET_METADATA_KEYS = {
    "password",
    "raw_password",
    "secret",
    "token",
}


class PostgresSSLMode(Enum):
    """Stable PostgreSQL SSL mode values."""

    DISABLE = "disable"
    ALLOW = "allow"
    PREFER = "prefer"
    REQUIRE = "require"
    VERIFY_CA = "verify-ca"
    VERIFY_FULL = "verify-full"


class PostgresConnectionStatus(Enum):
    """Stable local PostgreSQL connection lifecycle statuses."""

    CREATED = "CREATED"
    CONFIGURED = "CONFIGURED"
    READY = "READY"
    CONNECTED = "CONNECTED"
    DEGRADED = "DEGRADED"
    CLOSED = "CLOSED"
    FAILED = "FAILED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class PostgresConnectionConfig:
    """Immutable PostgreSQL connection configuration."""

    host: str
    port: int
    database: str
    user: str
    password_ref: str | None
    ssl_mode: PostgresSSLMode
    connect_timeout_seconds: int
    application_name: str
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        host = ensure_non_empty_string(self.host, "host")
        port = _ensure_port(self.port)
        database = ensure_non_empty_string(self.database, "database")
        user = ensure_non_empty_string(self.user, "user")
        password_ref = ensure_optional_non_empty_string(
            self.password_ref,
            "password_ref",
        )
        if not isinstance(self.ssl_mode, PostgresSSLMode):
            raise TypeError("ssl_mode must be PostgresSSLMode")
        connect_timeout_seconds = ensure_positive_int(
            self.connect_timeout_seconds,
            "connect_timeout_seconds",
        )
        application_name = ensure_non_empty_string(
            self.application_name,
            "application_name",
        )
        metadata = _copy_metadata(self.metadata)
        _reject_secret_metadata_keys(metadata)
        object.__setattr__(self, "host", host)
        object.__setattr__(self, "port", port)
        object.__setattr__(self, "database", database)
        object.__setattr__(self, "user", user)
        object.__setattr__(self, "password_ref", password_ref)
        object.__setattr__(
            self,
            "connect_timeout_seconds",
            connect_timeout_seconds,
        )
        object.__setattr__(self, "application_name", application_name)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary without raw secrets."""

        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password_ref": self.password_ref,
            "ssl_mode": self.ssl_mode.value,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "application_name": self.application_name,
            "metadata": dict(self.metadata),
        }

    def safe_dsn(self) -> str:
        """Return a safe DSN-like string without password material."""

        return (
            f"postgresql://{self.user}@{self.host}:{self.port}/{self.database}"
            f"?sslmode={self.ssl_mode.value}"
        )


@dataclass(frozen=True)
class PostgresConnectionIdentity:
    """Immutable identity descriptor for a PostgreSQL connection configuration."""

    connection_id: UUID
    config: PostgresConnectionConfig
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.connection_id, UUID):
            raise TypeError("connection_id must be UUID")
        if not isinstance(self.config, PostgresConnectionConfig):
            raise TypeError("config must be PostgresConnectionConfig")
        created_at = ensure_timezone_aware(self.created_at, "created_at")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "connection_id": str(self.connection_id),
            "config": self.config.to_dict(),
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PostgresConnectionState:
    """Immutable local connection lifecycle descriptor."""

    identity: PostgresConnectionIdentity
    status: PostgresConnectionStatus
    last_checked_at: datetime | None
    failure_reason: str | None
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, PostgresConnectionIdentity):
            raise TypeError("identity must be PostgresConnectionIdentity")
        if not isinstance(self.status, PostgresConnectionStatus):
            raise TypeError("status must be PostgresConnectionStatus")
        last_checked_at = None
        if self.last_checked_at is not None:
            last_checked_at = ensure_timezone_aware(
                self.last_checked_at,
                "last_checked_at",
            )
            if last_checked_at < self.identity.created_at:
                raise ValueError("last_checked_at must be >= identity.created_at")
        failure_reason = ensure_optional_non_empty_string(
            self.failure_reason,
            "failure_reason",
        )
        updated_at = ensure_timezone_aware(self.updated_at, "updated_at")
        if updated_at < self.identity.created_at:
            raise ValueError("updated_at must be >= identity.created_at")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "last_checked_at", last_checked_at)
        object.__setattr__(self, "failure_reason", failure_reason)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "identity": self.identity.to_dict(),
            "status": self.status.value,
            "last_checked_at": (
                None
                if self.last_checked_at is None
                else self.last_checked_at.isoformat()
            ),
            "failure_reason": self.failure_reason,
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    def is_available(self) -> bool:
        """Return whether the local descriptor marks the connection available."""

        return self.status in {
            PostgresConnectionStatus.READY,
            PostgresConnectionStatus.CONNECTED,
        }


def utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""

    return datetime.now(timezone.utc)


def new_postgres_connection_id() -> UUID:
    """Return a new PostgreSQL connection UUID."""

    return uuid4()


def ensure_non_empty_string(value: str, field_name: str) -> str:
    """Validate and normalize a non-empty string."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def ensure_optional_non_empty_string(value: str | None, field_name: str) -> str | None:
    """Validate and normalize an optional non-empty string."""

    if value is None:
        return None
    return ensure_non_empty_string(value, field_name)


def ensure_positive_int(value: int, field_name: str) -> int:
    """Validate a positive integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def ensure_non_negative_int(value: int, field_name: str) -> int:
    """Validate a non-negative integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def ensure_timezone_aware(value: datetime, field_name: str) -> datetime:
    """Validate a timezone-aware datetime."""

    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def create_postgres_connection_config(
    *,
    host: str,
    database: str,
    user: str,
    port: int = 5432,
    password_ref: str | None = None,
    ssl_mode: PostgresSSLMode = PostgresSSLMode.PREFER,
    connect_timeout_seconds: int = 10,
    application_name: str = "ceba",
    metadata: dict[str, object] | None = None,
) -> PostgresConnectionConfig:
    """Create an immutable PostgreSQL connection configuration."""

    required_metadata = {
        "source": "postgres_connection_config",
        "safe": True,
    }
    return PostgresConnectionConfig(
        host=host,
        port=port,
        database=database,
        user=user,
        password_ref=password_ref,
        ssl_mode=ssl_mode,
        connect_timeout_seconds=connect_timeout_seconds,
        application_name=application_name,
        metadata=_merge_metadata(required_metadata, metadata),
    )


def create_postgres_connection_identity(
    *,
    config: PostgresConnectionConfig,
    metadata: dict[str, object] | None = None,
) -> PostgresConnectionIdentity:
    """Create an immutable PostgreSQL connection identity descriptor."""

    if not isinstance(config, PostgresConnectionConfig):
        raise TypeError("config must be PostgresConnectionConfig")
    required_metadata = {
        "source": "postgres_connection_identity",
        "safe_dsn": config.safe_dsn(),
    }
    return PostgresConnectionIdentity(
        connection_id=new_postgres_connection_id(),
        config=config,
        created_at=utc_now(),
        metadata=_merge_metadata(required_metadata, metadata),
    )


def create_postgres_connection_state(
    *,
    identity: PostgresConnectionIdentity,
    status: PostgresConnectionStatus = PostgresConnectionStatus.CREATED,
    last_checked_at: datetime | None = None,
    failure_reason: str | None = None,
    metadata: dict[str, object] | None = None,
) -> PostgresConnectionState:
    """Create an immutable local PostgreSQL connection state descriptor."""

    if not isinstance(identity, PostgresConnectionIdentity):
        raise TypeError("identity must be PostgresConnectionIdentity")
    if not isinstance(status, PostgresConnectionStatus):
        raise TypeError("status must be PostgresConnectionStatus")
    required_metadata = {
        "source": "postgres_connection_state",
        "connection_id": str(identity.connection_id),
        "status": status.value,
    }
    return PostgresConnectionState(
        identity=identity,
        status=status,
        last_checked_at=last_checked_at,
        failure_reason=failure_reason,
        updated_at=utc_now(),
        metadata=_merge_metadata(required_metadata, metadata),
    )


def _ensure_port(value: int) -> int:
    port = ensure_positive_int(value, "port")
    if port > 65535:
        raise ValueError("port must be <= 65535")
    return port


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


def _reject_secret_metadata_keys(metadata: dict[str, object]) -> None:
    for key in metadata:
        if key in _SECRET_METADATA_KEYS:
            raise ValueError("metadata must not contain secret material keys")
