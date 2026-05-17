import argparse
import json
import sys
from dataclasses import dataclass, field
from enum import Enum

_FORBIDDEN_STATE_NAMES: frozenset[str] = frozenset(
    {"M_t", "Mt", "semantic_state", "runtime_semantic_state"}
)
_FORBIDDEN_SECRET_KEYS: frozenset[str] = frozenset(
    {
        "password",
        "raw_password",
        "secret",
        "raw_secret",
        "api_key",
        "token",
        "access_token",
        "refresh_token",
        "private_key",
        "credential",
    }
)


class MigrationDirection(Enum):
    UP = "UP"
    DOWN = "DOWN"
    VERIFY = "VERIFY"


@dataclass(frozen=True)
class MigrationInput:
    version: int
    filename: str
    checksum: str | None

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise TypeError("version must be an int")
        if self.version <= 0:
            raise ValueError("version must be positive")
        filename = _reject_forbidden_value(self.filename, "filename")
        if not filename.endswith(".sql"):
            raise ValueError("filename must end with .sql")
        object.__setattr__(self, "filename", filename)
        checksum = None if self.checksum is None else _reject_forbidden_value(self.checksum, "checksum")
        object.__setattr__(self, "checksum", checksum)

    def to_dict(self) -> dict[str, object]:
        return {"version": self.version, "filename": self.filename, "checksum": self.checksum}


@dataclass(frozen=True)
class MigrationPlan:
    direction: str
    current_version: int
    target_version: int | None
    migrations: tuple[MigrationInput, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.direction not in {item.value for item in MigrationDirection}:
            raise ValueError("direction must be UP, DOWN, or VERIFY")
        _ensure_non_negative_int(self.current_version, "current_version")
        if self.target_version is not None:
            _ensure_non_negative_int(self.target_version, "target_version")
        if not isinstance(self.migrations, tuple):
            object.__setattr__(self, "migrations", tuple(self.migrations))
        for migration in self.migrations:
            if not isinstance(migration, MigrationInput):
                raise TypeError("migrations must contain MigrationInput")
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "direction": self.direction,
            "current_version": self.current_version,
            "target_version": self.target_version,
            "migrations": [migration.to_dict() for migration in self.migrations],
            "metadata": dict(self.metadata),
        }


def _write_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def _safe_error_payload(message: str) -> dict[str, object]:
    return {"script": "migrate", "status": "error", "error": str(message)}


def _reject_forbidden_value(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    checked = value.strip()
    if not checked:
        raise ValueError(f"{field_name} must be non-empty")
    if checked in _FORBIDDEN_STATE_NAMES:
        raise ValueError(f"{field_name} must not be a forbidden runtime semantic-state name")
    return checked


def _reject_forbidden_keys(value: object, field_name: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field_name} keys must be strings")
            if key in _FORBIDDEN_STATE_NAMES or key in _FORBIDDEN_SECRET_KEYS:
                raise ValueError(f"{field_name} contains forbidden key")
            _reject_forbidden_keys(item, f"{field_name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_keys(item, f"{field_name}[{index}]")


def _copy_metadata(value: dict[str, object]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("metadata must be a dict")
    _reject_forbidden_keys(value, "metadata")
    return dict(value)


def _parse_metadata_json(value: str | None) -> dict[str, object]:
    if value is None:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("metadata-json must be a JSON object")
    return _copy_metadata(parsed)


def _ensure_non_negative_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _parse_migration(value: str) -> MigrationInput:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise ValueError("migration must use version:filename:checksum format")
    version_text, filename, checksum = parts
    checksum_value = None if checksum.strip() == "" else checksum
    return MigrationInput(version=int(version_text), filename=filename, checksum=checksum_value)


def _validate_unique(migrations: tuple[MigrationInput, ...]) -> None:
    versions = [migration.version for migration in migrations]
    filenames = [migration.filename for migration in migrations]
    if len(set(versions)) != len(versions):
        raise ValueError("duplicate migration versions rejected")
    if len(set(filenames)) != len(filenames):
        raise ValueError("duplicate migration filenames rejected")


def _planned_migrations(
    migrations: tuple[MigrationInput, ...],
    direction: MigrationDirection,
    current_version: int,
    target_version: int | None,
) -> tuple[MigrationInput, ...]:
    if direction is MigrationDirection.UP:
        selected = [
            migration
            for migration in migrations
            if migration.version > current_version
            and (target_version is None or migration.version <= target_version)
        ]
        return tuple(sorted(selected, key=lambda item: item.version))
    if direction is MigrationDirection.DOWN:
        selected = [
            migration
            for migration in migrations
            if migration.version <= current_version
            and (target_version is None or migration.version > target_version)
        ]
        return tuple(sorted(selected, key=lambda item: item.version, reverse=True))
    selected = [
        migration
        for migration in migrations
        if migration.version <= current_version
        and (target_version is None or migration.version <= target_version)
    ]
    return tuple(sorted(selected, key=lambda item: item.version))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare CEBA migration planning output.")
    parser.add_argument("--direction", required=True)
    parser.add_argument("--current-version", type=int, default=0)
    parser.add_argument("--target-version", type=int)
    parser.add_argument("--migration", action="append", default=[])
    parser.add_argument("--metadata-json")
    try:
        args = parser.parse_args(argv)
        direction = MigrationDirection[_reject_forbidden_value(args.direction, "direction").upper()]
        current_version = _ensure_non_negative_int(args.current_version, "current_version")
        target_version = (
            None
            if args.target_version is None
            else _ensure_non_negative_int(args.target_version, "target_version")
        )
        metadata = _parse_metadata_json(args.metadata_json)
        migrations = tuple(_parse_migration(value) for value in args.migration)
        _validate_unique(migrations)
        plan = MigrationPlan(
            direction=direction.value,
            current_version=current_version,
            target_version=target_version,
            migrations=_planned_migrations(migrations, direction, current_version, target_version),
            metadata=metadata,
        )
        _write_json({"script": "migrate", "status": "planned", "plan": plan.to_dict()})
        return 0
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        _write_json(_safe_error_payload(str(exc)))
        return 1
    except Exception as exc:
        _write_json(_safe_error_payload(f"unexpected error: {exc}"))
        return 2


if __name__ == "__main__":
    sys.exit(main())
