import argparse
import json
import sys

from ceba.persistence.postgres import (
    PostgresConnectionStatus,
    PostgresSSLMode,
    create_default_schema_manifest,
    create_postgres_connection_config,
    create_postgres_connection_identity,
    create_postgres_connection_state,
)

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


def _write_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def _safe_error_payload(message: str) -> dict[str, object]:
    return {"script": "init_db", "status": "error", "error": str(message)}


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


def _parse_metadata_json(value: str | None) -> dict[str, object]:
    if value is None:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("metadata-json must be a JSON object")
    _reject_forbidden_keys(parsed, "metadata")
    return parsed


def _ssl_mode(value: str) -> PostgresSSLMode:
    checked = _reject_forbidden_value(value, "ssl_mode").upper().replace("-", "_")
    try:
        return PostgresSSLMode[checked]
    except KeyError as exc:
        raise ValueError("ssl-mode must map to PostgresSSLMode") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare CEBA PostgreSQL initialization descriptors.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--database", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password-ref")
    parser.add_argument("--ssl-mode", default="PREFER")
    parser.add_argument("--application-name", default="ceba")
    parser.add_argument("--schema-name", default="ceba")
    parser.add_argument("--schema-version", type=int, default=1)
    parser.add_argument("--metadata-json")
    try:
        args = parser.parse_args(argv)
        metadata = _parse_metadata_json(args.metadata_json)
        config = create_postgres_connection_config(
            host=_reject_forbidden_value(args.host, "host"),
            port=args.port,
            database=_reject_forbidden_value(args.database, "database"),
            user=_reject_forbidden_value(args.user, "user"),
            password_ref=(
                None
                if args.password_ref is None
                else _reject_forbidden_value(args.password_ref, "password_ref")
            ),
            ssl_mode=_ssl_mode(args.ssl_mode),
            application_name=_reject_forbidden_value(args.application_name, "application_name"),
            metadata=metadata,
        )
        identity = create_postgres_connection_identity(config=config, metadata=metadata)
        state = create_postgres_connection_state(
            identity=identity,
            status=PostgresConnectionStatus.CONFIGURED,
            metadata=metadata,
        )
        manifest = create_default_schema_manifest(
            schema_name=_reject_forbidden_value(args.schema_name, "schema_name"),
            version=args.schema_version,
            metadata=metadata,
        )
        _write_json(
            {
                "script": "init_db",
                "status": "planned",
                "connection": state.to_dict(),
                "schema_manifest": manifest.to_dict(),
            }
        )
        return 0
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        _write_json(_safe_error_payload(str(exc)))
        return 1
    except Exception as exc:
        _write_json(_safe_error_payload(f"unexpected error: {exc}"))
        return 2


if __name__ == "__main__":
    sys.exit(main())
