import argparse
import json
import sys

from ceba.persistence.vector import (
    VectorBackend,
    VectorDistanceMetric,
    VectorIndexStatus,
    create_vector_index_config,
    create_vector_index_snapshot,
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
    return {"script": "rebuild_vector_index", "status": "error", "error": str(message)}


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
    return dict(parsed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare vector index rebuild descriptors.")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--collection-name", required=True)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--distance-metric", default="COSINE")
    parser.add_argument("--status", default="CREATED")
    parser.add_argument("--metadata-json")
    try:
        args = parser.parse_args(argv)
        metadata = _parse_metadata_json(args.metadata_json)
        config = create_vector_index_config(
            backend=VectorBackend[_reject_forbidden_value(args.backend, "backend").upper()],
            collection_name=_reject_forbidden_value(args.collection_name, "collection_name"),
            dimension=args.dimension,
            distance_metric=VectorDistanceMetric[
                _reject_forbidden_value(args.distance_metric, "distance_metric").upper()
            ],
            status=VectorIndexStatus[_reject_forbidden_value(args.status, "status").upper()],
            metadata=metadata,
        )
        snapshot = create_vector_index_snapshot(config=config, points=(), status=config.status, metadata=metadata)
        _write_json(
            {
                "script": "rebuild_vector_index",
                "status": "planned",
                "index_config": config.to_dict(),
                "snapshot": snapshot.to_dict(),
                "note": "vector DB is retrieval surface, not memory",
            }
        )
        return 0
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        _write_json(_safe_error_payload(str(exc)))
        return 1
    except Exception as exc:
        _write_json(_safe_error_payload(f"unexpected error: {exc}"))
        return 2


if __name__ == "__main__":
    sys.exit(main())
