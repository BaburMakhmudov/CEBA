import json
import math
from typing import TypeAlias

__all__ = [
    "JsonSafeScalar",
    "JsonSafeValue",
    "is_json_safe_scalar",
    "normalize_json_safe",
    "ensure_json_object",
    "json_dumps_deterministic",
]

JsonSafeScalar: TypeAlias = str | int | float | bool | None
JsonSafeValue: TypeAlias = JsonSafeScalar | dict[str, object] | list[object]

_FORBIDDEN_RUNTIME_STATE_NAMES: frozenset[str] = frozenset(
    {"M_t", "Mt", "semantic_state", "runtime_semantic_state"}
)
_FORBIDDEN_SECRET_MATERIAL_KEYS: frozenset[str] = frozenset(
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


def is_json_safe_scalar(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def normalize_json_safe(value: object, *, field_name: str = "value") -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must not contain non-finite floats")
        return value
    if isinstance(value, dict):
        normalized_dict: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field_name} keys must be strings")
            _reject_forbidden_key(key, f"{field_name}.{key}")
            normalized_dict[key] = normalize_json_safe(item, field_name=f"{field_name}.{key}")
        return normalized_dict
    if isinstance(value, (list, tuple)):
        return [
            normalize_json_safe(item, field_name=f"{field_name}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{field_name} contains unsupported JSON value type {type(value).__name__}")


def ensure_json_object(value: dict[str, object] | None, *, field_name: str = "value") -> dict[str, object]:
    if value is None:
        return {}
    normalized: object = normalize_json_safe(value, field_name=field_name)
    if not isinstance(normalized, dict):
        raise TypeError(f"{field_name} must be a JSON object")
    return normalized


def json_dumps_deterministic(value: object) -> str:
    normalized: object = normalize_json_safe(value)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _reject_forbidden_key(value: str, field_name: str) -> None:
    if value in _FORBIDDEN_RUNTIME_STATE_NAMES:
        raise ValueError(f"{field_name} must not be a forbidden runtime semantic-state key")
    if value in _FORBIDDEN_SECRET_MATERIAL_KEYS:
        raise ValueError(f"{field_name} must not be a forbidden secret material key")
