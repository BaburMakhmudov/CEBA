import math

__all__ = [
    "ensure_bool",
    "ensure_positive_int",
    "ensure_non_negative_int",
    "ensure_finite_number",
    "ensure_probability",
    "ensure_string_keyed_dict",
    "ensure_string_tuple",
    "ensure_unique_strings",
    "ensure_not_forbidden_key",
    "reject_forbidden_keys",
    "copy_string_keyed_dict",
]

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


def ensure_bool(value: bool, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool")
    return value


def ensure_positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def ensure_non_negative_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def ensure_finite_number(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number")
    checked: float = float(value)
    if not math.isfinite(checked):
        raise ValueError(f"{field_name} must be finite")
    return checked


def ensure_probability(value: float, field_name: str) -> float:
    checked: float = ensure_finite_number(value, field_name)
    if checked < 0.0 or checked > 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return checked


def ensure_string_keyed_dict(value: dict[str, object], field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a dict")
    reject_forbidden_keys(value, field_name)
    return value


def copy_string_keyed_dict(value: dict[str, object] | None, field_name: str = "metadata") -> dict[str, object]:
    if value is None:
        return {}
    return dict(ensure_string_keyed_dict(value, field_name))


def ensure_string_tuple(values: tuple[str, ...] | list[str], field_name: str) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be a tuple or list")
    checked: list[str] = []
    for index, value in enumerate(values):
        checked.append(ensure_not_forbidden_key(value, f"{field_name}[{index}]"))
    return tuple(checked)


def ensure_unique_strings(values: tuple[str, ...] | list[str], field_name: str) -> tuple[str, ...]:
    checked: tuple[str, ...] = ensure_string_tuple(values, field_name)
    if len(set(checked)) != len(checked):
        raise ValueError(f"{field_name} must contain unique strings")
    return checked


def ensure_not_forbidden_key(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    checked: str = value.strip()
    if not checked:
        raise ValueError(f"{field_name} must be non-empty")
    if checked in _FORBIDDEN_RUNTIME_STATE_NAMES:
        raise ValueError(f"{field_name} must not be a forbidden runtime semantic-state name")
    if checked in _FORBIDDEN_SECRET_MATERIAL_KEYS:
        raise ValueError(f"{field_name} must not be a forbidden secret material key")
    return checked


def reject_forbidden_keys(value: dict[str, object], field_name: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a dict")
    for key in value:
        ensure_not_forbidden_key(key, f"{field_name} key")
