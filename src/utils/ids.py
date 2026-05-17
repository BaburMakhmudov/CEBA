import re
from uuid import UUID, uuid4

__all__ = [
    "new_uuid",
    "uuid_to_str",
    "ensure_uuid",
    "ensure_uuid_string",
    "ensure_non_empty_string",
    "ensure_optional_non_empty_string",
    "normalize_identifier",
]

_FORBIDDEN_RUNTIME_STATE_NAMES: frozenset[str] = frozenset(
    {"M_t", "Mt", "semantic_state", "runtime_semantic_state"}
)
_IDENTIFIER_PATTERN: re.Pattern[str] = re.compile(r"^[a-z0-9_]+$")


def new_uuid() -> UUID:
    return uuid4()


def uuid_to_str(value: UUID, field_name: str = "uuid") -> str:
    return str(ensure_uuid(value, field_name))


def ensure_uuid(value: UUID, field_name: str) -> UUID:
    if not isinstance(value, UUID):
        raise TypeError(f"{field_name} must be a UUID")
    return value


def ensure_uuid_string(value: str, field_name: str) -> str:
    checked: str = ensure_non_empty_string(value, field_name)
    try:
        return str(UUID(checked))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a UUID string") from exc


def ensure_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    checked: str = value.strip()
    if not checked:
        raise ValueError(f"{field_name} must be non-empty")
    if checked in _FORBIDDEN_RUNTIME_STATE_NAMES:
        raise ValueError(f"{field_name} must not be a forbidden runtime semantic-state name")
    return checked


def ensure_optional_non_empty_string(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return ensure_non_empty_string(value, field_name)


def normalize_identifier(value: str, field_name: str = "identifier") -> str:
    checked: str = ensure_non_empty_string(value, field_name)
    normalized: str = checked.lower().replace(" ", "_").replace("-", "_")
    if not normalized:
        raise ValueError(f"{field_name} must normalize to a non-empty identifier")
    if normalized in _FORBIDDEN_RUNTIME_STATE_NAMES:
        raise ValueError(f"{field_name} must not be a forbidden runtime semantic-state name")
    if _IDENTIFIER_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{field_name} must contain only lowercase letters, digits, and underscores")
    return normalized
