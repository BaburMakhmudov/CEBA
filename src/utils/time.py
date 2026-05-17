from datetime import datetime, timezone

__all__ = [
    "utc_now",
    "ensure_timezone_aware",
    "ensure_utc",
    "datetime_to_iso",
    "datetime_from_iso",
    "duration_seconds",
    "is_expired",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_timezone_aware(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def ensure_utc(value: datetime, field_name: str) -> datetime:
    return ensure_timezone_aware(value, field_name).astimezone(timezone.utc)


def datetime_to_iso(value: datetime, field_name: str = "datetime") -> str:
    return ensure_timezone_aware(value, field_name).isoformat()


def datetime_from_iso(value: str, field_name: str = "datetime") -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized: str = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed: datetime = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO 8601 datetime") from exc
    return ensure_timezone_aware(parsed, field_name)


def duration_seconds(start: datetime, end: datetime) -> float:
    checked_start: datetime = ensure_timezone_aware(start, "start")
    checked_end: datetime = ensure_timezone_aware(end, "end")
    if checked_end < checked_start:
        raise ValueError("end must be greater than or equal to start")
    return float((checked_end - checked_start).total_seconds())


def is_expired(expires_at: datetime | None, *, at: datetime | None = None) -> bool:
    if expires_at is None:
        return False
    checked_expires_at: datetime = ensure_timezone_aware(expires_at, "expires_at")
    checked_at: datetime = utc_now() if at is None else ensure_timezone_aware(at, "at")
    return checked_at >= checked_expires_at
