__all__ = [
    "CebaError",
    "CebaValidationError",
    "CebaInvariantError",
    "CebaSerializationError",
    "CebaHashingError",
    "CebaConfigurationError",
]


class CebaError(Exception):
    """Base exception for CEBA utility-level errors."""


class CebaValidationError(CebaError):
    """Raised when validation fails."""


class CebaInvariantError(CebaError):
    """Raised when an internal invariant is violated."""


class CebaSerializationError(CebaError):
    """Raised when JSON-safe serialization or normalization fails."""


class CebaHashingError(CebaError):
    """Raised when deterministic hashing cannot be completed safely."""


class CebaConfigurationError(CebaError):
    """Raised when configuration-like values are invalid."""
