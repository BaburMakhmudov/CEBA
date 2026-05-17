"""Typed exceptions for CEBA World Model."""

from __future__ import annotations


__all__ = [
    "WorldModelError",
    "WorldModelValidationError",
    "WorldModelStateError",
    "WorldModelEvidenceError",
    "PredictionInputError",
    "PredictionError",
    "ExpectedObservationError",
    "WorldModelSnapshotError",
    "WorldModelInvariantError",
]


class WorldModelError(Exception):
    """Base exception for all CEBA world model errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "WORLD_MODEL_ERROR",
        context: dict[str, object] | None = None,
    ) -> None:
        if not isinstance(message, str):
            raise TypeError("message must be str")
        stripped_message = message.strip()
        if not stripped_message:
            raise ValueError("message must be non-empty")
        if not isinstance(code, str):
            raise TypeError("code must be str")
        stripped_code = code.strip()
        if not stripped_code:
            raise ValueError("code must be non-empty")
        copied_context = _copy_context({} if context is None else context)

        self.message = stripped_message
        self.code = stripped_code
        self.context = copied_context
        super().__init__(stripped_message)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly error dictionary."""

        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "code": self.code,
            "context": dict(self.context),
        }


class WorldModelValidationError(WorldModelError):
    """Raised for invalid world model input or schema-adjacent validation."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "WORLD_MODEL_VALIDATION_ERROR",
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, code=code, context=context)


class WorldModelStateError(WorldModelError):
    """Raised for invalid world state references or lifecycle failures."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "WORLD_MODEL_STATE_ERROR",
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, code=code, context=context)


class WorldModelEvidenceError(WorldModelError):
    """Raised for invalid evidence references or consistency failures."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "WORLD_MODEL_EVIDENCE_ERROR",
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, code=code, context=context)


class PredictionInputError(WorldModelError):
    """Raised for invalid prediction input construction or consistency."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "PREDICTION_INPUT_ERROR",
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, code=code, context=context)


class PredictionError(WorldModelError):
    """Raised when prediction cannot be produced by runtime logic."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "PREDICTION_ERROR",
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, code=code, context=context)


class ExpectedObservationError(WorldModelError):
    """Raised for expected observation construction or consistency failures."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "EXPECTED_OBSERVATION_ERROR",
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, code=code, context=context)


class WorldModelSnapshotError(WorldModelError):
    """Raised for invalid world model snapshot consistency."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "WORLD_MODEL_SNAPSHOT_ERROR",
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, code=code, context=context)


class WorldModelInvariantError(WorldModelError):
    """Raised for world model invariant violations."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "WORLD_MODEL_INVARIANT_ERROR",
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, code=code, context=context)


def _copy_context(context: dict[str, object]) -> dict[str, object]:
    if not isinstance(context, dict):
        raise TypeError("context must be dict[str, object]")
    copied = dict(context)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("context keys must be strings")
    return copied
