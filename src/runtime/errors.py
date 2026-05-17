"""Strict exception types for CEBA Runtime."""

__all__ = [
    "RuntimeErrorBase",
    "RuntimeValidationError",
    "RuntimeContextError",
    "RuntimeCycleError",
    "RuntimeEngineError",
    "RuntimeSchedulerError",
    "RuntimeTransactionError",
    "RuntimeCheckpointError",
    "RuntimeInvariantError",
]


class RuntimeErrorBase(Exception):
    """Base exception for all CEBA runtime errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "RUNTIME_ERROR",
        context: dict[str, object] | None = None,
    ) -> None:
        message_value = _ensure_non_empty_string(message, "message")
        code_value = _ensure_non_empty_string(code, "code")
        context_value = _copy_context({} if context is None else context)
        self.message = message_value
        self.code = code_value
        self.context = context_value
        super().__init__(message_value)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly error dictionary."""

        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "code": self.code,
            "context": dict(self.context),
        }


class RuntimeValidationError(RuntimeErrorBase):
    """Raised for invalid runtime inputs or unsafe object combinations."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "RUNTIME_VALIDATION_ERROR",
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, code=code, context=context)


class RuntimeContextError(RuntimeErrorBase):
    """Raised for invalid or inconsistent runtime context state."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "RUNTIME_CONTEXT_ERROR",
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, code=code, context=context)


class RuntimeCycleError(RuntimeErrorBase):
    """Raised when a runtime cycle cannot be executed safely."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "RUNTIME_CYCLE_ERROR",
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, code=code, context=context)


class RuntimeEngineError(RuntimeErrorBase):
    """Raised for runtime engine lifecycle or coordination failures."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "RUNTIME_ENGINE_ERROR",
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, code=code, context=context)


class RuntimeSchedulerError(RuntimeErrorBase):
    """Raised for scheduler consistency failures or unsafe task ordering."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "RUNTIME_SCHEDULER_ERROR",
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, code=code, context=context)


class RuntimeTransactionError(RuntimeErrorBase):
    """Raised for runtime transaction failures."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "RUNTIME_TRANSACTION_ERROR",
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, code=code, context=context)


class RuntimeCheckpointError(RuntimeErrorBase):
    """Raised for runtime checkpoint failures."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "RUNTIME_CHECKPOINT_ERROR",
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, code=code, context=context)


class RuntimeInvariantError(RuntimeErrorBase):
    """Raised for runtime invariant violations."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "RUNTIME_INVARIANT_ERROR",
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, code=code, context=context)


def _ensure_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must be non-empty")
    return stripped


def _copy_context(context: dict[str, object]) -> dict[str, object]:
    if not isinstance(context, dict):
        raise TypeError("context must be dict[str, object]")
    copied = dict(context)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("context keys must be strings")
    return copied
