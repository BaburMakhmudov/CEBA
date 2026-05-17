"""Deterministic runtime invariant validation utilities for CEBA Runtime."""

from __future__ import annotations

from dataclasses import dataclass, field

from ceba.runtime.context import (
    RuntimeContext,
    RuntimeInvariantFlag,
    ensure_non_empty_string,
)
from ceba.runtime.errors import RuntimeInvariantError


__all__ = [
    "RuntimeInvariantIssue",
    "RuntimeInvariantReport",
    "RuntimeInvariantPolicy",
    "RuntimeInvariantValidator",
    "validate_runtime_invariants",
    "require_runtime_invariants",
]


_VALID_SEVERITIES: frozenset[str] = frozenset({"INFO", "WARNING", "ERROR"})
_SEMANTIC_STATE_SYMBOLS: frozenset[str] = frozenset(
    {"M_t", "Mt", "semantic_state"}
)
_VECTOR_MEMORY_SURFACES: frozenset[str] = frozenset(
    {"vector_db", "qdrant", "pgvector", "lancedb"}
)
_STM_DATABASE_SURFACES: frozenset[str] = frozenset(
    {"stm_database", "short_term_database", "stm_db"}
)
_TEMPERATURE_MEMORY_SURFACES: frozenset[str] = frozenset(
    {"semantic_temperature", "temperature", "tau"}
)


@dataclass(frozen=True)
class RuntimeInvariantIssue:
    """Immutable runtime invariant issue descriptor."""

    invariant: RuntimeInvariantFlag
    code: str
    message: str
    severity: str
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.invariant, RuntimeInvariantFlag):
            raise TypeError("invariant must be RuntimeInvariantFlag")
        code = ensure_non_empty_string(self.code, "code")
        message = ensure_non_empty_string(self.message, "message")
        severity = ensure_non_empty_string(self.severity, "severity")
        if severity not in _VALID_SEVERITIES:
            raise ValueError("severity must be INFO, WARNING, or ERROR")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "invariant": self.invariant.value,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RuntimeInvariantReport:
    """Immutable report returned by runtime invariant validation."""

    valid: bool
    context_id: str
    issues: tuple[RuntimeInvariantIssue, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be bool")
        context_id = ensure_non_empty_string(self.context_id, "context_id")
        issues = _ensure_issues(self.issues)
        if self.valid and any(issue.severity == "ERROR" for issue in issues):
            raise ValueError("valid must be False when ERROR issues exist")
        object.__setattr__(self, "context_id", context_id)
        object.__setattr__(self, "issues", issues)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "valid": self.valid,
            "context_id": self.context_id,
            "issues": tuple(issue.to_dict() for issue in self.issues),
        }

    def error_count(self) -> int:
        """Return error issue count."""

        return _count_severity(self.issues, "ERROR")

    def warning_count(self) -> int:
        """Return warning issue count."""

        return _count_severity(self.issues, "WARNING")

    def info_count(self) -> int:
        """Return info issue count."""

        return _count_severity(self.issues, "INFO")

    def has_errors(self) -> bool:
        """Return whether the report contains error issues."""

        return self.error_count() > 0


@dataclass(frozen=True)
class RuntimeInvariantPolicy:
    """Configuration object controlling invariant severity and strictness."""

    require_world_model_before_delta: bool = True
    require_delta_only_writer: bool = True
    require_m_t_not_persisted: bool = True
    require_vector_db_not_memory: bool = True
    require_stm_not_database: bool = True
    require_temperature_not_memory: bool = True
    metadata_delta_attempt_key: str = "delta_attempted"
    metadata_memory_writer_key: str = "memory_writer"
    metadata_persisted_symbols_key: str = "persisted_symbols"
    metadata_memory_surfaces_key: str = "memory_surfaces"

    def __post_init__(self) -> None:
        for field_name in (
            "require_world_model_before_delta",
            "require_delta_only_writer",
            "require_m_t_not_persisted",
            "require_vector_db_not_memory",
            "require_stm_not_database",
            "require_temperature_not_memory",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")
        object.__setattr__(
            self,
            "metadata_delta_attempt_key",
            ensure_non_empty_string(
                self.metadata_delta_attempt_key,
                "metadata_delta_attempt_key",
            ),
        )
        object.__setattr__(
            self,
            "metadata_memory_writer_key",
            ensure_non_empty_string(
                self.metadata_memory_writer_key,
                "metadata_memory_writer_key",
            ),
        )
        object.__setattr__(
            self,
            "metadata_persisted_symbols_key",
            ensure_non_empty_string(
                self.metadata_persisted_symbols_key,
                "metadata_persisted_symbols_key",
            ),
        )
        object.__setattr__(
            self,
            "metadata_memory_surfaces_key",
            ensure_non_empty_string(
                self.metadata_memory_surfaces_key,
                "metadata_memory_surfaces_key",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "require_world_model_before_delta": (
                self.require_world_model_before_delta
            ),
            "require_delta_only_writer": self.require_delta_only_writer,
            "require_m_t_not_persisted": self.require_m_t_not_persisted,
            "require_vector_db_not_memory": self.require_vector_db_not_memory,
            "require_stm_not_database": self.require_stm_not_database,
            "require_temperature_not_memory": self.require_temperature_not_memory,
            "metadata_delta_attempt_key": self.metadata_delta_attempt_key,
            "metadata_memory_writer_key": self.metadata_memory_writer_key,
            "metadata_persisted_symbols_key": self.metadata_persisted_symbols_key,
            "metadata_memory_surfaces_key": self.metadata_memory_surfaces_key,
        }


class RuntimeInvariantValidator:
    """Deterministic validator for CEBA runtime architectural invariants."""

    def __init__(self, policy: RuntimeInvariantPolicy | None = None) -> None:
        self._policy = RuntimeInvariantPolicy() if policy is None else policy
        if not isinstance(self._policy, RuntimeInvariantPolicy):
            raise TypeError("policy must be RuntimeInvariantPolicy")

    def validate(
        self,
        *,
        context: RuntimeContext,
    ) -> RuntimeInvariantReport:
        """Validate runtime invariants for a context."""

        if not isinstance(context, RuntimeContext):
            raise TypeError("context must be RuntimeContext")
        issues: list[RuntimeInvariantIssue] = []
        if self._policy.require_world_model_before_delta:
            self._check_world_model_before_delta(context=context, issues=issues)
        if self._policy.require_delta_only_writer:
            self._check_delta_only_writer(context=context, issues=issues)
        if self._policy.require_m_t_not_persisted:
            self._check_m_t_not_persisted(context=context, issues=issues)
        if self._policy.require_vector_db_not_memory:
            self._check_vector_db_not_memory(context=context, issues=issues)
        if self._policy.require_stm_not_database:
            self._check_stm_not_database(context=context, issues=issues)
        if self._policy.require_temperature_not_memory:
            self._check_temperature_not_memory(context=context, issues=issues)
        self._add_satisfied_flag_issues(context=context, issues=issues)
        return _create_report(context_id=str(context.context_id), issues=tuple(issues))

    def require(
        self,
        *,
        context: RuntimeContext,
    ) -> RuntimeInvariantReport:
        """Validate runtime invariants and raise on errors."""

        report = self.validate(context=context)
        if report.has_errors():
            raise RuntimeInvariantError(
                "runtime invariant validation failed",
                context={"report": report.to_dict()},
            )
        return report

    def _check_world_model_before_delta(
        self,
        *,
        context: RuntimeContext,
        issues: list[RuntimeInvariantIssue],
    ) -> None:
        delta_attempted = _read_metadata_bool(
            context.metadata,
            self._policy.metadata_delta_attempt_key,
        )
        if delta_attempted and not _has_expected_observation(context):
            issues.append(
                _issue(
                    invariant=RuntimeInvariantFlag.WORLD_MODEL_BEFORE_DELTA,
                    code="WORLD_MODEL_BEFORE_DELTA_VIOLATION",
                    message="delta attempted before expected observation exists",
                    severity="ERROR",
                    metadata={
                        "metadata_key": self._policy.metadata_delta_attempt_key,
                    },
                )
            )

    def _check_delta_only_writer(
        self,
        *,
        context: RuntimeContext,
        issues: list[RuntimeInvariantIssue],
    ) -> None:
        memory_writer = _read_metadata_string(
            context.metadata,
            self._policy.metadata_memory_writer_key,
        )
        if memory_writer is not None and memory_writer not in {"delta", "none"}:
            issues.append(
                _issue(
                    invariant=RuntimeInvariantFlag.DELTA_ONLY_WRITER,
                    code="DELTA_ONLY_WRITER_VIOLATION",
                    message="non-delta memory writer detected",
                    severity="ERROR",
                    metadata={
                        "memory_writer": memory_writer,
                        "metadata_key": self._policy.metadata_memory_writer_key,
                    },
                )
            )

    def _check_m_t_not_persisted(
        self,
        *,
        context: RuntimeContext,
        issues: list[RuntimeInvariantIssue],
    ) -> None:
        persisted_symbols = _read_metadata_string_collection(
            context.metadata,
            self._policy.metadata_persisted_symbols_key,
        )
        matches = tuple(
            symbol for symbol in persisted_symbols if symbol in _SEMANTIC_STATE_SYMBOLS
        )
        if matches:
            issues.append(
                _issue(
                    invariant=RuntimeInvariantFlag.M_T_NOT_PERSISTED,
                    code="M_T_NOT_PERSISTED_VIOLATION",
                    message="runtime semantic state marked for persistence",
                    severity="ERROR",
                    metadata={
                        "persisted_symbols": matches,
                        "metadata_key": (
                            self._policy.metadata_persisted_symbols_key
                        ),
                    },
                )
            )

    def _check_vector_db_not_memory(
        self,
        *,
        context: RuntimeContext,
        issues: list[RuntimeInvariantIssue],
    ) -> None:
        memory_surfaces = _read_metadata_string_collection(
            context.metadata,
            self._policy.metadata_memory_surfaces_key,
        )
        matches = tuple(
            surface for surface in memory_surfaces if surface in _VECTOR_MEMORY_SURFACES
        )
        if matches:
            issues.append(
                _issue(
                    invariant=RuntimeInvariantFlag.VECTOR_DB_NOT_MEMORY,
                    code="VECTOR_DB_NOT_MEMORY_VIOLATION",
                    message="vector database marked as memory",
                    severity="ERROR",
                    metadata={
                        "memory_surfaces": matches,
                        "metadata_key": self._policy.metadata_memory_surfaces_key,
                    },
                )
            )

    def _check_stm_not_database(
        self,
        *,
        context: RuntimeContext,
        issues: list[RuntimeInvariantIssue],
    ) -> None:
        memory_surfaces = _read_metadata_string_collection(
            context.metadata,
            self._policy.metadata_memory_surfaces_key,
        )
        matches = tuple(
            surface for surface in memory_surfaces if surface in _STM_DATABASE_SURFACES
        )
        if matches:
            issues.append(
                _issue(
                    invariant=RuntimeInvariantFlag.STM_NOT_DATABASE,
                    code="STM_NOT_DATABASE_VIOLATION",
                    message="short-term memory marked as database",
                    severity="ERROR",
                    metadata={
                        "memory_surfaces": matches,
                        "metadata_key": self._policy.metadata_memory_surfaces_key,
                    },
                )
            )

    def _check_temperature_not_memory(
        self,
        *,
        context: RuntimeContext,
        issues: list[RuntimeInvariantIssue],
    ) -> None:
        memory_surfaces = _read_metadata_string_collection(
            context.metadata,
            self._policy.metadata_memory_surfaces_key,
        )
        matches = tuple(
            surface
            for surface in memory_surfaces
            if surface in _TEMPERATURE_MEMORY_SURFACES
        )
        if matches:
            issues.append(
                _issue(
                    invariant=RuntimeInvariantFlag.TEMPERATURE_NOT_MEMORY,
                    code="TEMPERATURE_NOT_MEMORY_VIOLATION",
                    message="semantic temperature marked as memory",
                    severity="ERROR",
                    metadata={
                        "memory_surfaces": matches,
                        "metadata_key": self._policy.metadata_memory_surfaces_key,
                    },
                )
            )

    def _add_satisfied_flag_issues(
        self,
        *,
        context: RuntimeContext,
        issues: list[RuntimeInvariantIssue],
    ) -> None:
        error_invariants = {
            issue.invariant for issue in issues if issue.severity == "ERROR"
        }
        for flag in context.invariant_flags:
            if flag in error_invariants:
                continue
            issues.append(
                _issue(
                    invariant=flag,
                    code=f"{flag.value}_SATISFIED",
                    message="runtime invariant satisfied",
                    severity="INFO",
                )
            )


def validate_runtime_invariants(
    *,
    context: RuntimeContext,
    policy: RuntimeInvariantPolicy | None = None,
) -> RuntimeInvariantReport:
    """Validate runtime invariants for a context."""

    validator = RuntimeInvariantValidator(policy)
    return validator.validate(context=context)


def require_runtime_invariants(
    *,
    context: RuntimeContext,
    policy: RuntimeInvariantPolicy | None = None,
) -> RuntimeInvariantReport:
    """Validate runtime invariants and raise on errors."""

    validator = RuntimeInvariantValidator(policy)
    return validator.require(context=context)


def _ensure_issues(
    values: tuple[RuntimeInvariantIssue, ...] | list[RuntimeInvariantIssue],
) -> tuple[RuntimeInvariantIssue, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("issues must be tuple or list")
    issues = tuple(values)
    for issue in issues:
        if not isinstance(issue, RuntimeInvariantIssue):
            raise TypeError("issues must contain RuntimeInvariantIssue")
    return issues


def _issue(
    *,
    invariant: RuntimeInvariantFlag,
    code: str,
    message: str,
    severity: str,
    metadata: dict[str, object] | None = None,
) -> RuntimeInvariantIssue:
    return RuntimeInvariantIssue(
        invariant=invariant,
        code=code,
        message=message,
        severity=severity,
        metadata={} if metadata is None else metadata,
    )


def _create_report(
    *,
    context_id: str,
    issues: tuple[RuntimeInvariantIssue, ...],
) -> RuntimeInvariantReport:
    valid = not any(issue.severity == "ERROR" for issue in issues)
    return RuntimeInvariantReport(
        valid=valid,
        context_id=context_id,
        issues=issues,
    )


def _count_severity(
    issues: tuple[RuntimeInvariantIssue, ...],
    severity: str,
) -> int:
    return sum(1 for issue in issues if issue.severity == severity)


def _read_metadata_bool(metadata: dict[str, object], key: str) -> bool:
    value = metadata.get(key)
    return value is True


def _read_metadata_string(
    metadata: dict[str, object],
    key: str,
) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    return value


def _read_metadata_string_collection(
    metadata: dict[str, object],
    key: str,
) -> tuple[str, ...]:
    value = metadata.get(key)
    if value is None:
        return ()
    if not isinstance(value, (tuple, list, set)):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _has_expected_observation(context: RuntimeContext) -> bool:
    if context.expected_observation is not None:
        return True
    if context.prediction_result is None:
        return False
    return context.prediction_result.expected_observation is not None


def _copy_metadata(metadata: dict[str, object]) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(metadata)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
    return copied
