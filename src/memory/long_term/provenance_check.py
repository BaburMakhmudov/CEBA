"""Deterministic provenance checking utilities for CEBA Long-Term Memory."""

from __future__ import annotations

from dataclasses import dataclass, field

from ceba.memory.long_term.schema import (
    LongTermTrace,
    ProvenanceCheck,
    ProvenanceStatus,
    create_provenance_check,
    ensure_non_empty_string,
    ensure_probability,
)


__all__ = [
    "ProvenanceCheckPolicy",
    "ProvenanceCheckInput",
    "ProvenanceCheckResult",
    "ProvenanceChecker",
    "check_trace_provenance",
    "check_traces_provenance",
]


@dataclass(frozen=True)
class ProvenanceCheckPolicy:
    """Configuration object controlling deterministic provenance checks."""

    min_source_count: int = 1
    min_confidence: float = 0.70
    weak_confidence_threshold: float = 0.30
    require_semantic_hash: bool = True
    allow_missing_sources: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.min_source_count, bool) or not isinstance(
            self.min_source_count,
            int,
        ):
            raise TypeError("min_source_count must be int")
        if self.min_source_count < 0:
            raise ValueError("min_source_count must be non-negative")
        min_confidence = ensure_probability(
            self.min_confidence,
            "min_confidence",
        )
        weak_threshold = ensure_probability(
            self.weak_confidence_threshold,
            "weak_confidence_threshold",
        )
        if min_confidence < weak_threshold:
            raise ValueError(
                "min_confidence must be >= weak_confidence_threshold"
            )
        if not isinstance(self.require_semantic_hash, bool):
            raise TypeError("require_semantic_hash must be bool")
        if not isinstance(self.allow_missing_sources, bool):
            raise TypeError("allow_missing_sources must be bool")
        object.__setattr__(self, "min_confidence", min_confidence)
        object.__setattr__(self, "weak_confidence_threshold", weak_threshold)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "min_source_count": self.min_source_count,
            "min_confidence": self.min_confidence,
            "weak_confidence_threshold": self.weak_confidence_threshold,
            "require_semantic_hash": self.require_semantic_hash,
            "allow_missing_sources": self.allow_missing_sources,
        }


@dataclass(frozen=True)
class ProvenanceCheckInput:
    """Explicit input object for checking one LongTermTrace provenance."""

    trace: LongTermTrace
    source_ids: tuple[str, ...] | list[str] = ()
    confidence: float = 1.0
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.trace, LongTermTrace):
            raise TypeError("trace must be LongTermTrace")
        source_ids = _ensure_string_tuple(self.source_ids, "source_ids")
        _ensure_unique_strings(source_ids, "source_ids")
        confidence = ensure_probability(self.confidence, "confidence")
        metadata = _copy_metadata(self.metadata)
        object.__setattr__(self, "source_ids", source_ids)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "trace": self.trace.to_dict(),
            "source_ids": self.source_ids,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ProvenanceCheckResult:
    """Immutable result object returned by provenance checking."""

    checks: tuple[ProvenanceCheck, ...]
    failed_trace_ids: tuple[str, ...]
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        checks = _ensure_checks(self.checks)
        failed_trace_ids = _ensure_string_tuple(
            self.failed_trace_ids,
            "failed_trace_ids",
        )
        failure_reasons = _ensure_string_tuple(
            self.failure_reasons,
            "failure_reasons",
        )
        if len(failed_trace_ids) != len(failure_reasons):
            raise ValueError("failed_trace_ids and failure_reasons must match")
        object.__setattr__(self, "checks", checks)
        object.__setattr__(self, "failed_trace_ids", failed_trace_ids)
        object.__setattr__(self, "failure_reasons", failure_reasons)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable-friendly dictionary."""

        return {
            "checks": tuple(check.to_dict() for check in self.checks),
            "failed_trace_ids": self.failed_trace_ids,
            "failure_reasons": self.failure_reasons,
        }

    def check_count(self) -> int:
        """Return provenance check count."""

        return len(self.checks)

    def failed_count(self) -> int:
        """Return failed trace count."""

        return len(self.failed_trace_ids)


class ProvenanceChecker:
    """Deterministic service that creates ProvenanceCheck objects."""

    def __init__(self, policy: ProvenanceCheckPolicy | None = None) -> None:
        self._policy = (
            ProvenanceCheckPolicy() if policy is None else policy
        )
        if not isinstance(self._policy, ProvenanceCheckPolicy):
            raise TypeError("policy must be ProvenanceCheckPolicy")

    def check_one(
        self,
        *,
        check_input: ProvenanceCheckInput,
    ) -> ProvenanceCheckResult:
        """Check one trace provenance input."""

        if not isinstance(check_input, ProvenanceCheckInput):
            raise TypeError("check_input must be ProvenanceCheckInput")
        trace = check_input.trace
        source_ids = (
            check_input.source_ids
            if check_input.source_ids
            else trace.source_ids
        )
        status = self._status_for_input(
            trace=trace,
            source_ids=source_ids,
            confidence=check_input.confidence,
            metadata=check_input.metadata,
        )
        metadata = self._metadata_for_input(
            check_input=check_input,
            source_ids=source_ids,
        )
        check = create_provenance_check(
            trace_id=trace.trace_id,
            status=status,
            source_ids=source_ids,
            confidence=check_input.confidence,
            metadata=metadata,
        )
        return ProvenanceCheckResult(
            checks=(check,),
            failed_trace_ids=(),
            failure_reasons=(),
        )

    def check_many(
        self,
        *,
        inputs: tuple[ProvenanceCheckInput, ...] | list[ProvenanceCheckInput],
    ) -> ProvenanceCheckResult:
        """Check many trace provenance inputs in input order."""

        input_values = _ensure_inputs(inputs)
        checks: list[ProvenanceCheck] = []
        failed_trace_ids: list[str] = []
        failure_reasons: list[str] = []
        for check_input in input_values:
            result = self.check_one(check_input=check_input)
            checks.extend(result.checks)
            failed_trace_ids.extend(result.failed_trace_ids)
            failure_reasons.extend(result.failure_reasons)
        return ProvenanceCheckResult(
            checks=tuple(checks),
            failed_trace_ids=tuple(failed_trace_ids),
            failure_reasons=tuple(failure_reasons),
        )

    def _status_for_input(
        self,
        *,
        trace: LongTermTrace,
        source_ids: tuple[str, ...],
        confidence: float,
        metadata: dict[str, object],
    ) -> ProvenanceStatus:
        if trace.status.value == "INVALID":
            return ProvenanceStatus.INVALID
        if not source_ids and not self._policy.allow_missing_sources:
            return ProvenanceStatus.MISSING
        if len(source_ids) < self._policy.min_source_count:
            return ProvenanceStatus.MISSING
        if self._policy.require_semantic_hash and trace.semantic_hash == "":
            return ProvenanceStatus.INVALID
        if metadata.get("conflicted") is True:
            return ProvenanceStatus.CONFLICTED
        if confidence >= self._policy.min_confidence:
            return ProvenanceStatus.VALID
        if confidence >= self._policy.weak_confidence_threshold:
            return ProvenanceStatus.WEAK
        return ProvenanceStatus.INVALID

    def _metadata_for_input(
        self,
        *,
        check_input: ProvenanceCheckInput,
        source_ids: tuple[str, ...],
    ) -> dict[str, object]:
        trace = check_input.trace
        metadata = _copy_metadata(check_input.metadata)
        metadata.update(
            {
                "source": "provenance_checker",
                "trace_id": str(trace.trace_id),
                "trace_status": trace.status.value,
                "source_count": len(source_ids),
                "semantic_hash": trace.semantic_hash,
            }
        )
        metadata["source"] = "provenance_checker"
        return metadata


def check_trace_provenance(
    *,
    check_input: ProvenanceCheckInput,
    policy: ProvenanceCheckPolicy | None = None,
) -> ProvenanceCheckResult:
    """Check one trace provenance input."""

    checker = ProvenanceChecker(policy)
    return checker.check_one(check_input=check_input)


def check_traces_provenance(
    *,
    inputs: tuple[ProvenanceCheckInput, ...] | list[ProvenanceCheckInput],
    policy: ProvenanceCheckPolicy | None = None,
) -> ProvenanceCheckResult:
    """Check many trace provenance inputs."""

    checker = ProvenanceChecker(policy)
    return checker.check_many(inputs=inputs)


def _ensure_string_tuple(
    values: tuple[str, ...] | list[str],
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple or list")
    return tuple(ensure_non_empty_string(value, field_name) for value in values)


def _ensure_unique_strings(values: tuple[str, ...], field_name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must contain unique strings")


def _ensure_checks(
    values: tuple[ProvenanceCheck, ...] | list[ProvenanceCheck],
) -> tuple[ProvenanceCheck, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("checks must be tuple or list")
    checks = tuple(values)
    for check in checks:
        if not isinstance(check, ProvenanceCheck):
            raise TypeError("checks must contain ProvenanceCheck")
    check_ids = tuple(check.check_id for check in checks)
    if len(set(check_ids)) != len(check_ids):
        raise ValueError("check_id values must be unique")
    return checks


def _ensure_inputs(
    values: tuple[ProvenanceCheckInput, ...] | list[ProvenanceCheckInput],
) -> tuple[ProvenanceCheckInput, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("inputs must be tuple or list")
    inputs = tuple(values)
    for value in inputs:
        if not isinstance(value, ProvenanceCheckInput):
            raise TypeError("inputs must contain ProvenanceCheckInput")
    return inputs


def _copy_metadata(metadata: dict[str, object]) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(metadata)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
    return copied
