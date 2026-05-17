"""Deterministic local metrics descriptors for CEBA observability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from ceba.observability.logging import (
    ensure_finite_number,
    ensure_non_empty_string,
    ensure_optional_non_empty_string,
    ensure_positive_int,
    ensure_timezone_aware,
    utc_now,
)


__all__ = [
    "MetricKind",
    "MetricUnit",
    "MetricStatus",
    "MetricSample",
    "MetricSeries",
    "MetricRegistryPolicy",
    "MetricRegistryResult",
    "MetricRegistry",
    "new_metric_sample_id",
    "create_metric_sample",
    "create_metric_series",
    "build_metric_registry",
]

_FORBIDDEN_STATE = {"M_t", "Mt", "semantic_state", "runtime_semantic_state"}
_FORBIDDEN_SECRET = {"password", "raw_password", "secret", "raw_secret", "api_key", "token", "access_token", "refresh_token", "private_key", "credential"}


class MetricKind(Enum):
    COUNTER = "COUNTER"
    GAUGE = "GAUGE"
    HISTOGRAM = "HISTOGRAM"
    TIMER = "TIMER"
    RATIO = "RATIO"
    UNKNOWN = "UNKNOWN"


class MetricUnit(Enum):
    COUNT = "COUNT"
    SECONDS = "SECONDS"
    MILLISECONDS = "MILLISECONDS"
    BYTES = "BYTES"
    PERCENT = "PERCENT"
    SCORE = "SCORE"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"


class MetricStatus(Enum):
    CREATED = "CREATED"
    RECORDED = "RECORDED"
    EXPORTED = "EXPORTED"
    DROPPED = "DROPPED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class MetricSample:
    sample_id: UUID
    name: str
    kind: MetricKind
    unit: MetricUnit
    value: float
    status: MetricStatus
    recorded_at: datetime
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, UUID):
            raise TypeError("sample_id must be UUID")
        name = ensure_non_empty_string(self.name, "name")
        _reject_name(name, "name")
        if not isinstance(self.kind, MetricKind):
            raise TypeError("kind must be MetricKind")
        if not isinstance(self.unit, MetricUnit):
            raise TypeError("unit must be MetricUnit")
        if not isinstance(self.status, MetricStatus):
            raise TypeError("status must be MetricStatus")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "value", ensure_finite_number(self.value, "value"))
        object.__setattr__(self, "recorded_at", ensure_timezone_aware(self.recorded_at, "recorded_at"))
        object.__setattr__(self, "tags", _copy_tags(self.tags))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {"sample_id": str(self.sample_id), "name": self.name, "kind": self.kind.value, "unit": self.unit.value, "value": self.value, "status": self.status.value, "recorded_at": self.recorded_at.isoformat(), "tags": dict(self.tags), "metadata": dict(self.metadata)}


@dataclass(frozen=True)
class MetricSeries:
    name: str
    samples: tuple[MetricSample, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = ensure_non_empty_string(self.name, "name")
        _reject_name(name, "name")
        samples = _sample_tuple(self.samples)
        if any(sample.name != name for sample in samples):
            raise ValueError("every sample.name must equal name")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "samples": [sample.to_dict() for sample in self.samples], "metadata": dict(self.metadata)}

    def sample_count(self) -> int:
        return len(self.samples)

    def latest(self) -> MetricSample | None:
        if not self.samples:
            return None
        return sorted(self.samples, key=lambda sample: (sample.recorded_at, str(sample.sample_id)), reverse=True)[0]


@dataclass(frozen=True)
class MetricRegistryPolicy:
    max_samples: int = 100000
    drop_unknown_metrics: bool = False
    require_non_negative_counters: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_samples", ensure_positive_int(self.max_samples, "max_samples"))
        if not isinstance(self.drop_unknown_metrics, bool):
            raise TypeError("drop_unknown_metrics must be bool")
        if not isinstance(self.require_non_negative_counters, bool):
            raise TypeError("require_non_negative_counters must be bool")


@dataclass(frozen=True)
class MetricRegistryResult:
    samples: tuple[MetricSample, ...] = ()
    series: tuple[MetricSeries, ...] = ()
    changed: bool = False
    reason: str = "metric registry result"
    rejected_sample_ids: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        samples = _sample_tuple(self.samples)
        series = _series_tuple(self.series)
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        rejected = _string_tuple(self.rejected_sample_ids, "rejected_sample_ids")
        reasons = _string_tuple(self.rejection_reasons, "rejection_reasons")
        if len(rejected) != len(reasons):
            raise ValueError("rejected_sample_ids and rejection_reasons must have same length")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "series", series)
        object.__setattr__(self, "reason", ensure_non_empty_string(self.reason, "reason"))
        object.__setattr__(self, "rejected_sample_ids", rejected)
        object.__setattr__(self, "rejection_reasons", reasons)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))


class MetricRegistry:
    def __init__(self, *, samples: tuple[MetricSample, ...] | list[MetricSample] = (), policy: MetricRegistryPolicy | None = None) -> None:
        self._policy = MetricRegistryPolicy() if policy is None else policy
        if not isinstance(self._policy, MetricRegistryPolicy):
            raise TypeError("policy must be MetricRegistryPolicy")
        self._samples = _sample_tuple(samples)

    def samples(self) -> tuple[MetricSample, ...]:
        return self._samples

    def sample_count(self) -> int:
        return len(self._samples)

    def record(self, *, sample: MetricSample) -> MetricRegistryResult:
        if not isinstance(sample, MetricSample):
            raise TypeError("sample must be MetricSample")
        reason = _reject_sample(sample, self._samples, self._policy)
        if reason is not None:
            return _result(reason=reason, rejected=(str(sample.sample_id),), rejection_reasons=(reason,))
        recorded = MetricSample(sample.sample_id, sample.name, sample.kind, sample.unit, sample.value, MetricStatus.RECORDED, sample.recorded_at, sample.tags, sample.metadata)
        self._samples = self._samples + (recorded,)
        return _result(samples=(recorded,), changed=True, reason="metric sample recorded")

    def record_value(self, *, name: str, kind: MetricKind, unit: MetricUnit = MetricUnit.NONE, value: float, tags: dict[str, str] | None = None, metadata: dict[str, object] | None = None) -> MetricRegistryResult:
        return self.record(sample=create_metric_sample(name=name, kind=kind, unit=unit, value=value, tags=tags, metadata=metadata))

    def query(self, *, name: str | None = None, kind: MetricKind | None = None, limit: int = 100) -> MetricRegistryResult:
        name = ensure_optional_non_empty_string(name, "name")
        if kind is not None and not isinstance(kind, MetricKind):
            raise TypeError("kind must be None or MetricKind")
        limit = ensure_positive_int(limit, "limit")
        samples = [sample for sample in self._samples if (name is None or sample.name == name) and (kind is None or sample.kind is kind)]
        samples.sort(key=lambda sample: (sample.recorded_at, str(sample.sample_id)), reverse=True)
        return _result(samples=tuple(samples[:limit]), reason="metric samples queried")

    def series(self, *, name: str) -> MetricRegistryResult:
        name = ensure_non_empty_string(name, "name")
        series = create_metric_series(name=name, samples=tuple(sample for sample in self._samples if sample.name == name))
        return _result(series=(series,), samples=series.samples, reason="metric series queried")

    def mark_exported(self, *, sample_id: str) -> MetricRegistryResult:
        sample_id = ensure_non_empty_string(sample_id, "sample_id")
        for index, sample in enumerate(self._samples):
            if str(sample.sample_id) == sample_id:
                updated = MetricSample(sample.sample_id, sample.name, sample.kind, sample.unit, sample.value, MetricStatus.EXPORTED, sample.recorded_at, sample.tags, sample.metadata)
                self._samples = self._samples[:index] + (updated,) + self._samples[index + 1 :]
                return _result(samples=(updated,), changed=True, reason="metric sample exported")
        return _result(reason="metric sample missing")

    def replace_samples(self, *, samples: tuple[MetricSample, ...] | list[MetricSample]) -> MetricRegistryResult:
        self._samples = _sample_tuple(samples)
        return _result(samples=self._samples, changed=True, reason="metric samples replaced")


def new_metric_sample_id() -> UUID:
    return uuid4()


def create_metric_sample(*, name: str, kind: MetricKind, unit: MetricUnit = MetricUnit.NONE, value: float, tags: dict[str, str] | None = None, metadata: dict[str, object] | None = None) -> MetricSample:
    return MetricSample(new_metric_sample_id(), name, kind, unit, value, MetricStatus.CREATED, utc_now(), {} if tags is None else dict(tags), _merge_metadata({"source": "metric_sample"}, metadata))


def create_metric_series(*, name: str, samples: tuple[MetricSample, ...] | list[MetricSample] = (), metadata: dict[str, object] | None = None) -> MetricSeries:
    return MetricSeries(name, tuple(samples), _merge_metadata({"source": "metric_series"}, metadata))


def build_metric_registry(*, samples: tuple[MetricSample, ...] | list[MetricSample] = (), policy: MetricRegistryPolicy | None = None) -> MetricRegistry:
    return MetricRegistry(samples=samples, policy=policy)


def _copy_metadata(value: dict[str, object]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(value)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
        if key in _FORBIDDEN_STATE or key in _FORBIDDEN_SECRET:
            raise ValueError("metadata contains forbidden key")
    return copied


def _merge_metadata(defaults: dict[str, object], metadata: dict[str, object] | None) -> dict[str, object]:
    merged = _copy_metadata(defaults)
    if metadata is not None:
        merged.update(_copy_metadata(metadata))
    if "source" in defaults:
        merged["source"] = defaults["source"]
    return merged


def _reject_name(value: str, field_name: str) -> None:
    if value in _FORBIDDEN_STATE:
        raise ValueError(f"{field_name} must not persist runtime semantic state")


def _copy_tags(value: dict[str, str]) -> dict[str, str]:
    if not isinstance(value, dict):
        raise TypeError("tags must be dict[str, str]")
    copied = dict(value)
    for key, item in copied.items():
        key = ensure_non_empty_string(key, "tag key")
        item = ensure_non_empty_string(item, "tag value")
        _reject_name(key, "tag key")
        _reject_name(item, "tag value")
    return copied


def _sample_tuple(value: tuple[MetricSample, ...] | list[MetricSample]) -> tuple[MetricSample, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("samples must be tuple[MetricSample, ...]")
    samples = tuple(value)
    ids = [sample.sample_id for sample in samples if isinstance(sample, MetricSample)]
    if len(ids) != len(set(ids)):
        raise ValueError("sample_id values must be unique")
    for sample in samples:
        if not isinstance(sample, MetricSample):
            raise TypeError("every sample must be MetricSample")
    return samples


def _series_tuple(value: tuple[MetricSeries, ...] | list[MetricSeries]) -> tuple[MetricSeries, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("series must be tuple[MetricSeries, ...]")
    series = tuple(value)
    for item in series:
        if not isinstance(item, MetricSeries):
            raise TypeError("every series must be MetricSeries")
    return series


def _string_tuple(value: tuple[str, ...] | list[str], field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{field_name} must be tuple[str, ...]")
    return tuple(ensure_non_empty_string(item, field_name) for item in value)


def _reject_sample(sample: MetricSample, samples: tuple[MetricSample, ...], policy: MetricRegistryPolicy) -> str | None:
    if len(samples) >= policy.max_samples:
        return "metric sample limit exceeded"
    if any(existing.sample_id == sample.sample_id for existing in samples):
        return "metric sample_id duplicated"
    if policy.drop_unknown_metrics and sample.kind is MetricKind.UNKNOWN:
        return "unknown metric dropped"
    if policy.require_non_negative_counters and sample.kind is MetricKind.COUNTER and sample.value < 0.0:
        return "counter metric must be non-negative"
    return None


def _result(*, samples: tuple[MetricSample, ...] = (), series: tuple[MetricSeries, ...] = (), changed: bool = False, reason: str, rejected: tuple[str, ...] = (), rejection_reasons: tuple[str, ...] = ()) -> MetricRegistryResult:
    return MetricRegistryResult(samples, series, changed, reason, rejected, rejection_reasons, {"source": "metric_registry_result"})
