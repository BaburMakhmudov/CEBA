"""Deterministic audit export descriptors."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from ceba.audit.schema import (
    AuditBundle,
    AuditEvent,
    AuditStatus,
    create_audit_bundle,
    ensure_non_empty_string,
    ensure_optional_non_empty_string,
    ensure_positive_int,
    ensure_timezone_aware,
    utc_now,
)


__all__ = [
    "AuditExportFormat",
    "AuditExportStatus",
    "AuditExportPolicy",
    "AuditExportDescriptor",
    "AuditExportResult",
    "AuditExporter",
    "build_audit_exporter",
]


_FORBIDDEN = {"M_t", "Mt", "semantic_state", "runtime_semantic_state"}


class AuditExportFormat(Enum):
    JSON = "JSON"
    JSONL = "JSONL"
    DICT = "DICT"
    UNKNOWN = "UNKNOWN"


class AuditExportStatus(Enum):
    CREATED = "CREATED"
    PREPARED = "PREPARED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class AuditExportPolicy:
    max_events: int = 100000
    allow_empty_export: bool = False
    include_metadata: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_events", ensure_positive_int(self.max_events, "max_events"))
        if not isinstance(self.allow_empty_export, bool):
            raise TypeError("allow_empty_export must be bool")
        if not isinstance(self.include_metadata, bool):
            raise TypeError("include_metadata must be bool")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_events": self.max_events,
            "allow_empty_export": self.allow_empty_export,
            "include_metadata": self.include_metadata,
        }


@dataclass(frozen=True)
class AuditExportDescriptor:
    export_id: UUID
    export_format: AuditExportFormat
    status: AuditExportStatus
    bundle: AuditBundle
    prepared_at: datetime
    label: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.export_id, UUID):
            raise TypeError("export_id must be UUID")
        if not isinstance(self.export_format, AuditExportFormat):
            raise TypeError("export_format must be AuditExportFormat")
        if not isinstance(self.status, AuditExportStatus):
            raise TypeError("status must be AuditExportStatus")
        if not isinstance(self.bundle, AuditBundle):
            raise TypeError("bundle must be AuditBundle")
        object.__setattr__(self, "prepared_at", ensure_timezone_aware(self.prepared_at, "prepared_at"))
        label = ensure_optional_non_empty_string(self.label, "label")
        if label in _FORBIDDEN:
            raise ValueError("label must not persist runtime semantic state")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "export_id": str(self.export_id),
            "export_format": self.export_format.value,
            "status": self.status.value,
            "bundle": self.bundle.to_dict(),
            "prepared_at": self.prepared_at.isoformat(),
            "label": self.label,
            "metadata": dict(self.metadata),
        }

    def event_count(self) -> int:
        return self.bundle.event_count()


@dataclass(frozen=True)
class AuditExportResult:
    descriptor: AuditExportDescriptor | None
    changed: bool
    reason: str
    blocked: bool
    payload: dict[str, object] | tuple[dict[str, object], ...] | None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.descriptor is not None and not isinstance(self.descriptor, AuditExportDescriptor):
            raise TypeError("descriptor must be None or AuditExportDescriptor")
        if not isinstance(self.changed, bool):
            raise TypeError("changed must be bool")
        object.__setattr__(self, "reason", ensure_non_empty_string(self.reason, "reason"))
        if not isinstance(self.blocked, bool):
            raise TypeError("blocked must be bool")
        if self.payload is not None and not isinstance(self.payload, (dict, tuple)):
            raise TypeError("payload must be None, dict, or tuple")
        if isinstance(self.payload, tuple):
            for item in self.payload:
                if not isinstance(item, dict):
                    raise TypeError("payload tuple values must be dict")
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "descriptor": None if self.descriptor is None else self.descriptor.to_dict(),
            "changed": self.changed,
            "reason": self.reason,
            "blocked": self.blocked,
            "payload": self.payload,
            "metadata": dict(self.metadata),
        }

    def is_success(self) -> bool:
        return self.descriptor is not None and self.changed and not self.blocked and self.descriptor.status is AuditExportStatus.PREPARED


class AuditExporter:
    def __init__(self, *, policy: AuditExportPolicy | None = None) -> None:
        self._policy = AuditExportPolicy() if policy is None else policy
        if not isinstance(self._policy, AuditExportPolicy):
            raise TypeError("policy must be AuditExportPolicy")

    def prepare_export(
        self,
        *,
        events: tuple[AuditEvent, ...] | list[AuditEvent],
        export_format: AuditExportFormat = AuditExportFormat.DICT,
        label: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> AuditExportResult:
        events_tuple = _event_tuple(events)
        if not isinstance(export_format, AuditExportFormat):
            raise TypeError("export_format must be AuditExportFormat")
        label = ensure_optional_non_empty_string(label, "label")
        if metadata is not None:
            metadata = _copy_metadata(metadata)
        if not events_tuple and not self._policy.allow_empty_export:
            return AuditExportResult(
                descriptor=None,
                changed=False,
                reason="audit export empty",
                blocked=True,
                payload=None,
                metadata={"source": "audit_exporter_result"},
            )
        limited = events_tuple[: self._policy.max_events]
        bundle = create_audit_bundle(events=limited, status=AuditStatus.EXPORTED, metadata={"source": "audit_exporter_bundle"})
        descriptor = AuditExportDescriptor(
            export_id=uuid4(),
            export_format=export_format,
            status=AuditExportStatus.PREPARED,
            bundle=bundle,
            prepared_at=utc_now(),
            label=label,
            metadata=_merge_metadata({"source": "audit_export_descriptor", "event_count": len(limited)}, metadata),
        )
        payload = _payload_for(descriptor, export_format)
        return AuditExportResult(
            descriptor=descriptor,
            changed=True,
            reason="audit export prepared",
            blocked=False,
            payload=payload,
            metadata={"source": "audit_exporter_result"},
        )


def build_audit_exporter(*, policy: AuditExportPolicy | None = None) -> AuditExporter:
    return AuditExporter(policy=policy)


def _event_tuple(value: tuple[AuditEvent, ...] | list[AuditEvent]) -> tuple[AuditEvent, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("events must be tuple[AuditEvent, ...]")
    events = tuple(value)
    for event in events:
        if not isinstance(event, AuditEvent):
            raise TypeError("every event must be AuditEvent")
    ids = [event.event_id for event in events]
    if len(ids) != len(set(ids)):
        raise ValueError("event_id values must be unique")
    return events


def _payload_for(descriptor: AuditExportDescriptor, export_format: AuditExportFormat) -> dict[str, object] | tuple[dict[str, object], ...]:
    if export_format is AuditExportFormat.DICT:
        return descriptor.to_dict()
    if export_format is AuditExportFormat.JSON:
        return {"json_ready": True, "data": descriptor.to_dict()}
    if export_format is AuditExportFormat.JSONL:
        return tuple(event.to_dict() for event in descriptor.bundle.events)
    return {"export_format": AuditExportFormat.UNKNOWN.value, "data": descriptor.to_dict()}


def _copy_metadata(value: dict[str, object]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("metadata must be dict[str, object]")
    copied = dict(value)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
        if key in _FORBIDDEN:
            raise ValueError("metadata must not persist runtime semantic state")
    return copied


def _merge_metadata(defaults: dict[str, object], metadata: dict[str, object] | None) -> dict[str, object]:
    merged = _copy_metadata(defaults)
    if metadata is not None:
        merged.update(_copy_metadata(metadata))
    if "source" in defaults:
        merged["source"] = defaults["source"]
    return merged
