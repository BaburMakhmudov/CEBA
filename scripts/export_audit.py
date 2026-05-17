import argparse
import json
import sys
from pathlib import Path

from ceba.audit import (
    AuditActorKind,
    AuditEventKind,
    AuditExportFormat,
    AuditSeverity,
    AuditStatus,
    build_audit_exporter,
    create_audit_actor,
    create_audit_event,
)

_FORBIDDEN_STATE_NAMES: frozenset[str] = frozenset(
    {"M_t", "Mt", "semantic_state", "runtime_semantic_state"}
)
_FORBIDDEN_SECRET_KEYS: frozenset[str] = frozenset(
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


def _write_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def _safe_error_payload(message: str) -> dict[str, object]:
    return {"script": "export_audit", "status": "error", "error": str(message)}


def _reject_forbidden_value(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    checked = value.strip()
    if not checked:
        raise ValueError(f"{field_name} must be non-empty")
    if checked in _FORBIDDEN_STATE_NAMES:
        raise ValueError(f"{field_name} must not be a forbidden runtime semantic-state name")
    return checked


def _reject_forbidden_keys(value: object, field_name: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field_name} keys must be strings")
            if key in _FORBIDDEN_STATE_NAMES or key in _FORBIDDEN_SECRET_KEYS:
                raise ValueError(f"{field_name} contains forbidden key")
            _reject_forbidden_keys(item, f"{field_name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_keys(item, f"{field_name}[{index}]")


def _parse_metadata_json(value: str | None) -> dict[str, object]:
    if value is None:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("metadata-json must be a JSON object")
    _reject_forbidden_keys(parsed, "metadata")
    return dict(parsed)


def _event_from_object(value: dict[str, object], metadata: dict[str, object]) -> object:
    _reject_forbidden_keys(value, "event")
    message = _reject_forbidden_value(str(value.get("message", "")), "message")
    severity = AuditSeverity[_reject_forbidden_value(str(value.get("severity", "")), "severity").upper()]
    actor_id = _reject_forbidden_value(str(value.get("actor_id", "")), "actor_id")
    actor_kind = AuditActorKind[_reject_forbidden_value(str(value.get("actor_kind", "")), "actor_kind").upper()]
    actor = create_audit_actor(actor_id=actor_id, kind=actor_kind, metadata=metadata)
    return create_audit_event(
        message=message,
        actor=actor,
        kind=AuditEventKind.EXPORT,
        severity=severity,
        status=AuditStatus.RECORDED,
        metadata=metadata,
    )


def _load_event_objects(event_json_values: list[str], events_file: str | None) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for value in event_json_values:
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("event-json must be a JSON object")
        objects.append(parsed)
    if events_file is not None:
        parsed_file = json.loads(Path(events_file).read_text(encoding="utf-8"))
        if not isinstance(parsed_file, list):
            raise ValueError("events-file must contain a JSON list")
        for item in parsed_file:
            if not isinstance(item, dict):
                raise ValueError("events-file entries must be JSON objects")
            objects.append(item)
    return objects


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare deterministic audit export descriptors.")
    parser.add_argument("--format", default="DICT")
    parser.add_argument("--label")
    parser.add_argument("--event-json", action="append", default=[])
    parser.add_argument("--events-file")
    parser.add_argument("--metadata-json")
    try:
        args = parser.parse_args(argv)
        metadata = _parse_metadata_json(args.metadata_json)
        event_objects = _load_event_objects(args.event_json, args.events_file)
        if not event_objects:
            event_objects = [
                {
                    "message": "audit export prepared",
                    "severity": "INFO",
                    "actor_id": "system",
                    "actor_kind": "SYSTEM",
                }
            ]
        events = tuple(_event_from_object(item, metadata) for item in event_objects)
        exporter = build_audit_exporter()
        result = exporter.prepare_export(
            events=events,
            export_format=AuditExportFormat[_reject_forbidden_value(args.format, "format").upper()],
            label=(
                None
                if args.label is None
                else _reject_forbidden_value(args.label, "label")
            ),
            metadata=metadata,
        )
        _write_json({"script": "export_audit", "status": "prepared", "export": result.to_dict()})
        return 0
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError) as exc:
        _write_json(_safe_error_payload(str(exc)))
        return 1
    except Exception as exc:
        _write_json(_safe_error_payload(f"unexpected error: {exc}"))
        return 2


if __name__ == "__main__":
    sys.exit(main())
