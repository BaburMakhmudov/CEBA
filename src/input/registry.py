"""CEBA Input adapter registry.

The registry stores technical capture adapters and resolves one adapter for a
signal. It does not compute meaning, intent, or semantic interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ois.input.adapters.api import create_api_adapter
from ois.input.adapters.base import InputAdapter, validate_adapter_identity
from ois.input.adapters.database import create_database_adapter
from ois.input.adapters.file import create_file_adapter
from ois.input.adapters.log import create_log_adapter
from ois.input.adapters.shell import create_shell_adapter
from ois.input.adapters.system import create_system_adapter
from ois.input.adapters.user import create_user_adapter
from ois.input.adapters.webhook import create_webhook_adapter
from ois.input.errors import AdapterResolutionError, SchemaViolationError, UnsupportedInputError
from ois.input.schema import InputSourceType


@dataclass(frozen=True, slots=True)
class AdapterInfo:
    name: str
    version: str
    source_type: InputSourceType
    supports_inline_payload: bool
    supports_payload_ref: bool
    supports_streaming: bool
    supports_binary: bool


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, InputAdapter] = {}

    def register(self, adapter: InputAdapter) -> None:
        if not isinstance(adapter, InputAdapter):
            raise SchemaViolationError(
                "adapter must satisfy InputAdapter protocol",
                field_path="adapter",
                details={"type": type(adapter).__name__},
            )
        name, _version, source_type = validate_adapter_identity(
            name=adapter.name,
            version=adapter.version,
            source_type=adapter.source_type,
        )
        if not hasattr(adapter, "capability"):
            raise SchemaViolationError(
                "adapter must define capability",
                field_path="adapter.capability",
                details={"adapter": name},
            )

        capability = adapter.capability

        if not hasattr(capability, "source_type"):
            raise SchemaViolationError(
                "adapter capability must define source_type",
                field_path="adapter.capability.source_type",
                details={"adapter": name},
            )

        if capability.source_type != source_type:
            raise SchemaViolationError(
                "adapter capability source_type must match adapter source_type",
                field_path="adapter.capability.source_type",
                details={
                    "adapter_source_type": source_type.value,
                    "capability_source_type": capability.source_type.value,
                },
            )
        key = _adapter_key(source_type=source_type, name=name)
        if key in self._adapters:
            raise AdapterResolutionError(
                "adapter is already registered",
                field_path="adapter",
                details={"key": key, "name": name, "source_type": source_type.value},
            )
        self._adapters[key] = adapter

    def unregister(self, *, name: str, source_type: InputSourceType | str) -> None:
        normalized_name = _normalize_adapter_name(name)
        _unused_name, _unused_version, normalized_source_type = validate_adapter_identity(
            name=normalized_name,
            version="registry",
            source_type=source_type,
        )
        key = _adapter_key(source_type=normalized_source_type, name=normalized_name)
        if key not in self._adapters:
            raise AdapterResolutionError(
                "adapter is not registered",
                field_path="adapter",
                details={
                    "key": key,
                    "name": normalized_name,
                    "source_type": normalized_source_type.value,
                },
            )
        del self._adapters[key]

    def get(self, *, name: str, source_type: InputSourceType | str) -> InputAdapter:
        normalized_name = _normalize_adapter_name(name)
        _unused_name, _unused_version, normalized_source_type = validate_adapter_identity(
            name=normalized_name,
            version="registry",
            source_type=source_type,
        )
        key = _adapter_key(source_type=normalized_source_type, name=normalized_name)
        try:
            return self._adapters[key]
        except KeyError as error:
            raise AdapterResolutionError(
                "adapter is not registered",
                field_path="adapter",
                details={
                    "key": key,
                    "name": normalized_name,
                    "source_type": normalized_source_type.value,
                },
            ) from error

    def list_adapters(self) -> tuple[AdapterInfo, ...]:
        return tuple(
            sorted(
                (_adapter_info(adapter) for adapter in self._adapters.values()),
                key=lambda info: (info.source_type.value, info.name),
            )
        )

    def resolve(self, signal: Any) -> InputAdapter:
        matches: list[InputAdapter] = []
        for adapter in self._adapters.values():
            try:
                if adapter.supports(signal):
                    matches.append(adapter)
            except Exception:
                continue

        if not matches:
            raise UnsupportedInputError(
                "no registered adapter supports signal",
                field_path="signal",
                details={"registered_adapters": len(self._adapters)},
            )
        if len(matches) > 1:
            raise AdapterResolutionError(
                "multiple registered adapters support signal",
                field_path="signal",
                details={
                    "matches": [
                        f"{adapter.source_type.value}:{adapter.name}"
                        for adapter in matches
                    ]
                },
            )
        return matches[0]

    def has_adapter(self, *, name: str, source_type: InputSourceType | str) -> bool:
        normalized_name = _normalize_adapter_name(name)
        _unused_name, _unused_version, normalized_source_type = validate_adapter_identity(
            name=normalized_name,
            version="registry",
            source_type=source_type,
        )
        key = _adapter_key(source_type=normalized_source_type, name=normalized_name)
        return key in self._adapters

    def clear(self) -> None:
        self._adapters.clear()


def create_default_registry() -> AdapterRegistry:
    registry = AdapterRegistry()
    registry.register(create_user_adapter())
    registry.register(create_api_adapter())
    registry.register(create_file_adapter())
    registry.register(create_shell_adapter())
    registry.register(create_system_adapter())
    registry.register(create_webhook_adapter())
    registry.register(create_database_adapter())
    registry.register(create_log_adapter())
    return registry


def register_default_adapters(registry: AdapterRegistry) -> None:
    normalized_registry = _ensure_registry(registry)
    normalized_registry.register(create_user_adapter())
    normalized_registry.register(create_api_adapter())
    normalized_registry.register(create_file_adapter())
    normalized_registry.register(create_shell_adapter())
    normalized_registry.register(create_system_adapter())
    normalized_registry.register(create_webhook_adapter())
    normalized_registry.register(create_database_adapter())
    normalized_registry.register(create_log_adapter())


_DEFAULT_REGISTRY = create_default_registry()


def get_default_registry() -> AdapterRegistry:
    return _DEFAULT_REGISTRY


def resolve_adapter(signal: Any, registry: AdapterRegistry | None = None) -> InputAdapter:
    active_registry = _DEFAULT_REGISTRY if registry is None else _ensure_registry(registry)
    return active_registry.resolve(signal)


def _adapter_key(*, source_type: InputSourceType, name: str) -> str:
    return f"{source_type.value}:{name}"


def _normalize_adapter_name(name: str) -> str:
    if not isinstance(name, str):
        raise SchemaViolationError(
            "adapter name must be a string",
            field_path="name",
            details={"type": type(name).__name__},
        )
    normalized = name.strip()
    if not normalized:
        raise SchemaViolationError(
            "adapter name must not be empty",
            field_path="name",
        )
    return normalized


def _ensure_registry(value: AdapterRegistry) -> AdapterRegistry:
    if not isinstance(value, AdapterRegistry):
        raise SchemaViolationError(
            "registry must be AdapterRegistry",
            field_path="registry",
            details={"type": type(value).__name__},
        )
    return value


def _adapter_info(adapter: InputAdapter) -> AdapterInfo:
    return AdapterInfo(
        name=adapter.name,
        version=adapter.version,
        source_type=adapter.source_type,
        supports_inline_payload=adapter.capability.supports_inline_payload,
        supports_payload_ref=adapter.capability.supports_payload_ref,
        supports_streaming=adapter.capability.supports_streaming,
        supports_binary=adapter.capability.supports_binary,
    )


__all__ = [
    "AdapterInfo",
    "AdapterRegistry",
    "create_default_registry",
    "get_default_registry",
    "register_default_adapters",
    "resolve_adapter",
]
