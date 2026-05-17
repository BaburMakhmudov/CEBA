"""CEBA Input dispatcher utilities.

The dispatcher routes canonical observations to technical handlers by
observation type. It does not compute meaning, intent, or semantic
interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable
from uuid import UUID

from ois.input.errors import DispatchError, SchemaViolationError
from ois.input.schema import CanonicalObservation, ObservationType
from ois.input.validation import require_valid_canonical_observation


class DispatchStatus(str, Enum):
    PENDING = "pending"
    DISPATCHED = "dispatched"
    FAILED = "failed"
    SKIPPED = "skipped"


@runtime_checkable
class DispatchHandler(Protocol):
    name: str

    def handle(self, observation: CanonicalObservation) -> None:
        ...


@dataclass(frozen=True, slots=True)
class DispatchRoute:
    observation_type: ObservationType
    route_name: str
    handler: DispatchHandler
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class DispatchResult:
    status: DispatchStatus
    observation_id: UUID
    trace_id: UUID
    observation_type: ObservationType
    route_name: str | None = None
    handler_name: str | None = None
    error_code: str | None = None
    error_message: str | None = None


class InputDispatcher:
    def __init__(self) -> None:
        self._routes: dict[ObservationType, DispatchRoute] = {}

    def register_route(self, route: DispatchRoute) -> None:
        normalized_route = _ensure_route(route)
        observation_type = _ensure_observation_type(normalized_route.observation_type)
        route_name = _normalize_route_name(normalized_route.route_name)
        handler = _ensure_handler(normalized_route.handler)
        key = _route_key(observation_type)
        if key in self._routes:
            raise DispatchError(
                "dispatch route is already registered",
                field_path="route.observation_type",
                details={"observation_type": observation_type.value},
            )
        self._routes[key] = DispatchRoute(
            observation_type=observation_type,
            route_name=route_name,
            handler=handler,
            enabled=normalized_route.enabled,
        )

    def unregister_route(self, observation_type: ObservationType) -> None:
        normalized_observation_type = _ensure_observation_type(observation_type)
        key = _route_key(normalized_observation_type)
        if key not in self._routes:
            raise DispatchError(
                "dispatch route is not registered",
                field_path="observation_type",
                details={"observation_type": normalized_observation_type.value},
            )
        del self._routes[key]

    def get_route(self, observation_type: ObservationType) -> DispatchRoute:
        normalized_observation_type = _ensure_observation_type(observation_type)
        key = _route_key(normalized_observation_type)
        try:
            return self._routes[key]
        except KeyError as error:
            raise DispatchError(
                "dispatch route is not registered",
                field_path="observation_type",
                details={"observation_type": normalized_observation_type.value},
            ) from error

    def has_route(self, observation_type: ObservationType) -> bool:
        normalized_observation_type = _ensure_observation_type(observation_type)
        return _route_key(normalized_observation_type) in self._routes

    def list_routes(self) -> tuple[DispatchRoute, ...]:
        return tuple(
            self._routes[observation_type]
            for observation_type in sorted(
                self._routes,
                key=lambda value: value.value,
            )
        )

    def clear(self) -> None:
        self._routes.clear()

    def dispatch(self, observation: CanonicalObservation) -> DispatchResult:
        normalized_observation = _ensure_observation(observation)
        require_valid_canonical_observation(normalized_observation)
        route = self.get_route(normalized_observation.observation_type)
        if route.enabled is False:
            return DispatchResult(
                status=DispatchStatus.SKIPPED,
                observation_id=normalized_observation.observation_id,
                trace_id=normalized_observation.trace_id,
                observation_type=normalized_observation.observation_type,
                route_name=route.route_name,
                handler_name=route.handler.name,
            )

        try:
            route.handler.handle(normalized_observation)
        except DispatchError as error:
            return DispatchResult(
                status=DispatchStatus.FAILED,
                observation_id=normalized_observation.observation_id,
                trace_id=normalized_observation.trace_id,
                observation_type=normalized_observation.observation_type,
                route_name=route.route_name,
                handler_name=route.handler.name,
                error_code=error.code,
                error_message=error.payload.message,
            )
        except Exception as error:
            wrapped = DispatchError(
                "dispatch handler failed",
                trace_id=normalized_observation.trace_id,
                observation_id=normalized_observation.observation_id,
                field_path="handler.handle",
                details={
                    "handler": route.handler.name,
                    "original_type": error.__class__.__name__,
                },
            )
            return DispatchResult(
                status=DispatchStatus.FAILED,
                observation_id=normalized_observation.observation_id,
                trace_id=normalized_observation.trace_id,
                observation_type=normalized_observation.observation_type,
                route_name=route.route_name,
                handler_name=route.handler.name,
                error_code=wrapped.code,
                error_message=wrapped.payload.message,
            )

        return DispatchResult(
            status=DispatchStatus.DISPATCHED,
            observation_id=normalized_observation.observation_id,
            trace_id=normalized_observation.trace_id,
            observation_type=normalized_observation.observation_type,
            route_name=route.route_name,
            handler_name=route.handler.name,
        )


def dispatch_observation(
    *,
    observation: CanonicalObservation,
    dispatcher: InputDispatcher,
) -> DispatchResult:
    normalized_dispatcher = _ensure_dispatcher(dispatcher)
    return normalized_dispatcher.dispatch(observation)


def build_dispatch_route(
    *,
    observation_type: ObservationType,
    route_name: str,
    handler: DispatchHandler,
    enabled: bool = True,
) -> DispatchRoute:
    return DispatchRoute(
        observation_type=_ensure_observation_type(observation_type),
        route_name=_normalize_route_name(route_name),
        handler=_ensure_handler(handler),
        enabled=enabled,
    )


def _ensure_observation(value: CanonicalObservation) -> CanonicalObservation:
    if not isinstance(value, CanonicalObservation):
        raise SchemaViolationError(
            "observation must be CanonicalObservation",
            field_path="observation",
            details={"type": type(value).__name__},
        )
    return value


def _ensure_observation_type(value: ObservationType) -> ObservationType:
    if not isinstance(value, ObservationType):
        raise SchemaViolationError(
            "observation_type must be ObservationType",
            field_path="observation_type",
            details={"type": type(value).__name__},
        )
    return value


def _ensure_dispatcher(value: InputDispatcher) -> InputDispatcher:
    if not isinstance(value, InputDispatcher):
        raise SchemaViolationError(
            "dispatcher must be InputDispatcher",
            field_path="dispatcher",
            details={"type": type(value).__name__},
        )
    return value


def _ensure_route(value: DispatchRoute) -> DispatchRoute:
    if not isinstance(value, DispatchRoute):
        raise SchemaViolationError(
            "route must be DispatchRoute",
            field_path="route",
            details={"type": type(value).__name__},
        )
    return value


def _ensure_handler(value: DispatchHandler) -> DispatchHandler:
    if not isinstance(value, DispatchHandler):
        raise SchemaViolationError(
            "handler must satisfy DispatchHandler protocol",
            field_path="handler",
            details={"type": type(value).__name__},
        )
    return value


def _normalize_route_name(value: str) -> str:
    if not isinstance(value, str):
        raise SchemaViolationError(
            "route_name must be a string",
            field_path="route_name",
            details={"type": type(value).__name__},
        )
    normalized = value.strip()
    if not normalized:
        raise SchemaViolationError(
            "route_name must not be empty",
            field_path="route_name",
        )
    return normalized


def _route_key(observation_type: ObservationType) -> ObservationType:
    return observation_type


__all__ = [
    "DispatchHandler",
    "DispatchResult",
    "DispatchRoute",
    "DispatchStatus",
    "InputDispatcher",
    "build_dispatch_route",
    "dispatch_observation",
]
