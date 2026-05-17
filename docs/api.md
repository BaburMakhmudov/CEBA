# CEBA API

## API Role

The API is the external interface boundary. It translates external requests into validated transport objects and delegates to approved CEBA public boundaries.

The API must not contain domain logic that belongs to core, runtime, memory, or persistence. It must not compute Delta, world model prediction, memory consolidation, or persistence writes directly.

## API Structure

Approved API structure:

```text
api/server.py
api/dependencies.py
api/errors.py
api/schemas.py
api/routes/
api/middleware/
```

- `server.py` defines server composition.
- `dependencies.py` wires approved dependencies.
- `errors.py` maps internal errors to safe transport errors.
- `schemas.py` defines transport schemas.
- `routes/` contains endpoint handlers.
- `middleware/` contains request-level boundary behavior.

## Routes

| Route File | Purpose | Allowed Responsibility | Forbidden Responsibility |
|---|---|---|---|
| `health.py` | Report API or local component health | Return safe health descriptors | Probe databases or external services unless an explicit adapter exists |
| `observations.py` | Accept observation submissions | Validate transport payloads and call approved input/runtime boundary | Infer meaning, telos, valuation, or intent |
| `runtime.py` | Coordinate runtime cycle requests | Invoke runtime public boundary | Compute Delta or world model prediction directly |
| `memory.py` | Access memory-facing operations | Call approved memory APIs | Treat vector DB or graph DB as memory |
| `telos.py` | Expose telos descriptors or configuration | Validate and delegate to telos boundary | Treat telos as plain string list |
| `delta.py` | Expose Delta operations | Call approved Delta boundary with prepared inputs | Compute Delta before expected observation |
| `constraints.py` | Expose constraint operations | Validate constraint transport data | Treat constraints as persistence or memory |
| `audit.py` | Expose audit records or export preparation | Return safe audit descriptors | Enforce invariants or write files |

## Middleware

| Middleware | Purpose | Boundary |
|---|---|---|
| `request_id.py` | Attach safe request identifiers | Must not encode secrets or semantic state |
| `auth.py` | Delegate authentication and authorization checks | Must delegate to security layer |
| `rate_limit.py` | Apply request limiting policy | Must not perform domain logic |
| `error_handler.py` | Convert exceptions to safe API responses | Must not expose secrets or internals |

## API Schemas

API schemas are transport schemas, not CEBA core schemas. They validate external request and response shape.

Core schemas remain in their own packages. API schemas may reference or convert to core/public descriptors, but they must not replace domain schemas or hide domain logic.

## Dependency Boundary

Dependencies wire components. They may provide configured public interfaces to routes, but they must not hide domain logic inside dependency constructors or closures.

Dependency setup should make coupling visible and testable.

## Error Boundary

API errors must wrap and expose safe errors only. Responses must not leak secrets, tokens, raw credentials, stack traces, or derived runtime semantic state.

Internal validation failures can be mapped to structured transport errors. Unexpected errors should be reported safely.

## Security Boundary

API responses must not include raw secrets. Auth middleware must delegate to the security layer.

No external auth calls should be embedded directly in middleware unless a future explicit adapter is added. Local security descriptors and validation reports remain separate from external identity-provider integration.

## Runtime Boundary

The API may call the runtime public boundary. It must not directly compute world model prediction, Delta, memory consolidation, or persistence writes outside an approved service flow.

Correct API behavior is request validation, delegation, safe response shaping, and error handling.

## Documentation of Non-Implemented Behavior

This file documents architectural API contracts. It does not guarantee that every route is deployed or implemented unless corresponding route implementations exist in the repository.

The contract is still binding for future route implementations.

## Summary

The API is a transport boundary. It validates external input, delegates to approved CEBA layers, and returns safe output. It must not become runtime, memory, Delta, world model, persistence, audit enforcement, or security provider logic.
