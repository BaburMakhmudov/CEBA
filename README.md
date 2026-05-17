# CEBA

## What CEBA Is

CEBA means Constrained Energy-Based Agent. CEBA is an architecture for domain-independent understanding update: observations are compared against expectations, differences are localized under constraints, and semantic state transitions are regulated through explicit boundaries.

This repository is an engineering implementation scaffold and specification. It is not a claim that completed AGI already exists.

## Core Principle

The core update cycle is:

```text
observation
→ expected observation
→ difference
→ constraint localization
→ metric/cost update
→ dynamics/regulation
→ telos/valuation
→ relation update
→ memory coordination
→ next prediction
```

The cycle updates understanding by handling differences under constraints. Runtime coordination may prepare and order work, but it must not bypass world model, Delta, memory, or persistence boundaries.

## Ontological Chain

The core ontological dependency chain is:

```text
difference
→ constraint
→ metric
→ dynamics
→ regulation
→ telos
→ valuation
→ relation
→ meaning-state stabilization
```

Ontological dependency order is not always the same as runtime execution order. Execution may be coordinated differently for operational reasons, but invariant dependencies must remain intact.

## Repository Structure

- `configs/`: static declarative configuration.
- `src/ceba/app/`: application composition and entrypoints.
- `src/ceba/api/`: external interface boundary.
- `src/ceba/input/`: raw observation input boundary.
- `src/ceba/parser/`: structured parsing boundary.
- `src/ceba/embedding/`: embedding and projection boundary.
- `src/ceba/difference/`: difference descriptors and mismatch inputs.
- `src/ceba/core/`: state, Delta, constraints, metrics, dynamics, regulation, telos, valuation, and relation layers.
- `src/ceba/memory/`: short-term memory, long-term memory, and retrieval coordination.
- `src/ceba/world_model/`: expectation and prediction boundary.
- `src/ceba/runtime/`: deterministic runtime coordination.
- `src/ceba/persistence/`: ledger, PostgreSQL, vector, and graph persistence surfaces.
- `src/ceba/audit/`: audit descriptors and local audit logs.
- `src/ceba/security/`: security descriptors, policies, and validation.
- `src/ceba/observability/`: logging, metrics, tracing, health, and diagnostics descriptors.
- `src/ceba/utils/`: dependency-free utility helpers.
- `scripts/`: operational descriptor and planning entrypoints.
- `docs/`: technical documentation and architecture contracts.

## Memory and Persistence Contract

STM holds active difference until resolution.
τ_t regulates strength of state transition.
LTM stores verified experience after consolidation.
Persistence stores ledger/repositories/index surfaces.
Vector DB is retrieval surface, not memory.
Graph DB is retrieval/index surface, not memory.
Canonical Ledger is persistence source-of-truth.
M_t is derived runtime semantic state and must not be persisted.

## Architectural Invariants

- `WORLD_MODEL_BEFORE_DELTA`
- `DELTA_ONLY_WRITER`
- `M_T_NOT_PERSISTED`
- `VECTOR_DB_NOT_MEMORY`
- `STM_NOT_DATABASE`
- `TEMPERATURE_NOT_MEMORY`

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Tooling

```bash
ruff check .
mypy src
pytest
```

`pytest` is configured for future verification. Test modules may be added later if the `tests/` folder is not currently used.

## Configuration

The `configs/` directory contains static YAML configuration for CEBA application identity, logging, database descriptors, embedding settings, vector index settings, graph store settings, runtime coordination, constraints, telos, and security.

Configuration is declarative infrastructure and policy data. It must not contain raw secrets, executable logic, `M_t` persistence, semantic-state storage targets, or claims that retrieval surfaces are memory.

## Scripts

The `scripts/` directory contains operational entrypoints that prepare descriptors and plans. They do not execute real database, vector, or graph operations. They emit JSON-safe planning or reporting output and preserve CEBA layer boundaries.

## Non-Goals

- No demo mode.
- No placeholder production logic.
- No raw secret storage.
- No vector DB as memory.
- No graph DB as memory.
- No `M_t` persistence.
- No unsupported AGI completion claims.

## License

See `LICENSE`.
