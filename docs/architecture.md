# CEBA Architecture

## Overview

CEBA means Constrained Energy-Based Agent. In this repository it is an architecture for domain-independent understanding update: observations are compared against expectations, differences are localized under constraints, and semantic state is updated through controlled dynamics.

This documentation does not claim that the repository is a completed AGI. It describes the intended engineering decomposition, boundaries, and invariants that keep CEBA components from collapsing into each other.

## Architectural Principle

The central CEBA cycle is:

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

The cycle updates understanding by handling differences under constraints. The expected observation is required before a difference can be computed. Constraints determine what the difference violates or satisfies. Metrics assign cost, distance, penalty, or relevance. Dynamics and regulation determine how strongly the system transitions. Telos and valuation orient and score the update. Relation updates structural connectivity. Memory coordination determines what remains active, what is consolidated, and what can be retrieved later.

This is not a simple linear pipe. Runtime execution may coordinate objects in operational order, but the ontology preserves the dependency chain:

```text
difference → constraint → metric → dynamics → regulation → telos → valuation → relation → meaning-state stabilization
```

## Approved Repository Structure

```text
src/ceba/
  app/
  api/
  input/
  parser/
  embedding/
  difference/
  core/state/
  core/delta/
  core/constraints/
  core/metrics/
  core/dynamics/
  core/regulation/
  core/telos/
  core/valuation/
  core/relation/
  memory/short_term/
  memory/long_term/
  memory/retrieval/
  world_model/
  runtime/
  persistence/ledger/
  persistence/postgres/
  persistence/vector/
  persistence/graph/
  audit/
  security/
  observability/
  utils/
scripts/
docs/
```

## Layer Responsibilities

| Layer | Responsibility | Must Not Do |
|---|---|---|
| app | Application composition and entrypoints | Hide domain logic or bypass runtime boundaries |
| api | External transport boundary | Compute Delta, memory consolidation, prediction, or persistence writes directly |
| input | Accept and validate raw observations | Infer meaning, telos, valuation, or intent |
| parser | Produce structured representations | Understand telos or finalize meaning |
| embedding | Produce representational vectors or references | Act as memory or source of truth |
| difference | Describe mismatch inputs and outputs | Write memory or persistence |
| core/state | Define local state descriptors | Persist derived runtime semantic state |
| core/delta | Compute structured mismatch under preconditions | Predict world model expectations or call retrieval stores |
| core/constraints | Represent rules and boundaries | Act as persistence or memory |
| core/metrics | Define costs, distances, penalties, and relevance | Replace valuation or telos |
| core/dynamics | Model state transition behavior | Store memory or persistence records |
| core/regulation | Control precision, update strength, consolidation, replay, and forgetting policy | Treat semantic temperature as memory |
| core/telos | Represent attractors, targets, and normative orientation | Reduce telos to a plain string list |
| core/valuation | Evaluate utility, free energy, entropy, or scores | Become relation runtime or graph persistence |
| core/relation | Manage semantic connectivity | Become the graph database |
| memory | Retain, consolidate, and retrieve experience through memory layers | Treat STM as a database or LTM as a vector DB |
| world_model | Produce expected observations and confidence-bearing predictions | Compute Delta or act as runtime scheduler |
| runtime | Coordinate prepared objects, invariants, transactions, cycles, and checkpoints | Compute domain logic directly |
| persistence | Store durable records, repositories, and retrieval/index surfaces | Persist `M_t` or act as memory itself |
| audit | Record evidence about events and decisions | Enforce invariants or execute runtime logic |
| security | Describe and validate boundary control | Call external providers inside pure local descriptors |
| observability | Describe logs, metrics, traces, health, and diagnostics | Send telemetry by itself |
| utils | Provide low-level deterministic helpers | Encode CEBA domain behavior |

## Runtime Boundary

Runtime coordinates prepared CEBA objects and invariant checks. It opens local coordination transactions, schedules tasks, creates checkpoints, and reports completion or failure.

Runtime must not compute world model prediction directly, compute Delta directly, mutate memory directly, or write persistence directly. Those operations belong to their own public boundaries and must be invoked through explicit architectural flows.

## Persistence Boundary

Persistence stores ledger records, repository records, and index surfaces.

- `persistence/ledger/` defines Canonical Ledger records and repository boundaries.
- `persistence/postgres/` defines PostgreSQL configuration, schema, JSONB, and transaction descriptors.
- `persistence/vector/` defines vector retrieval/index schemas and local adapter descriptors.
- `persistence/graph/` defines graph retrieval/index schemas and local adapter descriptors.

Persistence is not memory. PostgreSQL stores persistence records. Vector and graph stores are retrieval/index surfaces. `M_t` is derived runtime semantic state and must not be persisted.

## Memory Boundary

STM holds active difference until resolution. LTM stores verified experience after consolidation. Retrieval surfaces supply evidence and candidates, but retrieval surfaces are not memory themselves.

Vector DB is retrieval surface, not memory. Graph DB is retrieval/index surface, not memory. Canonical Ledger is persistence source-of-truth, not memory. Memory layers coordinate retention, consolidation, replay, and retrieval over these surfaces without collapsing into them.

## Engineering Non-Goals

- No demo mode.
- No placeholder logic.
- No hidden side effects.
- No database calls in schema/config descriptors.
- No `M_t` persistence.
- No external services inside pure schema or policy layers.
- No vector DB or graph DB treated as memory.
- No parser-level inference of meaning, telos, valuation, or intent.
- No runtime shortcuts around world model and Delta boundaries.

## Summary

CEBA is decomposed to keep observation, expectation, difference, constraints, metrics, dynamics, regulation, telos, valuation, relation, memory, runtime, and persistence distinct. The architecture is useful only if these boundaries remain explicit: the runtime coordinates, the world model predicts, Delta compares, memory consolidates, persistence records, retrieval surfaces retrieve, and `M_t` remains derived rather than stored.
