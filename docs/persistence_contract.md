# CEBA Persistence Contract

## Persistence Role

Persistence stores durable records and index surfaces. Persistence is not memory itself.

The persistence layer provides ledger records, PostgreSQL descriptors, vector retrieval/index surfaces, graph retrieval/index surfaces, and local repository boundaries. It must not persist derived runtime semantic state.

## Core Contract

STM holds active difference until resolution.
τ_t regulates strength of state transition.
LTM stores verified experience after consolidation.
Persistence stores ledger/repositories/index surfaces.
Vector DB is retrieval surface, not memory.
Graph DB is retrieval/index surface, not memory.
Canonical Ledger is persistence source-of-truth.
M_t is derived runtime semantic state and must not be persisted.

## Canonical Ledger

The Canonical Ledger is the persistence source of truth. It stores durable records, provenance, payload envelopes, snapshots, and local repository boundaries.

The ledger is not memory. It may contain records used by memory and retrieval flows, but it does not itself perform STM or LTM behavior. It must not store `M_t`.

## PostgreSQL

PostgreSQL is persistence infrastructure. The `persistence/postgres/` layer defines connection descriptors, schema models, JSONB utilities, and transaction descriptors.

PostgreSQL must not contain a raw semantic state table or `M_t` table. PostgreSQL can store persistence records and JSONB payloads that pass validation, but it must reject forbidden runtime semantic-state fields.

## Vector Persistence

Vector persistence covers Qdrant-style, pgvector-style, Lance-style, and in-memory vector index descriptors. It includes vector index policy, vector payloads, search queries, search matches, and snapshots.

Vector DB is retrieval surface only. It may store vectors and retrieval metadata. It is not memory, not LTM, and not source of truth.

## Graph Persistence

Graph persistence covers Neo4j-style adapter descriptors, concept repository utilities, relation repository utilities, and activation repository utilities.

Graph DB is retrieval/index surface only. It is not relation runtime and not memory. Relation computation belongs to `core/relation/`; graph persistence stores indexable graph records and retrieval metadata.

## M_T_NOT_PERSISTED

`M_T_NOT_PERSISTED` forbids persistence of derived runtime semantic state.

Forbidden runtime semantic-state names include:

- `M_t`
- `Mt`
- `semantic_state`
- `runtime_semantic_state`

These names are forbidden as payload keys, metadata keys, table names, descriptor names, and persisted record fields where relevant. Runtime semantic state must be derived during runtime, not stored as a durable object.

## Separation Table

| Surface | Source of truth? | Memory? | Stores `M_t`? | Purpose |
|---|---:|---:|---:|---|
| Canonical Ledger | Yes | No | No | Durable persistence records and provenance |
| PostgreSQL | Infrastructure for persistence | No | No | Stores records, schemas, JSONB payloads, transaction descriptors |
| Vector DB | No | No | No | Vector retrieval/index surface |
| Graph DB | No | No | No | Graph retrieval/index surface |
| STM | No | Yes | No | Active difference until resolution |
| LTM | No | Yes | No | Verified experience after consolidation |
| Runtime `M_t` | No | Derived state | No | Reconstructed runtime semantic state |

## Script Boundaries

Operational scripts prepare descriptors and plans. They do not perform real external operations in the current architecture.

Scripts must not execute SQL, connect to vector stores, connect to graph stores, write audit exports, or persist `M_t`. They emit JSON-safe planning or reporting outputs.

## Summary

Persistence records durable evidence and indexable surfaces. Memory remains a semantic retention layer. Retrieval surfaces retrieve. The Canonical Ledger is source of truth. PostgreSQL is storage infrastructure. Vector and graph stores are not memory. `M_t` is runtime-derived and must not be persisted.
