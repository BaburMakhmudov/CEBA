# CEBA Memory Model

## Memory Role

Memory is structured retention and consolidation of experience, not raw storage. CEBA memory layers decide what remains active, what becomes eligible for consolidation, what is verified, and what can be retrieved as evidence.

Memory must not be reduced to a vector store, graph store, database table, or ledger record.

## Core Contract

STM holds active difference until resolution.
τ_t regulates strength of state transition.
LTM stores verified experience after consolidation.
Persistence stores ledger/repositories/index surfaces.
Vector DB is retrieval surface, not memory.
Graph DB is retrieval/index surface, not memory.
Canonical Ledger is persistence source-of-truth.
M_t is derived runtime semantic state and must not be persisted.

## Short-Term Memory

Short-term memory is the runtime working layer for unresolved semantic work. It may hold active context, candidate traces, hypothesis sets, active differences, and local resolution status.

STM supports eviction and validation. It is not a database. It is not vector DB. It is not graph DB. It must not be treated as durable persistence.

## Long-Term Memory

Long-term memory stores verified experience after consolidation. LTM is reached through consolidation, trace building, replay, provenance checks, eligibility checks, and validation.

LTM is not raw vector storage. Vector representations may support retrieval, but the memory layer is responsible for validated experience and consolidation semantics.

## Retrieval

Retrieval includes query construction, ranking, evidence packs, vector retrieval, graph retrieval, and hybrid retrieval. Retrieval supplies candidates and evidence to other layers.

Retrieval is an access surface, not memory itself. A retrieval evidence pack can reference persistent and indexed records, but it is not equivalent to STM or LTM.

## Persistence vs Memory

| Component | What it stores | Whether it is memory | Why |
|---|---|---|---|
| STM | Active differences, working context, candidate traces | Yes, short-term | It holds unresolved active semantic work |
| LTM | Verified experience after consolidation | Yes, long-term | It retains consolidated experience |
| Canonical Ledger | Durable source-of-truth records and provenance | No | It is persistence, not memory semantics |
| PostgreSQL | Persistence records, schemas, JSONB payloads, descriptors | No | It stores records and infrastructure state |
| Vector DB | Vectors and retrieval metadata | No | It is a retrieval surface |
| Graph DB | Concept, relation, and activation index surfaces | No | It is retrieval/index infrastructure |
| Retrieval evidence pack | Ranked evidence and references | No | It is an access result over memory/persistence surfaces |

## Semantic Temperature

Semantic temperature is not memory. `τ_t` belongs to dynamics and regulation. Relevant boundaries include:

- `core/dynamics/semantic_temperature.py`
- `core/regulation/precision_controller.py`
- `core/regulation/update_policy.py`

`τ_t` regulates transition strength and plasticity. It may affect how strongly active differences update state, but it is not a remembered experience and not a persistence record.

## M_t

`M_t` is runtime-derived semantic state. It is reconstructed or derived during runtime from current context, model state, constraints, memory, and retrieval evidence.

`M_t` must not be persisted. Persistence may store records, provenance, repository entries, and index surfaces, but not the derived runtime semantic state itself.

## Common Mistakes

- “Vector DB is memory”: incorrect. Vector DB is retrieval surface.
- “STM is database”: incorrect. STM is active working memory.
- “Semantic temperature is memory”: incorrect. It belongs to dynamics and regulation.
- “Ledger is LTM”: incorrect. Ledger is persistence source-of-truth.
- “M_t should be stored”: incorrect. `M_t` is runtime-derived and must not be persisted.

## Summary

CEBA memory is a semantic retention and consolidation system. STM holds active unresolved difference; LTM stores verified consolidated experience; retrieval surfaces provide access; persistence stores durable records; `τ_t` regulates updates; `M_t` remains derived runtime state.
