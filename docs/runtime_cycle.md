# CEBA Runtime Cycle

## Runtime Role

Runtime coordinates prepared CEBA objects. Runtime does not perform world model prediction directly. Runtime does not compute Delta directly. Runtime does not mutate memory directly. Runtime does not write persistence directly.

Runtime is the operational boundary that checks invariants, orders prepared work, opens local coordination transactions, creates checkpoints, and reports completion or failure.

## Runtime Modules

| Module | Role | Allowed Responsibilities | Forbidden Responsibilities |
|---|---|---|---|
| `context.py` | Local coordination state | Hold cycle identifiers, prepared references, metadata, and local status | Act as memory or persistence |
| `invariants.py` | Runtime invariant checks | Validate architectural ordering and forbidden couplings | Execute domain logic or persist audit records |
| `transaction.py` | Local runtime transaction boundary | Track local cycle status and accepted operations | Act as database transaction |
| `checkpoints.py` | Local safety checkpoints | Describe restart/safety boundaries for prepared cycles | Store durable memory or persistence snapshots |
| `scheduler.py` | Deterministic ordering layer | Build stable task order from prepared work | Execute tasks or compute domain results |
| `cycle.py` | One-pass cycle coordination | Coordinate context, scheduling, transaction, checkpoints, and outcome | Compute world model prediction or Delta |
| `engine.py` | Top-level local coordinator | Run prepared cycles through runtime boundaries | Bypass component boundaries |

## Prepared Cycle

A prepared runtime cycle follows this coordination pattern:

1. Context creation.
2. Invariant validation.
3. Schedule building.
4. Transaction opening.
5. Task coordination.
6. Transaction commit or failure.
7. Checkpoint creation.
8. Completion or failure reporting.

The cycle coordinates prepared objects. It must not create hidden world model predictions, hidden Delta computations, hidden memory writes, or hidden persistence writes.

## RuntimeContext

`RuntimeContext` is local coordination state. It may carry identifiers, status, metadata, and prepared references needed for a cycle.

It is not memory, not STM, not LTM, and not persistence. It should not be used to store `M_t` as a durable object.

## RuntimeScheduler

`RuntimeScheduler` is a deterministic ordering layer. It builds stable ordering for prepared tasks so execution can be inspected and repeated.

The scheduler is not execution. It must not call model prediction, Delta computation, database clients, vector stores, graph stores, or LLMs.

## RuntimeTransaction

`RuntimeTransaction` is a local coordination boundary for runtime state transitions. It can describe opened, committed, failed, or blocked runtime work.

It is not a PostgreSQL transaction and must not execute SQL. Database transaction descriptors belong under `persistence/postgres/`.

## RuntimeCheckpoint

`RuntimeCheckpoint` is a local safety boundary. It can describe the state of prepared coordination at a point in the runtime flow.

It is not a persistence snapshot, not a memory snapshot, and not a vector or graph index snapshot.

## RuntimeCycleCoordinator

`RuntimeCycleCoordinator` coordinates one pass through the prepared cycle. It aligns context, invariant validation, scheduling, transaction state, checkpoint creation, and reporting.

It must not perform domain computation directly. World model prediction and Delta computation remain behind their own boundaries.

## RuntimeEngine

`RuntimeEngine` is the top-level local coordinator for prepared cycles. It may accept prepared cycle descriptors and drive them through runtime coordination.

The engine should not become a service locator for hidden side effects. Its role is orchestration, not domain computation.

## Runtime Invariant Enforcement

Runtime must preserve these invariants:

- `WORLD_MODEL_BEFORE_DELTA`
- `DELTA_ONLY_WRITER`
- `M_T_NOT_PERSISTED`
- `VECTOR_DB_NOT_MEMORY`
- `STM_NOT_DATABASE`
- `TEMPERATURE_NOT_MEMORY`

These invariants prevent runtime from executing out of order or collapsing semantic, memory, persistence, and retrieval layers.

## Forbidden Runtime Couplings

Runtime must not import or directly execute:

- database clients
- vector DB clients
- graph DB clients
- LLM calls
- world model prediction internals
- delta computation internals as execution shortcuts

Runtime may coordinate public boundaries. It must not bypass them.

## Summary

The runtime layer is a deterministic coordinator for prepared CEBA cycles. It checks invariants, orders work, tracks local coordination, and reports outcomes while leaving prediction, Delta, memory, and persistence to their own explicit layers.
