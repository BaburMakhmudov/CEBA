# CEBA Architectural Invariants

## Why Invariants Exist

Invariants prevent layer collapse and category errors. They make sure prediction, Delta, memory, persistence, retrieval, regulation, audit, security, and observability remain separate engineering concerns.

Without invariants, convenient shortcuts can turn parser into meaning inference, vector DB into memory, runtime into Delta computation, or persistence into derived semantic state storage.

## WORLD_MODEL_BEFORE_DELTA

Meaning: Delta requires an expected observation or world-model expectation boundary before mismatch can be computed.

Why it matters: Difference is only well-formed relative to expectation.

Violation example: The parser emits a structure and the Delta layer treats it as mismatch without an expected observation.

Correct behavior: The world model produces expected observation structures first. Delta compares observation-derived structures against that expectation.

## DELTA_ONLY_WRITER

Meaning: Only the Delta boundary may commit Delta-specific state transitions where architecturally permitted.

Why it matters: Delta updates must be traceable and cannot be hidden inside parser, runtime, memory, or persistence.

Violation example: Runtime directly changes Delta status after a parser result without invoking the Delta boundary.

Correct behavior: Runtime coordinates prepared Delta operations through the public Delta layer and records only local coordination status.

## M_T_NOT_PERSISTED

Meaning: `M_t` is derived runtime semantic state and must not be stored in persistence, vector DB, graph DB, ledger, or metadata payloads.

Why it matters: Persisting `M_t` collapses runtime state into durable storage and makes reconstruction, provenance, and invariants ambiguous.

Violation example: A PostgreSQL table named `semantic_state` stores complete runtime state snapshots.

Correct behavior: Persist records, provenance, repository data, and index surfaces. Reconstruct or derive `M_t` during runtime.

## VECTOR_DB_NOT_MEMORY

Meaning: Vector DB stores vectors and retrieval metadata. It is a retrieval surface, not memory.

Why it matters: Memory requires retention, consolidation, validation, eligibility, and replay semantics that vector indexing does not provide by itself.

Violation example: LTM is implemented as a raw vector collection.

Correct behavior: LTM stores verified experience after consolidation. Vector DB indexes retrieval references and metadata.

## STM_NOT_DATABASE

Meaning: STM is active working memory for unresolved differences, not a database.

Why it matters: STM must remain a runtime working layer with eviction, validation, candidate traces, and active context.

Violation example: A database table is treated as the authoritative STM state.

Correct behavior: STM may reference persisted records but holds active difference until resolution through memory-layer semantics.

## TEMPERATURE_NOT_MEMORY

Meaning: Semantic temperature `τ_t` belongs to dynamics and regulation, not memory.

Why it matters: Temperature controls transition strength or plasticity. It is not retained experience.

Violation example: `τ_t` is stored as a memory item or used as a substitute for STM state.

Correct behavior: Keep semantic temperature under dynamics/regulation boundaries such as precision and update policy.

## Invariant Enforcement Points

| Invariant | Enforced/checked by | Relevant folders | Forbidden coupling |
|---|---|---|---|
| `WORLD_MODEL_BEFORE_DELTA` | Runtime invariant validation, Delta preconditions, tests | `world_model/`, `core/delta/`, `runtime/` | Delta calling predictor as a shortcut or running without expectation |
| `DELTA_ONLY_WRITER` | Delta boundary, runtime transaction checks, tests | `core/delta/`, `runtime/` | Parser, runtime, memory, or persistence writing Delta state |
| `M_T_NOT_PERSISTED` | Schema validation, persistence validation, tests | `persistence/`, `runtime/`, `utils/` | Storing `M_t`, `semantic_state`, or runtime semantic state |
| `VECTOR_DB_NOT_MEMORY` | Vector schema/policy validation, memory contract tests | `persistence/vector/`, `memory/` | Treating vector collections as STM or LTM |
| `STM_NOT_DATABASE` | Memory layer validation, persistence boundary checks | `memory/short_term/`, `persistence/` | Database table as authoritative STM |
| `TEMPERATURE_NOT_MEMORY` | Dynamics/regulation boundaries, memory validation | `core/dynamics/`, `core/regulation/`, `memory/` | Storing `τ_t` as memory |

## Static vs Runtime Validation

Static validation includes source-level checks, schema validation, import boundary checks, and tests that reject forbidden names or dependencies.

Runtime validation includes invariant validation during prepared cycles and local transaction checks.

Audit records may describe invariant checks or violations, but audit does not enforce invariants. Observability may report diagnostics, but it does not enforce invariants. Security may validate boundary control, but it does not replace runtime invariant checks.

## Summary

CEBA invariants protect architectural meaning. They keep prediction before Delta, Delta writes explicit, `M_t` derived, vector and graph stores as retrieval surfaces, STM as working memory, and semantic temperature inside dynamics/regulation.
