# CEBA Delta Operator

## Delta Role

Delta is the structured difference or mismatch operator. It compares observation-derived structures against expected observation structures under constraints and metrics.

Delta is not parser output. Delta is not memory. Delta is not persistence. Delta is not world model.

## Delta Preconditions

`WORLD_MODEL_BEFORE_DELTA` means Delta requires an expected observation or world-model expectation boundary before mismatch can be computed.

Without an expectation, an observation can be parsed, embedded, or represented, but it cannot produce a well-formed Delta. Delta is the comparison between what was expected and what was observed.

## Delta Layer Modules

| Module | Role | Allowed Responsibility | Forbidden Responsibility |
|---|---|---|---|
| `schema.py` | Define Delta data structures | Describe typed Delta inputs, outputs, and references | Predict expectations or write memory |
| `delta_operator.py` | Main mismatch operator | Compute structured Delta from prepared inputs | Call world model predictor directly |
| `geodesic_mismatch.py` | Metric-aware mismatch | Measure mismatch under geometry | Reduce Delta to ordinary text similarity |
| `violation.py` | Constraint violation descriptors | Identify violated or satisfied constraints | Persist violation records directly |
| `localization.py` | Violation localization | Locate where mismatch applies | Mutate memory or runtime state |
| `telos_gap.py` | Telos deviation | Describe gap from target or attractor orientation | Infer telos inside parser |
| `structure_score.py` | Structural importance score | Score Delta quality, relevance, or impact | Replace valuation entirely |
| `commit_policy.py` | Controlled acceptance boundary | Decide whether a Delta update is admissible | Execute database commits |

## Delta Computation Boundary

Delta may consume:

- observation-derived structures
- expected observation references
- constraints
- metric or geometric mismatch inputs
- telos gap inputs

Delta must not do:

- world model prediction
- memory writes
- persistence writes
- vector DB calls
- graph DB calls
- LLM calls

## DELTA_ONLY_WRITER

`DELTA_ONLY_WRITER` means only the Delta boundary may commit Delta-specific state transitions where architecturally permitted.

Other layers may prepare inputs, constraints, metrics, or evidence, but they must not create hidden Delta state changes. This preserves traceability and prevents parser, runtime, memory, or persistence layers from silently becoming Delta computation.

## Geodesic Mismatch

Geodesic mismatch is mismatch under a metric. It is not ordinary text similarity and not embedding distance alone.

The mismatch can consider structure, constraints, metric geometry, and expected-observation context. Embeddings may be one input, but they do not define Delta by themselves.

## Violation and Localization

Violation describes which constraints are violated, satisfied, weakened, or activated by a difference. Localization identifies where the violation applies: which observation part, structure, relation, or semantic region is implicated.

Localization makes Delta actionable without letting Delta write memory or persistence directly.

## Telos Gap

Telos gap is deviation from a target, attractor, or normative direction. It is not moral judgment hardcoded into parser behavior.

Parser may produce structure. Telos gap computation belongs to Delta/core boundaries that have access to telos descriptors and expected-observation context.

## Structure Score

Structure score is a Delta quality or importance signal. It may express magnitude, relevance, risk, or impact under constraints and metrics.

It is not the entire valuation layer. Valuation may consume structure scores as inputs.

## Commit Policy

Commit policy controls acceptance of Delta updates. It determines whether a Delta result is admissible for the next stage of state transition.

This is not a persistence commit. It must not execute database writes or ledger writes.

## Summary

Delta is the controlled mismatch boundary. It requires expectation first, consumes prepared semantic structures, localizes violations, scores structure, and respects commit policy without becoming prediction, memory, persistence, or retrieval.
