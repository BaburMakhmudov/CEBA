# CEBA Ontology

## Ontological Purpose

CEBA is built around controlled update of semantic state, not answer generation. Its ontology describes how understanding changes when an observation differs from an expectation and that difference is handled under constraints.

The ontology is engineering-oriented. It defines component responsibilities and dependency order so implementation layers can remain separated.

## Core Chain

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

| Term | Engineering Meaning |
|---|---|
| difference | A mismatch, novelty, or residual signal between observation and expectation |
| constraint | A rule or boundary that determines what the difference violates, satisfies, or activates |
| metric | A cost, distance, penalty, relevance, or importance geometry |
| dynamics | A state transition model under constraints and metrics |
| regulation | Precision, step size, update, replay, consolidation, and forgetting policy |
| telos | Target, attractor, goal-shape, or normative orientation |
| valuation | Utility, free-energy, entropy, risk, or score evaluation |
| relation | Structural connectivity in the semantic field |
| meaning-state stabilization | A derived runtime state that has been updated consistently enough for the cycle |

This is not a simple linear pipe. Runtime execution may coordinate components in a different operational order, but must preserve these dependency constraints.

## Difference

Difference is a detected mismatch or novelty signal. It is not final meaning and does not by itself determine what the system understands.

Difference must be interpreted against an expected observation boundary. Without an expectation, there is no well-formed mismatch, only an observation.

## Constraint

Constraints are rules, boundaries, commitments, invariants, or permissible regions. They determine what a difference violates, satisfies, activates, or leaves unresolved.

Constraints are not persistence. They may be represented durably as records, but the constraint layer itself is a semantic boundary layer.

## Metric

Metric and cost define the geometry of importance, distance, penalty, priority, or relevance. Metrics make differences comparable and allow dynamics to choose update strength and direction.

Metrics are not valuation. A metric can supply the measurement space; valuation evaluates consequence or utility within or over that space.

## Dynamics

Dynamics describes state transition under metric and constraints. It models how an update moves from one semantic configuration toward another.

Dynamics is not memory. It does not store experience; it governs transition behavior.

## Regulation

Regulation controls precision, step size, update policy, consolidation policy, replay policy, and forgetting policy. It determines how much plasticity or resistance should apply during a state transition.

Regulation is where update control belongs. It must not be mixed into STM, LTM, persistence, or retrieval stores.

## Semantic Temperature

`τ_t` is semantic temperature.

- It belongs to dynamics and regulation.
- It controls transition strength and plasticity.
- It is not memory.
- It is not STM.
- It is not LTM.
- It is not persistence.

Semantic temperature may influence how strongly an active difference modifies state, but it is not the thing remembered and not a database field for semantic state.

## Telos

Telos is target, attractor, normative orientation, or goal-shape. It is not a plain string list. A string may name or refer to a telos descriptor, but telos itself is a structured orientation that influences update direction and evaluation.

Parser and input layers must not infer telos. Telos belongs to its own core boundary.

## Valuation

Valuation evaluates utility, free energy, entropy, risk, score, or other consequence measures. It is not relation and not metric alone.

Valuation depends on prior structure: differences, constraints, metrics, dynamics, regulation, and telos can all influence valuation inputs.

## Relation

Relation is structural connectivity inside the semantic field. It represents how concepts, observations, constraints, and states connect.

Relation is not graph DB. Relation runtime and graph persistence must remain separate. Graph persistence can index relation records for retrieval, but it is not the relation computation layer.

## Meaning-State Stabilization

`M_t` is derived runtime semantic state. It represents stabilized semantic state at a runtime boundary.

- `M_t` is not persisted.
- `M_t` is reconstructed or derived during runtime.
- Persistence stores records and retrieval surfaces, not `M_t`.

Persisting `M_t` would collapse runtime state into storage and violate the `M_T_NOT_PERSISTED` invariant.

## Execution Order vs Ontological Order

Runtime execution may coordinate objects in an operational order that differs from the ontological dependency order. For example, a runtime cycle may validate invariants, open a local transaction, schedule prepared tasks, and coordinate checkpoints before or after component-specific objects are assembled.

This does not change the ontology. Delta still requires an expectation. Valuation still must not replace metrics. Relation still must not become graph DB. Runtime scheduling must preserve the invariant dependencies even when execution order is optimized.

## Summary

CEBA’s ontology separates mismatch, constraint, measurement, transition, regulation, orientation, evaluation, relation, and stabilization. The purpose is to make semantic state updates controlled, inspectable, and bounded without confusing memory, persistence, retrieval, runtime, or derived `M_t`.
