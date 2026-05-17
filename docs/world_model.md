# CEBA World Model

## World Model Role

The world model is the expectation and prediction boundary. It produces expected observation structures or confidence-bearing predictions that Delta can compare against actual observations.

The world model defines what was expected before mismatch is computed.

## World Model Is Not

The world model is:

- not memory
- not delta
- not persistence
- not runtime scheduler
- not telos
- not valuation
- not `M_t`

It may use evidence and model state, but it must not collapse into those layers.

## World Model Modules

| Module | Role | Allowed Responsibility | Forbidden Responsibility |
|---|---|---|---|
| `schema.py` | World model data structures | Define expected observation and prediction descriptors | Store memory or persistence records |
| `errors.py` | World model errors | Represent safe local errors | Hide runtime or persistence failures |
| `state.py` | Model-state descriptors | Describe local model state | Persist `M_t` |
| `predictor.py` | Prediction boundary | Produce expected observation structures | Compute Delta directly |
| `expected_observation.py` | Expected observation schema | Represent predicted observation references | Mutate memory |
| `confidence.py` | Confidence descriptors | Represent uncertainty and quality | Replace valuation |
| `update.py` | Model update boundary | Apply model-state updates | Consolidate memory or write persistence |
| `evidence.py` | Evidence descriptors | Describe support for prediction or update | Act as memory itself |
| `validation.py` | Validation utilities | Validate world model descriptors | Execute runtime scheduling |

## Expected Observation

Expected observation must exist before Delta computation because Delta is the mismatch between expected and observed. Without an expectation, the system can record an observation but cannot compute a principled difference.

Expected observation may be structured, partial, uncertain, or confidence-bearing. It still establishes the comparison boundary.

## Confidence

Confidence is an uncertainty or quality descriptor. It can describe how reliable, complete, or supported a prediction is.

Confidence is not valuation. Valuation can use confidence, but confidence alone does not determine utility or update acceptance.

## Evidence

World model evidence supports prediction and update. Evidence may reference observations, traces, retrieval outputs, or prior records.

Evidence is not memory itself. Memory layers determine retention and consolidation; evidence descriptors provide support for a prediction or update.

## Update Boundary

World model update changes model state. It is not memory consolidation and not persistence.

An update may be informed by Delta, evidence, and validation, but durable storage and memory consolidation remain separate boundaries.

## Relation to Runtime

Runtime coordinates prepared cycles but does not directly implement prediction. Runtime may invoke a public world model boundary as part of a prepared flow, but it must not bypass the predictor or implement prediction as a shortcut.

## Relation to Delta

Delta compares observation against expected observation. Delta must not call predictor directly as a shortcut.

The correct dependency is:

```text
world model expectation → delta comparison
```

This preserves `WORLD_MODEL_BEFORE_DELTA`.

## Summary

The world model supplies expectation. It is separate from runtime, Delta, memory, persistence, telos, valuation, and derived `M_t`. This separation keeps mismatch computation well-formed and traceable.
