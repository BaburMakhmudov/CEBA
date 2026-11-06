# CEBA runnable Rust runtime (verified 2026-11-06)

This branch implements an **executable bounded CEBA vertical slice**, using two independent Tokio engines, PostgreSQL authoritative versioned state and hash-linked append-only commits, NATS JetStream durable event transport, PostgreSQL transactional outbox, persisted counterfactual experience records, dependency-driven invalidation and a causal instrumental goal loop.

## Start (Docker Desktop on Windows, Docker Engine on Linux)

```bash
git clone --branch build/ceba-runnable-runtime https://github.com/BaburMakhmudov/CEBA.git
cd CEBA
docker compose -f compose.yml up --build --abort-on-container-exit --exit-code-from ceba
```

Expected process output contains `"status": "PASS"`, `"state_version": 2`, `"canonical_types": 13`, `"viability_restored": true`, and `"replay_verified": true`. The first run will compile Rust in Docker and start persistent PostgreSQL and NATS containers. A repeat of the same Docker Compose command reuses the persistent database and must succeed without losing state.

Docker Compose environment values are local development examples. Change database secrets and protect network ports before any external deployment.

## Verified execution

GitHub Actions job: https://github.com/BaburMakhmudov/CEBA/actions/runs/42209892192

The CI job ran the five Rust causal core tests, two runtime unit tests, a real PostgreSQL + NATS JetStream end-to-end execution, and a second process invocation against the persisted database. All steps passed.

## What this executes

```text
ObservedFrame -> controlled PostgreSQL commit -> dependency closure
    -> transactional outbox -> JetStream -> Goal Engine
    -> causal evidence + Significance + GoalSpec + ActionProposal
    -> transactional outbox -> JetStream -> Action Engine
    -> simulated environment -> ExecutionResult
    -> controlled state commit -> changed event -> Goal Engine
    -> replay and viability verification
```

The 13 canonical artifact **types** are written to PostgreSQL; their current schemas are limited to this trial domain. The simulation starts with a failing `tests` capability, learns an action mapping using paired interventions, and restores capability through an action. `GoalSpec` is generated from experience and the configured viability requirement.

## Explicit scope boundaries

This executable is **not a finished general CEBA implementation**. The actuator is a controlled small service-world simulator. No Qwen/RWKV model inference is connected to this Rust service; the earlier Qwen experiments are separate. Real text/code/document/browser integrations, multi-host operation, durable intent reconciliation and full semantic/autonomous goal learning need further implementation. No untested general intelligence or unrestricted autonomy is claimed.

## Audit

- Ledger state source: `ceba_snapshot` + `ceba_commits`
- Counterfactual evidence: `ceba_experience`
- Engine outputs: `ceba_artifacts`
- Dependency invalidations: `ceba_invalidations`
- Outbox: `ceba_outbox`
- Validity markers: `ceba_validity`
- Rust core: `crates/ceba-causal`
- Rust runtime: `crates/ceba-runtime`

The code is isolated on `build/ceba-runnable-runtime`; `main` remains unchanged.
