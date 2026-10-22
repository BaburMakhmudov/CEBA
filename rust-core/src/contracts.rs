use blake3::Hasher;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeInterval { pub start: DateTime<Utc>, pub end: DateTime<Utc> }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectMeta {
    pub object_id: Uuid,
    pub parent_object_ids: Vec<Uuid>,
    pub schema_id: String,
    pub schema_version: String,
    pub model_context_hash: String,
    pub code_version_hash: String,
    pub timestamp_interval: TimeInterval,
    pub uncertainty: f64,
    pub provenance_refs: Vec<String>,
    pub payload_hash: String,
    pub signature_or_hash_chain: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all="snake_case")]
pub enum ObjectKind { ExpectedFrame, ObservedFrame, DeltaReport, ConstraintReport, RelationFrame, MetricReport, ActionProposal, StateTransitionPlan, ActuatorCommand, ExecutionResult, StateTransition, ExperienceTrace, LedgerCommit }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CebaObject { pub kind: ObjectKind, pub meta: ObjectMeta, pub payload: Value }

pub fn payload_hash(payload: &Value) -> anyhow::Result<String> {
    let bytes = serde_json::to_vec(payload)?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

pub fn chain_hash(previous: Option<&str>, object_id: Uuid, payload_hash: &str) -> String {
    let mut h = Hasher::new();
    if let Some(p) = previous { h.update(p.as_bytes()); }
    h.update(object_id.as_bytes());
    h.update(payload_hash.as_bytes());
    h.finalize().to_hex().to_string()
}

pub fn validate_object(o: &CebaObject) -> anyhow::Result<()> {
    anyhow::ensure!(o.meta.uncertainty.is_finite() && o.meta.uncertainty >= 0.0, "invalid uncertainty");
    anyhow::ensure!(o.meta.timestamp_interval.end >= o.meta.timestamp_interval.start, "invalid time interval");
    anyhow::ensure!(payload_hash(&o.payload)? == o.meta.payload_hash, "payload hash mismatch");
    if o.kind == ObjectKind::ObservedFrame { anyhow::ensure!(!o.meta.provenance_refs.is_empty(), "NO_OBSERVED_FRAME_WITHOUT_PROVENANCE"); }
    Ok(())
}
