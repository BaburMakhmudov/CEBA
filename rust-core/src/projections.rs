use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationDelta {
    pub subject: String, pub relation: String, pub object: String,
    pub context_hash: String, pub confidence: f64, pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceVector {
    pub experience_id: String, pub vector: Vec<f32>, pub context_hash: String,
    pub ledger_seq: i64, pub payload: serde_json::Value,
}

#[async_trait]
pub trait RelationProjection: Send + Sync {
    async fn project(&self, ledger_seq: i64, deltas: &[RelationDelta]) -> anyhow::Result<()>;
}

#[async_trait]
pub trait ExperienceProjection: Send + Sync {
    async fn upsert(&self, item: &ExperienceVector) -> anyhow::Result<()>;
    async fn similar(&self, vector: &[f32], limit: u64) -> anyhow::Result<Vec<ExperienceVector>>;
}

/// Projection metadata makes secondary stores rebuildable from the ledger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionCheckpoint {
    pub projector_version: String,
    pub last_applied_commit_id: String,
    pub last_applied_commit_hash: String,
    pub projection_checksum: String,
}
