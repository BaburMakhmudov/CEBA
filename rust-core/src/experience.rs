use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeVector {
    pub existence_delta: f64,
    pub evidence_correctability_delta: f64,
    pub coherence_delta: f64,
    pub agency_delta: f64,
    pub resources_delta: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceTrace {
    pub experience_id: String,
    pub state_before_id: String,
    pub action_id: String,
    pub state_after_id: String,
    pub outcome: OutcomeVector,
    pub prediction_error: f64,
    pub context_hash: String,
    pub evidence_refs: Vec<String>,
}

/// Online Gaussian sufficient statistics for Gamma_S consequence learning.
/// This deliberately does not let the language model assign confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineEffectModel {
    pub n: u64,
    pub mean: [f64; 5],
    pub m2: [f64; 5],
}

impl Default for OnlineEffectModel {
    fn default() -> Self {
        Self { n: 0, mean: [0.0; 5], m2: [0.0; 5] }
    }
}

impl OnlineEffectModel {
    pub fn update(&mut self, y: [f64; 5]) {
        self.n += 1;
        let n = self.n as f64;
        for i in 0..5 {
            let delta = y[i] - self.mean[i];
            self.mean[i] += delta / n;
            let delta2 = y[i] - self.mean[i];
            self.m2[i] += delta * delta2;
        }
    }

    pub fn variance(&self) -> [f64; 5] {
        if self.n < 2 { return [f64::INFINITY; 5]; }
        let mut out = [0.0; 5];
        for i in 0..5 { out[i] = self.m2[i] / (self.n - 1) as f64; }
        out
    }
}
