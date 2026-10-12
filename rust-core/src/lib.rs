use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Minimal continuation constraints C0 for a digital cognitive system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuationState {
    /// Runtime/service availability margin.
    pub existence: f64,
    /// Degree to which beliefs remain correctable by external evidence.
    pub evidence_correctability: f64,
    /// Internal unresolved-contradiction margin.
    pub coherence: f64,
    /// Remaining viable action-space margin.
    pub agency: f64,
    /// Remaining compute/time/storage/API resource margin.
    pub resources: f64,
}

impl ContinuationState {
    pub fn margins(&self) -> [f64; 5] {
        [
            self.existence,
            self.evidence_correctability,
            self.coherence,
            self.agency,
            self.resources,
        ]
    }

    pub fn hard_viable(&self) -> bool {
        self.margins().iter().all(|x| x.is_finite() && *x >= 0.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    pub state_id: String,
    pub continuation: ContinuationState,
    /// Additional learned state variables. C0 does not prescribe their meaning.
    pub learned: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionCandidate {
    pub action_id: String,
    pub description: String,
}

/// Gamma_S(X, context, action): experience-derived distribution over
/// system-relative consequences. It is learned evidence, not an LLM label.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionPrediction {
    pub action_id: String,
    pub predicted_next: StateSnapshot,
    /// Epistemic uncertainty in [0, 1].
    pub uncertainty: f64,
    /// Probability of violating any hard continuation constraint.
    pub hard_violation_probability: f64,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValuationConfig {
    pub horizon: usize,
    pub uncertainty_penalty: f64,
    pub risk_penalty: f64,
    pub max_hard_violation_probability: f64,
}

impl Default for ValuationConfig {
    fn default() -> Self {
        Self {
            horizon: 8,
            uncertainty_penalty: 0.25,
            risk_penalty: 1.0,
            max_hard_violation_probability: 0.01,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionValue {
    pub action_id: String,
    pub omega_now: f64,
    pub omega_after: f64,
    pub delta_omega: f64,
    pub uncertainty: f64,
    pub risk: f64,
    pub q_value: f64,
    pub viable: bool,
}

/// First executable approximation of mu_H(S): continuation opportunity measure.
/// Critical margins dominate via the geometric mean; agency explicitly preserves
/// future choice. Replaceable behind this function when the reachability backend lands.
pub fn omega_measure(s: &StateSnapshot) -> f64 {
    if !s.continuation.hard_viable() {
        return 0.0;
    }
    let eps = 1e-12;
    let margins = s.continuation.margins();
    let log_mean = margins
        .iter()
        .map(|x| (x.max(eps)).ln())
        .sum::<f64>()
        / margins.len() as f64;
    log_mean.exp() * (1.0 + s.continuation.agency.max(0.0)).ln_1p()
}

/// Q_t(a) = E[DeltaOmega | S_t,a] - lambda U - beta Risk.
/// Hard-risk gate executes before ranking.
pub fn value_action(
    current: &StateSnapshot,
    prediction: &DirectionPrediction,
    cfg: &ValuationConfig,
) -> ActionValue {
    let omega_now = omega_measure(current);
    let omega_after = omega_measure(&prediction.predicted_next);
    let delta_omega = omega_after - omega_now;
    let viable = prediction.predicted_next.continuation.hard_viable()
        && prediction.hard_violation_probability <= cfg.max_hard_violation_probability;

    let q_value = if viable {
        delta_omega
            - cfg.uncertainty_penalty * prediction.uncertainty
            - cfg.risk_penalty * prediction.hard_violation_probability
    } else {
        f64::NEG_INFINITY
    };

    ActionValue {
        action_id: prediction.action_id.clone(),
        omega_now,
        omega_after,
        delta_omega,
        uncertainty: prediction.uncertainty,
        risk: prediction.hard_violation_probability,
        q_value,
        viable,
    }
}

pub fn choose_action(
    current: &StateSnapshot,
    predictions: &[DirectionPrediction],
    cfg: &ValuationConfig,
) -> Option<ActionValue> {
    predictions
        .iter()
        .map(|p| value_action(current, p, cfg))
        .filter(|v| v.viable)
        .max_by(|a, b| a.q_value.total_cmp(&b.q_value))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(id: &str, m: [f64; 5]) -> StateSnapshot {
        StateSnapshot {
            state_id: id.into(),
            continuation: ContinuationState {
                existence: m[0],
                evidence_correctability: m[1],
                coherence: m[2],
                agency: m[3],
                resources: m[4],
            },
            learned: BTreeMap::new(),
        }
    }

    #[test]
    fn value_is_relative_to_current_state_and_hard_risk_is_gated() {
        let current = state("s0", [0.8, 0.15, 0.63, 0.5, 0.7]);
        let cfg = ValuationConfig::default();
        let predictions = vec![
            DirectionPrediction {
                action_id: "a1".into(),
                predicted_next: state("s1", [0.82, 0.10, 0.67, 0.48, 0.68]),
                uncertainty: 0.1,
                hard_violation_probability: 0.001,
                evidence_refs: vec!["exp:1".into()],
            },
            DirectionPrediction {
                action_id: "a2".into(),
                predicted_next: state("s2", [0.76, 0.48, 0.60, 0.62, 0.65]),
                uncertainty: 0.1,
                hard_violation_probability: 0.001,
                evidence_refs: vec!["exp:2".into()],
            },
            DirectionPrediction {
                action_id: "unsafe".into(),
                predicted_next: state("sx", [0.95, -0.1, 0.9, 0.9, 0.9]),
                uncertainty: 0.0,
                hard_violation_probability: 0.5,
                evidence_refs: vec![],
            },
        ];
        let best = choose_action(&current, &predictions, &cfg).unwrap();
        assert_eq!(best.action_id, "a2");
        assert!(best.delta_omega > 0.0);
    }
}
