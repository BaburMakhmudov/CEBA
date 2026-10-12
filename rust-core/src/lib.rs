pub mod contracts;
pub mod experience;
pub mod ledger;
pub mod projections;

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuationState { pub existence:f64, pub evidence_correctability:f64, pub coherence:f64, pub agency:f64, pub resources:f64 }
impl ContinuationState {
    pub fn margins(&self)->[f64;5]{[self.existence,self.evidence_correctability,self.coherence,self.agency,self.resources]}
    pub fn hard_viable(&self)->bool{self.margins().iter().all(|x|x.is_finite()&&*x>=0.0)}
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSnapshot { pub state_id:String, pub continuation:ContinuationState, pub learned:BTreeMap<String,f64> }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionCandidate { pub action_id:String, pub description:String }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionPrediction { pub action_id:String, pub predicted_next:StateSnapshot, pub uncertainty:f64, pub hard_violation_probability:f64, pub evidence_refs:Vec<String> }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValuationConfig { pub horizon:usize, pub uncertainty_penalty:f64, pub risk_penalty:f64, pub max_hard_violation_probability:f64 }
impl Default for ValuationConfig { fn default()->Self{Self{horizon:8,uncertainty_penalty:0.25,risk_penalty:1.0,max_hard_violation_probability:0.01}} }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionValue { pub action_id:String,pub omega_now:f64,pub omega_after:f64,pub delta_omega:f64,pub uncertainty:f64,pub risk:f64,pub q_value:f64,pub viable:bool }

pub fn omega_measure(s:&StateSnapshot)->f64{
    if !s.continuation.hard_viable(){return 0.0}
    let eps=1e-12; let m=s.continuation.margins();
    let log_mean=m.iter().map(|x|x.max(eps).ln()).sum::<f64>()/m.len() as f64;
    log_mean.exp()*(1.0+s.continuation.agency.max(0.0)).ln()
}
pub fn value_action(current:&StateSnapshot,p:&DirectionPrediction,cfg:&ValuationConfig)->ActionValue{
    let now=omega_measure(current); let after=omega_measure(&p.predicted_next); let d=after-now;
    let viable=p.predicted_next.continuation.hard_viable()&&p.hard_violation_probability<=cfg.max_hard_violation_probability;
    let q=if viable{d-cfg.uncertainty_penalty*p.uncertainty-cfg.risk_penalty*p.hard_violation_probability}else{f64::NEG_INFINITY};
    ActionValue{action_id:p.action_id.clone(),omega_now:now,omega_after:after,delta_omega:d,uncertainty:p.uncertainty,risk:p.hard_violation_probability,q_value:q,viable}
}
pub fn choose_action(current:&StateSnapshot,predictions:&[DirectionPrediction],cfg:&ValuationConfig)->Option<ActionValue>{
    predictions.iter().map(|p|value_action(current,p,cfg)).filter(|v|v.viable).max_by(|a,b|a.q_value.total_cmp(&b.q_value))
}

#[cfg(test)] mod tests { use super::*; fn s(id:&str,m:[f64;5])->StateSnapshot{StateSnapshot{state_id:id.into(),continuation:ContinuationState{existence:m[0],evidence_correctability:m[1],coherence:m[2],agency:m[3],resources:m[4]},learned:BTreeMap::new()}}
#[test] fn selects_state_relative_viable_action(){let cur=s("s0",[.8,.15,.63,.5,.7]);let cfg=ValuationConfig::default();let p=vec![DirectionPrediction{action_id:"a1".into(),predicted_next:s("s1",[.82,.10,.67,.48,.68]),uncertainty:.1,hard_violation_probability:.001,evidence_refs:vec![]},DirectionPrediction{action_id:"a2".into(),predicted_next:s("s2",[.76,.48,.60,.62,.65]),uncertainty:.1,hard_violation_probability:.001,evidence_refs:vec![]}];assert_eq!(choose_action(&cur,&p,&cfg).unwrap().action_id,"a2");}}
