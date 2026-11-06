//! CEBA causal-goal core: observation, counterfactual experience, relation
//! learning, significance, endogenous goal, action proposal, controlled commit.
use std::collections::{BTreeMap,BTreeSet};

pub type Facts = BTreeMap<String,bool>;
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct ObservedFrame {pub event_id:u64,pub source_sequence:u64,pub properties:Facts}
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct ExpectedFrame {pub state_version:u64,pub properties:Facts}
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct DeltaReport {pub observed_event:u64,pub changes:BTreeMap<String,(Option<bool>,bool)>}
pub fn difference(expected:&ExpectedFrame, observed:&ObservedFrame)->DeltaReport{
    let changes=observed.properties.iter().filter_map(|(key,&value)|{
        let before=expected.properties.get(key).copied();
        (before!=Some(value)).then_some((key.clone(),(before,value)))
    }).collect();
    DeltaReport{observed_event:observed.event_id,changes}
}
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct InterventionEvidence{
    pub experiment_id:u64,pub action:String,pub before:Facts,
    pub under_action:Facts,pub no_action:Facts,
}
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct CausalRelation{
    pub action:String,pub property:String,pub pre_value:bool,pub post_value:bool,
    pub support:usize,pub contradicting:usize,pub evidence_ids:Vec<u64>,
}
#[derive(Clone,Debug,Default,PartialEq,Eq)]
pub struct ExperienceMemory {pub interventions:Vec<InterventionEvidence>}
impl ExperienceMemory {
    pub fn remember(&mut self,e:InterventionEvidence)->Result<(),String>{
        if self.interventions.iter().any(|old|old.experiment_id==e.experiment_id){
            return Err("duplicate experiment id".into());
        }
        if e.before.keys().collect::<BTreeSet<_>>()!=e.under_action.keys().collect() ||
           e.before.keys().collect::<BTreeSet<_>>()!=e.no_action.keys().collect(){
            return Err("mismatched observation domains".into());
        }
        self.interventions.push(e);
        Ok(())
    }
    // Candidate effects require a paired do(action) vs do(nothing) comparison.
    pub fn relation(&self,action:&str,property:&str,before:bool,after:bool)->CausalRelation{
        let (mut support,mut contradicting)=(0,0);
        let mut ids=Vec::new();
        for e in self.interventions.iter().filter(|e|
            e.action==action && e.before.get(property)==Some(&before)){
            let with=e.under_action.get(property).copied();
            let without=e.no_action.get(property).copied();
            if with==Some(after) && without!=Some(after){
                support+=1;ids.push(e.experiment_id);
            } else {contradicting+=1;}
        }
        CausalRelation{action:action.into(),property:property.into(),
            pre_value:before,post_value:after,support,contradicting,evidence_ids:ids}
    }
}
#[derive(Clone,Debug,PartialEq)]
pub struct Significance{
    pub causal_relation:CausalRelation,
    pub delta_viability:f64,
}
#[derive(Clone,Debug,PartialEq)]
pub struct GoalSpec{
    pub goal_id:u64,pub originating_event:u64,
    pub property:String,pub target_value:bool,
    pub derived_significance:f64,pub causal_evidence:Vec<u64>,
}
#[derive(Clone,Debug,PartialEq)]
pub struct ActionProposal{
    pub goal_id:u64,pub action:String,pub expected_viability_gain:f64,
    pub from_relation:CausalRelation,
}
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct LedgerCommit{
    pub version:u64,pub event_id:u64,
    pub changed_fields:Vec<String>,pub new_facts:Facts,
}
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct StateSnapshot{
    pub version:u64,pub facts:Facts,pub last_source_sequence:u64,
}
#[derive(Default)]
pub struct CommitCoordinator {
    pub state:Option<StateSnapshot>,pub ledger:Vec<LedgerCommit>,
}
impl CommitCoordinator {
    pub fn observe(&mut self,base_version:u64,observed:ObservedFrame)->Result<DeltaReport,String>{
        let current=self.state.clone().unwrap_or(StateSnapshot{
            version:0,facts:Facts::new(),last_source_sequence:0});
        if base_version!=current.version{return Err("stale read-set version".into());}
        if observed.source_sequence!=current.last_source_sequence+1{
            return Err("out-of-order input".into());
        }
        if self.ledger.iter().any(|c|c.event_id==observed.event_id){
            return Err("duplicate event_id".into());
        }
        let delta=difference(&ExpectedFrame{
            state_version:current.version,properties:current.facts},&observed);
        let version=base_version+1;
        self.ledger.push(LedgerCommit{version,event_id:observed.event_id,
            changed_fields:delta.changes.keys().cloned().collect(),
            new_facts:observed.properties.clone()});
        self.state=Some(StateSnapshot{
            version,facts:observed.properties,
            last_source_sequence:observed.source_sequence});
        Ok(delta)
    }
    pub fn replay(&self)->StateSnapshot{
        let mut state=StateSnapshot{
            version:0,facts:Facts::new(),last_source_sequence:0};
        for event in &self.ledger{
            assert_eq!(event.version,state.version+1,"nonsequential ledger");
            state.version=event.version;
            state.last_source_sequence+=1;
            state.facts=event.new_facts.clone();
        }
        state
    }
}
#[derive(Clone,Debug)]
pub struct EndogenousCycle{
    pub goal:GoalSpec,pub proposal:ActionProposal,pub significance:Significance,
}
#[derive(Clone,Debug)]
pub struct Viability{pub mandatory_properties:BTreeSet<String>}
impl Viability{
    pub fn viable(&self,facts:&Facts)->bool{
        self.mandatory_properties.iter().all(|key|facts.get(key)==Some(&true))
    }
}
#[derive(Clone,Debug)]
pub struct GoalEngine{pub min_independent_interventions:usize}
impl GoalEngine{
    pub fn derive(&self,event:&ObservedFrame,memory:&ExperienceMemory,
        actions:&[String],viability:&Viability)->Option<EndogenousCycle>{
        if viability.viable(&event.properties){return None;}
        let mut best:Option<(String,CausalRelation,f64)>=None;
        for field in viability.mandatory_properties.iter().filter(|field|
            event.properties.get(*field)==Some(&false)){
            for action in actions{
                let relation=memory.relation(action,field,false,true);
                if relation.support<self.min_independent_interventions ||
                   relation.contradicting>0{continue;}
                let blocked=viability.mandatory_properties.iter().filter(|p|
                    event.properties.get(*p)!=Some(&true)).count();
                let gain=1.0/blocked as f64;
                if best.as_ref().map_or(true,|(_,_,v)|gain>*v){
                    best=Some((field.clone(),relation,gain));
                }
            }
        }
        best.map(|(field,relation,gain)|{
            let goal=GoalSpec{
                goal_id:event.event_id,originating_event:event.event_id,
                property:field,target_value:true,derived_significance:gain,
                causal_evidence:relation.evidence_ids.clone()};
            let proposal=ActionProposal{
                goal_id:goal.goal_id,action:relation.action.clone(),
                expected_viability_gain:gain,from_relation:relation.clone()};
            EndogenousCycle{goal,proposal,significance:Significance{
                causal_relation:relation,delta_viability:gain}}
        })
    }
}
// Test environment: action semantics are hidden and re-permuted for each seed.
#[derive(Clone,Debug)]
pub struct HiddenWorld{
    pub properties:Vec<String>,
    pub actions:Vec<String>,
    pub mapping:BTreeMap<String,String>,
}
impl HiddenWorld{
    pub fn new(seed:u64)->Self{
        let names=vec!["compiler","tests","artifact"].into_iter()
            .map(str::to_string).collect::<Vec<_>>();
        let labels=(0..5).map(|i|format!("unknown_operation_{i}")).collect::<Vec<_>>();
        let mut mapping=BTreeMap::new();
        // All 5P3 = 60 distinct injective action mappings are available.
        // The first 40 seeds therefore exercise 40 genuinely different worlds.
        let mut assignments=Vec::new();
        for a in 0..5{
            for b in 0..5{
                if b==a {continue;}
                for c in 0..5{
                    if c!=a && c!=b {assignments.push([a,b,c]);}
                }
            }
        }
        let assignment=assignments[(seed as usize)%assignments.len()];
        for (i,name) in names.iter().enumerate(){
            mapping.insert(labels[assignment[i]].clone(),name.clone());
        }
        Self{properties:names,actions:labels,mapping}
    }
    pub fn intervene(&self,state:&Facts,action:&str)->Facts{
        let mut out=state.clone();
        if let Some(key)=self.mapping.get(action){out.insert(key.clone(),true);}
        out
    }
    pub fn initial(&self,failure:&str)->Facts{
        self.properties.iter().map(|p|(p.clone(),p.as_str()!=failure)).collect()
    }
}
pub fn gather_interventions(world:&HiddenWorld,training_rounds:usize)->ExperienceMemory{
    let mut memory=ExperienceMemory::default();
    let mut id=0;
    for property in &world.properties{
        for _ in 0..training_rounds{
            for action in &world.actions{
                let before=world.initial(property);
                let after=world.intervene(&before,action);
                let no_action=before.clone();
                id+=1;
                memory.remember(InterventionEvidence{
                    experiment_id:id,action:action.clone(),before,
                    under_action:after,no_action}).unwrap();
            }
        }
    }
    memory
}
#[cfg(test)]
mod tests{
    use super::*;
    fn viability(w:&HiddenWorld)->Viability{
        Viability{mandatory_properties:w.properties.iter().cloned().collect()}
    }
    #[test]
    fn goals_emerge_from_experience_under_40_permutations(){
        let engine=GoalEngine{min_independent_interventions:2};
        for seed in 0..40 {
            let world=HiddenWorld::new(seed);
            let learned=gather_interventions(&world,3);
            for failure in &world.properties{
                let facts=world.initial(failure);
                let event=ObservedFrame{event_id:1,source_sequence:1,properties:facts.clone()};
                assert!(engine.derive(&event,&ExperienceMemory::default(),
                    &world.actions,&viability(&world)).is_none());
                let cycle=engine.derive(&event,&learned,&world.actions,
                    &viability(&world)).expect("no endogenous goal");
                assert_eq!(cycle.goal.property,*failure);
                assert!(cycle.goal.causal_evidence.len()>=2);
                assert!(!cycle.proposal.action.contains(failure));
                assert!(viability(&world).viable(&world.intervene(&facts,
                    &cycle.proposal.action)));
                let repaired=ObservedFrame{
                    event_id:2,source_sequence:2,
                    properties:world.intervene(&facts,&cycle.proposal.action)};
                assert!(engine.derive(&repaired,&learned,&world.actions,
                    &viability(&world)).is_none());
            }
        }
    }
    #[test]
    fn correlation_without_intervention_cannot_produce_goals(){
        let world=HiddenWorld::new(3);
        let engine=GoalEngine{min_independent_interventions:2};
        let facts=world.initial("tests");
        let mut memory=ExperienceMemory::default();
        for i in 1..5{
            let before=facts.clone();
            let after=world.intervene(&before,&world.actions[0]);
            memory.remember(InterventionEvidence{
                experiment_id:i,action:world.actions[0].clone(),before,
                under_action:after.clone(),no_action:after}).unwrap();
        }
        assert!(engine.derive(&ObservedFrame{event_id:11,source_sequence:1,properties:facts},
            &memory,&world.actions,&viability(&world)).is_none());
    }
    #[test]
    fn single_intervention_not_sufficient(){
        let world=HiddenWorld::new(2);
        let memory=gather_interventions(&world,1);
        let engine=GoalEngine{min_independent_interventions:2};
        assert!(engine.derive(&ObservedFrame{
            event_id:1,source_sequence:1,properties:world.initial("compiler")},
            &memory,&world.actions,&viability(&world)).is_none());
    }
    #[test]
    fn controlled_commits_replay_and_reject_stale(){
        let world=HiddenWorld::new(4);
        let memory=gather_interventions(&world,2);
        let engine=GoalEngine{min_independent_interventions:2};
        let observed=ObservedFrame{
            event_id:400,source_sequence:1,properties:world.initial("artifact")};
        let mut ledger=CommitCoordinator::default();
        assert_eq!(ledger.observe(0,observed.clone()).unwrap().changes.len(),
            world.properties.len());
        assert!(ledger.observe(0,observed.clone()).is_err());
        let cycle=engine.derive(&observed,&memory,&world.actions,
            &viability(&world)).unwrap();
        let result=ObservedFrame{event_id:401,source_sequence:2,
            properties:world.intervene(&observed.properties,&cycle.proposal.action)};
        ledger.observe(1,result.clone()).unwrap();
        assert!(viability(&world).viable(&ledger.state.as_ref().unwrap().facts));
        assert_eq!(ledger.state.as_ref().unwrap(),&ledger.replay());
        assert!(ledger.observe(1,result).is_err());
    }
    #[test]
    fn unrelated_actions_cannot_recover_viability(){
        let world=HiddenWorld::new(2);
        let memory=gather_interventions(&world,3);
        let engine=GoalEngine{min_independent_interventions:2};
        let facts=world.initial("tests");
        let cycle=engine.derive(&ObservedFrame{
            event_id:9,source_sequence:1,properties:facts.clone()},
            &memory,&world.actions,&viability(&world)).unwrap();
        for action in &world.actions{
            if action!=&cycle.proposal.action{
                assert!(!viability(&world).viable(&world.intervene(&facts,action)));
            }
        }
    }
}
