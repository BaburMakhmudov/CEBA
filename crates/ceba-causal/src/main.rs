use ceba_causal::{gather_interventions,CommitCoordinator,ExperienceMemory,
    GoalEngine,HiddenWorld,ObservedFrame,Viability};
fn main(){
    let engine=GoalEngine{min_independent_interventions:2};
    let (mut tested,mut recovered,mut baselines,mut replays)=(0,0,0,0);
    for seed in 0..40{
        let world=HiddenWorld::new(seed);
        let memory=gather_interventions(&world,3);
        let viability=Viability{
            mandatory_properties:world.properties.iter().cloned().collect()};
        for failure in &world.properties{
            let initial=world.initial(failure);
            let before=ObservedFrame{
                event_id:42,source_sequence:1,properties:initial.clone()};
            baselines+=usize::from(engine.derive(&before,
                &ExperienceMemory::default(),&world.actions,&viability).is_none());
            let cycle=engine.derive(&before,&memory,&world.actions,&viability)
                .expect("no learned endogenous goal");
            let after=world.intervene(&initial,&cycle.proposal.action);
            let mut ledger=CommitCoordinator::default();
            ledger.observe(0,before).expect("initial commit");
            ledger.observe(1,ObservedFrame{
                event_id:43,source_sequence:2,properties:after}).expect("outcome commit");
            recovered+=usize::from(viability.viable(
                &ledger.state.as_ref().unwrap().facts));
            replays+=usize::from(
                ledger.state.as_ref().unwrap()==&ledger.replay());
            tested+=1;
        }
    }
    println!("{{\"cases\":{tested},\"recovered\":{recovered},\"without_memory_abstained\":{baselines},\"exact_replays\":{replays},\"external_goal_specs\":0}}");
    assert_eq!(tested,120);
    assert_eq!(recovered,tested);
    assert_eq!(replays,tested);
    assert_eq!(baselines,tested);
}
