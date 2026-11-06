use anyhow::{anyhow,bail,Context,Result};
use async_nats::jetstream::{self,consumer,stream};
use ceba_causal::{gather_interventions,ExperienceMemory,Facts,GoalEngine,HiddenWorld,InterventionEvidence,ObservedFrame,Viability};
use futures_util::StreamExt;
use serde::{Deserialize,Serialize};
use serde_json::{json,Value};
use std::{collections::{BTreeMap,BTreeSet,VecDeque},env,time::Duration};
use tokio_postgres::{Client,NoTls,Transaction};

#[derive(Clone,Debug,Serialize,Deserialize)]
struct Snapshot {version:i64,sequence:i64,facts:Facts,hash:String}
#[derive(Clone,Debug,Serialize,Deserialize)]
struct Changed {version:i64,sequence:i64,event_id:String,changed:Vec<String>}
#[derive(Clone,Debug,Serialize,Deserialize)]
struct Action {id:String,base_version:i64,action:String,property:String,evidence:Vec<u64>}

async fn db(url:&str)->Result<Client>{
    let (client,connection)=tokio_postgres::connect(url,NoTls).await?;
    tokio::spawn(async move {if let Err(e)=connection.await {eprintln!("postgres: {e}");}});
    Ok(client)
}
async fn bootstrap(client:&Client,reset:bool)->Result<()>{
    client.batch_execute(r#"
    CREATE TABLE IF NOT EXISTS ceba_snapshot(id int PRIMARY KEY,version bigint NOT NULL,sequence bigint NOT NULL,facts text NOT NULL,last_hash text NOT NULL);
    CREATE TABLE IF NOT EXISTS ceba_commits(version bigint PRIMARY KEY,event_id text UNIQUE NOT NULL,source_sequence bigint UNIQUE NOT NULL,kind text NOT NULL,facts text NOT NULL,previous_hash text NOT NULL,content_hash text NOT NULL);
    CREATE TABLE IF NOT EXISTS ceba_artifacts(artifact_id text PRIMARY KEY,version bigint NOT NULL,kind text NOT NULL,body text NOT NULL);
    CREATE TABLE IF NOT EXISTS ceba_experience(experiment_id bigint PRIMARY KEY,body text NOT NULL);
    CREATE TABLE IF NOT EXISTS ceba_outbox(id bigserial PRIMARY KEY,subject text NOT NULL,body text NOT NULL,sent boolean NOT NULL DEFAULT false);
    CREATE TABLE IF NOT EXISTS ceba_invalidations(version bigint NOT NULL,engine_id text NOT NULL,PRIMARY KEY(version,engine_id));
    CREATE TABLE IF NOT EXISTS ceba_validity(engine_id text PRIMARY KEY,valid_version bigint NOT NULL);
    "#).await?;
    if reset {
        client.batch_execute("TRUNCATE ceba_validity,ceba_invalidations,ceba_outbox,ceba_experience,ceba_artifacts,ceba_commits,ceba_snapshot RESTART IDENTITY").await?;
    }
    client.execute("INSERT INTO ceba_snapshot VALUES(1,0,0,'{}','genesis') ON CONFLICT DO NOTHING",&[]).await?;
    Ok(())
}
async fn read(client:&Client)->Result<Snapshot>{
    let r=client.query_one("SELECT version,sequence,facts,last_hash FROM ceba_snapshot WHERE id=1",&[]).await?;
    Ok(Snapshot{version:r.get(0),sequence:r.get(1),facts:serde_json::from_str(&r.get::<_,String>(2))?,hash:r.get(3)})
}
fn hash(prev:&str,id:&str,version:i64,data:&str)->String{
    blake3::hash(format!("{prev}|{id}|{version}|{data}").as_bytes()).to_hex().to_string()
}
fn closure(changed:&[String])->BTreeSet<String>{
    let mut edges:BTreeMap<&str,Vec<&str>>=BTreeMap::new();
    for key in ["compiler","tests","artifact"]{edges.insert(key,vec!["world_model","goal"]);}
    edges.insert("world_model",vec!["delta"]);edges.insert("delta",vec!["relation"]);
    edges.insert("relation",vec!["significance"]);edges.insert("significance",vec!["goal"]);
    edges.insert("goal",vec!["action"]);
    let mut found=BTreeSet::new();
    let mut queue:VecDeque<String>=changed.iter().cloned().collect();
    while let Some(node)=queue.pop_front(){
        if !found.insert(node.clone()){continue;}
        if let Some(nodes)=edges.get(node.as_str()){for child in nodes{queue.push_back((*child).to_string());}}
    }
    found.retain(|s|["world_model","delta","relation","significance","goal","action"].contains(&s.as_str()));
    found
}
async fn artifact(tx:&Transaction<'_>,id:&str,version:i64,kind:&str,data:Value)->Result<()>{
    let body=serde_json::to_string(&data)?;
    tx.execute("INSERT INTO ceba_artifacts VALUES($1,$2,$3,$4) ON CONFLICT(artifact_id) DO NOTHING",
        &[&id,&version,&kind,&body]).await?;
    Ok(())
}
async fn commit(client:&mut Client,id:&str,base:i64,seq:i64,facts:Facts,kind:&str)->Result<bool>{
    let tx=client.transaction().await?;
    let row=tx.query_one("SELECT version,sequence,facts,last_hash FROM ceba_snapshot WHERE id=1 FOR UPDATE",&[]).await?;
    let version:i64=row.get(0);
    let last_seq:i64=row.get(1);
    let before:Facts=serde_json::from_str(&row.get::<_,String>(2))?;
    let previous:String=row.get(3);
    if tx.query_opt("SELECT 1 FROM ceba_commits WHERE event_id=$1",&[&id]).await?.is_some(){
        tx.rollback().await?;return Ok(false);
    }
    if version!=base{bail!("stale base_version {base}; actual {version}");}
    if seq!=last_seq+1{bail!("out-of-order source sequence {seq}; expected {}",last_seq+1);}
    let version=base+1;
    let mut changed:Vec<String>=facts.iter().filter(|(k,v)|before.get(*k)!=Some(*v))
        .map(|(k,_)|k.clone()).collect();
    changed.extend(before.keys().filter(|k|!facts.contains_key(*k)).cloned());
    changed.sort();
    let data=serde_json::to_string(&facts)?;
    let next_hash=hash(&previous,id,version,&data);
    tx.execute("INSERT INTO ceba_commits(version,event_id,source_sequence,kind,facts,previous_hash,content_hash) VALUES($1,$2,$3,$4,$5,$6,$7)",
        &[&version,&id,&seq,&kind,&data,&previous,&next_hash]).await?;
    tx.execute("UPDATE ceba_snapshot SET version=$1,sequence=$2,facts=$3,last_hash=$4 WHERE id=1",
        &[&version,&seq,&data,&next_hash]).await?;
    artifact(&tx,&format!("{id}/ExpectedFrame"),version,"ExpectedFrame",json!({"version":base,"properties":before})).await?;
    artifact(&tx,&format!("{id}/ObservedFrame"),version,"ObservedFrame",json!({"id":id,"sequence":seq,"properties":facts})).await?;
    artifact(&tx,&format!("{id}/DeltaReport"),version,"DeltaReport",json!({"changed":changed})).await?;
    artifact(&tx,&format!("{id}/ConstraintReport"),version,"ConstraintReport",json!({"unavailable":facts.iter().filter(|(_,v)|!**v).map(|(k,_)|k.clone()).collect::<Vec<_>>()})).await?;
    artifact(&tx,&format!("{id}/MetricReport"),version,"MetricReport",json!({"available":facts.values().filter(|x|**x).count(),"total":facts.len()})).await?;
    artifact(&tx,&format!("{id}/LedgerCommit"),version,"LedgerCommit",json!({"hash":next_hash,"previous_hash":previous})).await?;
    let affected=closure(&changed);
    for engine in &affected{
        tx.execute("INSERT INTO ceba_invalidations VALUES($1,$2) ON CONFLICT DO NOTHING",&[&version,&engine]).await?;
    }
    if affected.contains("goal"){
        let payload=serde_json::to_string(&Changed{version,sequence:seq,event_id:id.into(),changed})?;
        tx.execute("INSERT INTO ceba_outbox(subject,body) VALUES('ceba.state.changed',$1)",&[&payload]).await?;
    }
    tx.commit().await?;
    Ok(true)
}
async fn remember(client:&Client,world:&HiddenWorld)->Result<usize>{
    let experiences=gather_interventions(world,3);
    for e in &experiences.interventions{
        let body=serde_json::to_string(&json!({
            "experiment_id":e.experiment_id,"action":e.action,
            "before":e.before,"under_action":e.under_action,"no_action":e.no_action}))?;
        client.execute("INSERT INTO ceba_experience VALUES($1,$2) ON CONFLICT DO NOTHING",
            &[&(e.experiment_id as i64),&body]).await?;
    }
    Ok(experiences.interventions.len())
}
async fn recall(client:&Client)->Result<ExperienceMemory>{
    let mut memory=ExperienceMemory::default();
    for row in client.query("SELECT body FROM ceba_experience ORDER BY experiment_id",&[]).await?{
        let body:String=row.get(0);
        let v:Value=serde_json::from_str(&body)?;
        memory.remember(InterventionEvidence{
            experiment_id:v["experiment_id"].as_u64().context("experiment_id")?,
            action:v["action"].as_str().context("action")?.into(),
            before:serde_json::from_value(v["before"].clone())?,
            under_action:serde_json::from_value(v["under_action"].clone())?,
            no_action:serde_json::from_value(v["no_action"].clone())?,
        }).map_err(|s|anyhow!(s))?;
    }
    Ok(memory)
}
async fn propose(client:&mut Client,req:&Action,relation:Value,gain:f64)->Result<()>{
    let tx=client.transaction().await?;
    let version:i64=tx.query_one("SELECT version FROM ceba_snapshot WHERE id=1 FOR UPDATE",&[]).await?.get(0);
    if version!=req.base_version{tx.rollback().await?;return Ok(());}
    let goal_key=format!("{}/GoalSpec",req.id);
    let body=serde_json::to_string(req)?;
    let inserted=tx.execute("INSERT INTO ceba_artifacts VALUES($1,$2,'GoalSpec',$3) ON CONFLICT(artifact_id) DO NOTHING",
        &[&goal_key,&version,&body]).await?;
    if inserted==0{tx.rollback().await?;return Ok(());}
    artifact(&tx,&format!("{}/RelationFrame",req.id),version,"RelationFrame",relation).await?;
    artifact(&tx,&format!("{}/Significance",req.id),version,"Significance",json!({"delta_viability":gain})).await?;
    artifact(&tx,&format!("{}/ActionProposal",req.id),version,"ActionProposal",json!({"action":req.action,"base_version":version})).await?;
    artifact(&tx,&format!("{}/StateTransitionPlan",req.id),version,"StateTransitionPlan",json!({"property":req.property,"target":true})).await?;
    tx.execute("INSERT INTO ceba_outbox(subject,body) VALUES('ceba.action.requested',$1)",&[&body]).await?;
    tx.commit().await?;
    Ok(())
}
async fn make_valid(client:&Client,version:i64)->Result<()>{
    client.execute("INSERT INTO ceba_validity VALUES('goal',$1) ON CONFLICT(engine_id) DO UPDATE SET valid_version=GREATEST(ceba_validity.valid_version,EXCLUDED.valid_version)",
        &[&version]).await?;Ok(())
}
async fn send_outbox(client:&Client,js:&jetstream::Context)->Result<()>{
    let rows=client.query("SELECT id,subject,body FROM ceba_outbox WHERE NOT sent ORDER BY id LIMIT 20",&[]).await?;
    for row in rows{
        let id:i64=row.get(0);let subject:String=row.get(1);let body:String=row.get(2);
        js.publish(subject,body.into()).await?.await?;
        client.execute("UPDATE ceba_outbox SET sent=true WHERE id=$1",&[&id]).await?;
    }
    Ok(())
}
async fn goal_worker(db_url:String,consumer:consumer::PullConsumer,world:HiddenWorld)->Result<()>{
    let mut pg=db(&db_url).await?;
    let mut messages=consumer.messages().await?;
    while let Some(msg)=messages.next().await{
        let msg=msg?;
        let change:Changed=serde_json::from_slice(&msg.payload)?;
        let snapshot=read(&pg).await?;
        if snapshot.version==change.version{
            let memory=recall(&pg).await?;
            let criterion=Viability{mandatory_properties:world.properties.iter().cloned().collect()};
            let engine=GoalEngine{min_independent_interventions:2};
            let obs=ObservedFrame{event_id:snapshot.version as u64,source_sequence:snapshot.sequence as u64,properties:snapshot.facts.clone()};
            if let Some(cycle)=engine.derive(&obs,&memory,&world.actions,&criterion){
                let relation=&cycle.proposal.from_relation;
                let evidence=relation.evidence_ids.clone();
                let request=Action{
                    id:format!("goal/v{}",snapshot.version),
                    base_version:snapshot.version,
                    action:cycle.proposal.action,property:cycle.goal.property,evidence,
                };
                propose(&mut pg,&request,json!({"action":relation.action,"property":relation.property,
                    "support":relation.support,"contradicting":relation.contradicting,"evidence_ids":relation.evidence_ids}),
                    cycle.significance.delta_viability).await?;
            }
            make_valid(&pg,snapshot.version).await?;
        }
        msg.ack().await.map_err(|e|anyhow!(e.to_string()))?;
    }
    Ok(())
}
async fn action_worker(db_url:String,consumer:consumer::PullConsumer,world:HiddenWorld)->Result<()>{
    let mut pg=db(&db_url).await?;
    let mut messages=consumer.messages().await?;
    while let Some(msg)=messages.next().await{
        let msg=msg?;
        let request:Action=serde_json::from_slice(&msg.payload)?;
        let snapshot=read(&pg).await?;
        if snapshot.version==request.base_version{
            let after=world.intervene(&snapshot.facts,&request.action);
            let id=format!("execution/{}/{}",request.base_version,request.action);
            if commit(&mut pg,&id,snapshot.version,snapshot.sequence+1,after.clone(),"ActionEffectTransition").await?{
                let tx=pg.transaction().await?;
                artifact(&tx,&format!("{id}/ActuatorCommand"),snapshot.version,"ActuatorCommand",
                    json!({"action":request.action,"authorized":true,"base_version":snapshot.version})).await?;
                artifact(&tx,&format!("{id}/ExecutionResult"),snapshot.version+1,"ExecutionResult",
                    json!({"result":after,"action":request.action,"success":true})).await?;
                artifact(&tx,&format!("{id}/StateTransition"),snapshot.version+1,"StateTransition",
                    json!({"from":snapshot.version,"to":snapshot.version+1})).await?;
                artifact(&tx,&format!("{id}/ExperienceTrace"),snapshot.version+1,"ExperienceTrace",
                    json!({"goal":request.property,"action":request.action,"evidence":request.evidence,"result":after})).await?;
                tx.commit().await?;
            }
        }
        msg.ack().await.map_err(|e|anyhow!(e.to_string()))?;
    }
    Ok(())
}
async fn replay(client:&Client)->Result<(Snapshot,usize)>{
    let rows=client.query("SELECT version,event_id,source_sequence,facts,previous_hash,content_hash FROM ceba_commits ORDER BY version",&[]).await?;
    let mut state=Snapshot{version:0,sequence:0,facts:Facts::new(),hash:"genesis".into()};
    for row in &rows{
        let version:i64=row.get(0);let id:String=row.get(1);
        let seq:i64=row.get(2);let body:String=row.get(3);
        let prev:String=row.get(4);let h:String=row.get(5);
        if version!=state.version+1||seq!=state.sequence+1||prev!=state.hash||hash(&prev,&id,version,&body)!=h{
            bail!("ledger integrity violation at version {version}");
        }
        state=Snapshot{version,sequence:seq,facts:serde_json::from_str(&body)?,hash:h};
    }
    Ok((state,rows.len()))
}
#[tokio::main]
async fn main()->Result<()>{
    let pg_url=env::var("DATABASE_URL").unwrap_or_else(|_| "postgres://ceba:ceba_dev_only@localhost:5432/ceba".into());
    let nats_url=env::var("NATS_URL").unwrap_or_else(|_|"nats://localhost:4222".into());
    let reset=env::var("CEBA_RESET").as_deref()==Ok("1");
    let world=HiddenWorld::new(17);
    let mut pg=db(&pg_url).await?;
    bootstrap(&pg,reset).await?;
    let n_experience=remember(&pg,&world).await?;
    let client=async_nats::connect(nats_url).await?;
    let js=jetstream::new(client);
    let stream=js.get_or_create_stream(stream::Config{
        name:"CEBA_RUNTIME".into(),subjects:vec!["ceba.>".into()],..Default::default()
    }).await?;
    let goals=stream.get_or_create_consumer("ceba-goal-engine",consumer::pull::Config{
        durable_name:Some("ceba-goal-engine".into()),filter_subject:"ceba.state.changed".into(),
        ..Default::default()
    }).await?;
    let actions=stream.get_or_create_consumer("ceba-action-engine",consumer::pull::Config{
        durable_name:Some("ceba-action-engine".into()),filter_subject:"ceba.action.requested".into(),
        ..Default::default()
    }).await?;
    let g=tokio::spawn(goal_worker(pg_url.clone(),goals,world.clone()));
    let a=tokio::spawn(action_worker(pg_url.clone(),actions,world.clone()));
    if read(&pg).await?.version==0{
        let id=format!("observed/{}",uuid::Uuid::now_v7());
        commit(&mut pg,&id,0,1,world.initial("tests"),"ObservationUpdate").await?;
    }
    let viability=Viability{mandatory_properties:world.properties.iter().cloned().collect()};
    let started=tokio::time::Instant::now();
    let final_state=loop{
        send_outbox(&pg,&js).await?;
        let current=read(&pg).await?;
        let valid:Option<i64>=pg.query_opt("SELECT valid_version FROM ceba_validity WHERE engine_id='goal'",&[]).await?
            .map(|r|r.get(0));
        let pending:i64=pg.query_one("SELECT COUNT(*) FROM ceba_outbox WHERE NOT sent",&[]).await?.get(0);
        if viability.viable(&current.facts)&&valid==Some(current.version)&&pending==0{break current;}
        if started.elapsed()>Duration::from_secs(30){bail!("runtime deadline exceeded; state={current:?}; valid={valid:?}");}
        tokio::time::sleep(Duration::from_millis(40)).await;
    };
    let (restored,n_commits)=replay(&pg).await?;
    if restored.facts!=final_state.facts||restored.hash!=final_state.hash||restored.version!=final_state.version{
        bail!("restart replay mismatch");
    }
    let artifact_rows=pg.query("SELECT kind,COUNT(*) FROM ceba_artifacts GROUP BY kind",&[]).await?;
    let mut artifact_types=BTreeMap::new();
    for row in artifact_rows{
        let kind:String=row.get(0);let count:i64=row.get(1);
        artifact_types.insert(kind,count);
    }
    for required in ["ExpectedFrame","ObservedFrame","DeltaReport","ConstraintReport","RelationFrame",
        "MetricReport","StateTransitionPlan","ActionProposal","ActuatorCommand","ExecutionResult",
        "StateTransition","ExperienceTrace","LedgerCommit"] {
        if !artifact_types.contains_key(required){bail!("missing canonical type: {required}");}
    }
    let invalidations:i64=pg.query_one("SELECT COUNT(*) FROM ceba_invalidations",&[]).await?.get(0);
    println!("{}",serde_json::to_string_pretty(&json!({
        "status":"PASS","domain":"bounded service world",
        "postgres_ledger":true,"nats_jetstream":true,
        "independent_engines":2,"state_version":final_state.version,
        "ledger_commits":n_commits,"experience_records":n_experience,
        "dependency_invalidations":invalidations,
        "canonical_types":13,"artifacts":artifact_types,
        "endogenous_instrumental_goal":true,"viability_restored":true,
        "replay_verified":true,"state":final_state.facts,
        "real_world_actuation":false,"neural_model_integrated":false
    }))?);
    g.abort();a.abort();
    Ok(())
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test] fn transitive_invalidation(){
        let set=closure(&["tests".into()]);
        for e in ["world_model","delta","relation","significance","goal","action"]{assert!(set.contains(e));}
        assert!(closure(&["not_relevant".into()]).is_empty());
    }
    #[test] fn hash_chain_is_sensitive_to_changes(){
        let h=hash("genesis","obs",1,"{}");
        assert_eq!(h,hash("genesis","obs",1,"{}"));
        assert_ne!(h,hash("genesis","obs",2,"{}"));
        assert_ne!(h,hash("genesis","other",1,"{}"));
    }
}
