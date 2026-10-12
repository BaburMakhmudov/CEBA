use crate::contracts::{validate_object, CebaObject, ObjectKind};
use sqlx::{PgPool, Row};
use uuid::Uuid;

pub struct Ledger { pool: PgPool }

impl Ledger {
    pub async fn connect(url: &str) -> anyhow::Result<Self> { Ok(Self { pool: PgPool::connect(url).await? }) }

    pub async fn init(&self) -> anyhow::Result<()> {
        sqlx::query(r#"CREATE TABLE IF NOT EXISTS ceba_ledger (
            seq BIGSERIAL PRIMARY KEY,
            object_id UUID NOT NULL UNIQUE,
            kind TEXT NOT NULL,
            payload JSONB NOT NULL,
            payload_hash TEXT NOT NULL,
            chain_hash TEXT NOT NULL UNIQUE,
            parent_object_ids UUID[] NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );"#).execute(&self.pool).await?;
        Ok(())
    }

    pub async fn append(&self, object: &CebaObject) -> anyhow::Result<i64> {
        validate_object(object)?;
        anyhow::ensure!(object.kind == ObjectKind::LedgerCommit || !object.meta.signature_or_hash_chain.is_empty(), "missing hash chain");
        let mut tx = self.pool.begin().await?;
        let row = sqlx::query(r#"INSERT INTO ceba_ledger(object_id,kind,payload,payload_hash,chain_hash,parent_object_ids)
            VALUES($1,$2,$3,$4,$5,$6) RETURNING seq"#)
            .bind(object.meta.object_id)
            .bind(format!("{:?}", object.kind))
            .bind(&object.payload)
            .bind(&object.meta.payload_hash)
            .bind(&object.meta.signature_or_hash_chain)
            .bind(&object.meta.parent_object_ids)
            .fetch_one(&mut *tx).await?;
        let seq: i64 = row.try_get("seq")?;
        tx.commit().await?;
        Ok(seq)
    }

    pub async fn load(&self, id: Uuid) -> anyhow::Result<Option<CebaObject>> {
        let row = sqlx::query("SELECT payload FROM ceba_ledger WHERE object_id=$1").bind(id).fetch_optional(&self.pool).await?;
        Ok(row.and_then(|r| r.try_get::<serde_json::Value,_>("payload").ok()).and_then(|v| serde_json::from_value(v).ok()))
    }
}
