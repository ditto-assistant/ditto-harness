//! Turso (SQLite-family) database layer.
//!
//! Ports the Postgres schema from `db/migrations/000001_memory_schema.up.sql`
//! and the sqlc queries from `db/query/memory.sql` to Turso's SQLite dialect.
//!
//! Schema mapping decisions (see `rust/NOTES.md` for the spike verdict):
//! - `UUID` PKs -> `TEXT`, uuid v4 generated in Rust ([`new_row_id`]).
//! - `BIGSERIAL` -> `INTEGER PRIMARY KEY AUTOINCREMENT`.
//! - `TIMESTAMPTZ` -> `TEXT`, UTC RFC3339 via [`format_timestamp`] (fixed
//!   microsecond precision so lexicographic ordering matches time ordering).
//! - `JSONB` / `TEXT[]` -> `TEXT` containing JSON.
//! - `vector(768)` -> **native** `F32_BLOB(768)`; similarity via
//!   `vector_distance_cos(col, ?)` where `?` is bound as a little-endian f32
//!   blob ([`encode_f32_blob`]). Note `vector_distance_cos` returns cosine
//!   *distance*; similarity = `1 - distance` (mirrors pgvector `<=>`).
//! - HNSW indexes are skipped; per-user brute-force scans are fine locally.

use chrono::{DateTime, SecondsFormat, Utc};

use crate::types::{Error, Result};

/// Embedding dimension used across the harness (embeddinggemma).
pub const EMBEDDING_DIMS: usize = 768;

/// Full schema, one statement per slice entry, applied in order. Every
/// statement is idempotent (`IF NOT EXISTS`), so [`Db::migrate`] can run on
/// every open.
pub const SCHEMA_STATEMENTS: &[&str] = &[
    "CREATE TABLE IF NOT EXISTS harness_users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        uid TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    )",
    "CREATE TABLE IF NOT EXISTS memory_pairs (
        id TEXT PRIMARY KEY,
        firestore_pair_id TEXT NOT NULL,
        user_id TEXT NOT NULL REFERENCES harness_users(uid) ON UPDATE CASCADE ON DELETE CASCADE,
        kg_id TEXT NOT NULL,
        session_id TEXT,
        title TEXT,
        description TEXT,
        prompt TEXT,
        response TEXT,
        input TEXT,
        output TEXT,
        source TEXT,
        source_context TEXT,
        timestamp TEXT NOT NULL,
        timezone_offset INTEGER,
        seed_memories TEXT,
        retrieval_metadata TEXT,
        conversation_embedding F32_BLOB(768),
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        UNIQUE (user_id, firestore_pair_id)
    )",
    "CREATE TABLE IF NOT EXISTS subjects (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL REFERENCES harness_users(uid) ON UPDATE CASCADE ON DELETE CASCADE,
        kg_id TEXT NOT NULL,
        subject_text TEXT NOT NULL,
        description_text TEXT,
        is_key_subject INTEGER NOT NULL DEFAULT 0,
        embedding F32_BLOB(768),
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        UNIQUE (user_id, kg_id, subject_text)
    )",
    "CREATE TABLE IF NOT EXISTS subject_memory_pair_links (
        subject_id TEXT NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
        pair_id TEXT NOT NULL REFERENCES memory_pairs(id) ON DELETE CASCADE,
        user_id TEXT NOT NULL REFERENCES harness_users(uid) ON UPDATE CASCADE ON DELETE CASCADE,
        kg_id TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        PRIMARY KEY (subject_id, pair_id)
    )",
    "CREATE TABLE IF NOT EXISTS retrieval_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        kg_id TEXT NOT NULL,
        session_id TEXT NOT NULL DEFAULT '',
        request_path TEXT NOT NULL DEFAULT '',
        query TEXT NOT NULL DEFAULT '',
        query_embedding F32_BLOB(768),
        retrieved_pair_ids TEXT NOT NULL DEFAULT '[]',
        weights TEXT NOT NULL DEFAULT '{}',
        aux_features TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    )",
    "CREATE INDEX IF NOT EXISTS idx_memory_pairs_user_session_timestamp
        ON memory_pairs (user_id, session_id, timestamp DESC)",
    "CREATE INDEX IF NOT EXISTS idx_memory_pairs_user_kg_timestamp
        ON memory_pairs (user_id, kg_id, timestamp DESC)",
    "CREATE INDEX IF NOT EXISTS idx_subjects_user_kg_key
        ON subjects (user_id, kg_id, is_key_subject, updated_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_subject_memory_pair_links_pair
        ON subject_memory_pair_links (pair_id)",
    "CREATE INDEX IF NOT EXISTS idx_subject_memory_pair_links_user_kg
        ON subject_memory_pair_links (user_id, kg_id)",
    "CREATE INDEX IF NOT EXISTS idx_retrieval_events_user_created
        ON retrieval_events (user_id, created_at DESC)",
];

/// Handle to a Turso database with the harness schema applied.
#[derive(Clone)]
pub struct Db {
    #[allow(dead_code)]
    database: turso::Database,
    conn: turso::Connection,
}

impl Db {
    /// Opens (creating if needed) a database file and applies migrations.
    pub async fn open(path: &str) -> Result<Db> {
        let database = turso::Builder::new_local(path).build().await?;
        Db::from_database(database).await
    }

    /// Opens an in-memory database and applies migrations (tests).
    pub async fn open_memory() -> Result<Db> {
        let database = turso::Builder::new_local(":memory:").build().await?;
        Db::from_database(database).await
    }

    async fn from_database(database: turso::Database) -> Result<Db> {
        let conn = database.connect()?;
        let db = Db { database, conn };
        db.migrate().await?;
        Ok(db)
    }

    /// Applies the schema idempotently. Safe to call on every open.
    pub async fn migrate(&self) -> Result<()> {
        for stmt in SCHEMA_STATEMENTS {
            self.conn.execute(stmt, ()).await?;
        }
        Ok(())
    }

    /// Raw connection access for modules that run bespoke SQL
    /// (`retrieval::composite_retrieve`, `retrieval::log_event`).
    pub fn connection(&self) -> &turso::Connection {
        &self.conn
    }

    /// `-- name: UpsertUser :exec` — insert the uid, ignoring conflicts.
    pub async fn upsert_user(&self, uid: &str) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO harness_users (uid) VALUES (?) ON CONFLICT (uid) DO NOTHING",
                (turso::Value::Text(uid.to_string()),),
            )
            .await?;
        Ok(())
    }

    /// `-- name: CreateMemoryPair :one` — upsert on (user_id, firestore_pair_id),
    /// returning the stored row. Empty-string params for session_id/title/
    /// description/prompt/response/source/source_context are stored as NULL
    /// (Go `NULLIF(..., '')`); `updated_at` is refreshed on conflict.
    pub async fn create_memory_pair(
        &self,
        params: CreateMemoryPairParams,
    ) -> Result<MemoryPairRow> {
        let sql = format!(
            "INSERT INTO memory_pairs (
                id, firestore_pair_id, user_id, kg_id, session_id, title, description,
                prompt, response, input, output, source, source_context, timestamp,
                timezone_offset, seed_memories, retrieval_metadata, conversation_embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (user_id, firestore_pair_id) DO UPDATE SET
                kg_id = excluded.kg_id,
                session_id = excluded.session_id,
                title = excluded.title,
                description = excluded.description,
                prompt = excluded.prompt,
                response = excluded.response,
                input = excluded.input,
                output = excluded.output,
                source = excluded.source,
                source_context = excluded.source_context,
                timestamp = excluded.timestamp,
                timezone_offset = excluded.timezone_offset,
                seed_memories = excluded.seed_memories,
                retrieval_metadata = excluded.retrieval_metadata,
                conversation_embedding = excluded.conversation_embedding,
                updated_at = ?
            RETURNING {}",
            MEMORY_PAIR_COLUMNS
        );
        let args = vec![
            turso::Value::Text(new_row_id()),
            turso::Value::Text(params.firestore_pair_id),
            turso::Value::Text(params.user_id),
            turso::Value::Text(params.kg_id),
            text_or_null(&params.session_id),
            text_or_null(&params.title),
            text_or_null(&params.description),
            text_or_null(&params.prompt),
            text_or_null(&params.response),
            turso::Value::Text(params.input),
            turso::Value::Text(params.output),
            text_or_null(&params.source),
            text_or_null(&params.source_context),
            turso::Value::Text(format_timestamp(params.timestamp)),
            turso::Value::Integer(params.timezone_offset as i64),
            turso::Value::Text(params.seed_memories),
            turso::Value::Text(params.retrieval_metadata),
            embedding_or_null(params.conversation_embedding.as_deref())?,
            turso::Value::Text(format_timestamp(chrono::Utc::now())),
        ];
        let mut rows = self.conn.query(&sql, args).await?;
        match rows.next().await? {
            Some(row) => decode_memory_pair_row(&row, false),
            None => Err(Error::Other(
                "create_memory_pair returned no row".to_string(),
            )),
        }
    }

    /// `-- name: UpsertSubject :one` — upsert on (user_id, kg_id, subject_text).
    /// On conflict: COALESCE non-empty description, OR the key flag, COALESCE
    /// embedding, refresh updated_at. Returns the stored row.
    pub async fn upsert_subject(&self, params: UpsertSubjectParams) -> Result<SubjectRow> {
        let sql = "INSERT INTO subjects (
                id, user_id, kg_id, subject_text, description_text, is_key_subject, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (user_id, kg_id, subject_text) DO UPDATE SET
                description_text = COALESCE(excluded.description_text, description_text),
                is_key_subject = (is_key_subject OR excluded.is_key_subject),
                embedding = COALESCE(excluded.embedding, embedding),
                updated_at = ?
            RETURNING id, user_id, kg_id, subject_text, description_text, is_key_subject, embedding";
        let args = vec![
            turso::Value::Text(new_row_id()),
            turso::Value::Text(params.user_id),
            turso::Value::Text(params.kg_id),
            turso::Value::Text(params.subject_text),
            text_or_null(&params.description_text),
            turso::Value::Integer(params.is_key_subject as i64),
            embedding_or_null(params.embedding.as_deref())?,
            turso::Value::Text(format_timestamp(chrono::Utc::now())),
        ];
        let mut rows = self.conn.query(sql, args).await?;
        match rows.next().await? {
            Some(row) => decode_subject_row(&row, false),
            None => Err(Error::Other("upsert_subject returned no row".to_string())),
        }
    }

    /// `-- name: LinkSubjectMemoryPair :exec` — insert link, ignoring conflicts.
    pub async fn link_subject_memory_pair(
        &self,
        subject_id: &str,
        pair_id: &str,
        user_id: &str,
        kg_id: &str,
    ) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO subject_memory_pair_links (subject_id, pair_id, user_id, kg_id)
                 VALUES (?, ?, ?, ?)
                 ON CONFLICT (subject_id, pair_id) DO NOTHING",
                (
                    turso::Value::Text(subject_id.to_string()),
                    turso::Value::Text(pair_id.to_string()),
                    turso::Value::Text(user_id.to_string()),
                    turso::Value::Text(kg_id.to_string()),
                ),
            )
            .await?;
        Ok(())
    }

    /// `-- name: FetchMemories :many` — rows for the given public pair ids
    /// (`firestore_pair_id`), preserving the order of `pair_ids` (Go uses
    /// `array_position`; here order the results in Rust by input index).
    pub async fn fetch_memories(
        &self,
        user_id: &str,
        pair_ids: &[String],
    ) -> Result<Vec<MemoryPairRow>> {
        if pair_ids.is_empty() {
            return Ok(Vec::new());
        }
        let sql = format!(
            "SELECT {} FROM memory_pairs WHERE user_id = ? AND firestore_pair_id IN ({})",
            MEMORY_PAIR_COLUMNS,
            placeholders(pair_ids.len())
        );
        let mut args = Vec::with_capacity(1 + pair_ids.len());
        args.push(turso::Value::Text(user_id.to_string()));
        args.extend(pair_ids.iter().map(|id| turso::Value::Text(id.clone())));
        let mut rows = self.conn.query(&sql, args).await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(decode_memory_pair_row(&row, false)?);
        }
        // Preserve the input order (Go: ORDER BY array_position(...)).
        let index: std::collections::HashMap<&str, usize> = pair_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.as_str(), i))
            .collect();
        out.sort_by_key(|row| {
            index
                .get(row.firestore_pair_id.as_str())
                .copied()
                .unwrap_or(usize::MAX)
        });
        Ok(out)
    }

    /// `-- name: ListRecentMemories :many` — newest-first for the resolved
    /// session (`COALESCE(session_id,'main') = COALESCE(NULLIF(?,''),'main')`),
    /// excluding `exclude_pair_ids`, limited to `limit`.
    pub async fn list_recent_memories(
        &self,
        params: ListRecentMemoriesParams,
    ) -> Result<Vec<MemoryPairRow>> {
        let mut sql = format!(
            "SELECT {} FROM memory_pairs
             WHERE user_id = ? AND kg_id = ?
               AND COALESCE(session_id, 'main') = COALESCE(NULLIF(?, ''), 'main')",
            MEMORY_PAIR_COLUMNS
        );
        let mut args = vec![
            turso::Value::Text(params.user_id),
            turso::Value::Text(params.kg_id),
            turso::Value::Text(params.session_id),
        ];
        push_not_in_clause(
            &mut sql,
            &mut args,
            "firestore_pair_id",
            &params.exclude_pair_ids,
        );
        sql.push_str(" ORDER BY timestamp DESC LIMIT ?");
        args.push(turso::Value::Integer(params.limit));
        let mut rows = self.conn.query(&sql, args).await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(decode_memory_pair_row(&row, false)?);
        }
        Ok(out)
    }

    /// `-- name: SearchMemories :many` — vector search ordered by cosine
    /// distance then timestamp DESC, with `similarity = 1 - distance` filtered
    /// by `min_similarity`. Empty `session_id` searches all sessions.
    pub async fn search_memories(
        &self,
        params: SearchMemoriesParams,
    ) -> Result<Vec<MemoryPairRow>> {
        let blob = query_embedding_blob(&params.embedding)?;
        let mut sql = format!(
            "SELECT {}, (1.0 - vector_distance_cos(conversation_embedding, ?)) AS similarity
             FROM memory_pairs
             WHERE user_id = ? AND kg_id = ?
               AND conversation_embedding IS NOT NULL
               AND (? = '' OR COALESCE(session_id, 'main') = COALESCE(NULLIF(?, ''), 'main'))
               AND (1.0 - vector_distance_cos(conversation_embedding, ?)) >= ?",
            MEMORY_PAIR_COLUMNS
        );
        let mut args = vec![
            turso::Value::Blob(blob.clone()),
            turso::Value::Text(params.user_id),
            turso::Value::Text(params.kg_id),
            turso::Value::Text(params.session_id.clone()),
            turso::Value::Text(params.session_id),
            turso::Value::Blob(blob.clone()),
            turso::Value::Real(params.min_similarity),
        ];
        push_not_in_clause(
            &mut sql,
            &mut args,
            "firestore_pair_id",
            &params.exclude_pair_ids,
        );
        sql.push_str(
            " ORDER BY vector_distance_cos(conversation_embedding, ?), timestamp DESC LIMIT ?",
        );
        args.push(turso::Value::Blob(blob));
        args.push(turso::Value::Integer(params.limit));
        let mut rows = self.conn.query(&sql, args).await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(decode_memory_pair_row(&row, true)?);
        }
        Ok(out)
    }

    /// `-- name: SearchSubjects :many` — vector search over subjects with a
    /// LEFT JOIN link count (`memory_count`), ordered by distance, then
    /// memory_count DESC, then updated_at DESC.
    pub async fn search_subjects(&self, params: SearchSubjectsParams) -> Result<Vec<SubjectRow>> {
        let blob = query_embedding_blob(&params.embedding)?;
        let sql = "SELECT s.id, s.user_id, s.kg_id, s.subject_text, s.description_text,
                s.is_key_subject, s.embedding,
                (1.0 - vector_distance_cos(s.embedding, ?)) AS similarity,
                COUNT(smpl.pair_id) AS memory_count
            FROM subjects s
            LEFT JOIN subject_memory_pair_links smpl ON smpl.subject_id = s.id
            WHERE s.user_id = ? AND s.kg_id = ?
              AND s.embedding IS NOT NULL
              AND (1.0 - vector_distance_cos(s.embedding, ?)) >= ?
            GROUP BY s.id
            ORDER BY vector_distance_cos(s.embedding, ?), memory_count DESC, s.updated_at DESC
            LIMIT ?";
        let args = vec![
            turso::Value::Blob(blob.clone()),
            turso::Value::Text(params.user_id),
            turso::Value::Text(params.kg_id),
            turso::Value::Blob(blob.clone()),
            turso::Value::Real(params.min_similarity),
            turso::Value::Blob(blob),
            turso::Value::Integer(params.limit),
        ];
        let mut rows = self.conn.query(sql, args).await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(decode_subject_row(&row, true)?);
        }
        Ok(out)
    }

    /// `-- name: SearchMemoriesBySubject :many` — vector search restricted to
    /// pairs linked to `subject_id`, ordered by distance then timestamp DESC.
    pub async fn search_memories_by_subject(
        &self,
        params: SearchMemoriesBySubjectParams,
    ) -> Result<Vec<MemoryPairRow>> {
        let blob = query_embedding_blob(&params.embedding)?;
        let cols = memory_pair_columns_prefixed("mp.");
        let sql = format!(
            "SELECT {cols}, (1.0 - vector_distance_cos(mp.conversation_embedding, ?)) AS similarity
             FROM memory_pairs mp
             JOIN subject_memory_pair_links smpl ON smpl.pair_id = mp.id
             WHERE smpl.subject_id = ? AND mp.user_id = ?
               AND mp.conversation_embedding IS NOT NULL
               AND (1.0 - vector_distance_cos(mp.conversation_embedding, ?)) >= ?
             ORDER BY vector_distance_cos(mp.conversation_embedding, ?), mp.timestamp DESC
             LIMIT ?"
        );
        let args = vec![
            turso::Value::Blob(blob.clone()),
            turso::Value::Text(params.subject_id),
            turso::Value::Text(params.user_id),
            turso::Value::Blob(blob.clone()),
            turso::Value::Real(params.min_similarity),
            turso::Value::Blob(blob),
            turso::Value::Integer(params.limit),
        ];
        let mut rows = self.conn.query(&sql, args).await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(decode_memory_pair_row(&row, true)?);
        }
        Ok(out)
    }
}

/// SELECT column list shared by every memory-pair query (similarity, when
/// present, is appended as column 18).
const MEMORY_PAIR_COLUMNS: &str = "id, firestore_pair_id, user_id, kg_id, session_id, title, \
     description, prompt, response, input, output, source, source_context, timestamp, \
     timezone_offset, seed_memories, retrieval_metadata, conversation_embedding";

fn memory_pair_columns_prefixed(prefix: &str) -> String {
    MEMORY_PAIR_COLUMNS
        .split(", ")
        .map(|col| format!("{prefix}{}", col.trim()))
        .collect::<Vec<_>>()
        .join(", ")
}

/// `?, ?, ...` with `n` placeholders.
fn placeholders(n: usize) -> String {
    let mut out = String::with_capacity(n.saturating_mul(3));
    for i in 0..n {
        if i > 0 {
            out.push_str(", ");
        }
        out.push('?');
    }
    out
}

/// Appends `AND col NOT IN (?, ...)` when `ids` is non-empty (Go:
/// `!= ALL($x::text[])`, where a NULL array disables the filter).
fn push_not_in_clause(
    sql: &mut String,
    args: &mut Vec<turso::Value>,
    column: &str,
    ids: &[String],
) {
    if ids.is_empty() {
        return;
    }
    sql.push_str(&format!(
        " AND {column} NOT IN ({})",
        placeholders(ids.len())
    ));
    args.extend(ids.iter().map(|id| turso::Value::Text(id.clone())));
}

/// Go `NULLIF(arg, '')`: empty strings are stored as NULL.
fn text_or_null(s: &str) -> turso::Value {
    if s.is_empty() {
        turso::Value::Null
    } else {
        turso::Value::Text(s.to_string())
    }
}

/// `None`/empty embeddings bind as NULL; non-empty ones must be
/// [`EMBEDDING_DIMS`] long (Go's `vectorValue` panics on a dim mismatch).
fn embedding_or_null(v: Option<&[f32]>) -> Result<turso::Value> {
    match v {
        None | Some([]) => Ok(turso::Value::Null),
        Some(e) if e.len() != EMBEDDING_DIMS => Err(Error::InvalidArgument(format!(
            "embedding dimension = {}, want {EMBEDDING_DIMS}",
            e.len()
        ))),
        Some(e) => Ok(turso::Value::Blob(encode_f32_blob(e))),
    }
}

/// Query embeddings must always be present and exactly [`EMBEDDING_DIMS`].
fn query_embedding_blob(v: &[f32]) -> Result<Vec<u8>> {
    if v.len() != EMBEDDING_DIMS {
        return Err(Error::InvalidArgument(format!(
            "query embedding dimension = {}, want {EMBEDDING_DIMS}",
            v.len()
        )));
    }
    Ok(encode_f32_blob(v))
}

fn column_error(idx: usize, want: &str, got: &turso::Value) -> Error {
    Error::Other(format!("column {idx}: expected {want}, got {got:?}"))
}

fn get_text(row: &turso::Row, idx: usize) -> Result<String> {
    match row.get_value(idx)? {
        turso::Value::Text(s) => Ok(s),
        other => Err(column_error(idx, "text", &other)),
    }
}

fn get_opt_text(row: &turso::Row, idx: usize) -> Result<Option<String>> {
    match row.get_value(idx)? {
        turso::Value::Null => Ok(None),
        turso::Value::Text(s) => Ok(Some(s)),
        other => Err(column_error(idx, "text or null", &other)),
    }
}

fn get_opt_i64(row: &turso::Row, idx: usize) -> Result<Option<i64>> {
    match row.get_value(idx)? {
        turso::Value::Null => Ok(None),
        turso::Value::Integer(v) => Ok(Some(v)),
        other => Err(column_error(idx, "integer or null", &other)),
    }
}

fn get_i64(row: &turso::Row, idx: usize) -> Result<i64> {
    match row.get_value(idx)? {
        turso::Value::Integer(v) => Ok(v),
        other => Err(column_error(idx, "integer", &other)),
    }
}

fn get_f64(row: &turso::Row, idx: usize) -> Result<f64> {
    match row.get_value(idx)? {
        turso::Value::Real(v) => Ok(v),
        turso::Value::Integer(v) => Ok(v as f64),
        other => Err(column_error(idx, "real", &other)),
    }
}

fn get_opt_embedding(row: &turso::Row, idx: usize) -> Result<Option<Vec<f32>>> {
    match row.get_value(idx)? {
        turso::Value::Null => Ok(None),
        turso::Value::Blob(b) => Ok(Some(decode_f32_blob(&b))),
        other => Err(column_error(idx, "blob or null", &other)),
    }
}

/// Decodes a row in [`MEMORY_PAIR_COLUMNS`] order; `with_similarity` reads the
/// trailing similarity column produced by search queries.
fn decode_memory_pair_row(row: &turso::Row, with_similarity: bool) -> Result<MemoryPairRow> {
    let timestamp_text = get_text(row, 13)?;
    Ok(MemoryPairRow {
        id: get_text(row, 0)?,
        firestore_pair_id: get_text(row, 1)?,
        user_id: get_text(row, 2)?,
        kg_id: get_text(row, 3)?,
        session_id: get_opt_text(row, 4)?,
        title: get_opt_text(row, 5)?,
        description: get_opt_text(row, 6)?,
        prompt: get_opt_text(row, 7)?,
        response: get_opt_text(row, 8)?,
        input: get_opt_text(row, 9)?,
        output: get_opt_text(row, 10)?,
        source: get_opt_text(row, 11)?,
        source_context: get_opt_text(row, 12)?,
        timestamp: parse_timestamp(&timestamp_text)?,
        timezone_offset: get_opt_i64(row, 14)?.map(|v| v as i32),
        seed_memories: get_opt_text(row, 15)?,
        retrieval_metadata: get_opt_text(row, 16)?,
        conversation_embedding: get_opt_embedding(row, 17)?,
        similarity: if with_similarity {
            get_f64(row, 18)?
        } else {
            0.0
        },
    })
}

/// Decodes a subject row (`id, user_id, kg_id, subject_text, description_text,
/// is_key_subject, embedding [, similarity, memory_count]`).
fn decode_subject_row(row: &turso::Row, with_search_cols: bool) -> Result<SubjectRow> {
    Ok(SubjectRow {
        id: get_text(row, 0)?,
        user_id: get_text(row, 1)?,
        kg_id: get_text(row, 2)?,
        subject_text: get_text(row, 3)?,
        description_text: get_opt_text(row, 4)?,
        is_key_subject: get_i64(row, 5)? != 0,
        embedding: get_opt_embedding(row, 6)?,
        similarity: if with_search_cols {
            get_f64(row, 7)?
        } else {
            0.0
        },
        memory_count: if with_search_cols {
            get_i64(row, 8)?
        } else {
            0
        },
    })
}

/// Parameters for [`Db::create_memory_pair`]. String fields documented as
/// "empty -> NULL" mirror Go's `NULLIF(..., '')`.
#[derive(Debug, Clone)]
pub struct CreateMemoryPairParams {
    pub firestore_pair_id: String,
    pub user_id: String,
    pub kg_id: String,
    /// Empty -> NULL.
    pub session_id: String,
    /// Empty -> NULL.
    pub title: String,
    /// Empty -> NULL.
    pub description: String,
    /// Empty -> NULL.
    pub prompt: String,
    /// Empty -> NULL.
    pub response: String,
    /// JSON text (e.g. `[]` or `null`), stored as-is.
    pub input: String,
    /// JSON text, stored as-is.
    pub output: String,
    /// Empty -> NULL.
    pub source: String,
    /// Empty -> NULL.
    pub source_context: String,
    pub timestamp: DateTime<Utc>,
    pub timezone_offset: i32,
    /// JSON text, stored as-is.
    pub seed_memories: String,
    /// JSON text, stored as-is (`"null"` for absent metadata).
    pub retrieval_metadata: String,
    /// `None`/empty -> NULL; otherwise must be [`EMBEDDING_DIMS`] long.
    pub conversation_embedding: Option<Vec<f32>>,
}

/// Parameters for [`Db::upsert_subject`].
#[derive(Debug, Clone)]
pub struct UpsertSubjectParams {
    pub user_id: String,
    pub kg_id: String,
    pub subject_text: String,
    /// Empty -> NULL.
    pub description_text: String,
    pub is_key_subject: bool,
    /// `None`/empty -> NULL; otherwise must be [`EMBEDDING_DIMS`] long.
    pub embedding: Option<Vec<f32>>,
}

/// Parameters for [`Db::list_recent_memories`].
#[derive(Debug, Clone, Default)]
pub struct ListRecentMemoriesParams {
    pub user_id: String,
    pub kg_id: String,
    /// Empty resolves to "main" (matches NULL session rows too).
    pub session_id: String,
    pub exclude_pair_ids: Vec<String>,
    pub limit: i64,
}

/// Parameters for [`Db::search_memories`].
#[derive(Debug, Clone, Default)]
pub struct SearchMemoriesParams {
    pub embedding: Vec<f32>,
    pub user_id: String,
    pub kg_id: String,
    /// Empty searches all sessions; otherwise resolved like ListRecentMemories.
    pub session_id: String,
    pub exclude_pair_ids: Vec<String>,
    pub min_similarity: f64,
    pub limit: i64,
}

/// Parameters for [`Db::search_subjects`].
#[derive(Debug, Clone, Default)]
pub struct SearchSubjectsParams {
    pub embedding: Vec<f32>,
    pub user_id: String,
    pub kg_id: String,
    pub min_similarity: f64,
    pub limit: i64,
}

/// Parameters for [`Db::search_memories_by_subject`].
#[derive(Debug, Clone, Default)]
pub struct SearchMemoriesBySubjectParams {
    pub embedding: Vec<f32>,
    /// Subject row id (TEXT uuid).
    pub subject_id: String,
    pub user_id: String,
    pub min_similarity: f64,
    pub limit: i64,
}

/// A `memory_pairs` row. `similarity` is populated by search queries
/// (`1 - cosine distance`) and 0.0 elsewhere.
#[derive(Debug, Clone)]
pub struct MemoryPairRow {
    /// Internal row uuid (`memory_pairs.id`).
    pub id: String,
    /// Public pair id (`memory_pairs.firestore_pair_id`).
    pub firestore_pair_id: String,
    pub user_id: String,
    pub kg_id: String,
    pub session_id: Option<String>,
    pub title: Option<String>,
    pub description: Option<String>,
    pub prompt: Option<String>,
    pub response: Option<String>,
    /// JSON text.
    pub input: Option<String>,
    /// JSON text.
    pub output: Option<String>,
    pub source: Option<String>,
    pub source_context: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub timezone_offset: Option<i32>,
    /// JSON text.
    pub seed_memories: Option<String>,
    /// JSON text.
    pub retrieval_metadata: Option<String>,
    pub conversation_embedding: Option<Vec<f32>>,
    pub similarity: f64,
}

/// A `subjects` row. `similarity`/`memory_count` are populated by
/// [`Db::search_subjects`] and zero elsewhere.
#[derive(Debug, Clone)]
pub struct SubjectRow {
    pub id: String,
    pub user_id: String,
    pub kg_id: String,
    pub subject_text: String,
    pub description_text: Option<String>,
    pub is_key_subject: bool,
    pub embedding: Option<Vec<f32>>,
    pub similarity: f64,
    pub memory_count: i64,
}

/// Generates a new TEXT primary key (uuid v4) for rows the Postgres schema
/// created with `gen_random_uuid()`.
pub fn new_row_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// Formats a timestamp for storage: UTC RFC3339 with fixed microsecond
/// precision and `Z` suffix, so string comparison equals time comparison.
pub fn format_timestamp(ts: DateTime<Utc>) -> String {
    ts.to_rfc3339_opts(SecondsFormat::Micros, true)
}

/// Parses a stored timestamp (RFC3339; tolerant of varying precision).
pub fn parse_timestamp(s: &str) -> Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|err| Error::InvalidArgument(format!("invalid timestamp {s:?}: {err}")))
}

/// Encodes an embedding as the little-endian f32 blob layout used by Turso's
/// `F32_BLOB` columns (identical to what `vector32('[...]')` produces).
pub fn encode_f32_blob(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Decodes a little-endian f32 blob back into an embedding. Trailing bytes
/// that do not complete an f32 are ignored.
pub fn decode_f32_blob(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|chunk| {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(chunk);
            f32::from_le_bytes(buf)
        })
        .collect()
}

/// Rust-side cosine similarity. Returns `None` for empty or mismatched
/// lengths or zero-norm inputs (mirrors Go `retrieval.cosineSimilarity`).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.is_empty() || a.len() != b.len() {
        return None;
    }
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        let (fx, fy) = (*x as f64, *y as f64);
        dot += fx * fy;
        na += fx * fx;
        nb += fy * fy;
    }
    if na == 0.0 || nb == 0.0 {
        return None;
    }
    Some((dot / (na.sqrt() * nb.sqrt())) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn raw_conn() -> turso::Connection {
        let db = turso::Builder::new_local(":memory:")
            .build()
            .await
            .expect("open in-memory turso db");
        db.connect().expect("connect")
    }

    fn row_f64(row: &turso::Row, idx: usize) -> f64 {
        match row.get_value(idx).expect("get value") {
            turso::Value::Real(v) => v,
            turso::Value::Integer(v) => v as f64,
            other => panic!("expected numeric value, got {other:?}"),
        }
    }

    /// Permanent spike: asserts the native vector path chosen for the schema.
    #[tokio::test]
    async fn turso_native_vector_support() {
        let conn = raw_conn().await;
        conn.execute("CREATE TABLE t (id TEXT, emb F32_BLOB(4))", ())
            .await
            .expect("CREATE F32_BLOB");
        conn.execute(
            "INSERT INTO t VALUES ('a', vector32('[1.0, 0.0, 0.0, 0.0]'))",
            (),
        )
        .await
        .expect("INSERT vector32 literal");

        // Blob parameter binding must be interchangeable with vector32().
        let blob = encode_f32_blob(&[0.0, 1.0, 0.0, 0.0]);
        conn.execute(
            "INSERT INTO t VALUES ('b', ?)",
            (turso::Value::Blob(blob.clone()),),
        )
        .await
        .expect("INSERT blob param");

        // Distance semantics: identical vectors -> 0, orthogonal -> 1.
        let mut rows = conn
            .query(
                "SELECT id, vector_distance_cos(emb, ?) FROM t ORDER BY id",
                (turso::Value::Blob(encode_f32_blob(&[1.0, 0.0, 0.0, 0.0])),),
            )
            .await
            .expect("SELECT vector_distance_cos");
        let row_a = rows.next().await.expect("next").expect("row a");
        assert!(
            row_f64(&row_a, 1).abs() < 1e-6,
            "identical vectors -> distance 0"
        );
        let row_b = rows.next().await.expect("next").expect("row b");
        assert!(
            (row_f64(&row_b, 1) - 1.0).abs() < 1e-6,
            "orthogonal vectors -> distance 1"
        );
    }

    /// Probes used by the db port agent: upsert + RETURNING must work.
    #[tokio::test]
    async fn turso_upsert_and_returning_support() {
        let conn = raw_conn().await;
        conn.execute(
            "CREATE TABLE u (k TEXT PRIMARY KEY, v TEXT, n INTEGER NOT NULL DEFAULT 0)",
            (),
        )
        .await
        .expect("create");
        conn.execute("INSERT INTO u (k, v) VALUES ('a', 'first')", ())
            .await
            .expect("insert");
        let mut rows = conn
            .query(
                "INSERT INTO u (k, v) VALUES ('a', 'second')
                 ON CONFLICT (k) DO UPDATE SET v = excluded.v, n = u.n + 1
                 RETURNING k, v, n",
                (),
            )
            .await
            .expect("upsert with RETURNING");
        let row = rows.next().await.expect("next").expect("returned row");
        assert_eq!(
            row.get_value(1).expect("v"),
            turso::Value::Text("second".to_string())
        );
        assert_eq!(row.get_value(2).expect("n"), turso::Value::Integer(1));
    }

    #[tokio::test]
    async fn migrations_apply_idempotently() {
        let db = Db::open_memory().await.expect("open memory db");
        // Second run must be a no-op thanks to IF NOT EXISTS.
        db.migrate().await.expect("re-run migrations");
        // created_at defaults must fire.
        db.connection()
            .execute("INSERT INTO harness_users (uid) VALUES ('u1')", ())
            .await
            .expect("insert user");
        let mut rows = db
            .connection()
            .query("SELECT created_at FROM harness_users WHERE uid = 'u1'", ())
            .await
            .expect("select");
        let row = rows.next().await.expect("next").expect("row");
        match row.get_value(0).expect("created_at") {
            turso::Value::Text(ts) => {
                parse_timestamp(&ts).expect("default created_at parses as RFC3339");
            }
            other => panic!("expected text created_at, got {other:?}"),
        }
    }

    #[test]
    fn f32_blob_roundtrip_and_cosine() {
        let v = vec![1.5f32, -2.25, 0.0, 3.75];
        assert_eq!(decode_f32_blob(&encode_f32_blob(&v)), v);
        let sim = cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]).expect("similarity");
        assert!((sim - 1.0).abs() < 1e-6);
        assert!(cosine_similarity(&[], &[]).is_none());
        assert!(cosine_similarity(&[1.0], &[1.0, 2.0]).is_none());
        assert!(cosine_similarity(&[0.0], &[1.0]).is_none());
    }

    #[test]
    fn timestamp_format_is_sortable_and_parses() {
        let early = format_timestamp(parse_timestamp("2024-01-01T00:00:00Z").expect("parse"));
        let late = format_timestamp(parse_timestamp("2024-01-01T00:00:00.000001Z").expect("parse"));
        assert!(early < late, "lexicographic order must match time order");
    }

    /// 768-dim embedding with weight 1 on the given axes (only direction
    /// matters for cosine similarity).
    fn axis_embedding(axes: &[usize]) -> Vec<f32> {
        let mut v = vec![0f32; EMBEDDING_DIMS];
        for &axis in axes {
            v[axis % EMBEDDING_DIMS] = 1.0;
        }
        v
    }

    fn pair_params(uid: &str, pair_id: &str, ts: &str) -> CreateMemoryPairParams {
        CreateMemoryPairParams {
            firestore_pair_id: pair_id.to_string(),
            user_id: uid.to_string(),
            kg_id: format!("user_memories_{uid}"),
            session_id: String::new(),
            title: String::new(),
            description: String::new(),
            prompt: format!("prompt {pair_id}"),
            response: format!("response {pair_id}"),
            input: "[]".to_string(),
            output: "[]".to_string(),
            source: String::new(),
            source_context: String::new(),
            timestamp: parse_timestamp(ts).expect("parse test timestamp"),
            timezone_offset: 0,
            seed_memories: "null".to_string(),
            retrieval_metadata: "null".to_string(),
            conversation_embedding: None,
        }
    }

    #[tokio::test]
    async fn create_memory_pair_upserts_with_nullif_semantics() {
        let db = Db::open_memory().await.expect("open");
        db.upsert_user("u1").await.expect("upsert user");
        db.upsert_user("u1").await.expect("upsert user idempotent");

        let mut params = pair_params("u1", "pair-1", "2026-01-01T00:00:00Z");
        params.title = "first title".to_string();
        let first = db.create_memory_pair(params).await.expect("insert");
        assert_eq!(first.firestore_pair_id, "pair-1");
        assert_eq!(first.session_id, None, "empty session stored as NULL");
        assert_eq!(first.title.as_deref(), Some("first title"));

        // Same (user_id, firestore_pair_id) updates in place, keeping the row id.
        let mut params = pair_params("u1", "pair-1", "2026-01-02T00:00:00Z");
        params.session_id = "thread-9".to_string();
        params.conversation_embedding = Some(axis_embedding(&[1]));
        let second = db.create_memory_pair(params).await.expect("upsert");
        assert_eq!(second.id, first.id, "conflict must keep internal row id");
        assert_eq!(second.session_id.as_deref(), Some("thread-9"));
        assert_eq!(second.title, None, "empty title overwrites to NULL");
        assert!(second.conversation_embedding.is_some());
    }

    #[tokio::test]
    async fn upsert_subject_merges_on_conflict() {
        let db = Db::open_memory().await.expect("open");
        db.upsert_user("u1").await.expect("user");
        let first = db
            .upsert_subject(UpsertSubjectParams {
                user_id: "u1".to_string(),
                kg_id: "kg".to_string(),
                subject_text: "rust".to_string(),
                description_text: "the language".to_string(),
                is_key_subject: true,
                embedding: Some(axis_embedding(&[2])),
            })
            .await
            .expect("insert subject");

        // Conflict keeps description/key/embedding when the new row is empty.
        let second = db
            .upsert_subject(UpsertSubjectParams {
                user_id: "u1".to_string(),
                kg_id: "kg".to_string(),
                subject_text: "rust".to_string(),
                description_text: String::new(),
                is_key_subject: false,
                embedding: None,
            })
            .await
            .expect("upsert subject");
        assert_eq!(second.id, first.id);
        assert_eq!(second.description_text.as_deref(), Some("the language"));
        assert!(second.is_key_subject, "key flag is OR-ed");
        assert!(second.embedding.is_some(), "embedding COALESCEd");
    }

    #[tokio::test]
    async fn fetch_memories_preserves_input_order() {
        let db = Db::open_memory().await.expect("open");
        db.upsert_user("u1").await.expect("user");
        for pair_id in ["a", "b", "c"] {
            db.create_memory_pair(pair_params("u1", pair_id, "2026-01-01T00:00:00Z"))
                .await
                .expect("insert");
        }
        let rows = db
            .fetch_memories(
                "u1",
                &[
                    "c".to_string(),
                    "a".to_string(),
                    "b".to_string(),
                    "missing".to_string(),
                ],
            )
            .await
            .expect("fetch");
        let got: Vec<&str> = rows.iter().map(|r| r.firestore_pair_id.as_str()).collect();
        assert_eq!(got, ["c", "a", "b"]);
    }

    #[tokio::test]
    async fn list_recent_memories_resolves_sessions_and_excludes() {
        let db = Db::open_memory().await.expect("open");
        db.upsert_user("u1").await.expect("user");
        let main_pair = pair_params("u1", "main-1", "2026-01-01T00:00:00Z");
        db.create_memory_pair(main_pair).await.expect("insert");
        let mut thread_pair = pair_params("u1", "thread-1", "2026-01-02T00:00:00Z");
        thread_pair.session_id = "thread".to_string();
        db.create_memory_pair(thread_pair).await.expect("insert");

        let rows = db
            .list_recent_memories(ListRecentMemoriesParams {
                user_id: "u1".to_string(),
                kg_id: "user_memories_u1".to_string(),
                session_id: String::new(),
                exclude_pair_ids: Vec::new(),
                limit: 10,
            })
            .await
            .expect("list main");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].firestore_pair_id, "main-1");

        let rows = db
            .list_recent_memories(ListRecentMemoriesParams {
                user_id: "u1".to_string(),
                kg_id: "user_memories_u1".to_string(),
                session_id: "thread".to_string(),
                exclude_pair_ids: vec!["thread-1".to_string()],
                limit: 10,
            })
            .await
            .expect("list thread with exclusion");
        assert!(rows.is_empty(), "excluded pair must not be returned");
    }

    #[tokio::test]
    async fn search_memories_filters_and_orders_by_similarity() {
        let db = Db::open_memory().await.expect("open");
        db.upsert_user("u1").await.expect("user");
        let mut close = pair_params("u1", "close", "2026-01-01T00:00:00Z");
        close.conversation_embedding = Some(axis_embedding(&[0, 1]));
        db.create_memory_pair(close).await.expect("insert close");
        let mut far = pair_params("u1", "far", "2026-01-01T00:00:00Z");
        far.conversation_embedding = Some(axis_embedding(&[5]));
        db.create_memory_pair(far).await.expect("insert far");
        let none_pair = pair_params("u1", "none", "2026-01-01T00:00:00Z");
        db.create_memory_pair(none_pair).await.expect("insert none");

        let rows = db
            .search_memories(SearchMemoriesParams {
                embedding: axis_embedding(&[0]),
                user_id: "u1".to_string(),
                kg_id: "user_memories_u1".to_string(),
                session_id: String::new(),
                exclude_pair_ids: Vec::new(),
                min_similarity: 0.15,
                limit: 10,
            })
            .await
            .expect("search");
        assert_eq!(rows.len(), 1, "only the close vector passes 0.15: {rows:?}");
        assert_eq!(rows[0].firestore_pair_id, "close");
        assert!(
            (rows[0].similarity - 1.0 / 2f64.sqrt()).abs() < 1e-5,
            "similarity = {}",
            rows[0].similarity
        );
    }
}
