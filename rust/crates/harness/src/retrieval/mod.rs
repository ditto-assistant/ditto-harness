// SPDX-License-Identifier: AGPL-3.0-or-later
//! Composite retrieval, retrieval-event logging, auxiliary feature
//! extraction, and the loadable learned-weight MLP predictor.
//! Port of Go `pkg/retrieval`.

pub mod features;
pub mod mlp;

use std::collections::HashMap;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::db::Db;
use crate::types::{is_zero_f64, Error, Result};

pub use features::{
    extract_auxiliary_features, extract_auxiliary_features_context, AuxFeatureContext,
    AUX_FEATURE_DIM, LEGACY_AUX_FEATURE_DIM,
};
pub use mlp::{weights_from_slice, MlpPredictor};

/// Composite scoring variant (Go: `retrieval.Variant`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Variant {
    /// Cosine + linear recency + subject frequency (Go: `VariantLegacy`, "v1").
    #[default]
    #[serde(rename = "v1")]
    Legacy,
    /// Adds recency-exp, subject semantic match, session continuity, and
    /// neighbor density (Go: `VariantV2`, "v2").
    #[serde(rename = "v2")]
    V2,
}

impl std::fmt::Display for Variant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Variant::Legacy => f.write_str("v1"),
            Variant::V2 => f.write_str("v2"),
        }
    }
}

/// Indexes into the V2 weight vector (Go: `V2Weight*` consts).
pub const V2_WEIGHT_COSINE: usize = 0;
pub const V2_WEIGHT_RECENCY_LINEAR: usize = 1;
pub const V2_WEIGHT_RECENCY_EXP: usize = 2;
pub const V2_WEIGHT_SUBJECT_FREQUENCY: usize = 3;
pub const V2_WEIGHT_SUBJECT_SEM_MATCH: usize = 4;
pub const V2_WEIGHT_SESSION_CONTINUITY: usize = 5;
pub const V2_WEIGHT_NEIGHBOR_DENSITY: usize = 6;
pub const V2_NUM_WEIGHTS: usize = 7;

/// Time constant for the V2 recency-exp feature, in seconds
/// (Go inlines `14*24*3600.0`).
pub const V2_RECENCY_TAU_SECS: f64 = 14.0 * 24.0 * 3600.0;

/// Composite retrieval weights (Go: `Weights`). JSON matches Go tags.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Weights {
    pub cosine: f64,
    pub recency_linear: f64,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub recency_exp: f64,
    pub subject_frequency: f64,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub subject_sem_match: f64,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub session_continuity: f64,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub neighbor_density: f64,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub scale: f64,
}

impl Weights {
    /// The all-zero value (Go's `Weights{}`), used for "unset" checks.
    pub const ZERO: Weights = Weights {
        cosine: 0.0,
        recency_linear: 0.0,
        recency_exp: 0.0,
        subject_frequency: 0.0,
        subject_sem_match: 0.0,
        session_continuity: 0.0,
        neighbor_density: 0.0,
        scale: 0.0,
    };

    /// True when every weight (including scale) is zero, i.e. "unset".
    pub fn is_zero(&self) -> bool {
        *self == Weights::ZERO
    }
}

/// `Default` mirrors Go `DefaultWeights()`: cosine 0.65, recencyLinear 0.20,
/// subjectFrequency 0.15, scale 1.
impl Default for Weights {
    fn default() -> Weights {
        Weights {
            cosine: 0.65,
            recency_linear: 0.20,
            subject_frequency: 0.15,
            scale: 1.0,
            ..Weights::ZERO
        }
    }
}

/// Go: `DefaultWeights()`.
pub fn default_weights() -> Weights {
    Weights::default()
}

/// One scored candidate from composite retrieval (Go: `CompositeMemory`).
/// `pair_id` is the public pair id (`firestore_pair_id`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompositeMemory {
    pub pair_id: String,
    pub cosine_similarity: f64,
    pub recency_score: f64,
    pub frequency_score: f64,
    pub composite_score: f64,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub recency_exp: f64,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub subject_sem_match: f64,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub session_continuity: f64,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub neighbor_density: f64,
}

/// Parameters for [`composite_retrieve`] (Go: `CompositeParams`).
#[derive(Debug, Clone, Default)]
pub struct CompositeParams {
    pub embedding: Vec<f32>,
    pub user_id: String,
    pub kg_id: String,
    /// Empty searches all sessions.
    pub session_id: String,
    /// Used for the V2 session-continuity feature.
    pub current_session_id: String,
    /// `None` -> unix epoch (no lower bound in practice).
    pub min_timestamp: Option<DateTime<Utc>>,
    /// `0` -> 8.
    pub limit: usize,
    /// `0` -> `max(32, limit * 4)`.
    pub candidate_pool_size: usize,
    pub exclude_pair_ids: Vec<String>,
    /// `Weights::ZERO` -> defaults; `scale == 0` -> 1.
    pub weights: Weights,
    pub variant: Variant,
    pub request_path: String,
    pub query: String,
    /// When true, a retrieval event is logged (best effort, errors ignored).
    pub log_event: bool,
}

/// Auxiliary features fed to a [`WeightPredictor`] (Go: `Features`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Features {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub query: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub now: Option<DateTime<Utc>>,
    /// Never serialized (Go: `json:"-"`).
    #[serde(skip)]
    pub query_embedding: Vec<f32>,
    #[serde(default, skip_serializing_if = "crate::types::is_zero_i64")]
    pub short_term_memory_count: i64,
    #[serde(default, skip_serializing_if = "crate::types::is_zero_i64")]
    pub candidate_memory_count: i64,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub current_session_id: String,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub embedding_norm: f64,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub host_application_signal: String,
}

/// Predicts composite retrieval weights from query features
/// (Go: `WeightPredictor`).
#[async_trait]
pub trait WeightPredictor: Send + Sync {
    async fn predict(&self, features: &Features) -> Result<Weights>;
}

/// Default size of the candidate pool handed to a [`Reranker`] before
/// truncating to the caller's limit (Go production: `ceRerankPoolSize = 20`).
pub const RERANK_POOL_SIZE: usize = 20;

/// Second-stage reranker applied to the composite-ordered candidate pool
/// (Go production: the cross-encoder rerank in `pkg/services/retrieval/crossencoder`).
///
/// Mirrors the production pipeline shape: composite retrieval widens the pool to
/// [`RERANK_POOL_SIZE`], the reranker reorders it against `query`, and the result
/// is truncated to `top_n`. Implementations own their relevance model and how it
/// fuses with the incoming composite order (production uses Reciprocal Rank
/// Fusion of cross-encoder rank with composite rank). `pool` arrives ordered
/// best-first by composite score; the returned vec must be best-first and at most
/// `top_n` long.
///
/// The concrete model (e.g. an ONNX cross-encoder + tokenizer + weights) lives in
/// the consuming crate so this crate stays free of an inference runtime.
#[async_trait]
pub trait Reranker: Send + Sync {
    async fn rerank(
        &self,
        query: &str,
        pool: Vec<crate::types::Memory>,
        top_n: usize,
    ) -> Result<Vec<crate::types::Memory>>;
}

/// A predictor that always returns fixed weights (Go: `StaticPredictor`).
/// Zero weights fall back to [`default_weights`].
#[derive(Debug, Clone, Copy, Default)]
pub struct StaticPredictor {
    pub weights: Weights,
}

#[async_trait]
impl WeightPredictor for StaticPredictor {
    async fn predict(&self, _features: &Features) -> Result<Weights> {
        if self.weights.is_zero() {
            return Ok(default_weights());
        }
        Ok(self.weights)
    }
}

/// Runs composite retrieval (Go: `CompositeRetrieve`).
///
/// Implementation strategy for Turso: pull the candidate pool with a vector
/// query (`ORDER BY vector_distance_cos(conversation_embedding, ?) LIMIT
/// candidate_pool_size`, same WHERE filters as `Db::search_memories`), pull
/// subject-link aggregates with plain SQL, then compute recency / frequency /
/// (V2: recency-exp, subject sem match, session continuity, neighbor density)
/// and the weighted composite score in Rust. Result ordering: composite score
/// DESC, then timestamp DESC, then pair id DESC; truncated to `limit`.
/// When `log_event` is set, call [`log_event`] best-effort (ignore errors).
pub async fn composite_retrieve(
    db: &Db,
    mut params: CompositeParams,
) -> Result<Vec<CompositeMemory>> {
    if params.limit == 0 {
        params.limit = 8;
    }
    if params.candidate_pool_size == 0 {
        params.candidate_pool_size = 32.max(params.limit * 4);
    }
    let mut weights = if params.weights.is_zero() {
        default_weights()
    } else {
        params.weights
    };
    if weights.scale == 0.0 {
        weights.scale = 1.0;
    }

    let results = score_candidates(db, &params, weights).await?;

    if params.log_event {
        let ids: Vec<String> = results
            .iter()
            .filter(|r| !r.pair_id.is_empty())
            .map(|r| r.pair_id.clone())
            .collect();
        let _ = log_event(
            db,
            RetrievalEvent {
                user_id: params.user_id.clone(),
                kg_id: params.kg_id.clone(),
                session_id: params.session_id.clone(),
                request_path: params.request_path.clone(),
                query: params.query.clone(),
                query_embedding: params.embedding.clone(),
                retrieved_pair_ids: ids,
                weights,
                aux_features: Features {
                    query: params.query.clone(),
                    now: Some(Utc::now()),
                    current_session_id: params.current_session_id.clone(),
                    ..Features::default()
                },
            },
        )
        .await;
    }
    Ok(results)
}

/// One row of the candidate pool pulled by the vector query.
struct Candidate {
    /// Internal row id (`memory_pairs.id`), used for link lookups.
    row_id: String,
    /// Public pair id (`memory_pairs.firestore_pair_id`).
    pair_id: String,
    timestamp: DateTime<Utc>,
    session_id: String,
    cosine: f64,
}

/// Ports the Go `compositeSQLV1`/`compositeSQLV2` CTE queries: the candidate
/// pool comes from a Turso vector query; the freq / bounds / subject-match /
/// neighbor-density aggregates and the weighted score are computed in Rust.
async fn score_candidates(
    db: &Db,
    params: &CompositeParams,
    weights: Weights,
) -> Result<Vec<CompositeMemory>> {
    let conn = db.connection();
    if params.embedding.len() != crate::db::EMBEDDING_DIMS {
        return Err(Error::InvalidArgument(format!(
            "query embedding dimension = {}, want {}",
            params.embedding.len(),
            crate::db::EMBEDDING_DIMS
        )));
    }
    let blob = crate::db::encode_f32_blob(&params.embedding);
    let min_timestamp = params.min_timestamp.unwrap_or(DateTime::<Utc>::UNIX_EPOCH);

    // Candidate pool (Go: `candidates` CTE).
    let mut sql = String::from(
        "SELECT id, firestore_pair_id, timestamp, session_id,
                (1.0 - vector_distance_cos(conversation_embedding, ?)) AS cosine_sim
         FROM memory_pairs
         WHERE user_id = ? AND kg_id = ?
           AND (? = '' OR COALESCE(session_id, 'main') = COALESCE(NULLIF(?, ''), 'main'))
           AND timestamp >= ?
           AND conversation_embedding IS NOT NULL",
    );
    let mut args = vec![
        turso::Value::Blob(blob.clone()),
        turso::Value::Text(params.user_id.clone()),
        turso::Value::Text(params.kg_id.clone()),
        turso::Value::Text(params.session_id.clone()),
        turso::Value::Text(params.session_id.clone()),
        turso::Value::Text(crate::db::format_timestamp(min_timestamp)),
    ];
    if !params.exclude_pair_ids.is_empty() {
        sql.push_str(&format!(
            " AND firestore_pair_id NOT IN ({})",
            placeholders(params.exclude_pair_ids.len())
        ));
        args.extend(
            params
                .exclude_pair_ids
                .iter()
                .map(|id| turso::Value::Text(id.clone())),
        );
    }
    sql.push_str(" ORDER BY vector_distance_cos(conversation_embedding, ?) LIMIT ?");
    args.push(turso::Value::Blob(blob.clone()));
    args.push(turso::Value::Integer(params.candidate_pool_size as i64));

    let mut rows = conn.query(&sql, args).await?;
    let mut candidates: Vec<Candidate> = Vec::new();
    while let Some(row) = rows.next().await? {
        candidates.push(Candidate {
            row_id: get_text(&row, 0)?,
            pair_id: get_text(&row, 1)?,
            timestamp: crate::db::parse_timestamp(&get_text(&row, 2)?)?,
            session_id: get_opt_text(&row, 3)?.unwrap_or_default(),
            cosine: get_f64(&row, 4)?,
        });
    }
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    // Links for the candidate pool (basis for pair_freq / candidate_subjects).
    let candidate_row_ids: Vec<String> = candidates.iter().map(|c| c.row_id.clone()).collect();
    let sql = format!(
        "SELECT pair_id, subject_id FROM subject_memory_pair_links WHERE pair_id IN ({})",
        placeholders(candidate_row_ids.len())
    );
    let args: Vec<turso::Value> = candidate_row_ids
        .iter()
        .map(|id| turso::Value::Text(id.clone()))
        .collect();
    let mut rows = conn.query(&sql, args).await?;
    let mut pair_subjects: HashMap<String, Vec<String>> = HashMap::new();
    let mut subject_pairs: HashMap<String, Vec<String>> = HashMap::new();
    while let Some(row) = rows.next().await? {
        let pair_id = get_text(&row, 0)?;
        let subject_id = get_text(&row, 1)?;
        pair_subjects
            .entry(pair_id.clone())
            .or_default()
            .push(subject_id.clone());
        subject_pairs.entry(subject_id).or_default().push(pair_id);
    }

    // Per-subject global link counts (Go: `spc` subquery — counts over the
    // whole table, not just the candidate pool).
    let mut subject_link_counts: HashMap<String, i64> = HashMap::new();
    if !subject_pairs.is_empty() {
        let subject_ids: Vec<&String> = subject_pairs.keys().collect();
        let sql = format!(
            "SELECT subject_id, COUNT(*) FROM subject_memory_pair_links
             WHERE subject_id IN ({}) GROUP BY subject_id",
            placeholders(subject_ids.len())
        );
        let args: Vec<turso::Value> = subject_ids
            .iter()
            .map(|id| turso::Value::Text((*id).clone()))
            .collect();
        let mut rows = conn.query(&sql, args).await?;
        while let Some(row) = rows.next().await? {
            subject_link_counts.insert(get_text(&row, 0)?, get_i64(&row, 1)?);
        }
    }

    // pair_freq + max_freq (Go: `pair_freq` / `max_freq` CTEs).
    let mut pair_freq: HashMap<&str, f64> = HashMap::new();
    for (pair_id, subjects) in &pair_subjects {
        let total: i64 = subjects
            .iter()
            .map(|sid| subject_link_counts.get(sid).copied().unwrap_or(0))
            .sum();
        pair_freq.insert(pair_id.as_str(), total as f64);
    }
    let max_freq = pair_freq
        .values()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(1.0);

    // bounds (Go: `bounds` CTE).
    let oldest = candidates
        .iter()
        .map(|c| c.timestamp)
        .min()
        .unwrap_or(min_timestamp);
    let newest = candidates
        .iter()
        .map(|c| c.timestamp)
        .max()
        .unwrap_or(min_timestamp);
    let span_secs = features::duration_seconds(newest - oldest);

    // V2-only aggregates.
    let is_v2 = params.variant == Variant::V2;
    let mut subject_sem: HashMap<String, f64> = HashMap::new();
    let mut neighbor_density: HashMap<&str, f64> = HashMap::new();
    let mut density_max = 1.0f64;
    let now = Utc::now();
    if is_v2 {
        // subject_match (Go: `subject_match` CTE) — per-subject similarity,
        // folded into a per-pair MAX below.
        if !subject_pairs.is_empty() {
            let subject_ids: Vec<&String> = subject_pairs.keys().collect();
            let sql = format!(
                "SELECT id, (1.0 - vector_distance_cos(embedding, ?)) FROM subjects
                 WHERE embedding IS NOT NULL AND id IN ({})",
                placeholders(subject_ids.len())
            );
            let mut args = vec![turso::Value::Blob(blob)];
            args.extend(
                subject_ids
                    .iter()
                    .map(|id| turso::Value::Text((*id).clone())),
            );
            let mut rows = conn.query(&sql, args).await?;
            while let Some(row) = rows.next().await? {
                subject_sem.insert(get_text(&row, 0)?, get_f64(&row, 1)?);
            }
        }
        // neighbor_density (Go: `neighbor_density` CTE) — distinct candidate
        // pairs sharing at least one subject.
        for (pair_id, subjects) in &pair_subjects {
            let mut neighbors: std::collections::HashSet<&str> = std::collections::HashSet::new();
            for sid in subjects {
                if let Some(pairs) = subject_pairs.get(sid) {
                    for other in pairs {
                        if other != pair_id {
                            neighbors.insert(other.as_str());
                        }
                    }
                }
            }
            if !neighbors.is_empty() {
                neighbor_density.insert(pair_id.as_str(), neighbors.len() as f64);
            }
        }
        density_max = neighbor_density
            .values()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(1.0);
    }

    // Score (Go: final SELECT of compositeSQLV1/compositeSQLV2).
    let mut out: Vec<CompositeMemory> = Vec::with_capacity(candidates.len());
    for c in &candidates {
        let recency = if newest == oldest {
            1.0
        } else {
            features::duration_seconds(c.timestamp - oldest) / span_secs
        };
        let frequency = pair_freq.get(c.row_id.as_str()).copied().unwrap_or(0.0) / max_freq;
        let mut item = CompositeMemory {
            pair_id: c.pair_id.clone(),
            cosine_similarity: c.cosine,
            recency_score: recency,
            frequency_score: frequency,
            ..CompositeMemory::default()
        };
        if is_v2 {
            item.recency_exp =
                (-features::duration_seconds(now - c.timestamp) / V2_RECENCY_TAU_SECS).exp();
            item.subject_sem_match = pair_subjects
                .get(&c.row_id)
                .map(|subjects| {
                    subjects
                        .iter()
                        .filter_map(|sid| subject_sem.get(sid).copied())
                        .fold(f64::NEG_INFINITY, f64::max)
                })
                .filter(|v| v.is_finite())
                .unwrap_or(0.0);
            item.session_continuity = if params.current_session_id.is_empty() {
                0.0
            } else if c.session_id == params.current_session_id {
                1.0
            } else {
                0.0
            };
            item.neighbor_density = neighbor_density
                .get(c.row_id.as_str())
                .copied()
                .unwrap_or(0.0)
                / density_max;
            item.composite_score = weights.scale
                * (weights.cosine * item.cosine_similarity
                    + weights.recency_linear * item.recency_score
                    + weights.subject_frequency * item.frequency_score
                    + weights.recency_exp * item.recency_exp
                    + weights.subject_sem_match * item.subject_sem_match
                    + weights.session_continuity * item.session_continuity
                    + weights.neighbor_density * item.neighbor_density);
        } else {
            // V1 does NOT apply the scale factor (matches compositeSQLV1).
            item.composite_score = weights.cosine * item.cosine_similarity
                + weights.recency_linear * item.recency_score
                + weights.subject_frequency * item.frequency_score;
        }
        out.push(item);
    }

    // ORDER BY composite_score DESC, timestamp DESC, firestore_pair_id DESC.
    let ts_by_pair: HashMap<&str, DateTime<Utc>> = candidates
        .iter()
        .map(|c| (c.pair_id.as_str(), c.timestamp))
        .collect();
    out.sort_by(|a, b| {
        b.composite_score
            .total_cmp(&a.composite_score)
            .then_with(|| {
                let ta = ts_by_pair.get(a.pair_id.as_str());
                let tb = ts_by_pair.get(b.pair_id.as_str());
                tb.cmp(&ta)
            })
            .then_with(|| b.pair_id.cmp(&a.pair_id))
    });
    out.truncate(params.limit);
    Ok(out)
}

/// A retrieval event row for `retrieval_events` (Go: `LogEvent` args).
#[derive(Debug, Clone, Default)]
pub struct RetrievalEvent {
    pub user_id: String,
    pub kg_id: String,
    pub session_id: String,
    pub request_path: String,
    pub query: String,
    /// Empty -> NULL `query_embedding`.
    pub query_embedding: Vec<f32>,
    pub retrieved_pair_ids: Vec<String>,
    pub weights: Weights,
    pub aux_features: Features,
}

/// Inserts a row into `retrieval_events` (Go: `LogEvent`). `weights` and
/// `aux_features` are stored as JSON text; `retrieved_pair_ids` as a JSON
/// array of strings.
pub async fn log_event(db: &Db, event: RetrievalEvent) -> Result<()> {
    let weights_json = serde_json::to_string(&event.weights)?;
    let aux_json = serde_json::to_string(&event.aux_features)?;
    let ids_json = serde_json::to_string(&event.retrieved_pair_ids)?;
    let embedding = if event.query_embedding.is_empty() {
        turso::Value::Null
    } else {
        turso::Value::Blob(crate::db::encode_f32_blob(&event.query_embedding))
    };
    db.connection()
        .execute(
            "INSERT INTO retrieval_events (
                user_id, kg_id, session_id, request_path, query,
                query_embedding, retrieved_pair_ids, weights, aux_features
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            vec![
                turso::Value::Text(event.user_id),
                turso::Value::Text(event.kg_id),
                turso::Value::Text(event.session_id),
                turso::Value::Text(event.request_path),
                turso::Value::Text(event.query),
                embedding,
                turso::Value::Text(ids_json),
                turso::Value::Text(weights_json),
                turso::Value::Text(aux_json),
            ],
        )
        .await?;
    Ok(())
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

fn get_text(row: &turso::Row, idx: usize) -> Result<String> {
    match row.get_value(idx)? {
        turso::Value::Text(s) => Ok(s),
        other => Err(Error::Other(format!(
            "column {idx}: expected text, got {other:?}"
        ))),
    }
}

fn get_opt_text(row: &turso::Row, idx: usize) -> Result<Option<String>> {
    match row.get_value(idx)? {
        turso::Value::Null => Ok(None),
        turso::Value::Text(s) => Ok(Some(s)),
        other => Err(Error::Other(format!(
            "column {idx}: expected text or null, got {other:?}"
        ))),
    }
}

fn get_i64(row: &turso::Row, idx: usize) -> Result<i64> {
    match row.get_value(idx)? {
        turso::Value::Integer(v) => Ok(v),
        other => Err(Error::Other(format!(
            "column {idx}: expected integer, got {other:?}"
        ))),
    }
}

fn get_f64(row: &turso::Row, idx: usize) -> Result<f64> {
    match row.get_value(idx)? {
        turso::Value::Real(v) => Ok(v),
        turso::Value::Integer(v) => Ok(v as f64),
        other => Err(Error::Other(format!(
            "column {idx}: expected real, got {other:?}"
        ))),
    }
}

/// Names the dominant intent among the first three weights
/// (Go: `PredictedIntent`).
pub fn predicted_intent(w1: f64, w2: f64, w3: f64) -> &'static str {
    if w1 >= w2 && w1 >= w3 {
        return "semantic";
    }
    if w2 >= w1 && w2 >= w3 {
        return "temporal";
    }
    "frequency"
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{CreateMemoryPairParams, UpsertSubjectParams, EMBEDDING_DIMS};
    use chrono::TimeZone;

    /// Port of Go `ExampleStaticPredictor` + the zero-weight fallback.
    #[tokio::test]
    async fn static_predictor_predicts() {
        let predictor = StaticPredictor {
            weights: Weights {
                cosine: 0.7,
                recency_linear: 0.2,
                subject_frequency: 0.1,
                scale: 1.0,
                ..Weights::ZERO
            },
        };
        let weights = predictor
            .predict(&Features {
                query: "recent project preferences".to_string(),
                ..Features::default()
            })
            .await
            .expect("predict");
        assert_eq!(weights.cosine, 0.7);

        let fallback = StaticPredictor::default()
            .predict(&Features::default())
            .await
            .expect("predict");
        assert_eq!(fallback, default_weights());
    }

    #[test]
    fn predicted_intent_picks_dominant_weight() {
        assert_eq!(predicted_intent(0.5, 0.3, 0.2), "semantic");
        assert_eq!(predicted_intent(0.2, 0.5, 0.3), "temporal");
        assert_eq!(predicted_intent(0.1, 0.2, 0.7), "frequency");
        // Ties prefer semantic, then temporal (Go's >= ordering).
        assert_eq!(predicted_intent(0.4, 0.4, 0.2), "semantic");
    }

    #[test]
    fn weights_json_matches_go_tags() {
        let w = default_weights();
        let json = serde_json::to_value(w).expect("serialize");
        assert_eq!(json["cosine"], 0.65);
        assert_eq!(json["recencyLinear"], 0.2);
        assert_eq!(json["subjectFrequency"], 0.15);
        assert_eq!(json["scale"], 1.0);
        // omitempty zero-valued fields are skipped.
        assert!(json.get("recencyExp").is_none());
        assert!(json.get("neighborDensity").is_none());
    }

    fn axis_embedding(axes: &[usize]) -> Vec<f32> {
        let mut v = vec![0f32; EMBEDDING_DIMS];
        for &axis in axes {
            v[axis % EMBEDDING_DIMS] = 1.0;
        }
        v
    }

    fn pair_params(
        uid: &str,
        pair_id: &str,
        ts: &str,
        session: &str,
        embedding: &[usize],
    ) -> CreateMemoryPairParams {
        CreateMemoryPairParams {
            firestore_pair_id: pair_id.to_string(),
            user_id: uid.to_string(),
            kg_id: format!("user_memories_{uid}"),
            session_id: session.to_string(),
            title: String::new(),
            description: String::new(),
            prompt: format!("prompt {pair_id}"),
            response: format!("response {pair_id}"),
            input: "[]".to_string(),
            output: "[]".to_string(),
            source: String::new(),
            source_context: String::new(),
            timestamp: crate::db::parse_timestamp(ts).expect("parse test timestamp"),
            timezone_offset: 0,
            seed_memories: "null".to_string(),
            retrieval_metadata: "null".to_string(),
            conversation_embedding: Some(axis_embedding(embedding)),
        }
    }

    /// Seeds the scenario shared by the v1/v2 tests:
    /// - pair A: axis 0 (cosine 1 with the query), oldest, no session
    /// - pair B: axes 0+1 (cosine ~0.707), newest, no session
    /// - pair C: axis 1 (cosine 0), middle, session "thread-x"
    /// - subject s1 (axis-0 embedding) links A and B; s2 (no embedding) links B
    async fn seed_composite_fixture(db: &Db) {
        db.upsert_user("u1").await.expect("user");
        let a = db
            .create_memory_pair(pair_params(
                "u1",
                "pair-a",
                "2026-01-01T00:00:00Z",
                "",
                &[0],
            ))
            .await
            .expect("pair a");
        let b = db
            .create_memory_pair(pair_params(
                "u1",
                "pair-b",
                "2026-01-03T00:00:00Z",
                "",
                &[0, 1],
            ))
            .await
            .expect("pair b");
        db.create_memory_pair(pair_params(
            "u1",
            "pair-c",
            "2026-01-02T00:00:00Z",
            "thread-x",
            &[1],
        ))
        .await
        .expect("pair c");
        let s1 = db
            .upsert_subject(UpsertSubjectParams {
                user_id: "u1".to_string(),
                kg_id: "user_memories_u1".to_string(),
                subject_text: "subject-1".to_string(),
                description_text: String::new(),
                is_key_subject: false,
                embedding: Some(axis_embedding(&[0])),
            })
            .await
            .expect("subject 1");
        let s2 = db
            .upsert_subject(UpsertSubjectParams {
                user_id: "u1".to_string(),
                kg_id: "user_memories_u1".to_string(),
                subject_text: "subject-2".to_string(),
                description_text: String::new(),
                is_key_subject: false,
                embedding: None,
            })
            .await
            .expect("subject 2");
        for (sid, pid) in [(&s1.id, &a.id), (&s1.id, &b.id), (&s2.id, &b.id)] {
            db.link_subject_memory_pair(sid, pid, "u1", "user_memories_u1")
                .await
                .expect("link");
        }
    }

    fn base_params() -> CompositeParams {
        CompositeParams {
            embedding: axis_embedding(&[0]),
            user_id: "u1".to_string(),
            kg_id: "user_memories_u1".to_string(),
            ..CompositeParams::default()
        }
    }

    #[tokio::test]
    async fn composite_retrieve_v1_scores_and_orders() {
        let db = Db::open_memory().await.expect("open");
        seed_composite_fixture(&db).await;

        let results = composite_retrieve(&db, base_params()).await.expect("v1");
        let ids: Vec<&str> = results.iter().map(|r| r.pair_id.as_str()).collect();
        assert_eq!(ids, ["pair-b", "pair-a", "pair-c"]);

        // pair_freq: A = 2 (s1 count), B = 3 (s1 + s2), C = 0; max 3.
        // recency: A oldest = 0, B newest = 1, C middle = 0.5.
        let b = &results[0];
        assert!((b.cosine_similarity - 1.0 / 2f64.sqrt()).abs() < 1e-5);
        assert!((b.recency_score - 1.0).abs() < 1e-9);
        assert!((b.frequency_score - 1.0).abs() < 1e-9);
        let want_b = 0.65 * b.cosine_similarity + 0.20 + 0.15;
        assert!((b.composite_score - want_b).abs() < 1e-9);

        let a = &results[1];
        assert!((a.cosine_similarity - 1.0).abs() < 1e-5);
        assert!((a.recency_score - 0.0).abs() < 1e-9);
        assert!((a.frequency_score - 2.0 / 3.0).abs() < 1e-9);
        let want_a = 0.65 + 0.15 * (2.0 / 3.0);
        assert!((a.composite_score - want_a).abs() < 1e-5);

        let c = &results[2];
        assert!((c.recency_score - 0.5).abs() < 1e-9);
        assert_eq!(c.frequency_score, 0.0);
        // V1 leaves V2-only features zero.
        assert_eq!(b.recency_exp, 0.0);
        assert_eq!(b.subject_sem_match, 0.0);
        assert_eq!(b.neighbor_density, 0.0);
    }

    #[tokio::test]
    async fn composite_retrieve_v1_respects_limit_and_excludes() {
        let db = Db::open_memory().await.expect("open");
        seed_composite_fixture(&db).await;

        let mut params = base_params();
        params.exclude_pair_ids = vec!["pair-b".to_string()];
        params.limit = 1;
        let results = composite_retrieve(&db, params).await.expect("v1");
        let ids: Vec<&str> = results.iter().map(|r| r.pair_id.as_str()).collect();
        assert_eq!(ids, ["pair-a"]);
    }

    #[tokio::test]
    async fn composite_retrieve_v2_features_and_scale() {
        let db = Db::open_memory().await.expect("open");
        seed_composite_fixture(&db).await;

        let mut params = base_params();
        params.variant = Variant::V2;
        params.current_session_id = "thread-x".to_string();
        params.weights = Weights {
            cosine: 0.5,
            recency_linear: 0.2,
            subject_frequency: 0.1,
            recency_exp: 0.0,
            subject_sem_match: 0.1,
            session_continuity: 0.1,
            neighbor_density: 0.0,
            scale: 2.0,
        };
        let results = composite_retrieve(&db, params).await.expect("v2");
        let by_id: std::collections::HashMap<&str, &CompositeMemory> =
            results.iter().map(|r| (r.pair_id.as_str(), r)).collect();

        let a = by_id["pair-a"];
        let b = by_id["pair-b"];
        let c = by_id["pair-c"];

        // subject_match: s1 (axis 0) similarity 1 covers A and B; C has none.
        assert!((a.subject_sem_match - 1.0).abs() < 1e-5);
        assert!((b.subject_sem_match - 1.0).abs() < 1e-5);
        assert_eq!(c.subject_sem_match, 0.0);
        // session continuity only for pair C ("thread-x").
        assert_eq!(a.session_continuity, 0.0);
        assert_eq!(c.session_continuity, 1.0);
        // neighbor density: A and B share s1 (density 1 each, max 1); C none.
        assert!((a.neighbor_density - 1.0).abs() < 1e-9);
        assert!((b.neighbor_density - 1.0).abs() < 1e-9);
        assert_eq!(c.neighbor_density, 0.0);
        // recency_exp decays but stays positive.
        assert!(a.recency_exp > 0.0 && a.recency_exp < 1.0);

        // Composite applies the scale factor (x2) on top of the weighted sum.
        let want_a = 2.0
            * (0.5 * a.cosine_similarity
                + 0.2 * a.recency_score
                + 0.1 * a.frequency_score
                + 0.1 * a.subject_sem_match);
        assert!(
            (a.composite_score - want_a).abs() < 1e-6,
            "composite = {}, want {want_a}",
            a.composite_score
        );
        let want_c =
            2.0 * (0.5 * c.cosine_similarity + 0.2 * c.recency_score + 0.1 * c.session_continuity);
        assert!((c.composite_score - want_c).abs() < 1e-6);
    }

    #[tokio::test]
    async fn composite_retrieve_logs_event_when_requested() {
        let db = Db::open_memory().await.expect("open");
        seed_composite_fixture(&db).await;

        let mut params = base_params();
        params.session_id = String::new();
        params.current_session_id = "thread-x".to_string();
        params.request_path = "/test/composite".to_string();
        params.query = "axis zero things".to_string();
        params.log_event = true;
        let results = composite_retrieve(&db, params).await.expect("retrieve");
        assert_eq!(results.len(), 3);

        let mut rows = db
            .connection()
            .query(
                "SELECT user_id, request_path, retrieved_pair_ids, weights, aux_features,
                        query_embedding IS NOT NULL
                 FROM retrieval_events",
                (),
            )
            .await
            .expect("select events");
        let row = rows
            .next()
            .await
            .expect("next")
            .expect("one retrieval event row");
        assert_eq!(get_text(&row, 0).expect("user_id"), "u1");
        assert_eq!(get_text(&row, 1).expect("path"), "/test/composite");
        let ids: Vec<String> =
            serde_json::from_str(&get_text(&row, 2).expect("ids")).expect("ids json");
        assert_eq!(ids, ["pair-b", "pair-a", "pair-c"]);
        let weights: Weights =
            serde_json::from_str(&get_text(&row, 3).expect("weights")).expect("weights json");
        assert_eq!(weights, default_weights());
        let aux: serde_json::Value =
            serde_json::from_str(&get_text(&row, 4).expect("aux")).expect("aux json");
        assert_eq!(aux["query"], "axis zero things");
        assert_eq!(aux["currentSessionId"], "thread-x");
        assert!(aux.get("now").is_some());
        assert_eq!(get_i64(&row, 5).expect("embedding flag"), 1);
        assert!(rows.next().await.expect("end").is_none());
    }

    #[tokio::test]
    async fn log_event_with_empty_embedding_stores_null() {
        let db = Db::open_memory().await.expect("open");
        log_event(
            &db,
            RetrievalEvent {
                user_id: "u1".to_string(),
                kg_id: "kg".to_string(),
                // Go's zero-value Weights{} (Rust `Weights::default()` is
                // DefaultWeights, so zero must be explicit).
                weights: Weights::ZERO,
                ..RetrievalEvent::default()
            },
        )
        .await
        .expect("log event");
        let mut rows = db
            .connection()
            .query(
                "SELECT query_embedding IS NULL, retrieved_pair_ids, weights FROM retrieval_events",
                (),
            )
            .await
            .expect("select");
        let row = rows.next().await.expect("next").expect("row");
        assert_eq!(get_i64(&row, 0).expect("null flag"), 1);
        assert_eq!(get_text(&row, 1).expect("ids"), "[]");
        // Zero weights serialize to required-field JSON only.
        let weights: serde_json::Value =
            serde_json::from_str(&get_text(&row, 2).expect("weights")).expect("json");
        assert_eq!(weights["cosine"], 0.0);
    }

    #[tokio::test]
    async fn composite_retrieve_empty_pool_returns_empty() {
        let db = Db::open_memory().await.expect("open");
        db.upsert_user("u1").await.expect("user");
        let results = composite_retrieve(&db, base_params()).await.expect("empty");
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn composite_retrieve_session_filter() {
        let db = Db::open_memory().await.expect("open");
        seed_composite_fixture(&db).await;
        let mut params = base_params();
        params.session_id = "thread-x".to_string();
        let results = composite_retrieve(&db, params).await.expect("session");
        let ids: Vec<&str> = results.iter().map(|r| r.pair_id.as_str()).collect();
        assert_eq!(ids, ["pair-c"]);
    }

    #[tokio::test]
    async fn composite_retrieve_min_timestamp_filters() {
        let db = Db::open_memory().await.expect("open");
        seed_composite_fixture(&db).await;
        let mut params = base_params();
        params.min_timestamp = Some(
            Utc.with_ymd_and_hms(2026, 1, 2, 0, 0, 0)
                .single()
                .expect("ts"),
        );
        let results = composite_retrieve(&db, params).await.expect("filtered");
        let mut ids: Vec<&str> = results.iter().map(|r| r.pair_id.as_str()).collect();
        ids.sort_unstable();
        assert_eq!(ids, ["pair-b", "pair-c"]);
    }
}
