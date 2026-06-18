// SPDX-License-Identifier: AGPL-3.0-or-later
//! Memory store: ingestion, fetch, vector search, composite search, subject
//! search, subject-scoped memory search, prompt context, and slim payloads.
//! Port of Go `pkg/memory`.

pub mod prompt_context;
pub mod slim;
pub mod tools;

use std::collections::HashSet;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::db::{
    Db, MemoryPairRow, SearchMemoriesBySubjectParams, SubjectRow, UpsertSubjectParams,
};
use crate::retrieval::{self, Variant, WeightPredictor};
use crate::types::{
    EmbedRequest, EmbedResponse, Embedder, Error, Memory, Result, RetrievalMetadata, Subject,
};

pub use prompt_context::{
    build_prompt_long_term_json, prompt_memory_title, resolve_prompt_session_id, seed_memory_nodes,
    summarize_prompt_memories, PromptMemoryRequest, PromptMemoryResponse, PromptMemorySummary,
    PromptMemoryValue,
};
pub use slim::{
    slim_previews, slim_truncated, to_slim_memory, to_slim_memory_preview,
    to_slim_memory_truncated, SlimMemory, DEFAULT_FETCH_MAX_BYTES, DEFAULT_PREVIEW_LEN,
};
pub use tools::{memory_tools, memory_tools_with, ToolOptions};

/// Default result limit for searches (Go hardcodes 8).
pub const DEFAULT_SEARCH_LIMIT: usize = 8;
/// Default minimum similarity for memory searches (Go: 0.15).
pub const DEFAULT_MIN_SIMILARITY: f64 = 0.15;
/// Default minimum similarity for subject searches (Go: 0.10).
pub const DEFAULT_SUBJECT_MIN_SIMILARITY: f64 = 0.10;

/// Memory store backed by [`Db`] plus an [`Embedder`] and optional learned
/// [`WeightPredictor`] (Go: `memory.Store`). Cheap to clone.
#[derive(Clone)]
pub struct Store {
    db: Arc<Db>,
    embedder: Arc<dyn Embedder>,
    predictor: Option<Arc<dyn WeightPredictor>>,
    reranker: Option<Arc<dyn retrieval::Reranker>>,
}

/// Constructor options for [`Store::new`] (Go: `memory.Options`).
#[derive(Clone)]
pub struct StoreOptions {
    pub db: Arc<Db>,
    pub embedder: Arc<dyn Embedder>,
    /// `None` -> default weights + "semantic" intent in composite search.
    pub predictor: Option<Arc<dyn WeightPredictor>>,
    /// Optional second-stage reranker applied to the composite pool. `None` ->
    /// composite order is returned as-is (Go production: cross-encoder rerank).
    pub reranker: Option<Arc<dyn retrieval::Reranker>>,
}

/// Subject attached to a saved memory (Go: `SubjectInput`). Tool args use
/// lowercase keys; Go-marshaled payloads (`Text`, ...) are accepted as
/// aliases.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SubjectInput {
    #[serde(default, alias = "Text")]
    pub text: String,
    #[serde(default, alias = "Description")]
    pub description: String,
    #[serde(default, alias = "Key")]
    pub key: bool,
}

/// Request for [`Store::save_memory`] (Go: `SaveMemoryRequest`).
/// Defaults applied by the store: empty `kg_id` -> `kg_id(user_id)`, empty
/// `id` -> new uuid v4, `timestamp: None` -> now (UTC).
#[derive(Debug, Clone, Default)]
pub struct SaveMemoryRequest {
    pub user_id: String,
    pub kg_id: String,
    pub session_id: String,
    /// Public pair id; empty -> generated uuid v4.
    pub id: String,
    pub title: String,
    pub summary: String,
    pub prompt: String,
    pub response: String,
    pub input: Vec<crate::types::Content>,
    pub output: Vec<crate::types::Content>,
    pub source: String,
    pub source_context: String,
    pub timestamp: Option<DateTime<Utc>>,
    pub timezone_offset: i32,
    pub seed_memories: Vec<crate::types::SeedMemoryNode>,
    pub retrieval_metadata: Option<RetrievalMetadata>,
    pub subjects: Vec<SubjectInput>,
}

/// Request for [`Store::search_memories`] (Go: `SearchMemoriesRequest`).
/// Defaults: `limit` 0 -> 8, `min_similarity` 0 -> 0.15, empty `kg_id` ->
/// derived.
#[derive(Debug, Clone, Default)]
pub struct SearchMemoriesRequest {
    pub user_id: String,
    pub kg_id: String,
    /// Empty searches all sessions.
    pub session_id: String,
    pub queries: Vec<String>,
    pub limit: usize,
    pub min_similarity: f64,
    pub exclude_pair_ids: Vec<String>,
}

/// Request for [`Store::search_composite_memories`]
/// (Go: `CompositeSearchRequest`). Defaults: `limit` 0 -> 8,
/// `candidate_pool_size` 0 -> `max(32, limit*4)`, `variant` default Legacy.
#[derive(Debug, Clone, Default)]
pub struct CompositeSearchRequest {
    pub user_id: String,
    pub kg_id: String,
    pub session_id: String,
    pub query: String,
    pub limit: usize,
    pub candidate_pool_size: usize,
    pub exclude_pair_ids: Vec<String>,
    pub variant: Variant,
    pub request_path: String,
    pub log_event: bool,
}

/// Request for [`Store::search_subjects`] (Go: `SearchSubjectsRequest`).
/// Defaults: `limit` 0 -> 8, `min_similarity` 0 -> 0.10.
#[derive(Debug, Clone, Default)]
pub struct SearchSubjectsRequest {
    pub user_id: String,
    pub kg_id: String,
    pub queries: Vec<String>,
    pub limit: usize,
    pub min_similarity: f64,
}

/// One subject-scoped query (Go: `SubjectMemoryQuery`). JSON tags match Go:
/// `subject_id`, `query`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SubjectMemoryQuery {
    pub subject_id: String,
    pub query: String,
}

/// Request for [`Store::search_memories_in_subjects`]
/// (Go: `SearchMemoriesInSubjectsRequest`). Defaults: `limit` 0 -> 8,
/// `min_similarity` 0 -> 0.15.
#[derive(Debug, Clone, Default)]
pub struct SearchMemoriesInSubjectsRequest {
    pub user_id: String,
    pub queries: Vec<SubjectMemoryQuery>,
    pub limit: usize,
    pub min_similarity: f64,
}

/// Request for [`Store::fetch_memories`] (Go: `FetchMemoriesRequest`).
#[derive(Debug, Clone, Default)]
pub struct FetchMemoriesRequest {
    pub user_id: String,
    /// Public pair ids; result order preserves this order.
    pub pair_ids: Vec<String>,
}

/// Request for [`Store::list_recent_memories`]
/// (Go: `ListRecentMemoriesRequest`). `limit <= 0` returns an empty list;
/// empty `session_id` resolves to "main".
#[derive(Debug, Clone, Default)]
pub struct ListRecentMemoriesRequest {
    pub user_id: String,
    pub kg_id: String,
    pub session_id: String,
    pub limit: usize,
    pub exclude_pair_ids: Vec<String>,
}

impl Store {
    /// Creates a store (Go: `memory.NewStore`).
    pub fn new(opts: StoreOptions) -> Store {
        Store {
            db: opts.db,
            embedder: opts.embedder,
            predictor: opts.predictor,
            reranker: opts.reranker,
        }
    }

    /// Raw db access for sibling modules (chat/dream/retrieval glue).
    pub fn db(&self) -> &Db {
        &self.db
    }

    /// Saves a memory pair: embeds `prompt\nresponse\nsummary`, upserts the
    /// user and pair, then embeds/upserts/links each non-empty subject.
    /// Returns the stored memory (Go: `Store.SaveMemory`). Errors if
    /// `user_id` is blank.
    pub async fn save_memory(&self, mut req: SaveMemoryRequest) -> Result<Memory> {
        if req.user_id.trim().is_empty() {
            return Err(Error::InvalidArgument(
                "memory: user id is required".to_string(),
            ));
        }
        if req.kg_id.is_empty() {
            req.kg_id = crate::types::kg_id(&req.user_id);
        }
        if req.id.is_empty() {
            req.id = crate::db::new_row_id();
        }
        let timestamp = req.timestamp.unwrap_or_else(Utc::now);

        let text = format!("{}\n{}\n{}", req.prompt, req.response, req.summary)
            .trim()
            .to_string();
        let embed_resp = self.embed_texts(std::slice::from_ref(&text)).await?;
        let embedding = embedding_at(&embed_resp, 0);

        let input_json = serde_json::to_string(&req.input)?;
        let output_json = serde_json::to_string(&req.output)?;
        let seed_json = serde_json::to_string(&req.seed_memories)?;
        let metadata_json = serde_json::to_string(&req.retrieval_metadata)?;

        self.db.upsert_user(&req.user_id).await?;
        let row = self
            .db
            .create_memory_pair(crate::db::CreateMemoryPairParams {
                firestore_pair_id: req.id.clone(),
                user_id: req.user_id.clone(),
                kg_id: req.kg_id.clone(),
                session_id: req.session_id.clone(),
                title: req.title.clone(),
                description: req.summary.clone(),
                prompt: req.prompt.clone(),
                response: req.response.clone(),
                input: input_json,
                output: output_json,
                source: req.source.clone(),
                source_context: req.source_context.clone(),
                timestamp,
                timezone_offset: req.timezone_offset,
                seed_memories: seed_json,
                retrieval_metadata: metadata_json,
                conversation_embedding: embedding,
            })
            .await?;

        if !req.subjects.is_empty() {
            let subject_texts: Vec<String> = req
                .subjects
                .iter()
                .map(|subj| {
                    format!("{}\n{}", subj.text, subj.description)
                        .trim()
                        .to_string()
                })
                .collect();
            let subject_embeddings = self.embed_texts(&subject_texts).await?;
            for (i, subj) in req.subjects.iter().enumerate() {
                if subj.text.trim().is_empty() {
                    continue;
                }
                let srow = self
                    .db
                    .upsert_subject(UpsertSubjectParams {
                        user_id: req.user_id.clone(),
                        kg_id: req.kg_id.clone(),
                        subject_text: subj.text.clone(),
                        description_text: subj.description.clone(),
                        is_key_subject: subj.key,
                        embedding: embedding_at(&subject_embeddings, i),
                    })
                    .await?;
                self.db
                    .link_subject_memory_pair(&srow.id, &row.id, &req.user_id, &req.kg_id)
                    .await?;
            }
        }

        Ok(memory_from_row(row))
    }

    /// Vector search across one or more queries, de-duplicating results by
    /// pair id (excluded ids are pre-seeded) and tagging `similarity`
    /// (Go: `Store.SearchMemories`).
    pub async fn search_memories(&self, mut req: SearchMemoriesRequest) -> Result<Vec<Memory>> {
        if req.kg_id.is_empty() {
            req.kg_id = crate::types::kg_id(&req.user_id);
        }
        if req.limit == 0 {
            req.limit = DEFAULT_SEARCH_LIMIT;
        }
        if req.min_similarity == 0.0 {
            req.min_similarity = DEFAULT_MIN_SIMILARITY;
        }

        let embeddings = self.embed_texts(&req.queries).await?;
        let mut seen: HashSet<String> = req.exclude_pair_ids.iter().cloned().collect();
        let mut out = Vec::new();
        for embedding in &embeddings.embeddings {
            let rows = self
                .db
                .search_memories(crate::db::SearchMemoriesParams {
                    embedding: embedding.clone(),
                    user_id: req.user_id.clone(),
                    kg_id: req.kg_id.clone(),
                    session_id: req.session_id.clone(),
                    exclude_pair_ids: seen.iter().cloned().collect(),
                    min_similarity: req.min_similarity,
                    limit: req.limit as i64,
                })
                .await?;
            for row in rows {
                let mem = memory_from_row(row);
                if !seen.insert(mem.id.clone()) {
                    continue;
                }
                out.push(mem);
            }
        }
        Ok(out)
    }

    /// Composite (cosine + recency + frequency [+ V2 features]) search.
    /// Uses the predictor when present ("learned" intent) else default
    /// weights ("semantic"). Returns scored memories ordered by composite
    /// score plus the retrieval metadata describing weights/variant/pair ids
    /// (Go: `Store.SearchCompositeMemories`; metadata
    /// `query_embedding_model` is "host").
    pub async fn search_composite_memories(
        &self,
        mut req: CompositeSearchRequest,
    ) -> Result<(Vec<Memory>, Option<RetrievalMetadata>)> {
        if req.kg_id.is_empty() {
            req.kg_id = crate::types::kg_id(&req.user_id);
        }
        if req.limit == 0 {
            req.limit = DEFAULT_SEARCH_LIMIT;
        }
        // When a reranker is set, retrieve a WIDER composite pool, rerank it,
        // then truncate to the caller's limit (Go production: retrieveLimit =
        // max(rootCount, ceRerankPoolSize)).
        let requested_limit = req.limit;
        let pool_limit = if self.reranker.is_some() {
            retrieval::RERANK_POOL_SIZE.max(requested_limit)
        } else {
            requested_limit
        };
        if req.candidate_pool_size == 0 {
            req.candidate_pool_size = 32.max(pool_limit * 4);
        }
        let embed_resp = self.embed_texts(std::slice::from_ref(&req.query)).await?;
        let embedding = embedding_at(&embed_resp, 0).unwrap_or_default();

        let mut weights = retrieval::default_weights();
        let mut intent = "semantic";
        if let Some(predictor) = &self.predictor {
            weights = predictor
                .predict(&retrieval::Features {
                    query: req.query.clone(),
                    now: Some(Utc::now()),
                    query_embedding: embedding.clone(),
                    current_session_id: req.session_id.clone(),
                    ..retrieval::Features::default()
                })
                .await?;
            intent = "learned";
        }

        let results = retrieval::composite_retrieve(
            &self.db,
            retrieval::CompositeParams {
                embedding,
                user_id: req.user_id.clone(),
                kg_id: req.kg_id.clone(),
                session_id: req.session_id.clone(),
                current_session_id: req.session_id.clone(),
                min_timestamp: None,
                limit: pool_limit,
                candidate_pool_size: req.candidate_pool_size,
                exclude_pair_ids: req.exclude_pair_ids.clone(),
                weights,
                variant: req.variant,
                request_path: req.request_path.clone(),
                query: req.query.clone(),
                log_event: req.log_event,
            },
        )
        .await?;

        let pair_ids: Vec<String> = results.iter().map(|r| r.pair_id.clone()).collect();
        let score_by_id: std::collections::HashMap<&str, &retrieval::CompositeMemory> =
            results.iter().map(|r| (r.pair_id.as_str(), r)).collect();
        let mut memories = self
            .fetch_memories(FetchMemoriesRequest {
                user_id: req.user_id.clone(),
                pair_ids: pair_ids.clone(),
            })
            .await?;
        for mem in &mut memories {
            if let Some(score) = score_by_id.get(mem.id.as_str()) {
                mem.similarity = score.cosine_similarity;
                mem.recency_score = score.recency_score;
                mem.frequency_score = score.frequency_score;
                mem.composite_score = score.composite_score;
                mem.recency_exp = score.recency_exp;
                mem.subject_sem_match = score.subject_sem_match;
                mem.session_continuity = score.session_continuity;
                mem.neighbor_density = score.neighbor_density;
            }
        }

        // Second stage: rerank the composite pool, then truncate to the
        // caller's limit (Go production: cross-encoder rerank + RRF fusion).
        // `memories` arrive ordered best-first by composite score.
        if let Some(reranker) = &self.reranker {
            memories = reranker
                .rerank(&req.query, memories, requested_limit)
                .await?;
        }
        let retrieved_pair_ids: Vec<String> = memories.iter().map(|m| m.id.clone()).collect();

        let mut weight_map = std::collections::BTreeMap::new();
        weight_map.insert("cosine".to_string(), weights.cosine);
        weight_map.insert("recencyLinear".to_string(), weights.recency_linear);
        weight_map.insert("recencyExp".to_string(), weights.recency_exp);
        weight_map.insert("subjectFrequency".to_string(), weights.subject_frequency);
        weight_map.insert("subjectSemMatch".to_string(), weights.subject_sem_match);
        weight_map.insert("sessionContinuity".to_string(), weights.session_continuity);
        weight_map.insert("neighborDensity".to_string(), weights.neighbor_density);
        weight_map.insert("scale".to_string(), weights.scale);
        let metadata = RetrievalMetadata {
            intent: intent.to_string(),
            weights: weight_map,
            scale: weights.scale,
            variant: req.variant.to_string(),
            retrieved_pair_ids,
            query_embedding_model: "host".to_string(),
        };
        Ok((memories, Some(metadata)))
    }

    /// Vector search over the subject graph, de-duplicated across queries
    /// (Go: `Store.SearchSubjects`).
    pub async fn search_subjects(&self, mut req: SearchSubjectsRequest) -> Result<Vec<Subject>> {
        if req.kg_id.is_empty() {
            req.kg_id = crate::types::kg_id(&req.user_id);
        }
        if req.limit == 0 {
            req.limit = DEFAULT_SEARCH_LIMIT;
        }
        if req.min_similarity == 0.0 {
            req.min_similarity = DEFAULT_SUBJECT_MIN_SIMILARITY;
        }
        let embeddings = self.embed_texts(&req.queries).await?;
        let mut seen: HashSet<String> = HashSet::new();
        let mut out = Vec::new();
        for embedding in &embeddings.embeddings {
            let rows = self
                .db
                .search_subjects(crate::db::SearchSubjectsParams {
                    embedding: embedding.clone(),
                    user_id: req.user_id.clone(),
                    kg_id: req.kg_id.clone(),
                    min_similarity: req.min_similarity,
                    limit: req.limit as i64,
                })
                .await?;
            for row in rows {
                let subj = subject_from_row(row);
                if !seen.insert(subj.id.clone()) {
                    continue;
                }
                out.push(subj);
            }
        }
        Ok(out)
    }

    /// Vector search restricted to specific subject ids, de-duplicated
    /// (Go: `Store.SearchMemoriesInSubjects`). Errors on invalid subject ids.
    pub async fn search_memories_in_subjects(
        &self,
        mut req: SearchMemoriesInSubjectsRequest,
    ) -> Result<Vec<Memory>> {
        if req.limit == 0 {
            req.limit = DEFAULT_SEARCH_LIMIT;
        }
        if req.min_similarity == 0.0 {
            req.min_similarity = DEFAULT_MIN_SIMILARITY;
        }
        let texts: Vec<String> = req.queries.iter().map(|q| q.query.clone()).collect();
        let embeddings = self.embed_texts(&texts).await?;
        let mut seen: HashSet<String> = HashSet::new();
        let mut out = Vec::new();
        for (i, query) in req.queries.iter().enumerate() {
            uuid::Uuid::parse_str(&query.subject_id).map_err(|err| {
                Error::InvalidArgument(format!("parse subject id {:?}: {err}", query.subject_id))
            })?;
            // A missing embedding (blank query dropped by embed_texts) matches
            // nothing — same outcome as Go's NULL query vector.
            let Some(embedding) = embedding_at(&embeddings, i) else {
                continue;
            };
            let rows = self
                .db
                .search_memories_by_subject(SearchMemoriesBySubjectParams {
                    embedding,
                    subject_id: query.subject_id.clone(),
                    user_id: req.user_id.clone(),
                    min_similarity: req.min_similarity,
                    limit: req.limit as i64,
                })
                .await?;
            for row in rows {
                let mem = memory_from_row(row);
                if !seen.insert(mem.id.clone()) {
                    continue;
                }
                out.push(mem);
            }
        }
        Ok(out)
    }

    /// Fetches full memories for the given public pair ids, preserving input
    /// order (Go: `Store.FetchMemories`).
    pub async fn fetch_memories(&self, req: FetchMemoriesRequest) -> Result<Vec<Memory>> {
        let rows = self.db.fetch_memories(&req.user_id, &req.pair_ids).await?;
        Ok(rows.into_iter().map(memory_from_row).collect())
    }

    /// Lists the newest memories in a session (Go: `Store.ListRecentMemories`).
    pub async fn list_recent_memories(
        &self,
        mut req: ListRecentMemoriesRequest,
    ) -> Result<Vec<Memory>> {
        if req.kg_id.is_empty() {
            req.kg_id = crate::types::kg_id(&req.user_id);
        }
        if req.session_id.is_empty() {
            req.session_id = crate::types::MAIN_SESSION_ID.to_string();
        }
        if req.limit == 0 {
            return Ok(Vec::new());
        }
        let rows = self
            .db
            .list_recent_memories(crate::db::ListRecentMemoriesParams {
                user_id: req.user_id,
                kg_id: req.kg_id,
                session_id: req.session_id,
                exclude_pair_ids: req.exclude_pair_ids,
                limit: req.limit as i64,
            })
            .await?;
        Ok(rows.into_iter().map(memory_from_row).collect())
    }

    /// Drops blank texts, errors when nothing embeddable remains, and calls
    /// the embedder (Go: `Store.embedTexts`).
    pub(crate) async fn embed_texts(&self, texts: &[String]) -> Result<EmbedResponse> {
        let clean: Vec<String> = texts
            .iter()
            .filter(|text| !text.trim().is_empty())
            .cloned()
            .collect();
        if clean.is_empty() {
            return Err(Error::InvalidArgument(
                "memory: at least one non-empty query is required".to_string(),
            ));
        }
        self.embedder.embed(EmbedRequest { texts: clean }).await
    }
}

/// Embedding at `idx`, `None` when out of range (Go: `embeddingAt`).
fn embedding_at(resp: &EmbedResponse, idx: usize) -> Option<Vec<f32>> {
    resp.embeddings.get(idx).cloned()
}

/// Maps a db row to the public [`Memory`] shape (Go: `memoryFromRow`); JSON
/// columns that fail to parse degrade to empty values, matching Go's ignored
/// `json.Unmarshal` errors. `similarity` carries through from search rows.
pub(crate) fn memory_from_row(row: MemoryPairRow) -> Memory {
    let input = row
        .input
        .as_deref()
        .and_then(|raw| serde_json::from_str(raw).ok())
        .unwrap_or_default();
    let output = row
        .output
        .as_deref()
        .and_then(|raw| serde_json::from_str(raw).ok())
        .unwrap_or_default();
    let seed_memories = row
        .seed_memories
        .as_deref()
        .and_then(|raw| serde_json::from_str(raw).ok())
        .unwrap_or_default();
    let retrieval_metadata = row
        .retrieval_metadata
        .as_deref()
        .filter(|raw| !raw.is_empty() && *raw != "null")
        .and_then(|raw| serde_json::from_str::<RetrievalMetadata>(raw).ok());
    Memory {
        id: row.firestore_pair_id,
        source_pair_id: row.id,
        user_id: row.user_id,
        kg_id: row.kg_id,
        session_id: row.session_id.unwrap_or_default(),
        title: row.title.unwrap_or_default(),
        summary: row.description.unwrap_or_default(),
        prompt: row.prompt.unwrap_or_default(),
        response: row.response.unwrap_or_default(),
        input,
        output,
        source: row.source.unwrap_or_default(),
        source_context: row.source_context.unwrap_or_default(),
        timestamp: row.timestamp,
        timezone_offset: row.timezone_offset.unwrap_or_default(),
        seed_memories,
        retrieval_metadata,
        embedding: row.conversation_embedding.unwrap_or_default(),
        similarity: row.similarity,
        ..Memory::default()
    }
}

/// Maps a subject row to the public [`Subject`] shape (Go: `subjectFromSearch`).
fn subject_from_row(row: SubjectRow) -> Subject {
    Subject {
        id: row.id,
        user_id: row.user_id,
        kg_id: row.kg_id,
        text: row.subject_text,
        description: row.description_text.unwrap_or_default(),
        key: row.is_key_subject,
        embedding: row.embedding.unwrap_or_default(),
        similarity: row.similarity,
        memory_count: row.memory_count,
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    //! Shared fakes for memory tests (port of Go's `HashEmbedder`).

    use std::hash::{Hash, Hasher};
    use std::sync::Arc;

    use async_trait::async_trait;

    use super::{Store, StoreOptions};
    use crate::db::{Db, EMBEDDING_DIMS};
    use crate::types::{EmbedRequest, EmbedResponse, Embedder, Result};

    /// Deterministic bag-of-words embedder: each lowercase whitespace token
    /// bumps one hashed dimension, then the vector is L2-normalized.
    pub struct HashEmbedder;

    #[async_trait]
    impl Embedder for HashEmbedder {
        async fn embed(&self, req: EmbedRequest) -> Result<EmbedResponse> {
            Ok(EmbedResponse {
                embeddings: req.texts.iter().map(|t| hash_embedding(t)).collect(),
                ..EmbedResponse::default()
            })
        }
    }

    pub fn hash_embedding(text: &str) -> Vec<f32> {
        let mut vec = vec![0f32; EMBEDDING_DIMS];
        for token in text.to_lowercase().split_whitespace() {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            token.hash(&mut hasher);
            let idx = (hasher.finish() % EMBEDDING_DIMS as u64) as usize;
            vec[idx] += 1.0;
        }
        let norm: f64 = vec.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        if norm == 0.0 {
            vec[0] = 1.0;
            return vec;
        }
        let scale = (1.0 / norm.sqrt()) as f32;
        for v in &mut vec {
            *v *= scale;
        }
        vec
    }

    /// In-memory store wired to the hash embedder.
    pub async fn new_test_store() -> Store {
        let db = Db::open_memory().await.expect("open in-memory turso db");
        Store::new(StoreOptions {
            db: Arc::new(db),
            embedder: Arc::new(HashEmbedder),
            predictor: None,
            reranker: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::new_test_store;
    use super::*;
    use crate::types::Content;

    /// Port of Go `TestStoreSaveSearchFetchAndSubjects`.
    #[tokio::test]
    async fn store_save_search_fetch_and_subjects() {
        let store = new_test_store().await;

        let saved = store
            .save_memory(SaveMemoryRequest {
                user_id: "user-test".to_string(),
                prompt: "Remember that Peyton prefers direct engineering updates.".to_string(),
                response: "Use concise status notes and concrete file references.".to_string(),
                summary: "Peyton prefers direct engineering updates.".to_string(),
                input: vec![Content::text("direct updates")],
                output: vec![Content::text("concrete file references")],
                subjects: vec![SubjectInput {
                    text: "Engineering communication".to_string(),
                    description: "Preferences about direct updates".to_string(),
                    key: true,
                }],
                ..SaveMemoryRequest::default()
            })
            .await
            .expect("save_memory");
        assert!(!saved.id.is_empty(), "save_memory returned empty id");

        let memories = store
            .search_memories(SearchMemoriesRequest {
                user_id: "user-test".to_string(),
                queries: vec!["direct engineering updates".to_string()],
                limit: 5,
                ..SearchMemoriesRequest::default()
            })
            .await
            .expect("search_memories");
        assert_eq!(memories.len(), 1, "memories: {memories:?}");
        assert_eq!(memories[0].id, saved.id);
        assert!(memories[0].similarity > 0.0);

        let subjects = store
            .search_subjects(SearchSubjectsRequest {
                user_id: "user-test".to_string(),
                queries: vec!["engineering communication".to_string()],
                limit: 5,
                ..SearchSubjectsRequest::default()
            })
            .await
            .expect("search_subjects");
        assert_eq!(subjects.len(), 1, "subjects: {subjects:?}");
        assert_eq!(subjects[0].memory_count, 1);
        assert!(subjects[0].key);

        let in_subject = store
            .search_memories_in_subjects(SearchMemoriesInSubjectsRequest {
                user_id: "user-test".to_string(),
                queries: vec![SubjectMemoryQuery {
                    subject_id: subjects[0].id.clone(),
                    query: "concrete file references".to_string(),
                }],
                ..SearchMemoriesInSubjectsRequest::default()
            })
            .await
            .expect("search_memories_in_subjects");
        assert_eq!(in_subject.len(), 1, "in_subject: {in_subject:?}");
        assert_eq!(in_subject[0].id, saved.id);

        let fetched = store
            .fetch_memories(FetchMemoriesRequest {
                user_id: "user-test".to_string(),
                pair_ids: vec![saved.id.clone()],
            })
            .await
            .expect("fetch_memories");
        assert_eq!(fetched.len(), 1);
        assert!(!fetched[0].prompt.is_empty(), "prompt round-trips");
        assert_eq!(fetched[0].input.len(), 1, "input content round-trips");
    }

    #[tokio::test]
    async fn save_memory_requires_user_id() {
        let store = new_test_store().await;
        let err = store
            .save_memory(SaveMemoryRequest {
                prompt: "text".to_string(),
                ..SaveMemoryRequest::default()
            })
            .await
            .expect_err("blank user id must fail");
        assert!(err.to_string().contains("user id is required"), "{err}");
    }

    #[tokio::test]
    async fn search_memories_requires_a_non_blank_query() {
        let store = new_test_store().await;
        let err = store
            .search_memories(SearchMemoriesRequest {
                user_id: "u".to_string(),
                queries: vec!["  ".to_string()],
                ..SearchMemoriesRequest::default()
            })
            .await
            .expect_err("blank queries must fail");
        assert!(err.to_string().contains("non-empty query"), "{err}");
    }

    #[tokio::test]
    async fn search_memories_in_subjects_rejects_invalid_subject_id() {
        let store = new_test_store().await;
        let err = store
            .search_memories_in_subjects(SearchMemoriesInSubjectsRequest {
                user_id: "u".to_string(),
                queries: vec![SubjectMemoryQuery {
                    subject_id: "not-a-uuid".to_string(),
                    query: "anything".to_string(),
                }],
                ..SearchMemoriesInSubjectsRequest::default()
            })
            .await
            .expect_err("invalid subject id must fail");
        assert!(err.to_string().contains("parse subject id"), "{err}");
    }

    #[tokio::test]
    async fn list_recent_memories_zero_limit_returns_empty() {
        let store = new_test_store().await;
        let got = store
            .list_recent_memories(ListRecentMemoriesRequest {
                user_id: "u".to_string(),
                ..ListRecentMemoriesRequest::default()
            })
            .await
            .expect("zero limit is not an error");
        assert!(got.is_empty());
    }
}
