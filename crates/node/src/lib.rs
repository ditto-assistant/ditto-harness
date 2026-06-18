// SPDX-License-Identifier: AGPL-3.0-or-later
//! Node.js bindings for ditto-harness (napi-rs v3).
//!
//! Deliberately JSON-heavy: memories, subjects, and dream reports cross the
//! boundary as plain JSON objects (`serde_json::Value`), keeping the surface
//! small and honest while the Rust core stays the source of truth.

use std::sync::{Arc, Mutex};

use chrono::{DateTime, Duration, Utc};
use napi::bindgen_prelude::Result;
use napi::Error as NapiError;
use napi_derive::napi;
use serde::Deserialize;
use serde_json::Value;

use ditto_harness::agent::EventHandler;
use ditto_harness::chat::{
    Harness as ChatHarness, Options as ChatOptions, PrepareRequest, RunRequest,
};
use ditto_harness::db::{Db, EMBEDDING_DIMS};
use ditto_harness::dream::{DreamOptions, Dreamer};
use ditto_harness::memory::{
    SaveMemoryRequest, SearchMemoriesRequest, SearchSubjectsRequest, Store, StoreOptions,
    SubjectInput,
};
use ditto_harness::models::{
    ChatModelConfig, OllamaEmbedder, DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_CHAT_MODEL,
};
use ditto_harness::types::{EmbedRequest, EmbedResponse, Embedder, Model};

/// Returns the ditto-harness crate version.
#[napi]
pub fn harness_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

fn js_err(err: impl std::fmt::Display) -> NapiError {
    NapiError::from_reason(err.to_string())
}

fn to_json<T: serde::Serialize>(value: &T) -> Result<Value> {
    serde_json::to_value(value).map_err(js_err)
}

// ---------------------------------------------------------------------------
// Stub embedder (offline path)
// ---------------------------------------------------------------------------

/// Deterministic bag-of-words embedder used when no Ollama server is
/// available (mirrors the harness test `HashEmbedder`): each lowercase
/// whitespace token bumps one hashed dimension, then the vector is
/// L2-normalized. Useful for tests and offline smoke runs only.
struct HashEmbedder;

#[async_trait::async_trait]
impl Embedder for HashEmbedder {
    async fn embed(&self, req: EmbedRequest) -> ditto_harness::Result<EmbedResponse> {
        Ok(EmbedResponse {
            embeddings: req.texts.iter().map(|t| hash_embedding(t)).collect(),
            ..EmbedResponse::default()
        })
    }
}

fn hash_embedding(text: &str) -> Vec<f32> {
    use std::hash::{Hash, Hasher};
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

// ---------------------------------------------------------------------------
// Option objects (typed for .d.ts)
// ---------------------------------------------------------------------------

/// Options for [`Harness::open`].
#[napi(object)]
#[derive(Default)]
pub struct OpenOptions {
    /// Base URL for the local Ollama server used for embeddings.
    /// Default: `http://localhost:11434`.
    pub ollama_base_url: Option<String>,
    /// Embedder backend: `"ollama"` (default, embeddinggemma 768 dims) or
    /// `"hash"` (deterministic offline stub for tests/smoke runs).
    pub embedder: Option<String>,
}

/// Options for [`Harness::search_memories`].
#[napi(object)]
#[derive(Default)]
pub struct SearchOptions {
    /// Restrict to one session; omit to search all sessions.
    pub session_id: Option<String>,
    /// Max results. Default 8.
    pub limit: Option<u32>,
    /// Minimum cosine similarity. Default 0.15.
    pub min_similarity: Option<f64>,
}

/// Chat-model selection shared by [`Harness::dream`] and [`Harness::chat`].
#[napi(object)]
#[derive(Default)]
pub struct ModelOptions {
    /// `"ollama"` (default), `"openrouter"`, or `"vllm"`.
    pub provider: Option<String>,
    /// Model name. Default for ollama: `gemma3:4b`. Required for
    /// openrouter/vllm.
    pub model: Option<String>,
    /// Provider base URL (ollama/vllm). Required for vllm.
    pub base_url: Option<String>,
    /// API key for openrouter; falls back to `OPENROUTER_API_KEY`.
    pub api_key: Option<String>,
    /// Chat session id. Default `"main"`. (chat only)
    pub session_id: Option<String>,
    /// Max agent-loop turns. Default 8. (chat only)
    pub max_turns: Option<u32>,
    /// Persist the exchange as a memory pair. Default true. (chat only)
    pub save_memory: Option<bool>,
    /// Cap on memories considered, newest first. Default 128. (dream only)
    pub max_memories: Option<u32>,
    /// Run the dream refine stage. Default true. (dream only)
    pub refine: Option<bool>,
}

/// Result of one [`Harness::chat`] turn.
#[napi(object)]
pub struct ChatTurnResult {
    /// Final assistant text.
    pub response: String,
    /// Total monetary cost across model calls (0 for local providers).
    pub cost: f64,
    /// Names of the tools the agent called, in order.
    pub tool_calls: Vec<String>,
}

// ---------------------------------------------------------------------------
// saveMemory JSON payload
// ---------------------------------------------------------------------------

/// Accepted JSON shape for [`Harness::save_memory`] (camelCase keys; all but
/// `userId`, `prompt`, `response` optional).
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct SaveMemoryJson {
    user_id: String,
    kg_id: String,
    session_id: String,
    /// Stable public pair id; empty -> generated uuid v4.
    id: String,
    title: String,
    summary: String,
    prompt: String,
    response: String,
    source: String,
    source_context: String,
    /// RFC3339 timestamp; omitted -> now (or `daysAgo` backdating).
    timestamp: Option<String>,
    /// Backdates the memory N days (ignored when `timestamp` is set).
    days_ago: Option<i64>,
    timezone_offset: i32,
    subjects: Vec<SubjectInput>,
}

// ---------------------------------------------------------------------------
// EventHandler recording tool calls during chat
// ---------------------------------------------------------------------------

#[derive(Default)]
struct RecordingEvents {
    tool_calls: Mutex<Vec<String>>,
}

impl EventHandler for RecordingEvents {
    fn send_tool_call_progress(&self, _tool_call_id: &str, data: &Value) {
        let name = data
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        if let Ok(mut calls) = self.tool_calls.lock() {
            calls.push(name);
        }
    }
}

// ---------------------------------------------------------------------------
// Harness class
// ---------------------------------------------------------------------------

/// Agent memory harness over a local Turso/SQLite database: save and search
/// memories with vector similarity, search the subject graph, run the dream
/// (subject extraction) pipeline, and run one memory-augmented chat turn.
#[napi]
pub struct Harness {
    store: Arc<Store>,
    embedder: Arc<dyn Embedder>,
    ollama_base_url: String,
}

#[napi]
impl Harness {
    /// Opens (creating if needed) the database at `dbPath`, applies
    /// migrations, and wires the memory store with the configured embedder.
    /// Static factory: `const h = await Harness.open("./mem.db")`.
    #[napi]
    pub async fn open(db_path: String, opts: Option<OpenOptions>) -> Result<Harness> {
        let opts = opts.unwrap_or_default();
        let ollama_base_url = opts
            .ollama_base_url
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| DEFAULT_OLLAMA_BASE_URL.to_string());
        let embedder: Arc<dyn Embedder> = match opts.embedder.as_deref() {
            None | Some("ollama") => Arc::new(OllamaEmbedder::new(ollama_base_url.clone())),
            Some("hash") => Arc::new(HashEmbedder),
            Some(other) => {
                return Err(js_err(format!(
                    "unknown embedder {other:?}: expected \"ollama\" or \"hash\""
                )))
            }
        };
        let db = Db::open(&db_path).await.map_err(js_err)?;
        let store = Arc::new(Store::new(StoreOptions {
            db: Arc::new(db),
            embedder: Arc::clone(&embedder),
            predictor: None,
        }));
        Ok(Harness {
            store,
            embedder,
            ollama_base_url,
        })
    }

    /// Creates the user row if it does not exist (idempotent).
    #[napi]
    pub async fn seed_user(&self, uid: String) -> Result<()> {
        let store = Arc::clone(&self.store);
        store.db().upsert_user(&uid).await.map_err(js_err)
    }

    /// Saves one memory pair from a JSON object (camelCase keys:
    /// `userId`, `prompt`, `response`, plus optional `id`, `title`,
    /// `summary`, `sessionId`, `source`, `timestamp` (RFC3339), `daysAgo`,
    /// `subjects: [{text, description, key}]`). Embeds the pair and each
    /// subject; returns the stored memory as a JSON object.
    #[napi]
    pub async fn save_memory(&self, json: Value) -> Result<Value> {
        let req: SaveMemoryJson = serde_json::from_value(json).map_err(js_err)?;
        let timestamp: Option<DateTime<Utc>> = match (&req.timestamp, req.days_ago) {
            (Some(ts), _) => Some(
                DateTime::parse_from_rfc3339(ts)
                    .map_err(|e| js_err(format!("invalid timestamp {ts:?}: {e}")))?
                    .with_timezone(&Utc),
            ),
            (None, Some(days)) => Some(Utc::now() - Duration::days(days)),
            (None, None) => None,
        };
        let store = Arc::clone(&self.store);
        let memory = store
            .save_memory(SaveMemoryRequest {
                user_id: req.user_id,
                kg_id: req.kg_id,
                session_id: req.session_id,
                id: req.id,
                title: req.title,
                summary: req.summary,
                prompt: req.prompt,
                response: req.response,
                source: if req.source.is_empty() {
                    "node".to_string()
                } else {
                    req.source
                },
                source_context: req.source_context,
                timestamp,
                timezone_offset: req.timezone_offset,
                subjects: req.subjects,
                ..SaveMemoryRequest::default()
            })
            .await
            .map_err(js_err)?;
        to_json(&memory)
    }

    /// Vector-searches the user's memories for `query`; returns memory JSON
    /// objects ordered by descending similarity.
    #[napi]
    pub async fn search_memories(
        &self,
        uid: String,
        query: String,
        opts: Option<SearchOptions>,
    ) -> Result<Vec<Value>> {
        let opts = opts.unwrap_or_default();
        let store = Arc::clone(&self.store);
        let memories = store
            .search_memories(SearchMemoriesRequest {
                user_id: uid,
                session_id: opts.session_id.unwrap_or_default(),
                queries: vec![query],
                limit: opts.limit.unwrap_or(0) as usize,
                min_similarity: opts.min_similarity.unwrap_or(0.0),
                ..SearchMemoriesRequest::default()
            })
            .await
            .map_err(js_err)?;
        memories.iter().map(to_json).collect()
    }

    /// Vector-searches the user's subject graph for `query`; returns subject
    /// JSON objects ordered by descending similarity.
    #[napi]
    pub async fn search_subjects(&self, uid: String, query: String) -> Result<Vec<Value>> {
        let store = Arc::clone(&self.store);
        let subjects = store
            .search_subjects(SearchSubjectsRequest {
                user_id: uid,
                queries: vec![query],
                ..SearchSubjectsRequest::default()
            })
            .await
            .map_err(js_err)?;
        subjects.iter().map(to_json).collect()
    }

    /// Runs the dream pipeline over the user's recent memories: extracts
    /// durable subjects with the configured chat model, dedups/merges them
    /// into the subject graph, links them to source pairs, and optionally
    /// refines merged descriptions. Returns the DreamReport as JSON.
    #[napi]
    pub async fn dream(&self, uid: String, opts: Option<ModelOptions>) -> Result<Value> {
        let opts = opts.unwrap_or_default();
        let model = self.build_model(&opts)?;
        let store = Arc::clone(&self.store);
        let embedder = Arc::clone(&self.embedder);
        let dreamer = Dreamer::new(store, model, embedder);
        let report = dreamer
            .dream(
                &uid,
                DreamOptions {
                    max_memories: opts.max_memories.unwrap_or(0) as usize,
                    refine: opts.refine,
                    ..DreamOptions::default()
                },
            )
            .await
            .map_err(js_err)?;
        to_json(&report)
    }

    /// Runs one memory-augmented chat turn: retrieves prompt memories,
    /// exposes the five memory tools to the agent loop, and (by default)
    /// saves the exchange as a new memory pair.
    #[napi]
    pub async fn chat(
        &self,
        uid: String,
        message: String,
        opts: Option<ModelOptions>,
    ) -> Result<ChatTurnResult> {
        if message.trim().is_empty() {
            return Err(js_err("chat: message must not be empty"));
        }
        let opts = opts.unwrap_or_default();
        let model = self.build_model(&opts)?;
        let store = Arc::clone(&self.store);
        let harness = ChatHarness::new(ChatOptions {
            model,
            memory: Some(Arc::clone(&store)),
            tools: Vec::new(),
            include_memory_tools: true,
        });
        let events = RecordingEvents::default();
        let result = harness
            .run(
                RunRequest {
                    prepare: PrepareRequest {
                        user_id: uid,
                        session_id: opts.session_id.clone().unwrap_or_default(),
                        user_input: message,
                        use_composite: true,
                        ..PrepareRequest::default()
                    },
                    max_turns: opts.max_turns.unwrap_or(0) as usize,
                    save_memory: opts.save_memory.unwrap_or(true),
                    source: "node".to_string(),
                    ..RunRequest::default()
                },
                &events,
            )
            .await
            .map_err(js_err)?;
        let cost = result
            .result
            .costs
            .iter()
            .map(|c| c.cost.amount)
            .sum::<f64>();
        let tool_calls = events
            .tool_calls
            .into_inner()
            .map_err(|_| js_err("chat: tool-call recorder poisoned"))?;
        Ok(ChatTurnResult {
            response: result.result.text,
            cost,
            tool_calls,
        })
    }
}

impl Harness {
    /// Builds the chat model from `ModelOptions` (default: local Ollama with
    /// `gemma3:4b`).
    fn build_model(&self, opts: &ModelOptions) -> Result<Arc<dyn Model>> {
        let provider = opts.provider.as_deref().unwrap_or("ollama");
        let config = match provider {
            "ollama" => ChatModelConfig::ollama(
                opts.base_url
                    .clone()
                    .unwrap_or_else(|| self.ollama_base_url.clone()),
                opts.model
                    .clone()
                    .unwrap_or_else(|| DEFAULT_OLLAMA_CHAT_MODEL.to_string()),
            ),
            "openrouter" => {
                let api_key = opts
                    .api_key
                    .clone()
                    .filter(|k| !k.is_empty())
                    .or_else(|| std::env::var("OPENROUTER_API_KEY").ok())
                    .ok_or_else(|| {
                        js_err("openrouter requires apiKey (or OPENROUTER_API_KEY env)")
                    })?;
                let model = opts
                    .model
                    .clone()
                    .ok_or_else(|| js_err("openrouter requires a model name"))?;
                ChatModelConfig::openrouter(api_key, model)
            }
            "vllm" => {
                let base_url = opts
                    .base_url
                    .clone()
                    .ok_or_else(|| js_err("vllm requires baseUrl"))?;
                let model = opts
                    .model
                    .clone()
                    .ok_or_else(|| js_err("vllm requires a model name"))?;
                ChatModelConfig::vllm(base_url, model)
            }
            other => {
                return Err(js_err(format!(
                    "unknown provider {other:?}: expected \"ollama\", \"openrouter\", or \"vllm\""
                )))
            }
        };
        config.build().map_err(js_err)
    }
}
