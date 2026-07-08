// SPDX-License-Identifier: MIT
//! Core shared types, traits, and errors for the harness (plus the tiny
//! cost collector).
//!
//! Field names and JSON shapes intentionally match the original Ditto
//! backend's wire format for compatibility. Optional fields are skipped
//! during serialization when they hold their zero value.

use std::collections::BTreeMap;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Prefix used to derive a knowledge-graph id from a user id.
pub const DEFAULT_KG_PREFIX: &str = "user_memories_";

/// Session id used when none is provided.
pub const MAIN_SESSION_ID: &str = "main";

/// Returns the default knowledge-graph id for a user.
pub fn kg_id(user_id: &str) -> String {
    format!("{DEFAULT_KG_PREFIX}{user_id}")
}

/// Library-wide error type. CLI code wraps this in `anyhow`.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("database error: {0}")]
    Db(#[from] turso::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("memory: embedder is required")]
    NoEmbedder,
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("embedding error: {0}")]
    Embedding(String),
    #[error("model error: {0}")]
    Model(String),
    #[error("tool error: {0}")]
    Tool(String),
    #[error("{0}")]
    Other(String),
}

/// Crate-wide result alias.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Content part type. Serialized as the wire string values below.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContentType {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "text/markdown")]
    Markdown,
    #[serde(rename = "tool_call")]
    ToolCall,
    #[serde(rename = "tool_result")]
    ToolResult,
}

/// A model-issued tool call. `args` carries raw JSON.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub name: String,
    /// Raw JSON arguments; `Null` means absent.
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub args: Value,
}

/// The result of executing a tool call.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolCallResponse {
    #[serde(default)]
    pub id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
    /// Raw JSON output; `Null` means absent.
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub output: Value,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub error: String,
}

/// A single content part of a chat message or memory.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Content {
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub content_type: Option<ContentType>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call: Option<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_response: Option<ToolCallResponse>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Map<String, Value>>,
}

impl Content {
    /// Convenience constructor for a plain-text content part.
    pub fn text(content: impl Into<String>) -> Content {
        Content {
            content_type: Some(ContentType::Text),
            content: content.into(),
            ..Content::default()
        }
    }
}

/// A node in the seed-memory tree persisted with a saved pair.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SeedMemoryNode {
    #[serde(default)]
    pub pair_id: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<SeedMemoryNode>,
}

/// Metadata describing how a retrieval was performed.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetrievalMetadata {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub intent: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub weights: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub scale: f64,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub variant: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub retrieved_pair_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub query_embedding_model: String,
}

/// A stored memory pair plus retrieval scores.
///
/// `id` is the public pair id (`firestore_pair_id` column); `source_pair_id`
/// is the internal row UUID.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Memory {
    #[serde(default)]
    pub id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub source_pair_id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub user_id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub kg_id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub session_id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub title: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub summary: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub prompt: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub response: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input: Vec<Content>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub output: Vec<Content>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub source: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub source_context: String,
    /// Always serialized (never skipped as empty); RFC3339.
    pub timestamp: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "is_zero_i32")]
    pub timezone_offset: i32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub seed_memories: Vec<SeedMemoryNode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retrieval_metadata: Option<RetrievalMetadata>,
    /// Never serialized.
    #[serde(skip)]
    pub embedding: Vec<f32>,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub similarity: f64,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub recency_score: f64,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub frequency_score: f64,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
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

impl Default for Memory {
    fn default() -> Memory {
        Memory {
            id: String::new(),
            source_pair_id: String::new(),
            user_id: String::new(),
            kg_id: String::new(),
            session_id: String::new(),
            title: String::new(),
            summary: String::new(),
            prompt: String::new(),
            response: String::new(),
            input: Vec::new(),
            output: Vec::new(),
            source: String::new(),
            source_context: String::new(),
            timestamp: DateTime::<Utc>::UNIX_EPOCH,
            timezone_offset: 0,
            seed_memories: Vec::new(),
            retrieval_metadata: None,
            embedding: Vec::new(),
            similarity: 0.0,
            recency_score: 0.0,
            frequency_score: 0.0,
            composite_score: 0.0,
            recency_exp: 0.0,
            subject_sem_match: 0.0,
            session_continuity: 0.0,
            neighbor_density: 0.0,
        }
    }
}

/// A subject-graph node.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Subject {
    #[serde(default)]
    pub id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub user_id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub kg_id: String,
    /// Always serialized (never skipped as empty).
    pub text: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub key: bool,
    /// Never serialized.
    #[serde(skip)]
    pub embedding: Vec<f32>,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub similarity: f64,
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub memory_count: i64,
}

/// Token usage for a single model/embedding call.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub provider: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub model: String,
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub input_tokens: i64,
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub output_tokens: i64,
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub total_tokens: i64,
}

/// Monetary cost.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Cost {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub currency: String,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub amount: f64,
}

/// Usage paired with its cost. Both fields always serialize.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CostedUsage {
    pub usage: Usage,
    pub cost: Cost,
}

/// Request to embed one or more texts.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbedRequest {
    pub texts: Vec<String>,
}

/// Embedding response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbedResponse {
    pub embeddings: Vec<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<CostedUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Map<String, Value>>,
}

/// Produces embeddings for texts. Implementations must be
/// `Send + Sync` so they can be shared behind `Arc<dyn Embedder>`.
///
/// Contract:
/// - Every embedding must be exactly 768-dimensional
///   ([`crate::db::EMBEDDING_DIMS`]; the schema stores `F32_BLOB(768)`).
///   Other lengths error — at write time for stored embeddings, at query
///   time for query embeddings.
/// - `embeddings` must be 1:1 with `texts`, in order: callers index results
///   by input position. `Store::embed_texts` drops blank inputs before
///   calling, so implementations never see empty strings.
/// - The crate's similarity thresholds
///   ([`crate::memory::DEFAULT_MIN_SIMILARITY`] 0.15,
///   [`crate::dream::SUBJECT_MERGE_THRESHOLD`] 0.75) are calibrated for
///   embeddinggemma. A different embedder shifts what those cosine values
///   mean; expect to retune them.
#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, req: EmbedRequest) -> Result<EmbedResponse>;
}

/// A chat message in the agent loop.
/// `role` is one of "system" | "user" | "assistant" | "tool".
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatMessage {
    #[serde(default)]
    pub role: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub content: Vec<Content>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub tool_call_id: String,
}

/// One model turn result or stream delta.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatChunk {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "toolCall")]
    pub tool_call: Option<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<CostedUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Map<String, Value>>,
}

/// Callback receiving streamed [`ChatChunk`] deltas during [`Model::next_streaming`].
pub type OnChunk<'a> = &'a (dyn Fn(ChatChunk) + Send + Sync);

/// A chat model.
///
/// `next` performs one turn: given the conversation and available tool
/// definitions it returns either final text (`tool_call == None`) or a single
/// tool call to execute.
///
/// Streaming contract: `next_streaming` emits incremental [`ChatChunk`]
/// deltas (partial `text`, then optionally a `tool_call`, then a final chunk
/// carrying `cost`) through `on_chunk`, and returns the *aggregated* final
/// chunk — the same value `next` would have returned. The default
/// implementation calls `next` once and emits the full chunk a single time.
///
/// Note: the built-in agent loop drives `next` only (its `run_streaming`
/// streams loop *events*, not tokens); `next_streaming` is public API for
/// hosts that surface token-level deltas themselves.
#[async_trait]
pub trait Model: Send + Sync {
    async fn next(&self, messages: &[ChatMessage], tools: &[ToolDefinition]) -> Result<ChatChunk>;

    async fn next_streaming(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        on_chunk: OnChunk<'_>,
    ) -> Result<ChatChunk> {
        let chunk = self.next(messages, tools).await?;
        on_chunk(chunk.clone());
        Ok(chunk)
    }
}

/// JSON-schema description of a callable tool.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolDefinition {
    #[serde(default)]
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// Raw JSON schema for the tool input; `Null` means absent.
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub input_schema: Value,
}

/// A callable tool.
///
/// `execute` returns only the JSON output payload; the agent loop wraps it
/// into a [`ToolCallResponse`] (filling `id`/`name`, mapping `Err` to
/// `error`).
#[async_trait]
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    async fn execute(&self, args: Value) -> Result<Value>;
}

/// Accumulates per-call costs during an agent run.
#[derive(Debug, Clone, Default)]
pub struct CostCollector {
    items: Vec<CostedUsage>,
}

impl CostCollector {
    /// Records one cost item; `None` is ignored.
    pub fn add(&mut self, item: Option<&CostedUsage>) {
        if let Some(item) = item {
            self.items.push(item.clone());
        }
    }

    /// Returns the recorded items in insertion order.
    pub fn items(&self) -> &[CostedUsage] {
        &self.items
    }

    /// Consumes the collector, returning the recorded items.
    pub fn into_items(self) -> Vec<CostedUsage> {
        self.items
    }

    /// Sums recorded amounts; the last non-empty currency wins.
    pub fn total(&self) -> Cost {
        let mut total = Cost::default();
        for item in &self.items {
            if !item.cost.currency.is_empty() {
                total.currency = item.cost.currency.clone();
            }
            total.amount += item.cost.amount;
        }
        total
    }
}

pub(crate) fn is_zero_f64(v: &f64) -> bool {
    *v == 0.0
}

pub(crate) fn is_zero_i64(v: &i64) -> bool {
    *v == 0
}

pub(crate) fn is_zero_i32(v: &i32) -> bool {
    *v == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_json_matches_wire_field_names() {
        let mem = Memory {
            id: "pair-1".into(),
            source_pair_id: "row-uuid".into(),
            user_id: "u1".into(),
            kg_id: kg_id("u1"),
            session_id: "main".into(),
            summary: "s".into(),
            timezone_offset: -300,
            similarity: 0.5,
            ..Memory::default()
        };
        let v = serde_json::to_value(&mem).expect("serialize");
        let obj = v.as_object().expect("object");
        for key in [
            "id",
            "sourcePairId",
            "userId",
            "kgId",
            "sessionId",
            "summary",
            "timestamp",
            "timezoneOffset",
            "similarity",
        ] {
            assert!(obj.contains_key(key), "missing key {key}");
        }
        // Zero scores and empty strings are skipped during serialization.
        for key in ["title", "prompt", "recencyScore", "compositeScore", "input"] {
            assert!(!obj.contains_key(key), "unexpected key {key}");
        }
    }

    #[test]
    fn kg_id_derives_default_prefix() {
        assert_eq!(kg_id("abc"), "user_memories_abc");
    }
}
