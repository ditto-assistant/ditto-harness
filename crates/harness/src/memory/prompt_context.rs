// SPDX-License-Identifier: MIT
//! Prompt memory context: long/short-term retrieval bundling and the JSON
//! payload injected into the system prompt.
//!
//! What the model actually sees is a system message built by
//! `chat::memory_context_message`: the prefix
//! `"Relevant memory context for this turn:\n"` followed by one JSON object.
//! `longTerm` embeds [`build_prompt_long_term_json`]'s payload raw — the
//! first [`PROMPT_DETAILED_SEED_ROOT_COUNT`] entries are detailed (`summary`,
//! or `user`/`ditto` text when there is none, plus `parent`-stub children),
//! the rest are title-only. `shortTerm` entries are always compact. Shape
//! (whitespace added):
//!
//! ```json
//! {
//!   "longTerm": {"memories": [
//!     {"pairID": "p1", "timestamp": "2026-01-02T12:00:00Z",
//!      "summary": "User prefers Rust for backend work."},
//!     {"pairID": "c1", "parent": "p1"},
//!     {"pairID": "p2", "timestamp": "2026-01-01T09:30:00Z",
//!      "user": "original prompt text", "ditto": "original response text"},
//!     {"pairID": "p3", "timestamp": "2025-12-30T08:00:00Z",
//!      "title": "Home lab setup"}
//!   ]},
//!   "shortTerm": [
//!     {"pairID": "p4", "summary": "Chose Turso for local storage.",
//!      "timestamp": "2026-01-02T13:00:00Z", "title": "Chose Turso for local storage."}
//!   ]
//! }
//! ```

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::{ListRecentMemoriesRequest, Store};
use crate::retrieval::Variant;
use crate::types::{Memory, Result, RetrievalMetadata, SeedMemoryNode};

/// How many leading long-term memories get detailed JSON entries.
pub const PROMPT_DETAILED_SEED_ROOT_COUNT: usize = 2;

/// Summary row describing one prompt memory. JSON field names follow the
/// established wire format (note the capital-ID forms).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PromptMemorySummary {
    #[serde(rename = "pairID", default)]
    pub pair_id: String,
    #[serde(
        rename = "sessionID",
        default,
        skip_serializing_if = "String::is_empty"
    )]
    pub session_id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub source: String,
    /// Always serialized.
    pub timestamp: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub title: String,
    /// Always serialized.
    pub children: usize,
    #[serde(
        rename = "cosineSim",
        default,
        skip_serializing_if = "crate::types::is_zero_f64"
    )]
    pub cosine_sim: f64,
    #[serde(
        rename = "recencyScore",
        default,
        skip_serializing_if = "crate::types::is_zero_f64"
    )]
    pub recency_score: f64,
    #[serde(
        rename = "frequencyScore",
        default,
        skip_serializing_if = "crate::types::is_zero_f64"
    )]
    pub frequency_score: f64,
    #[serde(
        rename = "compositeScore",
        default,
        skip_serializing_if = "crate::types::is_zero_f64"
    )]
    pub composite_score: f64,
}

/// Request for [`Store::get_prompt_memories`].
/// Defaults: empty `kg_id` -> derived, empty `session_id` -> "main",
/// `long_term_limit` 0 -> 8. `short_term_limit` 0 -> no short-term lookup.
#[derive(Debug, Clone, Default)]
pub struct PromptMemoryRequest {
    pub user_id: String,
    pub kg_id: String,
    pub session_id: String,
    /// Empty query skips long-term retrieval entirely.
    pub query: String,
    pub long_term_limit: usize,
    pub short_term_limit: usize,
    pub candidate_pool_size: usize,
    pub exclude_pair_ids: Vec<String>,
    pub variant: Variant,
    pub request_path: String,
    pub log_retrieval: bool,
    /// When true, long-term uses composite search; otherwise plain vector
    /// search.
    pub use_composite: bool,
}

/// Diagnostics counter.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct PromptMemoryValue {
    pub count: usize,
}

/// Bundle of prompt memories.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptMemoryResponse {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub long_term: Vec<Memory>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub short_term: Vec<Memory>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub seed_memory_nodes: Vec<SeedMemoryNode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retrieval_metadata: Option<RetrievalMetadata>,
    #[serde(
        rename = "longTermJSON",
        default,
        skip_serializing_if = "String::is_empty"
    )]
    pub long_term_json: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub summary: Vec<PromptMemorySummary>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub diagnostics: BTreeMap<String, PromptMemoryValue>,
}

impl Store {
    /// Retrieves long-term (vector or composite) and short-term (recent)
    /// memories, excluding overlaps, and assembles the prompt bundle:
    /// seed nodes, ids, long-term JSON, summaries, and `longTerm`/`shortTerm`
    /// diagnostics counts.
    pub async fn get_prompt_memories(
        &self,
        mut req: PromptMemoryRequest,
    ) -> Result<PromptMemoryResponse> {
        if req.kg_id.is_empty() {
            req.kg_id = crate::types::kg_id(&req.user_id);
        }
        if req.session_id.is_empty() {
            req.session_id = crate::types::MAIN_SESSION_ID.to_string();
        }
        if req.long_term_limit == 0 {
            req.long_term_limit = super::DEFAULT_SEARCH_LIMIT;
        }

        let (long_term, retrieval_metadata) = self.get_prompt_long_term(&req).await?;
        let mut exclude = req.exclude_pair_ids.clone();
        exclude.extend(memory_ids(&long_term));

        let short_term = self
            .list_recent_memories(ListRecentMemoriesRequest {
                user_id: req.user_id.clone(),
                kg_id: req.kg_id.clone(),
                session_id: req.session_id.clone(),
                limit: req.short_term_limit,
                exclude_pair_ids: exclude,
            })
            .await?;

        let mut ids = memory_ids(&long_term);
        ids.extend(memory_ids(&short_term));
        let seed_nodes = seed_memory_nodes(&long_term, &short_term);
        let mut combined = long_term.clone();
        combined.extend(short_term.iter().cloned());
        let summary = summarize_prompt_memories(&combined, long_term.len() + short_term.len());
        let mut diagnostics = BTreeMap::new();
        diagnostics.insert(
            "longTerm".to_string(),
            PromptMemoryValue {
                count: long_term.len(),
            },
        );
        diagnostics.insert(
            "shortTerm".to_string(),
            PromptMemoryValue {
                count: short_term.len(),
            },
        );
        Ok(PromptMemoryResponse {
            long_term_json: build_prompt_long_term_json(&long_term),
            long_term,
            short_term,
            seed_memory_nodes: seed_nodes,
            retrieval_metadata,
            ids,
            summary,
            diagnostics,
        })
    }

    /// Long-term retrieval for the prompt: composite when requested, plain
    /// vector search otherwise; empty queries skip retrieval entirely.
    async fn get_prompt_long_term(
        &self,
        req: &PromptMemoryRequest,
    ) -> Result<(Vec<Memory>, Option<RetrievalMetadata>)> {
        if req.query.is_empty() {
            return Ok((Vec::new(), None));
        }
        if req.use_composite {
            return self
                .search_composite_memories(super::CompositeSearchRequest {
                    user_id: req.user_id.clone(),
                    kg_id: req.kg_id.clone(),
                    session_id: req.session_id.clone(),
                    query: req.query.clone(),
                    limit: req.long_term_limit,
                    candidate_pool_size: req.candidate_pool_size,
                    exclude_pair_ids: req.exclude_pair_ids.clone(),
                    variant: req.variant,
                    request_path: req.request_path.clone(),
                    log_event: req.log_retrieval,
                })
                .await;
        }
        let memories = self
            .search_memories(super::SearchMemoriesRequest {
                user_id: req.user_id.clone(),
                kg_id: req.kg_id.clone(),
                session_id: req.session_id.clone(),
                queries: vec![req.query.clone()],
                limit: req.long_term_limit,
                exclude_pair_ids: req.exclude_pair_ids.clone(),
                ..super::SearchMemoriesRequest::default()
            })
            .await?;
        Ok((memories, None))
    }
}

/// Empty session resolves to "main".
pub fn resolve_prompt_session_id(session_id: &str) -> String {
    if session_id.is_empty() {
        return crate::types::MAIN_SESSION_ID.to_string();
    }
    session_id.to_string()
}

/// Summarizes up to `limit` memories.
/// Carries pair id, resolved session, source, timestamp, title, child count,
/// cosine similarity, and composite score.
pub fn summarize_prompt_memories(memories: &[Memory], limit: usize) -> Vec<PromptMemorySummary> {
    if limit == 0 || memories.is_empty() {
        return Vec::new();
    }
    let limit = limit.min(memories.len());
    memories[..limit]
        .iter()
        .map(|mem| PromptMemorySummary {
            pair_id: mem.id.clone(),
            session_id: resolve_prompt_session_id(&mem.session_id),
            source: mem.source.clone(),
            timestamp: mem.timestamp,
            title: prompt_memory_title(mem),
            children: mem.seed_memories.len(),
            cosine_sim: mem.similarity,
            recency_score: mem.recency_score,
            frequency_score: mem.frequency_score,
            composite_score: mem.composite_score,
        })
        .collect()
}

/// Builds the `{"memories":[...]}` JSON string injected into the prompt.
/// The first
/// [`PROMPT_DETAILED_SEED_ROOT_COUNT`] memories are detailed (summary or
/// user/ditto text, plus child stubs with `parent` set); the rest are
/// title-only entries. Returns `{"memories":[]}` for empty input.
pub fn build_prompt_long_term_json(memories: &[Memory]) -> String {
    const EMPTY: &str = r#"{"memories":[]}"#;
    if memories.is_empty() {
        return EMPTY.to_string();
    }
    let mut items: Vec<serde_json::Value> = Vec::with_capacity(memories.len() * 2);
    for (i, mem) in memories.iter().enumerate() {
        let detailed = i < PROMPT_DETAILED_SEED_ROOT_COUNT;
        add_prompt_memory_json(&mut items, mem, "", detailed);
    }
    serde_json::to_string(&serde_json::json!({ "memories": items }))
        .unwrap_or_else(|_| EMPTY.to_string())
}

/// Best display title for a memory: title, else summary, else prompt, else
/// first non-empty input content.
pub fn prompt_memory_title(mem: &Memory) -> String {
    if !mem.title.is_empty() {
        return mem.title.clone();
    }
    if !mem.summary.is_empty() {
        return mem.summary.clone();
    }
    if !mem.prompt.is_empty() {
        return mem.prompt.clone();
    }
    for part in &mem.input {
        if !part.content.is_empty() {
            return part.content.clone();
        }
    }
    String::new()
}

/// Flattens long-term then short-term memories into seed nodes, each carrying
/// its own child seed memories.
pub fn seed_memory_nodes(long_term: &[Memory], short_term: &[Memory]) -> Vec<SeedMemoryNode> {
    long_term
        .iter()
        .chain(short_term.iter())
        .map(|mem| SeedMemoryNode {
            pair_id: mem.id.clone(),
            children: mem.seed_memories.clone(),
        })
        .collect()
}

/// One entry (plus child stubs when detailed) for the long-term prompt JSON.
fn add_prompt_memory_json(
    items: &mut Vec<serde_json::Value>,
    mem: &Memory,
    parent_id: &str,
    detailed: bool,
) {
    let mut item = serde_json::Map::new();
    item.insert("pairID".to_string(), serde_json::json!(mem.id));
    item.insert(
        "timestamp".to_string(),
        serde_json::json!(mem
            .timestamp
            .to_rfc3339_opts(chrono::SecondsFormat::Secs, true)),
    );
    if !parent_id.is_empty() {
        item.insert("parent".to_string(), serde_json::json!(parent_id));
    }
    if detailed {
        if !mem.summary.is_empty() {
            item.insert("summary".to_string(), serde_json::json!(mem.summary));
        } else {
            item.insert(
                "user".to_string(),
                serde_json::json!(content_text(&mem.input, &mem.prompt)),
            );
            item.insert(
                "ditto".to_string(),
                serde_json::json!(content_text(&mem.output, &mem.response)),
            );
        }
        items.push(serde_json::Value::Object(item));
        for child in &mem.seed_memories {
            items.push(serde_json::json!({
                "pairID": child.pair_id,
                "parent": mem.id,
            }));
        }
        return;
    }
    item.insert(
        "title".to_string(),
        serde_json::json!(prompt_memory_title(mem)),
    );
    items.push(serde_json::Value::Object(item));
}

/// Non-empty `fallback`, else the concatenated non-empty content parts.
fn content_text(parts: &[crate::types::Content], fallback: &str) -> String {
    if !fallback.is_empty() {
        return fallback.to_string();
    }
    let mut out = String::new();
    for part in parts {
        if !part.content.is_empty() {
            out.push_str(&part.content);
        }
    }
    out
}

/// Pair ids of memories with non-empty ids.
fn memory_ids(memories: &[Memory]) -> Vec<String> {
    memories
        .iter()
        .filter(|mem| !mem.id.is_empty())
        .map(|mem| mem.id.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;

    use super::super::test_support::new_test_store;
    use super::super::SaveMemoryRequest;
    use super::*;

    /// The first two long-term entries are detailed; later ones compress to
    /// title-only.
    #[test]
    fn build_prompt_long_term_json_compresses_after_first_two() {
        let ts = |day: u32| {
            Utc.with_ymd_and_hms(2026, 1, day, 12, 0, 0)
                .single()
                .expect("timestamp")
        };
        let memories = vec![
            Memory {
                id: "pair-1".to_string(),
                summary: "First summary".to_string(),
                timestamp: ts(1),
                ..Memory::default()
            },
            Memory {
                id: "pair-2".to_string(),
                prompt: "Second prompt".to_string(),
                response: "Second response".to_string(),
                timestamp: ts(2),
                ..Memory::default()
            },
            Memory {
                id: "pair-3".to_string(),
                title: "Third title".to_string(),
                prompt: "Third prompt".to_string(),
                response: "Third response".to_string(),
                timestamp: ts(3),
                ..Memory::default()
            },
        ];
        let raw = build_prompt_long_term_json(&memories);
        let payload: serde_json::Value = serde_json::from_str(&raw).expect("parse prompt json");
        let items = payload["memories"].as_array().expect("memories array");
        assert_eq!(items.len(), 3);
        assert!(
            items[0].get("summary").is_some(),
            "first memory should be detailed: {:?}",
            items[0]
        );
        assert!(
            items[1].get("user").is_some(),
            "second memory should include user text: {:?}",
            items[1]
        );
        assert!(
            items[2].get("title").is_some(),
            "third memory should be compressed title-only: {:?}",
            items[2]
        );
        assert!(
            items[2].get("user").is_none(),
            "third memory should not include full user text: {:?}",
            items[2]
        );
        assert_eq!(items[0]["timestamp"], "2026-01-01T12:00:00Z");
        assert_eq!(build_prompt_long_term_json(&[]), r#"{"memories":[]}"#);
    }

    /// Summaries resolve the session id and carry similarity/composite scores.
    #[test]
    fn summarize_prompt_memories_resolves_session_and_scores() {
        let memories = vec![Memory {
            id: "pair-1".to_string(),
            session_id: String::new(),
            source: "agent".to_string(),
            title: "Title".to_string(),
            timestamp: Utc
                .with_ymd_and_hms(2026, 1, 1, 12, 0, 0)
                .single()
                .expect("timestamp"),
            similarity: 0.8,
            composite_score: 0.7,
            ..Memory::default()
        }];
        let got = summarize_prompt_memories(&memories, 10);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].session_id, crate::types::MAIN_SESSION_ID);
        assert_eq!(got[0].title, "Title");
        assert_eq!(got[0].cosine_sim, 0.8);
        assert_eq!(got[0].composite_score, 0.7);
        assert!(summarize_prompt_memories(&memories, 0).is_empty());
    }

    /// The prompt bundle pairs long-term retrieval with recent short-term
    /// memories (non-composite long-term path; composite is owned by
    /// `retrieval`).
    #[tokio::test]
    async fn get_prompt_memories_returns_long_term_and_recent_short_term() {
        let store = new_test_store().await;
        let older = Utc
            .with_ymd_and_hms(2026, 1, 1, 10, 0, 0)
            .single()
            .expect("timestamp");

        let long_term = store
            .save_memory(SaveMemoryRequest {
                id: "long-term".to_string(),
                user_id: "prompt-user".to_string(),
                session_id: "thread-1".to_string(),
                prompt: "Remember the harness retrieves composite memory context.".to_string(),
                response: "Composite retrieval should seed the prompt.".to_string(),
                summary: "Harness composite retrieval context.".to_string(),
                timestamp: Some(older),
                ..SaveMemoryRequest::default()
            })
            .await
            .expect("save long-term");
        store
            .save_memory(SaveMemoryRequest {
                id: "recent-thread".to_string(),
                user_id: "prompt-user".to_string(),
                session_id: "thread-1".to_string(),
                prompt: "Recent thread note about packaging.".to_string(),
                response: "Keep the chat harness importable.".to_string(),
                summary: "Recent importable harness note.".to_string(),
                timestamp: Some(older + chrono::Duration::hours(1)),
                ..SaveMemoryRequest::default()
            })
            .await
            .expect("save recent");

        let got = store
            .get_prompt_memories(PromptMemoryRequest {
                user_id: "prompt-user".to_string(),
                session_id: "thread-1".to_string(),
                query: "composite memory context".to_string(),
                long_term_limit: 1,
                short_term_limit: 2,
                ..PromptMemoryRequest::default()
            })
            .await
            .expect("get_prompt_memories");
        assert_eq!(got.long_term.len(), 1, "long term: {:?}", got.long_term);
        assert_eq!(got.long_term[0].id, long_term.id);
        assert_eq!(got.short_term.len(), 1, "short term: {:?}", got.short_term);
        assert_eq!(got.short_term[0].id, "recent-thread");
        assert_eq!(got.seed_memory_nodes.len(), 2);
        assert!(!got.long_term_json.is_empty());
        assert_eq!(got.ids, vec!["long-term", "recent-thread"]);
        assert_eq!(
            got.diagnostics.get("longTerm").map(|value| value.count),
            Some(1)
        );
        assert_eq!(
            got.diagnostics.get("shortTerm").map(|value| value.count),
            Some(1)
        );
    }
}
